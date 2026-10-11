"""CPU contract fixtures, no actual source/checkpoint, GPU or real learning."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest

import numpy as np

from scripts import dl_reset_learning as learn


def inputs():
  calibration, contexts, pins = {}, {}, {}
  for episode in learn.EPISODES:
    # broadcasted immutable pixels are sufficient for synthetic contract data.
    first, last = np.zeros(1001, bool), np.zeros(1001, bool)
    first[0], last[-1] = True, True
    executed = np.ones(1001, bool)
    executed[-1] = False
    calibration[episode] = dict(
        image=np.broadcast_to(np.uint8(episode), (1001, 64, 64, 3)),
        reward=np.arange(1001, dtype=np.float32) / 1001,
        is_first=first, is_last=last, is_terminal=np.zeros(1001, bool),
        action=np.full((1001, 12), episode / 100, np.float32),
        action_executed=executed)
    calibration[episode]['action'][-1] = 0
    contexts[episode] = {'dyn/deter': np.full(4, episode, np.float32),
        'dyn/stoch': np.eye(2, dtype=np.float32)}
    pins[episode] = f'{episode:064x}'
  return calibration, pins, contexts


class Counter:
  def __init__(self, value=0):
    self.value = value

  def __int__(self):
    return self.value


class FixtureDyn:
  free_nats = 1

  def __init__(self):
    self._dl_reset_learning_branch_at_construction = 'B'

  def loss(self, *args):
    return args

  # This mock only certifies helper sequencing; the actual wrapper arithmetic
  # is checked against an independent CPU JAX oracle below.
  loss._dl_reset_learning_wrapper = 'B'


class FixtureAgent:
  """Tracks actual helper sequencing; not a production neural agent."""
  def __init__(self):
    self.config = types.SimpleNamespace(batch_size=16, batch_length=64, replay_context=1)
    self.jaxcfg = types.SimpleNamespace(enable_policy=False)
    self.model = types.SimpleNamespace(dyn=FixtureDyn(), scales={'rep': .1, 'dyn': 1},
        config=types.SimpleNamespace(rep_probe=types.SimpleNamespace(mode='off'),
            dt_latch=types.SimpleNamespace(mode='off'), ac_grads=False,
            reward_grad=True, repval_grad=True))
    self.spaces = dict(image=(np.uint8, (64, 64, 3)), reward=(np.float32, ()),
        is_first=(bool, ()), is_last=(bool, ()), is_terminal=(bool, ()),
        action=(np.float32, (12,)), consec=(np.int32, ()), stepid=(np.uint8, (20,)),
        **{'dyn/deter': (np.float32, (4,)), 'dyn/stoch': (np.float32, (2, 2))})
    self.train_mirrored = None
    self.init_calls = self.train_calls = self.drain_calls = 0
    self.seen_batches = []

  def _zeros(self, spaces, prefix):
    return {k: np.zeros((*prefix, *shape), dtype) for k, (dtype, shape) in spaces.items()}

  def load(self, checkpoint):
    self.params = copy.deepcopy(checkpoint['params'])
    saved = checkpoint['counters']
    self.n_updates = Counter(saved['updates'])
    self.n_batches = Counter(saved['updates'])
    self.n_actions = Counter(saved['actions'])
    self.n_eval_actions = Counter(saved['eval_actions'])
    self.eval_seed = saved['eval_seed']

  def save(self):
    return dict(params=copy.deepcopy(self.params), counters=learn._counters(self))

  def init_train(self, batch_size):
    assert batch_size == 16
    self.init_calls += 1
    return ('fixture-fresh-carry', self.init_calls)

  def _seeds(self, counter, sharding):
    return np.random.default_rng([7, counter]).integers(0, 2**32 - 1, 2, dtype=np.uint32)

  def prepare_batch(self, raw):
    counter = int(self.n_batches)
    self.n_batches.value += 1
    return {**raw, 'seed': self._seeds(counter, None)}

  def train(self, carry, prepared):
    assert carry[0] == 'fixture-fresh-carry'
    self.train_calls += 1
    prepared.pop('seed')
    self.seen_batches.append(learn.tree_digest(prepared))
    self.params['opt/step/value'] += np.int32(1)
    self.params['opt/momentum'] += np.float32(.01)
    self.n_updates.value += 1

  def take_train_result(self):
    self.drain_calls += 1
    return {}, {'opt/updates': np.int32(int(self.n_updates)), 'opt/grad_norm': np.float32(.4),
        'dl_reset/reset_count': np.int32(16), 'dl_reset/active_count': np.int32(8),
        'dl_reset/excess_sum': np.float32(3.5)}


def checkpoint():
  return dict(params={'enc/kernel': np.array([.1, .2], np.float32),
      'opt/step/value': np.array(11, np.int32), 'opt/momentum': np.array([.7], np.float32),
      'retnorm/lo': np.array(.3, np.float32), 'slowval_count': np.array(11, np.int32)},
      counters=dict(updates=11, batches=19, actions=300000, eval_actions=8, eval_seed=13))


class ResetLearningContracts(unittest.TestCase):
  def test_real_fields_terminal_placeholder_and_context_not_zero(self):
    cal, pins, contexts = inputs()
    agent = FixtureAgent()
    raw, receipt = learn.offline_batch(agent, cal, pins, contexts)
    self.assertEqual(raw['image'].shape, (16, 65, 64, 64, 3))
    self.assertEqual(receipt['balanced_duplicates_per_episode'], 2)
    self.assertTrue(receipt['offline_episode_concatenation'])
    self.assertFalse(receipt['original_continuous_replay_restored'])
    self.assertTrue(raw['is_first'][:, :2].all())
    self.assertFalse(raw['is_first'][:, 2:].any())
    self.assertTrue(raw['is_last'][:, 0].all())
    self.assertFalse(raw['is_last'][:, 1:].any())
    for index, episode in enumerate(learn.EPISODES * 2):
      previous = learn.EPISODES[(learn.EPISODES.index(episode) - 1) % 8]
      np.testing.assert_array_equal(raw['dyn/deter'][index, 0], contexts[previous]['dyn/deter'])
      np.testing.assert_array_equal(raw['action'][index, 0], np.zeros(12, np.float32))
      np.testing.assert_array_equal(raw['action'][index, 1:], cal[episode]['action'][:64])
      np.testing.assert_array_equal(raw['reward'][index, 1:], cal[episode]['reward'][:64])
      self.assertFalse(cal[previous]['is_first'][-1])
      np.testing.assert_array_equal(raw['stepid'][index], raw['stepid'][index % 8])

  def test_reject_bad_source_flags_executed_placeholder_and_dummy_context(self):
    mutations = (
        lambda c, x: c[9]['action_executed'].__setitem__(-1, True),
        lambda c, x: c[9]['action'].__setitem__((-1, 0), .1),
        lambda c, x: c[9]['is_first'].__setitem__(2, True),
        lambda c, x: c[9]['is_last'].__setitem__(12, True),
        lambda c, x: c[9]['is_terminal'].__setitem__(12, True),
        lambda c, x: x[9]['dyn/stoch'].fill(0),
        lambda c, x: x[9].pop('dyn/deter'),
    )
    for mutation in mutations:
      with self.subTest(mutation=mutation):
        cal, pins, contexts = inputs()
        mutation(cal, contexts)
        with self.assertRaises(ValueError):
          learn.offline_batch(FixtureAgent(), cal, pins, contexts)
    cal, pins, contexts = inputs()
    pins[9] = 'unverified'
    with self.assertRaises(ValueError):
      learn.offline_batch(FixtureAgent(), cal, pins, contexts)

  def test_exact_128_standard_restore_private_seeds_and_readonly_callbacks(self):
    cal, pins, contexts = inputs()
    initial, agent = checkpoint(), FixtureAgent()
    initial_digest = learn.tree_digest(initial)
    stages = []
    def feature_callback(current, stage):
      stages.append(stage)
      return {'h': np.full((8, 4), stage, np.float32)}
    result = learn.run_branch(agent, initial, cal, pins, contexts, 'B', feature_callback)
    self.assertEqual(stages, [0, 128])
    self.assertEqual((agent.init_calls, agent.train_calls, agent.drain_calls), (128, 128, 128))
    self.assertEqual(result['restored_counters']['batches'], 11)
    self.assertEqual(result['saved_source_counters']['batches'], 19)
    self.assertEqual(result['optimizer_counter'], dict(key='opt/step/value',
        dtype='int32', shape=[], initial=11, final=139, updates=128))
    self.assertEqual(result['final_checkpoint']['counters'],
        dict(updates=139, batches=139, actions=300000, eval_actions=8, eval_seed=13))
    self.assertEqual(result['total_loss_positions'], 131072)
    self.assertEqual(result['total_reset_loss_positions'], 2048)
    self.assertEqual(result['reset_pressure']['active_count'], 1024)
    self.assertEqual(result['reset_pressure']['excess_sum'], 448.)
    self.assertEqual(len(set(agent.seen_batches)), 1)
    self.assertEqual([r['seed_counter'] for r in result['receipts']], list(range(11, 139)))
    for row in result['receipts']:
      self.assertEqual(row['train_seed'], agent._seeds(row['seed_counter'], None).astype(int).tolist())
    self.assertEqual(learn.tree_digest(initial), initial_digest)
    np.testing.assert_array_equal(agent.params['retnorm/lo'], initial['params']['retnorm/lo'])
    self.assertGreater(agent.params['opt/momentum'][0], .7)
    self.assertFalse(result['method_effectiveness_established'])
    self.assertFalse(result['source_authentication_performed'])

  def test_callback_params_or_counter_mutation_fails_before_any_update(self):
    for kind in ('params', 'counter'):
      cal, pins, contexts = inputs()
      agent = FixtureAgent()
      def mutate(current, stage):
        if kind == 'params':
          current.params['retnorm/lo'] += 1
        else:
          current.n_eval_actions.value += 1
      with self.assertRaisesRegex(ValueError, 'callback modified'):
        learn.run_branch(agent, checkpoint(), cal, pins, contexts, 'B', mutate)
      self.assertEqual(agent.train_calls, 0)

  def test_nonfinite_metrics_wrong_seed_and_unexpected_counters_fail(self):
    for issue in ('metric', 'seed', 'action', 'missing_pressure', 'fractional_pressure'):
      cal, pins, contexts = inputs()
      agent = FixtureAgent()
      if issue == 'metric':
        agent.take_train_result = lambda: ({}, {'bad': np.float32(np.nan)})
      elif issue == 'seed':
        original = agent.prepare_batch
        def prepare(raw):
          data = original(raw)
          data['seed'] = np.zeros(2, np.uint32)
          return data
        agent.prepare_batch = prepare
      elif issue in ('missing_pressure', 'fractional_pressure'):
        original = agent.take_train_result
        def take():
          outs, metrics = original()
          if issue == 'missing_pressure':
            metrics.pop('dl_reset/reset_count')
          else:
            metrics['dl_reset/reset_count'] = np.float32(15.5)
          return outs, metrics
        agent.take_train_result = take
      else:
        original = agent.train
        def train(carry, data):
          original(carry, data)
          agent.n_actions.value += 1
        agent.train = train
      with self.subTest(issue=issue), self.assertRaises(ValueError):
        learn.run_branch(agent, checkpoint(), cal, pins, contexts, 'B')
      self.assertLessEqual(agent.train_calls, 1)

  def test_branch_patch_must_precede_agent_construction_and_full_state_required(self):
    cal, pins, contexts = inputs()
    with self.assertRaisesRegex(ValueError, 'patch'):
      learn.run_branch(FixtureAgent(), checkpoint(), cal, pins, contexts, 'R')
    original = FixtureDyn.loss
    with learn.rssm_branch('R', FixtureDyn):
      agent = FixtureAgent()
      result = learn.run_branch(agent, checkpoint(), cal, pins, contexts, 'R')
      self.assertTrue(result['reset_rep_gradient_stopped'])
      with self.assertRaisesRegex(ValueError, 'Nested'):
        with learn.rssm_branch('R', FixtureDyn):
          pass
    self.assertIs(FixtureDyn.loss, original)
    constructed_too_early = FixtureAgent()
    with learn.rssm_branch('R', FixtureDyn), self.assertRaisesRegex(ValueError, 'precede construction'):
      learn.run_branch(constructed_too_early, checkpoint(), cal, pins, contexts, 'R')
    bad = checkpoint()
    bad.pop('counters')
    with self.assertRaises(ValueError):
      learn.run_branch(FixtureAgent(), bad, cal, pins, contexts, 'B')

  def test_signed_zero_is_part_of_state_identity(self):
    self.assertNotEqual(learn.tree_digest(np.array([0.], np.float32)),
        learn.tree_digest(np.array([-0.], np.float32)))

  def test_optimizer_variable_leaf_and_scalar_format_guards(self):
    self.assertEqual(learn.optimizer_step({'opt/step/value': np.array(74494, np.int32)}), 74494)
    for params in ({'opt/step': np.array(74494, np.int32)}, {},
        {'opt/step/value': np.array([74494], np.int32)},
        {'opt/step/value': np.array(74494., np.float32)},
        {'opt/step/value': np.array(74494, np.int64)},
        {'opt/step/value': np.array(-1, np.int32)},
        {'opt/step/value': np.array(np.nan, np.float32)},
        {'opt/step/value': np.array(np.inf, np.float32)},
        {'opt/step/value': np.array(1.5, np.float32)},
        {'opt/step/value': np.array(True)}):
      with self.subTest(params=params), self.assertRaises(ValueError):
        learn.optimizer_step(params)

  def test_optimizer_wrong_increment_fails_with_unchanged_agent_counters(self):
    cal, pins, contexts = inputs()
    agent = FixtureAgent()
    train = agent.train
    def skip_optimizer_step(carry, data):
      train(carry, data)
      agent.params['opt/step/value'] -= np.int32(1)
    agent.train = skip_optimizer_step
    with self.assertRaisesRegex(ValueError, 'exactly 128'):
      learn.run_branch(agent, checkpoint(), cal, pins, contexts, 'B')
    self.assertEqual(agent.train_calls, 128)

  def test_true_bfloat16_identity_supported_without_accepting_arbitrary_void(self):
    import ml_dtypes
    values = np.array([0., -0., 1.], dtype=ml_dtypes.bfloat16)
    self.assertEqual(learn.tree_digest(values), learn.tree_digest(values.copy()))
    with self.assertRaisesRegex(ValueError, 'void/object'):
      learn.tree_digest(values.view('V2'))
    with self.assertRaisesRegex(ValueError, 'Nonfinite'):
      learn.tree_digest(np.array([np.nan], dtype=ml_dtypes.bfloat16))

  def test_cpu_jax_mask_forward_other_gradient_and_wrapper_contract(self):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--jax-fixture'],
        cwd=root, capture_output=True, text=True, timeout=60,
        env={**os.environ, 'PYTHONPATH': str(root), 'CUDA_VISIBLE_DEVICES': '', 'JAX_PLATFORMS': 'cpu'})
    self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-3000:])
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    self.assertTrue(receipt['rep_forward_exact'])
    self.assertTrue(receipt['nonrep_gradient_exact'])
    self.assertTrue(receipt['reset_rep_gradient_zero'])
    self.assertTrue(receipt['official_tiny_rssm_B_all_parameter_gradients_exact'])
    self.assertTrue(receipt['official_tiny_optimizer_variable_path_and_int32_exact'])
    self.assertFalse(receipt['actual_source_learning'])


def jax_fixture():
  dll_directory = (os.add_dll_directory(os.environ['DL_DLL_DIRECTORY'])
      if os.name == 'nt' and os.environ.get('DL_DLL_DIRECTORY') else None)
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
  import jax
  import jax.numpy as jnp
  from dreamerv3.rssm import RSSM
  import elements
  original = RSSM.loss
  # Fixture uses the production patch/official RSSM signature, replacing only
  # its body with known quadratic per-position terms so an independent oracle
  # can check free-nats-active/inactive gradients and all untouched outputs.
  def body(self, carry, tokens, acts, reset, training, rep_probe=False):
    return carry, {'entry': tokens}, {'rep': jnp.maximum(tokens ** 2, 1.),
        'dyn': acts ** 2}, tokens, {'fixture': acts}, {'fixture': tokens}
  RSSM.loss = body
  try:
    x = jnp.array([[2., .5, 3.], [4., 2., .25]], jnp.float32)
    y = jnp.full_like(x, .7)
    mask = jnp.array([[True, False, False], [True, False, False]])
    def values(xx, yy):
      return RSSM.loss(types.SimpleNamespace(free_nats=1), (), xx, yy, mask, True, True)[2]
    baseline = values(x, y)
    base_rep = jax.grad(lambda xx: values(xx, y)['rep'].sum())(x)
    base_other = jax.grad(lambda yy: values(x, yy)['dyn'].sum())(y)
    with learn.rssm_branch('B'):
      observed = values(x, y)
      np.testing.assert_array_equal(np.asarray(observed['rep']), np.asarray(baseline['rep']))
      np.testing.assert_array_equal(np.asarray(jax.grad(lambda xx: values(xx, y)['rep'].sum())(x)),
          np.asarray(base_rep))
      np.testing.assert_array_equal(np.asarray(jax.grad(lambda yy: values(x, yy)['dyn'].sum())(y)),
          np.asarray(base_other))
      metrics_b = RSSM.loss(types.SimpleNamespace(free_nats=1), (), x, y, mask, True)[4]
    with learn.rssm_branch('R'):
      instance = RSSM({'action': elements.Space(np.float32, (2,))}, deter=8,
          hidden=8, stoch=2, classes=2, blocks=2, name='tiny_construction_fixture')
      assert instance._dl_reset_rep_stop_at_construction
      protected = values(x, y)
      rep_grad = jax.grad(lambda xx: values(xx, y)['rep'].sum())(x)
      other_grad = jax.grad(lambda yy: values(x, yy)['dyn'].sum())(y)
      np.testing.assert_array_equal(np.asarray(protected['rep']), np.asarray(baseline['rep']))
      np.testing.assert_array_equal(np.asarray(protected['dyn']), np.asarray(baseline['dyn']))
      np.testing.assert_array_equal(np.asarray(other_grad), np.asarray(base_other))
      np.testing.assert_array_equal(np.asarray(rep_grad), np.where(np.asarray(mask), 0, np.asarray(base_rep)))
      np.testing.assert_array_equal(np.asarray(rep_grad)[np.asarray(mask)], [0., 0.])
      metrics_r = RSSM.loss(types.SimpleNamespace(free_nats=1), (), x, y, mask, True)[4]
      for name in ('active_count', 'excess_sum', 'reset_count'):
        np.testing.assert_array_equal(np.asarray(metrics_r['dl_reset/' + name]),
            np.asarray(metrics_b['dl_reset/' + name]))
      assert int(metrics_r['dl_reset/active_count']) == 2
      assert float(metrics_r['dl_reset/excess_sum']) == 18.
      assert int(metrics_r['dl_reset/reset_count']) == 2
      off_grad = jax.grad(lambda xx: learn.stop_reset_rep(xx ** 2, mask, False).sum())(x)
      np.testing.assert_array_equal(np.asarray(off_grad), np.asarray(2 * x))
    assert RSSM.loss is body
  finally:
    RSSM.loss = original
  # Independently compare the original official tiny RSSM against the B
  # observation wrapper, including every actually accessed parameter gradient.
  # This is CPU engineering evidence, not a real-data optimizer experiment.
  import ninjax as nj
  dyn = RSSM({'action': elements.Space(np.float32, (2,))}, deter=8,
      hidden=8, stoch=2, classes=2, blocks=2, obslayers=1, imglayers=1,
      dynlayers=1, norm='rms', act='silu', free_nats=1., name='official_tiny')
  carry = dyn.initial(2)
  tokens = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) / 24
  acts = {'action': jnp.full((2, 3, 2), .2, jnp.float32)}
  root = jnp.array([137, 251], jnp.uint32)
  def objective(ca, to, ac, re):
    out = dyn.loss(ca, to, ac, re, True, True)
    loss = out[2]['dyn'].mean() + .1 * out[2]['rep'].mean()
    loss += .01 * jnp.float32(jnp.square(out[3]['logit'])).mean()
    return loss, out
  state = nj.init(objective)({}, carry, tokens, acts, mask, seed=root)
  def gradient_function():
    return jax.jit(nj.pure(lambda ca, to, ac, re: nj.grad(
        objective, dyn, has_aux=True)(ca, to, ac, re)))
  original_state, original_out = gradient_function()(state, carry, tokens, acts, mask, seed=root)
  with learn.rssm_branch('B'):
    wrapped_state, wrapped_out = gradient_function()(state, carry, tokens, acts, mask, seed=root)
  jax.block_until_ready((original_out, wrapped_out))
  def exact(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for aa, bb in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
      aa, bb = np.asarray(aa), np.asarray(bb)
      assert aa.dtype == bb.dtype and aa.shape == bb.shape and aa.tobytes() == bb.tobytes()
  exact(original_state, wrapped_state)
  exact(original_out[:3], wrapped_out[:3])  # loss, full selected params, all gradients
  for index in (0, 1, 2, 3, 5):
    exact(original_out[3][index], wrapped_out[3][index])
  for name, value in original_out[3][4].items():
    exact(value, wrapped_out[3][4][name])
  assert len(original_out[2]) > 0
  # Use the actual official Optimizer and installed Ninjax Variable, rather
  # than reproducing their state path in a mock. No source/model data is read.
  from embodied.jax.opt import Optimizer
  import optax
  class TinyParameter(nj.Module):
    def __call__(self):
      return self.value('weight', jnp.array, [1.], jnp.float32)
  tiny = TinyParameter(name='tiny_optimizer_parameter')
  optimizer = Optimizer(tiny, optax.sgd(.1), name='opt')
  def optimize():
    return optimizer(lambda: jnp.square(tiny()).sum())
  actual_state = nj.init(optimize)({}, seed=root)
  assert 'opt/step/value' in actual_state and 'opt/step' not in actual_state
  initial_step = learn.optimizer_step(actual_state)
  actual_state, metrics = nj.pure(optimize)(actual_state, seed=root)
  jax.block_until_ready((actual_state, metrics))
  assert learn.optimizer_step(actual_state) == initial_step + 1
  assert np.asarray(metrics['opt/updates']).dtype == np.int32
  assert int(metrics['opt/updates']) == initial_step + 1
  print(json.dumps(dict(rep_forward_exact=True, nonrep_gradient_exact=True,
      reset_rep_gradient_zero=True, official_tiny_rssm_B_all_parameter_gradients_exact=True,
      official_tiny_optimizer_variable_path_and_int32_exact=True,
      actual_source_learning=False)))


if __name__ == '__main__':
  if sys.argv[1:] == ['--jax-fixture']:
    jax_fixture()
  else:
    unittest.main()
