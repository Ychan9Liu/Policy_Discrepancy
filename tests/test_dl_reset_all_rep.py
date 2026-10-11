"""CPU construction contracts for the adopted finite full-window A assay.

No actual source/checkpoint, GPU, simulator or real learning is used.
The sequencing mock establishes input/seed/restore guards; the separate
official tiny RSSM/Optimizer fixture establishes the numerical interfaces.
"""

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest

import numpy as np

from scripts import dl_reset_all_rep as draft
from scripts import dl_reset_learning as original


_fixture_spec = importlib.util.spec_from_file_location(
    'original_reset_learning_contract_fixture',
    Path(__file__).with_name('test_dl_reset_learning.py'))
fixture = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(fixture)


class AFixture(fixture.FixtureAgent):
  """Sequencing mock only; it makes no claim about optimizer learning."""

  def take_train_result(self):
    outs, metrics = super().take_train_result()
    metrics.update({'dl_allrep/active_count': np.int32(64),
        'dl_allrep/excess_sum': np.float32(10.5),
        'dl_allrep/position_count': np.int32(1024),
        'dl_allrep/nonreset_active_count': np.int32(56),
        'dl_allrep/nonreset_excess_sum': np.float32(7.)})
    return outs, metrics


class AllRepConstructionContracts(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.cal, cls.pins, cls.contexts = fixture.inputs()
    cls.initial = fixture.checkpoint()
    cls.reference_b = original.run_branch(fixture.FixtureAgent(), cls.initial,
        cls.cal, cls.pins, cls.contexts, 'B')

  def run_a(self, agent, reference_b=None, callback=None):
    return draft.run_a(agent, self.initial, self.cal, self.pins, self.contexts,
        callback, reference_b=(self.reference_b if reference_b is None else reference_b))

  def test_128_common_original_batch_keys_state_and_optimizer_contract(self):
    initial_digest = original.tree_digest(self.initial)
    stages = []
    def features(agent, stage):
      stages.append(stage)
      return {'fixture_only': np.array(stage, np.int32)}
    with draft.allrep_branch(fixture.FixtureDyn):
      agent = AFixture()
      result = self.run_a(agent, callback=features)
    self.assertEqual((agent.init_calls, agent.train_calls, agent.drain_calls), (128, 128, 128))
    self.assertEqual(stages, [0, 128])
    for name in ('binding', 'source_checkpoint_digest', 'initial_state_digest',
        'initial_params_digest', 'calibration_arrays_digest', 'restored_counters',
        'saved_source_counters', 'optimizer_counter'):
      self.assertEqual(result[name], self.reference_b[name])
    self.assertEqual(result['optimizer_counter']['key'], 'opt/step/value')
    self.assertEqual(result['optimizer_counter']['initial'], 11)
    self.assertEqual(result['optimizer_counter']['final'], 139)
    self.assertEqual([r['train_seed'] for r in result['receipts']],
        [r['train_seed'] for r in self.reference_b['receipts']])
    self.assertEqual([r['counters'] for r in result['receipts']],
        [r['counters'] for r in self.reference_b['receipts']])
    self.assertEqual(len(set(agent.seen_batches)), 1)
    self.assertEqual(result['whole_window_pressure']['active_count'], 8192)
    self.assertEqual(result['whole_window_pressure']['nonreset_active_count'], 7168)
    self.assertEqual(result['whole_window_pressure']['position_count'], 131072)
    self.assertEqual(result['whole_window_pressure']['excess_sum'], 1344.)
    self.assertEqual(result['reset_pressure']['reset_count'], 2048)
    self.assertEqual(result['reset_pressure']['excess_sum'], 448.)
    self.assertEqual(original.tree_digest(self.initial), initial_digest)
    self.assertTrue(result['entire_window_rep_gradient_stopped'])
    self.assertTrue(result['original_B_receipt_values_matched'])
    self.assertEqual(result['version'], 'DL-reset-allrep-learning-r1')
    self.assertEqual(result['layout_version'], 'DL-reset-learning-r1')
    self.assertTrue(result['protocol_adopted'])
    self.assertFalse(result['working_draft_not_adopted'])
    for name in ('source_authentication_performed', 'original_B_comparison_authenticated',
        'native_actor_evaluation_performed', 'method_effectiveness_established'):
      self.assertFalse(result[name])

  def test_wrong_original_b_input_seed_and_counter_reference_rejected(self):
    for issue in ('state', 'batch', 'optimizer', 'seed', 'counter'):
      with self.subTest(issue=issue), draft.allrep_branch(fixture.FixtureDyn):
        agent = AFixture()
        reference = copy.deepcopy(self.reference_b)
        if issue == 'state':
          reference['initial_state_digest'] = '0' * 64
        elif issue == 'batch':
          reference['binding']['batch_digest'] = '0' * 64
        elif issue == 'optimizer':
          reference['optimizer_counter']['initial'] += 1
        elif issue == 'seed':
          reference['receipts'][0]['train_seed'] = [0, 0]
        else:
          reference['receipts'][0]['counters']['actions'] += 1
        with self.assertRaises(ValueError):
          self.run_a(agent, reference)
        self.assertEqual(agent.train_calls, 1 if issue == 'counter' else 0)

  def test_missing_nonfinite_fractional_pressure_and_optimizer_metrics_rejected(self):
    for issue in ('missing', 'nonfinite', 'fractional', 'count_partition', 'opt_dtype'):
      with self.subTest(issue=issue), draft.allrep_branch(fixture.FixtureDyn):
        agent = AFixture()
        take_original = agent.take_train_result
        def take():
          outs, metrics = take_original()
          if issue == 'missing':
            metrics.pop('dl_allrep/position_count')
          elif issue == 'nonfinite':
            metrics['dl_allrep/nonreset_excess_sum'] = np.float32(np.nan)
          elif issue == 'fractional':
            metrics['dl_allrep/active_count'] = np.float32(63.5)
          elif issue == 'count_partition':
            metrics['dl_allrep/nonreset_active_count'] = np.int32(55)
          else:
            metrics['opt/updates'] = np.float32(metrics['opt/updates'])
          return outs, metrics
        agent.take_train_result = take
        with self.assertRaises(ValueError):
          self.run_a(agent)
        self.assertEqual(agent.train_calls, 1)

  def test_callback_state_mutation_rejected_before_first_update(self):
    with draft.allrep_branch(fixture.FixtureDyn):
      agent = AFixture()
      def mutate(current, stage):
        current.n_eval_actions.value += 1
      with self.assertRaisesRegex(ValueError, 'callback modified'):
        self.run_a(agent, callback=mutate)
      self.assertEqual(agent.train_calls, 0)

  def test_constructor_ownership_late_and_nested_patches_rejected_and_restored(self):
    old_loss, old_init = fixture.FixtureDyn.loss, fixture.FixtureDyn.__init__
    constructed_early = AFixture()
    with draft.allrep_branch(fixture.FixtureDyn):
      with self.assertRaisesRegex(ValueError, 'precede Agent construction'):
        draft._assert_a_agent(constructed_early)
      with self.assertRaisesRegex(ValueError, 'Nested'):
        with draft.allrep_branch(fixture.FixtureDyn):
          pass
    self.assertIs(fixture.FixtureDyn.loss, old_loss)
    self.assertIs(fixture.FixtureDyn.__init__, old_init)
    with original.rssm_branch('R', fixture.FixtureDyn):
      with self.assertRaisesRegex(ValueError, 'Nested'):
        with draft.allrep_branch(fixture.FixtureDyn):
          pass
    with draft.allrep_branch(fixture.FixtureDyn):
      constructed_previous = AFixture()
    with draft.allrep_branch(fixture.FixtureDyn):
      with self.assertRaisesRegex(ValueError, 'exact construction-time wrapper'):
        draft._assert_a_agent(constructed_previous)
      with original.rssm_branch('B', fixture.FixtureDyn):
        with self.assertRaisesRegex(ValueError, 'Independent A wrapper'):
          draft._assert_a_agent(AFixture())

  def test_cpu_official_tiny_forward_gradients_single_call_and_optimizer(self):
    root = Path(__file__).resolve().parents[1]
    outcome = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--jax-fixture'],
        cwd=root, capture_output=True, text=True, timeout=60,
        env={**os.environ, 'PYTHONPATH': str(root), 'CUDA_VISIBLE_DEVICES': '', 'JAX_PLATFORMS': 'cpu'})
    self.assertEqual(outcome.returncode, 0, outcome.stdout[-2000:] + outcome.stderr[-3000:])
    result = json.loads(outcome.stdout.strip().splitlines()[-1])
    for name in ('original_loss_called_once', 'entire_rep_gradient_zero',
        'all_forward_outputs_preserved', 'nonrep_official_tiny_rssm_parameter_gradients_byteexact',
        'official_tiny_optimizer_int32_variable_leaf_exact'):
      self.assertTrue(result[name])
    self.assertEqual(result['devices'], ['cpu'])
    self.assertFalse(result['actual_source_learning'])


def jax_fixture():
  dll_directory = (os.add_dll_directory(os.environ['DL_DLL_DIRECTORY'])
      if os.name == 'nt' and os.environ.get('DL_DLL_DIRECTORY') else None)
  import jax
  import jax.numpy as jnp
  import ninjax as nj
  import elements
  import optax
  from dreamerv3.rssm import RSSM
  from embodied.jax.opt import Optimizer
  assert all(device.platform == 'cpu' for device in jax.devices())
  loss_original = RSSM.loss
  calls = [0]
  # An independent quadratic oracle includes both active and free-nats-flat
  # rep entries at reset and nonreset positions. It is a numerical fixture.
  def body(self, carry, tokens, acts, reset, training, rep_probe=False):
    calls[0] += 1
    return carry, {'entry': tokens}, {'rep': jnp.maximum(tokens ** 2, 1.),
        'dyn': acts ** 2}, tokens, {'fixture': acts}, {'fixture': tokens}
  RSSM.loss = body
  x = jnp.array([[2., .5, 3.], [4., 2., .25]], jnp.float32)
  y = jnp.full_like(x, .7)
  mask = jnp.array([[True, False, False], [True, False, False]])
  def exact(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for aa, bb in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
      aa, bb = np.asarray(aa), np.asarray(bb)
      assert aa.dtype == bb.dtype and aa.shape == bb.shape and aa.tobytes() == bb.tobytes()
  try:
    def values(xx, yy, training=True):
      return RSSM.loss(types.SimpleNamespace(free_nats=1), (), xx, yy, mask, training, True)
    baseline = values(x, y)
    other_gradient = jax.grad(lambda yy: values(x, yy)[2]['dyn'].sum())(y)
    with draft.allrep_branch():
      instance = RSSM({'action': elements.Space(np.float32, (2,))}, deter=8,
          hidden=8, stoch=2, classes=2, blocks=2, name='tiny_draft_construction')
      assert instance._dl_allrep_loss_identity_at_construction == id(RSSM.loss)
      calls[0] = 0
      protected = values(x, y)
      assert calls[0] == 1
      for index in (0, 1, 2, 3, 5):
        exact(protected[index], baseline[index])
      exact(jax.grad(lambda xx: values(xx, y)[2]['rep'].sum())(x), np.zeros((2, 3), np.float32))
      exact(jax.grad(lambda yy: values(x, yy)[2]['dyn'].sum())(y), other_gradient)
      exact(jax.grad(lambda xx: values(xx, y, False)[2]['rep'].sum())(x),
          np.where(np.asarray(x) ** 2 > 1, 2 * np.asarray(x), 0))
      assert int(protected[4]['dl_allrep/active_count']) == 4
      assert float(protected[4]['dl_allrep/excess_sum']) == 29.
      assert int(protected[4]['dl_reset/active_count']) == 2
      assert float(protected[4]['dl_reset/excess_sum']) == 18.
    assert RSSM.loss is body
  finally:
    RSSM.loss = loss_original
  # Compare official RSSM full accessed parameter gradients for an untouched
  # dyn/features loss and every original forward output, with identical keys.
  root = jnp.array([137, 251], jnp.uint32)
  dyn = RSSM({'action': elements.Space(np.float32, (2,))}, deter=8,
      hidden=8, stoch=2, classes=2, blocks=2, obslayers=1, imglayers=1,
      dynlayers=1, norm='rms', act='silu', free_nats=1., name='official_tiny_draft')
  carry = dyn.initial(2)
  tokens = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) / 24
  acts = {'action': jnp.full((2, 3, 2), .2, jnp.float32)}
  def nonrep(ca, to, ac, re):
    out = dyn.loss(ca, to, ac, re, True, True)
    return out[2]['dyn'].mean() + .01 * jnp.float32(jnp.square(out[3]['logit'])).mean(), out
  def rep(ca, to, ac, re):
    out = dyn.loss(ca, to, ac, re, True, True)
    return out[2]['rep'].mean(), out
  state = nj.init(nonrep)({}, carry, tokens, acts, mask, seed=root)
  def gradient_function(objective):
    return jax.jit(nj.pure(lambda ca, to, ac, re: nj.grad(
        objective, dyn, has_aux=True)(ca, to, ac, re)))
  original_state, original_out = gradient_function(nonrep)(state, carry, tokens, acts, mask, seed=root)
  with draft.allrep_branch():
    a_state, a_out = gradient_function(nonrep)(state, carry, tokens, acts, mask, seed=root)
    _, rep_out = gradient_function(rep)(state, carry, tokens, acts, mask, seed=root)
  jax.block_until_ready((original_out, a_out, rep_out))
  exact(original_state, a_state)
  exact(original_out[:3], a_out[:3])
  for index in (0, 1, 2, 3, 5):
    exact(original_out[3][index], a_out[3][index])
  for name, value in original_out[3][4].items():
    exact(value, a_out[3][4][name])
  assert len(rep_out[2]) > 0
  for gradient in rep_out[2].values():
    assert np.all(np.asarray(gradient) == 0)
  # Exercise the installed official Optimizer/Ninjax state path and one real
  # tiny numerical update, without constructing a full Agent or reading data.
  class TinyParameter(nj.Module):
    def __call__(self):
      return self.value('weight', jnp.array, [1.], jnp.float32)
  tiny = TinyParameter(name='tiny_optimizer_draft')
  optimizer = Optimizer(tiny, optax.sgd(.1), name='opt')
  def optimize():
    return optimizer(lambda: jnp.square(tiny()).sum())
  optimizer_state = nj.init(optimize)({}, seed=root)
  assert 'opt/step/value' in optimizer_state and 'opt/step' not in optimizer_state
  initial_step = original.optimizer_step(optimizer_state)
  optimizer_state, metrics = nj.pure(optimize)(optimizer_state, seed=root)
  jax.block_until_ready((optimizer_state, metrics))
  assert original.optimizer_step(optimizer_state) == initial_step + 1
  assert np.asarray(metrics['opt/updates']).dtype == np.int32
  assert int(metrics['opt/updates']) == initial_step + 1
  print(json.dumps(dict(original_loss_called_once=True, entire_rep_gradient_zero=True,
      all_forward_outputs_preserved=True, nonrep_official_tiny_rssm_parameter_gradients_byteexact=True,
      official_tiny_optimizer_int32_variable_leaf_exact=True,
      devices=[device.platform for device in jax.devices()], actual_source_learning=False)))


if __name__ == '__main__':
  if sys.argv[1:] == ['--jax-fixture']:
    jax_fixture()
  else:
    unittest.main()
