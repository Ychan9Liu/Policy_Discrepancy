import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_native_actor as n


def obs(step=0, early=False):
  return dict(image=np.full((64, 64, 3), step % 256, np.uint8),
      reward=np.float32(0 if step == 0 else .25), is_first=np.bool_(step == 0),
      is_last=np.bool_(early or step == n.HORIZON), is_terminal=np.bool_(False))


class FakeActor:
  """Numerical construction fixture; never a model or scientific receipt."""
  initial_state_sha256 = 'a' * 64
  def assert_frozen(self, full=False): pass
  def initial(self, seed): return {'cursor': np.array(0, np.int32)}
  def step(self, carry, observation, seed, step):
    after = {'cursor': np.array(step + 1, np.int32)}
    key = n.private_root_key(seed, step)
    return after, np.full(12, .1, np.float32), dict(root_key=key,
        latent_key=key.copy(), actor_key=key.copy(), rng_consumption=np.int32(2),
        before_carry_sha256=n.tree_digest(carry), after_carry_sha256=n.tree_digest(after))


class FakeTransition:
  def __init__(self, change=None): self.step = 0; self.change = change
  def __call__(self, raw):
    self.step += 1
    row = dict(observation=obs(self.step), action_applied=raw.astype(np.float64),
        reward_f64=np.float64(.25), discount_f64=np.float64(1))
    if self.change: self.change(self, raw, row)
    return row


class NativeActorTest(unittest.TestCase):
  def test_import_has_no_loader_cli_device_or_env_side_effects(self):
    code = "import os,sys; before=dict(os.environ); import scripts.dl_native_actor; assert before==dict(os.environ); assert 'jax' not in sys.modules; assert 'dreamerv3.agent' not in sys.modules; assert not any(x.startswith('dm_control') for x in sys.modules)"
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
        capture_output=True, text=True, timeout=30,
        env=dict(os.environ, PYTHONPATH=str(ROOT)))
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_named_roots_match_official_seed0_eval(self):
    values = []
    for seed in n.SEEDS:
      for step in (0, 500, 999):
        expected = np.random.default_rng([0, 0x4556414C, seed, step]).integers(
            0, np.iinfo(np.uint32).max, (2,), np.uint32)
        np.testing.assert_array_equal(n.private_root_key(seed, step), expected)
        values.append(expected.tobytes())
    self.assertEqual(len(set(values)), 96)
    for seed, step in ((1023, 0), (1056, 0), (1024, -1), (1024, 1000),
        (True, 0), (np.int64(1024), 0), (1024, 0.), (1024, True)):
      with self.subTest(seed=seed, step=step), self.assertRaises(ValueError):
        n.private_root_key(seed, step)

  def test_observation_requires_exact_production_values(self):
    for change in ('image_dtype', 'image_shape', 'reward_dtype', 'nonfinite', 'flags', 'extra'):
      row = obs()
      if change == 'image_dtype': row['image'] = row['image'].astype(np.float32)
      if change == 'image_shape': row['image'] = row['image'][:-1]
      if change == 'reward_dtype': row['reward'] = np.float64(0)
      if change == 'nonfinite': row['reward'] = np.float32(np.nan)
      if change == 'flags': row['is_terminal'] = np.bool_(True)
      if change == 'extra': row['physics'] = np.float64(0)
      with self.subTest(change=change), self.assertRaises(ValueError): n.observation(row)

  def test_complete1000_episode_and_fixed_oracles(self):
    initial = obs(); before = n.tree_digest(initial); calls = []
    def oracle(**kwargs):
      calls.append(kwargs['step'])
      kwargs['raw'][0] = 100
      kwargs['observation']['image'][:] = 0
      kwargs['carry']['cursor'][...] = -10
      return {'synthetic_fixture_only': True}
    transition = FakeTransition()
    result = n.evaluate_episode(FakeActor(), 1024, initial, transition, oracle=oracle)
    self.assertEqual(calls, [0, 500, 999]); self.assertEqual(transition.step, 1000)
    self.assertEqual(n.tree_digest(initial), before)
    self.assertEqual(result['counts'], dict(transitions=1000, observations=1001,
        policy_calls=1000, resets=1, oracle_calls=3))
    self.assertEqual(result['return_f64'], 250.)
    self.assertEqual(result['raw'].shape, (1000, 12)); self.assertEqual(result['raw'].dtype, np.float32)
    self.assertEqual(result['applied'].dtype, np.float64)
    self.assertEqual(result['observations']['image'].shape, (1001, 64, 64, 3))
    self.assertEqual(len(result['carry_hashes']), 1001)
    self.assertEqual(result['root_keys'].shape, (1000, 2))
    self.assertTrue(result['observations']['is_last'][-1])
    self.assertFalse(result['observations']['is_terminal'].any())
    self.assertFalse(result['source_physics_renderer_resource_authenticated_by_this_helper'])

  def test_early_end_missing_last_reset_reward_and_typed_transition_fail(self):
    mutations = {
        'early_last': lambda tr, raw, row: row['observation'].update(is_last=np.bool_(True)),
        'reset': lambda tr, raw, row: row['observation'].update(is_first=np.bool_(True)),
        'reward_dtype': lambda tr, raw, row: row.update(reward_f64=np.float32(.25)),
        'wrong_reward': lambda tr, raw, row: row.update(reward_f64=np.float64(.5)),
        'discount': lambda tr, raw, row: row.update(discount_f64=np.float64(.5)),
        'applied_dtype': lambda tr, raw, row: row.update(action_applied=raw.copy()),
        'mutated_raw': lambda tr, raw, row: raw.__setitem__(0, 10),
        'extra': lambda tr, raw, row: row.update(extra=True),
        'missing_last': lambda tr, raw, row: row['observation'].update(is_last=np.bool_(False)),
    }
    for name, change in mutations.items():
      with self.subTest(name=name), self.assertRaises(ValueError):
        n.evaluate_episode(FakeActor(), 1024, obs(), FakeTransition(change))
    with self.assertRaises(ValueError):
      n.evaluate_episode(FakeActor(), 1024, {**obs(), 'reward': np.float32(1)}, FakeTransition())

  def test_deadline_failure_does_not_publish_partial_result(self):
    tr = FakeTransition()
    def check():
      if tr.step == 5: raise TimeoutError('Fixture whole deadline')
    with self.assertRaises(TimeoutError):
      n.evaluate_episode(FakeActor(), 1024, obs(), tr, check=check)
    self.assertEqual(tr.step, 5)

  def test_actual_tiny_official_policy_currentparams_and_guards(self):
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--fixture'],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
        env=dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT)))
    self.assertEqual(result.returncode, 0, result.stdout[-6500:] + result.stderr[-11000:])
    facts = json.loads(result.stdout.strip().splitlines()[-1])
    for name, value in facts.items(): self.assertTrue(value, name)


def fixture():
  dll = (os.add_dll_directory(os.environ['DL_DLL_DIRECTORY'])
      if os.name == 'nt' and os.environ.get('DL_DLL_DIRECTORY') else None)
  import elements
  import jax
  import jax.numpy as jnp
  import ninjax as nj
  from embodied.jax import internal
  from ruamel.yaml import YAML
  from scripts.dl_predictive_mc import agent_from_spaces
  from unittest import mock
  presets = YAML(typ='safe').load((ROOT / 'dreamerv3/configs.yaml').read_text(encoding='utf-8'))
  config = elements.Config(presets['defaults']).update(presets['debug'])
  config = elements.Config({**config.flat, **{'jax.platform': 'cpu',
      'jax.compute_dtype': 'bfloat16', 'jax.precompile': False, 'jax.profiler': False,
      'jax.enable_policy': True, 'jax.transfer_guard': True, 'batch_size': 2,
      'batch_length': 3, 'replay_context': 1, 'agent.imag_length': 2,
      'agent.imag_last': 2, 'agent.rep_probe.mode': 'off', 'agent.dt_latch.mode': 'off'}}).update(
          logdir='/tmp/dl-native-actor-fixture', seed=0)
  agent = agent_from_spaces(config)
  saved = n.full_digest(agent.save()); old_counters = n.counters(agent)
  actor = n.NativeActor(agent)
  carry = actor.initial(1024); initial = copy.deepcopy(carry)
  after, raw, proof = actor.step(carry, obs(), 1024, 0)
  assert n.tree_digest(carry) == n.tree_digest(initial)
  assert n.full_digest(agent.save()) == saved
  # Actual public eval policy uses its synchronized copy and identical roots.
  try:
    agent.set_eval_seed(1024)
    outer = agent.init_policy(1, mode='eval')
    public_after, public_actions, _ = agent.policy(outer,
        {k: v[None] for k, v in n.observation(obs()).items()}, mode='eval')
    public_raw = np.asarray(public_actions['action'][0])
    assert public_raw.dtype == raw.dtype and public_raw.tobytes() == raw.tobytes()
    public_after = n.host_copy(public_after)
    def stacked(tree):
      if isinstance(tree, dict): return {k: stacked(v) for k, v in tree.items()}
      if isinstance(tree, tuple): return tuple(stacked(v) for v in tree)
      if isinstance(tree, list): return np.stack(tree)
      return tree
    assert n.tree_digest(stacked(public_after)) == n.tree_digest(after)
  finally:
    agent.set_eval_seed(old_counters['eval_seed'])
    with agent.n_eval_actions.lock: agent.n_eval_actions.value = old_counters['eval_actions']
  assert n.full_digest(agent.save()) == saved
  expected_keys = n.host_copy(jax.jit(lambda k: jax.random.split(k, 16)[1:3])(
      internal.device_put(n.private_root_key(1024, 0), agent.train_mirrored)))
  assert proof['latent_key'].tobytes() == expected_keys[0].tobytes()
  assert proof['actor_key'].tobytes() == expected_keys[1].tobytes()
  agent.sync_policy()
  actual_oracle = actor.oracle(step=0, carry=copy.deepcopy(carry), observation=n.observation(obs()),
      root=n.private_root_key(1024, 0), after_carry=copy.deepcopy(after), raw=raw.copy(), proof=proof)
  assert actual_oracle['public_raw_byte_exact'] and actual_oracle['eval_counters_restored']
  assert n.full_digest(agent.save()) == saved and actor.receipt()['public_oracle_calls'] == 1
  for name, bad_value in (('policy_params', {}), ('pending_sync', {'pending': True})):
    with mock.patch.object(agent, name, bad_value):
      try:
        actor.oracle(step=0, carry=carry, observation=n.observation(obs()),
            root=n.private_root_key(1024, 0), after_carry=after, raw=raw, proof=proof)
      except ValueError: pass
      else: raise AssertionError('Stale/pending public policy copy accepted')
  def mode_action(carry):
    return agent.model.pol(agent.model.feat2tensor(carry[1]), 1)['action'].pred()
  placed_after = internal.device_put(after, agent.train_mirrored)
  _, mode = jax.jit(nj.pure(mode_action))(agent.params, placed_after,
      seed=internal.device_put(n.private_root_key(1024, 0), agent.train_mirrored),
      create=False, modify=False)
  assert n.host_copy(mode)[0].tobytes() != raw.tobytes()
  with mock.patch.multiple(agent, policy=mock.Mock(side_effect=AssertionError('public')),
      _seeds=mock.Mock(side_effect=AssertionError('public RNG')),
      _eval_seeds=mock.Mock(side_effect=AssertionError('public eval RNG')),
      policy_params={}):
    fresh = n.NativeActor(agent)
    again, again_raw, _ = fresh.step(fresh.initial(1024), obs(), 1024, 0)
    assert again_raw.tobytes() == raw.tobytes() and n.tree_digest(again) == n.tree_digest(after)
  # Legitimate posterior carry continues; no per-step initial/reset replacement.
  next_after, next_raw, _ = actor.step(after, obs(1), 1024, 1)
  assert next_raw.dtype == np.float32 and n.tree_digest(after) != n.tree_digest(next_after)
  with agent.n_eval_actions.lock: agent.n_eval_actions.value += 1
  try:
    try: actor.step(after, obs(1), 1024, 1)
    except AssertionError: pass
    else: raise AssertionError('Counter mutation accepted')
  finally:
    with agent.n_eval_actions.lock: agent.n_eval_actions.value -= 1
  original = agent.model.policy
  def writes(*args, **kwargs):
    name = next(iter(nj.context())); nj.context()[name] = nj.context()[name]
    return original(*args, **kwargs)
  def creates(*args, **kwargs):
    nj.context()['forbidden_native_variable'] = jnp.int32(0)
    return original(*args, **kwargs)
  def extra(*args, **kwargs):
    value = original(*args, **kwargs); nj.seed(); return value
  for replacement in (writes, creates, extra):
    agent.model.policy = replacement
    try:
      rejecting = n.NativeActor(agent)
      try: rejecting.step(rejecting.initial(1024), obs(), 1024, 0)
      except (AssertionError, RuntimeError): pass
      else: raise AssertionError('Forbidden policy trace accepted')
    finally: agent.model.policy = original
  actor.assert_frozen(full=True)
  print(json.dumps(dict(official_native_sample_raw_exact=True,
      official_after_carry_exact=True, actual_subkeys_exact=True,
      currentparams_stale_policycopy_ignored=True, public_rng_never_consumed=True,
      adapter_public_oracle_byteexact_and_counters_restored=True,
      stale_pending_oracle_copy_rejected=True, native_sample_not_mode=True,
      full_state_and_counters_unchanged=True, actual_bf16_carry_continues=True,
      attempted_write_create_extra_rng_rejected=True, counter_change_rejected=True,
      synthetic_only_not_real_model_acceptance=True)))
  if dll is not None: dll.close()


if __name__ == '__main__':
  if sys.argv[1:] == ['--fixture']: fixture()
  else: unittest.main()
