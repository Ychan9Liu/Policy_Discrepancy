"""Actual CPU MuJoCo/wrappers; policy and RGB are explicit engineering stubs."""
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('MUJOCO_GL', 'disable')

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_diagnostic as d
from scripts import dl_feedback_physics as f


def dummy_render(self, height=64, width=64, camera_id=2, **kwargs):
  # Depend on actual physics time, while never creating an EGL/GL context.
  value = int(round(self.data.time * 1000)) % 256
  return np.full((height, width, 3), value, np.uint8)


def initial_carry():
  return ({}, dict(deter=[np.zeros(4, np.float32)],
      stoch=[np.zeros((3, 4), np.float32)]), {},
      dict(action=[np.zeros(12, np.float32)]))


class FakePolicy:
  """Tests control flow/receipts, not actual neural policy or NJ key derivation."""
  def __init__(self):
    self.omits = []
    self.observation_hashes = []

  def step(self, carry, obs, root_key, omit=False):
    self.omits.append(omit)
    self.observation_hashes.append(f.source.tree_digest(obs))
    after = copy.deepcopy(carry)
    raw = np.linspace(-2.5, 2.5, 12, dtype=np.float32)
    after[1]['deter'][0] += np.float32(1)
    after[3]['action'][0] = raw.copy()
    proof = dict(root_key=root_key.copy(), latent_key=root_key.copy(),
        actor_key=root_key[::-1].copy(), selected_logit=np.zeros((3, 4), np.float32),
        h=after[1]['deter'][0].copy(), latent=after[1]['stoch'][0].copy(), raw=raw.copy(),
        rng_consumption=np.array(2, np.int32))
    return after, raw, proof


class FeedbackPhysicsTest(unittest.TestCase):
  def setUp(self):
    from dm_control.mujoco.engine import Physics
    self.render_patch = mock.patch.object(Physics, 'render', dummy_render)
    self.render_patch.start()
    self.adapter = f.FeedbackEnvironment(d.make_dmc('dmc_quadruped_walk', 77))
    self.obs = self.adapter.reset()
    self.snapshot = self.adapter.snapshot()
    self.addCleanup(self.adapter.close)
    self.addCleanup(self.render_patch.stop)

  def keys(self, horizon):
    return np.arange(horizon * 2, dtype=np.uint32).reshape(horizon, 2)

  def branch(self, horizon=10, prior=False, policy=None):
    obs, receipt = f.prepare_initial_observation(self.adapter, self.snapshot, self.obs)
    policy = policy or FakePolicy()
    record = f.run_branch(policy, self.adapter, self.snapshot, initial_carry(),
        self.keys(horizon), prior=prior, initial_obs=obs, horizon=horizon)
    return record, policy, receipt

  def test_actual_wrapper_branch_full_h100_and_reward_physics_replay(self):
    record, policy, receipt = self.branch(horizon=100, prior=True)
    self.assertEqual(record['executed_steps'], 100)
    self.assertEqual(len(record['snapshots']), 101)
    self.assertEqual(len(record['observations']), 101)
    self.assertEqual(policy.omits, [True] + [False] * 99)
    self.assertEqual(record['end_type'], 'window')
    self.assertEqual(receipt['fresh_render_count'], 1)
    self.assertEqual(receipt['maximum_state_error'], 0)
    self.assertEqual(record['reward_f64'].dtype, np.float64)
    self.assertEqual(record['reward_f32'].dtype, np.float32)
    self.assertTrue(np.any(np.abs(record['raw']) > 1))
    self.assertFalse(np.array_equal(record['raw'], record['applied']))
    # Quadruped's asymmetric actual bounds give zero raw lift an applied .05.
    self.assertNotEqual(float(record['applied'][0, 1]), float(record['raw'][0, 1]))
    self.assertGreater(len(set(policy.observation_hashes)), 1)
    with mock.patch.object(self.adapter.suite.physics, 'render',
        side_effect=AssertionError('physics replay must not render')):
      proof = f.verify_physics_replay(self.adapter, record)
    self.assertEqual(proof['physics_steps'], 100)
    self.assertEqual(proof['maximum_state_error'], 0)
    self.assertTrue(proof['reward_float64_byte_exact'])
    self.assertTrue(proof['reward_float32_byte_exact'])
    self.assertFalse(proof['policy_replayed'])
    self.assertFalse(proof['renderer_replayed'])

  def test_terminal_absorbs_without_policy_reset_or_fake_observations(self):
    cls = type(self.adapter.suite._task)
    with mock.patch.object(cls, 'get_termination', return_value=0.):
      record, policy, _ = self.branch(horizon=10, prior=True)
      self.assertEqual(record['executed_steps'], 1)
      self.assertEqual(policy.omits, [True])
      self.assertEqual(record['end_type'], 'true_terminal')
      self.assertEqual(len(record['observations']), 2)
      self.assertEqual(len(record['snapshots']), 2)
      self.assertEqual(len(record['proofs']), 1)
      self.assertEqual(record['action_executed'].tolist(), [True] + [False] * 9)
      self.assertEqual(record['observation_executed'].tolist(), [True, True] + [False] * 9)
      self.assertTrue(record['snapshots'][-1]['from_dm_done'])
      self.assertTrue(record['snapshots'][-1]['suite']['reset_next_step'])
      self.assertTrue(np.all(record['reward_f64'][1:] == 0))
      proof = f.verify_physics_replay(self.adapter, record)
      self.assertEqual(proof['physics_steps'], 1)
      self.assertEqual(proof['absorbing_padding_steps'], 9)
      invalid = copy.deepcopy(record)
      invalid['reward_f64'][-1] = .1
      with self.assertRaises(ValueError):
        f.validate_record(invalid)
      invalid = copy.deepcopy(record)
      invalid['observations'].append(copy.deepcopy(record['observations'][-1]))
      with self.assertRaises(ValueError):
        f.validate_record(invalid)
      invalid = copy.deepcopy(record)
      invalid['action_executed'][2] = True
      with self.assertRaises(ValueError):
        f.validate_record(invalid)

  def test_time_limit_only_final_action_allowed(self):
    self.adapter.suite._step_count = int(self.adapter.suite._step_limit) - 3
    self.snapshot = self.adapter.snapshot()
    record, _, _ = self.branch(horizon=3)
    self.assertEqual(record['end_type'], 'time_limit')
    self.assertFalse(record['is_terminal'][-1])
    self.assertEqual(f.verify_physics_replay(self.adapter, record)['maximum_state_error'], 0)
    self.adapter.restore(self.snapshot)
    self.adapter.suite._step_count = int(self.adapter.suite._step_limit) - 1
    self.snapshot = self.adapter.snapshot()
    with self.assertRaisesRegex(ValueError, 'Early or unexpected time limit'):
      self.branch(horizon=3)

  def test_f64_tamper_that_preserves_f32_is_rejected_byte_exact(self):
    record, _, _ = self.branch(horizon=3)
    damaged = copy.deepcopy(record)
    before = damaged['reward_f64'][0]
    damaged['reward_f64'][0] = np.nextafter(before, np.inf)
    self.assertEqual(np.float32(before).tobytes(), np.float32(damaged['reward_f64'][0]).tobytes())
    with self.assertRaisesRegex(AssertionError, 'F64/F32 reward'):
      f.verify_physics_replay(self.adapter, damaged)
    damaged = copy.deepcopy(record)
    damaged['applied'][0, 1] += .001
    with self.assertRaisesRegex(AssertionError, 'production mapping'):
      f.verify_physics_replay(self.adapter, damaged)
    damaged = copy.deepcopy(record)
    damaged['snapshots'][1]['suite']['integration'][0] += 1e-5
    with self.assertRaisesRegex(AssertionError, 'integration restoration'):
      f.verify_physics_replay(self.adapter, damaged)
    damaged = copy.deepcopy(record)
    damaged['snapshots'][1]['from_dm_done'] = True
    with self.assertRaises(AssertionError):
      f.verify_physics_replay(self.adapter, damaged)

  def test_nonfinite_invalid_dtype_mutation_keys_and_carry_refused(self):
    record, _, _ = self.branch(horizon=2)
    for path in ('reward', 'state', 'proof'):
      damaged = copy.deepcopy(record)
      if path == 'reward':
        damaged['reward_f64'][0] = np.nan
      elif path == 'state':
        damaged['snapshots'][0]['suite']['integration'][0] = np.inf
      else:
        damaged['proofs'][0]['h'][0] = np.nan
      with self.assertRaises((ValueError, AssertionError)):
        f.verify_physics_replay(self.adapter, damaged)
    damaged = copy.deepcopy(record)
    damaged['proofs'][0]['actor_key'] = np.zeros(2, np.int64)
    with self.assertRaises(ValueError):
      f.validate_record(damaged)
    damaged = copy.deepcopy(record)
    damaged['proofs'][0]['rng_consumption'] = np.array(1, np.int32)
    with self.assertRaises(AssertionError):
      f.validate_record(damaged)
    damaged = copy.deepcopy(record)
    damaged['final_carry'][3]['action'][0] = damaged['applied'][1].astype(np.float32)
    with self.assertRaises(AssertionError):
      f.validate_record(damaged)

    class MutatingPolicy(FakePolicy):
      def step(self, carry, obs, key, omit=False):
        carry[1]['deter'][0][0] = 42
        return super().step(carry, obs, key, omit)
    with self.assertRaisesRegex(AssertionError, 'mutated'):
      self.branch(horizon=2, policy=MutatingPolicy())

    class WrongDtypePolicy(FakePolicy):
      def step(self, *args, **kwargs):
        after, raw, proof = super().step(*args, **kwargs)
        return after, raw.astype(np.float64), proof
    with self.assertRaises(ValueError):
      self.branch(horizon=2, policy=WrongDtypePolicy())

    class NonfinitePolicy(FakePolicy):
      def step(self, *args, **kwargs):
        after, raw, proof = super().step(*args, **kwargs)
        raw[0] = np.nan
        return after, raw, proof
    with self.assertRaises(ValueError):
      self.branch(horizon=2, policy=NonfinitePolicy())

  def test_initial_render_once_preserves_source_scalar_fields_and_restore(self):
    original = copy.deepcopy(self.obs)
    original['image'][:] = 127
    before = f.source.tree_digest(original)
    with mock.patch.object(self.adapter.suite.physics, 'render',
        wraps=self.adapter.suite.physics.render) as render:
      fresh, proof = f.prepare_initial_observation(self.adapter, self.snapshot, original)
    self.assertEqual(render.call_count, 1)
    self.assertEqual(proof['maximum_pixel_difference'], 127)
    self.assertFalse(proof['renderer_byte_exact_claimed'])
    self.assertEqual(f.source.tree_digest(original), before)
    for key in f.OBS_KEYS[1:]:
      self.assertEqual(fresh[key].tobytes(), original[key].tobytes())
    self.assertEqual(f.source.state_error(self.adapter.snapshot(), self.snapshot), 0)
    bad = copy.deepcopy(self.snapshot)
    bad['from_dm_done'] = True
    with self.assertRaises(ValueError):
      f.prepare_initial_observation(self.adapter, bad, original)

  def test_source_array_index_to_root_observation_to_actual_prepare_contract(self):
    from scripts.dl_feedback import source_observation
    # Same array construction as a source trajectory, generated locally from
    # actual quad/production wrappers. No real source artifact is opened.
    current, _ = self.adapter.step(np.linspace(-1.5, 1.5, 12, dtype=np.float32))
    snapshot = self.adapter.snapshot()
    data = {key: np.stack([self.obs[key], current[key]]) for key in f.OBS_KEYS}
    original = {key: (value.dtype, value.shape, value.tobytes())
        for key, value in data.items()}
    # NumPy indexing returns scalars for the saved reward and flags. The r1
    # helper's scalar .copy() could not satisfy the explicit 0D-array boundary.
    self.assertIsInstance(data['reward'][1], np.float32)
    self.assertIsInstance(data['is_first'][1], np.bool_)
    indexed = {key: data[key][1].copy() for key in f.OBS_KEYS}
    with self.assertRaisesRegex(ValueError, 'production reward'):
      f.prepare_initial_observation(self.adapter, snapshot, indexed)
    observed = source_observation(data, 1)
    for key in f.OBS_KEYS:
      self.assertIsInstance(observed[key], np.ndarray)
      expected = np.asarray(data[key][1])
      self.assertEqual(observed[key].dtype, expected.dtype)
      self.assertEqual(observed[key].shape, expected.shape)
      self.assertEqual(observed[key].tobytes(), expected.tobytes())
      self.assertFalse(np.shares_memory(observed[key], data[key]))
    self.assertEqual(observed['reward'].shape, ())
    self.assertEqual(observed['reward'].dtype, np.float32)
    for key in f.OBS_KEYS[2:]:
      self.assertEqual(observed[key].shape, ())
      self.assertEqual(observed[key].dtype, np.bool_)
    fresh, proof = f.prepare_initial_observation(self.adapter, snapshot, observed)
    self.assertEqual(proof['fresh_render_count'], 1)
    self.assertEqual(proof['maximum_state_error'], 0)
    for key in f.OBS_KEYS[1:]:
      self.assertEqual(fresh[key].dtype, observed[key].dtype)
      self.assertEqual(fresh[key].shape, ())
      self.assertEqual(fresh[key].tobytes(), observed[key].tobytes())
    observed['image'][0, 0, 0] ^= np.uint8(1)
    observed['reward'][...] = np.float32(-1)
    for key, value in data.items():
      self.assertEqual((value.dtype, value.shape, value.tobytes()), original[key])
    self.assertEqual(f.source.state_error(self.adapter.snapshot(), snapshot), 0)


if __name__ == '__main__':
  unittest.main()
