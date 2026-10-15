"""Fresh CPU quad plus fake public Agent/stub RGB, never N1M/GPU acceptance.

The real wrapper controls, suite F64 reward, 1000-step clock and complete
physics snapshots are exercised. The fake Agent verifies the collector's
public counter/EVAL-key/carry bookkeeping; it is not a neural-policy oracle.
"""
import copy
import os
from pathlib import Path
import sys
import threading
import unittest
from unittest import mock

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MUJOCO_GL', 'disable')

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_diagnostic as d
from scripts import dl_feedback_model as model
from scripts import dl_feedback_physics as physics
from scripts import dl_r4_policy as keys
from scripts import dl_r4_utility as r


def dummy_render(self, height=64, width=64, camera_id=2, **kwargs):
  value = int(round(self.data.time * 1000)) % 256
  return np.full((height, width, 3), value, np.uint8)


class Counter:
  def __init__(self, value):
    self.value = value
    self.lock = threading.Lock()

  def __int__(self):
    return self.value


class FakePublicAgent:
  """A deterministic public-policy double, not the actual Dreamer model."""
  def __init__(self, *, skip_counter=False, fail_at=None, mutate_training=False):
    self.n_actions = Counter(71)
    self.n_batches = Counter(73)
    self.n_updates = Counter(79)
    self.n_eval_actions = Counter(17)
    self.eval_seed = 998877
    self.seed_calls = []
    self.calls = []
    self.skip_counter = skip_counter
    self.fail_at = fail_at
    self.mutate_training = mutate_training
    self.last_raw = np.zeros(12, np.float32)

  def set_eval_seed(self, seed):
    self.seed_calls.append(seed)
    self.eval_seed = int(seed)
    with self.n_eval_actions.lock:
      self.n_eval_actions.value = 0

  def init_policy(self, batch_size, mode):
    if batch_size != 1 or mode != 'eval':
      raise AssertionError('Fixture requires actual public eval B1 init')
    self.last_raw = np.zeros(12, np.float32)
    return ({}, dict(deter=[np.zeros(4, np.float32)],
        stoch=[np.eye(4, dtype=np.float32)[:3].copy()]), {},
        dict(action=[self.last_raw.copy()]))

  def policy(self, carry, obs, mode):
    if mode != 'eval' or set(obs) != set(physics.OBS_KEYS):
      raise AssertionError('Collector did not call public eval on original fields')
    for name, shape, dtype in (
        ('image', (1, 64, 64, 3), np.uint8), ('reward', (1,), np.float32),
        ('is_first', (1,), np.bool_), ('is_last', (1,), np.bool_),
        ('is_terminal', (1,), np.bool_)):
      value = obs[name]
      if value.shape != shape or value.dtype != np.dtype(dtype):
        raise AssertionError('Public source observation dtype/shape differs')
    step = int(self.n_eval_actions)
    if self.fail_at is not None and len(self.calls) == self.fail_at:
      raise RuntimeError('injected public policy failure')
    if carry[3]['action'][0].tobytes() != self.last_raw.tobytes():
      raise AssertionError('Public previous raw action was replaced by applied action')
    root = np.random.default_rng([0, 0x4556414C, self.eval_seed, step]).integers(
        0, np.iinfo(np.uint32).max, (2,), np.uint32)
    self.calls.append(dict(counter=step, eval_seed=self.eval_seed, root=root.copy(),
        pre_carry_sha256=model.tree_digest(carry), observation=copy.deepcopy(obs)))
    raw = np.random.default_rng(root).normal(0, .7, 12).astype(np.float32)
    after = copy.deepcopy(carry)
    after[1]['deter'][0] = carry[1]['deter'][0] + np.float32(1)
    after[1]['stoch'][0] = np.roll(carry[1]['stoch'][0], 1, axis=1).copy()
    after[3]['action'][0] = raw.copy()
    self.last_raw = raw.copy()
    self.calls[-1]['after_carry_sha256'] = model.tree_digest(after)
    self.calls[-1]['raw'] = raw.copy()
    with self.n_eval_actions.lock:
      if not (self.skip_counter and len(self.calls) == 11):
        self.n_eval_actions.value += 1
    if self.mutate_training and len(self.calls) == 11:
      self.n_actions.value += 1
    return after, dict(action=raw[None]), {}


class R4SourceIntegrationTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    from dm_control.mujoco.engine import Physics
    cls.render_patch = mock.patch.object(Physics, 'render', dummy_render)
    cls.render_patch.start()
    cls.episode_id = 3
    cls.adapter = physics.FeedbackEnvironment(d.make_dmc('dmc_quadruped_walk',
        keys.source_eval_seed(cls.episode_id)))
    cls.agent = FakePublicAgent()
    cls.saved = model.counters(cls.agent)
    cls.runtime_calls = []
    try:
      cls.collected = r.collect_episode(cls.agent, cls.adapter, cls.episode_id,
          lambda: cls.runtime_calls.append(int(cls.agent.n_eval_actions)))
    except Exception:
      cls.adapter.close()
      cls.render_patch.stop()
      raise

  @classmethod
  def tearDownClass(cls):
    cls.adapter.close()
    cls.render_patch.stop()

  def new_adapter(self):
    adapter = physics.FeedbackEnvironment(d.make_dmc('dmc_quadruped_walk',
        keys.source_eval_seed(self.episode_id)))
    self.addCleanup(adapter.close)
    return adapter

  def assert_restored(self, agent, original):
    self.assertEqual(model.counters(agent), original)
    self.assertEqual(agent.seed_calls[-1], original['eval_seed'])

  def test_complete_public_episode_frames_carries_keys_and_real_f64(self):
    data = self.collected
    arrays = data['arrays']
    self.assertEqual(data['episode_id'], self.episode_id)
    self.assertEqual(data['env_seed'], 4096 + self.episode_id)
    self.assertEqual(len(self.agent.calls), 1000)
    self.assertEqual(self.runtime_calls, [0])
    self.assertEqual(arrays['image'].shape, (1001, 64, 64, 3))
    self.assertEqual(arrays['image'].dtype, np.uint8)
    self.assertEqual(arrays['action'].shape, (1000, 12))
    self.assertEqual(arrays['action'].dtype, np.float32)
    self.assertEqual(arrays['action_applied'].dtype, np.float64)
    self.assertEqual(arrays['reward_f64'].shape, (1001,))
    self.assertEqual(arrays['reward_f64'].dtype, np.float64)
    self.assertEqual(arrays['reward'].dtype, np.float32)
    self.assertEqual(arrays['reward_f64'].astype(np.float32).tobytes(), arrays['reward'].tobytes())
    self.assertTrue(np.any(arrays['reward_f64'] != arrays['reward'].astype(np.float64)),
        'The fixture must retain real F64 information absent from F32 labels')
    self.assertEqual(np.flatnonzero(arrays['is_first']).tolist(), [0])
    self.assertEqual(np.flatnonzero(arrays['is_last']).tolist(), [1000])
    self.assertFalse(arrays['is_terminal'].any())
    self.assertTrue(arrays['action_executed'].all())
    self.assertEqual(arrays['action_executed'].shape, (1000,))
    self.assertEqual(len(data['snapshots']), 1001)
    self.assertEqual([int(s['suite']['step_count']) for s in data['snapshots']], list(range(1001)))
    self.assertEqual([x['position'] for x in data['records']], list(r.POSITIONS))
    for record in data['records']:
      t = record['position']
      self.assertEqual(record['eval_seed'], 4096 + self.episode_id)
      self.assertEqual(record['eval_counter'], t)
      self.assertEqual(record['carry_sha256'], model.tree_digest(record['carry']))
      self.assertEqual(record['carry_sha256'], self.agent.calls[t]['pre_carry_sha256'])
      self.assertEqual(record['after_carry_sha256'], self.agent.calls[t]['after_carry_sha256'])
      self.assertEqual(record['carry'][3]['action'][0].tobytes(), arrays['action'][t - 1].tobytes())
    for t, call in enumerate(self.agent.calls):
      self.assertEqual(call['counter'], t)
      np.testing.assert_array_equal(call['root'], keys.source_root_key(self.episode_id, t))
      self.assertEqual(call['raw'].tobytes(), arrays['action'][t].tobytes())
      self.assertEqual(call['observation']['image'][0].tobytes(), arrays['image'][t].tobytes())
    self.assertTrue(np.any(np.abs(arrays['action']) > 1))
    self.assertFalse(np.array_equal(arrays['action'], arrays['action_applied']))
    self.assert_restored(self.agent, self.saved)

  def test_source_reference_real_f64_and_exact_complete_snapshots(self):
    proof = r.verify_source_reference(self.adapter, self.collected)
    self.assertEqual(proof['physics_steps'], len(r.POSITIONS) * 100)
    self.assertTrue(proof['reward_float64_byte_exact'])
    self.assertTrue(proof['reward_float32_byte_exact'])
    self.assertTrue(proof['full_snapshot_byte_exact'])
    self.assertFalse(proof['renderer_replayed'])
    self.assertFalse(proof['policy_replayed'])
    for damage in ('f64_one_ulp', 'snapshot_one_ulp', 'control'):
      broken = copy.deepcopy(self.collected)
      t = r.POSITIONS[0]
      if damage == 'f64_one_ulp':
        old = broken['arrays']['reward_f64'][t + 1]
        broken['arrays']['reward_f64'][t + 1] = np.nextafter(old, np.inf)
        self.assertEqual(np.float32(old).tobytes(),
            np.float32(broken['arrays']['reward_f64'][t + 1]).tobytes())
      elif damage == 'snapshot_one_ulp':
        old = broken['snapshots'][t + 1]['suite']['integration'][0]
        broken['snapshots'][t + 1]['suite']['integration'][0] = np.nextafter(old, np.inf)
      else:
        broken['arrays']['action_applied'][t, 0] += .001
      with self.subTest(damage=damage), self.assertRaises((ValueError, AssertionError)):
        r.verify_source_reference(self.adapter, broken)

  def test_native_f32_suite_reward_not_promoted_to_f64(self):
    adapter, agent = self.new_adapter(), FakePublicAgent()
    original = model.counters(agent)
    step = adapter.step
    def narrowed(raw):
      result = step(raw)
      ts = adapter.recorder.last_timestep
      adapter.recorder.last_timestep = ts._replace(reward=np.float32(ts.reward))
      return result
    with mock.patch.object(adapter, 'step', narrowed):
      with self.assertRaisesRegex(ValueError, 'Actual source suite F64 reward'):
        r.collect_episode(agent, adapter, self.episode_id)
    self.assertEqual(len(agent.calls), 1)
    self.assert_restored(agent, original)

  def test_production_f32_reward_misaligned_with_actual_suite_f64_is_rejected(self):
    adapter, agent = self.new_adapter(), FakePublicAgent()
    original = model.counters(agent)
    step = adapter.step
    def misencoded(raw):
      obs, applied = step(raw)
      obs['reward'] = np.array(float(obs['reward']) + .125, np.float32)
      return obs, applied
    with mock.patch.object(adapter, 'step', misencoded):
      with self.assertRaisesRegex(ValueError, 'Actual source suite F64 reward'):
        r.collect_episode(agent, adapter, self.episode_id)
    self.assertEqual(len(agent.calls), 1)
    self.assert_restored(agent, original)

  def test_early_terminal_and_time_limit_rejected_without_episode_replacement(self):
    for discount in (0., 1.):
      adapter, agent = self.new_adapter(), FakePublicAgent()
      original = model.counters(agent)
      cls = type(adapter.suite._task)
      with self.subTest(discount=discount), mock.patch.object(cls, 'get_termination',
          return_value=discount), self.assertRaisesRegex(ValueError, 'reset/time-limit/terminal'):
        r.collect_episode(agent, adapter, self.episode_id)
      self.assertEqual(len(agent.calls), 1)
      self.assert_restored(agent, original)

  def test_public_counter_failure_policy_failure_and_training_mutation_fail_closed(self):
    for reason, agent, error in (
        ('missingcounter', FakePublicAgent(skip_counter=True), ValueError),
        ('policyfailure', FakePublicAgent(fail_at=65), RuntimeError),
        ('trainingmutation', FakePublicAgent(mutate_training=True), ValueError)):
      adapter = self.new_adapter()
      original = model.counters(agent)
      with self.subTest(reason=reason), self.assertRaises(error):
        r.collect_episode(agent, adapter, self.episode_id)
      self.assertEqual(int(agent.n_eval_actions), original['eval_actions'])
      self.assertEqual(agent.eval_seed, original['eval_seed'])
      if reason != 'trainingmutation': self.assert_restored(agent, original)
      else: self.assertNotEqual(model.counters(agent), original)

  def test_runtime_or_deadline_failure_also_restores_public_eval_state(self):
    adapter, agent = self.new_adapter(), FakePublicAgent()
    original = model.counters(agent)
    def fail_runtime(): raise RuntimeError('injected runtime mapping failure')
    with self.assertRaisesRegex(RuntimeError, 'mapping failure'):
      r.collect_episode(agent, adapter, self.episode_id, fail_runtime)
    self.assertEqual(agent.calls, [])
    self.assert_restored(agent, original)
    with mock.patch.object(r, 'check', side_effect=TimeoutError('injected owner deadline')):
      with self.assertRaises(TimeoutError):
        r.collect_episode(agent, adapter, self.episode_id)
    self.assertEqual(agent.calls, [])
    self.assert_restored(agent, original)

  def test_source_namespace_invalid_episode_refused_before_public_state_changes(self):
    for episode in (-1, 32, True, np.int64(1), 1.5):
      adapter, agent = self.new_adapter(), FakePublicAgent()
      original = model.counters(agent)
      with self.subTest(episode=episode), self.assertRaises(ValueError):
        r.collect_episode(agent, adapter, episode)
      self.assertEqual(agent.calls, [])
      self.assertEqual(agent.seed_calls, [])
      self.assertEqual(model.counters(agent), original)


if __name__ == '__main__':
  unittest.main()
