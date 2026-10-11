"""DL-source-r1 CPU trajectory semantics; RGB is stubbed, never GPU rendered."""
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_production_source as p
from scripts import dl_diagnostic as d


def dummy_render(self, height=64, width=64, camera_id=2, segmentation=False, **kw):
  if segmentation:
    return np.full((height, width, 2), -1, np.int32)
  return np.zeros((height, width, 3), np.uint8)


class FakePolicy:
  """Only the action/counter interface is stubbed; actual physics/wrappers run."""
  def __init__(self):
    import elements
    import jax
    self.n_eval_actions = elements.Counter()
    self.policy_mirrored = jax.sharding.NamedSharding(
        jax.sharding.Mesh(np.array(jax.devices()), ('d',)), jax.sharding.PartitionSpec())
    self.eval_seed = 0
  def set_eval_seed(self, seed):
    self.eval_seed, self.n_eval_actions.value = seed, 0
  def init_policy(self, batch, mode):
    return dict(previous=np.zeros((batch, 12), np.float32), calls=np.array([0]))
  def policy(self, carry, obs, mode):
    raw = np.linspace(-2.5, 2.5, 12, dtype=np.float32)[None]
    self.n_eval_actions.value += 1
    return dict(previous=raw.copy(), calls=np.array([int(self.n_eval_actions)])), dict(action=raw), {}


class ProductionSourceTest(unittest.TestCase):
  def adapter(self, seed=20261020):
    return p.ProductionEnvironment(d.make_dmc('dmc_quadruped_walk', seed))

  def test_actual_dmc_wrapper_restore_includes_hidden_done_and_terminal_reset(self):
    from dm_control.mujoco.engine import Physics
    with mock.patch.object(Physics, 'render', dummy_render):
      adapter = self.adapter()
      try:
        adapter.reset()
        snapshot = adapter.snapshot()
        raw = np.linspace(-3, 3, 12, dtype=np.float32)
        first, applied1 = adapter.step(raw)
        end1 = adapter.snapshot()
        adapter.from_dm._done = True  # Stale wrapper would otherwise silently reset.
        adapter.restore(snapshot)
        second, applied2 = adapter.step(raw)
        self.assertFalse(second['is_first'])
        np.testing.assert_array_equal(applied1, applied2)
        self.assertEqual(p.state_error(end1, adapter.snapshot()), 0)
        self.assertEqual(first['reward'].tobytes(), second['reward'].tobytes())
        # Non-first source near true task timeout: terminal flags and wrapper state.
        adapter.suite._step_count = int(adapter.suite._step_limit) - 1
        before_last = adapter.snapshot()
        last, _ = adapter.step(raw)
        self.assertTrue(last['is_last'])
        self.assertFalse(last['is_terminal'])
        final = adapter.snapshot()
        adapter.restore(before_last)
        replay_last, _ = adapter.step(raw)
        self.assertTrue(replay_last['is_last'])
        self.assertEqual(p.state_error(final, adapter.snapshot()), 0)
        adapter.restore(final)
        reset_obs, _ = adapter.step(raw) if not adapter.from_dm._done else (adapter.reset(), None)
        self.assertTrue(reset_obs['is_first'])
      finally:
        adapter.close()

  def test_source_episode_keeps_raw_and_actual_carries_and_full_reference_windows(self):
    from dm_control.mujoco.engine import Physics
    with mock.patch.object(Physics, 'render', dummy_render):
      adapter = self.adapter(7)
      try:
        entry = d.episode_plan(32, 8, p.SEED)[0]
        agent = FakePolicy()
        arrays, snapshots, carries, positions, proof = p.collect_episode(adapter, agent, entry)
        self.assertEqual(len(snapshots), 1001)
        self.assertEqual(len(positions), 16)
        self.assertEqual(proof['physics_steps'], 1600)
        self.assertEqual(proof['maximum_state_error'], 0)
        self.assertTrue(proof['policy_replay']['after_carry_byte_hash_exact'])
        self.assertEqual(int(agent.n_eval_actions), 1000)
        self.assertTrue(np.any(np.abs(arrays['action'][:-1]) > 1))
        self.assertEqual(arrays['action'].dtype, np.float32)
        self.assertEqual(arrays['action_applied'].dtype, np.float64)
        self.assertTrue(arrays['action_executed'][:-1].all())
        self.assertFalse(arrays['action_executed'][-1])
        self.assertFalse(np.array_equal(arrays['action'][:-1], arrays['action_applied'][:-1]))
        for saved in carries:
          t = saved['position']
          self.assertEqual(saved['eval_counter'], t)
          np.testing.assert_array_equal(saved['carry']['previous'][0], arrays['action'][t - 1])
          self.assertEqual(saved['carry_sha256'], p.tree_digest(saved['carry']))
        # Serialized snapshots/carry are CPU numeric objects, independent of devices.
        restored = pickle.loads(pickle.dumps((snapshots, carries), protocol=5))
        self.assertEqual(p.state_error(restored[0][42], snapshots[42]), 0)
        self.assertEqual(p.tree_digest(restored[1][0]['carry']), carries[0]['carry_sha256'])
        damaged = {k: v.copy() for k, v in arrays.items()}
        damaged['action_applied'][positions[0], 1] += .05
        with self.assertRaises(AssertionError):
          p.verify_reference_windows(adapter, damaged, snapshots, positions[:1])
      finally:
        adapter.close()

  def test_restoration_and_carry_hash_refuse_false_equivalence(self):
    from dm_control.mujoco.engine import Physics
    with mock.patch.object(Physics, 'render', dummy_render):
      adapter = self.adapter()
      try:
        adapter.reset()
        snap = adapter.snapshot()
        bad = pickle.loads(pickle.dumps(snap, protocol=5))
        bad['from_dm_done'] = not snap['from_dm_done']
        with self.assertRaises(AssertionError):
          p.state_error(snap, bad)
        for bad_value in (np.nan, np.inf):
          invalid = pickle.loads(pickle.dumps(snap, protocol=5))
          invalid['suite']['integration'][0] = bad_value
          with self.assertRaises(AssertionError):
            p.state_error(snap, invalid)
        self.assertNotEqual(p.tree_digest(dict(x=np.zeros(2, np.float32))),
            p.tree_digest(dict(x=np.zeros(2, np.float64))))
        with self.assertRaises(ValueError):
          p.tree_digest(np.array([dict(a=1)], object))
      finally:
        adapter.close()

  def test_supervisor_enforces_deadline_outside_the_cooperative_frame_loop(self):
    with tempfile.TemporaryDirectory() as directory:
      receipt = p.run_bounded([sys.executable, '-c',
          'import time; time.sleep(10)'], directory, time.monotonic() + .15)
      self.assertEqual(receipt['exit_code'], 124)
      self.assertTrue(receipt['timed_out'])
      self.assertTrue(receipt['worker_terminal'])
      self.assertEqual((Path(directory) / 'worker_exit_code.txt').read_text().strip(), '124')

  def test_public_tiny_policy_replays_host_bfloat16_carry_under_strict_transfer_guard(self):
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--policy-fixture'],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
        env=dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT)))
    self.assertEqual(child.returncode, 0, child.stdout[-3500:] + child.stderr[-4500:])
    self.assertIn('strict_guard_host_carry_replay_passed', child.stdout)


def policy_fixture():
  dll = (os.add_dll_directory(os.environ['DL_DLL_DIRECTORY'])
      if os.name == 'nt' and os.environ.get('DL_DLL_DIRECTORY') else None)
  import elements
  import jax
  from ruamel.yaml import YAML
  presets = YAML(typ='safe').load((ROOT/'dreamerv3/configs.yaml').read_text(encoding='utf-8'))
  config = elements.Config(presets['defaults']).update(presets['debug'])
  config = elements.Config({**config.flat, **{'jax.platform': 'cpu',
      'jax.compute_dtype': 'bfloat16', 'jax.precompile': False,
      'jax.profiler': False, 'jax.enable_policy': True, 'jax.transfer_guard': True,
      'batch_size': 2, 'batch_length': 3, 'replay_context': 0,
      'agent.imag_length': 2, 'agent.imag_last': 2,
      'agent.rep_probe.mode': 'off', 'agent.dt_latch.mode': 'off'}})
  config = config.update(logdir='/tmp/dl-source-policy-fixture', seed=7)
  agent = p.pinned.agent_from_spaces(config)
  agent.set_eval_seed(123)
  carry = agent.init_policy(1, mode='eval')
  obs0 = dict(image=np.zeros((1,64,64,3), np.uint8), reward=np.zeros(1,np.float32),
      is_first=np.ones(1,bool), is_last=np.zeros(1,bool), is_terminal=np.zeros(1,bool))
  carry, _, _ = agent.policy(carry, obs0, mode='eval')
  before = p.host_copy(carry)
  assert any(str(x.dtype) == 'bfloat16' for x in jax.tree.leaves(before))
  obs1 = {**obs0, 'is_first': np.zeros(1,bool)}
  after, raw, _ = agent.policy(carry, obs1, mode='eval')
  records = [dict(position=1, carry=before, carry_sha256=p.tree_digest(before),
      after_carry_sha256=p.tree_digest(p.host_copy(after)), eval_seed=123, eval_counter=1)]
  arrays = {k: np.concatenate([obs0[k], obs1[k]]) for k in obs0}
  arrays['action'] = np.concatenate([np.zeros((1,12),np.float32), raw['action']])
  receipt = p.verify_policy_records(agent, arrays, records)
  assert receipt['raw_sample_byte_exact'] and receipt['after_carry_byte_hash_exact']
  assert int(agent.n_eval_actions) == 2 and agent.eval_seed == 123
  assert jax.config.jax_transfer_guard == 'disallow'
  print('strict_guard_host_carry_replay_passed')


if __name__ == '__main__':
  if '--policy-fixture' in sys.argv:
    policy_fixture()
  else:
    unittest.main()
