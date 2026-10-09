"""Small action-grid, matching, and partial-driver boundary fixtures."""

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import elements
import jax
import numpy as np
import ruamel.yaml as yaml
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from embodied.core.driver import Driver
from embodied.run.protocol_v1 import (
    ProtocolState, array_hashes, audit_complete_outputs, audit_persisted_history,
    report_fingerprint, report_on_batch, seed32)
from dreamerv3.freeze_c import freeze


class TinyEnv:

  def __init__(self):
    self.count = 0
    self.act_space = {'action': elements.Space(np.float32, (1,)),
                      'reset': elements.Space(bool)}

  def step(self, action):
    if action['reset']:
      self.count = 0
      return dict(reward=np.float32(0), is_first=np.bool_(True),
                  is_last=np.bool_(False))
    self.count += 1
    return dict(reward=np.float32(1), is_first=np.bool_(False),
                is_last=np.bool_(self.count == 2))

  def close(self):
    pass


class ProtocolV1Test(unittest.TestCase):

  def test_report_slices_consumed_host_batch_under_transfer_guard(self):
    mesh = Mesh(np.array(jax.devices()[:1]), ('d',))
    class FakeAgent:
      train_sharded = NamedSharding(mesh, P('d', None))
      train_mirrored = NamedSharding(mesh, P())
      def init_report(self, batch_size):
        self.batch_size = batch_size
        return None
      def report(self, carry, batch):
        self.seen = batch
        return carry, {'reported': True}
    agent = FakeAgent()
    raw = dict(is_first=np.zeros((2, 4), bool),
               image=np.arange(24, dtype=np.uint8).reshape((2, 4, 3)))
    with jax.transfer_guard('disallow'):
      result = report_on_batch(agent, raw, 0, 0, 'dummycont_test', 3)
    self.assertEqual(result, {'reported': True})
    self.assertEqual(agent.batch_size, 2)
    self.assertEqual(agent.seen['image'].shape, (2, 3, 3))
    self.assertEqual(agent.seen['seed'].shape, (2,))
    np.testing.assert_array_equal(raw['image'], np.arange(
        24, dtype=np.uint8).reshape((2, 4, 3)))
    with self.assertRaisesRegex(TypeError, 'host replay batch'):
      report_on_batch(agent, agent.seen, 0, 0, 'dummycont_test', 3)

  def test_checkpoint_receipts_and_completed_outputs_must_agree(self):
    with TemporaryDirectory() as directory:
      root = Path(directory)
      state = ProtocolState(12, 3, 1, 3, 9)
      state.record_evaluation(0, [1], [2], [7])
      (root / 'evaluations.jsonl').write_text(json.dumps(dict(
          action_step=0, **state.evaluations[0])) + '\n')
      audit_persisted_history(root, state)
      orphan = dict(update_id=0, start_action=3, match_sum=0.5,
                    match_count=1, invalid_count=0)
      receipt_path = root / 'update_receipts.jsonl'
      receipt_path.write_text(json.dumps(orphan) + '\n')
      with self.assertRaisesRegex(RuntimeError, 'extra=\\[0\\]'):
        audit_persisted_history(root, state)
      self.assertEqual(len(receipt_path.read_text().splitlines()), 1)
      state.record_update(0, 3, 0.5, 1)
      audit_persisted_history(root, state)
      with self.assertRaisesRegex(RuntimeError, 'lacks final_state'):
        audit_complete_outputs(root, state, 'logging', 'fixture')

  def test_formal_preset_resolves_clean_visual_dmc(self):
    source = Path(__file__).resolve().parents[1] / 'dreamerv3' / 'configs.yaml'
    configs = yaml.YAML(typ='safe').load(source.read_text(encoding='utf-8'))
    config = elements.Config(configs['defaults']).update(configs['m2_v1'])
    self.assertEqual(config.script, 'protocol_v1')
    self.assertEqual(config.seed, 0)
    self.assertEqual(config.agent.rep_probe.mode, 'logging')
    self.assertEqual(config.agent.rep_probe.alpha, 20)
    self.assertEqual(config.run.action_budget, 1000000)
    self.assertEqual(config.run.eval_every_actions, 50000)
    self.assertEqual(config.run.eval_eps, 10)
    self.assertEqual(config.run.train_ratio, 256)
    self.assertFalse(config.env.dmc.proprio)
    self.assertTrue(config.env.dmc.image)
    self.assertTrue(config.env.dmc.use_seed)
    self.assertEqual(config.env.dmc.repeat, 1)
    self.assertEqual(config.run.engineering_trace_actions, 0)
    self.assertEqual(config.run.engineering_trace_updates, 0)

  def test_engineering_hashes_exclude_replay_uuid_and_detect_input_change(self):
    first = dict(image=np.zeros((2, 3, 3), np.uint8),
                 action=np.zeros((2, 3, 1), np.float32),
                 stepid=np.zeros((2, 3, 24), np.uint8))
    second = {k: v.copy() for k, v in first.items()}
    second['stepid'][0, 0, 0] = 1
    self.assertEqual(array_hashes(first, skip=('stepid',)),
                     array_hashes(second, skip=('stepid',)))
    second['image'][0, 0, 0] = 1
    self.assertNotEqual(array_hashes(first, skip=('stepid',))['image'],
                        array_hashes(second, skip=('stepid',))['image'])

  def test_physics_trace_is_opt_in_and_excluded_from_training_batch_hash(self):
    batch = {'image': np.zeros((1, 1, 3), np.uint8),
             'log/physics_state': np.array([1.0, 2.0], np.float64)}
    changed = {key: value.copy() for key, value in batch.items()}
    changed['log/physics_state'][0] = 3.0
    self.assertEqual(array_hashes(batch), array_hashes(changed))
    self.assertNotEqual(array_hashes(batch, include_logs=True),
                        array_hashes(changed, include_logs=True))

  def test_report_fingerprint_detects_state_and_batch_changes(self):
    class Agent:
      def __init__(self):
        self.params = {'weight': np.array([1, 2], np.float32)}
        self.counters = {'updates': 3, 'batches': 3}
      def save(self):
        return dict(params=self.params, counters=self.counters.copy())
    class Replay:
      def __init__(self):
        self.metrics = {'samples': 3}
        self.sampler = type('Sampler', (), {})()
        self.sampler.rng = np.random.default_rng(7)
    agent, replay = Agent(), Replay()
    batch = {'image': np.zeros((2, 3), np.uint8)}
    before = report_fingerprint(agent, replay, batch)
    self.assertEqual(before, report_fingerprint(agent, replay, batch))
    agent.params['weight'][0] = 3
    self.assertNotEqual(before, report_fingerprint(agent, replay, batch))
    agent.params['weight'][0] = 1
    batch['image'][0, 0] = 1
    self.assertNotEqual(before, report_fingerprint(agent, replay, batch))

  def test_partial_driver_and_exact_action_budget(self):
    driver = Driver([TinyEnv] * 3, parallel=False)
    state = ProtocolState(5, 1, 2, 1, 4)
    seen = []
    driver.on_step(lambda tran, worker: (
        state.on_train_transition(tran), seen.append((worker, bool(
            tran['is_first'])))))
    def policy(carry, obs):
      next_carry = {'value': [x + 1 for x in carry['value']]}
      acts = {'action': np.ones((len(obs['reward']), 1), np.float32)}
      return next_carry, acts, {}
    driver.reset(lambda count: {'value': [0] * count})
    while state.actions < state.budget:
      remaining = state.budget - state.actions
      driver.step_selected(policy, range(min(3, remaining)))
    self.assertEqual(state.actions, 5)
    self.assertGreaterEqual(state.resets, 3)
    self.assertEqual(driver.carry['value'][2], 2)
    self.assertEqual(seen[-1][0], 1)
    driver.close()

  def test_partial_driver_parallel_workers(self):
    driver = Driver([TinyEnv] * 3, parallel=True)
    seen = []
    driver.on_step(lambda tran, worker: seen.append(
        (worker, bool(tran['is_first']))))
    driver.reset(lambda count: {'value': [0] * count})
    policy = lambda carry, obs: (
        {'value': [x + 1 for x in carry['value']]},
        {'action': np.ones((len(obs['reward']), 1), np.float32)}, {})
    driver.step_selected(policy, [0, 1, 2])
    driver.step_selected(policy, [1])
    self.assertEqual(seen, [(0, True), (1, True), (2, True),
                            (1, False)])
    self.assertEqual(driver.carry['value'], [1, 2, 1])
    driver.close()

  def test_match_window_receipts_and_replay_repeat(self):
    state = ProtocolState(12, 3, 2, 3, 9)
    self.assertEqual(state.grid, (0, 3, 6, 9, 12))
    self.assertEqual(state.next_boundary(), 3)
    state.record_update(0, 2, 0.4, 1)
    state.record_update(1, 3, 0.2, 1)
    state.record_update(2, 8, 1.8, 3)
    state.record_update(3, 9, 0.5, 1)
    self.assertEqual(state.match_count, 4)
    self.assertAlmostEqual(state.match_sum, 2.0)
    self.assertAlmostEqual(state.frozen_c(), 0.5)
    self.assertFalse(state.record_update(2, 8, 1.8, 3))
    with self.assertRaisesRegex(ValueError, 'Conflicting'):
      state.record_update(2, 8, 1.9, 3)
    # Repeating the same replay sample under a new update ID counts again.
    state.record_update(4, 8, 1.8, 3)
    self.assertEqual(state.match_count, 7)
    self.assertAlmostEqual(state.frozen_c(), 3.8 / 7)

  def test_empty_invalid_and_checkpoint_restore(self):
    state = ProtocolState(12, 3, 2, 3, 9)
    state.record_update(0, 3, 0, 0)
    with self.assertRaisesRegex(ValueError, 'No active'):
      state.frozen_c()
    for kwargs in (
        dict(match_sum=float('nan'), match_count=1),
        dict(match_sum=1, match_count=1, invalid_count=1),
        dict(match_sum=2, match_count=1),
        dict(match_sum=0, match_count=0, finite=False),
    ):
      with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
        state.record_update(1, 3, **kwargs)
    state.actions = 3
    state.record_evaluation(3, [1, 2], [2, 2], [11, 12])
    restored = ProtocolState(12, 3, 2, 3, 9)
    restored.load(state.save())
    self.assertEqual(restored.save(), state.save())
    with self.assertRaisesRegex(ValueError, 'already recorded'):
      restored.record_evaluation(3, [1, 2], [2, 2], [11, 12])
    with self.assertRaisesRegex(ValueError, 'protocol mismatch'):
      ProtocolState(15, 3, 2, 3, 9).load(state.save())

  def test_seed_mapping_is_named_and_repeatable(self):
    first = seed32(0, 'dmc_reacher_hard', 'eval_env', 3, 7)
    self.assertEqual(first, seed32(0, 'dmc_reacher_hard', 'eval_env', 3, 7))
    self.assertNotEqual(first, seed32(
        0, 'dmc_reacher_hard', 'train_env', 3, 7))
    self.assertNotEqual(first, seed32(
        0, 'dmc_reacher_hard', 'eval_env', 3, 8))

  def test_freeze_uses_raw_receipts_and_is_immutable(self):
    with TemporaryDirectory() as directory:
      root = Path(directory)
      config = dict(
          [('agent.rep_probe.mode', 'logging'),
           ('agent.rep_probe.alpha', 20.0),
           ('agent.dyn.rssm.free_nats', 0.1),
           ('run.action_budget', 12), ('run.eval_every_actions', 3),
           ('run.eval_eps', 2), ('run.match_start', 3),
           ('run.match_end', 9), ('run.engineering_fixture', True)])
      (root / 'protocol_manifest.json').write_text(json.dumps(dict(
          protocol='M2-v1', task='dummycont_test',
          git_commit='fixture', config=config)))
      receipts = [
          dict(update_id=0, start_action=3, match_sum=0.2,
               match_count=1, invalid_count=0),
          dict(update_id=1, start_action=8, match_sum=1.8,
               match_count=3, invalid_count=0),
          dict(update_id=2, start_action=9, match_sum=1,
               match_count=1, invalid_count=0)]
      (root / 'update_receipts.jsonl').write_text(
          ''.join(json.dumps(row) + '\n' for row in receipts))
      (root / 'evaluations.jsonl').write_text(
          ''.join(json.dumps(dict(action_step=step)) + '\n'
                  for step in range(0, 13, 3)))
      (root / 'final_state.json').write_text(json.dumps(dict(
          git_commit='fixture', train_action_steps=12, updates=3,
          evaluations=5, match_sum=2.0, match_count=4)))
      (root / 'matching_result.json').write_text(json.dumps(dict(
          match_sum=2.0, match_count=4, c=0.5, start=3, end=9)))
      output = root / 'c.json'
      result = freeze(root, output, engineering_fixture=True)
      self.assertEqual(result['match_count'], 4)
      self.assertAlmostEqual(result['c'], 0.5)
      self.assertEqual(result['included_update_ids'], [0, 1])
      self.assertEqual(freeze(root, output, True), result)
      self.assertTrue((root / 'c.json.sha256').exists())
      final = json.loads((root / 'final_state.json').read_text())
      final['updates'] = 2
      (root / 'final_state.json').write_text(json.dumps(final))
      with self.assertRaisesRegex(ValueError, 'disagrees'):
        freeze(root, output, True)
      final['updates'] = 3
      (root / 'final_state.json').write_text(json.dumps(final))
      tampered = json.loads(output.read_text())
      tampered['c'] = 0.6
      output.write_text(json.dumps(tampered))
      with self.assertRaises(ValueError):
        freeze(root, output, True)


if __name__ == '__main__':
  unittest.main()
