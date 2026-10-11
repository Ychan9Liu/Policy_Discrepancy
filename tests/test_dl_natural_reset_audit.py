"""Meaningful synthetic CPU oracle failures; no real source or physics."""
import copy
import importlib.util
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('natural_reset_independent_audit',
    ROOT / 'analysis/dl_natural_reset_audit.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def state(index=0):
  return dict(suite=dict(integration=np.asarray([index, .125, .25, .5], np.float64),
      step_count=index, reset_next_step=False, task=dict(random_state=np.asarray([1, 2], np.uint32))),
      from_dm_done=False)


def record(control=0):
  raw = np.zeros((20, 12), np.float32); raw[0] = m.commands()[control]
  low = np.tile([-1., -1., -.8], 4).astype(np.float64)
  high = np.tile([1., 1.1, .8], 4).astype(np.float64)
  reward = np.arange(20, dtype=np.float64) / 32
  names = [f'{joint}_{leg}' for leg in ('front_left', 'front_right', 'back_right', 'back_left')
      for joint in ('yaw', 'lift', 'extend')]
  return dict(version='DL-action-information-r1', implementation='DL-action-information-physics-r1',
      action_contract=m.CONTRACT, horizon=20, executed_steps=20, end_type='window',
      first_raw=raw[0].copy(), raw=raw, applied=np.stack([m.mapping(x, low, high) for x in raw]),
      reward_f64=reward, reward_f32=reward.astype(np.float32), discount=np.ones(20, np.float64),
      is_first=np.zeros(20, bool), is_last=np.zeros(20, bool), is_terminal=np.zeros(20, bool),
      action_executed=np.ones(20, bool), snapshot_executed=np.ones(21, bool),
      snapshots=[state(t) for t in range(21)], model_receipt=dict(xml_sha256=m.XML_SHA,
          actuator_names=names, filter_tau=.1, physics_timestep=.005, control_timestep=.02,
          step_limit=1000, low=low.tolist(), high=high.tolist(), action_contract=m.CONTRACT))


def rewards_from_costs(costs):
  value = np.zeros((8, 7, 20), np.float64); value[:, :, 0] = costs
  return value


class NaturalResetAuditTest(unittest.TestCase):

  def test_closed_delta_boundary_and_winner_changes_are_not_necessity(self):
    costs = np.zeros((8, 7), np.float64)
    costs[0, 0] = .2; costs[1, 1] = .2
    value = m.statistics(rewards_from_costs(costs))
    self.assertEqual(value['status'], 'finite_reset_control_object_insufficient')
    self.assertTrue(all(value['delta_optimal_membership'][0]))
    pair = value['pairs'][0]
    self.assertFalse(pair['disjoint']); self.assertEqual(pair['minimax_minus_delta'], 0.)
    costs[0, 0] = np.nextafter(.2, np.inf); costs[1, 1] = np.nextafter(.2, np.inf)
    value = m.statistics(rewards_from_costs(costs))
    self.assertEqual(value['status'], 'finite_reset_control_object_sufficient')
    self.assertTrue(value['pairs'][0]['disjoint'])
    self.assertGreater(value['pairs'][0]['minimax_minus_delta'], 0.)
    self.assertEqual(len(value['pairs']), 28)

  def test_range_large_with_common_control_and_additive_rewards(self):
    costs = np.tile(np.asarray([1., 0., .1, .2, .3, .4, .5]), (8, 1))
    costs[1, 1] = 1.1
    value = m.statistics(rewards_from_costs(costs))
    self.assertGreater(value['range'][0], .2)
    self.assertEqual(value['status'], 'finite_reset_control_object_insufficient')
    changed = rewards_from_costs(costs) + np.arange(8)[:, None, None] * .25
    again = m.statistics(changed)
    self.assertEqual(value['delta_optimal_membership'], again['delta_optimal_membership'])
    m.compare_tree(value['pairs'], again['pairs'])
    self.assertEqual(value['status'], again['status'])

  def test_true_conflicting_sets_and_exhaustive_minimax(self):
    costs = np.zeros((8, 7), np.float64); costs[0, 2] = 1.; costs[1, 6] = .5
    value = m.statistics(rewards_from_costs(costs))
    self.assertEqual(value['delta_optimal_membership'][0], [False, False, True, False, False, False, False])
    self.assertEqual(value['delta_optimal_membership'][1], [False, False, False, False, False, False, True])
    self.assertEqual(value['pairs'][0]['episodes'], [9, 14])
    self.assertEqual(value['pairs'][0]['minimax_regret'], .5)
    self.assertAlmostEqual(value['pairs'][0]['minimax_minus_delta'], .3)
    self.assertTrue(value['pairs'][0]['sufficient_pair'])

  def test_fsum_independent_reduction_and_dtype_axis_guard(self):
    x = np.zeros((8, 7, 20), np.float64)
    x[..., :3] = [1e16, 1., -1e16]
    self.assertEqual(m.statistics(x)['costs'][0][0], 1.)
    for bad in (x.astype(np.float32), x[:, :6], np.full_like(x, np.nan)):
      with self.assertRaises(ValueError): m.statistics(bad)

  def test_numerical_tolerance_never_relaxes_membership_or_margin_side(self):
    value = m.statistics(np.zeros((8, 7, 20), np.float64))
    changed = copy.deepcopy(value); changed['costs'][0][0] += 1e-11
    m.compare_tree(changed, value)
    changed['costs'][0][0] += 1e-6
    with self.assertRaises(ValueError): m.compare_tree(changed, value)
    for key, replacement in (('delta_optimal_membership', [[False] * 7] * 8),
                             ('status', 'finite_reset_control_object_sufficient')):
      changed = copy.deepcopy(value); changed[key] = replacement
      with self.assertRaises(ValueError): m.compare_tree(changed, value)
    expected = dict(minimax_minus_delta=0.)
    with self.assertRaises(ValueError): m.compare_tree(dict(minimax_minus_delta=1e-12), expected)
    with self.assertRaises(ValueError): m.compare_tree(dict(minimax_minus_delta=-1e-12), expected)

  def test_mapping_zero_lift_offset_and_no_double_normalization(self):
    row = record(3)
    self.assertEqual(row['applied'][1, 1], .050000000000000044)
    self.assertTrue(m.byte_equal(row['raw'][1], np.zeros(12, np.float32)))
    changed = copy.deepcopy(row)
    low = np.tile([-1., -1., -.8], 4).astype(np.float64)
    high = np.tile([1., 1.1, .8], 4).astype(np.float64)
    changed['applied'][1] = m.mapping(changed['applied'][1].astype(np.float32), low, high)
    with self.assertRaises(ValueError): m.validate_record(changed, m.commands()[3], state(), 4)

  def test_record_source_binding_full_execution_and_terminal_guard(self):
    row = record(2)
    self.assertEqual(m.validate_record(row, m.commands()[2], state(), 4), 0.)
    failures = []
    for mutate in (
        lambda x: x.update(executed_steps=19),
        lambda x: x.update(end_type='true_terminal'),
        lambda x: x['is_last'].__setitem__(10, True),
        lambda x: x['snapshot_executed'].__setitem__(20, False),
        lambda x: x['raw'].__setitem__((1, 0), .25),
        lambda x: x['snapshots'][3]['suite'].__setitem__('step_count', 4),
        lambda x: x['reward_f32'].__setitem__(0, .1)):
      changed = copy.deepcopy(row); mutate(changed); failures.append(changed)
    for changed in failures:
      with self.assertRaises(ValueError): m.validate_record(changed, m.commands()[2], state(), 4)
    wrong = state(); wrong['suite']['task']['random_state'][0] = 9
    with self.assertRaises(ValueError): m.validate_record(row, m.commands()[2], wrong, 4)

  def test_new_state_byte_exact_including_signed_zero_and_original_guard(self):
    value = state(); other = copy.deepcopy(value)
    m.complete_state_exact(value, other)
    other['suite']['integration'][0] = -0.
    with self.assertRaises(ValueError): m.complete_state_exact(value, other)
    other = copy.deepcopy(value); other['suite']['integration'][0] += 1e-12
    with self.assertRaises(ValueError): m.complete_state_exact(value, other)
    other = copy.deepcopy(value); other['suite']['task']['random_state'][0] += 1
    with self.assertRaises(ValueError): m.complete_state_exact(value, other)
    row = record(); original = state(); original['suite']['integration'][0] += 1e-9
    self.assertAlmostEqual(m.validate_record(row, m.commands()[0], original, 4), 1e-9)
    original['suite']['integration'][0] = 1e-7
    with self.assertRaises(ValueError): m.validate_record(row, m.commands()[0], original, 4)

  def test_independent_stored_applied_replay_exact_and_corruption(self):
    row = record(1)
    class TimeStep:
      def __init__(self, t):
        self.reward = row['reward_f64'][t]; self.discount = row['discount'][t]
      def first(self): return False
      def last(self): return False
    class Environment:
      _reset_next_step = False
      def __init__(self): self.index = 0
      def step(self, action):
        if not m.byte_equal(action, row['applied'][self.index]): raise AssertionError('Remapped applied')
        result = TimeStep(self.index); self.index += 1; return result
    def restore(env, saved): env.index = saved['suite']['step_count']
    def snapshot(env, done): return copy.deepcopy(row['snapshots'][env.index])
    value = m.replay_record(Environment(), row, restore, snapshot)
    self.assertEqual(value['physics_steps'], 20)
    self.assertTrue(value['complete_state_byte_exact'])
    def bad_snapshot(env, done):
      result = snapshot(env, done)
      if env.index == 7: result['suite']['integration'][0] += 1e-12
      return result
    with self.assertRaises(ValueError): m.replay_record(Environment(), row, restore, bad_snapshot)

  def test_json_path_and_finite_exact_types(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp); path = root / 'a.json'; path.write_text('{"x":1,"x":2}')
      with self.assertRaises(ValueError): m.read_json(path)
      path.write_text('{"x":NaN}')
      with self.assertRaises(ValueError): m.read_json(path)
      with self.assertRaises(ValueError): m.safe_file(root, '../a.json')
      with self.assertRaises(ValueError): m.compare_tree({'count': True}, {'count': 1})

  def test_complete_published_statistics_wiring_not_producer_oracle(self):
    # The producer is the object under test; expected values come from this
    # audit's independent fsum/exhaustive-control computation.
    spec = importlib.util.spec_from_file_location('reset_producer_under_test',
        ROOT / 'scripts/dl_natural_reset.py')
    producer = importlib.util.module_from_spec(spec); spec.loader.exec_module(producer)
    for x in (np.zeros((8, 7, 20), np.float64),
              np.random.default_rng(27).uniform(0., 1., (8, 7, 20))):
      expected = m.analysis_report(m.statistics(x))
      m.compare_tree(producer.analyze(x), expected)
      changed = copy.deepcopy(producer.analyze(x)); changed['disjoint_pair_count'] += 1
      with self.assertRaises(ValueError): m.compare_tree(changed, expected)

  def test_npz_pickle_byte_binding_metadata_and_authenticated_decode(self):
    rows = [record(a) for a in range(7)]
    for a, value in enumerate(rows): value.update(episode=9, position=0, control=a)
    arrays = {name: np.stack([value[name] for value in rows]) for name in m.FIELDS}
    def payload(data, records=rows):
      stream = io.BytesIO()
      np.savez_compressed(stream, **data, episode=np.array(9, np.int32), position=np.array(0, np.int32),
          records=np.frombuffer(pickle.dumps(records, protocol=5), np.uint8))
      return stream.getvalue()
    packed = payload(arrays); digest = hashlib.sha256(packed).hexdigest()
    row, decoded = m.decode_producer_payload(packed, digest, 9)
    self.assertTrue(m.byte_equal(row['reward_f64'][0], decoded[0]['reward_f64']))
    with mock.patch.object(m.pickle, 'loads', side_effect=AssertionError('Unsafe unpickle')):
      with self.assertRaises(ValueError): m.decode_producer_payload(packed, '0' * 64, 9)
    changed = {name: value.copy() for name, value in arrays.items()}; changed['reward_f64'][0, 0] += .125
    packed = payload(changed)
    with self.assertRaises(ValueError): m.decode_producer_payload(packed, hashlib.sha256(packed).hexdigest(), 9)
    packed = payload(arrays)
    with self.assertRaises(ValueError): m.decode_producer_payload(packed, hashlib.sha256(packed).hexdigest(), 14)

  def test_source_exit_link_and_only_calibration_archives_are_decoded(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp); artifacts = []
      for ep in range(32):
        path = root / f'episode_{ep:03}.npz'
        if ep in m.EPISODES:
          first = np.zeros(1001, bool); first[0] = True
          last = np.zeros(1001, bool); last[-1] = True
          executed = np.ones(1001, bool); executed[-1] = False
          np.savez_compressed(path, reward=np.zeros(1001, np.float32),
              action=np.zeros((1001, 12), np.float32), action_applied=np.zeros((1001, 12), np.float64),
              action_executed=executed, is_first=first, is_last=last, is_terminal=np.zeros(1001, bool),
              action_contract=np.array(m.CONTRACT),
              snapshots=np.frombuffer(pickle.dumps([state()] * 1001, protocol=5), np.uint8))
        else:
          # These deliberately cannot be np.load-ed. They may only be hashed.
          path.write_bytes(b'blind archive must never be decoded')
        artifacts.append(dict(episode=ep, path=path.name, sha256=m.sha(path), seed=ep,
            split='calibration' if ep in m.EPISODES else 'blind'))
      plan = dict(version='DL-source-r1', implementation='DL-source-engineering-r2',
          action_contract=m.CONTRACT, metadata=dict(git_sha=m.SOURCE_GIT, **m.MODEL_PINS),
          environment=dict(packages={}))
      m.write_json(root / 'collection_plan.json', plan)
      complete = dict(version='DL-source-r1', implementation='DL-source-engineering-r2',
          action_contract=m.CONTRACT, plan_sha256=m.sha(root / 'collection_plan.json'), artifacts=artifacts,
          params_unchanged=True, training_counters_unchanged=True, optimizer_updates=0)
      m.write_json(root / 'collection_complete.json', complete)
      m.write_json(root / 'budget.json', dict(version='DL-source-r1', git_sha=m.SOURCE_GIT,
          seconds=3600, script_sha256='a' * 64))
      m.write_json(root / 'supervisor_complete.json', dict(version='DL-source-r1', git_sha=m.SOURCE_GIT,
          budget_sha256=m.sha(root / 'budget.json'), exit_code=0, timed_out=False,
          worker_terminal=True, elapsed_seconds=1.))
      (root / 'worker_exit_code.txt').write_bytes(b'0\n')
      metadata = {name: m.sha(root / name) for name in ('collection_plan.json', 'collection_complete.json',
          'budget.json', 'supervisor_complete.json', 'worker_exit_code.txt')}
      artifact_audit = root / 'artifact.json'; physics_audit = root / 'physics.json'
      m.write_json(artifact_audit, dict(source_contract_artifact_audit_passed=True,
          expected_source_sha=m.SOURCE_GIT, blind_arrays_read=False, metadata_sha256=metadata,
          supervisor_exit_code=0, source_file_sha256={'scripts/dl_production_source.py': 'a' * 64}))
      m.write_json(physics_audit, dict(expected_source_sha=m.SOURCE_GIT, status='passed',
          blind_npz_arrays_opened=False, physics_steps=12800,
          source_json_sha256=dict(complete=m.sha(root / 'collection_complete.json'),
              budget=metadata['budget.json'], supervisor=metadata['supervisor_complete.json'])))
      with mock.patch.dict(m.SOURCE_PINS, {name: metadata[name] for name in
              ('collection_plan.json', 'collection_complete.json')}, clear=True), \
          mock.patch.dict(m.AUDIT_PINS, dict(artifact=m.sha(artifact_audit), physics=m.sha(physics_audit)), clear=True):
        with mock.patch.object(m.np, 'load', wraps=np.load) as loaded:
          rows, source_plan, exits = m.load_source(root, artifact_audit, physics_audit)
          self.assertEqual(loaded.call_count, 8)
          self.assertEqual([row['artifact']['episode'] for row in rows], list(m.EPISODES))
          self.assertEqual(exits, metadata)
        (root / 'worker_exit_code.txt').write_text('1\n')
        with mock.patch.object(m.np, 'load', side_effect=AssertionError('Decode before exit binding')):
          with self.assertRaises(ValueError): m.load_source(root, artifact_audit, physics_audit)

  def test_generic_parent_scope_unknown_and_completion_rejected(self):
    with tempfile.TemporaryDirectory() as temp:
      output = Path(temp) / 'pass'
      value = m.supervise([sys.executable, '-c', 'pass'], output, seconds=5)
      self.assertEqual(value['exit_code'], 0); self.assertTrue(value['worker_terminal'])
      self.assertFalse(value['complete_verified'])
      self.assertIsNone(value['actual_gpu_devices_used']); self.assertIsNone(value['actual_physics_steps'])
      def reject(output, expected): raise ValueError('Missing scientific audit')
      value = m.supervise([sys.executable, '-c', 'pass'], Path(temp) / 'reject', seconds=5, verify=reject)
      self.assertEqual(value['exit_code'], 1); self.assertFalse(value['complete_verified'])
      self.assertTrue((Path(temp) / 'reject/completion_verification_failure.json').is_file())

  def test_generic_forged_callback_cannot_claim_real_gpu_or_physics_scope(self):
    with tempfile.TemporaryDirectory() as temp:
      def forged(output, expected):
        return dict(complete_verified=True, physics_steps=1120, actual_gpu_devices_used=0)
      value = m.supervise([sys.executable, '-c', 'pass'], Path(temp) / 'forged',
          seconds=5, verify=forged)
      self.assertEqual(value['exit_code'], 0)
      self.assertFalse(value['complete_verified'])
      self.assertIsNone(value['actual_gpu_devices_used']); self.assertIsNone(value['actual_physics_steps'])
      self.assertEqual(value['verification_kind'], 'generic_callback_unknown_actual_scope')

  def test_delayed_receipt_serialization_never_publishes_success(self):
    with tempfile.TemporaryDirectory() as temp:
      original = m.write_json
      def delayed(path, value):
        original(path, value)
        if Path(path).name == 'supervisor_receipt_pending.json': time.sleep(.08)
      with mock.patch.object(m, 'write_json', side_effect=delayed):
        value = m.supervise([sys.executable, '-c', 'pass'], Path(temp) / 'late', seconds=.07)
      self.assertEqual(value['exit_code'], 124); self.assertFalse(value['complete_verified'])
      self.assertTrue(value['timed_out'])
      self.assertTrue((Path(temp) / 'late/late_supervisor_receipt.json').exists())
      final = m.read_json(Path(temp) / 'late/supervisor_complete.json')
      self.assertEqual(final['exit_code'], 124); self.assertFalse(final['complete_verified'])

  @unittest.skipUnless(os.name == 'posix', 'owned POSIX group deadline fixture')
  def test_owned_parent_timeout_stops_worker_without_claiming_scientific_scope(self):
    with tempfile.TemporaryDirectory() as temp:
      value = m.supervise([sys.executable, '-c', 'import time; time.sleep(10)'],
          Path(temp) / 'timeout', seconds=.2)
      self.assertEqual(value['exit_code'], 124); self.assertTrue(value['timed_out'])
      self.assertTrue(value['worker_terminal']); self.assertTrue(value['owned_group_cleanup_requested'])
      self.assertIsNone(value['actual_physics_steps'])
      with self.assertRaises(ProcessLookupError): os.kill(value['worker_pid'], 0)


if __name__ == '__main__':
  unittest.main()
