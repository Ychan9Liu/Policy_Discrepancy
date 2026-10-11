"""Necessary CPU envelope contracts; no actual source/model/physics or GPU.

Synthetic statistics and snapshot bytes verify pairing and evidence contracts.
The real subprocess tests verify owned cleanup; injected late clocks verify
that a provisional parent receipt cannot complete beyond its entire budget.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

import numpy as np

from scripts import dl_rep_native as producer


def snapshot():
  return dict(suite=dict(integration=np.array([0., 1., 2.], np.float64),
      task={'random': np.array([11, 19], np.uint32)},
      step_count=0, reset_next_step=False), from_dm_done=False)


class NativeProducerContracts(unittest.TestCase):

  def test_same_payload_hash_before_decode_and_duplicate_json_rejection(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'fixture.pkl'
      pin = producer.save_pickle(path, {'fixture_only': 42})
      self.assertEqual(producer.load_payload(path, pin), {'fixture_only': 42})
      path.write_bytes(b'changed untrusted bytes')
      with mock.patch.object(producer.pickle, 'loads') as decode:
        with self.assertRaisesRegex(ValueError, 'before decode'):
          producer.load_payload(path, pin)
        decode.assert_not_called()
      path = Path(directory) / 'fixture.json'
      for payload in ('{"seed":1024,"seed":1025}', '{"value":NaN}'):
        path.write_text(payload, encoding='utf-8')
        with self.assertRaises(ValueError):
          producer.read_json(path)

  def test_frozen_fixed32_specs_and_bytes_reject_tampering(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source = root / 'fixture-only.bin'; source.write_bytes(b'fixed')
      frozen = dict(version=producer.VERSION, groups=['0', 'B', 'A'],
          evaluation_seeds=list(range(1024, 1056)), transitions_per_episode=1000,
          primary='A_minus_B', bootstrap_seed=20261026, bootstrap_draws=2000,
          new_evaluation_started=False, additional_F_measurements_performed=False,
          files_sha256={source.name: producer.sha(source)})
      producer.write_json(root / 'freeze.json', frozen)
      self.assertEqual(producer.verify_freeze(root), frozen)
      changed = copy.deepcopy(frozen); changed['evaluation_seeds'][-1] = 1056
      (root / 'freeze.json').write_text(json.dumps(changed), encoding='utf-8')
      with self.assertRaisesRegex(ValueError, 'Frozen object differs'):
        producer.verify_freeze(root)
      (root / 'freeze.json').write_text(json.dumps(frozen), encoding='utf-8')
      source.write_bytes(b'changed')
      with self.assertRaisesRegex(ValueError, 'bytes changed'):
        producer.verify_freeze(root)

  def test_whole32_pairing_signed_difference_and_shared_translation(self):
    # Individual returns vary widely, but true paired differences are constant.
    # Independent per-arm resampling or seed mispairing would widen this CI.
    base = np.arange(32, dtype=np.float64) * 31
    result = producer.statistics({'0': base - 2, 'B': base, 'A': base + 3})
    self.assertEqual(result['seeds'], list(range(1024, 1056)))
    self.assertEqual(result['groups'], ['0', 'B', 'A'])
    self.assertEqual(result['primary'], 'A_minus_B')
    self.assertEqual(result['bootstrap']['unit'], 'whole_episode')
    self.assertTrue(result['bootstrap']['paired'])
    paired = result['comparisons']['A_minus_B']
    self.assertEqual(paired['values'], [3.] * 32)
    self.assertEqual(paired['ci95'], [3., 3.])
    self.assertEqual(paired['ci_zero_side'], [1, 1])
    self.assertEqual(paired['classification'], 'positive_finite_net_effect')
    translated = producer.statistics({'0': base + 98, 'B': base + 100, 'A': base + 103})
    self.assertEqual(result['comparisons'], translated['comparisons'])
    scrambled = producer.statistics({'0': base - 2, 'B': base, 'A': base[::-1] + 3})
    self.assertEqual(scrambled['comparisons']['A_minus_B']['mean'], 3.)
    self.assertNotEqual(scrambled['comparisons']['A_minus_B']['ci95'], [3., 3.])
    self.assertEqual(scrambled['comparisons']['A_minus_B']['classification'], 'inconclusive')
    zero = producer.statistics({'0': base, 'B': base, 'A': base})
    self.assertEqual(zero['comparisons']['A_minus_B']['ci_zero_side'], [0, 0])
    self.assertEqual(zero['comparisons']['A_minus_B']['classification'], 'positive_effect_not_supported')
    self.assertFalse(result['method_effectiveness_established'])
    self.assertFalse(result['independent_second_signal_validated'])

  def test_incomplete_nonfinite_or_extra_arm_statistics_rejected(self):
    valid = {'0': np.zeros(32), 'B': np.zeros(32), 'A': np.zeros(32)}
    for issue in ('missing', 'short', 'nonfinite', 'extra'):
      with self.subTest(issue=issue):
        values = copy.deepcopy(valid)
        if issue == 'missing':
          values.pop('A')
        elif issue == 'short':
          values['B'] = np.zeros(31)
        elif issue == 'nonfinite':
          values['A'][15] = np.nan
        else:
          values['R'] = np.zeros(32)
        with self.assertRaises((ValueError, KeyError)):
          producer.statistics(values)

  def test_full_snapshot_signed_zero_rng_clock_and_flags_require_exact_bytes(self):
    original = snapshot()
    producer.assert_snapshot_exact(copy.deepcopy(original), original)
    same = producer.snapshot_digest(copy.deepcopy(original))
    self.assertEqual(same, producer.snapshot_digest(original))
    for issue in ('signed_zero', 'integration_bit', 'rng', 'clock', 'reset', 'done', 'dtype'):
      with self.subTest(issue=issue):
        changed = copy.deepcopy(original)
        state = changed['suite']
        if issue == 'signed_zero':
          state['integration'][0] = -0.
        elif issue == 'integration_bit':
          state['integration'][1] = np.nextafter(1., 2.)
        elif issue == 'rng':
          state['task']['random'][0] += np.uint32(1)
        elif issue == 'clock':
          state['step_count'] = 1
        elif issue == 'reset':
          state['reset_next_step'] = True
        elif issue == 'done':
          changed['from_dm_done'] = True
        else:
          state['integration'] = state['integration'].astype(np.float32)
        with self.assertRaisesRegex(ValueError, 'snapshot bytes differ'):
          producer.assert_snapshot_exact(changed, original)
        self.assertNotEqual(producer.snapshot_digest(changed), same)

  def test_real_owned_child_exit_timeout_and_late_precreation(self):
    with tempfile.TemporaryDirectory() as directory:
      with (Path(directory) / 'normal.log').open('w') as log:
        result = producer.run_owned([sys.executable, '-c', 'print(42)'], log,
            time.monotonic() + 10, dict(os.environ))
      self.assertEqual(result['exit_code'], 0)
      self.assertTrue(result['worker_terminal'])
      with (Path(directory) / 'timeout.log').open('w') as log:
        result = producer.run_owned([sys.executable, '-c', 'import time;time.sleep(10)'],
            log, time.monotonic() + .2, dict(os.environ))
      self.assertEqual(result['exit_code'], 124)
      self.assertTrue(result['timed_out'])
      self.assertTrue(result['worker_terminal'])
      self.assertFalse(producer.OWNED)
      with mock.patch.object(producer.subprocess, 'Popen') as create:
        with self.assertRaisesRegex(ValueError, 'Late before'):
          producer.run_owned(['must_not_start'], None, time.monotonic() - 1, {})
        create.assert_not_called()

  @unittest.skipUnless(os.name == 'posix' and Path('/proc').is_dir(), 'Linux real process groups')
  def test_real_owned_descendants_cleaned_and_unrelated_process_preserved(self):
    sentinel = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(20)'],
        start_new_session=True)
    try:
      with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); child_pid = root / 'descendant.pid'
        body = ('import subprocess,sys,time,pathlib; '
            'child=subprocess.Popen([sys.executable,"-c","import time;time.sleep(20)"]); '
            'pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(20)')
        with (root / 'owned.log').open('w') as log:
          result = producer.run_owned([sys.executable, '-c', body, str(child_pid)],
              log, time.monotonic() + .6, dict(os.environ))
        self.assertEqual(result['exit_code'], 124)
        self.assertIsNone(sentinel.poll())
        descendant = Path('/proc') / child_pid.read_text().strip() / 'stat'
        deadline = time.monotonic() + 2
        while (descendant.exists() and descendant.read_text().split()[2] != 'Z' and
            time.monotonic() < deadline):
          time.sleep(.02)
        self.assertTrue(not descendant.exists() or descendant.read_text().split()[2] == 'Z')
        self.assertFalse(producer.OWNED)
    finally:
      os.killpg(sentinel.pid, signal.SIGKILL); sentinel.wait(timeout=3)

  def test_late_parent_receipt_or_rename_cannot_publish_complete(self):
    # Only injected envelope callbacks run; no scientific phase, CUDA/EGL,
    # source or model exists. The real publication guard sees a late clock.
    for late_at in ('write', 'rename'):
      with self.subTest(late_at=late_at), tempfile.TemporaryDirectory() as directory:
        root = Path(directory); resource = root / 'fixture-resource.json'
        resource.write_text('{}', encoding='utf-8')
        output = root / 'output'
        args = types.SimpleNamespace(output=str(output), expected_git='a' * 40,
            resource_record=str(resource), seal=None, task='dmc_quadruped_walk',
            **{k: 'fixture-only' for k in ('dataset', 'artifact_audit', 'physics_audit',
                'checkpoint', 'config', 'old_primary', 'old_primary_cli_terminal',
                'old_audit', 'old_audit_cli_terminal')})
        clock = [0.]
        write_original, replace_original = producer.write_json, producer.os.replace
        require_original = producer.require
        def run_fixture(*unused):
          write_original(output / 'result.json', dict(record_manifest=[{}] * 96,
              actual_physics_transitions=96000))
          return dict(worker_pid=12345, worker_terminal=True,
              owned_group_cleanup_requested=True, exit_code=0, timed_out=False, elapsed_seconds=.01)
        def write(path, value):
          write_original(path, value)
          if late_at == 'write' and Path(path).name == 'supervisor-receipt-pending.json':
            clock[0] = 2.
        def replace(source, target):
          replace_original(source, target)
          if late_at == 'rename' and Path(target).name == 'supervisor_complete.json':
            clock[0] = 2.
        def fixture_require(condition, message):
          if message != 'Only assigned sv3 may execute':
            require_original(condition, message)
        with mock.patch.object(producer, 'require', side_effect=fixture_require), \
            mock.patch.object(producer, 'ENTRY_STARTED', 0.), \
            mock.patch.object(producer, 'SECONDS', 1), \
            mock.patch.object(producer, 'PHASES', ('statistics',)), \
            mock.patch.object(producer.time, 'monotonic', side_effect=lambda: clock[0]), \
            mock.patch.object(producer, 'git_binding', return_value={}), \
            mock.patch.object(producer, 'verify_freeze', return_value={}), \
            mock.patch.object(producer, 'run_owned', side_effect=run_fixture), \
            mock.patch.object(producer, 'write_json', side_effect=write), \
            mock.patch.object(producer.os, 'replace', side_effect=replace):
          self.assertEqual(producer.parent(args), 124)
        self.assertFalse((output / 'supervisor_complete.json').exists())
        self.assertTrue((output / 'late-supervisor.json').exists())
        if late_at == 'rename':
          self.assertTrue((output / 'late-provisional-supervisor.json').exists())


if __name__ == '__main__':
  unittest.main()
