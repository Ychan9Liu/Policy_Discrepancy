"""DL-r5 supervisor boundaries, with no real checkpoint/model/render/GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from scripts import dl_live_trial as r


class LiveTrialTests(unittest.TestCase):
  def test_json_overwrite_late_write_and_duplicate_guards(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)/'record.json'
      with mock.patch.object(r, 'DEADLINE', time.monotonic()-1):
        with self.assertRaisesRegex(ValueError, 'deadline'):
          r.write_json(path, dict(complete=True))
      self.assertFalse(path.exists())
      r.write_json(path, dict(complete=False))
      with self.assertRaises(FileExistsError):
        r.write_json(path, dict(complete=True))
      self.assertFalse(r.read_json(path)['complete'])
      for text in ('{"a":1,"a":2}', '{"a":NaN}'):
        path.write_text(text)
        with self.assertRaises(ValueError):
          r.read_json(path)

  def test_wrong_checkpoint_rejected_without_decoding(self):
    with tempfile.TemporaryDirectory() as directory:
      checkpoint, config = (Path(directory)/x for x in ('agent.pkl', 'config.yaml'))
      checkpoint.write_bytes(b'not a checkpoint')
      config.write_bytes(b'not the frozen config')
      with self.assertRaisesRegex(ValueError, 'checkpoint bytes'):
        r.authenticate_inputs(checkpoint, config)
      self.assertNotIn('jax', r.__dict__)

  def test_late_process_is_never_created(self):
    with mock.patch.object(r.subprocess, 'Popen') as popen:
      with self.assertRaisesRegex(ValueError, 'Late process'):
        r.spawn_owned(['unused'], deadline=time.monotonic()-1)
      popen.assert_not_called()

  def test_git_blobs_reject_hidden_worktree_tamper(self):
    # skip-worktree can hide a modified working file from git status. Verify
    # actual Git blob bytes, rather than accepting a clean-index claim.
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      names = ('scripts/dl_live_trial.py', 'scripts/dl_live_family.py',
          'scripts/dl_live_policy.py', 'scripts/dl_live_shadow.py',
          'analysis/dl_live_statistics.py', 'analysis/dl_live_audit.py',
          'docs/DL_R5_LIVE_TRIAL.md', 'AGENTS.md')
      for name in names:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'original\n')
      def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.STDOUT)
      git('init')
      git('add', '.')
      git('-c', 'user.name=DL test', '-c', 'user.email=fixture@example.invalid', 'commit', '-m', 'fixture')
      expected = git('rev-parse', 'HEAD').decode().strip()
      with mock.patch.object(r, 'ROOT', root):
        pins = r.source_binding(expected)
        self.assertEqual(set(pins), set(names))
        git('update-index', '--skip-worktree', 'scripts/dl_live_policy.py')
        (root/'scripts/dl_live_policy.py').write_bytes(b'changed\n')
        self.assertFalse(git('status', '--porcelain'))
        with self.assertRaisesRegex(ValueError, 'Checkout/blob mismatch'):
          r.source_binding(expected)

  def test_unknown_gpu_process_never_counts_as_owned(self):
    state = dict(gpu_uuid=r.GPU_UUID, processes=[dict(pid=11, type='C')],
        pmon_jobs=[dict(pid=11, type='G')])
    with mock.patch('scripts.dl_resources.live_state', return_value=state):
      self.assertEqual(r.sample_resource(11)['worker_pid'], 11)
      with self.assertRaisesRegex(ValueError, 'Unknown work'):
        r.sample_resource(12)
    with mock.patch('scripts.dl_resources.live_state', return_value=dict(state, gpu_uuid='other')):
      with self.assertRaisesRegex(ValueError, 'UUID differs'):
        r.sample_resource(11)

  def test_live_parent_budget_rejects_pid_reuse_hash_and_source_tamper(self):
    with tempfile.TemporaryDirectory() as directory:
      path, resource = (Path(directory)/name for name in ('budget.json', 'resource.json'))
      resource.write_text('{}')
      deadline = float(time.monotonic()+100)
      budget = dict(version=r.VERSION, seconds=7200, git_sha='a'*40,
          logical_server='sv3', hostname=r.HOST, physical_gpu=4, gpu_uuid=r.GPU_UUID,
          monotonic_deadline=deadline, parent_pid=42, parent_start_ticks=50,
          parent_pgid=40, source_files={'source': 'pin'}, input_pins={},
          resource_record=r.file_pin(resource))
      r.write_json(path, budget)
      identity = dict(pid=42, start_ticks=50, pgid=40, state='S')
      with mock.patch.object(r, 'require_host'), \
          mock.patch.object(r.socket, 'gethostname', return_value=r.HOST), \
          mock.patch.object(r.os, 'getppid', return_value=42), \
          mock.patch.object(r.os, 'getpid', return_value=70), \
          mock.patch.object(r.os, 'getpgid', return_value=70, create=True), \
          mock.patch.object(r, 'process_identity', return_value=identity) as proc, \
          mock.patch.object(r, 'source_binding', return_value={'source': 'pin'}) as source, \
          mock.patch.object(r, 'authenticate_inputs', return_value={}), \
          mock.patch.object(r, 'DEADLINE', deadline), \
          mock.patch.dict(os.environ, DL_R5_PARENT='42', DL_R5_BUDGET_SHA256=r.sha(path)):
        actual, checked = r.verify_budget(path, 'a'*40, 'unused', 'unused', resource)
        self.assertEqual(actual, budget)
        checked()
        proc.return_value = dict(identity, start_ticks=51)
        with self.assertRaisesRegex(ValueError, 'parent identity changed'):
          checked()
        proc.return_value = identity
        source.return_value = {'source': 'changed'}
        with self.assertRaisesRegex(ValueError, 'source bytes differ'):
          r.verify_budget(path, 'a'*40, 'unused', 'unused', resource)
        source.return_value = {'source': 'pin'}
        with mock.patch.dict(os.environ, DL_R5_BUDGET_SHA256='b'*64):
          with self.assertRaisesRegex(ValueError, 'budget raw hash differs'):
            r.verify_budget(path, 'a'*40, 'unused', 'unused', resource)

  def test_parent_calibration_seal_precedes_all_held_and_failure_stops(self):
    # Explicitly fake family/model data: this proves supervisor chronology,
    # never real-model execution or statistical/causal validity.
    for fail_calibration in (False, True):
      with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)/'run'
        args = mock.Mock(output=str(output), expected_git='a'*40,
            checkpoint='unused', config='unused', resource_record='unused')
        launched = []
        def family(args, root, index, budget):
          if index >= 8:
            seal = r.read_json(root/'calibration_frozen.json')
            self.assertEqual(seal['sealed_event'], 384)
            self.assertEqual(len(seal['calibration_units']), 128)
            self.assertEqual(seal['held_families_started'], 0)
          launched.append(index)
          return {}, dict(path=f'family_{index:02}/terminal.json', sha256='b'*64)
        def freeze(rows):
          self.assertEqual(len(rows), 128)
          self.assertEqual(launched, list(range(8)))
          if fail_calibration:
            raise ValueError('fixture cal failure')
          return {'frozen': True}
        with mock.patch.object(r, 'require_host'), \
            mock.patch.object(r, 'source_binding', return_value={}), \
            mock.patch.object(r, 'authenticate_inputs', return_value={}), \
            mock.patch.object(r, 'file_pin', side_effect=lambda p: {'path': str(p), 'sha256': 'c'*64, 'bytes': 0}), \
            mock.patch.object(r, 'process_identity', return_value=dict(start_ticks=1, pgid=2)), \
            mock.patch('scripts.dl_resources.require_resource', return_value=dict(physical_gpu=4, gpu_uuid=r.GPU_UUID)), \
            mock.patch.object(r, 'run_family', side_effect=family), \
            mock.patch.object(r, 'family_rows', side_effect=lambda root, index, complete: [dict(f=index, k=k) for k in range(16)]), \
            mock.patch('analysis.dl_live_statistics.freeze_calibration', side_effect=freeze), \
            mock.patch('analysis.dl_live_statistics.analyze', return_value={'support': False}), \
            mock.patch('builtins.print'):
          if fail_calibration:
            with self.assertRaisesRegex(ValueError, 'failed/incomplete'):
              r.parent(args)
          else:
            r.parent(args)
        self.assertEqual(launched, list(range(8 if fail_calibration else 32)))
        terminal = r.read_json(output/'supervisor_terminal.json')
        self.assertEqual(terminal['main_generation_complete'], not fail_calibration)
        self.assertFalse(terminal['scientific_support_claimed'])
        self.assertFalse(terminal['independent_audit_complete'])

  def test_family_rows_require_actual_matching_artifact_and_no_escape(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      family_dir = root/'family_00'
      family_dir.mkdir()
      entries = []
      for k in range(16):
        path = family_dir/f'unit_{k:02}.json'
        row = dict(family_id=0, unit_id=k)
        r.write_json(path, dict(version='DL-r5-live-family-r1', unit=k, unit_row=row))
        entries.append(dict(path=path.name, sha256=r.sha(path), unit=k, unit_row=row))
      complete = dict(version='DL-r5-live-family-r1', family=0, units=entries)
      self.assertEqual(len(r.family_rows(root, 0, complete)), 16)
      entries[5]['unit_row'] = dict(family_id=0, unit_id=6)
      with self.assertRaisesRegex(ValueError, 'identity differs'):
        r.family_rows(root, 0, complete)
      entries[5]['path'] = '../outside.json'
      r.write_json(root/'outside.json', {})
      with self.assertRaisesRegex(ValueError, 'path/hash differs'):
        r.family_rows(root, 0, complete)

  @unittest.skipUnless(os.name == 'posix', 'Linux process group contract')
  def test_owned_family_group_also_cleans_cpu_shadow(self):
    child = r.spawn_owned([sys.executable, '-c',
        'import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c",'
        '"import time;time.sleep(60)"]);print(p.pid,flush=True);time.sleep(60)'],
        stdout=subprocess.PIPE, text=True)
    try:
      shadow = int(child.stdout.readline())
      self.assertEqual(os.getpgid(shadow), child.pid)
    finally:
      r.cleanup(child)
      child.stdout.close()
    self.assertNotIn(child.pid, r.OWNED)
    try:
      self.assertEqual(r.process_identity(shadow)['state'], 'Z')
    except FileNotFoundError:
      pass


if __name__ == '__main__':
  unittest.main()
