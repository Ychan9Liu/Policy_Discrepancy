"""CPU metadata/path/failure tests; no real run, pickle, GPU or server access."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_r4_failure_capture as c

AUDIT_GIT = 'a' * 40
SOURCEPINS = {'scripts/dl_r4_utility.py': hashlib.sha256(b'fixture frozen source').hexdigest()}
BINDING = {'sourcepins': SOURCEPINS, 'audit_sourcepins': {c.SELF: 'b' * 64}}


def write_json(path, value):
  path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def file_sha(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


def absent_processes():
  return {name: dict(pid=pid, expected_starttime_ticks=start,
      checked_utc='2026-10-15T08:00:00+00:00', present=False,
      alive=False, identity_status='absent') for name, (pid, start) in c.IDENTITIES.items()}


def make_failure_run(path):
  """Minimal terminal metadata authority fixture, not scientific branch coverage.

  Independent auditor tests may append their own opaque artifacts before taking
  inventories. This helper never claims these bytes are actual production data.
  """
  path = Path(path).absolute()
  path.mkdir()
  write_json(path / 'budget.json', dict(version=c.RUN_VERSION, git_sha=c.RUN_GIT,
      parent_pid=c.IDENTITIES['parent'][0], source_files=copy.deepcopy(SOURCEPINS)))
  write_json(path / 'process.json', dict(parent_pid=c.IDENTITIES['parent'][0],
      worker_pid=c.IDENTITIES['worker'][0], worker_pgid=c.IDENTITIES['worker'][0],
      command=['/fixture/python', '/fixture/dl_r4_utility.py', 'worker',
          '--expected-git', c.RUN_GIT, '--output', str(path)]))
  (path / 'resources.jsonl').write_text('{"fixture":true}\n', encoding='utf-8')
  log = '\n'.join(json.dumps(dict(event='episode_complete', episode_id=i)) for i in range(7))
  (path / 'worker.log').write_text(log + '\nTraceback (most recent call last):\n'
      '  fixture traceback, not a real run\n' + c.ERROR + '\n', encoding='utf-8')
  evidence = []
  for episode in range(8):
    directory = path / f'episode_{episode:03}'
    directory.mkdir()
    write_json(directory / 'complete.json', dict(version=c.RUN_VERSION,
        episode_id=episode, model_full_state_unchanged=True))
    # These deliberately invalid pickle bytes prove the capture only hashes.
    (directory / 'opaque.pkl').write_bytes(b'not a pickle\x00' + bytes([episode]))
    evidence.append(dict(path=f'episode_{episode:03}/complete.json',
        sha256=file_sha(directory / 'complete.json')))
  write_json(path / 'calibration_frozen.json', dict(version=c.RUN_VERSION,
      frozen_models={'null_passed': False}, held_episodes_generated=0,
      calibration_evidence=evidence))
  write_json(path / 'supervisor_terminal.json', dict(version=c.RUN_VERSION,
      git_sha=c.RUN_GIT, worker_returncode=1, worker_terminal=True,
      elapsed_seconds=2622.79, worker_seconds=2621.12,
      worker_complete_sha256=None, worker_log_sha256=file_sha(path / 'worker.log'),
      resources_sha256=file_sha(path / 'resources.jsonl'), parent_failure=None,
      main_generation_complete=False, independent_audit_complete=False,
      scientific_support_claimed=False, owned_worker_group_cleaned=True))
  return path


class FailureCaptureTest(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.directory = Path(temporary.name)
    self.run = make_failure_run(self.directory / 'run')
    self.output = self.directory / 'capture.json'
    self.git = mock.patch.object(c, 'git_binding', return_value=copy.deepcopy(BINDING)).start()
    self.processes = mock.patch.object(c, 'process_set', side_effect=lambda: absent_processes()).start()
    self.addCleanup(mock.patch.stopall)

  def capture(self, **kwargs):
    return c.capture(kwargs.pop('run', self.run), kwargs.pop('output', self.output),
        kwargs.pop('run_git', c.RUN_GIT), kwargs.pop('audit_git', AUDIT_GIT), **kwargs)

  def metadata(self, name, change):
    path = self.run / name
    value = json.loads(path.read_text(encoding='utf-8'))
    change(value)
    write_json(path, value)

  def rehash_log(self):
    self.metadata('supervisor_terminal.json',
        lambda value: value.update(worker_log_sha256=file_sha(self.run / 'worker.log')))

  def test_complete_external_capture_all_opaque_bytes_two_passes_and_optional_pin(self):
    original, dirs = c.inventory(self.run)
    outside = self.directory / 'original-launch.json'
    outside.write_bytes(b'{"fixture_external_launch":true}\n')
    report = self.capture(external_pins=[outside])
    self.assertEqual(report['version'], c.VERSION)
    self.assertEqual(report['before_inventory'], original)
    self.assertEqual(report['after_inventory'], original)
    self.assertEqual(report['before_directories'], dirs)
    self.assertEqual(report['after_directories'], dirs)
    self.assertEqual(report['completed_episode_ids'], list(range(8)))
    self.assertEqual(report['held_episode_directories'], [])
    self.assertEqual(report['sourcepins'], SOURCEPINS)
    self.assertEqual(report['run_git'], c.RUN_GIT)
    self.assertEqual(report['audit_git'], AUDIT_GIT)
    self.assertTrue(report['stable'])
    for field in ('producer_closure_claimed', 'scientific_audit_complete',
        'pickle_decoded', 'run_files_written', 'calibration_null_passed'):
      self.assertFalse(report[field], field)
    self.assertEqual(report['terminal_class'], 'calibration_null_measurement_failed_held_unopened')
    self.assertEqual(report['final_error'], c.ERROR)
    self.assertEqual(report['external_pins'][0]['sha256'], file_sha(outside))
    self.assertEqual(c.inventory(self.run), (original, dirs))
    self.assertEqual(json.loads(self.output.read_text()), report)
    self.assertEqual(self.git.call_count, 2)
    self.assertEqual(self.processes.call_count, 2)
    self.assertEqual(set(report['process_observations']), {'before', 'after'})
    self.assertEqual(report['worker_log']['sha256'], file_sha(self.run / 'worker.log'))
    self.assertIn('episode_000/opaque.pkl', [p['path'] for p in report['before_inventory']])

  def test_wrong_git_and_live_unknown_or_mismatched_process_refused(self):
    for bad in ('0' * 40, '5ce1d37'):
      with self.assertRaises(ValueError): self.capture(run_git=bad)
    for bad in ('a' * 39, 'A' * 40):
      with self.assertRaises(ValueError): self.capture(audit_git=bad)
    for change in ('alive', 'present', 'identity', 'pid', 'missing'):
      observed = absent_processes()
      if change == 'alive': observed['worker']['alive'] = True
      elif change == 'present': observed['worker']['present'] = True
      elif change == 'identity': observed['worker']['identity_status'] = 'permission_denied'
      elif change == 'pid': observed['worker']['pid'] += 1
      else: observed.pop('worker')
      with self.subTest(change=change), mock.patch.object(c, 'process_set', return_value=observed):
        with self.assertRaises(ValueError): self.capture()
    self.assertFalse(self.output.exists())

  def test_fresh_output_outside_run_and_no_overwrite(self):
    with self.assertRaisesRegex(ValueError, 'outside failed run'):
      self.capture(output=self.run / 'capture.json')
    self.output.write_bytes(b'keep this report')
    with self.assertRaisesRegex(ValueError, 'Fresh external'):
      self.capture()
    self.assertEqual(self.output.read_bytes(), b'keep this report')

  def test_terminal_and_log_tampering_rejected_even_with_new_inventory_hashes(self):
    terminal = (self.run / 'supervisor_terminal.json').read_bytes()
    for field, value in (('worker_returncode', 0), ('worker_returncode', True),
        ('worker_terminal', False), ('main_generation_complete', True),
        ('independent_audit_complete', True), ('scientific_support_claimed', True),
        ('owned_worker_group_cleaned', False), ('parent_failure', 'injected'),
        ('worker_complete_sha256', 'e' * 64), ('elapsed_seconds', 21001),
        ('worker_log_sha256', '0' * 64), ('resources_sha256', '0' * 64)):
      (self.run / 'supervisor_terminal.json').write_bytes(terminal)
      self.metadata('supervisor_terminal.json', lambda row: row.update({field: value}))
      with self.subTest(field=field, value=value), self.assertRaises(ValueError): self.capture()
    (self.run / 'supervisor_terminal.json').write_bytes(terminal)
    old_log = (self.run / 'worker.log').read_bytes()
    for damaged in (old_log.replace(c.ERROR.encode(), b'ValueError: another failure'),
        old_log + b'continued after failure\n',
        old_log.replace(b'"episode_id": 6', b'"episode_id": 7')):
      (self.run / 'worker.log').write_bytes(damaged)
      self.rehash_log()
      with self.assertRaises(ValueError): self.capture()
    self.assertFalse(self.output.exists())

  def test_held_statistics_missing_cal_and_gate_metadata_refused(self):
    for path in ('episode_008', 'worker_complete.json', 'statistics_input.json',
        'statistics_provisional.json', 'statistics_unexpected.json'):
      target = self.run / path
      if path.startswith('episode_'): target.mkdir()
      else: target.write_bytes(b'{}')
      with self.subTest(path=path), self.assertRaises(ValueError): self.capture()
      if target.is_dir(): target.rmdir()
      else: target.unlink()
    directory = self.run / 'episode_007'
    complete = (directory / 'complete.json').read_bytes()
    (directory / 'complete.json').unlink()
    with self.assertRaises(ValueError): self.capture()
    (directory / 'complete.json').write_bytes(complete)
    frozen = (self.run / 'calibration_frozen.json').read_bytes()
    for change in ('nulltrue', 'held', 'evidencehash', 'coverage'):
      (self.run / 'calibration_frozen.json').write_bytes(frozen)
      def alter(row):
        if change == 'nulltrue': row['frozen_models']['null_passed'] = True
        elif change == 'held': row['held_episodes_generated'] = 1
        elif change == 'evidencehash': row['calibration_evidence'][0]['sha256'] = 'f' * 64
        else: row['calibration_evidence'].pop()
      self.metadata('calibration_frozen.json', alter)
      with self.subTest(change=change), self.assertRaises(ValueError): self.capture()
    self.assertFalse(self.output.exists())

  def test_source_pins_and_original_command_cannot_be_relabelled(self):
    for filename, field, value in (
        ('budget.json', 'git_sha', '0' * 40),
        ('budget.json', 'source_files', {}),
        ('budget.json', 'parent_pid', 1),
        ('process.json', 'worker_pid', 1),
        ('process.json', 'worker_pgid', 1),
        ('process.json', 'command', ['/x', '/x', 'worker', '--expected-git', c.RUN_GIT,
            '--output', str(self.directory / 'another')])):
      path = self.run / filename
      original = path.read_bytes()
      self.metadata(filename, lambda row: row.update({field: value}))
      with self.subTest(field=field), self.assertRaises(ValueError): self.capture()
      path.write_bytes(original)

  def test_changed_file_or_empty_directory_between_passes_is_refused(self):
    original = c.inventory
    for kind in ('content', 'directory'):
      calls = []
      def racing(root):
        calls.append(True)
        if len(calls) == 2:
          if kind == 'content': (self.run / 'episode_000/opaque.pkl').write_bytes(b'changed')
          else: (self.run / 'new-empty-directory').mkdir()
        return original(root)
      with self.subTest(kind=kind), mock.patch.object(c, 'inventory', side_effect=racing):
        with self.assertRaisesRegex(ValueError, 'inventory changed'):
          self.capture()
      self.assertFalse(self.output.exists())
      if kind == 'directory': (self.run / 'new-empty-directory').rmdir()

  def test_metadata_changed_after_inventory_is_refused(self):
    original = c.validate_failure
    def changed_metadata(root, *args):
      self.metadata('process.json', lambda row: row.update(worker_pid=1))
      return original(root, *args)
    with mock.patch.object(c, 'validate_failure', side_effect=changed_metadata):
      with self.assertRaisesRegex(ValueError, 'Metadata changed'):
        self.capture()
    self.assertFalse(self.output.exists())

  def test_symlink_run_entry_output_and_parent_paths_are_refused(self):
    target = self.directory / 'outside.bin'
    target.write_bytes(b'outside')
    link = self.run / 'bad-link'
    try:
      link.symlink_to(target)
    except OSError:
      self.skipTest('Symlink creation unavailable on this CPU test host')
    with self.assertRaisesRegex(ValueError, 'symlink|Symlink'):
      self.capture()
    link.unlink()
    linked_root = self.directory / 'linked-run'
    linked_root.symlink_to(self.run, target_is_directory=True)
    with self.assertRaisesRegex(ValueError, 'Symlink'):
      self.capture(run=linked_root)
    linked_output = self.directory / 'linked-output.json'
    linked_output.symlink_to(target)
    with self.assertRaisesRegex(ValueError, 'Symlink'):
      self.capture(output=linked_output)

  def test_duplicate_nonfinite_and_unsafe_relative_metadata_refused(self):
    for data in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e9999}'):
      with self.assertRaises(ValueError): c._json_bytes(data)
    for name in ('/escape', '../escape', 'a/../b', 'a//b', './x', 'a\\b', ''):
      with self.assertRaises(ValueError): c._relative(name)

  def test_host_only_import_and_bad_git_cli_fail_before_proc_or_output(self):
    code = "import os,sys; before=dict(os.environ); import scripts.dl_r4_failure_capture; assert before==dict(os.environ); assert 'numpy' not in sys.modules; assert 'jax' not in sys.modules; assert not any(x.startswith('dm_control') for x in sys.modules)"
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
        capture_output=True, text=True, timeout=30)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    result = subprocess.run([sys.executable, str(Path(c.__file__).resolve()),
        '--run', str(self.run), '--output', str(self.output), '--run-git', '0' * 40,
        '--audit-git', AUDIT_GIT], capture_output=True, text=True, timeout=30)
    self.assertNotEqual(result.returncode, 0)
    self.assertFalse(self.output.exists())


class GitAndProcessTests(unittest.TestCase):
  def archive(self, symlink=False):
    value = io.BytesIO()
    with tarfile.open(fileobj=value, mode='w') as stream:
      for name in ('scripts/dl_r4_utility.py', 'scripts/dl_r4_physics.py',
          'scripts/dl_r4_policy.py', 'AGENTS.md', 'docs/DL_R4_UTILITY.md', 'not-selected.png'):
        item = tarfile.TarInfo(name)
        data = name.encode()
        if symlink and name.endswith('dl_r4_policy.py'):
          item.type, item.linkname = tarfile.SYMTYPE, '/outside'
          stream.addfile(item)
        else:
          item.size = len(data)
          stream.addfile(item, io.BytesIO(data))
    return value.getvalue()

  def test_git_blob_archive_source_pins_and_clean_status_contract(self):
    for defect in (None, 'dirty', 'head', 'ownblob', 'symlink'):
      def run(command, **kwargs):
        args = command[3:]
        if args == ['rev-parse', '--show-toplevel']: out = str(c.ROOT).encode() + b'\n'
        elif args == ['rev-parse', 'HEAD']: out = (('0' * 40) if defect == 'head' else AUDIT_GIT).encode() + b'\n'
        elif args == ['status', '--porcelain=v1', '--untracked-files=all']: out = b'?? dirty\n' if defect == 'dirty' else b''
        elif args == ['show', AUDIT_GIT + ':' + c.SELF]: out = b'wrong' if defect == 'ownblob' else Path(c.__file__).read_bytes()
        elif args == ['archive', '--format=tar', c.RUN_GIT]: out = self.archive(defect == 'symlink')
        else: raise AssertionError('Unexpected Git query: ' + repr(args))
        return SimpleNamespace(returncode=0, stdout=out, stderr=b'')
      with self.subTest(defect=defect), mock.patch.object(c.subprocess, 'run', side_effect=run):
        if defect is not None:
          with self.assertRaises(ValueError): c.git_binding(c.RUN_GIT, AUDIT_GIT)
        else:
          result = c.git_binding(c.RUN_GIT, AUDIT_GIT)
          self.assertEqual(len(result['sourcepins']), 5)
          self.assertNotIn('not-selected.png', result['sourcepins'])
          self.assertEqual(result['audit_sourcepins'][c.SELF], file_sha(Path(c.__file__)))

  def test_present_original_pid_and_permission_ambiguity_are_never_accepted(self):
    # This fixture does not signal a process: mocked stat/cmdline paths emulate
    # each observation and the only kill API call below is a mocked signal 0.
    absent = FileNotFoundError('absent')
    with mock.patch.object(Path, 'read_text', side_effect=absent), \
        mock.patch.object(Path, 'exists', return_value=False), \
        mock.patch.object(c.os, 'kill', side_effect=ProcessLookupError) as probe:
      result = c.process_observation(247014, 407058309)
      self.assertFalse(result['alive'])
      probe.assert_called_once_with(247014, 0)
    for error in (PermissionError, None):
      with mock.patch.object(Path, 'read_text', side_effect=absent), \
          mock.patch.object(Path, 'exists', return_value=False), \
          mock.patch.object(c.os, 'kill', side_effect=error):
        with self.assertRaises(ValueError): c.process_observation(247014, 407058309)
    tail = ['S'] + ['1'] * 18 + ['407058309'] + ['0'] * 10
    data = '247014 (fixture process name) ' + ' '.join(tail)
    for state in ('S', 'Z'):
      fake = data.replace(') S ', ') ' + state + ' ')
      with mock.patch.object(Path, 'read_text', return_value=fake), \
          mock.patch.object(Path, 'read_bytes', return_value=b'fixture\x00command'):
        with self.assertRaisesRegex(ValueError, 'Original PID still present'):
          c.process_observation(247014, 407058309)


if __name__ == '__main__':
  unittest.main()
