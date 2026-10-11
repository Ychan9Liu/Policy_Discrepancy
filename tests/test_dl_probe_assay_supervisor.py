"""Real stdlib subprocess fixtures; no assay, NumPy, model or physics fits.

Windows fixtures certify only the directly owned simple child. Linux fixtures
exercise descendant cleanup and an unrelated, separately owned sentinel.
Successful mock verification is only the generic callback contract, never a
real synthetic artifact-chain audit or permission for scientific progression.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_probe_assay_supervisor as supervisor

FIXTURE_GIT = '0' * 40


class ProbeAssaySupervisorTest(unittest.TestCase):
  def command(self, script):
    # Windows venv python.exe may redirect to another interpreter. Use the
    # direct binary for the limited directly-owned-child fixture assertion.
    executable = getattr(sys, '_base_executable', sys.executable) if os.name == 'nt' else sys.executable
    return [executable, '-c', script]

  def invoke(self, script, output, seconds=3, verify=None):
    return supervisor.supervise(self.command(script), output, FIXTURE_GIT,
        seconds=seconds, verify=verify)

  def assert_terminal(self, receipt):
    self.assertIs(receipt['worker_terminal'], True)
    self.assertIsInstance(receipt['worker_pid'], int)
    self.assertGreater(receipt['worker_pid'], 0)
    self.assertEqual(receipt['owned_group_cleanup_requested'], os.name != 'nt')
    self.assertGreaterEqual(receipt['elapsed_seconds'], 0)
    self.assertLess(receipt['elapsed_seconds'], 3)

  def test_real_success_retained_without_fake_completion_authority(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'success'
      marker = output / 'owned-worker-proof.json'
      script = ('import json,os; from pathlib import Path; '
          f'Path({str(marker)!r}).write_text(json.dumps(dict(pid=os.getpid(),parent=os.getppid()))); '
          'print("owned-success",flush=True)')
      receipt = self.invoke(script, output)
      self.assertEqual(receipt['exit_code'], 0)
      self.assertIs(receipt['timed_out'], False)
      self.assertIs(receipt['complete_verified'], False)
      self.assert_terminal(receipt)
      proof = json.loads(marker.read_text())
      self.assertEqual(proof['pid'], receipt['worker_pid'])
      self.assertEqual(proof['parent'], os.getpid())
      self.assertFalse((output / 'assay_complete.json').exists())
      self.assertFalse((output / 'summary.json').exists())
      logs = list(output.glob('*.log'))
      self.assertTrue(logs)
      self.assertIn('owned-success', '\n'.join(path.read_text() for path in logs))

  def test_real_nonzero_exit_retained_and_verify_not_called(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'failure'
      verify = mock.Mock(side_effect=AssertionError('Nonzero worker reached verification'))
      receipt = self.invoke('print("owned-failure",flush=True); raise SystemExit(7)',
          output, verify=verify)
      self.assertEqual(receipt['exit_code'], 7)
      self.assertIs(receipt['timed_out'], False)
      self.assertIs(receipt['complete_verified'], False)
      self.assert_terminal(receipt)
      verify.assert_not_called()
      self.assertIn('owned-failure', '\n'.join(path.read_text() for path in output.glob('*.log')))
      self.assertFalse((output / 'assay_complete.json').exists())

  def test_new_directory_required_without_launch_or_overwrite(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'existing'
      output.mkdir()
      preserved = output / 'existing-evidence'
      preserved.write_bytes(b'KEEP-EXACTLY')
      marker = Path(directory) / 'must-not-launch'
      verify = mock.Mock(side_effect=AssertionError('Existing output reached verification'))
      with self.assertRaises(FileExistsError):
        self.invoke(f'from pathlib import Path; Path({str(marker)!r}).write_text("BAD")',
            output, verify=verify)
      self.assertFalse(marker.exists())
      self.assertEqual(preserved.read_bytes(), b'KEEP-EXACTLY')
      self.assertEqual(list(output.iterdir()), [preserved])
      verify.assert_not_called()

  def test_direct_child_uncooperative_sleep_hard_timeout(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'timeout'
      marker = output / 'late-proof'
      verify = mock.Mock(side_effect=AssertionError('Timed-out worker reached verification'))
      script = ('import time; from pathlib import Path; time.sleep(10); '
          f'Path({str(marker)!r}).write_text("BAD")')
      started = time.monotonic()
      receipt = self.invoke(script, output, seconds=.15, verify=verify)
      self.assertEqual(receipt['exit_code'], 124)
      self.assertIs(receipt['timed_out'], True)
      self.assertIs(receipt['complete_verified'], False)
      self.assert_terminal(receipt)
      self.assertLess(time.monotonic() - started, 3)
      self.assertFalse(marker.exists())
      self.assertFalse((output / 'assay_complete.json').exists())
      verify.assert_not_called()

  def test_completion_callback_rejects_mock_hash_chain_after_child_exit_zero(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'rejected-completion'
      verify = mock.Mock(side_effect=ValueError('fixture-complete-hash-chain-rejected'))
      receipt = self.invoke('print("worker-exited-zero",flush=True)', output, verify=verify)
      self.assertNotEqual(receipt['exit_code'], 0)
      self.assertIs(receipt['timed_out'], False)
      self.assertIs(receipt['complete_verified'], False)
      self.assert_terminal(receipt)
      verify.assert_called_once()
      # A successful process exit cannot stand in for the explicit validator.
      self.assertIn('fixture-complete-hash-chain-rejected',
          json.dumps(receipt) + '\n'.join(path.read_text() for path in output.glob('*.log')))

  def test_mock_completion_verification_is_explicit_separate_callback(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'mock-verified'
      seen = []
      def verify(*args, **kwargs):
        seen.append((args, kwargs))
        return {'fixture_only': True}
      receipt = self.invoke('print("mock-only",flush=True)', output, verify=verify)
      self.assertEqual(receipt['exit_code'], 0)
      self.assertIs(receipt['timed_out'], False)
      self.assertIs(receipt['complete_verified'], True)
      self.assert_terminal(receipt)
      self.assertEqual(len(seen), 1)
      self.assertFalse((output / 'assay_complete.json').exists())

  @unittest.skipIf(os.name == 'nt', 'POSIX alarm contract; Windows direct-child wait only')
  def test_hard_alarm_covers_slow_verification_after_worker_completed(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'verification-timeout'
      def verify(*args, **kwargs):
        time.sleep(10)
      started = time.monotonic()
      receipt = self.invoke('print("finished-before-verify",flush=True)',
          output, seconds=.2, verify=verify)
      self.assertEqual(receipt['exit_code'], 124)
      self.assertIs(receipt['timed_out'], True)
      self.assertIs(receipt['complete_verified'], False)
      self.assert_terminal(receipt)
      self.assertLess(time.monotonic() - started, 3)

  def sentinel(self, marker):
    script = ('import time; from pathlib import Path; time.sleep(.5); '
        f'Path({str(marker)!r}).write_text("OK"); time.sleep(10)')
    return subprocess.Popen(self.command(script), start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

  @unittest.skipIf(os.name == 'nt', 'POSIX final-success-write hard alarm contract')
  def test_hard_alarm_covers_final_success_receipt_write(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'success-write-timeout'
      original_write = supervisor.write_json
      delayed = []
      def slow_final_success(path, value):
        if (Path(path).name == 'supervisor_complete.json' and
            value.get('complete_verified') is True and not delayed):
          delayed.append(True)
          # The failure receipt may be saved after timeout; only the first
          # success write is delayed, so evidence recovery itself can finish.
          time.sleep(10)
        return original_write(path, value)
      verify = mock.Mock(return_value={'fixture_only': True})
      started = time.monotonic()
      with mock.patch.object(supervisor, 'write_json', side_effect=slow_final_success):
        receipt = self.invoke('print("worker-finished-before-success-write",flush=True)',
            output, seconds=.2, verify=verify)
      verify.assert_called_once()
      self.assertEqual(delayed, [True])
      self.assertEqual(receipt['exit_code'], 124)
      self.assertIs(receipt['timed_out'], True)
      self.assertIs(receipt['complete_verified'], False)
      self.assert_terminal(receipt)
      self.assertLess(time.monotonic() - started, 3)
      stored = json.loads((output / 'supervisor_complete.json').read_text())
      self.assertEqual(stored['exit_code'], 124)
      self.assertIs(stored['complete_verified'], False)

  def cleanup_sentinel(self, sentinel):
    # A single cleanup function always terminates before waiting; no LIFO
    # ordering can accidentally wait on a still-live unrelated process.
    if sentinel.poll() is None:
      sentinel.terminate()
    try:
      sentinel.wait(timeout=3)
    except subprocess.TimeoutExpired:
      sentinel.kill()
      sentinel.wait(timeout=3)

  @unittest.skipIf(os.name == 'nt', 'Linux owned process groups and unrelated sentinel')
  def test_linux_failed_leader_cleans_descendant_preserves_sentinel(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'failed-owned-group'
      late = Path(directory) / 'owned-descendant-survived'
      tree = Path(directory) / 'owned-tree.json'
      sentinel_marker = Path(directory) / 'independent-sentinel'
      sentinel = self.sentinel(sentinel_marker)
      self.addCleanup(self.cleanup_sentinel, sentinel)
      descendant = ('import time; from pathlib import Path; time.sleep(.5); '
          f'Path({str(late)!r}).write_text("BAD"); time.sleep(10)')
      leader = ('import json,os,subprocess,sys; from pathlib import Path; '
          f'child=subprocess.Popen([sys.executable,"-c",{descendant!r}]); '
          f'Path({str(tree)!r}).write_text(json.dumps([os.getpid(),child.pid])); '
          'raise SystemExit(7)')
      try:
        receipt = self.invoke(leader, output)
        self.assertEqual(receipt['exit_code'], 7)
        self.assertIs(receipt['complete_verified'], False)
        self.assert_terminal(receipt)
        parent, child = json.loads(tree.read_text())
        self.assertEqual(parent, receipt['worker_pid'])
        self.assertNotEqual(parent, child)
        time.sleep(.7)
        self.assertFalse(late.exists(), 'Failed leader left a running owned descendant')
        self.assertIsNone(sentinel.poll(), 'Unrelated sentinel was killed')
        self.assertTrue(sentinel_marker.exists())
      finally:
        self.cleanup_sentinel(sentinel)

  @unittest.skipIf(os.name == 'nt', 'Linux owned process groups and unrelated sentinel')
  def test_linux_timeout_cleans_descendant_preserves_sentinel(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'timed-out-owned-group'
      late = Path(directory) / 'owned-descendant-survived'
      tree = Path(directory) / 'owned-tree.json'
      sentinel_marker = Path(directory) / 'independent-sentinel'
      sentinel = self.sentinel(sentinel_marker)
      self.addCleanup(self.cleanup_sentinel, sentinel)
      descendant = ('import time; from pathlib import Path; time.sleep(.5); '
          f'Path({str(late)!r}).write_text("BAD"); time.sleep(10)')
      leader = ('import json,os,subprocess,sys,time; from pathlib import Path; '
          f'child=subprocess.Popen([sys.executable,"-c",{descendant!r}]); '
          f'Path({str(tree)!r}).write_text(json.dumps([os.getpid(),child.pid])); '
          'time.sleep(10)')
      try:
        receipt = self.invoke(leader, output, seconds=.25)
        self.assertEqual(receipt['exit_code'], 124)
        self.assertIs(receipt['timed_out'], True)
        self.assertIs(receipt['complete_verified'], False)
        self.assert_terminal(receipt)
        parent, child = json.loads(tree.read_text())
        self.assertEqual(parent, receipt['worker_pid'])
        self.assertNotEqual(parent, child)
        time.sleep(.7)
        self.assertFalse(late.exists(), 'Timed-out worker left a running owned descendant')
        self.assertIsNone(sentinel.poll(), 'Unrelated sentinel was killed')
        self.assertTrue(sentinel_marker.exists())
      finally:
        self.cleanup_sentinel(sentinel)


if __name__ == '__main__':
  unittest.main()
