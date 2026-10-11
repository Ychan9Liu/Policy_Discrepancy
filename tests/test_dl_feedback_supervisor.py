"""Real CPU subprocess tests of the feedback hard supervisor, never GPU work.

Linux additionally checks owned process-group descendants and an independent
sentinel. Windows only certifies the directly owned simple fixture process.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('MUJOCO_GL', 'disable')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_feedback as feedback


class FeedbackSupervisorTest(unittest.TestCase):
  def command(self, script):
    # Windows venv python.exe is a redirector that creates another interpreter.
    # The simple stdlib child fixture must use the direct binary to certify the
    # directly owned process (Linux process-tree coverage is a separate test).
    executable = getattr(sys, '_base_executable', sys.executable) if os.name == 'nt' else sys.executable
    return [executable, '-c', script]

  def invoke(self, script, output, seconds):
    return feedback.supervise(self.command(script), output, seconds=seconds)

  def test_real_child_success_is_bounded_and_receipted_without_fake_result(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'fresh'
      marker = output / 'worker-proof.json'
      script = ('import json,os,time; from pathlib import Path; time.sleep(.02); '
          f'Path({str(marker)!r}).write_text(json.dumps(dict(pid=os.getpid(),'
          'parent=os.getppid(), supervisor=os.environ.get("DL_FEEDBACK_PARENT"))))')
      started = time.monotonic()
      receipt = self.invoke(script, output, seconds=3)
      self.assertEqual(receipt['exit_code'], 0)
      self.assertFalse(receipt['timed_out'])
      self.assertTrue(receipt['worker_terminal'])
      actual = json.loads(marker.read_text())
      self.assertEqual(actual['pid'], receipt['worker_pid'])
      self.assertEqual(actual['parent'], os.getpid())
      self.assertEqual(actual['supervisor'], str(os.getpid()))
      self.assertLess(time.monotonic() - started, 3)
      self.assertFalse((output / 'result.json').exists())
      self.assertFalse((output / 'feedback_complete.json').exists())

  def test_failure_exit_is_preserved_and_existing_output_never_reused(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'failure'
      receipt = self.invoke('print("owned-failure-proof",flush=True); '
          'raise SystemExit(7)', output, seconds=3)
      self.assertEqual(receipt['exit_code'], 7)
      self.assertFalse(receipt['timed_out'])
      self.assertTrue(receipt['worker_terminal'])
      logs = list(output.glob('*.log'))
      self.assertTrue(logs, 'Failure stdout/stderr evidence was not saved')
      self.assertIn('owned-failure-proof', '\n'.join(x.read_text() for x in logs))
      self.assertFalse((output / 'result.json').exists())
      marker = Path(directory) / 'must-not-run'
      with self.assertRaises(FileExistsError):
        self.invoke(f'from pathlib import Path; Path({str(marker)!r}).write_text("BAD")',
            output, seconds=3)
      self.assertFalse(marker.exists())

  def test_statistical_overflow_fails_closed_before_a_classification(self):
    # Finite inputs do not guarantee finite accumulated/derived statistics.
    # This is a contract counterexample, not a physically realistic reward.
    rewards = np.zeros((8, 4, 4, 2, 100), np.float64)
    rewards[..., 0, :] = 1e308
    with np.errstate(over='ignore', invalid='ignore'):
      with self.assertRaises(ValueError):
        feedback.analyze_utility(rewards, np.ones_like(rewards, bool),
            np.array([9, 14, 15, 18, 20, 24, 30, 31], np.int64))

  def test_hard_timeout_includes_work_outside_any_cooperative_loop(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'timeout'
      marker = output / 'late-result.json'
      script = ('import time; from pathlib import Path; time.sleep(10); '
          f'Path({str(marker)!r}).write_text("BAD")')
      started = time.monotonic()
      receipt = self.invoke(script, output, seconds=.15)
      elapsed = time.monotonic() - started
      self.assertEqual(receipt['exit_code'], 124)
      self.assertTrue(receipt['timed_out'])
      self.assertTrue(receipt['worker_terminal'])
      self.assertLess(elapsed, 3)
      self.assertFalse(marker.exists())
      self.assertFalse((output / 'result.json').exists())
      self.assertFalse((output / 'feedback_complete.json').exists())

  @unittest.skipIf(os.name == 'nt', 'Owned Linux process-group contract; Windows direct-child fixture only')
  def test_linux_failed_leader_cannot_leave_owned_descendant(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory)/'early-failure'
      escaped = Path(directory)/'escaped'
      descendant = ('import time; from pathlib import Path; time.sleep(.5); '
          f'Path({str(escaped)!r}).write_text("BAD")')
      leader = ('import subprocess,sys; '
          f'subprocess.Popen([sys.executable,"-c",{descendant!r}]); raise SystemExit(7)')
      receipt = self.invoke(leader,output,seconds=3)
      self.assertEqual(receipt['exit_code'],7)
      self.assertFalse(receipt['timed_out'])
      time.sleep(.8)
      self.assertFalse(escaped.exists(),'Failed owned leader left a running descendant')

  @unittest.skipIf(os.name == 'nt', 'Owned Linux process-group contract; Windows direct-child fixture only')
  def test_linux_kills_owned_grandchild_and_preserves_independent_sentinel(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'tree-timeout'
      sentinel_marker = Path(directory) / 'sentinel-survived'
      late_marker = Path(directory) / 'owned-grandchild-survived'
      tree_marker = Path(directory) / 'tree-pids.json'
      sentinel_script = ('import time; from pathlib import Path; time.sleep(.7); '
          f'Path({str(sentinel_marker)!r}).write_text("OK"); time.sleep(10)')
      sentinel = subprocess.Popen(self.command(sentinel_script), start_new_session=True,
          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      try:
        grandchild_script = ('import time; from pathlib import Path; time.sleep(.7); '
            f'Path({str(late_marker)!r}).write_text("BAD"); time.sleep(10)')
        child_script = ('import json,os,subprocess,sys,time; from pathlib import Path; '
            f'child=subprocess.Popen([sys.executable,"-c",{grandchild_script!r}]); '
            f'Path({str(tree_marker)!r}).write_text(json.dumps([os.getpid(),child.pid])); '
            'time.sleep(10)')
        receipt = self.invoke(child_script, output, seconds=.25)
        self.assertEqual(receipt['exit_code'], 124)
        self.assertTrue(receipt['worker_terminal'])
        parent, grandchild = json.loads(tree_marker.read_text())
        self.assertEqual(parent, receipt['worker_pid'])
        self.assertNotEqual(grandchild, parent)
        time.sleep(.8)
        self.assertFalse(late_marker.exists(), 'Owned grandchild escaped the hard timeout')
        self.assertIsNone(sentinel.poll(), 'Unrelated independent sentinel was killed')
        self.assertTrue(sentinel_marker.exists())
      finally:
        if sentinel.poll() is None:
          sentinel.terminate()
        sentinel.wait(timeout=5)


if __name__ == '__main__':
  unittest.main()
