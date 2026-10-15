"""Small generated engineering fixtures, never actual full1M acceptance."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

from scripts import dl_normal_replay_probe as probe
from tests import test_dl_normal_replay as fixtures


class NormalReplayProbeTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    fixtures.NormalReplayTest.setUpClass()

  @classmethod
  def tearDownClass(cls):
    fixtures.NormalReplayTest.tearDownClass()

  def cache(self):
    return fixtures.NormalReplayTest().load()

  def test_fixture_cross_chunk_writeback_has_no_learning_or_real1m_claim(self):
    with mock.patch.dict(os.environ, probe.CPU_ENV):
      cache = self.cache()
      result = probe.exercise_cache(cache)
    self.assertEqual(len(result['plan']), 16)
    self.assertTrue(result['raw_fields_bytes_unchanged'])
    self.assertTrue(result['all65_cache_positions_checked'])
    self.assertTrue(result['pending_cleared'])
    self.assertFalse(result['network_output_used'])
    self.assertFalse(result['learning_performed'])
    self.assertFalse(result['real_1m_import_established'])
    self.assertFalse(cache.receipt()['original_inventory_authenticated'])
    self.assertLess(result['unique_written_stepids'], 16*64)
    actual = {p.name: p.read_bytes() for p in fixtures.NormalReplayTest.directory.iterdir()}
    self.assertEqual(actual, fixtures.NormalReplayTest.sourcebytes)

  def test_inspection_raw_pollution_is_rejected(self):
    cache, count = self.cache(), [0]
    original = cache.batch
    def polluted(starts):
      batch = original(starts)
      count[0] += 1
      if count[0] == 2:
        batch['reward'][0, 0] += np.float32(.25)
      return batch
    with mock.patch.dict(os.environ, probe.CPU_ENV), mock.patch.object(cache, 'batch', polluted):
      with self.assertRaisesRegex(ValueError, 'polluted raw field reward'):
        probe.exercise_cache(cache)

  def test_independent_readback_rejects_cache_damage_after_helper_validation(self):
    import elements
    cache = self.cache()
    original = cache.strict_update
    def damaged(out, submitted):
      result = original(out, submitted)
      first = out['stepid'][0, 0].tobytes()
      chunk = cache._replay.chunks[elements.UUID(first[:16])]
      chunk.data['dyn/deter'][int.from_bytes(first[16:], 'big'), 0] += np.float32(3)
      return result
    with mock.patch.dict(os.environ, probe.CPU_ENV), mock.patch.object(cache, 'strict_update', damaged):
      with self.assertRaisesRegex(ValueError, 'Independent stepid last-writer'):
        probe.exercise_cache(cache)

  def test_wrong_inventory_pin_fails_before_json_decode(self):
    with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, probe.CPU_ENV):
      path = Path(directory)/'inventory.json'
      path.write_bytes(b'not valid json')
      with mock.patch.object(probe.json, 'loads') as decode:
        with self.assertRaisesRegex(ValueError, 'SHA differs before decode'):
          probe.authenticated_inventory(path)
        decode.assert_not_called()

  def test_deadline_and_non_cpu_environment_rejected(self):
    with mock.patch.dict(os.environ, probe.CPU_ENV):
      probe.check()
      with mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0'):
        with self.assertRaisesRegex(ValueError, 'CPU environment changed'):
          probe.check()
      with mock.patch.object(probe, 'DEADLINE', time.monotonic()-1):
        with self.assertRaisesRegex(ValueError, '900s deadline'):
          probe.check()

  @unittest.skipUnless(sys.platform == 'linux', 'Linux real SIGALRM/owned process group contract')
  def test_real_alarm_kills_owned_helper_and_returns124(self):
    code = '''
import signal,sys,time
from scripts import dl_normal_replay_probe as p
from scripts import dl_live_trial as lifecycle
lifecycle.DEADLINE=time.monotonic()+.25
lifecycle.OWNED=p.OWNED
signal.signal(signal.SIGALRM,p.expire)
signal.setitimer(signal.ITIMER_REAL,.25)
child=lifecycle.spawn_owned([sys.executable,'-c','import time;time.sleep(60)'])
print(child.pid,flush=True)
time.sleep(60)
'''
    run = subprocess.run([sys.executable, '-c', code], cwd=probe.ROOT,
        capture_output=True, text=True, timeout=5)
    self.assertEqual(run.returncode, 124)
    pid = int(run.stdout.strip())
    status = Path(f'/proc/{pid}/stat')
    if status.exists():
      self.assertIn(status.read_text().rsplit(') ', 1)[1].split()[0], ('Z', 'X'))


if __name__ == '__main__':
  unittest.main()
