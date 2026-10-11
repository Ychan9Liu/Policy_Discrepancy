"""Synthetic producer envelope checks, never actual source/model evidence."""
import hashlib
import os
import signal
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock
import numpy as np
from scripts import dl_reset_joint as p


class ProducerTests(unittest.TestCase):
  def test_generated_payload_hash_before_decode(self):
    with tempfile.TemporaryDirectory() as d:
      path=Path(d)/'value.pkl'
      pin=p.save_pickle(path,{'good':42})
      self.assertEqual(p.load_generated(path,pin),{'good':42})
      path.write_bytes(b'not a pickle')
      with self.assertRaisesRegex(ValueError,'changed'):p.load_generated(path,pin)

  def test_all_fields_exact_promotion(self):
    import ml_dtypes
    v=dict(h=np.ones((8,4096),ml_dtypes.bfloat16),zq=np.ones((8,4,32,32),ml_dtypes.bfloat16),
        zp=np.zeros((8,4,32,32),ml_dtypes.bfloat16),D=np.arange(8,dtype=np.float32),
        K=np.zeros(8,np.float32))
    out=p.numeric_bundle(v,p.CAL,np.zeros((8,7),np.float64))
    self.assertEqual(set(out),{'episode_ids','h','zq','zp','returns','D','K'})
    for key in ('h','zq','zp','D','K','returns'):self.assertEqual(out[key].dtype,np.float64)
    np.testing.assert_array_equal(out['h'],v['h'].astype(np.float64))

  def test_freeze_tamper_rejected(self):
    with tempfile.TemporaryDirectory() as d:
      root=Path(d)
      names=[f'f{i}' for i in range(8)]
      for name in names:(root/name).write_bytes(name.encode())
      value=dict(version=p.VERSION,actual_git_sha='a'*40,blind_arrays_read=False,
          cal_episodes=list(p.CAL),blind_episodes=list(p.BLIND),groups=['0','B128','R128'],
          updates_per_arm=128,statistics_seed=20261026,source_files_sha256={},
          files={name:p.sha(root/name) for name in names})
      p.write_json(root/'freeze.json',value)
      with mock.patch.object(p,'git_binding',return_value={}):p.verify_freeze(root)
      (root/'f3').write_bytes(b'changed')
      with self.assertRaisesRegex(ValueError,'Frozen file changed'):p.verify_freeze(root)

  def test_child_exit_and_timeout(self):
    with tempfile.TemporaryDirectory() as d:
      with (Path(d)/'log').open('w') as log:
        result=p.run_owned([sys.executable,'-c','print(42)'],log,time.monotonic()+10,dict(os.environ))
      self.assertEqual(result['exit_code'],0);self.assertTrue(result['worker_terminal'])
      with (Path(d)/'slow').open('w') as log:
        result=p.run_owned([sys.executable,'-c','import time;time.sleep(10)'],log,
            time.monotonic()+.25,dict(os.environ))
      self.assertEqual(result['exit_code'],124);self.assertTrue(result['timed_out'])
      self.assertTrue(result['worker_terminal']);self.assertFalse(p.OWNED)

  def test_complete_episode_split(self):
    self.assertEqual(len(p.CAL),8);self.assertEqual(len(p.BLIND),24)
    self.assertFalse(set(p.CAL)&set(p.BLIND));self.assertEqual(set(p.CAL)|set(p.BLIND),set(range(32)))

  @unittest.skipUnless(os.name == 'posix' and Path('/proc').is_dir(),'Linux process inventory')
  def test_owned_descendant_cleanup_leaves_unrelated_process(self):
    import subprocess
    sentinel=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=True)
    try:
      with tempfile.TemporaryDirectory() as d:
        pidfile=Path(d)/'pid'
        code='import subprocess,sys,time,pathlib; c=subprocess.Popen([sys.executable,"-c","import time;time.sleep(20)"]);pathlib.Path(sys.argv[1]).write_text(str(c.pid));time.sleep(20)'
        with (Path(d)/'log').open('w') as log:
          result=p.run_owned([sys.executable,'-c',code,str(pidfile)],log,time.monotonic()+.6,dict(os.environ))
        self.assertEqual(result['exit_code'],124);self.assertIsNone(sentinel.poll())
        child=int(pidfile.read_text());stat=Path(f'/proc/{child}/stat')
        until=time.monotonic()+2
        while stat.exists() and stat.read_text().split()[2] != 'Z' and time.monotonic()<until:time.sleep(.02)
        self.assertTrue(not stat.exists() or stat.read_text().split()[2] == 'Z')
    finally:
      os.killpg(sentinel.pid,signal.SIGKILL);sentinel.wait(timeout=3)


if __name__ == '__main__':unittest.main()
