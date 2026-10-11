"""CPU synthetic provenance/features and real subprocess supervision.

BF16 arrays use the exact size50m interface shape, but are synthetic. The
fixture does not run a model, renderer, GPU or open any scientific artifact.
Linux-only tests verify ownership of real process groups and are skipped on
Windows; Windows tests cover only the direct child process contract.
"""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MUJOCO_GL'] = 'disable'
os.environ['JAX_PLATFORMS'] = 'cpu'

import ml_dtypes
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import dl_action_information as runner


class RunnerFeaturesTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.audit,_=runner.semantics.load_auditor(ROOT/'analysis/dl_feedback_physics_audit_v1.py')

  def setUp(self):
    self.records=[]
    hashes=[]
    for i,pos in enumerate(runner.POSITIONS):
      deter=(np.arange(4096)%8).astype(ml_dtypes.bfloat16)
      deter+=ml_dtypes.bfloat16(i)
      stoch=np.zeros((32,32),ml_dtypes.bfloat16)
      stoch[np.arange(32),(np.arange(32)+i)%32]=1
      carry=({},dict(deter=[deter.copy()],stoch=[stoch.copy()]),{},
          dict(action=[np.full(12,i/16,np.float32)]))
      digest=runner.tree_digest(carry);hashes.append(digest)
      snapshot=dict(suite=dict(integration=np.array([pos/50,.25],np.float64),
          step_count=pos,reset_next_step=False,task={'_desired_speed':.5,
              '_random':np.random.RandomState(100+i)}),from_dm_done=False)
      obs=dict(image=np.full((64,64,3),i,np.uint8),reward=np.array(.25,np.float32),
          is_first=np.array(False),is_last=np.array(False),is_terminal=np.array(False))
      for pair in range(4):
        key=self.audit.named_keys(9,pos,pair)[0]
        for branch in range(2):
          z=np.zeros((32,32),ml_dtypes.bfloat16)
          z[np.arange(32),(np.arange(32)+pair+branch)%32]=1
          raw=np.full(12,.125*(pair+1)*(1 if branch==0 else -1),np.float32)
          proof=dict(root_key=key.copy(),latent_key=key[::-1].copy(),actor_key=key^np.uint32(17),
              h=deter.copy(),latent=z,selected_logit=np.zeros((32,32),ml_dtypes.bfloat16),
              raw=raw.copy(),rng_consumption=np.array(2,np.int32))
          self.records.append(dict(episode=9,position=pos,pair=pair,branch=branch,
              initial_carry=copy.deepcopy(carry),pre_carry_sha256=[digest],
              snapshots=[copy.deepcopy(snapshot)],observations=[copy.deepcopy(obs)],proofs=[proof],
              keys=key[None].copy(),action_executed=np.array([True]),raw=raw[None].copy()))
    self.calibration=dict(definition=dict(episode=9),carry_positions=list(runner.POSITIONS),
        carry_sha256=hashes)

  def features(self,records=None,calibration=None):
    return runner.features(self.records if records is None else records,
        self.calibration if calibration is None else calibration,self.audit,(4096,32,32))

  def test_exact_BF16_size50m_interface_and_source_identity_readonly(self):
    original=runner.tree_digest(self.records[0]['initial_carry'])
    h,z,commands,receipts=self.features()
    self.assertEqual(h.shape,(4,4096));self.assertEqual(h.dtype,np.float64)
    self.assertEqual(z.shape,(4,4,2,32,32));self.assertEqual(z.dtype,np.float64)
    self.assertEqual(commands.shape,(4,8,12));self.assertEqual(commands.dtype,np.float32)
    self.assertEqual(len(receipts),32)
    for i in range(4):
      self.assertTrue(np.array_equal(h[i],self.records[8*i]['proofs'][0]['h'].astype(np.float64)))
      for j in range(8):
        pair,branch=divmod(j,2)
        self.assertTrue(np.array_equal(z[i,pair,branch],self.records[8*i+j]['proofs'][0]['latent'].astype(np.float64)))
        self.assertTrue(self.audit.byte_equal(commands[i,j],self.records[8*i+j]['raw'][0]))
    self.assertEqual(original,runner.tree_digest(self.records[0]['initial_carry']))
    h[0,0]=123;z[0,0,0,0,0]=123;commands[0,0,0]=123
    self.assertNotEqual(float(self.records[0]['proofs'][0]['h'][0]),123)
    self.assertNotEqual(float(self.records[0]['proofs'][0]['latent'][0,0]),123)
    self.assertNotEqual(float(self.records[0]['proofs'][0]['raw'][0]),123)
    self.assertNotIn('jax',sys.modules)

  def test_provenance_and_receipt_counterexamples_fail(self):
    cases={
        'source_carry':lambda r:r[0]['initial_carry'][1]['deter'][0].__setitem__(0,99),
        'pre_carry_hash':lambda r:r[1]['pre_carry_sha256'].__setitem__(0,'0'*64),
        'identity':lambda r:r[1].__setitem__('pair',3),
        'same_h':lambda r:r[1]['proofs'][0]['h'].__setitem__(0,99),
        'task_RNG':lambda r:r[1]['snapshots'][0]['suite']['task']['_random'].uniform(),
        'initial_rgb':lambda r:r[1]['observations'][0]['image'].__setitem__((0,0,0),255),
        'first_raw':lambda r:r[1]['proofs'][0]['raw'].__setitem__(0,99),
        'root_key':lambda r:r[1]['proofs'][0]['root_key'].__setitem__(0,np.uint32(123)),
        'named_key':lambda r:r[0]['keys'].__setitem__((0,0),np.uint32(123)),
        'paired_latent_key':lambda r:r[1]['proofs'][0]['latent_key'].__setitem__(0,np.uint32(123)),
        'paired_actor_key':lambda r:r[1]['proofs'][0]['actor_key'].__setitem__(0,np.uint32(123)),
        'rng_count':lambda r:r[1]['proofs'][0].__setitem__('rng_consumption',np.array(1,np.int32)),
        'probability_not_sample':lambda r:r[0]['proofs'][0].__setitem__('latent',np.full((32,32),1/32,ml_dtypes.bfloat16)),
        'multi_hot':lambda r:r[0]['proofs'][0]['latent'].__setitem__((0,1),1),
        'latent_shape':lambda r:r[0]['proofs'][0].__setitem__('latent',np.zeros((1,32,32),ml_dtypes.bfloat16)),
        'h_float32':lambda r:r[0]['proofs'][0].__setitem__('h',np.zeros(4096,np.float32)),
        'h_nonfinite':lambda r:r[0]['proofs'][0]['h'].__setitem__(0,np.nan),
        'latent_nonfinite':lambda r:r[0]['proofs'][0]['latent'].__setitem__((0,0),np.inf),
        'raw_nonfinite':lambda r:r[0]['proofs'][0]['raw'].__setitem__(0,np.nan),
        'raw_not_executed':lambda r:r[0]['action_executed'].__setitem__(0,False),
        'proof_field':lambda r:r[0]['proofs'][0].__setitem__('new_field',np.array(1)),
        'clock':lambda r:r[0]['snapshots'][0]['suite'].__setitem__('step_count',99),
    }
    for name,mutate in cases.items():
      with self.subTest(name=name):
        damaged=copy.deepcopy(self.records);mutate(damaged)
        with self.assertRaises(ValueError):
          self.features(damaged)
    damaged=copy.deepcopy(self.calibration);damaged['carry_sha256'][0]='0'*64
    with self.assertRaises(ValueError):
      self.features(calibration=damaged)

  def test_published_feedback_pins_and_exclusive_finite_JSON(self):
    self.assertEqual(runner.FEEDBACK_SHA,'0d49c589b525ba0c26f9f85bf52c23573dc19507')
    self.assertEqual(runner.JSON_PINS,dict(
        plan='f046120aa9a54fe52abdd6a9c86adaa679fc976239aff27f9304f90ca74e74f0',
        result='4592e363a54385a36e849f937fae259aba06890d0f5e11a86e44393135915309',
        complete='a4bfda1872553fb82706f90e79870d69cf69202ab1bd6885c4a3745c5917657c',
        budget='91e38a756d131ae279a8b8da0c6372422fafcd1ee49b1c76a88fc677fa81f4af',
        supervisor='69c1cac7e217dc9660779ed9661a54b986b1b71ade3b23063f124e431df0ed41'))
    with tempfile.TemporaryDirectory() as directory:
      path=Path(directory)/'receipt.json'
      runner.write_json(path,dict(version='fixture',a=1))
      before=path.read_bytes()
      with self.assertRaises(FileExistsError):
        runner.write_json(path,dict(a=2))
      self.assertEqual(path.read_bytes(),before)
      invalid=Path(directory)/'invalid.json';invalid.write_text('{"a":NaN}',encoding='utf-8')
      with self.assertRaises(ValueError):
        runner.read_json(invalid)


class RunnerSupervisorTest(unittest.TestCase):
  def setUp(self):
    self.directory=tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.root=Path(self.directory.name)

  def invoke(self,code,name='attempt',seconds=2,started=None):
    return runner.supervise([sys.executable,'-c',code],self.root/name,seconds,started)

  def test_success_failure_newoutput_and_fixed_production_budget(self):
    self.assertEqual(runner.SECONDS,180)
    success=self.invoke('print("synthetic worker complete")')
    self.assertEqual(success['exit_code'],0);self.assertFalse(success['timed_out'])
    self.assertTrue(success['worker_terminal']);self.assertEqual(success['reason'],'worker_completed')
    self.assertEqual(success['budget_sha256'],runner.sha(self.root/'attempt'/'budget.json'))
    self.assertFalse((self.root/'attempt'/'action_information_complete.json').exists())
    # Exit0 is leader completion only; this fixture intentionally does not
    # establish the scientific worker's complete artifact authority.
    with self.assertRaisesRegex(ValueError,'New attempt'):
      self.invoke('raise AssertionError("must not execute")')
    failed=self.invoke('import sys;sys.exit(7)',name='failed')
    self.assertEqual(failed['exit_code'],7);self.assertFalse(failed['timed_out'])
    self.assertEqual(failed['reason'],'worker_failed');self.assertTrue(failed['worker_terminal'])
    for bad in (0,-1,float('nan'),float('inf')):
      with self.assertRaises(ValueError):
        self.invoke('pass',name='bad',seconds=bad)
    self.assertFalse((self.root/'bad').exists())

  def test_timeout_and_parent_elapsed_budget(self):
    receipt=self.invoke('import time;time.sleep(20)',seconds=.15)
    self.assertEqual(receipt['exit_code'],124);self.assertTrue(receipt['timed_out'])
    self.assertTrue(receipt['worker_terminal']);self.assertEqual(receipt['reason'],'incomplete_hard_budget')
    self.assertFalse((self.root/'attempt'/'action_information_complete.json').exists())
    # Prior parent work consumes the same total budget; no child-only reset.
    receipt=self.invoke('import time;time.sleep(20)',name='parentspent',seconds=1,
        started=time.monotonic()-2)
    self.assertEqual(receipt['exit_code'],124);self.assertGreater(receipt['elapsed_seconds'],2)

  @unittest.skipUnless(os.name=='posix','Linux process-group proof cannot run on Windows')
  def test_owned_group_cleanup_after_early_failure_leaves_unrelated_process(self):
    sentinel=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
    grandchild=self.root/'grandchild.pid'
    def clean_sentinel():
      if sentinel.poll() is None:
        sentinel.kill()
      sentinel.wait(timeout=5)
    self.addCleanup(clean_sentinel)
    code=('import subprocess,sys,pathlib; '
        'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]); '
        f'pathlib.Path({str(grandchild)!r}).write_text(str(p.pid));sys.exit(7)')
    receipt=self.invoke(code)
    self.assertEqual(receipt['exit_code'],7);self.assertTrue(receipt['owned_group_cleanup_requested'])
    self.assertIsNone(sentinel.poll())
    pid=int(grandchild.read_text())
    for _ in range(50):
      status=Path(f'/proc/{pid}/stat')
      if not status.exists() or status.read_text().split()[2]=='Z':
        break
      time.sleep(.01)
    else:
      self.fail('Owned grandchild survived failed-leader group cleanup')

  @unittest.skipUnless(os.name=='posix','Linux process-group proof cannot run on Windows')
  def test_owned_group_timeout_cleans_grandchild(self):
    grandchild=self.root/'grandchild.pid'
    code=('import subprocess,sys,pathlib,time; '
        'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]); '
        f'pathlib.Path({str(grandchild)!r}).write_text(str(p.pid));time.sleep(30)')
    receipt=self.invoke(code,seconds=.5)
    self.assertEqual(receipt['exit_code'],124);self.assertTrue(receipt['owned_group_cleanup_requested'])
    pid=int(grandchild.read_text())
    for _ in range(50):
      status=Path(f'/proc/{pid}/stat')
      if not status.exists() or status.read_text().split()[2]=='Z':
        break
      time.sleep(.01)
    else:
      self.fail('Owned grandchild survived timeout group cleanup')


if __name__=='__main__':
  unittest.main()
