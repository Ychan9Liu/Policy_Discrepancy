"""Synthetic provenance/stats and official CPU engineering fixtures only.

These tests never open a scientific source or execute a model/renderer/GPU.
POSIX process ownership tests are skipped on Windows, then run on sv3 Linux.
"""
import copy
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import dl_natural_reset as reset


class ResetStatsTest(unittest.TestCase):
  def setUp(self):
    import numpy as np
    self.np = np

  def rewards(self, costs):
    np = self.np
    out = np.zeros((8,7,20),np.float64)
    out[:,:,0] = costs
    return out

  def test_state_sensitive_but_shared_near_optimal_control_is_insufficient(self):
    np = self.np
    costs = np.zeros((8,7),np.float64)
    costs[:,0] = 1
    costs[:,1] = np.linspace(.1,.8,8)
    result = reset.analyze(self.rewards(costs))
    self.assertTrue(all(result['sensitive']))
    self.assertFalse(result['sufficient'])
    self.assertEqual(result['status'],'finite_reset_control_object_insufficient')
    self.assertEqual(len(result['all_pair_minimax']),28)
    self.assertEqual(result['disjoint_pair_count'],0)

  def test_exact_closed_delta_and_one_ulp_disjoint_boundary(self):
    np = self.np
    costs = np.zeros((8,7),np.float64)
    costs[0,0] = .2;costs[1,1] = .2
    result = reset.analyze(self.rewards(costs))
    self.assertFalse(result['sufficient'])
    self.assertTrue(all(result['delta_optimal_membership'][0]))
    costs[0,0] = np.nextafter(.2,np.inf)
    costs[1,1] = np.nextafter(.2,np.inf)
    result = reset.analyze(self.rewards(costs))
    self.assertTrue(result['sufficient'])
    self.assertEqual(result['disjoint_pair_count'],1)
    pair = result['cross_episode_disjoint_pairs'][0]
    self.assertEqual((pair['episode_a'],pair['episode_b']),(9,14))
    self.assertGreater(pair['minimax_regret_minus_delta'],0)
    self.assertEqual(pair['minimax_regret'],float(np.nextafter(.2,np.inf)))
    self.assertFalse(result['minimax_margin_is_new_threshold'])
    self.assertFalse(result['display_tolerance_applied'])

  def test_display_winner_flip_with_tiny_cost_is_not_decision_evidence(self):
    np = self.np
    costs = np.zeros((8,7),np.float64)
    for i in range(8):costs[i,i%7] = 1e-9
    result = reset.analyze(self.rewards(costs))
    self.assertFalse(result['sufficient'])
    self.assertTrue(all(all(row) for row in result['delta_optimal_membership']))

  def test_finite_dtype_shape_and_overflow_rejected(self):
    np = self.np
    for data in (np.zeros((8,7,20),np.float32),np.zeros((8,7,19),np.float64),
        np.full((8,7,20),np.nan,np.float64)):
      with self.assertRaises(ValueError):reset.analyze(data)
    with np.errstate(over='ignore'):
      with self.assertRaisesRegex(ValueError,'cumulative'):
        reset.analyze(np.full((8,7,20),1e308,np.float64))


def synthetic_arrays():
  import numpy as np
  data = dict(reward=np.zeros(1001,np.float32),is_first=np.zeros(1001,np.bool_),
      is_last=np.zeros(1001,np.bool_),is_terminal=np.zeros(1001,np.bool_),
      action=np.zeros((1001,12),np.float32),action_applied=np.zeros((1001,12),np.float64),
      action_executed=np.ones(1001,np.bool_),action_contract=np.array(reset.CONTRACT),
      snapshots=np.frombuffer(pickle.dumps([{} for _ in range(1001)],protocol=5),np.uint8))
  data['is_first'][0]=True;data['is_last'][-1]=True;data['action_executed'][-1]=False
  return data


class ResetSourceTest(unittest.TestCase):
  def test_active_true_reset_complete_flags_and_numeric_contract(self):
    import numpy as np
    data = synthetic_arrays()
    reset.validate_source_arrays(data)
    cases = (
        ('not_reset',lambda x:x['is_first'].__setitem__(0,False)),
        ('first_again',lambda x:x['is_first'].__setitem__(2,True)),
        ('source_early_last',lambda x:x['is_last'].__setitem__(10,True)),
        ('source_terminal',lambda x:x['is_terminal'].__setitem__(999,True)),
        ('no_final_last',lambda x:x['is_last'].__setitem__(1000,False)),
        ('missing_action',lambda x:x['action_executed'].__setitem__(0,False)),
        ('bad_final_action',lambda x:x['action_executed'].__setitem__(1000,True)),
        ('reward_notzero',lambda x:x['reward'].__setitem__(0,.1)),
        ('reward_negativezero',lambda x:x['reward'].__setitem__(0,-0.)),
        ('wrong_dtype',lambda x:x.__setitem__('action',x['action'].astype(np.float64))),
        ('untrusted_object',lambda x:x.__setitem__('snapshots',np.array([object()],object))))
    for name,change in cases:
      with self.subTest(name=name):
        other=copy.deepcopy(data);change(other)
        with self.assertRaises(ValueError):reset.validate_source_arrays(other)

  def source(self, folder):
    import numpy as np
    folder=Path(folder)
    selected=set(np.random.default_rng(20261020).permutation(32)[:8])
    definitions=[dict(episode=i,seed=int(np.random.SeedSequence([20261020,i,0x444C]).generate_state(1)[0]),
        policy_seed=int(np.random.SeedSequence([20261020,i,0x504F4C]).generate_state(1)[0]),
        split='calibration' if i in selected else 'blind') for i in range(32)]
    plan=dict(version='DL-source-r1',implementation='DL-source-engineering-r2',
        action_contract=reset.CONTRACT,metadata=dict(git_sha=reset.SOURCE_GIT,
          config={'task':'dmc_quadruped_walk','env.dmc.proprio':False},**reset.MODEL_PINS),
        environment={'packages':{}},episodes=definitions)
    reset.write_json(folder/'collection_plan.json',plan)
    artifacts=[]
    for definition in definitions:
      path=folder/f'episode_{definition["episode"]:03}.npz'
      if definition['split']=='calibration':
        np.savez_compressed(path,**synthetic_arrays(),image=np.array([object()],object),
            before_carries=np.array([object()],object))
      else:
        # A blind archive is deliberately not even a valid zip.
        path.write_bytes(b'blind raw bytes only')
      artifacts.append(dict(**definition,path=path.name,sha256=reset.sha(path)))
    complete=dict(version='DL-source-r1',implementation='DL-source-engineering-r2',
        action_contract=reset.CONTRACT,params_unchanged=True,training_counters_unchanged=True,
        eval_counters_legitimately_advanced=True,optimizer_updates=0,
        plan_sha256=reset.sha(folder/'collection_plan.json'),artifacts=artifacts)
    reset.write_json(folder/'collection_complete.json',complete)
    cs=reset.sha(folder/'collection_complete.json')
    budget=dict(version='DL-source-r1',git_sha=reset.SOURCE_GIT,seconds=3600,script_sha256='c'*64)
    reset.write_json(folder/'budget.json',budget)
    supervisor=dict(version='DL-source-r1',git_sha=reset.SOURCE_GIT,budget_sha256=reset.sha(folder/'budget.json'),
        exit_code=0,timed_out=False,worker_terminal=True,elapsed_seconds=1.)
    reset.write_json(folder/'supervisor_complete.json',supervisor)
    (folder/'worker_exit_code.txt').write_bytes(b'0\n')
    terminal={name:reset.sha(folder/name) for name in reset.SOURCE_TERMINAL_NAMES}
    artifact=dict(source_contract_artifact_audit_passed=True,expected_source_sha=reset.SOURCE_GIT,
        blind_arrays_read=False,metadata_sha256={'collection_complete.json':cs,**terminal},
        source_file_sha256={'scripts/dl_production_source.py':'c'*64},supervisor_exit_code=0)
    physics=dict(expected_source_sha=reset.SOURCE_GIT,blind_npz_arrays_opened=False,status='passed',
        physics_steps=12800,source_json_sha256={'complete':cs,'budget':terminal['budget.json'],
          'supervisor':terminal['supervisor_complete.json']})
    reset.write_json(folder/'artifact.json',artifact);reset.write_json(folder/'physics.json',physics)
    pins=dict(plan=reset.sha(folder/'collection_plan.json'),complete=cs,
        artifact_audit=reset.sha(folder/'artifact.json'),physics_audit=reset.sha(folder/'physics.json'))
    return pins

  def test_all_bytes_before_decode_white_list_eight_and_no_blind_npz_open(self):
    import numpy as np
    with tempfile.TemporaryDirectory() as directory:
      folder=Path(directory);pins=self.source(folder)
      opens=[];original=np.load
      def load(path,*args,**kwargs):
        self.assertIsInstance(path,reset.io.BytesIO)
        opens.append(reset.hashlib.sha256(path.getvalue()).hexdigest())
        return original(path,*args,**kwargs)
      with mock.patch.dict(reset.SOURCE_PINS,pins,clear=True),mock.patch.object(np,'load',side_effect=load):
        selected,binding=reset.source_authority(folder,folder/'artifact.json',folder/'physics.json')
      self.assertEqual(opens,[reset.sha(folder/f'episode_{ep:03}.npz') for ep in reset.EPISODES])
      self.assertEqual([x['definition']['episode'] for x in selected],list(reset.EPISODES))
      self.assertEqual(len(binding['all_32_artifact_hashes']),32)
      self.assertFalse(binding['image_arrays_read']);self.assertFalse(binding['before_carries_read'])
      self.assertFalse(binding['blind_arrays_read'])
      opens.clear()
      (folder/'episode_029.npz').write_bytes(b'tampered blind raw bytes')
      with mock.patch.dict(reset.SOURCE_PINS,pins,clear=True),mock.patch.object(np,'load',side_effect=load):
        with self.assertRaisesRegex(ValueError,'archive bytes'):
          reset.source_authority(folder,folder/'artifact.json',folder/'physics.json')
      self.assertEqual(opens,[])

  def test_wrong_published_json_never_decodes_pickle(self):
    with tempfile.TemporaryDirectory() as directory:
      folder=Path(directory);pins=self.source(folder)
      (folder/'collection_plan.json').write_bytes(b'{}')
      with mock.patch.dict(reset.SOURCE_PINS,pins,clear=True),mock.patch.object(
          reset.pickle,'loads',side_effect=AssertionError('Untrusted pickle executed')):
        with self.assertRaisesRegex(ValueError,'Published'):
          reset.source_authority(folder,folder/'artifact.json',folder/'physics.json')


class ResetPhysicsFixtureTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    os.environ.update(CUDA_VISIBLE_DEVICES='',MUJOCO_GL='disable',JAX_PLATFORMS='cpu')
    from analysis import dl_task_semantics as semantics
    from scripts import dl_action_information_physics as physical
    cls.audit,_=semantics.load_auditor(ROOT/'analysis/dl_feedback_physics_audit_v1.py')
    cls.physical=physical

  def setUp(self):
    from dm_control import suite
    from dm_control.mujoco.engine import Physics
    self.suite=suite
    self.render_guard=mock.patch.object(Physics,'render',side_effect=AssertionError('No fixture render'))
    self.render_guard.start();self.addCleanup(self.render_guard.stop)
    self.env=suite.load('quadruped','walk',task_kwargs={'random':77})
    self.addCleanup(self.env.close)
    self.env.reset();self.initial=self.audit.snapshot(self.env,False)

  def source(self):
    import numpy as np
    data=synthetic_arrays();snapshots=[copy.deepcopy(self.initial)]
    rng=np.random.default_rng(77)
    low,high=self.env.action_spec().minimum,self.env.action_spec().maximum
    for i in range(20):
      raw=rng.uniform(-1.5,1.5,12).astype(np.float32)
      applied=self.physical.mapping(raw,low,high)
      ts=self.env.step(applied)
      data['action'][i]=raw;data['action_applied'][i]=applied
      data['reward'][i+1]=np.float32(ts.reward)
      snapshots.append(self.audit.snapshot(self.env,False))
    return dict(definition={'episode':9,'seed':77},data=data,snapshots=snapshots)

  def reference(self,source):
    other=self.suite.load('quadruped','walk',task_kwargs={'random':77})
    try:return reset.reference_replay(other,source,self.audit,self.physical)
    finally:other.close()

  def test_official_fresh_seed_reset_reference_raw_applied_reward_clock(self):
    source=self.source()
    receipt=self.reference(source)
    self.assertEqual(receipt['physics_steps'],20)
    self.assertTrue(receipt['fresh_official_seed_reset_verified'])
    self.assertEqual(receipt['fresh_reset_calls'],1)
    self.assertEqual(receipt['fresh_reset_complete_integration_error'],0)
    self.assertTrue(receipt['fresh_reset_integration_byte_exact'])
    self.assertEqual(receipt['maximum_complete_integration_error'],0)
    self.assertTrue(receipt['source_reward_float32_byte_exact'])
    self.assertFalse(receipt['source_float64_reward_byte_exact_claimed'])
    self.assertNotIn('jax',sys.modules)

  def test_reference_mismatches_rejected_before_any_new_control(self):
    import numpy as np
    source=self.source()
    cases=(('clock',lambda x:x['snapshots'][0]['suite'].__setitem__('step_count',1)),
        ('not_first',lambda x:x['data']['is_first'].__setitem__(0,False)),
        ('mapping',lambda x:x['data']['action_applied'].__setitem__((0,0),.99)),
        ('F32',lambda x:x['data']['reward'].__setitem__(4,np.nextafter(x['data']['reward'][4],np.float32(np.inf)))),
        ('state',lambda x:x['snapshots'][2]['suite']['integration'].__setitem__(0,999)),
        ('task',lambda x:x['snapshots'][0]['suite']['task'].__setitem__('_desired_speed',1.)))
    for name,change in cases:
      with self.subTest(name=name):
        damaged=copy.deepcopy(source);change(damaged)
        with self.assertRaises(ValueError):self.reference(damaged)

  def test_new_H20_f64_and_full_state_byte_exact_repeat_not_source_tolerance(self):
    from types import SimpleNamespace
    import numpy as np
    record=self.physical.run_sequence(self.env,self.initial,self.physical.controls(self.env)[3],self.audit)
    reset.complete_record(self.env,record,self.audit,self.physical)
    proxy=SimpleNamespace(**{k:getattr(self.audit,k) for k in
        ('validate_snapshot','array','byte_equal','restore','snapshot')},
        compare_state=lambda actual,expected:reset.compare_exact(self.audit,actual,expected))
    receipt=self.physical.replay_sequence(self.env,record,proxy)
    self.assertEqual(receipt['padding_steps'],0)
    self.assertEqual(receipt['maximum_integration_error'],0)
    actual=copy.deepcopy(record['snapshots'][1]);expected=copy.deepcopy(actual)
    actual['suite']['integration'][0]=np.nextafter(actual['suite']['integration'][0],np.inf)
    self.assertLess(self.audit.compare_state(actual,expected),1e-8)
    with self.assertRaisesRegex(ValueError,'byte'):
      reset.compare_exact(self.audit,actual,expected)
    damaged=copy.deepcopy(record)
    damaged['reward_f64'][3]=np.nextafter(damaged['reward_f64'][3],np.inf)
    with self.assertRaisesRegex(ValueError,'F64 reward'):
      self.physical.replay_sequence(self.env,damaged,proxy)

  def test_genuine_early_terminal_is_rejected_not_zero_padded_into_object(self):
    with mock.patch.object(type(self.env._task),'get_termination',return_value=0.):
      record=self.physical.run_sequence(self.env,self.initial,self.physical.controls(self.env)[0],self.audit)
    self.assertEqual(record['executed_steps'],1)
    with self.assertRaisesRegex(ValueError,'genuine complete'):
      reset.complete_record(self.env,record,self.audit,self.physical)


class ResetParentTest(unittest.TestCase):
  def test_parent_import_is_stdlib_and_keeps_environment(self):
    script="import os,sys; before=dict(os.environ); from scripts import dl_natural_reset; assert before==dict(os.environ); assert not any(x in sys.modules for x in ('numpy','jax','dm_control','mujoco')); print('stdlib parent')"
    result=subprocess.run([sys.executable,'-c',script],cwd=ROOT,capture_output=True,text=True,timeout=10)
    self.assertEqual(result.returncode,0,result.stderr)
    self.assertIn('stdlib parent',result.stdout)

  def test_generic_callback_cannot_claim_real_reads_gpu_or_completion(self):
    with tempfile.TemporaryDirectory() as directory:
      path=Path(directory)/'attempt'
      receipt=reset.supervise([sys.executable,'-c',"print('fixture')"],path,seconds=3)
      self.assertEqual(receipt['exit_code'],0)
      self.assertTrue(receipt['worker_terminal'])
      self.assertFalse(receipt['complete_verified'])
      self.assertIsNone(receipt['actual_numeric_episode_reads'])
      self.assertIsNone(receipt['actual_gpu_devices_used'])
      self.assertEqual(receipt['reason'],'generic_worker_exited')
      with self.assertRaisesRegex(ValueError,'must not exist'):
        reset.supervise([sys.executable,'-c','pass'],path,seconds=3)
      forged=reset.supervise([sys.executable,'-c','pass'],Path(directory)/'callback',seconds=3,
          verify=lambda *args:dict(complete_verified=True,
            verification_kind='schema_hash_chain_not_independent_arithmetic_or_physics'))
      self.assertFalse(forged['complete_verified'])
      self.assertIsNone(forged['actual_numeric_episode_reads'])
      self.assertIsNone(forged['actual_gpu_devices_used'])

  def test_timeout_only_own_worker_and_unknown_actual_science(self):
    with tempfile.TemporaryDirectory() as directory:
      receipt=reset.supervise([sys.executable,'-c','import time;time.sleep(20)'],
          Path(directory)/'attempt',seconds=.2)
      self.assertEqual(receipt['exit_code'],124)
      self.assertTrue(receipt['timed_out']);self.assertTrue(receipt['worker_terminal'])
      self.assertIsNone(receipt['actual_numeric_episode_reads'])

  @unittest.skipUnless(os.name=='posix','POSIX owned process groups run on Linux')
  def test_failed_leader_descendant_cleaned_unrelated_sentinel_survives(self):
    with tempfile.TemporaryDirectory() as directory:
      pidfile=Path(directory)/'descendant.pid'
      sentinel=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=True)
      try:
        code=f"import subprocess,sys,pathlib; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)']); pathlib.Path({str(pidfile)!r}).write_text(str(p.pid)); sys.exit(3)"
        receipt=reset.supervise([sys.executable,'-c',code],Path(directory)/'attempt',seconds=3)
        self.assertEqual(receipt['exit_code'],3)
        self.assertIsNone(sentinel.poll())
        pid=int(pidfile.read_text())
        path=Path(f'/proc/{pid}/stat')
        for _ in range(20):
          if not path.exists() or path.read_text().split()[2]=='Z':break
          time.sleep(.02)
        self.assertTrue(not path.exists() or path.read_text().split()[2]=='Z')
      finally:
        sentinel.terminate();sentinel.wait(timeout=3)

  @unittest.skipUnless(os.name=='posix','POSIX owned Git-helper groups run on Linux')
  def test_preworker_helper_timeout_cleans_own_descendant(self):
    with tempfile.TemporaryDirectory() as directory:
      pidfile=Path(directory)/'descendant.pid'
      code=f"import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)']); pathlib.Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(20)"
      with self.assertRaises(subprocess.TimeoutExpired):
        reset._owned_output([sys.executable,'-c',code],time.monotonic()+.3)
      pid=int(pidfile.read_text());path=Path(f'/proc/{pid}/stat')
      for _ in range(20):
        if not path.exists() or path.read_text().split()[2]=='Z':break
        time.sleep(.02)
      self.assertTrue(not path.exists() or path.read_text().split()[2]=='Z')

  def test_late_final_receipt_is_not_accepted_as_completion(self):
    with tempfile.TemporaryDirectory() as directory:
      original=reset.write_json
      def delayed(path,value):
        if Path(path).name=='supervisor_receipt_pending.json':time.sleep(.4)
        return original(path,value)
      with mock.patch.object(reset,'write_json',side_effect=delayed):
        result=reset.supervise([sys.executable,'-c','pass'],Path(directory)/'attempt',seconds=.3,
            verify=lambda *args:dict(complete_verified=True))
      self.assertEqual(result['exit_code'],124)
      self.assertTrue(result['timed_out']);self.assertFalse(result['complete_verified'])
      self.assertTrue((Path(directory)/'attempt'/'late_supervisor_receipt.json').is_file())
      saved=reset.read_json(Path(directory)/'attempt'/'supervisor_complete.json')
      self.assertEqual(saved['exit_code'],124)
      self.assertTrue(saved['timed_out']);self.assertFalse(saved['complete_verified'])

  def test_late_atomic_promotion_preserves_provisional_and_publishes_failure(self):
    with tempfile.TemporaryDirectory() as directory:
      original=reset.os.replace
      def delayed(source,target):
        if Path(source).name=='supervisor_receipt_pending.json':time.sleep(.4)
        return original(source,target)
      with mock.patch.object(reset.os,'replace',side_effect=delayed):
        result=reset.supervise([sys.executable,'-c','pass'],Path(directory)/'attempt',seconds=.3,
            verify=lambda *args:dict(complete_verified=True))
      self.assertEqual(result['exit_code'],124)
      self.assertFalse(result['complete_verified'])
      saved=reset.read_json(Path(directory)/'attempt'/'supervisor_complete.json')
      self.assertEqual(saved['exit_code'],124);self.assertFalse(saved['complete_verified'])
      self.assertTrue((Path(directory)/'attempt'/'late_provisional_supervisor_receipt.json').is_file())


if __name__=='__main__':
  unittest.main()
