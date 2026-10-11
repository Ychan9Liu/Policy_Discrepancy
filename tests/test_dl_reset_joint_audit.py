"""Synthetic audit failures, not a real model/source/physics acceptance."""
import copy
import hashlib
import importlib.util
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
import ml_dtypes

from scripts import dl_reset_joint_audit as m


def write(root,name,value):
  m.write_json(root/name,value);return m.sha(root/name)


def primary_fixture(root):
  primary=root/'primary';primary.mkdir()
  git='a'*40;pins={name:hashlib.sha256(name.encode()).hexdigest() for name in m.FILES}
  budget=dict(version=m.VERSION,actual_git_sha=git,parent_pid=123,seconds=3600,
      planned_physical_gpu=4,max_gpu_devices=1,max_gpu_hours=1,monotonic_start=10.,
      source_files_sha256=pins,script_sha256=pins['scripts/dl_reset_joint.py'])
  write(primary,'budget.json',budget)
  for name in m.FREEZE_FILES:
    if name != 'budget.json': (primary/name).write_bytes(b'synthetic '+name.encode())
  truth_pin=write(primary,'truth-complete.json',dict(synthetic=True))
  frozen=dict(version=m.VERSION,actual_git_sha=git,source_files_sha256=pins,
      cal_episodes=list(m.CAL),blind_episodes=list(m.BLIND),groups=['0','B128','R128'],
      updates_per_arm=128,statistics_seed=20261026,blind_arrays_read=False,
      selection='four-prediction-mean then exact argmax, lowest exact tie ID',
      monotonic_freeze=20.,files={name:m.sha(primary/name) for name in m.FREEZE_FILES})
  frozen_pin=write(primary,'freeze.json',frozen)
  write(primary,'freeze-seal-parent.json',dict(version=m.VERSION,actual_git_sha=git,
      freeze_sha256=frozen_pin,blind_started=False,monotonic_sealed=21.))
  result=dict(version=m.VERSION,actual_git_sha=git,freeze_sha256=frozen_pin,
      truth_sha256=truth_pin,source_inputs_unchanged_at_final_statistics=True,
      method_effectiveness_established=False,continuous_actor_supported=False,
      full_goal_complete=False,decision='stop_after_fixed_reset_joint_assay')
  result_pin=write(primary,'result.json',result)
  phases=[]
  for i,name in enumerate(m.PHASES):
    row=dict(phase=name,is_gpu=name in ('baseline','B','R','eval0','evalB','evalR'),
        worker_pid=200+i,exit_code=0,timed_out=False,worker_terminal=True,
        owned_group_cleanup_requested=True,elapsed_seconds=.1)
    write(primary,name+'-exit.json',row);phases.append(row)
  supervisor=dict(version=m.VERSION,actual_git_sha=git,parent_pid=123,
      exit_code=0,timed_out=False,complete_verified=True,actual_gpu_devices_used=1,
      budget_sha256=m.sha(primary/'budget.json'),result_sha256=result_pin,
      method_effectiveness_established=False,independent_audit_complete=False,
      elapsed_parent_entry_seconds=30.,phases=phases)
  write(primary,'supervisor_complete.json',supervisor)
  terminal=root/'external.json'
  m.write_json(terminal,dict(actual_git_sha=git,exit_code=0,timed_out=False,
      elapsed_external_seconds=31.,output=str(primary)))
  return primary,terminal,pins


def rewrite(path,change):
  value=m.read_json(path);change(value);path.unlink();m.write_json(path,value)


def learning_fixture(group='B'):
  source=dict(all_32_artifact_hashes=[dict(sha256=hashlib.sha256(str(e).encode()).hexdigest()) for e in range(32)])
  saved=dict(updates=10,batches=99,actions=100,eval_actions=0,eval_seed=0)
  base={**saved,'batches':10}
  layout=[dict(row=i,episode=e,context_episode=m.CAL[(m.CAL.index(e)-1)%8],
      context_position=1000,loss_positions=[0,63],terminal_action_executed=False,
      context_first_is_replay_annotation=True) for i,e in enumerate(m.CAL*2)]
  rows=[]
  for i in range(1,129):
    pressure=dict(active_count=i%3,excess_sum=(i%3)*.25,reset_count=16)
    rows.append(dict(update=i,seed_counter=10+i-1,train_seed=[1,2],
        counters={**base,'updates':10+i,'batches':10+i},reset_pressure=pressure,
        metrics={**{'dl_reset/'+k:v for k,v in pressure.items()},'opt/updates':20+i},metrics_digest='a'*64))
  value=dict(version='DL-reset-learning-r1',group=group,updates=128,receipts=rows,
      reset_rep_gradient_stopped=group == 'R',rep_forward_values_retained=True,
      historical_optimizer_momentum_retained=True,per_update_init_train=True,
      blind_arrays_read=False,simulator_calls=0,total_loss_positions=131072,total_reset_loss_positions=2048,
      binding=dict(source_hashes={str(e):source['all_32_artifact_hashes'][e]['sha256'] for e in m.CAL},
        offline_episode_concatenation=True,original_continuous_replay_restored=False,
        balanced_duplicates_per_episode=2,reset_loss_positions_per_update=16,
        loss_positions_per_update=1024,layout=layout),saved_source_counters=saved,restored_counters=base,
      reset_pressure=dict(active_count=sum(x['reset_pressure']['active_count'] for x in rows),
        reset_count=2048,excess_sum=sum(x['reset_pressure']['excess_sum'] for x in rows),
        entire_128_update_metrics_observed=True),
      optimizer_counter=dict(key='opt/step/value',dtype='int32',shape=[],initial=20,final=148,updates=128))
  return value,source


def features_fixture(n=8):
  bf=ml_dtypes.bfloat16;h=np.zeros((n,2),bf)
  z=np.zeros((n,4,32,32),bf);z[:,:,:,0]=1
  roots=np.stack([np.random.default_rng([20261026,0x52535231,k]).integers(
      0,np.iinfo(np.uint32).max,(2,),np.uint32) for k in range(4)])
  prior=np.zeros((n,4,32,32),bf);D=np.zeros(n,np.float32);K=D.copy()
  ref=dict(h=h[0].copy(),zp=z[0].copy(),prior_logit=prior[0,0].copy(),roots=roots.copy(),
      model_state_sha256='a'*64,verified_observations=8 if n == 8 else 32)
  aux=dict(h=np.repeat(h[:,None],4,1),zp=z.copy(),zq=z.copy(),prior_logit=prior,
      post_logit=prior.copy(),root_key=np.repeat(roots[None],n,0),
      rng_valid=np.ones((n,4),bool),rng_consumption=np.ones((n,4),np.int32),
      D=np.repeat(D[:,None],4,1),K=np.repeat(K[:,None],4,1),common_reference=ref,
      common_h_byte_exact=True,common_prior_byte_exact=True,common_zp_per_key_byte_exact=True,
      current_agent_params_used=True,public_policy_copy_used=False,
      sample_roots_have_episode_model_branch_id=False,source_q_sample_exact_claimed=False,
      learned_only=True,optimizer_updates=0,full_state_before_sha256='a'*64,
      full_state_after_sha256='a'*64,counters_before={'updates':0},counters_after={'updates':0})
  feature=dict(h=h,zq=z.copy(),zp=z.copy(),D=D,K=K,aux=aux)
  numeric={key:np.asarray(feature[key],np.float64) for key in ('h','zq','zp','D','K')}
  return feature,numeric


class JointAuditTest(unittest.TestCase):

  def test_stdlib_import_does_not_load_numpy_jax_or_models(self):
    code='import sys; from scripts import dl_reset_joint_audit; assert "numpy" not in sys.modules; assert "jax" not in sys.modules; assert "dreamerv3.agent" not in sys.modules'
    value=subprocess.run([sys.executable,'-c',code],cwd=m.ROOT,capture_output=True,text=True)
    self.assertEqual(value.returncode,0,value.stderr)

  def test_same_payload_sha_precedes_any_pickle_decode(self):
    with tempfile.TemporaryDirectory() as d:
      root=Path(d);payload=pickle.dumps(dict(synthetic=1));(root/'x.pkl').write_bytes(payload)
      with mock.patch.object(m.pickle,'loads',side_effect=AssertionError('Must not decode')) as decode:
        with self.assertRaises(ValueError): m.load_new(root,'x.pkl','0'*64)
        decode.assert_not_called()
      self.assertEqual(m.load_new(root,'x.pkl',hashlib.sha256(payload).hexdigest()),dict(synthetic=1))

  def test_primary_complete_chain_and_seal(self):
    with tempfile.TemporaryDirectory() as d:
      primary,terminal,pins=primary_fixture(Path(d))
      value=m.primary_authority(primary,terminal,pins)
      self.assertEqual(value['primary_git'],'a'*40)
      self.assertEqual(len(value['supervisor']['phases']),9)
      self.assertEqual(value['pins']['external_cli_terminal'],m.sha(terminal))

  def test_actual_launcher_schema_has_no_legacy_timeout_or_output_fields(self):
    with tempfile.TemporaryDirectory() as d:
      root=Path(d);engineering=root/'engineering';engineering.mkdir()
      launcher=engineering/'launch-primary02.py'
      # Self-contained generic bytes; no ignored launcher artifact is required
      # by fixtures deployed to a clean Linux checkout. Production pin stays fixed.
      self.assertEqual(m.PRIMARY_LAUNCHER_SHA,'29f21e81c2c8238ac344d0c4ca6502cba5433fce63c7812e12506abbc5f4253a')
      payload=b'"""Synthetic engineering launcher fixture only."""\n'
      pin=hashlib.sha256(payload).hexdigest()
      launcher.write_bytes(payload);terminal=engineering/'primary-terminal02.json'
      primary=root/'primary-attempt02';primary.mkdir()
      command=list(m.PRIMARY_LAUNCHER_COMMAND);command[command.index('--output')+1]=str(primary)
      external=dict(actual_git_sha=m.PRIMARY_LAUNCHER_GIT,exit_code=0,worker_pid=123,
          launcher_pid=122,worker_terminal=True,elapsed_cli_seconds=505.437680,command=command,
          physical_gpu=4,gpu_uuid=m.PRIMARY_GPU_UUID,launcher_sha256=pin)
      m.write_json(terminal,external)
      with (mock.patch.object(m,'PRIMARY_LAUNCHER_PATH',launcher),
          mock.patch.object(m,'PRIMARY_LAUNCHER_SHA',pin),
          mock.patch.object(m,'PRIMARY_LAUNCHER_COMMAND',tuple(command))):
        receipt=m.external_authority(terminal,external,primary,m.PRIMARY_LAUNCHER_GIT,123)
        self.assertFalse(receipt['external_timed_out_flag_recorded'])
        self.assertTrue(receipt['normal_terminal_verified'])
        self.assertEqual(receipt['launcher_sha256'],pin)
        for field,value in (('exit_code',False),('exit_code',124),('worker_terminal',False),
            ('worker_pid',124),('launcher_pid',123),('elapsed_cli_seconds',3600.1),
            ('elapsed_cli_seconds',0),('launcher_sha256','0'*64),('physical_gpu',5),
            ('gpu_uuid','GPU-wrong')):
          bad=copy.deepcopy(external);bad[field]=value
          with self.subTest(field=field,value=value),self.assertRaises(ValueError):
            m.external_authority(terminal,bad,primary,m.PRIMARY_LAUNCHER_GIT,123)
        for field in ('elapsed_cli_seconds','worker_terminal','launcher_sha256','command'):
          bad=copy.deepcopy(external);del bad[field]
          with self.subTest(missing=field),self.assertRaises((ValueError,KeyError)):
            m.external_authority(terminal,bad,primary,m.PRIMARY_LAUNCHER_GIT,123)
        bad=copy.deepcopy(external);bad['command'][-1]+='-changed'
        with self.assertRaises(ValueError):m.external_authority(terminal,bad,primary,m.PRIMARY_LAUNCHER_GIT,123)
        bad=copy.deepcopy(external);bad['command']+=['--worker-phase','statistics']
        with self.assertRaises(ValueError):m.external_authority(terminal,bad,primary,m.PRIMARY_LAUNCHER_GIT,123)
        launcher.write_bytes(payload+b'\n# tamper')
        with self.assertRaises(ValueError):m.external_authority(terminal,external,primary,m.PRIMARY_LAUNCHER_GIT,123)

  def test_actual_primary_cannot_fall_back_to_legacy_missing_launcher(self):
    with tempfile.TemporaryDirectory() as d:
      root=Path(d);launcher=root/'launch-primary02.py';launcher.write_bytes(b'not actual')
      terminal=root/'primary-terminal02.json';m.write_json(terminal,dict(exit_code=0))
      legacy=dict(actual_git_sha=m.PRIMARY_LAUNCHER_GIT,exit_code=0,timed_out=False,
          elapsed_external_seconds=500.,output=str(root))
      with mock.patch.object(m,'PRIMARY_LAUNCHER_PATH',launcher):
        with self.assertRaises(ValueError):m.external_authority(terminal,legacy,root,m.PRIMARY_LAUNCHER_GIT,123)

  def test_partial_exit_bool_exit_external_timeout_or_wrong_output_fail(self):
    for mutate in (
        lambda p,t:rewrite(p/'supervisor_complete.json',lambda x:x['phases'].pop()),
        lambda p,t:rewrite(p/'supervisor_complete.json',lambda x:x.update(exit_code=False)),
        lambda p,t:rewrite(t,lambda x:x.update(timed_out=True)),
        lambda p,t:rewrite(t,lambda x:x.update(elapsed_external_seconds=3600.1)),
        lambda p,t:rewrite(t,lambda x:x.update(output=str(p.parent))),
        lambda p,t:rewrite(p/'freeze-seal-parent.json',lambda x:x.update(blind_started=True)),
        lambda p,t:rewrite(p/'freeze-seal-parent.json',lambda x:x.update(monotonic_sealed=19.)),
        lambda p,t:(p/'late-supervisor.json').write_text('{}')):
      with self.subTest(mutate=mutate),tempfile.TemporaryDirectory() as d:
        primary,terminal,pins=primary_fixture(Path(d));mutate(primary,terminal)
        with self.assertRaises(ValueError):m.primary_authority(primary,terminal,pins)

  def test_frozen_full_checkpoint_raw_tamper_fails(self):
    with tempfile.TemporaryDirectory() as d:
      primary,terminal,pins=primary_fixture(Path(d));(primary/'B-checkpoint.pkl').write_bytes(b'tampered')
      with self.assertRaises(ValueError):m.primary_authority(primary,terminal,pins)

  def test_current_sha_may_differ_only_for_audit_or_results(self):
    self.assertTrue(m.allowed_diff('scripts/dl_reset_joint_audit.py'))
    self.assertTrue(m.allowed_diff('docs/DL_RESET_JOINT_RESULTS.md'))
    self.assertFalse(m.allowed_diff('dreamerv3/rssm.py'))
    self.assertFalse(m.allowed_diff('scripts/dl_reset_readout.py'))
    with mock.patch.object(m,'git_output',side_effect=[b'b'*40+b'\n',b'',b'dreamerv3/rssm.py\n']):
      with self.assertRaises(ValueError):m.git_binding('b'*40,'a'*40)

  def test_actual_128_counter_pressure_receipts_and_corruption(self):
    value,source=learning_fixture();result=m.validate_learning(value,'B',source)
    self.assertEqual(result['actual_updates'],128);self.assertEqual(result['reset_count'],2048)
    self.assertFalse(result['optimizer_independently_reexecuted'])
    for change in (lambda x:x['receipts'].pop(),
        lambda x:x['receipts'][65]['counters'].update(updates=0),
        lambda x:x['receipts'][4]['reset_pressure'].update(active_count=17),
        lambda x:x['binding']['layout'][0].update(context_position=0),
        lambda x:x['reset_pressure'].update(excess_sum=999.),
        lambda x:x.update(reset_rep_gradient_stopped=True)):
      bad=copy.deepcopy(value);change(bad)
      with self.assertRaises(ValueError):m.validate_learning(bad,'B',source)

  def test_native_features_common_32_and_private_roots(self):
    cal,cal_numeric=features_fixture();ref=m.validate_features(cal,cal_numeric,m.CAL)
    blind,numeric=features_fixture(24);m.validate_features(blind,numeric,m.BLIND,ref)
    for change in (lambda x:x['aux']['root_key'].__setitem__((7,1,0),123),
        lambda x:x['aux']['prior_logit'].__setitem__((4,2,0,0),1),
        lambda x:x['aux']['h'].__setitem__((2,3,0),1),
        lambda x:x['aux'].update(public_policy_copy_used=True),
        lambda x:x['aux'].update(full_state_after_sha256='b'*64),
        lambda x:x['aux']['common_reference'].update(verified_observations=24)):
      bad=copy.deepcopy(blind);change(bad)
      with self.assertRaises(ValueError):m.validate_features(bad,numeric,m.BLIND,ref)

  def test_sum_tolerance_never_relaxes_delta_or_signed_utility(self):
    labels=np.zeros((24,7),np.float64);labels[:,0]=.2
    m.verify_reward_boundaries(labels.copy(),labels)
    changed=labels.copy();changed[0,0]=np.nextafter(.2,np.inf)
    self.assertLess(np.max(np.abs(changed-labels)),1e-10)
    with self.assertRaises(ValueError):m.verify_reward_boundaries(changed,labels)
    labels[:]=0;changed=labels.copy();changed[0,4]=1e-12
    with self.assertRaises(ValueError):m.verify_reward_boundaries(changed,labels)

  def test_utility_bootstrap_exact_zero_side_is_not_a_numeric_tolerance(self):
    labels=np.zeros((24,7),np.float64);labels[:,1]=np.tile([1.,-1.],12)
    readouts=dict(frozen={'fit':{'constant_choice':0}},refits={g:{'constant_choice':0} for g in ('B','R')})
    result=dict(analysis={'models':{g:{'choices':np.tile([1,0],(24,1)).tolist()} for g in ('0','B','R')}})
    m.verify_assay_utility_boundaries(labels,labels,readouts,result)
    changed=labels.copy();changed[0,1]+=1e-12
    self.assertTrue(np.array_equal(np.sign(changed[:,1]),np.sign(labels[:,1])))
    with self.assertRaises(ValueError):m.verify_assay_utility_boundaries(changed,labels,readouts,result)

  def test_fixed_cal_truth_only_decodes_new_numeric_reward_products(self):
    with tempfile.TemporaryDirectory() as d:
      root=Path(d);cal=root/'calibration-attempt01';cal.mkdir()
      (root/'engineering').mkdir();(root/'independent-audit01').mkdir()
      rewards=np.tile(np.arange(20,dtype=np.float64)/32,(7,1));items=[]
      for ep in m.CAL:
        name=f'episode_{ep:03}.npz';np.savez(cal/name,reward_f64=rewards)
        items.append(dict(episode=ep,path=name,sha256=m.sha(cal/name)))
      budget_sha=write(cal,'budget.json',dict(synthetic=True))
      write(cal,'plan.json',dict(synthetic=True))
      result_sha=write(cal,'result.json',dict(evidence_artifacts=items))
      git='aa9fb7f777a28e8a82d57c31f9e749b5912af316'
      complete_sha=write(cal,'natural_reset_complete.json',dict(actual_git_sha=git,budget_sha256=budget_sha))
      write(cal,'supervisor_complete.json',dict(exit_code=0,timed_out=False,worker_terminal=True,
          complete_verified=True,budget_sha256=budget_sha))
      audit_path=root/'independent-audit01/audit.json'
      m.write_json(audit_path,dict(status='passed',primary_json_sha256={name:m.sha(cal/name) for name in
          ('budget.json','plan.json','result.json','natural_reset_complete.json','supervisor_complete.json')}))
      terminal=root/'engineering/primary-terminal01.json'
      m.write_json(terminal,dict(exit_code=0,actual_git_sha=git))
      pins={'result.json':result_sha,'natural_reset_complete.json':complete_sha,
          'audit':m.sha(audit_path),'external':m.sha(terminal)}
      Y=np.tile(rewards.sum(-1),(8,1));readouts=dict(cal={g:dict(returns=Y.copy()) for g in ('0','B','R')})
      with mock.patch.object(m,'CAL_ROOT',cal),mock.patch.object(m,'CAL_PINS',pins):
        report=m.calibration_truth(readouts)
        self.assertEqual(report['independent_physics_steps'],0)
        self.assertTrue(report['prior_completed_cal_audit_authenticated'])
        (cal/'episode_009.npz').write_bytes(b'tamper')
        with self.assertRaises(ValueError):m.calibration_truth(readouts)
  def test_original_source_authentication_only_hashes_archives(self):
    with tempfile.TemporaryDirectory() as d:
      root=Path(d);dataset=root/'collection';dataset.mkdir()
      episodes=[dict(episode=e,seed=100+e,split='calibration' if e in m.CAL else 'blind') for e in range(32)]
      artifacts=[];bound=[]
      for ep in range(32):
        path=dataset/f'episode_{ep:03}.npz';path.write_bytes(b'NOT AN NPZ OR PICKLE '+str(ep).encode())
        pin=m.sha(path);artifacts.append(dict(episode=ep,path=path.name,sha256=pin))
        bound.append(dict(episode=ep,path=str(path),sha256=pin))
      plan=dict(metadata=m.MODEL_PINS.copy(),episodes=episodes)
      complete=dict(artifacts=artifacts)
      plan_sha=write(dataset,'collection_plan.json',plan)
      complete_sha=write(dataset,'collection_complete.json',complete)
      budget_sha=write(dataset,'budget.json',dict(version='DL-source-r1',git_sha=m.SOURCE_GIT,seconds=3600))
      supervisor_sha=write(dataset,'supervisor_complete.json',dict(version='DL-source-r1',git_sha=m.SOURCE_GIT,
          exit_code=0,timed_out=False,worker_terminal=True,budget_sha256=budget_sha,elapsed_seconds=20.))
      (dataset/'worker_exit_code.txt').write_bytes(b'0\n')
      metadata={name:m.sha(dataset/name) for name in ('collection_plan.json','collection_complete.json',
          'budget.json','supervisor_complete.json','worker_exit_code.txt')}
      artifact_path=root/'artifact.json';physics_path=root/'physics.json'
      m.write_json(artifact_path,dict(source_contract_artifact_audit_passed=True,
          expected_source_sha=m.SOURCE_GIT,metadata_sha256=metadata))
      m.write_json(physics_path,dict(status='passed',expected_source_sha=m.SOURCE_GIT,
          blind_npz_arrays_opened=False,physics_steps=12800))
      pins=dict(plan=plan_sha,complete=complete_sha,artifact_audit=m.sha(artifact_path),physics_audit=m.sha(physics_path))
      binding=dict(source_git_sha=m.SOURCE_GIT,source_json_sha256=pins,
          source_model_metadata=plan['metadata'],
          source_original_exit_chain_verified=True,source_dataset=str(dataset),
          source_json_paths=dict(plan=str(dataset/'collection_plan.json'),complete=str(dataset/'collection_complete.json'),
            artifact_audit=str(artifact_path),physics_audit=str(physics_path)),
          source_terminal_paths={k:str(dataset/k) for k in metadata if k in
            ('budget.json','supervisor_complete.json','worker_exit_code.txt')},
          source_terminal_sha256={k:v for k,v in metadata.items() if k in
            ('budget.json','supervisor_complete.json','worker_exit_code.txt')},all_32_artifact_hashes=bound)
      with (mock.patch.object(m,'SOURCE_PINS',pins),
          mock.patch.object(np,'load',side_effect=AssertionError('No source arrays')),
          mock.patch.object(m.pickle,'loads',side_effect=AssertionError('No source pickle'))):
        self.assertEqual(m.authenticate_source(binding)['episodes'],episodes)
        (dataset/'episode_029.npz').write_bytes(b'TAMPERED')
        with self.assertRaises(ValueError):m.authenticate_source(binding)

  def test_owned_generic_success_never_claims_real_completion(self):
    with tempfile.TemporaryDirectory() as d:
      result=m.supervise([sys.executable,'-c','print("synthetic")'],Path(d)/'output',
          seconds=3,verify=lambda *_:dict(complete_verified=True))
      self.assertEqual(result['exit_code'],0);self.assertFalse(result['complete_verified'])
      self.assertIsNone(result['actual_physics_steps']);self.assertFalse(m.OWNED)

  def test_owned_timeout_and_disk_receipt(self):
    with tempfile.TemporaryDirectory() as d:
      output=Path(d)/'output'
      result=m.supervise([sys.executable,'-c','import time;time.sleep(3)'],output,seconds=.2)
      self.assertEqual(result['exit_code'],124);self.assertFalse(result['complete_verified'])
      self.assertTrue(result['worker_terminal']);self.assertFalse(m.OWNED)
      self.assertFalse(m.read_json(output/'supervisor_complete.json')['complete_verified'])

  def test_late_receipt_cannot_publish_stale_success(self):
    with tempfile.TemporaryDirectory() as d:
      output=Path(d)/'output';original=m.write_json
      def delayed(path,value):
        original(path,value)
        if Path(path).name == 'supervisor-receipt-pending.json':time.sleep(.3)
      with mock.patch.object(m,'write_json',side_effect=delayed):
        result=m.supervise([sys.executable,'-c','pass'],output,seconds=.25)
      self.assertEqual(result['exit_code'],124)
      self.assertEqual(m.read_json(output/'supervisor_complete.json')['exit_code'],124)
      self.assertFalse(m.read_json(output/'supervisor_complete.json')['complete_verified'])
      self.assertTrue((output/'late-supervisor.json').exists())

  def test_independent_replay_reuses_byte_oracle_and_whole_deadline(self):
    from tests.test_dl_natural_reset_audit import record
    from analysis import dl_natural_reset_audit as old
    row=record(1)
    class TimeStep:
      def __init__(self,t):self.reward=row['reward_f64'][t];self.discount=row['discount'][t]
      def first(self):return False
      def last(self):return False
    class Env:
      _reset_next_step=False
      def __init__(self):self.index=0
      def step(self,action):
        if not old.byte_equal(action,row['applied'][self.index]):raise AssertionError('Wrong applied')
        result=TimeStep(self.index);self.index+=1;return result
    def restore(env,saved):env.index=saved['suite']['step_count']
    def snapshot(env,done):return copy.deepcopy(row['snapshots'][env.index])
    with mock.patch.object(old,'check_deadline',m.check_deadline):
      receipt=old.replay_record(Env(),row,restore,snapshot)
      self.assertEqual(receipt['physics_steps'],20);self.assertTrue(receipt['complete_state_byte_exact'])
      with mock.patch.object(m.time,'monotonic',return_value=m.ENTRY_STARTED+300):
        with self.assertRaises(ValueError):old.replay_record(Env(),row,restore,snapshot)

  @unittest.skipUnless(os.name == 'posix','POSIX process group fixture')
  def test_only_owned_process_group_is_killed(self):
    unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'])
    try:
      with tempfile.TemporaryDirectory() as d:
        result=m.supervise([sys.executable,'-c',
            'import subprocess,sys,time;subprocess.Popen([sys.executable,"-c","import time;time.sleep(10)"]);time.sleep(10)'],
            Path(d)/'output',seconds=.2)
        self.assertEqual(result['exit_code'],124);self.assertIsNone(unrelated.poll())
    finally:unrelated.kill();unrelated.wait()


if __name__ == '__main__':unittest.main()
