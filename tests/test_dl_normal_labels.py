"""Generated real CPU prefixes and mocked512 graph; never real R5 acceptance.

RGB/raw are generated fixture data. Official reset/physics/reward/query really
execute in per-record tests; complete-source authority is separately mocked
only in explicitly named coordinator tests. Production CLI has no fixture mode.
"""
import copy
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('MUJOCO_GL','disable')
import numpy as np
from scripts import dl_normal_labels as labels
from scripts import dl_normal_source_index as source

ROOT=Path(__file__).resolve().parents[1]


def generated_record():
  from dm_control import suite
  from analysis import dl_live_audit as a
  unit=next(i for i in range(512) if source.plan(i)['t'] == 64)
  plan=source.plan(unit);t=plan['t'];n=t+1
  env=suite.load('quadruped','walk',task_kwargs={'random':plan['env_seed']})
  try:
    ts=env.reset();model=a.static_model(env);spec=env.action_spec()
    raw=np.linspace(-.7,.7,n*12,dtype=np.float32).reshape(n,12)
    applied=a.physics.mapping(raw,spec.minimum,spec.maximum)
    observations=[];snapshots=[a.physics.snapshot(env,False)];truth=None
    def observation(step,reward):
      return dict(image=np.full((64,64,3),step%256,np.uint8),reward=np.float32(reward),
          is_first=np.bool_(step == 0),is_last=np.bool_(False),is_terminal=np.bool_(False))
    observations.append(observation(0,0));rewards=[];discounts=[]
    for step in range(n):
      ts=env.step(applied[step]);rewards.append(np.asarray(ts.reward).copy());discounts.append(np.asarray(ts.discount).copy())
      snapshots.append(a.physics.snapshot(env,ts.last()));observations.append(observation(step+1,ts.reward))
      if step+1 == t:
        truth=dict(velocity_x=np.asarray(env.physics.torso_velocity()[0]).copy(),
            upright=np.asarray(env.physics.torso_upright()).copy(),reward=np.asarray(ts.reward).copy())
    rewards=np.stack(rewards);discounts=np.stack(discounts)
    record=dict(version='DL-r5-live-trajectory-r1',unit=unit,plan=plan,static_model=model,
        executed_steps=n,observations=observations,snapshots=snapshots,raw=raw,applied=applied,
        keys=np.stack([source.root_key(unit,i) for i in range(n)]),reward_f64=rewards,
        reward_f32=rewards.astype(np.float32),discount=discounts,
        **{name:np.zeros(n,bool) for name in ('is_first','is_last','is_terminal')},
        D=object(),e=object(),Y=object(),initial_carry=object(),bundle=object())
    trajectory=dict(path=f'family_{unit//16:02d}/unit_{unit%16:02d}/trajectory.pkl',bytes=123,sha256='a'*64)
    row=source.project_prefix(record,unit,trajectory,'b'*64)
    return record,row,truth
  finally:env.close()


def generated_index(args):
  from tests.test_dl_normal_source_index import ordered_rows
  rows=ordered_rows()
  for row in rows:row['prefix_typed_sha256']='f'*64
  pins={}
  for name,hash_name in labels.PIN_FIELDS:
    path=Path(args.run).parent/(name+'.json');path.write_text('{}')
    setattr(args,name,str(path));setattr(args,hash_name,source.digest(path.read_bytes()))
    pins[name]=source.raw_pin(path)
  return dict(version=source.VERSION,source_kind='complete-audited-R5-pretreatment',run_git=source.RUN_GIT,
      metadata_authentication_git=args.expected_git,prior_physics_audit_git=args.prior_physics_audit_git,
      run_directory=str(Path(args.run).resolve()),complete_units=512,complete_families=32,
      fit_families=list(range(8)),held_families=list(range(8,32)),fit_rows=128,held_rows=384,
      source_graph_and_prior_physics_audit_authenticated=True,rows=rows,run_inventory_sha256='b'*64,
      prior_dependency_pins={'generated':'fixture'},external_pins=pins,original_authority={'generated':'fixture'})


class NormalLabelsTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.record,cls.row,cls.truth=generated_record()

  def test_real_cpu_seed_to_current_labels_and_never_next_reward_or_action(self):
    from analysis import dl_live_audit as a
    from dm_control import suite
    from dm_control.suite import quadruped
    record=copy.deepcopy(self.record);t=record['plan']['t'];before=pickle.dumps(record,protocol=5)
    real_load=suite.load;loaded=[]
    def capture(*args,**kwargs):
      env=real_load(*args,**kwargs);loaded.append(env);return env
    with mock.patch.object(suite,'load',side_effect=capture), \
        mock.patch.object(quadruped.Physics,'render',side_effect=AssertionError('render forbidden')), \
        mock.patch.object(a.physics,'restore',side_effect=AssertionError('restore forbidden')):
      result=labels.label_record(record,self.row,'b'*64,a)
    self.assertEqual(len(loaded),1)
    self.assertEqual(loaded[0]._step_count,t)
    self.assertEqual(result['replayed_steps'],t)
    self.assertEqual(result['source_reward_index'],t-1)
    self.assertEqual(result['current_values']['velocity_x'],float(self.truth['velocity_x']))
    self.assertEqual(result['current_values']['upright'],float(self.truth['upright']))
    self.assertEqual(result['current_f64_hex']['current_reward_f64'],self.truth['reward'].tobytes().hex())
    self.assertEqual(result['source_id'],self.row['source_id'])
    self.assertEqual(result['prefix_typed_sha256'],self.row['prefix_typed_sha256'])
    self.assertNotEqual(result['source_snapshot_hash_format'],result['query_snapshot_hash_format'])
    self.assertEqual(result['query_before_snapshot_sha256'],result['query_after_snapshot_sha256'])
    self.assertEqual(result['task_rng_before_sha256'],result['task_rng_after_sha256'])
    self.assertFalse(result['current_source_raw_used'])
    self.assertEqual(pickle.dumps(record,protocol=5),before)
    # The randomized intervention/outcome bytes must not affect a current label.
    record['raw'][t]=2.75;record['applied'][t]=-999
    record['reward_f64'][t]=123.;record['reward_f32'][t]=123.
    record['observations'][t+1]['reward']=np.float32(123.)
    self.assertEqual(labels.label_record(record,self.row,'b'*64,a),result)

  def test_wrong_id_clock_snapshot_static_model_reward_or_mapping_rejected(self):
    from analysis import dl_live_audit as a
    for case in ('source_id','clock','snapshot_ulp','static_model','F64_ulp','mapping'):
      record=copy.deepcopy(self.record);row=copy.deepcopy(self.row);t=record['plan']['t']
      if case == 'source_id':row['source_id']='0'*64
      elif case == 'clock':record['snapshots'][t]['suite']['step_count']=t-1
      elif case == 'snapshot_ulp':
        value=record['snapshots'][t]['suite']['integration'][1]
        record['snapshots'][t]['suite']['integration'][1]=np.nextafter(value,np.float64(np.inf))
      elif case == 'static_model':record['static_model']['mjb_sha256']='0'*64
      elif case == 'F64_ulp':record['reward_f64'][t-1]=np.nextafter(record['reward_f64'][t-1],np.float64(np.inf))
      else:record['applied'][1,0]=np.nextafter(record['applied'][1,0],np.float64(np.inf))
      if case in ('snapshot_ulp','F64_ulp','mapping'):
        # Admit changed fixture metadata so the real physics/reward exact guard,
        # rather than the prefix hash alone, must reject these counterexamples.
        row=source.project_prefix(record,record['unit'],row['trajectory'],'b'*64)
      with self.subTest(case=case),self.assertRaises(ValueError):labels.label_record(record,row,'b'*64,a)

  def test_import_is_lazy_and_production_cli_requires_all_external_pins(self):
    code="import os,sys;before=dict(os.environ);import scripts.dl_normal_labels;assert before==dict(os.environ);assert not any(n.split('.')[0] in ('jax','dm_control','mujoco','OpenGL') for n in sys.modules)"
    result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=30)
    self.assertEqual(result.returncode,0,result.stderr)
    result=subprocess.run([sys.executable,str(Path(labels.__file__)),'--run','unused','--expected-git','a'*40],
        cwd=ROOT,capture_output=True,text=True,timeout=30)
    self.assertEqual(result.returncode,2)
    for flag in ('--prior-physics-audit-git','--launcher-terminal','--launcher-sha','--launcher-launch',
        '--launcher-launch-sha','--audit-report-sha','--audit-terminal-sha','--audit-launch-sha'):
      self.assertIn(flag,result.stderr)

  def test_generated512_coordinator_closes_only_all_rows_and_reauthenticates(self):
    # The real labels CLI is a fresh CPU process and must never inherit a model
    # runtime. Other modules in the combined engineering suite legitimately
    # import JAX; keep this mocked coordinator in the same isolation as its CLI.
    isolated='DL_NORMAL_LABELS_COORDINATOR_TEST_PROCESS'
    if os.environ.get(isolated) != '1':
      result=subprocess.run([sys.executable,'-m','unittest',
          'tests.test_dl_normal_labels.NormalLabelsTest.test_generated512_coordinator_closes_only_all_rows_and_reauthenticates','-v'],
          cwd=ROOT,capture_output=True,text=True,timeout=30,
          env=dict(os.environ,**{isolated:'1'},CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',MUJOCO_GL='disable'))
      self.assertEqual(result.returncode,0,result.stdout[-2000:]+result.stderr[-4000:])
      return
    self.assertNotIn('jax',sys.modules)
    for case in ('success','partial','reordered','late_failure'):
      with self.subTest(case=case),tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);run=root/'run';run.mkdir()
        args=SimpleNamespace(run=str(run),output=str(root/'labels.json'),expected_git='c'*40,prior_physics_audit_git='7'*40)
        index=generated_index(args);a=SimpleNamespace(DEADLINE=math.inf)
        if case == 'partial':index['rows'].pop()
        elif case == 'reordered':index['rows'][0],index['rows'][1]=index['rows'][1],index['rows'][0]
        def item(record,row,inventory,a):
          return dict(source_id=row['source_id'],replayed_steps=row['plan']['t'])
        with mock.patch.object(source,'authenticate_and_index',return_value=index) as auth, \
            mock.patch.object(source,'AUDITOR',a),mock.patch.object(labels,'_bind_current',return_value={'generated':'fixture'}), \
            mock.patch.object(labels,'_load_record',return_value={'generated':'fixture'}) as load, \
            mock.patch.object(labels,'label_record',side_effect=item) as label, \
            mock.patch.object(labels,'_reauthenticate',side_effect=ValueError('source changed') if case == 'late_failure' else None) as close:
          if case == 'success':
            result=labels.run(args);self.assertEqual(result['complete_units'],512)
            self.assertEqual(len(result['rows']),512);self.assertFalse(result['scientific_verdict'])
            self.assertFalse(result['D_e_Y_used_for_labels_or_selection'])
            self.assertEqual(label.call_count,512);close.assert_called_once_with(a,args,index)
            self.assertEqual(json.loads(Path(args.output).read_text()),result)
          else:
            with self.assertRaises(ValueError):labels.run(args)
            self.assertFalse(Path(args.output).exists())
            if case != 'late_failure':load.assert_not_called()
          auth.assert_called_once_with(args)
          for name,hash_name in labels.PIN_FIELDS:
            self.assertTrue(getattr(auth.call_args.args[0],name));self.assertTrue(getattr(auth.call_args.args[0],hash_name))

  def test_reauthentication_never_uses_fixture_bypass_or_drops_original_four_pins(self):
    with tempfile.TemporaryDirectory() as temporary:
      root=Path(temporary);run=root/'run';run.mkdir()
      args=SimpleNamespace(run=str(run),expected_git='c'*40,prior_physics_audit_git='7'*40)
      index=generated_index(args);stock=dict(files={},directories=[])
      index['run_inventory_sha256']=source.digest(source.canonical(stock))
      a=SimpleNamespace(CLI_AUTHORITY=None,OWNED={},external_authority=mock.Mock(return_value=index['original_authority']),
          inventory=mock.Mock(return_value=stock),authenticate=None)
      def authenticate(*call):
        self.assertIsNotNone(a.CLI_AUTHORITY)
        self.assertEqual(a.CLI_AUTHORITY.launcher_terminal,args.launcher_terminal)
        self.assertEqual(a.CLI_AUTHORITY.launcher_sha,args.launcher_sha)
        self.assertEqual(a.CLI_AUTHORITY.launcher_launch,args.launcher_launch)
        self.assertEqual(a.CLI_AUTHORITY.launcher_launch_sha,args.launcher_launch_sha)
        return dict(inventory=stock)
      a.authenticate=mock.Mock(side_effect=authenticate)
      with mock.patch.object(source,'_physics_authority',return_value=dict(inventory=stock)), \
          mock.patch.object(source,'_verify_dependencies',return_value=index['prior_dependency_pins']):
        labels._reauthenticate(a,args,index)
        a.authenticate.assert_called_once_with(args.run,source.RUN_GIT,args.expected_git)
        self.assertIsNone(a.CLI_AUTHORITY)
        args.launcher_launch_sha=None
        with self.assertRaises(ValueError):labels._reauthenticate(a,args,index)
        self.assertEqual(a.authenticate.call_count,1)


if __name__ == '__main__':unittest.main()
