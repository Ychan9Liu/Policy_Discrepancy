"""Generated source-index contracts; no real R5 values/source/N1M acceptance."""
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

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import dl_normal_source_index as s


def synthetic_record(unit=0):
  p=s.plan(unit);t=p['t'];n=t+1
  raw=np.linspace(-.8,.8,n*12,dtype=np.float32).reshape(n,12)
  rewards=np.arange(n,dtype=np.float64)/np.float64(1000)
  flags={name:np.zeros(n,bool) for name in ('is_first','is_last','is_terminal')}
  flags['is_last'][-1]=flags['is_terminal'][-1]=True
  observations=[];snapshots=[];rng=np.random.RandomState(p['env_seed']);rng.normal(size=3)
  for i in range(n+1):
    observations.append(dict(image=np.full((64,64,3),i%256,np.uint8),
        reward=np.float32(0 if i == 0 else rewards[i-1]),is_first=np.bool_(i == 0),
        is_last=np.bool_(i == n),is_terminal=np.bool_(i == n)))
    snapshots.append(dict(suite=dict(integration=np.array([i/50,.2],np.float64),step_count=i,
        reset_next_step=i == n,task=dict(_desired_speed=.5,_random=copy.deepcopy(rng))),from_dm_done=i == n))
  return dict(version='DL-r5-live-trajectory-r1',unit=unit,plan=p,executed_steps=n,
      observations=observations,snapshots=snapshots,raw=raw,applied=raw.astype(np.float64),
      keys=np.stack([s.root_key(unit,i) for i in range(n)]),reward_f64=rewards,
      reward_f32=rewards.astype(np.float32),discount=np.ones(n,np.float64),**flags,
      initial_carry=object(),bundle=object(),D=object(),e=object(),Y=object())


def trajectory_pin(unit):
  return dict(path=f'family_{unit//16:02d}/unit_{unit%16:02d}/trajectory.pkl',bytes=123,sha256='a'*64)


def ordered_rows():
  result=[]
  for i in range(512):
    p=s.plan(i);pin=trajectory_pin(i)
    result.append(dict(global_unit=i,family=i//16,replicate=i%16,plan=p,trajectory=pin,
        split='fit' if i//16 < 8 else 'held',
        source_id=s.digest(s.canonical(dict(run_inventory_sha256='b'*64,trajectory=pin,plan=p)))))
  return result


def arguments(root):
  run=root/'run';run.mkdir()
  values=dict(run=str(run),expected_git='c'*40,prior_physics_audit_git='7'*40)
  for field,hashfield in (('launcher_terminal','launcher_sha'),('launcher_launch','launcher_launch_sha'),
      ('audit_report','audit_report_sha'),('audit_terminal','audit_terminal_sha'),('audit_launch','audit_launch_sha')):
    path=root/(field+'.json');path.write_text('{}')
    values[field]=str(path);values[hashfield]=s.digest(path.read_bytes())
  return SimpleNamespace(**values)


class SourceIndexTests(unittest.TestCase):
  def test_import_has_no_model_environment_or_authority_side_effect_and_cli_needs_four_pins(self):
    code="import os,sys;before=dict(os.environ);import scripts.dl_normal_source_index;assert before==dict(os.environ);assert not any(n.split('.')[0] in ('jax','dm_control','mujoco','OpenGL') for n in sys.modules)"
    result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=30)
    self.assertEqual(result.returncode,0,result.stderr)
    result=subprocess.run([sys.executable,str(Path(s.__file__)), '--run','unused','--expected-git','a'*40],
        cwd=ROOT,capture_output=True,text=True,timeout=30)
    self.assertEqual(result.returncode,2)
    for flag in ('--prior-physics-audit-git','--launcher-terminal','--launcher-sha','--launcher-launch','--launcher-launch-sha'):
      self.assertIn(flag,result.stderr)

  def test_pretreatment_boundary_has_no_current_source_raw_or_old_latent(self):
    record=synthetic_record();t=record['plan']['t']
    row=s.project_prefix(record,0,trajectory_pin(0),'b'*64)
    self.assertEqual(row['bounds']['observations'],[0,t+1])
    self.assertEqual(row['bounds']['raw'],[0,t]);self.assertEqual(row['bounds']['applied'],[0,t])
    self.assertEqual(len(row['frames']),t+1)
    self.assertIsNone(row['frames'][0]['previous_raw_index'])
    self.assertEqual(row['frames'][-1]['previous_raw_index'],t-1)
    self.assertEqual(row['frames'][-1]['previous_raw_sha256'],s.tree_digest(record['raw'][t-1]))
    self.assertEqual(row['current']['preceding_reward_index'],t-1)
    self.assertEqual(row['current']['snapshot_index'],t)
    self.assertFalse(row['current']['source_raw_available'])
    self.assertFalse(row['current']['physical_is_last'])
    self.assertFalse(row['projection_authenticates_source'])
    self.assertFalse(row['old_latent_used_as_feature'])
    original=row['prefix_typed_sha256']
    # These are the randomized intervention/outcome side of the boundary.
    # They are not normal history actions and cannot alter its typed prefix.
    record['raw'][t]=.123;record['applied'][t]=.456
    record['observations'][t+1]['image'].fill(234)
    record['reward_f64'][t]=.777
    after=s.project_prefix(record,0,trajectory_pin(0),'b'*64)
    self.assertEqual(original,after['prefix_typed_sha256'])
    serialized=json.dumps(after)
    for forbidden in ('initial_carry','bundle','qlogits','psample','outcome','pretreatment'):
      self.assertNotIn(forbidden,serialized)

  def test_wrong_transition_offset_key_clock_or_plan_is_rejected(self):
    for name in ('reward_shift','flags_shift','current_key','clock','reset','raw_dtype','unit_plan'):
      record=synthetic_record();t=record['plan']['t']
      if name == 'reward_shift':record['observations'][t]['reward']=record['reward_f32'][t]
      elif name == 'flags_shift':record['is_last'][t-1]=True
      elif name == 'current_key':record['keys'][t]=record['keys'][t-1]
      elif name == 'clock':record['snapshots'][t]['suite']['step_count']-=1
      elif name == 'reset':record['observations'][t]['is_first']=np.bool_(True)
      elif name == 'raw_dtype':record['raw']=record['raw'].astype(np.float64)
      else:record['plan']=s.plan(1)
      with self.subTest(name=name),self.assertRaises(ValueError):
        s.project_prefix(record,0,trajectory_pin(0),'b'*64)

  def test_trusted_task_randomstate_hash_matches_original_r5_and_preserves_rng(self):
    from scripts import dl_live_shadow
    record=synthetic_record();t=record['plan']['t'];saved=record['snapshots'][t]
    rng=saved['suite']['task']['_random'];before=pickle.dumps(rng.get_state(),protocol=5)
    self.assertIs(type(rng),np.random.RandomState);self.assertEqual(rng.get_state()[3],1)
    original=s.snapshot_sha256(saved)
    self.assertEqual(original,dl_live_shadow.snapshot_sha256(saved))
    self.assertEqual(before,pickle.dumps(rng.get_state(),protocol=5))
    row=s.project_prefix(record,0,trajectory_pin(0),'b'*64)
    self.assertEqual(row['current']['snapshot_sha256'],original)
    self.assertEqual(row['current']['snapshot_hash_format'],s.SNAPSHOT_HASH_FORMAT)
    self.assertEqual(row['frames'][t]['snapshot_sha256'],original)
    self.assertEqual(before,pickle.dumps(rng.get_state(),protocol=5))
    rng.normal()
    self.assertNotEqual(original,s.snapshot_sha256(saved))
    changed=s.project_prefix(record,0,trajectory_pin(0),'b'*64)
    self.assertNotEqual(row['prefix_typed_sha256'],changed['prefix_typed_sha256'])
    restored=pickle.loads(pickle.dumps(saved,protocol=5))
    self.assertEqual(s.snapshot_sha256(restored),s.snapshot_sha256(saved))

  def test_equal_counts_do_not_authenticate_source_ids_order_or_split(self):
    original=ordered_rows();s.validate_order(original,'b'*64)
    self.assertEqual(sum(x['split'] == 'fit' for x in original),128)
    self.assertEqual(sum(x['split'] == 'held' for x in original),384)
    for case in ('swap','source_id','duplicate','trajectory','split','partial'):
      rows=copy.deepcopy(original)
      if case == 'swap':rows[0],rows[1]=rows[1],rows[0]
      elif case == 'source_id':rows[0]['source_id']=rows[1]['source_id']
      elif case == 'duplicate':rows[1]=copy.deepcopy(rows[0])
      elif case == 'trajectory':
        rows[0]['trajectory']=copy.deepcopy(rows[1]['trajectory'])
        rows[0]['source_id']=s.digest(s.canonical(dict(run_inventory_sha256='b'*64,
            trajectory=rows[0]['trajectory'],plan=rows[0]['plan'])))
      elif case == 'split':rows[128]['split']='fit'
      else:rows.pop()
      with self.subTest(case=case),self.assertRaises(ValueError):s.validate_order(rows,'b'*64)

  def test_output_resolves_dotdot_and_rejects_parent_alias_before_writing(self):
    with tempfile.TemporaryDirectory() as temporary:
      root=Path(temporary);run=root/'run';run.mkdir();external=root/'outside';external.mkdir()
      self.assertEqual(s.fresh_output(external/'index.json',run),(external/'index.json').resolve())
      with self.assertRaises(ValueError):s.fresh_output(external/'..'/'run'/'index.json',run)
      existing=external/'old.json';existing.write_bytes(b'preserve')
      with self.assertRaises(ValueError):s.fresh_output(existing,run)
      self.assertEqual(existing.read_bytes(),b'preserve')
      alias=root/'alias'
      try:alias.symlink_to(run,target_is_directory=True)
      except OSError:self.skipTest('OS symlink permission unavailable; clean Linux must exercise alias refusal')
      with self.assertRaises(ValueError):s.fresh_output(alias/'index.json',run)
      self.assertFalse((run/'index.json').exists())

  def test_bad_or_omitted_external_pin_is_rejected_before_import_or_decode(self):
    with tempfile.TemporaryDirectory() as temporary:
      args=arguments(Path(temporary))
      with mock.patch.object(s,'_load_auditor') as load:
        args.launcher_sha='0'*64
        with self.assertRaisesRegex(ValueError,'raw pin'):s.authenticate_and_index(args)
        args.launcher_sha=s.digest(Path(args.launcher_terminal).read_bytes());args.launcher_launch_sha=None
        with self.assertRaisesRegex(ValueError,'four producer pins'):s.authenticate_and_index(args)
        args.launcher_launch_sha=s.digest(Path(args.launcher_launch).read_bytes())
        for value in (None,'7'*7,'G'*40):
          args.prior_physics_audit_git=value
          with self.assertRaisesRegex(ValueError,'successful prior audit Git'):s.authenticate_and_index(args)
        load.assert_not_called()

  def test_metadata_dependencies_bind_the_explicit_prior_git_and_actual_imported_bytes(self):
    with tempfile.TemporaryDirectory() as temporary:
      root=Path(temporary);name='audit.py';body=b'generated dependency bytes\n'
      (root/name).write_bytes(body)
      module=SimpleNamespace(__file__=str(root/name));module.physics=module.primitive=module.statistics=module
      module.command_output=mock.Mock(return_value=b'0'*40+b' blob '+str(len(body)).encode()+b'\n'+body+b'\n')
      with mock.patch.object(s,'ROOT',root),mock.patch.object(s,'DEPENDENCIES',(name,)):
        result=s._verify_dependencies(module,'7'*40)
        self.assertEqual(result[name],dict(bytes=len(body),sha256=s.digest(body)))
        self.assertEqual(module.command_output.call_args.args[1],('7'*40+':'+name+'\n').encode())
        (root/name).write_bytes(b'different present code')
        with self.assertRaisesRegex(ValueError,'successful prior audit'):s._verify_dependencies(module,'7'*40)

  def test_partial_reordered_or_changed_whole_graph_fails_before_trusted_pickle(self):
    for case in ('partial','reordered','changed_graph'):
      with self.subTest(case=case),tempfile.TemporaryDirectory() as temporary:
        args=arguments(Path(temporary));stock=dict(files={},directories=[])
        units=[dict(unit=i,prefix=f'family_{i//16:02d}/unit_{i%16:02d}/') for i in range(512)]
        if case == 'partial':units.pop()
        elif case == 'reordered':units[0],units[1]=units[1],units[0]
        a=SimpleNamespace(CLI_AUTHORITY=None,DEADLINE=math.inf,OWNED={},
            external_authority=mock.Mock(return_value={}),
            authenticate=mock.Mock(return_value=dict(inventory=stock,units=units)),
            inventory=mock.Mock(return_value=stock if case != 'changed_graph' else dict(files={'extra':{}},directories=[])),
            trusted_pickle=mock.Mock())
        with mock.patch.object(s,'_load_auditor',return_value=a), \
            mock.patch.object(s,'_verify_dependencies',return_value={}), \
            mock.patch.object(s,'_physics_authority',return_value=dict(inventory=stock)):
          with self.assertRaises(ValueError):s.authenticate_and_index(args)
        a.trusted_pickle.assert_not_called()
        self.assertEqual(a.external_authority.call_args.args[0].launcher_launch_sha,args.launcher_launch_sha)
        self.assertIsNone(a.CLI_AUTHORITY)

  def test_actual_prior_audit_exit_schema_is_bound_without_reading_scientific_statistics(self):
    with tempfile.TemporaryDirectory() as temporary:
      root=Path(temporary);args=arguments(root)
      script=root/'launch.py';script.write_bytes(b'generated fixture controller')
      log=root/'audit.log';log.write_bytes(b'generated fixture log')
      pins={name:s.raw_pin(getattr(args,name)) for name in
          ('launcher_terminal','launcher_launch','audit_report','audit_terminal','audit_launch')}
      identity=dict(pid=321,pgid=321,sid=321,start_ticks=54,parent_pid=300)
      common=dict(version='DL-r5-audit-launcher-r1',run_directory=str(Path(args.run).resolve()),
          run_git=s.RUN_GIT,audit_git=args.prior_physics_audit_git,seconds=1800,parent_identity=identity,
          producer_launch=pins['launcher_launch'],producer_terminal=pins['launcher_terminal'],launcher=s.raw_pin(script))
      report=dict(version='DL-r5-independent-audit-r1',run_directory=common['run_directory'],
          run_git=s.RUN_GIT,audit_git=args.prior_physics_audit_git,seconds=1800,elapsed_seconds=2.,
          evidence_authenticated=True,all_physics_and_rewards_byte_exact=True,complete_units=512,complete_families=32,
          statistics={'generated_ignored_scientific_sign':'not inspected'})
      Path(args.audit_report).write_text(json.dumps(report));pins['audit_report']=s.raw_pin(args.audit_report)
      terminal=dict(common,returncode=0,external_failure=None,parent_identity_gone=True,
          owned_groups_cleaned=True,elapsed_seconds=3.,audit_output=pins['audit_report'],log=s.raw_pin(log))
      code=root/'code';launch=dict(common,cwd=str(code),environment=dict(CUDA_VISIBLE_DEVICES='',
          JAX_PLATFORMS='cpu',MUJOCO_GL='disable',MUJOCO_EGL_DEVICE_ID=None,PYOPENGL_PLATFORM=None))
      launch['command']=['/data/Policy_Discrepancy/envs/dreamer/bin/python',str(code/'analysis/dl_live_audit.py'),
          '--run',common['run_directory'],'--expected-run-git',s.RUN_GIT,'--expected-audit-git',args.prior_physics_audit_git,
          '--output',pins['audit_report']['path'],'--launcher-terminal',pins['launcher_terminal']['path'],
          '--launcher-sha',pins['launcher_terminal']['sha256'],'--launcher-launch',pins['launcher_launch']['path'],
          '--launcher-launch-sha',pins['launcher_launch']['sha256']]
      a=SimpleNamespace(original_inactive=mock.Mock(),original_group_inactive=mock.Mock())
      def write():
        Path(args.audit_terminal).write_text(json.dumps(terminal));Path(args.audit_launch).write_text(json.dumps(launch))
      with mock.patch.object(s,'AUDIT_LAUNCHER_SHA',s.digest(script.read_bytes())):
        write();self.assertEqual(s._physics_authority(args,a,pins),report)
        for case in ('exit','budget','identity','argv','producer_pin','partial','different_audit_git'):
          badt,badl,badr=copy.deepcopy(terminal),copy.deepcopy(launch),copy.deepcopy(report)
          if case == 'exit':badt['returncode']=1
          elif case == 'budget':badt['elapsed_seconds']=1800.1
          elif case == 'identity':badt['parent_identity_gone']=False
          elif case == 'argv':badl['command'][-1]='0'*64
          elif case == 'producer_pin':badt['producer_launch']['sha256']='0'*64
          elif case == 'partial':badr['complete_units']=511
          else:badr['audit_git']='8'*40
          Path(args.audit_terminal).write_text(json.dumps(badt));Path(args.audit_launch).write_text(json.dumps(badl))
          Path(args.audit_report).write_text(json.dumps(badr))
          with self.subTest(case=case),self.assertRaises(ValueError):s._physics_authority(args,a,pins)


if __name__ == '__main__':unittest.main()
