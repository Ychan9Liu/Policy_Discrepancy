"""Generated CPU interfaces only; no real R5/N1M, label or GPU acceptance."""
import copy
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import dl_normal_feature_run as f


def generated_labels(root):
  """512 compact fabricated rows; authority and Git commands explicitly mocked."""
  from scripts import dl_normal_source_index as s
  rows=[];labels=[]
  for u in range(512):
    plan=s.plan(u);t=plan['t']
    row=dict(source_id=f'{u:064x}',global_unit=u,family=u//16,replicate=u%16,split='fit' if u<128 else 'held',
        plan=plan,trajectory=dict(path=f'unit_{u}/trajectory.pkl',bytes=2,sha256='1'*64),prefix_typed_sha256='2'*64,
        current=dict(observation_sha256='3'*64,snapshot_sha256='4'*64,snapshot_hash_format=s.SNAPSHOT_HASH_FORMAT))
    label=copy.deepcopy(row);label.pop('current')
    values=dict(velocity_x=.125,upright=.75,upright_reward=.875,move_reward=.625,reward_component_product=.546875,current_reward_f64=.546875)
    label.update(source_reward_index=t-1,current_observation_index=t,current_observation_sha256='3'*64,
        current_source_snapshot_sha256='4'*64,source_snapshot_hash_format=s.SNAPSHOT_HASH_FORMAT,
        query_snapshot_hash_format='DL-normal-task-query-snapshot-r1',query_before_snapshot_sha256='5'*64,
        query_after_snapshot_sha256='5'*64,task_rng_before_sha256='6'*64,task_rng_after_sha256='6'*64,
        current_values=values,current_f64_hex={n:np.float64(v).tobytes().hex() for n,v in values.items()})
    rows.append(row);labels.append(label)
  index=dict(run_git=s.RUN_GIT,prior_physics_audit_git='7'*40,run_directory=str(root/'original'),rows=rows,
      external_pins={},run_inventory_sha256='8'*64)
  names=('scripts/dl_normal_labels.py','scripts/dl_normal_source_index.py','analysis/dl_normal_task_query.py')
  bodies={name:(ROOT/name).read_bytes() for name in names}
  report=dict(version='DL-normal-labels-r1',run_git=index['run_git'],prior_physics_audit_git='7'*40,
      run_directory=index['run_directory'],git_sha='9'*40,complete_units=512,complete_families=32,
      fit_families=list(range(8)),held_families=list(range(8,32)),source_complete_and_stable=True,
      current_labels_replayed=True,seconds=900,elapsed_seconds=1.,external_pins={},run_inventory_sha256='8'*64,
      rows=labels,current_source_pins={n:dict(bytes=len(b),sha256=f.sha_bytes(b)) for n,b in bodies.items()})
  output=root/'labels.json';f.write_json(output,report)
  authority=dict(version='DL-normal-labels-launcher-r1',git_sha='9'*40,run_directory=index['run_directory'],
      returncode=0,seconds=900,elapsed_seconds=1.25,success=True,labels_output=f.pin(output),
      parent_identity=dict(pid=101,pgid=101,sid=101,start_ticks=42),parent_identity_gone=True,owned_groups_cleaned=True,external_failure=None)
  terminal=root/'labels-terminal.json';f.write_json(terminal,authority)
  execution=dict(git_sha='9'*40,command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','sv3',
      '/data/Policy_Discrepancy/envs/dreamer/bin/python',
      '/data/Policy_Discrepancy/runs/dl-normal-labels-inputs-20261015-9999999/launch.py','--expected-git','9'*40],
      returncode=0,elapsed_seconds=1.5,GPU_used=False,log_sha256='a'*64,actual_outer_complete_within900=True)
  execution_path=root/'outer-execution.json';f.write_json(execution_path,execution)
  args=SimpleNamespace(labels=str(output),labels_sha=f.pin(output)['sha256'],labels_terminal=str(terminal),labels_terminal_sha=f.pin(terminal)['sha256'])
  args.fixture_execution_plan=dict(labels_outer_execution=f.pin(execution_path))
  def git(command,payload):
    assert command == ['git','cat-file','--batch']
    result=b''
    for line in payload.decode().splitlines():
      _,name=line.split(':',1);body=bodies[name]
      result+=b'0'*40+b' blob '+str(len(body)).encode()+b'\n'+body+b'\n'
    return result
  auditor=SimpleNamespace(original_inactive=mock.Mock(),original_group_inactive=mock.Mock(),command_output=git)
  return args,index,report,authority,auditor


def generated_worker_graph(output):
  """Fabricated512 publication graph, NOT source authentication or NN execution.

  Clock0/1 are intentionally outside production source.plan; only the post-child
  artifact verifier is exercised. Production source authentication is not patched
  or called, and this helper is never exposed by the CLI.
  """
  import ml_dtypes
  from scripts.dl_reset_learning import tree_digest
  arrays=dict(h=np.zeros(4096,ml_dtypes.bfloat16),q_prob=np.full((32,32),1/32,np.float32),
      p_prob=np.full((32,32),1/32,np.float32),qlogits=np.zeros((32,32),ml_dtypes.bfloat16),
      plogits=np.zeros((32,32),ml_dtypes.bfloat16),root_key=np.array([42,43],np.uint32),
      latent_key=np.array([44,45],np.uint32),actor_key=np.array([46,47],np.uint32),native_raw=np.zeros(12,np.float32))
  for name in ('source_index.json','budget.json','resources.jsonl','worker.log','process.json','worker_resources.jsonl'):
    (output/name).write_bytes(b'{}\n')
  f.write_json(output/'initial.json',dict(full_state_sha256=f.FULL_STATE,policy_copy_sha256='a'*64,counters=f.COUNTERS,
      model=dict(params_sha256=f.PARAMS,role='reference0')))
  f.write_json(output/'final.json',dict(full_state_sha256=f.FULL_STATE,policy_copy_sha256='a'*64,counters=f.COUNTERS,
      resets=512,steps=513,training_updates=0,environment_constructions=0,render_calls=0))
  for name in ('runtime_before.json','runtime_after.json'):f.write_json(output/name,{})
  rows=[];labels=[];units=[]
  for u in range(512):
    directory=output/f'unit_{u:03d}';directory.mkdir()
    t=1 if u==0 else 0
    row=dict(global_unit=u,source_id=f'{u:064x}',plan=dict(t=t),split='fit' if u<128 else 'held',prefix_typed_sha256='b'*64,
        frames=[dict(observation_sha256='c'*64,root_key=[42,43],previous_raw_sha256='d'*64) for _ in range(t+1)])
    label=dict(source_id=row['source_id'],generated_metadata_only=True)
    frame=dict(source_id=row['source_id'],counter=t,observation_sha256='c'*64,root_key=[42,43],latent_key=[44,45],actor_key=[46,47],
        source_prev_raw_sha256='d'*64,before_carry_sha256='f'*64 if t else '9'*64,source_raw_sha256=None,
        history_after_sha256=None,source_history_sha256='8'*64,
        native_after_sha256='e'*64,native_raw_sha256=tree_digest(arrays['native_raw']),source_raw_available=False,feature_only_final=True,
        original_nj_seed_calls=2,probability_nj_seed_calls=0,public_oracle=dict(public_policy_calls=1,public_key_byte_exact=True,
          native_raw_byte_exact=True,native_full_after_carry_byte_exact=True,counters_restored=True,before_host_override=True))
    specs=f.store_current(directory/'current.npz',arrays)
    frames=([dict(frame,counter=0,before_carry_sha256='9'*64,source_raw_sha256='d'*64,
        history_after_sha256='f'*64,source_raw_available=True,feature_only_final=False)] if t else [])+[frame]
    f.write_json(directory/'history.json',dict(source_id=row['source_id'],frames=frames))
    receipt=dict(version=f.VERSION,role='reference0',source_id=row['source_id'],global_unit=u,plan=row['plan'],split=row['split'],
        prefix_typed_sha256='b'*64,current_label_sha256=f.sha_bytes(f.canonical(label)),history_frames=t+1,public_policy_calls=t+1,
        reset=dict(root_key=[42,43],initial_carry_sha256='9'*64),
        current_source_raw_available=False,feature_schema=specs,artifacts={n:f.pin(directory/n) for n in ('current.npz','history.json')},
        seal=dict(full_state_policy_copy_counters_unchanged=True,full_state_sha256=f.FULL_STATE,counters=f.COUNTERS,
          fixture_geometry=False,feature_only_final=True,final_carry_sha256=None,final_native_after_sha256='e'*64,
          episode=dict(source_id=row['source_id'],next_counter=t+1,source_command_frames=t,history_sha256='8'*64)))
    f.write_json(directory/'complete.json',receipt);units.append(f.pin(directory/'complete.json'));rows.append(row);labels.append(label)
  complete=dict(version=f.VERSION,role='reference0',complete_units=512,real_N1M_inputs_authenticated=True,
      source_index_sha256=f.pin(output/'source_index.json')['sha256'],full_state_policy_copy_counters_unchanged=True,
      units=units,public_policy_calls=513,native_policy_calls=513,artifacts={n:f.pin(output/n) for n in
      ('initial.json','runtime_before.json','runtime_after.json','final.json','worker_resources.jsonl')})
  f.write_json(output/'worker_complete.json',complete)
  return dict(rows=rows),dict(rows=labels)


class NormalFeatureRunTests(unittest.TestCase):
  def test_initial_authentication_timeout_uses_loaded_registry_and_preserves_fences(self):
    from scripts import dl_normal_source_index as source
    from analysis import dl_live_audit as a
    owned=dict(pid=321,pgid=321,start_ticks=100)
    def stat(sid,start):return '321 (fixture) '+' '.join(['S','1','321',str(sid)]+['0']*15+[str(start)])
    real_path=Path
    with tempfile.TemporaryDirectory() as temp:
      proc=real_path(temp);(proc/'321').mkdir()
      def path(value):return proc if str(value)=='/proc' else proc/str(value).removeprefix('/proc/')
      for mode in ('reused_pid','foreign_sid'):
        (proc/'321/stat').write_text(stat(999 if mode=='foreign_sid' else 321,100))
        actual=dict(owned,start_ticks=101) if mode=='reused_pid' else owned
        with self.subTest(mode=mode),mock.patch.object(f,'AUDITOR',None),mock.patch.object(source,'AUDITOR',a), \
            mock.patch.dict(a.OWNED,{321:owned},clear=True),mock.patch.object(a,'Path',side_effect=path), \
            mock.patch.object(a,'process_identity',return_value=actual), \
            mock.patch.object(signal,'SIGKILL',9,create=True), \
            mock.patch.object(a.os,'killpg',create=True) as kill,mock.patch.object(a.os,'_exit',side_effect=SystemExit(124)), \
            mock.patch('builtins.__import__',side_effect=AssertionError('Signal handler must not import')):
          with self.assertRaises(SystemExit):f.expire(getattr(signal,'SIGALRM',14),None)
          self.assertNotIn(mock.call(321,signal.SIGKILL),kill.call_args_list)
          self.assertIn(mock.call(321,0),kill.call_args_list)

  @unittest.skipUnless(sys.platform.startswith('linux'),'Requires real Linux /proc, sessions and SIGALRM')
  def test_alarm_during_unreturned_authentication_cleans_real_owned_metadata_session(self):
    with tempfile.TemporaryDirectory() as temp:
      evidence=Path(temp)/'alarm.json'
      result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--alarm-fixture',str(evidence)],
          cwd=ROOT,capture_output=True,text=True,timeout=20,
          env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',MUJOCO_GL='disable'))
      self.assertEqual(result.returncode,124,result.stdout+result.stderr)
      value=f.decode(evidence.read_bytes())
      self.assertFalse(value['authentication_returned']);self.assertTrue(value['feature_auditor_unassigned'])
      self.assertEqual(len(value['owned']),1)
      from analysis import dl_live_audit as a
      for owned in value['owned']:
        until=time.monotonic()+3
        while a.group_alive(owned) and time.monotonic()<until:time.sleep(.01)
        self.assertFalse(a.group_alive(owned),'Original metadata session survived initial-auth alarm')

  def test_import_is_stdlib_only_and_cli_requires_explicit_plan_and_successful_labels(self):
    code="import os,sys;before=dict(os.environ);import scripts.dl_normal_feature_run;assert before==dict(os.environ);assert not any(n.split('.')[0] in ('numpy','jax','dm_control','mujoco','OpenGL') for n in sys.modules)"
    result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=30)
    self.assertEqual(result.returncode,0,result.stderr)
    result=subprocess.run([sys.executable,str(Path(f.__file__))],cwd=ROOT,capture_output=True,text=True,timeout=30)
    self.assertEqual(result.returncode,2)
    for name in ('execution-plan','execution-plan-sha','labels','labels-sha','labels-terminal','labels-terminal-sha',
        'prior-physics-audit-git','launcher-terminal','launcher-sha','launcher-launch','launcher-launch-sha'):
      self.assertIn('--'+name,result.stderr)

  def test_compute_environment_and_persist_before_unknown_work_failure(self):
    plan=dict(physical_gpu=5,gpu_uuid='GPU-1234')
    env=f.child_environment(plan)
    self.assertNotIn('MUJOCO_EGL_DEVICE_ID',env);self.assertNotIn('PYOPENGL_PLATFORM',env)
    with mock.patch.dict(os.environ,env,clear=True):f.compute_environment(plan)
    with mock.patch.dict(os.environ,dict(env,PYOPENGL_PLATFORM='egl'),clear=True),self.assertRaisesRegex(ValueError,'compute-only'):
      f.compute_environment(plan)
    from scripts import dl_resources as resources
    state=dict(gpu_uuid='GPU-1234',processes=[dict(pid=999,type='C')],pmon_jobs=[],raw_pmon='5 999 C ...')
    with tempfile.TemporaryDirectory() as temp,mock.patch.object(resources,'live_state',return_value=state):
      path=Path(temp)/'resources.jsonl'
      with self.assertRaisesRegex(ValueError,'Unknown'):f.sample_resource(plan,path,{123})
      self.assertEqual(f.decode(path.read_bytes())['live_state'],state)

  def test_labels_require_complete_512_actual_success_and_same_source(self):
    with tempfile.TemporaryDirectory() as temp:
      args,index,report,authority,a=generated_labels(Path(temp))
      pins,value=f.labels_authority(args,index,a,args.fixture_execution_plan)
      self.assertEqual(value,report);self.assertEqual(pins['output']['sha256'],args.labels_sha)
      a.original_inactive.assert_called_once_with(101,42);a.original_group_inactive.assert_called_once()
      cases=[]
      bad=copy.deepcopy(report);bad['rows'][13]['source_id']='0'*64;cases.append(('report',bad,'source ID'))
      bad=copy.deepcopy(report);bad['rows'][13]['current_f64_hex']['velocity_x']=np.nextafter(np.float64(.125),np.inf).tobytes().hex();cases.append(('report',bad,'F64'))
      bad=copy.deepcopy(report);bad['rows']=bad['rows'][:-1];cases.append(('report',bad,'Exactly512'))
      bad=copy.deepcopy(report);bad['current_source_pins']={};cases.append(('report',bad,'implementation'))
      bad=copy.deepcopy(authority);bad['success']=False;cases.append(('terminal',bad,'exit authority'))
      bad=copy.deepcopy(authority);bad['owned_groups_cleaned']=False;cases.append(('terminal',bad,'exit authority'))
      for which,bad,text in cases:
        with self.subTest(case=text):
          Path(args.labels).write_bytes(f.canonical(bad if which=='report' else report))
          current=f.pin(args.labels);term=copy.deepcopy(bad if which=='terminal' else authority);term['labels_output']=current
          Path(args.labels_terminal).write_bytes(f.canonical(term))
          args.labels_sha=current['sha256'];args.labels_terminal_sha=f.pin(args.labels_terminal)['sha256']
          with self.assertRaisesRegex(ValueError,text):f.labels_authority(args,index,a,args.fixture_execution_plan)
      Path(args.labels).write_bytes(f.canonical(report));args.labels_sha=f.pin(args.labels)['sha256']
      authority['labels_output']=f.pin(args.labels);Path(args.labels_terminal).write_bytes(f.canonical(authority))
      args.labels_terminal_sha=f.pin(args.labels_terminal)['sha256']
      execution_path=Path(args.fixture_execution_plan['labels_outer_execution']['path']);execution=f.decode(execution_path.read_bytes())
      for key,bad_value in (('returncode',1),('actual_outer_complete_within900',False),('elapsed_seconds',900.01),
          ('git_sha','a'*40),('command',['ssh','sv2','wrong'])):
        bad=dict(execution,**{key:bad_value});execution_path.write_bytes(f.canonical(bad))
        args.fixture_execution_plan['labels_outer_execution']=f.pin(execution_path)
        with self.subTest(outer=key),self.assertRaisesRegex(ValueError,'observed labels-controller'):
          f.labels_authority(args,index,a,args.fixture_execution_plan)

  def test_fresh_output_does_not_create_paths_in_immutable_input(self):
    with tempfile.TemporaryDirectory() as temp:
      root=Path(temp);source=root/'immutable';source.mkdir()
      with self.assertRaisesRegex(ValueError,'outside'):f.fresh_directory(root/'outside'/'..'/'immutable'/'new'/'result',[source])
      self.assertFalse((source/'new').exists())
      path=f.fresh_directory(root/'external'/'result',[source]);self.assertTrue(path.is_dir())
      with self.assertRaisesRegex(ValueError,'Fresh'):f.fresh_directory(path,[source])

  def test_partial_worker_is_rejected_before_tensor_decode(self):
    with tempfile.TemporaryDirectory() as temp:
      output=Path(temp);f.write_json(output/'source_index.json',{})
      f.write_json(output/'worker_complete.json',dict(version=f.VERSION,role='reference0',complete_units=511))
      with self.assertRaisesRegex(ValueError,'Complete actual reference0'):
        f.verify_worker(output,{},dict(rows=[]))

  def test_generated_512_artifact_closure_and_late_tamper_rejection(self):
    with tempfile.TemporaryDirectory() as temp:
      output=Path(temp).resolve();index,labels=generated_worker_graph(output)
      _,complete=f.verify_worker(output,index,labels)
      self.assertEqual(complete['complete_units'],512)  # Metadata graph, not real512 source or NN acceptance.
      paths=[output/'unit_000/current.npz',output/'unit_000/history.json',output/'unit_000/complete.json',output/'worker_complete.json']
      originals={p:p.read_bytes() for p in paths}
      def restore():
        for path,body in originals.items():path.write_bytes(body)
      def reseal():
        path=output/'unit_000/complete.json';receipt=f.decode(path.read_bytes())
        receipt['artifacts']={name:f.pin(output/'unit_000'/name) for name in ('current.npz','history.json')}
        path.write_bytes(f.canonical(receipt))
        top=f.decode((output/'worker_complete.json').read_bytes());top['units'][0]=f.pin(path)
        (output/'worker_complete.json').write_bytes(f.canonical(top))
      with np.load(paths[0],allow_pickle=False) as archive:arrays={n:archive[n] for n in archive.files}
      arrays['latent_key']=np.array([999,45],np.uint32)
      with paths[0].open('wb') as stream:np.savez(stream,**arrays)
      receipt=f.decode(paths[2].read_bytes());receipt['feature_schema']['latent_key']=f.typed_spec(arrays['latent_key'])
      paths[2].write_bytes(f.canonical(receipt));reseal()
      with self.assertRaisesRegex(ValueError,'feature/oracle key'):f.verify_worker(output,index,labels)
      restore()
      history=f.decode(paths[1].read_bytes());history['frames'][0]['history_after_sha256']='0'*64
      paths[1].write_bytes(f.canonical(history));reseal()
      with self.assertRaisesRegex(ValueError,'conditional carry chain'):f.verify_worker(output,index,labels)
      restore()
      (output/'unit_511'/'history.json').write_bytes(b'{}')
      with self.assertRaisesRegex(ValueError,'bytes changed'):f.verify_worker(output,index,labels)

  def test_real_tiny_64_frame_prefix_and_byte_storage(self):
    result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--fixture'],cwd=ROOT,capture_output=True,text=True,
        timeout=240,env=dict(os.environ,PYTHONPATH=str(ROOT),JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',MUJOCO_GL='disable'))
    self.assertEqual(result.returncode,0,result.stdout[-4000:]+result.stderr[-9000:])
    report=json.loads(result.stdout.strip().splitlines()[-1])
    self.assertEqual(report['source_command_frames'],64);self.assertEqual(report['actual_public_oracles'],65)
    for name in ('source_raw_t_not_feature_input','BF16_bits_preserved','full_state_counters_readonly','wrong_source_rejected','oracle_tamper_rejected'):
      self.assertTrue(report[name])
    self.assertFalse(report['real_N1M_authenticated']);self.assertFalse(report['source_authenticated']);self.assertFalse(report['GPU_used'])
    print(json.dumps(report,sort_keys=True))


def fixture():
  dll=(os.add_dll_directory(os.environ['DL_DLL_DIRECTORY']) if os.name=='nt' and os.environ.get('DL_DLL_DIRECTORY') else None)
  import elements
  import ml_dtypes
  from ruamel.yaml import YAML
  from scripts.dl_predictive_mc import agent_from_spaces
  from scripts import dl_normal_features as features
  from scripts import dl_normal_source_index as source
  from scripts.dl_reset_learning import tree_digest
  from tests.test_dl_normal_source_index import synthetic_record,trajectory_pin
  presets=YAML(typ='safe').load((ROOT/'dreamerv3/configs.yaml').read_text())
  config=elements.Config(presets['defaults']).update(presets['debug'])
  config=elements.Config({**config.flat,**{'jax.platform':'cpu','jax.compute_dtype':'bfloat16',
      'jax.precompile':False,'jax.profiler':False,'jax.enable_policy':True,'jax.transfer_guard':True,
      'batch_size':2,'batch_length':3,'replay_context':1,'agent.imag_length':2,'agent.imag_last':2,
      'agent.rep_probe.mode':'off','agent.dt_latch.mode':'off'}}).update(logdir='/tmp/dl-normal-feature-run-fixture',seed=0)
  agent=agent_from_spaces(config);agent.load(agent.save())
  initial=tree_digest(agent.save());before=features.counters(agent)
  unit=next(u for u in range(512) if source.plan(u)['t']==64)
  record=synthetic_record(unit);t=record['plan']['t'];row=source.project_prefix(record,unit,trajectory_pin(unit),'b'*64)
  wrapper=features.FrozenNormalFeatures(agent,fixture=True)
  wrong=copy.deepcopy(row);wrong['source_id']='c'*64
  try:f.extract_prefix(wrapper,record,wrong,'b'*64,fixture=True)
  except ValueError as error:assert 'source ID' in str(error)
  else:raise AssertionError('Wrong source accepted')
  # Unavailable raw[t] may not enter the prefix core. These generated fields
  # are outside the projector; no real source validation is claimed here.
  record['raw'][t]=np.float32(.123)
  assert source.project_prefix(record,unit,trajectory_pin(unit),'b'*64)==row
  result=f.extract_prefix(wrapper,record,row,'b'*64,fixture=True)
  f.canonical(dict(reset=result['reset'],frames=result['frames'],seal=result['seal']))
  assert len(result['frames'])==65 and result['seal']['episode']['source_command_frames']==64
  assert result['frames'][-1]['source_raw_sha256'] is None and result['frames'][-1]['history_after_sha256'] is None
  assert result['current']['source_raw'] is None
  with tempfile.TemporaryDirectory() as temp:
    path=Path(temp)/'current.npz';specs=f.store_current(path,result['current'])
    with np.load(path,allow_pickle=False) as stored:
      assert stored['h'].dtype==np.uint16
      assert stored['h'].tobytes()==result['current']['h'].tobytes()
      for name,spec in specs.items():
        value=stored[name].view(ml_dtypes.bfloat16) if spec['logical_dtype']=='bfloat16' else stored[name]
        assert f.typed_spec(value)==spec
  # Actual oracle executes, then a deliberately changed receipt is rejected
  # by the runner before publication. Underlying model stays readonly.
  other=features.FrozenNormalFeatures(agent,fixture=True);original=other.step
  def broken(*args,**kwargs):
    frame=original(*args,**kwargs);frame['public_oracle']['native_raw_byte_exact']=False;return frame
  other.step=broken
  try:f.extract_prefix(other,record,row,'b'*64,fixture=True)
  except ValueError as error:assert 'actual exact public/native' in str(error)
  else:raise AssertionError('Tampered oracle accepted')
  other.finish()
  assert tree_digest(agent.save())==initial and features.counters(agent)==before
  print(json.dumps(dict(source_command_frames=64,actual_public_oracles=65,source_raw_t_not_feature_input=True,
      BF16_bits_preserved=True,full_state_counters_readonly=True,wrong_source_rejected=True,oracle_tamper_rejected=True,
      real_N1M_authenticated=False,source_authenticated=False,GPU_used=False)),flush=True)
  if dll is not None:dll.close()


def alarm_fixture(evidence):
  """Actual Linux helper/session; source authentication deliberately has not returned."""
  from scripts import dl_normal_source_index as source
  from analysis import dl_live_audit as a
  source.AUDITOR=a;assert f.AUDITOR is None and not a.OWNED
  returned=False
  def alarm(signum,frame):
    f.write_json(evidence,dict(authentication_returned=returned,feature_auditor_unassigned=f.AUDITOR is None,
        owned=list(a.OWNED.values())))
    f.expire(signum,frame)
  def pending_authentication(args):
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,.25)
    # The real audited command_output creates/registers a separate owned session
    # before restoring the alarm mask. No real run/model/source is read here.
    a.command_output([sys.executable,'-c','import time;time.sleep(60)'])
  source.authenticate_and_index=pending_authentication
  source.authenticate_and_index(None);returned=True
  raise AssertionError('Pending authentication unexpectedly returned before alarm')


if __name__=='__main__':
  if '--alarm-fixture' in sys.argv:alarm_fixture(sys.argv[-1])
  elif '--fixture' in sys.argv:fixture()
  else:unittest.main()
