"""Finite producer engineering; no real N1M inputs, server, EGL or GPU.

One integration case uses real tiny F32 NN scratch cores on generated B2/T4
data. The full128/16x65 production entry is never certified by this fixture.
"""
import json
import datetime as dt
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
from scripts import dl_nm1_cal as p

ROOT=Path(__file__).resolve().parents[1]


class ProducerTests(unittest.TestCase):
  def test_import_is_stdlib_and_deadline_prevents_creation(self):
    result=subprocess.run([sys.executable,'-c',
        "import sys;import scripts.dl_nm1_cal;assert 'numpy' not in sys.modules;assert 'jax' not in sys.modules"],
        cwd=ROOT,capture_output=True,text=True,timeout=30)
    self.assertEqual(result.returncode,0,result.stderr)
    with tempfile.TemporaryDirectory() as directory,mock.patch.object(p,'DEADLINE',time.monotonic()-1), \
        mock.patch.object(p.subprocess,'Popen') as create:
      with self.assertRaisesRegex(ValueError,'deadline'):
        p.write_json(Path(directory)/'late.json',{})
      with self.assertRaisesRegex(ValueError,'deadline'):p.spawn(['unused'])
      create.assert_not_called()

  def test_exact_tensor_schema_preserves_bf16_bits_and_rejects_void_nan(self):
    import ml_dtypes
    from scripts.dl_reset_learning import tree_digest
    with tempfile.TemporaryDirectory() as directory:
      path=Path(directory)/'tensors.npz'
      data={'raw/f32':np.array([-0.,1.25],np.float32),
          'raw/bf16':np.array([-0.,1.25],ml_dtypes.bfloat16),'raw/id':np.arange(4,dtype=np.uint8)}
      specs=p.write_arrays(path,data)
      with np.load(path,allow_pickle=False) as stored:
        for key,value in data.items():
          spec=specs[key];actual=stored[spec['storage_key']]
          self.assertEqual(spec['typed_sha256'],tree_digest(value))
          self.assertEqual(actual.tobytes(),value.tobytes())
          if key.endswith('bf16'):
            self.assertEqual(actual.dtype,np.uint16)
            self.assertEqual(spec['logical_dtype'],'bfloat16')
      with self.assertRaisesRegex(ValueError,'dtype'):p.array_spec(np.zeros(1,'V2'))
      with self.assertRaisesRegex(ValueError,'Nonfinite'):p.array_spec(np.array([np.nan],np.float32))

  def test_input_pins_fail_before_pickle_and_stage_pair_is_explicit(self):
    with tempfile.TemporaryDirectory() as directory:
      root=Path(directory)
      for name in ('checkpoint','config','inventory','resource_record','stage_record'):(root/name).write_bytes(name.encode())
      args=mock.Mock(**{name:str(root/name) for name in ('checkpoint','config','inventory','resource_record')},
          stage_record=None,expected_stage_sha256=None)
      with self.assertRaisesRegex(ValueError,'before decoding'):p.input_pins(args)
      with mock.patch.object(p,'CHECKPOINT',p.sha(root/'checkpoint')), \
          mock.patch.object(p,'CONFIG',p.sha(root/'config')),mock.patch.object(p,'INVENTORY',p.sha(root/'inventory')):
        self.assertIsNone(p.input_pins(args)['stage_record'])
        args.stage_record=str(root/'stage_record')
        with self.assertRaisesRegex(ValueError,'together'):p.input_pins(args)
        args.expected_stage_sha256='0'*64
        with self.assertRaisesRegex(ValueError,'Stage raw'):p.input_pins(args)
        args.expected_stage_sha256=p.sha(root/'stage_record')
        self.assertEqual(p.input_pins(args)['stage_record']['sha256'],args.expected_stage_sha256)

  def test_fixed_budget_rejects_legacy_resource_and_constants(self):
    budget=dict(version=p.VERSION,schema=1,seconds=3600,git_sha='a'*40,logical_server='sv3',
        physical_gpu=5,gpu_uuid=p.GPU_UUID,constants=dict(p.CONSTANTS),monotonic_deadline=float(time.monotonic()+10))
    p.validate_budget(budget,'a'*40)
    for key,value in (('seconds',7200),('physical_gpu',4),('gpu_uuid','other'),('constants',{})):
      with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Fixed C'):
        p.validate_budget(dict(budget,**{key:value}),'a'*40)

  def test_compute_only_resources_need_no_probe_or_egl_and_keep_runtime_fences(self):
    from scripts import dl_resources as resources
    pid=os.getpid();instant=resources.now().isoformat()
    idle=dict(checked_at_utc=instant,physical_gpu=5,gpu_uuid=p.GPU_UUID,memory_used_mib=1,
        processes=[],pmon_jobs=[],raw_pmon='5 - - idle')
    environ={key:value for key,value in p.COMPUTE_ENV.items() if value is not None}
    with tempfile.TemporaryDirectory() as directory,mock.patch.object(p,'require_host'), \
        mock.patch.dict(os.environ,environ,clear=True),mock.patch.object(resources,'live_state',return_value=idle) as live, \
        mock.patch.object(resources,'mapping_evidence',side_effect=AssertionError('EGL probe forbidden')):
      record=p.compute_resource_record();path=Path(directory)/'resource.json';p.write_json(path,record)
      fresh=p.require_compute_resource(path);refreshed=p.refresh_compute_resource(fresh)
      self.assertIsNone(record['egl_uuid']);self.assertFalse(record['graphics_verified'])
      self.assertNotIn('device_probe_path',record)
      self.assertEqual(refreshed['resource_source_sha256'],p.sha(path))
      self.assertEqual(refreshed['resource_source_path'],str(path.resolve()))
      for key,value in (('physical_gpu',4),('gpu_uuid','GPU-other'),('egl_uuid',p.GPU_UUID),('version','DL-code-r1')):
        with self.subTest(key=key),self.assertRaises(ValueError):p.validate_compute_binding(dict(record,**{key:value}))
      for key,value in (('CUDA_VISIBLE_DEVICES','GPU-other'),('MUJOCO_GL','egl'),('PYOPENGL_PLATFORM','egl'),
          ('MUJOCO_EGL_DEVICE_ID','5')):
        with self.subTest(env=key),mock.patch.dict(os.environ,{key:value}),self.assertRaisesRegex(ValueError,'environment'):
          p.validate_compute_binding(record)
      stale=dict(record,checked_at_utc=(resources.now()-dt.timedelta(seconds=121)).isoformat())
      p.write_json(Path(directory)/'stale.json',stale)
      with self.assertRaisesRegex(ValueError,'120 seconds'):p.require_compute_resource(Path(directory)/'stale.json')
      own=dict(idle,processes=[dict(pid=pid,type='C')],pmon_jobs=[dict(pid=pid,type='C')],raw_pmon=f'5 {pid} C owned')
      live.return_value=own
      with mock.patch.object(p,'command_output',return_value=f'{pid}, {p.GPU_UUID}\n'.encode()):
        runtime=p.verify_compute_runtime(refreshed)
        self.assertEqual(runtime['compute_uuid'],[p.GPU_UUID]);self.assertFalse(runtime['graphics_verified'])
        for bad in (dict(own,processes=[dict(pid=pid,type='C+G')]),dict(own,raw_pmon=f'4 {pid} C wrong'),
            dict(own,pmon_jobs=[dict(pid=pid+1,type='C')])):
          live.return_value=bad
          with self.assertRaises(ValueError):p.verify_compute_runtime(refreshed)
      live.return_value=own
      with mock.patch.object(p,'command_output',return_value=f'{pid}, GPU-other\n'.encode()), \
          self.assertRaisesRegex(ValueError,'assigned UUID'):p.verify_compute_runtime(refreshed)
    code="""import os,sys
from unittest import mock
from scripts import dl_nm1_cal as p
from scripts import dl_resources as r
state=dict(checked_at_utc=r.now().isoformat(),physical_gpu=5,gpu_uuid=p.GPU_UUID,memory_used_mib=1,processes=[],pmon_jobs=[],raw_pmon='5 - - idle')
with mock.patch.object(p,'require_host'),mock.patch.dict(os.environ,{k:v for k,v in p.COMPUTE_ENV.items() if v is not None},clear=True),mock.patch.object(r,'live_state',return_value=state):
 p.compute_resource_record()
assert not any(n.split('.')[0] in ('jax','numpy','mujoco','dm_control','OpenGL') for n in sys.modules)
"""
    result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=30)
    self.assertEqual(result.returncode,0,result.stderr)

  def test_parent_saves_conflict_before_cleanup_and_actual_returncode(self):
    with tempfile.TemporaryDirectory() as directory:
      output=Path(directory)/'run'
      args=mock.Mock(output=str(output),expected_git='a'*40,checkpoint='unused',config='unused',
          inventory='unused',replay_directory='unused',resource_record='unused',
          stage_record=None,expected_stage_sha256=None)
      real_spawn=p.spawn
      def sleeper(command,**kwargs):
        return real_spawn([sys.executable,'-c','import time;time.sleep(60)'],
            stdout=kwargs['stdout'],stderr=kwargs['stderr'])
      state=dict(physical_gpu=5,gpu_uuid=p.GPU_UUID,processes=[dict(pid=999999,type='C')],
          pmon_jobs=[],raw_pmon='5 999999 C conflict',memory_used_mib=100)
      real_identity=p.identity
      with mock.patch.object(p,'require_host'),mock.patch.object(p,'source_binding',return_value={}), \
          mock.patch.object(p,'input_pins',return_value={}),mock.patch.object(p,'identity',
              side_effect=real_identity if os.name == 'posix' else
                  lambda pid:dict(pid=pid,parent_pid=1,pgid=pid,start_ticks=3)), \
          mock.patch.object(p,'require_compute_resource',return_value=dict(physical_gpu=5,gpu_uuid=p.GPU_UUID)), \
          mock.patch('scripts.dl_resources.live_state',return_value=state), \
          mock.patch.object(p,'spawn',side_effect=sleeper),mock.patch('builtins.print'):
        with self.assertRaisesRegex(ValueError,'C incomplete'):p.parent(args)
      terminal=p.read_json(output/'supervisor_terminal.json')
      self.assertIsInstance(terminal['worker_returncode'],int)
      self.assertNotEqual(terminal['worker_returncode'],0)
      self.assertTrue(terminal['worker_terminal'] and terminal['owned_group_cleaned'])
      self.assertIn('Unknown work appeared',terminal['failure'])
      sample=json.loads((output/'resources.jsonl').read_text())
      self.assertEqual(sample['raw_pmon'],state['raw_pmon'])
      self.assertEqual(sample['xml_process_summary'],state['processes'])
      self.assertIsNone(terminal['artifacts']['generation_complete.json'])
      self.assertEqual(terminal['artifacts']['resources.jsonl']['sha256'],p.sha(output/'resources.jsonl'))

  def test_cleanup_refuses_reused_leader_or_foreign_session_before_signal(self):
    real_path=p.Path;owned=dict(pid=321,parent_pid=1,pgid=321,start_ticks=100)
    def stat(sid,start):
      fields=['S','1','321',str(sid)]+['0']*15+[str(start)]
      return '321 (fixture) '+' '.join(fields)
    with tempfile.TemporaryDirectory() as directory:
      proc=real_path(directory);(proc/'321').mkdir();(proc/'321/stat').write_text(stat(321,100))
      def path(value):
        value=str(value)
        return proc if value == '/proc' else proc/value.removeprefix('/proc/')
      for mode in ('reused','foreign_session','older_member'):
        (proc/'321/stat').write_text(stat(999 if mode == 'foreign_session' else 321,
            99 if mode == 'older_member' else 100))
        current=dict(owned,start_ticks=101) if mode == 'reused' else owned
        child=mock.Mock(pid=321)
        with self.subTest(mode=mode),mock.patch.object(p,'Path',side_effect=path), \
            mock.patch.object(p,'identity',return_value=current),mock.patch.object(p.os,'name','posix'), \
            mock.patch.object(p.os,'killpg',create=True) as kill,mock.patch.dict(p.OWNED,{321:owned},clear=True):
          with self.assertRaisesRegex(ValueError,'reused|ownership'):p.cleanup(child)
          self.assertTrue(all(call.args[1] == 0 for call in kill.call_args_list))
          child.wait.assert_not_called()
          with mock.patch.object(p.os,'_exit') as exit_now:p.expire(None,None)
          exit_now.assert_called_once_with(124)
          self.assertTrue(all(call.args[1] == 0 for call in kill.call_args_list))

  @unittest.skipUnless(sys.platform == 'linux','Actual Linux session ownership fixture')
  def test_cleanup_reaped_leader_removes_only_its_live_descendant(self):
    command=[sys.executable,'-c',
        'import subprocess,sys; p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]); print(p.pid,flush=True)']
    child=p.spawn(command,stdout=subprocess.PIPE,text=True)
    owned=dict(p.OWNED[child.pid])
    try:
      grandchild=int(child.stdout.readline());child.wait(timeout=10)
      self.assertIn(grandchild,p.group_alive(owned))
      p.cleanup(child)
      self.assertEqual(p.group_alive(owned),[])
      self.assertNotIn(child.pid,p.OWNED)
    finally:
      if child.pid in p.OWNED:p.cleanup(child)
      child.stdout.close()

  def test_controls_use_real_jax_and_zero_denominator_is_complete_none(self):
    result=self.run_fixture('--controls')
    self.assertTrue(result['real_CPU_JAX_controls'])
    self.assertTrue(result['zero_denominator_controls_all_none'])
    self.assertFalse(result['real_N1M_calibration'])

  def test_one_real_tiny_batch_keeps_core_source_and_rejects_repeat_tamper(self):
    result=self.run_fixture('--tiny')
    self.assertTrue(result['source_unchanged'])
    self.assertTrue(result['real_producer_record'])
    self.assertTrue(result['repeat_tamper_rejected'])
    self.assertTrue(result['official_save_integer_counters_preserved'])
    self.assertTrue(result['full_tree_host_changes_typed_identity'])
    self.assertTrue(result['counter_type_value_and_param_tamper_rejected'])
    self.assertFalse(result['real_N1M_calibration'])

  def run_fixture(self,option):
    with tempfile.TemporaryDirectory() as directory:
      output=Path(directory)/'proof.json'
      result=subprocess.run([sys.executable,str(Path(__file__).resolve()),option,str(output)],
          cwd=ROOT,capture_output=True,text=True,timeout=240,
          env=dict(os.environ,PYTHONPATH=str(ROOT),JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',MUJOCO_GL='disable'))
      self.assertEqual(result.returncode,0,result.stdout[-2000:]+result.stderr[-6000:])
      return json.loads(output.read_text())


def controls_fixture(output):
  dll=(os.add_dll_directory(os.environ['DL_DLL_DIRECTORY']) if os.name == 'nt' and os.environ.get('DL_DLL_DIRECTORY') else None)
  import jax
  from analysis import dl_nm1_calibration as arithmetic
  from analysis.dl_nm1_cal_audit import permutation
  d=np.linspace(.01,2,1024,dtype=np.float32).reshape(16,64)
  e=np.linspace(.1,.9,1024,dtype=np.float32).reshape(16,64)
  f=np.float32(1)-np.float32(1)/(np.float32(1)+np.float32(20)*d)
  k=np.linspace(2,4,1024,dtype=np.float32).reshape(16,64)
  a=np.ones((16,64),bool);a[:,-1]=False
  ids=np.zeros((16,64,20),np.uint8)
  for row in range(16):
    for col in range(64):ids[row,col]=np.frombuffer((row+1).to_bytes(16,'big')+col.to_bytes(4,'big'),np.uint8)
  sig=dict(K=k,E=k-1,D=d,e=e,f=f,v_N=np.where(a,np.float32(1)-np.float32(.5)*e*f,np.float32(1)),
      m=a.copy(),A=a,stepid=ids,chain=np.repeat(np.arange(16,dtype=np.int32)[:,None],64,axis=1))
  frozen=arithmetic.freeze([sig],expected_batches=1)
  arrays,controls,available=p.control_arrays(sig,frozen)
  independent=permutation(a,k,d,frozen['bins']['D_f32'],frozen['bins']['K_f32'])
  for left,right in (('P_key','key'),('P_priorities','priorities'),('P_perm','perm'),('P_group','group'),('P_counts','counts')):
    np.testing.assert_array_equal(arrays[left],independent[right])
  assert all(available[name]['available'] for name in ('v_S','v_W','v_P'))
  empty=dict(sig,K=np.full_like(k,.5),E=np.zeros_like(k),e=np.zeros_like(e),v_N=np.ones_like(k))
  no=arithmetic.freeze([empty],expected_batches=1)
  missing,weights,availability=p.control_arrays(empty,no)
  assert set(missing) == {'P_key','P_priorities'} and all(v is None for v in weights.values())
  summary=arithmetic.summarize([empty],[dict(encoder_gradient_changed_elements=0,encoder_step_changed_elements=0,
      posterior_gradient_changed_elements=0,posterior_step_changed_elements=0)], [weights],no,expected_batches=1)
  assert not any(summary['qualification'].values())
  assert all(device.platform == 'cpu' for device in jax.devices())
  Path(output).write_text(json.dumps(dict(real_CPU_JAX_controls=True,zero_denominator_controls_all_none=True,
      real_N1M_calibration=False)))
  del dll


def tiny_fixture(output):
  dll=(os.add_dll_directory(os.environ['DL_DLL_DIRECTORY']) if os.name == 'nt' and os.environ.get('DL_DLL_DIRECTORY') else None)
  import elements
  import jax
  from ruamel.yaml import YAML
  from dreamerv3.agent import Agent
  from scripts.dl_nm1_counterfactual import Counterfactual, host
  from scripts.dl_reset_learning import tree_digest
  from embodied.run.protocol_v1 import parameter_digest
  cfg=elements.Config(YAML(typ='safe').load((ROOT/'dreamerv3/configs.yaml').read_text())['defaults'])
  presets=YAML(typ='safe').load((ROOT/'dreamerv3/configs.yaml').read_text())
  cfg=cfg.update(presets['debug'])
  cfg=elements.Config({**cfg.flat,**{'batch_size':2,'batch_length':4,'replay_context':1,
      'jax.platform':'cpu','jax.compute_dtype':'float32','jax.train_devices':[0],
      'jax.enable_policy':False,'jax.precompile':False,'jax.profiler':False,
      'agent.rep_probe.mode':'off','agent.dt_latch.mode':'off'}})
  obs=dict(vector=elements.Space(np.float32,(5,)),reward=elements.Space(np.float32),
      is_first=elements.Space(bool),is_last=elements.Space(bool),is_terminal=elements.Space(bool))
  act=dict(action=elements.Space(np.float32,(2,)))
  def agent(name,mode,rho):
    config=elements.Config({**cfg.agent.flat,**{'dt_latch.mode':mode,'dt_latch.rho':rho,'dt_latch.normal_only':True}})
    return Agent(obs,act,elements.Config(**config.flat,seed=0,jax=cfg.jax,batch_size=2,batch_length=4,
        replay_context=1,logdir=str(Path(output).parent/name)))
  b=agent('B','off',.5);raw=b._zeros(b.spaces,(2,5))
  raw['vector'][:]=np.arange(50,dtype=np.float32).reshape(2,5,5)/50
  raw['reward'][:]=[[0,.1,.3,.8,.9],[0,-.2,.4,.7,.5]]
  raw['is_first'][:,0]=True;raw['is_first'][1,3]=True
  raw['is_last'][:,-1]=True;raw['is_last'][1,2]=True
  raw['action'][:]=np.linspace(-.3,.3,20).reshape(2,5,2)
  raw['dyn/deter'][:]=.2;raw['dyn/stoch'][...,0]=1
  chains={}
  for row in range(2):
    uuid=(row+1).to_bytes(16,'big');chains[uuid]=row
    for col in range(5):raw['stepid'][row,col]=np.frombuffer(uuid+col.to_bytes(4,'big'),np.uint8)
  carry=b.init_train(2)
  _,oldout,oldmets=b.train(carry,dict(b.prepare_batch(raw)))
  assert not oldout and not oldmets;b.take_train_result()
  initial=b.save();digest=tree_digest(initial)
  variants={'B':Counterfactual(b)}
  for name,mode,rho in (('rho0','dtlatch',0.),('N','dtlatch',.5)):
    branch=agent(name,mode,rho);branch.load(initial);variants[name]=Counterfactual(branch)
  # Exercise the shared production load/final identity path with actual public
  # save/load trees. Fixture identities replace N1M pins only in this test.
  with mock.patch.object(p,'FULL_STATE',digest),mock.patch.object(p,'PARAMS',parameter_digest(b)), \
      mock.patch.object(p,'COUNTERS',initial['counters']), \
      mock.patch.object(p,'STEP',int(initial['params']['opt/step/value'])):
    for probe in variants.values():
      saved=p.verified_state(probe.agent)
      assert tree_digest(saved) == digest
      assert all(type(value) is int for value in saved['counters'].values())
      assert json.loads(json.dumps(saved['counters'])) == initial['counters']
    converted=host(initial)
    assert all(type(value) is np.ndarray and value.shape == () for value in converted['counters'].values())
    assert tree_digest(converted) != digest
    for case in ('counter_type','counter_value','parameter'):
      changed={name:dict(value) for name,value in initial.items()}
      if case == 'counter_type':changed['counters']['updates']=np.asarray(changed['counters']['updates'])
      elif case == 'counter_value':changed['counters']['updates']+=1
      else:
        key=next(k for k in changed['params'] if k.startswith('enc/'))
        value=changed['params'][key].copy()
        value.flat[0]=np.nextafter(value.flat[0],np.float32(np.inf))
        changed['params'][key]=value
      with mock.patch.object(b,'save',return_value=changed):
        try:p.verified_state(b)
        except ValueError as error:assert 'Loaded full state' in str(error)
        else:raise AssertionError('Typed full-state identity accepted '+case)
  carry=b.init_train(2)
  with mock.patch.object(p,'BATCH',2),mock.patch.object(p,'LENGTH',4),mock.patch.object(p,'STEP',1):
    signal,effects,forward=p.collect_batch(raw,variants,carry,0,chains,Path(output).parent/'batch_000')
    assert signal['K'].shape == (2,4) and set(effects) == {
        'encoder_gradient_changed_elements','encoder_step_changed_elements',
        'posterior_gradient_changed_elements','posterior_step_changed_elements'}
    record=p.read_json(Path(output).parent/'batch_000/record.json')
    assert record['identity_checks'] == dict(B_repeat_equal=True,rho0_equal=True)
    assert record['counter'] == 1
    expected_context=(({},dict(deter=raw['dyn/deter'][:,0],stoch=raw['dyn/stoch'][:,0]),{}),
        {key:raw[key][:,1:] for key in b.obs_space},
        {'action':raw['action'][:,:-1]},raw['stepid'][:,1:])
    assert record['context_sha256'] == tree_digest(expected_context)
    with np.load(Path(output).parent/'batch_000/evidence.npz',allow_pickle=False) as stored:
      assert set(stored.files) == {x['storage_key'] for x in record['arrays'].values()}
      for key in ('signal/K','signal/E','raw/consec','replay_B/stepid','replay_N/stepid'):
        assert key in record['arrays']
    original=variants['B'].production;calls=[0]
    def tamper(*args):
      value=original(*args);calls[0]+=1
      if calls[0] == 2:
        key=next(k for k in value['params'] if k.startswith('enc/'))
        tensor=value['params'][key];tensor.flat[0]=np.nextafter(tensor.flat[0],np.float32(np.inf))
      return value
    with mock.patch.object(variants['B'],'production',side_effect=tamper):
      try:p.collect_batch(raw,variants,carry,1,chains,Path(output).parent/'batch_bad')
      except ValueError as error:assert 'B/B' in str(error)
      else:raise AssertionError('Corrupt repeated production result accepted')
  for probe in variants.values():
    probe.fence(full=True);assert tree_digest(probe.agent.save()) == digest
  assert all(device.platform == 'cpu' for device in jax.devices())
  Path(output).write_text(json.dumps(dict(source_unchanged=True,real_producer_record=True,
      repeat_tamper_rejected=True,official_save_integer_counters_preserved=True,
      full_tree_host_changes_typed_identity=True,counter_type_value_and_param_tamper_rejected=True,
      real_N1M_calibration=False,scope='one generated B2/T4 F32 scratch batch')))
  del dll


if __name__ == '__main__':
  if len(sys.argv) == 3 and sys.argv[1] == '--controls':controls_fixture(sys.argv[2])
  elif len(sys.argv) == 3 and sys.argv[1] == '--tiny':tiny_fixture(sys.argv[2])
  else:unittest.main()
