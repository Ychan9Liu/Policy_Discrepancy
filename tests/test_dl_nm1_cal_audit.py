"""Independent audit arithmetic fixtures; never real N1M C qualification."""
import json
import hashlib
import math
import os
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import dl_nm1_cal_audit as audit


def write_json(path,value):
  path.write_text(json.dumps(value,sort_keys=True,allow_nan=False),encoding='utf-8')


def pin(path,label=None):
  return dict(path=label or str(path.resolve()),**audit.raw_hash(path))


def closed_graph(root):
  """655 authenticated opaque files; no fake successful NN/data qualification."""
  run=root/'run';run.mkdir();replay=root/'replay';replay.mkdir()
  inputs={}
  for name in ('checkpoint','config','inventory','resource_record'):
    path=root/name
    if name == 'inventory':write_json(path,dict(replay_directory=str(replay.resolve())))
    else:path.write_bytes(b'opaque-'+name.encode())
    inputs[name]=pin(path)
  inputs['stage_record']=None
  parent=dict(pid=123401,pgid=123400,start_ticks=17,parent_pid=10)
  worker=dict(pid=123402,pgid=123402,start_ticks=18,parent_pid=123401)
  budget=dict(version=audit.SCIENCE,schema=1,git_sha='a'*40,seconds=3600,
      hostname='lyg0326',logical_server='sv3',physical_gpu=5,gpu_uuid=audit.GPU_UUID,
      monotonic_deadline=3700.,parent_identity=parent,source_files={'fixture':{}},inputs=inputs,
      stage_expected_sha256=None,replay_directory=str(replay.resolve()),
      constants=dict(batches=128,batch_size=16,loss_length=64,context=1,plan_root=20261101,
          plan_tag=0x4E4D4341,model_seed=0,opt_step=249493))
  write_json(run/'budget.json',budget)
  write_json(run/'process.json',dict(version=audit.SCIENCE,schema=1,parent_identity=parent,
      worker_identity=worker,command=['fixture'],started_monotonic=100.))
  top={'budget.json','plan.json','initial.json','initial.npz','resource_refreshed.json',
      'runtime_before.json','runtime_after.json','frozen.json','summary.json','final.json'}
  for name in top-{'budget.json'}:(run/name).write_bytes(b'opaque')
  for name in ('worker.log','resources.jsonl'):(run/name).write_bytes(b'opaque')
  for i in range(128):
    directory=run/f'batch_{i:03d}';directory.mkdir()
    for name in ('record.json','evidence.npz','controls.json','controls.npz'):(directory/name).write_bytes(b'opaque')
    artifacts={name:pin(directory/name,name) for name in ('record.json','evidence.npz','controls.json','controls.npz')}
    write_json(directory/'complete.json',dict(version=audit.SCIENCE,schema=1,batch=i,artifacts=artifacts,
        record_sha256=artifacts['record.json']['sha256'],controls_sha256=artifacts['controls.json']['sha256']))
  write_json(run/'generation_complete.json',dict(version=audit.SCIENCE,schema=1,git_sha='a'*40,
      complete_batches=128,optimizer_updates_committed=0,replay_updates_committed=0,
      independent_audit_complete=False,subsequent_learning_authorized=False,
      top_artifacts={name:pin(run/name,name) for name in top},
      batches=[pin(run/f'batch_{i:03d}'/'complete.json',f'batch_{i:03d}/complete.json') for i in range(128)]))
  write_json(run/'supervisor_terminal.json',dict(version=audit.SCIENCE,schema=1,git_sha='a'*40,
      parent_identity=parent,worker_identity=worker,worker_returncode=0,failure=None,
      worker_terminal=True,owned_group_cleaned=True,elapsed_seconds=100.,
      artifacts={name:pin(run/name,name) for name in
          ('budget.json','process.json','worker.log','resources.jsonl','generation_complete.json')}))
  launcher=root/'launcher.json'
  write_json(launcher,dict(version='DL-NM1-launcher-r1',git_sha='a'*40,parent_returncode=0,
      parent_terminal=True,parent_identity_gone=True,watchdog_seconds=3610,elapsed_seconds=101.,
      run_directory=str(run.resolve()),parent_identity=parent,
      supervisor_terminal_sha256=pin(run/'supervisor_terminal.json')['sha256']))
  return run,launcher,inputs


def fixed_signals():
  records=[];rng=np.random.default_rng(482)
  for j in range(128):
    k=rng.uniform(.8,4,(16,64)).astype(np.float32);d=rng.uniform(0,1,(16,64)).astype(np.float32)
    m=rng.random((16,64))>.1;e=rng.uniform(0,1,(16,64)).astype(np.float32)
    f=rng.uniform(0,1,(16,64)).astype(np.float32);a=m.copy()
    ids=np.zeros((16,64,20),np.uint8)
    ids[:,:,12:]=np.arange(j*1024,(j+1)*1024,dtype='>u8').view(np.uint8).reshape(16,64,8)
    records.append(dict(K=k,E=np.maximum(k-np.float32(1),0),D=d,e=e,f=f,m=m,A=a,
        v_N=np.float32(1)-a*np.float32(.5)*e*f,stepid=ids,
        chain=np.broadcast_to(np.arange(16,dtype=np.int32)[:,None],(16,64)).copy()))
  return records


def controls_for(signals,frozen):
  controls=[]
  for s in signals:
    item={}
    for name,key in (('S','f'),('W','e')):
      coefficient=frozen[name]['coefficient_f32']
      item['v_'+name]=None if coefficient is None else np.float32(1)-s['A']*np.float32(coefficient)*s[key]
    item['v_P']=None
    if frozen['bins']['calibrated']:
      p=audit.permutation(s['A'],s['K'],s['D'],frozen['bins']['D_f32'],frozen['bins']['K_f32'])
      item['v_P']=s['v_N'].reshape(-1)[p['perm']]
    controls.append(item)
  return controls


def effect_counts(encoder=False,posterior=False):
  return {group+'_'+kind+'_changed_elements':int(enabled) for group,enabled in
      (('encoder',encoder),('posterior',posterior)) for kind in ('gradient','step')}


def batch_fixture(root):
  """One real NPZ/typed batch, synthetic values and tiny dimensions, no NN."""
  import ml_dtypes
  from scripts import dl_nm1_cal as writer
  from scripts.dl_reset_learning import tree_digest
  fields=dict(audit.RAW_FIELDS,image=('uint8',(2,2,3)),action=('float32',(2,)),
      **{'dyn/deter':('float32',(2,)),'dyn/stoch':('float32',(2,2))})
  chunks={};uids=['first','second']
  for i,length in enumerate((40,60)):
    data={key:np.zeros((length,*tail),dtype) for key,(dtype,tail) in fields.items()}
    data['dyn/stoch'][:,:,0]=1;data['dyn/deter'][:]=np.arange(length)[:,None]+i*40
    data['stepid'][:,0]=i+1;data['stepid'][:,16:]=np.arange(length,dtype='>u4').view(np.uint8).reshape(-1,4)
    if i==0:data['is_first'][0]=True
    else:data['is_last'][-1]=data['is_terminal'][-1]=True
    chunks[uids[i]]=dict(data=data,length=length,successor='second' if i==0 else 'end')
  source=dict(chunks=chunks,chain_by_uuid={'first':0,'second':0},plan=[[['first',i] for i in range(16)]])
  with mock.patch.object(audit,'RAW_FIELDS',fields):raw,chain=audit.reconstruct_batch(source,source['plan'][0])
  selected={'enc/w':np.array([1.,2.],np.float32),'dyn/obslogit/w':np.array([3.,4.],np.float32)}
  params=dict(selected,**{'opt/step/value':np.array(249493,np.int32)})
  seed=np.random.default_rng([0,249493]).integers(0,np.iinfo(np.uint32).max,(2,),np.uint32)
  carry=({},dict(deter=np.zeros((16,2),ml_dtypes.bfloat16),stoch=np.zeros((16,2,2),ml_dtypes.bfloat16)),{},
      dict(action=np.zeros((16,2),np.float32)))
  arrays={'raw/'+k:v for k,v in raw.items()};arrays['prepared_key']=seed
  arrays.update({'initial_carry/1/deter':carry[1]['deter'],'initial_carry/1/stoch':carry[1]['stoch'],
      'initial_carry/3/action':carry[3]['action']})
  m=np.ones((16,64),bool);m[:,-1]=False
  s=dict(K=np.full((16,64),2,np.float32),E=np.ones((16,64),np.float32),D=np.full((16,64),.25,np.float32),
      e=np.full((16,64),.5,np.float32),f=np.full((16,64),.8,np.float32),m=m,A=m.copy(),
      v_N=np.where(m,np.float32(.8),np.float32(1)),stepid=raw['stepid'][:,1:],chain=chain)
  arrays.update({'signal/'+k:v for k,v in s.items()})
  receipts={};reference={}
  for name,delta in (('B',.01),('N',.02)):
    updated=dict(params,**{k:v+np.float32(delta) for k,v in selected.items()})
    updated['opt/step/value']=np.array(249494,np.int32)
    for key,value in selected.items():
      arrays['grad_'+name+'/'+key]=np.full_like(value,1 if name=='B' else 2)
      arrays['step_'+name+'/'+key]=updated[key]
    dyn={k:np.ones((16,64,*fields['dyn/'+k][1]),ml_dtypes.bfloat16) for k in ('deter','stoch')}
    replay=dict(stepid=raw['stepid'][:,1:],**{'dyn/'+k:v for k,v in dyn.items()})
    arrays.update({'replay_'+name+'/'+k:v for k,v in replay.items()})
    arrays.update({'reference_replay_'+name+'/dyn/'+k:v for k,v in dyn.items()})
    receipts[name]=dict(state_sha256='a'*64,params={k:writer.array_spec(v) for k,v in updated.items()},
        carry_sha256='b'*64,replay_sha256=tree_digest(replay),metrics_sha256='c'*64)
    reference[name]=dict(state_scope='separate-complete-reference-JIT-not-production-intermediate',
        accessed=list(selected),modified=[],gradient_sha256='d'*64,
        forward_sha256=tree_digest(({},dyn,{})))
  arrays.update({'reference_B/K':s['K'],'reference_B/E':s['E'],
      'reference_B/rep_before':s['K'],'reference_B/weight':np.ones((16,64),np.float32),
      'reference_N/rep_before':s['K']})
  directory=root/'batch_000';directory.mkdir()
  record=dict(version=audit.SCIENCE,schema=1,batch=0,counter=249493,opt_step_before=249493,
      raw_sha256=tree_digest(raw),prepared_sha256=tree_digest(dict(raw,seed=seed)),
      carry_sha256=tree_digest(carry),context_sha256=tree_digest((
          ({},dict(deter=raw['dyn/deter'][:,0],stoch=raw['dyn/stoch'][:,0]),{}),
          {k:raw[k][:,1:] for k in ('image','reward','is_first','is_last','is_terminal')},
          dict(action=raw['action'][:,:-1]),raw['stepid'][:,1:])),
      production=dict(receipts,B_repeat=receipts['B'],rho0=receipts['B']),reference=reference,
      identity_checks=dict(B_repeat_equal=True,rho0_equal=True),effects=effect_counts(True,True),
      forward={key:True for key in ('production_replay_equal','reference_replay_equal',
          'production_vs_reference_B_equal','production_vs_reference_N_equal')})
  # Two elements per retained group, not one.
  record['effects']={key:2 for key in record['effects']}
  initial=dict(variant_seals={'B':dict(carry_sha256=tree_digest(carry))},state_leaves={k:writer.array_spec(v) for k,v in params.items()})
  def save():
    path=directory/'evidence.npz'
    if path.exists():path.unlink()
    record['arrays']=writer.write_arrays(path,arrays);write_json(directory/'record.json',record)
  save()
  return dict(run=root,original_params=params),source,initial,selected,fields,arrays,record,save


class CalibrationAuditTests(unittest.TestCase):
  def test_complete_closed_graph_and_all_failures_precede_numpy_or_pickle(self):
    with tempfile.TemporaryDirectory() as temporary:
      run,launcher,inputs=closed_graph(Path(temporary))
      with mock.patch.object(audit,'git_binding'),mock.patch.object(audit,'inactive'), \
          mock.patch.object(audit,'inactive_worker_group'), \
          mock.patch.object(audit,'CHECKPOINT',inputs['checkpoint']['sha256']), \
          mock.patch.object(audit,'CONFIG',inputs['config']['sha256']), \
          mock.patch.object(audit,'INVENTORY',inputs['inventory']['sha256']), \
          mock.patch.object(audit,'checkpoint',side_effect=AssertionError('pickle decoded')) as pickle_decode, \
          mock.patch.object(audit,'load_pack',side_effect=AssertionError('NPZ decoded')) as npz_decode, \
          mock.patch.object(audit,'replay_source',side_effect=AssertionError('source decoded')) as source_decode:
        auth=audit.authenticate(run,'a'*40,'b'*40,launcher,pin(launcher)['sha256'])
        self.assertEqual(len(auth['files']),655)
        original=launcher.read_bytes();value=audit.read_json(launcher)
        for key,replacement in (('parent_returncode',1),('parent_identity_gone',False),('elapsed_seconds',3600.01)):
          write_json(launcher,dict(value,**{key:replacement}))
          with self.subTest(key=key),self.assertRaisesRegex(ValueError,'external parent exit'):
            audit.analyze(run,'a'*40,'b'*40,launcher,pin(launcher)['sha256'])
        launcher.write_bytes(original)
        target=run/'batch_127'/'evidence.npz';target.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Raw file pin differs'):
          audit.analyze(run,'a'*40,'b'*40,launcher,pin(launcher)['sha256'])
        pickle_decode.assert_not_called();npz_decode.assert_not_called();source_decode.assert_not_called()

  def test_extra_file_and_live_original_session_cannot_open_data(self):
    with tempfile.TemporaryDirectory() as temporary:
      run,launcher,inputs=closed_graph(Path(temporary))
      with mock.patch.object(audit,'git_binding'),mock.patch.object(audit,'inactive'), \
          mock.patch.object(audit,'inactive_worker_group',side_effect=ValueError('live descendants')), \
          mock.patch.object(audit,'inventory') as scan:
        with self.assertRaisesRegex(ValueError,'live descendants'):
          audit.authenticate(run,'a'*40,'b'*40,launcher,pin(launcher)['sha256'])
        scan.assert_not_called()
      (run/'unplanned').write_bytes(b'x')
      with mock.patch.object(audit,'git_binding'),mock.patch.object(audit,'inactive'), \
          mock.patch.object(audit,'inactive_worker_group'):
        with self.assertRaisesRegex(ValueError,'missing/extra files'):
          audit.authenticate(run,'a'*40,'b'*40,launcher,pin(launcher)['sha256'])

  def test_full128_independent_pooled_math_matches_producer_and_or_is_per_batch(self):
    from analysis import dl_nm1_calibration as producer
    signals=fixed_signals();frozen=audit.independent_freeze(signals)
    self.assertEqual(frozen,producer.freeze(signals))
    effects=[effect_counts(i<8,8<=i<16) for i in range(128)]
    controls=controls_for(signals,frozen)
    result=audit.independent_summary(signals,effects,controls,frozen)
    self.assertEqual(result,producer.summarize(signals,effects,controls,frozen))
    self.assertEqual(result['numerical_effect_batches'],16)
    self.assertTrue(result['qualification']['C_N'])
    self.assertEqual(result['distinct_protected_chains'],16)
    self.assertFalse(result['subsequent_learning_authorized'])
    # Repeated visits cannot inflate distinct-source coverage.
    for s in signals:s['stepid'][:]=0;s['chain'][:]=0
    duplicate=audit.independent_summary(signals,effects,controls,frozen)
    self.assertEqual(duplicate['distinct_protected_stepids'],1)
    self.assertFalse(duplicate['qualification']['C_N_coverage'])
    signals[-1]['chain'][0,0]=1
    signals[-1]['A'][0,0]=True;signals[-1]['K'][0,0]=2;signals[-1]['v_N'][0,0]=.5
    with self.assertRaisesRegex(ValueError,'multiple chains'):
      audit.independent_summary(signals,effects,controls,frozen)

  def test_full128_zero_denominator_is_complete_negative_without_fallback(self):
    from analysis import dl_nm1_calibration as producer
    signals=fixed_signals()
    for s in signals:s['K'][:]=1;s['E'][:]=0;s['v_N'][:]=1
    frozen=audit.independent_freeze(signals)
    self.assertEqual(frozen,producer.freeze(signals))
    self.assertIsNone(frozen['S']['coefficient_f64']);self.assertIsNone(frozen['W']['coefficient_f32'])
    controls=controls_for(signals,frozen);effects=[effect_counts() for _ in signals]
    result=audit.independent_summary(signals,effects,controls,frozen)
    self.assertEqual(result,producer.summarize(signals,effects,controls,frozen))
    self.assertEqual(result['release_actual_f64'],dict(N=0.,S=None,W=None,P=None))
    self.assertIsNone(result['P_changed_protected_fraction'])
    self.assertFalse(any(result['qualification'].values()))

  def test_freeze_uses_one_full_position_fsum_and_cast_then_unique(self):
    signals=fixed_signals()
    for s in signals:s['A'][:]=False;s['e'][:]=1;s['f'][:]=1
    signals[0]['A'][0,:2]=True;signals[1]['A'][0,0]=True
    signals[0]['E'][0,:2]=[2**53,1];signals[1]['E'][0,0]=1
    signals[0]['D'][0,:2]=[1,np.nextafter(np.float32(1),np.float32(2))]
    signals[1]['D'][0,0]=1
    result=audit.independent_freeze(signals)
    self.assertEqual(result['aggregate64']['S_denominator'],float(2**53+2))
    perbatch=math.fsum(math.fsum(float(x) for x in s['E'][s['A']]) for s in signals)
    self.assertNotEqual(perbatch,result['aggregate64']['S_denominator'])
    self.assertEqual(result['bins']['D_f32'],sorted(set(np.float32(result['bins']['D_quantiles_f64']).tolist())))

  def test_actual_typed_cross_chunk_batch_context_and_inner_rejections(self):
    with tempfile.TemporaryDirectory() as temporary:
      auth,source,initial,selected,fields,arrays,record,save=batch_fixture(Path(temporary))
      with mock.patch.object(audit,'RAW_FIELDS',fields):
        signal,effects,stats,forward=audit.audit_batch(auth,source,initial,selected,0)
        self.assertEqual(effects,effect_counts(True,True)|{k:2 for k in effects})
        self.assertEqual(arrays['raw/dyn/deter'][0,40,0],40)
        self.assertEqual(stats['encoder']['retained_tensors'],1)
        self.assertTrue(all(forward.values()))
        original=record['context_sha256'];record['context_sha256']='0'*64;save()
        with self.assertRaisesRegex(ValueError,'context/previous raw'):audit.audit_batch(auth,source,initial,selected,0)
        record['context_sha256']=original
        m=arrays['signal/m'];arrays['signal/m']=m.astype(np.float32);save()
        with self.assertRaisesRegex(ValueError,'boolean m/A'):audit.audit_batch(auth,source,initial,selected,0)
        arrays['signal/m']=m;arrays['signal/chain']=arrays['signal/chain'][...,None];save()
        with self.assertRaisesRegex(ValueError,'typed stepid/source chain'):audit.audit_batch(auth,source,initial,selected,0)
        arrays['signal/chain']=arrays['signal/chain'][...,0];arrays['signal/D'][0,0]=-1e-6;save()
        with self.assertRaisesRegex(ValueError,'raw KL tolerance'):audit.audit_batch(auth,source,initial,selected,0)

  def test_signed_zero_is_not_numerical_effect_and_shape_changes_rejected(self):
    initial={'enc/w':np.array([0.,1.],np.float32),'dyn/obs/w':np.array([0.,2.],np.float32)}
    arrays={name+'/'+k:v.copy() for k,v in initial.items() for name in ('grad_B','grad_N','step_B','step_N')}
    for name in ('grad_N','step_N'):
      for k in initial:arrays[name+'/'+k][0]=-0.
    effects,stats=audit.effect_tensors(initial,arrays)
    self.assertFalse(any(effects.values()));self.assertIsNone(stats['encoder']['update_cosine'])
    self.assertFalse(stats['encoder']['gradient']['byte_identical'])
    self.assertEqual(stats['encoder']['gradient']['max_reference_ulp_difference'],0.)
    self.assertEqual(stats['encoder']['gradient']['reference_zero_elements'],1)
    arrays['grad_N/enc/w']=arrays['grad_N/enc/w'].reshape(1,2)
    with self.assertRaisesRegex(ValueError,'matched F32'):audit.effect_tensors(initial,arrays)

  def test_known_checkpoint_pin_and_restricted_numpy_decode(self):
    params={f'leaf{i}':np.array([i],np.float32) for i in range(293)}
    params['opt/step/value']=np.array(249493,np.int32)
    value=dict(params=params,counters=dict(audit.COUNTERS));body=pickle.dumps(value,protocol=5)
    with mock.patch.object(audit,'CHECKPOINT',hashlib.sha256(body).hexdigest()), \
        mock.patch.object(audit,'PARAMS',audit.parameter_hash(params)), \
        mock.patch.object(audit,'FULLSTATE',audit.typed_hash(value)):
      original,loaded=audit.checkpoint(body)
      self.assertEqual(audit.typed_hash(loaded),audit.typed_hash(value))
      self.assertEqual(original['params']['leaf3'].dtype,np.float32)
      with mock.patch.object(audit.NumpyCheckpoint,'load',side_effect=AssertionError('unsafe decode')) as load:
        with self.assertRaisesRegex(ValueError,'before decode'):audit.checkpoint(body+b'x')
        load.assert_not_called()
    evil=pickle.dumps(os.system)
    with mock.patch.object(audit,'CHECKPOINT',hashlib.sha256(evil).hexdigest()):
      with self.assertRaisesRegex(ValueError,'non-NumPy class'):audit.checkpoint(evil)

  @unittest.skipUnless(os.name=='posix','Linux /proc session identity semantics')
  def test_reaped_helper_leader_descendant_is_fenced_and_cleaned(self):
    child=subprocess.Popen([sys.executable,'-c',
        'import subprocess,sys;subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)'],
        start_new_session=True)
    owned=audit.process_identity(child.pid);audit.OWNED[child.pid]=owned
    try:
      child.wait(timeout=5)
      self.assertTrue(audit.group_alive(owned))
      with self.assertRaisesRegex(ValueError,'live descendants'):audit.inactive_worker_group(owned)
      audit.cleanup(child);self.assertFalse(audit.group_alive(owned));self.assertNotIn(child.pid,audit.OWNED)
    finally:
      if child.pid in audit.OWNED:audit.cleanup(child)

  def test_reused_pid_never_authorizes_group_signal(self):
    owned=dict(pid=42,pgid=42,start_ticks=100)
    with mock.patch.object(audit.os,'killpg',create=True) as kill, \
        mock.patch.object(audit,'process_identity',return_value=dict(pid=42,pgid=42,start_ticks=101)):
      with self.assertRaisesRegex(ValueError,'PID reused'):audit.group_alive(owned)
      self.assertEqual(kill.call_args_list,[mock.call(42,0)])

  def test_actual_control_npz_is_checked_against_fixed_rng_not_only_strata(self):
    from scripts import dl_nm1_cal as writer
    signals=fixed_signals();frozen=audit.independent_freeze(signals);s=signals[0]
    with tempfile.TemporaryDirectory() as temporary:
      root=Path(temporary);directory=root/'batch_000';directory.mkdir()
      p=audit.permutation(s['A'],s['K'],s['D'],frozen['bins']['D_f32'],frozen['bins']['K_f32'])
      arrays=dict(controls_for(signals[:1],frozen)[0],P_key=p['key'],P_priorities=p['priorities'],
          P_perm=p['perm'],P_group=p['group'],P_counts=p['counts'])
      record=dict(version=audit.SCIENCE,schema=1,batch=0,p_seed=0,p_step=249493,frozen_sha256='a'*64,
          availability={key:dict(available=True,reason=None) for key in ('v_S','v_W','v_P')})
      def save():
        path=directory/'controls.npz'
        if path.exists():path.unlink()
        record['arrays']=writer.write_arrays(path,arrays);write_json(directory/'controls.json',record)
      save();auth=dict(run=root,files={'frozen.json':dict(sha256='a'*64)})
      actual=audit.audit_control(auth,0,s,frozen)
      self.assertEqual(actual['v_P'].tobytes(),arrays['v_P'].tobytes())
      eligible=np.flatnonzero(s['A'].reshape(-1));pair=None
      for i in eligible:
        same=eligible[p['group'].reshape(-1)[eligible] == p['group'].reshape(-1)[i]]
        if len(same)>1:pair=same[:2];break
      a,b=pair;flat=arrays['P_perm'].reshape(-1);flat[a],flat[b]=flat[b],flat[a]
      arrays['v_P']=s['v_N'].reshape(-1)[arrays['P_perm']];save()
      with self.assertRaisesRegex(ValueError,'global P perm'):audit.audit_control(auth,0,s,frozen)

  def test_source_npz_graph_plan_is_independent_and_hashes_all_before_decode(self):
    fields=dict(audit.RAW_FIELDS,image=('uint8',(2,2,3)),action=('float32',(2,)),
        **{'dyn/deter':('float32',(2,)),'dyn/stoch':('float32',(2,2))})
    with tempfile.TemporaryDirectory() as temporary:
      directory=Path(temporary);names=[];summaries=[];inventory=[]
      identities=['0'*21+str(i) for i in (1,2,3)]
      for i,length in enumerate((40,60)):
        timestamp=f'20261015T000000F00000{i+1}'
        name=f'{timestamp}-{identities[i]}-{identities[i+1]}-{length}.npz';names.append(name)
        data={key:np.zeros((length,*tail),dtype) for key,(dtype,tail) in fields.items()}
        data['dyn/stoch'][:,:,0]=1;data['stepid'][:,:16]=np.frombuffer((i+1).to_bytes(16,'big'),np.uint8)
        data['stepid'][:,16:]=np.arange(length,dtype='>u4').view(np.uint8).reshape(-1,4)
        data['is_first'][0]=i==0
        if i==1:data['is_last'][-1]=data['is_terminal'][-1]=True
        np.savez_compressed(directory/name,**data)
        inventory.append(dict(path=name,**audit.raw_hash(directory/name)))
        summaries.append(dict(path=name,time=timestamp,uuid=identities[i],successor=identities[i+1],length=length,
            first=i==0,last=i==1,resets=int(i==0),terminals=int(i==1),ends=int(i==1),
            crc_verified=True,cached_context_validated=True))
      report=dict(version='DL-replay-inventory-r1',schema=1,source_git='13ec72a214b0c03b886b2c7fac52c8f089966dec',
          stable=True,before_inventory=inventory,after_inventory=inventory,rows=100,resets=1,train_actions=99,
          fields={k:[dtype,list(tail)] for k,(dtype,tail) in fields.items()},chunks=summaries,
          chains=[dict(head_uuid=identities[0],dangling_successor=identities[2],paths=names,rows=100,length65_windows=36)])
      with mock.patch.object(audit,'RAW_FIELDS',fields),mock.patch.object(audit,'COUNTS',(2,1,100,1,99)):
        source=audit.replay_source(directory,report)
        self.assertEqual(source['legal65_starts'],36)
        # All36 legal starts belong to first chunk; sample directly over this enumerable list.
        seed=int(np.random.SeedSequence([20261101,0x4E4D4341]).generate_state(1,dtype=np.uint64)[0])
        draws=np.random.default_rng(seed).integers(36,size=(128,16))
        self.assertEqual(source['plan'],[[[identities[0],int(i)] for i in row] for row in draws])
        (directory/names[-1]).write_bytes(b'tampered-tail')
        with mock.patch.object(np.lib.format,'read_array',side_effect=AssertionError('decode before hash')) as decode:
          with self.assertRaisesRegex(ValueError,'raw SHA differs'):audit.replay_source(directory,report)
          decode.assert_not_called()

  def test_compute_only_resource_receipts_reject_egl_graphics_and_wrong_gpu(self):
    from scripts import dl_nm1_cal as producer
    worker=1234;uuid=audit.GPU_UUID
    with tempfile.TemporaryDirectory() as temporary:
      root=Path(temporary);resource=root/'source.json'
      idle=dict(physical_gpu=5,gpu_uuid=uuid,processes=[],pmon_jobs=[],raw_pmon='5 - - idle')
      original=dict(version='DL-NM1-compute-resource-r1',logical_server='sv3',actual_hostname='lyg0326',
          physical_gpu=5,gpu_uuid=uuid,compute_uuid=uuid,egl_uuid=None,graphics_verified=False,
          environment=producer.COMPUTE_ENV,occupancy='idle',exclusive=True,live_check=idle,
          occupancy_basis='available complete XML process inventory and assigned-device pmon',
          source_mapping='not-applicable-no-environment-or-EGL-initialization')
      write_json(resource,original)
      refreshed=dict(original,resource_source_path=str(resource),resource_source_sha256=pin(resource)['sha256'])
      write_json(root/'resource_refreshed.json',refreshed)
      live=dict(idle,processes=[dict(pid=worker,type='C')],pmon_jobs=[dict(pid=worker,type='C')],raw_pmon='5 1234 C busy')
      sample=dict(live,worker_pid=worker,monotonic=101.,xml_process_summary=live['processes'])
      (root/'resources.jsonl').write_text(json.dumps(sample)+'\n')
      runtime=dict(pid=worker,compute_uuid=[uuid],physical_gpu=5,graphics_verified=False,pmon=live['raw_pmon'],live_state=live)
      for name in ('runtime_before.json','runtime_after.json'):write_json(root/name,runtime)
      paths={k:root/k for k in ('checkpoint','config','inventory')};paths.update(resource_record=resource,stage_record=None)
      command=['python','dl_nm1_cal.py','--worker','--expected-git','a'*40,'--output',str(root),'--replay-directory',str(root/'replay')]
      for key in ('checkpoint','config','inventory','resource_record'):command.extend(('--'+key.replace('_','-'),str(paths[key])))
      auth=dict(run=root,budget=dict(git_sha='a'*40,monotonic_deadline=3700.,inputs={'resource_record':pin(resource)}),
          terminal=dict(worker_identity=dict(pid=worker)),process=dict(command=command,started_monotonic=100.),
          directory=root/'replay',paths=paths)
      self.assertEqual(audit.resources(auth),1)
      refreshed['egl_uuid']=uuid;write_json(root/'resource_refreshed.json',refreshed)
      with self.assertRaisesRegex(ValueError,'resource binding'):audit.resources(auth)
      refreshed['egl_uuid']=None;write_json(root/'resource_refreshed.json',refreshed)
      sample['processes'][0]['type']='G';(root/'resources.jsonl').write_text(json.dumps(sample)+'\n')
      with self.assertRaisesRegex(ValueError,'graphics work'):audit.resources(auth)
      sample['processes'][0]['type']='C';sample['raw_pmon']='4 1234 C busy\n5 1234 C busy'
      (root/'resources.jsonl').write_text(json.dumps(sample)+'\n')
      with self.assertRaisesRegex(ValueError,'another GPU'):audit.resources(auth)

  def test_bf16_storage_preserves_signed_zero_and_rejects_nonfinite_bits(self):
    bits = np.array([0x0000, 0x8000, 0x3F80, 0x4000, 0xBF80, 0x7F7F], np.uint16)
    values = audit.bf16_values(bits)
    np.testing.assert_array_equal(values[:5], np.array([0., -0., 1., 2., -1.], np.float32))
    self.assertTrue(np.signbit(values[1]))
    self.assertEqual(values[-1], np.float32(np.ldexp(2 - 2 ** -7, 127)))
    self.assertEqual((values.view(np.uint32) >> np.uint32(16)).astype(np.uint16).tobytes(), bits.tobytes())
    for value in (0x7F80, 0xFF80, 0x7F81):
      with self.assertRaisesRegex(ValueError, 'Nonfinite'):
        audit.bf16_values(np.array([value], np.uint16))

  def test_independent_threefry_matches_real_jax033_cpu_fixed_c_keys_and_permutation(self):
    with tempfile.TemporaryDirectory(prefix='dl-nm1-prng-') as temporary:
      output = Path(temporary) / 'actual.npz'
      result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
          '--jax-fixture', str(output)], cwd=ROOT, capture_output=True, text=True,
          timeout=60, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', MUJOCO_GL='disable'))
      self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-3000:])
      with np.load(output, allow_pickle=False) as actual:
        key, priorities = audit.permutation_randomness()
        self.assertEqual(key.tobytes(), actual['key'].tobytes())
        self.assertEqual(priorities.tobytes(), actual['priorities'].tobytes())
        expected = audit.permutation(actual['mask'], actual['kl'], actual['D'], [.2, .6], [1.])
        self.assertEqual(expected['perm'].tobytes(), actual['perm'].tobytes())
        self.assertEqual(expected['group'].tobytes(), actual['group'].tobytes())
        self.assertEqual(audit.threefry(actual['adversarial_key'], actual['count']).tobytes(),
            actual['adversarial_bits'].tobytes())

  def test_threefry_input_rejects_odd_counters_or_wrong_dtype(self):
    with self.assertRaisesRegex(ValueError, 'even uint32'):
      audit.threefry(np.zeros(2, np.uint32), np.arange(3, dtype=np.uint32))
    with self.assertRaisesRegex(ValueError, 'key pair'):
      audit.threefry(np.zeros(2, np.int32), np.arange(4, dtype=np.uint32))
    with self.assertRaisesRegex(ValueError, 'uint32-valued'):
      audit.fold_in(np.zeros(2, np.uint32), 2 ** 32)

  def test_deadline_and_cpu_import_do_not_create_backend_or_data(self):
    with mock.patch.object(audit,'DEADLINE',time.monotonic()-1),mock.patch.object(audit.subprocess,'Popen') as create:
      with self.assertRaisesRegex(ValueError,'deadline'):audit.command_output(['unused'])
      create.assert_not_called()
    result=subprocess.run([sys.executable,'-c',
        "import sys;import analysis.dl_nm1_cal_audit;assert not any(k in sys.modules for k in ('jax','dreamerv3.agent','mujoco'))"],
        cwd=ROOT,capture_output=True,text=True,timeout=30,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',MUJOCO_GL='disable'))
    self.assertEqual(result.returncode,0,result.stderr)

  def test_real_cli_bad_authority_cannot_emit_pass_or_import_failure(self):
    with tempfile.TemporaryDirectory() as temporary:
      root=Path(temporary);run=root/'run';run.mkdir();output=root/'audit.json'
      result=subprocess.run([sys.executable,str(Path(audit.__file__).resolve()),'--run',str(run),
          '--run-git','a'*40,'--expected-git','b'*40,'--launcher-terminal',str(root/'missing.json'),
          '--launcher-sha','0'*64,'--output',str(output)],cwd=ROOT,capture_output=True,text=True,timeout=30)
      self.assertNotEqual(result.returncode,0)
      self.assertNotIn('ModuleNotFoundError',result.stderr)
      receipt=json.loads(output.read_text())
      self.assertFalse(receipt['audit_integrity_passed']);self.assertEqual(receipt['status'],'integrity-rejected')
      self.assertNotIn('scientific_qualification',receipt)


def jax_fixture(output):
  dll = (os.add_dll_directory(os.environ['DL_DLL_DIRECTORY'])
      if os.name == 'nt' and os.environ.get('DL_DLL_DIRECTORY') else None)
  import jax
  import jax.numpy as jnp
  from jax._src import prng
  assert jax.__version__ == '0.4.33'
  assert jax.config.jax_default_prng_impl == 'threefry2x32'
  assert not jax.config.jax_threefry_partitionable and not jax.config.jax_enable_x64
  assert all(d.platform == 'cpu' for d in jax.devices())
  key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(0), 0x444C), 249493)
  priorities = jax.random.uniform(key, (1024,), dtype=jnp.float32)
  kl = (np.arange(1024, dtype=np.float32) % 29 / 13).reshape(16, 64)
  discrepancy = (np.arange(1024, dtype=np.float32) % 17 / 17).reshape(16, 64)
  mask = (np.arange(1024) % 5 != 0).reshape(16, 64)
  def real_groups(mask, kl, discrepancy):
    valid, kl, d = mask.reshape(-1), kl.reshape(-1), discrepancy.reshape(-1)
    db = jnp.searchsorted(jnp.array([.2, .6], jnp.float32), d, side='right')
    kb = jnp.searchsorted(jnp.array([1.], jnp.float32), kl, side='right')
    group = (((valid.astype(jnp.int32) * 2 + (kl > 1)) * 3 + db) * 2 + kb)
    return group.reshape(16, 64)
  from dreamerv3 import dt_latch
  weight = np.where(mask, np.linspace(.6, .99, 1024, dtype=np.float32).reshape(16, 64), 1)
  _, _, perm = jax.jit(lambda w, m, k, d: dt_latch.conditional_shuffle(
      w, m, k, d, 1., [.2, .6], [1.], 0, 249493))(weight, mask, kl, discrepancy)
  group = jax.jit(real_groups)(mask, kl, discrepancy)
  adversarial_key = np.array([0xFFFFFFFF, 0xFFFFFFFF], np.uint32)
  count = np.array([0, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF], np.uint32)
  adversarial_bits = prng.threefry_2x32(jnp.asarray(adversarial_key), jnp.asarray(count))
  np.savez(output, key=key, priorities=priorities, perm=perm, group=group,
      mask=mask, kl=kl, D=discrepancy, adversarial_key=adversarial_key,
      count=count, adversarial_bits=adversarial_bits)
  del dll


if __name__ == '__main__':
  if len(sys.argv) == 3 and sys.argv[1] == '--jax-fixture':
    jax_fixture(sys.argv[2])
  else:
    unittest.main()
