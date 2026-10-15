"""Actual live producer/official CPU physics; NN, H1, public oracle and RGB fake.

The complete512/Git/resource entry is never certified by this single-unit
fixture. Its BF16 tensors use the actual frozen N1M shapes without a model.
"""
import copy
import contextlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('MUJOCO_GL','disable')
import numpy as np
import ml_dtypes

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis import dl_live_audit as a
from scripts import dl_live_family as producer
from scripts import dl_live_policy as live
from tests.test_dl_live_family import FakePolicy,stub_rgb


class R2MetadataTests(unittest.TestCase):
  def test_independent_engineering_binding_rejects_old_budget_and_missing_norm(self):
    source={a.ENGINEERING_NORM:dict(bytes=20,sha256='d'*64)}
    budget=dict(engineering_version='DL-r5-engineering-r2',seconds=12600,
        physical_gpu=5,gpu_uuid='GPU-79704918-e096-92b4-252c-e3244497a8b3',
        source_files=source,engineering_norm=dict(path=a.ENGINEERING_NORM,**source[a.ENGINEERING_NORM]))
    a.engineering_identity(budget,source)
    self.assertEqual(a.SECONDS,1800)
    for field,value in (('seconds',7200),('physical_gpu',4),('gpu_uuid','old GPU4 UUID'),
        ('engineering_version','DL-r5-engineering-r1'),('engineering_norm',{}),('source_files',{})):
      with self.subTest(field=field),self.assertRaisesRegex(ValueError,'engineering'):
        a.engineering_identity(dict(budget,**{field:value}),source)
    for field in ('engineering_version','engineering_norm'):
      missing=dict(budget);del missing[field]
      with self.subTest(missing=field),self.assertRaisesRegex(ValueError,'engineering'):
        a.engineering_identity(missing,source)
    with self.assertRaisesRegex(ValueError,'norm Git blob'):
      a.engineering_identity(budget,{})
    changed=copy.deepcopy(source);changed[a.ENGINEERING_NORM]['sha256']='e'*64
    with self.assertRaisesRegex(ValueError,'norm/source binding'):
      a.engineering_identity(budget,changed)


class KernelDouble(FakePolicy):
  """Actual tensor schema and hash contract, never an executed Dreamer model."""
  def initial(self,key):
    return ({},dict(deter=[np.zeros(4096,ml_dtypes.bfloat16)],
        stoch=[np.zeros((32,32),ml_dtypes.bfloat16)]),{},dict(action=[np.zeros(12,np.float32)]))

  def _inner(self,carry,obs,key):
    h=np.asarray(carry[1]['deter'][0].astype(np.float32)+.1,ml_dtypes.bfloat16)
    qz=np.eye(32,dtype=ml_dtypes.bfloat16);pz=np.roll(qz,1,axis=1)
    raw=np.linspace(-.4,.4,12,dtype=np.float32)+np.float32(obs['image'][0,0,0]*.0001)
    q,p=copy.deepcopy(carry),copy.deepcopy(carry)
    for after,z,action in ((q,qz,raw),(p,pz,-raw)):
      after[1]['deter'][0]=h.copy();after[1]['stoch'][0]=z.copy();after[3]['action'][0]=action.copy()
    value=dict(version='DL-r4-utility-r1',key_namespace='DL_R4_U1',
        current_full_state_sha256='a'*64,q_after=q,p_after=p,q_raw=raw,p_raw=-raw,
        h=h,qlogits=qz.copy(),plogits=pz.copy(),qsample=qz,psample=pz,qmode=qz.copy(),pmode=pz.copy(),
        root_key=key.copy(),latent_key=key[::-1].copy(),actor_key=(key^np.uint32(79)).copy(),
        before_carry_sha256=live.kernel.tree_digest(carry),observation_sha256=live.kernel.tree_digest(obs),
        D=np.asarray(.2,np.float32),Dmode=np.asarray(.2,np.float32),K=np.asarray(.3,np.float32),
        sample_D=np.asarray(.4,np.float32),original_nj_seed_calls=2,normal_policy_calls=1,
        derived_nj_seed_calls=0,extra_observe_calls=0)
    value['bundle_sha256']=live.kernel.full_digest(value)
    return value

  def current(self,carry,obs,key): return self._inner(carry,obs,key)
  def normal_step(self,carry,obs,key): return self._first(self._inner(carry,obs,key),'qq')
  def first(self,bundle,arm): return self._first(bundle,arm)
  def _validate_bundle(self,bundle):
    assert live.kernel.full_digest({k:v for k,v in bundle.items() if k != 'bundle_sha256'}) == bundle['bundle_sha256']
  def score_h1(self,bundle,raw,reward):
    value={key:bundle[key].copy() for key in ('D','Dmode','K','sample_D')}
    value.update(e=np.asarray(.1,np.float32),recorded_raw=raw.copy(),reward_f64=reward.copy(),
        reward_f32=np.asarray(reward,np.float32),source_bundle_sha256=bundle['bundle_sha256'],
        valid=np.asarray(True,np.bool_),mode_probe=True,native_policy_sample_is_distinct=True,
        nj_seed_calls=0,actual_policy_calls=0)
    return value


class PolicyDouble(live.LivePolicy):
  def __init__(self):
    self.kernel=KernelDouble();self._issued=set();self._sealed={};self._oracles=0
  def public_oracle(self,carry,obs,unit,step,*,expected_after,expected_raw):
    return dict(version=live.VERSION,key_namespace=live.KEY_NAMESPACE,unit=unit,step=step,
        eval_seed=live.unit_plan(unit)['eval_seed'],root_key=live.policy_root_key(unit,step),public_policy_calls=1,
        public_q_raw_byte_exact=True,public_q_full_after_carry_byte_exact=True,original_counters_restored=True,
        before_carry_sha256=live.kernel.tree_digest(carry),observation_sha256=live.kernel.tree_digest(obs),
        after_carry_sha256=live.kernel.tree_digest(expected_after))


class LiveAuditTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    from dm_control.mujoco.engine import Physics
    from scripts.dl_feedback_physics import FeedbackEnvironment
    from scripts import dl_diagnostic as diagnostic
    cls.temp=tempfile.TemporaryDirectory();cls.path=Path(cls.temp.name)/'unit'
    cls.rgb=mock.patch.object(Physics,'render',stub_rgb);cls.rgb.start()
    adapter=FeedbackEnvironment(diagnostic.make_dmc('dmc_quadruped_walk',8192))
    client=producer.ShadowClient(lambda:None)
    try:
      # Python3.12 Windows monotonic() can have15ms ticks; inject only the
      # actual higher-resolution Windows clock for this fixture. Linux uses
      # the unmodified production clock and the audit's strict < stays fixed.
      clock=(mock.patch.object(producer.time,'monotonic',producer.time.perf_counter)
          if os.name == 'nt' else contextlib.nullcontext())
      with clock:
        cls.complete=producer.run_unit(PolicyDouble(),adapter,client,7,cls.path,lambda:None)
    finally:
      client.close();adapter.close();cls.rgb.stop()
    cls.record,cls.request,cls.response,cls.seal=(pickle.loads((cls.path/name).read_bytes()) for name in
        ('trajectory.pkl','shadow_request.pkl','shadow_response.pkl','pretreatment.pkl'))
    cls.signals,cls.assignment=(a.read_json(cls.path/name) for name in ('signals_sealed.json','assignment.json'))

  @classmethod
  def tearDownClass(cls): cls.temp.cleanup()

  def validate(self,record=None,complete=None,request=None,response=None,seal=None,signals=None,assignment=None):
    return a.validate_record(self.record if record is None else record,
        self.complete if complete is None else complete,self.request if request is None else request,
        self.response if response is None else response,self.seal if seal is None else seal,
        self.signals if signals is None else signals,self.assignment if assignment is None else assignment,7,'a'*64)

  def test_actual_producer_unit_and_official_full_source_outcome_shadow_replay(self):
    row=self.validate()
    counts=a.replay(self.record,self.request,self.response)
    self.assertEqual(row,self.complete['unit_row'])
    self.assertEqual(counts,dict(source_steps=live.unit_plan(7)['t'],outcome_steps=100,
        shadow_steps=1,terminal_absorbing_steps=0))
    # Independently implemented typed/hash protocols agree at real boundaries.
    for value in (self.record['initial_carry'],self.record['bundle'],self.seal):
      self.assertEqual(a.full_digest(value),live.kernel.full_digest(value))
    self.assertEqual(a.snapshot_hash(self.request['actual_full_snapshot']),self.request['source_snapshot_sha256'])

  def test_actual_controls_and_sub_f32_f64_reward_tamper_refused(self):
    record=copy.deepcopy(self.record);record['applied'][0,0]=np.nextafter(record['applied'][0,0],np.inf)
    with self.assertRaisesRegex(ValueError,'mapping'): a.replay(record,self.request,self.response)
    record=copy.deepcopy(self.record);old=record['reward_f64'][0]
    record['reward_f64'][0]=np.nextafter(old,np.inf)
    self.assertEqual(np.float32(old).tobytes(),np.float32(record['reward_f64'][0]).tobytes())
    with self.assertRaisesRegex(ValueError,'F64 reward'): a.replay(record,self.request,self.response)
    response=copy.deepcopy(self.response);response['reward_f64']=np.nextafter(response['reward_f64'],np.inf)
    with self.assertRaisesRegex(ValueError,'F64 reward'): a.replay(self.record,self.request,response)

  def test_key_carry_first_assembly_and_ordinary_prefix_terminal_refused(self):
    record=copy.deepcopy(self.record);record['keys'][2,0]^=np.uint32(1)
    with self.assertRaisesRegex(ValueError,'policy root'): self.validate(record=record)
    record=copy.deepcopy(self.record);record['pre_carry_sha256'][3]='b'*64
    with self.assertRaisesRegex(ValueError,'chain discontinuity'): self.validate(record=record)
    record=copy.deepcopy(self.record);record['outcome_proofs'][0]['memory_source']=np.asarray(1,np.int8)
    with self.assertRaisesRegex(ValueError,'memory/NJ'): self.validate(record=record)
    record=copy.deepcopy(self.record);record['observations'][2]['is_last']=np.asarray(True,np.bool_)
    with self.assertRaisesRegex(ValueError,'reset mismatch|Premature prefix'): self.validate(record=record)

  def test_seal_assignment_chronology_and_source_baseline_not_shadow_reward(self):
    assignment=copy.deepcopy(self.assignment);assignment['monotonic']=self.signals['monotonic']
    with self.assertRaisesRegex(ValueError,'ordering'): self.validate(assignment=assignment)
    assignment=copy.deepcopy(self.assignment);assignment['label_receipt']['label']='pp'
    with self.assertRaisesRegex(ValueError,'label hash'): self.validate(assignment=assignment)
    seal=copy.deepcopy(self.seal);seal['r_t']=self.response['reward_f64'].copy()
    seal['pretreatment_sha256']=a.full_digest({k:v for k,v in seal.items() if k != 'pretreatment_sha256'})
    record=copy.deepcopy(self.record);record['seal']=copy.deepcopy(seal)
    with self.assertRaisesRegex(ValueError,'preceding F64'): self.validate(record=record,seal=seal)
    complete=copy.deepcopy(self.complete);complete['unit_row']['outcome']['reward_sum']+=1e-9
    with self.assertRaisesRegex(ValueError,'Compact statistics'): self.validate(complete=complete)

  def test_wire_hash_inventory_decode_and_partial512_fail_closed(self):
    response=copy.deepcopy(self.response);response['raw_q'][0]+=np.float32(.1)
    with self.assertRaisesRegex(ValueError,'wire object hashes'): self.validate(response=response)
    stock=a.inventory(self.path)
    auth=dict(root=self.path,inventory=stock)
    a.trusted_pickle(auth,'pretreatment.pkl')
    path=self.path/'temporary.pkl';path.write_bytes(b'not trusted pickle')
    try:
      auth['inventory']['files']['temporary.pkl']=dict(path='temporary.pkl',bytes=18,sha256='0'*64)
      with mock.patch.object(a.pickle,'loads',side_effect=AssertionError('decode must not run')):
        with self.assertRaisesRegex(ValueError,'before trusted decode'): a.trusted_pickle(auth,'temporary.pkl')
    finally: path.unlink()
    with self.assertRaisesRegex(ValueError,'Exactly32|Exactly 32'):
      a.statistics.validate_units([self.complete['unit_row']])
    with self.assertRaisesRegex(ValueError,'Duplicate JSON'): a.decode('{"x":1,"x":2}')

  def test_entry_deadline_and_import_never_loads_jax_or_renderer(self):
    with mock.patch.object(a,'DEADLINE',a.time.monotonic()-1):
      with self.assertRaisesRegex(ValueError,'deadline'): a.check()
    result=subprocess.run([sys.executable,'-c',
        "import sys,os; import analysis.dl_live_audit; assert 'jax' not in sys.modules; "
        "assert not any(x.startswith('dm_control') for x in sys.modules); "
        "assert os.environ['CUDA_VISIBLE_DEVICES']==''; assert os.environ['MUJOCO_GL']=='disable'"],
        cwd=ROOT,capture_output=True,text=True,timeout=30)
    self.assertEqual(result.returncode,0,result.stderr)


if __name__ == '__main__': unittest.main()
