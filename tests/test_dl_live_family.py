"""Actual CPU live quadruped; policy/H1/oracle and RGB are explicit fakes."""
import copy
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import dl_live_family as f
from scripts import dl_live_policy as live
from scripts import dl_live_shadow as shadow


class FakePolicy:
  """Engineering double; not actual Dreamer policy, H1, public oracle or keys."""
  def initial(self, unit):
    return ({},dict(deter=[np.zeros(4,np.float32)],stoch=[np.zeros((3,4),np.float32)]),
        {},dict(action=[np.zeros(12,np.float32)]))

  def _inner(self,carry,obs,key):
    h=carry[1]['deter'][0]+np.float32(.1)+np.float32(carry[3]['action'][0].sum()*.01)
    qz=np.eye(4,dtype=np.float32)[:3].copy()
    pz=np.roll(qz,1,axis=1)
    qraw=np.linspace(-.4,.4,12,dtype=np.float32)+np.float32(obs['image'][0,0,0]*.0001)
    praw=-qraw.copy()
    q,p=copy.deepcopy(carry),copy.deepcopy(carry)
    for after,z,raw in ((q,qz,qraw),(p,pz,praw)):
      after[1]['deter'][0]=h.copy();after[1]['stoch'][0]=z.copy();after[3]['action'][0]=raw.copy()
    return dict(q_after=q,p_after=p,q_raw=qraw,p_raw=praw,h=h,qsample=qz,psample=pz,
        qlogits=qz.copy(),plogits=pz.copy(),root_key=key.copy(),latent_key=key[::-1].copy(),
        actor_key=(key^np.uint32(79)).copy(),before_carry_sha256=live.kernel.tree_digest(carry),
        observation_sha256=live.kernel.tree_digest(obs),original_nj_seed_calls=2)

  def _first(self,inner,arm):
    action,memory=arm
    after=copy.deepcopy(inner[memory+'_after']);raw=inner[action+'_raw'].copy()
    after[3]['action'][0]=raw.copy()
    proof=dict(root_key=inner['root_key'].copy(),latent_key=inner['latent_key'].copy(),
        actor_key=inner['actor_key'].copy(),h=inner['h'].copy(),
        selected_logit=inner[memory+'logits'].copy(),latent=inner[memory+'sample'].copy(),
        action_logit=inner[action+'logits'].copy(),action_latent=inner[action+'sample'].copy(),
        raw=raw.copy(),rng_consumption=np.asarray(2,np.int32),
        memory_source=np.asarray(memory=='p',np.int8),action_source=np.asarray(action=='p',np.int8))
    return after,raw,proof

  def normal_step(self,carry,obs,unit,step):
    return self._first(self._inner(carry,obs,live.policy_root_key(unit,step)),'qq')

  def current(self,carry,obs,unit):
    plan=live.unit_plan(unit)
    inner=self._inner(carry,obs,live.policy_root_key(unit,plan['t']))
    bundle=dict(version=live.VERSION,key_namespace=live.KEY_NAMESPACE,unit=unit,
        kernel_bundle=inner,Dmode=np.asarray(.2,np.float32),K=np.asarray(.3,np.float32),
        q_raw=inner['q_raw'].copy(),q_after=copy.deepcopy(inner['q_after']),
        before_carry_sha256=inner['before_carry_sha256'],
        observation_sha256=inner['observation_sha256'],r_t_f32=obs['reward'].copy())
    bundle['bundle_sha256']=live.kernel.full_digest(bundle)
    return bundle

  def public_oracle(self,carry,obs,unit,step,*,expected_after,expected_raw):
    return dict(unit=unit,step=step,public_q_raw_byte_exact=True,
        public_q_full_after_carry_byte_exact=True,engineering_fake_oracle=True)

  def seal_pretreatment(self,bundle,response,*,live_before_sha256,live_after_sha256,r_t_f64):
    assert live_before_sha256==live_after_sha256==response['source_snapshot_sha256']
    assert np.asarray(r_t_f64,np.float32).tobytes()==bundle['r_t_f32'].tobytes()
    seal=dict(Dmode=bundle['Dmode'].copy(),K=bundle['K'].copy(),e=np.asarray(.1,np.float32),
        r_t=np.asarray(r_t_f64).copy(),r_t_f32=bundle['r_t_f32'].copy())
    seal['pretreatment_sha256']=live.kernel.full_digest(seal)
    return seal

  def first(self,bundle,label,*,label_receipt):
    assert label==label_receipt['label']
    return self._first(bundle['kernel_bundle'],'qq' if label in ('qqA','qqB') else label)

  def assert_frozen(self,full=True): pass


def stub_rgb(self,height=64,width=64,camera_id=2,**kwargs):
  return np.full((height,width,3),int(round(self.data.time*1000))%256,np.uint8)


class LiveFamilyTest(unittest.TestCase):
  def setUp(self):
    from dm_control.mujoco.engine import Physics
    self.patch=mock.patch.object(Physics,'render',stub_rgb)
    self.patch.start();self.addCleanup(self.patch.stop)
    self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
    self.root=Path(self.temp.name)

  def adapter(self):
    from scripts.dl_feedback_physics import FeedbackEnvironment
    from scripts import dl_diagnostic as d
    value=FeedbackEnvironment(d.make_dmc('dmc_quadruped_walk',8192))
    self.addCleanup(value.close)
    return value

  def client(self):
    value=f.ShadowClient(lambda:None)
    self.addCleanup(value.close)
    return value

  def test_real_cpu_complete_unit_durable_seal_before_label_and_no_live_restore(self):
    adapter=self.adapter();client=self.client();path=self.root/'unit'
    reveal=live.reveal_label
    def observed_reveal(unit,sealhash):
      signals=f.read_json(path/'signals_sealed.json')
      self.assertEqual(signals['pretreatment_sha256'],sealhash)
      self.assertFalse((path/'assignment.json').exists())
      self.assertEqual(f.sha(path/'pretreatment.pkl'),signals['pretreatment_file']['sha256'])
      return reveal(unit,sealhash)
    runtime=[]
    with (mock.patch.object(adapter,'restore',side_effect=AssertionError('No live restore')),
        mock.patch.object(live,'reveal_label',side_effect=observed_reveal)):
      result=f.run_unit(FakePolicy(),adapter,client,7,path,lambda:None,
          runtime_check=lambda:runtime.append('actual reset happened'))
    record=pickle.loads((path/'trajectory.pkl').read_bytes())
    t=live.unit_plan(7)['t'];n=t+100
    self.assertEqual(result['unit_row']['label'],'qqA')
    self.assertEqual(runtime,['actual reset happened'])
    self.assertEqual(len(record['observations']),n+1)
    self.assertEqual(len(record['snapshots']),n+1)
    self.assertEqual(len(record['raw']),n)
    self.assertEqual(len(record['after_carries']),100)
    self.assertEqual(len(record['outcome_proofs']),100)
    self.assertEqual(result['oracle_steps'],[0,t,t+99])
    for step in range(n):
      self.assertEqual(record['keys'][step].tobytes(),live.policy_root_key(7,step).tobytes())
    self.assertEqual(record['seal']['r_t'].tobytes(),record['reward_f64'][t-1].tobytes())
    self.assertEqual(result['unit_row']['outcome']['reward_sum'],
        __import__('math').fsum(float(x) for x in record['reward_f64'][t:]))
    self.assertEqual(result['unit_row']['chronology'],dict(signals_sealed=21,assignment=22,outcome_complete=23))
    signals=f.read_json(path/'signals_sealed.json');assignment=f.read_json(path/'assignment.json')
    self.assertLess(signals['monotonic'],assignment['monotonic'])
    self.assertLess(assignment['monotonic'],result['monotonic'])
    for name,entry in result['artifacts'].items():
      self.assertEqual(f.sha(path/name),entry['sha256'])
      self.assertEqual((path/name).stat().st_size,entry['bytes'])
    response=pickle.loads((path/'shadow_response.pkl').read_bytes())
    self.assertEqual(record['reward_f64'][t].tobytes(),response['reward_f64'].tobytes())
    self.assertEqual(shadow.snapshot_sha256(record['snapshots'][t+1]),response['next_snapshot_sha256'])
    with self.assertRaises(ValueError): f.run_unit(FakePolicy(),adapter,client,7,path,lambda:None)

  def test_wire_tamper_and_mutated_policy_inputs_refused_before_assignment(self):
    actual=self.client()
    class BrokenWire:
      def exchange(self,request):
        receipt,body,response=actual.exchange(request)
        changed=pickle.loads(body);changed['raw_q'][0]+=np.float32(.1)
        return receipt,pickle.dumps(changed,protocol=5),response
    path=self.root/'wire'
    with self.assertRaisesRegex(ValueError,'wire bodies'):
      f.run_unit(FakePolicy(),self.adapter(),BrokenWire(),7,path,lambda:None)
    self.assertFalse((path/'assignment.json').exists())
    class Mutating(FakePolicy):
      def normal_step(self,carry,obs,unit,step):
        result=super().normal_step(carry,obs,unit,step)
        carry[3]['action'][0][0]=1
        return result
    path=self.root/'mutation'
    with self.assertRaisesRegex(ValueError,'mutated'):
      f.run_unit(Mutating(),self.adapter(),actual,7,path,lambda:None)
    self.assertFalse((path/'assignment.json').exists())

  def test_prefix_terminal_refused_and_late_publication_never_succeeds(self):
    adapter=self.adapter()
    with mock.patch.object(type(adapter.suite._task),'get_termination',return_value=0.):
      with self.assertRaisesRegex(ValueError,'Premature prefix'):
        f.run_unit(FakePolicy(),adapter,self.client(),7,self.root/'terminal',lambda:None)
    self.assertFalse((self.root/'terminal'/'complete.json').exists())
    def expired(): raise ValueError('global deadline')
    with self.assertRaisesRegex(ValueError,'deadline'):
      f.write_json(self.root/'late.json',{'success':True},expired)
    self.assertFalse((self.root/'late.json').exists())

  def test_sealed_file_tamper_stops_before_assignment(self):
    path=self.root/'seal';original=live.reveal_label
    def altered(unit,digest):
      target=path/'pretreatment.pkl'
      body=target.read_bytes();target.write_bytes(body[:-1]+bytes([body[-1]^1]))
      return original(unit,digest)
    with mock.patch.object(live,'reveal_label',side_effect=altered):
      with self.assertRaisesRegex(ValueError,'Sealed pretreatment artifacts changed'):
        f.run_unit(FakePolicy(),self.adapter(),self.client(),7,path,lambda:None)
    self.assertFalse((path/'assignment.json').exists())
    self.assertFalse((path/'complete.json').exists())

  def test_engineered_outcome_true_terminal_stops_without_fabricated_future(self):
    # Only termination is engineered; actual CPU integration/control/reward and
    # the independent ordinary-q shadow transition remain real. This does not
    # authenticate the unmodified quadruped task or its independent audit.
    adapter=self.adapter();task=type(adapter.suite._task)
    original=task.get_termination
    t=live.unit_plan(7)['t'];threshold=(t+3)*adapter.suite.control_timestep()
    def stop(self,physics):
      return 0. if physics.data.time >= threshold-1e-10 else original(self,physics)
    path=self.root/'short'
    with mock.patch.object(task,'get_termination',new=stop):
      result=f.run_unit(FakePolicy(),adapter,self.client(),7,path,lambda:None)
    record=pickle.loads((path/'trajectory.pkl').read_bytes())
    self.assertEqual(record['outcome_steps'],3)
    self.assertEqual(record['end_type'],'true_terminal')
    self.assertEqual(len(record['raw']),t+3)
    self.assertEqual(len(record['observations']),t+4)
    self.assertEqual(len(record['after_carries']),3)
    self.assertTrue(record['is_terminal'][-1])
    self.assertEqual(result['oracle_steps'],[0,t,t+2])

  def test_import_does_not_load_model_environment_or_change_environ(self):
    result=subprocess.run([sys.executable,'-c',
        "import os,sys; before=dict(os.environ); import scripts.dl_live_family; "
        "assert dict(os.environ)==before; assert 'jax' not in sys.modules; "
        "assert not any(x.startswith('dm_control') for x in sys.modules)"],
        cwd=ROOT,capture_output=True,text=True,timeout=30)
    self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__': unittest.main()
