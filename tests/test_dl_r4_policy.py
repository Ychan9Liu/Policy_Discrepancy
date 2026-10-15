"""DL-r4 tiny CPU policy contracts; never real N1M/task/GPU acceptance."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_r4_policy as r


class R4PolicyTest(unittest.TestCase):
  def test_import_has_no_jax_env_loader_or_device_side_effect(self):
    code = "import os,sys; before=dict(os.environ); import scripts.dl_r4_policy; assert before==dict(os.environ); assert 'jax' not in sys.modules; assert not any(x.startswith('dm_control') for x in sys.modules)"
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
        capture_output=True, text=True, timeout=30,
        env=dict(os.environ, PYTHONPATH=str(ROOT)))
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_named_roots_are_injective_labels_and_original_eval_formula(self):
    labels = [(0, 64, 0), (0, 64, 15), (0, 65, 0), (1, 64, 0), (31, 896, 15)]
    seeds = [r.branch_eval_seed(*x) for x in labels]
    self.assertEqual(len(set(seeds)), len(labels))
    self.assertFalse(set(seeds) & set(range(1024, 1056)))
    self.assertFalse(set(seeds) & set(r.SOURCE_SEEDS))
    # Exact range packing is one-to-one for the entire declared domain.
    allseeds = [r.branch_eval_seed(ep, pos, pair) for ep in range(32)
        for pos in range(64, 897) for pair in range(16)]
    self.assertEqual(len(allseeds), len(set(allseeds)))
    self.assertEqual(max(allseeds)-min(allseeds)+1, len(allseeds))
    values = {}
    for ep, pos, pair in labels:
      for offset in (0, 50, 99):
        key = r.branch_root_key(ep, pos, pair, offset)
        expected = np.random.default_rng([0, 0x4556414C,
            r.branch_eval_seed(ep, pos, pair), offset]).integers(
                0, np.iinfo(np.uint32).max, (2,), np.uint32)
        np.testing.assert_array_equal(key, expected)
        values[(ep,pos,pair,offset)] = key.tobytes()
    for labels, value in reversed(list(values.items())):
      self.assertEqual(r.branch_root_key(*labels).tobytes(), value)
    for ep in (0,31):
      self.assertEqual(r.source_eval_seed(ep),4096+ep)
      expected=np.random.default_rng([0,0x4556414C,4096+ep,999]).integers(
          0,np.iinfo(np.uint32).max,(2,),np.uint32)
      np.testing.assert_array_equal(r.source_root_key(ep,999),expected)
    for labels in ((-1,64,0,0),(32,64,0,0),(0,63,0,0),(0,897,0,0),
        (0,64,16,0),(0,64,0,100),(True,64,0,0),(0,64,0,np.int64(0))):
      with self.assertRaises(ValueError):r.branch_root_key(*labels)
    for labels in ((32,0),(0,-1),(0,1000),(True,0)):
      with self.assertRaises(ValueError):r.source_root_key(*labels)

  def test_observation_rejects_last_dtype_reset_reward_and_omitted_fields(self):
    def obs():
      return dict(image=np.zeros((64,64,3),np.uint8),reward=np.float32(0),
          is_first=np.bool_(False),is_last=np.bool_(False),is_terminal=np.bool_(False))
    for change in ('last','terminal','reward','image','missing','resetreward'):
      row=obs()
      if change=='last':row['is_last']=np.bool_(True)
      if change=='terminal':row['is_terminal']=np.bool_(True)
      if change=='reward':row['reward']=np.float64(0)
      if change=='image':row['image']=np.zeros((64,64,3),np.float32)
      if change=='missing':row.pop('image')
      if change=='resetreward':row.update(is_first=np.bool_(True),reward=np.float32(1))
      with self.subTest(change=change),self.assertRaises(ValueError):r.observation(row)
    good=obs();after=r.observation(good);after['image'][...]=1
    self.assertFalse(good['image'].any())

  def test_actual_tiny_bf16_policy_three_arms_h1_oracle_and_negative_guards(self):
    result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--fixture'],
        cwd=ROOT,capture_output=True,text=True,timeout=240,
        env=dict(os.environ,JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT)))
    self.assertEqual(result.returncode,0,result.stdout[-6000:]+result.stderr[-11000:])
    facts=json.loads(result.stdout.strip().splitlines()[-1])
    for name,value in facts.items():self.assertTrue(value,name)


def fixture():
  dll=(os.add_dll_directory(os.environ['DL_DLL_DIRECTORY'])
      if os.name=='nt' and os.environ.get('DL_DLL_DIRECTORY') else None)
  import elements
  import jax
  import jax.numpy as jnp
  import ninjax as nj
  from embodied.jax import internal,nets as nn
  from ruamel.yaml import YAML
  from dreamerv3 import dt_latch
  from scripts.dl_predictive_mc import agent_from_spaces
  presets=YAML(typ='safe').load((ROOT/'dreamerv3/configs.yaml').read_text(encoding='utf-8'))
  config=elements.Config(presets['defaults']).update(presets['debug'])
  config=elements.Config({**config.flat,**{'jax.platform':'cpu','jax.compute_dtype':'bfloat16',
      'jax.precompile':False,'jax.profiler':False,'jax.enable_policy':True,'jax.transfer_guard':True,
      'batch_size':2,'batch_length':3,'replay_context':1,'agent.imag_length':2,'agent.imag_last':2,
      'agent.rep_probe.mode':'off','agent.dt_latch.mode':'off'}}).update(
          logdir='/tmp/dl-r4-policy-fixture',seed=0)
  agent=agent_from_spaces(config)
  checkpoint=agent.save()
  before_load=r.full_digest(checkpoint)
  agent.load(checkpoint)
  assert r.full_digest(agent.save())==before_load
  adapter=r.R4Policy(agent)
  saved=r.full_digest(agent.save())
  saved_counters=r.counters(agent)
  rng=np.random.default_rng(91)
  def obs(first=False):
    return dict(image=rng.integers(0,256,(64,64,3),np.uint8),reward=np.float32(0),
        is_first=np.bool_(first),is_last=np.bool_(False),is_terminal=np.bool_(False))
  initial_obs=obs(True);current_obs=obs();other_obs=obs()
  initial=adapter.initial(r.source_root_key(0,0))
  b0=adapter.current(initial,initial_obs,r.source_root_key(0,0))
  oracle0=adapter.public_oracle(initial,initial_obs,r.source_eval_seed(0),0,bundle=b0)
  assert oracle0['public_q_full_after_carry_byte_exact']
  carry=copy.deepcopy(b0['q_after'])
  original=r.tree_digest(carry)
  # A deterministic finite construction locates a nondegenerate test case.
  # Paired categorical draws can legitimately coincide; that is not a failure.
  for fixture_pair in range(16):
    key=r.branch_root_key(0,64,fixture_pair,0)
    bundle=adapter.current(carry,current_obs,key)
    if (bundle['qsample'].tobytes()!=bundle['psample'].tobytes() and
        bundle['q_raw'].tobytes()!=bundle['p_raw'].tobytes()):
      break
  else:raise AssertionError('Tiny fixture has no nondegenerate action/memory case')
  assert r.tree_digest(carry)==original
  actual_oracle=adapter.public_oracle(carry,current_obs,r.branch_eval_seed(0,64,fixture_pair),0,bundle=bundle)
  assert actual_oracle['public_q_raw_byte_exact']
  assert bundle['normal_policy_calls']==1 and bundle['original_nj_seed_calls']==2
  assert bundle['extra_observe_calls']==0 and bundle['derived_nj_seed_calls']==0
  qq,aq,qproof=adapter.first(bundle,'qq')
  pq,ap,pqproof=adapter.first(bundle,'pq')
  pp,app,pproof=adapter.first(bundle,'pp')
  assert set(qproof)==set(r.PROOF_KEYS)
  assert r.tree_digest(qq)==r.tree_digest(bundle['q_after'])
  assert r.tree_digest(pp)==r.tree_digest(bundle['p_after'])
  assert r.tree_digest(pq[1])==r.tree_digest(qq[1])
  assert r.tree_digest(pq[1])!=r.tree_digest(pp[1])
  assert ap.tobytes()==app.tobytes()==bundle['p_raw'].tobytes()
  assert pq[3]['action'][0].tobytes()==ap.tobytes()
  assert pqproof['latent'].tobytes()==bundle['qsample'].tobytes()
  assert pqproof['action_latent'].tobytes()==bundle['psample'].tobytes()
  assert int(pqproof['memory_source'])==0 and int(pqproof['action_source'])==1
  # Explicit actual subkeys, not inferred from branch call order.
  expectedkeys=r.host_copy(jax.jit(lambda k:jax.random.split(k,16)[1:3])(
      internal.device_put(key,agent.train_mirrored)))
  np.testing.assert_array_equal(bundle['latent_key'],expectedkeys[0])
  np.testing.assert_array_equal(bundle['actor_key'],expectedkeys[1])
  def reference_sample(logit,h,latent_key,actor_key):
    z=nn.cast(agent.model.dyn._dist(logit[None]).sample(latent_key))
    policy=agent.model.pol(agent.model.feat2tensor(dict(deter=h[None],stoch=z)),bdims=1)
    return z,policy['action'].sample(actor_key)
  reference=jax.jit(nj.pure(reference_sample))
  for logit_name,z_name,a_name in (('qlogits','qsample','q_raw'),('plogits','psample','p_raw')):
    args=internal.device_put(tuple(bundle[k] for k in (logit_name,'h','latent_key','actor_key')),
        agent.train_mirrored)
    _,(z,a)=reference(agent.params,*args,seed=internal.device_put(key,agent.train_mirrored),
        create=False,modify=False)
    assert r.host_copy(z)[0].tobytes()==bundle[z_name].tobytes()
    assert r.host_copy(a)[0].tobytes()==bundle[a_name].tobytes()
  again=adapter.current(carry,current_obs,key)
  assert again['bundle_sha256']==bundle['bundle_sha256']
  different=adapter.current(carry,other_obs,key)
  assert different['h'].tobytes()==bundle['h'].tobytes()
  assert different['plogits'].tobytes()==bundle['plogits'].tobytes()
  assert different['psample'].tobytes()==bundle['psample'].tobytes()
  assert different['p_raw'].tobytes()==bundle['p_raw'].tobytes()
  assert different['qlogits'].tobytes()!=bundle['qlogits'].tobytes()
  nextkey=r.branch_root_key(0,64,fixture_pair,1)
  after1,_,proof1=adapter.normal_step(pq,current_obs,nextkey)
  after2,_,proof2=adapter.normal_step(pq,other_obs,nextkey)
  assert proof1['selected_logit'].tobytes()!=proof2['selected_logit'].tobytes()
  assert int(proof1['memory_source'])==int(proof1['action_source'])==0
  assert after1[1]['deter'][0].tobytes()!=qq[1]['deter'][0].tobytes()
  # Correct vs wrong prevraw influences the actual next recurrent transition.
  wrong=copy.deepcopy(pq);wrong[3]['action'][0]=aq.copy()
  _,_,wrongproof=adapter.normal_step(wrong,current_obs,nextkey)
  assert wrongproof['h'].tobytes()!=proof1['h'].tobytes()
  # H1 uses common actual raw and NEXT reward, original deterministic mode API.
  reward=np.float64(.173)
  h1=adapter.score_h1(bundle,aq,reward)
  def reference_h1(h,qm,pm,raw,reward):
    q,p=dt_latch.one_step(agent.model,h[None],qm[None],pm[None],{'action':raw[None]})
    signals=dt_latch.reward_support(q,p,reward[None],jnp.ones((1,),jnp.bool_),
        config.agent.dt_latch.kl_tolerance)
    return signals,q.logits,p.logits,q.bins
  ref=jax.jit(nj.pure(reference_h1))
  args=internal.device_put((bundle['h'],bundle['qmode'],bundle['pmode'],aq,np.float32(reward)),
      agent.train_mirrored)
  _,(signals,ql,pl,bins)=ref(agent.params,*args,seed=internal.device_put(key,agent.train_mirrored),
      create=False,modify=False)
  signals,ql,pl,bins=r.host_copy((signals,ql,pl,bins))
  for name,value in signals.items():assert h1[name].tobytes()==value[0].tobytes(),name
  assert h1['raw_logits_q'].tobytes()==ql[0].tobytes()
  assert h1['raw_logits_p'].tobytes()==pl[0].tobytes()
  assert h1['bins'].tobytes()==bins.tobytes()
  assert h1['reward_f32'].tobytes()==np.asarray(reward,np.float32).tobytes()
  assert not adapter.score_h1(bundle,aq,np.float64(np.nan),valid=False)['e']
  for invalid in ('badarm','alterbundle','wrongroot','rawdtype','rewarddtype','badvalid'):
    try:
      if invalid=='badarm':adapter.first(bundle,'qp')
      elif invalid=='alterbundle':
        altered=copy.deepcopy(bundle);altered['q_raw'][0]+=1;adapter.first(altered,'qq')
      elif invalid=='wrongroot':adapter.public_oracle(carry,current_obs,4096,0,bundle=bundle)
      elif invalid=='rawdtype':adapter.score_h1(bundle,aq.astype(np.float64),reward)
      elif invalid=='rewarddtype':adapter.score_h1(bundle,aq,np.float32(reward))
      else:adapter.score_h1(bundle,aq,reward,valid=np.bool_(True))
    except ValueError:pass
    else:raise AssertionError('Invalid contract passed: '+invalid)
  # A normalizer value alteration with the same parameter dict is rejected.
  with agent.n_eval_actions.lock:agent.n_eval_actions.value+=1
  try:
    try:adapter.normal_step(pq,current_obs,nextkey)
    except AssertionError:pass
    else:raise AssertionError('Counter mutation accepted')
  finally:
    with agent.n_eval_actions.lock:agent.n_eval_actions.value-=1
  original_policy=agent.model.policy
  original_core=agent.model.dyn._core
  original_logit=agent.model.dyn._logit
  original_seed=nj.seed
  def writes(*args,**kwargs):
    name=next(iter(nj.context()));nj.context()[name]=nj.context()[name]
    return original_policy(*args,**kwargs)
  def creates(*args,**kwargs):
    nj.context()['r4_forbidden_created_state']=jnp.int32(0)
    return original_policy(*args,**kwargs)
  def extra(*args,**kwargs):
    result=original_policy(*args,**kwargs);nj.seed();return result
  for replacement in (writes,creates,extra):
    agent.model.policy=replacement
    try:
      rejecting=r.R4Policy(agent)
      try:rejecting.normal_step(pq,current_obs,nextkey)
      except (AssertionError,RuntimeError):pass
      else:raise AssertionError('Forbidden mutation/random consumption accepted')
      assert agent.model.dyn._core==original_core
      assert agent.model.dyn._logit==original_logit
      assert nj.seed is original_seed
    finally:agent.model.policy=original_policy
  assert r.full_digest(agent.save())==saved and r.counters(agent)==saved_counters
  adapter.assert_frozen(full=True)
  # Equal actual q/p distributions collapse all three intervention arms.
  def equal_heads(params):
    return {name:jnp.zeros_like(value) if name.startswith(('dyn/obslogit/','dyn/priorlogit/'))
        else value for name,value in params.items()}
  agent.params=jax.jit(equal_heads)(agent.params)
  agent.sync_policy()
  equal=r.R4Policy(agent)
  eb=equal.current(carry,current_obs,key)
  results=[equal.first(eb,arm) for arm in ('qq','pq','pp')]
  assert eb['qlogits'].tobytes()==eb['plogits'].tobytes()
  assert all(r.tree_digest(x[0])==r.tree_digest(results[0][0]) and
      x[1].tobytes()==results[0][1].tobytes() for x in results)
  assert float(eb['K'])==float(eb['D'])==float(eb['sample_D'])==0
  assert float(equal.score_h1(eb,eb['q_raw'],reward)['e'])==0
  print(json.dumps(dict(actual_standard_full_checkpoint_load=True,
      actual_public_source_and_branch_qq_exact=True, actual_qp_samples_and_subkeys_exact=True,
      current_one_native_observation_two_nj_seeds=True,three_action_memory_arms_exact=True,
      pq_actual_prevraw_reaches_next_recurrent_core=True,future_normal_q_uses_own_rgb=True,
      prior_current_rgb_independent=True,branch_order_and_repeat_invariant=True,
      original_h1_common_action_next_reward_exact=True,full_state_and_counters_unchanged=True,
      readonly_writes_creation_extra_rng_and_bad_contracts_rejected=True,
      hooks_restored_on_trace_failure=True,q_equals_p_all_three_arms_exact=True,
      fixture_only_not_real_N1M_task_or_GPU_acceptance=True)))
  if dll is not None:dll.close()


if __name__=='__main__':
  if sys.argv[1:]==['--fixture']:fixture()
  else:unittest.main()
