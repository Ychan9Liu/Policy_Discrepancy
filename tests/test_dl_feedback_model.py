"""Actual tiny BF16 policy properties, not real source or CUDA acceptance."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import dl_feedback_model as f


class FeedbackModelTest(unittest.TestCase):
  def test_named_root_keys_have_no_call_order_or_branch_state(self):
    requested = [(9,1,0,1),(9,1,0,2),(9,1,1,1),(14,1,0,1),(9,300,0,300),(31,900,3,999)]
    values = {x:f.named_root_key(*x) for x in requested}
    for x in reversed(requested):
      np.testing.assert_array_equal(values[x],f.named_root_key(*x))
      self.assertEqual(values[x].dtype,np.uint32)
      self.assertEqual(values[x].shape,(2,))
    self.assertEqual(len({x.tobytes() for x in values.values()}),len(requested))
    for args in ((9,1,4,1),(9,1,0,101),(9,0,0,1),(9,1,0,False),
        (9,300,0,299),(9,300,0,400),(9,901,0,901),(31,900,3,1000)):
      with self.assertRaises(ValueError):
        f.named_root_key(*args)

  def test_actual_tiny_bfloat16_production_policy_and_omission(self):
    child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--fixture'],
        cwd=ROOT,capture_output=True,text=True,timeout=180,
        env=dict(os.environ,JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT)))
    self.assertEqual(child.returncode,0,child.stdout[-4500:]+child.stderr[-6500:])
    facts=json.loads(child.stdout.strip().splitlines()[-1])
    for key in ('clean_actual_public_exact','q_equals_p_exact','prior_rgb_independent',
        'shared_h_exact','future_normal_posterior_uses_own_rgb','private_keys_exact',
        'call_order_invariant','bf16_host_carry_exact','strict_transfer_guard',
        'state_and_counters_unchanged','state_writes_rejected','actual_distribution_samples_exact'):
      self.assertTrue(facts[key],key)


def fixture():
  dll=(os.add_dll_directory(os.environ['DL_DLL_DIRECTORY'])
      if os.name=='nt' and os.environ.get('DL_DLL_DIRECTORY') else None)
  import elements
  import jax
  import jax.numpy as jnp
  import ninjax as nj
  from embodied.jax import internal
  from ruamel.yaml import YAML
  from scripts.dl_predictive_mc import agent_from_spaces
  presets=YAML(typ='safe').load((ROOT/'dreamerv3/configs.yaml').read_text(encoding='utf-8'))
  config=elements.Config(presets['defaults']).update(presets['debug'])
  config=elements.Config({**config.flat,**{'jax.platform':'cpu','jax.compute_dtype':'bfloat16',
      'jax.precompile':False,'jax.profiler':False,'jax.enable_policy':True,'jax.transfer_guard':True,
      'batch_size':2,'batch_length':3,'replay_context':0,'agent.imag_length':2,'agent.imag_last':2,
      'agent.rep_probe.mode':'off','agent.dt_latch.mode':'off'}}).update(logdir='/tmp/dl-feedback-model',seed=7)
  agent=agent_from_spaces(config)
  rng=np.random.default_rng(31)
  def obs(image,first=False):
    return dict(image=image,reward=np.float32(0),is_first=np.bool_(first),
        is_last=np.bool_(False),is_terminal=np.bool_(False))
  o0=obs(rng.integers(0,256,(64,64,3),np.uint8),True)
  o1=obs(rng.integers(0,256,(64,64,3),np.uint8))
  o2=obs(rng.integers(0,256,(64,64,3),np.uint8))
  agent.set_eval_seed(123)
  carry=agent.init_policy(1,mode='eval')
  carry,_,_=agent.policy(carry,{k:v[None] for k,v in o0.items()},mode='eval')
  before=f.host_copy(carry)
  public_after,expected_raw,_=agent.policy(carry,{k:v[None] for k,v in o1.items()},mode='eval')
  public_after=f.host_copy(public_after)
  record=dict(carry=before,carry_sha256=f.tree_digest(before),after_carry_sha256=f.tree_digest(public_after),
      eval_seed=123,eval_counter=1)
  adapter=f.FeedbackPolicy(agent)
  saved_counters=f.counters(agent)
  saved_params=f.tree_digest(f.host_copy(agent.params))
  receipt=adapter.oracle_source(record,o1,np.asarray(expected_raw['action'][0]))
  assert receipt['adapter_after_carry_byte_hash_exact']
  key=f.named_root_key(9,1,0,1)
  q,qa,qproof=adapter.step(before,o1,key,False)
  p,pa,pproof=adapter.step(before,o1,key,True)
  pr,par,prproof=adapter.step(before,o2,key,True)
  assert f.tree_digest(p)==f.tree_digest(pr) and pa.tobytes()==par.tobytes()
  assert qproof['h'].tobytes()==pproof['h'].tobytes()
  assert pproof['selected_logit'].tobytes()==prproof['selected_logit'].tobytes()
  # Independent expected NJ reserve ordering, plus actual distribution APIs.
  dk=internal.device_put(key,agent.policy_mirrored)
  njkeys=jax.jit(lambda k:jax.random.split(k,16)[1:3])(dk)
  expectedkeys=np.asarray(jax.device_get(njkeys))
  assert np.array_equal(qproof['latent_key'],expectedkeys[0])
  assert np.array_equal(qproof['actor_key'],expectedkeys[1])
  assert np.array_equal(qproof['latent_key'],pproof['latent_key'])
  assert np.array_equal(qproof['actor_key'],pproof['actor_key'])
  def sample_reference(logits,h,latent_key,actor_key):
    from embodied.jax import nets as nn
    z=nn.cast(agent.model.dyn._dist(logits[None]).sample(latent_key))
    feature=agent.model.feat2tensor(dict(deter=h[None],stoch=z))
    return z,agent.model.pol(feature,bdims=1)['action'].sample(actor_key)
  reference=jax.jit(nj.pure(sample_reference))
  for proof in (qproof,pproof):
    inputs=[internal.device_put(proof[k],agent.policy_mirrored)
        for k in ('selected_logit','h','latent_key','actor_key')]
    _,(z,a)=reference(agent.policy_params,*inputs,seed=dk,create=False,modify=False)
    assert np.asarray(jax.device_get(z))[0].tobytes()==proof['latent'].tobytes()
    assert np.asarray(jax.device_get(a))[0].tobytes()==proof['raw'].tobytes()
  # After initial omission, the SAME normal path consumes own current RGB.
  normal1,_,n1=adapter.step(p,o1,f.named_root_key(9,1,0,2),False)
  normal2,_,n2=adapter.step(p,o2,f.named_root_key(9,1,0,2),False)
  assert n1['selected_logit'].tobytes()!=n2['selected_logit'].tobytes()
  q_again,qa_again,qp_again=adapter.step(before,o1,key,False)
  assert f.tree_digest(q_again)==f.tree_digest(q) and qa_again.tobytes()==qa.tobytes()
  assert all(qproof[k].tobytes()==qp_again[k].tobytes() for k in qproof)
  assert all(str(x.dtype)=='bfloat16' for x in jax.tree.leaves(q[1]))
  assert f.counters(agent)==saved_counters and f.tree_digest(f.host_copy(agent.params))==saved_params
  # A state write cannot become an ignored scratch update.
  original_policy=agent.model.policy
  def writes(c,o,mode='eval'):
    name=next(iter(nj.context()))
    nj.context()[name]=nj.context()[name]
    return original_policy(c,o,mode=mode)
  agent.model.policy=writes
  try:
    rejecting=f.FeedbackPolicy(agent)
    try: rejecting.step(before,o1,key,False)
    except AssertionError as error: assert 'state writes' in str(error)
    else: raise AssertionError('Model write silently passed')
  finally:
    agent.model.policy=original_policy
  # Test-only zero final prior/posterior heads give equal actual distributions.
  def equal_heads(params):
    return {name:jnp.zeros_like(value) if name.startswith(('dyn/obslogit/','dyn/priorlogit/'))
        else value for name,value in params.items()}
  agent.policy_params=jax.jit(equal_heads)(agent.policy_params)
  equal_adapter=f.FeedbackPolicy(agent)
  eq_q,eq_qa,eq_qp=equal_adapter.step(before,o1,key,False)
  eq_p,eq_pa,eq_pp=equal_adapter.step(before,o1,key,True)
  assert np.array_equal(eq_qp['selected_logit'],eq_pp['selected_logit'])
  assert f.tree_digest(eq_q)==f.tree_digest(eq_p) and eq_qa.tobytes()==eq_pa.tobytes()
  assert f.counters(agent)==saved_counters
  print(json.dumps(dict(clean_actual_public_exact=True,q_equals_p_exact=True,prior_rgb_independent=True,
      shared_h_exact=True,future_normal_posterior_uses_own_rgb=True,private_keys_exact=True,
      call_order_invariant=True,bf16_host_carry_exact=True,strict_transfer_guard=jax.config.jax_transfer_guard=='disallow',
      state_and_counters_unchanged=True,state_writes_rejected=True,actual_distribution_samples_exact=True)))


if __name__=='__main__':
  if '--fixture' in sys.argv:fixture()
  else:unittest.main()
