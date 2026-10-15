"""Conditional-history engineering on a real tiny BF16 image64/action12 Agent.

All observations/actions are generated. No real N1M checkpoint, source,
environment, task label, trained endpoint, GPU or method verdict is used.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import dl_normal_features as f


class NormalFeaturesTests(unittest.TestCase):
  def test_import_and_source_keys_do_not_load_models_or_claim_source_authentication(self):
    command = "import os,sys;before=dict(os.environ);import scripts.dl_normal_features;assert before==dict(os.environ);assert not any(n.split('.')[0] in ('jax','dm_control','mujoco','OpenGL') for n in sys.modules)"
    result = subprocess.run([sys.executable,'-c',command],cwd=ROOT,capture_output=True,text=True,timeout=30)
    self.assertEqual(result.returncode,0,result.stderr)
    self.assertNotEqual(f.KEY_NAMESPACE,'DL_R4_U1')
    self.assertNotEqual(f.KEY_NAMESPACE,'DL_R5_LIVE_U1')
    for seed,counter in ((0,0),(0x52350000,896),(0x4E4D3100+31,999)):
      expected = np.random.default_rng([0,0x4556414C,seed,counter]).integers(
          0,np.iinfo(np.uint32).max,(2,),np.uint32)
      np.testing.assert_array_equal(f.policy_root_key(seed,counter),expected)
    for pair in ((True,0),(0,-1),(0,np.int64(1)),(2**32,0)):
      with self.assertRaises(ValueError):f.policy_root_key(*pair)

  def test_reject_malformed_observations_and_non_raw_actions(self):
    good = dict(image=np.zeros((64,64,3),np.uint8),reward=np.float32(0),
        is_first=np.bool_(False),is_last=np.bool_(False),is_terminal=np.bool_(False))
    for key,value in (('image',np.zeros((64,64,3),np.float32)),('reward',np.float64(0)),
        ('is_last',np.bool_(True)),('is_terminal',np.bool_(True))):
      with self.subTest(key=key),self.assertRaises(ValueError):f.observation(dict(good,**{key:value}))
    with self.assertRaises(ValueError):f.observation(dict(good,is_first=np.bool_(True),reward=np.float32(1)))
    for value in (np.zeros(12,np.float64),np.zeros((1,12),np.float32),np.full(12,np.nan,np.float32)):
      with self.assertRaises(ValueError):f.raw_action(value)

  def test_real_tiny_bf16_history_oracle_unimix_and_readonly_contracts(self):
    result = subprocess.run([sys.executable,str(Path(__file__).resolve()),'--fixture'],
        cwd=ROOT,capture_output=True,text=True,timeout=180,
        env=dict(os.environ,PYTHONPATH=str(ROOT),JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',MUJOCO_GL='disable'))
    self.assertEqual(result.returncode,0,result.stdout[-4000:]+result.stderr[-9000:])
    value = json.loads(result.stdout.strip().splitlines()[-1])
    self.assertTrue(value['public_native_before_source_override'])
    self.assertTrue(value['source_raw_changes_next_real_h'])
    self.assertTrue(value['official_unimix_not_mode_or_bare_softmax'])
    self.assertTrue(value['full_state_policy_copy_counters_unchanged'])
    self.assertFalse(value['source_authenticated'])
    self.assertFalse(value['real_N1M_authenticated'])
    self.assertFalse(value['GPU_used'])
    print(json.dumps(value,sort_keys=True))


def fixture():
  dll = (os.add_dll_directory(os.environ['DL_DLL_DIRECTORY'])
      if os.name == 'nt' and os.environ.get('DL_DLL_DIRECTORY') else None)
  import elements
  import jax
  import jax.numpy as jnp
  import ninjax as nj
  import ml_dtypes
  from embodied.jax import internal
  from ruamel.yaml import YAML
  from scripts.dl_predictive_mc import agent_from_spaces
  presets = YAML(typ='safe').load((ROOT/'dreamerv3/configs.yaml').read_text())
  config = elements.Config(presets['defaults']).update(presets['debug'])
  config = elements.Config({**config.flat,**{'jax.platform':'cpu','jax.compute_dtype':'bfloat16',
      'jax.precompile':False,'jax.profiler':False,'jax.enable_policy':True,'jax.transfer_guard':True,
      'batch_size':2,'batch_length':3,'replay_context':1,'agent.imag_length':2,'agent.imag_last':2,
      'agent.rep_probe.mode':'off','agent.dt_latch.mode':'off'}}).update(
          logdir='/tmp/dl-normal-features-fixture',seed=0)
  agent = agent_from_spaces(config)
  saved = agent.save();agent.load(saved)
  original_state = f.tree_digest(agent.save())
  original_counters = f.counters(agent)
  try:f.FrozenNormalFeatures(agent)
  except ValueError as error:assert 'geometry' in str(error)
  else:raise AssertionError('Tiny geometry silently accepted as production')
  a,b = f.FrozenNormalFeatures(agent,fixture=True),f.FrozenNormalFeatures(agent,fixture=True)
  rng = np.random.default_rng(20261108)
  def obs(first=False):
    return dict(image=rng.integers(0,256,(64,64,3),np.uint8),reward=np.float32(0),
        is_first=np.bool_(first),is_last=np.bool_(False),is_terminal=np.bool_(False))
  reset,current = obs(True),obs()
  source_a = np.linspace(-.9,.8,12,dtype=np.float32)
  source_b = -source_a
  zero = np.zeros(12,np.float32)
  def rejected(operation,fragment=None):
    try:operation()
    except (ValueError,RuntimeError,AssertionError) as error:
      if fragment is not None:assert fragment in str(error),(fragment,str(error))
    else:raise AssertionError('Invalid feature/history contract accepted')
  rejected(lambda:a.step(reset,source_a,0,source_prev_raw=zero),'Reset')
  rejected(lambda:a.reset(current,eval_seed=0x4E4D3100,source_id='generated'),'reset')
  a.reset(reset,eval_seed=0x4E4D3100,source_id='generated/A')
  b.reset(reset,eval_seed=0x4E4D3100,source_id='generated/B')
  inputs_before = f.tree_digest((reset,current,source_a,source_b))
  rejected(lambda:a.step(reset,source_a,1,source_prev_raw=zero),'counter')
  rejected(lambda:a.step(reset,source_a,0,source_prev_raw=source_a),'Previous')
  first_a = a.step(reset,source_a,0,source_prev_raw=zero)
  first_b = b.step(reset,source_b,0,source_prev_raw=zero)
  assert f.tree_digest(first_a['native_after']) == f.tree_digest(first_b['native_after'])
  assert f.tree_digest(first_a['native_after'][:3]) == f.tree_digest(first_a['history_after'][:3])
  assert first_a['native_raw'].tobytes() != source_a.tobytes()
  assert first_a['native_after'][3]['action'][0].tobytes() == first_a['native_raw'].tobytes()
  assert first_a['history_after'][3]['action'][0].tobytes() == source_a.tobytes()
  assert first_a['public_oracle']['before_host_override']
  assert first_a['public_oracle']['native_full_after_carry_byte_exact']
  rejected(lambda:a.step(current,source_a,1,source_prev_raw=first_a['native_raw']),'Previous')
  rejected(lambda:a.step(reset,source_a,1,source_prev_raw=source_a),'Exactly')
  # Caller output copies cannot replace this model's owned posterior history.
  first_a['history_after'][1]['deter'][0][0] = np.asarray(99,ml_dtypes.bfloat16)
  second_a = a.step(current,source_a,1,source_prev_raw=source_a)
  second_b = b.step(current,source_b,1,source_prev_raw=source_b)
  assert second_a['h'].tobytes() != second_b['h'].tobytes()
  assert second_a['q_prob'].shape == (2,4) and str(second_a['h'].dtype) == 'bfloat16'
  assert f.tree_digest((reset,current,source_a,source_b)) == inputs_before
  initial_carry = copy.deepcopy(a._carry)
  a._carry[1]['deter'][0][0] = np.asarray(99,ml_dtypes.bfloat16)
  rejected(lambda:a.step(current,source_a,2,source_prev_raw=source_a),'history')
  a._carry = initial_carry
  with agent.n_actions.lock:agent.n_actions.value += 1
  rejected(lambda:a.fence(),'counters')
  with agent.n_actions.lock:agent.n_actions.value -= 1
  # Separate actual official distribution query: same mode, distinct full
  # probabilities. The unimix prevents substitution with bare raw-logit softmax.
  logits_a = np.full((1,2,4),-4,ml_dtypes.bfloat16);logits_a[...,0] = 2
  logits_b = logits_a.copy();logits_b[...,0] = 4
  h = second_a['h'][None];key = second_a['root_key']
  qa = a._execute(a._query,(h,logits_a),key)
  qb = a._execute(a._query,(h,logits_b),key)
  assert np.array_equal(logits_a.argmax(-1),logits_b.argmax(-1))
  assert not np.array_equal(qa['q_prob'],qb['q_prob'])
  assert np.all(qa['q_prob'] > 0) and not np.all((qa['q_prob'] == 0) | (qa['q_prob'] == 1))
  reference = jax.jit(lambda x:(jax.nn.softmax(agent.model.dyn._dist(x).output.dist.logits,-1),
      jax.nn.softmax(x.astype(jnp.float32),-1)))
  official,bare = f.host_copy(reference(internal.device_put(logits_a,agent.train_mirrored)))
  assert official.tobytes() == qa['q_prob'].tobytes()
  assert bare.tobytes() != official.tobytes()
  # Tampered public output is rejected before the history is advanced or the
  # source raw is written to its host carry.
  public = agent.policy
  def broken_public(*args,**kwargs):
    after,actions,outs = public(*args,**kwargs)
    actions = {k:v.copy() for k,v in actions.items()}
    actions['action'][0,0] = np.nextafter(actions['action'][0,0],np.float32(np.inf))
    return after,actions,outs
  before = a._carry_hash
  agent.policy = broken_public
  try:rejected(lambda:a.step(current,source_a,2,source_prev_raw=source_a),'byte exact')
  finally:agent.policy = public
  assert a._carry_hash == before and a._episode['next_counter'] == 2
  def extra_counter(*args,**kwargs):
    result = public(*args,**kwargs)
    with agent.n_eval_actions.lock:agent.n_eval_actions.value += 1
    return result
  agent.policy = extra_counter
  try:rejected(lambda:a.step(current,source_a,2,source_prev_raw=source_a),'exactly one eval counter')
  finally:agent.policy = public
  assert a._carry_hash == before and a._episode['next_counter'] == 2
  # Re-tracing a model that attempts a state write is rejected, with temporary
  # hooks restored. No zero-tolerance readonly contract is weakened for fixture.
  model_policy = agent.model.policy
  core,logit,seed = agent.model.dyn._core,agent.model.dyn._logit,nj.seed
  def writes(*args,**kwargs):
    name = next(iter(nj.context()));nj.context()[name] = nj.context()[name]
    return model_policy(*args,**kwargs)
  agent.model.policy = writes
  try:
    bad = f.FrozenNormalFeatures(agent,fixture=True)
    bad.reset(reset,eval_seed=0x4E4D3100,source_id='generated/readonly-counterexample')
    rejected(lambda:bad.step(reset,source_a,0,source_prev_raw=zero),'writes')
  finally:agent.model.policy = model_policy
  assert agent.model.dyn._core == core and agent.model.dyn._logit == logit and nj.seed == seed
  seal_a,seal_b = a.finish(),b.finish()
  assert seal_a['full_state_policy_copy_counters_unchanged'] and seal_b['full_state_policy_copy_counters_unchanged']
  assert f.tree_digest(agent.save()) == original_state and f.counters(agent) == original_counters
  assert all(device.platform == 'cpu' for device in jax.devices())
  print(json.dumps(dict(version=f.VERSION,scope='Generated tiny BF16 image64/action12, two two-step histories',
      public_native_before_source_override=True,source_raw_changes_next_real_h=True,
      official_unimix_not_mode_or_bare_softmax=True,full_state_policy_copy_counters_unchanged=True,
      source_authenticated=False,real_N1M_authenticated=False,GPU_used=False)))
  del dll


if __name__ == '__main__':
  if len(sys.argv) == 2 and sys.argv[1] == '--fixture':fixture()
  else:unittest.main()
