"""Finite CPU scratch-update evidence; not real N1M calibration/learning."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.dl_nm1_counterfactual import neural_difference,neural_displacements


class CounterfactualTests(unittest.TestCase):
  def test_f32_ulp_and_neural_groups_exclude_optimizer_state_and_signed_zero(self):
    a={'enc/w':np.array([0.,1.,-2.,np.finfo(np.float32).max],np.float32),
        'dyn/obslogit/w':np.array([2.],np.float32),'opt/state/x':np.array([1],np.int32)}
    b={k:v.copy() for k,v in a.items()}
    b['enc/w'][0]=-0.;b['enc/w'][1]=np.nextafter(np.float32(1.),np.float32(2.))
    b['dyn/obslogit/w'][0]=3.;b['opt/state/x'][0]=17
    report=neural_difference(a,b)
    self.assertFalse(report['encoder']['byte_identical'])
    self.assertEqual(report['encoder']['changed_elements'],1)
    self.assertEqual(report['encoder']['max_reference_ulp_difference'],1.)
    self.assertEqual(report['posterior']['changed_elements'],1)
    self.assertEqual(report['all']['elements'],5)
    with self.assertRaisesRegex(ValueError,'finite F32'):
      neural_difference(a,dict(b,**{'enc/w':b['enc/w'].astype(np.float64)}))

  def test_real_agent_scratch_is_readonly_and_matches_public_train_with_old_moments(self):
    with tempfile.TemporaryDirectory(prefix='dl-nm1-scratch-') as directory:
      output=Path(directory)/'result.json'
      result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--fixture',str(output)],
          cwd=ROOT,capture_output=True,text=True,timeout=240,
          env=dict(os.environ,PYTHONPATH=str(ROOT),JAX_PLATFORMS='cpu',
              CUDA_VISIBLE_DEVICES='',MUJOCO_GL='disable'))
      self.assertEqual(result.returncode,0,result.stdout[-3000:]+result.stderr[-5000:])
      proof=json.loads(output.read_text())
      self.assertTrue(proof['complete_source_unchanged'])
      self.assertTrue(proof['public_train_matches_scratch'])
      self.assertEqual(proof['initial_opt_step'],1)
      self.assertFalse(proof['real_N1M_calibration'])
      self.assertFalse(proof['GPU_used'])
      print(json.dumps(proof,sort_keys=True))

  def test_endpoint_direction_is_not_actual_optimizer_displacement_direction(self):
    before={'enc/w':np.array([1e6,1e6],np.float32)}
    left={'enc/w':before['enc/w']+np.array([1,0],np.float32)}
    right={'enc/w':before['enc/w']+np.array([0,1],np.float32)}
    self.assertGreater(neural_difference(left,right)['all']['tensor_cosine'],.999999)
    actual=neural_displacements(before,left,right)['all']
    self.assertEqual(actual['update_cosine'],0.)
    self.assertEqual(actual['reference_update_l2'],1.)
    self.assertEqual(actual['alternative_update_l2'],1.)
    self.assertAlmostEqual(actual['update_difference_l2'],np.sqrt(2))
    self.assertIsNone(neural_displacements(before,before,right)['all']['update_cosine'])


def fixture(output):
  dll=(os.add_dll_directory(os.environ['DL_DLL_DIRECTORY'])
      if os.name == 'nt' and os.environ.get('DL_DLL_DIRECTORY') else None)
  import elements
  import jax
  from ruamel.yaml import YAML
  from dreamerv3.agent import Agent
  from scripts.dl_nm1_counterfactual import Counterfactual,host,prepare_readonly
  from scripts.dl_reset_learning import tree_digest

  presets=YAML(typ='safe').load((ROOT/'dreamerv3/configs.yaml').read_text())
  cfg=elements.Config(presets['defaults']).update(presets['debug'])
  cfg=elements.Config({**cfg.flat,**{'batch_size':2,'batch_length':4,'replay_context':1,
      'jax.platform':'cpu','jax.compute_dtype':'float32','jax.train_devices':[0],
      'jax.enable_policy':False,'jax.precompile':False,'jax.profiler':False,
      'agent.dt_latch.mode':'off','agent.rep_probe.mode':'off'}})
  agent=Agent(dict(vector=elements.Space(np.float32,(5,)),reward=elements.Space(np.float32),
      is_first=elements.Space(bool),is_last=elements.Space(bool),is_terminal=elements.Space(bool)),
      dict(action=elements.Space(np.float32,(2,))),elements.Config(**cfg.agent.flat,
          logdir=str(Path(output).parent/'agent'),seed=20261101,jax=cfg.jax,
          batch_size=2,batch_length=4,replay_context=1))
  assert all(d.platform == 'cpu' for d in jax.devices())
  assert agent.model.dyn.free_nats == 1.
  raw=agent._zeros(agent.spaces,(2,5))
  raw['vector'][:]=np.arange(50,dtype=np.float32).reshape(2,5,5)/50
  raw['reward'][:]=[[0,.1,.3,.8,.9],[0,-.2,.4,.7,.5]]
  raw['is_first'][:,0]=True;raw['is_last'][:,-1]=True
  raw['is_first'][1,3]=True;raw['is_last'][1,2]=True
  raw['dyn/deter'][:]=.2;raw['dyn/stoch'][...,0]=1
  raw['action'][:]=np.linspace(-.3,.3,20).reshape(2,5,2)
  for row in range(2):
    for col in range(5):raw['stepid'][row,col,:]=row*5+col
  carry=agent.init_train(2)
  carry,delayed_out,delayed_mets=agent.train(carry,dict(agent.prepare_batch(raw)))
  assert not delayed_out and not delayed_mets
  agent.take_train_result()
  initial=agent.save()
  assert int(initial['params']['opt/step/value']) == 1
  assert any(np.any(v != 0) for k,v in initial['params'].items()
      if k.startswith('opt/state/') and np.asarray(v).shape)
  source_digest=tree_digest(initial)
  before_batches=int(agent.n_batches)
  prepared=prepare_readonly(agent,raw,before_batches)
  assert int(agent.n_batches) == before_batches
  probe=Counterfactual(agent)
  ref1,ref2=probe.reference(carry,prepared),probe.reference(carry,prepared)
  original = host(ref1['aux'][2]['nm1_reference'])
  assert set(original) == {'K','E','rep_before','weight'}
  assert original['K'].dtype == original['E'].dtype == np.float32
  np.testing.assert_array_equal(original['E'],np.maximum(original['K']-np.float32(1),np.float32(0)))
  np.testing.assert_array_equal(original['rep_before'],np.maximum(original['K'],np.float32(1)))
  np.testing.assert_array_equal(original['weight'],np.ones((2,4),np.float32))
  np.testing.assert_array_equal(host(ref1['aux'][2]['losses']['rep']),original['rep_before'])
  assert tree_digest(ref1['grads']) == tree_digest(ref2['grads'])
  assert any(np.any(v != 0) for v in ref1['grads'].values())
  assert set(ref1['grads']) == set(ref1['params'])
  assert set(probe.reference_modified) <= {k for k in agent.params
      if k.startswith(('retnorm/','valnorm/','advnorm/'))}
  first,second=probe.production(carry,prepared),probe.production(carry,prepared)
  assert tree_digest((first['params'],first['carry'],first['outs'],first['metrics'])) == tree_digest(
      (second['params'],second['carry'],second['outs'],second['metrics']))
  assert int(first['params']['opt/step/value']) == 2
  assert neural_difference(initial['params'],first['params'])['all']['changed_elements'] > 0
  probe.fence(full=True)
  assert tree_digest(agent.save()) == source_digest
  # Use the real public wrapper only after the immutable probe has finished.
  # This validates placement/root equivalence and every resulting state leaf,
  # including optimizer moments, normalizers and slow target.
  actual=agent.prepare_batch(raw)
  assert tree_digest(host(actual)) == tree_digest(host(prepared))
  public_carry,delayed_out,delayed_mets=agent.train(carry,dict(actual))
  assert not delayed_out and not delayed_mets
  out,mets=agent.take_train_result()
  assert tree_digest(agent.save()['params']) == tree_digest(first['params'])
  assert tree_digest(host(public_carry)) == tree_digest(first['carry'])
  assert tree_digest(out) == tree_digest(first['outs'])
  assert tree_digest(mets) == tree_digest(first['metrics'])
  extra={}
  for name,rho in (('rho0',0.),('N_nr',.5)):
    branch_cfg=elements.Config({**cfg.agent.flat,**{'dt_latch.mode':'dtlatch',
        'dt_latch.rho':rho,'dt_latch.normal_only':True}})
    branch=Agent(dict(agent.obs_space),dict(agent.act_space),elements.Config(**branch_cfg.flat,
        logdir=str(Path(output).parent/name),seed=20261101,jax=cfg.jax,
        batch_size=2,batch_length=4,replay_context=1))
    branch.load(initial)
    branch_probe=Counterfactual(branch)
    branch_prepared=prepare_readonly(branch,raw,before_batches)
    branch_reference=branch_probe.reference(carry,branch_prepared)
    branch_raw=host(branch_reference['aux'][2]['nm1_reference'])
    np.testing.assert_array_equal(branch_raw['E'],np.maximum(branch_raw['K']-np.float32(1),np.float32(0)))
    np.testing.assert_array_equal(branch_raw['rep_before'],np.maximum(branch_raw['K'],np.float32(1)))
    branch_step=branch_probe.production(carry,branch_prepared)
    assert int(branch_step['params']['opt/step/value']) == 2
    branch_probe.fence(full=True)
    if name == 'rho0':
      assert tree_digest((branch_step['params'],branch_step['carry'],branch_step['outs'],branch_step['metrics'])) == tree_digest(
          (first['params'],first['carry'],first['outs'],first['metrics']))
      assert tree_digest(branch_reference['grads']) == tree_digest(ref1['grads'])
      extra['rho0_matches_B']=True
    else:
      signals=host(branch_reference['aux'][2]['dl'])
      expected=signals['m'] & ~raw['is_first'][:,1:]
      np.testing.assert_array_equal(signals['m_intervention'],expected)
      np.testing.assert_array_equal(branch_raw['weight'],signals['v_intervention'])
      np.testing.assert_array_equal(host(branch_reference['aux'][2]['losses']['rep']),
          branch_raw['weight']*branch_raw['rep_before'])
      assert signals['m'][1,2] and not signals['m_intervention'][1,2]
      extra['N_nr_finite_source_unchanged']=True
      extra['N_nr_active_update_difference_not_required_by_fixture']=True
    # Mutating a static mode after JIT construction must be rejected, not
    # silently interpreted as a new intervention with an old compiled graph.
    old_config=branch.model.config
    branch.model.config=old_config.update({'dt_latch.rho':.25})
    try:
      try:branch_probe.fence()
      except ValueError as error:assert 'configuration changed' in str(error)
      else:raise AssertionError('Static configuration mutation was not rejected')
    finally:branch.model.config=old_config
  Path(output).write_text(json.dumps(dict(version='DL-NM1-counterfactual-fixture-r2',
      complete_source_unchanged=True,public_train_matches_scratch=True,
      initial_opt_step=1,reference_gradient_scope=ref1['gradient_scope'],
      neural_difference=neural_difference(initial['params'],first['params']),
      real_N1M_calibration=False,GPU_used=False,**extra,
      scope='Generated tiny F32 vector/action2 B2/T4/context1; full loss/optimizer; not size50m/BF16/real65')))
  del dll


if __name__ == '__main__':
  if len(sys.argv) == 3 and sys.argv[1] == '--fixture':fixture(sys.argv[2])
  else:unittest.main()
