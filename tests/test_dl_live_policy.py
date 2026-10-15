"""Frozen real tiny BF16 model; synthetic RGB/labels, never N1M acceptance."""
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
from scripts import dl_live_policy as p


class LivePolicyTest(unittest.TestCase):
  def test_frozen_plan_namespace_absolute_keys_and_delayed_label_receipt(self):
    plans = [p.unit_plan(u) for u in range(512)]
    self.assertEqual(len({x['eval_seed'] for x in plans}), 512)
    self.assertEqual({x['env_seed'] for x in plans}, set(range(8192, 8224)))
    for unit in (0, 15, 16, 511):
      plan = plans[unit]
      self.assertNotIn('label', plan)
      self.assertEqual(plan['t'], 64+64*int(np.random.default_rng(
          [20261030,0x54494D45,unit]).integers(0,14)))
      for step in (0, plan['t'], plan['t']+99):
        expected = np.random.default_rng([0,0x4556414C,0x52350000+unit,step]).integers(
            0,np.iinfo(np.uint32).max,(2,),np.uint32)
        self.assertEqual(p.policy_root_key(unit,step).tobytes(),expected.tobytes())
      label = p.reveal_label(unit, 'a'*64)
      self.assertEqual(label['label'],p.LABELS[int(np.random.default_rng(
          [20261030,0x41524D53,unit]).integers(0,4))])
    for args in ((True,0),(512,0),(0,-1),(0,plans[0]['t']+100)):
      with self.assertRaises(ValueError): p.policy_root_key(*args)
    with self.assertRaises(ValueError): p.reveal_label(0, '')

  def test_real_tiny_policy_new_public_oracle_seal_and_native_future(self):
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--fixture'],
        cwd=ROOT, capture_output=True, text=True, timeout=240,
        env=dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT)))
    self.assertEqual(result.returncode, 0, result.stdout[-4000:]+result.stderr[-10000:])
    facts = json.loads(result.stdout.strip().splitlines()[-1])
    self.assertTrue(all(facts.values()),facts)


def fixture():
  import elements
  from ruamel.yaml import YAML
  from scripts.dl_predictive_mc import agent_from_spaces
  from scripts import dl_live_shadow as s
  cfg = YAML(typ='safe').load((ROOT/'dreamerv3/configs.yaml').read_text(encoding='utf-8'))
  config = elements.Config(cfg['defaults']).update(cfg['debug'])
  config = elements.Config({**config.flat, **{'jax.platform':'cpu','jax.compute_dtype':'bfloat16',
      'jax.precompile':False,'jax.profiler':False,'jax.enable_policy':True,'jax.transfer_guard':True,
      'batch_size':2,'batch_length':3,'replay_context':1,'agent.imag_length':2,'agent.imag_last':2,
      'agent.rep_probe.mode':'off','agent.dt_latch.mode':'off'}}).update(
          logdir='/tmp/dl-r5-live-policy-fixture',seed=0)
  agent = agent_from_spaces(config)
  checkpoint = agent.save(); agent.load(checkpoint)
  adapter = p.LivePolicy(agent)
  saved = p.kernel.full_digest(agent.save())
  rng = np.random.default_rng(69)
  def obs(first=False):
    return dict(image=rng.integers(0,256,(64,64,3),np.uint8),reward=np.float32(0),
        is_first=np.bool_(first),is_last=np.bool_(False),is_terminal=np.bool_(False))
  unit = 0; plan = p.unit_plan(unit)
  initial = adapter.initial(unit); carry = copy.deepcopy(initial)
  first_obs = obs(True)
  after, raw, proof = adapter.normal_step(carry,first_obs,unit,0)
  adapter.public_oracle(carry,first_obs,unit,0,expected_after=after,expected_raw=raw)
  carry = after
  # Actual tiny normal-prefix call sequence; all RGB is synthetic.
  for step in range(1,plan['t']): carry,_,_ = adapter.normal_step(carry,obs(),unit,step)
  current_obs = obs()
  bundle = adapter.current(carry,current_obs,unit)
  adapter.public_oracle(carry,current_obs,unit,plan['t'],
      expected_after=bundle['q_after'],expected_raw=bundle['q_raw'])
  assert bundle['kernel_namespace']==p.kernel.KEY_NAMESPACE
  assert bundle['key_namespace']!=bundle['kernel_namespace']
  fake_snapshot = dict(suite=dict(integration=np.zeros(3,np.float64),step_count=plan['t'],
      reset_next_step=False,task={}),from_dm_done=False)
  snap_sha = s.snapshot_sha256(fake_snapshot)
  shadow = dict(version=s.VERSION,unit=unit,env_seed=plan['env_seed'],request_id='fixture',
      request_sha256='b'*64,bundle_sha256=bundle['bundle_sha256'],
      source_snapshot_sha256=snap_sha,static_mjb_sha256='c'*64,raw_q=bundle['q_raw'].copy(),
      applied=np.zeros(12,np.float64),reward_f64=np.asarray(.173,np.float64),
      reward_f32=np.asarray(.173,np.float32),discount=np.asarray(1.,np.float64),
      is_first=np.bool_(False),is_last=np.bool_(False),is_terminal=np.bool_(False),
      next_snapshot=fake_snapshot,next_snapshot_sha256=snap_sha,
      official_steps=1,render_calls=0,model_calls=0,live_environment_touched=False)
  shadow['response_sha256'] = s.response_sha256(shadow)
  label = p.reveal_label(unit, 'a'*64)
  try: adapter.first(bundle,label['label'],label_receipt=label)
  except ValueError: pass
  else: raise AssertionError('Unsealed intervention accepted')
  seal = adapter.seal_pretreatment(bundle,shadow,
      live_before_sha256=snap_sha,live_after_sha256=snap_sha,r_t_f64=np.float64(0))
  label = p.reveal_label(unit,seal['pretreatment_sha256'])
  intervened,raw,proof = adapter.first(bundle,label['label'],label_receipt=label)
  assert set(proof)==set(p.PROOF_KEYS)
  assert intervened[3]['action'][0].tobytes()==raw.tobytes()
  reference = adapter.kernel.first(bundle['kernel_bundle'],
      'qq' if label['label'] in ('qqA','qqB') else label['label'])
  assert p.kernel.tree_digest(reference[0])==p.kernel.tree_digest(intervened)
  for invalid in ('label','unit','seal','bundle','live','reward','baseline','oracle'):
    try:
      if invalid in ('label','unit','seal'):
        broken = copy.deepcopy(label)
        if invalid=='label': broken['label']='pp' if label['label']!='pp' else 'pq'
        elif invalid=='unit': broken['unit']=1
        else: broken['pretreatment_sha256']='d'*64
        adapter.first(bundle,label['label'],label_receipt=broken)
      elif invalid=='bundle':
        broken=copy.deepcopy(bundle);broken['q_raw'][0]+=1
        adapter.first(broken,label['label'],label_receipt=label)
      elif invalid=='live':
        adapter.seal_pretreatment(bundle,shadow,live_before_sha256=snap_sha,
            live_after_sha256='e'*64,r_t_f64=np.float64(0))
      elif invalid=='reward':
        broken=copy.deepcopy(shadow);broken['reward_f64'][...]=np.nextafter(broken['reward_f64'],np.inf)
        adapter.score_h1(bundle,broken)
      elif invalid=='baseline':
        adapter.seal_pretreatment(bundle,shadow,live_before_sha256=snap_sha,
            live_after_sha256=snap_sha,r_t_f64=np.float32(0))
      else:
        wrong=bundle['q_raw'].copy();wrong[0]=np.nextafter(wrong[0],np.float32(np.inf))
        adapter.public_oracle(carry,current_obs,unit,plan['t'],
            expected_after=bundle['q_after'],expected_raw=wrong)
    except ValueError: pass
    else: raise AssertionError('Bad R5 contract accepted: '+invalid)
  carry = intervened
  for step in range(plan['t']+1,plan['t']+100):
    before = carry; current_obs=obs()
    carry,raw,proof=adapter.normal_step(before,current_obs,unit,step)
    assert int(proof['memory_source'])==int(proof['action_source'])==0
    if step==plan['t']+99:
      adapter.public_oracle(before,current_obs,unit,step,expected_after=carry,expected_raw=raw)
  # Each remaining assigned label follows the same kernel operation. This is
  # interface evidence on synthetic carry/RGB, not extra live task trajectories.
  for other_unit in (2,3,14):
    other_bundle=adapter.current(carry,obs(),other_unit)
    other_shadow=copy.deepcopy(shadow)
    other_snapshot=copy.deepcopy(fake_snapshot)
    other_snapshot['suite']['step_count']=p.unit_plan(other_unit)['t']
    other_sha=s.snapshot_sha256(other_snapshot)
    other_shadow.update(unit=other_unit,env_seed=p.unit_plan(other_unit)['env_seed'],
        bundle_sha256=other_bundle['bundle_sha256'],raw_q=other_bundle['q_raw'].copy(),
        source_snapshot_sha256=other_sha,next_snapshot=other_snapshot,next_snapshot_sha256=other_sha)
    other_shadow['response_sha256']=s.response_sha256(other_shadow)
    other_seal=adapter.seal_pretreatment(other_bundle,other_shadow,
        live_before_sha256=other_sha,live_after_sha256=other_sha,r_t_f64=np.float64(0))
    other_label=p.reveal_label(other_unit,other_seal['pretreatment_sha256'])
    result=adapter.first(other_bundle,other_label['label'],label_receipt=other_label)
    kernel_arm='qq' if other_label['label'] in ('qqA','qqB') else other_label['label']
    expected=adapter.kernel.first(other_bundle['kernel_bundle'],kernel_arm)
    assert p.kernel.tree_digest(result[0])==p.kernel.tree_digest(expected[0])
    assert result[1].tobytes()==expected[1].tobytes()
  assert p.kernel.tree_digest(adapter.initial(unit))==p.kernel.tree_digest(initial)
  adapter.assert_frozen(full=True)
  assert p.kernel.full_digest(agent.save())==saved
  assert adapter.receipt()['public_oracle_calls']==3
  print(json.dumps(dict(actual_tiny_bf16_new_domain_public_q_three_points_exact=True,
      full_synthetic_prefix_and_H100_native_q_future=True,seal_and_label_binding_rejects_tamper=True,
      first_kernel_memory_action_prevraw_exact=True,full_state_policycopy_counters_unchanged=True,
      all_four_assigned_labels_map_to_expected_original_kernel=True,
      fresh_initial_has_no_previous_unit_memory=True,synthetic_shadow_not_actual_environment=True,
      fixture_not_N1M_or_GPU_acceptance=True)))


if __name__=='__main__':
  if sys.argv[1:]==['--fixture']: fixture()
  else: unittest.main()
