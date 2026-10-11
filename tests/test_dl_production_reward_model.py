"""Native tiny BF16 CPU engineering; no real source or size50m skill claim."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_production_reward_model as m


class NativeRewardModelTest(unittest.TestCase):
  def test_native_roundtrip_rejects_lost_precision_and_nonsampled_latent(self):
    import ml_dtypes
    h = np.array([0., -.125, 1.5, -0.], ml_dtypes.bfloat16).astype(np.float64)
    z = np.zeros((4, 2, 2, 4), np.float64); z[..., 1] = 1
    a, b = m.restore_native(h, z)
    self.assertEqual(a.astype(np.float64).tobytes(), h.tobytes())
    self.assertEqual(b.astype(np.float64).tobytes(), z.tobytes())
    for value in (np.array([.1, 0., 0., 0.]), np.full(4, np.nan), h.astype(np.float32)):
      with self.assertRaises(ValueError): m.restore_native(value, z)
    invalid = z.copy(); invalid[..., 0] = .5; invalid[..., 1] = .5
    with self.assertRaisesRegex(ValueError, 'onehot'): m.restore_native(h, invalid)
    with self.assertRaisesRegex(ValueError, 'axes'): m.restore_native(h, z[0])

  def test_import_has_no_device_or_environment_side_effect(self):
    code = "import os,sys; before=dict(os.environ); import scripts.dl_production_reward_model; assert dict(os.environ)==before; assert 'jax' not in sys.modules; assert not any(x.startswith('dm_control') for x in sys.modules)"
    child = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=str(ROOT)), capture_output=True, text=True, timeout=30)
    self.assertEqual(child.returncode, 0, child.stdout + child.stderr)

  def test_actual_tiny_official_imagination_private_rng_and_frozen_state(self):
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--fixture'],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
        env=dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT)))
    self.assertEqual(child.returncode, 0, child.stdout[-5000:] + child.stderr[-8000:])
    facts = json.loads(child.stdout.strip().splitlines()[-1])
    for name in ('official_single_step_exact', 'next_reward_phase_exact',
        'q_equals_p_exact', 'initial_samples_fixed', 'future_key_prefix_exact',
        'future_action_prefix_exact', 'native_symmetric_pred_oracle', 'unimix_once',
        'all_future_steps_sampled', 'named_root_keys_exact', 'private_call_order_exact',
        'strict_transfer_guard', 'state_and_counters_unchanged', 'attempted_writes_rejected',
        'attempted_creation_rejected', 'counter_change_rejected',
        'extra_seed_rejected', 'input_guards', 'cancellation_counterexample_detected'):
      self.assertTrue(facts[name], name)


def fixture():
  dll = (os.add_dll_directory(os.environ['DL_DLL_DIRECTORY'])
      if os.name == 'nt' and os.environ.get('DL_DLL_DIRECTORY') else None)
  import elements
  import jax
  import jax.numpy as jnp
  import ml_dtypes
  import ninjax as nj
  from embodied.jax import internal
  from embodied.jax import nets as nn
  from dreamerv3.rssm import RSSM
  from ruamel.yaml import YAML
  from scripts.dl_predictive_mc import agent_from_spaces
  from unittest import mock
  presets = YAML(typ='safe').load((ROOT / 'dreamerv3/configs.yaml').read_text(encoding='utf-8'))
  config = elements.Config(presets['defaults']).update(presets['debug'])
  config = elements.Config({**config.flat, **{'jax.platform': 'cpu',
      'jax.compute_dtype': 'bfloat16', 'jax.transfer_guard': True,
      'jax.precompile': False, 'jax.profiler': False, 'jax.enable_policy': True,
      'batch_size': 2, 'batch_length': 3, 'replay_context': 0,
      'agent.imag_length': 2, 'agent.imag_last': 2,
      'agent.rep_probe.mode': 'off', 'agent.dt_latch.mode': 'off'}}).update(
          logdir='/tmp/dl-production-reward-model', seed=7)
  before_env_modules = {x for x in sys.modules if x.startswith('dm_control')}
  agent = agent_from_spaces(config)
  assert before_env_modules == {x for x in sys.modules if x.startswith('dm_control')}
  # Only test parameters are changed before freezing. A zero initial reward head
  # would hide causal-prefix failures by returning zero under every input.
  def nonuniform(params):
    result = dict(params)
    name = 'rew/head/logits/kernel'
    result[name] = .03*jnp.sin(jnp.arange(result[name].size, dtype=jnp.float32)).reshape(result[name].shape)
    # The first precommit near-uniform physical-bin fixture failed the fixed
    # CPU oracle because extreme-bin cancellation amplified rounding. Preserve
    # that separate counterexample below; here the predefined asymmetric bias
    # suppresses outer bins and keeps a nonzero, well-conditioned prediction.
    result['rew/head/logits/bias'] = jnp.array([-30., -3., -1., 0., -30.],jnp.float32)
    return result
  agent.params = jax.jit(nonuniform)(agent.params)
  d, s, c = config.agent.dyn.rssm.deter, config.agent.dyn.rssm.stoch, config.agent.dyn.rssm.classes
  h = np.linspace(-.5, .5, d).astype(ml_dtypes.bfloat16)
  z = np.zeros((4, 2, s, c), ml_dtypes.bfloat16)
  for initial in range(4): z[initial, :, :, initial % c] = 1
  h, z = m.restore_native(h.astype(np.float64), z.astype(np.float64))
  controls = np.zeros((7, 20, 12), np.float32)
  for a in range(7): controls[a, 0, a] = .25
  saved_h, saved_z = h.copy(), z.copy()
  saved_counters = m.counters(agent)
  saved_state = m.host_copy(agent.params)
  adapter = m.NativeRewardAdapter(agent)
  roots = m.private_keys(9, 1, 0, 0)
  roots_again = m.private_keys(9, 1, 0, 0)
  np.testing.assert_array_equal(roots, roots_again)
  # Scalar independent named-key reference fixes all fold order/endpoints.
  def key_reference():
    outputs = []
    for labels in ((m.KEY_TAG,9,1,0,0,0,1), (m.KEY_TAG,9,1,3,0,7,20)):
      key = jax.random.PRNGKey(20261024)
      for label in labels: key = jax.random.fold_in(key, jnp.uint32(label))
      outputs.append(key)
    return jnp.stack(outputs)
  references = m.host_copy(jax.jit(key_reference)())
  np.testing.assert_array_equal(roots[0,0,0], references[0])
  np.testing.assert_array_equal(roots[3,7,19], references[1])
  key_errors = ((0,1,0,0),(9,2,0,0),(9,1,4,0),(9,1,0,1),(9,1,0,32),(9,1,False,0))
  for args in key_errors:
    try: m.private_keys(*args)
    except ValueError: pass
    else: raise AssertionError('Invalid scientific key labels accepted')
  no_public_rng = mock.patch.multiple(agent, _seeds=mock.Mock(side_effect=AssertionError('public seeds')),
      _eval_seeds=mock.Mock(side_effect=AssertionError('public eval seeds')))
  with no_public_rng, mock.patch.object(agent.model.dyn, 'observe',
      side_effect=AssertionError('Initial observe prohibited')):
    result = adapter.call_chunk(h, z, controls, roots)
    repeated = adapter.call_chunk(h, z, controls, roots_again)
    changed_roots = roots.copy(); changed_roots[:,:,9:] = m.private_keys(9,1,1,0)[:,:,9:]
    changed = adapter.call_chunk(h, z, controls, changed_roots)
    changed_controls = controls.copy(); changed_controls[:,19,0] = .25
    action_changed = adapter.call_chunk(h,z,changed_controls,roots)
  assert result['reward'].shape == (4,2,8,7,20)
  assert result['logits'].shape == (4,2,8,7,20,config.agent.rewhead.bins)
  np.testing.assert_array_equal(result['logits'][:,0], result['logits'][:,1])
  np.testing.assert_array_equal(result['logits'], repeated['logits'])
  np.testing.assert_array_equal(result['logits'][...,:9,:], changed['logits'][...,:9,:])
  assert np.any(result['logits'][...,9:,:] != changed['logits'][...,9:,:])
  np.testing.assert_array_equal(result['logits'][...,:19,:], action_changed['logits'][...,:19,:])
  assert np.any(result['logits'][...,19,:] != action_changed['logits'][...,19,:])
  assert np.any(result['logits'][0,...,0,:] != result['logits'][1,...,0,:])
  np.testing.assert_array_equal(h, saved_h); np.testing.assert_array_equal(z, saved_z)
  assert set(result) == {'reward','logits','bins','root_keys','sample_keys'}
  expected_samples = m.sample_keys(roots)
  np.testing.assert_array_equal(result['sample_keys'], expected_samples)
  # Unmodified official API is the one-step oracle, with the identical private root.
  def official(h1,z1,a1):
    carry, (feat, _) = agent.model.dyn.imagine(dict(deter=h1[None],stoch=z1[None]),
        dict(action=a1[None]),1,training=False,single=True)
    reward = agent.model.rew(agent.model.feat2tensor(feat),1)
    return carry, reward.pred(), reward.logits, feat['logit']
  official_jit = jax.jit(nj.pure(official))
  single = jax.jit(m.readonly_single_step(agent.model))
  inputs = internal.device_put((h,z[0,0],controls[0,0],roots[0,0,0]),agent.train_mirrored)
  _, oracle = official_jit(agent.params,*inputs[:3],seed=inputs[3],create=False,modify=False)
  direct = m.host_copy(single(agent.params,*inputs))
  assert bool(direct['rng_valid']) and int(direct['rng_consumption']) == 1
  oracle = m.host_copy(oracle)
  np.testing.assert_array_equal(direct['h'],oracle[0]['deter'][0])
  np.testing.assert_array_equal(direct['z'],oracle[0]['stoch'][0])
  np.testing.assert_array_equal(direct['logits'],oracle[2][0])
  np.testing.assert_array_equal(direct['reward'],oracle[1][0])
  np.testing.assert_array_equal(result['logits'][0,0,0,0,0],oracle[2][0])
  sample_jit = jax.jit(lambda logits,key:nn.cast(agent.model.dyn._dist(logits).sample(seed=key)))
  sampled = m.host_copy(sample_jit(internal.device_put(oracle[3],agent.train_mirrored),
      internal.device_put(expected_samples[0,0,0],agent.train_mirrored)))
  np.testing.assert_array_equal(sampled[0],direct['z'])
  # Native symmetric F32 point formula, NOT a left-to-right dot-product oracle.
  logits = result['logits']; bins = result['bins']
  ex = np.exp(logits-logits.max(-1,keepdims=True)); probs = ex/ex.sum(-1,keepdims=True)
  mid = (len(bins)-1)//2
  expected_pred = (probs[...,mid:mid+1]*bins[mid:mid+1]).sum(-1) + (
      (probs[...,:mid]*bins[:mid])[...,::-1]+probs[...,mid+1:]*bins[mid+1:]).sum(-1)
  np.testing.assert_allclose(result['reward'],expected_pred,atol=m.PRED_ATOL,rtol=m.PRED_RTOL)
  # An extreme-bin near-uniform counterexample is an oracle limitation, NOT a
  # reason to enlarge the real B tolerance. Both the native point prediction and
  # independent symmetric CPU formula are finite but their rounding is exposed.
  from embodied.jax.outs import TwoHot
  def cancellation_example():
    logs = jnp.array([0.,0.,0.,0.,1e-6],jnp.float32)
    half = nn.symexp(jnp.linspace(-20,0,3,dtype=jnp.float32))
    physical = jnp.concatenate([half,-half[:-1][::-1]])
    out = TwoHot(logs,physical)
    return logs,physical,out.pred()
  bad_logs,bad_bins,bad_native = m.host_copy(jax.jit(cancellation_example)())
  bad_exp = np.exp(bad_logs-bad_logs.max()); bad_probs = bad_exp/bad_exp.sum()
  bad_cpu = (bad_probs[2:3]*bad_bins[2:3]).sum() + (
      (bad_probs[:2]*bad_bins[:2])[::-1]+bad_probs[3:]*bad_bins[3:]).sum()
  assert np.isfinite(bad_cpu) and np.isfinite(bad_native)
  assert not np.isclose(bad_native,bad_cpu,atol=m.PRED_ATOL,rtol=m.PRED_RTOL)
  # Production _dist applies its unimix once; test enough draws to expose a
  # duplicated mixing operation, and compare actual categorical draws per key.
  mix_dyn = RSSM(dict(action=elements.Space(np.float32,(12,))),unimix=.4,name='mix_fixture')
  def frequencies():
    logits = jnp.log(jnp.array([.8,.1,.1],jnp.float32))[None]
    keys = jax.random.split(jax.random.PRNGKey(71),30000)
    samples = jax.vmap(lambda key:mix_dyn._dist(logits).sample(seed=key))(keys)
    return samples.mean((0,1))
  freq = m.host_copy(jax.jit(frequencies)())
  np.testing.assert_allclose(freq,[.613333333,.193333333,.193333333],atol=.007)
  # Tracking rejects even writing an unchanged value; a second latent seed is
  # also rejected, rather than silently relabeled as a single sample.
  original = agent.model.dyn.imagine
  def writes(*args,**kwargs):
    name = next(iter(nj.context()))
    nj.context()[name] = nj.context()[name]
    return original(*args,**kwargs)
  def extra_seed(*args,**kwargs):
    output = original(*args,**kwargs); nj.seed(); return output
  for replacement, expected_message in ((writes,'state writes'),(extra_seed,'exactly one')):
    agent.model.dyn.imagine = replacement
    try:
      rejecting = m.NativeRewardAdapter(agent)
      try: rejecting.call_chunk(h,z,controls,roots)
      except AssertionError as error: assert expected_message in str(error), str(error)
      else: raise AssertionError('Forbidden trace did not fail')
    finally: agent.model.dyn.imagine = original
  def creates(*args,**kwargs):
    nj.context()['diagnostic_forbidden_scratch'] = jnp.int32(0)
    return original(*args,**kwargs)
  agent.model.dyn.imagine = creates
  try:
    rejecting = m.NativeRewardAdapter(agent)
    try: rejecting.call_chunk(h,z,controls,roots)
    except RuntimeError as error: assert 'create=True' in str(error), str(error)
    else: raise AssertionError('Created model state silently passed')
  finally: agent.model.dyn.imagine = original
  with agent.n_eval_actions.lock: agent.n_eval_actions.value += 1
  try:
    try: adapter.call_chunk(h,z,controls,roots)
    except AssertionError as error: assert 'unchanged' in str(error), str(error)
    else: raise AssertionError('Changed production counter accepted')
  finally:
    with agent.n_eval_actions.lock: agent.n_eval_actions.value -= 1
  for bad_h,bad_z,bad_a,bad_k in ((h.astype(np.float32),z,controls,roots),
      (h,z[:3],controls,roots),(h,z,controls[:6],roots),
      (h,z,controls,roots[:,:7]),(h,z,np.full_like(controls,np.nan),roots)):
    try: adapter.call_chunk(bad_h,bad_z,bad_a,bad_k)
    except ValueError: pass
    else: raise AssertionError('Input contract bypass')
  assert m.counters(agent) == saved_counters
  current = m.host_copy(agent.params)
  for name in saved_state: np.testing.assert_array_equal(saved_state[name],current[name])
  assert adapter.accessed_keys and all(x.startswith(('dyn/','rew/')) for x in adapter.accessed_keys)
  print(json.dumps(dict(official_single_step_exact=True,next_reward_phase_exact=True,
      q_equals_p_exact=True,initial_samples_fixed=True,future_key_prefix_exact=True,
      future_action_prefix_exact=True,native_symmetric_pred_oracle=True,unimix_once=True,
      all_future_steps_sampled=True,named_root_keys_exact=True,private_call_order_exact=True,
      strict_transfer_guard=jax.config.jax_transfer_guard=='disallow',state_and_counters_unchanged=True,
      attempted_writes_rejected=True,extra_seed_rejected=True,input_guards=True,
      attempted_creation_rejected=True,counter_change_rejected=True,
      cancellation_counterexample_detected=True,
      scope='Actual tiny BF16 CPU only; no checkpoint, source arrays, physics or CUDA')))


if __name__ == '__main__':
  fixture() if sys.argv[1:] == ['--fixture'] else unittest.main()
