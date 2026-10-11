"""Real CPU action-path evidence; fixture does not validate a research method."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_action_contract as a


class ActionContractTest(unittest.TestCase):
  def test_actual_wrappers_and_full_physics_without_render(self):
    from dm_control.mujoco.engine import Physics
    with mock.patch.object(Physics, 'render', side_effect=AssertionError('No CPU render')):
      report = a.run_cpu_audit()
    self.assertEqual(report['action_spec']['minimum'], [-1, -1, -.8]*4)
    self.assertEqual(report['action_spec']['maximum'], [1, 1.1, .8]*4)
    self.assertEqual(report['nonidentity_actuator_bounds'], 8)
    self.assertEqual(len(report['cases']), 8)
    self.assertTrue(all(c['mapping_byte_exact'] and c['raw_unchanged'] and
        c['explicit_physics_full_state_exact'] for c in report['cases']))
    self.assertTrue(any(c['legacy_max_integration_difference'] > 1e-8 for c in report['cases']))
    self.assertTrue(any(c['legacy_max_qvel_difference'] > 1e-8 for c in report['cases']))
    rounding = next(c for c in report['cases'] if c['name'] == 'rounding')
    self.assertNotEqual(rounding['stages']['applied']['values'][0],
        rounding['stages']['raw']['values'][0])  # Even physical [-1, 1] is not byte identity.
    zero = next(c for c in report['cases'] if c['name'] == 'zero')
    self.assertAlmostEqual(zero['stages']['applied']['values'][1], .05)
    self.assertEqual(zero['stages']['applied']['dtype'], 'float64')

  def test_wrong_dtype_nonfinite_and_wrong_shape_are_rejected(self):
    low, high = np.array([-1, -1, -.8]*4), np.array([1, 1.1, .8]*4)
    raw = np.zeros(12, np.float32)
    for bad in (raw.astype(np.float64), raw[:11], raw*float('nan')):
      with self.assertRaises(ValueError):
        a.production_action_stages(bad, low, high)
    with self.assertRaises(ValueError):
      a.production_action_stages(raw, low, high*float('inf'))
    result = a.production_action_stages(raw, low, high)
    result['clipped'][0] = .7
    np.testing.assert_array_equal(raw, np.zeros(12, np.float32))
    np.testing.assert_array_equal(result['raw'], raw)

  def test_actual_tiny_policy_returns_raw_prevact_and_wrapper_cannot_rewrite_it(self):
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--fixture'],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
        env=dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT)))
    self.assertEqual(child.returncode, 0, child.stdout[-3500:]+child.stderr[-4500:])
    facts = json.loads(child.stdout.strip().splitlines()[-1])
    for key in ('carry_is_actual_raw_sample', 'raw_sample_outside_normalized_range',
        'wrapper_does_not_rewrite_raw_or_carry', 'model_state_unchanged',
        'wrapper_mapping_exact', 'next_observe_receives_raw_prevact'):
      self.assertTrue(facts[key], key)


def fixture():
  dll = (os.add_dll_directory(os.environ['DL_DLL_DIRECTORY'])
      if os.name == 'nt' and os.environ.get('DL_DLL_DIRECTORY') else None)
  import elements
  import jax
  import jax.numpy as jnp
  from ruamel.yaml import YAML
  from scripts import dl_predictive_mc as mc
  from scripts.dl_predictive import readonly_jit
  presets = YAML(typ='safe').load((ROOT/'dreamerv3/configs.yaml').read_text(encoding='utf-8'))
  config = elements.Config(presets['defaults']).update(presets['debug'])
  config = elements.Config({**config.flat, **{'jax.platform': 'cpu', 'jax.compute_dtype': 'float32',
      'jax.precompile': False, 'jax.profiler': False, 'jax.enable_policy': False,
      'batch_size': 2, 'batch_length': 3, 'replay_context': 0,
      'agent.imag_length': 2, 'agent.imag_last': 2,
      'agent.rep_probe.mode': 'off', 'agent.dt_latch.mode': 'off'}})
  config = config.update(logdir='/tmp/dl-action-contract-fixture', seed=7)
  agent = mc.agent_from_spaces(config)
  carry = agent.init_train(8)
  obs = jax.jit(lambda: dict(image=jnp.zeros((8,64,64,3), jnp.uint8),
      reward=jnp.zeros(8, jnp.float32), is_first=jnp.ones(8, bool),
      is_last=jnp.zeros(8, bool), is_terminal=jnp.zeros(8, bool)))()
  key = jax.jit(lambda: jax.random.PRNGKey(77))()
  run = readonly_jit(lambda c, o: agent.model.policy(c, o, mode='eval'))
  state, (nextcarry, actions, _) = run(agent.params, carry, obs, seed=key)
  raw = np.asarray(jax.device_get(actions['action']))
  previous = np.asarray(jax.device_get(nextcarry[3]['action']))
  np.testing.assert_array_equal(raw, previous)
  assert np.any(np.abs(raw) > 1)
  assert all(np.array_equal(jax.device_get(v), jax.device_get(state[k]))
      for k, v in agent.params.items())
  env, recorder, wrapped = a.make_wrapped_suite(20261020)
  try:
    wrapped.step(dict(reset=np.bool_(True), action=np.zeros(12, np.float32)))
    before_raw, before_previous = raw.copy(), previous.copy()
    wrapped.step(dict(reset=np.bool_(False), action=raw[0]))
    stages = a.production_action_stages(raw[0], env.action_spec().minimum, env.action_spec().maximum)
    assert a._bytes_equal(stages['applied'], recorder.last_action)
    np.testing.assert_array_equal(raw, before_raw)
    np.testing.assert_array_equal(previous, before_previous)
  finally:
    env.close()
  # Inspect the actual next Agent.policy -> dyn.observe argument. Its _core
  # performs its own existing normalization; the interface still receives raw.
  incoming = []
  original_observe = agent.model.dyn.observe
  def capture(c, tokens, prevact, reset, **kwargs):
    incoming.append(prevact['action'])
    return original_observe(c, tokens, prevact, reset, **kwargs)
  second_obs = jax.jit(lambda o: {**o, 'is_first': jnp.zeros(8, bool)})(obs)
  agent.model.dyn.observe = capture
  try:
    rerun = readonly_jit(lambda c, o: agent.model.policy(c, o, mode='eval'))
    rerun(agent.params, nextcarry, second_obs, seed=key)
    assert len(incoming) == 1
    # incoming is a traced object; expose the same actual argument via wrapper
    # return so the equality check uses device data rather than a Python tracer.
    def call_with_receipt(c, o):
      incoming.clear()
      result = agent.model.policy(c, o, mode='eval')
      return result, incoming[-1]
    _, (_, received) = readonly_jit(call_with_receipt)(agent.params, nextcarry, second_obs, seed=key)
    np.testing.assert_array_equal(jax.device_get(received), raw)
  finally:
    agent.model.dyn.observe = original_observe
  print(json.dumps(dict(carry_is_actual_raw_sample=True,
      raw_sample_outside_normalized_range=True, wrapper_does_not_rewrite_raw_or_carry=True,
      model_state_unchanged=True, wrapper_mapping_exact=True,
      next_observe_receives_raw_prevact=True)))


if __name__ == '__main__':
  if '--fixture' in sys.argv:
    fixture()
  else:
    unittest.main()
