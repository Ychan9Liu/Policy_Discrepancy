"""DL-predictive-r1 CPU engineering properties; no scientific acceptance."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_predictive as p


class PredictiveTest(unittest.TestCase):
  def test_control_sequences_are_fixed_and_actual_actions_unmodified(self):
    original = np.linspace(-.9, .9, 120, dtype=np.float32).reshape(10, 12)
    before = original.copy()
    controls = p.control_sequences(original)
    self.assertEqual(controls.shape, (6, 10, 12))
    np.testing.assert_array_equal(controls[0], original)
    np.testing.assert_array_equal(original, before)
    for i in range(1, 6):
      np.testing.assert_array_equal(controls[i], np.repeat(controls[i, :1], 10, 0))
    for bad in (original[:9], original.astype(np.float64), original * np.nan, original * 2):
      with self.assertRaises(ValueError):
        p.control_sequences(bad)

  def test_real_constant_controls_and_recorded_full_window_restore(self):
    from scripts import dl_diagnostic as d
    for seed in (7, 17, 31):
      env = d.make_dmc('dmc_quadruped_walk', seed)
      try:
        env.reset()
        for _ in range(5):
          env.step(np.zeros(12))
        snapshot = d.physics_snapshot(env)
        sequence = np.linspace(-.25, .25, 120, dtype=np.float32).reshape(10, 12)
        recorded = np.asarray([env.step(a).reward for a in sequence], np.float32)
        for i, actions in enumerate(p.control_sequences(sequence)):
          reward, error, _ = d.verified_consequence(env, snapshot,
              actions[0], actions[1:], 10)
          self.assertEqual(error, 0.)
          if i == 0:
            np.testing.assert_array_equal(reward.astype(np.float32), recorded)
      finally:
        env.close()

  def test_actual_tiny_model_rollout_axes_no_seed_and_no_future_action_leak(self):
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--fixture'],
        cwd=ROOT, capture_output=True, text=True,
        env=dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT)))
    self.assertEqual(result.returncode, 0, result.stdout[-2500:] + result.stderr[-4000:])
    facts = json.loads(result.stdout.strip().splitlines()[-1])
    self.assertTrue(facts['q_equals_p_exact'])
    self.assertTrue(facts['h1_matches_existing_probe_exact'])
    self.assertTrue(facts['future_actions_do_not_change_prefix'])
    self.assertTrue(facts['state_unchanged'])

  def test_accurate_prediction_without_control_sensitivity_is_not_sufficient(self):
    rows = []
    for episode in range(8):
      logits = np.empty((16, 4, 2, 6, 10, 3))
      for variant in range(4):
        center = .8 if variant != 3 else .2
        logits[:, variant, 0] = np.log([(1-center)/2, center, (1-center)/2])
        logits[:, variant, 1] = np.log([.3, .4, .3])
      rows.append(dict(logits=logits, true_rewards=np.zeros((16, 6, 10)),
          episode=np.full(16, episode), physics_repeat_error=np.zeros(16),
          bins=np.array([-1,0,1], np.float32)))
    result = p.report(rows, [])
    self.assertTrue(result['prediction_skill_direction_supported'])
    self.assertFalse(result['control_sensitivity']['sufficient'])
    self.assertFalse(result['necessary_skill_supported_on_calibration'])
    self.assertFalse(result['training_authorized'])


def fixture():
  import elements
  import jax
  import jax.numpy as jnp
  import ninjax as nj
  from ruamel.yaml import YAML
  from dreamerv3.agent import Agent
  from dreamerv3 import dt_latch
  presets = YAML(typ='safe').load((ROOT / 'dreamerv3/configs.yaml').read_text())
  config = elements.Config(presets['defaults']).update(presets['debug'])
  config = elements.Config({**config.flat, **{'jax.platform': 'cpu',
      'jax.compute_dtype': 'float32', 'jax.precompile': False,
      'jax.profiler': False, 'jax.enable_policy': False,
      'batch_size': 2, 'batch_length': 3, 'replay_context': 0,
      'agent.imag_length': 2, 'agent.imag_last': 2,
      'agent.rep_probe.mode': 'off', 'agent.dt_latch.mode': 'off'}})
  agent = Agent(dict(vector=elements.Space(np.float32, (5,)),
      reward=elements.Space(np.float32), is_first=elements.Space(bool),
      is_last=elements.Space(bool), is_terminal=elements.Space(bool)),
      dict(action=elements.Space(np.float32, (2,))), elements.Config(**config.agent,
          logdir='/tmp/dl-predictive-fixture', seed=7, jax=config.jax,
          batch_size=2, batch_length=3, replay_context=0))
  def nonuniform(params):
    result = dict(params)
    key = 'rew/head/logits/kernel'
    result[key] = jnp.sin(jnp.arange(result[key].size, dtype=jnp.float32)).reshape(result[key].shape) * .03
    return result
  agent.params = jax.jit(nonuniform)(agent.params)
  carry = agent.init_train(3)
  h = carry[1]['deter']
  stoch, classes = config.agent.dyn.rssm.stoch, config.agent.dyn.rssm.classes
  action_values = p.control_sequences(np.linspace(-.2, .2, 20,
      dtype=np.float32).reshape(10, 2))
  pm, qm, actions = jax.jit(lambda: (
      jax.nn.one_hot(jnp.zeros((3, stoch), jnp.int32), classes),
      jax.nn.one_hot(jnp.ones((3, stoch), jnp.int32), classes),
      jnp.asarray(action_values)))()
  rollout = p.readonly_jit(lambda h, q, z, a: p.model_rollout(agent.model, h, q, z, a))
  state, same = rollout(agent.params, h, pm, pm, actions)
  _, result = rollout(agent.params, h, qm, pm, actions)
  altered = jax.jit(lambda a: a.at[:, 9].set(-a[:, 9]))(actions)
  _, changed = rollout(agent.params, h, qm, pm, altered)
  def h1(deter, q, z, a):
    q, z = dt_latch.one_step(agent.model, jnp.repeat(deter, 6, 0),
        jnp.repeat(q, 6, 0), jnp.repeat(z, 6, 0),
        {'action': jnp.tile(a[:, 0], (3, 1))})
    return jnp.stack([q.logits.reshape(3, 6, -1), z.logits.reshape(3, 6, -1)], 1)
  _, first = jax.jit(nj.pure(h1))(agent.params, h, qm, pm, actions)
  with jax._src.config.explicit_device_get_scope():
    same, result, changed, first = [jax.tree.map(np.asarray, x) for x in (same, result, changed, first)]
    np.testing.assert_array_equal(same['logits'][:, 0], same['logits'][:, 1])
    np.testing.assert_array_equal(result['logits'][..., 0, :], first)
    np.testing.assert_array_equal(result['logits'][..., :9, :], changed['logits'][..., :9, :])
    for key in state:
      np.testing.assert_array_equal(np.asarray(state[key]), np.asarray(agent.params[key]))
  print(json.dumps(dict(version=p.VERSION, q_equals_p_exact=True,
      h1_matches_existing_probe_exact=True, future_actions_do_not_change_prefix=True,
      state_unchanged=True, scope='Tiny float32 CPU model; no actual size50m skill acceptance')))


if __name__ == '__main__':
  if sys.argv[1:] == ['--fixture']:
    fixture()
  else:
    unittest.main()
