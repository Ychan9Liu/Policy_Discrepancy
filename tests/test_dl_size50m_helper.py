"""DL-engineering-r5 tiny production Agent regression, CPU only.

Executes the same accessed-parameter probe gradient helper used by the real
size50m worker. This is a small engineering fixture, not size50m evidence.
"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Size50mHelperTest(unittest.TestCase):

  def test_production_agent_probe_accessed_parameter_gradients(self):
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
        '--fixture'], cwd=ROOT, capture_output=True, text=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT), JAX_PLATFORMS='cpu'))
    self.assertEqual(result.returncode, 0,
        result.stdout[-4000:] + result.stderr[-4000:])
    report = json.loads(result.stdout.strip().splitlines()[-1])
    self.assertEqual(report['code_version'], 'DL-engineering-r5')
    self.assertGreater(report['tested_count'], 0)
    self.assertGreater(report['unaccessed_state_count'], 0)
    self.assertEqual(report['tested_modules'], ['dyn', 'pol', 'rew'])
    self.assertEqual(report['probe_gradient_norm'], 0.)
    self.assertTrue(report['state_unchanged'])
    self.assertTrue(report['logging_sampling_exact'])
    self.assertTrue(report['logging_state_unchanged'])


def production_agent_fixture():
  # Optional process-local Windows C++ runtime path; no system modifications.
  dll_directory = (os.add_dll_directory(os.environ['DL_DLL_DIRECTORY'])
      if os.name == 'nt' and os.environ.get('DL_DLL_DIRECTORY') else None)
  sys.path.insert(0, str(ROOT))
  import elements
  import embodied.jax.internal as internal
  import jax
  import jax.numpy as jnp
  import ninjax as nj
  import numpy as np
  import optax
  from ruamel.yaml import YAML
  from dreamerv3.agent import Agent

  spec = importlib.util.spec_from_file_location('dl_size50m_check',
      Path(__file__).with_name('dl_size50m_check.py'))
  helper = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(helper)
  configs = YAML(typ='safe').load((ROOT / 'dreamerv3/configs.yaml').read_text(
      encoding='utf-8'))
  config = elements.Config(configs['defaults']).update(configs['debug'])
  config = elements.Config({**config.flat, **{
      'batch_size': 2, 'batch_length': 3, 'replay_context': 0,
      'jax.platform': 'cpu', 'jax.compute_dtype': 'float32',
      'jax.train_devices': [0], 'jax.policy_devices': [0],
      'jax.precompile': False, 'jax.profiler': False,
      'jax.enable_policy': False, 'agent.imag_length': 2,
      'agent.imag_last': 2, 'agent.rep_probe.mode': 'off',
      'agent.dt_latch.mode': 'logging'}})
  obs_space = dict(vector=elements.Space(np.float32, (5,)),
      reward=elements.Space(np.float32), is_first=elements.Space(bool),
      is_last=elements.Space(bool), is_terminal=elements.Space(bool))
  act_space = {'action': elements.Space(np.float32, (2,))}
  agent = Agent(obs_space, act_space, elements.Config(**config.agent,
      logdir='/tmp/dl-engineering-r3-fixture', seed=7, jax=config.jax,
      batch_size=2, batch_length=3, replay_context=0))
  # Exercise a nonuniform real reward head rather than its zero initializer.
  def nonuniform(params):
    key = 'rew/head/logits/kernel'
    params = dict(params)
    params[key] = jnp.sin(jnp.arange(params[key].size, dtype=jnp.float32)).reshape(
        params[key].shape) * .03
    return params
  agent.params = jax.jit(nonuniform)(agent.params)
  raw = agent._zeros(agent.spaces, (2, 3))
  raw['vector'][:] = np.arange(30, dtype=np.float32).reshape(2, 3, 5) / 30
  raw['reward'][:] = [[-.5, .2, 1.], [.7, -.1, .5]]
  raw['is_first'][:, 0] = True
  raw['is_last'][:, -1] = True
  raw['is_terminal'][:, -1] = True
  raw['action'][:] = .2
  data = internal.device_put(raw, agent.train_sharded)
  carry = agent.init_train(2)
  loss_carry, obs, prevact, _ = jax.jit(agent.model._apply_replay_context)(carry, data)
  seed = agent._seeds(0, agent.train_mirrored)
  loss_fn = jax.jit(nj.pure(lambda c, o, p: agent.model.loss(c, o, p, True)))
  _, (_, aux) = loss_fn(agent.params, loss_carry, obs, prevact, seed=seed)
  feat = aux[2]['repfeat']
  gradient_fn = jax.jit(nj.pure(lambda f, o, p: helper.probe_shared_gradients(
      agent.model, f, o, p, 20., .5)))
  state, (_, selected, gradients) = gradient_fn(
      agent.params, feat, obs, prevact, seed=seed)
  jax.block_until_ready(gradients)
  assert set(selected) == set(gradients)
  assert set(gradients) < set(agent.params)
  modules = sorted({key.split('/')[0] for key in gradients})
  assert modules == ['dyn', 'pol', 'rew'], modules
  for block in ('dynin0', 'dynin1', 'dynin2', 'dynhid0', 'dyngru'):
    assert any(key.startswith(f'dyn/{block}/') for key in gradients), block
  assert any(key.startswith('dyn/prior') for key in gradients)
  # No unaccessed continuation/value/encoder/decoder parameter is claimed tested.
  assert not any(key.startswith(('con/', 'val/', 'enc/', 'dec/')) for key in gradients)
  with jax._src.config.explicit_device_get_scope():
    for key, value in gradients.items():
      np.testing.assert_array_equal(np.asarray(value), np.zeros_like(np.asarray(value)),
          err_msg=key)
    for key in agent.params:
      np.testing.assert_array_equal(np.asarray(state[key]), np.asarray(agent.params[key]))
    norm = float(jax.device_get(jax.jit(optax.global_norm)(gradients)))
  assert norm == 0.
  grad_fn = jax.jit(nj.pure(lambda c, o, p: nj.grad(
      agent.model.loss, agent.model.modules, has_aux=True)(
          c, o, p, True, dl_logging=False)))
  _, (_, _, _, grad_aux) = grad_fn(agent.params, loss_carry, obs, prevact, seed=seed)
  log_fn = jax.jit(lambda s, c, d, k: helper.read_only_logging(agent.model, s, c, d, k))
  log_state, (log_metrics, log_features, accessed) = log_fn(
      agent.params, carry, data, seed)
  jax.block_until_ready(log_features)
  with jax._src.config.explicit_device_get_scope():
    for a, b in zip(jax.tree.leaves(log_features), jax.tree.leaves(dict(
        tokens=grad_aux[2]['tokens'], repfeat=grad_aux[2]['repfeat']))):
      np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    for key in agent.params:
      np.testing.assert_array_equal(np.asarray(log_state[key]), np.asarray(agent.params[key]))
    assert int(np.asarray(accessed)) > 0
    assert float(np.asarray(log_metrics['dl/actual_v_mean'])) == 1.
  print(json.dumps(dict(code_version=helper.VERSION, fixture='tiny production Agent CPU',
      tested_count=len(gradients), unaccessed_state_count=len(agent.params) - len(gradients),
      tested_modules=modules, probe_gradient_norm=norm, state_unchanged=True,
      logging_sampling_exact=True, logging_state_unchanged=True)))


if __name__ == '__main__':
  if sys.argv[1:] == ['--fixture']:
    production_agent_fixture()
  else:
    unittest.main()
