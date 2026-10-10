"""CPU logging isolation: real tiny Agent updates, never rendering or a GPU.

Full state equality includes optimizer moments, normalization and replay carry.
It does not establish size50m CUDA equality or scientific method effectiveness.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def snapshot(mode, output):
  import elements
  import embodied.jax.internal as internal
  import jax
  import jax.numpy as jnp
  import ninjax as nj
  from ruamel.yaml import YAML
  from dreamerv3.agent import Agent

  presets = YAML(typ='safe').load((ROOT / 'dreamerv3/configs.yaml').read_text())
  config = elements.Config(presets['defaults']).update(presets['debug'])
  config = elements.Config({**config.flat, **{
      'batch_size': 2, 'batch_length': 3, 'replay_context': 1,
      'jax.platform': 'cpu', 'jax.compute_dtype': 'float32',
      'jax.train_devices': [0], 'jax.precompile': False,
      'jax.profiler': False, 'jax.enable_policy': False,
      'agent.imag_length': 2, 'agent.imag_last': 2,
      'agent.dyn.rssm.free_nats': 0., 'agent.rep_probe.mode': 'off',
      'agent.dt_latch.mode': 'logging' if mode == 'logging' else 'off'}})
  flat = config.agent.flat
  if mode == 'baseline':
    flat = {k: v for k, v in flat.items() if not k.startswith('dt_latch.')}
  agent = Agent(dict(vector=elements.Space(np.float32, (5,)),
      reward=elements.Space(np.float32), is_first=elements.Space(bool),
      is_last=elements.Space(bool), is_terminal=elements.Space(bool)),
      dict(action=elements.Space(np.float32, (2,))), elements.Config(**flat,
          logdir=str(Path(output).parent / mode), seed=7, jax=config.jax,
          batch_size=2, batch_length=3, replay_context=1))
  assert hasattr(agent, '_log_collect') == (mode == 'logging')
  data = agent._zeros(agent.spaces, (2, 4))
  data['vector'][:] = np.arange(40, dtype=np.float32).reshape(2, 4, 5) / 40
  data['reward'][:] = [[0, -.5, .25, 1], [.1, .5, -.25, .75]]
  data['is_first'][:, 0] = True
  data['is_last'][:, -1] = True
  data['is_terminal'][0, -1] = True
  data['action'][:] = np.linspace(-.3, .3, 16).reshape(2, 4, 2)
  # Exercise both cached and ordinary replay-context paths in the same batch.
  data['consec'][1, 0] = 1
  carry = agent.init_train(2)
  prefix_obs = {k: v[:, :1] for k, v in data.items() if k in agent.obs_space}
  prefix_act = {k: np.zeros_like(v[:, :1]) for k, v in data.items() if k in agent.act_space}
  prefix_obs, prefix_act = internal.device_put((prefix_obs, prefix_act), agent.train_sharded)
  def context_entries(ca, ob, pa):
    enc, dyn, _, _ = ca
    _, encent, tokens = agent.model.enc(enc, ob, ob['is_first'], training=False)
    _, dynent, _ = agent.model.dyn.observe(dyn, tokens, pa, ob['is_first'], training=False)
    return elements.tree.flatdict(dict(enc=encent, dyn=dynent))
  _, entries = jax.jit(nj.pure(context_entries))(agent.params, carry,
      prefix_obs, prefix_act, seed=agent._seeds(0x444C, agent.train_mirrored))
  with jax._src.config.explicit_device_get_scope():
    for key, value in entries.items():
      data[key][:, :1] = np.asarray(value)

  arrays, log_keys, sampling_checks = {}, [], 0
  def host(tree):
    with jax._src.config.explicit_device_get_scope():
      return jax.tree.map(lambda x: np.array(x, copy=True), jax.device_get(tree))
  def exact(lhs, rhs):
    left, ls = jax.tree.flatten(host(lhs))
    right, rs = jax.tree.flatten(host(rhs))
    assert ls == rs
    for a, b in zip(left, right):
      np.testing.assert_array_equal(a, b)
  def add(prefix, tree):
    for path, value in jax.tree_util.tree_flatten_with_path(host(tree))[0]:
      arrays[prefix + '/'.join(str(x) for x in path)] = value

  add('initial/', agent.params)
  add('context/', entries)
  grad_fn = jax.jit(nj.pure(lambda c, o, p: nj.grad(
      agent.model.loss, agent.model.modules, has_aux=True)(
          c, o, p, True, dl_logging=False)))
  for step in range(2):
    raw = {k: np.array(v, copy=True) for k, v in data.items()}
    if step:
      # Keep the real cached prefix, change the subsequent consumed batch.
      raw['vector'][:, 1:] *= -.75
      raw['reward'][:, 1:] += .125
      raw['action'][:, 1:] *= -1
    add(f'step{step}/input/', raw)
    prepared = agent.prepare_batch(raw)
    seed = prepared['seed']
    batch = {k: v for k, v in prepared.items() if k != 'seed'}
    loss_carry, obs, previous, _ = jax.jit(agent.model._apply_replay_context)(carry, batch)
    _, (loss, _, grads, aux) = grad_fn(agent.params, loss_carry, obs, previous, seed=seed)
    jax.block_until_ready(aux)
    add(f'step{step}/grads/', grads)
    add(f'step{step}/sampled/', aux[2]['repfeat'])
    add(f'step{step}/loss/', aux[2]['losses'])
    add(f'step{step}/rng/', seed)
    if mode == 'logging':
      before = host(agent.params)
      log, features = agent.log_train_diagnostics(carry, batch, seed,
          return_features=True)
      exact(before, agent.params)
      exact(features, dict(tokens=aux[2]['tokens'], repfeat=aux[2]['repfeat']))
      sampling_checks += 1
      assert 'dl/D_valid_mean' in log and float(host(log['dl/valid_count'])) > 0
      assert float(host(log['dl/actual_v_mean'])) == 1.
    carry, _, _ = agent.train(carry, dict(prepared))
    _, metrics = agent.take_train_result()
    if mode == 'logging':
      log_keys = sorted(k for k in metrics if k.startswith('dl/'))
      assert log_keys and 'dl/D_valid_mean' in log_keys
    else:
      assert not any(k.startswith('dl/') for k in metrics)
    add(f'step{step}/params/', agent.params)
    add(f'step{step}/carry/', carry)
    add(f'step{step}/counters/', agent.save()['counters'])
    # Normalization and every optimizer state entry are part of params above.
    add(f'step{step}/baseline_metrics/', {k: v for k, v in metrics.items()
        if not k.startswith('dl/')})
  write_rejected = False
  if mode == 'logging':
    original_score = agent.model.log_train_score
    def attempted_write(feat, rep, ob, pa):
      result = original_score(feat, rep, ob, pa)
      agent.model.opt.step.write(agent.model.opt.step.read() + 1)
      return result
    agent.model.log_train_score = attempted_write
    before = host(agent.params)
    try:
      jax.jit(agent._log_score_read_only)(agent.params, aux[2]['repfeat'],
          aux[2]['losses']['rep'], obs, previous, seed=seed)
    except RuntimeError as error:
      assert 'opt/step' in str(error)
      write_rejected = True
    finally:
      agent.model.log_train_score = original_score
    assert write_rejected
    exact(agent.params, before)
    original_loss = agent.model.loss
    def collector_write(*args, **kwargs):
      result = original_loss(*args, **kwargs)
      agent.model.opt.step.write(agent.model.opt.step.read() + 1)
      return result
    agent.model.loss = collector_write
    collector_rejected = False
    try:
      # A fresh function forces tracing after fault injection; jitting the same
      # previously compiled callable would legitimately reuse its old program.
      jax.jit(lambda *a, **k: agent._log_collect_read_only(*a, **k))(
          agent.params, loss_carry,
          obs, previous, seed=seed)
    except RuntimeError as error:
      assert 'opt/step' in str(error)
      collector_rejected = True
    finally:
      agent.model.loss = original_loss
    assert collector_rejected
    exact(agent.params, before)
  np.savez(output, **arrays)
  Path(str(output) + '.json').write_text(json.dumps(dict(mode=mode,
      updates=2, sampling_exact_checks=sampling_checks, dl_metric_keys=log_keys,
      attempted_state_write_rejected=write_rejected,
      replay_context='one cached row and one ordinary row',
      scope='Tiny vector CPU Agent; no size50m CUDA or method effectiveness claim')))


class DTLoggingIsolationTest(unittest.TestCase):

  def test_logging_does_not_change_two_complete_updates_or_loss_sampling(self):
    with tempfile.TemporaryDirectory(prefix='dl-logging-') as temporary:
      paths = {}
      for mode in ('baseline', 'off', 'logging'):
        path = Path(temporary) / f'{mode}.npz'
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
            '--snapshot', mode, str(path)], cwd=ROOT, capture_output=True, text=True,
            env=dict(os.environ, PYTHONPATH=str(ROOT), JAX_PLATFORMS='cpu',
                CUDA_VISIBLE_DEVICES=''))
        self.assertEqual(result.returncode, 0, result.stdout[-3500:] + result.stderr[-3500:])
        paths[mode] = path
      with np.load(paths['baseline']) as reference:
        for mode in ('off', 'logging'):
          with np.load(paths[mode]) as candidate:
            self.assertEqual(set(reference.files), set(candidate.files))
            for key in reference.files:
              np.testing.assert_array_equal(reference[key], candidate[key], err_msg=f'{mode}: {key}')
      proof = json.loads(Path(str(paths['logging']) + '.json').read_text())
      self.assertEqual(proof['sampling_exact_checks'], 2)
      self.assertIn('dl/D_valid_mean', proof['dl_metric_keys'])
      self.assertTrue(proof['attempted_state_write_rejected'])


if __name__ == '__main__':
  if len(sys.argv) == 4 and sys.argv[1] == '--snapshot':
    snapshot(sys.argv[2], sys.argv[3])
  else:
    unittest.main()
