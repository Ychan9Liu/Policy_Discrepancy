"""DL-NM1 finite normal-only contracts, never N1M or method acceptance."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

_dll_directory = (os.add_dll_directory(os.environ['DL_DLL_DIRECTORY'])
    if os.name == 'nt' and os.environ.get('DL_DLL_DIRECTORY') else None)

import jax
import jax.numpy as jnp
import numpy as np

from dreamerv3 import dt_latch
from tests.test_dt_latch import FakeAgent, fixture

ROOT = Path(__file__).resolve().parents[1]


def config(normal_only=None, mode='dtlatch', **changes):
  values = dict(mode=mode, alpha=20., rho=.5, kappa=.1,
      kl_tolerance=1e-5, d_bins=[1.], k_bins=[1.])
  if normal_only is not None:
    values['normal_only'] = normal_only
  values.update(changes)
  return SimpleNamespace(**values)


def legacy_overlay(agent, repfeat, prior_logit, rep_raw, rep_before, obs,
                   prevact, cfg, seed=0, step=0):
  """Frozen pre-NM1 overlay/calibration, independent regression reference."""
  signals = dt_latch.probe(agent, repfeat, prior_logit, obs, prevact,
      cfg.alpha, cfg.rho, cfg.kl_tolerance)
  m, d, e, f = [signals[k] for k in ('m', 'D', 'e', 'f_D')]
  dt_latch.require_valid(~m | (jnp.isfinite(rep_raw) &
      (rep_raw >= -cfg.kl_tolerance)))
  weight, shuffle_metrics = signals['v'], {}
  if cfg.mode == 'logging':
    weight = jnp.ones_like(weight)
  elif cfg.mode == 'strength':
    weight = jax.lax.stop_gradient(1-m*jnp.float32(cfg.kappa)*f)
  elif cfg.mode == 'reward_only':
    weight = jax.lax.stop_gradient(1-m*jnp.float32(cfg.rho)*e)
  elif cfg.mode == 'conditional_shuffle':
    weight, shuffle_metrics, _ = dt_latch.conditional_shuffle(weight, m,
        rep_raw, d, agent.dyn.free_nats, cfg.d_bins, cfg.k_bins, seed, step)
  elif cfg.mode != 'dtlatch':
    raise ValueError(cfg.mode)
  tau = agent.dyn.free_nats
  active, excess = rep_raw > tau, jnp.maximum(rep_raw-tau, 0)
  # Original calibration_stats is a separate call and recomputes excess.
  cal_excess = jnp.maximum(rep_raw-tau, 0)
  base = jnp.where(signals['m'], cal_excess*signals['f_D'], 0)
  num = jax.lax.stop_gradient((base*signals['e']).sum())
  den = jax.lax.stop_gradient(base.sum())
  valid_mean = lambda x: jnp.where(m.sum() > 0,
      jnp.where(m, x, jnp.zeros_like(x)).sum()/m.sum(), jnp.nan)
  metrics = {f'{key}_valid_mean': valid_mean(signals[key]) for key in (
      'ell_q', 'ell_p', 'd_q', 'd_p', 'u_q', 'u_p', 'loss_gap',
      'signed_support', 'e', 'D')}
  metrics.update(valid_count=m.sum(), unavailable_count=(~m).sum(),
      valid_frac=m.mean(), reward_outside_bins_count=signals['reward_outside_bins'].sum(),
      candidate_v_mean=signals['v'].mean(), actual_v_mean=weight.mean(),
      actual_v_min=weight.min(), actual_v_max=weight.max(),
      active_frac=active.mean(), valid_active_frac=(m & active).mean(),
      active_protected_frac=(m & active & (weight < 1-1e-6)).mean(),
      excess_release_mean=((1-weight)*excess).mean(),
      candidate_excess_release_mean=((1-signals['v'])*excess).mean(),
      weighted_excess_mean=(weight*excess).mean(),
      rep_before_mean=rep_before.mean(), rep_after_mean=(weight*rep_before).mean(),
      calibration_num=num, calibration_den=den, **shuffle_metrics)
  return jax.lax.stop_gradient(weight), {
      f'dl/{key}': jax.lax.stop_gradient(value) for key, value in metrics.items()}, signals


def ordinary_fixture():
  feat, prior, obs, act = fixture(b=1, t=6)
  obs['reward'] = jnp.ones((1, 6), jnp.float32)
  obs['is_first'] = jnp.array([[True, False, False, True, False, False]])
  obs['is_last'] = jnp.array([[False, False, True, False, True, False]])
  obs['is_terminal'] = jnp.array([[False, False, False, False, True, False]])
  return feat, prior, obs, act


class NormalOnlyTests(unittest.TestCase):
  def test_default_missing_and_false_are_exact_frozen_graph_and_values(self):
    feat, prior, obs, act = ordinary_fixture()
    raw = jnp.array([[2., 3., 4., 5., 6., .2]], jnp.float32)
    before = jnp.maximum(raw, 1.)
    args = (FakeAgent(), feat, prior, raw, before, obs, act)
    old = lambda x: legacy_overlay(*args[:3], x, jnp.maximum(x, 1.),
        *args[5:], config())
    # JAX creates a fresh DebugCallback Python wrapper for each trace. Keep all
    # callback/effect nodes; only its nondeterministic object address is erased.
    graph = lambda fn: re.sub(r' at 0x[0-9A-Fa-f]+', ' at <object-address>',
        str(jax.make_jaxpr(fn)(raw)))
    old_graph = graph(old)
    reference = jax.jit(old)(raw)
    for cfg in (config(), config(False)):
      new = lambda x: dt_latch.overlay(*args[:3], x, jnp.maximum(x, 1.),
          *args[5:], cfg)
      self.assertEqual(graph(new), old_graph)
      actual = jax.jit(new)(raw)
      left, ls = jax.tree.flatten(reference)
      right, rs = jax.tree.flatten(actual)
      self.assertEqual(ls, rs)
      for a, b in zip(left, right):
        self.assertEqual(np.asarray(a).dtype, np.asarray(b).dtype)
        self.assertEqual(np.asarray(a).tobytes(), np.asarray(b).tobytes())

  def test_h1_probe_is_unchanged_but_only_ordinary_interventions_are_eligible(self):
    feat, prior, obs, act = ordinary_fixture()
    raw = jnp.full((1, 6), 3., jnp.float32)
    original = dt_latch.probe(FakeAgent(), feat, prior, obs, act)
    weight, metrics, signals = dt_latch.overlay(FakeAgent(), feat, prior,
        raw, raw, obs, act, config(True))
    np.testing.assert_array_equal(original['m'], [[True, True, False, True, False, False]])
    np.testing.assert_array_equal(signals['m_intervention'], [[False, True, False, False, False, False]])
    for key in original:
      np.testing.assert_array_equal(signals[key], original[key], err_msg=key)
    self.assertLess(float(original['v'][0, 0]), 1.)
    self.assertLess(float(original['v'][0, 3]), 1.)
    self.assertLess(float(weight[0, 1]), 1.)
    np.testing.assert_array_equal(weight[0, [0, 2, 3, 4, 5]], 1.)
    self.assertEqual(float(metrics['dl/valid_count']), 3.)
    self.assertEqual(float(metrics['dl/intervention_valid_count']), 1.)
    num, den = dt_latch.calibration_stats(raw, signals, 1.)
    self.assertAlmostEqual(float(den), float(2*signals['f_D'][0, 1]), places=6)
    self.assertAlmostEqual(float(num), float(den*signals['e'][0, 1]), places=6)
    self.assertAlmostEqual(float(metrics['dl/candidate_excess_release_mean']),
        float(metrics['dl/excess_release_mean']), places=7)

  def test_actual_categorical_rep_gradient_keeps_reset_and_original_mean(self):
    feat, prior, obs, act = ordinary_fixture()
    agent = FakeAgent()
    def loss(qlog, normal=None):
      raw = agent.dyn._dist(qlog).kl(agent.dyn._dist(jax.lax.stop_gradient(prior)))
      rep = jnp.maximum(raw, agent.dyn.free_nats)
      if normal is not None:
        weight = dt_latch.overlay(agent, dict(feat, logit=qlog), prior,
            raw, rep, obs, act, config(normal))[0]
        rep = weight*rep
      return rep.mean()
    raw = agent.dyn._dist(feat['logit']).kl(agent.dyn._dist(prior))
    before = jnp.maximum(raw, 1.)
    weight, metrics, _ = dt_latch.overlay(agent, feat, prior, raw, before,
        obs, act, config(True))
    baseline = jax.grad(loss)(feat['logit'])
    candidate = jax.grad(lambda q: loss(q, True))(feat['logit'])
    legacy = jax.grad(lambda q: loss(q, False))(feat['logit'])
    np.testing.assert_allclose(candidate, weight[..., None, None]*baseline,
        rtol=2e-6, atol=1e-8)
    for at in (0, 3):
      self.assertGreater(float(jnp.linalg.norm(baseline[:, at])), 0.)
      np.testing.assert_array_equal(candidate[:, at], baseline[:, at])
      self.assertLess(float(jnp.linalg.norm(legacy[:, at])),
          float(jnp.linalg.norm(baseline[:, at])))
    # Other/unavailable positions retain their original loss in the full BT mean.
    expected = np.asarray(before).sum()/before.size
    release = np.asarray((1-weight)*before).sum()/before.size
    self.assertAlmostEqual(float(loss(feat['logit'], True)), expected-release, places=6)
    self.assertAlmostEqual(float(metrics['dl/rep_after_mean']), expected-release, places=6)
    below = jnp.full_like(raw, .2)
    grad = jax.grad(lambda value: (weight*jnp.maximum(value, 1.)).mean())(below)
    np.testing.assert_array_equal(grad, 0.)

  def test_controls_logging_stop_gradient_and_validation_boundaries(self):
    feat, prior, obs, act = ordinary_fixture()
    raw = jnp.full((1, 6), 3., jnp.float32)
    for mode in ('logging', 'strength', 'reward_only', 'conditional_shuffle'):
      weight, _, signals = dt_latch.overlay(FakeAgent(), feat, prior,
          raw, raw, obs, act, config(True, mode), seed=7, step=9)
      np.testing.assert_array_equal(weight[~signals['m_intervention']], 1.)
      if mode == 'logging':
        np.testing.assert_array_equal(weight, 1.)
      else:
        self.assertLess(float(weight[0, 1]), 1.)
    for alpha, rho in ((0., .5), (20., 0.)):
      weight = dt_latch.overlay(FakeAgent(), feat, prior, raw, raw,
          obs, act, config(True, alpha=alpha, rho=rho))[0]
      np.testing.assert_array_equal(weight, 1.)
    def weights(qlog, plog, h, action, reward):
      weight, _, _ = dt_latch.overlay(FakeAgent(), dict(deter=h, logit=qlog),
          plog, raw, raw, dict(obs, reward=reward), dict(action=action), config(True))
      return weight.sum()
    grads = jax.grad(weights, argnums=(0, 1, 2, 3, 4))(
        feat['logit'], prior, feat['deter'], act['action'], obs['reward'])
    for value in grads:
      np.testing.assert_array_equal(value, 0.)
    def full(cfg):
      return SimpleNamespace(dt_latch=cfg, rep_probe=SimpleNamespace(mode='off'),
          dyn=SimpleNamespace(typ='rssm'), rewhead=SimpleNamespace(output='symexp_twohot'),
          policy_dist_cont='bounded_normal')
    for enabled in (None, False, True):
      dt_latch.validate(full(config(enabled)), FakeAgent.act_space)
    for wrong in (0, 1, 'true', []):
      with self.assertRaisesRegex(ValueError, 'normal_only must be bool'):
        dt_latch.validate(full(config(wrong)), FakeAgent.act_space)

  def test_normal_exclusion_does_not_hide_invalid_observed_reset_h1(self):
    feat, prior, obs, act = ordinary_fixture()
    raw = jnp.full((1, 6), 3., jnp.float32)
    # Current reset0 cannot be protected, but its observed transition to1 is
    # still a valid H1 observation and must keep the original finite-label guard.
    bad = dict(obs, reward=obs['reward'].at[0, 1].set(jnp.nan))
    with self.assertRaises(Exception):
      jax.block_until_ready(jax.jit(lambda: dt_latch.overlay(FakeAgent(),
          feat, prior, raw, raw, bad, act, config(True)))())
    # last2 -> reset3 really is unavailable in the original probe; no new
    # eligibility flag may reinterpret it as an observed H1 transition.
    unavailable = dict(obs, reward=obs['reward'].at[0, 3].set(jnp.nan))
    weight = jax.jit(lambda: dt_latch.overlay(FakeAgent(), feat, prior,
        raw, raw, unavailable, act, config(True))[0])()
    self.assertTrue(np.isfinite(np.asarray(weight)).all())

  def test_real_tiny_agent_missing_and_false_keep_two_complete_updates_exact(self):
    with tempfile.TemporaryDirectory(prefix='dl-nm1-default-') as directory:
      paths = []
      for mode in ('missing', 'false'):
        path = Path(directory)/(mode+'.npz')
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
            '--snapshot', mode, str(path)], cwd=ROOT, capture_output=True,
            text=True, timeout=240, env=dict(os.environ, PYTHONPATH=str(ROOT),
                JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', MUJOCO_GL='disable'))
        self.assertEqual(result.returncode, 0, result.stdout[-2500:]+result.stderr[-2500:])
        paths.append(path)
      with np.load(paths[0]) as reference, np.load(paths[1]) as candidate:
        self.assertEqual(set(reference.files), set(candidate.files))
        for key in reference.files:
          self.assertEqual(reference[key].dtype, candidate[key].dtype, key)
          self.assertEqual(reference[key].shape, candidate[key].shape, key)
          self.assertEqual(reference[key].tobytes(), candidate[key].tobytes(), key)


def tiny_snapshot(mode, output):
  import elements
  from ruamel.yaml import YAML
  from dreamerv3.agent import Agent
  presets = YAML(typ='safe').load((ROOT/'dreamerv3/configs.yaml').read_text())
  cfg = elements.Config(presets['defaults']).update(presets['debug'])
  cfg = elements.Config({**cfg.flat, **{
      'batch_size': 2, 'batch_length': 4, 'replay_context': 0,
      'jax.platform': 'cpu', 'jax.compute_dtype': 'float32',
      'jax.train_devices': [0], 'jax.enable_policy': False,
      'jax.precompile': False, 'jax.profiler': False,
      'agent.rep_probe.mode': 'off', 'agent.dt_latch.mode': 'dtlatch'}})
  assert cfg.agent.dt_latch.normal_only is False
  flat = dict(cfg.agent.flat)
  if mode == 'missing':
    del flat['dt_latch.normal_only']
  else:
    assert mode == 'false'
  agent = Agent(dict(vector=elements.Space(np.float32, (5,)),
      reward=elements.Space(np.float32), is_first=elements.Space(bool),
      is_last=elements.Space(bool), is_terminal=elements.Space(bool)),
      dict(action=elements.Space(np.float32, (2,))), elements.Config(**flat,
          logdir=str(Path(output).parent/mode), seed=7, jax=cfg.jax,
          batch_size=2, batch_length=4, replay_context=0))
  assert all(device.platform == 'cpu' for device in jax.devices())
  assert agent.config.dyn.rssm.free_nats == 1.
  data = agent._zeros(agent.spaces, (2, 4))
  data['vector'][:] = np.arange(40, dtype=np.float32).reshape(2, 4, 5)/40
  data['reward'][:] = [[0, -.5, .25, 1], [.1, .5, -.25, .75]]
  data['is_first'][:, 0] = True
  data['is_first'][1, 2] = True
  data['is_last'][1, 1] = True
  data['is_last'][:, -1] = True
  data['is_terminal'][0, -1] = True
  data['action'][:] = np.linspace(-.3, .3, 16).reshape(2, 4, 2)
  arrays = {}
  def add(prefix, tree):
    with jax._src.config.explicit_device_get_scope():
      for path, value in jax.tree_util.tree_flatten_with_path(jax.device_get(tree))[0]:
        arrays[prefix+'/'.join(str(key) for key in path)] = np.array(value, copy=True)
  carry = agent.init_train(2)
  add('initial/', agent.save())
  for at in range(2):
    prepared = agent.prepare_batch({key: value.copy() for key, value in data.items()})
    add(f'step{at}/seed/', prepared['seed'])
    carry, delayed_out, delayed_metrics = agent.train(carry, dict(prepared))
    assert not delayed_out and not delayed_metrics
    out, metrics = agent.take_train_result()
    add(f'step{at}/state/', agent.save())
    add(f'step{at}/carry/', carry)
    add(f'step{at}/out/', out)
    add(f'step{at}/metrics/', metrics)
  np.savez(output, **arrays)
  print(json.dumps(dict(version='DL-NM1-normal-only-fixture-r1', mode=mode,
      updates=2, arrays=len(arrays), N1M_acceptance=False, GPU_used=False)))


if __name__ == '__main__':
  if len(sys.argv) == 4 and sys.argv[1] == '--snapshot':
    tiny_snapshot(sys.argv[2], sys.argv[3])
  else:
    unittest.main()
