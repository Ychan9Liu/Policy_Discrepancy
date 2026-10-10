"""DL-code-r1 engineering evidence, never a method-performance test."""

import unittest
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

# Optional process-local Windows runtime path, never changes system libraries.
_dll_directory = (os.add_dll_directory(os.environ['DL_DLL_DIRECTORY'])
    if os.name == 'nt' and os.environ.get('DL_DLL_DIRECTORY') else None)

import elements
import embodied.jax.outs as outs
import jax
import jax.numpy as jnp
import numpy as np

from dreamerv3 import dt_latch


def old_twohot_loss(output, target):
  # Frozen pre-DL source, used as an independent regression reference.
  target = jax.lax.stop_gradient(output.squash(target))
  below = (output.bins <= target[..., None]).astype(jnp.int32).sum(-1) - 1
  above = len(output.bins) - (
      output.bins > target[..., None]).astype(jnp.int32).sum(-1)
  below = jnp.clip(below, 0, len(output.bins) - 1)
  above = jnp.clip(above, 0, len(output.bins) - 1)
  equal = (below == above)
  dist_to_below = jnp.where(equal, 1, jnp.abs(output.bins[below] - target))
  dist_to_above = jnp.where(equal, 1, jnp.abs(output.bins[above] - target))
  total = dist_to_below + dist_to_above
  weight_below = dist_to_above / total
  weight_above = dist_to_below / total
  target = (
      jax.nn.one_hot(below, len(output.bins)) * weight_below[..., None] +
      jax.nn.one_hot(above, len(output.bins)) * weight_above[..., None])
  log_pred = output.logits - jax.scipy.special.logsumexp(
      output.logits, -1, keepdims=True)
  return -(target * log_pred).sum(-1)


class FakeDyn:
  free_nats = 1.0

  def __init__(self, scale=1.0):
    self.scale = scale

  def _dist(self, logits):
    return outs.Agg(outs.OneHot(logits, 0.01), 1)

  def _core(self, h, z, action):
    assert h.ndim == 2 and z.ndim == 3 and action.ndim == 2
    return h + self.scale * (z[:, 0, :1] - z[:, 0, 1:]) + action[:, :1]

  def _prior(self, h):
    return jnp.stack([h, -h], -1)


class FakeAgent:
  act_space = {'action': elements.Space(np.float32, (1,))}

  def __init__(self, core=1.0, reward=1.0, actor=1.0):
    self.dyn = FakeDyn(core)
    self.reward_scale, self.actor_scale = reward, actor

  def feat2tensor(self, feat):
    z = feat['stoch'].reshape((*feat['stoch'].shape[:-2], -1))
    return jnp.concatenate([feat['deter'], z], -1)

  def rew(self, x, bdims):
    assert bdims == 1
    logits = self.reward_scale * jnp.concatenate([-x[:, :1], 0 * x[:, :1],
        x[:, :1]], -1)
    return outs.TwoHot(logits, jnp.array([-1., 0., 1.], jnp.float32))

  def pol(self, x, bdims):
    assert bdims == 2
    mean = self.actor_scale * (x[..., 1:2] - x[..., 2:3])
    return {'action': outs.Agg(outs.Normal(mean, jnp.ones_like(mean)), 1)}


def fixture(b=2, t=5):
  post = jnp.broadcast_to(jnp.array([3., -3.]), (b, t, 1, 2))
  prior = -post
  feat = dict(deter=jnp.arange(b * t, dtype=jnp.float32).reshape(b, t, 1) / 20,
      logit=post)
  obs = dict(reward=jnp.broadcast_to(jnp.linspace(-1, 1, t), (b, t)),
      is_first=jnp.zeros((b, t), bool), is_last=jnp.zeros((b, t), bool),
      is_terminal=jnp.zeros((b, t), bool))
  act = dict(action=jnp.arange(b * t, dtype=jnp.float32).reshape(b, t, 1) / 30)
  return feat, prior, obs, act


class DtLatchTest(unittest.TestCase):

  def test_exact_twohot_loss_grad_and_jaxpr(self):
    bins = jnp.array([-3., -1., 0., .2, 1., 4.], jnp.float32)
    rewards = jnp.array([-10., -3., -2., -1., 0., .1, .2, 4., 10.], jnp.float32)
    logits = jnp.arange(rewards.size * bins.size, dtype=jnp.float32).reshape(
        rewards.size, bins.size) / 13
    for squash in (None, lambda x: jnp.sign(x) * jnp.log1p(jnp.abs(x))):
      new = lambda x: outs.TwoHot(x, bins, squash).loss(rewards)
      old = lambda x: old_twohot_loss(outs.TwoHot(x, bins, squash), rewards)
      np.testing.assert_array_equal(new(logits), old(logits))
      np.testing.assert_array_equal(jax.grad(lambda x: new(x).sum())(logits),
          jax.grad(lambda x: old(x).sum())(logits))
      self.assertEqual(str(jax.make_jaxpr(new)(logits)),
          str(jax.make_jaxpr(old)(logits)))
      target = outs.TwoHot(logits, bins, squash).target(rewards)
      np.testing.assert_array_equal(target.sum(-1), np.ones(rewards.size))

  def test_entropy_and_support_independent_closed_form(self):
    bins = jnp.array([-1., 0., 1.], jnp.float32)
    qprob = jnp.array([[.1, .2, .7], [.25, .25, .5]], jnp.float32)
    pprob = jnp.array([[.2, .2, .6], [.3, .3, .4]], jnp.float32)
    result = dt_latch.reward_support(outs.TwoHot(jnp.log(qprob), bins),
        outs.TwoHot(jnp.log(pprob), bins), jnp.array([1., .5]), jnp.ones(2, bool))
    y = np.array([[0., 0., 1.], [0., .5, .5]])
    h = np.array([0., np.log(2)])
    ellq, ellp = -(y * np.log(qprob)).sum(-1), -(y * np.log(pprob)).sum(-1)
    np.testing.assert_allclose(result['H_y'], h, atol=1e-7)
    np.testing.assert_allclose(result['ell_q'], ellq, atol=1e-7)
    np.testing.assert_allclose(result['d_q'], ellq - h, atol=1e-7)
    np.testing.assert_allclose(result['e'], np.maximum(
        np.exp(-(ellq - h)) - np.exp(-(ellp - h)), 0), atol=1e-7)

  def test_action_nextreward_no_futureobs_and_reset_mask(self):
    feat, prior, obs, act = fixture()
    obs['is_last'] = obs['is_last'].at[0, 1].set(True)
    obs['is_first'] = obs['is_first'].at[0, 2].set(True)
    obs['is_terminal'] = obs['is_terminal'].at[1, 3].set(True)
    obs['is_last'] = obs['is_last'].at[1, 3].set(True)
    result = dt_latch.probe(FakeAgent(), feat, prior, obs, act)
    np.testing.assert_array_equal(result['m'],
        [[True, False, True, True, False], [True, True, True, False, False]])
    # t=2 -> terminal t=3 retains its reward; t=3 -> following step is invalid.
    self.assertTrue(bool(result['m'][1, 2]))
    np.testing.assert_array_equal(result['v'][~result['m']], 1)
    agent = FakeAgent()
    qmode = agent.dyn._dist(feat['logit']).pred()
    pmode = agent.dyn._dist(prior).pred()
    for row, col in ((0, 0), (0, 2), (1, 2)):
      q, p = dt_latch.one_step(agent, feat['deter'][row:row+1, col],
          qmode[row:row+1, col], pmode[row:row+1, col],
          {'action': act['action'][row:row+1, col+1]})
      np.testing.assert_allclose(result['ell_q'][row, col],
          q.loss(obs['reward'][row:row+1, col+1])[0], rtol=1e-6)
      np.testing.assert_allclose(result['ell_p'][row, col],
          p.loss(obs['reward'][row:row+1, col+1])[0], rtol=1e-6)
    # API ignores any future image/vector field; it never encodes next obs.
    other = dt_latch.probe(agent, feat, prior,
        dict(obs, vector=jnp.full((2, 5, 8), jnp.nan)), act)
    np.testing.assert_array_equal(result['v'], other['v'])

  def test_invalid_reward_sanitized_only_when_unavailable(self):
    feat, prior, obs, act = fixture(b=1, t=3)
    obs['is_first'] = obs['is_first'].at[0, 1].set(True)
    obs['reward'] = obs['reward'].at[0, 1].set(jnp.nan)
    result = jax.jit(lambda: dt_latch.probe(FakeAgent(), feat, prior, obs, act))()
    self.assertTrue(np.isfinite(np.asarray(result['v'])).all())
    self.assertEqual(float(result['e'][0, 0]), 0.)
    bad = dict(obs, reward=obs['reward'].at[0, 2].set(jnp.nan))
    with self.assertRaises(Exception):
      jax.block_until_ready(jax.jit(lambda: dt_latch.probe(
          FakeAgent(), feat, prior, bad, act))())

  def test_all_probe_inputs_and_shared_module_params_detached(self):
    feat, prior, obs, act = fixture(b=1, t=3)
    def fn(core, reward, actor, h, qlog, plog, rew, action):
      result = dt_latch.probe(FakeAgent(core, reward, actor),
          dict(deter=h, logit=qlog), plog, dict(obs, reward=rew),
          dict(action=action))
      return sum(jnp.sum(x.astype(jnp.float32)) for x in result.values())
    args = (1., 1., 1., feat['deter'], feat['logit'], prior,
        obs['reward'], act['action'])
    grads = jax.grad(fn, argnums=tuple(range(len(args))))(*args)
    for grad in grads:
      np.testing.assert_array_equal(grad, np.zeros_like(grad))

  def test_gate_range_identity_and_local_rep_gradient(self):
    d = jnp.array([[0., .1, 1., 100.]])
    e = jnp.array([[1., 0., .2, 1.]])
    m = jnp.array([[True, True, True, False]])
    v, _ = dt_latch.gate(d, e, m, 20., .5)
    self.assertTrue(np.all((np.asarray(v) >= .5) & (np.asarray(v) <= 1)))
    np.testing.assert_array_equal(v[0, [0, 1, 3]], [1., 1., 1.])
    for alpha, rho in ((0., .5), (20., 0.)):
      np.testing.assert_array_equal(dt_latch.gate(d, e, m, alpha, rho)[0], 1)
    raw = jnp.array([[.2, 1., 2., 3.]])
    baseline = jax.grad(lambda x: jnp.maximum(x, 1.).mean())(raw)
    gated = jax.grad(lambda x: (v * jnp.maximum(x, 1.)).mean())(raw)
    np.testing.assert_array_equal(gated, v * baseline)
    self.assertEqual(float(gated[0, 0]), 0.)
    # Shared-module ordinary loss is untouched; these gradients stay nonzero.
    self.assertNotEqual(float(jax.grad(lambda r: FakeAgent(reward=r).rew(
        jnp.array([[1., 1., 0.]]), 1).loss(jnp.array([1.])).sum())(1.)), 0.)

  def test_uniform_initialization_and_short_chunk(self):
    feat, prior, obs, act = fixture()
    result = dt_latch.probe(FakeAgent(reward=0.), feat, prior, obs, act)
    np.testing.assert_array_equal(result['e'], 0)
    np.testing.assert_array_equal(result['v'], 1)
    feat, prior, obs, act = fixture(t=1)
    result = dt_latch.probe(FakeAgent(), feat, prior, obs, act)
    np.testing.assert_array_equal(result['m'], False)
    np.testing.assert_array_equal(result['v'], 1)

  def test_categorical_posterior_gradient_with_nontrivial_gate(self):
    feat, prior, obs, act = fixture(b=1, t=5)
    agent = FakeAgent()
    tau = .1
    def raw_rep(qlog):
      return agent.dyn._dist(qlog).kl(agent.dyn._dist(jax.lax.stop_gradient(prior)))
    def loss(qlog, gated):
      rep = jnp.maximum(raw_rep(qlog), tau)
      if gated:
        signal = dt_latch.probe(agent, dict(feat, logit=qlog), prior, obs, act)
        rep = signal['v'] * rep
      return rep.mean()
    signal = dt_latch.probe(agent, feat, prior, obs, act)
    self.assertTrue(np.any(np.asarray(signal['v']) < 1))
    baseline = jax.grad(lambda q: loss(q, False))(feat['logit'])
    gated = jax.grad(lambda q: loss(q, True))(feat['logit'])
    np.testing.assert_allclose(gated,
        signal['v'][..., None, None] * baseline, rtol=2e-6, atol=1e-8)
    self.assertGreater(float(jnp.linalg.norm(gated)), 0)

  def test_conditional_shuffle_strata_replay_and_calibration(self):
    shape = (4, 6)
    weight = jnp.linspace(.5, 1., 24).reshape(shape)
    m = jnp.ones(shape, bool).at[:, -1].set(False)
    weight = jnp.where(m, weight, 1)
    d = jnp.tile(jnp.array([.1, .1, .1, 3., 3., 3.]), (4, 1))
    raw = jnp.tile(jnp.array([.2, 2., 2., .2, 2., 2.]), (4, 1))
    shuffled, metrics, perm = dt_latch.conditional_shuffle(weight, m, raw, d,
        1., (1.,), (1.,), 7, 9)
    repeated = dt_latch.conditional_shuffle(weight, m, raw, d,
        1., (1.,), (1.,), 7, 9)[0]
    np.testing.assert_array_equal(shuffled, repeated)
    np.testing.assert_array_equal(shuffled[~m], 1)
    self.assertGreater(float(metrics['shuffle_moved_frac']), 0)
    take = lambda x: np.asarray(x).reshape(-1)[np.asarray(perm).reshape(-1)]
    np.testing.assert_array_equal(take(m), m.reshape(-1))
    np.testing.assert_array_equal(take(raw > 1), (raw > 1).reshape(-1))
    np.testing.assert_array_equal(take(d > 1), (d > 1).reshape(-1))
    for active in (False, True):
      for large_d in (False, True):
        mask = m & ((raw > 1) == active) & ((d > 1) == large_d)
        np.testing.assert_array_equal(np.sort(np.asarray(shuffled[mask])),
            np.sort(np.asarray(weight[mask])))
    signals = dict(m=m, f_D=jnp.full(shape, .5), e=jnp.full(shape, .2))
    num, den = dt_latch.calibration_stats(raw, signals, 1.)
    self.assertGreater(float(den), 0)
    self.assertAlmostEqual(float(.5 * num / den), .1, places=6)
    num, den = dt_latch.calibration_stats(jnp.zeros(shape), signals, 1.)
    self.assertEqual(float(den), 0)
    identical = jnp.where(m, .7, 1.)
    _, metric, _ = dt_latch.conditional_shuffle(identical, m, raw, d,
        1., (1.,), (1.,), 7, 9)
    self.assertGreater(float(metric['shuffle_active_protected_source_moved_frac']), 0)
    self.assertEqual(float(metric['shuffle_protected_weight_changed_frac']), 0)
    self.assertEqual(float(metric['shuffle_active_weight_changed_frac']), 0)
    self.assertEqual(float(metric['shuffle_excess_release_relative_deviation']), 0)

  def test_configuration_and_static_bypass(self):
    def config(mode='dtlatch', alpha=20., rho=.5, **kw):
      dl = dict(mode=mode, alpha=alpha, rho=rho, kappa=-1.,
          kl_tolerance=1e-5, d_bins=[], k_bins=[])
      dl.update(kw)
      return SimpleNamespace(dt_latch=SimpleNamespace(**dl),
          rep_probe=SimpleNamespace(mode='off'), dyn=SimpleNamespace(typ='rssm'),
          rewhead=SimpleNamespace(output='symexp_twohot'),
          policy_dist_cont='bounded_normal')
    act = FakeAgent.act_space
    dt_latch.validate(config(), act)
    for cfg in (config('off'), config(alpha=0), config(rho=0)):
      dt_latch.validate(cfg, act)
      self.assertFalse(dt_latch.active(cfg))
    for cfg in (config(rho=1.01), config(mode='strength'),
                config(mode='conditional_shuffle')):
      with self.assertRaises(ValueError):
        dt_latch.validate(cfg, act)
    dt_latch.validate(config(mode='strength', kappa=.1), act)
    dt_latch.validate(config(mode='conditional_shuffle', d_bins=[.1, 1.],
        k_bins=[1., 2.]), act)

  def test_conditional_shuffle_global_sharding(self):
    if len(jax.devices()) < 2:
      self.skipTest('requires two devices; CPU host devices also suffice')
    shape = (4, 6)
    weight = jnp.linspace(.5, .95, 24).reshape(shape)
    m = jnp.ones(shape, bool).at[:, -1].set(False)
    weight = jnp.where(m, weight, 1.)
    d, raw = jnp.full(shape, .1), jnp.full(shape, 2.)
    reference, _, perm_ref = dt_latch.conditional_shuffle(weight, m, raw, d,
        1., (1.,), (1.,), 7, 9)
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:2]), ('d',))
    sharded = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('d'))
    mirrored = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    fn = jax.jit(lambda w, m, k, d: dt_latch.conditional_shuffle(
        w, m, k, d, 1., (1.,), (1.,), 7, 9),
        in_shardings=(sharded,) * 4,
        out_shardings=(sharded, mirrored, sharded))
    with mesh:
      result, _, perm = fn(*(jax.device_put(x, sharded) for x in (weight, m, raw, d)))
    np.testing.assert_array_equal(result, reference)
    np.testing.assert_array_equal(perm, perm_ref)
    indices = np.asarray(perm).reshape(-1)
    self.assertTrue(np.any(indices // 12 != np.arange(24) // 12))

  def test_size50m_config_and_episode_reader_contract(self):
    import ruamel.yaml as yaml
    path = Path(__file__).with_name('dl_size50m_check.py')
    spec = importlib.util.spec_from_file_location('dl_size50m_check', path)
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    with tempfile.TemporaryDirectory(prefix='dl-size50m-reader-') as temp:
      temp = Path(temp)
      with (path.parents[1] / 'dreamerv3/configs.yaml').open(encoding='utf-8') as file:
        presets = yaml.YAML(typ='safe').load(file)
      config = elements.Config(presets['defaults']).update(presets['m2_v1'])
      config = config.update({'task': 'dmc_quadruped_walk'})
      config_path = temp / 'config.yaml'
      with config_path.open('w', encoding='utf-8') as file:
        yaml.YAML(typ='safe').dump(dict(config.flat), file)
      args = SimpleNamespace(config=str(config_path), output=str(temp / 'output'),
          platform='cpu', alpha=20., rho=.5)
      parsed = check.load_config(args, 'dtlatch')
      self.assertEqual(parsed.agent.dyn.rssm.deter, 4096)
      self.assertEqual(parsed.batch_size, 16)
      self.assertEqual(parsed.replay_context, 1)
      self.assertEqual(parsed.agent.dt_latch.mode, 'dtlatch')
      artifacts = []
      for episode in range(2):
        file = temp / f'episode_{episode:03}.npz'
        first, last = np.zeros(5, bool), np.zeros(5, bool)
        first[0], last[-1] = True, True
        np.savez(file, image=np.zeros((5, 64, 64, 3), np.uint8),
            reward=np.zeros(5, np.float32), is_first=first, is_last=last,
            is_terminal=np.zeros(5, bool), action=np.zeros((5, 2), np.float32))
        artifacts.append(dict(episode=episode, split='blind', path=file.name,
            sha256=check.sha256(file)))
      plan = temp / 'collection_plan.json'
      check.write_json(plan, dict(task='dmc_quadruped_walk',
          checkpoint_sha256='fixture-only'))
      check.write_json(temp / 'collection_complete.json', dict(
          plan_sha256=check.sha256(plan), params_unchanged=True, artifacts=artifacts))
      batch, _ = check.episode_batch(temp, batch_size=2, length=5)
      self.assertEqual(batch['image'].shape, (2, 5, 64, 64, 3))
      with self.assertRaises(ValueError):
        check.episode_batch(temp, batch_size=16, length=65)
      with (temp / artifacts[0]['path']).open('ab') as file:
        file.write(b'corrupt')
      with self.assertRaises(ValueError):
        check.episode_batch(temp, batch_size=2, length=5)

  @unittest.skipUnless(os.environ.get('DL_FULL_AGENT_TEST') == '1',
      'set DL_FULL_AGENT_TEST=1 for isolated full-agent CPU snapshots')
  def test_full_agent_bypasses_logging_rng_params_and_optimizer(self):
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='dl-identity-') as tmp:
      paths = {}
      for mode in ('baseline', 'off', 'alpha0', 'rho0', 'logging', 'strength0',
                     'dtlatch', 'invalid', 'context'):
        path = Path(tmp) / f'{mode}.npz'
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
            '--snapshot', mode, str(path)], cwd=root, capture_output=True,
            text=True, env=dict(os.environ, PYTHONPATH=str(root)))
        self.assertEqual(result.returncode, 0,
            result.stdout[-3000:] + result.stderr[-3000:])
        paths[mode] = path
      with np.load(paths['baseline']) as reference:
        for mode in ('off', 'alpha0', 'rho0', 'logging', 'strength0'):
          with np.load(paths[mode]) as candidate:
            self.assertEqual(set(reference.files), set(candidate.files))
            for key in reference.files:
              np.testing.assert_array_equal(reference[key], candidate[key],
                  err_msg=f'{mode}: {key}')
        with np.load(paths['dtlatch']) as candidate:
          self.assertEqual(set(reference.files), set(candidate.files))
          # Initialization, sampled paths, and first gradients remain baseline
          # because reward logits start uniform. Later gates may differ.
          for key in reference.files:
            if key.startswith(('init/', 'grad/', 'loss/', 'sampled/')):
              np.testing.assert_array_equal(reference[key], candidate[key],
                  err_msg=f'dtlatch init: {key}')


def full_agent_snapshot(mode, output):
  """Run a separate process so Ninjax creation and JAX RNG are comparable."""
  import embodied.jax.internal as internal
  import ninjax as nj
  import ruamel.yaml as yaml
  from dreamerv3.agent import Agent
  with open('dreamerv3/configs.yaml', encoding='utf-8') as file:
    configs = yaml.YAML(typ='safe').load(file)
  config = elements.Config(configs['defaults']).update(configs['debug'])
  overrides = dict(batch_size=2, batch_length=3, replay_context=0)
  overrides.update({
      'jax.platform': 'cpu', 'jax.compute_dtype': 'float32',
      'jax.train_devices': [0], 'jax.precompile': False,
      'jax.profiler': False, 'jax.enable_policy': False,
      'agent.imag_length': 2, 'agent.imag_last': 2,
      'agent.dyn.rssm.free_nats': 0.,
      'agent.rep_probe.mode': 'off', 'agent.dt_latch.mode':
          'off' if mode in ('baseline', 'off') else
          'logging' if mode == 'logging' else
          'strength' if mode == 'strength0' else 'dtlatch',
      'agent.dt_latch.alpha': 0. if mode == 'alpha0' else 20.,
      'agent.dt_latch.rho': 0. if mode == 'rho0' else .5,
      'agent.dt_latch.kappa': 0. if mode == 'strength0' else -1.,
  })
  config = elements.Config({**config.flat, **overrides})
  obs_space = dict(vector=elements.Space(np.float32, (5,)),
      reward=elements.Space(np.float32), is_first=elements.Space(bool),
      is_last=elements.Space(bool), is_terminal=elements.Space(bool))
  act_space = {'action': elements.Space(np.float32, (2,))}
  flat = config.agent.flat
  if mode == 'baseline':
    flat = {k: v for k, v in flat.items() if not k.startswith('dt_latch.')}
  agent_config = elements.Config({**flat, **dict(logdir='/tmp/dl-identity',
      seed=7, jax=config.jax, batch_size=config.batch_size,
      batch_length=config.batch_length, replay_context=config.replay_context)})
  agent = Agent(obs_space, act_space, agent_config)
  data = agent._zeros(agent.spaces, (2, 3))
  data['vector'][:] = np.arange(30, dtype=np.float32).reshape(2, 3, 5) / 30
  data['reward'][:] = np.array([[-.5, 0., 1.], [1., .5, -.5]], np.float32)
  data['is_first'][:, 0] = True
  data['is_first'][1, 2] = True
  data['is_last'][1, 1] = True
  data['is_terminal'][0, 2] = True
  data['is_last'][0, 2] = True
  data['action'][:] = .2
  if mode == 'invalid':
    data['reward'][0, 1] = np.nan  # valid H1 endpoint, must stop training.
  data = internal.device_put(data, agent.train_sharded)
  carry = agent.init_train(2)
  if mode == 'context':
    spec = importlib.util.spec_from_file_location('dl_size50m_check',
        Path(__file__).with_name('dl_size50m_check.py'))
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    prefix, previous = jax.jit(lambda values: (
        {k: v[:, :1] for k, v in values.items() if k in agent.obs_space},
        {k: jnp.zeros_like(v[:, :1]) for k, v in values.items() if k in agent.act_space}))(
            data)
    fn = jax.jit(nj.pure(lambda ca, ob, pa: check.reconstruct_context_entries(
        agent.model, ca, ob, pa)))
    cache_seed = agent._seeds(0x444C, agent.train_mirrored)
    state, entries = fn(agent.params, carry, prefix, previous, seed=cache_seed)
    repeat_state, repeated = fn(agent.params, carry, prefix, previous, seed=cache_seed)
    with jax._src.config.explicit_device_get_scope():
      for key in agent.params:
        np.testing.assert_array_equal(state[key], agent.params[key])
        np.testing.assert_array_equal(repeat_state[key], agent.params[key])
      for key in entries:
        np.testing.assert_array_equal(entries[key], repeated[key])
        assert entries[key].shape[:2] == (2, 1)
    assert set(entries) == {'dyn/deter', 'dyn/stoch'}
    np.savez(output, context_replayable=np.array(True))
    print(json.dumps(dict(mode=mode, context_replayable=True)))
    return
  loss_carry, obs, prevact, _ = jax.jit(agent.model._apply_replay_context)(carry, data)
  seed = agent._seeds(0, agent.train_mirrored)
  if mode == 'invalid':
    try:
      agent.train(carry, dict(data, seed=seed))
      jax.block_until_ready(agent.params)
    except Exception as error:
      if 'invalid signal at a valid H1 transition' not in str(error):
        raise
      np.savez(output, stopped=np.array(True))
      print(json.dumps(dict(mode=mode, stopped=True)))
      return
    raise AssertionError('DL invalid valid-position reward did not stop optimizer call')
  arrays = {}
  def add(prefix, tree):
    leaves = elements.tree.flatdict(tree)
    with jax._src.config.explicit_device_get_scope():
      for key, value in leaves.items():
        arrays[prefix + key] = np.asarray(value)
  add('init/', agent.params)
  grad_fn = jax.jit(nj.pure(lambda c, o, p: nj.grad(
      agent.model.loss, agent.model.modules, has_aux=True)(c, o, p, True)))
  state, (loss, _, grads, aux) = grad_fn(
      agent.params, loss_carry, obs, prevact, seed=seed)
  add('grad/', grads)
  add('loss/', aux[2]['losses'])
  add('sampled/', aux[2]['repfeat'])
  # Do not compare DL metrics to baseline: their presence is intended.
  for step in range(2):
    seed = agent._seeds(step, agent.train_mirrored)
    carry, _, _ = agent.train(carry, dict(data, seed=seed))
    jax.block_until_ready(agent.params)
    add(f'step{step}/', agent.params)
  np.savez(output, **arrays)
  print(json.dumps(dict(mode=mode, arrays=len(arrays))))


if __name__ == '__main__':
  if len(sys.argv) == 4 and sys.argv[1] == '--snapshot':
    full_agent_snapshot(sys.argv[2], sys.argv[3])
  else:
    unittest.main()
