"""DL-code-r1: detached H1 reward support for representation compression.

These probes are engineering interfaces, not evidence of task relevance.
Default off preserves the original model, parameter creation, and random stream.
"""

import math

import embodied.jax.nets as nn
import embodied.jax.outs as outs
import jax
import jax.numpy as jnp

from . import rep_probe

sg = jax.lax.stop_gradient
f32 = jnp.float32
MODES = ('off', 'logging', 'dtlatch', 'strength', 'conditional_shuffle',
         'reward_only')


def active(config):
  dl = getattr(config, 'dt_latch', None)
  return (dl is not None and dl.mode != 'off' and dl.alpha != 0 and dl.rho != 0
      and not (dl.mode == 'strength' and dl.kappa == 0))


def validate(config, act_space):
  dl = getattr(config, 'dt_latch', None)
  if dl is None or dl.mode == 'off':
    return
  if dl.mode not in MODES:
    raise ValueError(f'Unsupported dt_latch.mode: {dl.mode!r}')
  for key in ('alpha', 'rho', 'kl_tolerance'):
    value = getattr(dl, key)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
      raise ValueError(f'dt_latch.{key} must be finite')
  if dl.alpha < 0 or not 0 <= dl.rho <= 1 or not 0 <= dl.kl_tolerance <= 1e-3:
    raise ValueError('dt_latch requires alpha>=0, rho in [0,1], tolerance<=1e-3')
  if not active(config):
    return
  if config.dyn.typ != 'rssm' or config.rewhead.output != 'symexp_twohot':
    raise ValueError('dt_latch supports categorical RSSM and symexp_twohot reward')
  if config.policy_dist_cont != 'bounded_normal' or not act_space or any(
      space.discrete for space in act_space.values()):
    raise ValueError('dt_latch requires bounded_normal continuous action heads')
  if dl.mode != 'logging' and config.rep_probe.mode != 'off':
    raise ValueError('active dt_latch weighting requires rep_probe.mode=off')
  if dl.mode == 'strength' and not (
      math.isfinite(dl.kappa) and 0 <= dl.kappa <= dl.rho):
    raise ValueError('strength requires independently calibrated kappa in [0,rho]')
  if dl.mode == 'conditional_shuffle':
    for key in ('d_bins', 'k_bins'):
      values = tuple(getattr(dl, key))
      if not values or any(not math.isfinite(x) or x < 0 for x in values) or any(
          a >= b for a, b in zip(values, values[1:])):
        raise ValueError(f'conditional_shuffle requires frozen increasing {key}')


def _raise_invalid(_):
  raise ValueError('DL-code-r1: invalid signal at a valid H1 transition')


def require_valid(condition):
  """Fail under eager and JIT execution; no callback is run on valid inputs."""
  def fail(_):
    # Unordered callback avoids an implicit host-created ordered-effect token
    # under Dreamer's strict transfer guard; failures still abort the call.
    jax.debug.callback(_raise_invalid, jnp.int32(0))
    return jnp.int32(0)
  return jax.lax.cond(jnp.all(condition), lambda _: jnp.int32(0), fail, None)


def transition_mask(obs):
  first, last = obs['is_first'], obs['is_last']
  assert first.shape == last.shape and first.ndim == 2
  valid = (~last[:, :-1]) & (~first[:, 1:])
  return jnp.concatenate([valid, jnp.zeros_like(first[:, -1:])], 1)


def representative_signal(agent, repfeat, prior_logit):
  qmode = sg(agent.dyn._dist(sg(repfeat['logit'])).pred())
  pmode = sg(agent.dyn._dist(sg(prior_logit)).pred())
  h = sg(repfeat['deter'])
  qpolicy = agent.pol(sg(agent.feat2tensor(dict(deter=h, stoch=qmode))), 2)
  ppolicy = agent.pol(sg(agent.feat2tensor(dict(deter=h, stoch=pmode))), 2)
  d = rep_probe.discrepancy(qpolicy, ppolicy, agent.act_space)[0]
  return jax.tree.map(sg, dict(qmode=qmode, pmode=pmode, D=d))


def one_step(agent, deter, qmode, pmode, action):
  """Deterministic paired prior/reward paths, all inputs have batch axis N.

  No future observation, sampled latent, policy sampling, or Ninjax RNG call.
  The normal training path must have initialized these shared modules first.
  """
  n = deter.shape[0]
  flatten = nn.DictConcat(agent.act_space, 1)
  a = sg(flatten(nn.cast(action)))
  h = nn.cast(sg(jnp.concatenate([deter, deter], 0)))
  z = nn.cast(sg(jnp.concatenate([qmode, pmode], 0)))
  a = jnp.concatenate([a, a], 0)
  h = agent.dyn._core(h, z, a)
  z = nn.cast(sg(agent.dyn._dist(agent.dyn._prior(h)).pred()))
  prediction = agent.rew(agent.feat2tensor(dict(deter=h, stoch=z)), 1)
  if not isinstance(prediction, outs.TwoHot):
    raise ValueError('DL-code-r1 requires scalar TwoHot reward head')
  q = outs.TwoHot(sg(prediction.logits[:n]), prediction.bins,
      prediction.squash, prediction.unsquash)
  p = outs.TwoHot(sg(prediction.logits[n:]), prediction.bins,
      prediction.squash, prediction.unsquash)
  return q, p


def reward_support(q, p, reward, valid, kl_tolerance=1e-5):
  """Exact TwoHot CE/KL compatibility. Invalid labels are sanitized first."""
  reward = f32(reward)
  require_valid(~valid | jnp.isfinite(reward))
  safe_reward = jnp.where(valid & jnp.isfinite(reward), reward, f32(0))
  y = q.target(safe_reward)
  entropy = -jnp.sum(jax.scipy.special.xlogy(y, y), -1)
  ell_q, ell_p = q.loss(safe_reward), p.loss(safe_reward)
  d_q, d_p = ell_q - entropy, ell_p - entropy
  require_valid(~valid | (jnp.isfinite(ell_q) & jnp.isfinite(ell_p) &
      jnp.isfinite(d_q) & jnp.isfinite(d_p) &
      (d_q >= -kl_tolerance) & (d_p >= -kl_tolerance)))
  d_q, d_p = jnp.maximum(d_q, 0), jnp.maximum(d_p, 0)
  u_q, u_p = jnp.exp(-d_q), jnp.exp(-d_p)
  signed = u_q - u_p
  transformed = q.squash(safe_reward)
  signals = dict(ell_q=ell_q, ell_p=ell_p, H_y=entropy, d_q=d_q, d_p=d_p,
      u_q=u_q, u_p=u_p, signed_support=signed, loss_gap=ell_p - ell_q,
      e=jnp.maximum(signed, 0),
      reward_outside_bins=(transformed < q.bins[0]) | (transformed > q.bins[-1]))
  return jax.tree.map(lambda x: sg(jnp.where(valid, x, jnp.zeros_like(x))), signals)


def gate(d, e, m, alpha, rho):
  require_valid(~m | (jnp.isfinite(d) & (d >= 0) & jnp.isfinite(e) &
      (e >= 0) & (e <= 1)))
  d = jnp.where(m, d, f32(0))
  e = jnp.where(m, e, f32(0))
  f = 1 - 1 / (1 + f32(alpha) * d)
  return sg(1 - f32(rho) * e * f), sg(f)


def probe(agent, repfeat, prior_logit, obs, prevact, alpha=20.0, rho=0.5,
          kl_tolerance=1e-5, signal=None):
  """Return detached [B,T] signals; final position is explicitly unavailable."""
  b, t = obs['is_first'].shape
  assert repfeat['deter'].shape[:2] == (b, t)
  assert obs['reward'].shape == (b, t)
  assert all(x.shape[:2] == (b, t) for x in prevact.values())
  signal = signal or representative_signal(agent, repfeat, prior_logit)
  m = transition_mask(obs)
  if t < 2:
    zero = jnp.zeros((b, t), f32)
    result = {k: zero for k in ('ell_q', 'ell_p', 'H_y', 'd_q', 'd_p',
        'u_q', 'u_p', 'signed_support', 'loss_gap', 'e')}
    result.update(D=signal['D'], m=m, v=jnp.ones_like(zero), f_D=zero,
        reward_outside_bins=jnp.zeros_like(m))
    return jax.tree.map(sg, result)
  flat = lambda x: x[:, :-1].reshape((b * (t - 1), *x.shape[2:]))
  act = {k: sg(x[:, 1:].reshape((b * (t - 1), *x.shape[2:])))
      for k, x in prevact.items()}
  q, p = one_step(agent, flat(sg(repfeat['deter'])), flat(signal['qmode']),
      flat(signal['pmode']), act)
  valid = m[:, :-1].reshape(-1)
  signals = reward_support(q, p, obs['reward'][:, 1:].reshape(-1), valid,
      kl_tolerance)
  def pad(x):
    x = x.reshape((b, t - 1, *x.shape[1:]))
    return jnp.concatenate([x, jnp.zeros((b, 1, *x.shape[2:]), x.dtype)], 1)
  result = jax.tree.map(pad, signals)
  result['raw_logits_q'], result['raw_logits_p'] = pad(q.logits), pad(p.logits)
  result['D'], result['m'] = signal['D'], m
  result['v'], result['f_D'] = gate(signal['D'], result['e'], m, alpha, rho)
  return jax.tree.map(sg, result)


def calibration_stats(rep_raw, signals, tau):
  """Additive calibration statistics; zero denominator is not calibrated."""
  excess = jnp.maximum(rep_raw - tau, 0)
  base = jnp.where(signals['m'], excess * signals['f_D'], 0)
  return sg((base * signals['e']).sum()), sg(base.sum())


def conditional_shuffle(weight, m, rep_raw, d, tau, d_bins, k_bins, seed, step):
  """Global B*T within frozen m/activity/D/K strata; separate reproducible RNG."""
  valid, raw, d = m.reshape(-1), rep_raw.reshape(-1), d.reshape(-1)
  dbin = jnp.searchsorted(jnp.asarray(d_bins, f32), d, side='right')
  kbin = jnp.searchsorted(jnp.asarray(k_bins, f32), raw, side='right')
  nk, nd = len(k_bins) + 1, len(d_bins) + 1
  group = (((valid.astype(jnp.int32) * 2 + (raw > tau)) * nd + dbin) * nk + kbin)
  key = jax.random.fold_in(jax.random.PRNGKey(seed), 0x444C)
  key = jax.random.fold_in(key, jnp.uint32(step))
  positions = jnp.arange(weight.size)
  targets = jnp.argsort(group, stable=True)
  sources = jnp.lexsort((jax.random.uniform(key, (weight.size,)), group))
  perm = jnp.zeros_like(positions).at[targets].set(sources)
  perm = jnp.where(valid, perm, positions)
  shuffled = weight.reshape(-1)[perm].reshape(weight.shape)
  counts = jnp.bincount(group, length=4 * nd * nk)
  nvalid = valid.sum()
  source = weight.reshape(-1)
  active = valid & (raw > tau)
  protected = active & (source < 1 - 1e-6)
  changed = jnp.abs(source[perm] - source) > 1e-6
  excess = jnp.where(valid, jnp.maximum(raw - tau, 0), 0)
  candidate_release = ((1 - source) * excess).sum()
  actual_release = ((1 - source[perm]) * excess).sum()
  available_mean = lambda mask, event: jnp.where(mask.sum() > 0,
      (mask & event).sum() / mask.sum(), jnp.nan)
  result = dict(
      shuffle_moved_frac=jnp.where(nvalid > 0,
          (valid & (perm != positions)).sum() / nvalid, jnp.nan),
      shuffle_singleton_frac=jnp.where(nvalid > 0,
          (valid & (counts[group] == 1)).sum() / nvalid, jnp.nan),
      shuffle_active_source_protected_count=protected.sum(),
      shuffle_protected_available=(protected.sum() > 0).astype(f32),
      shuffle_protected_singleton_frac=available_mean(protected, counts[group] == 1),
      shuffle_active_protected_source_moved_frac=available_mean(
          protected, perm != positions),
      shuffle_active_weight_changed_frac=available_mean(active, changed),
      shuffle_protected_weight_changed_frac=available_mean(protected, changed),
      shuffle_release_available=(candidate_release > 0).astype(f32),
      shuffle_candidate_excess_release=candidate_release,
      shuffle_actual_excess_release=actual_release,
      shuffle_excess_release_relative_deviation=jnp.where(candidate_release > 0,
          jnp.abs(actual_release - candidate_release) / candidate_release, jnp.nan))
  return sg(shuffled), jax.tree.map(sg, result), sg(perm.reshape(weight.shape))


def overlay(agent, repfeat, prior_logit, rep_raw, rep_before, obs, prevact,
            config, seed=0, step=0, signal=None):
  signals = probe(agent, repfeat, prior_logit, obs, prevact,
      config.alpha, config.rho, config.kl_tolerance, signal)
  m, d, e, f = [signals[k] for k in ('m', 'D', 'e', 'f_D')]
  require_valid(~m | (jnp.isfinite(rep_raw) & (rep_raw >= -config.kl_tolerance)))
  weight = signals['v']
  shuffle_metrics = {}
  if config.mode == 'logging':
    weight = jnp.ones_like(weight)
  elif config.mode == 'strength':
    weight = sg(1 - m * f32(config.kappa) * f)
  elif config.mode == 'reward_only':
    weight = sg(1 - m * f32(config.rho) * e)
  elif config.mode == 'conditional_shuffle':
    weight, shuffle_metrics, _ = conditional_shuffle(weight, m, rep_raw, d,
        agent.dyn.free_nats, config.d_bins, config.k_bins, seed, step)
  elif config.mode != 'dtlatch':
    raise ValueError(config.mode)
  tau = agent.dyn.free_nats
  active = rep_raw > tau
  excess = jnp.maximum(rep_raw - tau, 0)
  num, den = calibration_stats(rep_raw, signals, tau)
  valid_mean = lambda x: jnp.where(m.sum() > 0,
      jnp.where(m, x, jnp.zeros_like(x)).sum() / m.sum(), jnp.nan)
  metrics = {f'{key}_valid_mean': valid_mean(signals[key]) for key in (
      'ell_q', 'ell_p', 'd_q', 'd_p', 'u_q', 'u_p', 'loss_gap',
      'signed_support', 'e', 'D')}
  metrics.update(valid_count=m.sum(), unavailable_count=(~m).sum(),
      valid_frac=m.mean(), reward_outside_bins_count=signals['reward_outside_bins'].sum(),
      candidate_v_mean=signals['v'].mean(), actual_v_mean=weight.mean(),
      actual_v_min=weight.min(), actual_v_max=weight.max(),
      active_frac=active.mean(), valid_active_frac=(m & active).mean(),
      active_protected_frac=(m & active & (weight < 1 - 1e-6)).mean(),
      excess_release_mean=((1 - weight) * excess).mean(),
      candidate_excess_release_mean=((1 - signals['v']) * excess).mean(),
      weighted_excess_mean=(weight * excess).mean(),
      rep_before_mean=rep_before.mean(), rep_after_mean=(weight * rep_before).mean(),
      calibration_num=num, calibration_den=den,
      **shuffle_metrics)
  return sg(weight), {f'dl/{k}': sg(v) for k, v in metrics.items()}, signals
