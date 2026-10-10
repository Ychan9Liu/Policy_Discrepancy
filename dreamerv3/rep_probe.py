"""Policy discrepancy weights and diagnostics for categorical RSSM training."""

import math

import embodied.jax.outs as outs
import jax
import jax.numpy as jnp
import numpy as np


def validate(config, act_space):
  """Validate the optional probe before any model parameters are created."""
  mode = config.rep_probe.mode
  if mode not in ('off', 'logging', 'dt', 'constant', 'shuffle'):
    raise ValueError(f'Unsupported rep_probe.mode: {mode!r}')
  if mode == 'off':
    return
  if mode == 'constant':
    c = config.rep_probe.c
    if not isinstance(c, (int, float)) or not math.isfinite(c) or not 0 < c <= 1:
      raise ValueError('constant rep_probe requires a finite 0 < c <= 1')
  alpha = config.rep_probe.alpha
  bypass = (mode == 'constant' and config.rep_probe.c == 1)
  if not bypass and (alpha is None or not isinstance(alpha, (int, float)) or (
      not math.isfinite(alpha) or alpha < 0)):
    raise ValueError(f'{mode} rep_probe requires a finite, nonnegative alpha')
  if mode == 'dt' and alpha == 0:
    return
  if bypass:
    return
  if config.dyn.typ != 'rssm' or config.policy_dist_cont != 'bounded_normal':
    raise ValueError('rep_probe supports only categorical RSSM and bounded_normal')
  if not act_space or any(space.discrete for space in act_space.values()):
    raise ValueError('rep_probe requires continuous action heads only')


def discrepancy(qpolicy, ppolicy, act_space):
  """Symmetric Gaussian KL per scalar action dimension, shape [B,T]."""
  total, mean_delta, std_delta = 0.0, 0.0, 0.0
  dims = 0
  valid = True
  for key, space in act_space.items():
    q, p = qpolicy[key], ppolicy[key]
    q = q.output if isinstance(q, outs.Agg) else q
    p = p.output if isinstance(p, outs.Agg) else p
    if not isinstance(q, outs.Normal) or not isinstance(p, outs.Normal):
      raise ValueError(f'rep_probe requires Normal action head: {key}')
    qm, pm = jnp.float32(q.mean), jnp.float32(p.mean)
    qs, ps = jnp.float32(q.stddev), jnp.float32(p.stddev)
    if qm.shape != pm.shape or qm.shape[2:] != space.shape:
      raise ValueError(f'Action head shape mismatch: {key}')
    axes = tuple(range(2, qm.ndim))
    qv, pv = jnp.square(qs), jnp.square(ps)
    diff = jnp.square(qm - pm)
    # 0.5 * (KL(q||p) + KL(p||q)); written without cancelling log terms.
    per_dim = 0.25 * (
        jnp.square(qv - pv) / (qv * pv) + diff * (1 / qv + 1 / pv))
    total = total + per_dim.sum(axes)
    mean_delta = mean_delta + jnp.abs(qm - pm).sum(axes)
    std_delta = std_delta + jnp.abs(qs - ps).sum(axes)
    valid = valid & jnp.all(jnp.isfinite(qm) & jnp.isfinite(pm) &
        jnp.isfinite(qs) & jnp.isfinite(ps) & (qs > 0) & (ps > 0), axis=axes)
    dims += int(np.prod(space.shape)) if space.shape else 1
  d = total / dims
  valid = valid & jnp.isfinite(d) & (d >= 0)
  return jnp.where(valid, d, jnp.nan), mean_delta / dims, std_delta / dims


def global_shuffle(weight, seed, step):
  """Permute all global B*T positions with a replayable, separate PRNG key."""
  key = jax.random.fold_in(jax.random.PRNGKey(seed), 0x4454)
  key = jax.random.fold_in(key, jnp.uint32(step))
  positions = jnp.arange(weight.size)
  permutation = jax.random.permutation(key, positions)
  shuffled = jnp.take(weight.reshape(-1), permutation).reshape(weight.shape)
  moved = jnp.mean(permutation != positions)
  return jax.lax.stop_gradient(shuffled), moved, permutation


def matching_stats(rep_raw, candidate_weight, threshold):
  """Global raw S/N over free-nats-active loss positions."""
  active = (rep_raw > threshold if threshold else
            jnp.ones_like(rep_raw, bool))
  return (jnp.where(active, candidate_weight, 0).sum(), active.sum())


def overlay(agent, repfeat, prior_logit, rep_raw, rep_before,
            mode, alpha, c=-1.0, seed=0, step=0, return_signal=False):
  """Probe the initialized actor; return the actual rep weight and metrics."""
  qmode = agent.dyn._dist(repfeat['logit']).pred()
  pmode = agent.dyn._dist(prior_logit).pred()
  qfeat = jax.lax.stop_gradient(agent.feat2tensor({
      'deter': repfeat['deter'], 'stoch': qmode}))
  pfeat = jax.lax.stop_gradient(agent.feat2tensor({
      'deter': repfeat['deter'], 'stoch': pmode}))
  qpolicy = agent.pol(qfeat, 2)
  ppolicy = agent.pol(pfeat, 2)
  d, mean_delta, std_delta = discrepancy(qpolicy, ppolicy, agent.act_space)
  d = jax.lax.stop_gradient(d)
  w = jax.lax.stop_gradient(1 / (1 + jnp.float32(alpha) * d))
  if mode == 'logging':
    weight = jnp.ones_like(w)
    moved = jnp.float32(0)
  elif mode == 'dt':
    weight = w
    moved = jnp.float32(0)
  elif mode == 'constant':
    weight = jnp.full_like(w, c)
    moved = jnp.float32(0)
  elif mode == 'shuffle':
    weight, moved, _ = global_shuffle(w, seed, step)
  else:
    raise ValueError(f'Unsupported active rep_probe mode: {mode!r}')
  active = rep_raw > agent.dyn.free_nats if agent.dyn.free_nats else jnp.ones_like(rep_raw, bool)
  count = active.sum()
  match_sum, match_count = matching_stats(
      rep_raw, w, agent.dyn.free_nats)
  tau = agent.dyn.free_nats if agent.dyn.free_nats else 0.0
  excess = jnp.maximum(rep_raw - tau, 0)
  active_mean = jnp.where(count > 0, (weight * active).sum() / count, jnp.nan)
  source_active_mean = jnp.where(count > 0, (w * active).sum() / count, jnp.nan)
  mode_diff = jnp.any(qmode != pmode, axis=-1)
  result = {
      'match_sum': match_sum,
      'match_count': match_count,
      'match_invalid_count': (
          ~jnp.isfinite(d) | ~jnp.isfinite(w) |
          ~jnp.isfinite(rep_raw)).sum(),
      'D_zero_count': (d == 0).sum(),
      'loss_position_count': jnp.int32(rep_raw.size),
      'D_mean': d.mean(), 'D_p10': jnp.percentile(d, 10),
      'D_p50': jnp.percentile(d, 50), 'D_p90': jnp.percentile(d, 90),
      'w_mean': w.mean(), 'w_p10': jnp.percentile(w, 10),
      'w_p50': jnp.percentile(w, 50), 'w_p90': jnp.percentile(w, 90),
      'actual_weight_mean': weight.mean(),
      'actual_weight_p10': jnp.percentile(weight, 10),
      'actual_weight_p50': jnp.percentile(weight, 50),
      'actual_weight_p90': jnp.percentile(weight, 90),
      'rep_raw_mean': rep_raw.mean(),
      'rep_raw_p10': jnp.percentile(rep_raw, 10),
      'rep_raw_p50': jnp.percentile(rep_raw, 50),
      'rep_raw_p90': jnp.percentile(rep_raw, 90),
      'rep_raw_D_cov': ((rep_raw - rep_raw.mean()) * (d - d.mean())).mean(),
      'rep_before_mean': rep_before.mean(),
      'rep_after_mean': (weight * rep_before).mean(),
      'rep_active_frac': active.mean(),
      'active_weight_available': (count > 0).astype(jnp.float32),
      'active_weight_mean': active_mean,
      'source_active_weight_mean': source_active_mean,
      'active_effective_frac': (active & (weight < 1 - 1e-6)).mean(),
      'active_weight_reduction_mean': jnp.where(
          count > 0, ((1 - weight) * active).sum() / count, jnp.nan),
      'weighted_excess_mean': (weight * excess).mean(),
      'excess_reduction_mean': ((1 - weight) * excess).mean(),
      'candidate_weighted_excess_mean': (w * excess).mean(),
      'actual_weight_raw_cov': ((weight - weight.mean()) *
          (rep_raw - rep_raw.mean())).mean(),
      'source_weight_raw_cov': ((w - w.mean()) *
          (rep_raw - rep_raw.mean())).mean(),
      'shuffle_moved_frac': moved,
      'mode_factor_diff_frac': (qmode != pmode).mean(),
      'mode_vector_diff_frac': mode_diff.mean(),
      'mean_abs_diff': mean_delta.mean(),
      'std_abs_diff': std_delta.mean(),
      'raw_active_D_near_zero_frac': (active & (d <= 1e-6)).mean(),
      'invalid_D_frac': (~jnp.isfinite(d)).mean(),
  }
  result = (weight, {f'dt/{key}': value for key, value in result.items()})
  if return_signal:  # DL-code-r1: reuse the existing actor calls in logging.
    return (*result, dict(qmode=qmode, pmode=pmode, D=d))
  return result


def metrics(agent, repfeat, prior_logit, rep_raw, rep_before, alpha):
  """Compatibility entry point for the first-stage logging-only fixtures."""
  return overlay(agent, repfeat, prior_logit, rep_raw, rep_before,
      'logging', alpha)[1]
