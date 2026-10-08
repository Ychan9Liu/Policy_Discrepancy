"""Read-only policy discrepancy diagnostics for categorical RSSM training."""

import math

import embodied.jax.outs as outs
import jax
import jax.numpy as jnp
import numpy as np


def validate(config, act_space):
  """Validate the optional probe before any model parameters are created."""
  mode = config.rep_probe.mode
  if mode not in ('off', 'logging'):
    raise ValueError(f'Unsupported rep_probe.mode: {mode!r}')
  if mode == 'off':
    return
  alpha = config.rep_probe.alpha
  if alpha is None or not isinstance(alpha, (int, float)) or (
      not math.isfinite(alpha) or alpha < 0):
    raise ValueError('logging rep_probe requires a finite, nonnegative alpha')
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


def metrics(agent, repfeat, prior_logit, rep_raw, rep_before, alpha):
  """Probe the already initialized actor without drawing actions or updating state."""
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
  active = rep_raw > agent.dyn.free_nats if agent.dyn.free_nats else jnp.ones_like(rep_raw, bool)
  count = active.sum()
  tau = agent.dyn.free_nats if agent.dyn.free_nats else 0.0
  excess = jnp.maximum(rep_raw - tau, 0)
  active_mean = jnp.where(count > 0, (w * active).sum() / count, jnp.nan)
  mode_diff = jnp.any(qmode != pmode, axis=-1)
  result = {
      'D_mean': d.mean(), 'D_p10': jnp.percentile(d, 10),
      'D_p50': jnp.percentile(d, 50), 'D_p90': jnp.percentile(d, 90),
      'w_mean': w.mean(), 'w_p10': jnp.percentile(w, 10),
      'w_p50': jnp.percentile(w, 50), 'w_p90': jnp.percentile(w, 90),
      'actual_weight_mean': jnp.ones_like(w).mean(),
      'rep_raw_mean': rep_raw.mean(),
      'rep_before_mean': rep_before.mean(),
      'rep_after_mean': rep_before.mean(),
      'rep_active_frac': active.mean(),
      'active_weight_available': (count > 0).astype(jnp.float32),
      'active_weight_mean': active_mean,
      'weighted_excess_mean': excess.mean(),
      'candidate_weighted_excess_mean': (w * excess).mean(),
      'mode_factor_diff_frac': (qmode != pmode).mean(),
      'mode_vector_diff_frac': mode_diff.mean(),
      'mean_abs_diff': mean_delta.mean(),
      'std_abs_diff': std_delta.mean(),
      'raw_active_D_near_zero_frac': (active & (d <= 1e-6)).mean(),
      'invalid_D_frac': (~jnp.isfinite(d)).mean(),
  }
  return {f'dt/{key}': value for key, value in result.items()}
