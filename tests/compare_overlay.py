"""Check full-agent overlay loss and gradient decomposition against baseline."""

import argparse
import json

import numpy as np


def load(named):
  name, path = named.split('=', 1)
  return name, np.load(path)


def equal_subset(base, run, prefixes):
  keys = [k for k in base.files if k.startswith(prefixes)]
  for key in keys:
    if key not in run or not np.array_equal(base[key], run[key]):
      raise AssertionError(f'{key} changed')
  return len(keys)


def gradients(run, prefix):
  return {key.removeprefix(prefix): run[key]
          for key in run.files if key.startswith(prefix)}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--baseline', required=True)
  parser.add_argument('--identity', action='append', default=[])
  parser.add_argument('--active', action='append', default=[])
  parser.add_argument('--rep_scale', type=float, default=0.1)
  parser.add_argument('--atol', type=float, default=2e-6)
  parser.add_argument('--rtol', type=float, default=2e-5)
  args = parser.parse_args()
  base = np.load(args.baseline)
  results = {}
  for named in args.identity:
    name, run = load(named)
    shared = {key for key in run.files if not key.startswith('diagnostic/')}
    assert shared == set(base.files), (name, len(shared), len(base.files))
    count = equal_subset(base, run, ('',))
    results[name] = {'exact_shared_arrays': count}

  before = base['loss/rep']
  base_grad = gradients(base, 'grad/')
  base_repgrad = gradients(base, 'repgrad/')
  active_runs = dict(load(named) for named in args.active)
  for name, run in active_runs.items():
    count = equal_subset(base, run, (
        'init/', 'repfeat/', 'gradstate/', 'dyngrad/', 'othergrad/'))
    for key in base.files:
      if key.startswith('loss/') and key != 'loss/rep':
        np.testing.assert_array_equal(base[key], run[key], err_msg=f'{name}:{key}')
        count += 1
    rep_after = run['loss/rep']
    assert rep_after.shape == before.shape
    diagnostics = {key.removeprefix('diagnostic/'): float(run[key])
                   for key in run.files if key.startswith('diagnostic/')}
    expected_loss = float(base['loss']) + args.rep_scale * float((
        rep_after - before).mean())
    np.testing.assert_allclose(float(run['loss']), expected_loss,
        rtol=args.rtol, atol=args.atol)
    np.testing.assert_allclose(diagnostics['dt/rep_before_mean'], before.mean(),
        rtol=args.rtol, atol=args.atol)
    np.testing.assert_allclose(diagnostics['dt/rep_after_mean'], rep_after.mean(),
        rtol=args.rtol, atol=args.atol)
    assert diagnostics['dt/rep_active_frac'] > 0
    assert diagnostics['dt/active_effective_frac'] > 0
    assert diagnostics['dt/excess_reduction_mean'] > 0
    assert diagnostics['dt/invalid_D_frac'] == 0
    assert np.isfinite(rep_after).all()
    grad = gradients(run, 'grad/')
    repgrad = gradients(run, 'repgrad/')
    assert grad.keys() == base_grad.keys()
    max_error = 0.0
    max_change = 0.0
    for key in grad:
      zero = np.zeros_like(base_grad[key])
      expected = (base_grad[key] + repgrad.get(key, zero) -
                  base_repgrad.get(key, zero))
      actual = grad[key]
      difference = np.abs(actual - expected)
      max_error = max(max_error, float(np.max(difference)))
      max_change = max(max_change, float(np.max(np.abs(actual - base_grad[key]))))
      np.testing.assert_allclose(actual, expected,
          rtol=args.rtol, atol=args.atol, err_msg=f'{name}:grad/{key}')
    assert max_change > args.atol, (name, max_change)
    results[name] = {
        'exact_unchanged_arrays': count,
        'loss': float(run['loss']),
        'rep_before_mean': float(before.mean()),
        'rep_after_mean': float(rep_after.mean()),
        'active_effective_frac': diagnostics['dt/active_effective_frac'],
        'active_weight_mean': diagnostics['dt/active_weight_mean'],
        'excess_reduction_mean': diagnostics['dt/excess_reduction_mean'],
        'shuffle_moved_frac': diagnostics['dt/shuffle_moved_frac'],
        'max_gradient_residual': max_error,
        'max_total_gradient_change': max_change,
    }

  if 'constant' in active_runs:
    ratio = active_runs['constant']['loss/rep'] / before
    np.testing.assert_allclose(ratio, 0.6, rtol=args.rtol, atol=args.atol)
  if 'dt' in active_runs and 'shuffle' in active_runs:
    dt_weight = active_runs['dt']['loss/rep'] / before
    shuffled_weight = active_runs['shuffle']['loss/rep'] / before
    np.testing.assert_allclose(np.sort(dt_weight.reshape(-1)),
        np.sort(shuffled_weight.reshape(-1)), rtol=args.rtol, atol=args.atol)
    assert np.max(np.abs(dt_weight - shuffled_weight)) > args.atol
  print(json.dumps(results, indent=2))


if __name__ == '__main__':
  main()
