"""Compare full-agent snapshots from baseline, off, and logging-only runs."""

import argparse
import json

import numpy as np


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('baseline')
  parser.add_argument('off')
  parser.add_argument('logging')
  args = parser.parse_args()
  runs = {name: np.load(path) for name, path in (
      ('baseline', args.baseline), ('off', args.off),
      ('logging', args.logging))}
  base = runs['baseline']
  counts = {}
  for name in ('off', 'logging'):
    run = runs[name]
    assert set(base.files) == {k for k in run.files if not k.startswith('diagnostic/')}
    max_error = 0.0
    for key in base.files:
      x, y = base[key], run[key]
      assert x.shape == y.shape and x.dtype == y.dtype, key
      if not np.array_equal(x, y):
        max_error = max(max_error, float(np.max(np.abs(x.astype(np.float64) - y))))
        raise AssertionError(f'{name}: {key} differs (max abs {max_error})')
      assert np.isfinite(y).all(), (name, key)
    counts[name] = len(base.files)
  logging = runs['logging']
  diagnostics = {key.removeprefix('diagnostic/'): float(logging[key])
                 for key in logging.files if key.startswith('diagnostic/')}
  assert diagnostics['dt/invalid_D_frac'] == 0
  assert 0 <= diagnostics['dt/rep_active_frac'] <= 1
  assert 0 < diagnostics['dt/w_mean'] <= 1
  assert diagnostics['dt/actual_weight_mean'] == 1
  if diagnostics['dt/rep_active_frac'] > 0:
    assert np.isfinite(diagnostics['dt/active_weight_mean'])
  else:
    assert np.isnan(diagnostics['dt/active_weight_mean'])
  print(json.dumps({'exact_shared_arrays': counts, 'diagnostics': diagnostics},
      indent=2))


if __name__ == '__main__':
  main()
