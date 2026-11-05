"""CPU exact snapshots across the original and isolated existing DtL modules.

Keeps the historical active-vs-baseline strict assertion and its failures. This
separate check answers only whether importing the existing method changed it.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def git(root, *args):
  return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def compare(first, second):
  changes = []
  with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
    if set(a.files) != set(b.files):
      raise ValueError('Snapshot array keys differ')
    for key in a.files:
      if a[key].dtype != b[key].dtype or a[key].shape != b[key].shape:
        raise ValueError('Snapshot schema differs: '+key)
      if not np.array_equal(a[key], b[key], equal_nan=True):
        delta = np.abs(a[key].astype(np.float64)-b[key].astype(np.float64))
        changes.append(dict(key=key, mismatched=int(np.count_nonzero(a[key] != b[key])),
                            max_abs_difference=float(delta.max())))
    return dict(array_count=len(a.files), exact=not changes, differences=changes)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--reference', type=Path, required=True)
  parser.add_argument('--candidate', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=False)
  versions = {}
  for label, root in (('reference',args.reference),('candidate',args.candidate)):
    if git(root, 'status', '--porcelain'):
      raise RuntimeError('Clean fixed checkout required: '+str(root))
    versions[label] = git(root, 'rev-parse', 'HEAD')
    for mode in ('baseline','dtlatch'):
      env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
          MUJOCO_GL='disable', PYTHONPATH=str(root),
          XLA_FLAGS='--xla_force_host_platform_device_count=2',
          OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
      out = args.output/f'{label}-{mode}.npz'
      with out.with_suffix('.log').open('x') as log:
        subprocess.run([sys.executable, str(root/'tests/test_dt_latch.py'),
            '--snapshot',mode,str(out)], cwd=root, env=env,
            stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
  imports = {mode:compare(args.output/f'reference-{mode}.npz',
                         args.output/f'candidate-{mode}.npz')
             for mode in ('baseline','dtlatch')}
  historical = {label:compare(args.output/f'{label}-baseline.npz',
                              args.output/f'{label}-dtlatch.npz')
                for label in ('reference','candidate')}
  report = dict(versions=versions, import_identity=imports,
      historical_active_vs_baseline=historical,
      faithful_import=all(x['exact'] for x in imports.values()),
      hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
              for p in args.output.glob('*.npz')},
      limitation='CPU generated tiny fixture; no tolerance relaxation, no CUDA equivalence or scientific claim')
  (args.output/'comparison.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
  print(json.dumps(dict(versions=versions,faithful_import=report['faithful_import'],
      historical_different_arrays={k:len(v['differences']) for k,v in historical.items()})))
  if not report['faithful_import']:
    raise AssertionError('Isolated import changed fixed-input original DtL computation')


if __name__ == '__main__':
  main()
