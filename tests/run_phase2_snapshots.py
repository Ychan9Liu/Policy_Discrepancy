"""Run each full-agent fixture in a fresh process at a fixed Git checkout."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_dir', required=True)
  parser.add_argument('--baseline_src', required=True)
  parser.add_argument('--platform', choices=('cpu', 'cuda'), default='cpu')
  parser.add_argument('--dtype', choices=('float32', 'bfloat16'), default='float32')
  parser.add_argument('--modes', default='baseline,off,logging,dt0,c1,dt,constant,shuffle')
  parser.add_argument('--alpha', type=float, default=50000.0)
  parser.add_argument('--c', type=float, default=0.6)
  parser.add_argument('--seed', type=int, default=7)
  args = parser.parse_args()
  repo = Path(__file__).resolve().parents[1]
  output = Path(args.output_dir).resolve()
  output.mkdir(parents=True, exist_ok=True)
  script = repo / 'tests' / 'snapshot_agent_identity.py'
  for name in args.modes.split(','):
    mode = {'dt0': 'dt', 'c1': 'constant', 'shuffle0': 'shuffle'}.get(name, name)
    alpha = 0 if name in ('dt0', 'shuffle0') else args.alpha
    c = 1 if name == 'c1' else args.c
    cwd = Path(args.baseline_src).resolve() if mode == 'baseline' else repo
    env = dict(os.environ, PYTHONPATH=str(cwd))
    path = output / f'{name}-{args.platform}.npz'
    command = [sys.executable, str(script), '--mode', mode,
        '--output', str(path), '--platform', args.platform,
        '--dtype', args.dtype, '--replay_context', '1',
        '--free_nats', '0.1', '--imag_last', '2',
        '--alpha', str(alpha), '--c', str(c), '--seed', str(args.seed)]
    print(f'RUN {name}: {path}', flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)
    print(f'PASS {name}: {path}', flush=True)


if __name__ == '__main__':
  main()
