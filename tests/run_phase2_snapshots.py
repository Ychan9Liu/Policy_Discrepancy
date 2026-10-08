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
  parser.add_argument('--replay_context', type=int, default=1)
  parser.add_argument('--free_nats', type=float, default=0.1)
  parser.add_argument('--imag_last', type=int, default=2)
  parser.add_argument('--sequence_case', choices=(
      'first_terminal', 'continuation', 'interior_reset', 'chunk_chain'),
      default='first_terminal')
  parser.add_argument('--second_chunk', action='store_true')
  parser.add_argument('--train_devices', nargs='+', type=int, default=[0])
  parser.add_argument('--bench_updates', type=int, default=0)
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
        '--dtype', args.dtype, '--replay_context', str(args.replay_context),
        '--free_nats', str(args.free_nats), '--imag_last', str(args.imag_last),
        '--sequence_case', args.sequence_case,
        '--train_devices', *(str(x) for x in args.train_devices),
        '--bench_updates', str(args.bench_updates),
        '--alpha', str(alpha), '--c', str(c), '--seed', str(args.seed)]
    if args.second_chunk:
      command.append('--second_chunk')
    print(f'RUN {name}: {path}', flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)
    print(f'PASS {name}: {path}', flush=True)


if __name__ == '__main__':
  main()
