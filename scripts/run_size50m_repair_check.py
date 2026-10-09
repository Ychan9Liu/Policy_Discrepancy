"""Run one isolated, reduced-budget size50m repair check on a pinned server.

Uses the shared supervisor's GPU occupancy guard and evidence capture. This is
an engineering fixture, never a formal protocol run.
"""

import argparse
from pathlib import Path
import subprocess

from scripts.run_size50m_integration import base_command, supervise


def replace(command, flag, value):
  command[command.index(flag) + 1] = str(value)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--directory', type=Path, required=True)
  parser.add_argument('--gpu', type=int, required=True)
  parser.add_argument('--kind', choices=('physics', 'report'), required=True)
  args = parser.parse_args()
  status = subprocess.check_output(['git', 'status', '--porcelain'], text=True)
  if status.strip():
    raise RuntimeError('Server checkout must be clean')
  if args.directory.exists():
    raise RuntimeError('Use a new independent output directory')
  command = base_command('dmc_hopper_hop', args.directory, 'logging',
                         episodes=1, report=2048 if args.kind == 'report' else 0)
  for flag, value in (('--run.action_budget', 2048),
                      ('--run.eval_every_actions', 2048),
                      ('--run.match_start', 2032),
                      ('--run.match_end', 2048)):
    replace(command, flag, value)
  if args.kind == 'physics':
    command += ['--run.engineering_trace_actions', '2048',
                '--run.engineering_trace_updates', '5',
                '--env.dmc.engineering_trace_physics', 'True']
  code = supervise(command, args.directory, 'initial', args.gpu)
  if code:
    raise SystemExit(code)


if __name__ == '__main__':
  main()
