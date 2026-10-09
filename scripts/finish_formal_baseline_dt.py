"""Dependent raw-return audit of the eight authorized baseline/Dt runs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def read(path):
  return json.loads(Path(path).read_text())

def main(args):
  import psutil
  watched = [(psutil.Process(pid), psutil.Process(pid).create_time())
             for pid in args.supervisor_pids]
  roots = [args.baseline_root, args.dt_root]
  while not all((r / 'supervisor-final.json').exists() for r in roots):
    for root, (process, created) in zip(roots, watched):
      if not (root / 'supervisor-final.json').exists() and (
          not process.is_running() or process.status() == psutil.STATUS_ZOMBIE
          or process.create_time() != created):
        raise RuntimeError('Supervisor stopped without final audit: ' + str(root))
    time.sleep(30)
  if args.output.exists():
    raise FileExistsError('Use a new comparison audit output')
  args.output.mkdir(parents=True)
  entries = []
  for root in roots:
    final = read(root / 'supervisor-final.json')
    if final['return_audit_exit_code'] != 0:
      raise RuntimeError('Source return audit failed')
    for row in final['runs']:
      expected = ('c_frozen_pending_return_audit' if row['group'] == 'baseline'
                  else 'training_finished_pending_audit')
      if row['state'] != expected:
        raise RuntimeError('Run incomplete or failed: ' + row['run_id'])
      entries.append(row)
  if len(entries) != 8 or len({(r['task'], r['group'], r['seed']) for r in entries}) != 8:
    raise RuntimeError('Eight unique predefined runs required')
  index = args.output / 'RUN_INDEX.json'
  index.write_text(json.dumps(dict(runs=entries), indent=2) + '\n')
  env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu')
  subprocess.run([sys.executable, '-m', 'analysis.return_metrics',
      '--index', str(index), '--output', str(args.output / 'return-audit')],
      cwd=args.repo, env=env, check=True)
  tables = read(args.output / 'return-audit/tables.json')
  if len(tables['seed_metrics']) != 8 or any(not r['auc_complete'] or
      not r['tail_complete'] for r in tables['seed_metrics']):
    raise RuntimeError('One of the baseline/Dt metrics is incomplete')
  difference = [r for r in tables['dt_differences'] if r['comparison'] == 'dt-baseline']
  if len(difference) != 5 or any(not r['auc_complete'] or not r['tail_complete'] for r in difference):
    raise RuntimeError('Four task and overall Dt-baseline differences incomplete')
  import wandb
  published = []
  for row in entries:
    metric = next(r for r in tables['seed_metrics'] if r['run_id'] == row['run_id'])
    values = {'formal/auc_return_per_budget': metric['auc_return_per_budget'],
              'formal/tail_mean_return': metric['tail_mean_return'],
              'formal/return_metrics_complete': True}
    if row['group'] == 'dt':
      task = next(r for r in difference if r['task'] == row['task'])
      overall = next(r for r in difference if r['level'] == 'overall')
      values.update({'formal/dt_minus_baseline_auc': task['auc_return_per_budget'],
          'formal/dt_minus_baseline_tail': task['tail_mean_return'],
          'formal/overall_dt_minus_baseline_auc': overall['auc_return_per_budget'],
          'formal/overall_dt_minus_baseline_tail': overall['tail_mean_return']})
    path = f'{row["wandb_entity"]}/PD_1/{row["wandb_id"]}'
    for attempt in range(3):
      try:
        remote = wandb.Api().run(path)
        if remote.state != 'finished':
          raise RuntimeError('Run/upload not finished: ' + path)
        remote.summary.update(values)
        verified = wandb.Api().run(path)
        if any(verified.summary.get(k) != v for k,v in values.items()):
          raise RuntimeError('W&B summary differs')
        published.append(dict(run_id=row['run_id'], wandb_path=path, values=values))
        break
      except Exception:
        if attempt == 2:
          raise
        time.sleep(30)
  result = dict(checked_utc=datetime.now(timezone.utc).isoformat(),
      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      published=published, baseline_dt_complete=True, full_four_groups_complete=False,
      interpretation='Single-seed descriptive comparisons only. Constant/shuffle absent; '
                     'device/resources and failures require final 04 review.')
  (args.output / 'completion.json').write_text(json.dumps(result, indent=2)+'\n')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--baseline-root', type=Path, required=True)
  parser.add_argument('--dt-root', type=Path, required=True)
  parser.add_argument('--supervisor-pids', nargs=2, type=int, required=True)
  parser.add_argument('--repo', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  try:
    main(args)
  except Exception as error:
    target = args.dt_root / 'comparison-postprocess-failure.json'
    target.write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
        error=type(error).__name__ + ': ' + str(error)), indent=2) + '\n')
    raise
