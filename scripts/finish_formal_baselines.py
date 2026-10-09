"""Wait for the authorized campaign, validate baseline outputs, publish summary.

No training, GPU use, recovery, seed selection, metric changes or other modes.
This is a dependent server job, not a scheduled chat or recurring automation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

def read(path):
  return json.loads(Path(path).read_text())

def sha(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def finish(root, supervisor_pid):
  import psutil
  supervisor = psutil.Process(supervisor_pid)
  created = supervisor.create_time()
  while not (root / 'supervisor-final.json').exists():
    if (not supervisor.is_running() or supervisor.status() == psutil.STATUS_ZOMBIE
        or supervisor.create_time() != created):
      raise RuntimeError('Supervisor ended without final audit index')
    time.sleep(30)
  final = read(root / 'supervisor-final.json')
  if final['return_audit_exit_code'] != 0:
    raise RuntimeError('Return audit failed; preserve inputs, no metric substitution')
  output = Path(final['output'])
  tables = read(output / 'tables.json')
  baseline = [r for r in tables['seed_metrics'] if r['group'] == 'baseline']
  tasks = {'dmc_hopper_hop', 'dmc_quadruped_run',
           'dmc_quadruped_walk', 'dmc_reacher_hard'}
  if len(baseline) != 4 or {r['task'] for r in baseline} != tasks:
    raise RuntimeError('Four unique baseline tasks are required')
  overall = next(r for r in tables['overall_metrics'] if r['group'] == 'baseline')
  assert overall['auc_complete'] and overall['tail_complete']
  results = []
  for row in final['runs']:
    if row['state'] != 'c_frozen_pending_return_audit':
      raise RuntimeError('Baseline did not finish and freeze: ' + row['task'])
    d = Path(row['directory'])
    manifest = read(d / 'protocol_manifest.json')
    config = manifest['config']
    if (manifest['git_commit'] != row['code_sha'] or config['run.engineering_fixture']
        or config['seed'] != 0 or config['logger.outputs'] != ['jsonl', 'wandb']
        or config['jax.prealloc'] is not False):
      raise RuntimeError('Completed manifest differs from approved execution')
    receipts = [json.loads(x) for x in (d / 'update_receipts.jsonl').read_text().splitlines()]
    ids = [r['update_id'] for r in receipts]
    if ids != list(range(len(ids))):
      raise RuntimeError('Persisted receipt IDs duplicated, missing or reordered')
    if any(r['invalid_count'] or r['match_count'] < 0 or
           not math.isfinite(r['match_sum']) for r in receipts):
      raise RuntimeError('Invalid matching receipt')
    artifact = read(d / 'c_frozen.json')
    if sha(d / 'c_frozen.json') != (d / 'c_frozen.json.sha256').read_text().strip():
      raise RuntimeError('Frozen c checksum mismatch')
    included = [r for r in receipts if 100000 <= r['start_action'] < 300000]
    total = sum(r['match_sum'] for r in included)
    count = sum(r['match_count'] for r in included)
    if (not count or total != artifact['match_sum'] or count != artifact['match_count']
        or artifact['c'] != total / count or artifact['engineering_fixture']
        or artifact['receipt_sha256'] != sha(d / 'update_receipts.jsonl')
        or not 0 < artifact['c'] <= 1):
      raise RuntimeError('Formal c source recomputation differs')
    state = read(d / 'final_state.json')
    if state['train_action_steps'] != 1000000 or state['updates'] != len(ids):
      raise RuntimeError('Final budget or successful-update count differs')
    metric = next(r for r in baseline if r['task'] == row['task'])
    if not metric['auc_complete'] or not metric['tail_complete'] or metric['seed'] != 0:
      raise RuntimeError('Baseline return incomplete; no shortened metrics')
    values = {'formal/c': artifact['c'], 'formal/match_sum': total,
        'formal/match_count': count, 'formal/c_receipt_sha256': artifact['receipt_sha256'],
        'formal/auc_return_per_budget': metric['auc_return_per_budget'],
        'formal/tail_mean_return': metric['tail_mean_return'],
        'formal/baseline_outputs_verified': True,
        'formal/overall_baseline_auc': overall['auc_return_per_budget'],
        'formal/overall_baseline_tail': overall['tail_mean_return']}
    results.append(dict(task=row['task'], directory=str(d), values=values,
        wandb_path=f'{row["wandb_entity"]}/PD_1/{row["wandb_id"]}',
        original_exit_code=row.get('exit_code'),
        exit_code_available=row.get('exit_code_available', True)))
  import wandb
  for row in results:
    for attempt in range(3):
      try:
        api = wandb.Api()
        remote = api.run(row['wandb_path'])
        if remote.state != 'finished':
          raise RuntimeError('W&B upload/run state is not finished')
        remote.summary.update(row['values'])
        refreshed = wandb.Api().run(row['wandb_path'])
        if any(refreshed.summary.get(k) != v for k, v in row['values'].items()):
          raise RuntimeError('Remote summary verification failed')
        row['wandb_summary_verified'] = True
        break
      except Exception:
        if attempt == 2:
          raise
        time.sleep(30)
  return dict(checked_utc=datetime.now(timezone.utc).isoformat(),
      postprocessor_sha256=sha(__file__), results=results,
      return_tables_sha256=sha(output / 'tables.json'),
      baseline_complete=True, all_four_groups_complete=False,
      interpretation='Single-seed baseline description; no M2 effectiveness conclusion. '
        'Adopted processes have unavailable original exit codes; inputs/snapshots and '
        'W&B finished status audited. Device/resource records still need final 04 review.')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--root', type=Path, required=True)
  parser.add_argument('--supervisor-pid', type=int, required=True)
  args = parser.parse_args()
  try:
    result = finish(args.root, args.supervisor_pid)
    target = args.root / 'baseline-completion-audit.json'
  except Exception as e:
    result = dict(utc=datetime.now(timezone.utc).isoformat(),
        error=type(e).__name__ + ': ' + str(e), baseline_complete=False)
    target = args.root / 'baseline-completion-audit-failed.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    raise
  target.write_text(json.dumps(result, indent=2) + '\n')
