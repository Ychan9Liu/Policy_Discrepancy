"""CPU-only, fresh-output audit and exact W&B transport of authorized runs.

Scientific inputs/metrics are unchanged. Numeric W&B mirrors are display fields;
the exact canonical JSON string and its SHA256 are verified byte for byte.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

def read(path):
  return json.loads(Path(path).read_text())

def sha(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value):
  Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                  allow_nan=False) + '\n')

def exact_payload(values):
  encoded = json.dumps(values, sort_keys=True, separators=(',', ':'),
                       allow_nan=False)
  return encoded, hashlib.sha256(encoded.encode()).hexdigest()

def verify_transport(summary, values, encoded, digest):
  if (summary.get('formal/exact_values_json') != encoded or
      summary.get('formal/exact_values_sha256') != digest):
    raise RuntimeError('Exact W&B payload/hash differs')
  differences = {}
  for key, value in values.items():
    actual = summary.get(key)
    if isinstance(value, float):
      if not isinstance(actual, (int, float)) or not math.isfinite(actual):
        raise RuntimeError('Missing/nonfinite numeric display field: ' + key)
      if actual != value:
        differences[key] = dict(authoritative=value, numeric_display=actual)
    elif actual != value:
      raise RuntimeError('Nonnumeric W&B field differs: ' + key)
  return differences

def publish(path, values):
  import wandb
  encoded, digest = exact_payload(values)
  for attempt in range(3):
    try:
      remote = wandb.Api().run(path)
      if remote.state != 'finished':
        raise RuntimeError('W&B run/upload is not finished: ' + path)
      remote.summary.update(dict(values, **{
          'formal/exact_values_json': encoded,
          'formal/exact_values_sha256': digest,
          'formal/numeric_fields_are_display_mirrors': True}))
      actual = wandb.Api().run(path)
      differences = verify_transport(actual.summary, values, encoded, digest)
      return dict(wandb_path=path, exact_payload_sha256=digest,
                  exact_transport_verified=True, numeric_differences=differences)
    except Exception:
      if attempt == 2:
        raise
      time.sleep(10)

def main(args):
  if args.output.exists():
    raise FileExistsError('Audit attempts must use new output directories')
  args.output.mkdir(parents=True)
  dump(args.output / 'waiting.json', dict(roots=[str(p) for p in args.roots],
       created_utc=datetime.now(timezone.utc).isoformat(), training_started=False))
  while not all((p / 'supervisor-final.json').exists() for p in args.roots):
    if not args.wait:
      raise RuntimeError('Training has not finished')
    import psutil
    for root in args.roots:
      if (root / 'supervisor-final.json').exists():
        continue
      pid = read(root / 'prelaunch.json')['supervisor_pid']
      try:
        proc = psutil.Process(pid)
        command = proc.cmdline()
        plans = [Path(x) for x in command if x.endswith('/plan.json')]
        if (proc.status() == psutil.STATUS_ZOMBIE or not any(
            'run_formal' in x for x in command) or not any(
            p.exists() and Path(read(p)['root']) == root for p in plans)):
          raise RuntimeError('Expected supervisor is absent: ' + str(root))
      except psutil.NoSuchProcess:
        raise RuntimeError('Supervisor exited without final audit: ' + str(root))
    time.sleep(30)
  rows = []
  for root in args.roots:
    final = read(root / 'supervisor-final.json')
    if final['return_audit_exit_code'] != 0:
      raise RuntimeError('Supervisor return audit failed')
    for row in final['runs']:
      if row['state'] not in ('c_frozen_pending_return_audit',
                               'training_finished_pending_audit'):
        raise RuntimeError('Training failed/incomplete: ' + row['run_id'])
      rows.append(row)
  if len(rows) != 4 * len(args.roots) or len({
      (r['task'], r['group'], r['seed']) for r in rows}) != len(rows):
    raise RuntimeError('Unique predefined task/group/seed runs required')
  index = args.output / 'RUN_INDEX.json'
  dump(index, dict(runs=rows))
  command = [sys.executable, '-m', 'analysis.return_metrics', '--index',
             str(index), '--output', str(args.output / 'return-audit')]
  if len(rows) == 16:
    command += ['--require-complete']
  with (args.output / 'return-audit.log').open('w') as log:
    subprocess.run(command, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu'), check=True)
  tables = read(args.output / 'return-audit/tables.json')
  if len(tables['seed_metrics']) != len(rows) or any(not r['auc_complete'] or
      not r['tail_complete'] for r in tables['seed_metrics']):
    raise RuntimeError('A completed run has incomplete formal metrics')
  sys.path.insert(0, str(args.repo))
  from dreamerv3.freeze_c import freeze
  verified = []
  for row in rows:
    directory = Path(row['directory'])
    manifest = read(directory / 'protocol_manifest.json')
    config = manifest['config']
    if (manifest['git_commit'] != row['code_sha'] or config['seed'] != 0 or
        config['run.engineering_fixture'] or config['jax.prealloc'] is not False
        or config['logger.outputs'] != ['jsonl', 'wandb'] or
        config != json.loads(json.dumps(row['config']))):
      raise RuntimeError('Execution manifest/config differs from the plan')
    final = read(directory / 'final_state.json')
    receipts = [json.loads(x) for x in
                (directory / 'update_receipts.jsonl').read_text().splitlines()]
    if ([r['update_id'] for r in receipts] != list(range(len(receipts))) or
        final['train_action_steps'] != 1000000 or final['updates'] != len(receipts)
        or final['evaluations'] != 21 or any(r['invalid_count'] or
        r['match_count'] < 0 or not math.isfinite(r['match_sum']) for r in receipts)):
      raise RuntimeError('Final action/update/evaluation/receipt audit failed')
    metric = next(m for m in tables['seed_metrics'] if m['run_id'] == row['run_id'])
    overall = next(m for m in tables['overall_metrics'] if m['group'] == row['group'])
    if not overall['auc_complete'] or not overall['tail_complete']:
      raise RuntimeError('Four-task group aggregate incomplete')
    values = {'formal/auc_return_per_budget': metric['auc_return_per_budget'],
        'formal/tail_mean_return': metric['tail_mean_return'],
        'formal/overall_group_auc': overall['auc_return_per_budget'],
        'formal/overall_group_tail': overall['tail_mean_return'],
        'formal/return_metrics_complete': True}
    if row['group'] in ('baseline', 'constant'):
      artifact = (directory / 'c_frozen.json' if row['group'] == 'baseline'
                  else Path(config['run.frozen_c_file']))
      before = sha(artifact)
      frozen = freeze(Path(read(artifact)['baseline_dir']), artifact)
      if (sha(artifact) != before or frozen['task'] != row['task'] or
          frozen['engineering_fixture'] or (row['group'] == 'constant' and
          config['agent.rep_probe.c'] != frozen['c'])):
        raise RuntimeError('Exact formal c/source audit failed')
      values.update({'formal/c': frozen['c'], 'formal/match_sum': frozen['match_sum'],
          'formal/match_count': frozen['match_count'],
          'formal/c_receipt_sha256': frozen['receipt_sha256'],
          'formal/c_artifact_sha256': before})
    if row['group'] == 'dt':
      for ref in ('baseline', 'constant', 'shuffle'):
        diff = [x for x in tables['dt_differences']
                if x['comparison'] == 'dt-' + ref and x['task'] == row['task']]
        if diff and diff[0]['auc_complete'] and diff[0]['tail_complete']:
          values.update({f'formal/dt_minus_{ref}_auc': diff[0]['auc_return_per_budget'],
                         f'formal/dt_minus_{ref}_tail': diff[0]['tail_mean_return']})
    transport = publish(f'{row["wandb_entity"]}/PD_1/{row["wandb_id"]}', values)
    verified.append(dict(run_id=row['run_id'], values=values, **transport,
        original_exit_code=row.get('exit_code'),
        original_exit_code_available=row.get('exit_code_available', True)))
    dump(args.output / 'published-progress.json', dict(runs=verified))
  dump(args.output / 'completion.json', dict(
      checked_utc=datetime.now(timezone.utc).isoformat(),
      source_sha256=sha(__file__), return_tables_sha256=sha(args.output /
      'return-audit/tables.json'), runs=verified, verified_run_count=len(rows),
      full_four_groups_complete=len(rows) == 16,
      interpretation='Single training seed descriptive results. No tolerance changes. '
        'W&B exact JSON/hash is authoritative; numeric mirrors may round. '
        'Original postprocess failures retained; no training was restarted.'))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--roots', nargs='+', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--repo', type=Path, required=True)
  parser.add_argument('--wait', action='store_true')
  args = parser.parse_args()
  try:
    main(args)
  except Exception as error:
    if args.output.exists():
      dump(args.output / 'failure.json', dict(
          utc=datetime.now(timezone.utc).isoformat(),
          error=type(error).__name__ + ': ' + str(error)))
    raise
