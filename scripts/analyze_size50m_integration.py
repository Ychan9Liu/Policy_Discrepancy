"""Audit supervisor artifacts and estimate resources from engineering runs.

Reads artifacts only. Throughput excludes the initial compilation and evaluation
pauses. Evaluation timing includes snapshot IO and serial episode execution.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
import statistics
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read(path):
  return json.loads(path.read_text())


def rows(path):
  return [json.loads(line) for line in path.read_text().splitlines() if line]


def file_sizes(directory):
  sizes = {}
  for subdir in ('ckpt', 'eval_snapshots', 'replay'):
    sizes[subdir] = sum(p.stat().st_size for p in (directory / subdir).rglob('*')
        if p.is_file())
  sizes['total'] = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
  return sizes


def checkpoint_events(path):
  events = []
  for line in path.read_text().splitlines():
    stamp, _, text = line.partition(' ')
    try:
      stamp = float(stamp)
    except ValueError:
      continue
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    if 'Saving checkpoint:' in text:
      events.append(dict(epoch=stamp, path=text.split('Saving checkpoint:', 1)[1].strip()))
  return events


def checkpoint_parameters(directory):
  latest = (directory / 'latest').read_text().strip()
  with (directory / latest / 'agent.pkl').open('rb') as file:
    data = pickle.load(file)
  digest = hashlib.sha256()
  for key, value in sorted(data['params'].items()):
    array = np.asarray(value)
    digest.update(key.encode())
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
  return dict(params_sha256=digest.hexdigest(), counters=data['counters'])


def analyze_run(directory):
  execution = read(directory / 'execution-initial.json')
  manifest = read(directory / 'protocol_manifest.json')
  resources = rows(directory / 'resources-initial.jsonl')
  final_path = directory / 'final_state.json'
  result = dict(directory=str(directory), task=manifest['task'],
      mode=manifest['config']['agent.rep_probe.mode'],
      config=manifest['config'], git_commit=manifest['git_commit'],
      execution=execution, sizes=file_sizes(directory))
  if not final_path.exists():
    result['complete'] = False
    return result
  result['complete'] = True
  result['final'] = read(final_path)
  result['audit'] = read(directory / 'audit.json') if (directory / 'audit.json').exists() else None
  evaluations = rows(directory / 'evaluations.jsonl')
  events = checkpoint_events(directory / 'stdout-initial.log')
  eval_times = []
  for event_index, event in enumerate(events):
    if '/eval_snapshots/' in event['path']:
      point = int(event['path'].split('/eval_snapshots/')[1].split('/')[0])
      following = next((e for e in events[event_index+1:] if '/ckpt/' in e['path']), None)
      if following:
        eval_times.append(dict(point=point, start=event['epoch'],
            end=following['epoch'], seconds=following['epoch']-event['epoch']))
  assert [e['point'] for e in eval_times] == [0, 2049, 4098]
  result['evaluation_timing'] = eval_times
  episodes = manifest['config']['run.eval_eps']
  result['eval_seconds_per_episode'] = statistics.median(
      e['seconds']/episodes for e in eval_times[1:])
  # Only the second training segment is timed, with 64 warmup updates excluded.
  lo, hi = eval_times[1]['end'], eval_times[2]['start']
  timed = [r for r in resources if lo < r['epoch'] < hi and r['last_receipt']]
  if timed:
    first_update = timed[0]['last_receipt']['update_id']
    timed = [r for r in timed if r['last_receipt']['update_id'] >= first_update+64]
  if len(timed) >= 2:
    first, last = timed[0], timed[-1]
    elapsed = last['epoch'] - first['epoch']
    actions = last['last_receipt']['start_action'] - first['last_receipt']['start_action']
    updates = last['last_receipt']['update_id'] - first['last_receipt']['update_id']
    result['steady_training'] = dict(seconds=elapsed, actions=actions,
        updates=updates, actions_per_second=actions/elapsed,
        updates_per_second=updates/elapsed, warmup_updates_excluded=64,
        note='wall throughput includes environment, replay and receipt IO; excludes evaluations')
  else:
    result['steady_training'] = None
  result['evaluation_lengths'] = [e['lengths'] for e in evaluations]
  result['snapshot_index'] = [dict(path=str(p.relative_to(directory)), bytes=p.stat().st_size)
      for p in (directory / 'eval_snapshots').rglob('*') if p.is_file()]
  result['snapshots'] = []
  for evaluation in evaluations:
    point = evaluation['action_step']
    saved = checkpoint_parameters(directory / 'eval_snapshots' / f'{point:07d}')
    assert saved['counters']['updates'] == evaluation['update_id']
    if point == 0:
      assert saved['counters']['actions'] == saved['counters']['updates'] == 0
    result['snapshots'].append(dict(action_step=point, **saved))
  assert result['snapshots'][-1]['params_sha256'] == result['final']['params_sha256']
  terminal = checkpoint_parameters(directory / 'ckpt')
  assert terminal['params_sha256'] == result['final']['params_sha256']
  result['endpoint_evaluation_did_not_modify_params'] = True
  result['artifact_sha256'] = {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
      for name in ('protocol_manifest.json', 'config.yaml', 'update_receipts.jsonl',
          'evaluations.jsonl', 'final_state.json')}
  from embodied.run.protocol_v1 import ProtocolState
  config = manifest['config']
  ledger = ProtocolState(*(config[k] for k in ('run.action_budget',
      'run.eval_every_actions', 'run.eval_eps', 'run.match_start', 'run.match_end')))
  receipts = rows(directory / 'update_receipts.jsonl')
  def record(row):
    return ledger.record_update(row['update_id'], row['start_action'],
        row['match_sum'], row['match_count'], row['invalid_count'])
  assert all(record(row) for row in receipts)
  before = (ledger.updates, ledger.match_sum, ledger.match_count)
  assert not any(record(row) for row in receipts)
  assert before == (ledger.updates, ledger.match_sum, ledger.match_count)
  conflicting = dict(receipts[0], match_sum=receipts[0]['match_sum']+.125)
  try:
    record(conflicting)
  except ValueError as error:
    assert 'Conflicting receipt' in str(error)
  else:
    raise AssertionError('Conflicting real receipt was accepted')
  result['receipt_audit'] = dict(unique_updates=ledger.updates,
      duplicate_replay_of_receipts_ignored=True, conflicting_duplicate_rejected=True,
      included_ids=[r['update_id'] for r in receipts if 2200 <= r['start_action'] < 3600])
  frozen_path = directory.parent / 'logging' / 'c_frozen_engineering.json'
  frozen = read(frozen_path)
  assert frozen['engineering_fixture'] and frozen['root_seed'] == 0
  assert frozen_path.with_suffix('.json.sha256').read_text().strip() == hashlib.sha256(
      frozen_path.read_bytes()).hexdigest()
  if result['mode'] == 'logging':
    assert ledger.match_sum == result['final']['match_sum']
    assert ledger.match_count == result['final']['match_count']
    result['recomputed_engineering_c'] = ledger.frozen_c()
    if directory.name == 'logging':
      assert ledger.match_sum == frozen['match_sum']
      assert ledger.match_count == frozen['match_count']
      assert hashlib.sha256((directory/'update_receipts.jsonl').read_bytes()).hexdigest() == frozen['receipt_sha256']
      assert ledger.frozen_c() == frozen['c']
  if result['mode'] == 'constant':
    assert config['agent.rep_probe.c'] == frozen['c']
    result['constant_source'] = dict(path=str(frozen_path), c=frozen['c'],
        sha256=hashlib.sha256(frozen_path.read_bytes()).hexdigest())
  return result


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--root', required=True)
  parser.add_argument('--output', required=True)
  args = parser.parse_args()
  root = Path(args.root)
  results = [analyze_run(p.parent) for p in sorted(root.rglob('execution-initial.json'))
      if (p.parent/'protocol_manifest.json').exists()]
  payload = dict(root=str(root), runs=results, engineering_only=True)
  Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True)+'\n')
  for r in results:
    print(json.dumps({k:r[k] for k in ('task', 'mode', 'directory', 'complete')}, sort_keys=True))


if __name__ == '__main__':
  main()
