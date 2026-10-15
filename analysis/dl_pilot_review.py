"""CPU-only evidence review for the completed DL-pilot-r3, not a new experiment.

Independently recomputes returns, checks persisted checkpoints and optimizer
receipts, and summarizes existing batch-level gate diagnostics. It never loads
an environment, agent, replay, or JAX backend, and never tunes a parameter.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import statistics

VERSION = 'DL-pilot-review-r1'
TRAIN_SHA = '13ec72a214b0c03b886b2c7fac52c8f089966dec'
TASK, BUDGET, INTERVAL, EPISODES, UPDATES = 'dmc_quadruped_walk', 1000000, 50000, 10, 249493
GRID = list(range(0, BUDGET + 1, INTERVAL))
INITIAL = '18199ec5fb45b13f18ec29ae6dcc438d3018807888ce9f2c5e6ee64b621a9267'


def require(condition, message):
  if not condition:
    raise ValueError(message)


def sha(path):
  value = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for block in iter(lambda: stream.read(2**20), b''):
      value.update(block)
  return value.hexdigest()


def read(path):
  return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonl(path):
  with Path(path).open(encoding='utf-8') as stream:
    for line in stream:
      yield json.loads(line)


def seed32(point, episode):
  parts = ['0', TASK, 'eval_env', str(point // INTERVAL), str(episode)]
  return int.from_bytes(hashlib.sha256('\0'.join(parts).encode()).digest()[:4], 'big')


def close(a, b):
  # Existing return_metrics.py arithmetic tolerances; no scientific gate.
  return math.isclose(a, b, abs_tol=1e-9, rel_tol=1e-12)


def evaluations(rows):
  require([r['action_step'] for r in rows] == GRID, 'Incomplete fixed 1M grid')
  result = []
  for row in rows:
    point = row['action_step']
    scores = row['returns_raw']
    require(len(scores) == len(row['episodes']) == EPISODES and
        row['scores'] == scores and all(math.isfinite(x) for x in scores), 'Invalid raw episodes')
    seeds = [seed32(point, i) for i in range(EPISODES)]
    require(row['seeds'] == seeds and row['lengths'] == [1000] * EPISODES, 'Seed/episode length differs')
    require(row['snapshot_action_step'] == row['target_action_step'] == point and
        row['snapshot_update_id'] == row['update_id'] == (point // 4 - 507 if point else 0),
        'Snapshot/action/update binding differs')
    require(row['reward_source'] == 'environment_interface.reward' and
        row['return_definition'] == 'undiscounted_sum_non_reset_rewards_repeat_already_summed',
        'Raw return definition differs')
    for i, episode in enumerate(row['episodes']):
      require(episode['complete'] and episode['episode_id'] == i and
          episode['environment_seed'] == seeds[i] and episode['length'] == 1000 and
          episode['return_raw'] == scores[i], 'Conflicting episode detail')
    mean = math.fsum(scores) / EPISODES
    require(close(mean, row['mean']) and close(mean, row['mean_return_raw']), 'Stored mean differs')
    result.append(dict(action_step=point, mean=mean, sd=statistics.stdev(scores),
        minimum=min(scores), maximum=max(scores), median=statistics.median(scores),
        returns_raw=scores, seeds=seeds, snapshot_id=row['snapshot_id']))
  return result


def area(points, start, end):
  points = [p for p in points if start <= p['action_step'] <= end]
  require(points[0]['action_step'] == start and points[-1]['action_step'] == end, 'Window edges absent')
  return math.fsum((b['action_step'] - a['action_step']) * (a['mean'] + b['mean']) / 2
      for a, b in zip(points, points[1:])) / (end - start)


def metrics(points):
  tail = [p['mean'] for p in points if p['action_step'] >= 800000]
  return dict(auc_per_1m_actions=area(points, 0, BUDGET),
      tail_mean_return=math.fsum(tail) / len(tail), endpoint_return=points[-1]['mean'],
      descriptive_windows={f'{a}_{b}': area(points, a, b)
          for a, b in ((0, 300000), (300000, 800000), (800000, 1000000))})


class Stat:
  def __init__(self):
    self.n = 0
    self.total = 0.
    self.correction = 0.
    self.minimum, self.maximum = math.inf, -math.inf

  def add(self, value):
    require(math.isfinite(value), 'Nonfinite diagnostic scalar')
    self.n += 1
    delta = value - self.correction
    updated = self.total + delta
    self.correction = (updated - self.total) - delta
    self.total = updated
    self.minimum, self.maximum = min(self.minimum, value), max(self.maximum, value)

  def result(self):
    return dict(n=self.n, mean=self.total / self.n if self.n else None,
        minimum=self.minimum if self.n else None, maximum=self.maximum if self.n else None)


def ratio(numerator, denominator):
  return numerator / denominator if denominator > 0 else None


def receipts(path, protocol):
  fields = ('actual_v_mean', 'actual_v_min', 'active_frac', 'valid_active_frac',
      'active_protected_frac', 'e_valid_mean', 'D_valid_mean', 'signed_support_valid_mean',
      'u_q_valid_mean', 'u_p_valid_mean', 'rep_before_mean', 'rep_after_mean',
      'excess_release_mean', 'weighted_excess_mean', 'calibration_num', 'calibration_den')
  total = {key: Stat() for key in fields}
  bins = {}
  valid_count = Stat()
  weighted = {key: Stat() for key in fields if key.endswith('_valid_mean')}
  unavailable = {key: 0 for key in fields}
  positive_updates, n = 0, 0
  for row in jsonl(path):
    update = row['update_id']
    require(update == n and row['start_action'] == 2032 + 4 * update,
        'Optimizer receipt sequence/action schedule differs')
    require(not row['invalid_count'] and row['match_sum'] == row['match_count'] == 0 and
        math.isfinite(row['optimizer_loss']) and math.isfinite(row['optimizer_grad_norm']), 'Invalid optimizer commit')
    require(tuple(protocol['records'][update]) == (row['start_action'], 0., 0, 0),
        'Final checkpoint ledger differs from persisted receipt')
    m = row['metrics']
    valid = float(m['dl/valid_count'])
    require(valid >= 0 and valid + m['dl/unavailable_count'] == 1024, 'Invalid batch mask count')
    valid_count.add(valid)
    require(.5 <= m['dl/actual_v_min'] <= m['dl/actual_v_mean'] <= m['dl/actual_v_max'] <= 1,
        'Actual gate outside frozen rho=.5 bounds')
    interval = min((row['start_action'] - 1) // INTERVAL, 19)
    if interval not in bins:
      bins[interval] = {key: Stat() for key in fields}
    for key in fields:
      value = float(m['dl/' + key])
      if not math.isfinite(value):
        require(valid == 0 and key.endswith('_valid_mean'), 'Nonfinite metric at valid transition')
        unavailable[key] += 1
        continue
      total[key].add(value)
      bins[interval][key].add(value)
      if key in weighted:
        weighted[key].add(value * valid)
    positive_updates += m['dl/excess_release_mean'] > 0
    n += 1
  require(n == UPDATES and len(protocol['records']) == n, 'Incomplete update ledger')
  reduced = {k: s.result() for k, s in total.items()}
  for key in weighted:
    reduced[key]['valid_position_weighted_mean'] = ratio(weighted[key].total, valid_count.total)
  before, after = total['rep_before_mean'].total, total['rep_after_mean'].total
  excess = total['excess_release_mean'].total + total['weighted_excess_mean'].total
  return dict(updates=n, unit='Replay update-batch statistics; resampled positions are not independent states',
      positive_release_updates=positive_updates, unavailable_updates=unavailable, statistics=reduced,
      mean_weight_reduction=1 - reduced['actual_v_mean']['mean'],
      protected_fraction_of_valid_rep_active_positions=ratio(
          total['active_protected_frac'].total, total['valid_active_frac'].total),
      observed_rep_loss_relative_reduction=ratio(before - after, before),
      observed_excess_relative_reduction=ratio(total['excess_release_mean'].total, excess),
      relative_to_D_only_release_on_valid_N_positions=ratio(
          .5 * total['calibration_num'].total, total['calibration_den'].total),
      counterfactual_strength_scope='Same N batches, same H1-valid mask, algebraic D-only excess release; not actual D training or a calibrated kappa',
      bins=[dict(action_start=i*INTERVAL, action_end=(i+1)*INTERVAL,
          **{k: v.result() for k, v in data.items()}) for i, data in sorted(bins.items())])


def checkpoint(path):
  import numpy as np
  data = pickle.loads(Path(path).read_bytes())  # Only hash-bound trusted project output.
  value = hashlib.sha256()
  leaves = {}
  for key, leaf in sorted(data['params'].items()):
    array = np.asarray(leaf)
    value.update(key.encode())
    descriptor = str(array.dtype).encode() + str(array.shape).encode()
    raw = array.tobytes()
    value.update(descriptor)
    value.update(raw)
    leaves[key] = hashlib.sha256(descriptor + raw).hexdigest()
  return dict(path=str(path), file_sha256=sha(path), params_sha256=value.hexdigest(),
      params_fingerprint_sha256=hashlib.sha256(json.dumps(leaves, sort_keys=True).encode()).hexdigest(),
      counters=data['counters'], parameter_leaves=len(data['params']))


def inspect(plan, output):
  plan, output = Path(plan).resolve(), Path(output).resolve()
  require(not output.exists(), 'Use a new evidence review directory')
  index = read(plan / 'RUN_INDEX.json')
  require(index['protocol'] == 'DL-pilot-r3' and index['git_sha'] == TRAIN_SHA and
      index['group'] == 'N' and index['seed'] == 0 and index['task'] == TASK, 'Run identity differs')
  directory = Path(index['logdir'])
  complete = read(plan / 'supervisor-attempt01/supervisor_complete.json')
  original_audit = read(directory / 'audit.json')
  final = read(directory / 'final_state.json')
  require(complete['worker_exit'] == 0 and complete['status'] == 'audited_complete' and
      complete['git_sha'] == TRAIN_SHA and complete['audit'] == original_audit,
      'Supervisor completion/audit binding differs')
  require(not (directory / 'FAILED.json').exists() and
      not (plan / 'supervisor-attempt01/supervisor_failed.json').exists(), 'Failed run cannot pass')
  sources = {}
  for name, expected in original_audit['artifact_sha256'].items():
    actual = sha(directory / name)
    require(actual == expected, 'Completed artifact changed: ' + name)
    sources[str(directory / name)] = actual
  for name, key in [('config.yaml', 'config_sha256'), ('historical_controls.json', 'controls_sha256')]:
    actual = sha(plan / name)
    require(actual == index[key], 'Frozen plan evidence changed')
    sources[str(plan / name)] = actual
  require(sha(directory / 'config.yaml') == index['config_sha256'], 'Actual config file differs')
  require(final['train_action_steps'] == BUDGET and final['updates'] == UPDATES and
      final['train_batch_keys'] == UPDATES and final['eval_action_steps'] == 210000 and
      final['evaluations'] == 21 and final['git_commit'] == TRAIN_SHA, 'Final counts differ')
  manifest = read(directory / 'protocol_manifest.json')
  controls = read(plan / 'historical_controls.json')
  allowed = {'script', 'logdir', 'logger.outputs', 'logger.filter', 'agent.rep_probe.mode'}
  config = manifest['config']
  base = controls['B']['manifest']['config']
  differences = {key: dict(historical=base.get(key), actual=config.get(key))
      for key in set(base) | set(config) if base.get(key) != config.get(key)}
  require(all(key in allowed or key.startswith('agent.dt_latch.') for key in differences),
      'Undeclared critical config differences')
  require(config['agent.dt_latch.mode'] == 'dtlatch' and config['agent.rep_probe.mode'] == 'off' and
      config['agent.dt_latch.alpha'] == 20 and config['agent.dt_latch.rho'] == .5 and
      config['run.action_budget'] == BUDGET and config['run.eval_eps'] == EPISODES and
      config['run.eval_every_actions'] == INTERVAL, 'Frozen DL treatment differs')
  points = {'N': evaluations(list(jsonl(directory / 'evaluations.jsonl')))}
  for group in ('B', 'D'):
    evidence = controls[group]
    for name, expected in evidence['source_sha256'].items():
      path = Path(evidence['directory']) / name
      require(sha(path) == expected, 'Historical control artifact changed')
      sources[str(path)] = expected
    actual_rows = list(jsonl(Path(evidence['directory']) / 'evaluations.jsonl'))
    require(actual_rows == evidence['evaluations'], 'Historical control mirror differs')
    points[group] = evaluations(actual_rows)
  state_audits = list(jsonl(directory / 'evaluation_state_audit.jsonl'))
  require([a['action_step'] for a in state_audits] == GRID and
      all(a['before'] == a['after'] for a in state_audits), 'Evaluation training state changed')
  snapshots = []
  for row, state in zip(points['N'], state_audits):
    snapshot_dir = (directory / row['snapshot_id']).resolve()
    require(directory.resolve() in snapshot_dir.parents and
        (snapshot_dir / 'done').exists(), 'Missing/outside snapshot evidence')
    snapshot = checkpoint(snapshot_dir / 'agent.pkl')
    expected_updates = row['action_step'] // 4 - 507 if row['action_step'] else 0
    require(snapshot['counters']['updates'] == snapshot['counters']['batches'] == expected_updates,
        'Actual snapshot optimizer counters differ')
    train_counters = {k: v for k, v in snapshot['counters'].items() if k not in ('eval_actions', 'eval_seed')}
    require(train_counters == state['before']['counters'] and
        snapshot['params_fingerprint_sha256'] == state['before']['params_sha256'],
        'Actual snapshot parameters/counters differ from evaluation state receipt')
    if not row['action_step']:
      require(snapshot['params_sha256'] == manifest['initial_params_sha256'] == INITIAL,
          'Actual common initial state differs')
    snapshots.append(dict(action_step=row['action_step'], **snapshot))
    print(json.dumps(dict(event='snapshot_verified', action_step=row['action_step'])), flush=True)
  require(snapshots[-1]['params_sha256'] == final['params_sha256'], 'Final evaluated parameters differ')
  checkpoint_dir = directory / 'ckpt' / (directory / 'ckpt/latest').read_text().strip()
  require((checkpoint_dir / 'done').exists(), 'Final checkpoint incomplete')
  final_agent = checkpoint(checkpoint_dir / 'agent.pkl')
  require(final_agent['params_sha256'] == final['params_sha256'] and
      final_agent['counters']['updates'] == UPDATES and
      final_agent['counters']['actions'] == final['train_policy_calls'], 'Final saved agent differs')
  step = pickle.loads((checkpoint_dir / 'step.pkl').read_bytes())
  protocol = pickle.loads((checkpoint_dir / 'protocol.pkl').read_bytes())
  require(step == protocol['actions'] == BUDGET and protocol['updates'] == UPDATES and
      protocol['eval_actions'] == 210000 and protocol['resets'] == final['train_resets'] and
      [int(k) for k in protocol['evaluations']] == GRID, 'Final saved protocol differs')
  actual_rows = list(jsonl(directory / 'evaluations.jsonl'))
  require(protocol['evaluations'] == {r['action_step']: {k: v for k, v in r.items() if k != 'action_step'}
      for r in actual_rows}, 'Final protocol evaluation ledger differs')
  gate = receipts(directory / 'update_receipts.jsonl', protocol)
  resource_rows = list(jsonl(plan / 'supervisor-attempt01/resources.jsonl'))
  devices = [device for row in resource_rows for device in row['actual_devices']]
  require(any('C' in d['type'] for d in devices) and any('G' in d['type'] for d in devices) and
      all(d['physical_gpu'] == 4 and d['uuid'] == index['uuid'] for d in devices),
      'Actual CUDA/EGL training mapping differs')
  peak_memory = max(float(row['memory_used'].split()[0]) for row in resource_rows)
  summaries = {group: metrics(data) for group, data in points.items()}
  require(close(summaries['N']['auc_per_1m_actions'], final['return_auc_per_action_budget']) and
      close(summaries['N']['tail_mean_return'], final['tail_mean_return']) and
      close(summaries['N']['endpoint_return'], final['endpoint_return_raw']), 'Independent final reduction differs')
  comparisons = {}
  for group in ('B', 'D'):
    changes = {key: dict(absolute=summaries['N'][key] - summaries[group][key],
        relative_percent=100 * (summaries['N'][key] / summaries[group][key] - 1))
        for key in ('auc_per_1m_actions', 'tail_mean_return', 'endpoint_return')}
    delta = [a - b for a, b in zip(points['N'][-1]['returns_raw'], points[group][-1]['returns_raw'])]
    changes['endpoint_eval_seed_differences'] = dict(values=delta, mean=math.fsum(delta) / EPISODES,
        sd=statistics.stdev(delta), positive_count=sum(d > 0 for d in delta),
        scope='Ten evaluation seeds for fixed trained policies; no training-seed uncertainty or pixel-pair claim')
    comparisons[group] = changes
  # Inputs must remain identical after checkpoint reads and full ledger review.
  require(all(sha(Path(path)) == expected for path, expected in sources.items()), 'Input changed during review')
  result = dict(version=VERSION, checked_utc=datetime.now(timezone.utc).isoformat(),
      review_source_sha256=sha(__file__), status='independent_evidence_review_passed', train_sha=TRAIN_SHA,
      source_run=str(directory), sources_sha256=sources, config_differences=differences,
      counts={k: final[k] for k in ('train_action_steps', 'updates', 'evaluations', 'eval_action_steps')},
      summaries=summaries, comparisons=comparisons, gate=gate,
      snapshots=snapshots, final_checkpoint=final_agent, resource_samples=len(resource_rows),
      sampled_peak_memory_mib=peak_memory, whole_entry_seconds=complete['elapsed_seconds'],
      occupied_gpu_hour_upper_bound=complete['elapsed_seconds'] / 3600,
      limitations=['One training seed; historical B/D and different code SHA.',
          'Gate logs are batch aggregates of resampled replay, not state-level utility truth.',
          'Read/hash/parameter/ledger/arithmetic audit; no new model training or episode reenactment.',
          'Segmented windows and endpoint episode differences are descriptive post hoc diagnostics.'],
      new_training=False, gpu_computation=False, scientific_method_effectiveness_established=False)
  output.mkdir(parents=True)
  (output / 'review.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
  (output / 'points.json').write_text(json.dumps(points, indent=2, sort_keys=True, allow_nan=False) + '\n')
  print(json.dumps(dict(status=result['status'], summaries=summaries,
      gate_summary={key: gate[key] for key in ('mean_weight_reduction',
          'observed_excess_relative_reduction', 'relative_to_D_only_release_on_valid_N_positions')})), flush=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--plan', required=True)
  parser.add_argument('--output', required=True)
  args = parser.parse_args()
  inspect(args.plan, args.output)
