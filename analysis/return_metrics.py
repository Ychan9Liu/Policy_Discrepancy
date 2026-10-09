"""Auditable raw-return tables for EXPERIMENTS C/C.1 (stdlib only).

CLI: python -m analysis.return_metrics --index RUN_INDEX.json --output NEW_DIR
The production CLI always uses the frozen million-action, seed0 protocol.
Reduced-budget sources are retained as engineering rows with null formal metrics.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess

from dreamerv3.return_stats import REWARD_SOURCE, RETURN_DEFINITION


TASKS = ('dmc_hopper_hop', 'dmc_quadruped_run',
         'dmc_quadruped_walk', 'dmc_reacher_hard')
GROUPS = ('baseline', 'dt', 'constant', 'shuffle')
# These source versions were inspected for raw reward and episodes=1 Driver IO.
# An arbitrary field called "score" is never enough to grant this mapping.
LEGACY_COMMITS = {
    '1393548fb6e46d28f92a834204f7b09495a3eab1',
    '63d36a111d247aa8cd05af173f74e43e4e35d2e1',
    '20ab3486bfaa067cc7950d3f5c8408fe2ed92ec3'}
MEAN_ATOL, MEAN_RTOL = 1e-9, 1e-12


@dataclass(frozen=True)
class MetricSpec:
  budget: int = 1000000
  grid: tuple = tuple(range(0, 1000001, 50000))
  tail: tuple = (800000, 850000, 900000, 950000, 1000000)
  episodes: int = 10
  seeds: tuple = (0,)
  tasks: tuple = TASKS
  scope: str = 'M2-v1-formal'

  def __post_init__(self):
    if (self.budget <= 0 or self.episodes <= 0 or not self.seeds or
        not self.tasks or not self.grid or self.grid[0] != 0 or
        self.grid[-1] != self.budget or tuple(sorted(set(self.grid))) != self.grid or
        not self.tail or not set(self.tail) <= set(self.grid) or
        len(set(self.seeds)) != len(self.seeds) or
        len(set(self.tasks)) != len(self.tasks)):
      raise ValueError('Invalid metric specification')
    if self.scope == 'M2-v1-formal' and (
        self.budget != 1000000 or self.grid != tuple(range(0, 1000001, 50000)) or
        self.tail != (800000, 850000, 900000, 950000, 1000000) or
        self.episodes != 10 or self.seeds != (0,) or self.tasks != TASKS):
      raise ValueError('Fixture variants require an explicit fixture scope')


def mean(values):
  return math.fsum(values) / len(values)


def trapezoid_auc(points, budget):
  """Integrate actual x; this primitive does not assert protocol completeness."""
  ordered = sorted(points)
  if budget <= 0 or len(ordered) < 2 or len({x for x, _ in ordered}) != len(ordered):
    raise ValueError('AUC needs distinct actual action steps and a fixed budget')
  if not all(math.isfinite(float(v)) for point in ordered for v in point):
    raise ValueError('Non-finite AUC input')
  return math.fsum((b[0] - a[0]) * (a[1] + b[1]) / 2
                   for a, b in zip(ordered, ordered[1:])) / budget


def integer(value):
  if isinstance(value, bool) or int(value) != value:
    raise ValueError(f'Not an integer: {value}')
  return int(value)


def sha(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_evidence(directory, targets, entry, manifest):
  """Prefer actual snapshot counters; support the preserved 04 read-only audit."""
  evidence, sources = {}, {}
  if entry.get('snapshot_audit_file'):
    path = Path(entry['snapshot_audit_file'])
    payload = json.loads(path.read_text(encoding='utf-8'))
    sources[str(path)] = sha(path)
    runs = [run for run in payload['runs']
            if run['directory'] == entry['source_directory']]
    if len(runs) != 1:
      raise ValueError('Snapshot audit does not identify exactly one source run')
    run = runs[0]
    if run['git_commit'] != manifest['git_commit']:
      raise ValueError('Snapshot audit/source commit disagreement')
    if run['artifact_sha256']['evaluations.jsonl'] != sha(directory / 'evaluations.jsonl'):
      raise ValueError('Snapshot audit/evaluation file hash disagreement')
    for item in run['snapshots']:
      candidate = dict(
          actual_step=item.get('actual_action_step', item['action_step']),
          update_id=item['counters']['updates'],
          snapshot_id=item.get('snapshot_id', f"eval_snapshots/{item['action_step']:07d}"),
          params_sha256=item['params_sha256'], basis='04-checkpoint-read-audit')
      if item['action_step'] in evidence and evidence[item['action_step']] != candidate:
        raise ValueError('Conflicting snapshot audit records')
      evidence[item['action_step']] = candidate
  else:
    # Reading trusted project checkpoints only, never arbitrary downloaded pickle.
    import pickle
    for point in targets:
      folder = directory / 'eval_snapshots' / f'{point:07d}'
      if not (folder / 'latest').exists():
        continue
      tag = (folder / 'latest').read_text().strip()
      if Path(tag).name != tag:
        raise ValueError('Invalid checkpoint tag')
      metadata_path = folder / 'evaluation_snapshot.json'
      if metadata_path.exists():
        item = json.loads(metadata_path.read_text(encoding='utf-8'))
        if (item['git_commit'] != manifest['git_commit'] or
            item['snapshot_id'] != f'eval_snapshots/{point:07d}/{tag}' or
            not (folder / tag / 'done').exists() or
            not (folder / tag / 'agent.pkl').exists()):
          raise ValueError('Snapshot metadata/checkpoint disagreement')
        sources[str(metadata_path)] = sha(metadata_path)
        sources[str(folder / 'latest')] = sha(folder / 'latest')
        sources[str(folder / tag / 'done')] = sha(folder / tag / 'done')
        evidence[point] = dict(actual_step=item['actual_action_step'],
            update_id=item['update_id'], snapshot_id=item['snapshot_id'],
            basis='runner-snapshot-metadata-and-completion-marker')
        continue
      path = folder / tag / 'agent.pkl'
      with path.open('rb') as file:
        saved = pickle.load(file)
      sources[str(folder / 'latest')] = sha(folder / 'latest')
      sources[str(path)] = sha(path)
      evidence[point] = dict(actual_step=point,
          update_id=saved['counters']['updates'],
          snapshot_id=f'eval_snapshots/{point:07d}/{tag}',
          basis='checkpoint-counter-read')
  return evidence, sources


def normalize_points(metadata, records, snapshots, spec=MetricSpec(), source_lines=None):
  """Preserve every source row, but never choose among conflicting snapshots."""
  episodes, points, issues = [], [], []
  grouped = {}
  source_lines = source_lines if source_lines is not None else range(1, len(records)+1)
  if len(source_lines) != len(records):
    raise ValueError('Source line mapping does not cover every record')
  for line, record in zip(source_lines, records):
    try:
      target = integer(record.get('target_action_step', record.get('action_step')))
      grouped.setdefault(target, []).append((line, record))
    except (TypeError, ValueError, OverflowError):
      issues.append(f'invalid_target_at_line:{line}')
  identity = {key: metadata[key] for key in ('task', 'group', 'seed', 'run_id')}
  identity.update({key: metadata[key] for key in (
      'source_commit', 'source_directory', 'engineering_fixture') if key in metadata})
  identity['metric_scope'] = spec.scope
  for target, entries in sorted(grouped.items()):
    first = entries[0][1]
    conflict = any(row != first for _, row in entries[1:])
    errors = ['conflicting_evaluation_records'] if conflict else []
    snapshot = snapshots.get(target)
    if snapshot is None:
      errors.append('missing_snapshot_evidence')
    # Conflicting rows remain in episode audit with separate source line IDs.
    unique = entries if conflict else entries[:1]
    point_episode_rows = []
    actual = None
    point_update = None
    for line, row in unique:
      row_errors = []
      legacy = row.get('schema_version') != 2
      try:
        values = row.get('returns_raw', row.get('scores'))
        values = [float(x) for x in values]
        lengths = [integer(x) for x in row['lengths']]
        seeds = [integer(x) for x in row['seeds']]
        point_update = integer(row['update_id'])
        actual = integer(row.get('snapshot_action_step', row.get('action_step')))
        if not values or len(values) != len(lengths) or len(values) != len(seeds):
          raise ValueError('Episode arrays have different lengths or are empty')
        if not all(math.isfinite(x) for x in values) or any(x <= 0 for x in lengths):
          raise ValueError('Non-finite return or invalid episode length')
        if 'returns_raw' in row and 'scores' in row and row['scores'] != values:
          row_errors.append('scores_returns_alias_conflict')
        for name in ('mean', 'mean_return_raw'):
          if name in row and not math.isclose(float(row[name]), mean(values),
                                              abs_tol=MEAN_ATOL, rel_tol=MEAN_RTOL):
            row_errors.append('declared_mean_conflict')
        if actual != target:
          row_errors.append('snapshot_action_step_mismatch')
        if 'action_step' in row and integer(row['action_step']) != actual:
          row_errors.append('action_step_snapshot_conflict')
        if snapshot and (snapshot['actual_step'] != actual or
                         snapshot['update_id'] != point_update):
          row_errors.append('snapshot_update_or_step_conflict')
        if legacy:
          if not metadata.get('legacy_raw_return_verified'):
            row_errors.append('unverified_legacy_score_semantics')
          details = [dict(episode_id=i, return_raw=value, length=length,
              environment_seed=seed, complete=bool(
                  metadata.get('legacy_raw_return_verified')),
              reward_source=REWARD_SOURCE, return_definition=RETURN_DEFINITION,
              action_rng=metadata.get('eval_policy', 'unrecorded'))
              for i, (value, length, seed) in enumerate(zip(values, lengths, seeds))]
        else:
          details = row['episodes']
          if (row.get('reward_source') != REWARD_SOURCE or
              row.get('return_definition') != RETURN_DEFINITION):
            row_errors.append('reward_semantics_conflict')
          if len(details) != len(values):
            raise ValueError('Episode details count mismatch')
          if snapshot and row.get('snapshot_id') != snapshot['snapshot_id']:
            row_errors.append('snapshot_id_conflict')
          if row.get('snapshot_update_id') != point_update:
            row_errors.append('snapshot_update_id_conflict')
        ids = [integer(item['episode_id']) for item in details]
        if sorted(ids) != list(range(len(values))):
          row_errors.append('episode_id_set_conflict')
        for i, (item, value, length, seed) in enumerate(zip(details, values, lengths, seeds)):
          item_errors = list(row_errors)
          if (item.get('return_raw') != value or item.get('length') != length or
              item.get('environment_seed') != seed):
            item_errors.append('episode_arrays_details_conflict')
          if item.get('complete') is not True:
            item_errors.append('incomplete_episode')
          if not item.get('action_rng'):
            item_errors.append('missing_action_rng')
          if (item.get('reward_source') != REWARD_SOURCE or
              item.get('return_definition') != RETURN_DEFINITION):
            item_errors.append('episode_reward_semantics_conflict')
          output = dict(identity, target_action_step=target,
              actual_snapshot_action_step=actual, snapshot_id=(snapshot or {}).get('snapshot_id'),
              snapshot_params_sha256=(snapshot or {}).get('params_sha256'),
              update_id=point_update, episode_id=item['episode_id'],
              environment_seed=seed, action_rng=item.get('action_rng'),
              return_raw=value, length=length, episode_complete=item.get('complete'),
              reward_source=REWARD_SOURCE, return_definition=RETURN_DEFINITION,
              evidence_basis='reviewed_legacy_source' if legacy else 'explicit_schema_v2',
              source_line=line, valid=not item_errors and not conflict,
              issues=item_errors + (['conflicting_evaluation_records'] if conflict else []))
          episodes.append(output)
          point_episode_rows.append(output)
          row_errors.extend(item_errors)
      except (KeyError, TypeError, ValueError, OverflowError) as error:
        row_errors.append(f'invalid_episode_record:{error}')
      errors.extend(row_errors)
    errors = sorted(set(errors))
    expected = metadata['source_eval_episodes']
    if len(point_episode_rows) != expected:
      errors.append(f'source_episode_count:{len(point_episode_rows)}_expected:{expected}')
    source_complete = not errors
    formal_errors = list(errors)
    if len(point_episode_rows) != spec.episodes:
      formal_errors.append(f'protocol_episode_count:{len(point_episode_rows)}_expected:{spec.episodes}')
    points.append(dict(identity, evaluation_point_id=target,
        target_action_step=target, actual_snapshot_action_step=actual,
        snapshot_id=(snapshot or {}).get('snapshot_id'), update_id=point_update,
        returns_raw=[item['return_raw'] for item in point_episode_rows],
        episode_count=len(point_episode_rows), expected_episode_count=spec.episodes,
        mean_return_raw=mean([item['return_raw'] for item in point_episode_rows])
            if source_complete else None,
        source_complete=source_complete, protocol_complete=not formal_errors,
        source_lines=[line for line, _ in entries],
        identical_rewrites_ignored=0 if conflict else len(entries)-1,
        issues=sorted(set(formal_errors))))
  return episodes, points, issues


def seed_metrics(metadata, points, issues, spec=MetricSpec()):
  point_map = {point['target_action_step']: point for point in points}
  def metric(required, auc):
    problems = list(issues)
    missing = [step for step in required if step not in point_map]
    problems += [f'missing_point:{step}' for step in missing]
    for step in required:
      if step in point_map and not point_map[step]['protocol_complete']:
        problems.append(f'invalid_point:{step}')
    if metadata['engineering_fixture']:
      problems.append('engineering_fixture_is_not_formal_data')
    if metadata.get('source_budget', spec.budget) != spec.budget:
      problems.append('source_budget_differs_from_fixed_budget')
    if problems:
      return None, False, sorted(set(problems))
    values = [(point_map[step]['actual_snapshot_action_step'],
               point_map[step]['mean_return_raw']) for step in required]
    return (trapezoid_auc(values, spec.budget) if auc else mean([v for _, v in values])), True, []
  auc, auc_complete, auc_issues = metric(spec.grid, True)
  tail, tail_complete, tail_issues = metric(spec.tail, False)
  return dict(metadata, fixed_budget=spec.budget,
      observed_action_steps=sorted(point_map), required_auc_steps=list(spec.grid),
      required_tail_steps=list(spec.tail), auc_return_per_budget=auc,
      auc_complete=auc_complete, auc_issues=auc_issues,
      tail_mean_return=tail, tail_complete=tail_complete, tail_issues=tail_issues)


def aggregate(seed_rows, spec=MetricSpec()):
  tasks, overall, differences = [], [], []
  for task in spec.tasks:
    for group in GROUPS:
      members = [row for row in seed_rows if row['task'] == task and row['group'] == group]
      output = dict(task=task, group=group, expected_seeds=list(spec.seeds),
                    source_runs=[row['run_id'] for row in members])
      for key, prefix in (('auc_return_per_budget', 'auc'), ('tail_mean_return', 'tail')):
        errors, values = [], []
        for seed in spec.seeds:
          found = [row for row in members if row['seed'] == seed]
          if len(found) != 1:
            errors.append(f'seed:{seed}_run_count:{len(found)}')
          elif not found[0][prefix + '_complete']:
            errors.append(f'incomplete_seed:{seed}')
          else:
            values.append(found[0][key])
        unexpected = sorted({row['seed'] for row in members} - set(spec.seeds))
        if unexpected:
          errors.append(f'unexpected_seeds:{unexpected}')
        output.update({key: mean(values) if not errors else None,
                       prefix + '_complete': not errors, prefix + '_issues': errors})
      tasks.append(output)
  for group in GROUPS:
    members = [row for row in tasks if row['group'] == group]
    output = dict(group=group, expected_tasks=list(spec.tasks),
                  task_weights={task: 1 / len(spec.tasks) for task in spec.tasks})
    for key, prefix in (('auc_return_per_budget', 'auc'), ('tail_mean_return', 'tail')):
      missing = [row['task'] for row in members if not row[prefix + '_complete']]
      output.update({key: mean([row[key] for row in members]) if not missing else None,
                     prefix + '_complete': not missing, prefix + '_issues': missing})
    overall.append(output)
  for scope, task, members in [('task', task, [row for row in tasks if row['task'] == task])
                               for task in spec.tasks] + [('overall', None, overall)]:
    by_group = {row['group']: row for row in members}
    for ref in ('baseline', 'constant', 'shuffle'):
      output = dict(level=scope, task=task, comparison='dt-' + ref)
      for key, prefix in (('auc_return_per_budget', 'auc'), ('tail_mean_return', 'tail')):
        complete = all(by_group[group][prefix + '_complete'] for group in ('dt', ref))
        output.update({key: by_group['dt'][key] - by_group[ref][key] if complete else None,
                       prefix + '_complete': complete,
                       prefix + '_issues': [] if complete else ['incomplete_comparison_members']})
      differences.append(output)
  return tasks, overall, differences


def analyze_run(entry, spec=MetricSpec()):
  directory = Path(entry['directory'])
  manifest_path = directory / 'protocol_manifest.json'
  evaluation_path = directory / 'evaluations.jsonl'
  manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
  config = manifest['config']
  source = entry.get('source_directory', str(directory.resolve()))
  metadata = dict(task=manifest['task'],
      group='baseline' if config['agent.rep_probe.mode'] == 'logging' else config['agent.rep_probe.mode'],
      seed=manifest['root_seed'], run_id=entry.get('run_id', source),
      source_directory=source, source_commit=manifest['git_commit'],
      source_host=manifest.get('host'),
      config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
      engineering_fixture=bool(config['run.engineering_fixture']),
      source_budget=config['run.action_budget'],
      source_eval_episodes=config['run.eval_eps'],
      eval_policy=manifest['eval_policy'],
      legacy_raw_return_verified=manifest['git_commit'] in LEGACY_COMMITS)
  final_path = directory / 'final_state.json'
  metadata['source_final_state_present'] = final_path.exists()
  records, source_lines = [], []
  parse_issues = [] if evaluation_path.exists() else ['missing_evaluations_file']
  if metadata['task'] not in spec.tasks or metadata['group'] not in GROUPS:
    parse_issues.append('outside_declared_task_or_group')
  texts = evaluation_path.read_text(encoding='utf-8').splitlines() if evaluation_path.exists() else []
  for line, text in enumerate(texts, 1):
    if not text.strip():
      continue
    try:
      row = json.loads(text)
      if not isinstance(row, dict):
        raise ValueError('Not an evaluation object')
      records.append(row)
      source_lines.append(line)
    except ValueError as error:
      parse_issues.append(f'malformed_json_at_line:{line}:{error}')
  targets = {row.get('target_action_step', row.get('action_step')) for row in records}
  targets = {step for step in targets if isinstance(step, int) and step >= 0}
  sources = {str(manifest_path): sha(manifest_path)}
  for name in ('evaluations.jsonl', 'config.yaml', 'execution-initial.json',
               'audit.json', 'final_state.json'):
    if (directory / name).exists():
      sources[str(directory / name)] = sha(directory / name)
  if entry.get('source_archive'):
    sources[entry['source_archive']] = sha(entry['source_archive'])
  try:
    snapshots, snapshot_sources = snapshot_evidence(directory, targets, entry, manifest)
    sources.update(snapshot_sources)
  except (KeyError, ValueError, OSError) as error:
    snapshots = {}
    parse_issues.append(f'snapshot_evidence_error:{error}')
  episodes, points, issues = normalize_points(
      metadata, records, snapshots, spec, source_lines=source_lines)
  summary = seed_metrics(metadata, points, parse_issues + issues, spec)
  audit = dict(metadata, input_sha256=sources,
      field_mapping={'scores': 'episode_return_raw', 'mean': 'mean_episode_return_raw',
                     'action_step': 'actual_training_action_step_at_snapshot'},
      completion_basis='explicit complete flag or reviewed per-episode Driver source plus snapshot audit',
      parse_issues=parse_issues + issues,
      identical_rewrites_ignored=sum(point['identical_rewrites_ignored'] for point in points))
  return episodes, points, summary, audit


def write_tables(output, tables, audit):
  output = Path(output)
  if output.exists():
    raise FileExistsError('Use a new analysis output directory')
  output.mkdir(parents=True)
  for name, rows in tables.items():
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (output / (name + '.csv')).open('w', newline='', encoding='utf-8-sig') as file:
      writer = csv.DictWriter(file, keys)
      writer.writeheader()
      for row in rows:
        writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                         if isinstance(value, (list, dict)) else value
                         for key, value in row.items()})
  (output / 'tables.json').write_text(json.dumps(tables, indent=2, ensure_ascii=False,
      allow_nan=False) + '\n', encoding='utf-8')
  (output / 'audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False,
      allow_nan=False) + '\n', encoding='utf-8')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--index', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--require-complete', action='store_true')
  args = parser.parse_args()
  entries = json.loads(args.index.read_text(encoding='utf-8'))['runs']
  tables = dict(episodes=[], evaluation_points=[], seed_metrics=[])
  audits = []
  for entry in entries:
    episodes, points, summary, audit = analyze_run(entry)
    tables['episodes'].extend(episodes)
    tables['evaluation_points'].extend(points)
    tables['seed_metrics'].append(summary)
    audits.append(audit)
  tasks, overall, differences = aggregate(tables['seed_metrics'])
  tables.update(task_metrics=tasks, overall_metrics=overall, dt_differences=differences)
  repo = Path(__file__).resolve().parents[1]
  commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
  audit = dict(schema_version=1, metric_spec=asdict(MetricSpec()),
      tool_commit=commit, tool_worktree_dirty=bool(subprocess.check_output(
          ['git', 'status', '--porcelain'], cwd=repo, text=True).strip()),
      index_path=str(args.index), index_sha256=sha(args.index), runs=audits,
      mean_tolerance=dict(atol=MEAN_ATOL, rtol=MEAN_RTOL),
      formal_metrics_complete=all(row['auc_complete'] and row['tail_complete'] for row in overall),
      interpretation='single-training-seed descriptive only; engineering rows are not formal M2 results')
  write_tables(args.output, tables, audit)
  print(json.dumps(dict(output=str(args.output), runs=len(audits),
      episodes=len(tables['episodes']), formal_metrics_complete=audit['formal_metrics_complete'])))
  if args.require_complete and not audit['formal_metrics_complete']:
    raise SystemExit(2)


if __name__ == '__main__':
  main()
