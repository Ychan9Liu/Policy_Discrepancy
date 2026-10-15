"""Plot the audited pilot and expose every point's fixed-AUC contribution."""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re


TRAIN_SHA = '13ec72a214b0c03b886b2c7fac52c8f089966dec'
TASK, BUDGET, INTERVAL, EPISODES, UPDATES = 'dmc_quadruped_walk', 1000000, 50000, 10, 249493
GROUPS = ('B', 'D', 'N')
GRID = list(range(0, BUDGET + 1, INTERVAL))


def require(condition, message):
  if not condition:
    raise ValueError(message)


def mapping(value, label):
  require(isinstance(value, dict), f'{label} must be an object')
  return value


def integer(value, label):
  require(type(value) is int, f'{label} must be an integer, not bool or float')
  return value


def number(value, label):
  require(type(value) in (int, float), f'{label} must be a finite number, not bool')
  try:
    finite = math.isfinite(value)
  except OverflowError:
    finite = False
  require(finite, f'{label} must be finite')
  return value


def close(actual, expected, label):
  require(math.isclose(number(actual, label), expected, abs_tol=1e-9, rel_tol=1e-12),
      f'{label} differs from raw episode arithmetic')


def unique_object(pairs):
  result = {}
  for key, value in pairs:
    require(key not in result, f'Duplicate JSON key: {key}')
    result[key] = value
  return result


def reject_constant(value):
  raise ValueError(f'Nonfinite JSON number: {value}')


def read_bound_json(path, expected):
  require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected),
      'Expected SHA256 must be 64 lowercase hexadecimal characters')
  payload = Path(path).read_bytes()
  actual = hashlib.sha256(payload).hexdigest()
  require(actual == expected, f'Input SHA256 differs: {path}')
  # Hash and decode the same bytes; never reread a possibly changed input later.
  value = json.loads(payload.decode('utf-8'), object_pairs_hook=unique_object,
      parse_constant=reject_constant, parse_float=lambda x: number(float(x), 'JSON number'))
  return value, actual


def validate_inputs(review_path, points_path, expected_review_sha, expected_points_sha):
  review, review_sha = read_bound_json(review_path, expected_review_sha)
  points, points_sha = read_bound_json(points_path, expected_points_sha)
  mapping(review, 'review')
  require(review.get('status') == 'independent_evidence_review_passed', 'Review did not pass')
  require(review.get('train_sha') == TRAIN_SHA, 'Training SHA differs from fixed pilot')
  source = review.get('source_run')
  require(isinstance(source, str) and PurePosixPath(source).is_absolute() and
      PurePosixPath(source).parts[-2:] == (TASK, 'N-seed0-attempt01'),
      'Review source_run does not identify the fixed task and arm')
  counts = mapping(review.get('counts'), 'review.counts')
  for key, expected in dict(train_action_steps=BUDGET, evaluations=len(GRID),
      eval_action_steps=len(GRID)*EPISODES*1000, updates=UPDATES).items():
    require(integer(counts.get(key), f'counts.{key}') == expected, f'counts.{key} differs')
  mapping(points, 'points')
  require(set(points) == set(GROUPS), 'Expected exactly B, D and N point groups')
  summaries = mapping(review.get('summaries'), 'review.summaries')
  require(set(summaries) == set(GROUPS), 'Expected exactly B, D and N summaries')
  for group in GROUPS:
    data = points[group]
    require(isinstance(data, list) and len(data) == len(GRID), f'{group}: incomplete fixed grid')
    means = []
    for point, expected_step in zip(data, GRID):
      mapping(point, f'{group} point')
      step = integer(point.get('action_step'), f'{group}.action_step')
      require(step == expected_step, f'{group}: fixed action grid differs')
      raw = point.get('returns_raw')
      require(isinstance(raw, list) and len(raw) == EPISODES,
          f'{group}/{step}: expected ten raw episode returns')
      mean = math.fsum(number(x, f'{group}/{step} return') for x in raw) / EPISODES
      close(point.get('mean'), mean, f'{group}/{step} mean')
      means.append(mean)
    summary = mapping(summaries[group], f'{group} summary')
    auc = math.fsum(INTERVAL*(a+b)/2 for a, b in zip(means, means[1:])) / BUDGET
    for key, value in dict(auc_per_1m_actions=auc,
        tail_mean_return=math.fsum(means[-5:])/5, endpoint_return=means[-1]).items():
      close(summary.get(key), value, f'{group} {key}')
  gate = mapping(review.get('gate'), 'review.gate')
  require(integer(gate.get('updates'), 'gate.updates') == UPDATES, 'Gate update count differs')
  bins = gate.get('bins')
  require(isinstance(bins, list) and len(bins) == len(GRID)-1, 'Incomplete gate bins')
  bin_updates = 0
  for i, entry in enumerate(bins):
    mapping(entry, 'gate bin')
    require(integer(entry.get('action_start'), 'bin.action_start') == GRID[i] and
        integer(entry.get('action_end'), 'bin.action_end') == GRID[i+1], 'Gate bin grid differs')
    values, sizes = {}, []
    for key in ('actual_v_mean', 'valid_active_frac', 'active_protected_frac'):
      stat = mapping(entry.get(key), f'bin.{key}')
      values[key] = number(stat.get('mean'), f'bin.{key}.mean')
      sizes.append(integer(stat.get('n'), f'bin.{key}.n'))
    require(sizes[0] > 0 and len(set(sizes)) == 1, 'Gate bin denominators differ')
    require(.5 <= values['actual_v_mean'] <= 1 and
        0 <= values['active_protected_frac'] <= values['valid_active_frac'] <= 1,
        'Gate bin mean or coverage outside fixed bounds')
    bin_updates += sizes[0]
  require(bin_updates == UPDATES, 'Gate bins do not cover all updates')
  comparisons = mapping(review.get('comparisons'), 'review.comparisons')
  contributions = {}
  for group in ('B', 'D'):
    table = []
    for i, (new, old) in enumerate(zip(points['N'], points[group])):
      weight = INTERVAL / BUDGET * (.5 if i in (0, len(GRID)-1) else 1)
      difference = new['mean'] - old['mean']
      table.append(dict(action_step=new['action_step'], mean_return_difference=difference,
          auc_weight=weight, auc_difference_contribution=weight*difference))
    total = math.fsum(p['auc_difference_contribution'] for p in table)
    comparison = mapping(comparisons.get(group), f'comparison {group}')
    for key in ('auc_per_1m_actions', 'tail_mean_return', 'endpoint_return'):
      metric = mapping(comparison.get(key), f'comparison {group}.{key}')
      close(metric.get('absolute'), summaries['N'][key]-summaries[group][key],
          f'comparison {group}.{key}')
    close(total, comparison['auc_per_1m_actions']['absolute'], f'{group} AUC decomposition')
    contributions[group] = dict(total=total, points=table,
        scope='Algebraic decomposition of the unchanged fixed 1M AUC; no points omitted or new metric substituted')
  return review, points, {'review.json': review_sha, 'points.json': points_sha}, contributions


def render(review_path, points_path, output, *, expected_review_sha, expected_points_sha):
  source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  review, points, input_sha, contributions = validate_inputs(
      review_path, points_path, expected_review_sha, expected_points_sha)
  output = Path(output)
  require(not output.exists(), 'Use a new plot directory')
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  output.mkdir(parents=True)
  plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
  fig = plt.figure(figsize=(12, 8), layout='constrained')
  grid = fig.add_gridspec(2, 2, height_ratios=[1.6, 1])
  curve, gate_ax, fraction_ax = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
  colors = {'B': '#6b7280', 'D': '#2563eb', 'N': '#d97706'}
  labels = {'B': 'Baseline (historical)', 'D': 'Dt (historical)', 'N': 'DtLatch (new)'}
  for group in ('B', 'D', 'N'):
    data = points[group]
    curve.plot([p['action_step']/100000 for p in data], [p['mean'] for p in data],
        marker='o', markersize=3.5, linewidth=1.8, color=colors[group], label=labels[group])
  curve.axvspan(8, 10, alpha=.05, color='black')
  curve.set(xlim=(0, 10), ylim=(0, 1050), xticks=range(11),
      xlabel='Training actions (100k)', ylabel='Mean raw episode return')
  curve.legend(loc='lower right', ncols=3, fontsize=9)
  curve.grid(alpha=.2)
  bins = review['gate']['bins']
  x = [(b['action_start'] + b['action_end']) / 200000 for b in bins]
  gate_ax.plot(x, [100*(1-b['actual_v_mean']['mean']) for b in bins], color=colors['N'], marker='o', markersize=3)
  gate_ax.set(title='Average protection strength', xlabel='Training actions (100k)', ylabel='Mean weight reduction, 1-v (%)')
  gate_ax.ticklabel_format(axis='y', style='plain', useOffset=False)
  fraction_ax.plot(x, [100*b['valid_active_frac']['mean'] for b in bins], color=colors['B'], label='Valid and rep-active')
  fraction_ax.plot(x, [100*b['active_protected_frac']['mean'] for b in bins], color=colors['N'], label='Active and protected')
  fraction_ax.set(title='Replay position coverage', xlabel='Training actions (100k)', ylabel='Fraction of batch positions (%)', ylim=(0, 103))
  fraction_ax.legend(fontsize=8, loc='lower right')
  for axis in (gate_ax, fraction_ax):
    axis.grid(alpha=.2)
    axis.set_xlim(0, 10)
  fig.suptitle('DtLatch pilot: quadruped_walk | one training seed | 1M actions\n'
      'Return curves: 10 evaluation episodes per point; gate panels: resampled replay batches', fontsize=13)
  fig.savefig(output/'learning_and_gate.png', dpi=180)
  fig.savefig(output/'learning_and_gate.pdf')
  plt.close(fig)
  result = dict(matplotlib_version=matplotlib.__version__,
      input_sha256=input_sha, plot_source_sha256=source_sha,
      files_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()},
      contributions=contributions)
  (output/'plot_and_contributions.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n', encoding='utf-8')
  print(json.dumps({'output': str(output), 'N_minus_B_at_350k': contributions['B']['points'][7]}, sort_keys=True))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--review', required=True)
  parser.add_argument('--points', required=True)
  parser.add_argument('--output', required=True)
  parser.add_argument('--expected-review-sha', required=True)
  parser.add_argument('--expected-points-sha', required=True)
  args = parser.parse_args()
  render(args.review, args.points, args.output,
      expected_review_sha=args.expected_review_sha, expected_points_sha=args.expected_points_sha)
