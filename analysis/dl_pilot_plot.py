"""Plot the audited pilot and expose every point's fixed-AUC contribution."""
import argparse
import hashlib
import json
from pathlib import Path


def render(review_path, points_path, output):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  review_path, points_path, output = map(Path, (review_path, points_path, output))
  review, points = [json.loads(p.read_text(encoding='utf-8')) for p in (review_path, points_path)]
  assert review['status'] == 'independent_evidence_review_passed'
  assert not output.exists(), 'Use a new plot directory'
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
  contributions = {}
  for group in ('B', 'D'):
    table = []
    for i, (new, old) in enumerate(zip(points['N'], points[group])):
      assert new['action_step'] == old['action_step']
      weight = .025 if i in (0, 20) else .05
      table.append(dict(action_step=new['action_step'], mean_return_difference=new['mean']-old['mean'],
          auc_weight=weight, auc_difference_contribution=weight*(new['mean']-old['mean'])))
    total = sum(p['auc_difference_contribution'] for p in table)
    assert abs(total-review['comparisons'][group]['auc_per_1m_actions']['absolute']) < 1e-9
    contributions[group] = dict(total=total, points=table,
        scope='Algebraic decomposition of the unchanged fixed 1M AUC; no points omitted or new metric substituted')
  result = dict(matplotlib_version=matplotlib.__version__,
      input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (review_path, points_path)},
      plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      files_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()},
      contributions=contributions)
  (output/'plot_and_contributions.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n', encoding='utf-8')
  print(json.dumps({'output': str(output), 'N_minus_B_at_350k': contributions['B']['points'][7]}, sort_keys=True))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--review', required=True)
  parser.add_argument('--points', required=True)
  parser.add_argument('--output', required=True)
  args = parser.parse_args()
  render(args.review, args.points, args.output)
