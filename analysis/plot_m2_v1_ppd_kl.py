"""Redraw the requested three-task presentation subset of the frozen M2 v1 data.

PPD-KL is a display alias for the existing dt group. No training, smoothing,
normalization, input replacement, or new across-task aggregate is performed.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess


TASKS = ('dmc_quadruped_run', 'dmc_quadruped_walk', 'dmc_reacher_hard')
GROUPS = ('baseline', 'dt', 'constant', 'shuffle')
LABELS = dict(baseline='baseline', dt='PPD-KL', constant='constant',
              shuffle='global shuffle')
COLORS = dict(baseline='#4b5563', dt='#c2410c', constant='#047857',
              shuffle='#2563eb')
GRID = list(range(0, 1000001, 50000))
ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n', encoding='utf-8')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def load_inputs(source):
    hashes = json.loads((source / 'output_hashes.json').read_text(encoding='utf-8'))
    inputs = {}
    for name in ('return-audit/tables.json', 'gate_intervals.csv'):
        expected = next((v for k, v in hashes.items()
                         if k.replace('\\', '/') == name), None)
        require(expected is not None and sha(source / name) == expected,
                'Input differs from the original analysis hash: ' + name)
        inputs[name] = dict(path=str(source / name), sha256=expected)
    tables = json.loads((source / 'return-audit/tables.json').read_text(encoding='utf-8'))
    with (source / 'gate_intervals.csv').open(encoding='utf-8-sig', newline='') as file:
        intervals = list(csv.DictReader(file))

    curves, differences, gates = [], [], []
    for task in TASKS:
        metrics = {}
        for group in GROUPS:
            points = sorted((r for r in tables['evaluation_points']
                             if r['task'] == task and r['group'] == group),
                            key=lambda r: r['actual_snapshot_action_step'])
            require(len(points) == 21 and
                    [r['actual_snapshot_action_step'] for r in points] == GRID and
                    [r['target_action_step'] for r in points] == GRID,
                    f'Missing or duplicate formal evaluation point: {task}/{group}')
            for point in points:
                require(point['seed'] == 0 and point['protocol_complete'] and
                        point['source_complete'] and not point['engineering_fixture'] and
                        point['episode_count'] == len(point['returns_raw']) == 10 and
                        len(set(r['run_id'] for r in points)) == 1,
                        f'Incomplete or conflicting seed0 evaluation: {task}/{group}')
                require(math.isclose(math.fsum(point['returns_raw']) / 10,
                                     point['mean_return_raw'], rel_tol=1e-12, abs_tol=1e-9),
                        'Episode mean differs from the audited point mean')
                if group in ('baseline', 'dt'):
                    curves.append(dict(task=task, group=group, label=LABELS[group],
                                       seed=0, action_step=point['actual_snapshot_action_step'],
                                       mean_return_raw=point['mean_return_raw'],
                                       run_id=point['run_id'], snapshot_id=point['snapshot_id']))
            means = [p['mean_return_raw'] for p in points]
            metrics[group] = dict(
                auc_return_per_budget=math.fsum((a+b)*25000 for a, b in zip(means, means[1:])) / 1e6,
                tail_mean_return=math.fsum(means[-5:]) / 5)

            bins = sorted((r for r in intervals if r['task'] == task and r['group'] == group),
                          key=lambda r: int(r['end_action']))
            require(len(bins) == 20 and [int(r['end_action']) for r in bins] == GRID[1:],
                    f'Missing or duplicate diagnostic interval: {task}/{group}')
            for row in bins:
                out = dict(task=task, group=group, label=LABELS[group], seed=int(row['seed']),
                           run_id=row['run_id'], start_action=int(row['start_action']),
                           end_action=int(row['end_action']), updates=int(row['updates']),
                           D_mean=float(row['D_mean']),
                           active_weight_mean=float(row['active_weight_mean']))
                require(out['seed'] == 0 and out['updates'] > 0 and
                        out['start_action'] == out['end_action'] - 50000 and
                        math.isfinite(out['D_mean']) and out['D_mean'] > 0 and
                        math.isfinite(out['active_weight_mean']) and
                        0 <= out['active_weight_mean'] <= 1.00000001,
                        'Invalid diagnostic bin')
                gates.append(out)

        for control in ('baseline', 'constant', 'shuffle'):
            rows = [r for r in tables['dt_differences'] if r['level'] == 'task' and
                    r['task'] == task and r['comparison'] == 'dt-' + control]
            require(len(rows) == 1 and rows[0]['auc_complete'] and rows[0]['tail_complete'],
                    'Missing or conflicting descriptive difference')
            row = rows[0]
            for metric in ('auc_return_per_budget', 'tail_mean_return'):
                require(math.isclose(metrics['dt'][metric] - metrics[control][metric],
                                     row[metric], rel_tol=1e-12, abs_tol=1e-9),
                        'Difference does not agree with unsmoothed episode means')
            differences.append(dict(task=task, comparison=row['comparison'],
                                    label='PPD-KL − ' + LABELS[control],
                                    auc_return_per_budget=row['auc_return_per_budget'],
                                    tail_mean_return=row['tail_mean_return']))
    return inputs, curves, differences, gates


def plot(output, curves, differences, gates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titlesize': 12, 'figure.titlesize': 17,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'm2-v1-ppd-kl'})

    def save(fig, name):
        fig.savefig(output / (name + '.png'), dpi=220, facecolor='white')
        fig.savefig(output / (name + '.svg'), metadata={'Date': None})
        plt.close(fig)

    def axis_style(ax):
        ax.grid(alpha=.18, linewidth=.7)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_locator(MultipleLocator(.2))
        ax.set_axisbelow(True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), sharex=True, sharey=True)
    for task, ax in zip(TASKS, axes):
        for group in ('baseline', 'dt'):
            points = [r for r in curves if r['task'] == task and r['group'] == group]
            ax.plot([r['action_step'] / 1e6 for r in points],
                    [r['mean_return_raw'] for r in points], color=COLORS[group],
                    label=LABELS[group], lw=2, marker='o', ms=3.6)
        ax.set_title(task.removeprefix('dmc_'), pad=10)
        ax.set_xlabel('Training actions (millions)')
        axis_style(ax)
    axes[0].set_ylabel('Raw episode return')
    axes[0].set_ylim(0, 1050)
    fig.suptitle('learning_curves', y=.98)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.5, .91), ncols=2, frameon=False)
    fig.text(.5, .025, 'seed0 | 10 episodes per evaluation | unsmoothed evaluation means',
             ha='center', fontsize=10, color='#555555')
    fig.subplots_adjust(left=.065, right=.985, bottom=.18, top=.75, wspace=.13)
    save(fig, 'learning_curves')

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.1))
    for ax, metric, title in zip(axes, ('auc_return_per_budget', 'tail_mean_return'),
                                ('Primary: raw return AUC / 1M',
                                 'Secondary: mean return, 800k–1M')):
        for j, control in enumerate(('baseline', 'constant', 'shuffle')):
            values = [next(r[metric] for r in differences if r['task'] == task and
                           r['comparison'] == 'dt-' + control) for task in TASKS]
            bars = ax.bar([i + (j-1)*.24 for i in range(3)], values, width=.23,
                          color=COLORS[control], label='PPD-KL − ' + LABELS[control])
            ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=9)
        ax.axhline(0, color='#333333', lw=.8)
        ax.set_xticks(range(3), [task.removeprefix('dmc_') for task in TASKS])
        ax.set_title(title, pad=12)
        ax.set_ylabel('Raw return difference')
        ax.grid(axis='y', alpha=.18, linewidth=.7)
        ax.set_axisbelow(True)
        ax.margins(y=.19)
    fig.suptitle('descriptive differences', y=.98)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.5, .91), ncols=3, frameon=False)
    fig.text(.5, .025, 'seed0 | descriptive single-training-seed comparisons',
             ha='center', fontsize=10, color='#555555')
    fig.subplots_adjust(left=.07, right=.985, bottom=.15, top=.74, wspace=.23)
    save(fig, 'descriptive_differences')

    fig, axes = plt.subplots(3, 2, figsize=(11.8, 9.6), sharex=True)
    for i, task in enumerate(TASKS):
        for j, field in enumerate(('D_mean', 'active_weight_mean')):
            ax = axes[i, j]
            for group in GROUPS:
                bins = [r for r in gates if r['task'] == task and r['group'] == group]
                ax.plot([r['end_action'] / 1e6 for r in bins], [r[field] for r in bins],
                        color=COLORS[group], label=LABELS[group], lw=1.8)
            if j == 0:
                ax.set_yscale('log')
                ax.set_ylabel(task.removeprefix('dmc_'), labelpad=10)
            else:
                ax.set_ylim(-.03, 1.03)
            if i == 0:
                ax.set_title(('Mean Dt (log scale)', 'Actual active weight')[j], pad=11)
            if i == 2:
                ax.set_xlabel('Training actions (millions)')
            axis_style(ax)
    fig.suptitle('gate diagnostics', y=.985)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.5, .952), ncols=4, frameon=False)
    fig.text(.5, .018, 'seed0 | update means within each 50k-action interval',
             ha='center', fontsize=10, color='#555555')
    fig.subplots_adjust(left=.115, right=.98, bottom=.09, top=.87, hspace=.22, wspace=.21)
    save(fig, 'gate_diagnostics')
    return dict(matplotlib=matplotlib.__version__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Existing audited analysis-final directory')
    parser.add_argument('--output', type=Path, required=True,
                        help='New output directory; existing paths are refused')
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    inputs, curves, differences, gates = load_inputs(source)
    output.mkdir(parents=True, exist_ok=False)
    versions = plot(output, curves, differences, gates)
    dump(output / 'plot_data.json', dict(learning_curves=curves,
                                        descriptive_differences=differences,
                                        gate_diagnostics=gates))
    manifest = dict(
        source_directory=str(source), inputs=inputs,
        script=dict(path=str(Path(__file__).resolve()), sha256=sha(Path(__file__)),
                    git_commit=git('rev-parse', 'HEAD'),
                    tracked_script_dirty=bool(git('status', '--porcelain', '--', __file__))),
        runtime=dict(python=platform.python_version(), **versions),
        tasks=TASKS, display_labels=LABELS, seed=0,
        learning_curve_groups=['baseline', 'dt'],
        diagnostic_groups=GROUPS,
        diagnostic_fields=['D_mean', 'active_weight_mean'],
        learning_point_count=len(curves), difference_count=len(differences),
        diagnostic_interval_count=len(gates), smoothing=False,
        scope='Presentation subset only; original four-task formal analysis unchanged.',
        checks='Original input hashes, 21x10 evaluations, point means, metric differences, 20 diagnostic bins verified.',
        output_sha256={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    dump(output / 'manifest.json', manifest)
    print(json.dumps(dict(output=str(output), figures=3, formats=['png', 'svg'],
                          evaluation_points=len(curves), diagnostic_intervals=len(gates)),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
