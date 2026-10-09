"""Generate reviewable, hand-calculated tables; no agent or environment runs."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

from analysis.return_metrics import TASKS, aggregate, write_tables
from tests.test_return_metrics import FIXTURE, calculate, metadata, point


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  tables = dict(episodes=[], evaluation_points=[], seed_metrics=[])
  for task, pair, delta in zip(TASKS, ((1,3),(2,6),(10,30),(20,40)),(2,-1,4,-2)):
    for group in ('baseline','dt','constant','shuffle'):
      for seed, value in enumerate(pair):
        value += delta if group == 'dt' else 0
        records = [point(step,(value,value)) for step in reversed(FIXTURE.grid)]
        episodes, points, metrics = calculate(records, metadata(task,group,seed))
        tables['episodes'].extend(episodes)
        tables['evaluation_points'].extend(points)
        tables['seed_metrics'].append(metrics)
  task_rows, overall, differences = aggregate(tables['seed_metrics'], FIXTURE)
  tables.update(task_metrics=task_rows,overall_metrics=overall,dt_differences=differences)
  assert next(row for row in overall if row['group']=='baseline')['auc_return_per_budget']==14
  assert next(row for row in differences if row['level']=='overall' and
      row['comparison']=='dt-baseline')['auc_return_per_budget']==.75
  commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
  audit = dict(tool_commit=commit,metric_spec=asdict(FIXTURE),fixture_only=True,
      hand_expected=dict(baseline_task_means=[2,4,20,30],baseline_overall=14,
          dt_task_deltas=[2,-1,4,-2],dt_overall_delta=.75),
      interpretation='synthetic statistics fixture; no additional training seeds authorized')
  write_tables(args.output,tables,audit)
  print(json.dumps(dict(output=str(args.output),fixture_only=True,
                        episodes=len(tables['episodes']),seed_rows=len(tables['seed_metrics']))))


if __name__ == '__main__':
  main()
