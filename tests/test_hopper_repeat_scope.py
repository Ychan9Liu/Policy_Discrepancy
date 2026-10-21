"""Approved seed-extension boundaries and complete-cohort arithmetic (no agent)."""

import unittest
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from dreamerv3.protocol_scope import V1, HOPPER_123, HOPPER_SCRIPT, validate_scope
from analysis.return_metrics import MetricSpec, aggregate
from tests.test_return_metrics import point


class HopperScopeTest(unittest.TestCase):
  def test_original_seed0_scope_is_not_relaxed(self):
    self.assertEqual(validate_scope('protocol_v1', 'dmc_hopper_hop', 0, 'logging'), V1)
    for seed in (1, 2, 3):
      with self.assertRaisesRegex(ValueError, 'seed 0'):
        validate_scope('protocol_v1', 'dmc_hopper_hop', seed, 'logging')
    self.assertEqual(MetricSpec().seeds, (0,))
    self.assertEqual(len(MetricSpec().tasks), 4)

  def test_extension_is_only_approved_task_modes_seeds(self):
    for seed in (1, 2, 3):
      for mode in ('logging', 'dt'):
        self.assertEqual(validate_scope(HOPPER_SCRIPT, 'dmc_hopper_hop', seed, mode), HOPPER_123)
    for task, seed, mode in (('dmc_reacher_hard', 1, 'dt'), ('dmc_hopper_hop', 0, 'dt'),
                            ('dmc_hopper_hop', 4, 'dt'), ('dmc_hopper_hop', True, 'dt'),
                            ('dmc_hopper_hop', 1, 'constant'), ('dmc_hopper_hop', 1, 'shuffle')):
      with self.subTest(task=task, seed=seed, mode=mode), self.assertRaises(ValueError):
        validate_scope(HOPPER_SCRIPT, task, seed, mode)

  def test_complete_seed_cohort_mean_and_missing_seed_no_reweight(self):
    spec = MetricSpec(seeds=(1, 2, 3), tasks=('dmc_hopper_hop',),
                      groups=('baseline', 'dt'), scope=HOPPER_123)
    rows = []
    for group, values in (('baseline', (1, 4, 9)), ('dt', (2, 6, 6))):
      for seed, value in enumerate(values, 1):
        rows.append(dict(task='dmc_hopper_hop', group=group, seed=seed,
            run_id=f'{group}/seed{seed}', auc_complete=True, tail_complete=True,
            auc_return_per_budget=value, tail_mean_return=value))
    tasks, overall, differences = aggregate(rows, spec)
    self.assertEqual(len(tasks), 2)
    self.assertEqual(len(overall), 2)
    self.assertEqual(len(differences), 2)  # Only dt-baseline, hopper and hopper-only overall.
    self.assertTrue(all(r['auc_return_per_budget'] == 0 for r in differences))
    tasks = aggregate(rows[:-1], spec)[0]
    self.assertFalse(next(r for r in tasks if r['group']=='dt')['auc_complete'])
    extra = dict(rows[0], seed=0, run_id='historical-seed0')
    self.assertFalse(aggregate(rows+[extra], spec)[0][0]['auc_complete'])
    with self.assertRaises(ValueError):
      MetricSpec(seeds=(0, 1, 2, 3), tasks=spec.tasks, groups=spec.groups, scope=HOPPER_123)

  def test_new_profile_cli_six_hand_runs_and_default_is_not_relaxed(self):
    # Synthetic complete tables and checkpoint markers, never model execution.
    with TemporaryDirectory() as directory:
      root=Path(directory); runs=[]
      for mode, values in (('logging',(1,4,9)),('dt',(2,6,6))):
        for seed,value in enumerate(values,1):
          folder=root/f'{mode}-{seed}'; folder.mkdir()
          config={'seed':seed,'agent.rep_probe.mode':mode,'run.engineering_fixture':False,
                  'run.action_budget':1000000,'run.eval_eps':10}
          (folder/'protocol_manifest.json').write_text(json.dumps(dict(
              protocol=HOPPER_123,task='dmc_hopper_hop',root_seed=seed,
              git_commit='hand-fixture-only',config=config,eval_policy='fixture-eval-rng')))
          records=[]
          for step in range(0,1000001,50000):
            row=point(step,(value,)*10)
            row['lengths']=[1]*10; row['seeds']=list(range(10))
            template=row['episodes'][0]
            row['episodes']=[dict(template,episode_id=i,environment_seed=i,length=1) for i in range(10)]
            records.append(row)
            snap=folder/'eval_snapshots'/f'{step:07d}'; (snap/'fixture').mkdir(parents=True)
            (snap/'latest').write_text('fixture'); (snap/'fixture'/'done').touch()
            (snap/'fixture'/'agent.pkl').write_bytes(b'non-executable-fixture-marker')
            (snap/'evaluation_snapshot.json').write_text(json.dumps(dict(
                snapshot_id=row['snapshot_id'],actual_action_step=step,update_id=step,
                git_commit='hand-fixture-only')))
          (folder/'evaluations.jsonl').write_text('\n'.join(json.dumps(row) for row in records))
          runs.append(dict(directory=str(folder),run_id=f'{mode}-seed{seed}'))
      index=root/'index.json'; index.write_text(json.dumps(dict(runs=runs)))
      command=[sys.executable,'-m','analysis.return_metrics','--index',str(index),'--require-complete']
      passed=subprocess.run(command+['--profile','hopper-seeds123','--output',str(root/'repeat')],
                            capture_output=True,text=True)
      self.assertEqual(passed.returncode,0,passed.stderr)
      tables=json.loads((root/'repeat/tables.json').read_text())
      self.assertEqual(len(tables['episodes']),1260)
      self.assertEqual([r['auc_return_per_budget'] for r in tables['paired_seed_differences']], [1,2,-3])
      self.assertTrue(all(r['auc_return_per_budget']==0 for r in tables['dt_differences']))
      old=subprocess.run(command+['--output',str(root/'default-v1')],capture_output=True,text=True)
      self.assertEqual(old.returncode,2)
      duplicate=subprocess.run(command+['--profile','hopper-seeds123','--output',str(root/'repeat')],
                               capture_output=True,text=True)
      self.assertNotEqual(duplicate.returncode,0)


if __name__ == '__main__':
  unittest.main()
