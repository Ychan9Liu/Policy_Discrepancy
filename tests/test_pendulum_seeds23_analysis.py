"""Offline scope/hash/aggregation checks, not real experiment acceptance."""
import copy
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis import pendulum_seeds23_analysis as a
from analysis.analyze_m2_v1 import FIELDS


class PendulumAnalysisTest(unittest.TestCase):
    def accepted(self):
        rows = []
        for (seed, group), (gpu, mode) in a.MATRIX.items():
            root = a.OLD_ROOT if group == 'baseline' else a.ROOT
            directory = f'{root}/seed{seed}/{mode}-attempt01'
            run_id = f'fixture-{seed}-{group}'
            audit = dict(run_id=run_id, seed=seed, group=group, updates=249493, evaluations=21)
            rows.append(dict(seed=seed, group=group, task=a.TASK, mode=mode, run_id=run_id,
                physical_gpu=gpu, directory=directory, code_sha=a.TRAIN_SHA, state='accepted', exit_code=0,
                run_audit=audit, config={'seed':seed, 'task':a.TASK, 'logdir':directory,
                    'agent.rep_probe.mode':mode, 'run.engineering_fixture':False, 'agent.rep_probe.alpha':20}))
        return dict(runs=rows, controller_sha=a.CONTROLLER_SHA), dict(verified_run_count=4, code_sha=a.TRAIN_SHA,
            runs=[r['run_audit'] for r in rows]), dict(complete=True, training_sha=a.TRAIN_SHA,
                audit_controller_sha=a.CONTROLLER_SHA, runs=rows)

    def test_exact_combined_terminal_cohort(self):
        index, completion, supervisor = self.accepted()
        result = a.cohort(index, completion, supervisor)
        self.assertEqual({r['physical_gpu'] for r in result}, {1,4,5,6})
        self.assertTrue(all(a.OLD_ROOT in r['directory'] for r in result if r['group']=='baseline'))

    def test_live_incomplete_and_duplicate_members_refused(self):
        for change in ('live', 'incomplete', 'duplicate', 'bad_receipt', 'nonzero_exit'):
            index, completion, supervisor = self.accepted()
            if change == 'live': index['runs'][0]['state'] = 'running'
            if change == 'incomplete': supervisor['complete'] = False
            if change == 'duplicate': index['runs'][1] = copy.deepcopy(index['runs'][0])
            if change == 'bad_receipt': completion['verified_run_count'] = 3
            if change == 'nonzero_exit': index['runs'][0]['exit_code'] = 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                a.cohort(index, completion, supervisor)

    def test_superseded_attempt_engineering_other_seed_or_science_refused(self):
        for change in ('old_dt','seed1','engineering','alpha','controller_sha_as_training'):
            index, completion, supervisor = self.accepted()
            row = next(r for r in index['runs'] if r['group'] == 'dt')
            if change == 'old_dt': row['directory'] = row['directory'].replace(a.ROOT,a.OLD_ROOT)
            if change == 'seed1': row['seed'] = 1
            if change == 'engineering': row['config']['run.engineering_fixture'] = True
            if change == 'alpha': row['config']['agent.rep_probe.alpha'] = 10
            if change == 'controller_sha_as_training': row['code_sha'] = 'b7f42d46d16d4eda5cff5a798ec7b60cb27a3790'
            with self.subTest(change=change), self.assertRaises(ValueError):
                a.cohort(index, completion, supervisor)

    def test_terminal_index_and_supervisor_conflict_refused(self):
        index, completion, supervisor = self.accepted()
        supervisor = copy.deepcopy(supervisor)
        supervisor['runs'][0]['run_id'] = 'different-attempt'
        with self.assertRaises(ValueError): a.cohort(index, completion, supervisor)

    def test_update_weighting_keeps_seeds_and_ratio_of_sums(self):
        rows = []
        for seed in (2,3):
            for i in range(20):
                r = dict(seed=seed, group='dt', start_action=i*50000, end_action=(i+1)*50000,
                         updates=1 if i == 0 else 3, **{f:0. for f in FIELDS})
                r.update(D_mean=seed if i == 0 else seed+2, D_zero_frac=.2, pooled_active_frac=.5,
                    pooled_source_active_weight=.8, weighted_excess_mean=3 if i == 0 else 1,
                    excess_reduction_mean=1 if i == 0 else 3, rep_before_mean=4, rep_after_mean=3)
                rows.append(r)
        summary = a.summarize(rows)
        first = next(r for r in summary if r['seed'] == 2 and r['stage'] == 'early')
        other = next(r for r in summary if r['seed'] == 3 and r['stage'] == 'early')
        self.assertEqual(first['D_mean'],3.5)
        self.assertEqual(other['D_mean'],4.5)
        self.assertEqual(first['excess_reduction_ratio'],.625)
        self.assertEqual(first['rep_value_reduction_ratio'],.25)
        self.assertAlmostEqual(first['pooled_source_active_weight'],.8)

    def test_mutated_mirror_is_not_accepted(self):
        with TemporaryDirectory() as tmp:
            source=Path(tmp); mirror=source/'inputs'; mirror.mkdir()
            data=b'original'; (mirror/'data.json').write_bytes(data)
            a.dump(mirror/'COLLECTION.json',dict(files={'data.json':dict(bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest())}))
            a.verify_mirror(source)
            (mirror/'data.json').write_bytes(b'modified')
            with self.assertRaises(ValueError): a.verify_mirror(source)


if __name__ == '__main__':
    unittest.main()
