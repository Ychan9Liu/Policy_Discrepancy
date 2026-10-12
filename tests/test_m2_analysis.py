"""Analysis checks: successful update units, snapshot stamps and pooled ratios."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from analysis.analyze_m2_v1 import FIELDS, receipts, summarize_diag


class M2AnalysisTest(unittest.TestCase):
    def make_receipts(self, root, rows):
        path = root / 'receipts.jsonl'
        path.write_text('\n'.join(json.dumps(r) for r in rows))
        # One receipt per 50k interval. No additional updates in later intervals.
        limits = [0, 1, 2, 3, 4, 5, 6] + [6]*14
        evaluations = [{'update_id':x} for x in limits]
        final = dict(updates=6,train_batch_keys=6,match_sum=8.0,match_count=12)
        return path, final, evaluations

    def rows(self):
        steps = [50000, 100000, 150000, 200000, 250000, 300000]
        return [dict(update_id=i,start_action=s,match_sum=2.0,match_count=3,invalid_count=0)
                for i,s in enumerate(steps)]

    def test_half_open_window_and_all_successful_units(self):
        with TemporaryDirectory() as temp:
            path, final, ev = self.make_receipts(Path(temp),self.rows())
            result = receipts(path,final,ev)
            self.assertEqual(result['included_update_ids'],[1,2,3,4])
            self.assertEqual(result['match_count'],12)
            self.assertEqual(sum(r['updates'] for r in result['intervals']),6)

    def test_nonlogging_zero_receipts_are_not_activity_measurements(self):
        with TemporaryDirectory() as temp:
            rows = [dict(r,match_sum=0.0,match_count=0) for r in self.rows()]
            path, final, ev = self.make_receipts(Path(temp),rows)
            final.update(match_sum=0.0,match_count=0)
            result = receipts(path,final,ev)
            self.assertEqual(result['updates'],6)
            self.assertEqual(result['match_count'],0)

    def test_duplicate_ids_and_conflicting_snapshot_stamps_refused(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            for field, value in (('update_id',0),('start_action',40000)):
                rows=self.rows()
                rows[1][field]=value
                path, final, ev=self.make_receipts(root,rows)
                with self.assertRaises(ValueError):
                    receipts(path,final,ev)

    def test_update_weighting_and_ratio_of_accumulations(self):
        rows=[]
        for start, updates, active, nearzero, weighted, reduced in (
                (0,1,.5,.1,3,1),(50000,3,1,.3,1,3)):
            r=dict(task='task',group='dt',start_action=start,end_action=start+50000,
                   updates=updates,**{f:.0 for f in FIELDS})
            r.update(D_mean=updates, D_zero_frac=.2,pooled_active_frac=active,
                     pooled_source_active_weight=.8,rep_active_frac=active,
                     raw_active_D_near_zero_frac=nearzero,active_D_near_zero_conditional=nearzero/active,
                     weighted_excess_mean=weighted,excess_reduction_mean=reduced,
                     rep_value_reduction_ratio=.2)
            rows.append(r)
        # Fill remaining intervals so production stage definitions stay meaningful.
        for start in range(100000,1000000,50000):
            rows.append(dict(rows[1],start_action=start,end_action=start+50000))
        with patch('analysis.analyze_m2_v1.rm.TASKS',('task',)), patch('analysis.analyze_m2_v1.rm.GROUPS',('dt',)):
            early=next(r for r in summarize_diag(rows) if r['stage']=='early')
        self.assertEqual(early['D_mean'],2.5)
        self.assertEqual(early['excess_reduction_ratio'],.625)
        self.assertAlmostEqual(early['active_D_near_zero_conditional'],1/3.5)


if __name__=='__main__':
    unittest.main()
