import unittest
from analysis.compare_seed01_dtl import GRID, analysis_tasks, evaluation_metrics, paired_means


class SeedAnalysisTest(unittest.TestCase):
    def evaluations(self):
        return [dict(action_step=s, mean=i, episodes=[dict(episode_id=j, complete=True,
            length=1000, return_raw=i, return_definition='undiscounted_sum_non_reset_rewards_repeat_already_summed')
            for j in range(10)]) for i,s in enumerate(GRID)]

    def test_hand_calculated_auc_and_tail(self):
        result=evaluation_metrics(self.evaluations())
        self.assertAlmostEqual(result['auc'],10)
        self.assertEqual(result['tail'],18)

    def test_missing_point_or_episode_rejected(self):
        ev=self.evaluations()
        with self.assertRaises(ValueError):evaluation_metrics(ev[:-1])
        ev[0]['episodes'].pop()
        with self.assertRaises(ValueError):evaluation_metrics(ev)

    def test_mean_uses_equal_training_seeds(self):
        rows=[dict(task='a',seed=s,group=g,auc=v,tail=v) for s,g,v in
              [(0,'baseline',10),(0,'dt',30),(1,'baseline',100),(1,'dt',80)]]
        r=paired_means(rows,['a'])[0]
        self.assertEqual((r['baseline_auc'],r['dt_auc'],r['delta_auc']),(55,55,0))
        self.assertEqual((r['seed0_delta_auc'],r['seed1_delta_auc']),(20,-20))
        with self.assertRaises(ValueError):paired_means(rows[:-1],['a'])
        with self.assertRaises(ValueError):paired_means(rows+[rows[0]],['a'])

    def test_requested_scope_cannot_drop_acrobot(self):
        keys=[(str(t),s,g) for t in range(19) for s in (0,1) for g in ('baseline','dt')]
        keys += [(str(t),0,'dtl') for t in range(3)]
        self.assertEqual(len(analysis_tasks(keys)),19)
        with self.assertRaises(ValueError):analysis_tasks(keys,True)
        keys += [('dmc_acrobot_swingup',s,g) for s in (0,1) for g in ('baseline','dt')]
        self.assertEqual(len(analysis_tasks(keys,True)),20)
        with self.assertRaises(ValueError):analysis_tasks(keys)


if __name__=='__main__':unittest.main()
