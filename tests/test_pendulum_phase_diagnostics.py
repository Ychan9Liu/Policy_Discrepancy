"""Hand-counted alignment/error cases; fixtures are not experiment acceptance."""
import copy
import unittest

from analysis.pendulum_seeds23_analysis import training_phase_diagnostics


class PhaseDiagnosticsTest(unittest.TestCase):
    def inputs(self):
        config = {'agent.loss_scales.dyn':1.,'agent.loss_scales.rep':.1,'agent.loss_scales.rec':1.}
        evaluations = []
        for i in range(21):
            values = [0]*10 if i == 0 else [10*(i-1)+j for j in range(10)]
            evaluations.append(dict(action_step=i*50000,update_id=0 if i == 0 else 11993+(i-1)*12500,
                episodes=[dict(complete=True,length=1000,return_raw=v) for v in values],
                mean=0 if i == 0 else 10*(i-1)+4.5))
        phases,logs = [],[]
        for i in range(20):
            phases.append(dict(start_action=i*50000,end_action=(i+1)*50000,action_count=50000,
                               positive_reward_count=5000*(i%9+1),reward_sum=5000*(i%9+1)))
            logs.append({'step':(i+1)*50000,'train/rew':.2,'train/ret':.6,
                'train/adv_mag':.5,'train/ent/action':1.2,'train/loss/rew':.3,'train/loss/image':4.,
                'train/opt/grad_norm':2.,'train/opt/update_rms':.001,'train/dt/active_weight_mean':.8,
                'train/dt/rep_active_frac':.25,'train/dt/D_mean':.01,
                'train/dt/mean_abs_diff':.04,'train/dt/std_abs_diff':.03,
                'train/loss/dyn':2.,'train/loss/rep':3.,'train/opt/loss':6.3,
                'train/opt/updates':5997. if i == 0 else 18243.5+(i-1)*12500})
        return config,evaluations,logs,phases

    def run_inputs(self,values):
        return training_phase_diagnostics(dict(seed=2,group='dt',run_id='fixture'),*values)

    def test_hand_counted_action_and_update_means_stay_separate(self):
        original=self.inputs(); saved=copy.deepcopy(original)
        first,second,*_=self.run_inputs(original)
        self.assertEqual((first['positive_reward_fraction'],second['positive_reward_fraction']),(.1,.2))
        self.assertEqual((first['evaluation_start_return'],first['evaluation_return'],second['evaluation_return']),(0,4.5,14.5))
        self.assertEqual((first['updates'],second['updates']),(11993,12500))
        self.assertEqual((first['imagined_reward'],first['policy_entropy'],first['advantage_magnitude']),(.2,1.2,.5))
        self.assertAlmostEqual(first['loss_reconstruction_abs_error'],0)
        self.assertEqual(original,saved)

    def test_bad_alignment_missing_data_and_nonfinite_logs_refused(self):
        for defect in ('reordered_phase','duplicate_log','missing_point','wrong_actions',
                       'nonbinary_sum','update_gap','wrong_update_mean','mean_conflict','nonfinite','loss_scale'):
            values=self.inputs(); config,ev,logs,phases=values
            if defect=='reordered_phase': phases[0],phases[1]=phases[1],phases[0]
            if defect=='duplicate_log': logs[1]=copy.deepcopy(logs[0])
            if defect=='missing_point': ev.pop()
            if defect=='wrong_actions': phases[0]['action_count']=49999
            if defect=='nonbinary_sum': phases[0]['reward_sum']=4999
            if defect=='update_gap': ev[1]['update_id']+=1
            if defect=='wrong_update_mean': logs[0]['train/opt/updates']+=1
            if defect=='mean_conflict': ev[1]['episodes'][0]['return_raw']=100
            if defect=='nonfinite': logs[0]['train/adv_mag']=float('nan')
            if defect=='loss_scale': config['agent.loss_scales.rep']=.2
            with self.subTest(defect=defect),self.assertRaises(ValueError):
                self.run_inputs(values)


if __name__=='__main__': unittest.main()
