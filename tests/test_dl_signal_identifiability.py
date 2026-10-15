"""Hand-derived bounds and an exhaustive finite counterexample, never RL tests."""
import copy
from fractions import Fraction
import json
import itertools
import math
import unittest

from analysis import dl_signal_identifiability as d


def gate(num=1.,den=4.):
  return dict(updates=2,relative_to_D_only_release_on_valid_N_positions=.5*num/den,
      statistics=dict(calibration_num=dict(n=2,mean=num),calibration_den=dict(n=2,mean=den),
          e_valid_mean=dict(valid_position_weighted_mean=.2),
          signed_support_valid_mean=dict(valid_position_weighted_mean=.1),
          u_q_valid_mean=dict(valid_position_weighted_mean=.6),
          u_p_valid_mean=dict(valid_position_weighted_mean=.5)))


class Identifiability(unittest.TestCase):
  def test_hand_derived_strength_feasibility_and_zero_signal(self):
    result=d.same_batch_bounds(gate())
    self.assertEqual(result['maximum_ratio_with_rho_at_most_one'],.25)
    self.assertEqual(result['rho_needed_to_equal_same_mask_D'],4.)
    self.assertFalse(result['legal_rho_can_match'])
    self.assertEqual(result['mean_absolute_support'],.30000000000000004)
    zero=d.same_batch_bounds(gate(0.,4.))
    self.assertIsNone(zero['rho_needed_to_equal_same_mask_D'])
    self.assertFalse(zero['finite_rho_can_match'])
    self.assertTrue(d.same_batch_bounds(gate(4.,4.))['legal_rho_can_match'])
    json.dumps(result,allow_nan=False)

  def test_invalid_and_inconsistent_marginals_are_not_evidence(self):
    for mutate in (
        lambda g:g.update(updates=True),
        lambda g:g['statistics']['calibration_den'].update(mean=0),
        lambda g:g['statistics']['calibration_num'].update(n=1),
        lambda g:g.update(relative_to_D_only_release_on_valid_N_positions=.5),
        lambda g:g['statistics']['e_valid_mean'].update(valid_position_weighted_mean=math.nan)):
      value=gate();mutate(value)
      with self.assertRaises(ValueError): d.same_batch_bounds(value)
    for payload in (b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":1e400}'):
      with self.assertRaises(ValueError): d.decode(payload)

  def test_three_world_actor_use_and_information_usefulness_are_distinct(self):
    result=d.categorical_witness()
    self.assertTrue(result['rep_active']);self.assertAlmostEqual(result['K'],1.2185938242800805)
    self.assertAlmostEqual(result['rep_dK_dconfidence_with_prior_fixed'],math.log(97))
    self.assertEqual(result['readouts_exhaustively_checked'],256)
    self.assertAlmostEqual(result['rep_finite_difference_check'],math.log(97),places=7)
    worlds=result['worlds']
    self.assertEqual(worlds['useful_current_readout']['current_actor_difference_exact'],'18/25')
    self.assertEqual(worlds['neutral']['current_actor_difference_exact'],'0')
    harmful=worlds['harmful_current_readout']
    self.assertEqual(harmful['current_actor_difference_exact'],'-18/25')
    self.assertEqual(harmful['best_readout_advantage_exact'],'6/25')
    self.assertEqual(len(harmful['all_readouts']),256)
    self.assertEqual(harmful['best_q_readout_witness'],[1,0,0,0])
    self.assertEqual([row['e'] for row in result['H1_label_receipts']],[0.,0.])
    self.assertTrue(result['H1_e_identically_zero'])
    self.assertEqual(result['N_weight_for_any_finite_legal_alpha_rho'],1.)
    boundary=d.categorical_witness(Fraction(1,4))
    self.assertEqual(boundary['K'],0.)
    self.assertFalse(boundary['rep_active'])
    self.assertTrue(all(w['best_readout_information_advantage']==0 for w in boundary['worlds'].values()))

  def test_incomplete_readouts_leaked_cue_bad_probabilities_and_H1_rejected(self):
    readouts=tuple(itertools.product(range(4),repeat=4))
    for bad in (readouts[:-1],readouts[:-1]+readouts[:1]):
      with self.assertRaises(ValueError):d.validate_readouts(bad)
    prior=((Fraction(1,4),)*4,)*4
    bad=[list(row) for row in prior];bad[0]=[Fraction(1,2),Fraction(1,6),Fraction(1,6),Fraction(1,6)]
    with self.assertRaisesRegex(ValueError,'cue'):d.validate_latents(prior,tuple(tuple(x) for x in bad))
    bad=[list(row) for row in prior];bad[0][0]=Fraction(0)
    with self.assertRaises(ValueError):d.validate_latents(tuple(tuple(x) for x in bad),prior)
    for label in (True,-1,2):
      with self.assertRaises(ValueError):d.one_hot_support((Fraction(99,100),Fraction(1,100)),label)
    with self.assertRaisesRegex(ValueError,'true law'):
      d.calibrated_H1((Fraction(9,10),Fraction(1,10)),(Fraction(99,100),Fraction(1,100)),
          (Fraction(99,100),Fraction(1,100)))


if __name__=='__main__': unittest.main()
