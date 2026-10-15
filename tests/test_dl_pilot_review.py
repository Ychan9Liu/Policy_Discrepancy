"""Hand-calculated return fixtures and corrupted-evidence review guards."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import dl_pilot_review as review


def rows():
  result = []
  for point in review.GRID:
    scores = [float(i + point / 10000) for i in range(10)]
    seeds = [review.seed32(point, i) for i in range(10)]
    result.append(dict(action_step=point, scores=scores, returns_raw=scores,
        seeds=seeds, lengths=[1000]*10, mean=math.fsum(scores)/10,
        mean_return_raw=math.fsum(scores)/10, snapshot_action_step=point,
        target_action_step=point, snapshot_update_id=point//4-507 if point else 0,
        update_id=point//4-507 if point else 0, snapshot_id=f'fixture/{point}',
        reward_source='environment_interface.reward',
        return_definition='undiscounted_sum_non_reset_rewards_repeat_already_summed',
        episodes=[dict(complete=True, episode_id=i, environment_seed=seeds[i],
            length=1000, return_raw=scores[i]) for i in range(10)]))
  return result


class ReviewContract(unittest.TestCase):
  def test_hand_calculated_linear_curve_and_frozen_tail(self):
    actual = review.metrics(review.evaluations(rows()))
    self.assertEqual(actual['auc_per_1m_actions'], 54.5)
    self.assertEqual(actual['tail_mean_return'], 94.5)
    self.assertEqual(actual['endpoint_return'], 104.5)
    self.assertEqual(actual['descriptive_windows']['0_300000'], 19.5)

  def test_rejects_missing_grid_incomplete_episode_and_wrong_rng(self):
    complete = rows()
    for name in ('grid', 'episode', 'rng', 'mean', 'snapshot'):
      altered = copy.deepcopy(complete)
      if name == 'grid': altered.pop(3)
      if name == 'episode': altered[8]['episodes'][0]['complete'] = False
      if name == 'rng': altered[8]['seeds'][0] += 1
      if name == 'mean': altered[8]['mean'] += .1
      if name == 'snapshot': altered[8]['snapshot_update_id'] += 1
      with self.subTest(name=name), self.assertRaises(ValueError):
        review.evaluations(altered)

  def exercise_receipt(self, *, invalid_action=False, unavailable=False):
    metric_names = ('actual_v_mean', 'actual_v_min', 'active_frac', 'valid_active_frac',
        'active_protected_frac', 'e_valid_mean', 'D_valid_mean', 'signed_support_valid_mean',
        'u_q_valid_mean', 'u_p_valid_mean', 'rep_before_mean', 'rep_after_mean',
        'excess_release_mean', 'weighted_excess_mean', 'calibration_num', 'calibration_den')
    metrics = {'dl/' + key: 0. for key in metric_names}
    metrics.update({'dl/actual_v_mean': 1., 'dl/actual_v_min': 1., 'dl/actual_v_max': 1.,
        'dl/valid_count': 0 if unavailable else 1008, 'dl/unavailable_count': 1024 if unavailable else 16,
        'dl/rep_before_mean': 10., 'dl/rep_after_mean': 10., 'dl/weighted_excess_mean': 9.,
        'dl/calibration_den': 0. if unavailable else 1.})
    if unavailable:
      for key in metrics:
        if key.endswith('_valid_mean'): metrics[key] = 'nan'
    row = dict(update_id=0, start_action=2033 if invalid_action else 2032, invalid_count=0,
        match_sum=0., match_count=0, optimizer_loss=1., optimizer_grad_norm=1., metrics=metrics)
    with patch.object(review, 'UPDATES', 1), patch.object(review, 'jsonl', return_value=iter([row])):
      return review.receipts('fixture', {'records': {0: (2032, 0., 0, 0)}})

  def test_receipt_schedule_and_final_checkpoint_ledger_binding(self):
    actual = self.exercise_receipt()
    self.assertEqual(actual['updates'], 1)
    self.assertEqual(actual['mean_weight_reduction'], 0.)
    self.assertEqual(actual['observed_excess_relative_reduction'], 0.)
    with self.assertRaisesRegex(ValueError, 'schedule'):
      self.exercise_receipt(invalid_action=True)

  def test_unavailable_signal_remains_unavailable_instead_of_becoming_zero(self):
    actual = self.exercise_receipt(unavailable=True)
    self.assertIsNone(actual['statistics']['e_valid_mean']['mean'])
    self.assertIsNone(actual['statistics']['e_valid_mean']['valid_position_weighted_mean'])
    self.assertEqual(actual['unavailable_updates']['e_valid_mean'], 1)
    self.assertIsNone(actual['relative_to_D_only_release_on_valid_N_positions'])


if __name__ == '__main__':
  unittest.main()
