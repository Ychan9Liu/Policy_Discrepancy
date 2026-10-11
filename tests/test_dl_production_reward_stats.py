"""Synthetic numeric tests only; no source arrays, Dreamer, physics or GPU."""
import itertools
import json
import os
from pathlib import Path
import sys
import unittest

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
  os.environ[name] = '1'
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import dl_production_reward_stats as stats


def fixture():
  truth = np.zeros(stats.TRUTH_SHAPE, np.float64)
  for state in range(4):
    truth[:, state, 2 + state % 2] = 1.
  predictions = np.zeros(stats.PREDICTION_SHAPE, np.float32)
  predictions[:, :, :, 0, :, :, 0] = truth[:, :, None, None].astype(np.float32)
  return predictions, truth, np.asarray(stats.EPISODES, np.int32)


def interval(point=.01, lower=.008, upper=.012):
  return dict(primary=float(point), lower95=float(lower), upper95=float(upper))


def reliability_fixture():
  return {name: dict(mse_gaps={key: interval() for key in ('p_minus_q', 'constant_minus_q')},
                    utility_gaps={key: interval() for key in ('q_minus_p', 'q_minus_constant')})
          for name in stats.SUBSETS}


class ProductionRewardStatsTest(unittest.TestCase):

  def test_offdiagonal_unbiased_estimator_independent_oracle(self):
    paths = np.array([[1., 2., -.5], [-2., .5, 3.], [2., -1., .25],
                      [.5, 2.5, 1.]], np.float64)
    truth = np.array([.25, 1.25, -.75], np.float64)
    actual = stats._moments(paths, truth)
    # Independently sum ordered unequal pairs of control-centered residuals.
    residual = paths - paths.sum(1, keepdims=True) / 3
    residual -= truth - truth.sum() / 3
    total = 0.
    for left in range(4):
      for right in range(4):
        if left != right:
          total += float(np.dot(residual[left], residual[right]) / 3)
    expected = total / (4 * 3)
    self.assertAlmostEqual(float(actual['bc_mse']), expected, places=13)
    self.assertAlmostEqual(float(actual['plugin_centered_mse'] - actual['mc_correction']),
                           expected, places=13)

  def test_heteroscedastic_paths_can_fake_plugin_skill_but_not_expected_bc_skill(self):
    direction = np.array([1., -1., 0.], np.float64)
    corrected, plugins = [], []
    # All four equally likely IID two-path outcomes. The truth and low-variance
    # interface mean are zero. The high-variance interface has the same mean.
    for signs in itertools.product((-1., 1.), repeat=2):
      value = stats._moments(np.asarray(signs)[:, None] * direction, np.zeros(3, np.float64))
      corrected.append(float(value['bc_mse']))
      plugins.append(float(value['plugin_centered_mse']))
    self.assertGreater(float(np.mean(plugins)), 0.)
    self.assertEqual(float(np.mean(corrected)), 0.)
    self.assertLess(min(corrected), 0.)  # Negative finite estimates must survive.

  def test_control_centering_precedes_variance_and_excludes_common_level(self):
    truth = np.array([1., 2., 3.], np.float64)
    common = np.array([0., 1., 2., 1.], np.float64)
    value = stats._moments(truth[None] + common[:, None], truth)
    self.assertEqual(float(value['bc_mse']), 0.)
    self.assertEqual(float(value['mc_correction']), 0.)
    self.assertEqual(float(value['plugin_level_mse']), 1.)
    np.testing.assert_array_equal(value['centered_path_sample_variance'], np.zeros(3))

  def test_full_report_positive_identity_and_preserves_all_numeric_inputs(self):
    predictions, truth, episodes = fixture()
    before = tuple(x.tobytes() for x in (predictions, truth, episodes))
    result = stats.report(predictions, truth, episodes)
    self.assertEqual(result['primary_status'], 'finite_native_mean_exploratory_supported')
    self.assertEqual(result['utility_reliability']['status'], 'finite_native_choice_exploratory_supported')
    for key in ('p_minus_q', 'constant_minus_q'):
      # Repeated averaging a non-binary constant has roundoff; it is recorded,
      # never snapped to zero or confused with the scientific .0004 gate.
      self.assertLess(result['mse_reliability']['metrics'][key]['epsilon'], 1e-14)
    self.assertEqual(result['subsets']['pooled128']['interfaces']['q']['bc_mse']['primary'], 0.)
    self.assertEqual(result['subsets']['pooled128']['utility_gaps']['q_minus_p']['primary'], 1.)
    self.assertEqual(result['subsets']['pooled128']['utility_gaps']['q_minus_constant']['primary'], .5)
    self.assertEqual(before, tuple(x.tobytes() for x in (predictions, truth, episodes)))
    self.assertFalse(result['proceed_short_training'])
    self.assertFalse(result['real_gate_authority'])
    self.assertFalse(result['source_provenance_verified_by_stats'])
    self.assertFalse(result['future_paths_independence_verified_by_stats'])
    json.dumps(result, allow_nan=False)
    predictions[:, :, :, 1] = predictions[:, :, :, 0]
    same = stats.report(predictions, truth, episodes)
    self.assertEqual(same['primary_status'], 'finite_native_mean_not_supported')
    self.assertEqual(same['subsets']['pooled128']['mse_gaps']['p_minus_q']['primary'], 0.)
    self.assertEqual(same['subsets']['pooled128']['mse_gaps']['p_minus_q']['upper95'], 0.)

  def test_perfect_residual_is_exact_zero_despite_direct_formula_roundoff(self):
    truth = np.array([1., 0., 0., 0., 0., 0., 0.], np.float64)
    paths = np.broadcast_to(truth, (128, 7)).copy()
    centered = paths - paths.mean(-1, keepdims=True)
    target = truth - truth.mean()
    # This is an arithmetic counterexample, not a new model/case selection.
    direct = np.mean((centered.mean(0) - target) ** 2 - centered.var(0, ddof=1) / 128)
    self.assertGreater(float(direct), 0.)
    actual = stats._moments(paths, truth)
    self.assertEqual(float(actual['bc_mse']), 0.)
    residual = centered - target
    offdiagonal = sum(float(np.dot(residual[i], residual[j]))
                      for i in range(128) for j in range(128) if i != j) / (128 * 127 * 7)
    self.assertEqual(offdiagonal, 0.)

  def test_initial_samples_average_after_square_and_selection(self):
    predictions = np.zeros(stats.PREDICTION_SHAPE, np.float32)
    truth = np.zeros(stats.TRUTH_SHAPE, np.float64)
    direction = np.array([1., -1., 0., 0., 0., 0., 0.], np.float32)
    for initial in range(4):
      predictions[:, :, initial, 0, :, :, 0] = (1 if initial % 2 == 0 else -1) * direction
    value = stats.report(predictions, truth, np.asarray(stats.EPISODES, np.int32))
    self.assertAlmostEqual(value['subsets']['pooled128']['interfaces']['q']['bc_mse']['primary'], 2 / 7)
    self.assertEqual(value['primary_status'], 'finite_native_mean_not_supported')
    choices = np.asarray(value['subsets']['pooled128']['interfaces']['q']['choices'])
    np.testing.assert_array_equal(choices[0, 0], [0, 1, 0, 1])
    # Averaging the four predictions first would yield zero error and control0
    # for every initial sample, hiding the conditional-interface distinction.
    self.assertEqual(float(predictions[:, :, :, 0, :, :, 0].mean(2).max()), 0.)

  def test_native_f32_rewards_sum_in_f64_and_subsets_are_predeclared(self):
    predictions = np.zeros(stats.PREDICTION_SHAPE, np.float32)
    truth = np.zeros(stats.TRUTH_SHAPE, np.float64)
    truth[..., 0] = 1.
    predictions[:, :, :, 0, :, 0, :3] = [1e8, 1., -1e8]
    result = stats.report(predictions, truth, np.asarray(stats.EPISODES, np.int32))
    mean = np.asarray(result['subsets']['pooled128']['cumulative_mean'])
    self.assertEqual(float(mean[0, 0, 0, 0, 0]), 1.)
    self.assertEqual(result['subsets']['pooled128']['interfaces']['q']['bc_mse']['primary'], 0.)
    self.assertEqual(result['subsets']['first32']['path_indices'], list(range(32)))
    self.assertEqual(result['subsets']['first64']['path_indices'], list(range(64)))
    self.assertEqual(result['subsets']['last64']['path_indices'], list(range(64, 128)))
    self.assertEqual(result['subsets']['block2']['path_indices'], list(range(64, 96)))

  def test_loeo_constant_excludes_the_heldout_episode_labels(self):
    _, truth, episodes = fixture()
    first, _, _ = stats._constant(truth, episodes)
    changed = truth.copy()
    changed[0] = np.array([2., 8., -1., 3., 1., 0., -2.])
    second, _, _ = stats._constant(changed, episodes)
    self.assertEqual(first['folds'][0], second['folds'][0])
    self.assertEqual(first['centered_cost_vectors'][0], second['centered_cost_vectors'][0])
    self.assertNotEqual(first['centered_cost_vectors'][1], second['centered_cost_vectors'][1])

  def test_whole_episode_bootstrap_not_rng_or_state_pseudoreplication(self):
    draws = np.random.Generator(np.random.PCG64(20261025)).integers(0, 8, (2000, 8))
    values = np.array([1., 0., 0., 0., 0., 0., 0., 0.], np.float64)
    result = stats._interval(values, draws)
    distribution = sorted(float(np.count_nonzero(row == 0) / 8) for row in draws)
    def quantile(q):
      coordinate = (len(distribution) - 1) * q
      left = int(np.floor(coordinate)); right = int(np.ceil(coordinate))
      return distribution[left] + (coordinate - left) * (distribution[right] - distribution[left])
    self.assertEqual(result['primary'], 1 / 8)
    self.assertEqual(result['lower95'], quantile(.025))
    self.assertEqual(result['upper95'], quantile(.975))
    per_initial = np.broadcast_to(values[:, None, None], (8, 4, 4)).copy()
    first = stats._aggregate(per_initial)
    repeated = stats._aggregate(np.repeat(per_initial, 3, axis=2))
    self.assertEqual(first['state_values'], repeated['state_values'])
    self.assertEqual(first['episode_values'], repeated['episode_values'])

  def test_global_tie_uses_maximum_set_and_ignores_mc_epsilon(self):
    values = np.array([0., .75e-10, 1.5e-10, -1.], np.float64)
    self.assertEqual(int(stats._argmax(values)), 1)  # No pairwise tie chaining.
    self.assertEqual(int(stats._argmax(np.array([0., stats.TIE]))), 0)
    self.assertEqual(int(stats._argmax(np.array([0., np.nextafter(stats.TIE, np.inf)]))), 1)
    self.assertEqual(int(stats._argmax(np.array([0., .0003]))), 1)

  def test_mse_and_true_choice_can_have_opposite_direction(self):
    truth = np.array([1., .99, 0.], np.float64)
    q = np.tile([.98, 1., 0.], (4, 1)).astype(np.float64)
    p = np.tile([1., .98, -1.], (4, 1)).astype(np.float64)
    q_error = float(stats._moments(q, truth)['bc_mse'])
    p_error = float(stats._moments(p, truth)['bc_mse'])
    self.assertLess(q_error, p_error)
    self.assertLess(truth[int(stats._argmax(q.mean(0)))], truth[int(stats._argmax(p.mean(0)))])

  def test_reliability_statuses_epsilon_boundary_and_strict_zero_sides(self):
    values = reliability_fixture()
    result = stats._reliability(values, 'mse_gaps', .0004, 'mean')
    self.assertEqual(result['status'], 'finite_native_mean_exploratory_supported')
    for data in values.values():
      data['mse_gaps']['p_minus_q'] = interval(0., 0., 0.)
    self.assertEqual(stats._reliability(values, 'mse_gaps', .0004, 'mean')['status'],
                     'finite_native_mean_not_supported')
    for data in values.values():
      data['mse_gaps']['p_minus_q'] = interval(0., -.01, .01)
    values['first32']['mse_gaps']['p_minus_q']['primary'] = .0004
    edge = stats._reliability(values, 'mse_gaps', .0004, 'mean')
    self.assertTrue(edge['mc_stable'])
    self.assertEqual(edge['status'], 'finite_native_mean_inconclusive')
    values['first32']['mse_gaps']['p_minus_q']['primary'] = float(np.nextafter(.0004, np.inf))
    self.assertEqual(stats._reliability(values, 'mse_gaps', .0004, 'mean')['status'], 'MC-inconclusive')
    # A tiny positive lower bound is not treated as zero by any audit tolerance.
    values = reliability_fixture()
    for data in values.values():
      data['mse_gaps']['p_minus_q'] = interval(.01, np.nextafter(0., 1.), .02)
    self.assertEqual(stats._reliability(values, 'mse_gaps', .0004, 'mean')['status'],
                     'finite_native_mean_exploratory_supported')

  def test_relative_margin_and_lower_minus_epsilon_are_necessary(self):
    values = reliability_fixture()
    for data in values.values():
      data['mse_gaps']['p_minus_q'] = interval(.001, .0004, .002)
    values['first32']['mse_gaps']['p_minus_q']['primary'] = .0013
    result = stats._reliability(values, 'mse_gaps', .0004, 'mean')
    self.assertTrue(result['mc_stable'])
    self.assertFalse(result['metrics']['p_minus_q']['positive_margin_robust'])
    self.assertEqual(result['status'], 'finite_native_mean_inconclusive')
    values = reliability_fixture()
    for data in values.values():
      data['mse_gaps']['p_minus_q'] = interval(.01, .0001, .02)
    values['first32']['mse_gaps']['p_minus_q']['primary'] = .0102
    self.assertFalse(stats._reliability(values, 'mse_gaps', .0004, 'mean')
                     ['metrics']['p_minus_q']['positive_margin_robust'])
    # Relative/lower bounds can hold for a large effect while the absolute
    # MC-resolution condition fails. It must never advertise robust support.
    values = reliability_fixture()
    for data in values.values():
      data['mse_gaps']['p_minus_q'] = interval(1., .9, 1.1)
    values['first32']['mse_gaps']['p_minus_q']['primary'] = 1.001
    result = stats._reliability(values, 'mse_gaps', .0004, 'mean')
    self.assertTrue(result['positive_margin_conditions_met'])
    self.assertFalse(result['necessary_directions_robust'])
    self.assertEqual(result['status'], 'MC-inconclusive')

  def test_half_block_disagreement_and_true_switch_cost_are_all_reported(self):
    first = dict(interfaces={name: dict(choices=[[[0, 1]]], selected_reward=dict(
        per_initial=[[[1., .9]]])) for name in ('q', 'p')})
    second = dict(interfaces={name: dict(choices=[[[0, 0]]], selected_reward=dict(
        per_initial=[[[1., 1.]]])) for name in ('q', 'p')})
    result = stats._choice_compare(first, second)
    self.assertEqual(result['q']['changed_choices'], 1)
    self.assertEqual(result['q']['agreement_fraction'], .5)
    self.assertAlmostEqual(result['q']['maximum_absolute_switch_cost'], .1)
    np.testing.assert_allclose(result['q']['absolute_switch_cost'], [[[0., .1]]], atol=1e-15)

  def test_schema_partial_axis_dtype_identity_nonfinite_are_rejected(self):
    predictions, truth, episodes = fixture()
    for wrong in (predictions[..., :19], predictions[:, :, :, :, :127],
                  predictions.swapaxes(3, 5), predictions.astype(np.float64)):
      with self.assertRaises(ValueError):
        stats.report(wrong, truth, episodes)
    for wrong in (truth.astype(np.float32), truth[..., :6]):
      with self.assertRaises(ValueError):
        stats.report(predictions, wrong, episodes)
    for wrong in (episodes.astype(np.int64), episodes[::-1],
                  np.array([9, 9, 15, 18, 20, 24, 30, 31], np.int32)):
      with self.assertRaises(ValueError):
        stats.report(predictions, truth, wrong)
    for array, index in ((predictions, (0, 0, 0, 0, 0, 0, 0)), (truth, (0, 0, 0))):
      old = array[index]
      for invalid in (np.nan, np.inf, -np.inf):
        array[index] = invalid
        with self.assertRaises(ValueError):
          stats.report(predictions, truth, episodes)
      array[index] = old


if __name__ == '__main__':
  unittest.main()
