"""Synthetic calibration-statistics evidence, not real control/method evidence.

No artifacts, model, simulator, JAX, or GPU are imported by these tests.
"""

import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_action_information_stats as stats


EPISODES = np.array([9, 14, 15, 18, 20, 24, 30, 31], np.int32)


def fixture():
  # Two task states need different controls. q carries this observable type,
  # whereas p and h are constant. This is a controlled readout fixture only.
  fixed = np.zeros((8, 4, 7), np.float64)
  latent = np.zeros((8, 4, 4, 2, 1, 2), np.float64)
  for position in range(4):
    kind = position % 2
    fixed[:, position, kind] = 1.
    latent[:, position, :, 0, 0, kind] = 1.
    latent[:, position, :, 1, 0, 0] = 1.
  actor = np.full((8, 4, 4, 2), .3, np.float64)
  actor[..., 0] = .2
  h = np.zeros((8, 4, 3), np.float64)
  return fixed, actor, h, latent, EPISODES.copy()


class ActionInformationStatsTest(unittest.TestCase):

  def test_mean_ridge_objective_and_unregularized_intercept(self):
    h = np.array([[-2.], [2.]], np.float64)
    z = np.full((2, 2), 8., np.float64)
    y = np.array([[2., 8.], [4., 12.]], np.float64)
    prediction, metadata = stats.ridge_predict(h, z, y, h.copy(), z.copy())
    # Normalized h is +/-1. With mean loss and lambda1 each slope is halved;
    # sum loss with lambda1 would instead retain 2/3 of the slope.
    np.testing.assert_allclose(prediction, [[2.5, 9.], [3.5, 11.]], atol=1e-14)
    self.assertFalse(np.allclose(prediction, [[3 - 2/3, 10 - 4/3],
                                             [3 + 2/3, 10 + 4/3]]))
    self.assertEqual(metadata['gram_diagonal_regularizer'], 2.)
    self.assertEqual(metadata['train_column_mean'], [3., 10.])
    self.assertFalse(metadata['intercept_regularized'])
    self.assertEqual(metadata['z_group']['zero_rms'], True)
    self.assertAlmostEqual(metadata['h_group']['rms'], 2.)
    self.assertAlmostEqual(metadata['regularized_objective'], 2.5)

  def test_group_total_energy_and_train_only_preprocessing(self):
    h = np.array([[-1., -1.], [1., 1.]], np.float64)
    z = np.array([[-2.], [2.]], np.float64)
    y = np.array([[-1.], [1.]], np.float64)
    p1, m1 = stats.ridge_predict(h, z, y, np.array([[0., 0.]]), np.array([[0.]]))
    p2, m2 = stats.ridge_predict(h, z, y, np.array([[1000., 1000.]]), np.array([[2000.]]))
    self.assertEqual(m1, m2)
    self.assertAlmostEqual(m1['h_group']['rms'], np.sqrt(2))
    for group in ('h_group', 'z_group'):
      self.assertAlmostEqual(m1[group]['train_mean_total_squared_norm'], 1.)
    self.assertAlmostEqual(float(p1[0, 0]), 0.)
    self.assertGreater(float(p2[0, 0]), 100.)

  def test_zero_group_out_of_train_range_is_zero_without_dropping_rows(self):
    h, z = np.ones((3, 2), np.float64), np.full((3, 1), 7., np.float64)
    targets = np.array([[2.], [4.], [6.]], np.float64)
    prediction, metadata = stats.ridge_predict(
        h, z, targets, np.array([[1e9, -1e9]]), np.array([[-1e9]]))
    np.testing.assert_array_equal(prediction, [[4.]])
    self.assertEqual(metadata['zero_rms_groups'], 2)
    self.assertEqual(metadata['training_rows'], 3)
    self.assertEqual(metadata['squared_weight_norm'], 0.)

  def test_closed_delta_boundary_is_exact_and_separate_from_tie(self):
    rewards = np.zeros((8, 4, 7), np.float64)
    rewards[..., 0] = .2
    rewards[..., 2] = -1.
    boundary = stats._external(rewards, EPISODES)
    self.assertTrue(boundary['delta_optimal_membership'][0][0][1])
    self.assertFalse(boundary['delta_optimal_membership'][0][0][2])
    changed = rewards.copy()
    changed[..., 1] = -1e-11  # well within ranking tie tolerance, outside delta.
    outside = stats._external(changed, EPISODES)
    self.assertFalse(outside['delta_optimal_membership'][0][0][1])
    spans = np.zeros_like(rewards)
    spans[..., 0] = .2
    self.assertEqual(stats._external(spans, EPISODES)['sensitive_episode_count'], 8)
    spans[..., 0] = np.nextafter(.2, -np.inf)
    self.assertEqual(stats._external(spans, EPISODES)['sensitive_episode_count'], 0)

  def test_tiny_winner_change_is_not_external_key_evidence(self):
    rewards = np.zeros((8, 4, 7), np.float64)
    rewards[..., :2] = 1.
    rewards[::2, :, 0] += 1e-9
    rewards[1::2, :, 1] += 1e-9
    external = stats._external(rewards, EPISODES)
    self.assertEqual(external['sensitive_episode_count'], 8)
    self.assertEqual(set(np.asarray(external['display_winner']).ravel()), {0, 1})
    self.assertEqual(external['disjoint_pair_count'], 0)
    self.assertFalse(external['sufficient'])

  def test_disjoint_sets_must_be_across_episodes_and_have_four_episode_coverage(self):
    rewards = np.zeros((8, 4, 7), np.float64)
    # Different near-optimal sets only inside episode0; other episodes include
    # all controls in their near-optimal sets, so no cross-episode witness.
    rewards[0, :2, 0] = 1.
    rewards[0, 2:, 1] = 1.
    external = stats._external(rewards, EPISODES)
    self.assertFalse(external['cross_episode_disjoint_set_exists'])
    rewards[1, :, 1] = 1.
    external = stats._external(rewards, EPISODES)
    self.assertTrue(external['cross_episode_disjoint_set_exists'])
    self.assertFalse(external['sensitivity_coverage_sufficient'])
    rewards[2:4, :, 0] = 1.
    external = stats._external(rewards, EPISODES)
    self.assertTrue(external['sufficient'])
    self.assertTrue(all(p['episode_a'] != p['episode_b']
                        for p in external['cross_episode_disjoint_pairs']))

  def test_loeo_real_sample_interface_selection_guard_and_json(self):
    arrays = fixture()
    before = [x.copy() for x in arrays]
    result = stats.analyze(*arrays)
    self.assertEqual(result['status'], 'interface_increment_exploratory_supported')
    self.assertEqual(result['decision'], 'may_design_finite_normal_loss_compression_mechanism')
    self.assertTrue(result['external_control']['sufficient'])
    readout = result['readout']
    self.assertEqual(readout['independent_episode_units'], 8)
    self.assertEqual(readout['states'], 32)
    self.assertEqual(readout['rows'], 128)
    for fold in readout['folds']:
      self.assertNotIn(fold['heldout_episode'], fold['training_episodes'])
      self.assertEqual(len(fold['training_episodes']), 7)
      self.assertEqual(fold['training_rows'], 112)
      self.assertEqual(fold['heldout_rows'], 16)
      self.assertEqual(fold['fits']['q']['feature_dimensions'], 5)
      self.assertEqual(fold['fits']['p']['feature_dimensions'], 5)
      self.assertEqual(fold['fits']['q']['gram_diagonal_regularizer'], 112.)
    self.assertAlmostEqual(readout['interfaces']['q']['primary_selected_reward'], 1.)
    self.assertAlmostEqual(readout['interfaces']['p']['primary_selected_reward'], .5)
    self.assertAlmostEqual(readout['interfaces']['q']['primary_finite_library_regret'], 0.)
    self.assertAlmostEqual(readout['comparisons']['q_minus_p']['lower95'], .5)
    self.assertAlmostEqual(readout['comparisons']['q_minus_constant']['lower95'], .5)
    self.assertEqual(result['actor']['status'], 'actor_first_action_exploratory_not_supported')
    for key in ('training_authorized', 'proceed_short_training', 'eligible_method_training',
                'method_effectiveness_established', 'blind_arrays_read',
                'current_rgb_increment_over_complete_history_established'):
      self.assertIs(result[key], False)
    json.dumps(result, allow_nan=False)
    for original, after in zip(before, arrays):
      np.testing.assert_array_equal(original, after)
    self.assertEqual(result, stats.analyze(*arrays))

  def test_heldout_targets_never_enter_fold_fit_or_constant(self):
    arrays = fixture()
    original = stats.analyze(*arrays)
    altered = [x.copy() for x in arrays]
    altered[0][0] += np.arange(7, dtype=np.float64)[None] * 100.
    changed = stats.analyze(*altered)
    self.assertEqual(original['readout']['folds'][0], changed['readout']['folds'][0])
    for branch in ('q', 'p'):
      np.testing.assert_array_equal(
          original['readout']['interfaces'][branch]['predicted_centered_reward'][0],
          changed['readout']['interfaces'][branch]['predicted_centered_reward'][0])
    self.assertNotEqual(original['readout']['folds'][1], changed['readout']['folds'][1])

  def test_sample_specific_selection_precedes_rng_average(self):
    fixed, actor, h, latent, episodes = fixture()
    # Three q samples identify the state correctly; the fourth reverses it.
    latent[:, :, 3, 0] = latent[:, :, 3, 0, :, ::-1]
    result = stats.analyze(fixed, actor, h, latent, episodes)
    selected = np.asarray(result['readout']['interfaces']['q']['selected_reward'])
    np.testing.assert_array_equal(selected[..., :3], np.ones((8, 4, 3)))
    np.testing.assert_array_equal(selected[..., 3], np.zeros((8, 4)))
    self.assertAlmostEqual(result['readout']['interfaces']['q']['primary_selected_reward'], .75)
    self.assertAlmostEqual(result['readout']['comparisons']['q_minus_p']['primary'], .25)
    self.assertEqual(result['readout']['interfaces']['q']['choices'][0][0], [0, 0, 0, 1])

  def test_rng_rows_are_not_bootstrap_episode_units(self):
    fixed, actor, h, latent, episodes = fixture()
    ep_signed = np.array([-4., -2., -1., 0., 1., 2., 3., 7.], np.float64)
    centered_rng = np.array([-100., -50., 50., 100.], np.float64)
    actor[..., 0] = ep_signed[:, None, None] + centered_rng[None, None, :]
    actor[..., 1] = 0.
    result = stats.analyze(fixed, actor, h, latent, episodes)
    draws = np.random.default_rng(20261021).integers(0, 8, (2000, 8))
    expected = np.quantile(ep_signed[draws].mean(1), [.025, .975], method='linear')
    interval = result['actor']['episode_interval']
    np.testing.assert_allclose([interval['lower95'], interval['upper95']], expected, atol=0)
    self.assertEqual(interval['episode_values'], ep_signed.tolist())
    self.assertEqual(result['bootstrap_draws'], 2000)
    self.assertEqual(result['actor']['status'], 'actor_first_action_exploratory_inconclusive')
    row_draws = np.random.default_rng(20261021).integers(0, 128, (2000, 128))
    wrong = np.quantile((actor[..., 0] - actor[..., 1]).ravel()[row_draws].mean(1),
                        [.025, .975], method='linear')
    self.assertGreater(np.max(np.abs(wrong - expected)), 1.)

  def test_identity_and_all_constant_degeneracy_cannot_be_supported(self):
    fixed, actor, h, latent, episodes = fixture()
    latent[..., 1, :, :] = latent[..., 0, :, :]
    result = stats.analyze(fixed, actor, h, latent, episodes)
    self.assertEqual(result['status'], 'interface_increment_exploratory_not_supported')
    self.assertEqual(result['readout']['comparisons']['q_minus_p']['upper95'], 0.)
    fixed.fill(2.)
    h.fill(1.)
    latent.fill(0.)
    constant = stats.analyze(fixed, actor, h, latent, episodes)
    self.assertEqual(constant['readout']['total_zero_rms_groups'], 32)
    self.assertFalse(constant['external_control']['sufficient'])
    self.assertEqual(constant['decision'], 'stop_finite_control_object_external_basis_insufficient')
    for interface in ('q', 'p'):
      self.assertEqual(constant['readout']['interfaces'][interface]['primary_selected_reward'], 2.)
      self.assertTrue(np.all(np.asarray(constant['readout']['interfaces'][interface]['choices']) == 0))

  def test_positive_readout_does_not_bypass_external_margin_gate(self):
    fixed, actor, h, latent, episodes = fixture()
    fixed *= .1
    result = stats.analyze(fixed, actor, h, latent, episodes)
    self.assertTrue(result['readout']['both_lower95_positive'])
    self.assertFalse(result['external_control']['sufficient'])
    self.assertNotEqual(result['status'], 'interface_increment_exploratory_supported')
    self.assertEqual(result['decision'], 'stop_finite_control_object_external_basis_insufficient')
    self.assertFalse(result['training_authorized'])

  def test_tie_rule_smallest_index_without_delta_relaxation(self):
    values = np.array([[1., 1. + 5e-11, 0.], [1., 1. + 2e-10, 0.]], np.float64)
    np.testing.assert_array_equal(stats._argmax(values), [0, 1])
    np.testing.assert_array_equal(stats._rankings(values), [[0, 1, 2], [1, 0, 2]])

  def test_reject_nonfinite_dtype_axis_or_episode_mismatch(self):
    for index in range(4):
      for bad in (np.nan, np.inf):
        with self.subTest(index=index, bad=bad):
          arrays = list(fixture())
          arrays[index].flat[0] = bad
          with self.assertRaises(ValueError):
            stats.analyze(*arrays)
    for index in range(5):
      with self.subTest(dtype=index):
        arrays = list(fixture())
        arrays[index] = arrays[index].astype(np.float32 if index < 4 else np.int64)
        with self.assertRaises(ValueError):
          stats.analyze(*arrays)
    arrays = list(fixture())
    arrays[3] = arrays[3].transpose(0, 1, 3, 2, 4, 5)
    with self.assertRaises(ValueError):
      stats.analyze(*arrays)
    arrays = list(fixture())
    arrays[4][0] = arrays[4][1]
    with self.assertRaises(ValueError):
      stats.analyze(*arrays)
    arrays = list(fixture())
    arrays[2] = np.zeros((8, 4, 0), np.float64)
    with self.assertRaises(ValueError):
      stats.analyze(*arrays)


if __name__ == '__main__':
  unittest.main()
