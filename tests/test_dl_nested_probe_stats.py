"""Small numerical/provenance-free fixtures, not a real nested-probe execution.

No artifacts, blind arrays, Dreamer, renderer, simulator or GPU are used.
The full synthetic assay is not rerun. Original lambda1 is a fixture oracle only.
"""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
  os.environ[name] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_action_information_stats as original
from scripts import dl_nested_probe_stats as stats


def fixture():
  # Four tiny groups, two states, two MC samples. No natural task labels.
  rewards = np.zeros((4, 2, 7), np.float64)
  rewards[:, 0, 0], rewards[:, 1, 1] = 1., 1.
  h = np.zeros((4, 2, 3), np.float64)
  latent = np.zeros((4, 2, 2, 2, 1, 2), np.float64)
  latent[:, 0, :, 0, 0, 0] = 1.
  latent[:, 1, :, 0, 0, 1] = 1.
  latent[:, :, :, 1, 0, 0] = 1.
  # Small deterministic cross-episode variation in h, shared by interfaces.
  h[:, :, 0] = np.arange(4, dtype=np.float64)[:, None] / 5
  return rewards, h, latent, np.array([9, 14, 20, 31], np.int32)


def candidate(rule, utility, mse):
  return {'lambda': rule, 'pooled_utility': float(utility), 'pooled_mse': float(mse)}


class NestedProbeStatsTest(unittest.TestCase):

  def test_lambda1_parity_with_original_mean_ridge(self):
    h = np.array([[-2., 1.], [-1., 0.], [0., 2.], [1., -2.], [2., 3.]], np.float64)
    z = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                  [1., 0., 0.], [0., 1., 0.]], np.float64)
    target = np.arange(35, dtype=np.float64).reshape((5, 7)) / 10
    he, ze = h[[0, 2, 4]].copy(), z[[0, 2, 4]].copy()
    expected, old = original.ridge_predict(h, z, target, he, ze)
    prediction, new = stats.ridge_predict(h, z, target, he, ze)
    np.testing.assert_allclose(prediction, expected, rtol=0, atol=1e-10)
    for group in ('h_group', 'z_group'):
      self.assertEqual(new[group], old[group])
    for key in ('train_column_mean', 'training_mean_sum_squared_error',
                'training_mse_per_control', 'squared_weight_norm', 'regularized_objective'):
      np.testing.assert_allclose(new[key], old[key], rtol=0, atol=1e-10)
    self.assertEqual(new['gram_diagonal_regularizer'], 5.)
    self.assertFalse(new['intercept_regularized'])
    self.assertTrue(new['eigen']['all_eigenvalues_retained'])
    self.assertFalse(new['eigen']['cutoff_applied'])

  def test_mean_ridge_analytic_solution_and_global_row_replication(self):
    h = np.array([[-2.], [2.]], np.float64)
    z = np.full((2, 2), 8., np.float64)
    target = np.array([[2., 8.], [4., 12.]], np.float64)
    prediction, fit = stats.ridge_predict(h, z, target, h, z)
    np.testing.assert_allclose(prediction, [[2.5, 9.], [3.5, 11.]], rtol=0, atol=1e-13)
    self.assertAlmostEqual(fit['regularized_objective'], 2.5, places=12)
    for rule in stats.LAMBDAS:
      base, _ = stats.ridge_predict(h, z, target, h, z, rule)
      doubled, meta = stats.ridge_predict(np.repeat(h, 3, 0), np.repeat(z, 3, 0),
          np.repeat(target, 3, 0), h, z, rule)
      np.testing.assert_allclose(base, doubled, rtol=0, atol=1e-10)
      self.assertEqual(meta['gram_diagonal_regularizer'], 6 * rule)

  def test_global_tie_sets_reject_nontransitive_chain_and_iteration_order(self):
    # Middle candidate is within the global maximum; lambda1 is within middle
    # but outside the global maximum. Pairwise chaining would admit lambda1.
    values = [candidate(1., -1.5e-10, 0.), candidate(.1, -.75e-10, 0.),
              candidate(.01, 0., 0.), candidate(.001, -1., 0.)]
    selected = stats._select_common_lambda(values)
    self.assertEqual(selected['selected_lambda'], .1)
    self.assertEqual(selected['utility_tie_candidates'], [.1, .01])
    self.assertEqual(stats._select_common_lambda(list(reversed(values))), selected)
    # Repeat nontransitive counterexample at the second, globally defined stage.
    values = [candidate(1., 0., 1.5e-10), candidate(.1, 0., .75e-10),
              candidate(.01, 0., 0.), candidate(.001, 0., 1.)]
    selected = stats._select_common_lambda(values)
    self.assertEqual(selected['selected_lambda'], .1)
    self.assertEqual(selected['mse_tie_candidates'], [.1, .01])
    tied = [candidate(rule, 2., .3) for rule in stats.LAMBDAS]
    self.assertEqual(stats._select_common_lambda(tied)['selected_lambda'], 1.)

  def test_outer_episode_excluded_from_every_selection_training_fold(self):
    inputs = fixture()
    result = stats.nested_readout(*inputs)
    episodes = inputs[-1].tolist()
    self.assertEqual(result['eigendecompositions'], 32)
    self.assertEqual(result['lambda_evaluations'], 104)
    for fold in result['outer_folds']:
      outer = fold['heldout_episode']
      self.assertEqual(len(fold['training_episodes']), 3)
      self.assertNotIn(outer, fold['training_episodes'])
      self.assertEqual(fold['training_rows'], 12)
      self.assertEqual(fold['heldout_rows'], 4)
      self.assertFalse(fold['outer_candidate_grid_evaluated'])
      self.assertEqual(len(fold['inner_folds']), 3)
      for inner in fold['inner_folds']:
        self.assertNotIn(outer, inner['training_episodes'])
        self.assertNotEqual(inner['validation_episode'], outer)
        self.assertNotIn(inner['validation_episode'], inner['training_episodes'])
        self.assertEqual(len(inner['training_episodes']), 2)
        self.assertEqual(inner['training_rows'], 8)
        self.assertEqual(inner['validation_rows'], 4)
        self.assertEqual(len(inner['candidates']), 4)
        for label in ('q', 'p'):
          fit = fold['outer_fits'][label]
          self.assertEqual(fit['regularization_lambda'], fold['selected_lambda'])
          self.assertEqual(fit['training_rows'], 12)
        self.assertEqual(sorted([outer] + fold['training_episodes']), sorted(episodes))

  def test_outer_target_mutation_does_not_select_lambda_or_preprocess(self):
    inputs = fixture()
    first = stats.nested_readout(*inputs)['outer_folds'][0]
    rewards = inputs[0].copy()
    rewards[0] = np.arange(14, dtype=np.float64).reshape((2, 7)) * 1000
    second = stats.nested_readout(rewards, *inputs[1:])['outer_folds'][0]
    for key in ('inner_folds', 'candidate_scores', 'common_lambda_selection',
                'selected_lambda', 'outer_fits', 'constant_control', 'constant_train_mean_rewards'):
      self.assertEqual(first[key], second[key])
    for branch in ('q', 'p'):
      self.assertEqual(first['outer_interfaces'][branch]['predicted_centered_reward'],
                       second['outer_interfaces'][branch]['predicted_centered_reward'])

  def test_outer_feature_mutation_does_not_select_or_fit_preprocessing(self):
    rewards, h, latent, episodes = fixture()
    first = stats.nested_readout(rewards, h, latent, episodes)['outer_folds'][0]
    changed_h, changed_z = h.copy(), latent.copy()
    changed_h[0] += 1000
    changed_z[0] += 200
    second = stats.nested_readout(rewards, changed_h, changed_z, episodes)['outer_folds'][0]
    for key in ('inner_folds', 'candidate_scores', 'common_lambda_selection', 'selected_lambda'):
      self.assertEqual(first[key], second[key])
    for branch in ('q', 'p'):
      # Output predictions legitimately change; training quantities must not.
      for key in ('h_group', 'z_group', 'train_column_mean', 'eigen',
                  'regularization_lambda', 'training_mean_sum_squared_error', 'squared_weight_norm'):
        self.assertEqual(first['outer_fits'][branch][key], second['outer_fits'][branch][key])

  def test_branch_swap_preserves_common_lambda_and_exchanges_outer_outputs(self):
    rewards, h, latent, episodes = fixture()
    first = stats.nested_readout(rewards, h, latent, episodes)
    second = stats.nested_readout(rewards, h, latent[:, :, :, ::-1].copy(), episodes)
    self.assertEqual(first['selected_common_lambdas'], second['selected_common_lambdas'])
    self.assertEqual(first['interfaces']['q'], second['interfaces']['p'])
    self.assertEqual(first['interfaces']['p'], second['interfaces']['q'])
    for left, right in zip(first['outer_folds'], second['outer_folds']):
      self.assertEqual(left['common_lambda_selection'], right['common_lambda_selection'])

  def test_copying_all_mc_rows_preserves_selected_procedure_and_predictions(self):
    rewards, h, latent, episodes = fixture()
    first = stats.nested_readout(rewards, h, latent, episodes)
    second = stats.nested_readout(rewards, h, np.repeat(latent, 2, 2), episodes)
    self.assertEqual(first['selected_common_lambdas'], second['selected_common_lambdas'])
    for branch in ('q', 'p'):
      left, right = first['interfaces'][branch], second['interfaces'][branch]
      prediction = np.asarray(left['predicted_centered_reward'])
      np.testing.assert_allclose(np.repeat(prediction, 2, 2),
          right['predicted_centered_reward'], rtol=0, atol=1e-10)
      np.testing.assert_array_equal(np.repeat(np.asarray(left['choices']), 2, 2), right['choices'])
      self.assertEqual(left['episode_selected_reward'], right['episode_selected_reward'])

  def test_within_state_control_centering_removes_common_reward_offsets(self):
    rewards, h, latent, episodes = fixture()
    first = stats.nested_readout(rewards, h, latent, episodes)
    offset = np.arange(8, dtype=np.float64).reshape((4, 2, 1)) * 10
    second = stats.nested_readout(rewards + offset, h, latent, episodes)
    self.assertEqual(first['selected_common_lambdas'], second['selected_common_lambdas'])
    for branch in ('q', 'p'):
      np.testing.assert_allclose(first['interfaces'][branch]['predicted_centered_reward'],
          second['interfaces'][branch]['predicted_centered_reward'], rtol=0, atol=1e-10)
      self.assertEqual(first['interfaces'][branch]['choices'], second['interfaces'][branch]['choices'])
    target = rewards - rewards.mean(-1, keepdims=True)
    for fold in first['outer_folds']:
      training = [episodes.tolist().index(ep) for ep in fold['training_episodes']]
      expected = target[training].mean((0, 1))
      for branch in ('q', 'p'):
        np.testing.assert_allclose(fold['outer_fits'][branch]['train_column_mean'],
                                  expected, rtol=0, atol=1e-10)

  def test_inner_candidate_scores_pool_episode_state_mc_and_both_branches_equally(self):
    rewards, h, latent, episodes = fixture()
    # Unequal reward magnitude rules out silently counting high-reward states
    # or branches as extra units. Each state's two samples still weigh equally.
    rewards *= np.array([1., 3.])[None, :, None]
    result = stats.nested_readout(rewards, h, latent, episodes)
    for fold in result['outer_folds']:
      for li, aggregate in enumerate(fold['candidate_scores']):
        utility, mse = [], []
        for inner in fold['inner_folds']:
          case = inner['candidates'][li]
          self.assertEqual(case['lambda_value'], aggregate['lambda'])
          utility.append([np.asarray(case['interfaces'][branch]['selected_reward']).mean()
                          for branch in ('q', 'p')])
          mse.append([np.asarray(case['interfaces'][branch]['sample_mse_per_control']).mean()
                      for branch in ('q', 'p')])
        self.assertAlmostEqual(aggregate['pooled_utility'], float(np.mean(utility)), places=13)
        self.assertAlmostEqual(aggregate['pooled_mse'], float(np.mean(mse)), places=13)
      self.assertEqual(fold['common_lambda_selection']['branch_weights'], [.5, .5])

  def test_strict_zero_group_discards_heldout_variation_without_dropping_rows(self):
    h, z = np.ones((3, 2), np.float64), np.full((3, 1), 7., np.float64)
    target = np.array([[2.], [4.], [6.]], np.float64)
    for rule in stats.LAMBDAS:
      prediction, fit = stats.ridge_predict(h, z, target,
          np.array([[1e9, -1e9]], np.float64), np.array([[-1e9]], np.float64), rule)
      np.testing.assert_array_equal(prediction, [[4.]])
      self.assertEqual(fit['zero_rms_groups'], 2)
      self.assertEqual(fit['training_rows'], 3)
      self.assertEqual(fit['squared_weight_norm'], 0.)
    tiny = np.array([[-1e-200], [1e-200]], np.float64)
    with self.assertRaisesRegex(ValueError, 'underflowed'):
      stats.ridge_predict(tiny, np.ones((2, 1)), np.ones((2, 1)), tiny, np.ones((2, 1)))

  def test_nonfinite_shape_dtype_episode_and_grid_rejection(self):
    baseline = fixture()
    for index in range(3):
      for bad in (np.nan, np.inf):
        values = [value.copy() for value in baseline]
        values[index].flat[0] = bad
        with self.subTest(array=index, bad=bad), self.assertRaises(ValueError):
          stats.nested_readout(*values)
    values = list(baseline)
    values[1] = values[1].astype(np.float32)
    with self.assertRaises(ValueError):
      stats.nested_readout(*values)
    values = list(baseline)
    values[2] = values[2][:, :, :, :1]
    with self.assertRaises(ValueError):
      stats.nested_readout(*values)
    values = list(baseline)
    values[-1] = np.array([9, 9, 20, 31], np.int32)
    with self.assertRaises(ValueError):
      stats.nested_readout(*values)
    for rule in (0., -1., float('nan'), .02, 1):
      with self.subTest(rule=rule), self.assertRaises(ValueError):
        stats.ridge_predict(np.ones((2, 1)), np.ones((2, 1)), np.ones((2, 1)),
                            np.ones((1, 1)), np.ones((1, 1)), rule)

  def test_eigensolver_psd_and_reconstruction_fail_closed(self):
    h, z, y = np.array([[-1.], [1.]]), np.ones((2, 1)), np.array([[-1.], [1.]])
    with mock.patch.object(np.linalg, 'eigh', return_value=(np.array([-.1, 1.]), np.eye(2))):
      with self.assertRaisesRegex(ValueError, 'PSD'):
        stats.ridge_predict(h, z, y, h, z)
    with mock.patch.object(np.linalg, 'eigh', return_value=(np.array([0., 1.]), np.eye(2))):
      with self.assertRaisesRegex(ValueError, 'reconstruction'):
        stats.ridge_predict(h, z, y, h, z)

  def test_input_immutable_json_and_whole_episode_bootstrap_boundary(self):
    inputs = fixture()
    copies = [value.copy() for value in inputs]
    result = stats.nested_readout(*inputs)
    for before, after in zip(copies, inputs):
      self.assertEqual(before.dtype, after.dtype)
      self.assertEqual(before.shape, after.shape)
      self.assertEqual(before.tobytes(), after.tobytes())
    json.dumps(result, allow_nan=False)
    q = np.asarray(result['interfaces']['q']['episode_selected_reward'])
    p = np.asarray(result['interfaces']['p']['episode_selected_reward'])
    values = q - p
    draws = np.random.default_rng(20261021).integers(0, 4, (2000, 4))
    low, high = np.quantile(values[draws].mean(1), [.025, .975], method='linear')
    interval = result['comparisons']['q_minus_p']
    self.assertEqual(interval['episode_values'], values.tolist())
    self.assertEqual(interval['lower95'], float(low))
    self.assertEqual(interval['upper95'], float(high))
    for key in ('training_authorized', 'proceed_short_training', 'eligible_method_training',
                'real_gate_authority', 'method_effectiveness_established',
                'source_provenance_verified_by_stats', 'blind_arrays_read',
                'equal_effective_degrees_of_freedom_claimed'):
      self.assertIs(result[key], False)
    self.assertIn('not the entire', result['negative_scope'])


if __name__ == '__main__':
  unittest.main()
