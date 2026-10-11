"""Synthetic finite NumPy fixtures only; no real reset/blind/model execution."""
import copy
import json
import os
from pathlib import Path
import sys
import unittest

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
  os.environ[name] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_reset_readout as stats


def fixture():
  def make(ids):
    ids = np.asarray(ids, np.int32)
    n = len(ids)
    cue = np.arange(n) % 2
    q = np.zeros((n, 4, 32, 32), np.float64)
    p = np.zeros_like(q)
    q[..., 0] = 1.
    p[..., 0] = 1.
    q[:, :, 0] = 0.
    p[:, :, 0] = 0.
    for i in range(n):
      q[i, :, 0, cue[i]] = 1.
    p[:, :, 0, 2] = 1.
    returns = np.zeros((n, 7), np.float64)
    returns[np.arange(n), cue] = 1.
    return dict(episode_ids=ids, h=np.zeros((n, 3), np.float64),
        zq=q, zp=p, returns=returns,
        D=(.1 + np.arange(n, dtype=np.float64) / 100.),
        K=(1. + cue.astype(np.float64) / 5.))
  cal = make(stats.CALIBRATION_EPISODES)
  blind = make([i for i in range(32) if i not in stats.CALIBRATION_EPISODES])
  return cal, blind


def joint(frozen, cal, blind):
  fit = stats.refit_calibration(cal, lambda_value=frozen['selected_lambda'])
  return stats.evaluate_joint(frozen, {'B': copy.deepcopy(fit), 'R': fit},
                              {name: copy.deepcopy(blind) for name in ('0', 'B', 'R')})


def force_model(fit, *, prior_action=0, q_constant=False):
  """Analytic shared F for checking R-B interpretation, not scientific fitting."""
  fit = copy.deepcopy(fit)
  model = fit['model']
  hd = len(model['h_mean'])
  model.update(h_mean=[0.] * hd, z_mean=[0.] * 1024, h_rms=0., z_rms=1.,
               intercept=[0.] * 7)
  weights = np.zeros((hd + 1024, 7), np.float64)
  weights[hd + 0, 0] = 1.
  weights[hd + 1, 0 if q_constant else 1] = 1.
  weights[hd + 2, prior_action] = 1.
  model['weights'] = weights.tolist()
  return fit


class ResetReadoutTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.cal, cls.blind = fixture()
    cls.frozen = stats.fit_calibration(cls.cal)

  def test_shared_pooled_fit_matches_independent_primal_normal_equation(self):
    cal = self.cal
    model = stats._fit_shared(cal, np.arange(8), .1)
    h, z, y = stats._pooled_rows(cal, np.arange(8))
    # A dual direct solve provides an independent oracle for the spectral fit.
    def normalize(v):
      c = v - v.mean(0)
      r = np.sqrt(np.mean(np.sum(c * c, axis=1)))
      return c / r if r else np.zeros_like(c)
    x = np.concatenate((normalize(h), normalize(z)), axis=1)
    bias = y.mean(0)
    dual = np.linalg.solve(x @ x.T + len(x) * .1 * np.eye(len(x)), y - bias)
    expected = x.T @ dual
    np.testing.assert_allclose(model['weights'], expected, rtol=0, atol=1e-10)
    np.testing.assert_allclose(model['intercept'], bias, rtol=0, atol=1e-14)
    self.assertEqual(model['metadata']['training_rows'], 64)
    self.assertTrue(model['metadata']['parameters_shared'])
    self.assertFalse(model['metadata']['branch_id_input'])

  def test_same_q_p_inputs_have_exact_same_prediction_signal_and_utility(self):
    batch = copy.deepcopy(self.blind)
    batch['zq'] = batch['zp'].copy()
    prediction, samples = stats._predict_shared(self.frozen['fit']['model'], batch)
    np.testing.assert_array_equal(samples[:, 0], samples[:, 1])
    decision = stats._decision(prediction, batch['returns'])
    np.testing.assert_array_equal(decision['choices'][:, 0], decision['choices'][:, 1])
    np.testing.assert_array_equal(decision['S'], np.zeros(24))
    np.testing.assert_array_equal(decision['U'], np.zeros(24))

  def test_mc_mean_precedes_argmax_and_differs_from_sample_majority(self):
    fit = force_model(self.frozen['fit'])
    model = fit['model']
    weights = np.asarray(model['weights'])
    weights[:] = 0.
    weights[3 + 0, 0] = 1.
    weights[3 + 1, 1] = 10.
    model['weights'] = weights.tolist()
    batch = copy.deepcopy(self.blind)
    batch['zq'][:, :, 0] = 0.
    batch['zq'][:, :3, 0, 0] = 1.
    batch['zq'][:, 3, 0, 1] = 1.
    prediction, samples = stats._predict_shared(model, batch)
    self.assertTrue(np.all(np.argmax(samples[:, 0], -1)[:, :3] == 0))
    self.assertTrue(np.all(np.argmax(prediction[:, 0], -1) == 1))
    np.testing.assert_array_equal(prediction[:, 0, :2], np.tile([.75, 2.5], (24, 1)))

  def test_exact_choice_tie_does_not_apply_lambda_near_tie_rule(self):
    prediction = np.zeros((2, 2, 7), np.float64)
    prediction[0, 0, 1] = 5e-11
    prediction[1, 0, :2] = 1.
    decision = stats._decision(prediction, np.zeros((2, 7), np.float64))
    self.assertEqual(decision['choices'][:, 0].tolist(), [1, 0])
    self.assertEqual(decision['S'].tolist(), [5e-11, 0.])

  def test_loeo_whole_episode_and_shared_global_lambda_selection(self):
    result = self.frozen
    self.assertEqual(len(result['fold_metadata']), 8)
    for fold in result['fold_metadata']:
      self.assertNotIn(fold['heldout_episode'], fold['training_episodes'])
      self.assertEqual(len(fold['training_episodes']), 7)
      self.assertEqual(len(fold['candidates']), 4)
      for candidate in fold['candidates']:
        meta = candidate['fit_metadata']
        self.assertEqual(meta['training_episodes'], fold['training_episodes'])
        self.assertEqual(meta['training_rows'], 56)
        self.assertEqual(np.asarray(candidate['mean_prediction']).shape, (2, 7))
        np.testing.assert_array_equal(np.asarray(candidate['mean_prediction']),
            np.asarray(candidate['sample_predictions']).mean(axis=1, dtype=np.float64))
    self.assertIn(result['selected_lambda'], stats.LAMBDAS)
    self.assertFalse(result['oof']['fully_nested_crossfit_claimed'])

  def test_global_lambda_ties_are_order_independent(self):
    candidates = [dict(**{'lambda': rule}, pooled_utility=0., pooled_mse=1.)
                  for rule in stats.LAMBDAS]
    self.assertEqual(stats._lambda_selection(candidates)['selected_lambda'], 1.)
    self.assertEqual(stats._lambda_selection(candidates),
                     stats._lambda_selection(list(reversed(candidates))))
    candidates[0]['pooled_utility'] = -1.5e-10
    candidates[1]['pooled_utility'] = -.75e-10
    self.assertEqual(stats._lambda_selection(candidates)['selected_lambda'], .1)

  def test_state_additive_return_offsets_preserve_choices_scores_and_U(self):
    cal, blind = fixture()
    shifted_cal, shifted_blind = copy.deepcopy(cal), copy.deepcopy(blind)
    shifted_cal['returns'] += np.arange(8, dtype=np.float64)[:, None] * 4.
    shifted_blind['returns'] += np.arange(24, dtype=np.float64)[:, None] * 4.
    shifted = stats.fit_calibration(shifted_cal)
    self.assertEqual(shifted['selected_lambda'], self.frozen['selected_lambda'])
    for key in ('S', 'U', 'choices'):
      np.testing.assert_allclose(shifted['oof'][key], self.frozen['oof'][key], rtol=0, atol=1e-10)
    baseline_result = joint(self.frozen, cal, blind)
    shifted_result = joint(shifted, shifted_cal, shifted_blind)
    for key in ('choices', 'S_decision', 'true_U'):
      np.testing.assert_allclose(shifted_result['models']['0'][key],
                                 baseline_result['models']['0'][key], rtol=0, atol=1e-10)
    for key in ('q_minus_p', 'q_minus_constant', 'q_centered_mse'):
      np.testing.assert_allclose(shifted_result['models']['0']['metrics'][key]['values'],
          baseline_result['models']['0']['metrics'][key]['values'], rtol=0, atol=1e-10)

  def test_high_predicted_S_can_support_wrong_q_with_negative_true_U(self):
    blind = copy.deepcopy(self.blind)
    prediction, _ = stats._predict_shared(self.frozen['fit']['model'], blind)
    decision = stats._decision(prediction, blind['returns'])
    # The symmetric artificial prior has a numerical near-tie; do not assume
    # which branch-independent action an eigensolver chooses. Assign synthetic
    # physics truth to foil the actual F choices, without changing q/p/D/K.
    blind['returns'][:] = 0.
    rows = np.arange(24)
    blind['returns'][rows, decision['choices'][:, 1]] = 1.
    disagreements = decision['choices'][:, 0] != decision['choices'][:, 1]
    blind['returns'][rows[disagreements], decision['choices'][disagreements, 0]] = 0.
    result = joint(self.frozen, self.cal, blind)
    model = result['models']['0']
    wrong = (np.asarray(model['S_decision']) >= .2) & (np.asarray(model['true_U']) < 0.)
    self.assertTrue(wrong.any())
    self.assertEqual(result['discrimination']['false_positive_episodes'],
                     blind['episode_ids'][wrong].tolist())
    self.assertFalse(model['finite_readout_opportunity_supported'])

  def test_identity_negatives_cannot_satisfy_wrong_choice_coverage(self):
    result = joint(self.frozen, self.cal, self.blind)
    coverage = result['discrimination']['coverage']['all']
    self.assertGreaterEqual(coverage['practical_positive'], 4)
    self.assertGreaterEqual(coverage['nonpositive'], 4)
    self.assertEqual(coverage['wrong_choice_negative'], 0)
    self.assertFalse(coverage['minimum_coverage_met'])
    self.assertTrue(result['models']['0']['finite_readout_opportunity_supported'])
    self.assertFalse(result['discrimination']['increments_over_D_only']['D+S'][
        'eligible_covered_overall_increment'])

  def test_high_D_threshold_is_calibration_only_and_ties_included(self):
    self.assertEqual(self.frozen['high_D_threshold'], np.median(self.cal['D']))
    decision = dict(U=np.r_[np.ones(12), -np.ones(12)],
                    choices=np.tile([0, 1], (24, 1)))
    D = np.full(24, self.frozen['high_D_threshold'], np.float64)
    result = stats._coverage(decision, D, self.frozen['high_D_threshold'], .2)
    self.assertEqual(result['high_D']['total'], 24)
    self.assertTrue(result['high_D']['minimum_coverage_met'])
    self.assertFalse(result['power_or_population_coverage_claimed'])

  def test_constant_target_and_high_D_are_not_signal_discrimination(self):
    cal, blind = fixture()
    cal['zq'] = cal['zp'].copy()
    blind['zq'] = blind['zp'].copy()
    cal['D'][:] = 100.
    blind['D'][:] = 100.
    frozen = stats.fit_calibration(cal)
    self.assertTrue(all(f['constant_target'] for f in frozen['regressors'].values()))
    result = joint(frozen, cal, blind)
    self.assertFalse(result['discrimination']['coverage']['high_D']['minimum_coverage_met'])
    self.assertFalse(result['models']['0']['finite_readout_opportunity_supported'])
    for record in result['discrimination']['regressors'].values():
      np.testing.assert_array_equal(record['predicted_U'], np.zeros(24))

  def test_same_two_slots_fixed_lambda_and_zero_missing_slot(self):
    for name, fit in self.frozen['regressors'].items():
      self.assertEqual(fit['nominal_slots'], 2)
      self.assertEqual(fit['nominal_coefficients'], 2)
      self.assertEqual(fit['lambda_value'], 1.)
      self.assertFalse(fit['equal_effective_capacity_claimed'])
      self.assertIn(fit['rank_defaulttol'], (0, 1, 2))
      if name in ('D-only', 'S-only'):
        self.assertEqual(fit['rms'][1], 0.)
        self.assertEqual(fit['weights'][1], 0.)
    S = np.asarray(self.frozen['oof']['S'])
    expected = np.sqrt(np.mean((S - S.mean()) ** 2))
    self.assertAlmostEqual(self.frozen['regressors']['D+S']['rms'][1], expected)

  def test_closed_delta_failures_and_rank_are_derived_without_refitting(self):
    result = joint(self.frozen, self.cal, self.blind)
    for model in result['models'].values():
      for branch in ('q', 'p'):
        count = int(np.sum(np.asarray(model['metrics'][branch + '_regret']['values']) > .2))
        self.assertEqual(model['closed_delta'][branch + '_failure_count'], count)
        self.assertEqual(model['closed_delta'][branch + '_failure_rate'], count / 24.)
    # A regret exactly delta belongs to the closed near-optimal set.
    self.assertFalse(float(.2) > stats.DELTA)
    for name, fit in self.frozen['regressors'].items():
      slots = stats._slots(name, self.cal['D'], self.cal['K'], np.asarray(self.frozen['oof']['S']))
      mean, rms = np.asarray(fit['mean']), np.asarray(fit['rms'])
      scaled = np.zeros_like(slots)
      active = rms > 0.
      scaled[:, active] = (slots[:, active] - mean[active]) / rms[active]
      self.assertEqual(fit['rank_defaulttol'], int(np.linalg.matrix_rank(scaled)))

  def test_prior_catchup_changes_gap_without_q_absolute_injury(self):
    frozen = copy.deepcopy(self.frozen)
    frozen['fit'] = force_model(frozen['fit'], prior_action=0)
    B = force_model(frozen['fit'], prior_action=1)
    R = force_model(frozen['fit'], prior_action=0)
    blind = copy.deepcopy(self.blind)
    # More cue1 states make prior action1 better, while both q heads are perfect.
    blind['zq'][:8, :, 0] = 0.
    blind['zq'][:8, :, 0, 1] = 1.
    blind['returns'][:8] = 0.
    blind['returns'][:8, 1] = 1.
    result = stats.evaluate_joint(frozen, {'B': B, 'R': R},
        {name: copy.deepcopy(blind) for name in ('0', 'B', 'R')})
    mechanism = result['mechanism']
    self.assertEqual(mechanism['R_minus_B_q_absolute']['mean'], 0.)
    self.assertLess(mechanism['R_minus_B_p_absolute']['mean'], 0.)
    self.assertGreater(mechanism['R_minus_B_q_minus_p']['mean'], 0.)

  def test_q_absolute_injury_is_reported_separately_from_prior(self):
    frozen = copy.deepcopy(self.frozen)
    frozen['fit'] = force_model(frozen['fit'])
    B = force_model(frozen['fit'], q_constant=True)
    R = force_model(frozen['fit'])
    result = stats.evaluate_joint(frozen, {'B': B, 'R': R},
        {name: copy.deepcopy(self.blind) for name in ('0', 'B', 'R')})
    self.assertGreater(result['mechanism']['R_minus_B_q_absolute']['mean'], 0.)
    self.assertEqual(result['mechanism']['R_minus_B_p_absolute']['mean'], 0.)

  def test_whole_episode_bootstrap_matches_direct_private_generator(self):
    result = joint(self.frozen, self.cal, self.blind)
    values = np.asarray(result['models']['0']['metrics']['q_minus_p']['values'])
    indices = np.random.Generator(np.random.PCG64(20261026)).integers(0, 24, (2000, 24))
    expected = np.percentile(values[indices].mean(1), [2.5, 97.5], method='linear')
    np.testing.assert_array_equal(result['models']['0']['metrics']['q_minus_p']['ci95'], expected)
    self.assertEqual(result['statistics']['bootstrap_indices_digest'], stats._digest(indices))
    self.assertFalse(result['statistics']['refit_or_training_uncertainty_included'])

  def test_blind_mutation_cannot_change_frozen_calibration(self):
    before = stats.dumps_frozen(self.frozen)
    blind = copy.deepcopy(self.blind)
    blind['D'] *= 10.
    blind['K'] += 50.
    blind['returns'][:] = 3.
    result = joint(self.frozen, self.cal, blind)
    self.assertEqual(stats.dumps_frozen(self.frozen), before)
    self.assertEqual(result['baseline_lambda_reused_for_B_R'], self.frozen['selected_lambda'])
    self.assertEqual(result['discrimination']['coverage']['high_D_threshold'],
                     self.frozen['high_D_threshold'])
    self.assertFalse(result['models']['0']['finite_readout_opportunity_supported'])

  def test_input_immutability_and_json_roundtrip(self):
    originals = {key: value.copy() for key, value in self.cal.items()}
    frozen = stats.fit_calibration(self.cal)
    serialized = stats.dumps_frozen(frozen)
    restored = stats.loads_frozen(serialized)
    self.assertEqual(stats.dumps_frozen(restored), serialized)
    result = joint(restored, self.cal, self.blind)
    json.dumps(result, allow_nan=False)
    for key, expected in originals.items():
      np.testing.assert_array_equal(self.cal[key], expected)
    with self.assertRaises(ValueError):
      stats.loads_frozen(serialized.replace('"delta":0.2', '"delta":0.3'))

  def test_refit_labels_lambda_and_joint_pairing_are_guarded(self):
    fit = stats.refit_calibration(self.cal, lambda_value=self.frozen['selected_lambda'])
    other = copy.deepcopy(fit)
    other['returns_digest']['sha256'] = '0' * 64
    with self.assertRaises(ValueError):
      stats.evaluate_joint(self.frozen, {'B': other, 'R': fit},
          {name: copy.deepcopy(self.blind) for name in ('0', 'B', 'R')})
    blind = {name: copy.deepcopy(self.blind) for name in ('0', 'B', 'R')}
    blind['B']['returns'][0, 0] += 1.
    with self.assertRaises(ValueError):
      stats.evaluate_joint(self.frozen, {'B': fit, 'R': fit}, blind)
    with self.assertRaises(ValueError):
      stats.refit_calibration(self.cal, lambda_value=.02)
    with self.assertRaises(ValueError):
      stats.evaluate_joint(self.frozen, {'B': fit, 'R': fit},
          {name: copy.deepcopy(self.blind) for name in ('0', 'B', 'R')}, seed=0)

  def test_nonfinite_dtype_onehot_divergence_and_axes_fail_closed(self):
    for key, mutation in (
        ('h', lambda a: a.astype(np.float32)),
        ('returns', lambda a: a + np.nan),
        ('D', lambda a: -a),
        ('zq', lambda a: a[:, :3]),
        ('zp', lambda a: a * .5)):
      batch = copy.deepcopy(self.cal)
      batch[key] = mutation(batch[key])
      with self.subTest(key=key), self.assertRaises(ValueError):
        stats.fit_calibration(batch)
    batch = copy.deepcopy(self.cal)
    batch['episode_ids'][0] = 100
    with self.assertRaises(ValueError):
      stats.fit_calibration(batch)

  def test_original_tiny_negative_raw_K_is_preserved_without_clipping(self):
    cal, blind = fixture()
    cal['K'][:] = -1e-5
    frozen = stats.fit_calibration(cal)
    self.assertEqual(frozen['oof']['K'], [-1e-5] * 8)
    blind['K'][:] = -5e-6
    result = joint(frozen, cal, blind)
    self.assertEqual(result['input_digests']['0']['K'], stats._digest(blind['K']))
    cal['K'][0] = -1.0001e-5
    with self.assertRaises(ValueError):
      stats.fit_calibration(cal)


if __name__ == '__main__':
  unittest.main()
