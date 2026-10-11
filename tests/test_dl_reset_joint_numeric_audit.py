"""Generated numeric fixtures only; independent direct-solve audit, no data/GPU."""
import copy
import hashlib
import os
from pathlib import Path
import pickle
import sys
import unittest
from unittest import mock

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
  os.environ[name] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import dl_reset_joint_numeric_audit as audit
from scripts import dl_reset_readout as producer


def fixture(*, constant=False, adverse=False):
  def make(ids):
    ids = np.asarray(ids, np.int32)
    n = len(ids)
    cue = np.arange(n) % 2
    zq = np.zeros((n, 4, 32, 32), np.float64)
    zp = np.zeros_like(zq)
    zq[..., 0] = 1.
    zp[..., 0] = 1.
    zq[:, :, 0] = 0.
    zp[:, :, 0] = 0.
    for i in range(n): zq[i, :, 0, cue[i]] = 1.
    zp[:, :, 0, 2] = 1.
    if constant: zq = zp.copy()
    Y = np.full((n, 7), .25, np.float64)
    Y[:, 0] += .125  # Avoid numerical artificial prior near-ties.
    Y[np.arange(n), cue] += 1.
    return dict(episode_ids=ids, h=np.zeros((n, 3), np.float64),
        zq=zq, zp=zp, returns=Y,
        D=.1 + np.arange(n, dtype=np.float64) / 100.,
        K=1. + cue.astype(np.float64) / 5.)
  cal0 = make(audit.CAL)
  blind0 = make([i for i in range(32) if i not in audit.CAL])
  cal = {name: copy.deepcopy(cal0) for name in ('0', 'B', 'R')}
  blind = {name: copy.deepcopy(blind0) for name in ('0', 'B', 'R')}
  # B loses the finite cue; R and 0 retain it. No actual optimizer claim.
  for batch in (cal['B'], blind['B']):
    batch['zq'][:, :, 0] = 0.
    batch['zq'][:, :, 0, 0] = 1.
    batch['h'] += 1.
  if adverse:
    for batch in blind.values():
      batch['returns'][:] = .25
      batch['returns'][:, 0] += 1.
  frozen = producer.fit_calibration(cal['0'])
  refits = {name: producer.refit_calibration(cal[name], lambda_value=frozen['selected_lambda'])
            for name in ('B', 'R')}
  result = producer.evaluate_joint(frozen, refits, blind)
  return dict(frozen=frozen, refits=refits, cal=cal), blind, result


class IndependentResetNumericAuditTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.readouts, cls.blind, cls.result = fixture()

  def test_complete_producer_output_matches_independent_solve_and_fsum(self):
    report = audit.audit_numeric(self.readouts, self.blind, {'analysis': self.result})
    self.assertTrue(report['passed'])
    self.assertEqual(report['independent_shared_head_fits'], 35)
    self.assertEqual(report['independent_kernel_solve_count'], 70)
    self.assertGreater(report['numeric_comparisons'], 25000)
    self.assertGreater(report['exact_comparisons'], 250)
    self.assertGreater(report['zero_side_comparisons'], 60)
    self.assertLessEqual(report['maximum_absolute_difference'], 1e-10)
    self.assertFalse(report['physical_model_state_and_runtime_authenticated_by_this_helper'])
    self.assertFalse(report['continuous_actor_or_method_effectiveness_established'])

  def test_no_producer_fit_prediction_or_eigh_is_an_audit_oracle(self):
    with mock.patch.object(producer, 'fit_calibration', side_effect=AssertionError('producer called')), \
         mock.patch.object(producer, '_fit_shared', side_effect=AssertionError('producer called')), \
         mock.patch.object(producer, '_predict_shared', side_effect=AssertionError('producer called')), \
         mock.patch.object(np.linalg, 'eigh', side_effect=AssertionError('eigh called')):
      self.assertTrue(audit.audit_numeric(self.readouts, self.blind, self.result)['passed'])

  def test_solver_never_constructs_high_dimension_primal_system(self):
    original = np.linalg.solve
    dimensions = []
    def checked(system, rhs):
      dimensions.append(system.shape)
      self.assertLessEqual(system.shape[0], 64)
      return original(system, rhs)
    with mock.patch.object(np.linalg, 'solve', side_effect=checked):
      audit.audit_numeric(self.readouts, self.blind, self.result)
    self.assertIn((64, 64), dimensions)
    self.assertIn((56, 56), dimensions)
    self.assertIn((2, 2), dimensions)

  def test_point_prediction_tamper_rejected(self):
    result = copy.deepcopy(self.result)
    result['models']['R']['sample_predictions'][0][0][0][0] += 1e-4
    with self.assertRaisesRegex(ValueError, 'sample_predictions'):
      audit.audit_numeric(self.readouts, self.blind, result)

  def test_calibration_per_sample_measurement_chain_is_verified(self):
    readouts = copy.deepcopy(self.readouts)
    readouts['frozen']['fold_metadata'][0]['candidates'][0]['sample_predictions'][0][0][0] += 1e-4
    with self.assertRaisesRegex(ValueError, 'sample_predictions'):
      audit.audit_numeric(readouts, self.blind, self.result)
    # Even a change below the point oracle cannot silently break recorded mean4.
    readouts = copy.deepcopy(self.readouts)
    readouts['frozen']['fold_metadata'][0]['candidates'][0]['sample_predictions'][0][0][0] += 1e-12
    with self.assertRaisesRegex(ValueError, 'recorded_mean4'):
      audit.audit_numeric(readouts, self.blind, self.result)

  def test_exact_choice_tamper_rejected(self):
    result = copy.deepcopy(self.result)
    result['models']['0']['choices'][0][0] = 6
    with self.assertRaisesRegex(ValueError, 'choices.*exact mismatch'):
      audit.audit_numeric(self.readouts, self.blind, result)

  def test_CI_zero_side_tamper_rejected_inside_numeric_tolerance(self):
    result = copy.deepcopy(self.result)
    result['mechanism']['R_minus_B_p_absolute']['ci95'][0] = -1e-12
    with self.assertRaisesRegex(ValueError, 'zero-side mismatch'):
      audit.audit_numeric(self.readouts, self.blind, result)

  def test_lambda_and_fold_isolation_tamper_rejected(self):
    readouts = copy.deepcopy(self.readouts)
    rule = readouts['frozen']['selected_lambda']
    readouts['frozen']['selected_lambda'] = next(v for v in audit.LAMBDAS if v != rule)
    with self.assertRaisesRegex(ValueError, 'selected_lambda'):
      audit.audit_numeric(readouts, self.blind, self.result)
    readouts = copy.deepcopy(self.readouts)
    fold = readouts['frozen']['fold_metadata'][0]
    fold['training_episodes'][0] = fold['heldout_episode']
    with self.assertRaisesRegex(ValueError, 'fold.train'):
      audit.audit_numeric(readouts, self.blind, self.result)

  def test_cal_and_blind_label_pairing_are_exact(self):
    readouts = copy.deepcopy(self.readouts)
    readouts['cal']['B']['returns'][0, 0] += 1e-14
    with self.assertRaisesRegex(ValueError, 'cal.Y.B'):
      audit.audit_numeric(readouts, self.blind, self.result)
    blind = copy.deepcopy(self.blind)
    blind['R']['returns'][0, 0] += 1e-14
    with self.assertRaisesRegex(ValueError, 'blind.Y.R'):
      audit.audit_numeric(self.readouts, blind, self.result)

  def test_bundle_digest_and_common_history_phase_guards(self):
    result = copy.deepcopy(self.result)
    result['input_digests']['0']['D']['sha256'] = '0' * 64
    with self.assertRaisesRegex(ValueError, 'input_digest'):
      audit.audit_numeric(self.readouts, self.blind, result)
    blind = copy.deepcopy(self.blind)
    blind['R']['h'][0, 0] = 5.
    result = copy.deepcopy(self.result)
    result['input_digests']['R']['h'] = audit.digest(blind['R']['h'])
    with self.assertRaisesRegex(ValueError, 'common h differs'):
      audit.audit_numeric(self.readouts, blind, result)

  def test_oof_and_signal_capacity_tamper_rejected(self):
    readouts = copy.deepcopy(self.readouts)
    readouts['frozen']['oof']['S'][0] += 1e-4
    with self.assertRaisesRegex(ValueError, 'oof.S'):
      audit.audit_numeric(readouts, self.blind, self.result)
    readouts = copy.deepcopy(self.readouts)
    readouts['frozen']['regressors']['D+K']['lambda_value'] = 2.
    with self.assertRaisesRegex(ValueError, 'signal.lambda.D\\+K'):
      audit.audit_numeric(readouts, self.blind, self.result)

  def test_coverage_rank_and_closed_delta_tamper_rejected(self):
    result = copy.deepcopy(self.result)
    result['discrimination']['coverage']['all']['wrong_choice_negative'] += 2
    with self.assertRaisesRegex(ValueError, 'coverage'):
      audit.audit_numeric(self.readouts, self.blind, result)
    readouts = copy.deepcopy(self.readouts)
    readouts['frozen']['regressors']['D-only']['rank_defaulttol'] = 2
    with self.assertRaisesRegex(ValueError, 'signal.rank'):
      audit.audit_numeric(readouts, self.blind, self.result)
    result = copy.deepcopy(self.result)
    result['models']['0']['closed_delta']['q_failure_count'] += 1
    with self.assertRaisesRegex(ValueError, 'closed_delta'):
      audit.audit_numeric(self.readouts, self.blind, result)

  def test_hash_mismatch_is_rejected_before_unpickle(self):
    payload = pickle.dumps(dict(only_new_generated_fixture=True))
    with mock.patch.object(pickle, 'loads', side_effect=AssertionError('decoded wrong hash')):
      with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
        audit.load_hash_pinned_new_payload(payload, '0' * 64)
    decoded = audit.load_hash_pinned_new_payload(payload, hashlib.sha256(payload).hexdigest())
    self.assertEqual(decoded, dict(only_new_generated_fixture=True))

  def test_scientific_negative_can_pass_numerical_audit(self):
    readouts, blind, result = fixture(adverse=True)
    self.assertFalse(result['models']['0']['finite_readout_opportunity_supported'])
    self.assertLess(result['models']['0']['metrics']['q_minus_p']['mean'], 0.)
    self.assertTrue(audit.audit_numeric(readouts, blind, result)['passed'])

  def test_constant_target_is_reported_without_fake_discrimination(self):
    readouts, blind, result = fixture(constant=True)
    report = audit.audit_numeric(readouts, blind, result)
    self.assertTrue(report['passed'])
    self.assertEqual(report['signal_design_rank']['D+S'], 1)
    self.assertTrue(result['discrimination']['regressors']['D+S']['fit_constant_target'])
    self.assertFalse(report['coverage']['all']['minimum_coverage_met'])


if __name__ == '__main__':
  unittest.main()
