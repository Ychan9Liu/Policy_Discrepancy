"""Synthetic arithmetic counterexamples; no actual C or NN efficacy claim."""
import copy
import math
import unittest

import numpy as np

from analysis.dl_nm1_calibration import freeze, release, summarize


def signal(index=0):
  shape = (16, 64)
  f = np.linspace(.1, .9, 1024, dtype=np.float32).reshape(shape)
  ids = np.zeros((*shape, 20), np.uint8)
  for value, row in zip(range(index*1024, (index+1)*1024), ids.reshape(-1, 20)):
    row[-8:] = np.frombuffer(value.to_bytes(8, 'big'), np.uint8)
  return dict(K=np.full(shape, 2., np.float32), E=np.ones(shape, np.float32),
      D=f.copy(), e=np.full(shape, .5, np.float32), f=f,
      v_N=np.float32(1)-np.float32(.25)*f,
      m=np.ones(shape, bool), A=np.ones(shape, bool), stepid=ids,
      chain=np.broadcast_to(np.arange(16, dtype=np.int32)[:, None], shape).copy())


def controls(record, frozen):
  # Synthetic formula fixture only; the real producer must retain actual JAX F32.
  a = record['A'].astype(np.float32)
  s = frozen['S']['coefficient_f32']
  w = frozen['W']['coefficient_f32']
  p = record['v_N'].reshape(-1)[::-1].reshape(16, 64).copy()
  return dict(v_S=None if s is None else np.float32(1)-a*np.float32(s)*record['f'],
      v_W=None if w is None else np.float32(1)-a*np.float32(w)*record['e'],
      v_P=p if frozen['bins']['calibrated'] else None)


def effect(eg=0, es=0, pg=0, ps=0):
  return dict(encoder_gradient_changed_elements=eg, encoder_step_changed_elements=es,
      posterior_gradient_changed_elements=pg, posterior_step_changed_elements=ps)


class CalibrationTests(unittest.TestCase):
  def test_complete_negative_zero_denominators_and_no_active_bins_are_not_exception(self):
    record = signal()
    record['K'][:] = .5
    record['E'][:] = 0
    frozen = freeze([record], expected_batches=1)
    self.assertFalse(frozen['S']['calibrated'])
    self.assertEqual(frozen['S']['reason'], 'zero_denominator')
    self.assertIsNone(frozen['W']['coefficient_f32'])
    self.assertFalse(frozen['bins']['calibrated'])
    result = summarize([record], [effect()], [controls(record, frozen)], frozen, expected_batches=1)
    self.assertFalse(any(result['qualification'].values()))
    self.assertEqual(result['release_actual_f64']['N'], 0.)
    self.assertIsNone(result['P_changed_protected_fraction'])
    self.assertEqual(result['relative_release_drift'], dict(S=None, W=None, P=None))

  def test_union_of_same_module_effects_and_full_coverage_qualify_not_method_efficacy(self):
    records = [signal(j) for j in range(128)]
    frozen = freeze(records)
    effects = [effect(1, 1) if j < 8 else effect(pg=1, ps=1) if j < 16 else effect()
        for j in range(128)]
    result = summarize(records, effects, [controls(row, frozen) for row in records], frozen)
    self.assertEqual(result['numerical_effect_batches'], 16)
    self.assertEqual(result['distinct_protected_stepids'], 128*1024)
    self.assertEqual(result['distinct_protected_chains'], 16)
    self.assertTrue(all(result['qualification'].values()))
    self.assertTrue(result['T1_necessary_qualification'])
    self.assertFalse(result['subsequent_learning_authorized'])
    self.assertFalse(result['method_efficacy_established'])

  def test_gradient_encoder_and_step_posterior_is_not_same_module_effect(self):
    records = [signal(j) for j in range(128)]
    frozen = freeze(records)
    result = summarize(records, [effect(eg=1, ps=1)]*128,
        [controls(row, frozen) for row in records], frozen)
    self.assertEqual(result['numerical_effect_batches'], 0)
    self.assertFalse(result['qualification']['C_N'])

  def test_repeated_exposure_does_not_increase_distinct_stepid_or_chain_coverage(self):
    record = signal()
    record['chain'][:] = 0
    record['stepid'][:] = 0
    records = [record]*128
    frozen = freeze(records)
    result = summarize(records, [effect(1, 1)]*128,
        [controls(row, frozen) for row in records], frozen)
    self.assertEqual(result['distinct_protected_stepids'], 1)
    self.assertEqual(result['distinct_protected_chains'], 1)
    self.assertFalse(result['qualification']['C_N_coverage'])
    bad = [copy.deepcopy(row) for row in records]
    bad[1]['chain'][:] = 1
    with self.assertRaisesRegex(ValueError, 'different source chains'):
      summarize(bad, [effect()]*128, [controls(row, frozen) for row in bad], frozen)

  def test_positive_analytic_release_can_round_to_no_actual_f32_protection(self):
    record = signal()
    record['e'][:] = np.float32(1e-9)
    record['v_N'][:] = np.float32(1)-np.float32(.5)*record['e']*record['f']
    frozen = freeze([record], expected_batches=1)
    self.assertGreater(frozen['aggregate64']['N_analytic_release'], 0.)
    result = summarize([record], [effect()], [controls(record, frozen)], frozen, expected_batches=1)
    self.assertEqual(result['release_actual_f64']['N'], 0.)
    self.assertFalse(result['qualification']['C_S'])
    self.assertEqual(result['distinct_protected_stepids'], 0)

  def test_actual_weights_not_analytic_identity_determine_scalar_qualification(self):
    record = signal()
    frozen = freeze([record], expected_batches=1)
    actual = controls(record, frozen)
    actual['v_S'][:] = 1.
    result = summarize([record], [effect()], [actual], frozen, expected_batches=1)
    self.assertEqual(result['relative_release_drift']['S'], 1.)
    self.assertFalse(result['qualification']['C_S'])

  def test_P_efficiency_pools_position_visits_instead_of_batch_proportions(self):
    records = [signal(j) for j in range(2)]
    for record, count in zip(records, (2, 100)):
      record['A'][:] = False
      record['A'].reshape(-1)[:count] = True
      record['v_N'][:] = 1
      record['v_N'].reshape(-1)[:count] = .8
    records[0]['v_N'].reshape(-1)[1] = .6
    frozen = freeze(records, expected_batches=2)
    actual = []
    for record in records:
      row = controls(record, frozen)
      row['v_P'] = record['v_N'].copy()
      actual.append(row)
    actual[0]['v_P'].reshape(-1)[:2] = records[0]['v_N'].reshape(-1)[:2][::-1]
    result = summarize(records, [effect()]*2, actual, frozen, expected_batches=2)
    self.assertEqual(result['P_protected_visits'], 102)
    self.assertEqual(result['P_changed_protected_visits'], 2)
    self.assertEqual(result['P_changed_protected_fraction'], 2/102)
    self.assertFalse(result['qualification']['C_P'])

  def test_quantiles_pool_repeated_visits_and_deduplicate_actual_f32_boundaries(self):
    record = signal()
    record['D'][:] = np.float32(.25)
    frozen = freeze([record], expected_batches=1)
    self.assertEqual(frozen['bins']['eligible_visits'], 1024)
    self.assertEqual(frozen['bins']['D_quantiles_f64'], [.25, .25])
    self.assertEqual(frozen['bins']['D_f32'], [.25])
    self.assertEqual(frozen['bins']['K_f32'], [2.])

  def test_release_lifts_f32_operands_before_subtraction_and_multiplication(self):
    weight = np.array([np.nextafter(np.float32(1), np.float32(0)), .8], np.float32)
    excess = np.array([1e18, .1], np.float32)
    expected = math.fsum((1-float(w))*float(e) for w, e in zip(weight, excess))
    self.assertEqual(release(weight, excess), expected)
    self.assertNotEqual(expected, float(np.sum((1-weight)*excess)))

  def test_incomplete_nonfinite_bad_E_or_controls_fail_integrity(self):
    record = signal()
    with self.assertRaisesRegex(ValueError, 'Complete fixed'):
      freeze([record])
    record['D'][0, 0] = np.nan
    with self.assertRaisesRegex(ValueError, 'Finite actual'):
      freeze([record], expected_batches=1)
    record = signal()
    record['E'][0, 0] = 0
    with self.assertRaisesRegex(ValueError, 'actual F32'):
      freeze([record], expected_batches=1)
    record = signal()
    frozen = freeze([record], expected_batches=1)
    with self.assertRaisesRegex(ValueError, 'actual numerical effect'):
      summarize([record], [effect(eg=True)], [controls(record, frozen)], frozen, expected_batches=1)
    broken = dict(controls(record, frozen), v_S=None)
    with self.assertRaisesRegex(ValueError, 'Actual finite GPU'):
      summarize([record], [effect()], [broken], frozen, expected_batches=1)


if __name__ == '__main__':
  unittest.main()
