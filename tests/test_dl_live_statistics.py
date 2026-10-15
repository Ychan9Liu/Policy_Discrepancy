"""Finite analytic CPU fixtures: not live model, randomization or physics evidence."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from analysis import dl_live_statistics as s


def fixture():
  units = []
  for i in range(512):
    position, label = s.assignment_plan(i)
    e = (i*7 % 13)/12
    y = .5+(.05 if label in ('qqA', 'qqB') else -.05)*(e-.5)
    event = 3*i+1+(i >= 128)
    units.append(dict(family_id=i//16, unit_id=i % 16,
        env_seed=8192+i//16, policy_seed=0x52350000+i,
        position=position, label=label,
        pretreatment=dict(D=(i % 16)/15, K=(i % 5)/7, e=e, r_t=.5, r_t_f32=.5),
        outcome=dict(reward_sum=100*y),
        chronology=dict(signals_sealed=event, assignment=event+1, outcome_complete=event+2)))
  data = dict(version=s.VERSION, horizon=100, bootstrap_seed=20261030,
      calibration_sealed_event=385, units=units, **s.PROVENANCE)
  data['frozen_models'] = s.freeze_calibration(units[:128])
  return data


class LiveStatisticsTests(unittest.TestCase):
  def test_real_cli_from_unrelated_directory_reaches_protocol_guard(self):
    with tempfile.TemporaryDirectory() as temporary:
      source, output = Path(temporary)/'input.json', Path(temporary)/'output.json'
      source.write_text('{}', encoding='utf-8')
      script = Path(__file__).resolve().parents[1]/'analysis'/'dl_live_statistics.py'
      run = subprocess.run([sys.executable, str(script), '--input', str(source),
          '--output', str(output)], cwd=temporary, capture_output=True, text=True, timeout=15)
      self.assertNotEqual(run.returncode, 0)
      self.assertIn('Exact root schema required', run.stderr)
      self.assertNotIn('ModuleNotFoundError', run.stderr)
      self.assertFalse(output.exists())

  def test_ht_finite_enumeration_and_shared_risk_noise_cancellation(self):
    # Exhaust all four equiprobable treatments, without sampling or regression.
    outcomes = [.9, .9, .2, .4]
    phi = [s.pseudo_outcomes(a, y, .3) for a, y in zip(s.LABELS, outcomes)]
    for response, expected in (('U_obs', .5), ('U_act', .7), ('U_memory', -.2), ('sham', 0)):
      self.assertAlmostEqual(sum(v[response] for v in phi)/4, expected)
    f0, f1 = -.1, .4
    observed = sum((v['U_obs']-f0)**2-(v['U_obs']-f1)**2 for v in phi)/4
    self.assertAlmostEqual(observed, (.5-f0)**2-(.5-f1)**2)
    # A treatment-dependent baseline b=Y destroys the intended HT estimand.
    contaminated = [s.pseudo_outcomes(a, y, y)['U_obs'] for a, y in zip(s.LABELS, outcomes)]
    self.assertEqual(sum(contaminated), 0)

  def test_pretreatment_control_variate_can_increase_variance(self):
    zero = [s.pseudo_outcomes(a, 0., 0.)['U_obs'] for a in s.LABELS]
    centered = [s.pseudo_outcomes(a, 0., 1.)['U_obs'] for a in s.LABELS]
    self.assertEqual(np.var(zero), 0)
    self.assertEqual(np.var(centered), 6)
    self.assertEqual(np.mean(centered), 0)

  def test_unit_target_ratio_and_empty_draw_not_repaired(self):
    rows = [dict(family_id=8+i//16) for i in range(384)]
    mask, values = np.zeros(384, bool), np.zeros(384)
    mask[[0, 16, 17, 18]] = True
    values[[0, 16, 17, 18]] = [1, 3, 3, 3]
    draws = np.array([[0, 1]+[2]*22, [0]*24, [1]*24])
    out = s.cluster_interval(rows, values, mask, draws)
    self.assertEqual(out['mean'], 2.5)  # Equal-family conditional mean would2.
    np.testing.assert_allclose(out['interval95'], [1.075, 2.975], rtol=0, atol=1e-15)
    empty = s.cluster_interval(rows, values, mask, np.array([[2]*24]))
    self.assertIsNone(empty['interval95'])
    self.assertEqual(empty['empty_bootstrap_denominators'], 1)

  def test_cal_freeze_ties_and_held_outcomes_cannot_change_fit(self):
    data = fixture()
    original = copy.deepcopy(data['frozen_models'])
    for u in data['units'][128:]:
      u['outcome']['reward_sum'] = 0.
      u['pretreatment']['e'] = 1.
    result = s.analyze(data)
    self.assertEqual(result['calibration_freeze'], original)
    self.assertEqual(result['coverage']['n_highD'], 96)
    self.assertFalse(result['primary_support_pattern'])
    self.assertFalse(result['sham_precision_is_support_gate'])
    self.assertEqual(result['weighted_gate_effect'], 'not_evaluated')
    self.assertFalse(result['training_authorized'])
    json.dumps(result, allow_nan=False)
    for u in data['units']:
      u['pretreatment']['D'] = 0.
    data['frozen_models'] = s.freeze_calibration(data['units'][:128])
    self.assertEqual(s.analyze(data)['coverage']['n_highD'], 384)

  def test_finite_support_pattern_does_not_require_sham_precision(self):
    result = s.analyze(fixture(), evidence_authenticated=True)
    self.assertTrue(result['primary_support_pattern'])
    self.assertFalse(result['sham_precision_flag'])
    self.assertFalse(result['implementation_concern'])
    self.assertFalse(result['method_effectiveness_established'])
    self.assertFalse(result['training_authorized'])

  def test_duplicate_and_quadratic_e_are_disabled_without_held_reactivation(self):
    for exponent in (1, 2):
      data = fixture()
      for unit in data['units'][:128]:
        unit['pretreatment']['e'] = unit['pretreatment']['D']**exponent
      frozen = s.freeze_calibration(data['units'][:128])
      control = frozen['aux_nonredundant_basis_control']
      self.assertTrue(control['projection']['residual_disabled'])
      self.assertLessEqual(control['projection']['cal_max_abs_residual'], 1e-10)
      self.assertEqual(control['base'], control['extended'])
      held = s.validate_units(data['units'])[128:]
      before = s.basis_predictions(held, control)
      np.testing.assert_array_equal(*before)
      for row in held:
        row['signals']['e'] = object()  # Disabled prediction must not even read it.
      after = s.basis_predictions(held, control)
      np.testing.assert_array_equal(before[0], after[1])
      data['frozen_models'] = frozen
      result = s.analyze(data, evidence_authenticated=True)
      auxiliary = result['aux_nonredundant_basis_control']
      self.assertEqual(auxiliary['highD']['mean'], 0.)
      self.assertEqual(auxiliary['highD']['interval95'], [0., 0.])
      self.assertFalse(auxiliary['support_pattern'])

  def test_projection_is_covariate_only_and_frozen_before_held(self):
    data = fixture()
    original = data['frozen_models']['aux_nonredundant_basis_control']['projection']
    self.assertFalse(original['residual_disabled'])
    for unit in data['units']:
      unit['outcome']['reward_sum'] = 0.
    for unit in data['units'][128:]:
      unit['pretreatment'].update(D=1000., K=2000., e=1.)
    changed = s.freeze_calibration(data['units'][:128])
    self.assertEqual(changed['aux_nonredundant_basis_control']['projection'], original)

  def test_sham_randomization_nonrejection_not_equivalence(self):
    rows = [dict(label='qqA', Y=1., signals=dict(r_t=0.)) for _ in range(20)]
    out = s.sham_test(rows, [True]*20)
    self.assertTrue(out['rejected'])
    self.assertLess(out['p_value'], .05)
    self.assertGreaterEqual(out['p_value'], 1/10001)
    for row in rows:
      row['Y'] = 0.
    self.assertEqual(s.sham_test(rows, [True]*20)['p_value'], 1.)

  def test_protocol_mapping_chronology_and_post_outcome_pollution_rejected(self):
    mutations = [lambda d: d.update(e_source='actual_assigned_next_reward'),
        lambda d: d.update(calibration_sealed_event=400),
        lambda d: d['units'][0]['pretreatment'].update(r_next=.3),
        lambda d: d['units'][0]['pretreatment'].update(r_t_f32=.5001),
        lambda d: d['units'][0]['chronology'].update(assignment=4),
        lambda d: d['units'][0].update(policy_seed=2),
        lambda d: d['units'][0].update(label='pp' if d['units'][0]['label'] != 'pp' else 'pq'),
        lambda d: d['units'][0]['outcome'].update(reward_sum=float('nan')),
        lambda d: d['frozen_models']['models']['U_obs']['DK'].update(intercept=88.)]
    base = fixture()
    for mutation in mutations:
      data = copy.deepcopy(base)
      mutation(data)
      with self.assertRaises(ValueError):
        s.analyze(data)
    del base['frozen_models']
    with self.assertRaisesRegex(ValueError, 'pre-held'):
      s.analyze(base, evidence_authenticated=True)


if __name__ == '__main__':
  unittest.main()
