"""DL-r4: analytic CPU counterexamples, not real task or physics validation."""
import copy
import json
import unittest

import numpy as np

from analysis import dl_r4_statistics as stats


def fixture():
  episodes = []
  for episode_id in range(32):
    candidates = [dict(position=p, D=1+p/1000, K=.2+p/2000,
        e=((episode_id+i) % 8)/7) for i, p in enumerate(stats.POSITIONS)]
    selected = []
    for p in (512, 832, 896):
      e = next(row['e'] for row in candidates if row['position'] == p)
      u = .08*(e-.5)
      qq = [50+100*u]*16
      selected.append(dict(position=p,
          returns=dict(qq=qq, pq=[50+50*u]*16, pp=[50.]*16),
          qq_null_return=qq[0], null_max_abs_step_reward_diff=0.))
    episodes.append(dict(episode_id=episode_id, env_seed=4096+episode_id,
        candidates=candidates, selected=selected))
  data = dict(version=stats.VERSION, horizon=100, mc_pairs=16,
      calibration_episode_ids=list(range(8)), bootstrap_seed=20261028,
      episodes=episodes)
  data['frozen_models'] = stats.freeze_calibration(episodes[:8])
  return data


class StatisticsTests(unittest.TestCase):
  def test_analytic_ridge_unpenalized_intercept(self):
    # Weighted x moments: mean0/variance1. Slope=1/(1+.1), intercept7.
    model = stats.fit_predictor([[-1.], [1.]], [6., 8.], [0, 1])
    self.assertAlmostEqual(model['intercept'], 7.)
    self.assertAlmostEqual(model['coefficient'][0], 10/11)
    np.testing.assert_allclose(stats.predict(model, [[-1.], [0.], [1.]]),
        [7-10/11, 7, 7+10/11], rtol=0, atol=2e-15)

  def test_episode_mass_and_duplicate_whole_episode_invariance(self):
    # EP0 contributes three rows, EP1 one: each episode still has mass1/2.
    weights = stats.episode_weights([0, 0, 0, 1])
    np.testing.assert_allclose(weights, [1/6, 1/6, 1/6, .5], atol=0, rtol=0)
    x, y, ids = [[0.], [2.], [8.]], [3., 1., 7.], [0, 0, 1]
    first = stats.fit_predictor(x, y, ids)
    duplicate = stats.fit_predictor(x[:2]*3+x[2:], y[:2]*3+y[2:], [0]*6+[1])
    np.testing.assert_allclose(stats.predict(first, [[-3.], [0.], [10.]]),
        stats.predict(duplicate, [[-3.], [0.], [10.]]), rtol=0, atol=4e-15)

  def test_constant_features_and_finite_rejection(self):
    model = stats.fit_predictor([[5.], [5.]], [2., 4.], [0, 1])
    np.testing.assert_array_equal(stats.predict(model, [[5.], [1e30]]), [3., 3.])
    for bad in (float('nan'), float('inf')):
      with self.assertRaises(ValueError):
        stats.fit_predictor([[0.], [bad]], [0., 1.], [0, 1])
    with self.assertRaisesRegex(ValueError, 'standardization'):
      stats.fit_predictor([[-1e308], [1e308]], [0., 1.], [0, 1])

  def test_d_selection_ties_and_anchor_overlap(self):
    candidates = [dict(position=p, D=0., K=0., e=1.) for p in stats.POSITIONS]
    top, union = stats.selected_positions(candidates)
    self.assertEqual(top, (64, 128))
    self.assertEqual(union, (64, 128, 512))
    candidates[7]['D'] = 9  # Position512 becomes top2; no replacement third state.
    self.assertEqual(stats.selected_positions(candidates), ((64, 512), (64, 512)))
    candidates[0]['e'] = 0
    self.assertEqual(stats.selected_positions(candidates)[0], (64, 512))

  def test_state_labels_are_uncertain_and_exploratory(self):
    self.assertEqual(stats.summary([.02]*16)['classification'], 'exploratory_useful')
    self.assertEqual(stats.summary([-.02]*16)['classification'], 'exploratory_harmful')
    self.assertEqual(stats.summary([0.]*16)['classification'], 'exploratory_practically_small')
    varying = stats.summary([-.5]*8+[.52]*8)
    self.assertGreater(varying['mean'], 0)
    self.assertEqual(varying['classification'], 'exploratory_unresolved')
    self.assertLess(varying['interval95'][0], 0)
    self.assertGreater(varying['interval95'][1], 0)

  def test_frozen_calibration_independent_of_held(self):
    data = fixture()
    original = copy.deepcopy(data['frozen_models'])
    # Held e/distribution and labels may be arbitrarily different, no cal refit.
    for episode in data['episodes'][8:]:
      for candidate in episode['candidates']:
        candidate['e'] = 1.
      for state in episode['selected']:
        state['returns'] = dict(qq=[10.]*16, pq=[50.]*16, pp=[90.]*16)
        state['qq_null_return'] = 10.
    result = stats.analyze(data)
    self.assertEqual(result['calibration_freeze'], original)
    self.assertEqual(result['predictors'], original['models'])
    self.assertEqual(result['status'], 'measurement-inconclusive')
    self.assertFalse(result['training_authorized'])
    self.assertFalse(result['proceed_short_training'])

  def test_synthetic_finite_support_scope_and_scales(self):
    result = stats.analyze(fixture(), evidence_authenticated=True)
    self.assertEqual(result['status'], 'exploratory-supporting-finite-procedure')
    self.assertTrue(result['coverage']['passed'])
    self.assertEqual(result['model_protocol']['primary'], 'U_obs / held D-top2 / DKe_vs_DK')
    self.assertFalse(result['method_effectiveness_established'])
    self.assertEqual(result['predictors']['U_obs']['DKe']['features'], ['D', 'K', 'e', 'time'])
    # Exactly48 held primary states, not72 union states, and 16 cal training rows.
    self.assertEqual(sum(x['top2'] and x['episode_id'] >= 8
        for x in result['selected_states']), 48)
    self.assertEqual(len(result['calibration_freeze']['training_weights']), 16)
    self.assertEqual(len(result['per_episode']), 32)
    for row in result['selected_states']:
      self.assertAlmostEqual(row['U_act']['mean']*2, row['U_obs']['mean'])
    json.dumps(result, allow_nan=False)

  def test_union_and_anchor_do_not_silently_become_primary(self):
    rows = []
    values = []
    for ep in range(8, 32):
      for pos, value in ((64, 1.), (128, 3.), (512, 20.)):
        rows.append(dict(episode_id=ep, top2=pos != 512, fixed512=pos == 512))
        values.append(value)
    self.assertEqual(stats.group_episode(rows, values, scope='top2'), [2.]*24)
    self.assertEqual(stats.group_episode(rows, values, scope='fixed512'), [20.]*24)
    self.assertEqual(stats.group_episode(rows, values, scope='union'), [8.]*24)

  def test_null_failure_cannot_be_promoted_by_good_prediction(self):
    data = fixture()
    data['episodes'][12]['selected'][0]['qq_null_return'] += .250001
    result = stats.analyze(data, evidence_authenticated=True)
    self.assertGreater(result['responses']['U_obs']['top2']['comparisons']
        ['DKe_vs_DK']['interval95'][0], 0)
    self.assertEqual(result['status'], 'measurement-inconclusive')
    self.assertFalse(result['measurement']['passed'])

  def test_missing_freeze_and_cal_or_model_tampering_rejected(self):
    data = fixture()
    del data['frozen_models']
    with self.assertRaisesRegex(ValueError, 'pre-held'):
      stats.analyze(data, evidence_authenticated=True)
    data = fixture()
    data['frozen_models']['models']['U_obs']['DKe']['coefficient'][0] += 1e-6
    with self.assertRaisesRegex(ValueError, 'arithmetic mismatch'):
      stats.analyze(data)
    data = fixture()
    data['episodes'][0]['selected'][1]['returns']['qq'][0] += .1
    with self.assertRaises(ValueError):
      stats.analyze(data)

  def test_schema_selection_count_seed_and_range_corruption(self):
    corruptions = [
        lambda x: x.update(version='old'),
        lambda x: x.update(horizon=64),
        lambda x: x.update(mc_pairs=4),
        lambda x: x.update(bootstrap_seed=3),
        lambda x: x['calibration_episode_ids'].__setitem__(0, 9),
        lambda x: x['episodes'][0].update(env_seed=20261020),
        lambda x: x['episodes'][1].update(episode_id=0),
        lambda x: x['episodes'][0]['selected'][0].update(position=64),
        lambda x: x['episodes'][0]['selected'][0]['returns']['qq'].pop(),
        lambda x: x['episodes'][0]['selected'][0]['returns']['qq'].__setitem__(0, 100.1),
        lambda x: x['episodes'][0]['selected'][0]['returns']['qq'].__setitem__(0, True),
        lambda x: x['episodes'][0]['selected'][0]['returns']['qq'].__setitem__(0, float('nan')),
        lambda x: x['episodes'][0]['candidates'][0].update(e=1.01),
        lambda x: x['episodes'][0]['candidates'][0].update(D=-1),
        lambda x: x['episodes'][0]['candidates'][0].update(K=float('inf')),
        lambda x: x['episodes'][0]['selected'][0].update(null_max_abs_step_reward_diff=1.01),
        lambda x: x.update(extra='not allowed'),
    ]
    base = fixture()
    for mutate in corruptions:
      with self.subTest(mutate=mutate):
        data = copy.deepcopy(base)
        mutate(data)
        with self.assertRaises(ValueError):
          stats.validate(data)
    with self.assertRaises(ValueError):
      stats.decode('{"version":1,"version":2}')

  def test_risk_difference_noise_square_cancels(self):
    # Same held noisy label is used for both fixed predictors. The difference
    # of squared errors is linear in that label; its noise² cancels exactly.
    labels = np.array([-.5, -.25, .25, .5])
    f0, f1 = -.2, .3
    observed = np.mean((labels-f0)**2-(labels-f1)**2)
    expected = (labels.mean()-f0)**2-(labels.mean()-f1)**2
    self.assertAlmostEqual(observed, expected, places=15)
    self.assertNotEqual(np.mean((labels-f0)**2), (labels.mean()-f0)**2)


if __name__ == '__main__':
  unittest.main()
