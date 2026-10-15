"""Finite arithmetic/identifiability cases, never a real-task acceptance."""
import copy
import unittest

import numpy as np

from analysis import dl_normal_readout as readout


class NormalReadoutTests(unittest.TestCase):
  def test_dual_fit_matches_independent_primal_and_effective_df_svd(self):
    rng = np.random.default_rng(4401)
    h, z, y = rng.normal(size=(12, 3)), rng.normal(size=(12, 2)), rng.normal(size=(12, 2))
    model = readout.fit(h, z, y)
    hc, zc = h-h.mean(0), z-z.mean(0)
    x = np.c_[hc/np.sqrt((hc*hc).sum(1).mean()), zc/np.sqrt((zc*zc).sum(1).mean())]
    primal = np.linalg.solve(x.T @ x/12+.1*np.eye(5), x.T @ (y-y.mean(0))/12)
    np.testing.assert_allclose(model['weights'], primal, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(readout.predict(model, h, z), x @ primal+y.mean(0), rtol=1e-12, atol=1e-12)
    singular = np.linalg.svd(x/np.sqrt(12), compute_uv=False)
    df = ((singular*singular)/(singular*singular+.1)).sum()
    self.assertAlmostEqual(model['effective_df'], df, places=12)
    self.assertAlmostEqual(model['kernel_trace'], 2., places=13)
    self.assertLessEqual(model['effective_df'], 20.)

  def test_group_rms_controls_coordinate_count_and_zero_groups_stay_zero(self):
    h = np.arange(16, dtype=np.float64).reshape(8, 2)
    z = np.sin(h)
    y = np.c_[h[:, 0], h[:, 0]**2]
    original = readout.fit(h, z, y)
    repeated = readout.fit(np.tile(h, (1, 7)), np.tile(z, (1, 11)), y)
    np.testing.assert_allclose(readout.predict(original, h, z),
        readout.predict(repeated, np.tile(h, (1, 7)), np.tile(z, (1, 11))), atol=1e-12, rtol=1e-12)
    zero = readout.fit(np.zeros_like(h), np.ones_like(z), y)
    self.assertEqual(zero['effective_df'], 0.)
    self.assertEqual(zero['kernel_trace'], 0.)
    np.testing.assert_array_equal(readout.predict(zero, h+100, z-100), np.tile(y.mean(0), (8, 1)))
    history = readout.fit(h, None, y)
    self.assertLessEqual(history['effective_df'], 10.)
    with self.assertRaisesRegex(ValueError, 'History-only'):
      readout.predict(history, h, z)

  def test_family_average_not_frame_pseudoreplication_and_pair_orientation(self):
    # Family0 has three perfectly predicted frames; family1 one error2 frame.
    target = np.zeros((4, 1), np.float64)
    family = np.array([0, 0, 0, 1], np.int64)
    ref = readout.family_risks(np.array([[0.], [0.], [0.], [2.]]), target, family)
    alt = readout.family_risks(np.zeros_like(target), target, family)
    self.assertEqual(ref['mean'][0], 2.)  # Naive frame mean would be1.
    result = readout.improvement(ref, alt)
    self.assertEqual(result['mean'][0], 2.)
    self.assertEqual(result['families'], 2)
    negative = readout.improvement(alt, ref)
    np.testing.assert_array_equal(negative['mean'], -result['mean'])
    np.testing.assert_array_equal(negative['pointwise_ci95'], -result['pointwise_ci95'][::-1])
    same = readout.improvement(ref, ref)
    np.testing.assert_array_equal(same['pointwise_ci95'], np.zeros((2, 1)))
    bad = copy.deepcopy(alt)
    bad['families'] = bad['families'][::-1]
    with self.assertRaisesRegex(ValueError, 'Paired families'):
      readout.improvement(ref, bad)

  def test_probability_feature_not_mode_or_sample_and_bad_inputs_rejected(self):
    q = np.array([[[.51, .49]], [[.99, .01]]], np.float64)
    # Same modal class carries different distribution features.
    np.testing.assert_array_equal(q.argmax(-1), np.zeros((2, 1), np.int64))
    self.assertFalse(np.array_equal(readout.belief_features(q)[0], readout.belief_features(q)[1]))
    for wrong in (q.astype(np.float32), q*.5, q*np.nan, np.ones((2, 1, 1))):
      with self.assertRaises(ValueError):
        readout.belief_features(wrong)
    h, y = np.ones((2, 2), np.float64), np.ones((2, 1), np.float64)
    for wrong in (h.astype(np.float32), h[:1], h*np.inf):
      with self.assertRaises(ValueError):
        readout.fit(wrong, None, y)


if __name__ == '__main__':
  unittest.main()
