"""NumPy-only DL-feedback-r1 counterexamples; never real opportunity evidence.

These cases test the frozen statistical object, including clustered episodes,
absorbing rewards and selective-positive cancellation. They do not certify
terminal causes, source restoration, policy sampling or method effectiveness.
"""
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.dl_feedback import analyze_utility


SHAPE = (8, 4, 4, 2, 100)
EPISODES = np.array([9, 14, 15, 18, 20, 24, 30, 31], np.int64)


def fixture():
  return np.full(SHAPE, .5, np.float64), np.ones(SHAPE, bool)


class FeedbackStatisticsTest(unittest.TestCase):

  def evaluate(self, rewards, executed):
    result = analyze_utility(rewards, executed, EPISODES)
    self.assertIs(result['proceed_short_training'], False)
    return result

  def test_signed_cumulative_prefixes_and_branch_order(self):
    rewards, executed = fixture()
    rewards[..., 0, :] = .2
    rewards[..., 1, :] = .1
    result = self.evaluate(rewards, executed)
    self.assertAlmostEqual(result['primary'], 10., places=12)
    self.assertEqual(result['status'], 'exploratory_average_opportunity_supported')
    for horizon in (1, 10, 50, 100):
      np.testing.assert_allclose(
          result['signed_prefix'][str(horizon)], .1 * horizon,
          rtol=0, atol=1e-12)
    reverse = self.evaluate(rewards[..., ::-1, :].copy(), executed.copy())
    np.testing.assert_allclose(reverse['u_pair'], -np.asarray(result['u_pair']),
                               rtol=0, atol=1e-12)
    self.assertAlmostEqual(reverse['primary'], -10., places=12)
    self.assertEqual(reverse['status'], 'exploratory_average_opportunity_not_supported')

  def test_cluster_bootstrap_keeps_16_repeats_inside_episode(self):
    rewards, executed = fixture()
    episode_values = np.array([-.14, -.10, -.06, -.02, .02, .06, .10, .14])
    rewards[..., 0, 0] += episode_values[:, None, None]
    result = self.evaluate(rewards, executed)
    # Each episode has identical nested repetitions. Flattening them would
    # manufacture 128 independent units and a substantially narrower interval.
    draws = np.random.default_rng(20261019).integers(0, 8, size=(2000, 8))
    expected = np.quantile(episode_values[draws].mean(1), [.025, .975],
                           method='linear')
    np.testing.assert_allclose(result['u_episode'], episode_values, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result['interval95'], expected, rtol=0, atol=1e-12)
    self.assertEqual(result['status'], 'exploratory_average_opportunity_inconclusive')
    wrong_draws = np.random.default_rng(20261019).integers(0, 128, size=(2000, 128))
    wrong = np.quantile(np.repeat(episode_values, 16)[wrong_draws].mean(1),
                        [.025, .975], method='linear')
    self.assertGreater(expected[1] - expected[0], 3 * (wrong[1] - wrong[0]))

  def test_terminal_padding_keeps_fixed_window_not_survivor_mean(self):
    rewards, executed = fixture()
    rewards[..., 0, :20] = .8
    rewards[..., 0, 20:] = 0.
    executed[..., 0, 20:] = False
    rewards[..., 1, :] = .1
    result = self.evaluate(rewards, executed)
    # q return 16, p return 10; the q terminal does not delete the paired p,
    # reduce H100 or turn this into a difference of per-alive-step means.
    self.assertAlmostEqual(result['primary'], 6., places=12)
    for horizon, expected in ((1, .7), (10, 7.), (50, 11.), (100, 6.)):
      np.testing.assert_allclose(result['signed_prefix'][str(horizon)], expected,
                                 rtol=0, atol=1e-12)
    np.testing.assert_allclose(result['interval95'], [6., 6.], rtol=0, atol=1e-12)

  def test_average_negative_does_not_erase_positive_state_subset(self):
    rewards, executed = fixture()
    rewards[:, 0, :, 0, 0] += .1
    rewards[:, 1:, :, 0, 0] -= .2
    result = self.evaluate(rewards, executed)
    self.assertAlmostEqual(result['primary'], -.125, places=12)
    self.assertEqual(result['status'], 'exploratory_average_opportunity_not_supported')
    utilities = np.asarray(result['u_pair'])
    self.assertTrue(np.all(utilities[:, 0] > 0))
    self.assertTrue(np.all(utilities[:, 1:] < 0))
    # A nonpositive average is not a proof that every selective target is absent.
    self.assertIs(result['proceed_short_training'], False)

  def test_rng_dispersion_and_descriptions_preserve_axes(self):
    rewards, executed = fixture()
    by_rng = np.array([-.12, -.04, .04, .12])
    rewards[..., 0, 0] += by_rng[None, None, :]
    result = self.evaluate(rewards, executed)
    np.testing.assert_allclose(result['per_rng'], by_rng, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result['within_state']['std'],
                               np.std(by_rng, ddof=1), rtol=0, atol=1e-12)
    np.testing.assert_allclose(result['within_state']['range'], .24, rtol=0, atol=1e-12)
    self.assertAlmostEqual(result['first_two'], -.08, places=12)
    self.assertAlmostEqual(result['last_two'], .08, places=12)
    expected_leave_one = [np.delete(by_rng, r).mean() for r in range(4)]
    np.testing.assert_allclose(result['leave_one_rng'], expected_leave_one,
                               rtol=0, atol=1e-12)
    self.assertAlmostEqual(result['primary'], 0., places=12)

  def test_zero_boundary_is_not_supported_and_rng_state_is_untouched(self):
    rewards, executed = fixture()
    rewards_before, executed_before = rewards.copy(), executed.copy()
    rng_before = np.random.get_state()
    first = self.evaluate(rewards, executed)
    second = self.evaluate(rewards, executed)
    rng_after = np.random.get_state()
    self.assertEqual(first['status'], 'exploratory_average_opportunity_not_supported')
    self.assertEqual(first['primary'], 0.)
    self.assertEqual(first['interval95'], [0., 0.])
    self.assertEqual(first, second)
    np.testing.assert_array_equal(rewards, rewards_before)
    np.testing.assert_array_equal(executed, executed_before)
    self.assertEqual(rng_before[0], rng_after[0])
    np.testing.assert_array_equal(rng_before[1], rng_after[1])
    self.assertEqual(rng_before[2:], rng_after[2:])

  def test_rejects_dtype_shape_and_episode_identity_substitutions(self):
    rewards, executed = fixture()
    invalid = [
        (rewards.astype(np.float32), executed, EPISODES),
        (rewards, executed.astype(np.int8), EPISODES),
        (rewards[:7], executed[:7], EPISODES[:7]),
        (rewards, executed[..., :99], EPISODES),
        (rewards, executed, np.repeat(9, 8)),
        (rewards, executed, EPISODES.astype(np.float64)),
        (rewards, executed, EPISODES.astype(str)),
        (rewards, executed, np.ones(8, bool)),
        (rewards, executed, EPISODES[:, None]),
    ]
    for index, arguments in enumerate(invalid):
      with self.subTest(index=index), self.assertRaises(ValueError):
        analyze_utility(*arguments)

  def test_rejects_mask_holes_missing_initial_step_and_fake_padding(self):
    rewards, executed = fixture()
    invalid = []
    hole = executed.copy()
    hole[0, 0, 0, 0, 20] = False  # Later True would mean reset/resumption.
    invalid.append((rewards.copy(), hole))
    empty = executed.copy()
    empty[0, 0, 0, 0] = False
    empty_rewards = rewards.copy()
    empty_rewards[0, 0, 0, 0] = 0.
    invalid.append((empty_rewards, empty))
    padded = executed.copy()
    padded[0, 0, 0, 0, 20:] = False
    nonzero_padding = rewards.copy()
    nonzero_padding[0, 0, 0, 0, 20:] = 0.
    nonzero_padding[0, 0, 0, 0, 99] = 1e-30
    invalid.append((nonzero_padding, padded))
    for index, arguments in enumerate(invalid):
      with self.subTest(index=index), self.assertRaises(ValueError):
        self.evaluate(*arguments)

  def test_rejects_nonfinite_values_even_in_absorbing_padding(self):
    rewards, executed = fixture()
    for bad, in_padding in ((np.nan, False), (np.inf, False), (-np.inf, False),
                             (np.nan, True)):
      candidate, mask = rewards.copy(), executed.copy()
      if in_padding:
        mask[0, 0, 0, 0, 20:] = False
        candidate[0, 0, 0, 0, 20:] = 0.
      candidate[0, 0, 0, 0, 99] = bad
      with self.subTest(bad=bad, padding=in_padding), self.assertRaises(ValueError):
        self.evaluate(candidate, mask)


if __name__ == '__main__':
  unittest.main()
