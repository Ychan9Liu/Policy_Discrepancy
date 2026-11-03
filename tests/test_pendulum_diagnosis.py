"""Hand-counted replay cases, independent of the million-action artifacts."""
import unittest
import numpy as np

from analysis.diagnose_pendulum_v1 import episodes_from_chain, phase, replay_order


def chain(reward, action, first, last):
    return dict(reward=np.asarray(reward, dtype=np.float32),
                action=np.asarray(action, dtype=np.float32)[:, None],
                is_first=np.asarray(first, dtype=bool),
                is_last=np.asarray(last, dtype=bool),
                is_terminal=np.zeros(len(reward), dtype=bool))


class ReplayForensicsTest(unittest.TestCase):
    def setUp(self):
        self.chains = [
            chain([0, 1, 2, 0, 3], [10, 11, 0, 13, 14], [1, 0, 0, 1, 0], [0, 0, 1, 0, 0]),
            chain([0, 4, 5], [20, 21, 22], [1, 0, 0], [0, 0, 0]),
        ]

    def test_boundaries_reset_exclusion_and_previous_action(self):
        fields, stamps, by_worker = replay_order(self.chains, budget=5, interval=3)
        np.testing.assert_array_equal(stamps, [0, 0, 1, 2, 3, 3, 4, 5])
        np.testing.assert_array_equal(fields['reward'], [0, 0, 1, 4, 2, 0, 5, 3])
        use = ~fields['is_first']
        np.testing.assert_array_equal(fields['applied_raw_action'][use, 0], [10, 20, 11, 21, 13])
        self.assertEqual(phase(fields, stamps, 0, 3)['reward_sum'], 7)
        self.assertEqual(phase(fields, stamps, 3, 5)['reward_sum'], 8)
        complete, tail = episodes_from_chain(self.chains[0], by_worker[0], 0)
        self.assertEqual(len(complete), 1)
        self.assertEqual((complete[0]['length'], complete[0]['return_raw']), (2, 3))
        self.assertEqual((complete[0]['first_positive_step'], complete[0]['positive_bouts'], complete[0]['longest_positive_bout']), (1, 1, 2))
        self.assertEqual((tail['length'], tail['return_raw']), (1, 3))

    def test_delayed_reward_entry_and_disjoint_positive_bouts(self):
        sample = chain([0, 0, 1, 0, 1, 1, 0], [0]*7, [1, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0])
        complete, tail = episodes_from_chain(sample, np.asarray([0, 1, 2, 3, 4, 5, 5]), 0)
        episode = complete[0]
        self.assertEqual(episode['first_action_reward'], 0)
        self.assertEqual((episode['first_positive_step'], episode['last_positive_step']), (2, 5))
        self.assertEqual((episode['positive_bouts'], episode['longest_positive_bout']), (2, 2))
        self.assertEqual((tail['length'], tail['return_raw']), (0, 0))

    def test_incomplete_or_extra_replay_is_rejected(self):
        with self.assertRaises(ValueError):
            replay_order(self.chains, budget=6, interval=3)
        with self.assertRaises(ValueError):
            replay_order(self.chains, budget=4, interval=3)

    def test_internal_reset_without_episode_end_is_rejected(self):
        broken = chain([0, 1, 0], [1, 2, 3], [1, 0, 1], [0, 0, 0])
        with self.assertRaises(ValueError):
            episodes_from_chain(broken, np.asarray([0, 1, 1]), 0)


if __name__ == '__main__':
    unittest.main()
