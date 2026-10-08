"""Small action-grid, matching, and partial-driver boundary fixtures."""

import unittest

import elements
import numpy as np

from embodied.core.driver import Driver
from embodied.run.protocol_v1 import ProtocolState, seed32


class TinyEnv:

  def __init__(self):
    self.count = 0
    self.act_space = {'action': elements.Space(np.float32, (1,)),
                      'reset': elements.Space(bool)}

  def step(self, action):
    if action['reset']:
      self.count = 0
      return dict(reward=np.float32(0), is_first=np.bool_(True),
                  is_last=np.bool_(False))
    self.count += 1
    return dict(reward=np.float32(1), is_first=np.bool_(False),
                is_last=np.bool_(self.count == 2))

  def close(self):
    pass


class ProtocolV1Test(unittest.TestCase):

  def test_partial_driver_and_exact_action_budget(self):
    driver = Driver([TinyEnv] * 3, parallel=False)
    state = ProtocolState(5, 1, 2, 1, 4)
    seen = []
    driver.on_step(lambda tran, worker: (
        state.on_train_transition(tran), seen.append((worker, bool(
            tran['is_first'])))))
    def policy(carry, obs):
      next_carry = {'value': [x + 1 for x in carry['value']]}
      acts = {'action': np.ones((len(obs['reward']), 1), np.float32)}
      return next_carry, acts, {}
    driver.reset(lambda count: {'value': [0] * count})
    while state.actions < state.budget:
      remaining = state.budget - state.actions
      driver.step_selected(policy, range(min(3, remaining)))
    self.assertEqual(state.actions, 5)
    self.assertGreaterEqual(state.resets, 3)
    self.assertEqual(driver.carry['value'][2], 2)
    self.assertEqual(seen[-1][0], 1)
    driver.close()

  def test_match_window_receipts_and_replay_repeat(self):
    state = ProtocolState(12, 3, 2, 3, 9)
    self.assertEqual(state.grid, (0, 3, 6, 9, 12))
    self.assertEqual(state.next_boundary(), 3)
    state.record_update(0, 2, 0.4, 1)
    state.record_update(1, 3, 0.2, 1)
    state.record_update(2, 8, 1.8, 3)
    state.record_update(3, 9, 0.5, 1)
    self.assertEqual(state.match_count, 4)
    self.assertAlmostEqual(state.match_sum, 2.0)
    self.assertAlmostEqual(state.frozen_c(), 0.5)
    self.assertFalse(state.record_update(2, 8, 1.8, 3))
    with self.assertRaisesRegex(ValueError, 'Conflicting'):
      state.record_update(2, 8, 1.9, 3)
    # Repeating the same replay sample under a new update ID counts again.
    state.record_update(4, 8, 1.8, 3)
    self.assertEqual(state.match_count, 7)
    self.assertAlmostEqual(state.frozen_c(), 3.8 / 7)

  def test_empty_invalid_and_checkpoint_restore(self):
    state = ProtocolState(12, 3, 2, 3, 9)
    state.record_update(0, 3, 0, 0)
    with self.assertRaisesRegex(ValueError, 'No active'):
      state.frozen_c()
    for kwargs in (
        dict(match_sum=float('nan'), match_count=1),
        dict(match_sum=1, match_count=1, invalid_count=1),
        dict(match_sum=2, match_count=1),
        dict(match_sum=0, match_count=0, finite=False),
    ):
      with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
        state.record_update(1, 3, **kwargs)
    state.actions = 3
    state.record_evaluation(3, [1, 2], [2, 2], [11, 12])
    restored = ProtocolState(12, 3, 2, 3, 9)
    restored.load(state.save())
    self.assertEqual(restored.save(), state.save())
    with self.assertRaisesRegex(ValueError, 'already recorded'):
      restored.record_evaluation(3, [1, 2], [2, 2], [11, 12])
    with self.assertRaisesRegex(ValueError, 'protocol mismatch'):
      ProtocolState(15, 3, 2, 3, 9).load(state.save())

  def test_seed_mapping_is_named_and_repeatable(self):
    first = seed32(0, 'dmc_reacher_hard', 'eval_env', 3, 7)
    self.assertEqual(first, seed32(0, 'dmc_reacher_hard', 'eval_env', 3, 7))
    self.assertNotEqual(first, seed32(
        0, 'dmc_reacher_hard', 'train_env', 3, 7))
    self.assertNotEqual(first, seed32(
        0, 'dmc_reacher_hard', 'eval_env', 3, 8))


if __name__ == '__main__':
  unittest.main()
