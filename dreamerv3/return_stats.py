"""Raw environment-interface episode return; no repeat or reward transform."""

import math


REWARD_SOURCE = 'environment_interface.reward'
RETURN_DEFINITION = 'undiscounted_sum_non_reset_rewards_repeat_already_summed'


class EpisodeReturn:

  def __init__(self):
    self.started = False
    self.complete = False
    self.return_raw = 0.0
    self.length = 0

  def add(self, transition):
    reward = float(transition['reward'])
    if not math.isfinite(reward):
      raise ValueError('Non-finite environment reward')
    if bool(transition['is_first']):
      if self.started:
        raise ValueError('Reset would splice two evaluation episodes')
      if reward != 0 or bool(transition['is_last']):
        raise ValueError('Reset must have zero reward and not end an episode')
      self.started = True
      return
    if not self.started or self.complete:
      raise ValueError('Reward outside the current evaluation episode')
    # ActionRepeat has already summed primitive rewards at the env interface.
    self.return_raw += reward
    self.length += 1
    self.complete = bool(transition['is_last'])

  def result(self):
    if not self.complete or self.length < 1:
      raise ValueError('Evaluation episode is incomplete')
    return dict(return_raw=self.return_raw, length=self.length, complete=True,
                reward_source=REWARD_SOURCE,
                return_definition=RETURN_DEFINITION)
