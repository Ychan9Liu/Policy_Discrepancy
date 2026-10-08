"""Short continuous-action environment for protocol engineering fixtures."""

import elements
import embodied
import numpy as np


class DummyCont(embodied.Env):

  def __init__(self, task, length=3, seed=0):
    del task
    self.length = int(length)
    self.rng = np.random.default_rng(seed)
    self.count = 0
    self.done = False
    self.offset = np.float32(0)

  @property
  def obs_space(self):
    return dict(
        vector=elements.Space(np.float32, (5,)),
        reward=elements.Space(np.float32),
        is_first=elements.Space(bool),
        is_last=elements.Space(bool),
        is_terminal=elements.Space(bool))

  @property
  def act_space(self):
    return dict(
        action=elements.Space(np.float32, (2,)),
        reset=elements.Space(bool))

  def step(self, action):
    if action['reset'] or self.done:
      self.count = 0
      self.done = False
      self.offset = np.float32(self.rng.uniform(-0.5, 0.5))
      return self._obs(0, True)
    self.count += 1
    self.done = self.count >= self.length
    reward = np.float32(1 + self.offset + 0.01 * action['action'][0])
    return self._obs(reward, False)

  def _obs(self, reward, first):
    return dict(
        vector=np.full((5,), self.offset + self.count / 10, np.float32),
        reward=np.float32(reward),
        is_first=np.bool_(first),
        is_last=np.bool_(self.done),
        is_terminal=np.bool_(self.done))

  def close(self):
    pass
