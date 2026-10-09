"""Capture seeded DMC image and physics state under fixed actions."""

import argparse
from pathlib import Path

import numpy as np

from embodied.envs.dmc import DMC


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--task', default='hopper_hop')
  parser.add_argument('--seed', type=int, required=True)
  parser.add_argument('--steps', type=int, default=64)
  parser.add_argument('--output', required=True)
  args = parser.parse_args()
  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  env = DMC(args.task, seed=args.seed, image=True, proprio=False)
  images, states, rewards = [], [], []
  zero = np.zeros(env.act_space['action'].shape, np.float32)
  for index in range(args.steps + 1):
    obs = env.step(dict(action=zero, reset=index == 0))
    images.append(obs['image'].copy())
    states.append(env._dmenv.physics.get_state().copy())
    rewards.append(float(obs['reward']))
  env.close()
  np.savez_compressed(output, images=np.stack(images),
                      physics=np.stack(states), rewards=np.array(rewards))
  print(dict(task=args.task, seed=args.seed, steps=args.steps,
             output=str(output)), flush=True)


if __name__ == '__main__':
  main()
