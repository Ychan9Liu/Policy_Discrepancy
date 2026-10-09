"""Repeat one size50m CUDA train path with frozen synthetic visual input.

Engineering fixture only: it isolates model/update computation from DMC,
replay selection, and evaluation without changing the M2 protocol.
"""

import argparse
import json
from pathlib import Path

import elements
import numpy as np
import ruamel.yaml as yaml

from dreamerv3.main import make_agent
from embodied.run.protocol_v1 import array_hashes, parameter_digest


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', required=True)
  args = parser.parse_args()
  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  source = Path(__file__).resolve().parents[1] / 'dreamerv3' / 'configs.yaml'
  configs = yaml.YAML(typ='safe').load(source.read_text(encoding='utf-8'))
  config = elements.Config(configs['defaults']).update(configs['m2_v1'])
  config = config.update(dict(task='dmc_hopper_hop',
      logdir=str(output.parent), **{'jax.prealloc': False,
      'jax.profiler': False}))
  agent = make_agent(config)
  batch_size = config.batch_size
  length = config.batch_length + config.replay_context
  raw = agent._zeros(agent.spaces, (batch_size, length))
  rng = np.random.default_rng(12345)
  raw['image'][:] = rng.integers(
      0, 256, raw['image'].shape, dtype=np.uint8)
  raw['action'][:] = rng.uniform(
      -0.2, 0.2, raw['action'].shape).astype(np.float32)
  raw['reward'][:] = rng.uniform(
      -1, 1, raw['reward'].shape).astype(np.float32)
  raw['is_first'][:, 0] = True
  carry = agent.init_train(batch_size)
  result = dict(task=config.task, size='size50m', seed=config.seed,
      batch_size=batch_size, length=length,
      input_hashes=array_hashes(raw),
      initial_params_sha256=parameter_digest(agent), updates=[])
  for update_id in range(4):
    batch = agent.prepare_batch(raw)
    carry, _, _ = agent.train(carry, dict(batch))
    _, metrics = agent.take_train_result()
    result['updates'].append(dict(
        update_id=update_id,
        loss=float(metrics['opt/loss']),
        match_sum=float(metrics['dt/match_sum']),
        match_count=int(metrics['dt/match_count']),
        params_sha256=parameter_digest(agent),
        train_batches=int(agent.n_batches),
        train_updates=int(agent.n_updates)))
  output.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n',
                    encoding='utf-8')
  print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
  main()
