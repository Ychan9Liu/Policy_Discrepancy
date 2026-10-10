"""DL-code-r1: real CPU quadruped state-restoration engineering audit.

No rendering, GPU, checkpoint, training, gate, or method-performance test.
Repeats fixed-action/common-continuation consequences from complete snapshots
and compares actual stored float32 reward encoding without loosening double
precision repeat tolerance. JSON records file hashes and actual versions.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import platform
import time
from unittest import mock

import dm_control
import importlib.metadata
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('dl_diagnostic', ROOT / 'scripts/dl_diagnostic.py')
dl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dl)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', required=True)
  parser.add_argument('--horizon', type=int, choices=(10, 100), default=10,
      help='H100 is DL-opportunity-r1 mechanics only; H10 remains original audit')
  args = parser.parse_args()
  started = time.perf_counter()
  result = dict(code_version=dl.VERSION, git_sha=dl.git_sha(),
      source_sha256=dl.sha256(ROOT / 'scripts/dl_diagnostic.py'),
      fixture_sha256=dl.sha256(__file__), scope='CPU real physics restoration; no rendering or model',
      python=platform.python_version(), dm_control=importlib.metadata.version('dm_control'),
      mujoco=mujoco.__version__, numpy=np.__version__, tolerance=1e-8,
      horizon=args.horizon, seeds=[719, 1729, 20261010], positions=[1, 10, 50, 100], records=[])
  for seed in result['seeds']:
    env = dl.make_dmc('dmc_quadruped_walk', seed)
    try:
      env.reset()
      rng = np.random.default_rng([seed, 0x444C])
      actions = rng.uniform(-.5, .5, (102 + args.horizon, *env.action_spec().shape)).astype(np.float32)
      snapshots, rewards = [], []
      with mock.patch.object(env.physics, 'render', side_effect=AssertionError('Rendering prohibited')):
        for action in actions:
          snapshots.append(dl.physics_snapshot(env))
          ts = env.step(action)
          rewards.append(np.float32(ts.reward))
        for position in result['positions']:
          snapshot = snapshots[position]
          continuation = actions[position + 1:position + args.horizon]
          actual, error, digest = dl.verified_consequence(env, snapshot,
              actions[position], continuation, args.horizon)
          if not np.array_equal(np.asarray(actual, np.float32),
              np.asarray(rewards[position:position + args.horizon], np.float32)):
            raise AssertionError('Restored full-window label encoding differs from collected labels')
          candidates = np.concatenate([dl.fixed_actions(env.action_spec().shape),
              actions[position:position + 2]])
          for index, action in enumerate(candidates):
            sequence, repeat_error, endpoint = dl.verified_consequence(
                env, snapshot, action, continuation, args.horizon)
            # Independent prefix rollouts must exactly match the corresponding
            # prefix of the long rollout, from the complete same snapshot.
            prefixes = (1, 10, 50, 100) if args.horizon == 100 else (1, 10)
            prefix_errors = []
            for prefix in prefixes:
              shorter, shorter_error, _ = dl.verified_consequence(
                  env, snapshot, action, continuation, prefix)
              np.testing.assert_array_equal(shorter, sequence[:prefix])
              prefix_errors.append(shorter_error)
            result['records'].append(dict(seed=seed, position=position, candidate=index,
                integration_size=len(snapshot['integration']),
                repeat_error=max(error, repeat_error, *prefix_errors), true_h1=float(sequence[0]),
                true_h10_mean=float(sequence[:10].mean()),
                true_horizon_mean=float(sequence.mean()), true_horizon_sum=float(sequence.sum()),
                prefixes_exact=True, endstate_sha256=endpoint,
                reference_endstate_sha256=digest))
    finally:
      env.close()
  result.update(passed=True, checks=len(result['records']),
      maximum_repeat_error=max(x['repeat_error'] for x in result['records']),
      elapsed_seconds=time.perf_counter() - started,
      limitation='Real simulator mechanics only. No segmentation, frozen size50m, independent signal or gate evidence.')
  dl.write_json(args.output, result)
  print(json.dumps({k: result[k] for k in (
      'git_sha', 'passed', 'checks', 'maximum_repeat_error', 'elapsed_seconds', 'scope')}), flush=True)


if __name__ == '__main__':
  main()
