"""CPU evidence for the production raw-action / physical-control contract.

The actual main.wrap_env chain is the oracle. This helper neither fixes old
datasets nor establishes a research result. No checkpoint or RGB render is used.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VERSION = 'DL-action-contract-r1'


def array_hash(value):
  value = np.ascontiguousarray(value)
  return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() +
      value.tobytes()).hexdigest()


def production_action_stages(raw, low, high):
  """Mirror the *ordered dtype operations*, without mutating raw or carry.

  Do not substitute an algebraically equivalent all-float64 expression. The
  normalized Space supplied to UnifyDtypes is float32; real DMC bounds are
  float64, so promotion happens only after (float32_action + 1) / 2.
  The caller must still use actual production wrappers for environment steps.
  """
  raw, low, high = np.asarray(raw), np.asarray(low), np.asarray(high)
  if (raw.dtype != np.float32 or raw.ndim != 1 or
      raw.shape != low.shape or low.shape != high.shape or
      not np.issubdtype(low.dtype, np.floating) or low.dtype != high.dtype or
      not all(np.isfinite(x).all() for x in (raw, low, high)) or
      not np.all(low < high)):
    raise ValueError('Require finite raw float32 and matching finite float bounds')
  mask = np.isfinite(low) & np.isfinite(high)
  inner_low, inner_high = np.where(mask, low, -1), np.where(mask, high, 1)
  clipped = np.clip(raw, -1, 1)
  unified = np.asarray(clipped, np.float32)
  applied = (unified + 1) / 2 * (inner_high - inner_low) + inner_low
  applied = np.where(mask, applied, unified)
  return {name: value.copy() for name, value in dict(
      raw=raw, clipped=clipped, unified=unified, applied=applied).items()}


def control_cases(size=12):
  """Predeclared cases independent of simulator outcome or any model signal."""
  if size != 12:
    raise ValueError('This audit is frozen to quadruped action12')
  odd = np.array([.1, -.1, .3, -.3, .12345679, -.12345679,
      np.nextafter(np.float32(0), np.float32(1)),
      np.nextafter(np.float32(0), np.float32(-1)),
      np.nextafter(np.float32(1), np.float32(0)),
      np.nextafter(np.float32(-1), np.float32(0)), .9, -.9], np.float32)
  return dict(zero=np.zeros(size, np.float32),
      normalized_low=np.full(size, -1, np.float32),
      normalized_high=np.ones(size, np.float32),
      below_normalized=np.full(size, -2.75, np.float32),
      above_normalized=np.full(size, 2.75, np.float32),
      mixed_outside=np.linspace(-3, 3, size, dtype=np.float32),
      rounding=np.full(size, .1, np.float32), odd_floats=odd)


class SuiteActionRecorder:
  """Transparent dm_env proxy records the control actually passed to suite."""
  def __init__(self, env):
    self.env = env
    self.last_action = None
    self.step_calls = 0
    self.reset_calls = 0

  def __getattr__(self, name):
    return getattr(self.env, name)

  def step(self, action):
    self.last_action = np.asarray(action).copy()
    self.step_calls += 1
    return self.env.step(action)

  def reset(self):
    self.reset_calls += 1
    return self.env.reset()


def wrap_recorded_suite(env):
  """The real action path, omitting DMC's unconditional image render only."""
  import embodied
  from embodied.envs.from_dm import FromDM
  from dreamerv3.main import wrap_env
  recorder = SuiteActionRecorder(env)
  leaf = FromDM(recorder)
  wrapped = wrap_env(embodied.wrappers.ActionRepeat(leaf, 1), None)
  return recorder, wrapped


def make_wrapped_suite(seed=20261020):
  from dm_control import suite
  env = suite.load('quadruped', 'walk', task_kwargs={'random': int(seed)})
  recorder, wrapped = wrap_recorded_suite(env)
  return env, recorder, wrapped


def snapshot_receipt(snapshot):
  task = pickle.dumps(snapshot['task'], protocol=5)
  return dict(integration_sha256=array_hash(snapshot['integration']),
      task_sha256=hashlib.sha256(task).hexdigest(),
      step_count=int(snapshot['step_count']), reset_next_step=bool(snapshot['reset_next_step']))


def _bytes_equal(a, b):
  return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def run_cpu_audit(seed=20261020):
  """Actual CPU physics: oracle vs exact mapping vs legacy direct raw control."""
  from scripts import dl_diagnostic as d
  env, recorder, wrapped = make_wrapped_suite(seed)
  try:
    spec = env.action_spec()
    if spec.shape != (12,) or spec.dtype != np.float64:
      raise ValueError('Actual quadruped action spec differs from audit contract')
    if not np.array_equal(np.stack((spec.minimum, spec.maximum), -1),
        env.physics.model.actuator_ctrlrange):
      raise AssertionError('dm_env bounds differ from actual actuator ctrlrange')
    # Main wrap_env reset semantics, before any nonterminal comparison.
    wrapped.step(dict(reset=np.bool_(True), action=np.zeros(12, np.float32)))
    initial = d.physics_snapshot(env)
    if wrapped._done or initial['reset_next_step']:
      raise AssertionError('Oracle comparison requires a normal nonterminal state')
    cases = []
    for name, raw in control_cases().items():
      before = raw.copy()
      stages = production_action_stages(raw, spec.minimum, spec.maximum)
      d.restore_physics(env, initial)
      obs = wrapped.step(dict(reset=np.bool_(False), action=raw))
      physical = recorder.last_action.copy()
      oracle = d.physics_snapshot(env)
      oracle_qpos, oracle_qvel = env.physics.data.qpos.copy(), env.physics.data.qvel.copy()
      if not _bytes_equal(physical, stages['applied']) or not _bytes_equal(raw, before):
        raise AssertionError('Wrapper control mismatch or raw input mutation')
      d.restore_physics(env, initial)
      expected_step = env.step(stages['applied'])
      expected = d.physics_snapshot(env)
      if (snapshot_receipt(oracle) != snapshot_receipt(expected) or
          obs['reward'].tobytes() != np.float32(expected_step.reward).tobytes()):
        raise AssertionError('Full-state physical consequence differs from wrapper oracle')
      d.restore_physics(env, initial)
      legacy_step = env.step(raw)
      legacy = d.physics_snapshot(env)
      cases.append(dict(name=name,
          stages={k: dict(dtype=str(v.dtype), values=v.tolist(), sha256=array_hash(v))
              for k, v in stages.items()},
          wrapper_applied_sha256=array_hash(physical), mapping_byte_exact=True,
          raw_unchanged=True, explicit_physics_full_state_exact=True,
          legacy_raw_control_differs=bool(np.any(physical != raw)),
          legacy_max_integration_difference=float(np.max(np.abs(
              oracle['integration'] - legacy['integration']))),
          legacy_max_qpos_difference=float(np.max(np.abs(oracle_qpos - env.physics.data.qpos))),
          legacy_max_qvel_difference=float(np.max(np.abs(oracle_qvel - env.physics.data.qvel))),
          oracle_reward=float(obs['reward']), legacy_reward=float(legacy_step.reward),
          oracle=snapshot_receipt(oracle), legacy=snapshot_receipt(legacy)))
    versions = {}
    for name in ('numpy', 'dm-control', 'mujoco', 'elements'):
      versions[name] = importlib.metadata.version(name)
    files = ('dreamerv3/main.py', 'embodied/core/wrappers.py',
        'embodied/envs/from_dm.py', 'dreamerv3/agent.py', 'scripts/dl_action_contract.py')
    result = dict(version=VERSION, task='dmc_quadruped_walk', seed=int(seed),
        platform='CPU physics only; no RGB render or JAX execution',
        git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_file_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
        python=sys.version, dependencies=versions,
        action_spec=dict(dtype=str(spec.dtype), shape=list(spec.shape),
            minimum=spec.minimum.tolist(), maximum=spec.maximum.tolist(),
            actuator_names=[env.physics.model.id2name(i, 'actuator') for i in range(12)]),
        execution_order=['ClipAction', 'CheckSpaces', 'UnifyDtypes', 'NormalizeAction',
            'ActionRepeat(1)', 'FromDM', 'suite.step'],
        normalized_input_dtype=str(wrapped.act_space['action'].dtype),
        physics_step_calls=recorder.step_calls + 2*len(cases),
        nonidentity_actuator_bounds=int(np.sum((spec.minimum != -1) | (spec.maximum != 1))),
        cases=cases, all_mapping_byte_exact=True, all_raw_inputs_unchanged=True,
        all_explicit_physics_full_state_exact=True,
        old_artifacts_modified=False, training_authorized=False,
        scope='Normal nonterminal single transitions from the same reset state. '
            'Suite task/integration/counters are compared; this is not a complete wrapper '
            'snapshot restore protocol. Agent raw-carry behavior is a separate CPU fixture. '
            'No claim that old direct-raw-suite history is production history.')
    return result
  finally:
    env.close()


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', required=True, type=Path,
      help='New JSON file (existing output is never overwritten)')
  parser.add_argument('--seed', type=int, default=20261020)
  args = parser.parse_args(argv)
  if args.output.exists():
    raise FileExistsError(args.output)
  result = run_cpu_audit(args.seed)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open('x', encoding='utf-8') as file:
    json.dump(result, file, ensure_ascii=False, sort_keys=True, indent=2)
    file.write('\n')
  print(json.dumps(dict(output=str(args.output), version=VERSION,
      nonidentity_actuator_bounds=result['nonidentity_actuator_bounds'],
      all_mapping_byte_exact=True, all_explicit_physics_full_state_exact=True)))


if __name__ == '__main__':
  main()
