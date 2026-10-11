"""DL-source-r1: fresh production-action trajectories, never legacy relabeling.

The collector executes the actual DMC and main.wrap_env chain. Raw actor
samples remain the model's previous actions; final physical controls are
recorded separately. No optimizer or gate fitting occurs. Source verification
uses trusted project state pickles, not arbitrary external files.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_diagnostic as d
from scripts import dl_predictive_mc as pinned

VERSION = 'DL-source-r1'
IMPLEMENTATION = 'DL-source-engineering-r2'
CONTRACT = 'production_clip_unify_normalize_v1'
SEED, EPISODES, CALIBRATION, POSITIONS, HORIZON = 20261020, 32, 8, 16, 100
BUDGET = 3600


class RecordingSuite:
  """Transparent dm_env proxy records the argument actually received by suite."""
  def __init__(self, suite):
    self.suite = suite
    self.last_action = None
    self.steps = 0

  def __getattr__(self, name):
    return getattr(self.suite, name)

  def reset(self):
    self.last_action = None
    return self.suite.reset()

  def step(self, action):
    self.last_action = np.asarray(action).copy()
    self.steps += 1
    return self.suite.step(action)


class ProductionEnvironment:
  def __init__(self, suite):
    from dreamerv3.main import wrap_env
    from embodied.envs.dmc import DMC
    self.suite = suite
    self.recorder = RecordingSuite(suite)
    self.dmc = DMC(self.recorder, repeat=1, size=(64, 64),
        proprio=False, image=True, camera=2)
    self.env = wrap_env(self.dmc, None)
    # Exact current production chain: FromDM is the sole mutable step wrapper.
    self.from_dm = self.dmc._env.env
    if type(self.from_dm).__name__ != 'FromDM':
      raise ValueError('Unexpected production step wrapper state')
    spec = suite.action_spec()
    self.low, self.high = spec.minimum.copy(), spec.maximum.copy()
    self.chain = []
    current = self.env
    while current is not self.dmc:
      self.chain.append(type(current).__name__)
      current = current.env
    self.chain += ['DMC', type(self.dmc._env).__name__, 'FromDM']
    if self.chain != ['ClipAction', 'CheckSpaces', 'UnifyDtypes',
        'NormalizeAction', 'DMC', 'ActionRepeat', 'FromDM']:
      raise ValueError('Unreviewed production wrapper chain')

  def reset(self):
    return self.env.step(dict(reset=True, action=np.zeros(12, np.float32)))

  def step(self, raw):
    from scripts.dl_action_contract import production_action_stages
    stages = production_action_stages(raw, self.low, self.high)
    before = raw.copy()
    obs = self.env.step(dict(reset=False, action=raw))
    if (not np.array_equal(raw, before) or self.recorder.last_action is None or
        self.recorder.last_action.dtype != stages['applied'].dtype or
        self.recorder.last_action.tobytes() != stages['applied'].tobytes()):
      raise AssertionError('Actual wrapper action disagrees or mutated raw sample')
    return obs, self.recorder.last_action.copy()

  def snapshot(self):
    return dict(suite=d.physics_snapshot(self.suite),
        from_dm_done=bool(self.from_dm._done))

  def restore(self, snapshot):
    d.restore_physics(self.suite, snapshot['suite'])
    self.from_dm._done = bool(snapshot['from_dm_done'])
    self.recorder.last_action = None

  def close(self):
    self.env.close()


def state_error(actual, expected):
  a, b = actual['suite'], expected['suite']
  for state in (a['integration'], b['integration']):
    if (not isinstance(state, np.ndarray) or state.dtype != np.float64 or
        state.ndim != 1 or not np.isfinite(state).all()):
      raise AssertionError('Complete physics state must be finite float64 vector')
  if a['integration'].shape != b['integration'].shape:
    raise AssertionError('Complete physics state shape differs')
  if (actual['from_dm_done'] != expected['from_dm_done'] or
      a['step_count'] != b['step_count'] or
      a['reset_next_step'] != b['reset_next_step'] or
      pickle.dumps(a['task'], protocol=5) != pickle.dumps(b['task'], protocol=5)):
    raise AssertionError('Production task/counter/reset/wrapper restoration differs')
  value = float(np.max(np.abs(a['integration'] - b['integration'])))
  if value > 1e-8:
    raise AssertionError('Production complete integration restoration differs')
  return value


def host_copy(tree):
  import jax
  with jax._src.config.explicit_device_get_scope():
    return jax.tree.map(lambda x: np.asarray(x).copy(), tree)


def tree_digest(tree):
  """Numeric leaves with explicit container structure, dtype and shape."""
  digest = hashlib.sha256()
  def visit(node):
    if isinstance(node, dict):
      digest.update(b'dict')
      for key in sorted(node):
        digest.update(str(key).encode())
        visit(node[key])
    elif isinstance(node, (list, tuple)):
      digest.update(type(node).__name__.encode() + str(len(node)).encode())
      for value in node:
        visit(value)
    else:
      value = np.ascontiguousarray(node)
      if value.dtype.hasobject:
        raise ValueError('Object arrays are not carry evidence')
      digest.update(str(value.dtype).encode() + str(value.shape).encode())
      digest.update(value.tobytes())
  visit(tree)
  return digest.hexdigest()


def counters(agent):
  return dict(updates=int(agent.n_updates), batches=int(agent.n_batches),
      actions=int(agent.n_actions), eval_actions=int(agent.n_eval_actions),
      eval_seed=int(agent.eval_seed))


def verify_policy_records(agent, arrays, records):
  """Replay actual eval keys/carry on frozen RGB; restore eval counters afterward."""
  from embodied.jax import internal
  previous_seed, previous_counter = int(agent.eval_seed), int(agent.n_eval_actions)
  try:
    for record in records:
      t = record['position']
      if tree_digest(record['carry']) != record['carry_sha256']:
        raise AssertionError('Stored actual source carry hash differs')
      agent.set_eval_seed(record['eval_seed'])
      with agent.n_eval_actions.lock:
        agent.n_eval_actions.value = record['eval_counter']
      obs = {k: arrays[k][t][None] for k in
          ('image', 'reward', 'is_first', 'is_last', 'is_terminal')}
      # Public policy expects its split carry already on the policy devices.
      # Restoring trusted host leaves must be explicit under the unchanged
      # production transfer guard; retain every dtype, key and numeric value.
      device_carry = internal.device_put(record['carry'],
          internal.local_sharding(agent.policy_mirrored))
      after, action, _ = agent.policy(device_carry, obs, mode='eval')
      raw = np.asarray(action['action'][0])
      if (raw.dtype != arrays['action'].dtype or
          raw.tobytes() != arrays['action'][t].tobytes() or
          tree_digest(host_copy(after)) != record['after_carry_sha256']):
        raise AssertionError('Actual production source policy/carry/key replay differs')
  finally:
    agent.set_eval_seed(previous_seed)
    with agent.n_eval_actions.lock:
      agent.n_eval_actions.value = previous_counter
  return dict(positions=len(records), raw_sample_byte_exact=True,
      after_carry_byte_hash_exact=True, actual_eval_keys_replayed=True,
      eval_counter_restored=True, frozen_source_rgb_used=True)


def verify_reference_windows(adapter, arrays, snapshots, positions):
  """No render or model: exact source actions/rewards and complete end states."""
  from scripts.dl_action_contract import production_action_stages
  maximum = 0.
  for start in positions:
    adapter.restore(snapshots[start])
    if adapter.from_dm._done:
      raise ValueError('Reference starts at last/reset state')
    for t in range(int(start), int(start) + HORIZON):
      if not arrays['action_executed'][t]:
        raise ValueError('Reference window includes terminal no-action placeholder')
      applied = production_action_stages(arrays['action'][t],
          adapter.low, adapter.high)['applied']
      recorded = arrays['action_applied'][t]
      if applied.dtype != recorded.dtype or applied.tobytes() != recorded.tobytes():
        raise AssertionError('Recorded source raw/applied mapping differs')
      ts = adapter.suite.step(applied)
      adapter.from_dm._done = ts.last()
      if np.float32(ts.reward or 0).tobytes() != arrays['reward'][t + 1].tobytes():
        raise AssertionError('Source reward does not match production controls')
      if bool(ts.last()) != bool(arrays['is_last'][t + 1]):
        raise AssertionError('Source terminal alignment differs')
      if (bool(ts.first()) != bool(arrays['is_first'][t + 1]) or
          bool(ts.last() and ts.discount == 0) != bool(arrays['is_terminal'][t + 1])):
        raise AssertionError('Source first/terminal flags differ')
      maximum = max(maximum, state_error(adapter.snapshot(), snapshots[t + 1]))
  return dict(windows=len(positions), horizon=HORIZON,
      physics_steps=len(positions) * HORIZON, maximum_state_error=maximum,
      raw_applied_exact=True, rewards_float32_exact=True,
      integration_task_counter_reset_wrapper_verified=True)


def collect_episode(adapter, agent, entry, check_runtime=None, deadline=None):
  import mujoco
  agent.set_eval_seed(entry['policy_seed'])
  carry = agent.init_policy(1, mode='eval')
  obs = adapter.reset()
  if check_runtime:
    check_runtime()
  length = int(adapter.suite._step_limit) + 1
  if length != 1001:
    raise ValueError('Frozen source requires 1001-frame complete episodes')
  positions = d.sample_positions(length, POSITIONS, HORIZON)
  arrays = {key: [] for key in ('image', 'preserve', 'body', 'reward',
      'is_first', 'is_last', 'is_terminal', 'action', 'action_applied', 'action_executed')}
  snapshots, before_carries = [], []
  for t in range(length):
    if deadline is not None and time.monotonic() >= deadline:
      raise TimeoutError('DL-source-r1 hard 3600s budget exhausted')
    snapshots.append(adapter.snapshot())
    if t in positions:
      saved = host_copy(carry)
      before_carries.append(dict(position=t, carry=saved,
          carry_sha256=tree_digest(saved), eval_counter=int(agent.n_eval_actions),
          eval_seed=int(agent.eval_seed)))
    seg = adapter.suite.physics.render(64, 64, camera_id=2, segmentation=True)
    preserve, body = d.segmentation_masks(seg, adapter.suite.physics.model.geom_bodyid,
        int(mujoco.mjtObj.mjOBJ_GEOM))
    for key in obs:
      arrays[key].append(np.asarray(obs[key]).copy())
    arrays['preserve'].append(preserve)
    arrays['body'].append(body)
    if obs['is_last']:
      if t != length - 1:
        raise ValueError('Early terminal source; no episode replacement allowed')
      arrays['action'].append(np.zeros(12, np.float32))
      arrays['action_applied'].append(np.zeros(12, np.float64))
      arrays['action_executed'].append(False)
      break
    if t == length - 1:
      raise ValueError('Expected time limit absent at final source frame')
    carry, action, _ = agent.policy(carry,
        {k: np.asarray(v)[None] for k, v in obs.items()}, mode='eval')
    raw = np.asarray(action['action'][0]).copy()
    if raw.dtype != np.float32:
      raise ValueError('Actual actor sample must be float32; no silent dtype conversion')
    if t in positions:
      before_carries[-1]['after_carry_sha256'] = tree_digest(host_copy(carry))
    obs, applied = adapter.step(raw)
    arrays['action'].append(raw)
    arrays['action_applied'].append(applied)
    arrays['action_executed'].append(True)
  arrays = {k: np.asarray(v) for k, v in arrays.items()}
  if int(agent.n_eval_actions) != length - 1 or len(before_carries) != POSITIONS:
    raise AssertionError('Policy call/carry source coverage differs')
  proof = verify_reference_windows(adapter, arrays, snapshots, positions)
  proof['policy_replay'] = verify_policy_records(agent, arrays, before_carries)
  return arrays, snapshots, before_carries, positions, proof


def collect(args):
  from scripts.dl_resources import require_resource, verify_runtime
  pinned.require_clean()
  budget = json.loads((Path(args.output) / 'budget.json').read_text(encoding='utf-8'))
  if (budget['parent_pid'] != os.getppid() or
      budget['git_sha'] != d.git_sha() or budget['seconds'] != BUDGET or
      int(os.environ.get('DL_SOURCE_PARENT', '-1')) != os.getppid()):
    raise ValueError('Source worker requires its live bounded parent supervisor')
  resource = require_resource(args.resource_record)
  if resource['physical_gpu'] != 4:
    raise ValueError('DL-source-r1 is frozen to sv3 physical GPU4')
  from embodied.run.protocol_v1 import parameter_digest
  deadline = budget['monotonic_start'] + BUDGET
  agent, config, metadata = pinned.load_agent_no_env(args)
  metadata.update(code_version=IMPLEMENTATION, source_version=VERSION,
      space_contract='Production DMC spaces; collection executes actual DMC + main.wrap_env')
  if metadata['params_sha256'] != pinned.PARAMS_SHA:
    raise ValueError('Frozen 300k parameter digest differs')
  frozen = counters(agent)
  resource['runtime_compute'] = verify_runtime(resource, compute=True)
  plan = d.episode_plan(EPISODES, CALIBRATION, SEED)
  output = Path(args.output)
  source_files = ('scripts/dl_production_source.py', 'scripts/dl_action_contract.py',
      'dreamerv3/main.py', 'embodied/core/wrappers.py', 'embodied/core/driver.py',
      'embodied/envs/dmc.py', 'embodied/envs/from_dm.py', 'dreamerv3/agent.py',
      'embodied/jax/agent.py')
  d.write_json(output / 'collection_plan.json', dict(version=VERSION, implementation=IMPLEMENTATION,
      action_contract=CONTRACT, metadata=metadata, episodes=plan, seed=SEED,
      positions=POSITIONS, eligibility_horizon=HORIZON, expected_frames=1001,
      camera=2, shape=[64, 64], behavior='actual sampled eval policy; raw carry',
      carries='actual pre-policy carry at all 16 predetermined positions',
      source_files_sha256={name: d.sha256(ROOT / name) for name in source_files},
      environment=dict(python=sys.version, packages={name: importlib.metadata.version(name)
          for name in ('jax', 'jaxlib', 'numpy', 'dm-control', 'mujoco', 'elements', 'ninjax')}),
      terminal_action_placeholder='action_executed=False; never mapped or executed',
      budget_seconds=BUDGET, resource=resource, optimizer_updates=0,
      gate_changed=False, legacy_source_reused=False))
  artifacts = []
  for entry in plan:
    adapter = ProductionEnvironment(d.make_dmc(args.task, entry['seed']))
    try:
      runtime = (lambda: resource.update(runtime_cuda_egl=verify_runtime(
          resource, compute=True, graphics=True))) if not artifacts else None
      arrays, snapshots, carries, positions, proof = collect_episode(
          adapter, agent, entry, runtime, deadline)
      path = output / f'episode_{entry["episode"]:03}.npz'
      if path.exists():
        raise FileExistsError(path)
      np.savez_compressed(path, **arrays, positions=positions,
          snapshots=np.frombuffer(pickle.dumps(snapshots, protocol=5), np.uint8),
          before_carries=np.frombuffer(pickle.dumps(carries, protocol=5), np.uint8),
          action_low=adapter.low, action_high=adapter.high,
          action_contract=np.array(CONTRACT))
      artifacts.append(dict(**entry, path=path.name, sha256=d.sha256(path),
          frames=len(snapshots), wrapper_chain=adapter.chain, verification=proof,
          action_spec=dict(low=adapter.low.tolist(), high=adapter.high.tolist(),
              dtype=str(adapter.low.dtype), raw_dtype=str(arrays['action'].dtype)),
          eval_counters=counters(agent), carry_positions=positions.tolist(),
          carry_sha256=[x['carry_sha256'] for x in carries]))
      print(json.dumps(dict(stage='source', episode=entry['episode'],
          split=entry['split'], frames=len(snapshots), verification=proof)), flush=True)
    finally:
      adapter.close()
  final = counters(agent)
  if (parameter_digest(agent) != metadata['params_sha256'] or
      any(final[k] != frozen[k] for k in ('updates', 'batches', 'actions'))):
    raise AssertionError('Frozen training parameter/state counters changed')
  d.write_json(output / 'collection_complete.json', dict(version=VERSION, implementation=IMPLEMENTATION,
      action_contract=CONTRACT, artifacts=artifacts, params_unchanged=True,
      training_counters_unchanged=True, eval_counters_legitimately_advanced=True,
      counters_before=frozen, counters_after=final, resource=resource,
      plan_sha256=d.sha256(output / 'collection_plan.json'),
      optimizer_updates=0, method_effectiveness_established=False))


def run_bounded(command, output, deadline):
  """Hard wall-time bound covering loading, JIT, reference replay and writes."""
  output = Path(output)
  with (output / 'source.log').open('x', encoding='utf-8') as log:
    child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=os.name != 'nt', env=dict(os.environ,
            DL_SOURCE_PARENT=str(os.getpid())))
    timed_out = False
    try:
      code = child.wait(timeout=max(.001, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
      timed_out = True
      if os.name != 'nt':
        os.killpg(child.pid, signal.SIGKILL)  # Only this newly owned process group.
      else:
        child.kill()
      child.wait()
      code = 124
  (output / 'worker_exit_code.txt').write_text(str(code) + '\n', encoding='utf-8')
  return dict(worker_pid=child.pid, exit_code=code, timed_out=timed_out,
      worker_terminal=child.poll() is not None)


def supervise(args):
  pinned.require_clean()
  output = Path(args.output).resolve()
  if output.exists():
    raise FileExistsError('New source attempt output must not exist')
  output.mkdir(parents=True)
  started = time.monotonic()
  d.write_json(output / 'budget.json', dict(version=VERSION,
      parent_pid=os.getpid(), git_sha=d.git_sha(), seconds=BUDGET,
      monotonic_start=started, script_sha256=d.sha256(Path(__file__))))
  command = [sys.executable, str(Path(__file__).resolve()), '--worker',
      '--checkpoint', args.checkpoint, '--config', args.config,
      '--output', str(output), '--resource-record', args.resource_record,
      '--task', args.task, '--platform', args.platform]
  receipt = run_bounded(command, output, started + BUDGET)
  receipt.update(version=VERSION, git_sha=d.git_sha(),
      budget_sha256=d.sha256(output / 'budget.json'),
      elapsed_seconds=time.monotonic() - started,
      method_effectiveness_established=False)
  d.write_json(output / 'supervisor_complete.json', receipt)
  print(json.dumps(receipt), flush=True)
  if receipt['exit_code']:
    raise SystemExit(receipt['exit_code'])


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--config', required=True)
  parser.add_argument('--output', required=True)
  parser.add_argument('--resource-record', required=True)
  parser.add_argument('--task', default='dmc_quadruped_walk',
      choices=['dmc_quadruped_walk'])
  parser.add_argument('--platform', default='cuda', choices=['cuda'])
  parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
  args = parser.parse_args()
  (collect if args.worker else supervise)(args)


if __name__ == '__main__':
  main()
