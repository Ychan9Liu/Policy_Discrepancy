"""DL-r4-utility-r1: bounded, frozen N1M action/memory diagnostic.

Only standard library at entry. No training, checkpoint rewrite, remote access,
device selection, or experiment is performed merely by importing this module.
"""
import time
ENTRY_STARTED = time.monotonic()
import os
import signal
import sys

VERSION = 'DL-r4-utility-r1'
SECONDS = 21000
DEADLINE = ENTRY_STARTED + SECONDS
OWNED = set()


def expire(signum, frame):
  for pid in tuple(OWNED):
    try: os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError: pass
  os._exit(124)


if __name__ == '__main__' and os.name == 'posix':
  signal.signal(signal.SIGALRM, expire)
  signal.signal(signal.SIGTERM, expire)
  signal.setitimer(signal.ITIMER_REAL, max(.001, DEADLINE-time.monotonic()))

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import re
import socket
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CHECKPOINT_SHA = 'c6a7d92e24669a515bb44693faf73e442d9868f9fc814eec0e22e069aa594d24'
CONFIG_SHA = '877f8470cb4f7f45d143e3099fbd3f55a828991afe33779b7b04c4264f6c8377'
PARAMS_SHA = 'cffc13c0dcdfbd0c8b0e5c503c9d6251f4b1550c08b85a8742a638655d66d372'
EXPECTED_COUNTERS = dict(actions=62563, batches=249493, updates=249493,
    eval_actions=1001, eval_seed=1289196729)
POSITIONS = tuple(range(64, 897, 64))
SOURCE_LIMIT, QUALIFICATION_LIMIT, LABEL_LIMIT = 1800, 1200, 18000


def require(condition, message):
  if not condition: raise ValueError(message)


def check():
  require(time.monotonic() < DEADLINE, 'Whole DL-r4 entry deadline exceeded')


def arm_phase(seconds):
  """A phase has a real alarm as well as cooperative checks; never extends run."""
  global DEADLINE
  previous = DEADLINE
  require(seconds > 0, 'Cumulative phase budget exhausted')
  DEADLINE = min(previous, time.monotonic()+seconds)
  if os.name == 'posix':
    signal.setitimer(signal.ITIMER_REAL, max(.001, DEADLINE-time.monotonic()))
  return previous


def restore_phase(previous):
  global DEADLINE
  require(time.monotonic() < DEADLINE, 'Phase deadline exceeded before restore')
  DEADLINE = previous
  if os.name == 'posix':
    signal.setitimer(signal.ITIMER_REAL, max(.001, DEADLINE-time.monotonic()))


def sha(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
  return digest.hexdigest()


def read_json(path):
  def pairs(items):
    result = {}
    for key, value in items:
      require(key not in result, 'Duplicate JSON key'); result[key] = value
    return result
  return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs,
      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def write_json(path, value):
  check()
  with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write('\n')


def save_pickle(path, value):
  check()
  with Path(path).open('xb') as stream: pickle.dump(value, stream, protocol=5)
  check()
  return dict(path=Path(path).name, sha256=sha(path))


def spawn_owned(command, deadline, *, own_group=True, **kwargs):
  require(time.monotonic() < deadline, 'Late process creation')
  if os.name != 'posix':
    child = subprocess.Popen(command, **kwargs)
    if own_group: OWNED.add(child.pid)
    return child
  previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM, signal.SIGTERM})
  try:
    child = subprocess.Popen(command, start_new_session=own_group,
        preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, previous), **kwargs)
    if own_group: OWNED.add(child.pid)
    return child
  finally: signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def cleanup(child):
  if os.name == 'posix':
    try: os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError: pass
  elif child.poll() is None: child.kill()
  child.wait(timeout=5)
  OWNED.discard(child.pid)


def owned_output(command):
  own_group = 'DL_R4_PARENT' not in os.environ
  child = spawn_owned(command, DEADLINE, own_group=own_group,
      cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  try:
    out, error = child.communicate(timeout=min(30, max(.001, DEADLINE-time.monotonic())))
    require(child.returncode == 0, 'Metadata command failed: '+error.decode(errors='replace')[:256])
    check(); return out
  finally:
    if own_group: cleanup(child)
    elif child.poll() is None: child.kill(); child.wait(timeout=5)
    for pipe in (child.stdout,child.stderr):
      if pipe is not None: pipe.close()


def git_binding(expected):
  require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{40}', expected), 'Full Git SHA required')
  require(owned_output(['git', 'rev-parse', 'HEAD']).decode().strip() == expected, 'Git HEAD differs')
  require(not owned_output(['git', 'status', '--porcelain', '--untracked-files=all']), 'Clean checkout required')
  names = owned_output(['git', 'ls-files']).decode().splitlines()
  names = [x for x in names if x.endswith(('.py', '.yaml', '.yml', '.lock')) or
      x in ('AGENTS.md', 'docs/DL_R4_UTILITY.md')]
  required = {'scripts/dl_r4_utility.py', 'scripts/dl_r4_policy.py',
      'scripts/dl_r4_physics.py', 'analysis/dl_r4_statistics.py', 'docs/DL_R4_UTILITY.md'}
  require(required <= set(names), 'Tracked r4 implementation incomplete')
  # git show per file has a bounded owned helper; authenticates checkout bytes,
  # not merely index cleanliness or a plausible HEAD string.
  pins = {}
  for name in names:
    blob = owned_output(['git', 'show', expected+':'+name])
    require((ROOT/name).read_bytes() == blob, 'Checkout/blob mismatch: '+name)
    pins[name] = hashlib.sha256(blob).hexdigest()
  return pins


def authenticate_inputs(checkpoint, config):
  checkpoint, config = Path(checkpoint).resolve(strict=True), Path(config).resolve(strict=True)
  require(sha(checkpoint) == CHECKPOINT_SHA, 'N1M checkpoint bytes differ')
  require(sha(config) == CONFIG_SHA, 'N1M configuration bytes differ')
  return dict(checkpoint_path=str(checkpoint), checkpoint_sha256=CHECKPOINT_SHA,
      config_path=str(config), config_sha256=CONFIG_SHA)


def load_agent(checkpoint, config_path, output):
  """Authenticate original bytes before pickle decode or model construction."""
  pins = authenticate_inputs(checkpoint, config_path)
  data = Path(checkpoint).read_bytes()
  require(hashlib.sha256(data).hexdigest() == CHECKPOINT_SHA, 'Checkpoint changed before decode')
  payload = pickle.loads(data)  # SHA-pinned, trusted project artifact only.
  del data
  from scripts.dl_reset_learning import tree_digest
  require(set(payload) == {'params', 'counters'} and payload['counters'] == EXPECTED_COUNTERS,
      'Full N1M counters differ')
  require(len(payload['params']) == 294, 'Full N1M leaf count differs')
  original_state = tree_digest(payload)
  from scripts.dl_diagnostic import load_config
  from scripts.dl_predictive_mc import agent_from_spaces
  from embodied.run.protocol_v1 import parameter_digest
  config = load_config(config_path, 'dmc_quadruped_walk', output, 'cuda')
  require(config.seed == 0 and str(config.jax.compute_dtype) == 'bfloat16', 'N1M model seed/dtype differs')
  agent = agent_from_spaces(config)
  agent.load(payload)
  require(tree_digest(agent.save()) == original_state, 'Loaded full params/optimizer/norm/counters differ')
  require(parameter_digest(agent) == PARAMS_SHA, 'Loaded N1M parameter digest differs')
  del payload
  pins.update(full_state_sha256=original_state, params_sha256=PARAMS_SHA,
      config=config.flat, counters=EXPECTED_COUNTERS,
      packages={x: importlib.metadata.version(x) for x in
          ('jax', 'jaxlib', 'numpy', 'mujoco', 'dm-control', 'ninjax', 'elements')})
  return agent, pins


def restore_eval(agent, saved):
  agent.set_eval_seed(saved['eval_seed'])
  with agent.n_eval_actions.lock: agent.n_eval_actions.value = saved['eval_actions']


def select_positions(candidates):
  require([x['position'] for x in candidates] == list(POSITIONS), 'All 14 ordinary candidates required')
  require(all(type(x['D']) in (int, float) and 0 <= x['D'] < float('inf') for x in candidates),
      'Candidate D must be finite nonnegative')
  top = [x['position'] for x in sorted(candidates, key=lambda x: (-x['D'], x['position']))[:2]]
  return sorted(set(top + [512])), top


def collect_episode(agent, adapter, episode_id, runtime_check=None):
  """Actual public eval policy; full source arrays, no segmentation/render extras."""
  import numpy as np
  from scripts.dl_feedback_model import host_copy, tree_digest, counters
  require(type(episode_id) is int and 0 <= episode_id < 32, 'New source episode namespace required')
  saved = counters(agent)
  arrays = {k: [] for k in ('image', 'reward', 'reward_f64', 'is_first', 'is_last', 'is_terminal',
      'action', 'action_applied')}
  snapshots, records = [], []
  try:
    agent.set_eval_seed(4096+episode_id)
    carry = agent.init_policy(1, mode='eval')
    obs = adapter.reset()
    if runtime_check: runtime_check()
    for t in range(1001):
      check()
      snapshots.append(adapter.snapshot())
      for k in ('image', 'reward', 'is_first', 'is_last', 'is_terminal'):
        arrays[k].append(np.asarray(obs[k]).copy())
      genuine_reward = np.float64(0) if t == 0 else np.asarray(adapter.recorder.last_timestep.reward)
      require(genuine_reward.dtype == np.float64 and genuine_reward.shape == () and
          np.isfinite(genuine_reward) and np.float32(genuine_reward).tobytes() == obs['reward'].tobytes(),
          'Actual source suite F64 reward or its production encoding differs')
      arrays['reward_f64'].append(genuine_reward.copy())
      require(bool(obs['is_first']) == (t == 0) and bool(obs['is_last']) == (t == 1000)
          and not bool(obs['is_terminal']), 'New source reset/time-limit/terminal differs')
      if t == 1000: break
      if t in POSITIONS:
        before = host_copy(carry)
        records.append(dict(position=t, carry=before, carry_sha256=tree_digest(before),
            eval_seed=4096+episode_id, eval_counter=int(agent.n_eval_actions)))
      carry, actions, _ = agent.policy(carry, {k: np.asarray(v)[None] for k, v in obs.items()}, mode='eval')
      raw = np.asarray(actions['action'][0]).copy()
      require(raw.dtype == np.float32 and raw.shape == (12,) and np.isfinite(raw).all(),
          'Source actual raw action invalid')
      if t in POSITIONS: records[-1]['after_carry_sha256'] = tree_digest(host_copy(carry))
      obs, applied = adapter.step(raw)
      arrays['action'].append(raw); arrays['action_applied'].append(applied)
    require(int(agent.n_eval_actions) == 1000 and len(records) == 14, 'Source key/candidate coverage differs')
  finally:
    restore_eval(agent, saved)
    require(counters(agent) == saved, 'Source public counters not restored')
  arrays = {k: np.asarray(v) for k, v in arrays.items()}
  arrays['action_executed'] = np.ones(1000, bool)
  return dict(episode_id=episode_id, env_seed=4096+episode_id,
      arrays=arrays, snapshots=snapshots, records=records)


def verify_source_reference(adapter, collected):
  """Actual stored controls, F64/F32 rewards and complete source state replay."""
  import numpy as np
  from scripts.dl_action_contract import production_action_stages
  from scripts.dl_r4_physics import _snapshot_exact
  a, snapshots = collected['arrays'], collected['snapshots']
  steps = 0
  for t in POSITIONS:
    check()
    adapter.restore(snapshots[t])
    _snapshot_exact(adapter.snapshot(), snapshots[t])
    for offset in range(100):
      check()
      i = t+offset
      applied = production_action_stages(a['action'][i],adapter.low,adapter.high)['applied']
      require(applied.dtype == a['action_applied'].dtype and
          applied.tobytes() == a['action_applied'][i].tobytes(), 'Source raw/applied reference differs')
      ts = adapter.suite.step(applied.copy())
      actual = np.asarray(ts.reward)
      require(actual.dtype == np.float64 and actual.shape == () and
          actual.tobytes() == a['reward_f64'][i+1].tobytes() and
          np.float32(actual).tobytes() == a['reward'][i+1].tobytes(), 'Source F64/F32 reference reward differs')
      adapter.from_dm._done = bool(ts.last())
      require(tuple(bool(a[k][i+1]) for k in ('is_first','is_last','is_terminal')) ==
          (ts.first(),ts.last(),ts.last() and ts.discount == 0), 'Source reference flags differ')
      _snapshot_exact(adapter.snapshot(), snapshots[i+1])
      steps += 1
  return dict(windows=14, horizon=100, physics_steps=steps,
      reward_float64_byte_exact=True, reward_float32_byte_exact=True,
      raw_applied_mapping_byte_exact=True, full_snapshot_byte_exact=True,
      policy_replayed=False, renderer_replayed=False)


def worker(args):
  global DEADLINE
  require(os.name == 'posix' and socket.gethostname().split('.')[0] == 'lyg0326', 'sv3 Linux required')
  output = Path(args.output).resolve(strict=True)
  budget = read_json(output/'budget.json')
  require(budget['parent_pid'] == os.getppid() == int(os.environ.get('DL_R4_PARENT', '-1')) and
      budget['version'] == VERSION and budget['seconds'] == SECONDS and
      budget['git_sha'] == args.expected_git, 'Live bounded parent required')
  os.kill(os.getppid(), 0)
  DEADLINE = budget['monotonic_deadline']
  signal.setitimer(signal.ITIMER_REAL, max(.001, DEADLINE-time.monotonic()))
  pins = git_binding(args.expected_git)
  require(pins == budget['source_files'], 'Parent/worker source bytes differ')
  from scripts.dl_resources import require_resource, verify_runtime
  resource = require_resource(args.resource_record)
  require(resource['physical_gpu'] == 4, 'This finite object is assigned sv3 physical4')
  qualification_start = time.monotonic()
  whole_deadline = arm_phase(QUALIFICATION_LIMIT)
  agent, metadata = load_agent(args.checkpoint, args.config, output)
  from scripts.dl_r4_policy import R4Policy, source_root_key, branch_root_key, branch_eval_seed
  from scripts import dl_r4_physics as physics
  from scripts.dl_feedback_physics import FeedbackEnvironment
  from scripts import dl_diagnostic as d
  from scripts.dl_feedback_model import tree_digest, counters
  from analysis.dl_r4_statistics import analyze, freeze_calibration
  import numpy as np
  policy = R4Policy(agent)
  initial_counter = counters(agent)
  source_seconds = labels_seconds = 0.
  qualification_seconds = time.monotonic()-qualification_start
  require(qualification_seconds < QUALIFICATION_LIMIT, 'Model qualification deadline exceeded')
  restore_phase(whole_deadline)
  write_json(output/'model_authenticated.json', dict(version=VERSION, metadata=metadata,
      model_qualification_seconds=qualification_seconds, resource=resource))
  episodes, evidence = [], []
  frozen_models = None
  for episode_id in range(32):
    check()
    episode_started = time.monotonic()
    if episode_id == 8:
      require(frozen_models is not None and (output/'calibration_frozen.json').is_file(),
          'Held data must not open before cal fit seal')
    epdir = output/f'episode_{episode_id:03}'
    epdir.mkdir()
    adapter = None
    try:
      started = time.monotonic()
      whole_deadline = arm_phase(SOURCE_LIMIT-source_seconds)
      adapter = FeedbackEnvironment(d.make_dmc('dmc_quadruped_walk', 4096+episode_id))
      collected = collect_episode(agent, adapter, episode_id,
          (lambda: write_json(output/'runtime_cuda_egl.json', verify_runtime(resource,
              compute=True, graphics=True))) if episode_id == 0 else None)
      source_seconds += time.monotonic()-started
      require(source_seconds < SOURCE_LIMIT, 'Cumulative source budget exhausted')
      restore_phase(whole_deadline)
      source_entry = save_pickle(epdir/'source.pkl', collected)
      policy.assert_frozen(full=True)
      visual = physics.visual_model_identity(adapter)
      candidates, bundles, oracles = [], {}, []
      a = collected['arrays']
      started = time.monotonic()
      whole_deadline = arm_phase(LABEL_LIMIT-labels_seconds)
      for record in collected['records']:
        t = record['position']
        obs = {k: np.asarray(a[k][t]).copy() for k in ('image','reward','is_first','is_last','is_terminal')}
        bundle = policy.current(record['carry'], obs, source_root_key(episode_id, t))
        # Source raw/after and public oracle are separate byte checks.
        require(bundle['q_raw'].tobytes() == a['action'][t].tobytes() and
            tree_digest(bundle['q_after']) == record['after_carry_sha256'], 'Actual source q adapter differs')
        oracles.append(policy.public_oracle(record['carry'], obs, 4096+episode_id, t, bundle=bundle))
        signal_values = policy.score_h1(bundle, a['action'][t], a['reward_f64'][t+1], valid=True)
        candidates.append(dict(position=t, D=float(bundle['Dmode']), K=float(bundle['K']),
            e=float(signal_values['e'])))
        bundles[t] = dict(bundle=bundle, h1=signal_values)
      selected, top2 = select_positions(candidates)
      signals_entry = save_pickle(epdir/'signals.pkl', dict(bundles=bundles, public_oracles=oracles,
          candidates=candidates, selected=selected, top2=top2, visual_identity=visual))
      # Source reference physics at all ordinary windows; actual production
      # actions/control mapping/rewards/full state, without model or render.
      source_proof = verify_source_reference(adapter, collected)
      rows, branches = [], []
      records_by_t = {x['position']: x for x in collected['records']}
      for t in selected:
        initial_obs = {k: np.asarray(a[k][t]).copy() for k in ('image','reward','is_first','is_last','is_terminal')}
        row = dict(position=t, returns={arm: [] for arm in ('qq','pq','pp')})
        for pair in range(16):
          check()
          keys = np.stack([branch_root_key(episode_id, t, pair, offset) for offset in range(100)])
          bundle = policy.current(records_by_t[t]['carry'], initial_obs, keys[0])
          if pair == 0:
            branch_oracle = policy.public_oracle(records_by_t[t]['carry'], initial_obs,
                branch_eval_seed(episode_id,t,pair), 0, bundle=bundle)
            save_pickle(epdir/f't{t:03}_branch_public_oracle.pkl', branch_oracle)
          pair_records = {}
          for arm in physics.branch_order(pair):
            rec = physics.run_branch(policy, adapter, collected['snapshots'][t],
                records_by_t[t]['carry'], keys, arm=arm, bundle=bundle,
                initial_obs=initial_obs, visual_identity=visual, check=check)
            phys = physics.verify_physics_replay(adapter, rec)
            cached = physics.replay_cached_policy(policy, rec)
            pair_records[arm] = rec
            artifact = save_pickle(epdir/f't{t:03}_pair{pair:02}_{arm}.pkl', rec)
            branches.append(dict(position=t, pair=pair, arm=arm, **artifact,
                physics=phys, cached_policy=cached))
            row['returns'][arm].append(float(np.sum(rec['reward_f64'], dtype=np.float64)))
          physics.validate_pair(pair_records)
          if pair == 0:
            null = physics.run_branch(policy, adapter, collected['snapshots'][t],
                records_by_t[t]['carry'], keys, arm='qq', bundle=bundle,
                initial_obs=initial_obs, visual_identity=visual, check=check)
            phys = physics.verify_physics_replay(adapter, null)
            cached = physics.replay_cached_policy(policy, null)
            artifact = save_pickle(epdir/f't{t:03}_pair00_qq_null.pkl', null)
            branches.append(dict(position=t, pair=0, arm='qq_null', **artifact,
                physics=phys, cached_policy=cached))
            row['qq_null_return'] = float(np.sum(null['reward_f64'], dtype=np.float64))
            row['null_max_abs_step_reward_diff'] = float(np.max(np.abs(
                null['reward_f64']-pair_records['qq']['reward_f64'])))
        rows.append(row)
        write_json(epdir/f't{t:03}_summary.json', row)
      labels_seconds += time.monotonic()-started
      require(labels_seconds < LABEL_LIMIT, 'Cumulative labels/replay budget exhausted')
      restore_phase(whole_deadline)
      policy.assert_frozen(full=True)
      require(counters(agent) == initial_counter, 'Frozen counters changed')
      episode = dict(episode_id=episode_id, env_seed=4096+episode_id,
          candidates=candidates, selected=rows)
      receipt = dict(version=VERSION, episode_id=episode_id, source=source_entry,
          signals=signals_entry, source_reference=source_proof, branches=branches,
          branch_public_oracles=[dict(path=f't{t:03}_branch_public_oracle.pkl',
              sha256=sha(epdir/f't{t:03}_branch_public_oracle.pkl')) for t in selected],
          episode_started_monotonic=episode_started, episode_completed_monotonic=time.monotonic(),
          source_seconds_cumulative=source_seconds, labels_seconds_cumulative=labels_seconds,
          model_full_state_unchanged=True)
      write_json(epdir/'complete.json', receipt)
      episodes.append(episode)
      evidence.append(dict(path=f'{epdir.name}/complete.json', sha256=sha(epdir/'complete.json')))
      if episode_id == 7:
        frozen_models = freeze_calibration(episodes)
        write_json(output/'calibration_frozen.json', dict(version=VERSION,
            calibration_evidence=evidence.copy(), frozen_models=frozen_models,
            calibration_episodes=episodes.copy(), held_episodes_generated=0,
            sealed_monotonic=time.monotonic()))
        require(frozen_models['null_passed'], 'Calibration null measurement failed; held remains unopened')
      print(json.dumps(dict(event='episode_complete', episode_id=episode_id,
          source_seconds=source_seconds, labels_seconds=labels_seconds)), flush=True)
    finally:
      if adapter is not None:
        closing_started = time.monotonic()
        closing_deadline = arm_phase(SOURCE_LIMIT-source_seconds)
        adapter.close()
        source_seconds += time.monotonic()-closing_started
        restore_phase(closing_deadline)
  payload = dict(version=VERSION, horizon=100, mc_pairs=16,
      calibration_episode_ids=list(range(8)), bootstrap_seed=20261028,
      frozen_models=frozen_models, episodes=episodes)
  write_json(output/'statistics_input.json', payload)
  # A main-worker replay is not an independent audit. Default False prevents
  # publishing a scientific-support verdict before independent verification.
  statistics = analyze(payload, evidence_authenticated=False)
  write_json(output/'statistics_provisional.json', statistics)
  policy.assert_frozen(full=True)
  require(authenticate_inputs(args.checkpoint, args.config) == budget['input_pins'], 'Inputs changed during run')
  require(git_binding(args.expected_git) == pins, 'Source changed during run')
  write_json(output/'worker_complete.json', dict(version=VERSION, git_sha=args.expected_git,
      complete_episodes=32, evidence=evidence, statistics_input_sha256=sha(output/'statistics_input.json'),
      provisional_statistics_sha256=sha(output/'statistics_provisional.json'),
      fixed_artifact_hashes={name: sha(output/name) for name in ('calibration_frozen.json',
          'model_authenticated.json','runtime_cuda_egl.json','budget.json','process.json')},
      independent_audit_complete=False, source_seconds=source_seconds,
      labels_seconds=labels_seconds, qualification_seconds=qualification_seconds,
      full_model_state_unchanged=True, optimizer_updates=0,
      policy_receipt=policy.receipt(),
      training_actions=0, runtime_final=verify_runtime(resource, compute=True, graphics=True)))


def sample_resource(worker_pid):
  """Bounded read-only XML telemetry; unknown jobs cause our own worker to stop."""
  raw = owned_output(['nvidia-smi','-i','4','-q','-x'])
  gpu = ET.fromstring(raw).find('gpu')
  require(gpu is not None and gpu.find('processes') is not None, 'GPU XML process inventory missing')
  processes = [dict(pid=int(x.findtext('pid')), type=x.findtext('type'))
      for x in gpu.find('processes').findall('process_info')]
  require(all(x['pid'] == worker_pid for x in processes),
      'Unknown work appeared on assigned GPU4; stop only this owned worker')
  return dict(checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
      monotonic=time.monotonic(), physical_gpu=4, gpu_uuid=gpu.findtext('uuid'),
      memory_used=gpu.findtext('fb_memory_usage/used'),
      utilization=gpu.findtext('utilization/gpu_util'), processes=processes)


def parent(args):
  require(os.name == 'posix' and socket.gethostname().split('.')[0] == 'lyg0326', 'sv3 Linux required')
  pins = git_binding(args.expected_git)
  input_pins = authenticate_inputs(args.checkpoint, args.config)
  output = Path(args.output).resolve()
  require(not output.exists(), 'Fresh output directory required')
  output.mkdir(parents=True)
  write_json(output/'budget.json', dict(version=VERSION, parent_pid=os.getpid(),
      seconds=SECONDS, monotonic_deadline=DEADLINE, git_sha=args.expected_git,
      source_files=pins, input_pins=input_pins))
  command = [sys.executable, str(Path(__file__).resolve()), 'worker',
      '--expected-git', args.expected_git, '--checkpoint', args.checkpoint,
      '--config', args.config, '--output', str(output), '--resource-record', args.resource_record]
  env = dict(os.environ, DL_R4_PARENT=str(os.getpid()))
  started = time.monotonic()
  with (output/'worker.log').open('xb') as log:
    child = spawn_owned(command, DEADLINE, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    write_json(output/'process.json', dict(parent_pid=os.getpid(), worker_pid=child.pid,
        worker_pgid=child.pid, command=command, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    failure = None
    try:
      with (output/'resources.jsonl').open('x',encoding='utf-8') as telemetry:
        while child.poll() is None:
          check()
          sample = sample_resource(child.pid)
          telemetry.write(json.dumps(sample,sort_keys=True,allow_nan=False)+'\n'); telemetry.flush()
          try: child.wait(timeout=min(15,max(.001,DEADLINE-time.monotonic())))
          except subprocess.TimeoutExpired: pass
      result = child.returncode
    except Exception as error:
      result, failure = 1, repr(error)
    finally:
      cleanup(child)
  check()
  complete = output/'worker_complete.json'
  verified = result == 0 and complete.is_file()
  write_json(output/'supervisor_terminal.json', dict(version=VERSION,
      git_sha=args.expected_git, worker_returncode=result, worker_terminal=child.poll() is not None,
      elapsed_seconds=time.monotonic()-ENTRY_STARTED, worker_seconds=time.monotonic()-started,
      worker_complete_sha256=sha(complete) if complete.is_file() else None,
      worker_log_sha256=sha(output/'worker.log'),
      resources_sha256=sha(output/'resources.jsonl') if (output/'resources.jsonl').is_file() else None,
      parent_failure=failure,
      main_generation_complete=verified, independent_audit_complete=False,
      scientific_support_claimed=False, owned_worker_group_cleaned=True))
  require(verified, 'r4 failed or incomplete; preserve all artifacts, no scientific verdict')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('command', choices=('run','worker'))
  for name in ('expected-git','checkpoint','config','output','resource-record'):
    parser.add_argument('--'+name, required=True)
  args = parser.parse_args()
  (parent if args.command == 'run' else worker)(args)


if __name__ == '__main__': main()
