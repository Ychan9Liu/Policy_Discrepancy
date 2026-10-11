"""DL-natural-reset-calibration-r1: eight pinned reset states, CPU only.

The CLI parent uses only the standard library and owns a 180-second entry
budget. Numerical/official-physics imports occur only in its worker. This
object tests a finite control requirement, not RGB, a model, or a method.
"""
import time
ENTRY_STARTED = time.monotonic()
import os
import signal
import sys
_OWNED_GROUPS = set()


def _expire(signum, frame):
  for pid in tuple(_OWNED_GROUPS):
    try: os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError: pass
  os._exit(124)


if __name__ == '__main__' and os.name == 'posix':
  signal.signal(signal.SIGALRM, _expire)
  signal.setitimer(signal.ITIMER_REAL, max(1e-6, 180 - (time.monotonic() - ENTRY_STARTED)))

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import pickle
import re
import socket
import subprocess
from types import SimpleNamespace

VERSION = 'DL-natural-reset-calibration-r1'
IMPLEMENTATION = 'DL-natural-reset-engineering-r1'
ROOT = Path(__file__).resolve().parents[1]
SECONDS, HORIZON, DELTA = 180, 20, .2
EPISODES = (9, 14, 15, 18, 20, 24, 30, 31)
CONTROL_NAMES = ('zero', 'all+', 'all-', 'lift-LR+', 'lift-LR-', 'extend-diag+', 'extend-diag-')
CONTRACT = 'production_clip_unify_normalize_v1'
SOURCE_GIT = 'f2d24da1831da2ffca22d313e7f534c96fc0d3eb'
SOURCE_PINS = {
    'plan': '633efe28d551d5793ac8f0ed5f0e36e937e7f1af3c8edefc1d9f5fa278146fcc',
    'complete': 'd155f577d19d2bbc051ec56b95f15e26afc774c130c3c9728a7fe599f295f3cf',
    'artifact_audit': '8f93788ca212118139fab135036fd47ccbad79f04a34139b5dfa7d2dd26f7516',
    'physics_audit': '8ed3528fae18dc7a3043c767da87f873642a40a1a028ce86247f0521c3264722'}
MODEL_PINS = {
    'checkpoint_sha256': '2ab693c8ded4d8f2186bd489aa8b77498448ba2f2c331090159da4e5f08056b9',
    'params_sha256': '14859c4e129b51aa488c10eab09f16d56fd110c76239b2c9072f415829d111c2',
    'config_source_sha256': '11428512fdc9b9e04a5384e419c5abcef93e5429db0a3327d05ccb0d4bd20445'}
SOURCE_FILES = ('scripts/dl_natural_reset.py', 'tests/test_dl_natural_reset.py',
    'docs/DL_NATURAL_RESET.md', 'scripts/dl_action_information_physics.py',
    'analysis/dl_task_semantics.py', 'analysis/dl_feedback_physics_audit_v1.py')
FIELDS = ('reward', 'is_first', 'is_last', 'is_terminal', 'action',
    'action_applied', 'action_executed', 'action_contract', 'snapshots')
SOURCE_TERMINAL_NAMES = ('budget.json','supervisor_complete.json','worker_exit_code.txt')
NO_AUTHORITY = dict(blind_arrays_read=False, current_rgb_increment_established=False,
    shared_model_history_verified=False, active_rep_compression_effect_established=False,
    model_calls=0, render_calls=0, optimizer_updates=0, gpu_devices_used=0,
    real_gate_authority=False, training_authorized=False, eligible_method_training=False,
    method_effectiveness_established=False, proceed_short_training=False)


def require(condition, message):
  if not condition:
    raise ValueError(message)


def sha(path, check=lambda: None):
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b''):
      check(); digest.update(block)
  return digest.hexdigest()


def read_json(path):
  def reject(value):
    raise ValueError('Nonfinite JSON constant: ' + value)
  return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)


def write_json(path, value):
  with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write('\n')


def _kill_owned(child):
  if os.name == 'nt':
    if child.poll() is None: child.kill()
  else:
    try: os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError: pass
  _OWNED_GROUPS.discard(child.pid)
  child.wait(timeout=2)


def _owned_output(command, deadline):
  require(time.monotonic() < deadline, 'Entry deadline exhausted before Git helper')
  child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
      stderr=subprocess.PIPE, start_new_session=os.name != 'nt')
  _OWNED_GROUPS.add(child.pid)
  try:
    stdout, stderr = child.communicate(timeout=max(1e-6, deadline-time.monotonic()))
    if child.returncode:
      raise subprocess.CalledProcessError(child.returncode, command, stdout, stderr)
    return stdout
  finally:
    _kill_owned(child)


def git_binding(expected, deadline):
  require(re.fullmatch('[0-9a-f]{40}', expected) is not None, 'Full execution Git SHA required')
  require(_owned_output(['git', 'rev-parse', 'HEAD'], deadline).decode().strip() == expected,
      'Execution Git differs')
  require(not _owned_output(['git', 'status', '--porcelain'], deadline), 'Clean checkout required')
  proofs = {}
  for name in SOURCE_FILES:
    path = (ROOT/name).resolve(strict=True)
    require(path.is_relative_to(ROOT), 'Unsafe execution dependency')
    blob = _owned_output(['git', 'show', expected+':'+name], deadline)
    require(path.read_bytes() == blob, 'Execution dependency differs from Git: '+name)
    proofs[name] = hashlib.sha256(blob).hexdigest()
  return proofs


def analyze(reward_f64):
  """No fitted model or sampling: the frozen finite-library requirement."""
  import numpy as np
  require(isinstance(reward_f64, np.ndarray) and reward_f64.dtype == np.float64 and
      reward_f64.shape == (8, 7, 20) and np.isfinite(reward_f64).all(),
      'Complete finite F64 reset rewards required')
  costs = reward_f64.sum(-1, dtype=np.float64)
  require(np.isfinite(costs).all(), 'Nonfinite cumulative reset rewards')
  regret = costs.max(-1, keepdims=True) - costs
  near = regret <= DELTA
  ranges = np.ptp(costs, axis=-1)
  pairs, margins = [], []
  for i in range(8):
    for j in range(i+1, 8):
      item = dict(episode_a=EPISODES[i], episode_b=EPISODES[j],
          minimax_regret=float(np.minimum.reduce(np.maximum(regret[i], regret[j]))),
          minimax_regret_minus_delta=float(np.minimum.reduce(
              np.maximum(regret[i], regret[j]))-DELTA),
          closed_delta_optimal_sets_disjoint=not bool(np.any(near[i] & near[j])))
      margins.append(item)
      if item['closed_delta_optimal_sets_disjoint']:
        require(ranges[i] >= DELTA and ranges[j] >= DELTA,
            'Disjoint closed sets require both state ranges at least delta')
        pairs.append(item.copy())
  sufficient = bool(pairs)
  return dict(episodes=list(EPISODES), position=0, horizon=20,
      control_names=list(CONTROL_NAMES), delta=DELTA, rewards=costs.tolist(),
      range=ranges.tolist(), finite_library_regret=regret.tolist(),
      delta_optimal_membership=near.tolist(), delta_optimal_boundary='max_reward - reward <= delta',
      sensitive=(ranges >= DELTA).tolist(), cross_episode_disjoint_pairs=pairs,
      disjoint_pair_count=len(pairs), all_pair_minimax=margins,
      sufficient=sufficient,
      status='finite_reset_control_object_sufficient' if sufficient else
          'finite_reset_control_object_insufficient',
      decision='stop_after_fixed_natural_reset_calibration',
      minimax_margin_is_new_threshold=False, display_tolerance_applied=False,
      causal_rgb_or_model_information_established=False)


def source_authority(dataset, artifact_audit, physics_audit, check=lambda: None):
  """All 32 raw archive hashes; decode only eight calibration white lists."""
  import numpy as np
  dataset = Path(dataset).resolve(strict=True)
  paths = dict(plan=dataset/'collection_plan.json', complete=dataset/'collection_complete.json',
      artifact_audit=Path(artifact_audit).resolve(strict=True),
      physics_audit=Path(physics_audit).resolve(strict=True))
  for key, path in paths.items():
    require(sha(path, check) == SOURCE_PINS[key], 'Published source JSON differs: '+key)
  plan, complete, audit, physics = [read_json(paths[k]) for k in paths]
  require(plan['version'] == complete['version'] == 'DL-source-r1' and
      plan['implementation'] == complete['implementation'] == 'DL-source-engineering-r2' and
      plan['action_contract'] == complete['action_contract'] == CONTRACT and
      plan['metadata']['git_sha'] == SOURCE_GIT and
      complete['plan_sha256'] == SOURCE_PINS['plan'], 'Source production version differs')
  require(all(plan['metadata'][k] == v for k, v in MODEL_PINS.items()) and
      complete['params_unchanged'] is True and complete['training_counters_unchanged'] is True and
      complete['eval_counters_legitimately_advanced'] is True and complete['optimizer_updates'] == 0,
      'Source immutable model/counter proof differs')
  config = plan['metadata']['config']
  require(config['task'] == 'dmc_quadruped_walk' and config['env.dmc.proprio'] is False,
      'Unreviewed source task/observation')
  require(audit['source_contract_artifact_audit_passed'] is True and
      audit['expected_source_sha'] == physics['expected_source_sha'] == SOURCE_GIT and
      audit['blind_arrays_read'] is False and physics['status'] == 'passed' and
      physics['blind_npz_arrays_opened'] is False and physics['physics_steps'] == 12800 and
      physics['source_json_sha256']['complete'] == SOURCE_PINS['complete'] and
      audit['metadata_sha256']['collection_complete.json'] == SOURCE_PINS['complete'],
      'Source completed audits missing')
  terminal_paths = {name:dataset/name for name in SOURCE_TERMINAL_NAMES}
  terminal_hashes = {name:audit['metadata_sha256'][name] for name in SOURCE_TERMINAL_NAMES}
  for name,path in terminal_paths.items():
    require(re.fullmatch('[0-9a-f]{64}',terminal_hashes[name]) is not None and
        sha(path,check) == terminal_hashes[name], 'Source terminal metadata bytes differ: '+name)
  source_budget = read_json(terminal_paths['budget.json'])
  source_supervisor = read_json(terminal_paths['supervisor_complete.json'])
  require(source_budget['version'] == source_supervisor['version'] == 'DL-source-r1' and
      source_budget['git_sha'] == source_supervisor['git_sha'] == SOURCE_GIT and
      source_budget['seconds'] == 3600 and
      source_budget['script_sha256'] == audit['source_file_sha256']['scripts/dl_production_source.py'] and
      source_supervisor['budget_sha256'] == terminal_hashes['budget.json'] and
      type(source_supervisor['exit_code']) is int and source_supervisor['exit_code'] == 0 and
      source_supervisor['timed_out'] is False and source_supervisor['worker_terminal'] is True and
      0 <= source_supervisor['elapsed_seconds'] <= 3600 and
      audit['supervisor_exit_code'] == 0 and terminal_paths['worker_exit_code.txt'].read_bytes() == b'0\n' and
      physics['source_json_sha256']['budget'] == terminal_hashes['budget.json'] and
      physics['source_json_sha256']['supervisor'] == terminal_hashes['supervisor_complete.json'],
      'Source original bounded terminal/exit chain differs')
  require({name:importlib.metadata.version(name) for name in plan['environment']['packages']} ==
      plan['environment']['packages'], 'Source runtime package versions differ')
  order = np.random.default_rng(20261020).permutation(32)
  calibration = set(order[:8].tolist())
  definitions = [dict(episode=i, seed=int(np.random.SeedSequence(
      [20261020, i, 0x444C]).generate_state(1)[0]),
      policy_seed=int(np.random.SeedSequence([20261020, i, 0x504F4C]).generate_state(1)[0]),
      split='calibration' if i in calibration else 'blind') for i in range(32)]
  require(plan['episodes'] == definitions and len(complete['artifacts']) == 32 and
      tuple(sorted(calibration)) == EPISODES, 'Source episode allocation differs')
  selected, all_hashes, calibrated_items = [], [], []
  for definition, item in zip(definitions, complete['artifacts']):
    check()
    require({k:item[k] for k in definition} == definition and
        item['path'] == f'episode_{definition["episode"]:03}.npz', 'Source artifact identity/path differs')
    path = (dataset/item['path']).resolve(strict=True)
    require(path.parent == dataset and sha(path, check) == item['sha256'], 'Source archive bytes differ')
    all_hashes.append(dict(episode=definition['episode'], path=str(path), sha256=item['sha256']))
    if definition['split'] == 'calibration':
      calibrated_items.append((definition,item,path))
  # No pickle or numerical array opens until all 32 raw bytes are verified.
  for definition,item,path in calibrated_items:
    check()
    payload = path.read_bytes()
    check()
    require(hashlib.sha256(payload).hexdigest() == item['sha256'],
        'Source archive changed before calibrated decode')
    # The verified payload, not another pathname read, supplies every decoded
    # array and the trusted snapshot bytes used for pickle.loads below.
    with np.load(io.BytesIO(payload), allow_pickle=False) as artifact:
      data = {key:artifact[key].copy() for key in FIELDS}
    validate_source_arrays(data)
    # This is only the fixed, hash-authenticated project-generated snapshot pickle.
    snapshots = pickle.loads(data.pop('snapshots').tobytes())
    require(isinstance(snapshots, list) and len(snapshots) == 1001,
        'Source complete snapshot coverage differs')
    selected.append(dict(definition=definition, artifact=item, data=data, snapshots=snapshots))
  require(tuple(x['definition']['episode'] for x in selected) == EPISODES, 'Partial calibration source')
  return selected, dict(source_git_sha=SOURCE_GIT, source_json_sha256=SOURCE_PINS,
      source_json_paths={k:str(v) for k, v in paths.items()}, source_dataset=str(dataset),
      source_terminal_paths={k:str(v) for k,v in terminal_paths.items()},
      source_terminal_sha256=terminal_hashes,source_original_exit_chain_verified=True,
      all_32_artifact_hashes=all_hashes, numeric_episode_reads=8, trusted_snapshot_archives_decoded=8,
      image_arrays_read=False, before_carries_read=False, blind_arrays_read=False,
      actual_fields_read=list(FIELDS), source_model_metadata=plan['metadata'])


def validate_source_arrays(data):
  import numpy as np
  for name, dtype, shape in (
      ('reward', np.float32, (1001,)), ('is_first', np.bool_, (1001,)),
      ('is_last', np.bool_, (1001,)), ('is_terminal', np.bool_, (1001,)),
      ('action', np.float32, (1001,12)), ('action_applied', np.float64, (1001,12)),
      ('action_executed', np.bool_, (1001,))):
    value = data[name]
    require(isinstance(value, np.ndarray) and value.dtype == np.dtype(dtype) and
        value.shape == shape and np.isfinite(value).all(), 'Source numeric contract differs: '+name)
  require(isinstance(data['snapshots'], np.ndarray) and data['snapshots'].dtype == np.uint8 and
      data['snapshots'].ndim == 1 and data['snapshots'].size > 0, 'Trusted snapshot payload differs')
  require(data['action_contract'].shape == () and data['action_contract'].item() == CONTRACT,
      'Source action contract differs')
  require(bool(data['is_first'][0]) and not np.any(data['is_first'][1:]) and
      not np.any(data['is_last'][:1000]) and bool(data['is_last'][1000]) and
      not np.any(data['is_terminal']) and np.all(data['action_executed'][:1000]) and
      not bool(data['action_executed'][1000]) and data['reward'][0].tobytes() == np.float32(0).tobytes(),
      'Source is not a complete actual reset/time-limit episode')


def compare_exact(audit, actual, expected):
  """New repeated physics uses byte identity, not the source restore tolerance."""
  audit.compare_state(actual, expected)
  require(audit.byte_equal(actual['suite']['integration'], expected['suite']['integration']),
      'New complete integration byte replay differs')
  # compare_state already requires exact task/RNG/clock/reset/done identity.
  return 0.


def reference_replay(env, source, audit, physical, check=lambda: None):
  import numpy as np
  import mujoco
  initial = source['snapshots'][0]
  size = mujoco.mj_stateSize(env.physics.model.ptr, mujoco.mjtState.mjSTATE_INTEGRATION)
  audit.validate_snapshot(initial, size)
  require(initial['suite']['step_count'] == 0 and not initial['suite']['reset_next_step'] and
      not initial['from_dm_done'] and bool(source['data']['is_first'][0]),
      'Source is not actual active reset frame0')
  receipt = physical.verify_model(env)
  # A new official environment with the original fixed seed checks the reset
  # provenance directly. This is a reset, not a new control transition.
  fresh = env.reset()
  require(fresh.first() and not fresh.last() and fresh.reward is None,
      'Official fresh reset semantics differ')
  fresh_saved = audit.snapshot(env,False)
  fresh_error = audit.compare_state(fresh_saved,initial)
  fresh_integration_exact = audit.byte_equal(fresh_saved['suite']['integration'],
      initial['suite']['integration'])
  audit.restore(env, initial)
  require(physical.verify_model(env) == receipt, 'Restored source task/model differs')
  maximum = audit.compare_state(audit.snapshot(env, False), initial)
  initial_error = maximum
  spec, data = env.action_spec(), source['data']
  for step in range(20):
    check()
    require(not env._reset_next_step, 'Reference would reset')
    raw = data['action'][step].copy()
    applied = physical.mapping(raw, spec.minimum, spec.maximum)
    require(audit.byte_equal(applied, data['action_applied'][step]),
        'Source raw-to-applied byte mapping differs')
    ts = env.step(applied)
    reward = np.asarray(ts.reward)
    audit.array(reward, np.float64, (), 'official source reference reward')
    require(np.asarray(reward, np.float32).tobytes() == data['reward'][step+1].tobytes(),
        'Source reference F32 reward byte differs')
    flags = (bool(ts.first()), bool(ts.last()), bool(ts.last() and ts.discount == 0))
    require(flags == tuple(bool(data[k][step+1]) for k in ('is_first','is_last','is_terminal')) and
        flags == (False,False,False) and float(ts.discount) == 1.,
        'Source reference flags/discount or full H20 window differs')
    saved = source['snapshots'][step+1]
    audit.validate_snapshot(saved, size)
    require(saved['suite']['step_count'] == step+1, 'Source reference absolute clock differs')
    maximum = max(maximum, audit.compare_state(audit.snapshot(env,False), saved))
  return dict(episode=source['definition']['episode'], position=0, physics_steps=20,
      original_raw_applied_byte_exact=True, source_reward_float32_byte_exact=True,
      source_float64_reward_byte_exact_claimed=False, flags_exact=True,
      source_complete_state_tolerance=1e-8, maximum_restore_error=initial_error,
      fresh_official_seed_reset_verified=True,fresh_reset_calls=1,
      fresh_reset_complete_integration_error=fresh_error,
      fresh_reset_integration_byte_exact=fresh_integration_exact,
      fresh_reset_snapshot_sha256=hashlib.sha256(pickle.dumps(fresh_saved,protocol=5)).hexdigest(),
      maximum_complete_integration_error=maximum, complete_task_rng_clock_reset_done_exact=True,
      source_initial_snapshot_sha256=hashlib.sha256(pickle.dumps(initial,protocol=5)).hexdigest())


def complete_record(env, record, audit, physical):
  physical.validate_record(env, record, audit)
  require(record['executed_steps'] == 20 and record['end_type'] == 'window' and
      record['snapshots'][0]['suite']['step_count'] == 0 and
      not record['is_first'].any() and not record['is_last'].any() and
      not record['is_terminal'].any() and record['action_executed'].all() and
      record['snapshot_executed'].all(), 'New reset object requires genuine complete H20, no padding')


def worker(args):
  output = Path(args.output).resolve(strict=True)
  budget = read_json(output/'budget.json')
  require(budget['parent_pid'] == os.getppid() == int(os.environ.get('DL_NATURAL_RESET_PARENT','-1')) and
      budget['seconds'] == 180 and budget['script_sha256'] == sha(__file__) and
      budget['actual_git_sha'] == args.expected_git, 'Live owned parent/Git/budget required')
  os.kill(os.getppid(),0)
  deadline = budget['monotonic_start'] + SECONDS
  def check():
    require(time.monotonic() < deadline, 'Incomplete at whole-entry 180-second deadline')
  check()
  sources = git_binding(args.expected_git, deadline)
  os.environ.update(CUDA_VISIBLE_DEVICES='', MUJOCO_GL='disable', JAX_PLATFORMS='cpu',
      OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
  sys.path.insert(0,str(ROOT))
  import numpy as np
  from analysis import dl_task_semantics as semantics
  from scripts import dl_action_information_physics as physical
  audit, dependency = semantics.load_auditor(ROOT/'analysis/dl_feedback_physics_audit_v1.py')
  suite, _, _, official, packages = semantics.official_modules()
  selected, binding = source_authority(args.dataset,args.artifact_audit,args.physics_audit,check)
  check()
  references = []
  # Every source prefix passes before any new control outcome is generated.
  for source in selected:
    env = suite.load('quadruped','walk',task_kwargs={'random':source['definition']['seed']})
    try:
      references.append(reference_replay(env,source,audit,physical,check))
    finally:
      env.close()
  require(len(references) == 8 and sum(x['physics_steps'] for x in references) == 160,
      'Incomplete source reference coverage')
  plan = dict(version=VERSION,implementation=IMPLEMENTATION,actual_git_sha=args.expected_git,
      seconds=SECONDS,hostname=socket.gethostname(),platform=sys.platform,python=sys.version,
      episodes=list(EPISODES),position=0,horizon=20,controls=7,delta=DELTA,
      control_names=list(CONTROL_NAMES),common_tail='19 raw float32 normalized-zero commands',
      source_binding=binding,source_files_sha256=sources,derived_physics_auditor=dependency,
      official_modules=official,packages=packages,reference_receipts=references,
      budget_sha256=sha(output/'budget.json'),max_reference_steps=160,max_main_steps=1120,
      max_replay_steps=1120,exploratory_calibration_only=True,**NO_AUTHORITY)
  write_json(output/'plan.json',plan);check()
  rewards, artifacts, main_steps, replay_steps = [], [], 0, 0
  strict_audit = SimpleNamespace(**{name:getattr(audit,name) for name in
      ('validate_snapshot','array','byte_equal','restore','snapshot')},
      compare_state=lambda actual,expected:compare_exact(audit,actual,expected))
  for source in selected:
    ep = source['definition']['episode']
    env = suite.load('quadruped','walk',task_kwargs={'random':source['definition']['seed']})
    records, receipts = [], []
    try:
      controls = physical.controls(env)
      for control, raw in enumerate(controls):
        check()
        record = physical.run_sequence(env,source['snapshots'][0],raw,audit,check)
        record.update(episode=ep,position=0,control=control)
        try:
          complete_record(env,record,audit,physical)
          receipt = physical.replay_sequence(env,record,strict_audit,check)
          require(receipt['padding_steps'] == 0 and receipt['maximum_integration_error'] == 0,
              'New repeated sequence is not complete/byte-exact')
        except Exception:
          with (output/f'failed-ep{ep:03}-control{control:02}.npz').open('xb') as stream:
            np.savez_compressed(stream,records=np.frombuffer(pickle.dumps(record,protocol=5),np.uint8))
          raise
        records.append(record);receipts.append(dict(control=control,**receipt,
            complete_integration_task_rng_clock_reset_done_byte_exact=True))
        main_steps += 20;replay_steps += 20
        require(main_steps <= 1120 and replay_steps <= 1120, 'Frozen step budget exceeded')
      arrays = {name:np.stack([record[name] for record in records]) for name in physical.ARRAYS}
      path = output/f'episode_{ep:03}.npz'
      with path.open('xb') as stream:
        np.savez_compressed(stream,**arrays,episode=np.array(ep,np.int32),position=np.array(0,np.int32),
            records=np.frombuffer(pickle.dumps(records,protocol=5),np.uint8))
      artifacts.append(dict(episode=ep,path=path.name,sha256=sha(path,check),complete_records=7,
          main_physics_steps=140,replay_physics_steps=140,replay_receipts=receipts,
          original_initial_snapshot_sha256=references[len(artifacts)]['source_initial_snapshot_sha256']))
      rewards.append(arrays['reward_f64'])
      print(json.dumps(dict(stage='reset_controls',episode=ep,main_steps=main_steps,
          replay_steps=replay_steps)),flush=True)
    finally:
      env.close()
  require(main_steps == replay_steps == 1120 and len(artifacts) == 8 and
      'jax' not in sys.modules, 'Incomplete fixed CPU object or JAX imported')
  verify_source_unchanged(binding,check)
  require(git_binding(args.expected_git,deadline) == sources, 'Execution sources changed')
  result = dict(version=VERSION,implementation=IMPLEMENTATION,actual_git_sha=args.expected_git,
      source_files_sha256=sources,source_inputs_unchanged=True,source_binding=binding,
      reference_physics_steps=160,main_physics_steps=main_steps,replay_physics_steps=replay_steps,
      total_physics_steps=2400,evidence_artifacts=artifacts,reference_receipts=references,
      analysis=analyze(np.stack(rewards)),budget_sha256=sha(output/'budget.json'),
      plan_sha256=sha(output/'plan.json'),exploratory_calibration_only=True,
      elapsed_parent_entry_seconds=time.monotonic()-budget['monotonic_start'],
      actual_from_dm_wrapper_executed=False,new_independent_physics_audit=False,
      stopped_after_fixed_object=True,**NO_AUTHORITY)
  write_json(output/'result.json',result);check()
  write_json(output/'natural_reset_complete.json',dict(version=VERSION,implementation=IMPLEMENTATION,
      actual_git_sha=args.expected_git,source_files_sha256=sources,complete_episodes=8,
      complete_states=8,complete_records=56,reference_physics_steps=160,main_physics_steps=1120,
      replay_physics_steps=1120,total_physics_steps=2400,source_inputs_unchanged=True,
      budget_sha256=sha(output/'budget.json'),plan_sha256=sha(output/'plan.json'),
      result_sha256=sha(output/'result.json'),status=result['analysis']['status'],**NO_AUTHORITY))
  check()


def verify_source_unchanged(binding,check=lambda:None):
  for key,path in binding['source_json_paths'].items():
    require(sha(path,check) == SOURCE_PINS[key], 'Source JSON changed')
  require(tuple(binding['source_terminal_paths']) == SOURCE_TERMINAL_NAMES and
      set(binding['source_terminal_sha256']) == set(SOURCE_TERMINAL_NAMES), 'Partial source terminal binding')
  for name,path in binding['source_terminal_paths'].items():
    require(sha(path,check) == binding['source_terminal_sha256'][name], 'Source terminal metadata changed')
  require(len(binding['all_32_artifact_hashes']) == 32, 'Partial source byte binding')
  for item in binding['all_32_artifact_hashes']:
    require(sha(item['path'],check) == item['sha256'], 'Source archive changed')


def verify_completion(output,expected_git,deadline):
  """Stdlib chain check. Independent arithmetic/physics audit is separate."""
  output = Path(output)
  require(not (output/'late_supervisor_receipt.json').exists(), 'Late parent receipt is not completion')
  budget,plan,result,complete = [read_json(output/name) for name in
      ('budget.json','plan.json','result.json','natural_reset_complete.json')]
  for value in (budget,plan,result,complete):
    require(value['version'] == VERSION and value['actual_git_sha'] == expected_git,
        'Wrong fixed artifact version/Git')
  for value in (plan,result,complete):
    require(all(value[k] is v if type(v) is bool else type(value[k]) is int and value[k] == v
        for k,v in NO_AUTHORITY.items()), 'Scientific authority boundary changed')
  require(budget['seconds'] == plan['seconds'] == SECONDS and
      budget['script_sha256'] == sha(__file__) and budget['parent_pid'] == os.getpid() and
      plan['episodes'] == list(EPISODES) and plan['position'] == 0 and plan['horizon'] == 20 and
      plan['controls'] == 7 and plan['delta'] == DELTA and
      plan['hostname'] == 'lyg0326' and plan['platform'].startswith('linux'),
      'Wrong actual runtime/fixed object')
  sources = git_binding(expected_git,deadline)
  require(all(value['source_files_sha256'] == sources for value in (plan,result,complete)),
      'Incomplete source code binding')
  require(sha(output/'budget.json') == plan['budget_sha256'] == result['budget_sha256'] ==
      complete['budget_sha256'] and sha(output/'plan.json') == result['plan_sha256'] ==
      complete['plan_sha256'] and sha(output/'result.json') == complete['result_sha256'],
      'Budget/plan/result completion hash differs')
  for key,value in dict(complete_episodes=8,complete_states=8,complete_records=56,
      reference_physics_steps=160,main_physics_steps=1120,replay_physics_steps=1120,
      total_physics_steps=2400).items():
    require(type(complete[key]) is int and complete[key] == value, 'Partial completion: '+key)
  require(result['reference_physics_steps'] == 160 and result['main_physics_steps'] ==
      result['replay_physics_steps'] == 1120 and result['total_physics_steps'] == 2400 and
      result['source_inputs_unchanged'] is True and result['stopped_after_fixed_object'] is True and
      result['elapsed_parent_entry_seconds'] <= SECONDS, 'Incomplete fixed physics or late worker')
  require(plan['source_binding'] == result['source_binding'] and
      len(plan['reference_receipts']) == 8 and result['reference_receipts'] == plan['reference_receipts'],
      'Source/reference chain differs')
  verify_source_unchanged(plan['source_binding'],lambda:require(time.monotonic()<deadline,'Late source verification'))
  artifacts = result['evidence_artifacts']
  require(len(artifacts) == 8 and tuple(item['episode'] for item in artifacts) == EPISODES,
      'Partial/reordered result episodes')
  for item in artifacts:
    require(item['path'] == f'episode_{item["episode"]:03}.npz' and
        (output/item['path']).resolve(strict=True).parent == output.resolve() and
        sha(output/item['path']) == item['sha256'] and item['complete_records'] == 7 and
        len(item['replay_receipts']) == 7, 'Changed/partial outcome artifact')
  require(complete['status'] == result['analysis']['status'] and
      result['analysis']['decision'] == 'stop_after_fixed_natural_reset_calibration' and
      result['analysis']['status'] in ('finite_reset_control_object_sufficient',
        'finite_reset_control_object_insufficient'), 'Wrong finite stopping decision')
  return dict(complete_verified=True,verification_kind='schema_hash_chain_not_independent_arithmetic_or_physics',
      numeric_episode_reads=8,trusted_snapshot_archives_decoded=8,blind_arrays_read=False,
      actual_gpu_devices_used=0,physics_steps=2400,model_calls=0,render_calls=0,optimizer_updates=0,
      budget_sha256=sha(output/'budget.json'),plan_sha256=sha(output/'plan.json'),
      result_sha256=sha(output/'result.json'),complete_sha256=sha(output/'natural_reset_complete.json'))


def supervise(command, output, *, seconds=SECONDS, started=None, expected_git=None, verify=None):
  """Generic fixtures report unknown actual scientific scope; CLI is fixed."""
  require(math.isfinite(seconds) and seconds > 0, 'Positive finite deadline required')
  started = time.monotonic() if started is None else started
  deadline = started + seconds
  output = Path(output).resolve()
  require(not output.exists(), 'New attempt directory must not exist')
  output.mkdir(parents=True)
  write_json(output/'budget.json',dict(version=VERSION,implementation=IMPLEMENTATION,
      actual_git_sha=expected_git,parent_pid=os.getpid(),seconds=seconds,monotonic_start=started,
      script_sha256=sha(__file__),planned_gpu_devices=0,actual_gpu_devices_used=None,
      planned_numeric_episode_reads=8,actual_numeric_episode_reads=None))
  require(time.monotonic() < deadline, 'Late before worker creation')
  timed_out, verification, child = False, None, None
  with (output/'natural_reset.log').open('x',encoding='utf-8') as log:
    child = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
        start_new_session=os.name != 'nt',env=dict(os.environ,DL_NATURAL_RESET_PARENT=str(os.getpid()),
            CUDA_VISIBLE_DEVICES='',MUJOCO_GL='disable',JAX_PLATFORMS='cpu'))
    _OWNED_GROUPS.add(child.pid)
    try:
      code = child.wait(timeout=max(1e-6,deadline-time.monotonic()))
    except subprocess.TimeoutExpired:
      timed_out = True;code = 124
    finally:
      _kill_owned(child)
  if not timed_out and code == 0 and verify is not None:
    try:
      verification = verify(output,expected_git,deadline)
    except Exception as error:
      code = 1
      write_json(output/'completion_verification_failure.json',dict(error=repr(error)))
  if time.monotonic() >= deadline:
    timed_out = True;code = 124;verification = None
  real_verified = bool(verify is verify_completion and verification and
      verification.get('verification_kind') == 'schema_hash_chain_not_independent_arithmetic_or_physics')
  receipt = dict(version=VERSION,implementation=IMPLEMENTATION,actual_git_sha=expected_git,
      parent_pid=os.getpid(),worker_pid=child.pid,exit_code=code,timed_out=timed_out,
      worker_terminal=child.poll() is not None,owned_group_cleanup_requested=os.name != 'nt',
      elapsed_parent_entry_seconds=time.monotonic()-started,verification=verification,
      complete_verified=real_verified,
      actual_numeric_episode_reads=8 if real_verified else None,
      actual_gpu_devices_used=0 if real_verified else None,
      reason='incomplete_hard_budget' if timed_out else 'verified_fixed_cpu_object' if real_verified else
          'generic_worker_exited' if code == 0 else 'worker_or_chain_failed',
      budget_sha256=sha(output/'budget.json'))
  # Stage the receipt first. A delayed write cannot publish a success-looking
  # final receipt; only the complete write before the deadline is promoted.
  pending = output/'supervisor_receipt_pending.json'
  write_json(pending,receipt)
  if time.monotonic() >= deadline:
    write_json(output/'late_supervisor_receipt.json',dict(incomplete=True,exit_code=124))
    receipt.update(exit_code=124,timed_out=True,complete_verified=False,verification=None,
        actual_numeric_episode_reads=None,actual_gpu_devices_used=None,
        reason='incomplete_hard_budget',elapsed_parent_entry_seconds=time.monotonic()-started)
    write_json(output/'supervisor_complete.json',receipt)
  else:
    os.replace(pending,output/'supervisor_complete.json')
    if time.monotonic() >= deadline:
      # Preserve the provisional receipt as failure evidence, then publish an
      # unambiguously failed receipt. Late cleanup is never valid completion.
      os.replace(output/'supervisor_complete.json',output/'late_provisional_supervisor_receipt.json')
      write_json(output/'late_supervisor_receipt.json',dict(incomplete=True,exit_code=124))
      receipt.update(exit_code=124,timed_out=True,complete_verified=False,verification=None,
          actual_numeric_episode_reads=None,actual_gpu_devices_used=None,
          reason='incomplete_hard_budget',elapsed_parent_entry_seconds=time.monotonic()-started)
      write_json(output/'supervisor_complete.json',receipt)
  return receipt


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for field in ('dataset','artifact-audit','physics-audit','output','expected-git'):
    parser.add_argument('--'+field,required=True)
  parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
  args = parser.parse_args()
  require(os.name == 'posix' and socket.gethostname() == 'lyg0326', 'Frozen actual execution is sv3 Linux CPU')
  if args.worker:
    worker(args);return 0
  deadline = ENTRY_STARTED + SECONDS
  git_binding(args.expected_git,deadline)
  command = [sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker']
  receipt = supervise(command,args.output,started=ENTRY_STARTED,expected_git=args.expected_git,
      verify=verify_completion)
  print(json.dumps(receipt),flush=True)
  return receipt['exit_code']


if __name__ == '__main__':
  sys.exit(main())
