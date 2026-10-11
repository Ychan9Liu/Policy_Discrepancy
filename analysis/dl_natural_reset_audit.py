"""DL-natural-reset-calibration-r1 independent CPU statistics/physics audit.

No producer statistics, model, renderer, JAX or blind arrays are executed.
Complete-state restore/snapshot primitives are explicitly reused from the
already reviewed independent feedback auditor; this is not another simulator.
Only hash-authenticated project pickle bytes may be decoded.
"""
import os
import time

ENTRY_STARTED = time.monotonic()
SECONDS = 180
VERSION = 'DL-natural-reset-calibration-r1'
IMPLEMENTATION = 'DL-natural-reset-audit-r1'
EPISODES = (9, 14, 15, 18, 20, 24, 30, 31)
SOURCE_GIT = 'f2d24da1831da2ffca22d313e7f534c96fc0d3eb'
SOURCE_PINS = {
    'collection_plan.json': '633efe28d551d5793ac8f0ed5f0e36e937e7f1af3c8edefc1d9f5fa278146fcc',
    'collection_complete.json': 'd155f577d19d2bbc051ec56b95f15e26afc774c130c3c9728a7fe599f295f3cf'}
AUDIT_PINS = {
    'artifact': '8f93788ca212118139fab135036fd47ccbad79f04a34139b5dfa7d2dd26f7516',
    'physics': '8ed3528fae18dc7a3043c767da87f873642a40a1a028ce86247f0521c3264722'}
MODEL_PINS = dict(
    checkpoint_sha256='2ab693c8ded4d8f2186bd489aa8b77498448ba2f2c331090159da4e5f08056b9',
    params_sha256='14859c4e129b51aa488c10eab09f16d56fd110c76239b2c9072f415829d111c2',
    config_source_sha256='11428512fdc9b9e04a5384e419c5abcef93e5429db0a3327d05ccb0d4bd20445')
CONTRACT = 'production_clip_unify_normalize_v1'
XML_SHA = 'c6b0ade5f3160b155bbe673aec3e0430eed3273ae13b60880855647e82bf8334'
DELTA = .2
FIELDS = {
    'raw': ('float32', (7, 20, 12)), 'applied': ('float64', (7, 20, 12)),
    'reward_f64': ('float64', (7, 20)), 'reward_f32': ('float32', (7, 20)),
    'discount': ('float64', (7, 20)), 'is_first': ('bool', (7, 20)),
    'is_last': ('bool', (7, 20)), 'is_terminal': ('bool', (7, 20)),
    'action_executed': ('bool', (7, 20)),
    'snapshot_executed': ('bool', (7, 21))}

import argparse
import copy
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import pickle
import re
import signal
import socket
import subprocess
import sys

if __name__ != '__main__' or '--worker' in sys.argv:
  import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
  if not condition:
    raise ValueError(message)


def check_deadline():
  if time.monotonic() - ENTRY_STARTED >= SECONDS:
    raise TimeoutError('Independent natural-reset whole-entry 180s expired')


def sha(path):
  result = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
      check_deadline(); result.update(chunk)
  return result.hexdigest()


def read_json(path):
  def pairs(items):
    result = {}
    for key, value in items:
      require(key not in result, 'Duplicate JSON key: ' + key)
      result[key] = value
    return result
  def reject(value):
    raise ValueError('Nonfinite JSON: ' + value)
  return json.loads(Path(path).read_text(encoding='utf-8'),
      object_pairs_hook=pairs, parse_constant=reject)


def safe_file(root, name):
  root = Path(root).resolve(strict=True)
  require(type(name) is str and Path(name).name == name and not Path(name).is_absolute(),
          'Unsafe artifact filename')
  value = (root / name).resolve(strict=True)
  require(value.parent == root and value.is_file(), 'Artifact escapes its directory')
  return value


def numeric(value, dtype, shape, name):
  require(type(value) is np.ndarray and value.dtype == np.dtype(dtype) and
      value.shape == shape and not value.dtype.hasobject and np.isfinite(value).all(),
      'Malformed finite numeric array: ' + name)
  return value


def byte_equal(a, b):
  return (type(a) is np.ndarray and type(b) is np.ndarray and
      a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes())


def commands():
  """Independent enumeration of the frozen seven normalized controls."""
  value = np.zeros((7, 12), np.float32)
  value[1] = .25; value[2] = -.25
  value[3, [1, 10]] = .25; value[3, [4, 7]] = -.25
  value[4] = -value[3]
  value[5, [2, 8]] = .25; value[5, [5, 11]] = -.25
  value[6] = -value[5]
  return value


def mapping(raw, low, high):
  numeric(raw, 'float32', (12,), 'raw')
  numeric(low, 'float64', (12,), 'low')
  numeric(high, 'float64', (12,), 'high')
  clipped = np.clip(raw, -1., 1.).astype(np.float32, copy=False)
  return ((clipped + 1.) / 2.) * (high - low) + low


def statistics(rewards):
  """Independent fsum and exhaustive pair minimax, never producer formulas."""
  numeric(rewards, 'float64', (8, 7, 20), 'all F64 rewards')
  costs = np.asarray([[math.fsum(float(x) for x in rewards[e, a])
      for a in range(7)] for e in range(8)], np.float64)
  best = np.asarray([max(row) for row in costs], np.float64)
  regret = best[:, None] - costs
  ranges = np.asarray([max(row) - min(row) for row in costs], np.float64)
  membership = regret <= DELTA
  pairs = []
  for left in range(8):
    for right in range(left + 1, 8):
      common = bool(any(membership[left, a] and membership[right, a] for a in range(7)))
      minimax = min(max(float(regret[left, a]), float(regret[right, a])) for a in range(7))
      eligible = bool(ranges[left] >= DELTA and ranges[right] >= DELTA and not common)
      pairs.append(dict(episodes=[EPISODES[left], EPISODES[right]],
          disjoint=not common, ranges_at_least_delta=bool(
              ranges[left] >= DELTA and ranges[right] >= DELTA),
          minimax_regret=minimax, minimax_minus_delta=minimax - DELTA,
          sufficient_pair=eligible))
  sufficient = any(pair['sufficient_pair'] for pair in pairs)
  return dict(costs=costs.tolist(), regret=regret.tolist(), range=ranges.tolist(),
      delta_optimal_membership=membership.tolist(), pairs=pairs,
      status=('finite_reset_control_object_sufficient' if sufficient else
              'finite_reset_control_object_insufficient'))


def compare_tree(actual, expected, path='root'):
  """Numerical oracle 1e-10; discrete identity and margin zero sides exact."""
  if isinstance(expected, dict):
    require(isinstance(actual, dict) and set(actual) == set(expected), 'Fields differ: ' + path)
    for key, value in expected.items(): compare_tree(actual[key], value, path + '.' + key)
  elif isinstance(expected, list):
    require(isinstance(actual, list) and len(actual) == len(expected), 'Axes differ: ' + path)
    for index, value in enumerate(expected): compare_tree(actual[index], value, f'{path}[{index}]')
  elif isinstance(expected, float):
    require(type(actual) in (int, float) and not isinstance(actual, bool) and
        math.isfinite(actual) and abs(actual - expected) <= 1e-10,
        'Numerical oracle differs: ' + path)
    if path.endswith('minus_delta'):
      require((actual > 0, actual == 0, actual < 0) ==
          (expected > 0, expected == 0, expected < 0), 'Margin zero side differs: ' + path)
  else:
    require(type(actual) is type(expected) and actual == expected, 'Exact field differs: ' + path)


def complete_state_exact(actual, expected):
  """New branch replays demand bytes, never the legacy 1e-8 state guard."""
  require(type(actual) is dict and type(expected) is dict and
      set(actual) == set(expected) == {'suite', 'from_dm_done'}, 'Malformed complete state')
  a, b = actual['suite'], expected['suite']
  require(type(a) is dict and type(b) is dict and set(a) == set(b) ==
      {'integration', 'step_count', 'reset_next_step', 'task'}, 'Incomplete suite state')
  numeric(a['integration'], 'float64', b['integration'].shape, 'actual integration')
  numeric(b['integration'], 'float64', a['integration'].shape, 'expected integration')
  require(byte_equal(a['integration'], b['integration']), 'Integration state bytes differ')
  for key in ('step_count', 'reset_next_step'):
    require(type(a[key]) is type(b[key]) and a[key] == b[key], 'Clock/reset state differs')
  require(type(actual['from_dm_done']) is type(expected['from_dm_done']) and
      actual['from_dm_done'] == expected['from_dm_done'], 'Done state differs')
  require(pickle.dumps(a['task'], protocol=5) == pickle.dumps(b['task'], protocol=5),
      'Task/RNG serialized state bytes differ')


def load_source(dataset, artifact_audit, physics_audit):
  """Hash-authenticated authority; only eight calibration NPZs are opened."""
  dataset = Path(dataset).resolve(strict=True)
  for name, expected in SOURCE_PINS.items():
    require(sha(safe_file(dataset, name)) == expected, 'Fixed source JSON differs: ' + name)
  for key, path in (('artifact', artifact_audit), ('physics', physics_audit)):
    require(sha(path) == AUDIT_PINS[key], 'Fixed source audit differs: ' + key)
  plan = read_json(dataset / 'collection_plan.json')
  complete = read_json(dataset / 'collection_complete.json')
  artifact_proof, physical_proof = read_json(artifact_audit), read_json(physics_audit)
  require(artifact_proof['source_contract_artifact_audit_passed'] is True and
      artifact_proof['expected_source_sha'] == physical_proof['expected_source_sha'] == SOURCE_GIT and
      artifact_proof['blind_arrays_read'] is False and physical_proof['status'] == 'passed' and
      physical_proof['blind_npz_arrays_opened'] is False and physical_proof['physics_steps'] == 12800 and
      physical_proof['source_json_sha256']['complete'] == SOURCE_PINS['collection_complete.json'],
      'Completed fixed source audits missing')
  exit_names = ('collection_plan.json', 'collection_complete.json', 'budget.json',
      'supervisor_complete.json', 'worker_exit_code.txt')
  exit_digests = {name: sha(safe_file(dataset, name)) for name in exit_names}
  require(all(exit_digests[name] == artifact_proof['metadata_sha256'][name] for name in exit_names),
      'Actual source budget/supervisor/worker exit bytes differ from pinned audit')
  source_budget = read_json(dataset / 'budget.json')
  source_supervisor = read_json(dataset / 'supervisor_complete.json')
  require(source_budget['version'] == source_supervisor['version'] == 'DL-source-r1' and
      source_budget['git_sha'] == source_supervisor['git_sha'] == SOURCE_GIT and
      source_budget['seconds'] == 3600 and
      source_budget['script_sha256'] == artifact_proof['source_file_sha256']['scripts/dl_production_source.py'] and
      source_supervisor['budget_sha256'] == exit_digests['budget.json'] and
      type(source_supervisor['exit_code']) is int and source_supervisor['exit_code'] == 0 and
      source_supervisor['timed_out'] is False and source_supervisor['worker_terminal'] is True and
      math.isfinite(source_supervisor['elapsed_seconds']) and
      0 < source_supervisor['elapsed_seconds'] <= 3600 and
      type(artifact_proof['supervisor_exit_code']) is int and artifact_proof['supervisor_exit_code'] == 0 and
      physical_proof['source_json_sha256']['budget'] == exit_digests['budget.json'] and
      physical_proof['source_json_sha256']['supervisor'] == exit_digests['supervisor_complete.json'] and
      (dataset / 'worker_exit_code.txt').read_bytes() == b'0\n',
      'Actual source completed exit/whole budget differs')
  require(plan['version'] == complete['version'] == 'DL-source-r1' and
      plan['implementation'] == complete['implementation'] == 'DL-source-engineering-r2' and
      plan['action_contract'] == complete['action_contract'] == CONTRACT and
      plan['metadata']['git_sha'] == SOURCE_GIT and
      complete['plan_sha256'] == SOURCE_PINS['collection_plan.json'], 'Source science/Git identity differs')
  for name, expected in MODEL_PINS.items():
    require(plan['metadata'][name] == expected, 'Source model pin differs: ' + name)
  require(complete['params_unchanged'] is True and complete['training_counters_unchanged'] is True and
      complete['optimizer_updates'] == 0 and len(complete['artifacts']) == 32, 'Source completion differs')
  packages = plan['environment']['packages']
  require({name: importlib.metadata.version(name) for name in packages} == packages,
      'Source and actual dependency versions differ')
  # All bytes may be hashed; blind arrays are never opened or decoded.
  entries = {}
  for entry in complete['artifacts']:
    ep = entry['episode']
    require(type(ep) is int and 0 <= ep < 32 and ep not in entries and
        entry['path'] == f'episode_{ep:03}.npz', 'Source episode/path differs')
    path = safe_file(dataset, entry['path'])
    require(sha(path) == entry['sha256'], 'Source NPZ bytes differ')
    entries[ep] = entry
  require({ep for ep, entry in entries.items() if entry['split'] == 'calibration'} == set(EPISODES),
      'Source calibration split differs')
  result = []
  for ep in EPISODES:
    path = safe_file(dataset, entries[ep]['path'])
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == entries[ep]['sha256'],
        'Source trusted bytes changed before decode')
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
      row = {name: archive[name].copy() for name in ('reward', 'action', 'action_applied',
          'action_executed', 'is_first', 'is_last', 'is_terminal', 'action_contract', 'snapshots')}
    require(row.pop('action_contract').item() == CONTRACT, 'Decoded source action contract differs')
    numeric(row['reward'], 'float32', (1001,), 'source reward')
    numeric(row['action'], 'float32', (1001, 12), 'source raw')
    numeric(row['action_applied'], 'float64', (1001, 12), 'source applied')
    for name in ('action_executed', 'is_first', 'is_last', 'is_terminal'):
      numeric(row[name], 'bool', (1001,), name)
    packed = row.pop('snapshots')
    require(packed.dtype == np.uint8 and packed.ndim == 1 and not packed.dtype.hasobject,
        'Trusted source snapshot bytes malformed')
    row['snapshots'] = pickle.loads(packed.tobytes())
    require(len(row['snapshots']) == 1001 and bool(row['is_first'][0]) and
        not row['is_last'][0] and not row['is_terminal'][0] and row['action_executed'][:20].all(),
        'True reset source and twenty actual actions required')
    row['artifact'] = entries[ep]
    result.append(row)
  return result, plan, exit_digests


def verify_official_model(env):
  import mujoco
  from dm_control.suite import quadruped
  require(sha(Path(quadruped.__file__).with_suffix('.xml')) == XML_SHA,
      'Official quadruped XML pin differs')
  require(type(env._task) is quadruped.Move and type(env.physics) is quadruped.Physics and
      env._task._desired_speed == .5, 'Official task/physics differs')
  names = tuple(f'{joint}_{leg}' for leg in
      ('front_left', 'front_right', 'back_right', 'back_left') for joint in ('yaw', 'lift', 'extend'))
  model = env.physics.model
  require(tuple(model.id2name(i, 'actuator') for i in range(model.nu)) == names and
      np.all(model.actuator_dyntype == int(mujoco.mjtDyn.mjDYN_FILTER)) and
      np.array_equal(model.actuator_dynprm[:, 0], np.full(12, .1)), 'Actuator identity/filter differs')
  require(model.opt.timestep == .005 and env.control_timestep() == .02 and
      env._step_limit == 1000, 'Official time contract differs')
  spec = env.action_spec()
  low, high = spec.minimum.copy(), spec.maximum.copy()
  numeric(low, 'float64', (12,), 'official low'); numeric(high, 'float64', (12,), 'official high')
  require(np.array_equal(low, np.tile([-1., -1., -.8], 4)) and
      np.array_equal(high, np.tile([1., 1.1, .8], 4)) and
      np.array_equal(np.stack([low, high], -1), model.actuator_ctrlrange), 'Official bounds differ')
  return low, high


def validate_record(record, control, original, state_size):
  require(type(record) is dict and record['version'] == 'DL-action-information-r1' and
      record['implementation'] == 'DL-action-information-physics-r1' and
      record['action_contract'] == CONTRACT and record['horizon'] == 20 and
      type(record['executed_steps']) is int and record['executed_steps'] == 20 and
      record['end_type'] == 'window' and len(record['snapshots']) == 21,
      'Record is not a full genuine reset H20 branch')
  for name, (dtype, shape) in FIELDS.items():
    numeric(record[name], dtype, shape[1:], 'record ' + name)
  numeric(record['first_raw'], 'float32', (12,), 'first raw')
  require(byte_equal(record['first_raw'], control) and byte_equal(record['raw'][0], control) and
      record['action_executed'].all() and record['snapshot_executed'].all() and
      not record['is_first'].any() and not record['is_last'].any() and
      not record['is_terminal'].any() and np.all(record['discount'] == 1.),
      'Control/terminal/full-execution semantics differ')
  require(byte_equal(record['raw'][1:], np.zeros((19, 12), np.float32)),
      'Common normalized-zero tail differs')
  low = np.tile([-1., -1., -.8], 4).astype(np.float64)
  high = np.tile([1., 1.1, .8], 4).astype(np.float64)
  names = [f'{joint}_{leg}' for leg in
      ('front_left', 'front_right', 'back_right', 'back_left') for joint in ('yaw', 'lift', 'extend')]
  require(record['model_receipt'] == dict(xml_sha256=XML_SHA, actuator_names=names,
      filter_tau=.1, physics_timestep=.005, control_timestep=.02, step_limit=1000,
      low=low.tolist(), high=high.tolist(), action_contract=CONTRACT), 'Record model receipt differs')
  for t in range(20):
    require(byte_equal(mapping(record['raw'][t], low, high), record['applied'][t]),
        'Independent mapped control differs')
  require(byte_equal(record['reward_f64'].astype(np.float32), record['reward_f32']),
      'F64/F32 reward encoding differs')
  require(type(original['suite']['step_count']) is int and original['suite']['step_count'] == 0,
      'Original source does not start at reset clock zero')
  require(original['from_dm_done'] is False and original['suite']['reset_next_step'] is False,
      'Original reset source has done/reset-next state')
  # Initial source -> primary uses the original <=1e-8 guard, recorded explicitly.
  old = np.asarray(original['suite']['integration'])
  initial = record['snapshots'][0]
  numeric(old, 'float64', (state_size,), 'source reset integration')
  numeric(initial['suite']['integration'], 'float64', (state_size,), 'primary reset integration')
  error = float(np.max(np.abs(old - initial['suite']['integration'])))
  require(error <= 1e-8 and pickle.dumps(original['suite']['task'], protocol=5) ==
      pickle.dumps(initial['suite']['task'], protocol=5), 'Primary/source reset binding differs')
  for t, saved in enumerate(record['snapshots']):
    numeric(saved['suite']['integration'], 'float64', (state_size,), 'snapshot integration')
    require(type(saved['suite']['step_count']) is int and saved['suite']['step_count'] == t and
        saved['from_dm_done'] is False and saved['suite']['reset_next_step'] is False,
        'Recorded reset/full-window clock or done differs')
  return error


def replay_record(env, record, restore, snapshot):
  """Exactly twenty stored-applied actions, independent byte-level oracle."""
  restore(env, record['snapshots'][0])
  complete_state_exact(snapshot(env, False), record['snapshots'][0])
  for t in range(20):
    check_deadline()
    require(not env._reset_next_step, 'Independent replay would reset')
    ts = env.step(record['applied'][t].copy())
    reward, discount = np.asarray(ts.reward), np.asarray(ts.discount, np.float64)
    numeric(reward, 'float64', (), 'official replay reward')
    numeric(discount, 'float64', (), 'official replay discount')
    require(reward.tobytes() == record['reward_f64'][t].tobytes() and
        np.asarray(reward, np.float32).tobytes() == record['reward_f32'][t].tobytes() and
        discount.tobytes() == record['discount'][t].tobytes(), 'Official reward/discount bytes differ')
    require((bool(ts.first()), bool(ts.last()), bool(ts.last() and discount == 0)) ==
        tuple(bool(record[name][t]) for name in ('is_first', 'is_last', 'is_terminal')),
        'Official flags differ')
    complete_state_exact(snapshot(env, bool(ts.last())), record['snapshots'][t + 1])
  return dict(physics_steps=20, complete_state_byte_exact=True,
      float64_reward_byte_exact=True, float32_reward_encoding_byte_exact=True,
      discount_byte_exact=True, flags_exact=True, padding_steps=0)


def analysis_report(value):
  """Translate independently computed results into the published whole schema."""
  all_pairs = [dict(episode_a=pair['episodes'][0], episode_b=pair['episodes'][1],
      minimax_regret=pair['minimax_regret'], minimax_regret_minus_delta=pair['minimax_minus_delta'],
      closed_delta_optimal_sets_disjoint=pair['disjoint']) for pair in value['pairs']]
  selected = [pair for pair in all_pairs if pair['closed_delta_optimal_sets_disjoint']]
  return dict(episodes=list(EPISODES), position=0, horizon=20,
      control_names=['zero', 'all+', 'all-', 'lift-LR+', 'lift-LR-', 'extend-diag+', 'extend-diag-'],
      delta=DELTA, rewards=value['costs'], range=value['range'],
      finite_library_regret=value['regret'], delta_optimal_membership=value['delta_optimal_membership'],
      delta_optimal_boundary='max_reward - reward <= delta',
      sensitive=[x >= DELTA for x in value['range']], cross_episode_disjoint_pairs=selected,
      disjoint_pair_count=len(selected), all_pair_minimax=all_pairs, sufficient=bool(selected),
      status=value['status'], decision='stop_after_fixed_natural_reset_calibration',
      minimax_margin_is_new_threshold=False, display_tolerance_applied=False,
      causal_rgb_or_model_information_established=False)


SOURCE_FILES = (
    'scripts/dl_natural_reset.py', 'tests/test_dl_natural_reset.py', 'docs/DL_NATURAL_RESET.md',
    'scripts/dl_action_information_physics.py', 'analysis/dl_task_semantics.py',
    'analysis/dl_feedback_physics_audit_v1.py', 'analysis/dl_natural_reset_audit.py',
    'tests/test_dl_natural_reset_audit.py')
RESTORE_SHA = '54e5b5a24fc9971ef8f454979a1ee9a9bc47c536f52bcac42b4c50efd96ef1ab'
OFFICIAL_PINS = {
    'dm_control.suite': '9b7002e4b70d7223376258173fff8ff7fa8ce53dd3535992f9e6dcb21a233760',
    'dm_control.suite.quadruped': '154662d7fb3ab576a013dbf3741ffb9d83009d791309fb50f2d33d9d5b04c91f',
    'dm_control.utils.rewards': '009f02389fd2af84b2955324707638e79a0e18b6a540dd7c254abf5096e157f3',
    'dm_control.rl.control': '3ac4b726242ea31f28235cfc94ff87ebbd7238bc71a673024281c74b9aef1606'}
AUTHORITY = dict(blind_arrays_read=False, current_rgb_increment_established=False,
    shared_model_history_verified=False, active_rep_compression_effect_established=False,
    model_calls=0, render_calls=0, optimizer_updates=0, gpu_devices_used=0,
    real_gate_authority=False, training_authorized=False, eligible_method_training=False,
    method_effectiveness_established=False, proceed_short_training=False)
OWNED_GROUPS = set()


def write_json(path, value):
  with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')


def kill_owned(child):
  if os.name == 'posix':
    try: os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError: pass
  elif child.poll() is None:
    child.kill()
  OWNED_GROUPS.discard(child.pid)
  child.wait(timeout=2)


def git_output(args):
  check_deadline()
  child = subprocess.Popen(['git', *args], cwd=ROOT, stdout=subprocess.PIPE,
      stderr=subprocess.PIPE, start_new_session=os.name != 'nt')
  OWNED_GROUPS.add(child.pid)
  try:
    out, err = child.communicate(timeout=max(1e-6, SECONDS - (time.monotonic() - ENTRY_STARTED)))
    require(child.returncode == 0, 'Git audit helper failed: ' + err.decode(errors='replace')[-300:])
    return out
  finally:
    kill_owned(child)


def git_binding(expected):
  require(re.fullmatch('[0-9a-f]{40}', expected or '') is not None, 'Full expected Git SHA required')
  require(git_output(['rev-parse', 'HEAD']).decode().strip() == expected and
      not git_output(['status', '--porcelain']).strip(), 'Actual clean execution Git differs')
  result = {}
  for name in SOURCE_FILES:
    payload = git_output(['show', expected + ':' + name])
    require((ROOT / name).read_bytes() == payload, 'Source differs from clean Git: ' + name)
    result[name] = hashlib.sha256(payload).hexdigest()
  return result


def primary_authority(directory, expected, code_pins):
  directory = Path(directory).resolve(strict=True)
  names = ('budget.json', 'plan.json', 'result.json', 'natural_reset_complete.json', 'supervisor_complete.json')
  files = {name: safe_file(directory, name) for name in names}
  digests = {name: sha(path) for name, path in files.items()}
  budget, plan, result, complete, supervisor = [read_json(files[name]) for name in names]
  for item in (budget, plan, result, complete, supervisor):
    require(item['version'] == VERSION and item['implementation'] == 'DL-natural-reset-engineering-r1' and
        item['actual_git_sha'] == expected, 'Primary object version/Git differs')
  require(budget['seconds'] == plan['seconds'] == 180 and
      budget['script_sha256'] == code_pins['scripts/dl_natural_reset.py'], 'Primary budget/source differs')
  for item in (plan, result, complete):
    require(item['source_files_sha256'] == {name: code_pins[name] for name in item['source_files_sha256']} and
        set(item['source_files_sha256']) == set(SOURCE_FILES[:6]), 'Primary source pins differ')
    for key, value in AUTHORITY.items():
      require(type(item[key]) is type(value) and item[key] == value, 'Primary authority boundary changed')
    require(item['budget_sha256'] == digests['budget.json'], 'Primary budget binding differs')
  require(result['plan_sha256'] == complete['plan_sha256'] == digests['plan.json'] and
      complete['result_sha256'] == digests['result.json'] and
      supervisor['budget_sha256'] == digests['budget.json'], 'Primary completion chain differs')
  require(supervisor['exit_code'] == 0 and supervisor['timed_out'] is False and
      supervisor['worker_terminal'] is True and supervisor['owned_group_cleanup_requested'] is True and
      supervisor['complete_verified'] is True and supervisor['reason'] == 'verified_fixed_cpu_object' and
      supervisor['elapsed_parent_entry_seconds'] <= 180 and
      not (directory / 'late_supervisor_receipt.json').exists(), 'Primary did not finish complete owned budget')
  require(plan['episodes'] == list(EPISODES) and plan['position'] == 0 and plan['horizon'] == 20 and
      plan['controls'] == 7 and plan['delta'] == DELTA and plan['hostname'] == 'lyg0326' and
      plan['platform'].startswith('linux'), 'Primary fixed runtime/object differs')
  for key, value in dict(complete_episodes=8, complete_states=8, complete_records=56,
      reference_physics_steps=160, main_physics_steps=1120, replay_physics_steps=1120,
      total_physics_steps=2400).items():
    require(type(complete[key]) is int and complete[key] == value, 'Partial primary completion: ' + key)
  require(result['source_inputs_unchanged'] is True and complete['source_inputs_unchanged'] is True and
      result['stopped_after_fixed_object'] is True and result['elapsed_parent_entry_seconds'] <= 180 and
      complete['status'] == result['analysis']['status'] and
      plan['source_binding'] == result['source_binding'], 'Primary source/stopping/report differs')
  require(result['reference_receipts'] == plan['reference_receipts'] and
      len(plan['reference_receipts']) == 8, 'Partial/different primary reference receipts')
  verify = supervisor['verification']
  require(verify['complete_verified'] is True and verify['physics_steps'] == 2400 and
      verify['actual_gpu_devices_used'] == 0 and verify['numeric_episode_reads'] == 8 and
      verify['blind_arrays_read'] is False and verify['result_sha256'] == digests['result.json'] and
      verify['complete_sha256'] == digests['natural_reset_complete.json'], 'Primary supervisor verification differs')
  return plan, result, digests


def load_restore():
  import importlib.util
  filename = ROOT / 'analysis/dl_feedback_physics_audit_v1.py'
  payload = filename.read_bytes()
  require(hashlib.sha256(payload).hexdigest() == RESTORE_SHA, 'Reused restore auditor bytes differ')
  spec = importlib.util.spec_from_file_location('natural_reset_reused_restore', filename)
  module = importlib.util.module_from_spec(spec)
  exec(compile(payload, str(filename), 'exec'), module.__dict__)
  require(module.VERSION == 'DL-independent-feedback-physics-audit-v1' and 'jax' not in sys.modules,
      'Reused restore dependency identity or no-JAX boundary differs')
  return module


def official_modules():
  from dm_control import suite
  from dm_control.suite import quadruped
  from dm_control.utils import rewards
  from dm_control.rl import control
  receipt = {}
  for module in (suite, quadruped, rewards, control):
    filename = Path(module.__file__).resolve(strict=True)
    digest = sha(filename)
    require(digest == OFFICIAL_PINS[module.__name__], 'Official module byte pin differs')
    receipt[module.__name__] = dict(path=str(filename), sha256=digest)
  return suite, receipt


def validate_npz(archive, episode):
  require(set(archive.files) == {'episode', 'position', 'records', *FIELDS}, 'Producer NPZ field axes differ')
  row = {name: archive[name].copy() for name in archive.files}
  numeric(row['episode'], 'int32', (), 'episode'); numeric(row['position'], 'int32', (), 'position')
  require(int(row['episode']) == episode and int(row['position']) == 0, 'NPZ episode/reset identity differs')
  for name, (dtype, shape) in FIELDS.items(): numeric(row[name], dtype, shape, name)
  packed = row['records']
  require(packed.dtype == np.uint8 and packed.ndim == 1 and packed.size > 0,
      'Trusted producer record payload malformed')
  records = pickle.loads(packed.tobytes())
  require(type(records) is list and len(records) == 7, 'Partial trusted producer record list')
  for a, record in enumerate(records):
    require(record['episode'] == episode and record['position'] == 0 and record['control'] == a,
        'Pickle record episode/reset/control identity differs')
    for name in FIELDS:
      require(byte_equal(row[name][a], record[name]), 'NPZ/pickle numeric input differs: ' + name)
  return row, records


def decode_producer_payload(payload, expected_sha, episode):
  """Authenticate the same bytes opened and unpickled; no pathname reread."""
  require(type(payload) is bytes and re.fullmatch('[0-9a-f]{64}', expected_sha or '') is not None and
      hashlib.sha256(payload).hexdigest() == expected_sha, 'Trusted payload SHA differs before decode')
  with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
    return validate_npz(archive, episode)


def audit(args):
  require(os.name == 'posix' and sys.platform.startswith('linux') and socket.gethostname() == 'lyg0326' and
      os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('MUJOCO_GL') == 'disable' and
      os.environ.get('JAX_PLATFORMS') == 'cpu', 'Actual audit requires assigned sv3 Linux CPU only')
  output = Path(args.output).resolve(strict=True)
  budget = read_json(output / 'budget.json')
  require(budget['parent_pid'] == os.getppid() == int(os.environ.get('DL_RESET_AUDIT_PARENT', '-1')) and
      budget['actual_git_sha'] == args.expected_git and budget['seconds'] == 180 and
      budget['script_sha256'] == sha(__file__), 'Live owned independent audit parent required')
  os.kill(os.getppid(), 0)
  code_pins = git_binding(args.expected_git)
  primary = Path(args.dataset).resolve(strict=True)
  plan, result, primary_pins = primary_authority(primary, args.expected_git, code_pins)
  sources, source_plan, source_exit_digests = load_source(
      args.source_dataset, args.artifact_audit, args.physics_audit)
  source_binding = plan['source_binding']
  source_root = Path(args.source_dataset).resolve(strict=True)
  source_complete = read_json(source_root / 'collection_complete.json')
  expected_hashes = [dict(episode=item['episode'], path=str(safe_file(source_root, item['path'])),
      sha256=item['sha256']) for item in source_complete['artifacts']]
  require(source_binding['source_json_sha256'] == dict(plan=SOURCE_PINS['collection_plan.json'],
      complete=SOURCE_PINS['collection_complete.json'], artifact_audit=AUDIT_PINS['artifact'],
      physics_audit=AUDIT_PINS['physics']) and
      source_binding['source_json_paths'] == dict(plan=str(source_root / 'collection_plan.json'),
          complete=str(source_root / 'collection_complete.json'),
          artifact_audit=str(Path(args.artifact_audit).resolve()),
          physics_audit=str(Path(args.physics_audit).resolve())) and
      source_binding['all_32_artifact_hashes'] == expected_hashes and
      source_binding['source_terminal_paths'] == {name: str(source_root / name) for name in
          ('budget.json', 'supervisor_complete.json', 'worker_exit_code.txt')} and
      source_binding['source_terminal_sha256'] == {name: source_exit_digests[name] for name in
          ('budget.json', 'supervisor_complete.json', 'worker_exit_code.txt')} and
      source_binding['source_original_exit_chain_verified'] is True and
      source_binding['actual_fields_read'] == ['reward', 'is_first', 'is_last', 'is_terminal',
          'action', 'action_applied', 'action_executed', 'action_contract', 'snapshots'] and
      source_binding['source_dataset'] == str(source_root) and
      source_binding['source_git_sha'] == SOURCE_GIT and source_binding['numeric_episode_reads'] == 8 and
      source_binding['trusted_snapshot_archives_decoded'] == 8 and
      source_binding['blind_arrays_read'] is False and source_binding['image_arrays_read'] is False and
      source_binding['before_carries_read'] is False and
      source_binding['source_model_metadata'] == source_plan['metadata'], 'Primary actual source binding differs')
  restore = load_restore(); suite, modules = official_modules()
  import mujoco
  artifacts = result['evidence_artifacts']
  require(len(artifacts) == 8 and tuple(item['episode'] for item in artifacts) == EPISODES,
      'Partial/reordered primary artifacts')
  receipts, all_rewards = [], []
  maximum_source_binding_error = 0.
  for index, episode in enumerate(EPISODES):
    check_deadline()
    item, source, reference = artifacts[index], sources[index], plan['reference_receipts'][index]
    require(reference['episode'] == episode and reference['position'] == 0 and reference['physics_steps'] == 20 and
        reference['source_complete_state_tolerance'] == 1e-8 and
        reference['source_float64_reward_byte_exact_claimed'] is False and
        reference['fresh_official_seed_reset_verified'] is True and reference['fresh_reset_calls'] == 1 and
        reference['fresh_reset_complete_integration_error'] <= 1e-8 and
        reference['maximum_complete_integration_error'] <= 1e-8 and
        reference['original_raw_applied_byte_exact'] is True and
        reference['source_reward_float32_byte_exact'] is True and reference['flags_exact'] is True and
        reference['complete_task_rng_clock_reset_done_exact'] is True,
        'Source reference/fresh reset receipt differs')
    original = source['snapshots'][0]
    original_sha = hashlib.sha256(pickle.dumps(original, protocol=5)).hexdigest()
    require(reference['source_initial_snapshot_sha256'] == item['original_initial_snapshot_sha256'] == original_sha,
        'Original source reset snapshot digest differs')
    require(item['path'] == f'episode_{episode:03}.npz' and item['complete_records'] == 7 and
        item['main_physics_steps'] == item['replay_physics_steps'] == 140 and len(item['replay_receipts']) == 7,
        'Producer artifact completion identity differs')
    path = safe_file(primary, item['path']); payload = path.read_bytes()
    row, records = decode_producer_payload(payload, item['sha256'], episode)
    seed = int(np.random.SeedSequence([20261020, episode, 0x444C]).generate_state(1)[0])
    require(seed == source['artifact']['seed'], 'Independent official source environment seed differs')
    env = suite.load('quadruped', 'walk', task_kwargs={'random': seed})
    try:
      verify_official_model(env)
      size = mujoco.mj_stateSize(env.physics.model.ptr, mujoco.mjtState.mjSTATE_INTEGRATION)
      episode_receipts = []
      for a, record in enumerate(records):
        error = validate_record(record, commands()[a], original, size)
        maximum_source_binding_error = max(error, maximum_source_binding_error)
        replay = item['replay_receipts'][a]
        require(replay['control'] == a and replay['physics_steps'] == 20 and replay['padding_steps'] == 0 and
            replay['complete_integration_task_rng_clock_reset_done_byte_exact'] is True and
            replay['maximum_integration_error'] == 0., 'Primary replay receipt differs')
        episode_receipts.append(dict(control=a, **replay_record(env, record, restore.restore, restore.snapshot)))
      receipts.append(dict(episode=episode, source_reset_sha256=original_sha,
          path=item['path'], sha256=item['sha256'], replays=episode_receipts))
    finally:
      env.close()
    all_rewards.append(row['reward_f64'])
  numerical = statistics(np.stack(all_rewards)); oracle = analysis_report(numerical)
  compare_tree(result['analysis'], oracle)
  # Independently rehash both authorities and every input after computation.
  for name, digest in primary_pins.items(): require(sha(primary / name) == digest, 'Primary JSON changed')
  for item in artifacts: require(sha(primary / item['path']) == item['sha256'], 'Primary arrays changed')
  for name, expected in SOURCE_PINS.items(): require(sha(Path(args.source_dataset) / name) == expected, 'Source JSON changed')
  for name, digest in source_exit_digests.items():
    require(sha(source_root / name) == digest, 'Source terminal budget/exit metadata changed')
  for path, expected in ((args.artifact_audit, AUDIT_PINS['artifact']), (args.physics_audit, AUDIT_PINS['physics'])):
    require(sha(path) == expected, 'Source audit changed')
  for item in source_binding['all_32_artifact_hashes']:
    require(sha(item['path']) == item['sha256'], 'Source array bytes changed')
  require(git_binding(args.expected_git) == code_pins and 'jax' not in sys.modules,
      'Code changed or prohibited model runtime imported')
  check_deadline()
  report = dict(version=VERSION, implementation=IMPLEMENTATION, actual_git_sha=args.expected_git,
      hostname=socket.gethostname(), platform=sys.platform, python=sys.version,
      status='passed', source_files_sha256=code_pins, primary_json_sha256=primary_pins,
      source_json_sha256=SOURCE_PINS, source_audit_sha256=AUDIT_PINS,
      source_terminal_metadata_sha256=source_exit_digests, source_original_exit_code=0,
      original_source_git=SOURCE_GIT, source_numeric_episode_reads=8, primary_numeric_episode_reads=8,
      source_snapshot_archives_decoded=8, primary_trusted_record_archives_decoded=8,
      physics_steps=1120, new_official_reset_calls=0, primary_fresh_resets_authenticated=8,
      source_f32_reference_authenticated_steps=160, source_f64_reference_claimed=False,
      repeated_restore_primitives_sha256=RESTORE_SHA, official_modules=modules,
      independent_statistics='F64 math.fsum / exhaustive seven-control minimax and closed delta membership',
      numerical_absolute_tolerance=1e-10, discrete_membership_classification_margin_zero_sides_exact=True,
      maximum_source_reset_binding_integration_error=maximum_source_binding_error,
      new_complete_state_and_rewards_byte_exact=True, independent_analysis=oracle,
      replay_receipts=receipts, actual_from_dm_wrapper_executed=False,
      independent_simulator=False, primary_reference_physics_reexecuted=False,
      elapsed_worker_entry_seconds=time.monotonic() - ENTRY_STARTED,
      budget_sha256=sha(output / 'budget.json'), **AUTHORITY)
  write_json(output / 'audit.json', report); check_deadline()


def verify_audit_receipt(output, expected):
  report = read_json(Path(output) / 'audit.json')
  require(report['version'] == VERSION and report['implementation'] == IMPLEMENTATION and
      report['actual_git_sha'] == expected and report['status'] == 'passed' and
      report['budget_sha256'] == sha(Path(output) / 'budget.json') and
      report['source_files_sha256'] == git_binding(expected), 'Independent receipt version/Git/source differs')
  for key, value in AUTHORITY.items():
    require(type(report[key]) is type(value) and report[key] == value, 'Independent authority boundary differs')
  for key, value in dict(physics_steps=1120, source_numeric_episode_reads=8,
      primary_numeric_episode_reads=8, source_snapshot_archives_decoded=8,
      primary_trusted_record_archives_decoded=8, new_official_reset_calls=0,
      primary_fresh_resets_authenticated=8, source_f32_reference_authenticated_steps=160).items():
    require(type(report[key]) is int and report[key] == value, 'Partial independent scope: ' + key)
  require(report['new_complete_state_and_rewards_byte_exact'] is True and
      report['discrete_membership_classification_margin_zero_sides_exact'] is True and
      report['source_f64_reference_claimed'] is False and
      report['primary_reference_physics_reexecuted'] is False and
      report['independent_simulator'] is False and report['actual_from_dm_wrapper_executed'] is False and
      report['numerical_absolute_tolerance'] == 1e-10 and len(report['replay_receipts']) == 8,
      'Independent scope/evidence boundary differs')
  require(sum(item['physics_steps'] for ep in report['replay_receipts'] for item in ep['replays']) == 1120,
      'Incomplete independent step receipts')
  return dict(complete_verified=True, actual_gpu_devices_used=0,
      source_numeric_episode_reads=8, primary_numeric_episode_reads=8, physics_steps=1120,
      model_calls=0, render_calls=0, optimizer_updates=0, audit_sha256=sha(Path(output) / 'audit.json'))


def supervise(command, output, *, seconds=SECONDS, started=None, expected_git=None, verify=None):
  """Whole-entry owned worker; generic fixtures have unknown actual scope."""
  require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0,
      'Positive finite independent budget required')
  started = time.monotonic() if started is None else started
  deadline = started + seconds
  output = Path(output).resolve()
  require(not output.exists(), 'Fresh independent audit directory required')
  output.mkdir(parents=True)
  write_json(output / 'budget.json', dict(version=VERSION, implementation=IMPLEMENTATION,
      parent_pid=os.getpid(), seconds=seconds, monotonic_start=started,
      actual_git_sha=expected_git, script_sha256=sha(__file__),
      planned_gpu_devices=0, planned_physics_steps=1120, actual_gpu_devices_used=None,
      actual_physics_steps=None))
  require(time.monotonic() < deadline, 'Independent budget expired before creation')
  timed_out, verification = False, None
  with (output / 'audit.log').open('x', encoding='utf-8') as log:
    child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=os.name != 'nt', env=dict(os.environ, DL_RESET_AUDIT_PARENT=str(os.getpid()),
          CUDA_VISIBLE_DEVICES='', MUJOCO_GL='disable', JAX_PLATFORMS='cpu',
          OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1'))
    OWNED_GROUPS.add(child.pid)
    try:
      code = child.wait(timeout=max(1e-6, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
      timed_out = True; code = 124
    finally:
      kill_owned(child)
  if code == 0 and not timed_out and verify is not None:
    try:
      verification = verify(output, expected_git)
    except Exception as error:
      write_json(output / 'completion_verification_failure.json', dict(error=repr(error))); code = 1
  if time.monotonic() >= deadline:
    timed_out = True; code = 124; verification = None
  real_verified = verify is verify_audit_receipt and bool(
      verification and verification.get('complete_verified') is True)
  receipt = dict(version=VERSION, implementation=IMPLEMENTATION, actual_git_sha=expected_git,
      parent_pid=os.getpid(), worker_pid=child.pid, exit_code=code, timed_out=timed_out,
      worker_terminal=child.poll() is not None, owned_group_cleanup_requested=os.name != 'nt',
      elapsed_parent_entry_seconds=time.monotonic() - started,
      complete_verified=real_verified, verification=verification,
      verification_kind='actual_whole_audit_chain' if real_verified else 'generic_callback_unknown_actual_scope',
      actual_gpu_devices_used=0 if real_verified else None,
      actual_physics_steps=1120 if real_verified else None,
      reason='incomplete_hard_budget' if timed_out else 'verified_independent_reset_audit' if real_verified else
          'generic_worker_exited' if code == 0 else 'worker_or_chain_failed',
      budget_sha256=sha(output / 'budget.json'))
  # Publish success only after receipt serialization remains inside the budget.
  # A delayed write or rename retains the provisional evidence and publishes an
  # explicitly failed final receipt, never a green stale completion file.
  pending = output / 'supervisor_receipt_pending.json'
  write_json(pending, receipt)
  late = time.monotonic() >= deadline
  if not late:
    os.replace(pending, output / 'supervisor_complete.json')
    late = time.monotonic() >= deadline
  if late and not timed_out:
    write_json(output / 'late_supervisor_receipt.json', dict(incomplete=True, exit_code=124))
    provisional = output / 'supervisor_complete.json'
    if provisional.exists(): os.replace(provisional, output / 'provisional_supervisor_receipt.json')
    receipt.update(exit_code=124, timed_out=True, complete_verified=False, verification=None,
        actual_gpu_devices_used=None, actual_physics_steps=None, reason='incomplete_receipt_hard_budget')
    write_json(output / 'supervisor_complete.json', receipt)
  elif late:
    os.replace(pending, output / 'supervisor_complete.json')
  return receipt


def _expire(signum, frame):
  for pid in tuple(OWNED_GROUPS):
    try: os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError: pass
  os._exit(124)


def main():
  if os.name == 'posix':
    signal.signal(signal.SIGALRM, _expire)
    signal.setitimer(signal.ITIMER_REAL, max(1e-6, 180 - (time.monotonic() - ENTRY_STARTED)))
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('dataset', 'source-dataset', 'artifact-audit', 'physics-audit', 'output', 'expected-git'):
    parser.add_argument('--' + name, required=True)
  parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.worker:
    audit(args); return 0
  receipt = supervise([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker'],
      args.output, started=ENTRY_STARTED, expected_git=args.expected_git, verify=verify_audit_receipt)
  print(json.dumps(receipt, sort_keys=True), flush=True)
  return receipt['exit_code']


if __name__ == '__main__':
  raise SystemExit(main())
