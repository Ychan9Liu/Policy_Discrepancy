"""Independent, bounded CPU physics audit of trusted DL-feedback-r1 records.

Uses stdlib, NumPy, ml_dtypes (only to register BF16 trusted-pickle dtype), and
official dm_control/MuJoCo. Never imports project code, JAX, a model, or a
renderer. Reuses this reviewer's independently written source-audit restore
approach; this is independent from the production helper, not an independent
simulator or a second author. No utility, D/K/C, bootstrap or policy is computed.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import pickle
import re
import signal
import sys
import time

os.environ['MUJOCO_GL'] = 'disable'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
import numpy as np
import ml_dtypes  # BF16 registration for the complete trusted record only.

VERSION = 'DL-independent-feedback-physics-audit-v1'
SCIENCE = 'DL-feedback-r1'
ENGINEERING = 'DL-feedback-engineering-r2'
PHYSICAL = 'DL-feedback-physics-r1'
CONTRACT = 'production_clip_unify_normalize_v1'
CALIBRATION = [9, 14, 15, 18, 20, 24, 30, 31]
POSITIONS = [1, 300, 600, 900]
HORIZON = 100
TOLERANCE = 1e-8  # Original complete-state contract, not a new allowance.
PINS = dict(
    source_implementation_sha='f2d24da1831da2ffca22d313e7f534c96fc0d3eb',
    source_plan_sha256='633efe28d551d5793ac8f0ed5f0e36e937e7f1af3c8edefc1d9f5fa278146fcc',
    source_complete_sha256='d155f577d19d2bbc051ec56b95f15e26afc774c130c3c9728a7fe599f295f3cf',
    artifact_audit_sha256='8f93788ca212118139fab135036fd47ccbad79f04a34139b5dfa7d2dd26f7516',
    physics_audit_sha256='8ed3528fae18dc7a3043c767da87f873642a40a1a028ce86247f0521c3264722')
MODEL_PINS = dict(
    checkpoint_sha256='2ab693c8ded4d8f2186bd489aa8b77498448ba2f2c331090159da4e5f08056b9',
    params_sha256='14859c4e129b51aa488c10eab09f16d56fd110c76239b2c9072f415829d111c2',
    config_source_sha256='11428512fdc9b9e04a5384e419c5abcef93e5429db0a3327d05ccb0d4bd20445')
ARRAYS = {
    'keys': (np.uint32, (100, 2)), 'raw': (np.float32, (100, 12)),
    'applied': (np.float64, (100, 12)), 'reward_f64': (np.float64, (100,)),
    'reward_f32': (np.float32, (100,)), 'discount': (np.float64, (100,)),
    'is_first': (np.bool_, (100,)), 'is_last': (np.bool_, (100,)),
    'is_terminal': (np.bool_, (100,)), 'action_executed': (np.bool_, (100,)),
    'observation_executed': (np.bool_, (101,))}
OBS_KEYS = {'image', 'reward', 'is_first', 'is_last', 'is_terminal'}


class AuditError(ValueError):
  pass


def require(condition, message):
  if not condition:
    raise AuditError(message)


def sha256(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def json_file(path):
  def reject(value):
    raise AuditError('Nonfinite JSON constant: ' + value)
  return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)


def byte_equal(a, b):
  return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and
      a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes())


def array(value, dtype, shape, name):
  require(isinstance(value, np.ndarray) and value.dtype == np.dtype(dtype) and
      value.shape == shape and not value.dtype.hasobject, name + ': dtype/shape/object differs')
  require(np.isfinite(value).all(), name + ': nonfinite')


def named_keys(episode, position, pair):
  return np.stack([np.random.default_rng([20261018, 0x46445231,
      episode, position, pair, position + step]).integers(
      0, 4294967295, size=(2,), dtype=np.uint32) for step in range(HORIZON)])


def environment_seed(episode):
  return int(np.random.SeedSequence([20261020, episode, 0x444C]).generate_state(1)[0])


def mapping(raw, low, high):
  clipped = np.clip(raw, -1, 1).astype(np.float32, copy=False)
  fraction = (clipped + 1) / 2
  return fraction * (high - low) + low


def snapshot(env, done):
  import mujoco
  bits = mujoco.mjtState.mjSTATE_INTEGRATION
  state = np.empty(mujoco.mj_stateSize(env.physics.model.ptr, bits), np.float64)
  mujoco.mj_getState(env.physics.model.ptr, env.physics.data.ptr, state, bits)
  return dict(suite=dict(integration=state, step_count=env._step_count,
      reset_next_step=env._reset_next_step, task=copy.deepcopy(env._task.__dict__)),
      from_dm_done=bool(done))


def validate_snapshot(value, size):
  require(isinstance(value, dict) and set(value) == {'suite', 'from_dm_done'},
      'Malformed complete snapshot')
  state = value['suite']
  require(isinstance(state, dict) and set(state) ==
      {'integration', 'step_count', 'reset_next_step', 'task'}, 'Incomplete suite state')
  array(state['integration'], np.float64, (size,), 'integration')
  require(type(state['step_count']) is int and state['step_count'] >= 0,
      'Nonnegative Python suite clock required')
  require(isinstance(value['from_dm_done'], (bool, np.bool_)) and
      isinstance(state['reset_next_step'], (bool, np.bool_)) and
      isinstance(state['task'], dict), 'Invalid wrapper/reset/task receipt')


def restore(env, saved):
  import mujoco
  state = saved['suite']
  env._task.__dict__.clear()
  env._task.__dict__.update(copy.deepcopy(state['task']))
  env._step_count = state['step_count']
  env._reset_next_step = bool(state['reset_next_step'])
  mujoco.mj_setState(env.physics.model.ptr, env.physics.data.ptr,
      state['integration'], mujoco.mjtState.mjSTATE_INTEGRATION)
  env.physics.forward()


def compare_state(actual, expected):
  a, b = actual['suite'], expected['suite']
  array(a['integration'], np.float64, b['integration'].shape, 'replayed integration')
  require(actual['from_dm_done'] == expected['from_dm_done'], 'Expected FromDM done differs')
  require(a['step_count'] == b['step_count'] and
      a['reset_next_step'] == b['reset_next_step'], 'Suite clock/reset differs')
  require(pickle.dumps(a['task'], protocol=5) == pickle.dumps(b['task'], protocol=5),
      'Complete task/RNG bytes differ')
  error = float(np.max(np.abs(a['integration'] - b['integration'])))
  require(np.isfinite(error) and error <= TOLERANCE,
      'Complete integration exceeds original 1e-8: ' + str(error))
  return error


def observation(obs):
  require(isinstance(obs, dict) and set(obs) == OBS_KEYS, 'Observation fields differ')
  array(obs['image'], np.uint8, (64, 64, 3), 'cached image')
  array(obs['reward'], np.float32, (), 'cached production reward')
  for key in ('is_first', 'is_last', 'is_terminal'):
    array(obs[key], np.bool_, (), key)
  require(not bool(obs['is_terminal']) or bool(obs['is_last']), 'Terminal is not last')


def validate_record(env, record):
  import mujoco
  require(record['version'] == SCIENCE and record['implementation'] == PHYSICAL and
      record['action_contract'] == CONTRACT and record['horizon'] == HORIZON,
      'Legacy/unknown branch contract')
  count = record['executed_steps']
  require(type(count) is int and 1 <= count <= HORIZON and type(record['prior']) is bool,
      'Branch count/prior differs')
  for name, (dtype, shape) in ARRAYS.items():
    array(record[name], dtype, shape, name)
  require(byte_equal(record['action_executed'], np.arange(HORIZON) < count) and
      byte_equal(record['observation_executed'], np.arange(HORIZON + 1) <= count),
      'Genuine executed prefix mask differs')
  for name in set(ARRAYS) - {'keys', 'action_executed', 'observation_executed'}:
    require(byte_equal(record[name][count:], np.zeros_like(record[name][count:])),
        'Nonzero/signed-zero absorbing padding: ' + name)
  require(len(record['observations']) == len(record['snapshots']) == count + 1 and
      len(record['proofs']) == count, 'Genuine observation/snapshot/proof coverage differs')
  size = mujoco.mj_stateSize(env.physics.model.ptr, mujoco.mjtState.mjSTATE_INTEGRATION)
  for obs, saved in zip(record['observations'], record['snapshots']):
    observation(obs)
    validate_snapshot(saved, size)
  initial = record['snapshots'][0]
  require(not initial['from_dm_done'] and not initial['suite']['reset_next_step'] and
      not bool(record['observations'][0]['is_first']) and
      not bool(record['observations'][0]['is_last']), 'Inactive/reset initial state')
  if 'position' in record:
    require(initial['suite']['step_count'] == record['position'], 'Initial absolute clock differs')
  for step in range(count):
    obs, saved = record['observations'][step + 1], record['snapshots'][step + 1]
    require(record['reward_f32'][step].tobytes() == obs['reward'].tobytes() and
        np.asarray(record['reward_f64'][step], np.float32).tobytes() == obs['reward'].tobytes(),
        'F64/F32/cached reward encoding differs')
    require(record['discount'][step] in (0, 1), 'Discount outside frozen 0/1 contract')
    flags = tuple(bool(record[k][step]) for k in ('is_first', 'is_last', 'is_terminal'))
    require(flags == tuple(bool(obs[k]) for k in ('is_first', 'is_last', 'is_terminal')) and
        not flags[0] and flags[2] == (record['discount'][step] == 0) and
        (not flags[1] or step == count - 1), 'Transition flags/terminal alignment differs')
    require(saved['suite']['step_count'] == initial['suite']['step_count'] + step + 1 and
        bool(saved['from_dm_done']) == flags[1] and
        bool(saved['suite']['reset_next_step']) == flags[1], 'Saved counter/done/reset differs')
  last, terminal = bool(record['is_last'][count - 1]), bool(record['is_terminal'][count - 1])
  end = 'true_terminal' if terminal else ('time_limit' if last else 'window')
  require(record['end_type'] == end and (count == HORIZON or terminal),
      'Incomplete window is not genuine absorbing terminal')
  return count


def replay_record(env, record, check_deadline=lambda: None):
  count = validate_record(env, record)
  spec = env.action_spec()
  require(spec.dtype == np.float64 and spec.shape == (12,), 'Official action spec differs')
  require(byte_equal(mapping(record['raw'][:count], spec.minimum, spec.maximum),
      record['applied'][:count]), 'Independent raw/applied byte mapping differs')
  restore(env, record['snapshots'][0])
  restore_error = compare_state(snapshot(env, False), record['snapshots'][0])
  maximum = restore_error
  for step in range(count):
    check_deadline()
    require(not env._reset_next_step, 'Replay would call reset after last')
    # Actual recorded applied control, never remapped or re-normalized.
    ts = env.step(record['applied'][step].copy())
    reward, discount = np.asarray(ts.reward), np.asarray(ts.discount, np.float64)
    array(reward, np.float64, (), 'official reward')
    array(discount, np.float64, (), 'official discount')
    require(not ts.first() and discount in (0, 1) and
        (discount != 0 or ts.last()), 'Official transition/discount semantics differ')
    require(reward.tobytes() == record['reward_f64'][step].tobytes(),
        'Official F64 reward byte replay differs')
    require(np.asarray(reward, np.float32).tobytes() == record['reward_f32'][step].tobytes(),
        'Official F32 reward encoding byte replay differs')
    require(discount.tobytes() == record['discount'][step].tobytes(),
        'Official discount byte replay differs')
    flags = (bool(ts.first()), bool(ts.last()), bool(ts.last() and discount == 0))
    require(flags == tuple(bool(record[k][step]) for k in ('is_first', 'is_last', 'is_terminal')),
        'Official first/last/terminal flags differ')
    maximum = max(maximum, compare_state(snapshot(env, ts.last()), record['snapshots'][step + 1]))
    require(not ts.last() or step == count - 1, 'Replay crossed true terminal/reset')
    if ts.last() and discount == 1:
      require(step == HORIZON - 1 and env._step_count == env._step_limit,
          'Early/unexpected time limit')
  return dict(physics_steps=count, padding_steps=HORIZON-count, end_type=record['end_type'],
      maximum_integration_error=maximum, maximum_restore_error=restore_error)


def authority(root, expected_sha):
  require(re.fullmatch('[0-9a-f]{40}', expected_sha or '') is not None, 'Full expected Git SHA required')
  names = dict(plan='plan.json', result='result.json', complete='feedback_complete.json',
      budget='budget.json', supervisor='supervisor_complete.json')
  paths = {key: root/name for key, name in names.items()}
  hashes = {key: sha256(path) for key, path in paths.items()}
  plan, result, complete, budget, supervisor = (json_file(paths[x]) for x in names)
  require(all(x['version'] == SCIENCE and x['implementation'] == ENGINEERING
      for x in (plan, result, complete, budget, supervisor)), 'Science/engineering version differs')
  require(plan['metadata']['git_sha'] == budget['git_sha'] == result['actual_git_sha'] ==
      complete['actual_git_sha'] == supervisor['actual_git_sha'] == expected_sha, 'Run Git binding differs')
  require(result['plan_sha256'] == complete['plan_sha256'] == hashes['plan'] and
      complete['result_sha256'] == hashes['result'] and
      supervisor['budget_sha256'] == hashes['budget'], 'Complete/result/plan/budget binding differs')
  require(supervisor['exit_code'] == 0 and supervisor['timed_out'] is False and
      supervisor['worker_terminal'] is True and supervisor['reason'] == 'worker_completed' and
      (root/'worker_exit_code.txt').read_text(encoding='utf-8').strip() == '0',
      'Authority requires complete successful terminal supervisor')
  require(plan['budget_seconds'] == budget['seconds'] == 3600 and
      np.isfinite(supervisor['elapsed_seconds']) and 0 < supervisor['elapsed_seconds'] <= 3600,
      'Complete run exceeded/falsified hard budget')
  require(np.isfinite(result['elapsed_seconds']) and
      0 < result['elapsed_seconds'] <= supervisor['elapsed_seconds'], 'Worker/supervisor timing differs')
  require(complete['complete_episodes'] == 8 and complete['complete_records'] == 256 and
      plan['action_contract'] == result['action_contract'] == CONTRACT and
      plan['positions'] == POSITIONS and plan['position_indices'] == [0,5,10,15] and
      plan['pairs'] == 4 and plan['horizon'] == HORIZON and
      plan['private_seed'] == 20261018 and plan['stat_seed'] == 20261019 and
      plan['branch_order'] == ['posterior','one_step_prior_then_posterior'], 'Frozen scope differs')
  require(plan['key_convention'] == dict(tag=0x46445231, absolute_step='position+s',
      high_exclusive=4294967295, branch_absent=True), 'Named key convention differs')
  for key, value in MODEL_PINS.items():
    require(plan['metadata'][key] == value, 'Model provenance differs: ' + key)
  require(plan['metadata']['code_version'] == SCIENCE, 'Model science binding differs')
  for obj in (plan, result):
    binding = obj['source_binding']
    require(all(binding[k] == v for k,v in PINS.items()) and
        binding['calibration_episodes'] == CALIBRATION and
        binding['all_32_artifact_hashes_verified'] is True and binding['blind_arrays_read'] is False,
        'Completed corrected-source provenance differs')
  for obj in (plan, result, complete):
    require(obj['optimizer_updates'] == 0 and obj['training_authorized'] is False,
        'Unexpected training/update authorization')
  for obj in (result, complete):
    require(obj['params_unchanged'] is True and obj['counters_unchanged'] is True,
        'Read-only parameters/counters not verified')
  require(result['counters_before'] == result['counters_after'] and
      plan['gate_changed'] is False and plan['blind_arrays_read'] is False and
      result['blind_arrays_read'] is False and
      result['full_closed_loop_bit_exact_established'] is False and
      result['renderer_byte_exact_established'] is False, 'Readonly/scope flags differ')
  for obj in (result, complete, supervisor):
    for key in ('training_authorized','proceed_short_training','eligible_method_training',
                'method_effectiveness_established'):
      require(obj[key] is False, 'Forbidden method/training conclusion: ' + key)
  require([x['definition']['episode'] for x in plan['calibration']] == CALIBRATION and
      len(plan['source_oracles']) == 32, 'Calibration/source-oracle coverage differs')
  for name in ('numpy','dm-control','mujoco'):
    require(plan['environment']['packages'][name] == importlib.metadata.version(name),
        'Official CPU runtime package differs: ' + name)
  artifacts = result['evidence_artifacts']
  require([x['episode'] for x in artifacts] == CALIBRATION and len(artifacts) == 8,
      'Feedback artifact calibration coverage differs')
  require({p.name for p in root.glob('*.npz')} == {f'episode_{ep:03}.npz' for ep in CALIBRATION},
      'Unexpected/missing/failed NPZ artifacts')
  for item in artifacts:
    require(item['path'] == f"episode_{item['episode']:03}.npz" and item['records'] == 32 and
        re.fullmatch('[0-9a-f]{64}', item['sha256']) is not None and
        sha256(root/item['path']) == item['sha256'], 'Original artifact byte hash/path differs')
    require(len(item['verifications']) == 32, 'Producer per-branch verification coverage differs')
  return plan, result, hashes


def audit_dataset(dataset, expected_sha, seconds):
  started = time.monotonic()
  require(np.isfinite(seconds) and 0 < seconds <= 600, 'CPU audit deadline must be 0..600 seconds')
  def check():
    if time.monotonic() - started >= seconds:
      raise TimeoutError('Independent CPU audit deadline exhausted; incomplete')
  root = Path(dataset).resolve(strict=True)
  plan, result, json_hashes = authority(root, expected_sha)
  check()
  from dm_control import suite
  per_episode, maximum, restore_maximum, total, padding, branches = [], 0., 0., 0, 0, 0
  for item, cal in zip(result['evidence_artifacts'], plan['calibration']):
    check()
    ep = item['episode']
    require(cal['definition']['seed'] == environment_seed(ep) and
        cal['definition']['split'] == 'calibration' and cal['carry_positions'] == POSITIONS,
        'Source environment seed/position/split differs')
    payload = (root/item['path']).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == item['sha256'], 'Artifact changed after initial hash')
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
      require(set(archive.files) == set(ARRAYS) | {'episode','positions','source_image','fresh_image','records'},
          'NPZ schema differs')
      values = {name: archive[name] for name in archive.files}
    array(values['episode'], np.int32, (), 'episode')
    array(values['positions'], np.int32, (4,), 'positions')
    require(int(values['episode']) == ep and byte_equal(values['positions'], np.array(POSITIONS,np.int32)),
        'NPZ episode/position binding differs')
    for name in ('source_image','fresh_image'):
      array(values[name], np.uint8, (4,64,64,3), name)
    for name, (dtype, shape) in ARRAYS.items():
      array(values[name], dtype, (4,4,2) + shape, 'NPZ '+name)
    raw_pickle = values['records']
    require(isinstance(raw_pickle,np.ndarray) and raw_pickle.dtype == np.uint8 and
        raw_pickle.ndim == 1 and raw_pickle.size > 0, 'Trusted records envelope differs')
    # Full pickle is decoded only after successful completed-run/hash authority.
    records = pickle.loads(raw_pickle.tobytes())
    require(isinstance(records,list) and len(records) == 32, 'Trusted record coverage differs')
    env = suite.load('quadruped','walk',task_kwargs={'random':environment_seed(ep)})
    episode_receipts, reference = [], {}
    try:
      spec = env.action_spec()
      source_spec = cal['source_artifact']['action_spec']
      require(source_spec['dtype'] == 'float64' and source_spec['raw_dtype'] == 'float32' and
          spec.dtype == np.float64 and spec.shape == (12,) and
          byte_equal(spec.minimum,np.asarray(source_spec['low'],np.float64)) and
          byte_equal(spec.maximum,np.asarray(source_spec['high'],np.float64)) and
          byte_equal(np.stack([spec.minimum,spec.maximum],-1),
              np.asarray(env.physics.model.actuator_ctrlrange)), 'Actual/source/actuator control bounds differ')
      for index, record in enumerate(records):
        check()
        position_index, pair, branch = np.unravel_index(index,(4,4,2))
        position = POSITIONS[position_index]
        require((record['episode'],record['position'],record['pair'],record['branch']) ==
            (ep,position,int(pair),int(branch)) and record['prior'] is bool(branch),
            'Trusted record branch/order identity differs')
        for name in ARRAYS:
          require(byte_equal(record[name],values[name][position_index,pair,branch]),
              'NPZ/trusted-record numeric bytes differ: '+name)
        require(byte_equal(record['keys'],named_keys(ep,position,int(pair))), 'Private named keys differ')
        require(byte_equal(record['observations'][0]['image'],values['fresh_image'][position_index]),
            'Recorded initial cached RGB differs from shared fresh initial frame')
        receipt = replay_record(env,record,check)
        if position not in reference:
          reference[position] = record
        else:
          compare_state(record['snapshots'][0],reference[position]['snapshots'][0])
          for key in OBS_KEYS:
            require(byte_equal(record['observations'][0][key],reference[position]['observations'][0][key]),
                'Paired branches have different initial observations')
          require(pickle.dumps(record['initial_carry'],protocol=5) ==
              pickle.dumps(reference[position]['initial_carry'],protocol=5),
              'Paired initial full carry bytes differ')
        verification = item['verifications'][index]
        require((verification['position'],verification['pair'],verification['branch']) ==
            (position,int(pair),int(branch)) and verification['executed_steps'] == receipt['physics_steps'] and
            verification['end_type'] == receipt['end_type'], 'Producer verification branch/count differs')
        maximum = max(maximum,receipt['maximum_integration_error'])
        restore_maximum = max(restore_maximum,receipt['maximum_restore_error'])
        total += receipt['physics_steps']; padding += receipt['padding_steps']; branches += 1
        episode_receipts.append(dict(position=position,pair=int(pair),branch=int(branch),**receipt))
      require(item['physics_steps'] == sum(x['physics_steps'] for x in episode_receipts),
          'Producer physical replay step total differs')
    finally:
      env.close()
    per_episode.append(dict(episode=ep,artifact_sha256=item['sha256'],branches=episode_receipts))
    print(json.dumps(dict(stage='independent_physics',episode=ep,records=32,
        steps=sum(x['physics_steps'] for x in episode_receipts))),flush=True)
  check()
  require(branches == 256 and total+padding == 25600, 'Bounded complete physical coverage differs')
  return dict(version=VERSION,status='passed',expected_feedback_sha=expected_sha,dataset=str(root),
      science=SCIENCE,implementation=ENGINEERING,source_binding=PINS,feedback_json_sha256=json_hashes,
      calibration_episodes=CALIBRATION,complete_records=branches,physics_steps=total,padding_steps=padding,
      maximum_integration_error=maximum,maximum_restore_error=restore_maximum,
      float64_reward_byte_exact=True,float32_reward_encoding_byte_exact=True,discount_byte_exact=True,
      first_last_terminal_exact=True,task_rng_pickle_exact=True,suite_counter_reset_exact=True,
      from_dm_expected_done_exact=True,raw_applied_mapping_byte_exact=True,
      model_executed=False,jax_imported=('jax' in sys.modules),renderer_executed=False,
      actual_from_dm_wrapper_executed=False,original_source_initial_arrays_reopened=False,
      model_policy_rng_verified=False,utility_or_signals_computed=False,blind_arrays_read=False,
      training_authorized=False,method_effectiveness_established=False,
      ml_dtypes_dtype_registration_only=True,deadline_seconds=seconds,elapsed_seconds=time.monotonic()-started,
      environment={name:importlib.metadata.version(name) for name in ('numpy','ml_dtypes','dm-control','mujoco')},
      episodes=per_episode,limitations=[
          'Conditional replay from hash-bound saved initial snapshots, not fresh independent source-policy generation.',
          'No model, latent/actor-key execution, rendering or actual FromDM wrapper; done is checked against official last.',
          'Same official physics backend and source-auditor author; independent of production helper, not second simulator.',
          'Does not recompute utility/statistics or establish representation/method effectiveness.'])


def self_test():
  from dm_control import suite
  from unittest import mock
  env = suite.load('quadruped','walk',task_kwargs={'random':environment_seed(9)})
  tests=[]
  try:
    env.reset()
    env.step(np.zeros(12,np.float64))
    def obs(ts):
      return dict(image=np.zeros((64,64,3),np.uint8),reward=np.asarray(ts.reward,np.float32),
          is_first=np.asarray(ts.first(),np.bool_),is_last=np.asarray(ts.last(),np.bool_),
          is_terminal=np.asarray(ts.last() and ts.discount == 0,np.bool_))
    record=dict(version=SCIENCE,implementation=PHYSICAL,action_contract=CONTRACT,horizon=100,
        prior=False,executed_steps=100,end_type='window',observations=[],snapshots=[snapshot(env,False)],
        proofs=[{} for _ in range(100)])
    for name,(dtype,shape) in ARRAYS.items():
      record[name]=np.zeros(shape,dtype)
    record['observations']=[dict(image=np.zeros((64,64,3),np.uint8),
        reward=np.array(0,np.float32),is_first=np.array(False),is_last=np.array(False),is_terminal=np.array(False))]
    record['action_executed'][:]=True; record['observation_executed'][:]=True
    record['raw'][:]=np.random.default_rng(71).normal(size=(100,12)).astype(np.float32)
    spec=env.action_spec()
    record['applied'][:]=mapping(record['raw'],spec.minimum,spec.maximum)
    for step in range(100):
      ts=env.step(record['applied'][step].copy())
      record['reward_f64'][step]=ts.reward
      record['reward_f32'][step]=np.float32(ts.reward)
      record['discount'][step]=ts.discount
      for name in ('is_first','is_last','is_terminal'):
        record[name][step]=obs(ts)[name]
      record['observations'].append(obs(ts));record['snapshots'].append(snapshot(env,ts.last()))
    result=replay_record(env,record)
    require(result['physics_steps']==100 and result['maximum_integration_error']==0,
        'Valid official quadruped selftest failed')
    tests.append('official_quadruped_100_steps_full_state_and_F64_F32_bytes')
    def rejected(changed,label):
      try:
        replay_record(env,changed)
      except (AuditError,ValueError,TypeError):
        tests.append(label);return
      raise AuditError('Tampered fixture accepted: '+label)
    changed=copy.deepcopy(record)
    original=changed['reward_f64'][3]
    changed['reward_f64'][3]=np.nextafter(original,np.float64(np.inf))
    require(changed['reward_f64'][3].tobytes()!=original.tobytes() and
        np.float32(changed['reward_f64'][3]).tobytes()==np.float32(original).tobytes(),
        'One-bit F64 tamper fixture not meaningful')
    rejected(changed,'F64_single_ulp_change_same_F32_rejected')
    changed=copy.deepcopy(record);changed['snapshots'][0]['suite']['integration'][0]=np.nan
    rejected(changed,'nonfinite_complete_integration_rejected')
    changed=copy.deepcopy(record);changed['reward_f64'][4]=np.nan
    rejected(changed,'nonfinite_reward_rejected')
    changed=copy.deepcopy(record);changed['applied'][0,0]=np.nextafter(changed['applied'][0,0],np.inf)
    rejected(changed,'recorded_applied_single_ulp_tamper_rejected')
    changed=copy.deepcopy(record);changed['snapshots'][1]['from_dm_done']=True
    rejected(changed,'expected_FromDM_done_tamper_rejected')
    changed=copy.deepcopy(record);changed['action_executed'][4]=False
    rejected(changed,'executed_mask_gap_rejected')
    bf16=np.array([1,2],dtype=ml_dtypes.bfloat16)
    require(byte_equal(pickle.loads(pickle.dumps(bf16,protocol=5)),bf16), 'BF16 registration failed')
    tests.append('full_trusted_BF16_dtype_registration_no_JAX')
    def bounded_fixture(start, terminal=False):
      restore(env,start)
      other=copy.deepcopy(record)
      for name,(dtype,shape) in ARRAYS.items():
        other[name]=np.zeros(shape,dtype)
      other['raw'][:]=record['raw']
      other['applied'][:]=record['applied']
      other['observations']=[copy.deepcopy(record['observations'][0])]
      other['snapshots']=[snapshot(env,False)];other['proofs']=[]
      other['observation_executed'][0]=True
      for step in range(100):
        ts=env.step(other['applied'][step].copy())
        next_obs=obs(ts)
        other['reward_f64'][step]=ts.reward
        other['reward_f32'][step]=np.float32(ts.reward)
        other['discount'][step]=ts.discount
        for name in ('is_first','is_last','is_terminal'):
          other[name][step]=next_obs[name]
        other['action_executed'][step]=True;other['observation_executed'][step+1]=True
        other['observations'].append(next_obs);other['snapshots'].append(snapshot(env,ts.last()))
        other['proofs'].append({})
        if ts.last():
          break
      count=len(other['proofs'])
      other['executed_steps']=count
      other['end_type']='true_terminal' if terminal else 'time_limit'
      other['raw'][count:]=0;other['applied'][count:]=0
      return other
    with mock.patch.object(type(env._task),'get_termination',return_value=0.0):
      terminal=bounded_fixture(record['snapshots'][0],True)
      terminal_receipt=replay_record(env,terminal)
      require(terminal_receipt['physics_steps']==1 and terminal_receipt['padding_steps']==99,
          'True terminal fixture crossed reset or padding')
      tests.append('official_task_patched_terminal_one_step_99_absorbing_no_reset')
      changed=copy.deepcopy(terminal);changed['reward_f64'][3]=1
      rejected(changed,'nonzero_absorbing_padding_rejected')
    start=copy.deepcopy(record['snapshots'][0])
    start['suite']['step_count']=int(env._step_limit)-HORIZON
    timeout=bounded_fixture(start)
    require(replay_record(env,timeout)['physics_steps']==100 and timeout['end_type']=='time_limit',
        'Final time limit fixture not accepted')
    tests.append('official_final_H100_time_limit_discount1_accepted')
    changed=copy.deepcopy(timeout)
    changed['snapshots'][0]['suite']['step_count']+=1
    # Consistent saved clocks force replay to detect the earlier real last.
    for saved in changed['snapshots'][1:]:
      saved['suite']['step_count']+=1
    rejected(changed,'early_time_limit_cannot_be_absorbing_padding')
    require('jax' not in sys.modules,'CPU selftest imported JAX')
    return dict(version=VERSION,status='selftest_passed',tests=tests,physics_steps=100,
        boundary_fixture_physics_steps=101,
        maximum_integration_error=result['maximum_integration_error'],actual_arrays_opened=False,
        model_executed=False,renderer_executed=False,jax_imported=False)
  finally:
    env.close()


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--dataset')
  parser.add_argument('--expected-feedback-sha')
  parser.add_argument('--output',required=True)
  parser.add_argument('--deadline-seconds',type=float,default=120)
  parser.add_argument('--self-test',action='store_true')
  args=parser.parse_args()
  output=Path(args.output)
  if output.exists():
    raise FileExistsError('New audit receipt path required')
  started=time.monotonic()
  timer_armed=False
  try:
    require(args.self_test or (args.dataset and args.expected_feedback_sha),
        'Dataset and expected full Git SHA required')
    require(np.isfinite(args.deadline_seconds) and 0 < args.deadline_seconds <= 600,
        'CPU audit deadline must be positive finite and at most 600 seconds')
    if not args.self_test and hasattr(signal,'setitimer'):
      def interrupted(signum,frame):
        raise TimeoutError('Independent CPU audit wall deadline exhausted; incomplete')
      signal.signal(signal.SIGALRM,interrupted)
      signal.setitimer(signal.ITIMER_REAL,args.deadline_seconds)
      timer_armed=True
    receipt=self_test() if args.self_test else audit_dataset(
        args.dataset,args.expected_feedback_sha,args.deadline_seconds)
    code=0
  except Exception as exc:
    receipt=dict(version=VERSION,status='failed',complete=False,error_type=type(exc).__name__,
        error=str(exc),expected_feedback_sha=args.expected_feedback_sha,
        elapsed_seconds=time.monotonic()-started,model_executed=False,renderer_executed=False,
        utility_or_signals_computed=False,training_authorized=False)
    code=1
  finally:
    if timer_armed:
      signal.setitimer(signal.ITIMER_REAL,0)
  receipt['audit_script_sha256']=sha256(Path(__file__))
  receipt['deadline_enforcement']='POSIX real-time signal plus monotonic checks' if timer_armed else (
      'selftest' if args.self_test else 'monotonic checks; Windows has no POSIX signal timer')
  receipt['auditor_current_git_sha_environment']=os.environ.get('DL_AUDITOR_GIT_SHA')
  output.parent.mkdir(parents=True,exist_ok=True)
  with output.open('x',encoding='utf-8') as stream:
    json.dump(receipt,stream,indent=2,sort_keys=True,allow_nan=False)
    stream.write('\n')
  print(json.dumps(dict(status=receipt['status'],output=str(output),
      elapsed_seconds=time.monotonic()-started,error=receipt.get('error')),allow_nan=False),flush=True)
  return code


if __name__=='__main__':
  raise SystemExit(main())
