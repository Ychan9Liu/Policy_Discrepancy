"""Independent numerical/physics audit for the finite native actor assay.

This is a new H1000 verifier, not the old H20/H100 record validator. It does
not import a model, JAX, a renderer, the learning helper or producer statistics.
Only newly generated project records may be decoded after byte authentication.
The CLI owns a1200s CPU parent covering imports, hashes, replay and full exit.
"""
import time
ENTRY_STARTED = time.monotonic()
import os
import signal
import sys
OWNED = set()
SECONDS = 1200


def _expire(signum, frame):
  for pid in tuple(OWNED):
    try: os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError: pass
  os._exit(124)


if __name__ == '__main__' and os.name == 'posix':
  signal.signal(signal.SIGALRM, _expire)
  signal.setitimer(signal.ITIMER_REAL, max(.001, SECONDS-(time.monotonic()-ENTRY_STARTED)))

import argparse
import copy
import hashlib
import importlib.metadata
import io
import json
import math
import pickle
import re
from pathlib import Path
import subprocess

import numpy as np

VERSION = 'DL-rep-native-assay-r1'
IMPLEMENTATION = 'DL-rep-native-independent-audit-r1'
GROUPS = ('0', 'B', 'A')
SEEDS = tuple(range(1024, 1056))
HORIZON = 1000
BOOTSTRAP_SEED = 20261026
BOOTSTRAPS = 2000
ABS_TOLERANCE = 1e-10
OBS_KEYS = {'image', 'reward', 'is_first', 'is_last', 'is_terminal'}
ROOT = Path(__file__).resolve().parents[1]
PHASES = ('authenticate', 'A', 'freeze', 'eval0', 'evalB', 'evalA', 'statistics')
GPU_UUID = 'GPU-9c070d5a-ac68-4c26-9522-a790ad874b23'
OLD_PRIMARY_GIT = 'c030e26f688ed227aa43866bd68dcf4352a073e3'
OLD_AUDITOR_GIT = 'd987650c749b8917e2ea80a719d586232088f501'
OLD_AUDIT_SHA = 'ba4c7d9c5d47baeb83f48eb2cde8eca5637186c1abe6ae42a961d13fe4fc8a22'
OLD_AUDIT_SUPERVISOR_SHA = '3496eadb986937e2d04c061863e5a065c08fca0d302b4a231b4c4fc4ba4e2732'
OLD_AUDIT_TERMINAL_SHA = '6a97200e5511212ae28481615dbe733e89e7c03af0d75ee8392365b043bc2b83'
OLD_B_COMPLETE_SHA = 'b2596ff268873725e54a4008d760961e55a67c00c5318477b5ccc98ff280b8fb'
INITIAL_STATE = '216fe9be592190709fe4f22ad23ab89808baa9eb49b618046855861a11276c78'
B_STATE = '9e670873990510f97174f07b04a407ed290ad762057a33e8884818d5e2fd013d'
INITIAL_CHECKPOINT = '2ab693c8ded4d8f2186bd489aa8b77498448ba2f2c331090159da4e5f08056b9'
B_CHECKPOINT = 'c878652b0fca1313ed9d1536eec838e2972e3dea611e4b309f94b32a63272670'
CONFIG_SHA = '11428512fdc9b9e04a5384e419c5abcef93e5429db0a3327d05ccb0d4bd20445'
CONTEXT_SHA = 'c8a6a964455e2a579c4ed76f1de19cc38931955279b36b45445310414451e1ab'


def require(condition, message):
  if not condition:
    raise ValueError(message)


def _array(value, dtype, shape, name):
  require(isinstance(value, np.ndarray) and value.dtype == np.dtype(dtype) and
      value.shape == shape and not value.dtype.hasobject and
      np.isfinite(value).all(), name + ': finite typed host array required')
  return value


def _equal(a, b, name):
  require(isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and
      a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(),
      name + ': bytes/dtype/shape differ')


def _hash(value, name):
  require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value),
      name + ': SHA256 required')


def _snapshot(value):
  require(type(value) is dict and set(value) == {'suite', 'from_dm_done'},
      'Incomplete full snapshot')
  state = value['suite']
  require(type(state) is dict and set(state) ==
      {'integration', 'task', 'step_count', 'reset_next_step'},
      'Incomplete suite integration/task/clock/reset snapshot')
  require(isinstance(state['integration'], np.ndarray) and
      state['integration'].ndim == 1 and state['integration'].size > 0,
      'Nonempty integration vector required')
  _array(state['integration'], np.float64, state['integration'].shape, 'integration')
  require(type(state['step_count']) is int and state['step_count'] >= 0 and
      isinstance(state['reset_next_step'], (bool, np.bool_)) and
      isinstance(value['from_dm_done'], (bool, np.bool_)) and
      type(state['task']) is dict, 'Invalid snapshot task/clock/flags')
  return state


def compare_snapshot(actual, expected):
  a, b = _snapshot(actual), _snapshot(expected)
  _equal(a['integration'], b['integration'], 'Complete integration state')
  require(a['step_count'] == b['step_count'] and
      a['reset_next_step'] == b['reset_next_step'] and
      actual['from_dm_done'] == expected['from_dm_done'],
      'Complete task clock/reset/wrapper done differs')
  require(pickle.dumps(a['task'], protocol=5) == pickle.dumps(b['task'], protocol=5),
      'Complete task/RNG state differs')


def snapshot_hash(value):
  """Byte digest of a trusted complete snapshot, with fixed protocol5."""
  _snapshot(value)
  return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def numeric_digest(tree):
  """Independent implementation of the producer's typed numeric encoding."""
  digest = hashlib.sha256()
  def visit(value):
    if isinstance(value, dict):
      digest.update(b'dict')
      for key in sorted(value):
        require(type(key) is str, 'String numeric tree keys required')
        key = key.encode()
        digest.update(str(len(key)).encode()+b':'+key)
        visit(value[key.decode()])
    elif isinstance(value, (tuple, list)):
      digest.update(type(value).__name__.encode()+str(len(value)).encode()+b':')
      for child in value:
        visit(child)
    else:
      value = np.asarray(value)
      require(not value.dtype.hasobject and value.dtype.kind not in 'SU' and
          np.isfinite(value).all(), 'Finite numeric digest leaves required')
      digest.update(str(value.dtype).encode()+b':'+str(value.shape).encode()+b':')
      digest.update(np.ascontiguousarray(value).tobytes())
  visit(tree)
  return digest.hexdigest()


def load_record(root, filename, pin, check=lambda: None):
  """Decode one NEW record from the same hash-verified payload, never source."""
  _hash(pin, 'record byte hash')
  require(type(filename) is str and Path(filename).name == filename and
      not Path(filename).is_absolute() and filename.endswith('.pkl'),
      'Unsafe new record filename')
  directory = Path(root).resolve(strict=True)
  path = (directory / filename).resolve(strict=True)
  require(path.parent == directory and path.is_file(), 'Record escapes new output directory')
  check(); payload = path.read_bytes(); check()
  require(hashlib.sha256(payload).hexdigest() == pin,
      'New record byte hash differs before trusted pickle decode')
  return pickle.load(io.BytesIO(payload))


def named_roots(seed):
  require(type(seed) is int and seed in SEEDS, 'Unfrozen environment seed')
  return np.stack([np.random.default_rng([0, 0x4556414C, seed, step]).integers(
      0, np.iinfo(np.uint32).max, (2,), dtype=np.uint32)
      for step in range(HORIZON)])


def _mapping(raw, low, high):
  """Independent ordered Clip/Unify/Normalize, preserving F32 arithmetic."""
  _array(raw, np.float32, (12,), 'raw action')
  _array(low, np.float64, (12,), 'suite low bound')
  _array(high, np.float64, (12,), 'suite high bound')
  require(np.all(low < high), 'Invalid suite action bounds')
  clipped = np.clip(raw, -1, 1)
  unified = np.asarray(clipped, np.float32)
  applied = (unified + 1) / 2 * (high - low) + low
  return dict(raw=raw, clipped=clipped, unified=unified, applied=applied)


def _observations(value):
  require(type(value) is dict and set(value) == OBS_KEYS,
      'Frozen vision-only observations required')
  for name, dtype, shape in (
      ('image', np.uint8, (1001, 64, 64, 3)),
      ('reward', np.float32, (1001,)),
      ('is_first', bool, (1001,)), ('is_last', bool, (1001,)),
      ('is_terminal', bool, (1001,))):
    _array(value[name], dtype, shape, 'observations/' + name)
  first, last, terminal = (value[k] for k in ('is_first', 'is_last', 'is_terminal'))
  require(np.array_equal(first, np.arange(1001) == 0) and
      np.array_equal(last, np.arange(1001) == 1000) and
      not terminal[:1000].any() and value['reward'][0].tobytes() == np.float32(0).tobytes(),
      'Early termination/reset/padding or missing final last')


def validate_record(record):
  """Validate genuine complete H1000 evidence without model/physics calls."""
  require(type(record) is dict and record.get('version') == VERSION and
      record.get('group') in GROUPS and type(record.get('seed')) is int and
      record['seed'] in SEEDS, 'Unknown record version/group/seed')
  trajectory = record.get('trajectory')
  require(type(trajectory) is dict, 'Missing trajectory')
  _observations(trajectory.get('observations'))
  for name, dtype, shape in (
      ('raw', np.float32, (1000, 12)), ('applied', np.float64, (1000, 12)),
      ('reward_f64', np.float64, (1000,)), ('discount_f64', np.float64, (1000,)),
      ('root_keys', np.uint32, (1000, 2)), ('latent_keys', np.uint32, (1000, 2)),
      ('actor_keys', np.uint32, (1000, 2))):
    _array(trajectory.get(name), dtype, shape, 'trajectory/' + name)
  _equal(trajectory['root_keys'], named_roots(record['seed']), 'Actual private roots')
  _equal(trajectory['reward_f64'].astype(np.float32),
      trajectory['observations']['reward'][1:], 'F64 to actual F32 reward')
  discount = trajectory['discount_f64']
  require(np.isin(discount, (0, 1)).all() and (discount[:999] == 1).all() and
      np.array_equal(trajectory['observations']['is_terminal'][1:], discount == 0),
      'Discount/terminal mismatch or early terminal')
  hashes = trajectory.get('carry_hashes')
  require(isinstance(hashes, (list, tuple)) and len(hashes) == 1001,
      'Complete carry hash chain coverage required')
  for value in hashes:
    _hash(value, 'carryhash')
  require(trajectory.get('version') == VERSION and trajectory.get('seed') == record['seed'] and
      trajectory.get('counts') == dict(transitions=1000, observations=1001,
          policy_calls=1000, resets=1, oracle_calls=3) and
      trajectory.get('readonly_full_state_and_counters_unchanged') is True and
      trajectory.get('public_policy_copy_used_for_scientific_calls') is False,
      'Incomplete policy trajectory/read-only/oracle counts')
  _hash(trajectory.get('model_state_sha256'), 'Current full model state digest')
  numerical = {name: trajectory[name] for name in ('observations', 'raw', 'applied',
      'reward_f64', 'discount_f64', 'root_keys', 'latent_keys', 'actor_keys')}
  require(trajectory.get('arrays_sha256') == numeric_digest(numerical),
      'Actual numeric trajectory digest differs')
  oracles = trajectory.get('oracle_receipts')
  require(type(oracles) is list and len(oracles) == 3 and
      [row.get('step') for row in oracles] == [0, 500, 999] and
      all(type(row.get('receipt')) is dict and row['receipt'] for row in oracles),
      'Production oracle receipt coverage differs')
  oracle_copy = None
  for row in oracles:
    receipt = row['receipt']
    require(all(receipt.get(name) is True for name in ('public_raw_byte_exact',
        'public_after_carry_byte_exact', 'currentparams_policycopy_typed_byte_exact',
        'eval_counters_restored', 'full_state_and_counters_unchanged', 'actual_native_sample')) and
        receipt.get('public_model_calls') == 1, 'Production oracle semantics not authenticated')
    _hash(receipt.get('policy_copy_sha256'), 'Oracle synchronized policy copy')
    _equal(receipt.get('root_key'), trajectory['root_keys'][row['step']], 'Production oracle root')
    if oracle_copy is None: oracle_copy = receipt['policy_copy_sha256']
    else: require(oracle_copy == receipt['policy_copy_sha256'], 'Within episode policy copy changed')
  require(type(trajectory.get('return_f64')) is float and
      abs(trajectory['return_f64'] - math.fsum(map(float, trajectory['reward_f64']))) <=
      ABS_TOLERANCE, 'Trajectory return not bound to all1000 actual rewards')
  snapshots = record.get('snapshots')
  require(isinstance(snapshots, (list, tuple)) and len(snapshots) == 1001,
      'Exactly 1001 actual snapshots required')
  for step, value in enumerate(snapshots):
    state = _snapshot(value)
    require(state['step_count'] == step and
        bool(state['reset_next_step']) == (step == 1000) and
        bool(value['from_dm_done']) == (step == 1000),
        'Snapshot reset/wrapper clock differs from genuine 1000-step episode')
  stages = record.get('stages')
  require(type(stages) is list and len(stages) == 1000 and
      all(type(row) is dict and set(row) == {'raw', 'clipped', 'unified', 'applied'} for row in stages),
      'Actual ordered production action stages required')
  for step, row in enumerate(stages):
    for name, dtype in (('raw', np.float32), ('clipped', np.float32),
        ('unified', np.float32), ('applied', np.float64)):
      _array(row[name], dtype, (12,), 'stages/' + name)
    _equal(row['raw'], trajectory['raw'][step], 'Stage raw sample')
    _equal(row['applied'], trajectory['applied'][step], 'Stage stored applied control')
  reset = record.get('reset_receipt')
  require(type(reset) is dict and reset.get('seed') == record['seed'] and
      reset.get('rendering_full_state_byte_exact') is True,
      'Fresh reset seed/render purity receipt required')
  require(reset.get('initial_snapshot_sha256') == snapshot_hash(snapshots[0]) and
      reset.get('initial_rgb_sha256') == hashlib.sha256(
          trajectory['observations']['image'][0].tobytes()).hexdigest(),
      'Reset snapshot/RGB receipt not bound to actual data')
  require(record.get('camera') == 2 and record.get('size') == [64, 64] and
      record.get('proprio') is False and record.get('action_repeat') == 1 and
      record.get('actual_physics_transitions') == 1000 and record.get('actual_RGB_frames') == 1001 and
      record.get('production_chain') == ['ClipAction', 'CheckSpaces', 'UnifyDtypes',
          'NormalizeAction', 'DMC', 'ActionRepeat', 'FromDM'], 'Frozen environment contract differs')
  return trajectory


class OfficialPhysics:
  """Official CPU suite; independent restore, no production wrappers/render."""
  def __init__(self, seed):
    from dm_control import suite
    self.suite = suite.load('quadruped', 'walk', task_kwargs={'random': seed})
    self.done = False
    spec = self.suite.action_spec()
    self.low, self.high = spec.minimum.copy(), spec.maximum.copy()
    require(int(self.suite._step_limit) == HORIZON, 'Official episode length differs')
    self.suite.reset()

  def snapshot(self):
    import mujoco
    bits = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(self.suite.physics.model.ptr, bits), np.float64)
    mujoco.mj_getState(self.suite.physics.model.ptr, self.suite.physics.data.ptr, state, bits)
    return dict(suite=dict(integration=state,
        task=copy.deepcopy(self.suite._task.__dict__),
        step_count=self.suite._step_count, reset_next_step=self.suite._reset_next_step),
        from_dm_done=bool(self.done))

  def restore(self, saved):
    import mujoco
    state = _snapshot(saved)
    self.suite._task.__dict__.clear()
    self.suite._task.__dict__.update(copy.deepcopy(state['task']))
    self.suite._step_count = state['step_count']
    self.suite._reset_next_step = bool(state['reset_next_step'])
    self.done = bool(saved['from_dm_done'])
    mujoco.mj_setState(self.suite.physics.model.ptr, self.suite.physics.data.ptr,
        state['integration'], mujoco.mjtState.mjSTATE_INTEGRATION)
    self.suite.physics.forward()

  def step(self, applied):
    timestep = self.suite.step(applied)
    self.done = bool(timestep.last())
    return timestep

  def close(self):
    self.suite.close()


def replay_record(record, adapter, check=lambda: None):
  """Full stored-applied replay with byte-exact reward/flags/all-state."""
  trajectory = validate_record(record)
  check()
  # The newly seeded official reset is checked before restoring the receipt.
  compare_snapshot(adapter.snapshot(), record['snapshots'][0])
  adapter.restore(record['snapshots'][0])
  compare_snapshot(adapter.snapshot(), record['snapshots'][0])
  observations = trajectory['observations']
  rewards = []
  for step in range(HORIZON):
    check()
    mapped = _mapping(trajectory['raw'][step], adapter.low, adapter.high)
    for name in mapped:
      _equal(mapped[name], record['stages'][step][name], 'Production stage/' + name)
    timestep = adapter.step(trajectory['applied'][step].copy())
    reward, discount = np.asarray(timestep.reward), np.asarray(timestep.discount)
    _array(reward, np.float64, (), 'Official reward')
    require(discount.shape == () and np.isfinite(discount) and discount in (0, 1),
        'Official discount must be finite scalar 0/1')
    discount = np.asarray(discount, np.float64)
    _equal(reward, np.asarray(trajectory['reward_f64'][step]), 'Actual F64 reward')
    _equal(discount, np.asarray(trajectory['discount_f64'][step]), 'Actual F64 discount')
    expected_flags = (False, step == 999, bool(discount == 0))
    require((bool(timestep.first()), bool(timestep.last()), bool(discount == 0)) ==
        expected_flags and tuple(bool(observations[name][step+1]) for name in
        ('is_first', 'is_last', 'is_terminal')) == expected_flags,
        'Official/observed flags differ or premature last')
    compare_snapshot(adapter.snapshot(), record['snapshots'][step+1])
    rewards.append(float(reward))
  check()
  return dict(physics_steps=HORIZON, return_fsum=math.fsum(rewards),
      initial_fresh_reset_byte_exact=True, all_state_byte_exact=True,
      all_reward_discount_flags_byte_exact=True, applied_stage_mapping_byte_exact=True,
      model_replayed=False, renderer_replayed=False, optimizer_replayed=False)


def _summary(values, draws):
  values = [float(x) for x in values]
  require(len(values) == 32 and all(math.isfinite(x) for x in values),
      'Full finite 32-episode values required')
  mean = math.fsum(values) / 32
  samples = [math.fsum(values[int(i)] for i in row) / 32 for row in draws]
  ci = np.quantile(np.asarray(samples, np.float64), [.025, .975], method='linear').tolist()
  return dict(values=values, mean=mean,
      std=math.sqrt(math.fsum((x-mean)**2 for x in values) / 32),
      minimum=min(values), maximum=max(values), ci95=ci)


def independent_statistics(returns):
  """All32 paired fsum bootstrap; never select a prefix or arm/seed."""
  require(type(returns) is dict and set(returns) == set(GROUPS),
      'Exactly initial0, B and A returns required')
  draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 32, size=(BOOTSTRAPS, 32))
  models = {group: _summary(returns[group], draws) for group in GROUPS}
  comparisons = {}
  for name, left, right in (('A_minus_B', 'A', 'B'),
      ('A_minus_0', 'A', '0'), ('B_minus_0', 'B', '0')):
    summary = _summary([float(x)-float(y) for x, y in zip(returns[left], returns[right])], draws)
    low, high = summary['ci95']
    summary['ci_zero_side'] = [int(np.sign(low)), int(np.sign(high))]
    summary['classification'] = ('positive_finite_net_effect' if low > 0 else
        'positive_effect_not_supported' if high <= 0 else 'inconclusive')
    comparisons[name] = summary
  return dict(version=VERSION, seeds=list(SEEDS), groups=list(GROUPS), horizon=HORIZON,
      bootstrap=dict(seed=BOOTSTRAP_SEED, samples=BOOTSTRAPS, unit='whole_episode',
          method='percentile_linear', paired=True), models=models,
      comparisons=comparisons, primary='A_minus_B',
      method_effectiveness_established=False, independent_second_signal_validated=False)


def compare_statistics(expected, actual):
  """Tolerance only for arithmetic; CI zero side and classification exact."""
  if isinstance(expected, dict):
    require(type(actual) is dict and set(actual) == set(expected), 'Statistics schema differs')
    if 'ci95' in expected:
      require(isinstance(actual['ci95'], (list, tuple)) and len(actual['ci95']) == 2 and
          all(type(x) in (int, float) and math.isfinite(x) for x in actual['ci95']) and
          [int(np.sign(x)) for x in actual['ci95']] ==
          [int(np.sign(x)) for x in expected['ci95']], 'CI crossed exact zero-side boundary')
    for name, value in expected.items():
      compare_statistics(value, actual[name])
  elif isinstance(expected, (list, tuple)):
    require(isinstance(actual, (list, tuple)) and len(actual) == len(expected),
        'Statistics array coverage differs')
    for left, right in zip(expected, actual):
      compare_statistics(left, right)
  elif type(expected) is float:
    require(type(actual) in (int, float) and math.isfinite(actual) and
        abs(expected-actual) <= ABS_TOLERANCE, 'Statistics arithmetic differs')
  else:
    require(type(actual) is type(expected) and actual == expected,
        'Statistics exact metadata/classification differs')


def audit_records(records, result, *, adapter_factory=OfficialPhysics, check=lambda: None):
  """Stream exactly96 authenticated records; caller owns pins and timeout.

  records is an iterable in group0/B/A then ascendingseed order. Only one full
  RGB record is retained at a time. result is the producer statistics subtree.
  Model/current-params/private-key/oracle claims remain receipt authentication,
  never an independent model or renderer replay.
  """
  returns = {group: [] for group in GROUPS}
  reset_signatures = {}
  count = 0
  iterator = iter(records)
  for expected in ((g, s) for g in GROUPS for s in SEEDS):
    try:
      record = next(iterator)
    except StopIteration:
      raise ValueError('Missing model/seed record') from None
    group, seed = expected
    require((record.get('group'), record.get('seed')) == expected,
        'Missing/duplicated/reordered model/seed record')
    adapter = adapter_factory(seed)
    try:
      replay = replay_record(record, adapter, check)
    finally:
      adapter.close()
    signature = (record['reset_receipt']['initial_snapshot_sha256'],
        record['reset_receipt']['initial_rgb_sha256'])
    if group == '0':
      reset_signatures[seed] = signature
    else:
      require(signature == reset_signatures[seed], 'Cross-arm fresh reset/RGB differs')
    returns[group].append(replay['return_fsum'])
    count += 1
  try:
    next(iterator)
  except StopIteration:
    pass
  else:
    raise ValueError('Unexpected extra model/seed record')
  require(count == 96 and all(len(returns[g]) == 32 for g in GROUPS),
      'Full96 episodes/96000 transitions required')
  statistics = independent_statistics(returns)
  compare_statistics(statistics, result)
  return dict(version=IMPLEMENTATION, passed=True, episodes=count,
      physics_steps=count*HORIZON, statistics=statistics,
      all_stored_applied_physics_replayed=True, independent_fsum_statistics=True,
      original_rgb_only_authenticated=True, policy_replayed=False,
      renderer_replayed=False, optimizer_replayed=False,
      full_method_effectiveness_established=False)


def _write_json(path, value):
  with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
    stream.write('\n')


def _json_payload(payload):
  def pairs(items):
    result = {}
    for key, value in items:
      require(key not in result, 'Duplicate JSON key: ' + key)
      result[key] = value
    return result
  def reject(value):
    raise ValueError('Nonfinite JSON value: ' + value)
  return json.loads(payload, object_pairs_hook=pairs, parse_constant=reject)


def _safe_file(root, name):
  require(type(name) is str and Path(name).name == name and not Path(name).is_absolute(),
      'Unsafe generated filename')
  root = Path(root).resolve(strict=True)
  path = (root / name).resolve(strict=True)
  require(path.parent == root and path.is_file(), 'Generated artifact escapes output directory')
  return path


def _sha(path, check=lambda: None):
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024*1024), b''):
      check(); digest.update(chunk)
  return digest.hexdigest()


def _read_json(path, pin=None, check=lambda: None):
  check(); payload = Path(path).read_bytes(); check()
  if pin is not None:
    _hash(pin, 'JSON byte pin')
    require(hashlib.sha256(payload).hexdigest() == pin, 'JSON pin differs before decode')
  return _json_payload(payload)


def _terminal(receipt, *, seconds, elapsed_key):
  require(type(receipt) is dict and type(receipt.get('exit_code')) is int and
      receipt['exit_code'] == 0 and receipt.get('timed_out') is False and
      receipt.get('worker_terminal') is True, 'Successful owned worker terminal required')
  elapsed = receipt.get(elapsed_key)
  require(type(elapsed) in (int, float) and math.isfinite(elapsed) and
      0 < elapsed <= seconds, 'Whole-entry time bound differs')


def _metadata(command, check=lambda: None):
  check()
  own_group = os.name == 'posix' and 'DL_NATIVE_AUDIT_PARENT' not in os.environ
  child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
      stderr=subprocess.PIPE, start_new_session=own_group)
  if own_group: OWNED.add(child.pid)
  try:
    out, err = child.communicate(timeout=30)
    require(child.returncode == 0, 'Metadata command failed: ' + err.decode(errors='replace')[:200])
    check(); return out
  finally:
    if own_group:
      try: os.killpg(child.pid, signal.SIGKILL)
      except ProcessLookupError: pass
      OWNED.discard(child.pid)
    elif child.poll() is None:
      child.kill()
    child.wait(timeout=5)


def _code_binding(current_git, primary_git, pins, check):
  require(type(current_git) is str and re.fullmatch('[0-9a-f]{40}', current_git) and
      type(primary_git) is str and re.fullmatch('[0-9a-f]{40}', primary_git), 'Full Git SHA required')
  require(_metadata(['git','rev-parse','HEAD'], check).decode().strip() == current_git and
      not _metadata(['git','status','--porcelain','--untracked-files=all'], check),
      'Independent audit requires current clean checkout')
  exceptions = {'analysis/dl_rep_native_audit.py', 'tests/test_dl_rep_native_audit.py'}
  require(type(pins) is dict and {'scripts/dl_rep_native.py', 'scripts/dl_native_actor.py',
      'scripts/dl_reset_all_rep.py', 'docs/DL_REP_NATIVE_ASSAY.md'} <= set(pins),
      'Missing primary scientific source pins')
  for name, pin in pins.items():
    _hash(pin, 'Primary source SHA')
    require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts,
        'Unsafe source file path')
    blob = _metadata(['git','show', primary_git+':'+name], check)
    require(hashlib.sha256(blob).hexdigest() == pin, 'Primary source not bound to actual Git blob')
    if name not in exceptions:
      require((ROOT / name).read_bytes() == blob, 'Producer source changed since primary execution')
  own = {}
  for name in sorted(exceptions):
    blob = _metadata(['git','show', current_git+':'+name], check)
    require((ROOT / name).read_bytes() == blob, 'Current independent audit source differs from Git')
    own[name] = hashlib.sha256(blob).hexdigest()
  return own


def _device(receipt, worker_pid, *, graphics):
  require(receipt.get('physical_gpu') == 4 and receipt.get('gpu_uuid') == GPU_UUID and
      receipt.get('compute_uuid') == receipt.get('egl_uuid') == GPU_UUID and
      receipt.get('logical_server') == 'sv3' and receipt.get('actual_hostname') == 'lyg0326' and
      receipt.get('cuda_visible_devices') == GPU_UUID and
      str(receipt.get('mujoco_egl_device_id')) == '4' and
      receipt.get('occupancy') == 'idle' and receipt.get('exclusive') is True,
      'Actual exclusive sv3 physicalGPU4 CUDA/EGL receipt differs')
  names = ['runtime_compute_start',
      'runtime_compute_and_graphics_start', 'runtime_compute_and_graphics_end'] if graphics else [
      'runtime_compute_start', 'runtime_compute_end']
  for name in names:
    proof = receipt.get(name, {})
    require(proof.get('pid') == worker_pid and proof.get('physical_gpu') == 4 and
        proof.get('compute_uuid') == [GPU_UUID] and
        proof.get('graphics_verified') is (graphics and name != 'runtime_compute_start'),
        'Actual runtime compute/graphics identity differs')
    state = proof.get('live_state', {})
    require(state.get('gpu_uuid') == GPU_UUID and state.get('physical_gpu') == 4 and
        type(state.get('processes')) is list and type(state.get('pmon_jobs')) is list and
        all(row.get('pid') == worker_pid for row in state['processes']+state['pmon_jobs']),
        'Runtime resource ownership differs')
    if graphics and name != 'runtime_compute_start':
      require(any(row.get('pid') == worker_pid and 'G' in row.get('type', '')
          for row in state['pmon_jobs']), 'Actual EGL graphics process proof missing')


def _native_receipt(receipt, state):
  require(receipt.get('version') == VERSION and receipt.get('current_full_state_sha256') == state and
      all(receipt.get(name) is True for name in ('currentparams_identity_unchanged',
          'scientific_policy_reads_currentparams', 'readonly_full_state_and_counters_unchanged',
          'attempted_model_writes_rejected', 'attempted_model_creation_rejected')) and
      receipt.get('public_policy_copy_used_for_scientific_calls') is False and
      receipt.get('source_physics_renderer_resource_authenticated_by_this_helper') is False and
      receipt.get('private_initial_calls') == 32 and receipt.get('private_policy_calls') == 32000 and
      receipt.get('public_oracle_calls') == 96 and
      type(receipt.get('accessed_parameter_keys')) is list and receipt['accessed_parameter_keys'],
      'Current-params native/public oracle readonly receipt differs')
  _hash(receipt.get('synchronized_policy_copy_sha256'), 'Synchronized policy copy SHA')


def _pressure_receipts(learning):
  reset_rows, whole_rows = [], []
  for row in learning['receipts']:
    reset, whole = row['reset_pressure'], row['whole_window_pressure']
    for mapping, names in ((reset, ('active_count','reset_count')),
        (whole, ('active_count','position_count','nonreset_active_count'))):
      require(all(type(mapping.get(name)) is int and mapping[name] >= 0 for name in names),
          'Pressure count must be exact nonnegative integer')
    require(reset['reset_count'] == 16 and reset['active_count'] <= 16 and
        whole['position_count'] == 1024 and whole['active_count'] <= 1024 and
        whole['nonreset_active_count'] <= 1008 and
        whole['active_count'] == reset['active_count']+whole['nonreset_active_count'],
        'All-window/reset pressure count partition differs')
    for count,excess in ((reset['active_count'],reset['excess_sum']),
        (whole['active_count'],whole['excess_sum']),
        (whole['nonreset_active_count'],whole['nonreset_excess_sum'])):
      require(type(excess) in (int,float) and math.isfinite(excess) and excess >= 0 and
          (count > 0) == (excess > 0), 'Finite strict pressure active/excess invariant differs')
    reset_rows.append(reset); whole_rows.append(whole)
  for label, rows, counts, sums in (
      ('reset_pressure', reset_rows, ('active_count','reset_count'), ('excess_sum',)),
      ('whole_window_pressure', whole_rows, ('active_count','position_count','nonreset_active_count'),
          ('excess_sum','nonreset_excess_sum'))):
    total = learning[label]
    require(total.get('entire_128_update_metrics_observed') is True,
        'All128 pressure evidence required')
    require(all(total[name] == sum(row[name] for row in rows) for name in counts) and
        all(abs(total[name]-math.fsum(row[name] for row in rows)) <= ABS_TOLERANCE for name in sums),
        'Pressure aggregate not bound to all128 current metrics')
  require(learning.get('total_loss_positions') == 131072 and
      learning.get('total_reset_loss_positions') == 2048, 'Whole learning window counts differ')
  return dict(post_free_nats_current_A_pressure_observed=True,
      actual_nonzero_rep_gradient_or_information_deletion_independently_proved=False,
      active_count=learning['whole_window_pressure']['active_count'],
      excess_sum=learning['whole_window_pressure']['excess_sum'],
      original_B_whole_window_pressure_measured=False)


def authenticate_primary(primary, external_cli_terminal, current_git, check=lambda: None):
  """Independently authenticate typed receipts; decode no models/old sources."""
  root = Path(primary).resolve(strict=True)
  require(not any((root/name).exists() for name in ('late-supervisor.json',
      'late-provisional-supervisor.json', 'supervisor-receipt-pending.json')),
      'Late/provisional primary output forbidden')
  supervisor = _read_json(root/'supervisor_complete.json', check=check)
  require(supervisor.get('version') == VERSION and supervisor.get('exit_code') == 0 and
      supervisor.get('timed_out') is False and supervisor.get('complete_verified') is True and
      supervisor.get('actual_gpu_devices_used') == 1 and
      supervisor.get('independent_audit_complete') is False and
      supervisor.get('method_effectiveness_established') is False and
      0 < supervisor.get('elapsed_parent_entry_seconds', math.inf) <= 7200,
      'Complete successful7200s primary parent required')
  primary_git = supervisor['actual_git_sha']
  external = _read_json(external_cli_terminal, check=check)
  require(external.get('exit_code') == 0 and external.get('worker_terminal') is True and
      external.get('actual_git_sha') == primary_git and
      external.get('worker_pid') == supervisor.get('parent_pid') and
      0 < external.get('elapsed_cli_seconds', math.inf) <= 7200,
      'Independent complete external primary CLI exit0 required')
  budget = _read_json(root/'budget.json', supervisor['budget_sha256'], check)
  require(budget.get('version') == VERSION and budget.get('seconds') == 7200 and
      budget.get('actual_git_sha') == primary_git and
      budget.get('parent_pid') == supervisor.get('parent_pid') and
      budget.get('planned_physical_gpu') == 4 and budget.get('max_gpu_devices') == 1 and
      budget.get('max_gpu_hours') == 2, 'Fixed primary resource/whole-entry budget differs')
  audit_sources = _code_binding(current_git, primary_git, budget['source_files_sha256'], check)
  phases = supervisor.get('phases')
  require(type(phases) is list and [row.get('phase') for row in phases] == list(PHASES),
      'Seven complete primary phases required')
  for row in phases:
    _terminal(row, seconds=7200, elapsed_key='elapsed_seconds')
    require(row.get('is_gpu') is (row['phase'] in ('A','eval0','evalB','evalA')),
        'Fixed GPU/CPU phase map differs')
    require(_read_json(root/(row['phase']+'-exit.json'), check=check) == row,
        'Worker phase exit not bound to complete parent')
  require(math.fsum(row['elapsed_seconds'] for row in phases) <=
      supervisor['elapsed_parent_entry_seconds'], 'Worker time exceeds whole-entry parent')
  result = _read_json(root/'result.json', supervisor['result_sha256'], check)
  require(result.get('version') == VERSION and result.get('actual_git_sha') == primary_git and
      result.get('actual_physics_transitions') == 96000 and result.get('actual_RGB_frames') == 96096 and
      result.get('actual_training_updates') == 128 and result.get('original_B_retrained') is False and
      result.get('original_inputs_unchanged') is True and
      all(result.get(name) is False for name in ('independent_signal_identified',
          'selective_gate_validated', 'ordinary_replay_training_validated',
          'method_effectiveness_established', 'minimal_practical_effect_declared', 'full_goal_complete')),
      'Complete finite result counts/research boundary differs')
  freeze = _read_json(root/'freeze.json', result['freeze_sha256'], check)
  seal = _read_json(root/'freeze-seal-parent.json', result['freeze_seal_parent_sha256'], check)
  require(freeze.get('version') == VERSION and freeze.get('actual_git_sha') == primary_git and
      freeze.get('groups') == list(GROUPS) and freeze.get('evaluation_seeds') == list(SEEDS) and
      freeze.get('transitions_per_episode') == 1000 and
      freeze.get('expected_physics_transitions') == 96000 and freeze.get('primary') == 'A_minus_B' and
      freeze.get('bootstrap_seed') == 20261026 and freeze.get('bootstrap_draws') == 2000 and
      freeze.get('root_definition') == 'default_rng([0,0x4556414C,environment_seed,step]).uint32_key2' and
      freeze.get('new_evaluation_started') is False and
      freeze.get('additional_F_measurements_performed') is False and
      freeze.get('old_B_authenticated_reuse') is True and freeze.get('old_B_retrained') is False and
      freeze.get('native_actor_action') == 'official_sample' and
      seal.get('version') == VERSION and seal.get('actual_git_sha') == primary_git and
      seal.get('freeze_sha256') == result['freeze_sha256'] and seal.get('new_evaluation_started') is False and
      budget['monotonic_start'] <= freeze['monotonic_frozen'] <= seal['monotonic_sealed'] <=
      budget['monotonic_start']+7200, 'Pre-evaluation parent seal/frozen scope differs')
  require(set(freeze['files_sha256']) == {'budget.json','plan.json',
      'old-authenticated.json','A-complete.json','A-checkpoint.pkl'},
      'Frozen input files differ')
  for name, pin in freeze['files_sha256'].items():
    require(_sha(_safe_file(root,name), check) == pin, 'Frozen input byte pin differs')
  plan = _read_json(root/'plan.json', freeze['files_sha256']['plan.json'], check)
  require(plan.get('version') == VERSION and plan.get('groups') == list(GROUPS) and
      plan.get('evaluation_seeds') == list(SEEDS) and plan.get('primary') == 'A_minus_B' and
      plan.get('transitions_per_episode') == 1000 and
      plan.get('actual_training_updates_per_new_arm') == 128 and plan.get('old_B_retrained') is False and
      all(plan.get(name) is False for name in ('additional_F_measurements_performed',
          'independent_signal_identified','selective_gate_validated')),
      'Frozen plan/fixed finite primary differs')
  auth = _read_json(root/'old-authenticated.json', result['old_authentication_sha256'], check)
  require(freeze['old_authentication_sha256'] == result['old_authentication_sha256'] and
      auth.get('version') == VERSION and auth.get('actual_git_sha') == primary_git and
      auth.get('primary_git_sha') == OLD_PRIMARY_GIT and auth.get('auditor_git_sha') == OLD_AUDITOR_GIT and
      auth.get('old_audit_sha256') == OLD_AUDIT_SHA and
      all(auth.get(name) is True for name in ('original_19_producer_files_unchanged',
          'old_full_exit_and_independent_audit_authenticated')) and
      auth.get('old_B_retrained') is False and auth.get('original_blind_arrays_decoded') is False,
      'Original B authenticated reuse scope differs')
  inputs = auth['immutable_inputs_sha256']
  require(type(inputs) is dict and inputs, 'No immutable original input pins')
  require({INITIAL_CHECKPOINT,B_CHECKPOINT,CONFIG_SHA,CONTEXT_SHA,
      OLD_B_COMPLETE_SHA,OLD_AUDIT_SHA,OLD_AUDIT_SUPERVISOR_SHA,OLD_AUDIT_TERMINAL_SHA} <= set(inputs.values()),
      'Missing known original model/config/context/B/full audit endpoint byte pins')
  old_sources = auth['source_binding']['all_32_artifact_hashes']
  require(type(old_sources) is list and len(old_sources) == 32 and
      all(inputs.get(row['path']) == row['sha256'] for row in old_sources) and
      auth['old_files_sha256'].get('B-checkpoint.pkl') == B_CHECKPOINT and
      auth['old_files_sha256'].get('contexts.pkl') == CONTEXT_SHA,
      'Original complete32 source hash-only and B/context binding differs')
  for path, pin in inputs.items():
    require(Path(path).is_absolute() and _sha(path, check) == pin,
        'Original input bytes changed; audit only hashes models/source')
  old = Path(auth['primary']).resolve(strict=True)
  old_b = _read_json(old/'B-complete.json', OLD_B_COMPLETE_SHA, check)
  require(old_b['learning'] == auth['original_B_learning'], 'Original B JSON not bound to known endpoint')
  audit_paths = [path for path,pin in inputs.items() if Path(path).name == 'audit.json' and pin == OLD_AUDIT_SHA]
  require(len(audit_paths) == 1, 'Original audited B endpoint missing')
  old_audit = _read_json(audit_paths[0], OLD_AUDIT_SHA, check)
  require(old_audit.get('status') == 'passed' and old_audit.get('primary_git_sha') == OLD_PRIMARY_GIT and
      old_audit.get('auditor_git_sha') == OLD_AUDITOR_GIT and old_audit.get('primary') == str(old) and
      old_audit.get('complete_primary_exit_and_parent_preblind_seal_verified') is True and
      old_audit.get('actual_gpu_devices_used') == 0 and
      old_audit.get('actual_independent_physics_steps') == 3360 and
      OLD_AUDIT_SUPERVISOR_SHA in inputs.values() and OLD_AUDIT_TERMINAL_SHA in inputs.values(),
      'Original independent B result not authenticated')
  states = freeze['full_state_digests']
  require(set(states) == set(GROUPS) and states['0'] == INITIAL_STATE and states['B'] == B_STATE and
      freeze['complete_checkpoint_sha256']['0'] == INITIAL_CHECKPOINT and
      freeze['complete_checkpoint_sha256']['B'] == B_CHECKPOINT and
      freeze['complete_checkpoint_sha256']['A'] == freeze['files_sha256']['A-checkpoint.pkl'],
      'Known initial/B checkpoint/fullstate identity differs')
  a = _read_json(root/'A-complete.json', freeze['files_sha256']['A-complete.json'], check)
  learning, original = a['learning'], old_b['learning']
  require(a.get('phase') == 'A' and a.get('version') == VERSION and a.get('actual_git_sha') == primary_git and
      a.get('old_authentication_sha256') == result['old_authentication_sha256'] and
      a.get('original_B_learning_reference_authenticated') is True and
      a.get('original_blind_arrays_decoded') is False and a.get('additional_F_measurements_performed') is False and
      a.get('cal_image_frames_decoded') == 8008 and
      a['source_binding']['all_32_artifact_hashes'] == old_sources and
      a.get('checkpoint_sha256') == freeze['complete_checkpoint_sha256']['A'] and
      learning.get('version') == 'DL-reset-allrep-learning-r1' and learning.get('group') == 'A' and
      learning.get('protocol_adopted') is True and learning.get('working_draft_not_adopted') is False and
      learning.get('layout_version') == 'DL-reset-learning-r1' and learning.get('updates') == 128 and
      learning.get('final_state_digest') == states['A'] and
      all(learning.get(name) is True for name in ('entire_window_rep_gradient_stopped',
          'rep_forward_values_retained', 'other_loss_branches_retained',
          'posterior_features_and_recurrent_carry_not_stopped',
          'historical_optimizer_momentum_retained', 'per_update_init_train',
          'original_B_receipt_values_matched')),
      'All-window current-rep A intervention/128 complete state differs')
  for name in ('binding','source_checkpoint_digest','initial_state_digest','initial_params_digest',
      'calibration_arrays_digest','restored_counters','saved_source_counters','optimizer_counter'):
    require(learning[name] == original[name], 'A/B matched initial/input/layout differs: '+name)
  require(learning['optimizer_counter'] == dict(key='opt/step/value', dtype='int32', shape=[],
      initial=74493, final=74621, updates=128) and len(learning['receipts']) == len(original['receipts']) == 128,
      'Official optimizer or complete128 update receipts differ')
  for index, (left,right) in enumerate(zip(learning['receipts'], original['receipts'])):
    require(left['update'] == right['update'] == index+1 and
        all(left[name] == right[name] for name in ('seed_counter','train_seed','counters')),
        'A/B actual original train root/counter progression differs')
  pressure = _pressure_receipts(learning)
  _device(a['device'], phases[1]['worker_pid'], graphics=False)
  evaluations = {}
  manifest = []
  for group in GROUPS:
    value = _read_json(root/('eval'+group+'-complete.json'), result['evaluation_complete_sha256'][group], check)
    require(value.get('version') == VERSION and value.get('actual_git_sha') == primary_git and
        value.get('group') == group and value.get('phase') == 'eval'+group and
        value.get('freeze_sha256') == result['freeze_sha256'] and
        value.get('actual_physics_transitions') == 32000 and value.get('actual_RGB_frames') == 32032 and
        value.get('native_model_calls') == 32000 and value.get('production_oracle_calls') == 96 and
        value.get('additional_F_measurements_performed') is False and
        value.get('complete_normal_posterior_feedback') is True and
        value.get('full_state_digest_before_and_after') == states[group],
        'Evaluation typed model/full-feedback/count receipt differs')
    _native_receipt(value['native_interface_receipt'], states[group])
    _device(value['device'], phases[PHASES.index('eval'+group)]['worker_pid'], graphics=True)
    rows = value['records']
    require(type(rows) is list and len(rows) == 32 and [row['seed'] for row in rows] == list(SEEDS) and
        all(row['group'] == group and row['file'] == f'{group}-episode-{row["seed"]}.pkl' for row in rows),
        'Complete fixed model/seed record order differs')
    resets = value.get('reset_receipts')
    require(type(resets) is list and len(resets) == 32 and
        [entry.get('seed') for entry in resets] == list(SEEDS) and
        all(entry.get('rendering_full_state_byte_exact') is True for entry in resets),
        'All32 typed fresh-reset/render receipts required')
    for entry in resets:
      _hash(entry.get('initial_snapshot_sha256'), 'Fresh reset snapshot SHA')
      _hash(entry.get('initial_rgb_sha256'), 'Fresh reset RGB SHA')
    for row in rows:
      require(_sha(_safe_file(root,row['file']), check) == row['sha256'], 'New record byte hash differs')
    evaluations[group] = value; manifest.extend(rows)
  require(result['record_manifest'] == manifest and
      evaluations['0']['reset_receipts'] == evaluations['B']['reset_receipts'] == evaluations['A']['reset_receipts'],
      'Full96 manifest or cross-arm fresh reset/image pairing differs')
  return dict(root=root, primary_git=primary_git, current_git=current_git, result=result,
      evaluations=evaluations, freeze=freeze, immutable=inputs,
      audit_sources=audit_sources, budget=budget, pressure=pressure,
      endpoint_sha256={name:_sha(root/name,check) for name in ('result.json','freeze.json',
          'supervisor_complete.json','budget.json','plan.json','freeze-seal-parent.json',
          'old-authenticated.json','A-complete.json','eval0-complete.json','evalB-complete.json',
          'evalA-complete.json')},
      external_cli_terminal_sha256=_sha(external_cli_terminal,check))


def audit_authenticated(authority, check=lambda: None):
  def records():
    for group in GROUPS:
      evaluation = authority['evaluations'][group]
      for index, row in enumerate(evaluation['records']):
        record = load_record(authority['root'], row['file'], row['sha256'], check)
        require(record.get('group') == group and record.get('seed') == row['seed'] and
            record['trajectory']['model_state_sha256'] == authority['freeze']['full_state_digests'][group] and
            record['reset_receipt'] == evaluation['reset_receipts'][index] and
            abs(record['trajectory']['return_f64']-row['return_f64']) <= ABS_TOLERANCE and
            all(entry['receipt']['policy_copy_sha256'] ==
                evaluation['native_interface_receipt']['synchronized_policy_copy_sha256']
                for entry in record['trajectory']['oracle_receipts']),
            'Actual trajectory not bound to full model/current copy/seed/return receipt')
        yield record
  result = audit_records(records(), authority['result']['statistics'], check=check)
  for path,pin in authority['immutable'].items():
    require(_sha(path,check) == pin, 'Original input changed during independent replay')
  for name,pin in authority['endpoint_sha256'].items():
    require(_sha(authority['root']/name,check) == pin, 'Primary endpoint changed during audit')
  for row in authority['result']['record_manifest']:
    require(_sha(_safe_file(authority['root'],row['file']),check) == row['sha256'],
        'New record changed during audit')
  for name,pin in authority['budget']['source_files_sha256'].items():
    if name not in authority['audit_sources']:
      require(_sha(ROOT/name,check) == pin, 'Producer source changed during audit')
  for name,pin in authority['audit_sources'].items():
    require(_sha(ROOT/name,check) == pin, 'Independent audit source changed during replay')
  require(_metadata(['git','rev-parse','HEAD'],check).decode().strip() == authority['current_git'] and
      not _metadata(['git','status','--porcelain','--untracked-files=all'],check),
      'Audit terminal Git binding differs')
  return dict(**result, status='passed', primary=str(authority['root']),
      primary_git_sha=authority['primary_git'], auditor_git_sha=authority['current_git'],
      primary_endpoint_sha256=authority['endpoint_sha256'],
      primary_external_cli_terminal_sha256=authority['external_cli_terminal_sha256'],
      audit_source_sha256=authority['audit_sources'], actual_gpu_devices_used=0,
      A_pressure=authority['pressure'],
      complete_primary_exit_and_parent_pre_evaluation_seal_verified=True,
      current_model_and_optimizer_only_byte_authenticated=True,
      native_policy_and_renderer_only_typed_receipts_authenticated=True,
      original_source_or_checkpoint_decoded=False, old_B_retrained=False,
      conditional_on='same official CPU simulator; authenticated producer model/key/oracle/render receipts')


def _cleanup(child):
  if os.name == 'posix':
    try: os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError: pass
  elif child.poll() is None:
    child.kill()
  child.wait(timeout=5); OWNED.discard(child.pid)


def _run_owned(command, log, deadline, env):
  require(time.monotonic() < deadline, 'Late worker creation')
  child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
      stderr=subprocess.STDOUT, start_new_session=os.name == 'posix')
  OWNED.add(child.pid); start = time.monotonic(); timed_out = False
  try:
    code = child.wait(timeout=max(.001,deadline-time.monotonic()))
  except subprocess.TimeoutExpired:
    code = 124; timed_out = True
  finally:
    _cleanup(child)
  return dict(worker_pid=child.pid, worker_terminal=child.poll() is not None,
      exit_code=code, timed_out=timed_out, elapsed_seconds=time.monotonic()-start,
      owned_group_cleanup_requested=os.name == 'posix')


def _worker(args):
  output = Path(args.output).resolve(strict=True)
  budget = _read_json(output/'budget.json')
  require(budget.get('version') == VERSION and budget.get('seconds') == SECONDS and
      budget.get('actual_git_sha') == args.expected_git and
      os.environ.get('DL_NATIVE_AUDIT_PARENT') == str(budget['parent_pid']) and
      os.getppid() == budget['parent_pid'], 'Live owned CPU audit parent required')
  os.kill(os.getppid(),0)
  deadline = budget['monotonic_start']+SECONDS
  def check(): require(time.monotonic() < deadline, 'Whole-entry CPU1200s budget exhausted')
  require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('MUJOCO_GL') == 'disable' and
      os.environ.get('JAX_PLATFORMS') == 'cpu', 'CPU/no-render worker environment differs')
  authority = authenticate_primary(args.primary,args.external_cli_terminal,args.expected_git,check)
  result = audit_authenticated(authority,check)
  result['environment'] = {name:importlib.metadata.version(name) for name in ('numpy','mujoco','dm-control')}
  require(all(authority['result']['environment'][name] == version for name,version in
      result['environment'].items()), 'Same official simulator/NumPy version required')
  require('jax' not in sys.modules and 'dreamerv3.agent' not in sys.modules and
      'embodied.jax.agent' not in sys.modules, 'Independent audit imported forbidden model/JAX module')
  check(); _write_json(output/'audit.json',result); check()


def _parent(args):
  require(os.name == 'posix', 'Owned whole-entry CPU audit requires POSIX process groups')
  output = Path(args.output).resolve()
  require(not output.exists(), 'New audit attempt directory required')
  output.mkdir(parents=True)
  deadline = ENTRY_STARTED+SECONDS
  _write_json(output/'budget.json',dict(version=VERSION,actual_git_sha=args.expected_git,
      parent_pid=os.getpid(),seconds=SECONDS,monotonic_start=ENTRY_STARTED,
      actual_gpu_devices_used=0,scope='imports,authentication,replay,statistics,write,exit'))
  command = [sys.executable,'-m','analysis.dl_rep_native_audit','--worker']
  for key in ('primary','external_cli_terminal','expected_git','output'):
    command += ['--'+key.replace('_','-'),str(getattr(args,key))]
  env = dict(os.environ,DL_NATIVE_AUDIT_PARENT=str(os.getpid()),CUDA_VISIBLE_DEVICES='',
      MUJOCO_GL='disable',JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
  with (output/'worker.log').open('x',encoding='utf-8') as log:
    terminal = _run_owned(command,log,deadline,env)
  _write_json(output/'worker_exit_code.txt',terminal['exit_code'])
  complete = terminal['exit_code'] == 0 and not terminal['timed_out'] and time.monotonic() < deadline
  pin = None
  if complete:
    result = _read_json(output/'audit.json')
    require(result.get('status') == 'passed' and result.get('passed') is True and
        result.get('version') == IMPLEMENTATION and result.get('auditor_git_sha') == args.expected_git and
        result.get('episodes') == 96 and result.get('physics_steps') == 96000 and
        result.get('actual_gpu_devices_used') == 0,
        'Independent complete96000-step result required')
    pin = _sha(output/'audit.json')
  receipt = dict(version=VERSION,implementation=IMPLEMENTATION,actual_git_sha=args.expected_git,
      parent_pid=os.getpid(),**{**terminal, 'exit_code': terminal['exit_code'] if
          complete or terminal['exit_code'] else 1},complete_verified=complete,
      elapsed_parent_entry_seconds=time.monotonic()-ENTRY_STARTED,
      budget_sha256=_sha(output/'budget.json'),worker_exit_sha256=_sha(output/'worker_exit_code.txt'),
      audit_sha256=pin,actual_gpu_devices_used=0 if complete else None,
      method_effectiveness_established=False)
  pending = output/'supervisor-receipt-pending.json'
  _write_json(pending,receipt)
  if time.monotonic() >= deadline:
    _write_json(output/'late-supervisor.json',dict(exit_code=124,incomplete=True)); return 124
  os.replace(pending,output/'supervisor_complete.json')
  if time.monotonic() >= deadline:
    os.replace(output/'supervisor_complete.json',output/'late-provisional-supervisor.json')
    _write_json(output/'late-supervisor.json',dict(exit_code=124,incomplete=True)); return 124
  return receipt['exit_code'] if complete or receipt['exit_code'] else 1


def verify_completion(root, external_cli_terminal=None):
  """Verify full published audit; external exit is needed for full acceptance."""
  root = Path(root).resolve(strict=True)
  require(not any((root/name).exists() for name in ('late-supervisor.json',
      'late-provisional-supervisor.json','supervisor-receipt-pending.json')),
      'Late/provisional CPU audit cannot complete')
  receipt = _read_json(root/'supervisor_complete.json')
  _terminal(receipt,seconds=SECONDS,elapsed_key='elapsed_parent_entry_seconds')
  require(receipt.get('version') == VERSION and receipt.get('implementation') == IMPLEMENTATION and
      receipt.get('complete_verified') is True and receipt.get('actual_gpu_devices_used') == 0 and
      receipt.get('method_effectiveness_established') is False,
      'Complete finite CPU audit receipt required')
  budget = _read_json(root/'budget.json',receipt['budget_sha256'])
  require(budget.get('version') == VERSION and budget.get('seconds') == SECONDS and
      budget.get('actual_git_sha') == receipt['actual_git_sha'] and
      budget.get('parent_pid') == receipt['parent_pid'] and budget.get('actual_gpu_devices_used') == 0 and
      _sha(root/'worker_exit_code.txt') == receipt.get('worker_exit_sha256') and
      _read_json(root/'worker_exit_code.txt') == 0, 'CPU audit budget/worker exit chain differs')
  result = _read_json(root/'audit.json',receipt['audit_sha256'])
  require(result.get('version') == IMPLEMENTATION and result.get('status') == 'passed' and
      result.get('passed') is True and result.get('auditor_git_sha') == receipt['actual_git_sha'] and
      result.get('episodes') == 96 and result.get('physics_steps') == 96000 and
      result.get('actual_gpu_devices_used') == 0, 'Complete96-episode independent result required')
  if external_cli_terminal is not None:
    external = _read_json(external_cli_terminal)
    require(external.get('actual_git_sha') == receipt['actual_git_sha'] and
        external.get('exit_code') == 0 and external.get('worker_terminal') is True and
        external.get('worker_pid') == receipt['parent_pid'] and
        0 < external.get('elapsed_cli_seconds',math.inf) <= SECONDS,
        'Complete external CPU audit CLI exit0 required')
  return dict(audit_sha256=receipt['audit_sha256'],physics_steps=96000,
      complete_published_parent_verified=True,
      complete_external_exit_verified=external_cli_terminal is not None,
      method_effectiveness_established=False)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('primary','external-cli-terminal','expected-git','output'):
    parser.add_argument('--'+name,required=True)
  parser.add_argument('--worker',action='store_true')
  args = parser.parse_args(argv)
  if args.worker:
    _worker(args); return 0
  return _parent(args)


if __name__ == '__main__':
  sys.exit(main())
