"""DL-feedback-r1 production branch recording and conditional physics replay.

This module never initializes JAX or samples a policy. A host-only callback
supplies the actual policy samples and numeric key receipts. Physics replay
uses recorded applied controls without rendering or normalizing them again;
it does not establish policy or renderer equivalence.
"""
import copy
import hashlib
import re

import numpy as np

from scripts import dl_production_source as source

VERSION = 'DL-feedback-r1'
IMPLEMENTATION = 'DL-feedback-physics-r1'
OBS_KEYS = ('image', 'reward', 'is_first', 'is_last', 'is_terminal')
PROOF_KEYS = ('root_key', 'latent_key', 'actor_key', 'selected_logit',
              'h', 'latent', 'raw', 'rng_consumption')


def _array(value, dtype, shape, name):
  if (not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype) or
      value.shape != shape or not np.isfinite(value).all()):
    raise ValueError(f'{name} must be a finite host {dtype} array with shape {shape}')
  return value


def _numeric_tree(tree):
  if isinstance(tree, dict):
    if not all(isinstance(k, str) for k in tree):
      raise ValueError('Numeric tree dictionary keys must be strings')
    for value in tree.values():
      _numeric_tree(value)
  elif isinstance(tree, (list, tuple)):
    for value in tree:
      _numeric_tree(value)
  else:
    if not isinstance(tree, (np.ndarray, np.generic, bool, int, float)):
      raise ValueError('Policy receipts must already be host numeric leaves')
    value = np.asarray(tree)
    if (value.dtype.hasobject or value.dtype.kind in 'SU' or
        not np.isfinite(value).all()):
      raise ValueError('Nonfinite or nonnumeric policy receipt')


def _carry(carry, raw=None):
  _numeric_tree(carry)
  if (not isinstance(carry, (list, tuple)) or len(carry) != 4 or
      not isinstance(carry[0], dict) or carry[0] or
      not isinstance(carry[2], dict) or carry[2] or
      not isinstance(carry[1], dict) or set(carry[1]) != {'deter', 'stoch'} or
      not isinstance(carry[3], dict) or set(carry[3]) != {'action'}):
    raise ValueError('Unreviewed complete production carry structure')
  for key in ('deter', 'stoch'):
    if not isinstance(carry[1][key], (list, tuple)) or len(carry[1][key]) != 1:
      raise ValueError('Feedback requires one unbatched production policy carry')
  deter, stoch = carry[1]['deter'][0], carry[1]['stoch'][0]
  if (not isinstance(deter, np.ndarray) or deter.ndim != 1 or not deter.size or
      not isinstance(stoch, np.ndarray) or stoch.ndim != 2 or not stoch.size or
      deter.dtype != stoch.dtype or str(deter.dtype) not in ('bfloat16', 'float32')):
    raise ValueError('Unreviewed production latent carry shape or floating dtype')
  prev = carry[3]['action']
  if not isinstance(prev, (list, tuple)) or len(prev) != 1:
    raise ValueError('Previous raw action must retain the actual B1 leaf')
  _array(prev[0], np.float32, (12,), 'carry previous raw action')
  if raw is not None and prev[0].tobytes() != raw.tobytes():
    raise AssertionError('After carry must contain the raw policy sample')
  return source.tree_digest(carry)


def _observation(obs):
  if not isinstance(obs, dict) or set(obs) != set(OBS_KEYS):
    raise ValueError('Feedback requires the frozen vision-only production observation')
  _array(obs['image'], np.uint8, (64, 64, 3), 'image')
  _array(obs['reward'], np.float32, (), 'production reward')
  for key in OBS_KEYS[2:]:
    _array(obs[key], bool, (), key)
  if bool(obs['is_terminal']) and not bool(obs['is_last']):
    raise ValueError('Terminal observation must also be last')


def _proof(proof, key, raw):
  if not isinstance(proof, dict) or set(proof) != set(PROOF_KEYS):
    raise ValueError('Missing or unexpected numeric policy proof field')
  _numeric_tree(proof)
  for name in ('root_key', 'latent_key', 'actor_key'):
    _array(proof[name], np.uint32, (2,), name)
  if proof['root_key'].tobytes() != key.tobytes():
    raise AssertionError('Policy consumed a different private root key')
  _array(proof['rng_consumption'], np.int32, (), 'actual NJ RNG consumption count')
  if int(proof['rng_consumption']) != 2:
    raise AssertionError('Unexpected production NJ RNG consumption count')
  _array(proof['raw'], np.float32, (12,), 'proof raw')
  if proof['raw'].tobytes() != raw.tobytes():
    raise AssertionError('Proof raw differs from the actual policy return')
  h, logits, latent = (proof[name] for name in ('h', 'selected_logit', 'latent'))
  if (not isinstance(h, np.ndarray) or h.ndim != 1 or not h.size or
      not isinstance(logits, np.ndarray) or logits.ndim != 2 or not logits.size or
      not isinstance(latent, np.ndarray) or latent.shape != logits.shape or
      h.dtype != logits.dtype or latent.dtype != logits.dtype or
      str(h.dtype) not in ('bfloat16', 'float32')):
    raise ValueError('Policy h/logit/latent shapes or actual floating dtypes differ')


def _snapshot(snapshot):
  # source.state_error also checks finite F64, full shape and task/counter state.
  source.state_error(snapshot, snapshot)
  if (not isinstance(snapshot['from_dm_done'], (bool, np.bool_)) or
      not isinstance(snapshot['suite']['reset_next_step'], (bool, np.bool_))):
    raise ValueError('Snapshot wrapper/reset flags must be booleans')
  count = snapshot['suite']['step_count']
  if (isinstance(count, (bool, np.bool_)) or
      not isinstance(count, (int, np.integer)) or count < 0):
    raise ValueError('Snapshot clock must be a nonnegative integer')


def _timestep(ts):
  if ts is None or ts.first():
    raise ValueError('Unexpected reset/first timestep inside a feedback branch')
  reward = np.asarray(ts.reward)
  discount = np.asarray(ts.discount)
  if (reward.shape != () or reward.dtype != np.float64 or
      not np.isfinite(reward) or discount.shape != () or
      not np.isfinite(discount) or discount not in (0, 1)):
    raise ValueError('Official timestep reward must be finite F64 and discount 0/1')
  if discount == 0 and not ts.last():
    raise ValueError('Discount-zero timestep must be a true terminal')
  return reward.copy(), np.asarray(discount, np.float64)


def _flags(obs, ts):
  expected = (ts.first(), ts.last(), ts.last() and ts.discount == 0)
  if tuple(bool(obs[k]) for k in OBS_KEYS[2:]) != expected:
    raise AssertionError('Actual production observation and official flags disagree')


class ReceiptRecordingSuite(source.RecordingSuite):
  def __init__(self, suite):
    super().__init__(suite)
    self.last_timestep = None

  def reset(self):
    self.last_timestep = super().reset()
    return self.last_timestep

  def step(self, action):
    self.last_timestep = super().step(action)
    return self.last_timestep


class FeedbackEnvironment(source.ProductionEnvironment):
  """The unchanged actual wrapper chain plus an official timestep receipt."""
  def __init__(self, suite):
    super().__init__(suite)
    self.recorder = ReceiptRecordingSuite(suite)
    self.dmc._dmenv = self.recorder
    self.from_dm._env = self.recorder

  def restore(self, snapshot):
    _snapshot(snapshot)
    super().restore(snapshot)
    self.recorder.last_timestep = None


def prepare_initial_observation(adapter, snapshot, source_obs):
  """One fresh image per source state; preserve the existing F32 reward/flags."""
  _snapshot(snapshot)
  _observation(source_obs)
  if (snapshot['from_dm_done'] or snapshot['suite']['reset_next_step'] or
      bool(source_obs['is_last']) or bool(source_obs['is_terminal'])):
    raise ValueError('Feedback cannot start from a last/reset source state')
  adapter.restore(snapshot)
  before = source.state_error(adapter.snapshot(), snapshot)
  image = adapter.suite.physics.render(64, 64, camera_id=2)
  _array(image, np.uint8, (64, 64, 3), 'fresh actual RGB')
  after = source.state_error(adapter.snapshot(), snapshot)
  obs = copy.deepcopy(source_obs)
  obs['image'] = image.copy()
  return obs, dict(source_image_sha256=hashlib.sha256(source_obs['image'].tobytes()).hexdigest(),
      fresh_image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
      maximum_pixel_difference=int(np.max(np.abs(image.astype(np.int16) -
          source_obs['image'].astype(np.int16)))),
      maximum_state_error=max(before, after), fresh_render_count=1,
      renderer_byte_exact_claimed=False)


def run_branch(policy_callback, adapter, snapshot, hostcarry, keys, *,
               prior=False, initial_obs, horizon=100):
  """Run actual production controls, never policy/env calls after terminal.

  observations/snapshots have executed_steps+1 genuine entries. Only the H
  reward/action arrays have absorbing padding, identified by explicit masks.
  Model/key semantics and cached-RGB policy replay are separate oracle checks.
  """
  if type(horizon) is not int or not 1 <= horizon <= 100 or type(prior) is not bool:
    raise ValueError('Invalid bounded feedback horizon or branch')
  _array(keys, np.uint32, (horizon, 2), 'private named keys')
  _snapshot(snapshot)
  _observation(initial_obs)
  if (snapshot['from_dm_done'] or snapshot['suite']['reset_next_step'] or
      bool(initial_obs['is_last'])):
    raise ValueError('Feedback starts at an inactive source state')
  _carry(hostcarry)
  adapter.restore(snapshot)
  source.state_error(adapter.snapshot(), snapshot)
  carry, obs = copy.deepcopy(hostcarry), copy.deepcopy(initial_obs)
  record = dict(version=VERSION, implementation=IMPLEMENTATION,
      action_contract=source.CONTRACT, horizon=horizon, prior=prior,
      initial_carry=copy.deepcopy(carry), observations=[copy.deepcopy(obs)],
      snapshots=[adapter.snapshot()], keys=keys.copy(),
      raw=np.zeros((horizon, 12), np.float32),
      applied=np.zeros((horizon, 12), np.float64),
      reward_f64=np.zeros(horizon, np.float64), reward_f32=np.zeros(horizon, np.float32),
      discount=np.zeros(horizon, np.float64),
      is_first=np.zeros(horizon, bool), is_last=np.zeros(horizon, bool),
      is_terminal=np.zeros(horizon, bool), action_executed=np.zeros(horizon, bool),
      observation_executed=np.zeros(horizon + 1, bool),
      pre_carry_sha256=[''] * horizon, after_carry_sha256=[''] * horizon, proofs=[])
  record['observation_executed'][0] = True
  end_type = 'window'
  for step in range(horizon):
    before = _carry(carry)
    obs_before = source.tree_digest(obs)
    key = keys[step].copy()
    result = policy_callback.step(carry, obs, key, omit=(prior and step == 0))
    if _carry(carry) != before or source.tree_digest(obs) != obs_before:
      raise AssertionError('Policy callback mutated the host input carry/observation')
    if key.tobytes() != keys[step].tobytes():
      raise AssertionError('Policy callback mutated the private named key')
    if not isinstance(result, tuple) or len(result) != 3:
      raise ValueError('Policy callback must return after-hostcarry, raw, proof')
    after, raw, proof = result
    _array(raw, np.float32, (12,), 'actual raw sample')
    after_hash = _carry(after, raw)
    _proof(proof, keys[step], raw)
    carry = copy.deepcopy(after)
    next_obs, applied = adapter.step(raw.copy())
    _observation(next_obs)
    _array(applied, np.float64, (12,), 'actual applied control')
    ts = adapter.recorder.last_timestep
    reward, discount = _timestep(ts)
    _flags(next_obs, ts)
    if np.float32(reward).tobytes() != next_obs['reward'].tobytes():
      raise AssertionError('Actual F64 reward disagrees with production F32 encoding')
    state = adapter.snapshot()
    _snapshot(state)
    if bool(state['from_dm_done']) != bool(ts.last()):
      raise AssertionError('FromDM done receipt disagrees with official last')
    record['raw'][step], record['applied'][step] = raw, applied
    record['reward_f64'][step], record['reward_f32'][step] = reward, next_obs['reward']
    record['discount'][step] = discount
    for name in OBS_KEYS[2:]:
      record[name][step] = next_obs[name]
    record['action_executed'][step] = True
    record['observation_executed'][step + 1] = True
    record['pre_carry_sha256'][step] = before
    record['after_carry_sha256'][step] = after_hash
    record['proofs'].append(copy.deepcopy(proof))
    record['observations'].append(copy.deepcopy(next_obs))
    record['snapshots'].append(state)
    obs = next_obs
    if ts.last():
      if discount == 0:
        end_type = 'true_terminal'
      else:
        if (step != horizon - 1 or
            adapter.suite._step_count != adapter.suite._step_limit):
          raise ValueError('Early or unexpected time limit cannot be padded as terminal')
        end_type = 'time_limit'
      break
  record.update(executed_steps=len(record['proofs']), end_type=end_type,
      final_carry=copy.deepcopy(carry))
  validate_record(record)
  return record


def validate_record(record):
  """Fail closed on malformed/padded evidence before any replay call."""
  if (record['version'] != VERSION or record['implementation'] != IMPLEMENTATION or
      record['action_contract'] != source.CONTRACT):
    raise ValueError('Legacy or unknown feedback/action record')
  h, count = record['horizon'], record['executed_steps']
  if (type(h) is not int or not 1 <= h <= 100 or type(count) is not int or
      not 1 <= count <= h or type(record['prior']) is not bool):
    raise ValueError('Invalid feedback record lengths')
  for name, dtype, shape in (
      ('keys', np.uint32, (h, 2)), ('raw', np.float32, (h, 12)),
      ('applied', np.float64, (h, 12)), ('reward_f64', np.float64, (h,)),
      ('reward_f32', np.float32, (h,)), ('discount', np.float64, (h,)),
      ('is_first', bool, (h,)), ('is_last', bool, (h,)),
      ('is_terminal', bool, (h,)), ('action_executed', bool, (h,)),
      ('observation_executed', bool, (h + 1,))):
    _array(record[name], dtype, shape, name)
  if (not np.array_equal(record['action_executed'], np.arange(h) < count) or
      not np.array_equal(record['observation_executed'], np.arange(h + 1) <= count)):
    raise ValueError('Executed masks must mark precisely the genuine consecutive prefix')
  for name in ('raw', 'applied', 'reward_f64', 'reward_f32', 'discount',
               'is_first', 'is_last', 'is_terminal'):
    if np.any(record[name][count:] != 0):
      raise ValueError('Nonzero absorbing padding is not an actual transition')
  if (len(record['observations']) != count + 1 or
      len(record['snapshots']) != count + 1 or len(record['proofs']) != count):
    raise ValueError('Padded placeholder observations or states are forbidden')
  _carry(record['initial_carry'])
  for name in ('pre_carry_sha256', 'after_carry_sha256'):
    if (len(record[name]) != h or
        any(not isinstance(x, str) or not re.fullmatch('[0-9a-f]{64}', x)
            for x in record[name][:count]) or any(record[name][count:])):
      raise ValueError('Carry hash coverage differs from executed policy calls')
  if record['pre_carry_sha256'][0] != _carry(record['initial_carry']):
    raise AssertionError('Initial carry hash differs')
  if (record['pre_carry_sha256'][1:count] != record['after_carry_sha256'][:count - 1] or
      _carry(record['final_carry'], record['raw'][count - 1]) !=
      record['after_carry_sha256'][count - 1]):
    raise AssertionError('Policy carry hash chain is incomplete')
  for obs, state in zip(record['observations'], record['snapshots']):
    _observation(obs)
    _snapshot(state)
  if (record['snapshots'][0]['from_dm_done'] or
      record['snapshots'][0]['suite']['reset_next_step'] or
      bool(record['observations'][0]['is_last'])):
    raise ValueError('Inactive initial branch state')
  for step in range(count):
    _proof(record['proofs'][step], record['keys'][step], record['raw'][step])
    obs, state = record['observations'][step + 1], record['snapshots'][step + 1]
    if (record['reward_f32'][step].tobytes() != obs['reward'].tobytes() or
        np.float32(record['reward_f64'][step]).tobytes() != obs['reward'].tobytes() or
        tuple(bool(record[k][step]) for k in OBS_KEYS[2:]) !=
        tuple(bool(obs[k]) for k in OBS_KEYS[2:]) or
        bool(obs['is_first']) or
        bool(state['from_dm_done']) != bool(obs['is_last']) or
        bool(state['suite']['reset_next_step']) != bool(obs['is_last']) or
        record['discount'][step] not in (0, 1) or
        bool(obs['is_terminal']) != bool(record['discount'][step] == 0) or
        (bool(obs['is_last']) and step != count - 1)):
      raise AssertionError('Transition reward/flags/reset semantics disagree')
  last, terminal = bool(record['is_last'][count - 1]), bool(record['is_terminal'][count - 1])
  expected_end = 'true_terminal' if terminal else ('time_limit' if last else 'window')
  if record['end_type'] != expected_end or (count != h and not terminal):
    raise ValueError('Incomplete window is not absorbing terminal evidence')
  return count


def verify_physics_replay(adapter, record):
  """Recorded-control official physics replay; no renderer/policy equivalence."""
  from scripts.dl_action_contract import production_action_stages
  count = validate_record(record)
  adapter.restore(record['snapshots'][0])
  maximum = source.state_error(adapter.snapshot(), record['snapshots'][0])
  for step in range(count):
    stages = production_action_stages(record['raw'][step], adapter.low, adapter.high)
    applied = record['applied'][step]
    if (stages['applied'].dtype != applied.dtype or
        stages['applied'].tobytes() != applied.tobytes()):
      raise AssertionError('Recorded applied control differs from production mapping')
    ts = adapter.suite.step(applied.copy())
    reward, discount = _timestep(ts)
    adapter.from_dm._done = bool(ts.last())
    if (reward.tobytes() != record['reward_f64'][step].tobytes() or
        np.float32(reward).tobytes() != record['reward_f32'][step].tobytes() or
        discount.tobytes() != record['discount'][step].tobytes()):
      raise AssertionError('Official F64/F32 reward or discount byte replay differs')
    _flags(record['observations'][step + 1], ts)
    maximum = max(maximum, source.state_error(adapter.snapshot(), record['snapshots'][step + 1]))
    if ts.last() and discount == 1 and (
        step != record['horizon'] - 1 or adapter.suite._step_count != adapter.suite._step_limit):
      raise ValueError('Time-limit branch does not end at the declared clock/window')
  return dict(version=IMPLEMENTATION, physics_steps=count,
      absorbing_padding_steps=record['horizon'] - count,
      maximum_state_error=maximum, reward_float64_byte_exact=True,
      reward_float32_byte_exact=True, applied_mapping_byte_exact=True,
      integration_task_rng_counter_reset_wrapper_verified=True,
      policy_replayed=False, renderer_replayed=False)
