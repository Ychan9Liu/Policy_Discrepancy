"""DL-r4: actual three-arm action/memory branches and conditional replays.

This host-only module constructs no model, checkpoint or GPU context. The
caller supplies a frozen R4Policy and one live FeedbackEnvironment. RGB from
the real source observation is shared only at the current decision; every
executed future observation comes from that branch's normal wrapper render.
Stored-control physics replay and cached-own-RGB policy replay are separate
claims. Neither establishes deterministic rendering or complete closed-loop
byte replay.
"""
import copy
import hashlib
import itertools
import pickle
import re

import numpy as np

from scripts import dl_feedback_physics as feedback
from scripts import dl_production_source as source
from scripts.dl_reset_features import tree_digest as policy_tree_digest

VERSION = 'DL-r4-action-memory-r1'
IMPLEMENTATION = 'DL-r4-physics-r1'
SCHEMA = 1
HORIZON = 100
POSITIONS = tuple(range(64, 897, 64))
ARMS = ('qq', 'pq', 'pp')
ARM_ORDERS = tuple(itertools.permutations(ARMS))
OBS_KEYS = feedback.OBS_KEYS
PROOF_KEYS = ('root_key', 'latent_key', 'actor_key', 'h', 'selected_logit',
              'latent', 'action_logit', 'action_latent', 'raw',
              'rng_consumption', 'memory_source', 'action_source')
BUNDLE_FIELDS = ('q_after', 'p_after', 'q_raw', 'p_raw', 'h', 'qlogits',
                 'plogits', 'qsample', 'psample', 'root_key', 'latent_key',
                 'actor_key', 'original_nj_seed_calls', 'before_carry_sha256',
                 'observation_sha256')


def require(condition, message):
  if not condition:
    raise ValueError(message)


def _same(actual, expected, message):
  require(isinstance(actual, (np.ndarray, np.generic))
      and isinstance(expected, (np.ndarray, np.generic))
      and actual.dtype == expected.dtype and actual.shape == expected.shape
      and actual.tobytes() == expected.tobytes(), message)


def _digest(value):
  return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def _array(value, dtype, shape, name):
  return feedback._array(value, dtype, shape, name)


def _carry(value, raw=None):
  return feedback._carry(value, raw)


def _snapshot_exact(actual, expected):
  """The old tolerant primitive is an input guard, not the r4 acceptance rule."""
  feedback._snapshot(actual)
  feedback._snapshot(expected)
  source.state_error(actual, expected)
  _same(actual['suite']['integration'], expected['suite']['integration'],
      'R4 complete integration snapshot differs byte-exactly')
  # state_error checks complete task RNG, clock, reset and wrapper flags.
  return 0.


def visual_model_identity(adapter):
  """Static CPU model serialization identity; deliberately no GL identity claim."""
  model = adapter.suite.physics.model
  binary = model.to_bytes()
  require(isinstance(binary, bytes) and len(binary) > 0,
      'An actual nonempty official model serialization is required')
  require(int(model.ncam) > 2, 'Actual model has no fixed camera 2')
  return dict(mjb_sha256=hashlib.sha256(binary).hexdigest(),
      mjb_bytes=len(binary), camera=2, camera_name=model.id2name(2, 'camera'),
      adapter_identity=id(adapter), model_identity=id(model),
      complete_effective_render_inputs_authenticated=False,
      renderer_byte_exact_established=False)


def _visual(adapter, expected, *, same_live=True):
  require(isinstance(expected, dict), 'Explicit source visual-model identity required')
  actual = visual_model_identity(adapter)
  fields = ('mjb_sha256', 'mjb_bytes', 'camera', 'camera_name')
  if same_live:
    fields += ('adapter_identity', 'model_identity')
  require(all(actual.get(k) == expected.get(k) for k in fields),
      'Static visual model or live adapter identity changed')
  require(expected.get('complete_effective_render_inputs_authenticated') is False
      and expected.get('renderer_byte_exact_established') is False,
      'Static CPU model identity cannot authenticate GPU/render state')
  return actual


def _bundle(bundle, key=None):
  require(isinstance(bundle, dict) and all(k in bundle for k in BUNDLE_FIELDS),
      'Missing R4 current q/p bundle fields')
  for name in ('root_key', 'latent_key', 'actor_key'):
    _array(bundle[name], np.uint32, (2,), 'bundle ' + name)
  for name in ('before_carry_sha256', 'observation_sha256'):
    require(isinstance(bundle[name], str) and re.fullmatch('[0-9a-f]{64}', bundle[name]),
        'Actual current bundle input hash is missing or malformed')
  if key is not None:
    _same(bundle['root_key'], key, 'Current bundle uses a different root')
  count = np.asarray(bundle['original_nj_seed_calls'])
  require(count.shape == () and count.dtype.kind in 'iu' and int(count) == 2,
      'Current bundle must record the original two actual NJ seed calls')
  for prefix, memory in (('q', 'q_after'), ('p', 'p_after')):
    raw = bundle[prefix + '_raw']
    _array(raw, np.float32, (12,), 'bundle raw')
    _carry(bundle[memory], raw)
    h, z = bundle[memory][1]['deter'][0], bundle[memory][1]['stoch'][0]
    _same(h, bundle['h'], 'q/p bundle must share the same actual h')
    _same(z, bundle[prefix + 'sample'], 'Bundle sample differs from after memory')
    logit = bundle[prefix + 'logits']
    require(isinstance(logit, np.ndarray) and logit.shape == z.shape
        and logit.dtype == z.dtype and np.isfinite(logit).all(),
        'Bundle current distribution dtype/shape/finite differs')
  return source.tree_digest({k: bundle[k] for k in BUNDLE_FIELDS})


def _proof(proof, key, raw, after, *, arm='qq'):
  """New schema, checked directly; never projected into old PROOF_KEYS."""
  require(isinstance(proof, dict) and set(proof) == set(PROOF_KEYS),
      'Missing or unexpected R4 numeric proof field')
  feedback._numeric_tree(proof)
  for name in ('root_key', 'latent_key', 'actor_key'):
    _array(proof[name], np.uint32, (2,), name)
  _same(proof['root_key'], key, 'Policy used a different private root')
  _array(proof['rng_consumption'], np.int32, (), 'actual NJ consumption')
  require(int(proof['rng_consumption']) == 2, 'Actual original NJ consumption differs')
  expected = dict(qq=(0, 0), pq=(0, 1), pp=(1, 1))[arm]
  for name, value in zip(('memory_source', 'action_source'), expected):
    _array(proof[name], np.int8, (), name)
    require(int(proof[name]) == value, 'R4 memory/action source differs')
  _array(proof['raw'], np.float32, (12,), 'proof raw')
  _same(proof['raw'], raw, 'Actual raw action differs from proof')
  _carry(after, raw)
  _same(proof['h'], after[1]['deter'][0], 'Proof h differs from actual outgoing memory')
  _same(proof['latent'], after[1]['stoch'][0],
      'Proof latent differs from actual outgoing memory')
  h, z = proof['h'], proof['latent']
  for name in ('selected_logit', 'action_logit', 'action_latent'):
    value = proof[name]
    require(isinstance(value, np.ndarray) and value.dtype == z.dtype
        and value.shape == z.shape and np.isfinite(value).all(),
        'R4 current logit/latent proof differs: ' + name)
  require(h.dtype == z.dtype, 'R4 actual latent and h dtype differ')
  if arm != 'pq':
    _same(proof['action_logit'], proof['selected_logit'],
        'Same-source logits differ')
    _same(proof['action_latent'], z, 'Same-source action/memory latents differ')


def _first_matches(bundle, arm, after, raw, proof):
  memory, action = ('p' if arm[1] == 'p' else 'q'), arm[0]
  expected = copy.deepcopy(bundle[memory + '_after'])
  expected[3]['action'][0] = bundle[action + '_raw'].copy()
  require(_carry(after, raw) == _carry(expected, bundle[action + '_raw']),
      'First arm does not combine the declared memory and actual raw action')
  _same(raw, bundle[action + '_raw'], 'First raw action differs from q/p bundle')
  for name in ('root_key', 'latent_key', 'actor_key', 'h'):
    _same(proof[name], bundle[name], 'First receipt differs from current bundle: ' + name)
  for name, field in (('selected_logit', memory + 'logits'),
      ('latent', memory + 'sample'), ('action_logit', action + 'logits'),
      ('action_latent', action + 'sample')):
    _same(proof[name], bundle[field], 'First memory/action proof differs: ' + name)


def prepare_initial_observation(adapter, snapshot, source_obs, *, visual_identity):
  """Restore physics and share the original actually observed current RGB only."""
  feedback._snapshot(snapshot)
  feedback._observation(source_obs)
  require(not snapshot['from_dm_done'] and not snapshot['suite']['reset_next_step']
      and not bool(source_obs['is_first']) and not bool(source_obs['is_last']),
      'R4 requires an ordinary active nonreset source observation')
  require(int(snapshot['suite']['step_count']) in POSITIONS,
      'R4 ordinary source clock is outside the frozen positions')
  _visual(adapter, visual_identity)
  adapter.restore(snapshot)
  _snapshot_exact(adapter.snapshot(), snapshot)
  _visual(adapter, visual_identity)
  return copy.deepcopy(source_obs), dict(shared_actual_source_rgb=True,
      source_image_sha256=hashlib.sha256(source_obs['image'].tobytes()).hexdigest(),
      fresh_initial_render_count=0, physical_restore_byte_exact=True,
      renderer_byte_exact_established=False,
      complete_effective_render_inputs_authenticated=False)


def branch_order(pair):
  require(type(pair) is int and 0 <= pair < 16, 'R4 pair must be in 0..15')
  return ARM_ORDERS[pair % len(ARM_ORDERS)]


def run_branch(policy, adapter, snapshot, hostcarry, keys, *, arm, bundle,
               initial_obs, visual_identity, check=lambda: None):
  """H100 normal closed loop after one explicitly crossed current interface."""
  require(arm in ARMS, 'Unknown R4 arm')
  _array(keys, np.uint32, (HORIZON, 2), 'R4 named private keys')
  _carry(hostcarry)
  bundle_hash = _bundle(bundle, keys[0])
  obs, initial_receipt = prepare_initial_observation(adapter, snapshot, initial_obs,
      visual_identity=visual_identity)
  carry = copy.deepcopy(hostcarry)
  require(bundle['before_carry_sha256'] == policy_tree_digest(carry)
      and bundle['observation_sha256'] == policy_tree_digest(obs),
      'Current bundle was generated from different history or observation')
  record = dict(version=VERSION, implementation=IMPLEMENTATION, schema=SCHEMA,
      action_contract=source.CONTRACT, horizon=HORIZON, arm=arm,
      initial_carry=copy.deepcopy(carry), current_bundle=copy.deepcopy(bundle),
      current_bundle_sha256=bundle_hash, observations=[copy.deepcopy(obs)],
      snapshots=[adapter.snapshot()], keys=keys.copy(),
      visual_identity=copy.deepcopy(visual_identity), initial_receipt=initial_receipt,
      raw=np.zeros((HORIZON, 12), np.float32),
      applied=np.zeros((HORIZON, 12), np.float64),
      reward_f64=np.zeros(HORIZON, np.float64),
      reward_f32=np.zeros(HORIZON, np.float32),
      discount=np.zeros(HORIZON, np.float64),
      is_first=np.zeros(HORIZON, bool), is_last=np.zeros(HORIZON, bool),
      is_terminal=np.zeros(HORIZON, bool), action_executed=np.zeros(HORIZON, bool),
      observation_executed=np.zeros(HORIZON + 1, bool),
      pre_carry_sha256=[''] * HORIZON, after_carry_sha256=[''] * HORIZON,
      proofs=[], after_carries=[], original_current_nj_seed_calls=2,
      first_assembly_model_calls=0, first_assembly_nj_seed_calls=0,
      future_normal_policy_calls=0, renderer_byte_exact_established=False)
  record['observation_executed'][0] = True
  end_type = 'window'
  for step in range(HORIZON):
    check()
    before, obs_hash = _carry(carry), source.tree_digest(obs)
    key, key_bytes = keys[step].copy(), keys[step].tobytes()
    if step == 0:
      untouched = _digest(bundle)
      result = policy.first(bundle, arm=arm)
      require(_digest(bundle) == untouched, 'First assembly mutated the current bundle')
    else:
      result = policy.normal_step(carry, obs, key)
      record['future_normal_policy_calls'] += 1
    require(_carry(carry) == before and source.tree_digest(obs) == obs_hash
        and key.tobytes() == key_bytes, 'Policy mutated a carry/observation/key input')
    require(isinstance(result, tuple) and len(result) == 3,
        'R4 policy must return after carry, raw, numeric proof')
    after, raw, proof = result
    _array(raw, np.float32, (12,), 'actual R4 raw action')
    after_hash = _carry(after, raw)
    _proof(proof, keys[step], raw, after, arm=arm if step == 0 else 'qq')
    if step == 0:
      _first_matches(bundle, arm, after, raw, proof)
    next_obs, applied = adapter.step(raw.copy())
    feedback._observation(next_obs)
    _array(applied, np.float64, (12,), 'actual applied control')
    ts = adapter.recorder.last_timestep
    reward, discount = feedback._timestep(ts)
    feedback._flags(next_obs, ts)
    require(np.float32(reward).tobytes() == next_obs['reward'].tobytes(),
        'F64 reward differs from the production F32 encoding')
    state = adapter.snapshot()
    feedback._snapshot(state)
    require(bool(state['from_dm_done']) == bool(ts.last()),
        'Production wrapper done state differs')
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
    record['after_carries'].append(copy.deepcopy(after))
    record['observations'].append(copy.deepcopy(next_obs))
    record['snapshots'].append(state)
    carry, obs = copy.deepcopy(after), next_obs
    if ts.last():
      if discount == 0:
        end_type = 'true_terminal'
      else:
        require(step == HORIZON - 1
            and adapter.suite._step_count == adapter.suite._step_limit,
            'Early or unexpected time limit is not absorbing terminal')
        end_type = 'time_limit'
      break
  record.update(executed_steps=len(record['proofs']), end_type=end_type,
      final_carry=copy.deepcopy(carry), visual_identity_after=_visual(adapter, visual_identity))
  validate_record(record)
  check()
  return record


def validate_record(record):
  require(isinstance(record, dict) and record.get('version') == VERSION
      and record.get('implementation') == IMPLEMENTATION and record.get('schema') == SCHEMA
      and record.get('action_contract') == source.CONTRACT,
      'Legacy or unknown R4 branch schema')
  require(record['horizon'] == HORIZON and type(record['horizon']) is int
      and record['arm'] in ARMS and type(record['executed_steps']) is int
      and 1 <= record['executed_steps'] <= HORIZON,
      'R4 branch has invalid fixed horizon, arm or executed count')
  count, arm = record['executed_steps'], record['arm']
  for name, dtype, shape in (
      ('keys', np.uint32, (HORIZON, 2)), ('raw', np.float32, (HORIZON, 12)),
      ('applied', np.float64, (HORIZON, 12)), ('reward_f64', np.float64, (HORIZON,)),
      ('reward_f32', np.float32, (HORIZON,)), ('discount', np.float64, (HORIZON,)),
      ('is_first', bool, (HORIZON,)), ('is_last', bool, (HORIZON,)),
      ('is_terminal', bool, (HORIZON,)), ('action_executed', bool, (HORIZON,)),
      ('observation_executed', bool, (HORIZON + 1,))):
    _array(record[name], dtype, shape, name)
  require(np.array_equal(record['action_executed'], np.arange(HORIZON) < count)
      and np.array_equal(record['observation_executed'], np.arange(HORIZON + 1) <= count),
      'Actual execution masks must describe exactly the consecutive prefix')
  for name in ('raw', 'applied', 'reward_f64', 'reward_f32', 'discount',
      'is_first', 'is_last', 'is_terminal'):
    require(not np.any(record[name][count:] != 0), 'Nonzero absorbing padding')
  require(len(record['observations']) == count + 1
      and len(record['snapshots']) == count + 1
      and len(record['proofs']) == count and len(record['after_carries']) == count,
      'Missing genuine evidence or fabricated padded observation/state')
  require(record['current_bundle_sha256'] == _bundle(record['current_bundle'], record['keys'][0]),
      'Current bundle content/hash differs')
  require(record['current_bundle']['before_carry_sha256']
        == policy_tree_digest(record['initial_carry'])
      and record['current_bundle']['observation_sha256']
        == policy_tree_digest(record['observations'][0]),
      'Recorded current bundle input history/observation differs')
  for name in ('pre_carry_sha256', 'after_carry_sha256'):
    require(len(record[name]) == HORIZON
        and all(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v)
            for v in record[name][:count]) and not any(record[name][count:]),
        'Carry hash execution coverage differs')
  previous = _carry(record['initial_carry'])
  initial = record['snapshots'][0]
  require(int(initial['suite']['step_count']) in POSITIONS
      and not initial['from_dm_done'] and not initial['suite']['reset_next_step']
      and not bool(record['observations'][0]['is_first'])
      and not bool(record['observations'][0]['is_last']), 'Invalid ordinary initial state')
  for obs, snapshot in zip(record['observations'], record['snapshots']):
    feedback._observation(obs)
    feedback._snapshot(snapshot)
  for step in range(count):
    require(previous == record['pre_carry_sha256'][step], 'Broken pre-carry hash chain')
    after, raw, proof = (record['after_carries'][step], record['raw'][step],
                        record['proofs'][step])
    _proof(proof, record['keys'][step], raw, after, arm=arm if step == 0 else 'qq')
    previous = _carry(after, raw)
    require(previous == record['after_carry_sha256'][step], 'Broken after-carry hash chain')
    if step == 0:
      _first_matches(record['current_bundle'], arm, after, raw, proof)
    obs, snapshot = record['observations'][step + 1], record['snapshots'][step + 1]
    require(record['reward_f32'][step].tobytes() == obs['reward'].tobytes()
        and np.float32(record['reward_f64'][step]).tobytes() == obs['reward'].tobytes()
        and tuple(bool(record[k][step]) for k in OBS_KEYS[2:])
          == tuple(bool(obs[k]) for k in OBS_KEYS[2:])
        and not bool(obs['is_first'])
        and bool(snapshot['from_dm_done']) == bool(obs['is_last'])
        and bool(snapshot['suite']['reset_next_step']) == bool(obs['is_last'])
        and int(snapshot['suite']['step_count']) == int(initial['suite']['step_count']) + step + 1
        and record['discount'][step] in (0, 1)
        and bool(obs['is_terminal']) == bool(record['discount'][step] == 0)
        and (not bool(obs['is_last']) or step == count - 1),
        'R4 transition reward, clock, flags or reset semantics differ')
  require(_carry(record['final_carry'], record['raw'][count - 1]) == previous,
      'Final carry differs from complete hash chain')
  terminal, last = bool(record['is_terminal'][count - 1]), bool(record['is_last'][count - 1])
  expected_end = 'true_terminal' if terminal else ('time_limit' if last else 'window')
  require(record['end_type'] == expected_end and (count == HORIZON or terminal),
      'An incomplete window is not an absorbing terminal')
  require(record['original_current_nj_seed_calls'] == 2
      and record['first_assembly_model_calls'] == 0
      and record['first_assembly_nj_seed_calls'] == 0
      and record['future_normal_policy_calls'] == count - 1,
      'R4 current generation/host assembly/normal call receipt differs')
  require(record['renderer_byte_exact_established'] is False,
      'R4 does not establish renderer byte equivalence')
  for name in ('visual_identity', 'visual_identity_after'):
    require(record[name].get('renderer_byte_exact_established') is False
        and record[name].get('complete_effective_render_inputs_authenticated') is False,
        'A static model receipt cannot prove renderer identity')
  require(record['visual_identity'] == record['visual_identity_after'],
      'Static model or live adapter identity changed during branch')
  receipt = record['initial_receipt']
  require(receipt.get('shared_actual_source_rgb') is True
      and receipt.get('fresh_initial_render_count') == 0
      and receipt.get('physical_restore_byte_exact') is True
      and receipt.get('source_image_sha256') == hashlib.sha256(
          record['observations'][0]['image'].tobytes()).hexdigest()
      and receipt.get('renderer_byte_exact_established') is False,
      'Initial shared-source RGB/restore receipt differs')
  return count


def verify_physics_replay(adapter, record, *, check=lambda: None):
  """Stored-applied official suite replay, no model or render calls."""
  from scripts.dl_action_contract import production_action_stages
  count = validate_record(record)
  _visual(adapter, record['visual_identity'], same_live=False)
  adapter.restore(record['snapshots'][0])
  _snapshot_exact(adapter.snapshot(), record['snapshots'][0])
  for step in range(count):
    check()
    applied = production_action_stages(record['raw'][step], adapter.low, adapter.high)['applied']
    _same(applied, record['applied'][step], 'Stored applied action differs from actual mapping')
    ts = adapter.suite.step(record['applied'][step].copy())
    reward, discount = feedback._timestep(ts)
    adapter.from_dm._done = bool(ts.last())
    require(reward.tobytes() == record['reward_f64'][step].tobytes()
        and np.float32(reward).tobytes() == record['reward_f32'][step].tobytes()
        and discount.tobytes() == record['discount'][step].tobytes(),
        'Stored-control physics F64/F32 reward or discount byte replay differs')
    feedback._flags(record['observations'][step + 1], ts)
    _snapshot_exact(adapter.snapshot(), record['snapshots'][step + 1])
    if ts.last() and discount == 1:
      require(step == HORIZON - 1 and adapter.suite._step_count == adapter.suite._step_limit,
          'Time-limit replay does not end at the frozen clock/window')
  _visual(adapter, record['visual_identity'], same_live=False)
  check()
  return dict(version=VERSION, implementation=IMPLEMENTATION, schema=SCHEMA,
      physics_steps=count, absorbing_padding_steps=HORIZON - count,
      reward_float64_byte_exact=True, reward_float32_byte_exact=True,
      applied_mapping_byte_exact=True, full_snapshot_byte_exact=True,
      policy_replayed=False, renderer_replayed=False)


def replay_cached_policy(policy, record, *, check=lambda: None):
  """Recompute current bundle and all policy calls on each branch's own saved RGB."""
  count = validate_record(record)
  carry = copy.deepcopy(record['initial_carry'])
  check()
  initial_obs = copy.deepcopy(record['observations'][0])
  key = record['keys'][0].copy()
  before, obs_hash, key_bytes = _carry(carry), source.tree_digest(initial_obs), key.tobytes()
  bundle = policy.current(carry, initial_obs, key)
  require(_carry(carry) == before and source.tree_digest(initial_obs) == obs_hash
      and key.tobytes() == key_bytes, 'Cached current call mutated its inputs')
  require(_bundle(bundle, key) == record['current_bundle_sha256'],
      'Cached current q/p bundle differs byte-exactly')
  for step in range(count):
    check()
    require(_carry(carry) == record['pre_carry_sha256'][step], 'Cached pre-carry differs')
    obs, key = copy.deepcopy(record['observations'][step]), record['keys'][step].copy()
    before, obs_hash, key_bytes = _carry(carry), source.tree_digest(obs), key.tobytes()
    if step == 0:
      bundle_before = _digest(bundle)
      after, raw, proof = policy.first(bundle, arm=record['arm'])
      require(_digest(bundle) == bundle_before, 'Cached first assembly mutated bundle')
    else:
      after, raw, proof = policy.normal_step(carry, obs, key)
    require(_carry(carry) == before and source.tree_digest(obs) == obs_hash
        and key.tobytes() == key_bytes, 'Cached policy call mutated its inputs')
    _array(raw, np.float32, (12,), 'cached raw')
    _proof(proof, key, raw, after, arm=record['arm'] if step == 0 else 'qq')
    _same(raw, record['raw'][step], 'Cached policy raw differs byte-exactly')
    require(_carry(after, raw) == record['after_carry_sha256'][step],
        'Cached outgoing carry differs byte-exactly')
    for name in PROOF_KEYS:
      _same(proof[name], record['proofs'][step][name],
          'Cached numeric policy receipt differs: ' + name)
    carry = after
  check()
  return dict(version=VERSION, implementation=IMPLEMENTATION, schema=SCHEMA,
      policy_steps=count, current_bundle_recomputed=True,
      current_model_calls=1, first_assembly_model_calls=0,
      raw_byte_exact=True, full_carry_byte_exact=True,
      numeric_proofs_byte_exact=True, cached_own_rgb=True,
      physics_replayed=False, renderer_replayed=False)


def validate_pair(records):
  require(isinstance(records, dict) and set(records) == set(ARMS),
      'Exactly qq/pq/pp branch records are required')
  for arm in ARMS:
    validate_record(records[arm])
    require(records[arm]['arm'] == arm, 'Arm label mismatch')
  qq, pq, pp = (records[a] for a in ARMS)
  for other in (pq, pp):
    _same(other['keys'], qq['keys'], 'Three arms must share all planned private roots')
    _snapshot_exact(other['snapshots'][0], qq['snapshots'][0])
    require(source.tree_digest(other['observations'][0]) == source.tree_digest(qq['observations'][0])
        and _carry(other['initial_carry']) == _carry(qq['initial_carry'])
        and other['current_bundle_sha256'] == qq['current_bundle_sha256']
        and other['visual_identity'] == qq['visual_identity'],
        'Three arms do not share current history/observation/model/bundle')
  require(source.tree_digest(qq['after_carries'][0][1])
      == source.tree_digest(pq['after_carries'][0][1]), 'qq/pq current q memory differs')
  for name in ('raw', 'applied', 'reward_f64', 'reward_f32', 'discount'):
    _same(pp[name][0], pq[name][0], 'pq/pp first physical action/reward differs: ' + name)
  _snapshot_exact(pp['snapshots'][1], pq['snapshots'][1])
  for name in ('is_first', 'is_last', 'is_terminal'):
    require(pp[name][0] == pq[name][0], 'pq/pp first physical flags differ')
  return dict(version=VERSION, arms=list(ARMS), qq_pq_current_memory_byte_exact=True,
      pq_pp_first_raw_applied_reward_snapshot_byte_exact=True,
      common_current_rgb_history=True, common_private_roots=True,
      future_rendered_pixels_paired=False, interaction_estimated=False)


def run_state_pair(policy, adapter, snapshot, hostcarry, keys, *, initial_obs,
                   visual_identity, pair, check=lambda: None):
  """Three cyclically ordered arms; pair0 also has one full qq->qq null run."""
  order = branch_order(pair)
  check()
  before, obs_hash = _carry(hostcarry), source.tree_digest(initial_obs)
  _array(keys, np.uint32, (HORIZON, 2), 'R4 private key sequence')
  root = keys[0].copy()
  bundle = policy.current(hostcarry, initial_obs, root)
  require(_carry(hostcarry) == before and source.tree_digest(initial_obs) == obs_hash
      and root.tobytes() == keys[0].tobytes(), 'Current bundle generation mutated inputs')
  _bundle(bundle, keys[0])
  records = {}
  for arm in order:
    check()
    records[arm] = run_branch(policy, adapter, snapshot, hostcarry, keys,
        arm=arm, bundle=bundle, initial_obs=initial_obs,
        visual_identity=visual_identity, check=check)
  pairing = validate_pair(records)
  null = None
  if pair == 0:
    null = run_branch(policy, adapter, snapshot, hostcarry, keys, arm='qq', bundle=bundle,
        initial_obs=initial_obs, visual_identity=visual_identity, check=check)
    _same(null['raw'][0], records['qq']['raw'][0], 'qq null current action differs')
    require(null['after_carry_sha256'][0] == records['qq']['after_carry_sha256'][0],
        'qq null current carry differs')
  check()
  return dict(version=VERSION, implementation=IMPLEMENTATION, schema=SCHEMA,
      pair=pair, order=list(order), records=records, pairing=pairing,
      null_record=null, null_performed=pair == 0,
      null_reward_difference=None if null is None else
        (records['qq']['reward_f64'] - null['reward_f64']).copy(),
      renderer_byte_exact_established=False)
