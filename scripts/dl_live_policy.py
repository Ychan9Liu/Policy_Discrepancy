"""DL-r5 thin frozen-policy adapter: one live intervention, continuous keys.

The unchanged R4 kernel remains honestly named in each current envelope. This
module has no environment, loader, resource allocator, experiment or statistics.
The owning runner authenticates external artifacts and seal-before-reveal order.
"""
import copy
import re

import numpy as np

from scripts import dl_r4_policy as kernel
from scripts import dl_live_shadow as shadow

VERSION = 'DL-r5-live-r1'
KEY_NAMESPACE = 'DL_R5_LIVE_U1'
SEED = 20261030
EVAL_SEED_BASE = 0x52350000
LABELS = ('qqA', 'qqB', 'pq', 'pp')
PROOF_KEYS = kernel.PROOF_KEYS
require = kernel.require


def _unit(unit):
  require(type(unit) is int and 0 <= unit < 512, 'Require unit0..511')
  return unit


def _hash(value):
  require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'Invalid SHA256')
  return value


def unit_plan(unit):
  unit = _unit(unit)
  index = int(np.random.default_rng([SEED, 0x54494D45, unit]).integers(0, 14))
  return dict(unit=unit, family=unit // 16, replicate=unit % 16,
      env_seed=8192 + unit // 16, eval_seed=EVAL_SEED_BASE + unit,
      t_index=index, t=64 + 64 * index)


def policy_root_key(unit, step):
  plan = unit_plan(unit)
  require(type(step) is int and 0 <= step <= plan['t'] + 99,
      'Require the continuous absolute live policy counter')
  return np.random.default_rng([0, kernel.EVAL_TAG, plan['eval_seed'], step]).integers(
      0, np.iinfo(np.uint32).max, (2,), np.uint32)


def reveal_label(unit, pretreatment_sha256):
  """Call only after the owner has durably sealed all pretreatment signals."""
  _unit(unit)
  _hash(pretreatment_sha256)
  index = int(np.random.default_rng([SEED, 0x41524D53, unit]).integers(0, 4))
  return dict(version=VERSION, key_namespace=KEY_NAMESPACE, unit=unit,
      label_index=index, label=LABELS[index], pretreatment_sha256=pretreatment_sha256)


def _digest(value, field):
  return kernel.full_digest({k: v for k, v in value.items() if k != field})


class LivePolicy:
  def __init__(self, agent):
    self.kernel = kernel.R4Policy(agent)
    self.agent = agent
    self._issued, self._sealed = set(), {}
    self._oracles = 0

  def initial(self, unit):
    return self.kernel.initial(policy_root_key(unit, 0))

  def normal_step(self, carry, obs, unit, step):
    return self.kernel.normal_step(carry, obs, policy_root_key(unit, step))

  def current(self, carry, obs, unit):
    plan = unit_plan(unit)
    inner = self.kernel.current(carry, obs, policy_root_key(unit, plan['t']))
    bundle = dict(version=VERSION, key_namespace=KEY_NAMESPACE, unit=unit,
        step=plan['t'], eval_seed=plan['eval_seed'], kernel_version=kernel.VERSION,
        kernel_namespace=kernel.KEY_NAMESPACE, kernel_bundle=inner,
        r_t_f32=np.asarray(obs['reward']).copy())
    for name in ('Dmode', 'K', 'q_raw', 'q_after', 'root_key',
        'before_carry_sha256', 'observation_sha256'):
      bundle[name] = copy.deepcopy(inner[name])
    bundle['bundle_sha256'] = _digest(bundle, 'bundle_sha256')
    self._issued.add(bundle['bundle_sha256'])
    return bundle

  def _bundle(self, bundle):
    require(type(bundle) is dict and bundle.get('version') == VERSION and
        bundle.get('key_namespace') == KEY_NAMESPACE and
        bundle.get('bundle_sha256') in self._issued and
        _digest(bundle, 'bundle_sha256') == bundle['bundle_sha256'],
        'Require unchanged issued R5 current envelope')
    plan = unit_plan(bundle['unit'])
    require(bundle['step'] == plan['t'] and bundle['eval_seed'] == plan['eval_seed'] and
        bundle['root_key'].tobytes() == policy_root_key(plan['unit'], plan['t']).tobytes(),
        'R5 current unit/counter/root differs')
    self.kernel._validate_bundle(bundle['kernel_bundle'])
    return bundle['kernel_bundle']

  def score_h1(self, bundle, shadow_receipt):
    inner = self._bundle(bundle)
    require(shadow.response_sha256(shadow_receipt) == shadow_receipt['response_sha256'],
        'Shadow response SHA differs')
    require(shadow_receipt.get('version') == shadow.VERSION and
        shadow_receipt.get('unit') == bundle['unit'] and
        shadow_receipt.get('env_seed') == unit_plan(bundle['unit'])['env_seed'] and
        shadow_receipt.get('bundle_sha256') == bundle['bundle_sha256'] and
        shadow_receipt.get('render_calls') == shadow_receipt.get('model_calls') == 0 and
        shadow_receipt.get('official_steps') == 1 and
        shadow_receipt.get('live_environment_touched') is False and
        shadow_receipt['raw_q'].dtype == np.float32 and
        shadow_receipt['raw_q'].tobytes() == inner['q_raw'].tobytes(),
        'H1 requires the bound independent baseline-q shadow receipt')
    _hash(shadow_receipt['request_sha256'])
    _hash(shadow_receipt['source_snapshot_sha256'])
    require(shadow.snapshot_sha256(shadow_receipt['next_snapshot']) ==
        shadow_receipt['next_snapshot_sha256'], 'Shadow next snapshot SHA differs')
    reward = np.asarray(shadow_receipt['reward_f64'])
    require(np.asarray(shadow_receipt['reward_f32']).dtype == np.float32 and
        np.asarray(shadow_receipt['reward_f32']).tobytes() == np.asarray(reward, np.float32).tobytes(),
        'Shadow next F64/F32 encoding differs')
    return self.kernel.score_h1(inner, inner['q_raw'], reward)

  def seal_pretreatment(self, bundle, shadow_receipt, *, live_before_sha256,
                       live_after_sha256, r_t_f64):
    require(_hash(live_before_sha256) == _hash(live_after_sha256) ==
        shadow_receipt['source_snapshot_sha256'], 'Shadow wait changed the live full state')
    score = self.score_h1(bundle, shadow_receipt)
    current_reward = np.asarray(r_t_f64)
    require(current_reward.dtype == np.float64 and current_reward.shape == () and
        np.isfinite(current_reward) and 0 <= current_reward <= 1 and
        np.asarray(current_reward, np.float32).tobytes() == bundle['r_t_f32'].tobytes(),
        'Pretreatment baseline requires actual current F64 reward and exact F32 encoding')
    seal = dict(version=VERSION, key_namespace=KEY_NAMESPACE, unit=bundle['unit'],
        bundle_sha256=bundle['bundle_sha256'],
        source_snapshot_sha256=live_before_sha256,
        shadow_request_sha256=shadow_receipt['request_sha256'],
        Dmode=bundle['Dmode'].copy(), K=bundle['K'].copy(), e=score['e'].copy(),
        r_t=current_reward.copy(), r_t_f32=bundle['r_t_f32'].copy(),
        score_h1=score, label_revealed=False)
    seal['pretreatment_sha256'] = _digest(seal, 'pretreatment_sha256')
    self._sealed[bundle['bundle_sha256']] = seal['pretreatment_sha256']
    return seal

  def first(self, bundle, label, *, label_receipt):
    inner = self._bundle(bundle)
    require(type(label) is str and label in LABELS and
        bundle['bundle_sha256'] in self._sealed and
        label_receipt == reveal_label(bundle['unit'], self._sealed[bundle['bundle_sha256']]) and
        label_receipt['label'] == label, 'First intervention requires its sealed label receipt')
    return self.kernel.first(inner, 'qq' if label in ('qqA', 'qqB') else label)

  def public_oracle(self, carry, obs, unit, step, *, expected_after, expected_raw):
    """Actual unchanged public policy at the new domain and absolute counter."""
    import jax
    from embodied.jax import internal
    key = policy_root_key(unit, step)
    plan = unit_plan(unit)
    self.assert_frozen(full=True)
    original = kernel.counters(self.agent)
    carry_sha, obs_sha = kernel.tree_digest(carry), kernel.tree_digest(kernel.observation(obs))
    expected_sha = kernel.tree_digest(expected_after)
    require(type(expected_raw) is np.ndarray and expected_raw.dtype == np.float32 and
        expected_raw.shape == (12,), 'Expected actual private q raw required')
    agent = self.agent
    try:
      agent.set_eval_seed(plan['eval_seed'])
      with agent.n_eval_actions.lock:
        agent.n_eval_actions.value = step
      actual_key = kernel.host_copy(agent._eval_seeds(step, agent.policy_mirrored))
      require(actual_key.tobytes() == key.tobytes(), 'Original public R5 key differs')
      placed = internal.device_put(copy.deepcopy(carry), internal.local_sharding(agent.policy_mirrored))
      actual_after, actual_action, _ = agent.policy(placed,
          {k: v[None] for k, v in kernel.observation(obs).items()}, mode='eval')
      jax.block_until_ready(actual_after)
      actual_after = kernel.host_copy(actual_after)
      raw = np.asarray(actual_action['action'][0])
      require(raw.dtype == expected_raw.dtype and raw.shape == expected_raw.shape and
          raw.tobytes() == expected_raw.tobytes() and
          kernel.tree_digest(actual_after) == expected_sha, 'Public/private q output not byte exact')
    finally:
      agent.set_eval_seed(original['eval_seed'])
      with agent.n_eval_actions.lock:
        agent.n_eval_actions.value = original['eval_actions']
      self.assert_frozen(full=True)
    require(kernel.tree_digest(carry) == carry_sha and
        kernel.tree_digest(kernel.observation(obs)) == obs_sha, 'Oracle mutated live policy inputs')
    self._oracles += 1
    return dict(version=VERSION, key_namespace=KEY_NAMESPACE, unit=unit, step=step,
        eval_seed=plan['eval_seed'], root_key=key, public_policy_calls=1,
        public_q_raw_byte_exact=True, public_q_full_after_carry_byte_exact=True,
        original_counters_restored=True, before_carry_sha256=carry_sha,
        observation_sha256=obs_sha, after_carry_sha256=expected_sha)

  def assert_frozen(self, full=True):
    self.kernel.assert_frozen(full=full)

  def receipt(self):
    return dict(version=VERSION, key_namespace=KEY_NAMESPACE,
        kernel=self.kernel.receipt(), public_oracle_calls=self._oracles,
        sealed_current_bundles=len(self._sealed), label_not_in_policy_root=True,
        public_rng_not_consumed_by_private_calls=True)
