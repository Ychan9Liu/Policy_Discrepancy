"""DL-rep-native-assay-r1: current-params readonly policy and full episode.

No loader, CLI, simulator, renderer, device selection or source authorization.
The driver authenticates inputs, actual environment/rendering, oracle receipts,
physics snapshots, resource mapping and the whole owned-process deadline.
"""
import contextlib
import copy
import threading

import numpy as np

from scripts.dl_reset_features import counters, host_copy, tree_digest
from scripts.dl_reset_learning import tree_digest as full_digest

VERSION = 'DL-rep-native-assay-r1'
SEEDS = tuple(range(1024, 1056))
HORIZON = 1000
ORACLE_STEPS = (0, 500, 999)
OBS_KEYS = ('image', 'reward', 'is_first', 'is_last', 'is_terminal')
_TRACE_LOCK = threading.RLock()


def require(condition, message):
  if not condition:
    raise ValueError(message)


def private_root_key(seed, step):
  """Exact official seed0 evaluation root; no group or call-order label."""
  require(type(seed) is int and seed in SEEDS and type(step) is int and
      0 <= step < HORIZON, 'Require fixed evaluation seed and step')
  return np.random.default_rng([0, 0x4556414C, seed, step]).integers(
      0, np.iinfo(np.uint32).max, (2,), np.uint32)


def observation(obs):
  require(isinstance(obs, dict) and set(obs) == set(OBS_KEYS),
      'Exact production observation fields required')
  result = {}
  for name, dtype, shape in (
      ('image', np.uint8, (64, 64, 3)), ('reward', np.float32, ()),
      ('is_first', np.bool_, ()), ('is_last', np.bool_, ()),
      ('is_terminal', np.bool_, ())):
    value = np.asarray(obs[name])
    require(value.dtype == np.dtype(dtype) and value.shape == shape and
        np.isfinite(value).all(), 'Observation dtype/shape/finite differs: ' + name)
    result[name] = value.copy()
  require(not bool(result['is_terminal']) or bool(result['is_last']),
      'Terminal must also be last')
  return result


@contextlib.contextmanager
def _sampling_keys():
  import ninjax as nj
  original = nj.seed
  keys = []
  def traced(*args, **kwargs):
    require(not args and not kwargs, 'Unexpected native sampling seed signature')
    key = original()
    keys.append(key)
    return key
  with _TRACE_LOCK:
    nj.seed = traced
    try:
      yield keys
    finally:
      nj.seed = original


def _readonly(fn, accessed):
  import ninjax as nj
  pure = nj.pure(fn)
  def call(params, *args, root):
    _, result, read, modified, created = pure(params, *args, seed=root,
        create=False, modify=True, ignore=False, track=True)
    if modified or created:
      raise AssertionError('Native policy attempted model state writes')
    accessed.update(read)
    return result
  return call


class NativeActor:
  """One frozen current Agent; carry is the official model's batched B1 tree.

  A new instance is required after training/load. No public policy copy, public
  RNG method or public counter is read/advanced by this adapter.
  """
  def __init__(self, agent):
    import jax
    require(len(agent.train_devices) == 1 and not agent.jaxcfg.use_shardmap,
        'Require one current train device without shardmap')
    require(int(agent.config.seed) == 0 and
        str(agent.config.jax.compute_dtype) == 'bfloat16', 'Require seed0 BF16 model')
    require(set(agent.obs_space) == set(OBS_KEYS) and
        set(agent.model.enc.obs_space) == {'image'} and
        set(agent.act_space) == {'action'} and agent.act_space['action'].shape == (12,),
        'Require original vision64/action12 spaces')
    self.agent = agent
    self.initial_params = agent.params
    self.initial_values = tuple((k, id(v)) for k, v in sorted(agent.params.items()))
    self.initial_counters = counters(agent)
    self.initial_state_sha256 = self._state_hash()
    self.accessed_keys = set()
    self._init_calls = self._private_calls = self._public_oracle_calls = 0
    self._oracle_policy_copy_sha256 = None
    self._init = jax.jit(_readonly(lambda: agent.model.init_policy(1), self.accessed_keys))
    def policy(carry, obs):
      with _sampling_keys() as keys:
        after, actions, out = agent.model.policy(carry, obs, mode='eval')
      if len(keys) != 2:
        raise AssertionError('Native policy must consume exactly two nj.seed calls')
      return after, actions, out, dict(latent_key=keys[0], actor_key=keys[1],
          rng_consumption=np.int32(2))
    self._policy = jax.jit(_readonly(policy, self.accessed_keys))

  def _state_hash(self):
    # Agent.save acquires the non-reentrant train_lock. This one-device view
    # has identical typed leaves and is safe while the adapter holds that lock.
    return full_digest(dict(params=host_copy(self.agent.params), counters=counters(self.agent)))

  def assert_frozen(self, full=False):
    if (self.agent.params is not self.initial_params or
        tuple((k, id(v)) for k, v in sorted(self.agent.params.items())) != self.initial_values or
        counters(self.agent) != self.initial_counters):
      raise AssertionError('Native actor requires unchanged current params/counters')
    if full and self._state_hash() != self.initial_state_sha256:
      raise AssertionError('Native actor full optimizer/model/normalizer state changed')

  def _carry(self, carry, initial=False):
    require(type(carry) is tuple and len(carry) == 4 and carry[0] == {} and
        carry[2] == {} and isinstance(carry[1], dict) and
        set(carry[1]) == {'deter', 'stoch'} and isinstance(carry[3], dict) and
        set(carry[3]) == {'action'}, 'Original batched B1 carry structure required')
    cfg = self.agent.config.dyn.rssm
    for name, shape in (('deter', (1, cfg.deter)),
        ('stoch', (1, cfg.stoch, cfg.classes))):
      value = carry[1][name]
      require(type(value) is np.ndarray and str(value.dtype) == 'bfloat16' and
          value.shape == shape and np.isfinite(value).all(), 'Carry differs: ' + name)
    raw = carry[3]['action']
    require(type(raw) is np.ndarray and raw.dtype == np.float32 and
        raw.shape == (1, 12) and np.isfinite(raw).all(), 'Previous raw action differs')
    z = carry[1]['stoch'].astype(np.float32)
    if initial:
      require(np.all(carry[1]['deter'] == 0) and np.all(z == 0) and np.all(raw == 0),
          'Official initial carry must be zero')
    else:
      require(np.all((z == 0) | (z == 1)) and np.all(z.sum(-1) == 1),
          'Actual posterior sample must be categorical onehot')
    tree_digest(carry)
    return carry

  def initial(self, seed):
    import jax
    from embodied.jax import internal
    root = private_root_key(seed, 0)
    with self.agent.train_lock:
      self.assert_frozen(full=True)
      placed = internal.device_put(root, self.agent.train_mirrored)
      result = self._init(self.agent.params, root=placed)
      jax.block_until_ready(result)
      carry = host_copy(result)
      self._carry(carry, initial=True)
      self.assert_frozen(full=True)
    self._init_calls += 1
    return carry

  def step(self, carry, obs, seed, step):
    import jax
    from embodied.jax import internal
    root = private_root_key(seed, step)
    validated = observation(obs)
    self._carry(carry, initial=(step == 0))
    require(bool(validated['is_first']) == (step == 0) and
        not bool(validated['is_last']), 'Policy reset/last position differs')
    if step == 0:
      require(validated['reward'] == 0, 'Reset reward must be zero')
    before_hash = tree_digest(carry)
    obs_hash = tree_digest(validated)
    with self.agent.train_lock:
      self.assert_frozen()
      placed_carry, placed_obs, placed_root = internal.device_put(
          (carry, {k: v[None] for k, v in validated.items()}, root),
          self.agent.train_mirrored)
      result = self._policy(self.agent.params, placed_carry, placed_obs, root=placed_root)
      jax.block_until_ready(result)
      after, actions, out, proof = host_copy(result)
      self.assert_frozen()
    self._carry(after)
    require(isinstance(actions, dict) and set(actions) == {'action'}, 'Action fields differ')
    value = actions['action']
    require(value.dtype == np.float32 and value.shape == (1, 12) and
        np.isfinite(value).all(), 'Native action must be finite float32 B1 action12')
    raw = value[0].copy()
    require(after[3]['action'].tobytes() == value.tobytes(), 'Raw action/carry differ')
    finite = out.get('finite', {})
    require(finite and all(bool(np.asarray(x).all()) for x in finite.values()),
        'Official production policy finite checks failed')
    for key in ('latent_key', 'actor_key'):
      require(proof[key].dtype == np.uint32 and proof[key].shape == (2,),
          'Actual native sampling key differs')
    if tree_digest(carry) != before_hash or tree_digest(validated) != obs_hash:
      raise AssertionError('Native policy mutated input carry/observation')
    proof.update(root_key=root.copy(), seed=seed, step=step,
        before_carry_sha256=before_hash, after_carry_sha256=tree_digest(after),
        observation_sha256=obs_hash, current_full_state_sha256=self.initial_state_sha256)
    self._private_calls += 1
    return after, raw, proof

  def _policy_copy_digest(self):
    require(self.agent.jaxcfg.enable_policy and self.agent.pending_sync is None,
        'Public oracle requires enabled synchronized policy with no pending copy')
    current = {key: self.agent.params[key] for key in self.agent.policy_keys}
    copy_hash = tree_digest(host_copy(self.agent.policy_params))
    require(copy_hash == tree_digest(host_copy(current)),
        'Public oracle policy copy is not typed-byte equal to current params')
    if self._oracle_policy_copy_sha256 is not None:
      require(copy_hash == self._oracle_policy_copy_sha256, 'Oracle policy copy changed')
    return copy_hash

  def oracle(self, *, step, carry, observation, root, after_carry, raw, proof):
    """Public original policy equivalence, restoring its temporary eval counters."""
    import jax
    from embodied.jax import internal
    seed = proof['seed']
    require(step in ORACLE_STEPS and proof['step'] == step and
        root.tobytes() == private_root_key(seed, step).tobytes(), 'Fixed oracle key differs')
    self.assert_frozen(full=True)
    copy_hash = self._policy_copy_digest()
    original = counters(self.agent)
    agent = self.agent
    try:
      agent.set_eval_seed(seed)
      with agent.n_eval_actions.lock: agent.n_eval_actions.value = step
      actual_root = host_copy(agent._eval_seeds(step, agent.policy_mirrored))
      require(actual_root.dtype == root.dtype and actual_root.shape == root.shape and
          actual_root.tobytes() == root.tobytes(), 'Public eval seed derivation differs')
      split_carry = jax.tree.map(lambda value: [value[0].copy()], carry)
      placed = internal.device_put(split_carry, internal.local_sharding(agent.policy_mirrored))
      public_after, public_action, _ = agent.policy(placed,
          {key: value[None] for key, value in observation.items()}, mode='eval')
      actual_raw = np.asarray(public_action['action'][0])
      require(actual_raw.dtype == raw.dtype and actual_raw.shape == raw.shape and
          actual_raw.tobytes() == raw.tobytes(), 'Public/private native raw action differs')
      # Public wrapper returns one list element per environment; private keeps B1.
      def stack(tree):
        if isinstance(tree, dict): return {key: stack(value) for key, value in tree.items()}
        if isinstance(tree, tuple): return tuple(stack(value) for value in tree)
        if isinstance(tree, list): return np.stack(tree)
        return tree
      require(tree_digest(stack(host_copy(public_after))) == tree_digest(after_carry),
          'Public/private complete after carry differs')
    finally:
      agent.set_eval_seed(original['eval_seed'])
      with agent.n_eval_actions.lock: agent.n_eval_actions.value = original['eval_actions']
      self.assert_frozen(full=True)
      require(self._policy_copy_digest() == copy_hash, 'Oracle changed synchronized policy copy')
    self._oracle_policy_copy_sha256 = copy_hash
    self._public_oracle_calls += 1
    return dict(public_raw_byte_exact=True, public_after_carry_byte_exact=True,
        currentparams_policycopy_typed_byte_exact=True, eval_counters_restored=True,
        full_state_and_counters_unchanged=True, actual_native_sample=True,
        policy_copy_sha256=copy_hash, root_key=root.copy(), public_model_calls=1)

  def receipt(self):
    self.assert_frozen(full=True)
    if self._oracle_policy_copy_sha256 is not None: self._policy_copy_digest()
    return dict(version=VERSION, current_full_state_sha256=self.initial_state_sha256,
        currentparams_identity_unchanged=True, scientific_policy_reads_currentparams=True,
        readonly_full_state_and_counters_unchanged=True,
        public_policy_copy_used_for_scientific_calls=False,
        synchronized_policy_copy_sha256=self._oracle_policy_copy_sha256,
        private_initial_calls=self._init_calls, private_policy_calls=self._private_calls,
        public_oracle_calls=self._public_oracle_calls,
        accessed_parameter_keys=sorted(self.accessed_keys),
        attempted_model_writes_rejected=True, attempted_model_creation_rejected=True,
        source_physics_renderer_resource_authenticated_by_this_helper=False)


def evaluate_episode(adapter, seed, initial_observation, transition, *,
                     check=lambda: None, oracle=None):
  """Exactly1000 steps; callback owns physics, renderer and source authority.

  transition(raw.copy()) -> observation/action_applied/reward_f64/discount_f64.
  oracle receives keyword step, carry, observation, root, after_carry, raw,
  proof (copies), at fixed0/500/999; it must not step/reset/render physics.
  All returned carry values are official model B1 trees, not public split lists.
  Failures raise and publish no partial result or padding.
  """
  private_root_key(seed, 0)
  obs = observation(initial_observation)
  require(bool(obs['is_first']) and not bool(obs['is_last']) and
      not bool(obs['is_terminal']) and obs['reward'] == 0, 'Genuine reset observation required')
  check(); adapter.assert_frozen(full=True)
  carry = adapter.initial(seed)
  observations = {k: [v.copy()] for k, v in obs.items()}
  raw_rows, applied_rows, rewards, discounts = [], [], [], []
  roots, latent_keys, actor_keys, hashes = [], [], [], [tree_digest(carry)]
  oracle_receipts = []
  for step in range(HORIZON):
    check(); adapter.assert_frozen()
    after, raw, proof = adapter.step(carry, obs, seed, step)
    require(type(raw) is np.ndarray and raw.dtype == np.float32 and raw.shape == (12,) and
        np.isfinite(raw).all(), 'Raw actor action differs')
    root = private_root_key(seed, step)
    require(proof['root_key'].dtype == root.dtype and proof['root_key'].shape == root.shape and
        proof['root_key'].tobytes() == root.tobytes() and int(proof['rng_consumption']) == 2 and
        proof['before_carry_sha256'] == hashes[-1] and
        proof['after_carry_sha256'] == tree_digest(after), 'Policy proof/carry/key mismatch')
    if oracle is not None and step in ORACLE_STEPS:
      receipt = oracle(step=step, carry=copy.deepcopy(carry),
          observation=copy.deepcopy(obs), root=root.copy(), after_carry=copy.deepcopy(after),
          raw=raw.copy(), proof=copy.deepcopy(proof))
      require(isinstance(receipt, dict), 'Oracle must return an explicit receipt')
      oracle_receipts.append(dict(step=step, receipt=receipt))
      adapter.assert_frozen(full=True)
    check()
    supplied = raw.copy()
    row = transition(supplied)
    require(supplied.tobytes() == raw.tobytes(), 'Transition callback mutated supplied raw action')
    require(isinstance(row, dict) and set(row) ==
        {'observation', 'action_applied', 'reward_f64', 'discount_f64'},
        'Exact transition callback fields required')
    next_obs = observation(row['observation'])
    applied, reward, discount = [np.asarray(row[k]) for k in
        ('action_applied', 'reward_f64', 'discount_f64')]
    require(applied.dtype == np.float64 and applied.shape == (12,) and
        np.isfinite(applied).all(), 'Actual applied control must be finite F64 action12')
    require(reward.dtype == np.float64 and reward.shape == () and np.isfinite(reward),
        'Official actual reward must be finite F64 scalar')
    require(discount.dtype == np.float64 and discount.shape == () and
        np.isfinite(discount) and discount in (0, 1), 'Official discount differs')
    require(next_obs['reward'].tobytes() == np.asarray(reward, np.float32).tobytes(),
        'Production F32 reward differs from actual F64 reward')
    final = (step == HORIZON - 1)
    require(not bool(next_obs['is_first']) and bool(next_obs['is_last']) == final and
        bool(next_obs['is_terminal']) == bool(discount == 0),
        'Unexpected reset/early-last/final-last/terminal; no padding allowed')
    for key in observations: observations[key].append(next_obs[key].copy())
    raw_rows.append(raw.copy()); applied_rows.append(applied.copy())
    rewards.append(reward.copy()); discounts.append(discount.copy())
    roots.append(root); latent_keys.append(proof['latent_key'].copy())
    actor_keys.append(proof['actor_key'].copy()); hashes.append(proof['after_carry_sha256'])
    carry, obs = after, next_obs
  check(); adapter.assert_frozen(full=True)
  arrays = dict(observations={k: np.stack(v) for k, v in observations.items()},
      raw=np.stack(raw_rows), applied=np.stack(applied_rows),
      reward_f64=np.stack(rewards), discount_f64=np.stack(discounts),
      root_keys=np.stack(roots), latent_keys=np.stack(latent_keys), actor_keys=np.stack(actor_keys))
  return dict(version=VERSION, seed=seed, **arrays, carry_hashes=hashes,
      counts=dict(transitions=HORIZON, observations=HORIZON + 1, policy_calls=HORIZON,
          resets=1, oracle_calls=len(oracle_receipts)),
      return_f64=float(np.sum(arrays['reward_f64'], dtype=np.float64)),
      arrays_sha256=tree_digest(arrays), oracle_receipts=oracle_receipts,
      model_state_sha256=adapter.initial_state_sha256,
      readonly_full_state_and_counters_unchanged=True,
      public_policy_copy_used_for_scientific_calls=False,
      source_physics_renderer_resource_authenticated_by_this_helper=False,
      independent_model_or_optimizer_replay_performed=False)
