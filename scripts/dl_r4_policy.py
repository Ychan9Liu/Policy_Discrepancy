"""DL-r4-utility-r1: frozen native policy, action/memory separation and H1.

No loader, CLI, environment, source authorization or device allocation lives
here. The owner authenticates a full N1M checkpoint and the actual physics,
rendering, resource mapping, labels and whole-entry deadline separately.

Host carries have the public policy's outer tuple/list1 format. A current()
call runs the original normal posterior policy exactly once. A separate pure
JIT derives the paired prior sample/action from its actual sampling subkeys;
it neither repeats encoding/recurrent observation nor calls nj.seed(). first()
only assembles a host carry. Its proof's rng_consumption=2 describes the
originating current() policy trace, not a second random or policy execution.
"""
import contextlib
import copy
import threading

import numpy as np

from scripts.dl_reset_features import counters, host_copy, tree_digest
from scripts.dl_reset_learning import tree_digest as full_digest

VERSION = 'DL-r4-utility-r1'
KEY_NAMESPACE = 'DL_R4_U1'
EPISODES = 32
HORIZON = 100
SOURCE_SEEDS = tuple(range(4096, 4128))
SOURCE_HORIZON = 1000
POSITION_MIN, POSITION_MAX = 64, 896
PAIRS = 16
EVAL_TAG = 0x4556414C
# Injective label packing (not a random 32-bit hash with birthday collisions).
BRANCH_SEED_BASE = 0x52345531  # ASCII R4U1, a domain distinct from source seeds.
BRANCH_SEED_COUNT = EPISODES * (POSITION_MAX - POSITION_MIN + 1) * PAIRS
OBS_KEYS = ('image', 'reward', 'is_first', 'is_last', 'is_terminal')
PROOF_KEYS = ('root_key', 'latent_key', 'actor_key', 'h', 'selected_logit',
    'latent', 'action_logit', 'action_latent', 'raw', 'rng_consumption',
    'memory_source', 'action_source')
_TRACE_LOCK = threading.RLock()


def require(condition, message):
  if not condition:
    raise ValueError(message)


def _integer(value, low, high, name):
  require(type(value) is int and low <= value <= high, 'Invalid ' + name)
  return value


def source_eval_seed(episode):
  return 4096 + _integer(episode, 0, EPISODES - 1, 'source episode')


def branch_eval_seed(episode, position, pair):
  _integer(episode, 0, EPISODES - 1, 'branch episode')
  _integer(position, POSITION_MIN, POSITION_MAX, 'branch position')
  _integer(pair, 0, PAIRS - 1, 'branch pair')
  return BRANCH_SEED_BASE + ((episode * (POSITION_MAX - POSITION_MIN + 1) +
      position - POSITION_MIN) * PAIRS + pair)


def _eval_root(eval_seed, counter):
  return np.random.default_rng([0, EVAL_TAG, eval_seed, counter]).integers(
      0, np.iinfo(np.uint32).max, (2,), np.uint32)


def source_root_key(episode, step):
  _integer(step, 0, SOURCE_HORIZON - 1, 'source step')
  return _eval_root(source_eval_seed(episode), step)


def branch_root_key(episode, position, pair, offset):
  _integer(offset, 0, HORIZON - 1, 'branch offset')
  return _eval_root(branch_eval_seed(episode, position, pair), offset)


def _root(value):
  require(type(value) is np.ndarray and value.dtype == np.uint32 and
      value.shape == (2,), 'Require explicit host uint32 root[2]')
  return value.copy()


def observation(obs):
  require(type(obs) is dict and set(obs) == set(OBS_KEYS),
      'Exact production observation fields required')
  out = {}
  for name, dtype, shape in (
      ('image', np.uint8, (64, 64, 3)), ('reward', np.float32, ()),
      ('is_first', np.bool_, ()), ('is_last', np.bool_, ()),
      ('is_terminal', np.bool_, ())):
    value = np.asarray(obs[name])
    require(value.dtype == np.dtype(dtype) and value.shape == shape and
        np.isfinite(value).all(), 'Observation dtype/shape/finite differs: ' + name)
    out[name] = value.copy()
  require(not bool(out['is_last']) and not bool(out['is_terminal']),
      'Policy cannot execute on a last/terminal observation')
  if out['is_first']:
    require(out['reward'] == 0, 'Reset reward must be zero')
  return out


@contextlib.contextmanager
def _trace(model):
  """Trace the unchanged original policy, restoring all hooks on failure."""
  import ninjax as nj
  dyn = model.dyn
  original_core, original_logit, original_seed = dyn._core, dyn._logit, nj.seed
  receipt, keys = {}, []
  def core(*args, **kwargs):
    value = original_core(*args, **kwargs)
    if 'h' in receipt:
      raise AssertionError('Expected exactly one current recurrent core')
    receipt['h'] = value
    return value
  def logit(name, value):
    result = original_logit(name, value)
    if name == 'obslogit':
      if 'qlogits' in receipt:
        raise AssertionError('Expected exactly one current posterior logit')
      receipt['qlogits'] = result
    return result
  def seed(*args, **kwargs):
    if args or kwargs:
      raise AssertionError('Unexpected native nj.seed signature')
    value = original_seed()
    keys.append(value)
    return value
  with _TRACE_LOCK:
    dyn._core, dyn._logit, nj.seed = core, logit, seed
    try:
      yield receipt, keys
    finally:
      dyn._core, dyn._logit, nj.seed = original_core, original_logit, original_seed


def _readonly(fn, accessed):
  import ninjax as nj
  pure = nj.pure(fn)
  def invoke(params, *args, root):
    _, out, read, modified, created = pure(params, *args, seed=root,
        create=False, modify=True, ignore=False, track=True)
    if modified or created:
      raise AssertionError('R4 read-only call attempted model state writes/creation')
    accessed.update(read)
    return out
  return invoke


def _bundle_digest(bundle):
  return full_digest({k: v for k, v in bundle.items() if k != 'bundle_sha256'})


class R4Policy:
  """One frozen loaded Agent; current params are the scientific authority."""
  def __init__(self, agent):
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    from embodied.jax import nets as nn
    from embodied.jax import outs
    from dreamerv3 import dt_latch, rep_probe
    require(agent.jaxcfg.enable_policy and len(agent.train_devices) ==
        len(agent.policy_devices) == 1 and not agent.jaxcfg.use_shardmap and
        agent.train_devices[0] == agent.policy_devices[0],
        'Require one shared train/policy device without shardmap')
    require(int(agent.config.seed) == 0 and
        str(agent.config.jax.compute_dtype) == 'bfloat16', 'Require seed0 BF16 Agent')
    require(set(agent.obs_space) == set(OBS_KEYS) and
        set(agent.model.enc.obs_space) == {'image'} and
        set(agent.act_space) == {'action'} and agent.act_space['action'].shape == (12,),
        'Require production vision64/action12 spaces')
    self.agent = agent
    self.initial_params = agent.params
    self.initial_policy_params = agent.policy_params
    self.initial_values = tuple((k, id(v)) for k, v in sorted(agent.params.items()))
    self.initial_policy_values = tuple((k, id(v)) for k, v in sorted(agent.policy_params.items()))
    self.initial_counters = counters(agent)
    self.initial_state_sha256 = self._state_hash()
    self.initial_policy_sha256 = self._policy_copy_hash()
    self.accessed_keys = set()
    self._issued = set()
    self._calls = dict(initial=0, normal=0, current=0, derived=0, first=0, h1=0, oracle=0)
    model = agent.model
    self._initial = jax.jit(_readonly(lambda: model.init_policy(1), self.accessed_keys))
    def native(carry, obs):
      with _trace(model) as (receipt, keys):
        after, actions, out = model.policy(carry, obs, mode='eval')
      if set(receipt) != {'h', 'qlogits'} or len(keys) != 2:
        raise AssertionError('Original native policy core/logit/two-key trace differs')
      return after, actions, out, dict(**receipt, latent_key=keys[0], actor_key=keys[1])
    self._native = jax.jit(_readonly(native, self.accessed_keys))
    def derive(h, qlogits, qsample, latent_key, actor_key):
      plogits = model.dyn._prior(h)
      psample = nn.cast(model.dyn._dist(plogits).sample(seed=latent_key))
      def actor(z):
        return model.pol(model.feat2tensor(dict(deter=h, stoch=z)), bdims=1)['action']
      qpolicy, ppolicy = actor(qsample), actor(psample)
      praw = ppolicy.sample(actor_key)
      btfeat = dict(deter=h[:, None], logit=qlogits[:, None])
      mode = dt_latch.representative_signal(model, btfeat, plogits[:, None])
      sample_policies = [model.pol(model.feat2tensor(dict(deter=h[:, None],
          stoch=z[:, None])), bdims=2) for z in (qsample, psample)]
      sample_d = rep_probe.discrepancy(*sample_policies, model.act_space)[0]
      def moments(z):
        value = actor(z)
        value = value.output if isinstance(value, outs.Agg) else value
        if not isinstance(value, outs.Normal):
          raise AssertionError('R4 requires actual Normal actor head')
        return value.mean, value.stddev
      qm, qs = moments(mode['qmode'][:, 0])
      pm, ps = moments(mode['pmode'][:, 0])
      sqm, sqs = moments(qsample)
      spm, sps = moments(psample)
      # Extra derived sampling has explicit keys and must consume no nj.seed.
      return dict(plogits=plogits, psample=psample, p_raw=praw,
          K=model.dyn._dist(qlogits).kl(model.dyn._dist(plogits)),
          D=mode['D'][:, 0], Dmode=mode['D'][:, 0], sample_D=sample_d[:, 0],
          qmode=mode['qmode'][:, 0], pmode=mode['pmode'][:, 0],
          q_mode_mean=qm, p_mode_mean=pm, q_mode_std=qs, p_mode_std=ps,
          q_sample_mean=sqm, p_sample_mean=spm, q_sample_std=sqs, p_sample_std=sps)
    def no_seed_derive(*args):
      original = nj.seed
      def forbidden(*_a, **_kw):
        raise AssertionError('Derived q/p outputs must not consume nj.seed')
      with _TRACE_LOCK:
        nj.seed = forbidden
        try:
          return derive(*args)
        finally:
          nj.seed = original
    self._derive = jax.jit(_readonly(no_seed_derive, self.accessed_keys))
    def h1(h, qm, pm, raw, reward, valid):
      q, p = dt_latch.one_step(model, h, qm, pm, {'action': raw})
      result = dt_latch.reward_support(q, p, reward, valid,
          float(model.config.dt_latch.kl_tolerance))
      result.update(raw_logits_q=q.logits, raw_logits_p=p.logits,
          target=q.target(jnp.where(valid, reward, 0)), bins=q.bins)
      return result
    def no_seed_h1(*args):
      original = nj.seed
      def forbidden(*_a, **_kw):
        raise AssertionError('H1 scoring must not consume nj.seed')
      with _TRACE_LOCK:
        nj.seed = forbidden
        try:
          return h1(*args)
        finally:
          nj.seed = original
    self._h1 = jax.jit(_readonly(no_seed_h1, self.accessed_keys))

  def _state_hash(self):
    return full_digest(dict(params=host_copy(self.agent.params), counters=counters(self.agent)))

  def _policy_copy_hash(self):
    agent = self.agent
    require(agent.pending_sync is None, 'Require no pending policy synchronization')
    current = {k: agent.params[k] for k in agent.policy_keys}
    actual = tree_digest(host_copy(agent.policy_params))
    require(actual == tree_digest(host_copy(current)),
        'Public policy copy must be typed-byte equal to current policy params')
    return actual

  def assert_frozen(self, full=True):
    agent = self.agent
    if (agent.params is not self.initial_params or
        agent.policy_params is not self.initial_policy_params or
        tuple((k, id(v)) for k, v in sorted(agent.params.items())) != self.initial_values or
        tuple((k, id(v)) for k, v in sorted(agent.policy_params.items())) != self.initial_policy_values or
        agent.pending_sync is not None or counters(agent) != self.initial_counters):
      raise AssertionError('R4 requires unchanged complete params/policycopy/counters')
    if full and (self._state_hash() != self.initial_state_sha256 or
        self._policy_copy_hash() != self.initial_policy_sha256):
      raise AssertionError('R4 full optimizer/model/normalizer/policy state changed')

  def _carry(self, carry, initial=False):
    require(type(carry) is tuple and len(carry) == 4 and carry[0] == {} and
        carry[2] == {} and type(carry[1]) is dict and
        set(carry[1]) == {'deter', 'stoch'} and type(carry[3]) is dict and
        set(carry[3]) == {'action'}, 'Require original outer public-policy carry')
    cfg = self.agent.config.dyn.rssm
    for name, shape in (('deter', (cfg.deter,)), ('stoch', (cfg.stoch, cfg.classes))):
      item = carry[1][name]
      require(type(item) is list and len(item) == 1 and type(item[0]) is np.ndarray and
          str(item[0].dtype) == 'bfloat16' and item[0].shape == shape and
          np.isfinite(item[0]).all(), 'Outer BF16 latent carry differs: ' + name)
    prev = carry[3]['action']
    require(type(prev) is list and len(prev) == 1 and type(prev[0]) is np.ndarray and
        prev[0].dtype == np.float32 and prev[0].shape == (12,) and
        np.isfinite(prev[0]).all(), 'Require actual previous raw float32 action')
    z = carry[1]['stoch'][0].astype(np.float32)
    if initial:
      require(np.all(carry[1]['deter'][0] == 0) and np.all(z == 0) and
          np.all(prev[0] == 0), 'Initial native carry must be zero')
    else:
      require(np.all((z == 0) | (z == 1)) and np.all(z.sum(-1) == 1),
          'Actual current memory must be sampled onehot')
    tree_digest(carry)
    return carry

  @staticmethod
  def _batched(carry):
    import jax
    return jax.tree.map(lambda x: np.stack(x) if isinstance(x, list) else x,
        carry, is_leaf=lambda x: isinstance(x, list))

  @staticmethod
  def _outer(carry):
    import jax
    return jax.tree.map(lambda value: [value[0].copy()], carry)

  def _execute(self, function, args, root):
    import jax
    from embodied.jax import internal
    key = _root(root)
    with self.agent.train_lock:
      self.assert_frozen(full=False)
      placed = internal.device_put((args, key), self.agent.train_mirrored)
      result = function(self.agent.params, *placed[0], root=placed[1])
      jax.block_until_ready(result)
      result = host_copy(result)
      self.assert_frozen(full=False)
    return result

  def initial(self, root_key):
    self.assert_frozen(full=True)
    result = self._execute(self._initial, (), root_key)
    carry = self._outer(result)
    self._carry(carry, initial=True)
    self.assert_frozen(full=True)
    self._calls['initial'] += 1
    return carry

  def _native_step(self, carry, obs, root_key):
    validated = observation(obs)
    self._carry(carry, initial=bool(validated['is_first']))
    before = tree_digest(carry)
    seen = tree_digest(validated)
    after, actions, out, receipt = self._execute(self._native,
        (self._batched(carry), {k: v[None] for k, v in validated.items()}), root_key)
    require(type(actions) is dict and set(actions) == {'action'}, 'Native action fields differ')
    raw = actions['action'][0].copy()
    require(raw.dtype == np.float32 and raw.shape == (12,) and np.isfinite(raw).all(),
        'Native raw sample must be finite float32 action12')
    after = self._outer(after)
    self._carry(after)
    require(after[3]['action'][0].tobytes() == raw.tobytes(), 'Native raw/after-prevact differs')
    finite = out.get('finite', {})
    require(finite and all(bool(np.asarray(x).all()) for x in finite.values()),
        'Original native policy finite check failed')
    require(tree_digest(carry) == before and tree_digest(validated) == seen,
        'Native policy mutated input carry/observation')
    return after, raw, receipt, validated, before

  def current(self, carry, obs, root_key):
    """One normal q policy plus paired p derived from the same actual subkeys."""
    key = _root(root_key)
    after, raw, receipt, validated, before = self._native_step(carry, obs, key)
    qsample = np.stack(after[1]['stoch'])
    derived = self._execute(self._derive, (receipt['h'], receipt['qlogits'],
        qsample, receipt['latent_key'], receipt['actor_key']), key)
    p_after = copy.deepcopy(after)
    p_after[1]['stoch'][0] = derived['psample'][0].copy()
    p_after[3]['action'][0] = derived['p_raw'][0].copy()
    self._carry(p_after)
    bundle = dict(version=VERSION, key_namespace=KEY_NAMESPACE,
        current_full_state_sha256=self.initial_state_sha256,
        before_carry_sha256=before, observation_sha256=tree_digest(validated),
        q_after=after, p_after=p_after, q_raw=raw, p_raw=derived.pop('p_raw')[0].copy(),
        h=receipt['h'][0].copy(), qlogits=receipt['qlogits'][0].copy(),
        qsample=qsample[0].copy(), root_key=key,
        latent_key=receipt['latent_key'].copy(), actor_key=receipt['actor_key'].copy(),
        original_nj_seed_calls=2, normal_policy_calls=1,
        derived_nj_seed_calls=0, extra_observe_calls=0)
    bundle.update({name: value[0].copy() for name, value in derived.items()})
    for name in ('D', 'K', 'sample_D'):
      require(np.asarray(bundle[name]).shape == () and np.isfinite(bundle[name]) and
          float(bundle[name]) >= 0, 'Invalid actual R4 signal: ' + name)
    require(bundle['p_raw'].dtype == np.float32 and bundle['p_raw'].shape == (12,) and
        np.isfinite(bundle['p_raw']).all(), 'Paired prior raw action differs')
    bundle['bundle_sha256'] = _bundle_digest(bundle)
    self._issued.add(bundle['bundle_sha256'])
    self._calls['current'] += 1
    self._calls['derived'] += 1
    return bundle

  def _validate_bundle(self, bundle):
    require(type(bundle) is dict and bundle.get('version') == VERSION and
        bundle.get('key_namespace') == KEY_NAMESPACE and
        bundle.get('current_full_state_sha256') == self.initial_state_sha256 and
        bundle.get('bundle_sha256') in self._issued and
        _bundle_digest(bundle) == bundle['bundle_sha256'],
        'Require an unchanged current bundle issued by this frozen adapter')
    self.assert_frozen(full=False)

  @staticmethod
  def _proof(bundle, memory, action):
    qmemory, qaction = memory == 0, action == 0
    return dict(root_key=bundle['root_key'].copy(),
        latent_key=bundle['latent_key'].copy(), actor_key=bundle['actor_key'].copy(),
        h=bundle['h'].copy(),
        selected_logit=bundle['qlogits' if qmemory else 'plogits'].copy(),
        latent=bundle['qsample' if qmemory else 'psample'].copy(),
        action_logit=bundle['qlogits' if qaction else 'plogits'].copy(),
        action_latent=bundle['qsample' if qaction else 'psample'].copy(),
        raw=bundle['q_raw' if qaction else 'p_raw'].copy(),
        rng_consumption=np.array(2, np.int32), memory_source=np.array(memory, np.int8),
        action_source=np.array(action, np.int8))

  def first(self, bundle, arm='qq'):
    """Pure host assembly: action source first, current memory source second."""
    require(type(arm) is str and arm in ('qq', 'pq', 'pp'), 'Only qq/pq/pp arms are defined')
    self._validate_bundle(bundle)
    action, memory = int(arm[0] == 'p'), int(arm[1] == 'p')
    after = copy.deepcopy(bundle['p_after' if memory else 'q_after'])
    raw = bundle['p_raw' if action else 'q_raw'].copy()
    after[3]['action'][0] = raw.copy()
    self._carry(after)
    proof = self._proof(bundle, memory, action)
    self._calls['first'] += 1
    return after, raw, proof

  def normal_step(self, carry, obs, root_key):
    """Normal posterior after the initial one-step intervention, for every arm."""
    key = _root(root_key)
    after, raw, receipt, _, _ = self._native_step(carry, obs, key)
    h, logit, latent = receipt['h'][0], receipt['qlogits'][0], after[1]['stoch'][0]
    bundle = dict(root_key=key, latent_key=receipt['latent_key'], actor_key=receipt['actor_key'],
        h=h, qlogits=logit, qsample=latent, q_raw=raw)
    proof = self._proof(bundle, 0, 0)
    self._calls['normal'] += 1
    return after, raw, proof

  def score_h1(self, bundle, recorded_raw, next_reward, valid=True):
    """Original mode H1 probe conditional on one common recorded raw action.

    next_reward is the genuine F64 reward after that action, then explicitly
    encoded F32 exactly as production obs. It is not the difference of the
    separate q-action and p-action rewards. Invalid labels return the original
    masked zero support and do not become positive evidence.
    """
    self._validate_bundle(bundle)
    require(type(valid) is bool, 'H1 valid must be an explicit bool')
    require(type(recorded_raw) is np.ndarray and recorded_raw.dtype == np.float32 and
        recorded_raw.shape == (12,) and np.isfinite(recorded_raw).all(),
        'H1 requires the actual common recorded raw float32 action12')
    reward = np.asarray(next_reward)
    require(reward.dtype == np.float64 and reward.shape == () and
        (not valid or np.isfinite(reward)), 'H1 requires actual scalar F64 next reward')
    encoded = np.asarray(reward if valid else 0, np.float32)
    result = self._execute(self._h1, (bundle['h'][None], bundle['qmode'][None],
        bundle['pmode'][None], recorded_raw[None], encoded[None],
        np.asarray([valid], np.bool_)), bundle['root_key'])
    result = {name: value.copy() if name == 'bins' else value[0].copy()
        for name, value in result.items()}
    result.update(D=bundle['D'].copy(), Dmode=bundle['Dmode'].copy(), K=bundle['K'].copy(),
        sample_D=bundle['sample_D'].copy(), valid=np.asarray(valid, np.bool_),
        recorded_raw=recorded_raw.copy(), reward_f64=reward.copy() if valid else None,
        reward_f32=encoded, source_bundle_sha256=bundle['bundle_sha256'],
        mode_probe=True, native_policy_sample_is_distinct=True,
        nj_seed_calls=0, actual_policy_calls=0)
    self._calls['h1'] += 1
    return result

  def public_oracle(self, carry, obs, eval_seed, counter, bundle=None):
    """Actual original public policy; no patched RNG or injected public output."""
    import jax
    from embodied.jax import internal
    _integer(eval_seed, 0, np.iinfo(np.uint32).max, 'oracle eval seed')
    _integer(counter, 0, SOURCE_HORIZON - 1, 'oracle counter')
    source = eval_seed in SOURCE_SEEDS
    branch = BRANCH_SEED_BASE <= eval_seed < BRANCH_SEED_BASE + BRANCH_SEED_COUNT
    require(source or branch, 'Oracle requires the declared new source/branch seed domain')
    require(source or counter < HORIZON, 'Branch oracle exceeds the declared H100 namespace')
    key = _eval_root(eval_seed, counter)
    if bundle is None:
      bundle = self.current(carry, obs, key)
    self._validate_bundle(bundle)
    require(bundle['root_key'].tobytes() == key.tobytes() and
        bundle['before_carry_sha256'] == tree_digest(carry) and
        bundle['observation_sha256'] == tree_digest(observation(obs)),
        'Public oracle root/input differs from the actual current bundle')
    self.assert_frozen(full=True)
    original = counters(self.agent)
    agent = self.agent
    try:
      agent.set_eval_seed(eval_seed)
      with agent.n_eval_actions.lock:
        agent.n_eval_actions.value = counter
      actual_key = host_copy(agent._eval_seeds(counter, agent.policy_mirrored))
      require(actual_key.tobytes() == key.tobytes(), 'Original public root derivation differs')
      placed = internal.device_put(copy.deepcopy(carry), internal.local_sharding(agent.policy_mirrored))
      actual_after, actual_action, _ = agent.policy(placed,
          {k: v[None] for k, v in observation(obs).items()}, mode='eval')
      jax.block_until_ready(actual_after)
      actual_after = host_copy(actual_after)
      raw = np.asarray(actual_action['action'][0])
      require(raw.dtype == bundle['q_raw'].dtype and raw.shape == bundle['q_raw'].shape and
          raw.tobytes() == bundle['q_raw'].tobytes(), 'Public/current q raw is not byte exact')
      require(tree_digest(actual_after) == tree_digest(bundle['q_after']),
          'Public/current q complete after carry is not byte exact')
    finally:
      agent.set_eval_seed(original['eval_seed'])
      with agent.n_eval_actions.lock:
        agent.n_eval_actions.value = original['eval_actions']
      self.assert_frozen(full=True)
    self._calls['oracle'] += 1
    return dict(public_original_policy_called=True, public_q_raw_byte_exact=True,
        public_q_full_after_carry_byte_exact=True, original_public_root_byte_exact=True,
        eval_counters_restored=True, full_state_and_policycopy_unchanged=True,
        root_key=key.copy(), eval_seed=eval_seed, eval_counter=counter,
        public_policy_calls=1, source_physics_or_renderer_authenticated=False)

  def receipt(self):
    self.assert_frozen(full=True)
    return dict(version=VERSION, key_namespace=KEY_NAMESPACE, calls=dict(self._calls),
        current_full_state_sha256=self.initial_state_sha256,
        synchronized_policycopy_sha256=self.initial_policy_sha256,
        scientific_reads_current_params=True, full_state_and_counters_unchanged=True,
        public_rng_not_consumed_by_private_calls=True,
        current_original_nj_seed_calls=2, derived_nj_seed_calls=0, h1_nj_seed_calls=0,
        first_is_host_assembly_no_policy_or_random_call=True,
        accessed_parameter_keys=sorted(self.accessed_keys),
        source_checkpoint_physics_renderer_resource_authenticated=False,
        independent_optimizer_replay_performed=False, method_effectiveness_established=False)
