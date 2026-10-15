"""Frozen normal-history features; no source loader, runner or GPU authority.

The caller authenticates RGB/raw histories and source keys. This adapter owns
its newly reset posterior carry. Each step checks the unchanged public policy
on that same conditional input BEFORE replacing only its outgoing previous
action with the supplied source raw. This is not a native closed-loop rollout.
No checkpoint, source file, environment, label or readout is opened here.
"""
import contextlib
import copy
import threading

import numpy as np

from scripts.dl_reset_features import counters, host_copy
from scripts.dl_reset_learning import tree_digest

VERSION = 'DL-normal-features-r1'
KEY_NAMESPACE = 'DL_NORMAL_FEATURES_U1'
EVAL_TAG = 0x4556414C
OBS_KEYS = ('image', 'reward', 'is_first', 'is_last', 'is_terminal')
_TRACE_LOCK = threading.RLock()


def require(value, message):
  if not value:
    raise ValueError(message)


def policy_root_key(eval_seed, counter):
  """Original public EVAL formula; namespace is a receipt, not an extra key."""
  require(type(eval_seed) is int and 0 <= eval_seed < 2**32 and
      type(counter) is int and counter >= 0, 'Explicit source eval seed/counter required')
  return np.random.default_rng([0, EVAL_TAG, eval_seed, counter]).integers(
      0, np.iinfo(np.uint32).max, (2,), np.uint32)


def observation(value):
  require(type(value) is dict and set(value) == set(OBS_KEYS), 'Complete observation required')
  result = {}
  for name, dtype, shape in (('image', np.uint8, (64,64,3)), ('reward', np.float32, ()),
      ('is_first', np.bool_, ()), ('is_last', np.bool_, ()), ('is_terminal', np.bool_, ())):
    item = np.asarray(value[name])
    require(item.dtype == dtype and item.shape == shape and np.isfinite(item).all(),
        'Observation dtype/shape/finite differs: '+name)
    result[name] = item.copy()
  require(not result['is_last'] and not result['is_terminal'], 'No policy call on terminal/last frame')
  require(not result['is_first'] or result['reward'] == 0, 'Reset observation reward must be zero')
  return result


def raw_action(value):
  require(type(value) is np.ndarray and value.dtype == np.float32 and value.shape == (12,) and
      np.isfinite(value).all(), 'Finite original raw float32 action12 required')
  return value.copy()


@contextlib.contextmanager
def _trace(model):
  """Observe the original policy graph; restore hooks even on failed tracing."""
  import ninjax as nj
  dyn = model.dyn
  core, logit, seed = dyn._core, dyn._logit, nj.seed
  found, keys = {}, []
  def traced_core(*args, **kwargs):
    result = core(*args, **kwargs)
    require('h' not in found, 'Exactly one native recurrent core required')
    found['h'] = result
    return result
  def traced_logit(name, value):
    result = logit(name, value)
    if name == 'obslogit':
      require('qlogits' not in found, 'Exactly one native posterior logit required')
      found['qlogits'] = result
    return result
  def traced_seed(*args, **kwargs):
    require(not args and not kwargs, 'Unexpected native seed signature')
    result = seed(); keys.append(result)
    return result
  with _TRACE_LOCK:
    dyn._core, dyn._logit, nj.seed = traced_core, traced_logit, traced_seed
    try:
      yield found, keys
    finally:
      dyn._core, dyn._logit, nj.seed = core, logit, seed


def _readonly(function, accessed):
  import ninjax as nj
  pure = nj.pure(function)
  def invoke(params, *args, root):
    _, result, read, modified, created = pure(params, *args, seed=root,
        create=False, modify=True, ignore=False, track=True)
    require(not modified and not created, 'Feature query attempted state writes/creation')
    accessed.update(read)
    return result
  return invoke


class FrozenNormalFeatures:
  """Single-threaded conditional-history adapter for one frozen loaded Agent.

  fixture=True explicitly permits tiny geometry; neither setting authenticates
  N1M weights or a source. No caller-supplied latent/carry is accepted.
  Public oracle checks run at every step in this narrow engineering interface.
  """
  def __init__(self, agent, *, fixture=False):
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    require(type(fixture) is bool, 'Explicit fixture flag required')
    require(agent.jaxcfg.enable_policy and len(agent.train_devices) == len(agent.policy_devices) == 1 and
        agent.train_devices[0] == agent.policy_devices[0] and not agent.jaxcfg.use_shardmap,
        'One shared train/policy device without shardmap required')
    require(agent.config.seed == 0 and agent.config.jax.compute_dtype == 'bfloat16', 'Seed0 BF16 Agent required')
    require(set(agent.obs_space) == set(OBS_KEYS) and set(agent.model.enc.obs_space) == {'image'} and
        set(agent.act_space) == {'action'} and agent.act_space['action'].shape == (12,),
        'Actual image64/action12 interface required')
    cfg = agent.model.dyn
    self.shape = (cfg.deter, cfg.stoch, cfg.classes)
    require(fixture or self.shape == (4096,32,32), 'Production feature geometry must be4096/32/32')
    require(float(cfg.unimix) == .01, 'Official unimix .01 required')
    self.agent, self.fixture = agent, fixture
    self._model = agent.model
    self._configuration = tree_digest(dict(agent.model.config.flat))
    self._params, self._policy_params = agent.params, agent.policy_params
    self._identities = tuple((k,id(v)) for k,v in sorted(agent.params.items()))
    self._policy_identities = tuple((k,id(v)) for k,v in sorted(agent.policy_params.items()))
    self._counters = counters(agent)
    self._state = tree_digest(agent.save())
    self._policy = self._policy_hash()
    self._carry = self._carry_hash = self._episode = None
    self._resets = self._steps = 0
    self.accessed_keys = set()
    def native(carry, obs):
      with _trace(agent.model) as (found, keys):
        after, actions, out = agent.model.policy(carry, obs, mode='eval')
      require(set(found) == {'h','qlogits'} and len(keys) == 2, 'Original native policy/two-key trace differs')
      return after, actions, out, dict(found, latent_key=keys[0], actor_key=keys[1])
    self._native = jax.jit(_readonly(native, self.accessed_keys))
    def query(h, qlogits):
      plogits = agent.model.dyn._prior(h)
      # Exactly the official categorical distribution after unimix. Do not
      # replace this with mode, sampled onehot, or softmax of raw logits.
      probability = lambda logits: jax.nn.softmax(agent.model.dyn._dist(logits).output.dist.logits, -1)
      return dict(q_prob=probability(qlogits), p_prob=probability(plogits), plogits=plogits)
    def no_seed_query(*args):
      original = nj.seed
      def forbidden(*_args, **_kwargs):
        raise ValueError('Probability query must consume no nj.seed')
      with _TRACE_LOCK:
        nj.seed = forbidden
        try:
          return query(*args)
        finally:
          nj.seed = original
    self._query = jax.jit(_readonly(no_seed_query, self.accessed_keys))
    self.fence(full=True)

  def _policy_hash(self):
    require(all(getattr(self.agent,k) is None for k in ('pending_outs','pending_mets','pending_sync')),
        'No pending train outputs/policy synchronization allowed')
    actual = tree_digest(host_copy(self.agent.policy_params))
    require(actual == tree_digest(host_copy({k:self.agent.params[k] for k in self.agent.policy_keys})),
        'Policy copy must equal current model params')
    return actual

  def fence(self, *, full=False):
    a = self.agent
    require(a.model is self._model and tree_digest(dict(a.model.config.flat)) == self._configuration,
        'Frozen model/configuration changed')
    require(a.params is self._params and a.policy_params is self._policy_params and
        tuple((k,id(v)) for k,v in sorted(a.params.items())) == self._identities and
        tuple((k,id(v)) for k,v in sorted(a.policy_params.items())) == self._policy_identities and
        all(not v.is_deleted() for v in (*a.params.values(),*a.policy_params.values())),
        'Frozen model/policy buffers changed or were donated')
    require(counters(a) == self._counters and
        all(getattr(a,k) is None for k in ('pending_outs','pending_mets','pending_sync')),
        'Frozen counters/pending state changed')
    if full:
      require(tree_digest(a.save()) == self._state and self._policy_hash() == self._policy,
          'Complete original state/policy copy changed')

  @staticmethod
  def _batched(carry):
    import jax
    return jax.tree.map(lambda x:np.stack(x) if isinstance(x,list) else x,
        carry,is_leaf=lambda x:isinstance(x,list))

  @staticmethod
  def _outer(carry):
    import jax
    return jax.tree.map(lambda value:[value[0].copy()],carry)

  def _execute(self, function, args, key):
    import jax
    from embodied.jax import internal
    self.fence()
    with self.agent.train_lock:
      placed = internal.device_put((args,key),self.agent.train_mirrored)
      result = function(self.agent.params,*placed[0],root=placed[1])
      jax.block_until_ready(result)
      result = host_copy(result)
    self.fence()
    return result

  def reset(self, obs, *, eval_seed, source_id):
    """Initialize from true reset, with no old latent/carry input path."""
    require(self._episode is None, 'Finish the preceding conditional history before reset')
    obs = observation(obs)
    require(bool(obs['is_first']), 'True reset observation required')
    require(type(source_id) is str and source_id, 'Caller source identity required, not authenticated here')
    key = policy_root_key(eval_seed,0)
    self.fence(full=True)
    a, original = self.agent, counters(self.agent)
    try:
      a.set_eval_seed(eval_seed)
      with a.n_eval_actions.lock:a.n_eval_actions.value = 0
      carry = host_copy(a.init_policy(1,mode='eval'))
      require(counters(a) == dict(original,eval_seed=eval_seed,eval_actions=0),
          'Public initialization changed counters')
    finally:
      a.set_eval_seed(original['eval_seed'])
      with a.n_eval_actions.lock:a.n_eval_actions.value = original['eval_actions']
      self.fence(full=True)
    require(not any(np.any(v) for v in (carry[1]['deter'][0],carry[1]['stoch'][0],carry[3]['action'][0])),
        'True public initial carry must be zero')
    self._carry, self._carry_hash = carry, tree_digest(carry)
    self._episode = dict(source_id=source_id,eval_seed=eval_seed,next_counter=0,
        reset_observation_sha256=tree_digest(obs),history_sha256=tree_digest(dict(obs=obs,root=key)))
    self._resets += 1
    return dict(version=VERSION,key_namespace=KEY_NAMESPACE,root_key=key,
        initial_carry_sha256=self._carry_hash,source_authenticated=False)

  def _public_oracle(self, carry, obs, counter, expected_after, expected_raw, key):
    """Compare genuine native outputs, before any source-raw host override."""
    import jax
    from embodied.jax import internal
    a, original = self.agent, counters(self.agent)
    before = tree_digest((carry,obs))
    try:
      a.set_eval_seed(self._episode['eval_seed'])
      with a.n_eval_actions.lock:a.n_eval_actions.value = counter
      actual_key = host_copy(a._eval_seeds(counter,a.policy_mirrored))
      require(actual_key.tobytes() == key.tobytes(), 'Original public EVAL key differs')
      placed = internal.device_put(copy.deepcopy(carry),internal.local_sharding(a.policy_mirrored))
      after, action, _ = a.policy(placed,{k:v[None] for k,v in obs.items()},mode='eval')
      jax.block_until_ready(after);after = host_copy(after)
      raw = np.asarray(action['action'][0])
      require(raw.dtype == expected_raw.dtype and raw.shape == expected_raw.shape and
          raw.tobytes() == expected_raw.tobytes() and tree_digest(after) == tree_digest(expected_after),
          'Public/native raw or full after-carry is not byte exact')
      require(counters(a) == dict(original,eval_seed=self._episode['eval_seed'],eval_actions=counter+1),
          'Public oracle must consume exactly one eval counter')
    finally:
      a.set_eval_seed(original['eval_seed'])
      with a.n_eval_actions.lock:a.n_eval_actions.value = original['eval_actions']
      self.fence()
    require(tree_digest((carry,obs)) == before, 'Public oracle changed history inputs')
    return dict(public_policy_calls=1,public_key_byte_exact=True,native_raw_byte_exact=True,
        native_full_after_carry_byte_exact=True,counters_restored=True,before_host_override=True)

  def step(self, obs, source_raw, counter, *, source_prev_raw):
    """One q step on the common external history; public oracle then override."""
    require(self._episode is not None, 'Reset a new conditional history first')
    require(type(counter) is int and counter == self._episode['next_counter'], 'Contiguous source counter required')
    self.fence()
    require(tree_digest(self._carry) == self._carry_hash, 'Own posterior history carry changed')
    obs, source_raw, source_prev_raw = observation(obs), raw_action(source_raw), raw_action(source_prev_raw)
    require(bool(obs['is_first']) == (counter == 0), 'Exactly the first source frame must be reset')
    require(counter != 0 or tree_digest(obs) == self._episode['reset_observation_sha256'], 'Reset frame changed')
    require(self._carry[3]['action'][0].tobytes() == source_prev_raw.tobytes(), 'Previous source raw/history differs')
    before = copy.deepcopy(self._carry);before_sha = tree_digest(before)
    key = policy_root_key(self._episode['eval_seed'],counter)
    after, actions, outs, trace = self._execute(self._native,
        (self._batched(before),{k:v[None] for k,v in obs.items()}),key)
    native_after = self._outer(after);native_raw = raw_action(actions['action'][0])
    require(outs.get('finite') and all(bool(np.asarray(x).all()) for x in outs['finite'].values()),
        'Original policy finite checks failed')
    require(native_after[3]['action'][0].tobytes() == native_raw.tobytes(), 'Native carry/raw mismatch')
    oracle = self._public_oracle(before,obs,counter,native_after,native_raw,key)
    features = self._execute(self._query,(trace['h'],trace['qlogits']),key)
    h, qlogits = trace['h'][0], trace['qlogits'][0]
    for name in ('q_prob','p_prob'):
      value = features[name][0]
      require(value.dtype == np.float32 and value.shape == self.shape[1:] and
          np.isfinite(value).all() and np.all((value >= 0) & (value <= 1)) and
          np.all(np.abs(value.sum(-1)-1) <= 1e-6), 'Official probability dtype/shape/range differs')
    require(h.shape == (self.shape[0],) and str(h.dtype) == 'bfloat16' and
        qlogits.shape == self.shape[1:] and str(qlogits.dtype) == 'bfloat16', 'Actual BF16 h/q geometry differs')
    require(h.tobytes() == native_after[1]['deter'][0].tobytes(), 'Feature h is not the native posterior carry h')
    history_after = copy.deepcopy(native_after)
    history_after[3]['action'][0] = source_raw.copy()
    require(tree_digest(history_after[:3]) == tree_digest(native_after[:3]), 'Host override changed posterior memory')
    require(tree_digest(self._carry) == before_sha, 'Step changed incoming history')
    self._carry, self._carry_hash = history_after, tree_digest(history_after)
    self._episode['next_counter'] += 1;self._steps += 1
    self._episode['history_sha256'] = tree_digest(dict(previous=self._episode['history_sha256'],
        counter=counter,obs=obs,source_raw=source_raw,source_prev_raw=source_prev_raw,root_key=key))
    return dict(version=VERSION,key_namespace=KEY_NAMESPACE,source_id=self._episode['source_id'],
        eval_seed=self._episode['eval_seed'],counter=counter,root_key=key,
        latent_key=trace['latent_key'],actor_key=trace['actor_key'],original_nj_seed_calls=2,
        probability_nj_seed_calls=0,h=h.copy(),qlogits=qlogits.copy(),
        plogits=features['plogits'][0].copy(),q_prob=features['q_prob'][0].copy(),
        p_prob=features['p_prob'][0].copy(),native_raw=native_raw,native_after=native_after,
        history_after=copy.deepcopy(history_after),source_raw=source_raw,source_prev_raw=source_prev_raw,
        observation_sha256=tree_digest(obs),before_carry_sha256=before_sha,
        history_after_sha256=self._carry_hash,source_history_sha256=self._episode['history_sha256'],
        public_oracle=oracle,source_authenticated=False,real_N1M_authenticated=False,
        conditional_common_history=True,native_closed_loop_trajectory=False,fixture_geometry=self.fixture)

  def finish(self):
    require(self._episode is not None and self._episode['next_counter'] > 0, 'Nonempty conditional history required')
    require(tree_digest(self._carry) == self._carry_hash, 'Own history changed before final seal')
    self.fence(full=True)
    result = dict(version=VERSION,key_namespace=KEY_NAMESPACE,episode=copy.deepcopy(self._episode),
        final_carry_sha256=self._carry_hash,full_state_sha256=self._state,policy_copy_sha256=self._policy,
        counters=dict(self._counters),full_state_policy_copy_counters_unchanged=True,
        resets=self._resets,steps=self._steps,accessed_keys=sorted(self.accessed_keys),
        source_authenticated=False,real_N1M_authenticated=False,conditional_common_history=True,
        native_closed_loop_trajectory=False,fixture_geometry=self.fixture)
    self._carry = self._carry_hash = self._episode = None
    return result
