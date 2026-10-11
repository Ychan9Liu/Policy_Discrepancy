"""DL-feedback-r1: read-only B1 production policy, one posterior omission.

No environment, source artifact, checkpoint, optimizer or GPU is constructed
by this module. The worker supplies an already frozen actual Agent and trusted
host carry. Trace-local hooks preserve actual policy/RSSM/sample operations.
"""
import contextlib
import hashlib
import threading

import numpy as np

VERSION = 'DL-feedback-r1'
PRIVATE_SEED, KEY_TAG = 20261018, 0x46445231
_HOOK_LOCK = threading.RLock()


def named_root_key(episode, position, pair, absolute_step):
  """Named root stream; branch/call order/counters are deliberately absent."""
  step = absolute_step
  values = (episode, position, pair, step)
  if (any(type(x) is not int for x in values) or not 0 <= episode < 32 or
      not 1 <= position <= 900 or not 0 <= pair < 4 or
      not position <= step < position+100 or step > 999):
    raise ValueError('Require fixed episode/position/pair/absolute-step labels')
  rng = np.random.default_rng([PRIVATE_SEED, KEY_TAG, *values])
  return rng.integers(0, np.iinfo(np.uint32).max, (2,), np.uint32)


private_root_key = named_root_key


def tree_digest(tree):
  digest = hashlib.sha256()
  def visit(node):
    if isinstance(node, dict):
      digest.update(b'dict')
      for key in sorted(node):
        digest.update(str(key).encode()); visit(node[key])
    elif isinstance(node, (list, tuple)):
      digest.update(type(node).__name__.encode()+str(len(node)).encode())
      for value in node:
        visit(value)
    else:
      value = np.ascontiguousarray(node)
      if value.dtype.hasobject or not np.isfinite(value).all():
        raise ValueError('Require finite numeric host carry leaves')
      digest.update(str(value.dtype).encode()+str(value.shape).encode())
      digest.update(value.tobytes())
  visit(tree)
  return digest.hexdigest()


def host_copy(tree):
  import jax
  with jax._src.config.explicit_device_get_scope():
    return jax.tree.map(lambda x: np.asarray(x).copy(), tree)


def counters(agent):
  return dict(updates=int(agent.n_updates), batches=int(agent.n_batches),
      actions=int(agent.n_actions), eval_actions=int(agent.n_eval_actions),
      eval_seed=int(agent.eval_seed))


@contextlib.contextmanager
def _policy_trace_receipts(model, omit):
  """Hooks exist only during JIT tracing; restore Python methods on all exits.

  The actual _observe, _dist, Normal.sample and model.policy remain the caller.
  Replacing only obslogit uses the actual same-h prior without first sampling q.
  Parameters already exist; _prior itself consumes no additional nj.seed.
  """
  import ninjax as nj
  dyn = model.dyn
  original_core, original_logit, original_seed = dyn._core, dyn._logit, nj.seed
  receipt, sampled_keys = {}, []
  def core(*args, **kwargs):
    h = original_core(*args, **kwargs)
    if 'h' in receipt:
      raise AssertionError('Expected exactly one B1 policy core transition')
    receipt['h'] = h
    return h
  def logit(name, x):
    if name == 'obslogit':
      selected = dyn._prior(receipt['h']) if omit else original_logit(name, x)
      if 'selected_logit' in receipt:
        raise AssertionError('Expected exactly one current latent distribution')
      receipt['selected_logit'] = selected
      return selected
    return original_logit(name, x)
  def seed(*args, **kwargs):
    value = original_seed(*args, **kwargs)
    # Frozen initialized policy calls sample one latent and one action only.
    # Amount/reservation changes are not silently relabeled as those two tags.
    if args or kwargs:
      raise AssertionError('Unexpected production nj.seed consumption signature')
    sampled_keys.append(value)
    return value
  with _HOOK_LOCK:
    dyn._core, dyn._logit, nj.seed = core, logit, seed
    try:
      yield receipt, sampled_keys
    finally:
      dyn._core, dyn._logit, nj.seed = original_core, original_logit, original_seed


class FeedbackPolicy:
  """Actual outer host carry -> host carry/raw/proof, without public counters."""
  def __init__(self, agent):
    import jax
    import ninjax as nj
    from embodied.jax import transform
    if (not agent.jaxcfg.enable_policy or len(agent.policy_devices) != 1 or
        agent.jaxcfg.use_shardmap or set(agent.act_space) != {'action'} or
        agent.act_space['action'].shape != (12,)):
      raise ValueError('Require actual frozen B1 action12 policy on one device')
    self.agent = agent
    self.initial_counters = counters(agent)
    self.initial_params = agent.params
    self.initial_policy_params = agent.policy_params
    pp = agent.policy_params_sharding
    pm, ps = agent.policy_mirrored, agent.policy_sharded
    self.calls = {}
    for omit in (False, True):
      def policy(carry, obs, mode, omit=omit):
        with _policy_trace_receipts(agent.model, omit) as (receipt, keys):
          result = agent.model.policy(carry, obs, mode=mode)
        if len(keys) != 2 or set(receipt) != {'h','selected_logit'}:
          raise AssertionError('Production policy sampling/core/logit receipt differs')
        after, actions, out = result
        receipt.update(latent_key=keys[0], actor_key=keys[1],
            latent=after[1]['stoch'], raw=actions['action'])
        return after, actions, out, receipt
      pure = nj.pure(policy)
      def readonly(state, carry, obs, mode, *, seed, pure=pure):
        state, out, _, modified, created = pure(state, carry, obs, mode,
            seed=seed, create=False, modify=True, track=True)
        if modified or created:
          raise AssertionError('Feedback policy attempted model state writes')
        return state, out
      readonly._is_pure = True
      self.calls[omit] = transform.apply(readonly, agent.policy_mesh,
          (pp,pm,ps,ps), (ps,ps,ps,pm), agent.partition_rules[1],
          static_argnums=(4,), use_shardmap=False)

  def _obs(self, obs):
    if set(obs) != set(self.agent.obs_space):
      raise ValueError('Require exact production observation fields')
    result = {}
    for name, space in self.agent.obs_space.items():
      value = np.asarray(obs[name])
      if value.dtype != space.dtype or not np.isfinite(value).all():
        raise ValueError(f'Observation dtype/finite contract differs: {name}')
      if value.shape == space.shape:
        value = value[None]
      if value.shape != (1,*space.shape):
        raise ValueError(f'Expected unbatched or B1 observation: {name}')
      result[name] = value.copy()
    return result

  def _carry(self, carry):
    if (type(carry) is not tuple or len(carry) != 4 or
        carry[0] != {} or carry[2] != {} or set(carry[1]) != {'deter','stoch'} or
        set(carry[3]) != {'action'}):
      raise ValueError('Require actual outer simple enc/dyn/dec/prevraw carry')
    cfg = self.agent.config.dyn.rssm
    expected = dict(deter=(cfg.deter,), stoch=(cfg.stoch,cfg.classes))
    for name, value in carry[1].items():
      if (type(value) is not list or len(value) != 1 or type(value[0]) is not np.ndarray or
          value[0].shape != expected[name] or
          str(value[0].dtype) != str(self.agent.config.jax.compute_dtype) or
          not np.isfinite(value[0]).all()):
        raise ValueError('Require B1 original numeric dtype dyn carry leaves')
    previous = carry[3]['action']
    if (type(previous) is not list or len(previous) != 1 or type(previous[0]) is not np.ndarray or
        previous[0].dtype != np.float32 or previous[0].shape != (12,) or
        not np.isfinite(previous[0]).all()):
      raise ValueError('Require actual preceding raw float32 sample')
    tree_digest(carry)
    return carry

  def step(self, hostcarry, obs, root_key, omit=False):
    import jax
    from embodied.jax import internal
    if type(omit) is not bool:
      raise ValueError('omit must explicitly select only the intended step')
    key = np.asarray(root_key)
    if key.dtype != np.uint32 or key.shape != (2,):
      raise ValueError('Require private root uint32 key[2]')
    before = counters(self.agent)
    carry = self._carry(hostcarry)
    observed = self._obs(obs)
    agent = self.agent
    if (agent.params is not self.initial_params or agent.policy_params is not self.initial_policy_params or
        agent.pending_sync is not None):
      raise AssertionError('Feedback requires frozen model and policy parameter dictionaries')
    device_carry = internal.device_put(carry, internal.local_sharding(agent.policy_mirrored))
    global_carry = internal.to_global(agent._stack(device_carry), agent.policy_sharded)
    observed = internal.device_put(observed, agent.policy_sharded)
    device_key = internal.device_put(key, agent.policy_mirrored)
    after, action, out, proof = self.calls[omit](agent.policy_params, device_key,
        global_carry, observed, 'eval')
    jax.block_until_ready((after,action,out,proof))
    finite = host_copy(out).get('finite',{})
    if not finite or not all(bool(x.all()) for x in jax.tree.leaves(finite)):
      raise AssertionError('Actual production policy finite checks failed')
    after = host_copy(agent._split(internal.to_local(after)))
    raw = host_copy(action)['action'][0]
    proof = host_copy(proof)
    proof = {name:value[0] if name not in ('latent_key','actor_key') else value
        for name,value in proof.items()}
    proof['root_key'] = key.copy()
    proof['rng_consumption'] = np.array(2,np.int32)
    if (raw.dtype != np.float32 or raw.shape != (12,) or not np.isfinite(raw).all() or
        raw.tobytes() != proof['raw'].tobytes() or tree_digest(after[3]) != tree_digest(dict(action=[raw]))):
      raise AssertionError('Actual actor raw sample/carry/receipt differs')
    if counters(agent) != before:
      raise AssertionError('Private feedback step changed production counters')
    return after, raw.copy(), {name:value.copy() for name,value in proof.items()}

  def oracle_source(self, record, sourceobs, raw_expected):
    """Public actual eval oracle plus adapter q, finally restore eval state."""
    from embodied.jax import internal
    agent = self.agent
    original = counters(agent)
    hostcarry = self._carry(record['carry'])
    if tree_digest(hostcarry) != record['carry_sha256']:
      raise AssertionError('Actual source pre-policy carry hash differs')
    expected = np.asarray(raw_expected)
    if expected.dtype != np.float32 or expected.shape != (12,):
      raise ValueError('Expected exact source raw sample')
    try:
      agent.set_eval_seed(int(record['eval_seed']))
      with agent.n_eval_actions.lock:
        agent.n_eval_actions.value = int(record['eval_counter'])
      key = np.random.default_rng([agent.config.seed,0x4556414C,
          int(record['eval_seed']),int(record['eval_counter'])]).integers(
              0,np.iinfo(np.uint32).max,(2,),np.uint32)
      carry = internal.device_put(hostcarry, internal.local_sharding(agent.policy_mirrored))
      public_after, public_action, _ = agent.policy(carry,self._obs(sourceobs),mode='eval')
      public_after, public_raw = host_copy(public_after), np.asarray(public_action['action'][0])
      adapted_after, adapted_raw, proof = self.step(hostcarry,sourceobs,key,omit=False)
      if (public_raw.dtype != expected.dtype or public_raw.tobytes() != expected.tobytes() or
          adapted_raw.tobytes() != expected.tobytes() or
          tree_digest(public_after) != record['after_carry_sha256'] or
          tree_digest(adapted_after) != record['after_carry_sha256']):
        # Small byte receipts distinguish public-source and adapter failures;
        # they do not relax exactness or expose the full model/checkpoint.
        details = dict(expected_raw_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
            public_raw_sha256=hashlib.sha256(public_raw.tobytes()).hexdigest(),
            adapter_raw_sha256=hashlib.sha256(adapted_raw.tobytes()).hexdigest(),
            expected_after_sha256=record['after_carry_sha256'],
            public_after_sha256=tree_digest(public_after),
            adapter_after_sha256=tree_digest(adapted_after))
        raise AssertionError('Feedback clean path is not byte-exact actual production source policy: '
            + str(details))
      return dict(public_raw_byte_exact=True,adapter_raw_byte_exact=True,
          public_after_carry_byte_hash_exact=True,adapter_after_carry_byte_hash_exact=True,
          eval_key=key.tolist(),latent_key=proof['latent_key'].tolist(),actor_key=proof['actor_key'].tolist())
    finally:
      agent.set_eval_seed(original['eval_seed'])
      with agent.n_eval_actions.lock:
        agent.n_eval_actions.value = original['eval_actions']
      if counters(agent) != original:
        raise AssertionError('Source oracle changed frozen production counters')
