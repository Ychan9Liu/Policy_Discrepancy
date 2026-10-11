"""Current-model, learned-only reset features; no source/checkpoint/device loader.

The caller supplies authenticated cached production observations and a loaded
Agent. Reads always use agent.params, including after normal B/R updates; the
possibly stale public policy copy is never consulted. These features are a
finite diagnostic, not an approved continuous Dreamer deployment signal.
"""
import hashlib

import numpy as np

VERSION = 'DL-reset-shared-readout-r1'
IMPLEMENTATION = 'DL-reset-features-r1'
SEED, KEY_TAG, SAMPLES, SOURCE_FRAMES = 20261026, 0x52535231, 4, 1001
OBS_KEYS = ('image', 'reward', 'is_first', 'is_last', 'is_terminal')


def require(condition, message):
  if not condition:
    raise ValueError(message)


def private_root_key(sample_index):
  """Four common roots: no episode, model, branch or public RNG state."""
  require(type(sample_index) is int and 0 <= sample_index < SAMPLES,
      'Require fixed sample index 0..3')
  return np.random.default_rng([SEED, KEY_TAG, sample_index]).integers(
      0, np.iinfo(np.uint32).max, (2,), np.uint32)


def private_roots():
  return np.stack([private_root_key(k) for k in range(SAMPLES)])


def host_copy(tree):
  import jax
  with jax._src.config.explicit_device_get_scope():
    return jax.tree.map(lambda x: np.asarray(x).copy(), tree)


def counters(agent):
  return dict(updates=int(agent.n_updates), batches=int(agent.n_batches),
      actions=int(agent.n_actions), eval_actions=int(agent.n_eval_actions),
      eval_seed=int(agent.eval_seed))


def tree_digest(tree):
  """Typed numeric tree bytes, including full optimizer/normalizer parameters."""
  digest = hashlib.sha256()
  def visit(node):
    if isinstance(node, dict):
      digest.update(b'dict')
      for key in sorted(node):
        require(type(key) is str, 'Tree dictionary keys must be strings')
        encoded = key.encode()
        digest.update(str(len(encoded)).encode()+b':'+encoded)
        visit(node[key])
    elif isinstance(node, (tuple, list)):
      digest.update(type(node).__name__.encode()+str(len(node)).encode()+b':')
      for value in node:
        visit(value)
    else:
      value = np.asarray(node)
      require(not value.dtype.hasobject and value.dtype.kind not in 'SU' and
          np.isfinite(value).all(), 'Require finite numeric tree leaves')
      digest.update(str(value.dtype).encode()+b':'+str(value.shape).encode()+b':')
      digest.update(np.ascontiguousarray(value).tobytes())
  visit(tree)
  return digest.hexdigest()


def _same(a, b):
  return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _tracked(fn, accessed):
  import ninjax as nj
  pure = nj.pure(fn)
  def invoke(params, *args, root):
    _, out, read, modified, created = pure(params, *args, seed=root,
        create=False, modify=True, ignore=False, track=True)
    if modified or created:
      raise AssertionError('Reset feature/annotation attempted model state writes')
    accessed.update(read)
    return out
  return invoke


def readonly_reset_features(model, accessed=None):
  """Actual single observation/sample; mode D is the original BT probe."""
  import jax
  import jax.numpy as jnp
  import ninjax as nj
  from embodied.jax import nets as nn
  from embodied.jax import outs
  from dreamerv3 import dt_latch, rep_probe
  accessed = set() if accessed is None else accessed
  def step(obs):
    ctx = nj.context()
    root = ctx.seed
    enc, dyn, dec, previous = model.init_policy(1)
    reset = obs['is_first']
    enc, _, tokens = model.enc(enc, obs, reset, training=False, single=True)
    _, _, feat = model.dyn.observe(dyn, tokens, previous, reset,
        training=False, single=True)
    prior = model.dyn._prior(feat['deter'])
    if len(ctx.reserve) != 14:
      raise AssertionError('Reset observation must consume exactly one nj.seed()')
    split = jax.random.split(root, 16)
    valid = jnp.all(ctx.seed == split[0]) & jnp.all(jnp.stack(ctx.reserve) == split[2:])
    # q was sampled by actual _observe. p uses the same actual private sample
    # key and official unimix distribution, without a second Ninjax seed call.
    pz = nn.cast(model.dyn._dist(prior).sample(seed=split[1]))
    raw_kl = model.dyn._dist(feat['logit']).kl(model.dyn._dist(prior))
    btfeat = dict(deter=feat['deter'][:, None], logit=feat['logit'][:, None])
    signal = dt_latch.representative_signal(model, btfeat, prior[:, None])
    def actor(z):
      value = model.pol(model.feat2tensor(dict(deter=btfeat['deter'], stoch=z)), 2)['action']
      value = value.output if isinstance(value, outs.Agg) else value
      if not isinstance(value, outs.Normal):
        raise ValueError('Reset D requires actual bounded_normal action head')
      return value
    qm, pm = actor(signal['qmode']), actor(signal['pmode'])
    qs = model.pol(model.feat2tensor(dict(deter=btfeat['deter'],
        stoch=feat['stoch'][:, None])), 2)
    ps = model.pol(model.feat2tensor(dict(deter=btfeat['deter'], stoch=pz[:, None])), 2)
    sample_d = rep_probe.discrepancy(qs, ps, model.act_space)[0]
    return dict(h=feat['deter'][0], zq=feat['stoch'][0], zp=pz[0],
        post_logit=feat['logit'][0], prior_logit=prior[0], D=signal['D'][0, 0],
        K=raw_kl[0], qmode=signal['qmode'][0, 0], pmode=signal['pmode'][0, 0],
        qmean=qm.mean[0, 0], pmean=pm.mean[0, 0], qstd=qm.stddev[0, 0],
        pstd=pm.stddev[0, 0], sample_D=sample_d[0, 0],
        sample_key=split[1], root_key=root, rng_valid=valid,
        rng_consumption=jnp.int32(1))
  invoke = _tracked(step, accessed)
  return lambda params, obs, root: invoke(params, obs, root=root)


def readonly_annotations(model, accessed=None):
  """Official batched encoder and RSSM sequence observe, with given raw acts.

  This is a new private annotation, not a replay of source policy samples or
  stored source carries. No actor call or environment action is executed.
  """
  accessed = set() if accessed is None else accessed
  def sequence(obs, previous):
    import ninjax as nj
    batch = obs['is_first'].shape[0]
    enc, dyn, dec, _ = model.init_policy(batch)
    root = nj.context().seed
    enc, enc_entry, tokens = model.enc(enc, obs, obs['is_first'],
        training=False, single=False)
    dyn, dyn_entry, _ = model.dyn.observe(dyn, tokens, previous,
        obs['is_first'], training=False, single=False)
    # Current simple encoder/decoder have empty carry and entry dictionaries.
    # Reject a changed model instead of inventing decoder annotations.
    if dec or model.dec.entry_space:
      raise ValueError('Terminal annotation requires the current empty decoder entries')
    return dict(entries=dict(enc=enc_entry, dyn=dyn_entry, dec={}),
        carry=(enc, dyn, dec), root_key=root)
  invoke = _tracked(sequence, accessed)
  return lambda params, obs, previous, root: invoke(params, obs, previous, root=root)


def annotation_inputs(episodes):
  """Validate actual full episodes and shift raw actions including raw999."""
  require(isinstance(episodes, (list, tuple)) and len(episodes) > 0,
      'Require a positive sequence of complete source episodes')
  fields = {key: [] for key in OBS_KEYS}
  actions = []
  for episode in episodes:
    require(isinstance(episode, dict), 'Require episode array dictionaries')
    for key in (*OBS_KEYS, 'action', 'action_executed'):
      require(key in episode, 'Missing actual source episode field: '+key)
    for key in OBS_KEYS:
      value = np.asarray(episode[key])
      shape = (SOURCE_FRAMES, 64, 64, 3) if key == 'image' else (SOURCE_FRAMES,)
      dtype = np.uint8 if key == 'image' else np.float32 if key == 'reward' else np.bool_
      require(value.shape == shape and value.dtype == np.dtype(dtype) and
          np.isfinite(value).all(), 'Invalid full episode '+key)
      fields[key].append(value.copy())
    first, last, terminal = [fields[key][-1] for key in OBS_KEYS[2:]]
    require(first[0] and not first[1:].any() and last[-1] and not last[:-1].any() and
        not terminal.any(), 'Require actual normal reset/full time-limit episode')
    raw = np.asarray(episode['action'])
    executed = np.asarray(episode['action_executed'])
    require(raw.dtype == np.float32 and raw.shape == (SOURCE_FRAMES, 12) and
        np.isfinite(raw).all() and executed.dtype == np.bool_ and
        executed.shape == (SOURCE_FRAMES,) and executed[:-1].all() and not executed[-1] and
        np.all(raw[-1] == 0), 'Invalid actual raw/terminal placeholder contract')
    actions.append(raw.copy())
  obs = {key: np.stack(values) for key, values in fields.items()}
  raw = np.stack(actions)
  previous = np.concatenate([np.zeros((len(episodes), 1, 12), np.float32), raw[:, :-1]], 1)
  return obs, dict(action=previous)


class ResetFeatureAdapter:
  """A frozen current params view; a new instance is needed after an update."""
  def __init__(self, agent, common_reference=None):
    import jax
    require(len(agent.train_devices) == 1 and not agent.jaxcfg.use_shardmap,
        'Require one current-model train device without shardmap')
    require(str(agent.config.jax.compute_dtype) == 'bfloat16', 'Require actual BF16 model')
    require(set(agent.obs_space) == set(OBS_KEYS) and
        set(agent.model.enc.obs_space) == {'image'} and set(agent.act_space) == {'action'} and
        agent.act_space['action'].shape == (12,), 'Require production RGB-only/action12 spaces')
    self.agent = agent
    self.initial_params = agent.params
    self.initial_values = tuple((k, id(v)) for k, v in sorted(agent.params.items()))
    self.initial_counters = counters(agent)
    self.initial_state_sha256 = self._state_hash()
    self.accessed_keys, self.annotation_accessed_keys = set(), set()
    self._call = jax.jit(readonly_reset_features(agent.model, self.accessed_keys))
    self._annotate = jax.jit(readonly_annotations(agent.model, self.annotation_accessed_keys))
    self.reference, self.verified_observations = None, 0
    if common_reference is not None:
      self._import_reference(common_reference)

  def _state_hash(self):
    return tree_digest(dict(params=host_copy(self.agent.params), counters=counters(self.agent)))

  def assert_frozen(self, full=False):
    if (self.agent.params is not self.initial_params or
        tuple((k, id(v)) for k, v in sorted(self.agent.params.items())) != self.initial_values or
        counters(self.agent) != self.initial_counters):
      raise AssertionError('Reset adapter requires unchanged current params/counters')
    if full and self._state_hash() != self.initial_state_sha256:
      raise AssertionError('Reset adapter full parameter/optimizer/normalizer state changed')

  def _import_reference(self, reference):
    cfg = self.agent.config.dyn.rssm
    require(isinstance(reference, dict) and set(reference) ==
        {'h', 'prior_logit', 'zp', 'roots', 'model_state_sha256', 'verified_observations'},
        'Invalid common reference fields')
    require(reference['model_state_sha256'] == self.initial_state_sha256 and
        type(reference['verified_observations']) is int and reference['verified_observations'] > 0,
        'Common reference must belong to this frozen complete model state')
    shapes = dict(h=(cfg.deter,), prior_logit=(cfg.stoch, cfg.classes),
        zp=(SAMPLES, cfg.stoch, cfg.classes), roots=(SAMPLES, 2))
    for key, shape in shapes.items():
      value = reference[key]
      dtype = 'uint32' if key == 'roots' else 'bfloat16'
      require(type(value) is np.ndarray and value.shape == shape and
          str(value.dtype) == dtype and np.isfinite(value).all(), 'Invalid common reference '+key)
    require(_same(reference['roots'], private_roots()), 'Common root reference differs')
    z = reference['zp'].astype(np.float64)
    require(np.logical_or(z == 0, z == 1).all() and np.all(z.sum(-1) == 1),
        'Common prior samples must be official categorical onehot')
    self.reference = {key: reference[key].copy() for key in shapes}
    self.verified_observations = reference['verified_observations']

  def _observation(self, obs):
    require(isinstance(obs, dict) and set(obs) == set(OBS_KEYS), 'Require exact production observation')
    result = {}
    for key, space in self.agent.obs_space.items():
      value = np.asarray(obs[key])
      require(value.shape == space.shape and value.dtype == space.dtype and np.isfinite(value).all(),
          'Invalid reset observation '+key)
      result[key] = value[None].copy()
    require(bool(result['is_first'][0]) and not bool(result['is_last'][0]) and
        not bool(result['is_terminal'][0]) and result['reward'][0] == 0,
        'Require real first reset observation with zero reward')
    return result

  def extract(self, observations, check=lambda: None):
    """Return h[E,H], zq/zp[E,4,S,C], mode D/raw K[E], and typed receipts.

    Reuse this adapter across cal/blind, or import its authenticated common
    reference into a fresh adapter. Source authentication and once-only blind
    authorization belong to the caller, not this array-only module.
    """
    import jax
    from embodied.jax import internal
    require(isinstance(observations, (tuple, list)) and len(observations) > 0,
        'Require a positive sequence of reset observations')
    prepared = [self._observation(obs) for obs in observations]
    input_sha = tree_digest(prepared)
    roots = private_roots()
    records = []
    with self.agent.train_lock:
      check(); self.assert_frozen(full=True)
      before = self.initial_state_sha256
      for obs in prepared:
        draws = []
        for root in roots:
          check(); self.assert_frozen()
          placed = internal.device_put((obs, root), self.agent.train_mirrored)
          out = self._call(self.agent.params, *placed)
          jax.block_until_ready(out)
          draws.append(host_copy(out))
        cfg = self.agent.config.dyn.rssm
        for value in draws:
          require(all(np.isfinite(x).all() for x in value.values()) and
              bool(value['rng_valid']) and int(value['rng_consumption']) == 1,
              'Reset feature finite/RNG contract differs')
          for key, shape in dict(h=(cfg.deter,), post_logit=(cfg.stoch, cfg.classes),
              prior_logit=(cfg.stoch, cfg.classes), zq=(cfg.stoch, cfg.classes),
              zp=(cfg.stoch, cfg.classes)).items():
            require(str(value[key].dtype) == 'bfloat16' and value[key].shape == shape,
                'Actual reset BF16 shape/dtype differs: '+key)
          for key in ('zq', 'zp'):
            z = value[key].astype(np.float64)
            require(np.logical_or(z == 0, z == 1).all() and np.all(z.sum(-1) == 1),
                'Actual reset latent is not sampled categorical onehot')
          require(value['D'].dtype == np.float32 and value['K'].dtype == np.float32 and
              value['D'] >= 0 and value['K'] >= -1e-5, 'Invalid original D/raw K')
        deterministic = ('h', 'post_logit', 'prior_logit', 'D', 'K', 'qmode', 'pmode',
            'qmean', 'pmean', 'qstd', 'pstd')
        require(all(_same(draws[0][key], other[key]) for other in draws[1:] for key in deterministic),
            'Reset h/logits/mode signal changed with private sample root')
        record = {key: np.stack([value[key] for value in draws]) for key in draws[0]}
        require(_same(record['root_key'], roots), 'Actual root receipt differs')
        reference = dict(h=record['h'][0], prior_logit=record['prior_logit'][0],
            zp=record['zp'], roots=roots)
        if self.reference is None:
          self.reference = {key: value.copy() for key, value in reference.items()}
        require(all(_same(reference[key], self.reference[key]) for key in reference),
            'Reset common h/prior/per-key prior sample bytes differ across observations')
        records.append(record)
      check(); self.assert_frozen(full=True)
      require(tree_digest(prepared) == input_sha, 'Reset observation bytes changed')
    combined = {key: np.stack([row[key] for row in records]) for key in records[0]}
    self.verified_observations += len(records)
    reference = dict(**{key: value.copy() for key, value in self.reference.items()},
        model_state_sha256=before, verified_observations=self.verified_observations)
    aux = dict(**combined, common_reference=reference, counters_before=dict(self.initial_counters),
        counters_after=counters(self.agent), full_state_before_sha256=before,
        full_state_after_sha256=before, input_sha256=input_sha,
        accessed_parameter_keys=sorted(self.accessed_keys),
        common_h_byte_exact=True, common_prior_byte_exact=True, common_zp_per_key_byte_exact=True,
        sample_roots_have_episode_model_branch_id=False, source_q_sample_exact_claimed=False,
        current_agent_params_used=True, public_policy_copy_used=False, learned_only=True,
        continuous_deployment_signal_established=False, optimizer_updates=0)
    return dict(h=combined['h'][:, 0].copy(), zq=combined['zq'].copy(), zp=combined['zp'].copy(),
        D=combined['D'][:, 0].copy(), K=combined['K'][:, 0].copy(), aux=aux)

  def annotate_episode_terminal(self, episodes, key, check=lambda: None):
    """Batched full1001 actual obs/raw-action annotation for frozen context."""
    import jax
    from embodied.jax import internal
    obs, previous = annotation_inputs(episodes)
    key = np.asarray(key)
    require(key.dtype == np.uint32 and key.shape == (2,), 'Require explicit private annotation root U32[2]')
    input_sha = tree_digest(dict(obs=obs, previous=previous, key=key))
    with self.agent.train_lock:
      check(); self.assert_frozen(full=True)
      placed = internal.device_put((obs, previous, key.copy()), self.agent.train_mirrored)
      out = self._annotate(self.agent.params, *placed)
      jax.block_until_ready(out)
      out = host_copy(out)
      check(); self.assert_frozen(full=True)
    require(_same(out['root_key'], key), 'Annotation private root differs')
    enc, dyn, dec = out['entries']['enc'], out['entries']['dyn'], out['entries']['dec']
    require(not enc and not dec and set(dyn) == {'deter', 'stoch'}, 'Current annotation entry structure differs')
    require(all(str(value.dtype) == 'bfloat16' and np.isfinite(value).all() for value in dyn.values()),
        'Actual annotation must retain finite BF16 entries')
    cfg = self.agent.config.dyn.rssm
    require(dyn['deter'].shape == (len(episodes), SOURCE_FRAMES, cfg.deter) and
        dyn['stoch'].shape == (len(episodes), SOURCE_FRAMES, cfg.stoch, cfg.classes),
        'Actual full annotation dimensions differ')
    z = dyn['stoch'].astype(np.float64)
    require(np.logical_or(z == 0, z == 1).all() and np.all(z.sum(-1) == 1),
        'Actual annotation categorical entries must be sampled onehot')
    entries = {'dyn/'+name: value.copy() for name, value in dyn.items()}
    terminal = {name: value[:, -1].copy() for name, value in entries.items()}
    require(all(_same(out['carry'][1][name], dyn[name][:, -1]) for name in dyn),
        'Actual terminal carry/entry bytes differ')
    full_carry = (*out['carry'], {'action': previous['action'][:, -1].copy()})
    require(tree_digest(dict(obs=obs, previous=previous, key=key)) == input_sha,
        'Annotation input bytes changed')
    return dict(entries=entries, terminal_entry=terminal, carry=full_carry,
        aux=dict(root_key=key.copy(), frames=SOURCE_FRAMES, episodes=len(episodes),
            actual_fields_used=[*OBS_KEYS, 'action', 'action_executed'],
            annotation_is_new_private_representation=True, source_original_carry_exact_claimed=False,
            carry_is_observation_annotation=True, carry_is_source_policy_carry=False,
            previous_terminal_action_is_raw999=True, terminal_placeholder_executed=False,
            actor_calls=0, environment_actions=0, optimizer_updates=0,
            all_entries_sha256=tree_digest(entries), terminal_entry_sha256=tree_digest(terminal),
            actual_carry_sha256=tree_digest(full_carry), input_sha256=input_sha,
            full_state_before_sha256=self.initial_state_sha256,
            full_state_after_sha256=self.initial_state_sha256,
            counters_before=dict(self.initial_counters), counters_after=counters(self.agent),
            accessed_parameter_keys=sorted(self.annotation_accessed_keys)))


def annotate_episode_terminal(agent, arrays, key, check=lambda: None):
  """Convenience wrapper; arrays is a sequence of full episode dictionaries."""
  return ResetFeatureAdapter(agent).annotate_episode_terminal(arrays, key, check)
