"""DL-production-reward-r1: frozen native RSSM/reward conditional rollouts.

No environment, checkpoint, source arrays, optimizer, or resource allocation is
performed here. An already loaded Agent and verified current h/z are supplied by
the worker. Four initial samples are fixed; only future latent paths are sampled.
"""
import functools

import numpy as np

VERSION = 'DL-production-reward-r1'
IMPLEMENTATION = 'DL-production-reward-model-r1'
SEED, KEY_TAG = 20261024, 0x50525231
HORIZON, INITIALS, BRANCHES, CHUNK, CONTROLS, BLOCKS, PER_BLOCK = 20, 4, 2, 8, 7, 4, 32
EPISODES = (9, 14, 15, 18, 20, 24, 30, 31)
POSITIONS = (1, 300, 600, 900)
PRED_ATOL, PRED_RTOL = 1e-6, 1e-5


def require(condition, message):
  if not condition:
    raise ValueError(message)


def host_copy(tree):
  import jax
  with jax._src.config.explicit_device_get_scope():
    return jax.tree.map(lambda x: np.asarray(x).copy(), tree)


def counters(agent):
  return dict(updates=int(agent.n_updates), batches=int(agent.n_batches),
      actions=int(agent.n_actions), eval_actions=int(agent.n_eval_actions),
      eval_seed=int(agent.eval_seed))


def restore_native(h, z):
  """Strictly reverse the source's BF16 -> F64 promotion, on the CPU."""
  import ml_dtypes
  values = []
  for name, value in (('h', h), ('z', z)):
    require(type(value) is np.ndarray and value.dtype == np.float64 and
        np.isfinite(value).all(), 'Require finite source F64 array: ' + name)
    native = value.astype(ml_dtypes.bfloat16)
    back = native.astype(np.float64)
    require(back.shape == value.shape and back.tobytes() == value.tobytes(),
        'Source is not an exact BF16 promotion: ' + name)
    values.append(native)
  require(values[0].ndim == 1 and values[1].ndim == 4 and
      values[1].shape[:2] == (INITIALS, BRANCHES), 'Invalid initial feature axes')
  require(np.logical_or(z == 0, z == 1).all() and np.all(z.sum(-1) == 1),
      'Require fixed actual sampled onehot latents')
  return tuple(values)


@functools.lru_cache(maxsize=1)
def _key_function():
  import jax
  import jax.numpy as jnp
  def build(labels):
    root = jax.random.fold_in(jax.random.PRNGKey(SEED), jnp.uint32(KEY_TAG))
    root = jax.random.fold_in(root, labels[0])
    root = jax.random.fold_in(root, labels[1])
    def initial_keys(initial):
      base = jax.random.fold_in(root, initial)
      base = jax.random.fold_in(base, labels[2])
      def path_keys(path):
        path_root = jax.random.fold_in(base, path)
        return jax.vmap(lambda step: jax.random.fold_in(path_root, step))(
            jnp.arange(1, HORIZON + 1, dtype=jnp.uint32))
      return jax.vmap(path_keys)(jnp.arange(CHUNK, dtype=jnp.uint32) + labels[3])
    return jax.vmap(initial_keys)(jnp.arange(INITIALS, dtype=jnp.uint32))
  return jax.jit(build)


def private_keys(episode, position, block, sample_start):
  """Host U32 roots[4,8,20,2]; branches/controls deliberately share roots."""
  import jax
  labels = (episode, position, block, sample_start)
  require(all(type(x) is int for x in labels) and episode in EPISODES and
      position in POSITIONS and 0 <= block < BLOCKS and
      sample_start in (0, 8, 16, 24), 'Invalid frozen MC labels/chunk')
  # The Agent enables strict transfer guards. Constants and PRNG construction
  # are inside JIT; only explicitly placed dynamic host labels cross the guard.
  placed = jax.device_put(np.asarray(labels, np.uint32))
  return host_copy(_key_function()(placed))


@functools.lru_cache(maxsize=1)
def _sample_key_function():
  import jax
  def build(roots):
    return jax.vmap(lambda key: jax.random.split(key, 16)[1])(
        roots.reshape((-1, 2))).reshape(roots.shape)
  return jax.jit(build)


def sample_keys(roots):
  """Canonical expected sample keys; actual private context is checked at run."""
  import jax
  require(type(roots) is np.ndarray and roots.dtype == np.uint32 and
      roots.shape == (INITIALS, CHUNK, HORIZON, 2), 'Require canonical explicit roots')
  return host_copy(_sample_key_function()(jax.device_put(roots)))


def readonly_single_step(model, accessed=None):
  """Actual official B1 imagine, with one private Ninjax root per transition.

  The private context's seed/reserve is checked against exactly one ordinary
  nj.seed() call. This identifies its actual sample key without a global hook or
  inventing a new categorical sampler. Writes, including value-preserving writes,
  are rejected by Ninjax attempted-access tracking.
  """
  import jax
  import jax.numpy as jnp
  import ninjax as nj
  def step(h, z, action):
    ctx = nj.context()
    root = ctx.seed
    carry, (feat, _) = model.dyn.imagine(
        dict(deter=h[None], stoch=z[None]), dict(action=action[None]),
        1, training=False, single=True)
    reward = model.rew(model.feat2tensor(feat), 1)
    if len(ctx.reserve) != 14:
      raise AssertionError('Native imagination must consume exactly one nj.seed()')
    split = jax.random.split(root, 16)
    rng_valid = jnp.all(ctx.seed == split[0]) & jnp.all(jnp.stack(ctx.reserve) == split[2:])
    return dict(h=carry['deter'][0], z=carry['stoch'][0],
        reward=reward.pred()[0], logits=reward.logits[0], bins=reward.bins,
        sample_key=split[1], rng_valid=rng_valid, rng_consumption=jnp.int32(1))
  pure = nj.pure(step)
  def invoke(params, h, z, action, root):
    _, out, read, modified, created = pure(params, h, z, action, seed=root,
        create=False, modify=True, ignore=False, track=True)
    if modified or created:
      raise AssertionError('Native reward prediction attempted model state writes')
    if accessed is not None:
      accessed.update(read)
    return out
  return invoke


class NativeRewardAdapter:
  """One state x four initial samples x two branches x chunk8 x seven actions."""
  def __init__(self, agent):
    import jax
    import jax.numpy as jnp
    require(str(agent.config.jax.compute_dtype) == 'bfloat16', 'Require native BF16 model')
    require(len(agent.train_devices) == 1 and not agent.jaxcfg.use_shardmap,
        'Require single-device native adapter')
    require(agent.config.rewhead.output == 'symexp_twohot', 'Require native symexp_twohot head')
    require(set(agent.act_space) == {'action'} and agent.act_space['action'].shape == (12,),
        'Require exact normalized action12 space')
    self.agent = agent
    self.initial_counters = counters(agent)
    self.initial_params = agent.params
    self.initial_values = tuple((k, id(v)) for k, v in sorted(agent.params.items()))
    self.initial_policy_params = getattr(agent, 'policy_params', None)
    self.initial_policy_values = (tuple((k, id(v)) for k, v in sorted(self.initial_policy_params.items()))
        if self.initial_policy_params is not None else None)
    self.accessed_keys = set()
    single = readonly_single_step(agent.model, self.accessed_keys)
    many = jax.vmap(single, in_axes=(None, 0, 0, 0, 0))
    bshape = (INITIALS, BRANCHES, CHUNK, CONTROLS)
    def rollout(params, h, z, controls, roots):
      # No observe, reset, current transition, or new initial latent draw.
      current_h = jnp.broadcast_to(h, (*bshape, h.shape[-1])).reshape((-1, h.shape[-1]))
      current_z = jnp.broadcast_to(z[:, :, None, None], (*bshape, *z.shape[-2:]))
      current_z = current_z.reshape((-1, *z.shape[-2:]))
      acts = jnp.broadcast_to(controls[None, None, None], (*bshape, HORIZON, 12))
      keys = jnp.broadcast_to(roots[:, None, :, None], (*bshape, HORIZON, 2))
      acts = jnp.moveaxis(acts, -2, 0).reshape((HORIZON, -1, 12))
      keys = jnp.moveaxis(keys, -2, 0).reshape((HORIZON, -1, 2))
      def body(carry, inputs):
        out = many(params, *carry, *inputs)
        return (out['h'], out['z']), {name: out[name] for name in
            ('reward', 'logits', 'bins', 'sample_key', 'rng_valid', 'rng_consumption')}
      final, outs = jax.lax.scan(body, (current_h, current_z), (acts, keys))
      def shaped(name):
        value = outs[name]
        return jnp.moveaxis(value.reshape((HORIZON, *bshape, *value.shape[2:])), 0, 4)
      return dict(reward=shaped('reward'), logits=shaped('logits'),
          bins=outs['bins'][0, 0], sample_keys=shaped('sample_key'),
          rng_valid=shaped('rng_valid'), rng_consumption=shaped('rng_consumption'),
          final_h=final[0].reshape((*bshape, h.shape[-1])),
          final_z=final[1].reshape((*bshape, *z.shape[-2:])))
    # One fixed graph, including logits, for every production chunk.
    self._call = jax.jit(rollout)

  def assert_frozen(self):
    agent = self.agent
    if (agent.params is not self.initial_params or
        tuple((k, id(v)) for k, v in sorted(agent.params.items())) != self.initial_values or
        getattr(agent, 'policy_params', None) is not self.initial_policy_params or
        (tuple((k, id(v)) for k, v in sorted(agent.policy_params.items()))
          if self.initial_policy_params is not None else None) != self.initial_policy_values or
        getattr(agent, 'pending_sync', None) is not None or counters(agent) != self.initial_counters):
      raise AssertionError('Native prediction requires unchanged parameters/policy/counters')

  def call_chunk(self, h, z, controls, roots):
    import jax
    from embodied.jax import internal
    cfg = self.agent.config.dyn.rssm
    expected = dict(h=(cfg.deter,), z=(INITIALS, BRANCHES, cfg.stoch, cfg.classes))
    for name, value in (('h', h), ('z', z)):
      require(type(value) is np.ndarray and str(value.dtype) == 'bfloat16' and
          value.shape == expected[name] and np.isfinite(value).all(), 'Invalid native initial ' + name)
    z64 = z.astype(np.float64)
    require(np.logical_or(z64 == 0, z64 == 1).all() and np.all(z64.sum(-1) == 1),
        'Require actual fixed initial onehot samples')
    require(type(controls) is np.ndarray and controls.dtype == np.float32 and
        controls.shape == (CONTROLS, HORIZON, 12) and np.isfinite(controls).all(),
        'Require raw F32 control sequences[7,20,12]')
    require(type(roots) is np.ndarray and roots.dtype == np.uint32 and
        roots.shape == (INITIALS, CHUNK, HORIZON, 2), 'Require explicit roots[4,8,20,2]')
    self.assert_frozen()
    placed = internal.device_put((h, z, controls, roots), self.agent.train_mirrored)
    out = self._call(self.agent.params, *placed)
    jax.block_until_ready(out)
    out = host_copy(out)
    self.assert_frozen()
    require(out['rng_valid'].all() and np.all(out['rng_consumption'] == 1),
        'Private Ninjax RNG consumption differed')
    sample = out['sample_keys'][:, 0, :, 0]
    require(np.array_equal(out['sample_keys'], np.broadcast_to(
        sample[:, None, :, None], out['sample_keys'].shape)), 'Branch/control sample keys differed')
    require(out['reward'].dtype == np.float32 and out['logits'].dtype == np.float32 and
        out['bins'].dtype == np.float32 and all(np.isfinite(x).all() for x in out.values()),
        'Native output dtype/finite contract failed')
    return dict(reward=out['reward'], logits=out['logits'], bins=out['bins'],
        sample_keys=sample.copy(), root_keys=roots.copy())
