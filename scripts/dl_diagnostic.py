"""DL-code-r1: frozen-model, independently labelled DtLatch diagnostics.

collect -> score -> analyze are separate, auditable stages. No training occurs.
Only quadruped_walk is supported: segmentation preserves ALL geometries,
including ground and contact shadows. Body removal is explicitly a cue
deprivation intervention, never a label of task irrelevance. Simulator rewards
compare the first q/p mean action followed by nine shared recorded actions.
They are finite action comparisons, not optimal-action or causal cue labels.

Checkpoint and environment state pickles must be trusted project artifacts.
New output files are never silently overwritten. collect/score require fresh
sv3 GPU4--7 resource and actual CUDA/EGL mapping evidence, including CPU model
inference because those stages still render. analyze never uses a GPU guard.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VERSION = 'DL-code-r1'
VARIANTS = ('clean', 'background_a', 'background_b', 'cue_deprivation')


def sha256(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for block in iter(lambda: stream.read(2 ** 20), b''):
      digest.update(block)
  return digest.hexdigest()


def git_sha():
  return subprocess.check_output(
      ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def write_json(path, value):
  path = Path(path)
  if path.exists():
    raise FileExistsError(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value, indent=2, sort_keys=True,
      allow_nan=False) + '\n', encoding='utf-8')


def dilate(mask, radius=1):
  """Conservative edge preservation without an image-processing dependency."""
  mask = np.asarray(mask, bool)
  padded = np.pad(mask, radius)
  result = np.zeros_like(mask)
  h, w = mask.shape
  for y in range(2 * radius + 1):
    for x in range(2 * radius + 1):
      result |= padded[y:y + h, x:x + w]
  return result


def segmentation_masks(segmentation, geom_bodyid, geom_type=5):
  """MuJoCo segmentation is [object ID, object type]; preserve unknown objects."""
  seg = np.asarray(segmentation)
  if seg.ndim != 3 or seg.shape[-1] != 2:
    raise ValueError('Expected dm_control [H,W,2] segmentation')
  object_id, object_type = seg[..., 0], seg[..., 1]
  geom = (object_type == geom_type) & (object_id >= 0)
  if geom.any() and object_id[geom].max() >= len(geom_bodyid):
    raise ValueError('Segmentation geom ID exceeds model.ngeom')
  body = np.zeros_like(geom)
  body[geom] = np.asarray(geom_bodyid)[object_id[geom]] > 0
  # Ground geometry contains contact shadows; preserving the whole rendered
  # ground is stricter than identifying a narrow task-dependent contact patch.
  preserve = dilate((object_id >= 0) | geom)
  return preserve, dilate(body)


def intervention_images(image, preserve, body, seed):
  image = np.asarray(image, np.uint8)
  preserve, body = np.asarray(preserve, bool), np.asarray(body, bool)
  if image.shape[:2] != preserve.shape or preserve.shape != body.shape:
    raise ValueError('Image and mask shapes differ')
  rng = np.random.default_rng(seed)
  h, w = preserve.shape
  colors = rng.integers(16, 240, (4, 3), dtype=np.uint8)
  mix = np.linspace(0, 1, h)[:, None, None]
  texture_a = np.broadcast_to((colors[0] * (1 - mix) +
      colors[1] * mix).astype(np.uint8), (h, w, 3))
  yy, xx = np.indices((h, w))
  texture_b = np.where(((yy // 4 + xx // 4) % 2)[..., None],
      colors[2], colors[3]).astype(np.uint8)
  a, b, cue = image.copy(), image.copy(), image.copy()
  a[~preserve], b[~preserve] = texture_a[~preserve], texture_b[~preserve]
  cue[body] = 127
  if not np.array_equal(a[preserve], image[preserve]) or not np.array_equal(
      b[preserve], image[preserve]):
    raise AssertionError('Nuisance intervention changed protected geometry')
  return np.stack([image, a, b, cue])


def episode_plan(count=32, calibration=8, seed=20261010):
  if count < 2 or not 0 < calibration < count:
    raise ValueError('Require nonempty independent calibration and blind splits')
  order = np.random.default_rng(seed).permutation(count)
  calib = set(order[:calibration].tolist())
  return [dict(episode=i, seed=int(np.random.SeedSequence(
      [seed, i, 0x444C]).generate_state(1)[0]),
      policy_seed=int(np.random.SeedSequence([seed, i, 0x504F4C]).generate_state(1)[0]),
      split='calibration' if i in calib else 'blind') for i in range(count)]


def sample_positions(length, maximum=32, horizon=10):
  """Predetermined outcome-independent positions with full continuation."""
  last = length - horizon - 1
  if maximum < 1 or horizon < 1:
    raise ValueError('Sample count and horizon must be positive')
  if last < 1:
    return np.array([], np.int32)
  return np.unique(np.linspace(1, last, min(maximum, last), dtype=np.int32))


def fixed_actions(shape, magnitude=0.25):
  """Cue-independent fixed action set, not an optimization procedure."""
  n = int(np.prod(shape))
  pattern = np.where(np.arange(n) % 2, magnitude, -magnitude)
  return np.stack([np.zeros(n), np.full(n, magnitude),
      np.full(n, -magnitude), pattern, -pattern]).reshape((5, *shape)).astype(
          np.float32)


def compatibility_numpy(qlogits, plogits, reward, bins):
  """Exact identity-squash TwoHot target; CPU negative-control rescoring.

  Restricted to this repo's symexp_twohot reward head. Numeric fixtures compare
  the target and score to the actual TwoHot implementation, including endpoints.
  """
  reward, bins = np.asarray(reward, np.float32), np.asarray(bins, np.float32)
  q, p = np.asarray(qlogits, np.float64), np.asarray(plogits, np.float64)
  if q.shape != p.shape or q.shape[:-1] != reward.shape or q.shape[-1] != len(bins):
    raise ValueError('Reward/logit/bin shape mismatch')
  if not all(np.isfinite(x).all() for x in (q, p, reward, bins)) or not np.all(
      np.diff(bins) > 0):
    raise ValueError('Nonfinite scores or unordered bins')
  below = np.clip((bins <= reward[..., None]).sum(-1) - 1, 0, len(bins) - 1)
  above = np.clip(len(bins) - (bins > reward[..., None]).sum(-1), 0, len(bins) - 1)
  same = below == above
  db = np.where(same, 1, np.abs(bins[below] - reward))
  da = np.where(same, 1, np.abs(bins[above] - reward))
  target = (np.eye(len(bins))[below] * (da / (db + da))[..., None] +
      np.eye(len(bins))[above] * (db / (db + da))[..., None])
  entropy = -np.sum(np.where(target > 0, target * np.log(
      np.maximum(target, np.finfo(float).tiny)), 0), -1)
  def ce(logits):
    maximum = logits.max(-1, keepdims=True)
    logz = maximum + np.log(np.exp(logits - maximum).sum(-1, keepdims=True))
    return -(target * (logits - logz)).sum(-1)
  ell_q, ell_p = ce(q), ce(p)
  dq, dp = ell_q - entropy, ell_p - entropy
  if min(dq.min(initial=0), dp.min(initial=0)) < -1e-5:
    raise ValueError('Materially negative compatibility KL')
  uq, up = np.exp(-np.maximum(dq, 0)), np.exp(-np.maximum(dp, 0))
  return dict(target=target, ell_q=ell_q, ell_p=ell_p, H_y=entropy,
      signed_support=uq - up, e=np.maximum(uq - up, 0), loss_gap=ell_p - ell_q)


def auc(labels, scores):
  labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
  if labels.shape != scores.shape or not np.isfinite(scores).all():
    raise ValueError('Invalid AUROC input')
  positive, negative = scores[labels], scores[~labels]
  if not len(positive) or not len(negative):
    return None
  # Count ties exactly and avoid a quadratic Npositive*Nnegative allocation.
  negative = np.sort(negative)
  lower = np.searchsorted(negative, positive, side='left')
  upper = np.searchsorted(negative, positive, side='right')
  return float(np.mean((lower + upper) / (2 * len(negative))))


def cluster_interval(values, episodes, statistic=None, draws=2000, seed=17):
  """Bootstrap complete episodes; scalar means give every episode equal weight."""
  values, episodes = np.asarray(values), np.asarray(episodes)
  unique = np.unique(episodes)
  if not len(unique):
    return dict(point=None, lower95=None, upper95=None, episodes=0)
  groups = [np.flatnonzero(episodes == e) for e in unique]
  if statistic is None:
    means = np.asarray([values[x].mean() for x in groups])
    point = float(means.mean())
    sampled = np.random.default_rng(seed).integers(
        len(groups), size=(draws, len(groups)))
    estimates = means[sampled].mean(-1)
  else:
    point = statistic(np.arange(len(values)))
    rng, estimates = np.random.default_rng(seed), []
    for _ in range(draws):
      idx = np.concatenate([groups[i] for i in rng.integers(
          len(groups), size=len(groups))])
      result = statistic(idx)
      if result is not None:
        estimates.append(result)
    estimates = np.asarray(estimates)
  if point is None or len(estimates) < 0.95 * draws:
    return dict(point=point, lower95=None, upper95=None, episodes=len(unique),
        valid_bootstrap_draws=len(estimates))
  return dict(point=point, lower95=float(np.quantile(estimates, .025)),
      upper95=float(np.quantile(estimates, .975)), episodes=len(unique),
      valid_bootstrap_draws=len(estimates))


def different_episode_labels(reward, episodes, seed=19):
  """Rotate whole label episodes; never assign a label from the same episode.

  Positional quantiles match unequal episode sample counts. The caller applies
  this separately in calibration/blind splits, preventing split leakage.
  """
  reward, episodes = np.asarray(reward), np.asarray(episodes)
  unique = np.unique(episodes)
  if len(unique) < 2:
    raise ValueError('Cross-episode label control requires at least two episodes')
  order = np.random.default_rng(seed).permutation(unique)
  result, source = np.empty_like(reward), np.empty_like(episodes)
  for a, b in zip(order, np.roll(order, 1)):
    dest, donor = np.flatnonzero(episodes == a), np.flatnonzero(episodes == b)
    chosen = donor[np.minimum((np.arange(len(dest)) * len(donor) /
        len(dest)).astype(int), len(donor) - 1)]
    result[dest], source[dest] = reward[chosen], b
  assert np.all(source != episodes)
  return result, source


def load_config(path, task, output, platform):
  import elements
  from ruamel.yaml import YAML
  yaml = YAML(typ='safe')
  presets = yaml.load((ROOT / 'dreamerv3/configs.yaml').read_text(encoding='utf-8'))
  if not path:
    raise ValueError('--config must name the checkpoint run\'s complete actual configuration')
  config = elements.Config(yaml.load(Path(path).read_text(encoding='utf-8')))
  if config.task != task:
    raise ValueError('Checkpoint configuration task differs from diagnostic task')
  changes = dict(task=task, logdir=str(output), **{
      'jax.platform': platform, 'jax.policy_devices': [0], 'jax.train_devices': [0],
      'jax.prealloc': False, 'jax.precompile': False, 'jax.profiler': False,
      'jax.enable_policy': True, 'agent.dt_latch.mode': 'off',
      'agent.rep_probe.mode': 'off'})
  config = elements.Config({**config.flat, **changes})
  rssm = config.agent.dyn.rssm
  expected = dict(deter=4096, hidden=512, stoch=32, classes=32, unimix=.01,
      blocks=8, imglayers=2, obslayers=1, dynlayers=1, absolute=False, norm='rms', act='silu')
  if (any(getattr(rssm, k) != v for k, v in expected.items()) or
      config.agent.enc.typ != 'simple' or config.agent.dyn.typ != 'rssm' or
      config.agent.enc.simple.depth != 32 or config.agent.dec.simple.depth != 32 or
      config.agent.policy.units != 512 or config.agent.policy_dist_cont != 'bounded_normal' or
      config.agent.rewhead.output != 'symexp_twohot' or config.agent.rewhead.bins != 255 or
      config.env.dmc.repeat != 1 or config.env.dmc.proprio or not config.env.dmc.image or
      tuple(config.env.dmc.size) != (64, 64) or config.env.dmc.camera not in (-1, 2)):
    raise ValueError('DL diagnostic requires real size50m, vision64, action repeat1')
  return config


def load_agent(args):
  from dreamerv3.main import make_agent
  from embodied.run.protocol_v1 import parameter_digest
  config = load_config(args.config, args.task, args.output, args.platform)
  agent = make_agent(config)
  path = Path(args.checkpoint)
  if path.is_dir() and (path / 'latest').exists():
    path = path / (path / 'latest').read_text().strip()
  if path.is_dir():
    path = path / 'agent.pkl'
  with path.open('rb') as file:
    checkpoint = pickle.load(file)
  agent.load(checkpoint)
  metadata = dict(checkpoint=str(path.resolve()), checkpoint_sha256=sha256(path),
      checkpoint_counters=checkpoint['counters'], params_sha256=parameter_digest(agent),
      config=config.flat, config_source_sha256=sha256(args.config),
      git_sha=git_sha(), code_version=VERSION)
  return agent, config, metadata


def make_dmc(task, seed):
  from dm_control import suite
  if task != 'dmc_quadruped_walk':
    raise ValueError('DL-code-r1 freezes diagnostics to dmc_quadruped_walk')
  return suite.load('quadruped', 'walk', task_kwargs={'random': int(seed)})


def physics_snapshot(env):
  import mujoco
  bits = mujoco.mjtState.mjSTATE_INTEGRATION
  state = np.empty(mujoco.mj_stateSize(env.physics.model.ptr, bits), np.float64)
  mujoco.mj_getState(env.physics.model.ptr, env.physics.data.ptr, state, bits)
  return dict(integration=state, step_count=env._step_count,
      reset_next_step=env._reset_next_step,
      task=copy.deepcopy(env._task.__dict__))


def restore_physics(env, snapshot):
  import mujoco
  env._task.__dict__.clear()
  env._task.__dict__.update(copy.deepcopy(snapshot['task']))
  env._step_count = snapshot['step_count']
  env._reset_next_step = snapshot['reset_next_step']
  mujoco.mj_setState(env.physics.model.ptr, env.physics.data.ptr,
      snapshot['integration'], mujoco.mjtState.mjSTATE_INTEGRATION)
  env.physics.forward()


def true_consequence(env, snapshot, first_action, continuation, horizon):
  restore_physics(env, snapshot)
  rewards = []
  actions = [first_action, *list(continuation[:horizon - 1])]
  if len(actions) != horizon:
    raise ValueError('Insufficient shared recorded continuation')
  for action in actions:
    timestep = env.step(np.asarray(action, np.float64))
    rewards.append(float(timestep.reward))
    if timestep.last() and len(rewards) < horizon:
      raise ValueError('Unexpected terminal in external consequence window')
  return np.asarray(rewards, np.float64)


def verified_consequence(env, snapshot, action, continuation, horizon):
  """Repeat rewards AND full integration/task/counter outcome, not reward alone."""
  reward = true_consequence(env, snapshot, action, continuation, horizon)
  first = physics_snapshot(env)
  repeat = true_consequence(env, snapshot, action, continuation, horizon)
  second = physics_snapshot(env)
  error = max(float(np.max(np.abs(reward - repeat))), float(np.max(np.abs(
      first['integration'] - second['integration']))))
  task1 = pickle.dumps(first['task'], protocol=5)
  task2 = pickle.dumps(second['task'], protocol=5)
  if (error > 1e-8 or task1 != task2 or first['step_count'] != second['step_count']
      or first['reset_next_step'] != second['reset_next_step']):
    raise AssertionError('DL complete-state simulator restoration is not reproducible')
  digest = hashlib.sha256(first['integration'].tobytes() + task1).hexdigest()
  return reward, error, digest


def consequence_prefixes(payoffs, horizons):
  """DL-opportunity-r1: summaries of the same true recorded continuation.

  The last candidate is the prior; earlier candidates follow VARIANTS order.
  This changes only external labels, never the H1 reward probe or its gate.
  """
  payoffs = np.asarray(payoffs, np.float64)
  horizons = tuple(horizons)
  if (payoffs.ndim != 2 or payoffs.shape[0] != len(VARIANTS) + 1 or
      not np.isfinite(payoffs).all() or not horizons or
      any(type(h) is not int or h < 1 or h > payoffs.shape[1] for h in horizons) or
      any(a >= b for a, b in zip(horizons, horizons[1:]))):
    raise ValueError('Invalid external consequence prefix dimensions or horizons')
  sums = np.stack([payoffs[:, :h].sum(axis=-1) for h in horizons], axis=-1)
  means = sums / np.asarray(horizons)
  return dict(consequence_prefixes=np.asarray(horizons, np.int32),
      true_q_prefix_sum=sums[:-1], true_p_prefix_sum=sums[-1],
      true_q_prefix_mean=means[:-1], true_p_prefix_mean=means[-1])


def collect(args):
  from scripts.dl_resources import require_resource, verify_runtime
  resource = require_resource(args.resource_record)
  import mujoco
  from embodied.run.protocol_v1 import parameter_digest
  output = Path(args.output)
  agent, config, metadata = load_agent(args)
  resource['runtime_compute'] = verify_runtime(resource, compute=args.platform == 'cuda')
  metadata['resource'] = resource
  plan = episode_plan(args.episodes, args.calibration_episodes, args.seed)
  write_json(output / 'collection_plan.json', dict(**metadata, task=args.task,
      camera=2, shape=[64, 64], seed=args.seed, episodes=plan,
      mask='all segmented objects + 1px dilation; bodyid>0 cue deprivation',
      physics='mjSTATE_INTEGRATION + environment step/reset + deep-copied task',
      behavior='checkpoint sampled eval policy; independent environment seeds'))
  artifacts = []
  graphics_checked = False
  for entry in plan:
    env = make_dmc(args.task, entry['seed'])
    try:
      agent.set_eval_seed(entry['policy_seed'])
      ts, carry = env.reset(), agent.init_policy(1, mode='eval')
      arrays = {k: [] for k in ('image', 'preserve', 'body', 'reward', 'is_first',
          'is_last', 'is_terminal', 'action')}
      states = []
      while True:
        image = env.physics.render(64, 64, camera_id=2)
        seg = env.physics.render(64, 64, camera_id=2, segmentation=True)
        if not graphics_checked:
          resource['runtime_cuda_egl'] = verify_runtime(resource,
              compute=args.platform == 'cuda', graphics=True)
          graphics_checked = True
        preserve, body = segmentation_masks(seg, env.physics.model.geom_bodyid,
            int(mujoco.mjtObj.mjOBJ_GEOM))
        obs = dict(image=image, reward=np.float32(ts.reward or 0),
            is_first=np.bool_(ts.first()), is_last=np.bool_(ts.last()),
            is_terminal=np.bool_(ts.last() and ts.discount == 0))
        states.append(physics_snapshot(env))
        for key in obs:
          arrays[key].append(obs[key])
        arrays['preserve'].append(preserve)
        arrays['body'].append(body)
        if ts.last():
          arrays['action'].append(np.zeros(env.action_spec().shape, np.float32))
          break
        carry, action, _ = agent.policy(carry,
            {k: np.asarray(v)[None] for k, v in obs.items()}, mode='eval')
        action = np.asarray(action['action'][0], np.float32)
        arrays['action'].append(action)
        ts = env.step(action)
      path = output / f'episode_{entry["episode"]:03}.npz'
      if path.exists():
        raise FileExistsError(path)
      np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()},
          snapshots=np.frombuffer(pickle.dumps(states, protocol=5), np.uint8))
      artifacts.append(dict(**entry, path=path.name, sha256=sha256(path),
          frames=len(states), return_raw=float(np.sum(arrays['reward'])),
          body_pixel_mean=float(np.mean(arrays['body'])),
          background_pixel_mean=float(1 - np.mean(arrays['preserve']))))
      print(json.dumps(dict(stage='collect', **artifacts[-1])), flush=True)
    finally:
      env.close()
  unchanged = parameter_digest(agent) == metadata['params_sha256']
  if not unchanged:
    raise AssertionError('Frozen checkpoint parameters changed during collection')
  write_json(output / 'collection_complete.json', dict(artifacts=artifacts,
      params_unchanged=unchanged, plan_sha256=sha256(output / 'collection_plan.json'),
      resource=resource))


class FrozenScorer:
  """Single clean history; all current-image alternatives start at the same h."""

  def __init__(self, agent, alpha, rho, tolerance=1e-5):
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import embodied.jax.outs as outs
    from dreamerv3 import dt_latch
    self.agent = agent
    model = agent.model
    def observe(carry, images, previous, first):
      n = images.shape[0]
      enc, dyn = jax.tree.map(lambda x: jnp.repeat(x, n, axis=0), carry)
      reset = jnp.full((n,), first)
      _, _, tokens = model.enc(enc, {'image': images}, reset,
          training=False, single=True)
      dyn, _, feat = model.dyn.observe(dyn, tokens,
          {'action': jnp.broadcast_to(previous, (n, *previous.shape))}, reset,
          training=False, single=True)
      newcarry = (carry[0], jax.tree.map(lambda x: x[:1], dyn))
      return newcarry, feat
    def advance(carry, image, previous, first):
      return observe(carry, image[None], previous, first)[0]
    def diagnose(carry, images, previous, first, action, reward, fixed, fixed_reward):
      newcarry, feat = observe(carry, images, previous, first)
      h, logits = feat['deter'], feat['logit']
      # Images differ after the recurrent core; shared h is an invariant.
      dt_latch.require_valid(jnp.all(h == h[:1]))
      prior = model.dyn._prior(h)
      qmode, pmode = model.dyn._dist(logits).pred(), model.dyn._dist(prior).pred()
      repfeat = {k: jnp.repeat(feat[k][:, None], 2, 1) for k in ('deter', 'logit')}
      n = len(images)
      obs = dict(reward=jnp.stack([jnp.zeros(n), jnp.full(n, reward)], 1),
          is_first=jnp.zeros((n, 2), bool), is_last=jnp.zeros((n, 2), bool))
      prevact = {'action': jnp.broadcast_to(action, (n, 2, *action.shape))}
      signals = dt_latch.probe(model, repfeat, jnp.repeat(prior[:, None], 2, 1),
          obs, prevact, alpha, rho, tolerance)
      result = {k: v[:, 0] for k, v in signals.items()}
      result['K'] = model.dyn._dist(logits).kl(model.dyn._dist(prior))
      qpol = model.pol(model.feat2tensor(dict(deter=h, stoch=qmode)), 1)['action']
      ppol = model.pol(model.feat2tensor(dict(deter=h, stoch=pmode)), 1)['action']
      qpol = qpol.output if isinstance(qpol, outs.Agg) else qpol
      ppol = ppol.output if isinstance(ppol, outs.Agg) else ppol
      result['q_action'], result['p_action'] = jnp.clip(qpol.mean, -1, 1), jnp.clip(ppol.mean, -1, 1)
      result['q_std'], result['p_std'] = qpol.stddev, ppol.stddev
      axes = tuple(range(1, qpol.mean.ndim))
      qv, pv = jnp.square(jnp.float32(qpol.stddev)), jnp.square(jnp.float32(ppol.stddev))
      result['D_mean_component'] = .25 * (jnp.square(jnp.float32(qpol.mean) -
          jnp.float32(ppol.mean)) * (1 / qv + 1 / pv)).mean(axes)
      result['D_std_component'] = .25 * (jnp.square(qv - pv) / (qv * pv)).mean(axes)
      dt_latch.require_valid(jnp.abs(result['D_mean_component'] +
          result['D_std_component'] - result['D']) <= 1e-5 * (1 + result['D']))
      for name, x in (('q', logits), ('p', prior)):
        ordered = jnp.sort(jnp.float32(x), -1)
        margin = ordered[..., -1] - ordered[..., -2]
        result[f'{name}_margin_min'] = margin.min(-1)
        result[f'{name}_margin_mean'] = margin.mean(-1)
      result['mode_factor_diff'] = jnp.any(qmode != pmode, -1).mean(-1)
      def near_tie_swap(x):
        x = jnp.float32(x)
        top = x.argmax(-1)
        candidates = jnp.where(jax.nn.one_hot(top, x.shape[-1], dtype=bool), -jnp.inf, x)
        second = candidates.argmax(-1)
        topvalue = jnp.take_along_axis(x, top[..., None], -1)[..., 0]
        secondvalue = jnp.take_along_axis(x, second[..., None], -1)[..., 0]
        margin = topvalue - secondvalue
        factor = margin.argmin(-1)
        rows = jnp.arange(n)
        one, two = top[rows, factor], second[rows, factor]
        changed = x.at[rows, factor, one].set(x[rows, factor, two])
        # Promote the runner-up slightly so exact ties actually change mode.
        changed = changed.at[rows, factor, two].set(x[rows, factor, one] + .001)
        nearby = margin.min(-1) < .01
        changed = jnp.where(nearby[:, None, None], changed, x)
        return changed, nearby
      qnear, qflip = near_tie_swap(logits)
      pnear, pflip = near_tie_swap(prior)
      qnear_mode, pnear_mode = model.dyn._dist(qnear).pred(), model.dyn._dist(pnear).pred()
      qm, pm = dt_latch.one_step(model, h, qnear_mode, pnear_mode,
          {'action': jnp.broadcast_to(action, (n, *action.shape))})
      mode_support = dt_latch.reward_support(qm, pm, jnp.full(n, reward),
          jnp.ones(n, bool), tolerance)
      altered_feat = dict(deter=repfeat['deter'],
          logit=jnp.repeat(qnear[:, None], 2, 1))
      altered = dt_latch.representative_signal(model, altered_feat,
          jnp.repeat(pnear[:, None], 2, 1))
      altered_v, _ = dt_latch.gate(altered['D'][:, 0], mode_support['e'],
          jnp.ones(n, bool), alpha, rho)
      result.update(mode_perturb_D=altered['D'][:, 0],
          mode_perturb_e=mode_support['e'], mode_perturb_C=1 - altered_v,
          mode_perturb_applied=qflip | pflip,
          mode_perturb_q_K=model.dyn._dist(logits).kl(model.dyn._dist(qnear)),
          mode_perturb_p_K=model.dyn._dist(prior).kl(model.dyn._dist(pnear)))
      a = fixed.shape[0]
      repeat = lambda x: jnp.repeat(x, a, 0)
      aq, ap = dt_latch.one_step(model, repeat(h), repeat(qmode), repeat(pmode),
          {'action': jnp.tile(fixed, (n, *([1] * (fixed.ndim - 1))))})
      fixed_signals = dt_latch.reward_support(aq, ap,
          jnp.tile(fixed_reward, n), jnp.ones((n * a,), bool), tolerance)
      for key in ('e', 'signed_support', 'ell_q', 'ell_p', 'loss_gap'):
        result[f'fixed_{key}'] = fixed_signals[key].reshape((n, a))
      result['bins'] = aq.bins
      return newcarry, result
    self.advance = jax.jit(nj.pure(advance))
    self.diagnose = jax.jit(nj.pure(diagnose))

  def call(self, function, carry, *args, seed):
    import jax
    import embodied.jax.internal as internal
    inputs = internal.device_put(args, self.agent.train_mirrored)
    rng = internal.device_put(np.asarray([seed, 0], np.uint32),
        self.agent.train_mirrored)
    state, output = function(self.agent.params, carry, *inputs, seed=rng)
    if set(state) != set(self.agent.params):
      raise AssertionError('Diagnostic created parameters outside frozen checkpoint')
    return output


def score(args):
  prefixes = tuple(getattr(args, 'consequence_prefixes', ()))
  if prefixes and (prefixes != (1, 10, 50, 100) or args.horizon != 100 or
      args.samples != 16 or args.seed != 20261012):
    raise ValueError('DL-opportunity-r1 requires frozen H100/16 positions/new seed')
  from scripts.dl_resources import require_resource, verify_runtime
  resource = require_resource(args.resource_record)
  import jax
  from embodied.run.protocol_v1 import parameter_digest
  source, output = Path(args.dataset), Path(args.output)
  plan = json.loads((source / 'collection_plan.json').read_text())
  completed = json.loads((source / 'collection_complete.json').read_text())
  if plan['task'] != args.task or not np.isfinite(args.alpha) or args.alpha < 0 or not (
      np.isfinite(args.rho) and 0 <= args.rho <= 1):
    raise ValueError('Dataset task mismatch or invalid declared gate parameters')
  if sha256(source / 'collection_plan.json') != completed['plan_sha256']:
    raise ValueError('Collection plan hash mismatch')
  if prefixes:
    expected = episode_plan(32, 8, 20261012)
    keys = ('episode', 'split', 'seed', 'policy_seed')
    actual = [{k: x[k] for k in keys} for x in completed['artifacts']]
    checkpoint = Path(args.checkpoint)
    if checkpoint.is_dir() and (checkpoint / 'latest').exists():
      checkpoint /= (checkpoint / 'latest').read_text().strip()
    if checkpoint.is_dir():
      checkpoint /= 'agent.pkl'
    sources = {
        '2ab693c8ded4d8f2186bd489aa8b77498448ba2f2c331090159da4e5f08056b9':
            '14859c4e129b51aa488c10eab09f16d56fd110c76239b2c9072f415829d111c2',
        'c9c893081485c8ad22468611d046bc5bf9fbe463d55996fda15cf85b2a0323fe':
            'fb8c429cb13861d90f2aeba32842eda8958bd17d1b80f2dfa15c0363aca7549d'}
    fingerprint = sha256(checkpoint)
    config_fingerprint = '11428512fdc9b9e04a5384e419c5abcef93e5429db0a3327d05ccb0d4bd20445'
    if (plan.get('seed') != 20261012 or plan.get('episodes') != expected or
        actual != expected or not completed.get('params_unchanged') or
        args.alpha != 20 or args.rho != .5 or fingerprint not in sources or
        plan.get('checkpoint_sha256') != fingerprint or
        plan.get('params_sha256') != sources[fingerprint] or
        sha256(args.config) != config_fingerprint or
        plan.get('config_source_sha256') != config_fingerprint):
      raise ValueError('DL-opportunity-r1 requires frozen independent collection and gate')
  agent, config, metadata = load_agent(args)
  resource['runtime_compute'] = verify_runtime(resource, compute=args.platform == 'cuda')
  metadata['resource'] = resource
  if metadata['checkpoint_sha256'] != plan['checkpoint_sha256'] or metadata[
      'params_sha256'] != plan['params_sha256']:
    raise ValueError('DL-r1 requires independent episodes from this same frozen checkpoint')
  scorer = FrozenScorer(agent, args.alpha, args.rho)
  if prefixes:
    output.mkdir(parents=True, exist_ok=True)
    for filename in ('collection_plan.json', 'collection_complete.json'):
      copied = output / filename
      if copied.exists():
        raise FileExistsError(copied)
      copied.write_bytes((source / filename).read_bytes())
  extra_plan = dict(consequence_prefixes=list(prefixes),
      diagnostic_seed=args.seed, dataset_complete_sha256=sha256(
          source / 'collection_complete.json'),
      opportunity_version='DL-opportunity-r1') if prefixes else {}
  write_json(output / 'score_plan.json', dict(**metadata, **extra_plan,
      dataset=str(source.resolve()), dataset_plan_sha256=completed['plan_sha256'],
      collection_checkpoint_sha256=plan['checkpoint_sha256'],
      collection_params_sha256=plan['params_sha256'], cross_model_behavior=False,
      task=args.task, sample_count=args.samples, horizon=args.horizon,
      alpha=args.alpha, rho=args.rho, free_nats=config.agent.dyn.rssm.free_nats,
      variants=VARIANTS, consequence='first q/p action; shared recorded continuation',
      history_rng='Every frame advances clean B1; B4 diagnostic carry is discarded',
      mode_fragility='Swap closest top-two if logit margin<.01; promote runner-up .001 '
          'to break exact ties; report perturbation KL',
      restoration_tolerance=1e-8, fixed_action_magnitude=.25))
  artifacts, started = [], time.perf_counter()
  for entry in completed['artifacts']:
    path = source / entry['path']
    if sha256(path) != entry['sha256']:
      raise ValueError(f'Collection artifact hash mismatch: {path}')
    with np.load(path, allow_pickle=False) as archive:
      data = {k: archive[k] for k in archive.files}
    states = pickle.loads(data.pop('snapshots').tobytes())
    positions = sample_positions(len(data['reward']), args.samples, args.horizon)
    if not len(positions):
      raise ValueError('Episode too short for frozen diagnostic window')
    selected = set(positions.tolist())
    env = make_dmc(args.task, entry['seed'])
    env.reset()
    carry = agent.init_train(1)[:2]
    fixed = fixed_actions(data['action'].shape[1:])
    records, bins = [], None
    try:
      for t in range(int(positions[-1]) + 1):
        previous = data['action'][t - 1] if t else np.zeros_like(data['action'][0])
        seed = int(np.random.SeedSequence([args.seed, entry['episode'], t]).generate_state(1)[0])
        before = carry
        carry = scorer.call(scorer.advance, before, data['image'][t], previous,
            data['is_first'][t], seed=seed)
        if t not in selected:
          continue
        continuation = data['action'][t + 1:t + args.horizon]
        reference, error, reference_hash = verified_consequence(
            env, states[t], data['action'][t], continuation,
            args.horizon if prefixes else 1)
        # Collection uses float32 labels. Compare their exact encoding without
        # loosening the independent float64 repeated-physics tolerance.
        if not np.array_equal(np.asarray(reference, np.float32),
            data['reward'][t + 1:t + 1 + len(reference)]):
          raise AssertionError(f'Restored reward encoding differs at ep={entry["episode"]},t={t}')
        fixed_reward = []
        for action in fixed:
          true, err, _ = verified_consequence(env, states[t], action, continuation, 1)
          fixed_reward.append(true[0])
          error = max(error, err)
        fixed_reward = np.asarray(fixed_reward, np.float32)
        images = intervention_images(data['image'][t], data['preserve'][t],
            data['body'][t], [args.seed, entry['episode']])
        _, result = scorer.call(scorer.diagnose, before, images, previous,
            data['is_first'][t], data['action'][t], data['reward'][t + 1],
            fixed, fixed_reward, seed=seed)
        with jax._src.config.explicit_device_get_scope():
          result = jax.tree.map(lambda x: np.asarray(x, np.float32), result)
        bins = result.pop('bins')
        qa, pa = result['q_action'], result['p_action']
        if not np.array_equal(pa, np.repeat(pa[:1], len(VARIANTS), 0)):
          raise AssertionError('Paired interventions changed prior action')
        # Repeat every external candidate from the same complete state. Common
        # continuation controls subsequent actions and prevents reset leakage.
        payoffs = []
        payoff_hashes = []
        for action in [*list(qa), pa[0]]:
          true, err, digest = verified_consequence(
              env, states[t], action, continuation, args.horizon)
          error = max(error, err)
          payoffs.append(true)
          payoff_hashes.append(digest)
        payoffs = np.asarray(payoffs)
        result.update(position=t, next_reward=data['reward'][t + 1],
            fixed_reward=fixed_reward, true_q_h1=payoffs[:-1, 0],
            true_p_h1=payoffs[-1, 0], true_q_horizon=payoffs[:-1].mean(-1),
            true_p_horizon=payoffs[-1].mean(), physics_repeat_error=error,
            reference_endstate_hash=reference_hash, payoff_endstate_hash=payoff_hashes,
            background_frac=float((~data['preserve'][t]).mean()),
            body_frac=float(data['body'][t].mean()))
        if prefixes:
          result.update(consequence_prefixes(payoffs, prefixes))
          result['true_reward_sequences'] = payoffs
          result['reference_reward_sequence'] = reference
          result['reference_full_window_verified'] = True
          result['reference_prefix_sum'] = np.asarray([
              reference[:h].sum() for h in prefixes])
        records.append(result)
      target = output / f'episode_{entry["episode"]:03}.npz'
      if target.exists():
        raise FileExistsError(target)
      np.savez_compressed(target, **{k: np.asarray([r[k] for r in records])
          for k in records[0]}, bins=bins)
      artifacts.append(dict(episode=entry['episode'], split=entry['split'],
          path=target.name, sha256=sha256(target), positions=len(records)))
      print(json.dumps(dict(stage='score', **artifacts[-1])), flush=True)
    finally:
      env.close()
  if parameter_digest(agent) != metadata['params_sha256']:
    raise AssertionError('Frozen checkpoint parameters changed during scoring')
  write_json(output / 'score_complete.json', dict(artifacts=artifacts,
      params_unchanged=True, elapsed_seconds=time.perf_counter() - started,
      plan_sha256=sha256(output / 'score_plan.json')))


def analyze(args):
  source = Path(args.scores)
  plan = json.loads((source / 'score_plan.json').read_text())
  if plan.get('horizon', 10) != 10 or plan.get('opportunity_version'):
    raise ValueError('DL-protocol-r1 analysis requires original H10 labels')
  complete = json.loads((source / 'score_complete.json').read_text())
  if sha256(source / 'score_plan.json') != complete['plan_sha256']:
    raise ValueError('Scoring plan hash mismatch')
  pieces = []
  reference_bins = None
  for entry in complete['artifacts']:
    path = source / entry['path']
    if sha256(path) != entry['sha256']:
      raise ValueError('Score artifact changed')
    with np.load(path, allow_pickle=False) as file:
      piece = {k: file[k] for k in file.files if k != 'bins'}
      bins = file['bins']
    if reference_bins is None:
      reference_bins = bins.copy()
    elif not np.array_equal(reference_bins, bins):
      raise ValueError('Reward bins changed across frozen score artifacts')
    piece['episode'] = np.full(entry['positions'], entry['episode'])
    piece['blind'] = np.full(entry['positions'], entry['split'] == 'blind')
    pieces.append(piece)
  data = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
  blind, calibration = data['blind'], ~data['blind']
  if not blind.any() or not calibration.any():
    raise ValueError('Both declared splits are required')
  episode, tau = data['episode'], plan['free_nats']
  C = 1 - data['v']
  utility = data['true_q_horizon'][:, 0] - data['true_p_horizon']
  positive, negative = utility >= .01, utility <= 0
  labelled = blind & (positive | negative)
  sufficient = ((blind & positive).sum() >= 128 and (blind & negative).sum() >= 128
      and len(np.unique(episode[blind & positive])) >= 8
      and len(np.unique(episode[blind & negative])) >= 8)
  interval = lambda x, mask: cluster_interval(x[mask], episode[mask],
      draws=args.bootstrap, seed=args.seed)
  result = dict(code_version=VERSION, git_sha=git_sha(), score_plan=plan,
      score_complete_sha256=sha256(source / 'score_complete.json'),
      statistical_unit='independent episode; observations within episode clustered',
      label='q/p first mean action, common recorded continuation; >=.01 positive, <=0 negative',
      blind_episodes=len(np.unique(episode[blind])), calibration_episodes=len(
          np.unique(episode[calibration])), blind_positions=int(blind.sum()),
      positive_positions=int((blind & positive).sum()), negative_positions=int((blind & negative).sum()),
      gray_positions=int((blind & ~(positive | negative)).sum()),
      physics_max_repeat_error=float(data['physics_repeat_error'].max()),
      background_available=bool((data['background_frac'][blind] > 0).all()),
      body_available=bool((data['body_frac'][blind] > 0).all()),
      sufficient_label_coverage=bool(sufficient))
  result['signal_auc'] = {}
  candidates = {key: data[key][:, 0] for key in (
      'e', 'signed_support', 'loss_gap', 'D', 'K', 'f_D')}
  candidates['C'] = C[:, 0]
  for key, all_values in candidates.items():
    values, labels = all_values[labelled], positive[labelled]
    result['signal_auc'][key] = cluster_interval(values, episode[labelled],
        statistic=lambda idx, v=values, y=labels: auc(y[idx], v[idx]),
        draws=args.bootstrap, seed=args.seed)
  actual, dt = C[labelled, 0], data['D'][labelled, 0]
  labels = positive[labelled]
  result['paired_auc_C_minus_D'] = cluster_interval(actual, episode[labelled],
      statistic=lambda idx: (auc(labels[idx], actual[idx]) - auc(labels[idx], dt[idx]))
      if np.any(labels[idx]) and np.any(~labels[idx]) else None,
      draws=args.bootstrap, seed=args.seed)
  reward_score = data['e'][labelled, 0]
  result['paired_auc_C_minus_reward_only'] = cluster_interval(actual, episode[labelled],
      statistic=lambda idx: (auc(labels[idx], actual[idx]) - auc(labels[idx], reward_score[idx]))
      if np.any(labels[idx]) and np.any(~labels[idx]) else None,
      draws=args.bootstrap, seed=args.seed)
  result['nuisance_delta_C'] = {VARIANTS[v]: interval(C[:, v] - C[:, 0], blind)
      for v in (1, 2)}
  cue_cost = data['true_q_horizon'][:, 0] - data['true_q_horizon'][:, 3]
  useful_cue = blind & (cue_cost >= .01)
  cue_coverage = int(useful_cue.sum()) >= 128 and len(np.unique(episode[useful_cue])) >= 8
  result['cue_deprivation_all_delta_C'] = interval(C[:, 0] - C[:, 3], blind)
  result['cue_deprivation_true_cost'] = interval(cue_cost, blind)
  result['useful_cue_positions'] = int(useful_cue.sum())
  result['useful_cue_episodes'] = len(np.unique(episode[useful_cue]))
  result['useful_cue_retention_delta_C'] = interval(C[:, 0] - C[:, 3], useful_cue)
  active = blind & (data['K'][:, 0] > tau) & data['m'][:, 0].astype(bool)
  excess = np.maximum(data['K'][:, 0] - tau, 0)
  result['active_positions'] = int(active.sum())
  result['active_episodes'] = len(np.unique(episode[active]))
  result['active_protection'] = interval(C[:, 0], active)
  result['weighted_excess_release'] = interval(C[:, 0] * excess, blind)
  result['weighted_excess_release_fraction'] = (float(np.sum(C[blind, 0] * excess[blind]) /
      np.sum(excess[blind])) if np.sum(excess[blind]) > 0 else None)
  result['mode_margin_min'] = interval(data['q_margin_min'][:, 0], blind)
  result['near_mode_tie_fraction'] = interval((data['q_margin_min'][:, 0] < .01).astype(float), blind)
  mode_affected = blind & data['mode_perturb_applied'][:, 0].astype(bool)
  result['mode_fragility'] = dict(affected_positions=int(mode_affected.sum()),
      affected_episodes=len(np.unique(episode[mode_affected])),
      absolute_delta_C=interval(np.abs(data['mode_perturb_C'][:, 0] - C[:, 0]), mode_affected),
      absolute_delta_e=interval(np.abs(data['mode_perturb_e'][:, 0] - data['e'][:, 0]), mode_affected),
      q_perturb_K=interval(data['mode_perturb_q_K'][:, 0], mode_affected),
      p_perturb_K=interval(data['mode_perturb_p_K'][:, 0], mode_affected),
      limitation='Local top-two swap diagnostic; no posterior authenticity label')
  result['fixed_action_support'] = {}
  for key in ('e', 'signed_support'):
    result['fixed_action_support'][key] = interval(data['fixed_' + key][:, 0].mean(-1), blind)
    result['fixed_action_support'][key + '_minus_behavior'] = interval(
        data['fixed_' + key][:, 0].mean(-1) - data[key][:, 0], blind)
  result['label_shuffle'] = {}
  for split_name, mask in (('calibration', calibration), ('blind', blind)):
    rewards, _ = different_episode_labels(data['next_reward'][mask], episode[mask], args.seed)
    shuffled = compatibility_numpy(data['raw_logits_q'][mask, 0],
        data['raw_logits_p'][mask, 0], rewards, bins)
    result['label_shuffle'][split_name] = {key: cluster_interval(
        shuffled[key], episode[mask], draws=args.bootstrap, seed=args.seed)
        for key in ('e', 'signed_support', 'loss_gap')}
    result['label_shuffle'][split_name]['real_minus_shuffled_e'] = cluster_interval(
        data['e'][mask, 0] - shuffled['e'], episode[mask], draws=args.bootstrap, seed=args.seed)
  # Only calibration data determine bins and strength, never the blind outcomes.
  d_bins = np.unique(np.quantile(data['D'][calibration, 0], [1/3, 2/3])).tolist()
  active_calibration = calibration & (data['K'][:, 0] > tau)
  k_bins = ([float(np.median(data['K'][active_calibration, 0]))]
      if active_calibration.any() else [])
  base = excess * data['f_D'][:, 0] * data['m'][:, 0]
  denominator = np.sum(base[calibration])
  kappa = (float(plan['rho'] * np.sum(base[calibration] * data['e'][calibration, 0]) /
      denominator) if denominator > 0 else None)
  reward_base = excess * data['e'][:, 0] * data['m'][:, 0]
  reward_denominator = np.sum(reward_base[calibration])
  reward_lambda = (float(plan['rho'] * np.sum(base[calibration] * data['e'][calibration, 0]) /
      reward_denominator) if reward_denominator > 0 else None)
  result['calibration_only'] = dict(d_bins=d_bins, k_bins=k_bins, kappa=kappa,
      reward_only_lambda=reward_lambda,
      source='independent frozen diagnostics; not training-window c matching',
      warning='offline diagnostic estimate; training strength protocol must declare its own source')
  std_fraction = np.divide(data['D_std_component'][:, 0], data['D'][:, 0],
      out=np.zeros(len(C)), where=data['D'][:, 0] > 0)
  high_d = blind & (data['D'][:, 0] >= d_bins[-1]) & (data['D'][:, 0] > 0)
  result['policy_mean_std_diagnostic'] = dict(
      D_mean_component=interval(data['D_mean_component'][:, 0], blind),
      D_std_component=interval(data['D_std_component'][:, 0], blind),
      high_D_std_dominated_fraction=interval((std_fraction > .5).astype(float), high_d),
      limitation='True mean-action payoffs do not validate differences in policy variance. '
      'Variance-dominated discrepancies require common-noise sampled-action follow-up.')
  std_dominated = bool(high_d.any() and (std_fraction[high_d] > .5).mean() >= .5)
  result['policy_mean_std_diagnostic']['std_dominates_majority_high_D'] = std_dominated
  result['offline_strength_controls'] = {}
  for name, weight in (('S', kappa), ('W', reward_lambda)):
    if weight is None:
      result['offline_strength_controls'][name] = dict(status='uncalibrated')
      continue
    comparison_C = data['m'][:, 0] * weight * (
        data['f_D'][:, 0] if name == 'S' else data['e'][:, 0])
    result['offline_strength_controls'][name] = dict(status='calibrated',
        calibration_release_difference=float(np.sum((comparison_C - C[:, 0])[
            calibration] * excess[calibration])),
        blind_release=interval(comparison_C * excess, blind))
  # Diagnostic-only permutation, separate PRNG and calibration/blind pools.
  # Training performs its own within-current-batch permutation of complete v.
  rng = np.random.default_rng(args.seed + 0x4450)
  perm = np.arange(len(C))
  group_sizes = np.zeros(len(C), int)
  active_all = data['K'][:, 0] > tau
  db = np.searchsorted(d_bins, data['D'][:, 0], side='right')
  kb = np.searchsorted(k_bins, data['K'][:, 0], side='right')
  for split in (False, True):
    for activity in (False, True):
      for d in range(len(d_bins) + 1):
        for k in range(len(k_bins) + 1):
          idx = np.flatnonzero((blind == split) & (active_all == activity) &
              (db == d) & (kb == k) & data['m'][:, 0].astype(bool))
          group_sizes[idx] = len(idx)
          perm[idx] = rng.permutation(idx)
  protected = active & (C[:, 0] > 1e-6)
  moved = float((perm[protected] != np.flatnonzero(protected)).mean()) if protected.any() else None
  changed_values = (float((np.abs(C[perm[protected], 0] - C[protected, 0]) > 1e-6).mean())
      if protected.any() else None)
  original_release = float(np.sum(C[blind, 0] * excess[blind]))
  perm_release = float(np.sum(C[perm[blind], 0] * excess[blind]))
  changed_release = abs(perm_release - original_release) / original_release if original_release > 0 else None
  permutation_valid = (changed_values is not None and changed_values >= .5 and
      changed_release is not None and changed_release <= .10 and len(k_bins) > 0)
  result['offline_permutation_control'] = dict(
      status='usable' if permutation_valid else 'inconclusive',
      active_protected_moved_fraction=moved, blind_weighted_excess_relative_change=changed_release,
      active_protected_value_changed_fraction=changed_values,
      blind_moved_release_mass=float(np.sum(np.abs(C[perm[blind], 0] - C[blind, 0]) * excess[blind])),
      blind_singleton_fraction=float((group_sizes[blind] == 1).mean()),
      warning='Offline diagnostic only; does not certify training-batch control validity')
  result['blind_strata'] = []
  for d in range(len(d_bins) + 1):
    for k in range(len(k_bins) + 1):
      mask = labelled & (db == d) & (kb == k)
      if mask.any():
        result['blind_strata'].append(dict(d_bin=d, k_bin=k, count=int(mask.sum()),
            episode_count=len(np.unique(episode[mask])), auc_e=auc(positive[mask], data['e'][mask, 0]),
            positive_count=int(positive[mask].sum()), negative_count=int(negative[mask].sum()),
            auc_C=auc(positive[mask], C[mask, 0]), auc_D=auc(positive[mask], data['D'][mask, 0]),
            mean_signed_support=float(data['signed_support'][mask, 0].mean())))
  signal = result['signal_auc']['C']
  supported = (sufficient and signal['lower95'] is not None and signal['lower95'] > .5
      and signal['point'] >= .60)
  nuisance = result['background_available'] and all(x['upper95'] is not None and
      x['upper95'] <= .01 for x in result['nuisance_delta_C'].values())
  cue = result['useful_cue_retention_delta_C']
  cue_supported = (result['body_available'] and cue_coverage and cue['lower95'] is not None and
      cue['lower95'] > 0 and cue['point'] >= .01)
  release = result['weighted_excess_release_fraction']
  mechanism = int(active.sum()) >= 128 and result['active_episodes'] >= 8 and release is not None and release >= .01
  increment = result['paired_auc_C_minus_D']
  increment_supported = sufficient and increment['lower95'] is not None and increment['lower95'] > 0
  result['screening'] = dict(signal='supported' if supported else (
      'inconclusive' if not sufficient or signal['lower95'] is None else 'not_supported'),
      nuisance='supported' if nuisance else ('inconclusive' if not result[
          'background_available'] else 'not_supported'),
      cue_deprivation='supported' if cue_supported else ('inconclusive' if not result[
          'body_available'] or not cue_coverage or cue['lower95'] is None else 'not_supported'),
      active_release='supported' if mechanism else ('not_supported' if
          int(active.sum()) >= 128 and result['active_episodes'] >= 8 and
          release is not None else 'inconclusive'),
      added_signal_beyond_D='supported' if increment_supported else 'inconclusive',
      full_policy_distribution='inconclusive' if std_dominated else 'mean_action_scope_only',
      eligible_for_short_training_review=bool(supported and nuisance and cue_supported and
          mechanism and increment_supported and permutation_valid and kappa is not None and
          reward_lambda is not None and not std_dominated),
      proceed_short_training=False,
      remaining_engineering='Size50m DL-B, segmentation visual review and training controls '
      'must be independently audited; this analysis cannot authorize training',
      limitation='Single task/checkpoint screening; no formal performance conclusion. '
      'Cue deprivation and finite action utility do not prove posterior causal correctness.')
  write_json(args.output, result)
  print(json.dumps(result['screening'], sort_keys=True), flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest='stage', required=True)
  for name, function in (('collect', collect), ('score', score)):
    command = sub.add_parser(name)
    command.set_defaults(function=function)
    command.add_argument('--checkpoint', required=True)
    command.add_argument('--config', required=True)
    command.add_argument('--task', default='dmc_quadruped_walk')
    command.add_argument('--output', required=True)
    command.add_argument('--platform', choices=('cpu', 'cuda'), default='cuda')
    command.add_argument('--resource-record', required=True,
        help='fresh assigned sv3 GPU4--7 record with actual CUDA/EGL probe evidence')
    command.add_argument('--seed', type=int, default=20261010)
  collect_parser = sub.choices['collect']
  collect_parser.add_argument('--episodes', type=int, default=32)
  collect_parser.add_argument('--calibration-episodes', type=int, default=8)
  score_parser = sub.choices['score']
  score_parser.add_argument('--dataset', required=True)
  score_parser.add_argument('--samples', type=int, default=32)
  score_parser.add_argument('--horizon', type=int, default=10)
  score_parser.add_argument('--alpha', type=float, default=20)
  score_parser.add_argument('--rho', type=float, default=.5)
  score_parser.add_argument('--consequence-prefixes', type=int, nargs='+', default=[],
      help='DL-opportunity-r1 external H1/10/50/100 labels; H1 gate remains unchanged')
  analyze_parser = sub.add_parser('analyze')
  analyze_parser.set_defaults(function=analyze)
  analyze_parser.add_argument('--scores', required=True)
  analyze_parser.add_argument('--output', required=True)
  analyze_parser.add_argument('--bootstrap', type=int, default=2000)
  analyze_parser.add_argument('--seed', type=int, default=17)
  args = parser.parse_args()
  args.function(args)


if __name__ == '__main__':
  main()
