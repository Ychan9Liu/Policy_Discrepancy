"""DL-engineering-r3: real-image size50m engineering and compression inspection.

Runs isolated paired groups from a trusted checkpoint and independently
collected episode NPZs. No environment construction, rendering, or training
continuation occurs. A single update is replayed from the same checkpoint to
measure warmed execution and verify restored state. Parameter tensors stay in
memory; artifacts contain hashes and small per-position diagnostics only.
This is real-image engineering evidence, never method-effectiveness evidence.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import re
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VERSION = 'DL-engineering-r3'
GROUPS = ('off', 'alpha0', 'rho0', 'logging', 'dtlatch')


def sha256(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for block in iter(lambda: stream.read(2 ** 20), b''):
      digest.update(block)
  return digest.hexdigest()


def git_state():
  def command(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
  return dict(sha=command('rev-parse', 'HEAD'),
      status=command('status', '--porcelain'), diff_sha256=hashlib.sha256(
          command('diff', '--binary', 'HEAD').encode()).hexdigest())


def write_json(path, data):
  path = Path(path)
  if path.exists():
    raise FileExistsError(path)
  path.write_text(json.dumps(data, sort_keys=True, indent=2,
      allow_nan=False) + '\n', encoding='utf-8')


def checkpoint_file(path):
  path = Path(path)
  if path.is_dir() and (path / 'latest').exists():
    path = path / (path / 'latest').read_text(encoding='utf-8').strip()
  if path.is_dir():
    path = path / 'agent.pkl'
  if not path.is_file():
    raise FileNotFoundError(path)
  return path.resolve()


def load_config(args, group):
  import elements
  from ruamel.yaml import YAML
  yaml = YAML(typ='safe')
  config = elements.Config(yaml.load(Path(args.config).read_text(encoding='utf-8')))
  defaults = yaml.load((ROOT / 'dreamerv3/configs.yaml').read_text(encoding='utf-8'))
  dl = elements.Config(defaults['defaults']).agent.dt_latch
  rssm = config.agent.dyn.rssm
  expected = dict(deter=4096, hidden=512, stoch=32, classes=32, unimix=.01,
      blocks=8, imglayers=2, obslayers=1, dynlayers=1, absolute=False,
      norm='rms', act='silu')
  if any(getattr(rssm, k) != v for k, v in expected.items()):
    raise ValueError('Checkpoint actual configuration is not supported size50m RSSM')
  for head in ('enc.simple', 'dec.simple', 'rewhead', 'conhead', 'policy', 'value'):
    flat = config.agent.flat
    if flat[f'{head}.units'] != 512:
      raise ValueError(f'Unexpected size50m {head}.units')
  if (config.agent.enc.typ != 'simple' or config.agent.dec.typ != 'simple' or
      config.agent.dyn.typ != 'rssm' or config.agent.enc.simple.depth != 32 or
      config.agent.dec.simple.depth != 32 or
      config.agent.policy_dist_cont != 'bounded_normal' or
      config.agent.rewhead.output != 'symexp_twohot' or
      config.agent.rewhead.bins != 255 or config.env.dmc.proprio or
      not config.env.dmc.image or tuple(config.env.dmc.size) != (64, 64) or
      config.env.dmc.repeat != 1 or config.batch_size != 16 or
      config.batch_length != 64 or config.replay_context != 1):
    raise ValueError('Requires production vision64 size50m, B16/T64/context1/repeat1')
  if config.agent.dyn.rssm.free_nats != 1 or config.agent.loss_scales.rep != .1 or (
      config.agent.loss_scales.dyn != 1 or config.agent.ac_grads):
    raise ValueError('Requires frozen original KL/gradient protocol')
  changes = {'logdir': str(args.output), 'jax.platform': args.platform,
      'jax.policy_devices': [0], 'jax.train_devices': [0], 'jax.prealloc': False,
      'jax.precompile': False, 'jax.profiler': False, 'jax.enable_policy': False,
      'agent.rep_probe.mode': 'off', **{
          f'agent.dt_latch.{k}': v for k, v in dl.flat.items()},
      'agent.dt_latch.mode': 'off' if group == 'off' else
          'logging' if group == 'logging' else 'dtlatch',
      'agent.dt_latch.alpha': 0. if group == 'alpha0' else args.alpha,
      'agent.dt_latch.rho': 0. if group == 'rho0' else args.rho}
  return elements.Config({**config.flat, **changes})


def episode_batch(dataset, batch_size=16, length=65, split='blind'):
  """Outcome-independent first windows of distinct complete reset episodes."""
  dataset = Path(dataset)
  plan_path, complete_path = dataset / 'collection_plan.json', dataset / 'collection_complete.json'
  plan = json.loads(plan_path.read_text(encoding='utf-8'))
  complete = json.loads(complete_path.read_text(encoding='utf-8'))
  if complete['plan_sha256'] != sha256(plan_path) or not complete['params_unchanged']:
    raise ValueError('Collection completion/plan integrity check failed')
  candidates = sorted((x for x in complete['artifacts'] if x['split'] == split),
      key=lambda x: x['episode'])
  if len(candidates) < batch_size:
    raise ValueError(f'Require {batch_size} distinct {split} episodes; found {len(candidates)}')
  chosen = candidates[:batch_size]
  rows = []
  keys = ('image', 'reward', 'is_first', 'is_last', 'is_terminal', 'action')
  for item in chosen:
    path = (dataset / item['path']).resolve()
    if not path.is_relative_to(dataset.resolve()) or sha256(path) != item['sha256']:
      raise ValueError('Episode path/hash integrity failed')
    with np.load(path, allow_pickle=False) as episode:
      if any(len(episode[k]) < length for k in keys):
        raise ValueError('Chosen episode is too short; no outcome-dependent replacement')
      row = {k: episode[k][:length].copy() for k in keys}
    if row['image'].dtype != np.uint8 or row['image'].shape != (length, 64, 64, 3):
      raise ValueError('Requires real uint8 RGB 64x64 episode frames')
    if not row['is_first'][0] or np.any(row['is_first'][1:]):
      raise ValueError('Window must begin at exactly one genuine episode reset')
    if any(not np.isfinite(x).all() for x in row.values()):
      raise ValueError('Nonfinite real episode data')
    rows.append(row)
  batch = {k: np.stack([r[k] for r in rows]) for k in keys}
  if len({x['episode'] for x in chosen}) != batch_size:
    raise ValueError('Duplicate episode IDs')
  return batch, dict(plan=plan, plan_sha256=sha256(plan_path),
      complete_sha256=sha256(complete_path), split=split, episodes=chosen,
      source_window='first65 frames of distinct reset episodes; no repetition/padding')


def make_agent(config, batch):
  """Production model, spaces read from recorded real data, never make_env."""
  import elements
  from dreamerv3.agent import Agent
  obs_space = {k: elements.Space(batch[k].dtype, batch[k].shape[2:])
      for k in ('image', 'reward', 'is_first', 'is_last', 'is_terminal')}
  act_space = {'action': elements.Space(batch['action'].dtype, batch['action'].shape[2:])}
  if batch['action'].dtype != np.float32 or len(act_space['action'].shape) != 1:
    raise ValueError('Requires float32 continuous vector actions')
  return Agent(obs_space, act_space, elements.Config(**config.agent,
      logdir=config.logdir, seed=config.seed, jax=config.jax,
      batch_size=config.batch_size, batch_length=config.batch_length,
      replay_context=config.replay_context, report_length=config.report_length,
      replica=config.replica, replicas=config.replicas))


def reconstruct_context_entries(model, carry, obs, prevact):
  """Rebuild observed prefix entries only; no future frame or model update."""
  import elements
  ec, dc, cc, _ = carry
  ec, ee, tokens = model.enc(ec, obs, obs['is_first'], False)
  dc, de, feat = model.dyn.observe(dc, tokens, prevact, obs['is_first'], False)
  cc, ce, _ = model.dec(cc, feat, obs['is_first'], False)
  return elements.tree.flatdict(dict(enc=ee, dyn=de, dec=ce))


def probe_shared_gradients(model, repfeat, obs, prevact, alpha, rho):
  """Impure Ninjax helper: differentiate every actually read probe parameter.

  Ninjax rejects requested state entries that the objective does not read.
  Track its real accesses before selecting exact leaf targets; state entries
  not read by this objective are independent of it, not directly gradient
  tested. No model state may be created or modified by the detached probe.
  """
  import ninjax as nj
  from dreamerv3 import dt_latch

  def objective(feat, ob, pa):
    prior = model.dyn._prior(feat['deter'])
    signal = dt_latch.probe(model, feat, prior, ob, pa, alpha, rho)
    return sum(signal[k].sum() for k in ('D', 'ell_q', 'ell_p', 'e', 'v'))

  _, _, accessed, modified, created = nj.pure(objective, nested=True)(
      dict(nj.context()), repfeat, obs, prevact,
      create=False, modify=False, track=True)
  if modified or created:
    raise AssertionError('Detached probe attempted to create or modify model state')
  if not accessed:
    raise AssertionError('Detached probe did not access any shared parameters')
  # Escape full leaf names: Ninjax accepts regular-expression state targets.
  targets = tuple(re.escape(key) for key in sorted(accessed))
  return nj.grad(objective, targets)(repfeat, obs, prevact)


def worker(args):
  if os.name == 'nt' and args.platform == 'cuda':
    raise RuntimeError('Real CUDA verification runs on assigned Linux server only')
  if git_state()['status']:
    raise RuntimeError('Size50m worker requires clean committed checkout')
  resource = None
  if args.platform == 'cuda':
    from scripts.dl_resources import require_resource, require_session, verify_runtime
    if args.resource_session:
      resource = require_session(args.resource_session)
    elif args.resource_record:
      resource = require_resource(args.resource_record)
    else:
      raise ValueError('CUDA worker requires a current resource record or live paired session')
  import elements
  import embodied.jax.internal as internal
  import jax
  import jax.numpy as jnp
  import ninjax as nj
  import optax
  from dreamerv3 import dt_latch
  from embodied.run.protocol_v1 import array_hashes, tree_hash
  output = Path(args.output)
  output.mkdir(parents=True, exist_ok=False)
  config = load_config(args, args.worker)
  batch, provenance = episode_batch(args.dataset, split=args.split)
  checkpoint_path = checkpoint_file(args.checkpoint)
  checkpoint_sha = sha256(checkpoint_path)
  if checkpoint_sha != provenance['plan']['checkpoint_sha256'] or config.task != (
      provenance['plan']['task']):
    raise ValueError('Dataset checkpoint/task does not match supplied checkpoint')
  if sha256(args.config) != provenance['plan']['config_source_sha256']:
    raise ValueError('Actual configuration source differs from collection source')
  with checkpoint_path.open('rb') as file:
    checkpoint = pickle.load(file)  # Trusted project checkpoint only.
  start = time.perf_counter()
  agent = make_agent(config, batch)
  if resource is not None:
    resource['runtime_compute'] = verify_runtime(resource)
  agent.load(checkpoint)
  init_seconds = time.perf_counter() - start
  model, b, length = agent.model, config.batch_size, config.batch_length + 1
  raw = agent._zeros(agent.spaces, (b, length))
  raw.update(batch)
  # These are reconstructed cached entries, not original replay restoration.
  raw['consec'][:] = 0
  for row, episode in enumerate(provenance['episodes']):
    for col in range(length):
      raw['stepid'][row, col] = np.frombuffer(hashlib.sha256(
          f"{episode['sha256']}:{col}".encode()).digest()[:20], np.uint8)
  carry = agent.init_train(b)
  c = config.replay_context
  prefix_obs = {k: raw[k][:, :c] for k in agent.obs_space}
  prefix_act = {k: np.zeros_like(raw[k][:, :c]) for k in agent.act_space}
  prefix_obs, prefix_act = internal.device_put((prefix_obs, prefix_act), agent.train_sharded)
  cache_seed = agent._seeds(args.cache_seed_counter, agent.train_mirrored)
  cache_fn = jax.jit(nj.pure(lambda ca, ob, pa: reconstruct_context_entries(
      model, ca, ob, pa)))
  start = time.perf_counter()
  cache_state, entries = cache_fn(agent.params, carry, prefix_obs, prefix_act, seed=cache_seed)
  jax.block_until_ready(entries)
  cache_seconds = time.perf_counter() - start
  with jax._src.config.explicit_device_get_scope():
    for key, value in entries.items():
      raw[key][:, :c] = np.asarray(value)
  prepared = agent.prepare_batch(raw)
  seed = prepared['seed']
  loss_carry, obs, prevact, _ = jax.jit(model._apply_replay_context)(carry,
      {k: v for k, v in prepared.items() if k != 'seed'})
  def host(tree):
    with jax._src.config.explicit_device_get_scope():
      return jax.tree.map(lambda x: np.asarray(x), jax.device_get(tree))
  def hashes(tree):
    return array_hashes(host(tree))
  initial_hashes = hashes(agent.params)
  if initial_hashes != array_hashes(checkpoint['params']):
    raise AssertionError('Loaded full checkpoint parameters/state changed')
  if hashes(cache_state) != initial_hashes:
    raise AssertionError('Context reconstruction changed checkpoint parameters/state')
  grad_fn = jax.jit(nj.pure(lambda c, o, p: nj.grad(model.loss, model.modules,
      has_aux=True)(c, o, p, True)))
  start = time.perf_counter()
  _, (total_loss, _, grads, aux) = grad_fn(agent.params, loss_carry, obs, prevact, seed=seed)
  jax.block_until_ready((grads, aux))
  grad_seconds = time.perf_counter() - start
  repfeat = aux[2]['repfeat']
  def signals(feat, obs, prevact):
    prior = model.dyn._prior(feat['deter'])
    raw_k = model.dyn._dist(feat['logit']).kl(model.dyn._dist(jax.lax.stop_gradient(prior)))
    signals = dt_latch.probe(model, feat, prior, obs, prevact, args.alpha, args.rho)
    return prior, raw_k, signals
  signal_fn = jax.jit(nj.pure(signals))
  signal_state, (prior, raw_k, dl) = signal_fn(agent.params, repfeat, obs, prevact, seed=seed)
  if hashes(signal_state) != initial_hashes:
    raise AssertionError('Detached probe changed model state')
  probe_grad_fn = jax.jit(nj.pure(lambda f, o, p: probe_shared_gradients(
      model, f, o, p, args.alpha, args.rho)))
  _, (_, _, probe_grads) = probe_grad_fn(agent.params, repfeat, obs, prevact, seed=seed)
  norm = jax.jit(optax.global_norm)
  probe_norm = float(host(norm(probe_grads)))
  probe_all_zero = bool(host(jax.jit(lambda tree: jnp.all(jnp.stack([
      jnp.all(value == 0) for value in jax.tree.leaves(tree)])))(probe_grads)))
  if probe_norm != 0 or not probe_all_zero:
    raise AssertionError('Probe leaked gradient into shared model parameters')
  def other_loss(c, o, p):
    losses = model.loss(c, o, p, True)[1][2]['losses']
    return sum(x.mean() * model.scales[k] for k, x in losses.items() if k != 'rep')
  other_fn = jax.jit(nj.pure(lambda c, o, p: nj.grad(other_loss,
      model.modules)(c, o, p)))
  _, (_, _, other_grads) = other_fn(agent.params, loss_carry, obs, prevact, seed=seed)
  actual_v = jax.jit(lambda v: v if args.worker == 'dtlatch' else
      jnp.ones_like(v))(dl['v'])
  # Float32 independent posterior logits isolate the local rep gradient.
  post, frozen_prior = jax.jit(lambda q, p: (jnp.float32(q),
      jax.lax.stop_gradient(p)))(repfeat['logit'], prior)
  def local_rep(q):
    k = model.dyn._dist(q).kl(model.dyn._dist(frozen_prior))
    return jnp.maximum(k, model.dyn.free_nats)
  base_local = jax.jit(jax.grad(lambda q: local_rep(q).mean()))(post)
  gated_local = jax.jit(jax.grad(lambda q: (jax.lax.stop_gradient(actual_v) *
      local_rep(q)).mean()))(post)
  expected_local = jax.jit(lambda v, g: v[..., None, None] * g)(actual_v, base_local)
  g1, ge = host(gated_local), host(expected_local)
  np.testing.assert_allclose(g1, ge, rtol=2e-6, atol=1e-8)
  first_update = time.perf_counter()
  agent.train(carry, dict(prepared))
  agent.take_train_result()
  jax.block_until_ready(agent.params)
  update_seconds = time.perf_counter() - first_update
  updated_hashes = hashes(agent.params)
  # Re-load, repeat exactly one update; not a continuing training run.
  agent.load(checkpoint)
  carry2 = agent.init_train(b)
  prepared2 = agent.prepare_batch(raw)
  warm_start = time.perf_counter()
  agent.train(carry2, dict(prepared2))
  agent.take_train_result()
  jax.block_until_ready(agent.params)
  warm_seconds = time.perf_counter() - warm_start
  if hashes(agent.params) != updated_hashes:
    raise AssertionError('Restored checkpoint/optimizer/RNG update is not replayable')
  signal_host, raw_host, actual = host(dl), host(raw_k), host(actual_v)
  excess = np.maximum(raw_host - model.dyn.free_nats, 0)
  active = raw_host > model.dyn.free_nats
  summary = dict(code_version=VERSION, group=args.worker, git=git_state(),
      resource=resource,
      checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_sha,
      checkpoint_counters=checkpoint['counters'], actual_config=config.flat,
      config_source_sha256=sha256(args.config), provenance=provenance,
      shape=dict(batch_size=b, batch_length=64, replay_context=1),
      cache_seed_counter=args.cache_seed_counter, cache_rng=host(cache_seed).tolist(),
      train_rng=host(seed).tolist(), raw_input_hashes=array_hashes(raw),
      context_hashes=hashes(entries), initial_param_hashes=initial_hashes,
      updated_param_hashes=updated_hashes,
      normalization_initial_hashes={k: v for k, v in initial_hashes.items()
          if k.startswith(('retnorm/', 'valnorm/', 'advnorm/'))},
      normalization_updated_hashes={k: v for k, v in updated_hashes.items()
          if k.startswith(('retnorm/', 'valnorm/', 'advnorm/'))},
      sampled_latent_hashes=hashes(repfeat), gradient_hashes=hashes(grads),
      other_gradient_hashes=hashes(other_grads),
      total_loss=float(host(total_loss)), total_gradient_norm=float(host(norm(grads))),
      other_gradient_norm=float(host(norm(other_grads))),
      probe_shared_gradient_norm=probe_norm,
      probe_shared_gradient_all_zero=probe_all_zero,
      probe_gradient_tested_keys=sorted(probe_grads),
      probe_gradient_tested_count=len(probe_grads),
      probe_unaccessed_state_count=len(agent.params) - len(probe_grads),
      probe_gradient_scope='All actually accessed probe parameters; unread state entries are mathematically independent, not directly gradient tested',
      local_gradient_max_abs_error=float(np.max(np.abs(g1 - ge))),
      branch_losses={k: float(host(v).mean()) for k, v in aux[2]['losses'].items()},
      active_fraction=float(active.mean()), valid_active_fraction=float(
          (active & signal_host['m']).mean()),
      active_protected_fraction=float((active & (actual < 1 - 1e-6)).mean()),
      excess_release_mean=float(((1 - actual) * excess).mean()),
      candidate_excess_release_mean=float(((1 - signal_host['v']) * excess).mean()),
      actual_v_min=float(actual.min()), actual_v_mean=float(actual.mean()),
      timings_seconds=dict(init_load=init_seconds, cache_compile=cache_seconds,
          full_gradient_compile=grad_seconds, first_update=update_seconds,
          warmed_replayed_update=warm_seconds),
      optimizer_replay_equal=True, evidence='real vision + reconstructed context engineering only',
      environment=dict(python=platform.python_version(), system=platform.platform(),
          packages={p: importlib.metadata.version(p) for p in
              ('jax', 'jaxlib', 'elements', 'ninjax', 'numpy', 'optax', 'chex')}))
  np.savez_compressed(output / 'positions.npz', **{k: signal_host[k] for k in
      ('D', 'm', 'e', 'v', 'f_D', 'ell_q', 'ell_p', 'u_q', 'u_p', 'signed_support')},
      raw_kl=raw_host, actual_v=actual)
  write_json(output / 'summary.json', summary)
  print(json.dumps(dict(group=args.worker, complete=True,
      active_protected_fraction=summary['active_protected_fraction'],
      warmed_update_seconds=warm_seconds)), flush=True)


def paired(args):
  output = Path(args.output)
  output.mkdir(parents=True, exist_ok=False)
  if git_state()['status']:
    raise RuntimeError('Size50m execution requires clean committed checkout')
  session_path = None
  if args.platform == 'cuda':
    from scripts.dl_resources import start_session
    if not args.resource_record:
      raise ValueError('CUDA paired execution requires a fresh assigned resource record')
    session_path = output / 'resource-session.json'
    start_session(args.resource_record, session_path)
  for group in GROUPS:
    command = [sys.executable, str(Path(__file__).resolve()),
        '--worker', group, '--checkpoint', args.checkpoint, '--config', args.config,
        '--dataset', args.dataset, '--output', str(output / group),
        '--platform', args.platform, '--split', args.split,
        '--alpha', str(args.alpha), '--rho', str(args.rho),
        '--cache-seed-counter', str(args.cache_seed_counter)]
    if session_path:
      command += ['--resource-session', str(session_path)]
    subprocess.run(command, check=True, cwd=ROOT)
  summaries = {g: json.loads((output / g / 'summary.json').read_text(
      encoding='utf-8')) for g in GROUPS}
  base = summaries['off']
  errors = []
  for group, candidate in summaries.items():
    keys = ('checkpoint_sha256', 'raw_input_hashes', 'context_hashes',
        'initial_param_hashes', 'sampled_latent_hashes', 'train_rng', 'cache_rng')
    if group != 'dtlatch':
      keys += ('gradient_hashes', 'branch_losses', 'total_loss',
          'updated_param_hashes', 'other_gradient_hashes')
    for key in keys:
      if candidate[key] != base[key]:
        errors.append(f'{group}: {key} changed')
    if group == 'dtlatch':
      for key in base['branch_losses']:
        if key != 'rep' and candidate['branch_losses'][key] != base['branch_losses'][key]:
          errors.append(f'dtlatch: ordinary {key} loss changed')
      if candidate['other_gradient_hashes'] != base['other_gradient_hashes']:
        errors.append('dtlatch: ordinary non-rep parameter gradients changed')
  report = dict(code_version=VERSION, git=git_state(), checks_passed=not errors,
      errors=errors, local_gradient_rtol=2e-6, local_gradient_atol=1e-8,
      real_size50m_production_shape=True,
      compression_exercised=summaries['dtlatch']['active_protected_fraction'] > 0,
      limitation='Engineering and local compression only; no task-selection or performance conclusion')
  write_json(output / 'paired_check.json', report)
  if errors:
    raise AssertionError('; '.join(errors))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', required=True, help='trusted project agent checkpoint')
  parser.add_argument('--config', required=True, help='complete actual checkpoint run config')
  parser.add_argument('--dataset', required=True, help='DL collection directory')
  parser.add_argument('--output', required=True, help='new output directory')
  parser.add_argument('--platform', choices=('cuda', 'cpu'), default='cuda')
  parser.add_argument('--split', choices=('blind', 'calibration'), default='blind')
  parser.add_argument('--alpha', type=float, default=20.)
  parser.add_argument('--rho', type=float, default=.5)
  parser.add_argument('--cache-seed-counter', type=int, default=0x444C)
  parser.add_argument('--worker', choices=GROUPS)
  parser.add_argument('--resource-record',
      help='fresh assigned sv3 GPU4--7 resource record; required for CUDA execution')
  parser.add_argument('--resource-session', help=argparse.SUPPRESS)
  parser.add_argument('--validate-only', action='store_true',
      help='read config and shape/provenance without model, GPU, or checkpoint unpickle')
  args = parser.parse_args()
  if args.validate_only:
    config = load_config(args, args.worker or 'off')
    batch, provenance = episode_batch(args.dataset, split=args.split)
    path = checkpoint_file(args.checkpoint)
    if sha256(path) != provenance['plan']['checkpoint_sha256']:
      raise ValueError('Checkpoint hash differs from collection')
    if sha256(args.config) != provenance['plan']['config_source_sha256']:
      raise ValueError('Actual configuration source differs from collection source')
    print(json.dumps(dict(validated=True, task=config.task,
        shapes={k: list(v.shape) for k, v in batch.items()}, git=git_state())))
  elif args.worker:
    worker(args)
  else:
    paired(args)


if __name__ == '__main__':
  main()
