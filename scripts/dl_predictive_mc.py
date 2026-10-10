"""DL-predictive-MC-r1: finite, calibration-only latent-path integration.

The public CLI supervises a single read-only CUDA worker for at most 3600s.
No environment is constructed, blind array opened, gate fitted, or optimizer
updated. CPU functions are exposed for engineering and independent rescoring.
"""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_diagnostic as d
from scripts import dl_predictive as modal

VERSION = 'DL-predictive-MC-r1'
HISTORY_SEED, MC_SEED, STAT_SEED = 20261012, 20261017, 20261016
BLOCKS, PER_BLOCK, MAX_CHUNK, HORIZON, BUDGET = 4, 32, 8, 10, 3600
BRANCHES = ('clean', 'background_a', 'background_b', 'cue_deprivation', 'prior')
CHECKPOINT_SHA = '2ab693c8ded4d8f2186bd489aa8b77498448ba2f2c331090159da4e5f08056b9'
PARAMS_SHA = '14859c4e129b51aa488c10eab09f16d56fd110c76239b2c9072f415829d111c2'
CONFIG_SHA = '11428512fdc9b9e04a5384e419c5abcef93e5429db0a3327d05ccb0d4bd20445'
COLLECTION_SHA = '5765c893a419a59db46e5a3e3197af20b9059757'
MODAL_SHA = 'be9e32aa018d8b61c0b2931fe44f72a6f480d4b5'
MODAL_RESULT_SHA = 'cc4ccf9e4842cd6fe2e0e94a85f7be2fceecb50996aa68d7fda134f87c778d2a'
MODAL_PLAN_SHA = '4ac004571cd60211400a28d30c965556f2e127b1826e38b29db2fca6bce6b843'
MODAL_AUDIT_SHA = '4bd7e91af939d12834f989999dcccc3080577563eee1f28c50b11f77282a99d5'
GAPS = ('clean_minus_prior_log_score', 'clean_over_deprived_log_score')


def require(condition, message):
  if not condition:
    raise ValueError(message)


def array_hash(value):
  value = np.ascontiguousarray(value)
  return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() +
      value.tobytes()).hexdigest()


def safe_artifact(root, entry):
  path = (root / entry['path']).resolve()
  require(path.is_relative_to(root) and path.suffix == '.npz' and path.is_file(),
      'Unsafe or missing calibration artifact')
  require(d.sha256(path) == entry['sha256'], 'Calibration artifact hash differs')
  return path


def checkpoint_path(path):
  path = Path(path)
  if path.is_dir() and (path / 'latest').is_file():
    path /= (path / 'latest').read_text().strip()
  return path / 'agent.pkl' if path.is_dir() else path


def agent_from_spaces(config):
  """Exact fixed DMC/NormalizeAction/UnifyDtypes spaces, without make_env."""
  import elements
  from dreamerv3.agent import Agent
  obs_space = dict(image=elements.Space(np.uint8, (64, 64, 3)),
      reward=elements.Space(np.float32), is_first=elements.Space(bool),
      is_last=elements.Space(bool), is_terminal=elements.Space(bool))
  act_space = dict(action=elements.Space(np.float32, (12,), -1, 1))
  return Agent(obs_space, act_space, elements.Config(**config.agent,
      logdir=config.logdir, seed=config.seed, jax=config.jax,
      batch_size=config.batch_size, batch_length=config.batch_length,
      replay_context=config.replay_context, report_length=config.report_length,
      replica=config.replica, replicas=config.replicas))


def load_agent_no_env(args):
  from embodied.run.protocol_v1 import parameter_digest
  path = checkpoint_path(args.checkpoint)
  require(d.sha256(path) == CHECKPOINT_SHA and d.sha256(args.config) == CONFIG_SHA,
      'Trusted checkpoint/config byte verification must precede loading')
  config = d.load_config(args.config, args.task, args.output, args.platform)
  require(not config.random_agent, 'Require actual frozen Dreamer model')
  agent = agent_from_spaces(config)
  with path.open('rb') as file:
    checkpoint = pickle.load(file)  # Only the pinned trusted project checkpoint.
  agent.load(checkpoint)
  metadata = dict(checkpoint=str(path.resolve()), checkpoint_sha256=CHECKPOINT_SHA,
      checkpoint_counters=checkpoint['counters'], params_sha256=parameter_digest(agent),
      config=config.flat, config_source_sha256=CONFIG_SHA, git_sha=d.git_sha(),
      code_version=VERSION,
      space_contract='Production DMC vision64/propriofalse/action12 NormalizeAction + UnifyDtypes; no environment constructed')
  return agent, config, metadata


def load_sources(dataset, modal_scores, audit_path):
  """Verify metadata for all episodes; open arrays ONLY for eight calibration IDs."""
  source, scores = Path(dataset).resolve(), Path(modal_scores).resolve()
  result_path, score_plan_path = scores / 'result.json', scores / 'plan.json'
  audit_path = Path(audit_path)
  require(d.sha256(result_path) == MODAL_RESULT_SHA and
      d.sha256(score_plan_path) == MODAL_PLAN_SHA and
      d.sha256(audit_path) == MODAL_AUDIT_SHA, 'Pinned modal result/plan/audit hash differs')
  result = json.loads(result_path.read_text(encoding='utf-8'))
  score_plan = json.loads(score_plan_path.read_text(encoding='utf-8'))
  audit = json.loads(audit_path.read_text(encoding='utf-8'))
  plan_path, complete_path = source / 'collection_plan.json', source / 'collection_complete.json'
  plan = json.loads(plan_path.read_text(encoding='utf-8'))
  complete = json.loads(complete_path.read_text(encoding='utf-8'))
  expected = d.episode_plan(32, 8, HISTORY_SEED)
  calibration = [x for x in expected if x['split'] == 'calibration']
  metadata = score_plan['metadata']
  require(complete['plan_sha256'] == d.sha256(plan_path) and
      score_plan['source_complete_sha256'] == d.sha256(complete_path) and
      complete['params_unchanged'] is True, 'Collection completion proof differs')
  require(score_plan['source'] == plan and plan['git_sha'] == COLLECTION_SHA and
      plan['seed'] == HISTORY_SEED and plan['episodes'] == expected and
      plan['task'] == 'dmc_quadruped_walk' and plan['camera'] == 2 and
      plan['shape'] == [64, 64], 'Frozen collection or seed/window object differs')
  require(plan['checkpoint_sha256'] == metadata['checkpoint_sha256'] == CHECKPOINT_SHA and
      plan['params_sha256'] == metadata['params_sha256'] == PARAMS_SHA and
      plan['config_source_sha256'] == metadata['config_source_sha256'] == CONFIG_SHA and
      metadata['checkpoint_counters']['updates'] == 74493,
      'Frozen 300k checkpoint/config/parameter proof differs')
  require(score_plan['version'] == result['version'] == modal.VERSION and
      score_plan['horizon'] == result['primary_horizon'] == HORIZON and
      score_plan['controls'] == 6 and score_plan['prefixes'] == [1, 3, 10] and
      score_plan['diagnostic_seed'] == HISTORY_SEED and
      score_plan['statistical_seed'] == STAT_SEED and
      score_plan['selected_episodes'] == [x['episode'] for x in calibration] and
      score_plan['gate_changed'] is False and score_plan['optimizer_updates'] == 0 and
      result['analysis_git_sha'] == MODAL_SHA and result['params_unchanged'] is True and
      result['plan_sha256'] == MODAL_PLAN_SHA and result['positions'] == 128 and
      result['calibration_episodes'] == 8 and result['maximum_physics_repeat_error'] == 0,
      'Modal diagnostic is not the original finite calibration object')
  require(audit['source_result_sha256'] == MODAL_RESULT_SHA and
      audit['all_eight_calibration_artifact_hashes_verified'] is True and
      audit['blind_arrays_read'] is False and audit['prior_rollouts_equal'] is True and
      audit['fixed_controls_exact'] is True and audit['maximum_physics_repeat_error'] == 0,
      'Independent modal audit proof differs')
  artifacts = complete['artifacts']
  require(len(artifacts) == 32 and
      [{k: x[k] for k in ('episode', 'split', 'seed', 'policy_seed')} for x in artifacts] == expected and
      len({x['path'] for x in artifacts}) == 32, 'Collection artifacts differ from complete seed plan')
  modal_entries = result['evidence_artifacts']
  require(len(modal_entries) == 8 and [x['episode'] for x in modal_entries] ==
      [x['episode'] for x in calibration], 'Modal artifacts differ from calibration IDs')
  selected = []
  for definition, entry in zip(calibration, modal_entries):
    raw_entry = artifacts[definition['episode']]
    require(entry['positions'] == 16 and entry['source_sha256'] == raw_entry['sha256'],
        'Modal artifact source or position count differs')
    raw_path, score_path = safe_artifact(source, raw_entry), safe_artifact(scores, entry)
    # snapshots are deliberately not read or unpickled, and no blind NPZ is opened.
    with np.load(raw_path, allow_pickle=False) as file:
      raw = {k: file[k].copy() for k in ('image', 'preserve', 'body', 'reward',
          'is_first', 'is_last', 'is_terminal', 'action')}
    with np.load(score_path, allow_pickle=False) as file:
      scored = {k: file[k].copy() for k in file.files}
    frames = raw_entry['frames']
    require(type(frames) is int and len(raw['reward']) == frames and
        raw['image'].shape == (frames, 64, 64, 3) and raw['image'].dtype == np.uint8 and
        raw['action'].shape == (frames, 12) and raw['action'].dtype == np.float32 and
        np.isfinite(raw['action']).all() and np.isfinite(raw['reward']).all(),
        'Invalid calibration source image/action/reward contract')
    require(raw['preserve'].shape == raw['body'].shape == (frames, 64, 64) and
        raw['preserve'].dtype == raw['body'].dtype == np.bool_, 'Invalid saved image masks')
    for flag in ('is_first', 'is_last', 'is_terminal'):
      require(raw[flag].shape == (frames,) and raw[flag].dtype == np.bool_, 'Invalid source reset flags')
    require(raw['is_first'][0] and raw['is_last'][-1] and not raw['is_first'][1:].any() and
        not raw['is_last'][:-1].any() and not raw['is_terminal'][:-1].any(),
        'Calibration source is not one complete uninterrupted episode')
    positions = d.sample_positions(frames, 16, 100)
    require(len(positions) == 16 and np.array_equal(scored['position'], positions) and
        np.array_equal(scored['episode'], np.full(16, definition['episode'])),
        'Modal positions are not the outcome-independent source selection')
    truth, controls = scored['true_rewards'], scored['controls']
    require(truth.shape == (16, 6, 10) and truth.dtype == np.float64 and
        np.isfinite(truth).all() and controls.shape == (16, 6, 10, 12) and
        controls.dtype == np.float32 and np.isfinite(controls).all() and
        np.array_equal(scored['physics_repeat_error'], np.zeros(16)),
        'Invalid verified finite-control truth')
    require(scored['logits'].shape == (16, 4, 2, 6, 10, 255) and
        np.isfinite(scored['logits']).all() and scored['bins'].shape == (255,) and
        scored['bins'].dtype == np.float32 and np.all(np.diff(scored['bins']) > 0),
        'Invalid original modal reward distribution')
    require(np.array_equal(scored['logits'][:, :, 1],
        np.repeat(scored['logits'][:, :1, 1], 4, axis=1)), 'Original modal prior variants differ')
    for index, position in enumerate(positions):
      require(np.array_equal(controls[index], modal.control_sequences(raw['action'][position:position+10])) and
          np.array_equal(truth[index, 0].astype(np.float32), raw['reward'][position+1:position+11]),
          'Recorded actions or encoded real continuation rewards differ')
    selected.append(dict(definition=definition, raw=raw, modal=scored,
        source_artifact=raw_entry, modal_artifact=entry))
  true = np.concatenate([x['modal']['true_rewards'] for x in selected])
  episodes = np.repeat([x['definition']['episode'] for x in selected], 16)
  sensitive = np.ptp(true[:, 1:].sum(-1), axis=-1) >= .1
  require(int(sensitive.sum()) == 68 and len(np.unique(episodes[sensitive])) == 8 and
      result['control_sensitivity']['positions'] == 68 and
      result['control_sensitivity']['episodes'] == 8, 'Original control sensitivity proof differs')
  proof = dict(collection_plan_sha256=d.sha256(plan_path),
      collection_complete_sha256=d.sha256(complete_path), modal_result_sha256=MODAL_RESULT_SHA,
      modal_plan_sha256=MODAL_PLAN_SHA, modal_audit_sha256=MODAL_AUDIT_SHA,
      calibration_episodes=[x['episode'] for x in calibration], blind_arrays_read=False,
      control_sensitivity_positions=68, control_sensitivity_episodes=8,
      checkpoint_sha256=CHECKPOINT_SHA, config_sha256=CONFIG_SHA, params_sha256=PARAMS_SHA,
      artifacts=[dict(episode=x['definition']['episode'],
          source_sha256=x['source_artifact']['sha256'], modal_sha256=x['modal_artifact']['sha256'],
          controls_sha256=array_hash(x['modal']['controls']),
          true_rewards_sha256=array_hash(x['modal']['true_rewards'])) for x in selected])
  return selected, proof


@functools.lru_cache(maxsize=MAX_CHUNK)
def _key_function(count):
  import jax
  import jax.numpy as jnp
  def build(labels):
    root = jax.random.PRNGKey(MC_SEED)
    for label in labels[:3]:
      root = jax.random.fold_in(root, label)
    paths = jnp.arange(count, dtype=jnp.uint32) + labels[3]
    controls, steps = jnp.arange(6, dtype=jnp.uint32), jnp.arange(HORIZON + 1, dtype=jnp.uint32)
    def path_keys(path):
      base = jax.random.fold_in(root, path)
      return jax.vmap(lambda control: jax.vmap(lambda step: jax.random.fold_in(
          jax.random.fold_in(base, control), step))(steps))(controls)
    return jax.vmap(path_keys)(paths)
  return jax.jit(build)


def private_keys(episode, position, block, sample_start, count):
  """Explicit transfer, then JIT constants; shape alone selects compilation."""
  import jax
  require(1 <= count <= MAX_CHUNK and 0 <= block < BLOCKS and
      0 <= sample_start and sample_start + count <= PER_BLOCK, 'Invalid finite MC key range')
  labels = jax.device_put(np.asarray([episode, position, block, sample_start], np.uint32))
  return _key_function(count)(labels)


def coupled_sample(model, logits, keys):
  """Use the actual RSSM OneHot distribution once; same key across branches."""
  import jax
  from embodied.jax import nets as nn
  # logits [branches,samples,controls,factors,classes], keys [samples,controls,2]
  sample = lambda logit, key: model.dyn._dist(logit).sample(seed=key)
  per_control = jax.vmap(sample)
  per_sample = jax.vmap(per_control)
  return nn.cast(jax.vmap(lambda x: per_sample(x, keys))(logits))


def model_chunk(model, deter, qlogits, plogit, sequences, keys):
  """All initial and future latents sampled; no NJ seed, reward, or future image input."""
  import jax.numpy as jnp
  from embodied.jax import nets as nn
  require(qlogits.shape[0] == 4 and plogit.shape == qlogits.shape[1:] and
      deter.ndim == 1 and sequences.shape[0] == 6 and sequences.shape[1] == HORIZON and
      keys.shape[1:] == (6, HORIZON + 1, 2), 'Invalid MC model chunk axes')
  size = keys.shape[0]
  initial = jnp.concatenate([qlogits, plogit[None]], 0)
  initial = jnp.broadcast_to(initial[:, None, None], (5, size, 6, *initial.shape[1:]))
  h = nn.cast(jnp.broadcast_to(deter, (5, size, 6, len(deter))))
  z = coupled_sample(model, initial, keys[:, :, 0])
  flat = lambda x: x.reshape((-1, *x.shape[3:]))
  logits = []
  for step in range(HORIZON):
    action = nn.cast(jnp.broadcast_to(sequences[None, None, :, step],
        (5, size, 6, sequences.shape[-1])))
    h = model.dyn._core(flat(h), flat(z), flat(action)).reshape(h.shape)
    prior = model.dyn._prior(flat(h)).reshape(initial.shape)
    z = coupled_sample(model, prior, keys[:, :, step + 1])
    reward = model.rew(model.feat2tensor(dict(deter=flat(h), stoch=flat(z))), 1)
    logits.append(reward.logits.reshape((5, size, 6, -1)))
  return dict(logits=jnp.stack(logits, 3), bins=reward.bins)


class Predictor:
  def __init__(self, agent):
    import jax
    self.agent = agent
    history = modal.Predictor(agent)
    self.advance = history.advance
    self.mode_predict = history.predict
    model = agent.model
    def prepare(carry, images, previous, first):
      import jax.numpy as jnp
      n = len(images)
      enc, dyn = jax.tree.map(lambda x: jnp.repeat(x, n, 0), carry)
      reset = jnp.full((n,), first)
      _, _, tokens = model.enc(enc, {'image': images}, reset, False, single=True)
      _, _, feat = model.dyn.observe(dyn, tokens,
          {'action': jnp.broadcast_to(previous, (n, *previous.shape))}, reset, False, single=True)
      from dreamerv3.dt_latch import require_valid
      require_valid(jnp.all(feat['deter'] == feat['deter'][:1]))
      return feat['deter'][0], feat['logit'], model.dyn._prior(feat['deter'][:1])[0]
    self.prepare = modal.readonly_jit(prepare)
    # This is nj.pure only to read parameters; model_chunk never calls nj.seed.
    self.chunk = modal.readonly_jit(lambda h, q, p, controls, keys:
        model_chunk(model, h, q, p, controls, keys))
    self.history_call = history.call

  def call_chunk(self, h, q, p, controls, keys):
    import embodied.jax.internal as internal
    inputs = internal.device_put((h, q, p, controls, keys), self.agent.train_mirrored)
    _, output = self.chunk(self.agent.params, *inputs)
    return output


def log_probabilities(logits):
  logits = np.asarray(logits, np.float64)
  require(np.isfinite(logits).all(), 'Nonfinite raw MC reward logits')
  maximum = logits.max(-1, keepdims=True)
  return logits - maximum - np.log(np.exp(logits - maximum).sum(-1, keepdims=True))


def log_probability_sum(logits):
  values = log_probabilities(logits)
  maximum = values.max(1)
  return maximum + np.log(np.exp(values - maximum[:, None]).sum(1))


def scores_from_sums(log_sums, samples, truth, bins):
  """Score marginal distributions; no product of time marginals or path CE mean."""
  mixture = log_sums - np.log(samples)
  require(mixture.shape[1:3] == (5, 6) and mixture.shape[3] == HORIZON,
      'Mixture branch/control/time axes differ')
  p = np.repeat(mixture[:, 4:5], 4, 1)
  labels = np.broadcast_to(truth[:, None], mixture[:, :4].shape[:-1]).astype(np.float32)
  scores = d.compatibility_numpy(mixture[:, :4], p, labels, bins)
  return dict(ce_q=scores['ell_q'], ce_p=scores['ell_p'])


def gap_values(scores, horizon=10, fixed=True):
  controls = slice(1, 6) if fixed else slice(0, 1)
  q, p = scores['ce_q'][..., :horizon].mean(-1), scores['ce_p'][..., :horizon].mean(-1)
  return {GAPS[0]: (p[:, 0, controls] - q[:, 0, controls]).mean(-1),
      GAPS[1]: (q[:, 3, controls] - q[:, 0, controls]).mean(-1)}


def reliability(gap32, gap64, gap128, last64, block_gaps, primary):
  result = {}
  for key in GAPS:
    terms = dict(nested32_64=abs(gap32[key] - gap64[key]),
        nested64_128=abs(gap64[key] - gap128[key]),
        independent_halves=abs(gap64[key] - last64[key]),
        block_t_margin=3.182 * np.std([b[key] for b in block_gaps], ddof=1) / 2)
    epsilon = float(max(terms.values()))
    point, lower = primary[key]['point'], primary[key]['lower95']
    supported = point > 0 and epsilon <= point * .2 and lower is not None and lower - epsilon > 0
    result[key] = dict(epsilon=epsilon, components={k: float(v) for k, v in terms.items()},
        stable=epsilon <= .001, positive_margin_robust=bool(supported))
  stable = all(x['stable'] for x in result.values())
  directions = all(x['positive_margin_robust'] for x in result.values())
  status = ('MC-inconclusive' if not stable else
      'design_independent_revision_only' if directions else 'stop_reward_support_candidate')
  return dict(metrics=result, mc_stable=stable, directions_robust=directions,
      decision=status, interpretation='Finite numerical reliability indicator, not a rigorous integration-error bound')


def report(block_log_sums, truth, episodes, bins):
  # [position, block, 5 branches, 6 controls, 10 steps, bins]
  sums = np.asarray(block_log_sums, np.float64)
  require(sums.shape[:5] == (len(truth), 4, 5, 6, 10) and np.isfinite(sums).all(),
      'Invalid block probability sums')
  pooled = lambda ids: np.logaddexp.reduce(sums[:, ids], axis=1)
  scores = {32: scores_from_sums(pooled([0]), 32, truth, bins),
      64: scores_from_sums(pooled([0, 1]), 64, truth, bins),
      128: scores_from_sums(pooled([0, 1, 2, 3]), 128, truth, bins)}
  half = scores_from_sums(pooled([2, 3]), 64, truth, bins)
  blocks = [scores_from_sums(sums[:, b], 32, truth, bins) for b in range(4)]
  by_prefix = {}
  for horizon in (1, 3, 10):
    groups = {}
    for name, fixed in (('recorded', False), ('fixed', True)):
      gaps = gap_values(scores[128], horizon, fixed)
      interval = lambda x: d.cluster_interval(x, episodes, draws=2000, seed=STAT_SEED)
      controls = slice(1, 6) if fixed else slice(0, 1)
      q = scores[128]['ce_q'][..., :horizon].mean(-1)
      groups[name] = {key: interval(value) for key, value in gaps.items()}
      groups[name].update(background_a_minus_clean_CE=interval((q[:, 1, controls]-q[:, 0, controls]).mean(-1)),
          background_b_minus_clean_CE=interval((q[:, 2, controls]-q[:, 0, controls]).mean(-1)),
          mean_CE_q=float(q[:, 0, controls].mean()),
          mean_CE_p=float(scores[128]['ce_p'][:, 0, controls, :horizon].mean()))
    by_prefix[str(horizon)] = groups
  per_control = {}
  q10, p10 = scores[128]['ce_q'].mean(-1), scores[128]['ce_p'].mean(-1)
  for control, name in enumerate(('recorded', 'zero', 'positive', 'negative', 'pattern', 'inverse_pattern')):
    per_control[name] = {GAPS[0]: d.cluster_interval(p10[:,0,control]-q10[:,0,control],
        episodes, draws=2000, seed=STAT_SEED),
        GAPS[1]: d.cluster_interval(q10[:,3,control]-q10[:,0,control],
        episodes, draws=2000, seed=STAT_SEED)}
  aggregate = lambda score: {k: float(v.mean()) for k, v in gap_values(score).items()}
  primary = by_prefix['10']['fixed']
  numeric = reliability(aggregate(scores[32]), aggregate(scores[64]), aggregate(scores[128]),
      aggregate(half), [aggregate(x) for x in blocks], primary)
  span = np.ptp(truth[:, 1:].sum(-1), axis=-1)
  sensitive = span >= .1
  coverage = dict(positions=int(sensitive.sum()), episodes=len(np.unique(episodes[sensitive])),
      sufficient=int(sensitive.sum()) >= 16 and len(np.unique(episodes[sensitive])) >= 4)
  numeric['numeric_decision'] = numeric['decision']
  if not coverage['sufficient'] and numeric['decision'] == 'design_independent_revision_only':
    numeric['decision'] = 'stop_reward_support_candidate'
    numeric['control_coverage_missing'] = True
  descriptive = {}
  for count in (32, 64, 128):
    descriptive[str(count)] = aggregate(scores[count])
  # This set is selected solely by pre-existing true fixed-control sensitivity.
  sensitive_scores = {k: d.cluster_interval(v[sensitive], episodes[sensitive],
      draws=2000, seed=STAT_SEED) for k, v in gap_values(scores[128]).items()}
  pointwise = {key: dict(mean_absolute_half_difference=float(np.abs(
      gap_values(scores[64])[key] - gap_values(half)[key]).mean()),
      p95_absolute_half_difference=float(np.quantile(np.abs(
          gap_values(scores[64])[key] - gap_values(half)[key]), .95))) for key in GAPS}
  return dict(version=VERSION, primary_horizon=10, samples=128, blocks=4,
      paths_per_block=32, prefixes=by_prefix, per_control_descriptive=per_control,
      numerical_reliability=numeric,
      nested_counts_descriptive=descriptive, pointwise_mc_descriptive=pointwise,
      sensitive_subset_descriptive=sensitive_scores, control_sensitivity=coverage,
      necessary_skill_supported_on_calibration=bool(numeric['mc_stable'] and
          numeric['directions_robust'] and coverage['sufficient']),
      training_authorized=False, proceed_short_training=False, eligible_method_training=False,
      interpretation='Calibration-only finite marginal skill; neither causal correctness nor independent method validation')


def require_clean():
  if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
    raise RuntimeError('Actual MC diagnostic requires clean committed checkout')


def check_deadline(start):
  if time.monotonic() - start >= BUDGET:
    raise TimeoutError('Frozen 3600s budget exhausted; no extra samples or horizon extension')


def worker(args):
  require(args.platform == 'cuda' and args.task == 'dmc_quadruped_walk' and
      args.sample_chunk == MAX_CHUNK, 'Frozen CUDA task/sample-chunk contract differs')
  output = Path(args.output).resolve()
  budget_path = output / 'budget.json'
  budget = json.loads(budget_path.read_text(encoding='utf-8'))
  require(budget['parent_pid'] == os.getppid() == int(os.environ.get('DL_MC_PARENT', '-1')) and
      budget['seconds'] == BUDGET and budget['git_sha'] == d.git_sha(),
      'Worker must be a direct child of the bounded read-only supervisor')
  started = budget['monotonic_start']
  from scripts.dl_resources import require_resource, verify_runtime
  resource = require_resource(args.resource_record)
  require(resource['physical_gpu'] == 4, 'MC-r1 is limited to assigned sv3 physical GPU4')
  require_clean()
  selected, proof = load_sources(args.dataset, args.modal_scores, args.modal_audit)
  require(d.sha256(checkpoint_path(args.checkpoint)) == CHECKPOINT_SHA and
      d.sha256(args.config) == CONFIG_SHA, 'Actual checkpoint/config bytes differ')
  check_deadline(started)
  import jax
  from embodied.run.protocol_v1 import parameter_digest
  agent, config, metadata = load_agent_no_env(args)
  require(metadata['params_sha256'] == PARAMS_SHA and
      metadata['checkpoint_counters']['updates'] == 74493, 'Loaded checkpoint state differs')
  resource['runtime_compute'] = verify_runtime(resource, compute=True, graphics=False)
  d.write_json(output / 'plan.json', dict(version=VERSION, metadata=metadata,
      proof=proof, resource=resource, history_seed=HISTORY_SEED, mc_seed=MC_SEED,
      statistical_seed=STAT_SEED, branches=list(BRANCHES), blocks=4, paths_per_block=32,
      samples=128, sample_chunk=args.sample_chunk, horizon=10, controls=6,
      physics_steps=0, render_calls=0, optimizer_updates=0, gate_changed=False,
      hard_budget_seconds=BUDGET, latent_reward_steps=4915200))
  predictor, artifacts, records = Predictor(agent), [], []
  for selected_episode in selected:
    episode = selected_episode['definition']['episode']
    raw, scored = selected_episode['raw'], selected_episode['modal']
    positions = scored['position']
    carry, position_records = agent.init_train(1)[:2], []
    for t in range(int(positions[-1]) + 1):
      check_deadline(started)
      previous = raw['action'][t-1] if t else np.zeros_like(raw['action'][0])
      seed = int(np.random.SeedSequence([HISTORY_SEED, episode, t]).generate_state(1)[0])
      before = carry
      carry = predictor.history_call(predictor.advance, before, raw['image'][t],
          previous, raw['is_first'][t], seed=seed)
      if t not in positions:
        continue
      index = int(np.flatnonzero(positions == t)[0])
      images = d.intervention_images(raw['image'][t], raw['preserve'][t], raw['body'][t], [HISTORY_SEED, episode])
      h, q, p = predictor.history_call(predictor.prepare, before, images, previous,
          raw['is_first'][t], seed=seed)
      controls = scored['controls'][index]
      # Replay the unchanged mode function, under identical history and seed.
      # This is attribution evidence, not a new scientific acceptance gate.
      mode_replay = predictor.history_call(predictor.mode_predict, before, images,
          previous, raw['is_first'][t], controls, seed=seed)
      with jax._src.config.explicit_device_get_scope():
        mode_replay = jax.tree.map(np.asarray, mode_replay)
        initial_h, initial_q, initial_p = map(np.asarray, (h, q, p))
      compute_dtype = str(initial_h.dtype)
      require(np.array_equal(mode_replay['bins'], scored['bins']) and
          mode_replay['logits'].shape == scored['logits'][index].shape and
          np.isfinite(mode_replay['logits']).all(), 'Invalid modal replay attribution evidence')
      expected_mode = scored['logits'][index]
      mode_error = float(np.abs(mode_replay['logits'].astype(np.float64)-expected_mode).max())
      block_sums, chunk_entries = [], []
      for block in range(BLOCKS):
        total = None
        for sample in range(0, PER_BLOCK, args.sample_chunk):
          check_deadline(started)
          count = min(args.sample_chunk, PER_BLOCK-sample)
          keys = private_keys(episode, t, block, sample, count)
          result = predictor.call_chunk(h, q, p, controls, keys)
          with jax._src.config.explicit_device_get_scope():
            result = jax.tree.map(np.asarray, result)
            key_values = np.asarray(keys)
          require(np.array_equal(result['bins'], scored['bins']), 'Original reward bins changed')
          chunk_sum = log_probability_sum(result['logits'])
          total = chunk_sum if total is None else np.logaddexp(total, chunk_sum)
          chunk_path = output / f'episode{episode:03}-position{t:04}-block{block}-sample{sample:02}.npz'
          if chunk_path.exists():
            raise FileExistsError(chunk_path)
          np.savez_compressed(chunk_path, logits=result['logits'], keys=key_values,
              bins=result['bins'], log_probability_sum=chunk_sum,
              probability_sum=np.exp(chunk_sum), sample_start=sample, sample_count=count,
              block=block, position=t, episode=episode)
          chunk_entries.append(dict(path=chunk_path.name, sha256=d.sha256(chunk_path),
              block=block, sample_start=sample, sample_count=count))
        block_sums.append(total)
      position_records.append(dict(position=t, block_log_probability_sums=np.stack(block_sums),
          true_rewards=scored['true_rewards'][index], controls=controls,
          initial_deter=initial_h.astype(np.float32),
          initial_q_logits=initial_q.astype(np.float32), initial_p_logits=initial_p.astype(np.float32),
          initial_compute_dtype=compute_dtype,
          mode_replay_logits=mode_replay['logits'], mode_replay_max_abs_error=mode_error,
          mode_replay_byte_exact=array_hash(mode_replay['logits']) == array_hash(expected_mode),
          mode_replay_sha256=array_hash(mode_replay['logits']),
          source_mode_sha256=array_hash(expected_mode)))
      artifacts.append(dict(episode=episode, position=t, chunks=chunk_entries))
      print(json.dumps(dict(episode=episode, position=t, completed_samples=128)), flush=True)
    row = {key: np.stack([x[key] for x in position_records]) for key in position_records[0]}
    row['block_probability_sums'] = np.exp(row['block_log_probability_sums'])
    row['bins'] = scored['bins']
    row['episode'] = np.full(16, episode, np.int32)
    path = output / f'episode_{episode:03}.npz'
    if path.exists():
      raise FileExistsError(path)
    np.savez_compressed(path, **row)
    records.append(row)
    artifacts.append(dict(episode=episode, path=path.name, sha256=d.sha256(path), positions=16))
  require(parameter_digest(agent) == PARAMS_SHA, 'MC diagnostic changed checkpoint parameters/state')
  require(agent.save()['counters'] == metadata['checkpoint_counters'],
      'MC diagnostic advanced original policy/training counters or seeds')
  check_deadline(started)
  summary = report(np.concatenate([x['block_log_probability_sums'] for x in records]),
      np.concatenate([x['true_rewards'] for x in records]),
      np.concatenate([x['episode'] for x in records]), records[0]['bins'])
  summary.update(actual_git_sha=d.git_sha(), plan_sha256=d.sha256(output / 'plan.json'),
      evidence_artifacts=artifacts, params_unchanged=True, counters_unchanged=True, proof=proof,
      elapsed_seconds=time.monotonic()-started,
      mode_replay=dict(byte_exact_all=bool(np.concatenate([x['mode_replay_byte_exact'] for x in records]).all()),
          maximum_absolute_error=float(max(x['mode_replay_max_abs_error'].max() for x in records)),
          attribution='Replay is an audit; nonexact results retain floating-point/history confounding, with no tolerance pass'))
  d.write_json(output / 'result.json', summary)
  print(json.dumps(dict(complete=True, decision=summary['numerical_reliability']['decision'],
      training_authorized=False)), flush=True)


def supervise(args):
  require(args.platform == 'cuda' and args.task == 'dmc_quadruped_walk' and
      args.sample_chunk == MAX_CHUNK, 'Frozen CUDA task/sample-chunk contract differs')
  require_clean()
  from scripts.dl_resources import require_resource
  resource = require_resource(args.resource_record)
  require(resource['physical_gpu'] == 4, 'MC-r1 is limited to assigned sv3 physical GPU4')
  output = Path(args.output).resolve()
  if output.exists():
    raise FileExistsError(output)
  output.mkdir(parents=True)
  started = time.monotonic()
  d.write_json(output / 'budget.json', dict(version=VERSION, parent_pid=os.getpid(),
      seconds=BUDGET, monotonic_start=started, git_sha=d.git_sha(), resource=resource))
  env = dict(os.environ, DL_MC_PARENT=str(os.getpid()))
  try:
    run = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--_worker'],
        cwd=ROOT, env=env, timeout=BUDGET)
    code, reason = run.returncode, 'worker_completed' if run.returncode == 0 else 'worker_failed'
  except subprocess.TimeoutExpired:
    code, reason = 124, 'hard_budget_exhausted_MC_inconclusive'
  d.write_json(output / 'supervisor_complete.json', dict(version=VERSION, exit_code=code,
      reason=reason, elapsed_seconds=time.monotonic()-started, actual_git_sha=d.git_sha(),
      training_authorized=False, proceed_short_training=False))
  return code


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for field in ('checkpoint', 'config', 'dataset', 'modal_scores', 'output', 'resource_record'):
    parser.add_argument('--' + field.replace('_', '-'), required=True)
  parser.add_argument('--modal-audit')
  parser.add_argument('--sample-chunk', type=int, default=8)
  parser.add_argument('--task', default='dmc_quadruped_walk')
  parser.add_argument('--platform', default='cuda')
  parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
  args = parser.parse_args()
  args.modal_audit = args.modal_audit or str(Path(args.modal_scores).parent / 'independent-predictive-audit-v1.json')
  if args._worker:
    worker(args)
    return 0
  return supervise(args)


if __name__ == '__main__':
  sys.exit(main())
