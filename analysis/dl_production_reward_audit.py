"""Independent CPU arithmetic and artifact audit for DL-production-reward-r1.

No project producer module, pickle, model, environment or renderer is imported.
The prescribed JAX CPU PRNG primitives verify named keys, not an independent
implementation of Threefry or a new Ninjax/model execution. Physics/carry truth
is authenticated reuse of the source receipts. --selftest never reads data.
"""
import os
import signal
import time

ENTRY_STARTED = time.monotonic()
SECONDS = 120


def _expire(signum, frame):
  raise TimeoutError('Independent audit whole-entry 120s budget expired')


if __name__ == '__main__':
  for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
  os.environ['JAX_PLATFORMS'] = 'cpu'
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  if os.name == 'posix':
    signal.signal(signal.SIGALRM, _expire)
    signal.setitimer(signal.ITIMER_REAL, max(1e-6, SECONDS - (time.monotonic() - ENTRY_STARTED)))

import argparse
import functools
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

VERSION = 'DL-production-reward-r1'
IMPLEMENTATION = 'DL-production-reward-audit-r1'
EPISODES = (9, 14, 15, 18, 20, 24, 30, 31)
POSITIONS = (1, 300, 600, 900)
SHAPE = (8, 4, 4, 2, 128, 7, 20)
CONTROL_NAMES = ('zero', 'all+', 'all-', 'lift-LR+', 'lift-LR-', 'extend-diag+', 'extend-diag-')
SOURCE_GIT = 'f1ae4d6513e81ed680992b64ba8ebade7d256d52'
SOURCE_PINS = {
    'plan.json': '474bad6adf6558b7a47c40711546dc4cf2adc870684b8cff2766a802605f44d6',
    'result.json': '06df6efa5f9ce84b428921b59fa508df8badd1dd54e30335bf2d3a273e585fad',
    'action_information_complete.json': '28481a749476be0b3e8c05f28e719cb22e4cfffb638e56c04d8f86128026d37e',
    'budget.json': 'ddc8f43a3f4b57c11858be61f4c31315484c8c42b1e887f48391e71d077e2a68',
    'supervisor_complete.json': 'a920a0191429c7ffd05af5a897a30839da47d046d9126a1002a4997928dc1ea3',
    'independent-numeric-audit-v1.json': 'cbb99e43ad72a6960d86e5044455a18fecc38f068172defe60deb0d82f8678e5'}
MODEL_PINS = dict(checkpoint_sha256='2ab693c8ded4d8f2186bd489aa8b77498448ba2f2c331090159da4e5f08056b9',
    params_sha256='14859c4e129b51aa488c10eab09f16d56fd110c76239b2c9072f415829d111c2',
    config_source_sha256='11428512fdc9b9e04a5384e419c5abcef93e5429db0a3327d05ccb0d4bd20445')
COUNTERS = dict(updates=74493, batches=74493, actions=18769, eval_actions=1001, eval_seed=3404107006)
SOURCE_FILES = (
    'scripts/dl_production_reward.py', 'scripts/dl_production_reward_source.py',
    'scripts/dl_production_reward_model.py', 'scripts/dl_production_reward_stats.py',
    'scripts/dl_production_reward_supervisor.py', 'docs/DL_PRODUCTION_REWARD.md',
    'scripts/dl_predictive_mc.py', 'scripts/dl_predictive.py', 'scripts/dl_diagnostic.py',
    'scripts/dl_resources.py', 'dreamerv3/agent.py', 'dreamerv3/rssm.py',
    'embodied/jax/agent.py', 'embodied/jax/heads.py', 'embodied/jax/outs.py',
    'embodied/jax/nets.py', 'scripts/probe_formal_devices.py',
    'tests/test_dl_production_reward_source.py', 'tests/test_dl_production_reward_model.py',
    'tests/test_dl_production_reward_stats.py', 'tests/test_dl_production_reward_supervisor.py',
    'tests/test_dl_production_reward_worker.py', 'analysis/dl_production_reward_audit.py',
    'tests/test_dl_production_reward_audit.py')
WHITE_FIELDS = ('episode', 'positions', 'h', 'latent', 'reward_f64', 'first_commands')
SUBSETS = dict(first32=(0,), first64=(0, 1), pooled128=(0, 1, 2, 3), last64=(2, 3),
               block0=(0,), block1=(1,), block2=(2,), block3=(3,))
AUTHORITY = dict(training_authorized=False, method_evidence=False, real_gate_authority=False,
    eligible_method_training=False, proceed_short_training=False, method_effectiveness_established=False,
    current_rgb_increment_over_complete_history_established=False, continuing_kl_compression_effect_established=False)


def require(condition, message):
  if not condition:
    raise ValueError(message)


def check_deadline():
  if time.monotonic() - ENTRY_STARTED >= SECONDS:
    raise TimeoutError('Independent audit whole-entry 120s budget expired')


def sha(path):
  h = hashlib.sha256()
  with Path(path).open('rb') as f:
    for block in iter(lambda: f.read(1024 * 1024), b''):
      check_deadline(); h.update(block)
  return h.hexdigest()


def safe_file(root, name):
  require(type(name) is str and not Path(name).is_absolute() and Path(name).name == name,
          'Unsafe artifact name')
  path = (root / name).resolve(strict=True)
  require(path.parent == root, 'Artifact path escapes directory')
  return path


def read_json(path):
  def pairs(items):
    result = {}
    for key, value in items:
      require(key not in result, 'Duplicate JSON key: ' + key)
      result[key] = value
    return result
  def nonfinite(value):
    raise ValueError('Nonfinite JSON: ' + value)
  return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs, parse_constant=nonfinite)


def raw_digest(a, dtype_name=False):
  a = np.asarray(a)
  value = dict(dtype=a.dtype.str, shape=list(a.shape),
               sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
  if dtype_name: value['dtype_name'] = str(a.dtype)
  return value


def typed_sha(a):
  a = np.ascontiguousarray(a)
  return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def numeric(a, dtype, shape, name):
  require(type(a) is np.ndarray and a.dtype == np.dtype(dtype) and a.shape == shape and
          not a.dtype.hasobject and np.isfinite(a).all(), 'Invalid array: ' + name)
  return a


def commands():
  values = np.zeros((7, 12), np.float32)
  values[1] = .25; values[2] = -.25
  values[3, [1, 10]] = .25; values[3, [4, 7]] = -.25; values[4] = -values[3]
  values[5, [2, 8]] = .25; values[5, [5, 11]] = -.25; values[6] = -values[5]
  return values


def choose(values):
  """One global admissible set; lowest ID, including closed 1e-10 boundary."""
  require(np.isfinite(values).all(), 'Nonfinite choice')
  admissible = values.max(axis=-1, keepdims=True) - values <= 1e-10
  return np.argmax(admissible, axis=-1).astype(np.int32)


def moments(paths, target):
  """Independent off-diagonal residual U statistic, never mean CE or clipping."""
  require(paths.dtype == target.dtype == np.float64 and paths.ndim >= 2 and
          paths.shape[-2] >= 2 and target.shape == paths.shape[:-2] + paths.shape[-1:] and
          np.isfinite(paths).all() and np.isfinite(target).all(), 'Invalid moment arrays')
  count = paths.shape[-2]
  centered = paths - paths.mean(-1, keepdims=True)
  truth = target - target.mean(-1, keepdims=True)
  residual = centered - truth[..., None, :]
  residual_sum = residual.sum(-2)
  residual_mean = residual_sum / count
  # Sum of cross-products over i != j, divided by N(N-1).
  cross = (residual_sum * residual_sum - (residual * residual).sum(-2)) / (count * (count - 1))
  deviation = residual - residual_mean[..., None, :]
  variance = (deviation * deviation).sum(-2) / (count - 1)
  require(np.isfinite(cross).all() and np.isfinite(variance).all(), 'Moment overflow')
  return dict(cumulative_mean=paths.mean(-2), centered_cumulative_mean=centered.mean(-2),
      centered_path_sample_variance=variance, centered_variance_over_n=variance / count,
      bc_mse=cross.mean(-1), plugin_centered_mse=(residual_mean ** 2).mean(-1),
      mc_correction=(variance / count).mean(-1),
      plugin_level_mse=((paths.mean(-2) - target) ** 2).mean(-1))


def aggregate(values):
  require(values.dtype == np.float64 and values.ndim == 3 and np.isfinite(values).all(), 'Bad metric axes')
  state = values.sum(2) / values.shape[2]
  episode = state.sum(1) / values.shape[1]
  return dict(per_initial=values.tolist(), state_values=state.tolist(), episode_values=episode.tolist(),
              primary=float(episode.sum() / len(episode)))


def linear_percentile(values, fraction):
  ordered = np.sort(values)
  index = (len(ordered) - 1) * fraction
  lower, upper = math.floor(index), math.ceil(index)
  return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower))


def gap(values, draws):
  result = aggregate(values)
  scores = np.asarray(result['episode_values'], np.float64)
  samplemeans = scores[draws].sum(1) / draws.shape[1]
  return dict(result, lower95=linear_percentile(samplemeans, .025),
              upper95=linear_percentile(samplemeans, .975))


def constant(truth, episodes):
  centered = truth - truth.mean(-1, keepdims=True)
  vectors, folds = [], []
  for held, episode in enumerate(episodes):
    train = [x for x in range(len(episodes)) if x != held]
    vector = centered[train].mean((0, 1))
    vectors.append(vector)
    folds.append(dict(heldout_episode=int(episode), training_episodes=[int(episodes[x]) for x in train],
                      centered_cost_vector=vector.tolist(), training_states=len(train) * truth.shape[1]))
  vectors = np.stack(vectors)
  choices = choose(vectors)
  selected = np.stack([truth[e, :, choices[e]] for e in range(len(episodes))])
  mse = ((vectors[:, None] - centered) ** 2).mean(-1)
  regret = truth.max(-1) - selected
  result = dict(folds=folds, centered_cost_vectors=vectors.tolist(), control_by_episode=choices.tolist(),
      state_mse=mse.tolist(), episode_mse=mse.mean(1).tolist(), primary_mse=float(mse.mean(1).mean()),
      state_selected_reward=selected.tolist(), episode_selected_reward=selected.mean(1).tolist(),
      primary_selected_reward=float(selected.mean(1).mean()), state_finite_library_regret=regret.tolist(),
      primary_finite_library_regret=float(regret.mean()), correction=0.,
      labels_used='other seven episodes; each of their four states equally weighted')
  return result, mse, selected


def score(paths, truth, constant_mse, constant_selected, draws, indexes, blocks):
  e, s, i, b, n, a = paths.shape
  target = np.broadcast_to(truth[:, :, None, None], (e, s, i, b, a))
  moment = moments(paths, target)
  choices = choose(moment['cumulative_mean'])
  selected = np.take_along_axis(target, choices[..., None], -1)[..., 0]
  regret = target.max(-1) - selected
  interfaces = {}
  for branch, name in enumerate(('q', 'p')):
    metrics = {key: aggregate(moment[key][..., branch]) for key in
               ('bc_mse', 'plugin_centered_mse', 'mc_correction', 'plugin_level_mse')}
    metrics.update(choices=choices[..., branch].tolist(), selected_reward=aggregate(selected[..., branch]),
        finite_library_regret=aggregate(regret[..., branch]),
        negative_bc_mse_estimates=int((moment['bc_mse'][..., branch] < 0).sum()))
    interfaces[name] = metrics
  return dict(paths=n, path_indices=indexes.tolist(), blocks=list(blocks),
      **{key: moment[key].tolist() for key in ('cumulative_mean', 'centered_cumulative_mean',
          'centered_path_sample_variance', 'centered_variance_over_n')}, interfaces=interfaces,
      mse_gaps=dict(p_minus_q=gap(moment['bc_mse'][..., 1] - moment['bc_mse'][..., 0], draws),
                   constant_minus_q=gap(constant_mse[:, :, None] - moment['bc_mse'][..., 0], draws)),
      utility_gaps=dict(q_minus_p=gap(selected[..., 0] - selected[..., 1], draws),
                       q_minus_constant=gap(selected[..., 0] - constant_selected[:, :, None], draws)))


def reliability(subsets, group, limit, family):
  metrics = {}
  for key, value in subsets['pooled128'][group].items():
    points = {name: item[group][key]['primary'] for name, item in subsets.items()}
    blocks = np.array([points['block' + str(i)] for i in range(4)], np.float64)
    components = dict(nested32_64=abs(points['first32'] - points['first64']),
        nested64_128=abs(points['first64'] - points['pooled128']),
        independent_halves=abs(points['first64'] - points['last64']),
        block_t_margin=float(3.182 * np.sqrt(((blocks - blocks.mean()) ** 2).sum() / 3) / 2))
    epsilon, point = float(max(components.values())), value['primary']
    positive = point > 0 and epsilon <= .2 * point and value['lower95'] - epsilon > 0
    metrics[key] = dict(epsilon=epsilon, components=components, stable=epsilon <= limit, point=point,
        lower95=value['lower95'], upper95=value['upper95'],
        lower95_minus_epsilon=value['lower95'] - epsilon, upper95_plus_epsilon=value['upper95'] + epsilon,
        positive_margin_robust=bool(positive), negative_margin_robust=bool(value['upper95'] + epsilon <= 0),
        relative_positive_margin_limit=.2)
  stable = all(x['stable'] for x in metrics.values())
  positive = all(x['positive_margin_robust'] for x in metrics.values())
  negative = any(x['negative_margin_robust'] for x in metrics.values())
  status = ('MC-inconclusive' if not stable else 'finite_native_' + family +
            ('_not_supported' if negative else '_exploratory_supported' if positive else '_inconclusive'))
  return dict(metrics=metrics, epsilon_limit=limit, mc_stable=stable,
      positive_margin_conditions_met=positive, necessary_directions_robust=bool(stable and positive), status=status,
      interpretation='Finite numerical reliability indicator, not a rigorous MC integration-error bound')


def choice_compare(left, right):
  result = {}
  for branch in ('q', 'p'):
    a, b = left['interfaces'][branch], right['interfaces'][branch]
    same = np.asarray(a['choices'], np.int32) == np.asarray(b['choices'], np.int32)
    change = np.asarray(a['selected_reward']['per_initial']) - np.asarray(b['selected_reward']['per_initial'])
    result[branch] = dict(equal_choices=same.tolist(), agreement_fraction=float(same.mean()),
        changed_choices=int((~same).sum()), signed_selected_reward_change=change.tolist(),
        absolute_switch_cost=np.abs(change).tolist(), maximum_absolute_switch_cost=float(np.abs(change).max()),
        mean_absolute_switch_cost=float(np.abs(change).mean()))
  return result


def recompute(predictions, truth, episodes):
  numeric(predictions, np.float32, SHAPE, 'reward'); numeric(truth, np.float64, (8, 4, 7), 'truth')
  numeric(episodes, np.int32, (8,), 'episodes')
  require(np.array_equal(episodes, EPISODES), 'Unapproved episode axes')
  digests = {k: raw_digest(v) for k, v in dict(predictions=predictions, truth=truth, episodes=episodes).items()}
  returns = predictions.sum(-1, dtype=np.float64)
  const, cmse, cselected = constant(truth, episodes)
  draws = np.random.default_rng(20261025).integers(0, 8, (2000, 8))
  subsets = {}
  with np.errstate(over='raise', divide='raise', invalid='raise'):
    for name, blocks in SUBSETS.items():
      check_deadline()
      ids = np.array([32 * block + path for block in blocks for path in range(32)], np.int32)
      subsets[name] = score(returns[..., ids, :], truth, cmse, cselected, draws, ids, blocks)
  mse, utility = reliability(subsets, 'mse_gaps', .0004, 'mean'), reliability(subsets, 'utility_gaps', .002, 'choice')
  comparisons = {name + '_vs_pooled128': choice_compare(item, subsets['pooled128'])
                 for name, item in subsets.items() if name != 'pooled128'}
  comparisons['first64_vs_last64'] = choice_compare(subsets['first64'], subsets['last64'])
  extrema = dict(native_reward_minimum=float(predictions.min()), native_reward_maximum=float(predictions.max()),
      cumulative_path_minimum=float(returns.min()), cumulative_path_maximum=float(returns.max()),
      per_initial_native_minimum=predictions.min(axis=(-1, -2, -3)).tolist(),
      per_initial_native_maximum=predictions.max(axis=(-1, -2, -3)).tolist(),
      per_initial_cumulative_minimum=returns.min(axis=(-1, -2)).tolist(),
      per_initial_cumulative_maximum=returns.max(axis=(-1, -2)).tolist())
  require(digests == {k: raw_digest(v) for k, v in dict(predictions=predictions, truth=truth, episodes=episodes).items()},
          'Audit modified input arrays')
  return dict(version=VERSION, implementation='DL-production-reward-stats-engineering-r1',
      episodes=list(EPISODES), positions=list(POSITIONS), prediction_shape=list(SHAPE), truth_shape=[8, 4, 7],
      controls=list(CONTROL_NAMES), branches=['q', 'p'], horizon=20, blocks=4, paths_per_block=32, future_paths=128,
      initial_samples=4, axes=['episode', 'state', 'initial_sample', 'branch', 'future_path', 'control', 'step'],
      primary_subset='pooled128', constant=const, subsets=subsets, mse_reliability=mse, utility_reliability=utility,
      primary_status=mse['status'], choice_stability=comparisons, path_extrema=extrema,
      bc_mse_definition='mean_controls((mean_paths(control_centered_return)-centered_truth)^2 - ddof1_var/N)',
      bc_mse_numeric_coordinates='control-centered residual paths; fixed-target translation leaves variance unchanged',
      plugin_level_mse_definition='MSE of uncentered conditional path mean; no MC bias correction',
      negative_bc_mse_clipped=False, initial_samples_averaged_before_loss=False,
      future_paths_independence_verified_by_stats=False, source_provenance_verified_by_stats=False,
      source_or_model_execution_verified_by_stats=False, tie_tolerance=1e-10,
      tie_rule='lowest control ID with max(value)-value<=1e-10', bootstrap_draws=2000, statistical_seed=20261025,
      percentile_method='linear', bootstrap_unit='whole calibration episodes; fixed model/LOEO-constant episode scores',
      interval_scope='Nominal calibration description; LOEO constants overlap, no refitting uncertainty or independent confirmation',
      mc_error_bound_claimed=False, inputs_unchanged=True, input_digests=digests, calibration_only=True,
      execution_scope='numeric statistics only; runtime model/GPU verification belongs to caller',
      model_calls=0, physics_steps=0, optimizer_updates=0, gpu_devices_used=0, blind_arrays_read=False,
      records_decoded=False, current_rgb_increment_established=False, active_rep_compression_effect_established=False,
      method_effectiveness_established=False, method_evidence=False, actor_applicable=False, real_gate_authority=False,
      eligible_method_training=False, training_authorized=False, proceed_short_training=False,
      decision='stop_after_fixed_production_reward_diagnostic',
      negative_scope='Only the specified native H20 conditional-mean reward-support estimator; not H1 head, latent information, rawK or future learning')


def compare_tree(expected, published):
  summary = dict(max_abs_error=0., compared_numeric_leaves=0, compared_exact_leaves=0, zero_side_checks=0,
                 absolute_tolerance=1e-10, relative_tolerance=0., full_statistic_tree_verified=True)
  critical = {'primary', 'lower95', 'upper95', 'point', 'lower95_minus_epsilon', 'upper95_plus_epsilon'}
  def visit(a, b, path, sensitive=False):
    sensitive = sensitive or (bool(path) and path[-1] in critical)
    if type(a) is dict:
      require(type(b) is dict and set(a) == set(b), 'Statistic keys differ: ' + '.'.join(path))
      for key in a: visit(a[key], b[key], (*path, key), sensitive)
    elif type(a) is list:
      require(type(b) is list and len(a) == len(b), 'Statistic axes differ: ' + '.'.join(path))
      for i, (x, y) in enumerate(zip(a, b)): visit(x, y, (*path, str(i)), sensitive)
    elif type(a) is float:
      require(type(b) is float and math.isfinite(a) and math.isfinite(b), 'Statistic numeric type/nonfinite')
      error = abs(a - b)
      require(error <= 1e-10, 'Statistic arithmetic mismatch: ' + '.'.join(path))
      if sensitive:
        require((a > 0) == (b > 0) and (a < 0) == (b < 0), 'Statistic zero side mismatch: ' + '.'.join(path))
        summary['zero_side_checks'] += 1
      summary['compared_numeric_leaves'] += 1
      summary['max_abs_error'] = max(error, summary['max_abs_error'])
    else:
      require(type(a) is type(b) and a == b, 'Exact statistic mismatch: ' + '.'.join(path))
      summary['compared_exact_leaves'] += 1
  visit(expected, published, ())
  return summary


def source_authority(dataset):
  documents = {}
  for name, digest in SOURCE_PINS.items():
    path = safe_file(dataset, name)
    require(sha(path) == digest, 'Unapproved source JSON: ' + name)
    documents[name] = read_json(path)
  plan, result, complete, budget, parent, audit = [documents[k] for k in (
      'plan.json', 'result.json', 'action_information_complete.json', 'budget.json',
      'supervisor_complete.json', 'independent-numeric-audit-v1.json')]
  for value in (plan, result, complete, parent):
    require(value['version'] == 'DL-action-information-r1' and value['actual_git_sha'] == SOURCE_GIT,
            'Source scientific identity differs')
  require(budget['version'] == 'DL-action-information-r1' and budget['git_sha'] == SOURCE_GIT and
          budget['seconds'] == 180 and complete['plan_sha256'] == SOURCE_PINS['plan.json'] and
          complete['result_sha256'] == SOURCE_PINS['result.json'] and parent['budget_sha256'] == SOURCE_PINS['budget.json'] and
          parent['exit_code'] == 0 and parent['timed_out'] is False and parent['worker_terminal'] is True and
          complete['complete_episodes'] == 8 and complete['complete_states'] == 32 and complete['complete_records'] == 480,
          'Source incomplete/failed')
  require(plan['episodes'] == list(EPISODES) and plan['positions'] == list(POSITIONS) and plan['horizon'] == 20 and
          plan['delta'] == .2 and plan['actor_command_order'] == [f'pair{i}{b}' for i in range(4) for b in ('q', 'p')],
          'Source preselected object differs')
  require(all(plan['model_provenance'][k] == v for k, v in MODEL_PINS.items()) and
          plan['model_provenance']['checkpoint_counters'] == COUNTERS, 'Source model identity differs')
  expected_config = {'task': 'dmc_quadruped_walk', 'env.dmc.proprio': False, 'env.dmc.size': [64, 64],
      'env.dmc.repeat': 1, 'jax.compute_dtype': 'bfloat16', 'agent.dyn.typ': 'rssm', 'agent.dyn.rssm.deter': 4096,
      'agent.dyn.rssm.stoch': 32, 'agent.dyn.rssm.classes': 32, 'agent.dyn.rssm.unimix': .01,
      'agent.rewhead.output': 'symexp_twohot', 'agent.dt_latch.mode': 'off', 'agent.rep_probe.mode': 'off'}
  config = plan['model_provenance']['config']
  require(all(type(config[k]) is type(v) and config[k] == v for k, v in expected_config.items()) and
          config['env.dmc.camera'] in (-1, 2), 'Source actual interface differs')
  require(audit['status'] == 'passed' and audit['expected_git_sha'] == SOURCE_GIT and
          audit['source_h_sampled_z_and_actor_raw_byte_exact'] is True and audit['source_feedback_bytes_verified'] is True and
          audit['blind_arrays_read'] is False, 'Source authenticated feature proof missing')
  require(result['main_physics_steps'] == result['replay_physics_steps'] == 9600 and
          result['fresh_model_calls'] == result['render_calls'] == result['fresh_actor_rng_calls'] == 0 and
          result['source_artifacts_unchanged'] is True, 'Source physical coverage receipt missing')
  artifacts = result['evidence_artifacts']
  require(len(artifacts) == 8 and [x['episode'] for x in artifacts] == list(EPISODES), 'Source episode coverage differs')
  for item in artifacts:
    require(item['path'] == f"episode_{item['episode']:03}.npz" and
            sha(safe_file(dataset, item['path'])) == item['sha256'], 'Source NPZ bytes differ')
    require(item['records'] == 60 and item['main_steps'] == item['replay_steps'] == 1200 and len(item['verifications']) == 60,
            'Source per-episode coverage differs')
    for position in POSITIONS:
      proofs = [x for x in item['verifications'] if x['position'] == position]
      require(len(proofs) == 15 and [x['control'] for x in proofs] == list(range(15)), 'Source control proof coverage differs')
      require(all(x['executed_steps'] == x['physics_steps'] == 20 and x['padding_steps'] == 0 and x['end_type'] == 'window' and
          x['float64_reward_byte_exact'] is True and x['float32_reward_encoding_byte_exact'] is True and x['discount_byte_exact'] is True
          for x in proofs), 'Source early terminal or unverifiable H20 truth')
  return plan, result, artifacts


def load_source_numeric(path, episode):
  with np.load(path, allow_pickle=False) as archive:
    require(set(WHITE_FIELDS).issubset(archive.files), 'Missing source whitelist field')
    data = {name: archive[name].copy() for name in WHITE_FIELDS}
  numeric(data['episode'], np.int32, (), 'source episode')
  require(int(data['episode']) == episode and episode in EPISODES, 'Source numeric episode differs')
  numeric(data['positions'], np.int32, (4,), 'source positions')
  require(np.array_equal(data['positions'], POSITIONS), 'Source numeric positions differ')
  for name, dtype, dimensions in (('h', np.float64, (4, 4096)), ('latent', np.float64, (4, 4, 2, 32, 32)),
      ('reward_f64', np.float64, (4, 15, 20)), ('first_commands', np.float32, (4, 15, 12))):
    numeric(data[name], dtype, dimensions, name)
  z = data['latent']
  require(((z == 0) | (z == 1)).all() and (z.sum(-1) == 1).all(), 'Source actual sample is not onehot')
  library = np.ascontiguousarray(data['first_commands'][:, :7])
  require(library.tobytes() == np.broadcast_to(commands(), (4, 7, 12)).tobytes(), 'Source controls differ bytewise')
  # Dtype registration is NumPy-only; no JAX, model, source module or pickle.
  import ml_dtypes
  native = {}
  for name, value in (('h_native', data['h']), ('latent_native', z)):
    native[name] = value.astype(ml_dtypes.bfloat16)
    require(np.isfinite(native[name]).all() and native[name].astype(np.float64).tobytes() == value.tobytes(),
            'Source native BF16 promotion is not exact')
  truth = data['reward_f64'][:, :7].sum(-1, dtype=np.float64)
  require(np.isfinite(truth).all(), 'Source truth sum overflow')
  receipts = {name: raw_digest(value, True) for name, value in data.items()}
  receipts.update({name: raw_digest(value, True) for name, value in native.items()})
  receipts.update(controls=raw_digest(library, True), true_cost=raw_digest(truth, True))
  return truth, library, receipts


def git_binding(root, expected, receipts):
  require(re.fullmatch('[0-9a-f]{40}', expected) is not None and set(receipts) == set(SOURCE_FILES), 'Incomplete source binding')
  require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() == expected,
          'Execution checkout Git differs')
  require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=root), 'Execution checkout is dirty')
  for name, digest in receipts.items():
    check_deadline()
    blob = subprocess.check_output(['git', 'show', expected + ':' + name], cwd=root)
    require(hashlib.sha256(blob).hexdigest() == digest and (root / name).read_bytes() == blob,
            'Source bytes differ from committed Git: ' + name)
  require(Path(__file__).resolve() == root / 'analysis/dl_production_reward_audit.py', 'Auditor is not execution Git file')
  return dict(receipts)


def same_fields(value, expected, context):
  for key, item in expected.items():
    require(type(value[key]) is type(item) and value[key] == item, context + ': ' + key)


def b_authority(directory, dataset, expected, execution):
  names = ('budget.json', 'plan.json', 'result.json', 'production_reward_complete.json', 'supervisor_complete.json')
  docs = {name: read_json(safe_file(directory, name)) for name in names}
  hashes = {name: sha(safe_file(directory, name)) for name in names}
  budget, plan, result, complete, parent = [docs[k] for k in names]
  for value in docs.values(): same_fields(value, dict(version=VERSION, actual_git_sha=expected), 'B identity differs')
  same_fields(budget, dict(seconds=1800, planned_numeric_episode_reads=8, actual_numeric_episode_reads=None,
                           planned_gpu_devices=1, planned_physical_gpu=4), 'B budget differs')
  same_fields(plan, dict(episodes=list(EPISODES), positions=list(POSITIONS), horizon=20, initial_samples=4,
      branches=['q', 'p'], controls=7, future_paths=128, blocks=4, paths_per_block=32, sample_chunk=8,
      latent_reward_steps=4587520, private_seed=20261024, statistical_seed=20261025, seconds=1800,
      oracle_atol=1e-6, oracle_rtol=1e-5, stats_audit_atol=1e-10,
      source_git_sha=SOURCE_GIT, source_json_sha256=SOURCE_PINS,
      hostname='lyg0326', params_unchanged=True, gate_changed=False, physics_provenance_reused=True), 'B fixed plan differs')
  require(plan['platform'].startswith('linux') and Path(plan['source_dataset']).resolve(strict=True) == dataset,
          'B dataset/platform differs')
  for metadata in (plan['source_model_provenance'], plan['model_metadata']):
    same_fields(metadata, dict(MODEL_PINS, checkpoint_counters=COUNTERS), 'B checkpoint binding differs')
  require(plan['model_metadata']['code_version'] == VERSION and plan['packages']['numpy'] == '1.26.4' and
          all(plan['packages'][k] == v for k, v in {'jax': '0.4.33', 'jaxlib': '0.4.33', 'ninjax': '3.6.3'}.items()),
          'B model/runtime differs')
  same_fields(plan['model_metadata'], dict(loader_code_version='DL-predictive-MC-r1',
      loader_scope='Pinned model factory/checkpoint loader only; no legacy assay inputs'), 'B loader attribution differs')
  boundary = dict(AUTHORITY, exploratory_calibration_only=True, source_provenance_verified=True,
      real_dataset_reads=8, blind_arrays_read=False, physics_steps=0, render_calls=0, optimizer_updates=0,
      gpu_devices_used=1, records_decoded=False, initial_latents_resampled=False, actor_calls=0)
  for value in (plan, result, complete): same_fields(value, boundary, 'B actual receipt boundary differs')
  same_fields(result, dict(checkpoint_state_unchanged=True, state_unchanged=True, params_unchanged=True,
      counters_unchanged=True, source_inputs_unchanged=True, stopped_after_fixed_object=True,
      checkpoint_and_config_inputs_unchanged=True), 'B readonly receipts differ')
  require(result['checkpoint_state_sha256'] == plan['checkpoint_state_sha256'] and
          re.fullmatch('[0-9a-f]{64}', plan['checkpoint_state_sha256']) is not None,
          'B checkpoint complete-state receipt differs')
  require(result['elapsed_parent_entry_seconds'] <= 1800, 'B worker exceeded budget')
  source_map = plan['source_files_sha256']
  require(result['source_files_sha256'] == complete['source_files_sha256'] == source_map and
          budget['supervisor_sha256'] == source_map['scripts/dl_production_reward_supervisor.py'], 'B source hash chain differs')
  git_binding(execution, expected, source_map)
  require(plan['budget_sha256'] == result['budget_sha256'] == complete['budget_sha256'] == parent['budget_sha256'] == hashes['budget.json'] and
          result['plan_sha256'] == complete['plan_sha256'] == hashes['plan.json'] and
          complete['result_sha256'] == hashes['result.json'], 'B metadata hash chain differs')
  same_fields(complete, dict(complete_episodes=8, complete_states=32, complete_chunks=512, latent_reward_steps=4587520,
      initial_samples=4, branches=2, controls=7, future_paths=128, horizon=20), 'B incomplete coverage')
  require(complete['primary_status'] == result['analysis']['primary_status'], 'B primary status differs')
  same_fields(parent, dict(exit_code=0, timed_out=False, worker_terminal=True, owned_group_cleanup_requested=True,
      complete_verified=True, real_artifact_chain_verified=True, generic_fixture_without_scientific_verification=False,
      actual_numeric_episode_reads=8, actual_gpu_devices_used=1), 'B supervisor not successful terminal real worker')
  require(parent['failure'] is None and parent['elapsed_parent_entry_seconds'] <= 1800 and
          parent['hostname'] == 'lyg0326' and parent['platform'].startswith('linux'), 'B supervisor final receipt differs')
  verify = parent['verification']
  same_fields(verify, dict(verification_kind='production_reward_artifact_chain_schema_and_hashes',
      budget_sha256=hashes['budget.json'], plan_sha256=hashes['plan.json'], result_sha256=hashes['result.json'],
      complete_sha256=hashes['production_reward_complete.json']), 'B parent verified hash chain differs')
  resource, runtime = plan['resource'], plan['resource']['runtime_compute']
  require(resource['physical_gpu'] == runtime['physical_gpu'] == 4 and resource['logical_server'] == 'sv3' and
          resource['exclusive'] is True and resource['gpu_uuid'] == resource['compute_uuid'] == resource['egl_uuid'] == resource['cuda_visible_devices'] and
          runtime['compute_uuid'] == [resource['gpu_uuid']] and runtime['graphics_verified'] is False and
          runtime['pid'] == parent['worker_pid'] == verify['worker_pid'], 'B owned compute resource receipts differ')
  require(sha(resource['resource_source_path']) == resource['resource_source_sha256'], 'B resource source bytes changed')
  final_runtime = result['final_runtime_compute']
  require(final_runtime['physical_gpu'] == 4 and final_runtime['pid'] == runtime['pid'] and
          final_runtime['compute_uuid'] == [resource['gpu_uuid']] and final_runtime['graphics_verified'] is False,
          'B final owned CUDA runtime identity receipt differs')
  same_fields(parent['verified_worker_boundary'], dict(AUTHORITY, source_provenance_verified=True,
      actual_numeric_episode_reads=8, actual_gpu_devices_used=1, real_dataset_reads=8, gpu_devices_used=1,
      records_decoded=False, physics_steps=0, render_calls=0, optimizer_updates=0, actor_calls=0,
      initial_latents_resampled=False, exploratory_calibration_only=True), 'B parent boundary differs')
  require(parent['verified_worker_boundary'] == verify['worker_boundary'], 'B parent verification boundary differs')
  accessed = result['native_model_accessed_keys']
  require(type(accessed) is list and accessed == sorted(set(accessed)) and accessed and
          all(type(x) is str and x.startswith(('dyn/', 'rew/')) for x in accessed) and
          any(x.startswith('dyn/') for x in accessed) and any(x.startswith('rew/') for x in accessed), 'B actual parameter-read receipt differs')
  return docs, hashes


@functools.lru_cache(maxsize=1)
def _key_engine():
  import jax
  import jax.numpy as jnp
  require(jax.__version__ == '0.4.33' and all(x.platform == 'cpu' for x in jax.devices()), 'Require pinned CPU key engine')
  def one(label):
    root = jax.random.fold_in(jax.random.PRNGKey(20261024), jnp.uint32(0x50525231))
    for index in range(6): root = jax.random.fold_in(root, label[index])
    return root, jax.random.split(root, 16)[1]
  return jax.jit(jax.vmap(one))


def independent_keys(episode, position):
  """Prescribed fold domain using CPU JAX primitives, with no producer import."""
  import jax
  labels = np.array([[episode, position, i, b, p, t]
                    for i in range(4) for b in range(4) for p in range(32) for t in range(1, 21)], np.uint32)
  roots, samples = _key_engine()(jax.device_put(labels))
  roots, samples = jax.device_get((roots, samples))
  return np.asarray(roots).reshape(4, 128, 20, 2), np.asarray(samples).reshape(4, 128, 20, 2)


def symmetric_mean(logits, bins):
  require(logits.dtype == bins.dtype == np.float32 and logits.shape[-1] == 255 and bins.shape == (255,) and
          np.isfinite(logits).all() and np.isfinite(bins).all() and (np.diff(bins) > 0).all(), 'Invalid native oracle inputs')
  scores = logits.astype(np.float64)
  probabilities = np.exp(scores - scores.max(-1, keepdims=True))
  probabilities /= probabilities.sum(-1, keepdims=True)
  terms = probabilities * bins.astype(np.float64)
  pairs = terms[..., :127][..., ::-1] + terms[..., 128:]
  return terms[..., 127] + pairs.sum(-1)


def oracle_check(directory, published, first):
  require(published['path'] == 'first_chunk_oracle.npz', 'Unexpected first oracle file')
  path = safe_file(directory, published['path'])
  require(sha(path) == published['sha256'], 'First oracle bytes differ')
  with np.load(path, allow_pickle=False) as archive:
    require(set(archive.files) == {'reward', 'logits', 'bins', 'root_keys', 'sample_keys'}, 'First oracle schema differs')
    data = {name: archive[name].copy() for name in archive.files}
  numeric(data['reward'], np.float32, (4, 2, 8, 7, 20), 'first reward')
  numeric(data['logits'], np.float32, (4, 2, 8, 7, 20, 255), 'first logits')
  for key in ('root_keys', 'sample_keys'):
    numeric(data[key], np.uint32, (4, 8, 20, 2), 'first ' + key)
    require(data[key].tobytes() == first[key][0, :, :8].tobytes(), 'First oracle keys are not first source chunk')
  require(data['reward'].tobytes() == first['reward'][0, :, :, :8].tobytes() and
          data['bins'].tobytes() == first['bins'].tobytes(), 'First oracle predictions do not match saved trajectory')
  expected = symmetric_mean(data['logits'], data['bins'])
  errors = np.abs(data['reward'].astype(np.float64) - expected)
  failed = int((errors > 1e-6 + 1e-5 * np.abs(expected)).sum())
  receipt = dict(path=path.name, sha256=sha(path), max_abs_error=float(errors.max()), atol=1e-6, rtol=1e-5,
      finite_point_oracle_passed=failed == 0, full_model_independently_reproduced=False,
      compared_points=int(errors.size), failed_points=failed)
  require(failed == 0, 'Native symmetric mean fixed-tolerance oracle failed')
  compare_tree(receipt, published)
  checkpath = safe_file(directory, 'first_chunk_oracle_check.json')
  compare_tree(receipt, read_json(checkpath))
  return dict(receipt, oracle_check_sha256=sha(checkpath), oracle_dtype='float64',
              scope='first saved chunk only; no independent full-model or all-logit reproduction')


def audit(dataset, directory, expected, execution):
  dataset, directory, execution = [Path(x).resolve(strict=True) for x in (dataset, directory, execution)]
  check_deadline()
  require(np.__version__ == '1.26.4', 'Require pinned independent NumPy1.26.4')
  source_plan, source_result, source_items = source_authority(dataset)
  documents, bhashes = b_authority(directory, dataset, expected, execution)
  plan, result = documents['plan.json'], documents['result.json']
  require(plan['source_model_provenance'] == source_plan['model_provenance'], 'B source model metadata not authenticated source')
  require(plan['packages'].items() >= source_plan['packages'].items(), 'B package/source binding differs')
  source_receipts, truths, libraries = [], [], []
  for item in source_items:
    check_deadline()
    path = safe_file(dataset, item['path'])
    require(sha(path) == item['sha256'], 'Source bytes changed before read')
    truth, library, proof = load_source_numeric(path, item['episode'])
    require(sha(path) == item['sha256'], 'Source bytes changed during read')
    truths.append(truth); libraries.append(library)
    source_receipts.append(dict(episode=item['episode'], path=item['path'], sha256=item['sha256'], typed_hashes=proof))
  require(source_receipts == plan['input_artifacts'], 'B typed source receipts differ from independent numeric reads')
  truth = np.stack(truths)
  original_truth = np.asarray(source_result['external_control']['rewards'], np.float64)
  require(original_truth.shape == (8, 4, 7) and original_truth.tobytes() == truth.tobytes(), 'Source published truth bytes differ')
  artifacts = result['evidence_artifacts']
  require(type(artifacts) is list and len(artifacts) == 8 and [x['episode'] for x in artifacts] == list(EPISODES),
          'B episode artifacts incomplete/reordered')
  # Hash all eight B files before accessing any of their numeric contents.
  for item in artifacts:
    require(item['path'] == f"episode_{item['episode']:03}.npz" and sha(safe_file(directory, item['path'])) == item['sha256'],
            'B episode raw hash/path differs')
  rows, key_digests, first, common_bins = [], [], None, None
  for ei, item in enumerate(artifacts):
    check_deadline()
    path = safe_file(directory, item['path'])
    with np.load(path, allow_pickle=False) as archive:
      require(set(archive.files) == {'reward', 'root_keys', 'sample_keys', 'bins', 'episode', 'positions', 'true_cost', 'controls'},
              'B numeric archive schema differs')
      data = {name: archive[name].copy() for name in archive.files}
    numeric(data['episode'], np.int32, (), 'B episode'); numeric(data['positions'], np.int32, (4,), 'B positions')
    require(int(data['episode']) == item['episode'] and np.array_equal(data['positions'], POSITIONS), 'B numeric source axes differ')
    numeric(data['reward'], np.float32, SHAPE[1:], 'B rewards')
    numeric(data['true_cost'], np.float64, (4, 7), 'B truth'); numeric(data['controls'], np.float32, (4, 7, 12), 'B controls')
    require(data['true_cost'].tobytes() == truths[ei].tobytes() and data['controls'].tobytes() == libraries[ei].tobytes(),
            'B trajectory source truth/control mismatch')
    numeric(data['bins'], np.float32, (255,), 'B bins')
    require((np.diff(data['bins']) > 0).all(), 'B native bins not ordered')
    if common_bins is None: common_bins = data['bins'].copy()
    require(data['bins'].tobytes() == common_bins.tobytes(), 'B bins vary by episode')
    for name in ('root_keys', 'sample_keys'):
      numeric(data[name], np.uint32, (4, 4, 128, 20, 2), name)
      require(typed_sha(data[name]) == item[name + '_typed_sha256'], 'B key typed digest differs')
    require(typed_sha(data['reward']) == item['reward_typed_sha256'], 'B reward typed digest differs')
    for si, position in enumerate(POSITIONS):
      check_deadline()
      roots, samples = independent_keys(item['episode'], position)
      require(roots.tobytes() == data['root_keys'][si].tobytes() and samples.tobytes() == data['sample_keys'][si].tobytes(),
              'Named MC key domain or actual recorded sample keys differ')
    key_digests.append(dict(episode=item['episode'], root_keys=raw_digest(data['root_keys']), sample_keys=raw_digest(data['sample_keys'])))
    require(sha(path) == item['sha256'], 'B episode bytes changed during read')
    if ei == 0: first = data
    rows.append(data['reward'])
  point_oracle = oracle_check(directory, result['first_chunk_oracle'], first)
  independent = recompute(np.stack(rows), truth, np.asarray(EPISODES, np.int32))
  numeric_comparison = compare_tree(independent, result['analysis'])
  # Repeat raw-byte and exact committed-code checks after all computation.
  source_authority(dataset)
  again, again_hashes = b_authority(directory, dataset, expected, execution)
  require(again_hashes == bhashes and again == documents, 'B metadata changed during audit')
  for item in artifacts: require(sha(safe_file(directory, item['path'])) == item['sha256'], 'B outputs changed during audit')
  require(sha(safe_file(directory, 'first_chunk_oracle.npz')) == point_oracle['sha256'] and
          sha(safe_file(directory, 'first_chunk_oracle_check.json')) == point_oracle['oracle_check_sha256'], 'Oracle changed during audit')
  return dict(version=VERSION, implementation=IMPLEMENTATION, status='passed',
      expected_git_sha=expected, actual_git_sha=expected, source_git_sha=SOURCE_GIT,
      source_json_sha256=dict(SOURCE_PINS), b_json_sha256=bhashes,
      source_files_sha256=plan['source_files_sha256'], source_inputs=source_receipts, b_prediction_artifacts=artifacts,
      independent_statistics=numeric_comparison, native_point_oracle=point_oracle,
      independent_keys=dict(all_root_and_sample_key_bytes_exact=True, root_keys_checked=327680,
          sample_keys_checked=327680, domains=['episode', 'position', 'initial', 'block', 'path', 'step'],
          private_seed=20261024, tag=0x50525231, sample_seed_contract='ninjax3.6.3 split(private_root,16)[1]',
          branch_control_common=True, path_order='block0..3, path0..31; step1..20', evidence=key_digests,
          scope='Independent enumeration; reused JAX0.4.33 CPU PRNG primitives, not independent PRNG implementation or new NJ/model execution'),
      primary_status=independent['primary_status'], utility_status=independent['utility_reliability']['status'],
      source_proof_scope='Authenticated source carry/physical/action/key receipts reused; no new physics/carry/model re-execution',
      source_physics_independently_reexecuted=False, source_model_independently_reexecuted=False,
      b_runtime_scope='Artifact and owned-supervisor/CUDA identity receipts checked; this CPU audit does not inspect live GPU processes',
      first_chunk_only_point_oracle=True, full_model_independently_reproduced=False,
      actual_numeric_episode_reads=8, actual_prediction_episode_reads=8, actual_gpu_devices_used=0,
      gpu_devices_used=0, model_calls=0, physics_steps=0, render_calls=0, optimizer_updates=0,
      records_decoded=False, blind_arrays_read=False, synthetic_fixture=False,
      inputs_unchanged=True, code_before_after_git_exact=True,
      elapsed_parent_entry_seconds=time.monotonic() - ENTRY_STARTED, seconds=SECONDS,
      audit_python=sys.version, audit_numpy=np.__version__, audit_jax=sys.modules['jax'].__version__,
      pointwise_bc_sign_scope='All F64 abs1e-10; primary/CI zero sides and negative-count integer exact, not each individual BC sign',
      own_posix_alarm=os.name == 'posix', **AUTHORITY)


def selftest():
  paths = np.array([[1., 0.], [0., 1.]], np.float64)
  value = moments(paths, np.zeros(2, np.float64))
  require(value['bc_mse'] == -.25 and value['plugin_centered_mse'] == 0. and value['mc_correction'] == .25,
          'Signed off-diagonal counterexample failed')
  target = np.array([1., .25, -.5], np.float64)
  perfect = moments(np.tile(target, (4, 1)), target)
  require(perfect['bc_mse'] == perfect['plugin_centered_mse'] == perfect['mc_correction'] == 0., 'Perfect residual zero failed')
  require(int(choose(np.array([0., 1e-10]))) == 0 and int(choose(np.array([0., np.nextafter(1e-10, np.inf)]))) == 1,
          'Closed tie boundary failed')
  logits, bins = np.zeros((1, 255), np.float32), np.arange(-127, 128, dtype=np.float32)
  require(float(symmetric_mean(logits, bins)[0]) == 0., 'Symmetric point expectation failed')
  truth = np.zeros((8, 4, 7), np.float64); truth[0, :, 6] = 5.; truth[1:, :, 2] = 1.
  const, _, _ = constant(truth, np.asarray(EPISODES, np.int32))
  require(const['control_by_episode'][0] == 2 and const['folds'][0]['training_states'] == 28,
          'LOEO leakage counterexample failed')
  return dict(version=VERSION, implementation=IMPLEMENTATION, status='synthetic_selftest_passed',
      synthetic_fixture=True, actual_numeric_episode_reads=0, actual_prediction_episode_reads=0,
      gpu_devices_used=0, model_calls=0, physics_steps=0, records_decoded=False, blind_arrays_read=False,
      test_scope='Small independent moment, perfect-zero, LOEO, tie-boundary and symmetric-probability examples only',
      elapsed_parent_entry_seconds=time.monotonic() - ENTRY_STARTED, **AUTHORITY)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--selftest', action='store_true')
  for name in ('dataset', 'result-dir', 'execution-root', 'output'):
    parser.add_argument('--' + name, type=Path)
  parser.add_argument('--expected-git')
  args = parser.parse_args()
  if args.selftest:
    require(all(getattr(args, k) is None for k in ('dataset', 'result_dir', 'execution_root', 'expected_git')),
            'Selftest must not access real sources')
    value = selftest()
  else:
    require(all(getattr(args, k) is not None for k in ('dataset', 'result_dir', 'execution_root', 'expected_git', 'output')),
            'Real audit requires all five bound CLI arguments')
    require(args.output.parent.resolve(strict=True) != args.dataset.resolve(strict=True) and
            not args.output.exists(), 'Use a fresh audit output outside immutable source artifacts')
    value = audit(args.dataset, args.result_dir, args.expected_git, args.execution_root)
  check_deadline()
  if args.output is not None:
    require(not args.output.exists(), 'Audit output already exists')
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
      json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    check_deadline()
  else:
    print(json.dumps(value, sort_keys=True, allow_nan=False))
  return 0


if __name__ == '__main__':
  try:
    raise SystemExit(main())
  except TimeoutError:
    os._exit(124)
