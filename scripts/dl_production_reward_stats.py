"""Pure NumPy statistics for the frozen DL-production-reward-r1 object.

This module receives numeric arrays only. Source identity, planned independent
MC keys, production model semantics and execution completion are the caller's
responsibility. No artifact, model, environment or production RNG is accessed.
Signed BC-MSE estimates conditional-mean error, not a reward distribution.
"""
import hashlib

import numpy as np


VERSION = 'DL-production-reward-r1'
IMPLEMENTATION = 'DL-production-reward-stats-engineering-r1'
EPISODES = (9, 14, 15, 18, 20, 24, 30, 31)
POSITIONS = (1, 300, 600, 900)
PREDICTION_SHAPE = (8, 4, 4, 2, 128, 7, 20)
TRUTH_SHAPE = (8, 4, 7)
CONTROL_NAMES = ('zero', 'all+', 'all-', 'lift-LR+', 'lift-LR-',
                 'extend-diag+', 'extend-diag-')
TIE = 1e-10
MSE_EPSILON_LIMIT = .0004
UTILITY_EPSILON_LIMIT = .002
POSITIVE_RELATIVE_LIMIT = .2
STAT_SEED = 20261025
BOOTSTRAP_DRAWS = 2000
SUBSETS = {
    'first32': (0,), 'first64': (0, 1), 'pooled128': (0, 1, 2, 3),
    'last64': (2, 3), 'block0': (0,), 'block1': (1,),
    'block2': (2,), 'block3': (3,)}


def _require(condition, message):
  if not condition:
    raise ValueError(message)


def _numeric(value, dtype, shape, name):
  _require(isinstance(value, np.ndarray) and value.dtype == np.dtype(dtype) and
           value.shape == shape and not value.dtype.hasobject,
           f'{name}: require {np.dtype(dtype)} shape {shape}')
  _require(np.isfinite(value).all(), f'{name}: nonfinite input')


def _digest(value):
  value = np.ascontiguousarray(value)
  return dict(dtype=value.dtype.str, shape=list(value.shape),
              sha256=hashlib.sha256(value.tobytes()).hexdigest())


def _argmax(value):
  _require(np.isfinite(value).all(), 'Nonfinite choice input')
  return np.argmax(value.max(-1, keepdims=True) - value <= TIE, -1).astype(np.int32)


def _aggregate(per_initial):
  """One branch, [episode, state, fixed initial sample]; never pool before loss."""
  _require(per_initial.dtype == np.float64 and per_initial.ndim == 3 and
           np.isfinite(per_initial).all(), 'Invalid per-initial metric')
  state = per_initial.mean(2, dtype=np.float64)
  episode = state.mean(1, dtype=np.float64)
  return dict(per_initial=per_initial.tolist(), state_values=state.tolist(),
              episode_values=episode.tolist(), primary=float(episode.mean(dtype=np.float64)))


def _interval(episode_values, draws):
  _require(np.isfinite(episode_values).all(), 'Nonfinite episode values')
  boot = episode_values[draws].mean(1, dtype=np.float64)
  lower, upper = np.quantile(boot, [.025, .975], method='linear')
  return dict(primary=float(episode_values.mean(dtype=np.float64)), lower95=float(lower),
              upper95=float(upper), episode_values=episode_values.tolist())


def _gap(per_initial, draws):
  value = _aggregate(per_initial)
  return dict(value, **{k: v for k, v in _interval(
      np.asarray(value['episode_values'], np.float64), draws).items()
      if k not in ('primary', 'episode_values')})


def _moments(path_returns, target):
  """Generic numeric fixture helper; production shape is enforced by report.

  path_returns has [..., N, control], target [..., control]. Both are float64.
  MC paths must be independent across N according to verified private keys.
  Copying paths and pretending to increase N is not a valid invocation.
  """
  _require(isinstance(path_returns, np.ndarray) and path_returns.dtype == np.float64 and
           path_returns.ndim >= 2 and path_returns.shape[-2] >= 2 and
           path_returns.shape[-1] > 0, 'Finite F64 paths with N>=2 required')
  _numeric(target, np.float64, path_returns.shape[:-2] + (path_returns.shape[-1],), 'target')
  _require(np.isfinite(path_returns).all(), 'Nonfinite cumulative path return')
  n = path_returns.shape[-2]
  centered = path_returns - path_returns.mean(-1, keepdims=True, dtype=np.float64)
  centered_target = target - target.mean(-1, keepdims=True, dtype=np.float64)
  mean = path_returns.mean(-2, dtype=np.float64)
  centered_mean = centered.mean(-2, dtype=np.float64)
  # Translation by the fixed target does not change path variance or the
  # scientific BC-MSE. Residual coordinates avoid repeatedly averaging a
  # non-binary-representable constant (e.g. 1/7) before subtracting itself.
  # Perfect paths then yield exact zero, consistent with the U-statistic.
  residual = centered - centered_target[..., None, :]
  variance = residual.var(-2, ddof=1, dtype=np.float64)
  per_control_plugin = np.square(residual.mean(-2, dtype=np.float64))
  correction = variance / n
  bc = (per_control_plugin - correction).mean(-1, dtype=np.float64)
  _require(np.isfinite(bc).all() and np.isfinite(variance).all(), 'Nonfinite BC-MSE')
  return dict(cumulative_mean=mean, centered_cumulative_mean=centered_mean,
      centered_path_sample_variance=variance, centered_variance_over_n=correction,
      bc_mse=bc, plugin_centered_mse=per_control_plugin.mean(-1, dtype=np.float64),
      mc_correction=correction.mean(-1, dtype=np.float64),
      plugin_level_mse=np.square(mean - target).mean(-1, dtype=np.float64))


def _constant(truth, episodes):
  centered = truth - truth.mean(-1, keepdims=True, dtype=np.float64)
  vectors, folds = [], []
  for held in range(len(episodes)):
    train = [i for i in range(len(episodes)) if i != held]
    vector = centered[train].mean((0, 1), dtype=np.float64)
    vectors.append(vector)
    folds.append(dict(heldout_episode=int(episodes[held]),
        training_episodes=[int(episodes[i]) for i in train],
        centered_cost_vector=vector.tolist(), training_states=len(train) * truth.shape[1]))
  vectors = np.stack(vectors)
  choices = _argmax(vectors)
  selected = np.take_along_axis(truth, np.broadcast_to(
      choices[:, None, None], (len(episodes), truth.shape[1], 1)), -1)[..., 0]
  mse = np.square(vectors[:, None] - centered).mean(-1, dtype=np.float64)
  regret = truth.max(-1) - selected
  return dict(folds=folds, centered_cost_vectors=vectors.tolist(),
      control_by_episode=choices.tolist(), state_mse=mse.tolist(),
      episode_mse=mse.mean(1, dtype=np.float64).tolist(),
      primary_mse=float(mse.mean(1, dtype=np.float64).mean(dtype=np.float64)),
      state_selected_reward=selected.tolist(),
      episode_selected_reward=selected.mean(1, dtype=np.float64).tolist(),
      primary_selected_reward=float(selected.mean(1, dtype=np.float64).mean(dtype=np.float64)),
      state_finite_library_regret=regret.tolist(),
      primary_finite_library_regret=float(regret.mean(dtype=np.float64)),
      correction=0., labels_used='other seven episodes; each of their four states equally weighted'), mse, selected


def _score(path_returns, truth, constant_mse, constant_selected, draws, path_indices):
  e, s, r, b, n, a = path_returns.shape
  target = np.broadcast_to(truth[:, :, None, None], (e, s, r, b, a))
  moments = _moments(path_returns, target)
  choices = _argmax(moments['cumulative_mean'])
  selected = np.take_along_axis(target, choices[..., None], -1)[..., 0]
  regret = target.max(-1) - selected
  interfaces = {}
  for branch, name in enumerate(('q', 'p')):
    metrics = {key: _aggregate(moments[key][..., branch]) for key in
               ('bc_mse', 'plugin_centered_mse', 'mc_correction', 'plugin_level_mse')}
    metrics.update(choices=choices[..., branch].tolist(),
        selected_reward=_aggregate(selected[..., branch]),
        finite_library_regret=_aggregate(regret[..., branch]),
        negative_bc_mse_estimates=int(np.count_nonzero(moments['bc_mse'][..., branch] < 0)))
    interfaces[name] = metrics
  mse_gaps = dict(p_minus_q=_gap(moments['bc_mse'][..., 1] - moments['bc_mse'][..., 0], draws),
      constant_minus_q=_gap(constant_mse[:, :, None] - moments['bc_mse'][..., 0], draws))
  utility_gaps = dict(q_minus_p=_gap(selected[..., 0] - selected[..., 1], draws),
      q_minus_constant=_gap(selected[..., 0] - constant_selected[:, :, None], draws))
  return dict(paths=n, path_indices=path_indices.tolist(),
      cumulative_mean=moments['cumulative_mean'].tolist(),
      centered_cumulative_mean=moments['centered_cumulative_mean'].tolist(),
      centered_path_sample_variance=moments['centered_path_sample_variance'].tolist(),
      centered_variance_over_n=moments['centered_variance_over_n'].tolist(),
      interfaces=interfaces, mse_gaps=mse_gaps, utility_gaps=utility_gaps)


def _reliability(subsets, group, epsilon_limit, family):
  primary = subsets['pooled128'][group]
  metrics = {}
  for key, value in primary.items():
    points = {name: data[group][key]['primary'] for name, data in subsets.items()}
    block_values = np.asarray([points['block' + str(i)] for i in range(4)], np.float64)
    components = dict(nested32_64=abs(points['first32'] - points['first64']),
        nested64_128=abs(points['first64'] - points['pooled128']),
        independent_halves=abs(points['first64'] - points['last64']),
        block_t_margin=float(3.182 * block_values.std(ddof=1) / 2))
    epsilon = float(max(components.values()))
    point = value['primary']
    robust = (point > 0 and epsilon <= POSITIVE_RELATIVE_LIMIT * point and
              value['lower95'] - epsilon > 0)
    metrics[key] = dict(epsilon=epsilon, components=components,
        stable=epsilon <= epsilon_limit, point=point, lower95=value['lower95'],
        upper95=value['upper95'], lower95_minus_epsilon=value['lower95'] - epsilon,
        upper95_plus_epsilon=value['upper95'] + epsilon,
        positive_margin_robust=bool(robust),
        negative_margin_robust=bool(value['upper95'] + epsilon <= 0),
        relative_positive_margin_limit=POSITIVE_RELATIVE_LIMIT)
  stable = all(value['stable'] for value in metrics.values())
  positive = all(value['positive_margin_robust'] for value in metrics.values())
  negative = any(value['negative_margin_robust'] for value in metrics.values())
  if not stable:
    status = 'MC-inconclusive'
  elif negative:
    status = 'finite_native_' + family + '_not_supported'
  elif positive:
    status = 'finite_native_' + family + '_exploratory_supported'
  else:
    status = 'finite_native_' + family + '_inconclusive'
  return dict(metrics=metrics, epsilon_limit=epsilon_limit, mc_stable=stable,
      positive_margin_conditions_met=positive,
      necessary_directions_robust=bool(stable and positive), status=status,
      interpretation='Finite numerical reliability indicator, not a rigorous MC integration-error bound')


def _choice_compare(left, right):
  comparisons = {}
  for name in ('q', 'p'):
    first = left['interfaces'][name]
    second = right['interfaces'][name]
    same = np.asarray(first['choices'], np.int32) == np.asarray(second['choices'], np.int32)
    change = (np.asarray(first['selected_reward']['per_initial'], np.float64) -
              np.asarray(second['selected_reward']['per_initial'], np.float64))
    comparisons[name] = dict(equal_choices=same.tolist(),
        agreement_fraction=float(same.mean(dtype=np.float64)),
        changed_choices=int(np.count_nonzero(~same)), signed_selected_reward_change=change.tolist(),
        absolute_switch_cost=np.abs(change).tolist(),
        maximum_absolute_switch_cost=float(np.abs(change).max()),
        mean_absolute_switch_cost=float(np.abs(change).mean(dtype=np.float64)))
  return comparisons


def report(predictions, truth, episodes):
  """Return the complete frozen statistic tree, without execution authority.

  predictions: F32[8,4,4,2,128,7,20], branch q0/p1, ordered block/path.
  truth: F64[8,4,7], actual H20 totals. episodes: the fixed int32[8] IDs.
  Source/MC-key identity and completion must be independently verified outside.
  """
  _numeric(predictions, np.float32, PREDICTION_SHAPE, 'predictions')
  _numeric(truth, np.float64, TRUTH_SHAPE, 'truth')
  _numeric(episodes, np.int32, (8,), 'episodes')
  _require(np.array_equal(episodes, EPISODES), 'Fixed unique episode identity/order changed')
  before = {key: _digest(value) for key, value in
            dict(predictions=predictions, truth=truth, episodes=episodes).items()}
  with np.errstate(over='raise', invalid='raise', divide='raise'):
    cumulative = predictions.sum(-1, dtype=np.float64)
    constant, constant_mse, constant_selected = _constant(truth, episodes)
    draws = np.random.default_rng(STAT_SEED).integers(0, 8, (BOOTSTRAP_DRAWS, 8))
    subsets = {}
    for name, blocks in SUBSETS.items():
      indices = np.concatenate([np.arange(block * 32, (block + 1) * 32, dtype=np.int32)
                                for block in blocks])
      subsets[name] = _score(cumulative[..., indices, :], truth, constant_mse,
                            constant_selected, draws, indices)
      subsets[name]['blocks'] = list(blocks)
    mse = _reliability(subsets, 'mse_gaps', MSE_EPSILON_LIMIT, 'mean')
    utility = _reliability(subsets, 'utility_gaps', UTILITY_EPSILON_LIMIT, 'choice')
    choice_stability = {name + '_vs_pooled128': _choice_compare(value, subsets['pooled128'])
                        for name, value in subsets.items() if name != 'pooled128'}
    choice_stability['first64_vs_last64'] = _choice_compare(subsets['first64'], subsets['last64'])
    extrema = dict(native_reward_minimum=float(predictions.min()),
        native_reward_maximum=float(predictions.max()),
        cumulative_path_minimum=float(cumulative.min()), cumulative_path_maximum=float(cumulative.max()),
        per_initial_native_minimum=predictions.min(axis=(-1, -2, -3)).tolist(),
        per_initial_native_maximum=predictions.max(axis=(-1, -2, -3)).tolist(),
        per_initial_cumulative_minimum=cumulative.min(axis=(-1, -2)).tolist(),
        per_initial_cumulative_maximum=cumulative.max(axis=(-1, -2)).tolist())
  after = {key: _digest(value) for key, value in
           dict(predictions=predictions, truth=truth, episodes=episodes).items()}
  _require(before == after, 'Production reward statistics modified inputs')
  return dict(version=VERSION, implementation=IMPLEMENTATION, episodes=list(EPISODES),
      positions=list(POSITIONS), prediction_shape=list(PREDICTION_SHAPE),
      truth_shape=list(TRUTH_SHAPE), controls=list(CONTROL_NAMES), branches=['q', 'p'],
      horizon=20, blocks=4, paths_per_block=32, future_paths=128, initial_samples=4,
      axes=['episode', 'state', 'initial_sample', 'branch', 'future_path', 'control', 'step'],
      primary_subset='pooled128', constant=constant, subsets=subsets,
      mse_reliability=mse, utility_reliability=utility, primary_status=mse['status'],
      choice_stability=choice_stability, path_extrema=extrema,
      bc_mse_definition='mean_controls((mean_paths(control_centered_return)-centered_truth)^2 - ddof1_var/N)',
      bc_mse_numeric_coordinates='control-centered residual paths; fixed-target translation leaves variance unchanged',
      plugin_level_mse_definition='MSE of uncentered conditional path mean; no MC bias correction',
      negative_bc_mse_clipped=False, initial_samples_averaged_before_loss=False,
      future_paths_independence_verified_by_stats=False, source_provenance_verified_by_stats=False,
      source_or_model_execution_verified_by_stats=False,
      tie_tolerance=TIE, tie_rule='lowest control ID with max(value)-value<=1e-10',
      bootstrap_draws=BOOTSTRAP_DRAWS, statistical_seed=STAT_SEED, percentile_method='linear',
      bootstrap_unit='whole calibration episodes; fixed model/LOEO-constant episode scores',
      interval_scope='Nominal calibration description; LOEO constants overlap, no refitting uncertainty or independent confirmation',
      mc_error_bound_claimed=False, inputs_unchanged=True, input_digests=before,
      calibration_only=True, execution_scope='numeric statistics only; runtime model/GPU verification belongs to caller',
      model_calls=0, physics_steps=0, optimizer_updates=0, gpu_devices_used=0,
      blind_arrays_read=False, records_decoded=False, current_rgb_increment_established=False,
      active_rep_compression_effect_established=False, method_effectiveness_established=False,
      method_evidence=False, actor_applicable=False,
      real_gate_authority=False, eligible_method_training=False, training_authorized=False,
      proceed_short_training=False, decision='stop_after_fixed_production_reward_diagnostic',
      negative_scope='Only the specified native H20 conditional-mean reward-support estimator; not H1 head, latent information, rawK or future learning')
