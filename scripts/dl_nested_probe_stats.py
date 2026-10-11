"""Pure CPU statistics for DL-nested-probe-r1.

No artifacts, actors, model, simulator or supervisor are loaded here. The caller
binds the actual eight-episode numeric inputs and enforces the runtime budget.
General finite float64 fixtures with at least three episodes are also accepted.
The unchanged original DL-action-information-r1 remains a different object.
"""
import hashlib

import numpy as np


VERSION = 'DL-nested-probe-r1'
IMPLEMENTATION = 'DL-nested-probe-stats-engineering-r1'
LAMBDAS = (1., .1, .01, .001)
TIE = 1e-10
NUMERIC_TOL = 1e-10
STAT_SEED = 20261021
BOOTSTRAP_DRAWS = 2000


def _require(condition, message):
  if not condition:
    raise ValueError(message)


def _numeric(value, shape, name):
  _require(isinstance(value, np.ndarray) and value.dtype == np.float64 and
           value.shape == shape and not value.dtype.hasobject,
           f'{name}: require float64 shape {shape}')
  _require(np.isfinite(value).all(), f'{name}: nonfinite input')


def _digest(value):
  value = np.ascontiguousarray(value)
  _require(not value.dtype.hasobject, 'Object dtype cannot be hashed')
  return dict(dtype=value.dtype.str, shape=list(value.shape),
              sha256=hashlib.sha256(value.tobytes()).hexdigest())


def _argmax(values):
  _require(np.isfinite(values).all(), 'Nonfinite choice input')
  return np.argmax(values.max(-1, keepdims=True) - values <= TIE,
                   axis=-1).astype(np.int32)


def _standardize(train, heldout):
  mean = train.mean(0, dtype=np.float64)
  centered = train - mean
  rms = float(np.sqrt(np.square(centered).sum(1).mean(dtype=np.float64)))
  _require(np.isfinite(rms), 'Nonfinite training feature-group RMS')
  zero = rms == 0.
  if zero:
    _require(np.all(centered == 0), 'Feature-group RMS underflowed for nonzero data')
    fitted, transformed = np.zeros_like(train), np.zeros_like(heldout)
  else:
    fitted, transformed = centered / rms, (heldout - mean) / rms
  _require(np.isfinite(fitted).all() and np.isfinite(transformed).all(),
           'Nonfinite standardized feature group')
  return fitted, transformed, dict(mean=_digest(mean), rms=rms, zero_rms=zero,
      dimensions=train.shape[1], train_mean_total_squared_norm=float(
          np.square(fitted).sum(1).mean(dtype=np.float64)),
      zero_rms_rule='zero group for train and heldout; no rows removed')


class _Kernel:
  """One train-only preprocessing/eigendecomposition, reused across fixed lambdas."""

  def __init__(self, h_train, z_train, targets, h_test, z_test):
    _require(isinstance(h_train, np.ndarray) and h_train.ndim == 2 and
             min(h_train.shape) > 0, 'Positive two-dimensional train h required')
    n, hd = h_train.shape
    _require(isinstance(z_train, np.ndarray) and z_train.ndim == 2 and
             z_train.shape[0] == n and z_train.shape[1] > 0, 'Train z dimensions differ')
    _require(isinstance(targets, np.ndarray) and targets.ndim == 2 and
             targets.shape[0] == n and targets.shape[1] > 0, 'Train targets dimensions differ')
    _require(isinstance(h_test, np.ndarray) and h_test.ndim == 2 and
             len(h_test) > 0, 'Positive two-dimensional heldout h required')
    nt, zd, outputs = len(h_test), z_train.shape[1], targets.shape[1]
    for value, shape, name in (
        (h_train, (n, hd), 'train h'), (z_train, (n, zd), 'train z'),
        (targets, (n, outputs), 'train targets'), (h_test, (nt, hd), 'heldout h'),
        (z_test, (nt, zd), 'heldout z')):
      _numeric(value, shape, name)
    ht, he, hm = _standardize(h_train, h_test)
    zt, ze, zm = _standardize(z_train, z_test)
    self.x = np.concatenate((ht, zt), 1)
    xe = np.concatenate((he, ze), 1)
    self.n, self.outputs = n, outputs
    self.intercept = targets.mean(0, dtype=np.float64)
    self.y = targets - self.intercept
    self.kernel = (self.x @ self.x.T) / n
    self.cross = (xe @ self.x.T) / n
    self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.kernel)
    _require(np.isfinite(self.eigenvalues).all() and np.isfinite(self.eigenvectors).all(),
             'Nonfinite kernel eigendecomposition')
    _require(float(self.eigenvalues.min()) >= -NUMERIC_TOL, 'Kernel PSD check failed')
    orth = float(np.max(np.abs(self.eigenvectors.T @ self.eigenvectors - np.eye(n))))
    reconstruction = float(np.max(np.abs(
        (self.eigenvectors * self.eigenvalues) @ self.eigenvectors.T - self.kernel)))
    trace = float(self.eigenvalues.sum())
    energy = hm['train_mean_total_squared_norm'] + zm['train_mean_total_squared_norm']
    _require(max(orth, reconstruction, abs(trace - energy)) <= NUMERIC_TOL,
             'Kernel eigenvectors/reconstruction/trace check failed')
    self.projected_targets = self.eigenvectors.T @ self.y
    self.preprocessing = dict(training_rows=n, heldout_rows=nt,
        feature_dimensions=hd + zd, output_dimensions=outputs,
        train_column_mean=self.intercept.tolist(), intercept_regularized=False,
        h_group=hm, z_group=zm,
        zero_rms_groups=int(hm['zero_rms']) + int(zm['zero_rms']),
        eigen=dict(minimum_eigenvalue=float(self.eigenvalues.min()),
            maximum_eigenvalue=float(self.eigenvalues.max()), trace=trace,
            eigenvalue_count=n, all_eigenvalues_retained=True, cutoff_applied=False,
            orthogonality_max_abs_error=orth, reconstruction_max_abs_error=reconstruction))

  def predict(self, mean_lambda):
    _require(type(mean_lambda) is float and mean_lambda in LAMBDAS,
             'Lambda must be one of the fixed positive mean-objective rules')
    denominator = self.eigenvalues + mean_lambda
    _require(np.isfinite(denominator).all() and np.all(denominator > 0),
             'Invalid ridge eigenvalue denominator')
    # Retain every eigenvalue, including numerical near-zero values. No cutoff.
    coefficient = self.eigenvectors @ (self.projected_targets / denominator[:, None])
    prediction = self.cross @ coefficient + self.intercept
    equation_error = float(np.max(np.abs(
        (self.kernel + mean_lambda * np.eye(self.n)) @ coefficient - self.y)))
    _require(np.isfinite(prediction).all() and np.isfinite(coefficient).all() and
             equation_error <= NUMERIC_TOL, 'Ridge numerical equation check failed')
    weights = (self.x.T @ coefficient) / self.n
    residual = self.kernel @ coefficient - self.y
    error = float(np.square(residual).sum(1).mean(dtype=np.float64))
    norm = float(np.square(weights).sum(dtype=np.float64))
    _require(np.isfinite(error) and np.isfinite(norm), 'Nonfinite ridge objective')
    metadata = dict(self.preprocessing, regularization_lambda=mean_lambda,
        gram_diagonal_regularizer=self.n * mean_lambda,
        objective='mean_rows(sum_outputs(squared_error)) + lambda * ||W||_F^2',
        solver='untruncated symmetric kernel eigh; K=X X.T/n',
        ridge_equation_max_abs_error=equation_error,
        training_mean_sum_squared_error=error, training_mse_per_control=error / self.outputs,
        squared_weight_norm=norm, regularized_objective=error + mean_lambda * norm,
        maximum_ridge_eigengain=float(np.max(self.eigenvalues / denominator)))
    return prediction, metadata


def ridge_predict(h_train, z_train, targets, h_test, z_test, mean_lambda=1.):
  """Generic finite fixture interface; caller supplies already row-centered labels."""
  with np.errstate(over='raise', invalid='raise', divide='raise'):
    return _Kernel(h_train, z_train, targets, h_test, z_test).predict(mean_lambda)


def _select_common_lambda(candidates):
  """Global set rules are order-independent; no iterative pairwise tie chaining."""
  _require(isinstance(candidates, (list, tuple)) and len(candidates) == len(LAMBDAS),
           'All fixed lambda candidates required')
  _require({x['lambda'] for x in candidates} == set(LAMBDAS), 'Changed/duplicate lambda grid')
  for candidate in candidates:
    _require(type(candidate['lambda']) is float and
             type(candidate['pooled_utility']) is float and
             type(candidate['pooled_mse']) is float and
             np.isfinite(candidate['pooled_utility']) and
             np.isfinite(candidate['pooled_mse']) and candidate['pooled_mse'] >= 0,
             'Nonfinite or malformed common candidate score')
  maximum = max(x['pooled_utility'] for x in candidates)
  utility_set = [x for x in candidates if x['pooled_utility'] >= maximum - TIE]
  minimum = min(x['pooled_mse'] for x in utility_set)
  mse_set = [x for x in utility_set if x['pooled_mse'] <= minimum + TIE]
  selected = max(x['lambda'] for x in mse_set)
  return dict(selected_lambda=selected, maximum_pooled_utility=maximum,
      utility_tie_candidates=sorted((x['lambda'] for x in utility_set), reverse=True),
      minimum_mse_within_utility_set=minimum,
      mse_tie_candidates=sorted((x['lambda'] for x in mse_set), reverse=True),
      rule='global U>=max(U)-1e-10; within set MSE<=min(MSE)+1e-10; largest lambda',
      q_p_selection_shared=True, branch_weights=[.5, .5],
      equal_effective_degrees_of_freedom_claimed=False)


def _validate(rewards, h, latent, episodes):
  _require(isinstance(rewards, np.ndarray) and rewards.ndim == 3 and
           rewards.shape[0] >= 3 and rewards.shape[1] > 0 and rewards.shape[2] >= 2,
           'At least three episodes, positive states, and two controls required')
  e, s, a = rewards.shape
  _require(isinstance(h, np.ndarray) and h.ndim == 3 and h.shape[-1] > 0,
           'Positive h feature dimension required')
  _require(isinstance(latent, np.ndarray) and latent.ndim == 6 and
           latent.shape[2] > 0 and latent.shape[3] == 2 and min(latent.shape[-2:]) > 0,
           'Positive samples, two interfaces, and positive latent dimensions required')
  r, f, c = latent.shape[2], latent.shape[-2], latent.shape[-1]
  _numeric(rewards, (e, s, a), 'rewards')
  _numeric(h, (e, s, h.shape[-1]), 'h')
  _numeric(latent, (e, s, r, 2, f, c), 'latent')
  _require(isinstance(episodes, np.ndarray) and episodes.dtype == np.int32 and
           episodes.shape == (e,) and np.all(episodes >= 0) and
           len(np.unique(episodes)) == e, 'Unique nonnegative int32 episode IDs required')
  return e, s, r, a


def _rows(h, latent, targets, training, heldout, branch):
  s, r, hd = h.shape[1], latent.shape[2], h.shape[-1]
  zd, outputs = int(np.prod(latent.shape[-2:])), targets.shape[-1]
  n = len(training) * s * r
  ht = np.repeat(h[training, :, None], r, 2).reshape((n, hd))
  he = np.repeat(h[heldout, :, None], r, 1).reshape((s * r, hd))
  zt = latent[training, :, :, branch].reshape((n, zd))
  ze = latent[heldout, :, :, branch].reshape((s * r, zd))
  yt = np.repeat(targets[training, :, None], r, 2).reshape((n, outputs))
  return ht, zt, yt, he, ze


def _evaluate(prediction, rewards, target, samples):
  states, controls = rewards.shape
  prediction = prediction.reshape((states, samples, controls))
  choices = _argmax(prediction)
  truth = np.broadcast_to(rewards[:, None], (states, samples, controls))
  selected = np.take_along_axis(truth, choices[..., None], -1)[..., 0]
  regret = rewards.max(-1)[:, None] - selected
  mse = np.square(prediction - target[:, None]).mean(-1, dtype=np.float64)
  return dict(predicted_centered_reward=prediction.tolist(), choices=choices.tolist(),
      selected_reward=selected.tolist(), finite_library_regret=regret.tolist(),
      state_selected_reward=selected.mean(1, dtype=np.float64).tolist(),
      mean_selected_reward=float(selected.mean(1, dtype=np.float64).mean(dtype=np.float64)),
      sample_mse_per_control=mse.tolist(),
      mean_mse_per_control=float(mse.mean(1, dtype=np.float64).mean(dtype=np.float64)))


def _interval(values, draws):
  bootstrap = values[draws].mean(1, dtype=np.float64)
  low, high = np.quantile(bootstrap, [.025, .975], method='linear')
  _require(np.isfinite(bootstrap).all(), 'Nonfinite constructed episode bootstrap')
  return dict(primary=float(values.mean(dtype=np.float64)), lower95=float(low),
              upper95=float(high), episode_values=values.tolist())


def nested_readout(rewards, h, latent, episodes):
  """Select one common lambda inside each outer episode-isolated training set.

  Production shapes/onehot/source proof are enforced by the caller. This core
  accepts smaller finite fixtures; sample/episode indices are never features.
  Only the chosen common rule is evaluated on each outer heldout episode.
  """
  e, s, r, a = _validate(rewards, h, latent, episodes)
  before = {k: _digest(v) for k, v in dict(rewards=rewards, h=h, latent=latent, episodes=episodes).items()}
  with np.errstate(over='raise', invalid='raise', divide='raise'):
    targets = rewards - rewards.mean(-1, keepdims=True, dtype=np.float64)
    predictions = np.empty((e, s, r, 2, a), np.float64)
    choices = np.empty((e, s, r, 2), np.int32)
    selected = np.empty((e, s, r, 2), np.float64)
    mses = np.empty((e, s, r, 2), np.float64)
    constant_controls = np.empty(e, np.int32)
    constant_rewards = np.empty((e, s), np.float64)
    outer_folds, decompositions, evaluations = [], 0, 0
    for outer in range(e):
      outer_training = np.asarray([index for index in range(e) if index != outer], np.int32)
      # Each inner validation episode gets its own train-only h/z normalization.
      inner_folds = []
      score_utility = np.empty((e - 1, len(LAMBDAS), 2), np.float64)
      score_mse = np.empty_like(score_utility)
      for inner_index, inner in enumerate(outer_training):
        training = outer_training[outer_training != inner]
        kernels = {name: _Kernel(*_rows(h, latent, targets, training, inner, branch))
                   for branch, name in enumerate(('q', 'p'))}
        decompositions += 2
        candidate_results = []
        for li, rule in enumerate(LAMBDAS):
          branch_results = {}
          for branch, name in enumerate(('q', 'p')):
            prediction, fit = kernels[name].predict(rule)
            evaluation = _evaluate(prediction, rewards[inner], targets[inner], r)
            score_utility[inner_index, li, branch] = evaluation['mean_selected_reward']
            score_mse[inner_index, li, branch] = evaluation['mean_mse_per_control']
            branch_results[name] = dict(evaluation, fit=fit)
            evaluations += 1
          candidate_results.append(dict(lambda_value=rule, interfaces=branch_results))
        inner_folds.append(dict(validation_episode=int(episodes[inner]),
            training_episodes=episodes[training].astype(int).tolist(),
            training_rows=len(training) * s * r, validation_rows=s * r,
            train_state_weight=1 / (len(training) * s),
            train_row_weight=1 / (len(training) * s * r),
            preprocess={name: kernel.preprocessing for name, kernel in kernels.items()},
            candidates=candidate_results))
      candidate_scores = []
      for li, rule in enumerate(LAMBDAS):
        by_branch = {name: dict(
            inner_episode_selected_reward=score_utility[:, li, branch].tolist(),
            mean_selected_reward=float(score_utility[:, li, branch].mean(dtype=np.float64)),
            inner_episode_mse=score_mse[:, li, branch].tolist(),
            mean_mse=float(score_mse[:, li, branch].mean(dtype=np.float64)))
            for branch, name in enumerate(('q', 'p'))}
        candidate_scores.append({'lambda': rule,
            'pooled_utility': float(score_utility[:, li].mean(0, dtype=np.float64).mean(dtype=np.float64)),
            'pooled_mse': float(score_mse[:, li].mean(0, dtype=np.float64).mean(dtype=np.float64)),
            'interfaces': by_branch})
      common = _select_common_lambda(candidate_scores)
      chosen = common['selected_lambda']
      outer_fits, outer_results = {}, {}
      for branch, name in enumerate(('q', 'p')):
        kernel = _Kernel(*_rows(h, latent, targets, outer_training, outer, branch))
        decompositions += 1
        prediction, fit = kernel.predict(chosen)
        evaluation = _evaluate(prediction, rewards[outer], targets[outer], r)
        predictions[outer, :, :, branch] = np.asarray(evaluation['predicted_centered_reward'])
        choices[outer, :, :, branch] = np.asarray(evaluation['choices'], np.int32)
        selected[outer, :, :, branch] = np.asarray(evaluation['selected_reward'], np.float64)
        mses[outer, :, :, branch] = np.asarray(evaluation['sample_mse_per_control'], np.float64)
        outer_fits[name], outer_results[name] = fit, evaluation
        evaluations += 1
      constant_mean = rewards[outer_training].mean((0, 1), dtype=np.float64)
      constant = int(_argmax(constant_mean))
      constant_controls[outer] = constant
      constant_rewards[outer] = rewards[outer, :, constant]
      outer_folds.append(dict(heldout_episode=int(episodes[outer]),
          training_episodes=episodes[outer_training].astype(int).tolist(),
          training_rows=(e - 1) * s * r, heldout_rows=s * r,
          inner_folds=inner_folds, candidate_scores=candidate_scores,
          common_lambda_selection=common, selected_lambda=chosen,
          outer_fits=outer_fits, outer_interfaces=outer_results,
          outer_candidate_grid_evaluated=False,
          constant_control=constant, constant_train_mean_rewards=constant_mean.tolist()))
    state_rewards = selected.mean(2, dtype=np.float64)
    episode_rewards = state_rewards.mean(1, dtype=np.float64)
    constant_episode = constant_rewards.mean(1, dtype=np.float64)
    interfaces = {}
    for branch, name in enumerate(('q', 'p')):
      values = selected[..., branch]
      regret = rewards.max(-1)[..., None] - values
      interfaces[name] = dict(predicted_centered_reward=predictions[..., branch, :].tolist(),
          choices=choices[..., branch].tolist(), selected_reward=values.tolist(),
          finite_library_regret=regret.tolist(), state_selected_reward=state_rewards[..., branch].tolist(),
          episode_selected_reward=episode_rewards[..., branch].tolist(),
          primary_selected_reward=float(episode_rewards[..., branch].mean(dtype=np.float64)),
          state_finite_library_regret=regret.mean(2, dtype=np.float64).tolist(),
          episode_finite_library_regret=regret.mean((1, 2), dtype=np.float64).tolist(),
          primary_finite_library_regret=float(regret.mean(dtype=np.float64)),
          sample_mse_per_control=mses[..., branch].tolist(),
          primary_mse_per_control=float(mses[..., branch].mean(dtype=np.float64)))
    qp = episode_rewards[:, 0] - episode_rewards[:, 1]
    qc = episode_rewards[:, 0] - constant_episode
    draws = np.random.default_rng(STAT_SEED).integers(0, e, (BOOTSTRAP_DRAWS, e))
    comparisons = dict(q_minus_p=_interval(qp, draws), q_minus_constant=_interval(qc, draws))
    supported = all(value['lower95'] > 0 for value in comparisons.values())
    negative = any(value['upper95'] <= 0 for value in comparisons.values())
    status = ('selected_procedure_exploratory_supported' if supported else
              'selected_procedure_exploratory_not_supported' if negative else
              'selected_procedure_exploratory_inconclusive')
  after = {k: _digest(v) for k, v in dict(rewards=rewards, h=h, latent=latent, episodes=episodes).items()}
  _require(before == after, 'Nested readout modified its inputs')
  _require(decompositions == 2 * e * (e - 1) + 2 * e and
           evaluations == 2 * e * (e - 1) * len(LAMBDAS) + 2 * e,
           'Incomplete nested evaluation count')
  return dict(version=VERSION, implementation=IMPLEMENTATION,
      episodes=episodes.astype(int).tolist(), states_per_episode=s, samples_per_state=r,
      controls=a, h_dimensions=h.shape[-1], latent_shape=list(latent.shape[-2:]),
      lambda_grid=list(LAMBDAS), objective='mean_rows(sum_outputs(squared_error)) + lambda * ||W||_F^2',
      labels='within-state control centering; train-column centering with unregularized intercept',
      group_normalization='train-only total energy RMS for h and z; strict zero group',
      tie_tolerance=TIE, tie_rule='lowest control index within tolerance of maximum',
      numeric_absolute_tolerance=NUMERIC_TOL, eigen_cutoff_applied=False,
      outer_folds=outer_folds, interfaces=interfaces,
      constant=dict(control_by_episode=constant_controls.tolist(), state_selected_reward=constant_rewards.tolist(),
          episode_selected_reward=constant_episode.tolist(), primary_selected_reward=float(constant_episode.mean()),
          finite_library_regret=(rewards.max(-1) - constant_rewards).tolist()),
      comparisons=comparisons, selected_procedure_status=status,
      eigendecompositions=decompositions, lambda_evaluations=evaluations,
      selected_common_lambdas=[fold['selected_lambda'] for fold in outer_folds],
      q_p_selection_shared=True, equal_effective_degrees_of_freedom_claimed=False,
      bootstrap_draws=BOOTSTRAP_DRAWS, statistical_seed=STAT_SEED, percentile_method='linear',
      bootstrap_unit='whole calibration episodes; fixed nested cross-fitted scores',
      interval_scope='Nominal calibration description; overlapping training sets, no refitting uncertainty or independent confirmation',
      input_arrays_unchanged=True, input_array_digests=before,
      calibration_only=True, readout_fitted=True, real_gate_authority=False,
      eligible_method_training=False, training_authorized=False, proceed_short_training=False,
      method_effectiveness_established=False, dreamer_optimizer_updates=0,
      source_provenance_verified_by_stats=False, blind_arrays_read=False,
      model_calls=0, environment_calls=0,
      current_rgb_increment_over_complete_history_established=False,
      continuing_kl_compression_effect_established=False,
      negative_scope='Only the fixed selected nested procedure; not the entire information/function family',
      limitations=['Eight previously observed calibration episodes are not new blind confirmation.',
          'Common lambda does not ensure matched effective degrees of freedom across q and p.',
          'Readout differences need not imply information beyond complete history or causal RGB cues.',
          'No active KL compression, protection, gate discrimination, actor learning or model performance is evaluated.'])
