"""Frozen, pure NumPy calibration statistics for DL-action-information-r1.

The caller verifies source hashes, actual feature/action contracts, and physics
receipts. This module neither loads artifacts nor executes a model/environment.
Readout fits are analysis; they do not update Dreamer or authorize training.
"""

import hashlib

import numpy as np


VERSION = 'DL-action-information-r1'
IMPLEMENTATION = 'DL-action-information-stats-r1'
DELTA = .2
TIE = 1e-10
RIDGE = 1.
BOOTSTRAP_DRAWS = 2000
STAT_SEED = 20261021
POSITIONS = (1, 300, 600, 900)
CONTROL_NAMES = ('zero', 'all+', 'all-', 'lift-LR+', 'lift-LR-',
                 'extend-diag+', 'extend-diag-')


def _require(condition, message):
  if not condition:
    raise ValueError(message)


def _numeric(value, shape, name):
  _require(isinstance(value, np.ndarray) and value.dtype == np.float64 and
           value.shape == shape, f'{name}: require float64 shape {shape}')
  _require(np.isfinite(value).all(), f'{name}: nonfinite input')


def _digest(value):
  value = np.ascontiguousarray(value)
  return dict(dtype=value.dtype.str, shape=list(value.shape),
              sha256=hashlib.sha256(value.tobytes()).hexdigest())


def _argmax(values):
  """Lowest index within the frozen absolute tie margin of the maximum."""
  _require(np.isfinite(values).all(), 'Nonfinite ranking values')
  tied = values.max(axis=-1, keepdims=True) - values <= TIE
  return np.argmax(tied, axis=-1).astype(np.int32)


def _rankings(values):
  """Display ranking; tie handling never modifies the delta-optimal set."""
  rows = values.reshape((-1, values.shape[-1]))
  ranked = []
  for row in rows:
    remaining = list(range(len(row)))
    ordering = []
    while remaining:
      best = max(row[i] for i in remaining)
      winner = min(i for i in remaining if best - row[i] <= TIE)
      ordering.append(winner)
      remaining.remove(winner)
    ranked.append(ordering)
  return np.asarray(ranked, np.int32).reshape(values.shape)


def _standardize(train, test):
  mean = train.mean(0, dtype=np.float64)
  centered = train - mean
  rms = float(np.sqrt(np.mean(np.square(centered).sum(1), dtype=np.float64)))
  _require(np.isfinite(rms), 'Nonfinite feature-group RMS')
  zero = rms == 0.
  if zero:
    _require(np.all(centered == 0), 'Feature RMS underflowed for a nonzero group')
    # A training-constant group has no estimable slope. It contributes zero
    # for heldout samples too, including values outside that training constant.
    standardized_train = np.zeros_like(train)
    standardized_test = np.zeros_like(test)
  else:
    standardized_train = centered / rms
    standardized_test = (test - mean) / rms
  receipt = dict(mean=_digest(mean), rms=rms, zero_rms=zero,
                 dimensions=train.shape[1],
                 train_mean_total_squared_norm=float(np.mean(
                     np.square(standardized_train).sum(1), dtype=np.float64)),
                 zero_rms_rule='zero group for train and heldout; no rows removed')
  return standardized_train, standardized_test, receipt


def ridge_predict(h_train, z_train, targets, h_test, z_test):
  """Train-only group normalization and fixed mean-objective ridge.

  This helper accepts an arbitrary positive output dimension for small numerical
  fixtures. The production analyze() contract always has seven outputs.
  The unregularized intercept is the training column mean of targets.
  """
  _require(isinstance(h_train, np.ndarray) and h_train.ndim == 2 and
           h_train.shape[0] > 0 and h_train.shape[1] > 0,
           'Positive two-dimensional training h required')
  n, hd = h_train.shape
  _require(isinstance(z_train, np.ndarray) and z_train.ndim == 2 and
           z_train.shape[0] == n and z_train.shape[1] > 0,
           'Training latent rows/dimensions differ')
  zd = z_train.shape[1]
  _require(isinstance(targets, np.ndarray) and targets.ndim == 2 and
           targets.shape[0] == n and targets.shape[1] > 0,
           'Training target rows/dimensions differ')
  _require(isinstance(h_test, np.ndarray) and h_test.ndim == 2,
           'Two-dimensional heldout h required')
  nt = len(h_test)
  for value, shape, name in (
      (h_train, (n, hd), 'train h'), (z_train, (n, zd), 'train latent'),
      (targets, (n, targets.shape[1]), 'train targets'),
      (h_test, (nt, hd), 'heldout h'), (z_test, (nt, zd), 'heldout latent')):
    _numeric(value, shape, name)
  with np.errstate(over='raise', invalid='raise', divide='raise'):
    ht, he, h_receipt = _standardize(h_train, h_test)
    zt, ze, z_receipt = _standardize(z_train, z_test)
    x = np.concatenate((ht, zt), axis=1)
    xe = np.concatenate((he, ze), axis=1)
    intercept = targets.mean(0, dtype=np.float64)
    y = targets - intercept
    # Dual solution avoids a potentially 10k x 10k feature Gram matrix.
    # mean_i ||y_i - x_i W||^2 + lambda ||W||_F^2 uses n * lambda.
    gram = x @ x.T
    gram.flat[::n + 1] += n * RIDGE
    dual = np.linalg.solve(gram, y)
    weights = x.T @ dual
    prediction = xe @ weights + intercept
    residual = x @ weights - y
    data_objective = float(np.square(residual).sum(1).mean(dtype=np.float64))
    squared_weight_norm = float(np.square(weights).sum(dtype=np.float64))
    _require(np.isfinite(prediction).all() and np.isfinite(weights).all() and
             np.isfinite(data_objective) and np.isfinite(squared_weight_norm),
             'Nonfinite ridge solution')
  return prediction, dict(
      training_rows=n, feature_dimensions=hd + zd,
      output_dimensions=targets.shape[1], regularization_lambda=RIDGE,
      gram_diagonal_regularizer=n * RIDGE,
      objective='mean_rows(sum_outputs(squared_error)) + lambda * ||W||_F^2',
      solver='dual (X X.T + n*lambda*I)', intercept_regularized=False,
      train_column_mean=intercept.tolist(), h_group=h_receipt, z_group=z_receipt,
      weights=_digest(weights), dual_coefficients=_digest(dual),
      training_mean_sum_squared_error=data_objective,
      training_mse_per_control=data_objective / targets.shape[1],
      squared_weight_norm=squared_weight_norm,
      regularized_objective=data_objective + RIDGE * squared_weight_norm,
      zero_rms_groups=int(h_receipt['zero_rms']) + int(z_receipt['zero_rms']))


def _interval(episode_values, draws):
  primary = float(episode_values.mean(dtype=np.float64))
  bootstrap = episode_values[draws].mean(1, dtype=np.float64)
  lower, upper = np.quantile(bootstrap, [.025, .975], method='linear')
  _require(np.isfinite(bootstrap).all(), 'Nonfinite bootstrap reduction')
  return dict(primary=primary, lower95=float(lower), upper95=float(upper),
              episode_values=episode_values.tolist())


def _external(rewards, episodes):
  best = rewards.max(-1)
  regret = best[..., None] - rewards
  spans = np.ptp(rewards, axis=-1)
  # No tie tolerance is applied to either scientific boundary.
  near = regret <= DELTA
  sensitive = spans >= DELTA
  coverage = [int(ep) for i, ep in enumerate(episodes) if sensitive[i].any()]
  pairs = []
  flat_sets = near.reshape((32, 7))
  for a in range(32):
    ea, sa = divmod(a, 4)
    for b in range(a + 1, 32):
      eb, sb = divmod(b, 4)
      if ea != eb and not np.any(flat_sets[a] & flat_sets[b]):
        pairs.append(dict(episode_a=int(episodes[ea]), position_a=POSITIONS[sa],
                          episode_b=int(episodes[eb]), position_b=POSITIONS[sb]))
  supported = len(coverage) >= 4 and bool(pairs)
  return dict(
      delta=DELTA, delta_optimal_boundary='max_reward - reward <= delta',
      rewards=rewards.tolist(), finite_library_regret=regret.tolist(),
      range=spans.tolist(), rankings=_rankings(rewards).tolist(),
      display_winner=_argmax(rewards).tolist(), delta_optimal_membership=near.tolist(),
      sensitive=sensitive.tolist(), sensitive_states=int(sensitive.sum()),
      sensitive_episodes=coverage, sensitive_episode_count=len(coverage),
      minimum_sensitive_episodes=4, sensitivity_coverage_sufficient=len(coverage) >= 4,
      cross_episode_disjoint_pairs=pairs, disjoint_pair_count=len(pairs),
      cross_episode_disjoint_set_exists=bool(pairs), sufficient=supported,
      status='finite_control_selection_basis_supported' if supported else
          'finite_control_selection_basis_not_supported',
      interpretation='Conservative sufficient evidence within the fixed seven-command library; '
          'failure does not exclude useful state information or other action libraries')


def _readouts(rewards, h, latent, episodes):
  targets = rewards - rewards.mean(-1, keepdims=True, dtype=np.float64)
  hd, zd = h.shape[-1], int(np.prod(latent.shape[-2:]))
  predictions = np.empty((8, 4, 4, 2, 7), np.float64)
  choices = np.empty((8, 4, 4, 2), np.int32)
  selected = np.empty((8, 4, 4, 2), np.float64)
  errors = np.empty((8, 4, 4, 2), np.float64)
  constants = np.empty(8, np.int32)
  constant_rewards = np.empty((8, 4), np.float64)
  folds = []
  for heldout in range(8):
    train = np.arange(8) != heldout
    # Every state supplies exactly four rows: identical state total weights.
    ht = np.repeat(h[train, :, None, :], 4, axis=2).reshape((112, hd))
    he = np.repeat(h[heldout, :, None, :], 4, axis=1).reshape((16, hd))
    yt = np.repeat(targets[train, :, None, :], 4, axis=2).reshape((112, 7))
    constant_means = rewards[train].mean((0, 1), dtype=np.float64)
    constant = int(_argmax(constant_means))
    constants[heldout] = constant
    constant_rewards[heldout] = rewards[heldout, :, constant]
    fit = {}
    for branch, name in enumerate(('q', 'p')):
      zt = latent[train, :, :, branch].reshape((112, zd))
      ze = latent[heldout, :, :, branch].reshape((16, zd))
      prediction, metadata = ridge_predict(ht, zt, yt, he, ze)
      prediction = prediction.reshape((4, 4, 7))
      choice = _argmax(prediction)
      true = np.broadcast_to(rewards[heldout, :, None, :], (4, 4, 7))
      values = np.take_along_axis(true, choice[..., None], -1)[..., 0]
      predictions[heldout, :, :, branch] = prediction
      choices[heldout, :, :, branch] = choice
      selected[heldout, :, :, branch] = values
      errors[heldout, :, :, branch] = np.square(
          prediction - targets[heldout, :, None, :]).mean(-1, dtype=np.float64)
      fit[name] = metadata
    _require(fit['q']['feature_dimensions'] == fit['p']['feature_dimensions'],
             'q/p readout capacity differs')
    folds.append(dict(
        heldout_episode=int(episodes[heldout]),
        training_episodes=episodes[train].astype(int).tolist(), training_states=28,
        heldout_states=4, samples_per_state=4, training_rows=112, heldout_rows=16,
        state_training_weight=1 / 28, per_row_training_weight=1 / 112,
        constant_control=constant, constant_train_mean_rewards=constant_means.tolist(),
        fits=fit))
  state_rewards = selected.mean(2, dtype=np.float64)
  episode_rewards = state_rewards.mean(1, dtype=np.float64)
  constant_episode = constant_rewards.mean(1, dtype=np.float64)
  interfaces = {}
  for branch, name in enumerate(('q', 'p')):
    values = selected[..., branch]
    regrets = rewards.max(-1)[..., None] - values
    interfaces[name] = dict(
        predicted_centered_reward=predictions[..., branch, :].tolist(),
        choices=choices[..., branch].tolist(), selected_reward=values.tolist(),
        finite_library_regret=regrets.tolist(),
        state_selected_reward=state_rewards[..., branch].tolist(),
        episode_selected_reward=episode_rewards[..., branch].tolist(),
        primary_selected_reward=float(episode_rewards[..., branch].mean()),
        state_finite_library_regret=regrets.mean(2, dtype=np.float64).tolist(),
        episode_finite_library_regret=regrets.mean((1, 2), dtype=np.float64).tolist(),
        primary_finite_library_regret=float(regrets.mean(dtype=np.float64)),
        sample_mse_per_control=errors[..., branch].tolist(),
        primary_mse_per_control=float(errors[..., branch].mean(dtype=np.float64)))
  result = dict(
      interface_order=['q', 'p'], features='actual first sampled (h, stoch); no logits or mean probabilities',
      samples_per_state=4, independent_episode_units=8, states=32, rows=128,
      labels='seven rewards minus each state seven-control mean; train-column centered',
      prediction_units='centered cumulative reward with training column intercept restored',
      fold_design='leave entire episode, all states and RNG samples together',
      interfaces=interfaces, folds=folds,
      constant=dict(control_by_episode=constants.tolist(),
                    state_selected_reward=constant_rewards.tolist(),
                    episode_selected_reward=constant_episode.tolist(),
                    primary_selected_reward=float(constant_episode.mean()),
                    finite_library_regret=(rewards.max(-1) - constant_rewards).tolist()),
      total_zero_rms_groups=sum(f['fits'][k]['zero_rms_groups']
          for f in folds for k in ('q', 'p')))
  return result, episode_rewards[:, 0] - episode_rewards[:, 1], \
      episode_rewards[:, 0] - constant_episode


def analyze(fixed_rewards, actor_rewards, h, latent, episodes):
  """Analyze the frozen eight-episode, four-state, four-pair H20 object.

  Input arrays are never modified. Strict source identities, configured feature
  dimensions, shared-h proofs, and physical replay belong to the caller.
  """
  _numeric(fixed_rewards, (8, 4, 7), 'fixed rewards')
  _numeric(actor_rewards, (8, 4, 4, 2), 'actor rewards')
  _require(isinstance(h, np.ndarray) and h.ndim == 3 and h.shape[-1] > 0,
           'Positive h feature dimension required')
  _numeric(h, (8, 4, h.shape[-1]), 'h')
  _require(isinstance(latent, np.ndarray) and latent.ndim == 6 and
           latent.shape[-2] > 0 and latent.shape[-1] > 0,
           'Positive sampled latent factor/class dimensions required')
  _numeric(latent, (8, 4, 4, 2, *latent.shape[-2:]), 'latent')
  _require(isinstance(episodes, np.ndarray) and episodes.dtype == np.int32 and
           episodes.shape == (8,) and np.all(episodes >= 0) and
           len(np.unique(episodes)) == 8, 'Require eight unique nonnegative int32 episode IDs')
  try:
    with np.errstate(over='raise', invalid='raise', divide='raise'):
      external = _external(fixed_rewards, episodes)
      readout, qp, qc = _readouts(fixed_rewards, h, latent, episodes)
      draws = np.random.default_rng(STAT_SEED).integers(0, 8, (BOOTSTRAP_DRAWS, 8))
      comparisons = dict(q_minus_p=_interval(qp, draws), q_minus_constant=_interval(qc, draws))
      both_positive = all(x['lower95'] > 0 for x in comparisons.values())
      any_nonpositive = any(x['upper95'] <= 0 for x in comparisons.values())
      if both_positive and external['sufficient']:
        status = 'interface_increment_exploratory_supported'
      elif any_nonpositive:
        status = 'interface_increment_exploratory_not_supported'
      else:
        status = 'interface_increment_exploratory_inconclusive'
      readout.update(comparisons=comparisons, both_lower95_positive=both_positive,
                     any_upper95_nonpositive=any_nonpositive, status=status)
      signed = actor_rewards[..., 0] - actor_rewards[..., 1]
      actor_interval = _interval(signed.mean((1, 2), dtype=np.float64), draws)
      if actor_interval['lower95'] > 0:
        actor_status = 'actor_first_action_exploratory_supported'
      elif actor_interval['upper95'] <= 0:
        actor_status = 'actor_first_action_exploratory_not_supported'
      else:
        actor_status = 'actor_first_action_exploratory_inconclusive'
      actor = dict(
          actual_rewards=actor_rewards.tolist(), signed_q_minus_p=signed.tolist(),
          episode_interval=actor_interval,
          state_signed_mean=signed.mean(-1, dtype=np.float64).tolist(),
          within_state=dict(std=signed.std(-1, ddof=1).tolist(), range=np.ptp(signed, axis=-1).tolist()),
          per_rng_mean=signed.mean((0, 1), dtype=np.float64).tolist(),
          position_mean=signed.mean((0, 2), dtype=np.float64).tolist(),
          finite_library_reference_gap=(fixed_rewards.max(-1)[..., None, None] - actor_rewards).tolist(),
          reference_gap_is_global_regret=False, status=actor_status,
          exact_state_labels_established=False,
          current_actor_benefit_is_required_for_future_learning_route=False)
  except (FloatingPointError, np.linalg.LinAlgError) as exc:
    raise ValueError('Invalid numerical reduction in frozen statistics') from exc
  if not external['sufficient']:
    decision = 'stop_finite_control_object_external_basis_insufficient'
  elif status != 'interface_increment_exploratory_supported':
    decision = 'do_not_advance_gate_or_training'
  elif actor_status == 'actor_first_action_exploratory_supported':
    decision = 'may_design_independent_signal_and_causal_cue_validation'
  else:
    decision = 'may_design_finite_normal_loss_compression_mechanism'
  return dict(
      version=VERSION, implementation=IMPLEMENTATION, episodes=episodes.astype(int).tolist(),
      positions=list(POSITIONS), control_names=list(CONTROL_NAMES), horizon=20,
      tie_tolerance=TIE, tie_rule='lowest control index within tolerance of maximum',
      external_control=external, readout=readout, actor=actor, status=status, decision=decision,
      bootstrap_draws=BOOTSTRAP_DRAWS, statistical_seed=STAT_SEED,
      percentile_method='linear', bootstrap_unit='eight whole episodes; fixed cross-fitted scores',
      interval_scope='Nominal calibration description: overlapping LOEO fits, fixed four latent samples; '
          'no refitting uncertainty, independent blind confirmation, or training-seed generalization',
      calibration_only=True, readout_fitted=True,
      dreamer_optimizer_updates=0, dreamer_training_performed=False,
      model_calls=0, environment_calls=0, source_provenance_verified_by_stats=False,
      blind_arrays_read=False, training_authorized=False, proceed_short_training=False,
      eligible_method_training=False, method_effectiveness_established=False,
      current_rgb_increment_over_complete_history_established=False,
      continuing_kl_compression_effect_established=False,
      limitations=[
          'Fixed seven first commands and normalized-zero tail; not normal closed-loop or globally optimal control.',
          'Posterior-interface readout may exploit a nonlinear recoding of h; no causal RGB or complete-history increment proof.',
          'Four sampled latents are actor-input samples, not 128 independent states or a precise latent integral.',
          'Eight calibration episode bootstrap is nominal for overlapping cross-fitted scores, not independent confirmation.',
          'Negative readout results only concern this fixed linear family, ridge, data, and sample budget.',
          'No active-rep erosion, future actor learning, gate discrimination, OOD exclusion, or performance effect established.'])
