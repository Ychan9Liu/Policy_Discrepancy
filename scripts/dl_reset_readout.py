"""Pure NumPy diagnostics for the sealed reset0 0/B/R object.

No files, artifacts, model, simulator, rendering or GPU are accessed here.
The producer authenticates the 8 calibration / 24 held-out episodes, common
history/prior and physical return source before passing numeric arrays.
The shared seven-control readout is a diagnostic, not a deployed Dreamer head.
"""
import hashlib
import json

import numpy as np


VERSION = 'DL-reset-readout-r1'
LAMBDAS = (1., .1, .01, .001)
TIE = 1e-10
NUMERIC_TOL = 1e-10
STAT_SEED = 20261026
BOOTSTRAP_DRAWS = 2000
DELTA = .2
SIGNAL_NAMES = ('D-only', 'D+K', 'D+S', 'S-only')
CALIBRATION_EPISODES = (9, 14, 15, 18, 20, 24, 30, 31)


def _require(condition, message):
  if not condition:
    raise ValueError(message)


def _array(value, name, shape=None):
  _require(isinstance(value, np.ndarray) and value.dtype == np.float64 and
           not value.dtype.hasobject, f'{name}: float64 NumPy array required')
  _require(shape is None or value.shape == shape, f'{name}: shape mismatch')
  _require(np.isfinite(value).all(), f'{name}: nonfinite values')
  return value


def _digest(value):
  value = np.ascontiguousarray(value)
  _require(not value.dtype.hasobject, 'Object arrays cannot be hashed')
  return dict(dtype=value.dtype.str, shape=list(value.shape),
              sha256=hashlib.sha256(value.tobytes()).hexdigest())


def _validate_batch(batch, count, name):
  _require(isinstance(batch, dict) and set(batch) ==
           {'episode_ids', 'h', 'zq', 'zp', 'returns', 'D', 'K'},
           f'{name}: unexpected/missing fields')
  ids = batch['episode_ids']
  _require(isinstance(ids, np.ndarray) and ids.dtype.kind in 'iu' and
           ids.shape == (count,) and np.all(ids >= 0) and
           len(np.unique(ids)) == count, f'{name}: unique integer episodes required')
  h = batch['h']
  _require(isinstance(h, np.ndarray) and h.ndim == 2 and h.shape[1] > 0,
           f'{name}: positive history dimension required')
  _array(h, name + '.h', (count, h.shape[1]))
  for branch in ('zq', 'zp'):
    value = _array(batch[branch], name + '.' + branch, (count, 4, 32, 32))
    _require(np.all((value == 0.) | (value == 1.)) and
             np.all(value.sum(-1) == 1.), f'{name}.{branch}: exact one-hot required')
  _array(batch['returns'], name + '.returns', (count, 7))
  for key in ('D', 'K'):
    value = _array(batch[key], name + '.' + key, (count,))
    minimum = 0. if key == 'D' else -1e-5
    _require(np.all(value >= minimum), f'{name}.{key}: negative divergence outside guard')
  if count == 8:
    _require(set(ids.tolist()) == set(CALIBRATION_EPISODES),
             f'{name}: changed calibration episode set')
  return batch


def _center_targets(returns):
  with np.errstate(over='raise', invalid='raise'):
    # First remove a common anchor. This equivalent form preserves exact input
    # pairwise differences under additive offsets before the /7 mean rounding.
    differences = returns - returns[:, :1]
    result = differences - differences.mean(-1, keepdims=True, dtype=np.float64)
  _require(np.isfinite(result).all(), 'Nonfinite action-centered returns')
  return result


def _group_fit(values):
  mean = values.mean(0, dtype=np.float64)
  centered = values - mean
  rms = float(np.sqrt(np.square(centered).sum(1).mean(dtype=np.float64)))
  _require(np.isfinite(rms), 'Nonfinite training group RMS')
  if rms == 0.:
    _require(np.all(centered == 0.), 'Nonzero group RMS underflow')
    result = np.zeros_like(values)
  else:
    result = centered / rms
  _require(np.isfinite(result).all(), 'Nonfinite group normalization')
  return result, mean, rms


def _group_apply(values, mean, rms):
  if rms == 0.:
    return np.zeros_like(values)
  return (values - mean) / rms


def _pooled_rows(batch, indices):
  """Each episode has eight equally weighted rows: q4 then p4."""
  h = np.repeat(batch['h'][indices], 8, axis=0)
  z = np.concatenate((batch['zq'][indices], batch['zp'][indices]), axis=1)
  z = z.reshape((len(indices) * 8, 1024))
  targets = np.repeat(_center_targets(batch['returns'][indices]), 8, axis=0)
  return h, z, targets


def _fit_shared(batch, indices, lambda_value):
  _require(type(lambda_value) is float and lambda_value in LAMBDAS,
           'Changed shared-readout lambda')
  h, z, targets = _pooled_rows(batch, indices)
  with np.errstate(over='raise', invalid='raise', divide='raise'):
    ht, hm, hr = _group_fit(h)
    zt, zm, zr = _group_fit(z)
    x = np.concatenate((ht, zt), axis=1)
    n = len(x)
    intercept = targets.mean(0, dtype=np.float64)
    y = targets - intercept
    kernel = x @ x.T / n
    eigval, eigvec = np.linalg.eigh(kernel)
    _require(np.isfinite(eigval).all() and np.isfinite(eigvec).all() and
             eigval.min() >= -NUMERIC_TOL, 'Invalid shared-readout eigensystem')
    orth = float(np.max(np.abs(eigvec.T @ eigvec - np.eye(n))))
    reconstruction = float(np.max(np.abs(
        (eigvec * eigval) @ eigvec.T - kernel)))
    energy = float(np.square(x).sum(1).mean(dtype=np.float64))
    _require(max(orth, reconstruction, abs(float(eigval.sum()) - energy)) <=
             NUMERIC_TOL, 'Shared-readout eigen reconstruction failed')
    denom = eigval + lambda_value
    _require(np.all(denom > 0.), 'Invalid ridge denominator')
    dual = eigvec @ ((eigvec.T @ y) / denom[:, None])
    equation_error = float(np.max(np.abs(
        (kernel + lambda_value * np.eye(n)) @ dual - y)))
    _require(equation_error <= NUMERIC_TOL, 'Shared-readout ridge equation failed')
    weights = x.T @ dual / n
    residual = x @ weights + intercept - targets
    error = float(np.square(residual).sum(1).mean(dtype=np.float64))
    norm = float(np.square(weights).sum(dtype=np.float64))
  _require(np.isfinite(weights).all() and np.isfinite(intercept).all(),
           'Nonfinite shared-readout fit')
  return dict(h_mean=hm.tolist(), z_mean=zm.tolist(), h_rms=hr, z_rms=zr,
      weights=weights.tolist(), intercept=intercept.tolist(), lambda_value=lambda_value,
      metadata=dict(training_episodes=batch['episode_ids'][indices].tolist(),
          training_rows=n, rows_per_episode=8, q_p_weights=[.5, .5],
          parameters_shared=True, branch_id_input=False, action_centered_targets=True,
          target_centering='d=Y-Y[:,0:1]; centered=d-mean_controls(d)',
          mean_objective=True, intercept_regularized=False,
          h_mean=_digest(hm), z_mean=_digest(zm), targets=_digest(targets),
          h_zero_rms=hr == 0., z_zero_rms=zr == 0.,
          ridge_equation_max_abs_error=equation_error,
          eigenvalues=eigval.tolist(), all_eigenvalues_retained=True,
          eigen_orthogonality_error=orth, eigen_reconstruction_error=reconstruction,
          effective_df=float(np.sum(eigval / denom)),
          training_mse_per_control=error / 7., squared_weight_norm=norm,
          regularized_objective=error + lambda_value * norm))


def _predict_shared(model, batch):
  hm = np.asarray(model['h_mean'], np.float64)
  zm = np.asarray(model['z_mean'], np.float64)
  weights = np.asarray(model['weights'], np.float64)
  intercept = np.asarray(model['intercept'], np.float64)
  _require(hm.shape == (batch['h'].shape[1],) and zm.shape == (1024,) and
           weights.shape == (len(hm) + len(zm), 7) and intercept.shape == (7,),
           'Fitted/readout dimensions differ')
  _require(all(np.isfinite(v).all() for v in (hm, zm, weights, intercept)) and
           np.isfinite(model['h_rms']) and np.isfinite(model['z_rms']) and
           model['h_rms'] >= 0. and model['z_rms'] >= 0., 'Invalid fitted readout')
  h = np.repeat(batch['h'][:, None], 8, axis=1).reshape((-1, len(hm)))
  z = np.concatenate((batch['zq'], batch['zp']), axis=1).reshape((-1, 1024))
  with np.errstate(over='raise', invalid='raise', divide='raise'):
    x = np.concatenate((_group_apply(h, hm, model['h_rms']),
                        _group_apply(z, zm, model['z_rms'])), axis=1)
    all_predictions = (x @ weights + intercept).reshape((len(batch['h']), 2, 4, 7))
    # This is deliberately different from taking argmax of each MC row first.
    prediction = all_predictions.mean(axis=2, dtype=np.float64)
  _require(np.isfinite(all_predictions).all(), 'Nonfinite readout prediction')
  return prediction, all_predictions


def _decision(prediction, returns):
  _require(prediction.shape == (len(returns), 2, 7), 'Wrong prediction axes')
  # Exact ties use the first index. Near-tie sets are used only for lambda selection.
  choices = np.argmax(prediction, axis=-1).astype(np.int32)
  rows = np.arange(len(returns))
  aq, ap = choices[:, 0], choices[:, 1]
  reward_q, reward_p = returns[rows, aq], returns[rows, ap]
  signal = prediction[rows, 0, aq] - prediction[rows, 0, ap]
  utility = reward_q - reward_p
  _require(np.all(signal >= 0.) and np.isfinite(signal).all() and
           np.isfinite(utility).all(), 'Invalid decision signal/utility')
  return dict(choices=choices, S=signal, U=utility,
              reward_q=reward_q, reward_p=reward_p)


def _lambda_selection(candidates):
  _require(len(candidates) == 4 and {c['lambda'] for c in candidates} == set(LAMBDAS),
           'Changed lambda candidate set')
  _require(all(np.isfinite(c['pooled_utility']) and np.isfinite(c['pooled_mse']) and
               c['pooled_mse'] >= 0. for c in candidates),
           'Nonfinite/negative lambda candidate score')
  maximum = max(c['pooled_utility'] for c in candidates)
  u_set = [c for c in candidates if c['pooled_utility'] >= maximum - TIE]
  minimum = min(c['pooled_mse'] for c in u_set)
  m_set = [c for c in u_set if c['pooled_mse'] <= minimum + TIE]
  selected = max(c['lambda'] for c in m_set)
  return dict(selected_lambda=selected,
      utility_tie_candidates=sorted((c['lambda'] for c in u_set), reverse=True),
      mse_tie_candidates=sorted((c['lambda'] for c in m_set), reverse=True),
      rule='global pooled utility within 1e-10; then pooled MSE within 1e-10; largest lambda')


def _slots(name, D, K, S):
  zero = np.zeros_like(D)
  values = {'D-only': (D, zero), 'D+K': (D, K),
            'D+S': (D, S), 'S-only': (S, zero)}
  _require(name in values, 'Unknown signal family')
  return np.stack(values[name], axis=-1)


def _fit_regressor(slots, target):
  """Fixed two-slot, mean-objective lambda1 ridge; no model selection."""
  mean = slots.mean(0, dtype=np.float64)
  centered = slots - mean
  rms = np.sqrt(np.square(centered).mean(0, dtype=np.float64))
  _require(np.isfinite(rms).all(), 'Nonfinite signal-slot RMS')
  zero = rms == 0.
  _require(np.all(centered[:, zero] == 0.), 'Signal-slot RMS underflow')
  scaled = np.zeros_like(slots)
  scaled[:, ~zero] = centered[:, ~zero] / rms[~zero]
  intercept = float(target.mean(dtype=np.float64))
  gram = scaled.T @ scaled / len(slots)
  rhs = scaled.T @ (target - intercept) / len(slots)
  weights = np.linalg.solve(gram + np.eye(2), rhs)
  _require(np.isfinite(weights).all(), 'Nonfinite signal regressor')
  eigval = np.linalg.eigvalsh(gram)
  return dict(mean=mean.tolist(), rms=rms.tolist(), weights=weights.tolist(),
      intercept=intercept, lambda_value=1., zero_rms=zero.tolist(),
      nominal_slots=2, nominal_coefficients=2, intercept_regularized=False,
      rank_defaulttol=int(np.linalg.matrix_rank(scaled)),
      rank_rule='numpy.linalg.matrix_rank on standardized calibration 8x2 design; default tolerance',
      effective_df=float(np.sum(eigval / (eigval + 1.))),
      constant_target=bool(np.all(target == target[0])),
      target_minimum=float(target.min()), target_maximum=float(target.max()),
      equal_effective_capacity_claimed=False,
      training_predictions=(scaled @ weights + intercept).tolist())


def _predict_regressor(fit, slots):
  mean, rms, weights = (np.asarray(fit[k], np.float64)
                         for k in ('mean', 'rms', 'weights'))
  _require(mean.shape == rms.shape == weights.shape == (2,) and
           all(np.isfinite(v).all() for v in (mean, rms, weights)) and
           np.all(rms >= 0.) and np.isfinite(fit['intercept']) and
           fit['lambda_value'] == 1. and fit['nominal_slots'] == 2,
           'Invalid signal fit')
  scaled = np.zeros_like(slots)
  active = rms > 0.
  scaled[:, active] = (slots[:, active] - mean[active]) / rms[active]
  result = scaled @ weights + fit['intercept']
  _require(np.isfinite(result).all(), 'Nonfinite signal prediction')
  return result


def refit_calibration(cal, *, lambda_value):
  """Fit shared F at the frozen baseline lambda, using only all eight cal EPs."""
  _validate_batch(cal, 8, 'calibration')
  _require(type(lambda_value) is float and lambda_value in LAMBDAS,
           'Refit must use one fixed baseline lambda')
  model = _fit_shared(cal, np.arange(8), lambda_value)
  constant_values = cal['returns'].mean(0, dtype=np.float64)
  return dict(model=model, lambda_value=lambda_value,
      episode_ids=cal['episode_ids'].tolist(), returns_digest=_digest(cal['returns']),
      constant_values=constant_values.tolist(),
      constant_choice=int(np.argmax(constant_values)))


def fit_calibration(cal, *, delta=DELTA):
  """Freeze F0 lambda, F0, cal-OOF signal heads and high-D threshold before blind."""
  _require(type(delta) is float and delta == DELTA, 'Changed external delta')
  _validate_batch(cal, 8, 'calibration')
  oof = {rule: np.empty((8, 2, 7), np.float64) for rule in LAMBDAS}
  fold_metadata = []
  all_indices = np.arange(8)
  for heldout in range(8):
    train = all_indices[all_indices != heldout]
    test = {key: value[[heldout]] for key, value in cal.items()}
    fold = dict(heldout_episode=int(cal['episode_ids'][heldout]),
                training_episodes=cal['episode_ids'][train].tolist(), candidates=[])
    for rule in LAMBDAS:
      model = _fit_shared(cal, train, rule)
      prediction, samples = _predict_shared(model, test)
      oof[rule][heldout] = prediction[0]
      decision = _decision(prediction, test['returns'])
      mse = float(np.square(prediction - _center_targets(test['returns'])[:, None])
                  .mean(dtype=np.float64))
      fold['candidates'].append(dict(lambda_value=rule,
          mean_prediction=prediction[0].tolist(), choices=decision['choices'][0].tolist(),
          sample_predictions=samples[0].tolist(),
          pooled_utility=float((decision['reward_q'][0] + decision['reward_p'][0]) / 2.),
          pooled_mse=mse, fit_metadata=model['metadata']))
    fold_metadata.append(fold)
  centered = _center_targets(cal['returns'])
  candidates = []
  for rule in LAMBDAS:
    decision = _decision(oof[rule], cal['returns'])
    candidates.append(dict(**{'lambda': rule},
        pooled_utility=float((decision['reward_q'] + decision['reward_p']).mean() / 2.),
        pooled_mse=float(np.square(oof[rule] - centered[:, None]).mean()),
        oof_mean_prediction=oof[rule].tolist()))
  selection = _lambda_selection(candidates)
  rule = selection['selected_lambda']
  decision = _decision(oof[rule], cal['returns'])
  regressors = {name: _fit_regressor(_slots(name, cal['D'], cal['K'], decision['S']),
                                    decision['U']) for name in SIGNAL_NAMES}
  return dict(version=VERSION, delta=delta, lambda_grid=list(LAMBDAS),
      selected_lambda=rule, selection=selection, candidates=candidates,
      fold_metadata=fold_metadata, fit=refit_calibration(cal, lambda_value=rule),
      oof=dict(episode_ids=cal['episode_ids'].tolist(),
          prediction=oof[rule].tolist(), choices=decision['choices'].tolist(),
          S=decision['S'].tolist(), U=decision['U'].tolist(),
          D=cal['D'].tolist(), K=cal['K'].tolist(),
          global_cal_lambda_selection=True, fully_nested_crossfit_claimed=False),
      regressors=regressors, high_D_threshold=float(np.median(cal['D'])),
      signal_primary_model='0', blind_seen=False,
      role='diagnostic shared readout; no deployment/continuous-actor claim')


def _validate_frozen(frozen):
  _require(isinstance(frozen, dict) and frozen.get('version') == VERSION and
           frozen.get('blind_seen') is False and frozen.get('delta') == DELTA and
           frozen.get('lambda_grid') == list(LAMBDAS), 'Invalid frozen calibration')
  rule = frozen.get('selected_lambda')
  _require(type(rule) is float and rule in LAMBDAS and
           frozen['fit']['lambda_value'] == rule and
           set(frozen['regressors']) == set(SIGNAL_NAMES) and
           np.isfinite(frozen['high_D_threshold']), 'Malformed frozen procedure')
  _require(set(frozen['fit']['episode_ids']) == set(CALIBRATION_EPISODES),
           'Changed frozen calibration episodes')
  # Reject JSON encodings containing nonfinite numbers anywhere, including metadata.
  json.dumps(frozen, allow_nan=False)


def dumps_frozen(frozen):
  """Deterministic JSON text for producer hashing; no filesystem side effects."""
  _validate_frozen(frozen)
  return json.dumps(frozen, sort_keys=True, separators=(',', ':'), allow_nan=False)


def loads_frozen(payload):
  """Load newly produced trusted JSON text, never a source artifact/pickle."""
  _require(isinstance(payload, str), 'Frozen JSON text required')
  frozen = json.loads(payload)
  _validate_frozen(frozen)
  return frozen


def _summary(values, indices):
  _require(values.ndim == 1 and values.shape == (24,) and np.isfinite(values).all(),
           'Twenty-four finite episode values required')
  draws = values[indices].mean(axis=1, dtype=np.float64)
  ci = np.percentile(draws, [2.5, 97.5], method='linear')
  return dict(values=values.tolist(), mean=float(values.mean(dtype=np.float64)),
      ci95=ci.tolist(), bootstrap_unit='whole episode', bootstrap_draws=len(indices),
      interpretation='descriptive held-out interval; not independent training replicates')


def _coverage(decision, D, threshold, delta):
  utility = decision['U']
  positive, nonpositive = utility >= delta, utility <= 0.
  disagree = decision['choices'][:, 0] != decision['choices'][:, 1]
  wrong = disagree & (utility < 0.)
  high = D >= threshold
  def counts(mask):
    pos = int(np.sum(mask & positive))
    nonpos = int(np.sum(mask & nonpositive))
    bad = int(np.sum(mask & wrong))
    return dict(total=int(mask.sum()), practical_positive=pos, nonpositive=nonpos,
        wrong_choice_negative=bad,
        identity_nonpositive=int(np.sum(mask & nonpositive & ~disagree)),
        minimum_coverage_met=bool(pos >= 4 and nonpos >= 4 and bad >= 2))
  return dict(all=counts(np.ones(24, bool)), high_D=counts(high),
      high_D_threshold=threshold, high_D_rule='D >= calibration median; ties included',
      positive_rule='U >= .2', nonpositive_rule='U <= 0',
      wrong_rule='aq != ap and U < 0',
      requirements='at least 4 positive / 4 nonpositive / 2 wrong-choice negative EPs',
      power_or_population_coverage_claimed=False)


def _evaluate_fit(fit, batch, indices):
  prediction, samples = _predict_shared(fit['model'], batch)
  decision = _decision(prediction, batch['returns'])
  rows = np.arange(24)
  constant_choice = fit['constant_choice']
  constant_reward = batch['returns'][rows, constant_choice]
  centered = _center_targets(batch['returns'])
  mse_q = np.square(prediction[:, 0] - centered).mean(axis=1, dtype=np.float64)
  mse_p = np.square(prediction[:, 1] - centered).mean(axis=1, dtype=np.float64)
  metrics = dict(q_absolute=_summary(decision['reward_q'], indices),
      p_absolute=_summary(decision['reward_p'], indices),
      constant_absolute=_summary(constant_reward, indices),
      q_minus_p=_summary(decision['U'], indices),
      q_minus_constant=_summary(decision['reward_q'] - constant_reward, indices),
      q_centered_mse=_summary(mse_q, indices), p_centered_mse=_summary(mse_p, indices),
      q_regret=_summary(batch['returns'].max(-1) - decision['reward_q'], indices),
      p_regret=_summary(batch['returns'].max(-1) - decision['reward_p'], indices))
  opportunity = bool(metrics['q_minus_p']['ci95'][0] > 0. and
                     metrics['q_minus_constant']['ci95'][0] > 0. and
                     metrics['q_minus_p']['mean'] > 0. and
                     metrics['q_minus_constant']['mean'] > 0.)
  q_failures = int(np.sum(np.asarray(metrics['q_regret']['values']) > DELTA))
  p_failures = int(np.sum(np.asarray(metrics['p_regret']['values']) > DELTA))
  return dict(episode_ids=batch['episode_ids'].tolist(),
      predictions=prediction.tolist(), sample_predictions=samples.tolist(),
      choices=decision['choices'].tolist(), S_decision=decision['S'].tolist(),
      true_U=decision['U'].tolist(), constant_choice=constant_choice,
      closed_delta=dict(delta=DELTA, failure_rule='regret > delta; closed near-optimal set',
          q_failure_count=q_failures, p_failure_count=p_failures,
          q_failure_rate=q_failures / 24., p_failure_rate=p_failures / 24.),
      metrics=metrics, finite_readout_opportunity_supported=opportunity,
      interpretation='limited shared seven-control readout, not actor/model performance'), decision


def evaluate_joint(frozen, refits, blind, *, seed=STAT_SEED,
                   bootstrap_count=BOOTSTRAP_DRAWS, delta=DELTA):
  """Evaluate all three previously frozen fits together on the 24 held-out EPs.

  The caller enforces the once-only blind opening and records the pre-opening
  frozen JSON hash. This pure function has no persistent execution state.
  """
  _validate_frozen(frozen)
  _require(type(seed) is int and seed == STAT_SEED and
           type(bootstrap_count) is int and bootstrap_count == BOOTSTRAP_DRAWS and
           type(delta) is float and delta == DELTA, 'Changed statistics protocol')
  _require(isinstance(refits, dict) and set(refits) == {'B', 'R'} and
           isinstance(blind, dict) and set(blind) == {'0', 'B', 'R'},
           'Joint 0/B/R frozen fits and held-out batches required')
  fits = {'0': frozen['fit'], **refits}
  for name in ('0', 'B', 'R'):
    _validate_batch(blind[name], 24, 'blind.' + name)
    _require(fits[name]['lambda_value'] == frozen['selected_lambda'] and
             fits[name]['model']['lambda_value'] == frozen['selected_lambda'] and
             fits[name]['model']['metadata']['training_episodes'] ==
                 fits[name]['episode_ids'] and
             fits[name]['model']['metadata']['training_rows'] == 64 and
             fits[name]['model']['metadata']['parameters_shared'] is True and
             fits[name]['model']['metadata']['branch_id_input'] is False and
             fits[name]['episode_ids'] == fits['0']['episode_ids'] and
             fits[name]['returns_digest'] == fits['0']['returns_digest'] and
             fits[name]['constant_choice'] == fits['0']['constant_choice'] and
             fits[name]['constant_values'] == fits['0']['constant_values'],
             'Changed calibration labels/order/lambda/constant across models')
    _require(np.array_equal(blind[name]['episode_ids'], blind['0']['episode_ids']) and
             np.array_equal(blind[name]['returns'], blind['0']['returns']),
             'Unpaired held-out episodes/physical returns across models')
  _require(set(blind['0']['episode_ids'].tolist()).isdisjoint(fits['0']['episode_ids']),
           'Calibration/held-out episode overlap')
  _require(set(blind['0']['episode_ids'].tolist()) | set(fits['0']['episode_ids']) ==
           set(range(32)), 'Changed complete production episode set')
  indices = np.random.Generator(np.random.PCG64(seed)).integers(
      0, 24, size=(bootstrap_count, 24))
  models, decisions = {}, {}
  for name in ('0', 'B', 'R'):
    models[name], decisions[name] = _evaluate_fit(fits[name], blind[name], indices)
  batch, decision = blind['0'], decisions['0']
  coverage = _coverage(decision, batch['D'], frozen['high_D_threshold'], delta)
  target = decision['U']
  signal_predictions, signal_errors, signal_metrics = {}, {}, {}
  for name in SIGNAL_NAMES:
    prediction = _predict_regressor(frozen['regressors'][name],
        _slots(name, batch['D'], batch['K'], decision['S']))
    error = np.square(prediction - target)
    signal_predictions[name], signal_errors[name] = prediction, error
    signal_metrics[name] = dict(predicted_U=prediction.tolist(),
        mse=_summary(error, indices),
        fit_constant_target=frozen['regressors'][name]['constant_target'])
  increments = {}
  for name in ('D+K', 'D+S', 'S-only'):
    difference = signal_errors['D-only'] - signal_errors[name]
    summary = _summary(difference, indices)
    summary['positive_increment_descriptive'] = bool(summary['mean'] > 0. and
                                                    summary['ci95'][0] > 0.)
    summary['high_D_mean_increment'] = (float(difference[
        batch['D'] >= frozen['high_D_threshold']].mean()) if
        np.any(batch['D'] >= frozen['high_D_threshold']) else None)
    # Coverage permits interpreting the all-episode increment; the high-D
    # subset mean alone does not establish a subset-specific improvement.
    summary['eligible_covered_overall_increment'] = bool(
        coverage['high_D']['minimum_coverage_met'] and
        not frozen['regressors'][name]['constant_target'] and
        summary['positive_increment_descriptive'])
    increments[name] = summary
  false_positive = (decision['S'] >= delta) & (target <= 0.)
  false_negative = (decision['S'] < delta) & (target >= delta)
  mechanism = {}
  for metric in ('q_absolute', 'p_absolute', 'q_minus_p', 'q_centered_mse',
                 'p_centered_mse'):
    values = (np.asarray(models['R']['metrics'][metric]['values'], np.float64) -
              np.asarray(models['B']['metrics'][metric]['values'], np.float64))
    mechanism['R_minus_B_' + metric] = _summary(values, indices)
  return dict(version=VERSION, models=models,
      statistics=dict(seed=seed, bootstrap_draws=bootstrap_count,
          bootstrap_indices_digest=_digest(indices), episode_count=24,
          paired_indices_across_all_comparisons=True, fixed_fit_uncertainty_only=True,
          refit_or_training_uncertainty_included=False, descriptive=True),
      discrimination=dict(primary_model='0', target='true signed U=Y(aq)-Y(ap)',
          blind_target_constant=bool(np.all(target == target[0])),
          blind_target_minimum=float(target.min()), blind_target_maximum=float(target.max()),
          blind_S_constant=bool(np.all(decision['S'] == decision['S'][0])),
          coverage=coverage, regressors=signal_metrics, increments_over_D_only=increments,
          S_threshold=delta, false_positive_episodes=batch['episode_ids'][false_positive].tolist(),
          false_negative_episodes=batch['episode_ids'][false_negative].tolist(),
          identical_choice_episodes=batch['episode_ids'][
              decision['choices'][:, 0] == decision['choices'][:, 1]].tolist(),
          conditional_independence_or_universal_identification_claimed=False,
          equal_two_slot_architecture_not_equal_effective_df=True,
          nuisance_wrong_observation_and_continuous_actor_not_tested=True),
      mechanism=mechanism,
      role='finite reset diagnostic; no gate/deployment/formal-performance conclusion',
      baseline_lambda_reused_for_B_R=frozen['selected_lambda'],
      input_digests={name: {key: _digest(value) for key, value in blind[name].items()}
                     for name in ('0', 'B', 'R')})
