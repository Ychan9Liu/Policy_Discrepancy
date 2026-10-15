"""DL normal-task distribution readout, pure finite NumPy analysis.

Fixed ridge/group-RMS definitions only. No artifact authority, model, simulator,
GPU, learning intervention, hyperparameter search or method-support verdict.
Probability readouts are richer than the actor's single sampled latent.
"""
import numpy as np

VERSION = 'DL-normal-task-readout-r1'
RIDGE = .1
BOOTSTRAPS, STAT_SEED = 10000, 20261040
NUMERIC_TOL = 1e-10


def require(value, message):
  if not value:
    raise ValueError(message)


def array(value, name, ndim):
  require(type(value) is np.ndarray and value.dtype == np.float64 and
      value.ndim == ndim and all(value.shape) and np.isfinite(value).all(),
      'Finite nonempty float64 '+name+' required')
  return value


def belief_features(probabilities):
  """Caller must authenticate these as actual official unimix probabilities.

  Geometry is generic for small tests; this function authenticates no source
  or N1M shape. Production input validation must separately bind32x32.
  """
  value = array(probabilities, 'belief probabilities', 3)
  require(value.shape[-1] >= 2 and np.all((0 <= value) & (value <= 1)) and
      np.all(np.abs(value.sum(-1)-1) <= 1e-6), 'Invalid categorical probabilities')
  return value.reshape((len(value), -1)).copy()


def _fit_group(value):
  mean = value.mean(0, dtype=np.float64)
  centered = value-mean
  rms = float(np.sqrt(np.square(centered).sum(1).mean(dtype=np.float64)))
  require(np.isfinite(rms), 'Nonfinite fitting group RMS')
  if rms == 0:
    require(not np.any(centered), 'Nonzero group RMS underflow')
    normalized = np.zeros_like(value)
  else:
    normalized = centered/rms
  return normalized, dict(mean=mean, rms=rms, dimension=value.shape[1])


def _apply_group(value, group):
  require(value.shape[1] == group['dimension'], 'Feature dimension changed')
  return (value-group['mean'])/group['rms'] if group['rms'] else np.zeros_like(value)


def fit(h, belief, targets):
  """Fit mean_n ||y-Xb-a||^2 + .1||b||^2; intercept unpenalized.

  h and optional belief are independently centered/group-RMS normalized using
  fitting rows only. Each nonconstant group contributes kernel trace1; adding
  more coordinates cannot inflate that group's total energy. Different
  models/q/p are fitted separately; coefficients are never cross-transplanted.
  """
  h, targets = array(h, 'history', 2), array(targets, 'targets', 2)
  require(len(h) == len(targets) and len(h) >= 2, 'Fitting row count differs/too small')
  if belief is not None:
    array(belief, 'belief features', 2)
    require(len(belief) == len(h), 'Belief fitting row count differs')
  with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
    xh, gh = _fit_group(h)
    if belief is None:
      x, gz = xh, None
    else:
      xz, gz = _fit_group(belief)
      x = np.concatenate((xh, xz), axis=1)
    n = len(x)
    intercept = targets.mean(0, dtype=np.float64)
    yc = targets-intercept
    kernel = x @ x.T/n
    eigenvalues = np.linalg.eigvalsh(kernel)
    require(eigenvalues.min() >= -NUMERIC_TOL, 'Non-PSD normalized feature kernel')
    dual = np.linalg.solve(kernel+RIDGE*np.eye(n), yc)
    equation_error = float(np.max(np.abs((kernel+RIDGE*np.eye(n)) @ dual-yc)))
    require(equation_error <= NUMERIC_TOL, 'Ridge equation residual too large')
    weights = x.T @ dual/n
    spectrum = np.maximum(eigenvalues, 0)
    trace = float(np.trace(kernel))
    groups = int(gh['rms'] > 0)+int(gz is not None and gz['rms'] > 0)
    require(abs(trace-groups) <= NUMERIC_TOL, 'Group normalization trace differs')
    df = float(np.sum(spectrum/(spectrum+RIDGE)))
    require(df <= groups/RIDGE+NUMERIC_TOL, 'Normalized effective degrees-of-freedom bound failed')
    rank_threshold = max(x.shape)*np.finfo(np.float64).eps*float(spectrum.max())
    normalized_train_risk = np.square(x @ weights+intercept-targets).mean(0)
  require(np.isfinite(weights).all() and np.isfinite(intercept).all(), 'Nonfinite fitted weights')
  return dict(version=VERSION, ridge=RIDGE, fitting_rows=n, history=gh, belief=gz,
      weights=weights, intercept=intercept, kernel_eigenvalues=eigenvalues,
      kernel_trace=trace, effective_df=df,
      numerical_kernel_rank=int(np.sum(spectrum > rank_threshold)),
      kernel_rank_threshold=rank_threshold, equation_error=equation_error,
      fitting_risk_original_units=normalized_train_risk,
      claim='Fixed linear distribution accessibility; no total-information or actor-use claim')


def predict(model, h, belief):
  require(model['version'] == VERSION and model['ridge'] == RIDGE, 'Changed readout definition')
  h = array(h, 'evaluation history', 2)
  with np.errstate(over='raise', invalid='raise', divide='raise'):
    xh = _apply_group(h, model['history'])
    if model['belief'] is None:
      require(belief is None, 'History-only model cannot accept belief features')
      x = xh
    else:
      array(belief, 'evaluation belief', 2)
      require(len(belief) == len(h), 'Evaluation belief row count differs')
      x = np.concatenate((xh, _apply_group(belief, model['belief'])), axis=1)
    result = x @ model['weights']+model['intercept']
  require(np.isfinite(result).all(), 'Nonfinite evaluation prediction')
  return result


def family_risks(predictions, targets, families):
  """Average states within family first; frames are never independent units."""
  predictions, targets = array(predictions, 'predictions', 2), array(targets, 'targets', 2)
  require(predictions.shape == targets.shape and type(families) is np.ndarray and
      families.dtype.kind in 'iu' and families.shape == (len(targets),) and
      np.all(families >= 0), 'Aligned predictions/targets/family IDs required')
  ids = np.unique(families)
  with np.errstate(over='raise', invalid='raise'):
    error = np.square(predictions-targets)
    risks = np.stack([error[families == value].mean(0, dtype=np.float64) for value in ids])
  require(np.isfinite(risks).all(), 'Nonfinite physical-unit family risk')
  return dict(families=ids, counts=np.array([(families == value).sum() for value in ids], np.int64),
      risks=risks, mean=risks.mean(0, dtype=np.float64))


def improvement(reference, alternative):
  """Pointwise paired family bootstrap; positive favors the alternative.

  The same fixed resamples are used for all contrasts with aligned families.
  No multiplicity correction, scientific decision, training-seed replication or
  unique mediation is inferred here. Input must be authenticated held families.
  """
  require(np.array_equal(reference['families'], alternative['families']) and
      np.array_equal(reference['counts'], alternative['counts']), 'Paired families/counts differ')
  left, right = array(reference['risks'], 'reference family risks', 2), array(
      alternative['risks'], 'alternative family risks', 2)
  require(left.shape == right.shape and len(left) >= 2, 'Paired risk shape/family count differs')
  delta = left-right
  require(np.isfinite(delta).all(), 'Nonfinite paired risk difference')
  draws = np.random.default_rng(STAT_SEED).integers(0, len(delta), (BOOTSTRAPS, len(delta)))
  values = delta[draws].mean(1, dtype=np.float64)
  return dict(mean=delta.mean(0, dtype=np.float64),
      pointwise_ci95=np.quantile(values, [.025, .975], axis=0, method='linear'),
      family_differences=delta, families=len(delta), draws=BOOTSTRAPS, seed=STAT_SEED,
      positive_means='reference risk minus alternative risk; lower alternative error',
      inferential_scope='Fixed family sampling / fitted readout / one training object only')
