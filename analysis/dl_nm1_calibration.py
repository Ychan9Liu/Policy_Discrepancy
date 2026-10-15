"""DL-NM1 C producer arithmetic. No NN, GPU execution or audit authority.

All scalars below use the stored actual F32 forward/weight values. Controls
must be produced by the original JAX F32 implementation, never by this module.
An uncalibrated control is a complete negative scientific result, not a reason
to replace its coefficient or add calibration batches.
"""
import math

import numpy as np

VERSION = 'DL-NM1-calibration-r1'
SHAPE = (16, 64)
FLOAT_FIELDS = ('K', 'E', 'D', 'e', 'f', 'v_N')
FIELDS = {*FLOAT_FIELDS, 'm', 'A', 'stepid', 'chain'}


def require(condition, message):
  if not condition:
    raise ValueError(message)


def _signals(signals, expected_batches):
  require(type(expected_batches) is int and expected_batches > 0,
      'Positive explicit expected batch count required')
  require(type(signals) in (list, tuple) and len(signals) == expected_batches,
      'Complete fixed batch list required')
  for record in signals:
    require(type(record) is dict and set(record) == FIELDS, 'Exact raw signal fields required')
    for key in FLOAT_FIELDS:
      value = record[key]
      require(type(value) is np.ndarray and value.dtype == np.float32 and
          value.shape == SHAPE and np.isfinite(value).all(), 'Finite actual F32 field: '+key)
    for key in ('m', 'A'):
      value = record[key]
      require(type(value) is np.ndarray and value.dtype == np.bool_ and value.shape == SHAPE,
          'Actual boolean mask required: '+key)
    require(not np.any(record['A'] & ~record['m']), 'Intervention must be original H1 subset')
    require(np.all(record['K'] >= -1e-5) and np.all(record['D'] >= -1e-5),
        'Original KL validity tolerance exceeded')
    for key in ('e', 'f', 'v_N'):
      require(np.all((record[key] >= 0) & (record[key] <= 1)), 'Weight signal outside unit range: '+key)
    require(np.array_equal(record['E'], np.maximum(record['K']-np.float32(1), np.float32(0))),
        'E must be actual F32 max(K-1,0)')
    require(np.all(record['v_N'][~record['A']] == 1), 'N changes a position outside A')
    require(type(record['stepid']) is np.ndarray and record['stepid'].dtype == np.uint8 and
        record['stepid'].shape == (*SHAPE, 20), 'Full64 uint8 stepid required')
    require(type(record['chain']) is np.ndarray and record['chain'].dtype == np.int32 and
        record['chain'].shape == SHAPE and np.all((record['chain'] >= 0) & (record['chain'] < 16)),
        'Original inventory chain index required')


def _sum_products(signals, fields):
  def products():
    for record in signals:
      arrays = [record[key][record['A']].reshape(-1) for key in fields]
      for values in zip(*arrays):
        product = 1.
        for value in values:
          product *= float(value)
        yield product
  return math.fsum(products())


def _coefficient(numerator, denominator):
  value = numerator / denominator if denominator > 0 and math.isfinite(denominator) else None
  valid = value is not None and math.isfinite(value) and 0 <= value <= .5
  return dict(calibrated=bool(valid),
      reason=None if valid else ('zero_denominator' if denominator == 0 else 'invalid_coefficient'),
      coefficient_f64=float(value) if valid else None,
      coefficient_f32=float(np.float32(value)) if valid else None,
      denominator_f64=float(denominator))


def freeze(signals, expected_batches=128):
  """Freeze F64 analytic coefficients and the actual F32 P boundaries."""
  _signals(signals, expected_batches)
  numerator = .5 * _sum_products(signals, ('E', 'e', 'f'))
  denominator_s = _sum_products(signals, ('E', 'f'))
  denominator_w = _sum_products(signals, ('E', 'e'))
  active = [record['A'] & (record['K'] > 1) for record in signals]
  count = sum(int(mask.sum()) for mask in active)
  if count:
    d = np.concatenate([record['D'][mask] for record, mask in zip(signals, active)])
    k = np.concatenate([record['K'][mask] for record, mask in zip(signals, active)])
    dq = np.quantile(d.astype(np.float64), [1/3, 2/3], method='linear')
    kq = np.quantile(k.astype(np.float64), [.5], method='linear')
    bins = dict(calibrated=True, reason=None, eligible_visits=count,
        D_quantiles_f64=dq.tolist(), K_quantiles_f64=kq.tolist(),
        D_f32=np.unique(dq.astype(np.float32)).tolist(),
        K_f32=np.unique(kq.astype(np.float32)).tolist())
  else:
    bins = dict(calibrated=False, reason='no_active_positions', eligible_visits=0,
        D_quantiles_f64=[], K_quantiles_f64=[], D_f32=[], K_f32=[])
  return dict(version=VERSION, batches=expected_batches, positions=expected_batches*1024,
      aggregate64=dict(N_analytic_release=numerator,
          S_denominator=denominator_s, W_denominator=denominator_w),
      S=_coefficient(numerator, denominator_s),
      W=_coefficient(numerator, denominator_w), bins=bins)


def release(weights, excess):
  """Implemented release, F64 fsum of the actual stored F32 operands."""
  return math.fsum((1-float(v))*float(e) for v, e in
      zip(weights.reshape(-1), excess.reshape(-1)))


def summarize(signals, effects, controls, frozen, expected_batches=128):
  """Report fixed C qualifications; this does not authorize T or claim efficacy."""
  _signals(signals, expected_batches)
  require(frozen == freeze(signals, expected_batches), 'Frozen coefficient object changed')
  require(type(effects) in (list, tuple) and type(controls) in (list, tuple) and
      len(effects) == len(controls) == expected_batches, 'Complete effects and controls required')
  effect_keys = {'encoder_gradient_changed_elements', 'encoder_step_changed_elements',
      'posterior_gradient_changed_elements', 'posterior_step_changed_elements'}
  positions, chains, chain_by_stepid = set(), set(), {}
  effect_batches = 0
  release_terms = {name: [] for name in ('N', 'S', 'W', 'P')}
  protected_count = changed_count = moved_weight_visits = 0
  for record, effect, control in zip(signals, effects, controls):
    require(type(effect) is dict and set(effect) == effect_keys and
        all(type(value) is int and value >= 0 for value in effect.values()),
        'Four nonnegative actual numerical effect counts required')
    effect_batches += int(any(effect[prefix+'_gradient_changed_elements'] > 0 and
        effect[prefix+'_step_changed_elements'] > 0 for prefix in ('encoder', 'posterior')))
    eligible = record['A'] & (record['K'] > 1) & (record['v_N'] < 1)
    for value, chain in zip(record['stepid'][eligible], record['chain'][eligible]):
      key, chain = value.tobytes(), int(chain)
      require(key not in chain_by_stepid or chain_by_stepid[key] == chain,
          'One original stepid mapped to different source chains')
      chain_by_stepid[key] = chain
      positions.add(key)
      chains.add(chain)
    require(type(control) is dict and set(control) == {'v_S', 'v_W', 'v_P'},
        'Exact control weight schema required')
    release_terms['N'].extend((1-float(v))*float(e) for v, e in
        zip(record['v_N'].reshape(-1), record['E'].reshape(-1)))
    for name in ('S', 'W', 'P'):
      value = control['v_'+name]
      calibrated = frozen['bins' if name == 'P' else name]['calibrated']
      if not calibrated:
        require(value is None, 'Uncalibrated control must explicitly be None: '+name)
        continue
      require(type(value) is np.ndarray and value.dtype == np.float32 and value.shape == SHAPE and
          np.isfinite(value).all() and np.all((value >= 0) & (value <= 1)),
          'Actual finite GPU F32 control required: '+name)
      require(np.all(value[~record['A']] == 1), 'Control changes a position outside A: '+name)
      release_terms[name].extend((1-float(v))*float(e) for v, e in
          zip(value.reshape(-1), record['E'].reshape(-1)))
    protected = record['A'] & (record['K'] > 1) & (record['v_N'] < np.float32(1-1e-6))
    protected_count += int(protected.sum())
    if control['v_P'] is not None:
      changed = np.abs(control['v_P']-record['v_N']) > np.float32(1e-6)
      changed_count += int((protected & changed).sum())
      moved_weight_visits += int((record['A'] & changed).sum())
  totals = {name: math.fsum(terms) if len(terms) == expected_batches*1024 else None
      for name, terms in release_terms.items()}
  nrelease = totals['N']
  drifts = {name: abs(totals[name]-nrelease)/nrelease
      if totals[name] is not None and nrelease > 0 else None for name in ('S', 'W', 'P')}
  fraction = changed_count/protected_count if protected_count else None
  coverage = len(positions) >= 128 and len(chains) >= 8
  effect = effect_batches >= 16
  qualification = dict(C_N_coverage=coverage, C_N_effect=effect, C_N=coverage and effect,
      C_S=drifts['S'] is not None and drifts['S'] <= .01,
      C_W=drifts['W'] is not None and drifts['W'] <= .01,
      C_P=fraction is not None and fraction >= .5 and drifts['P'] is not None and drifts['P'] <= .10)
  return dict(version=VERSION, batches=expected_batches,
      distinct_protected_stepids=len(positions), distinct_protected_chains=len(chains),
      numerical_effect_batches=effect_batches, release_actual_f64=totals,
      relative_release_drift=drifts, P_protected_visits=protected_count,
      P_changed_protected_visits=changed_count, P_changed_protected_fraction=fraction,
      P_changed_intervention_visits=moved_weight_visits,
      qualification=qualification, T1_necessary_qualification=bool(qualification['C_N'] and qualification['C_S']),
      T2_control_qualification=bool(qualification['C_P'] and qualification['C_W']),
      subsequent_learning_authorized=False, method_efficacy_established=False)
