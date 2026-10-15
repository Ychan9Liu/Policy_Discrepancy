"""DL live RCT: fixed CPU statistics; raw chronology/randomization need external audit.

Ridge arithmetic is explicitly shared with R4, whose protocol is unchanged.
No per-state utility truth, protection effectiveness or training permission follows.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))
from analysis.dl_r4_statistics import (authenticate_frozen, decode, finite,
    fit_predictor, keys, predict, require)

VERSION = 'DL-live-utility-r1'
HORIZON, DELTA, DRAWS, BOOTSTRAP_SEED = 100, .01, 10000, 20261030
SHAM_SEED, SHAM_ALPHA = 20261031, .05
LABELS = ('qqA', 'qqB', 'pq', 'pp')
POSITIONS = tuple(range(64, 897, 64))
FEATURES = {'D': ('D', 'time', 'r_t'), 'DK': ('D', 'K', 'time', 'r_t'),
    'DKe': ('D', 'K', 'e', 'time', 'r_t'), 'e': ('e', 'time', 'r_t')}
BASIS = ('D', 'K', 'D_squared', 'D_times_K', 'K_squared', 'time', 'r_t')
COMPARISONS = {'DKe_vs_DK': ('DK', 'DKe'), 'DK_vs_D': ('D', 'DK'),
    'DKe_vs_D': ('D', 'DKe'), 'e_vs_D': ('D', 'e')}
PROVENANCE = dict(e_source='cpu_shadow_baseline_q_before_assignment',
    sham_interface='identical_qq', randomization='iid_uniform4_independent_pretreatment')


def assignment_plan(index):
  """Frozen independent namespaces; actual reveal must follow signal sealing."""
  j = int(np.random.default_rng([BOOTSTRAP_SEED, 0x54494D45, index]).integers(0, 14))
  arm = int(np.random.default_rng([BOOTSTRAP_SEED, 0x41524D53, index]).integers(0, 4))
  return POSITIONS[j], LABELS[arm]


def pseudo_outcomes(label, y, baseline):
  z = y-baseline
  qq = label in ('qqA', 'qqB')
  return dict(U_obs=(2*qq-4*(label == 'pp'))*z,
      U_act=(2*qq-4*(label == 'pq'))*z,
      U_memory=(4*(label == 'pq')-4*(label == 'pp'))*z,
      sham=(4*(label == 'qqA')-4*(label == 'qqB'))*z)


def validate_units(units, families=32):
  require(type(units) is list and len(units) == 16*families,
      f'Exactly {families} complete families x16 units required')
  rows, previous = [], -1
  for index, unit in enumerate(units):
    keys(unit, ('family_id', 'unit_id', 'env_seed', 'policy_seed', 'position',
        'label', 'pretreatment', 'outcome', 'chronology'), 'unit')
    expected = dict(family_id=index//16, unit_id=index % 16,
        env_seed=8192+index//16, policy_seed=0x52350000+index)
    for name, value in expected.items():
      require(type(unit[name]) is int and unit[name] == value, f'{name} mapping mismatch')
    position, label = assignment_plan(index)
    require(type(unit['position']) is int and unit['position'] == position and
        unit['label'] == label, 'Frozen time/label namespace plan mismatch')
    pre, outcome, chronology = (unit[k] for k in ('pretreatment', 'outcome', 'chronology'))
    keys(pre, ('D', 'K', 'e', 'r_t', 'r_t_f32'), 'pretreatment')
    signals = {k: finite(pre[k], 0, 1 if k in ('e', 'r_t', 'r_t_f32') else None)
        for k in pre}
    require(signals['r_t_f32'].hex() == float(np.float32(signals['r_t'])).hex(),
        'Stored F32 reward encoding must match pre-treatment F64 suite reward')
    keys(outcome, ('reward_sum',), 'outcome')
    y = finite(outcome['reward_sum'], 0, HORIZON)/HORIZON
    keys(chronology, ('signals_sealed', 'assignment', 'outcome_complete'), 'chronology')
    events = [chronology[k] for k in ('signals_sealed', 'assignment', 'outcome_complete')]
    require(all(type(v) is int and v >= 0 for v in events) and
        previous < events[0] < events[1] < events[2],
        'Strict serial pre-treatment sealing/assignment/outcome chronology required')
    previous = events[2]
    signals['time'] = position/1000
    rows.append(dict(family_id=index//16, unit_id=index % 16, label=label,
        signals=signals, Y=y, **pseudo_outcomes(label, y, signals['r_t'])))
  return rows


def quadratic_basis(rows):
  x = np.array([[r['signals']['D'], r['signals']['K'], r['signals']['D']**2,
      r['signals']['D']*r['signals']['K'], r['signals']['K']**2,
      r['signals']['time'], r['signals']['r_t']] for r in rows], np.float64)
  require(np.isfinite(x).all(), 'Nonfinite fixed quadratic basis')
  return x


def projected_residual(rows, projection):
  x = quadratic_basis(rows)
  z = (x-np.array(projection['center']))/np.array(projection['scale'])
  z[:, np.array(projection['constant'], bool)] = 0
  design = np.column_stack((np.ones(len(rows)), z))
  residual = np.array([r['signals']['e'] for r in rows])-design @ projection['coefficient']
  require(np.isfinite(residual).all(), 'Nonfinite calibration-frozen projected residual')
  return residual


def freeze_basis_control(rows):
  """Only cal covariates choose the projection; no Y, held data or tuning."""
  x = quadratic_basis(rows)
  center, scale = x.mean(0), x.std(0)
  require(np.isfinite(center).all() and np.isfinite(scale).all(), 'Nonfinite basis standardization')
  constant = scale == 0
  scale = np.where(constant, 1., scale)
  z = (x-center)/scale
  z[:, constant] = 0
  coefficient, _, rank, singular = np.linalg.lstsq(np.column_stack((np.ones(len(rows)), z)),
      [r['signals']['e'] for r in rows], rcond=None)
  projection = dict(center=center.tolist(), scale=scale.tolist(), constant=constant.tolist(),
      coefficient=coefficient.tolist(), rank=int(rank), singular_values=singular.tolist(),
      solver='numpy.linalg.lstsq SVD, rcond=None; unpenalized intercept and slopes',
      outcome_used=False, residual_disable_threshold=1e-10)
  residual = projected_residual(rows, projection)
  projection['cal_max_abs_residual'] = float(np.max(np.abs(residual)))
  projection['residual_disabled'] = projection['cal_max_abs_residual'] <= 1e-10
  ids, y = [r['family_id'] for r in rows], [r['U_obs'] for r in rows]
  base = fit_predictor(x, y, ids)
  base['features'] = list(BASIS)
  extended = base
  if not projection['residual_disabled']:
    extended = fit_predictor(np.column_stack((x, residual)), y, ids)
    extended['features'] = [*BASIS, 'residual_e']
  for model in (base, extended):
    model.update(objective='unit mean squared loss + .1*sum(standardized slopes**2); intercept unpenalized',
        standardization='all128 calibration units only; each family1/8 and each unit1/16')
  return dict(projection=projection, base=base, extended=extended, basis=list(BASIS))


def basis_predictions(rows, control):
  x = quadratic_basis(rows)
  base = predict(control['base'], x)
  if control['projection']['residual_disabled']:
    return base, base  # Never read held e to reactivate a disabled column.
  extended = predict(control['extended'], np.column_stack((x,
      projected_residual(rows, control['projection']))))
  return base, extended


def freeze_calibration(units128):
  rows = validate_units(units128, 8)
  models = {}
  ids = [r['family_id'] for r in rows]
  for response in ('U_obs', 'U_act'):
    models[response] = {}
    for name, names in FEATURES.items():
      x = [[r['signals'][key] for key in names] for r in rows]
      model = fit_predictor(x, [r[response] for r in rows], ids)
      model.update(features=list(names),
          objective='unit mean squared loss + .1*sum(standardized slopes**2); intercept unpenalized',
          standardization='all128 calibration units only; each family1/8 and each unit1/16')
      models[response][name] = model
  canonical = json.dumps(units128, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
  return dict(version=VERSION, models=models, aux_nonredundant_basis_control=freeze_basis_control(rows),
      calibration_canonical_input_sha256=hashlib.sha256(canonical).hexdigest(),
      D75=float(np.quantile([r['signals']['D'] for r in rows], .75, method='linear')),
      threshold_definition='cal128 D quantile(.75, method=linear); held D >= threshold; retain ties',
      training_weights=[1/128]*128, training_family_ids=list(range(8)), ridge=.1)


def cluster_interval(rows, values, mask, draws):
  """Equal selected-unit target; bootstrap all24 families, including empty ones."""
  values, mask = np.asarray(values, np.float64), np.asarray(mask, bool)
  require(values.shape == mask.shape == (384,) and np.isfinite(values).all(),
      'All384 finite held unit values and mask required')
  totals, counts = [], []
  for family in range(8, 32):
    take = mask & np.array([r['family_id'] == family for r in rows])
    totals.append(float(values[take].sum()))
    counts.append(int(take.sum()))
  totals, counts = np.array(totals), np.array(counts)
  denominator = counts[draws].sum(1)
  empty = int((denominator == 0).sum())
  mean = float(totals.sum()/counts.sum()) if counts.sum() else None
  ci = None if empty or not counts.sum() else np.quantile(
      totals[draws].sum(1)/denominator, [.025, .975], method='linear').tolist()
  return dict(mean=mean, interval95=ci, n_units=int(counts.sum()),
      per_family_sum=totals.tolist(), per_family_count=counts.tolist(),
      empty_bootstrap_denominators=empty, weighting='equal selected units; ratio of family sums/counts')


def sham_test(rows, mask):
  """Conditional qq-membership, iid Rademacher labels, not fixed arm counts."""
  selected = [r for r, keep in zip(rows, mask) if keep and r['label'] in LABELS[:2]]
  if not selected:
    return dict(p_value=None, rejected=False, n_qq=0, status='unresolved')
  z = np.array([r['Y']-r['signals']['r_t'] for r in selected])
  signs = np.array([1 if r['label'] == 'qqA' else -1 for r in selected])
  observed = float(np.sum(signs*z))
  random_signs = 2*np.random.default_rng(SHAM_SEED).integers(0, 2, (DRAWS, len(z)))-1
  simulated = np.sum(random_signs*z, axis=1)
  p = float((1+np.count_nonzero(np.abs(simulated) >= abs(observed)))/(DRAWS+1))
  return dict(p_value=p, rejected=p < SHAM_ALPHA, n_qq=len(z),
      statistic_signed_mean=4*observed/int(np.asarray(mask).sum()),
      seed=SHAM_SEED, draws=DRAWS, alpha=SHAM_ALPHA, status='diagnostic_only',
      null='sharp qqA/qqB equality; conditional qq membership, no cross-unit label interference')


def analyze(data, *, evidence_authenticated=False):
  root = {'version', 'horizon', 'bootstrap_seed', 'units', 'calibration_sealed_event', *PROVENANCE}
  require(type(data) is dict and set(data) in (root, root | {'frozen_models'}), 'Exact root schema required')
  require(type(evidence_authenticated) is bool, 'External authentication must be bool')
  for key, value in dict(version=VERSION, horizon=HORIZON, bootstrap_seed=BOOTSTRAP_SEED, **PROVENANCE).items():
    require(type(data[key]) is type(value) and data[key] == value, f'{key} protocol mismatch')
  rows = validate_units(data['units'])
  seal = data['calibration_sealed_event']
  require(type(seal) is int and data['units'][127]['chronology']['outcome_complete'] <
      seal < data['units'][128]['chronology']['signals_sealed'], 'Cal seal must precede all held signals')
  frozen = freeze_calibration(data['units'][:128])
  supplied = data.get('frozen_models')
  if supplied is not None:
    authenticate_frozen(frozen, supplied)
  require(not evidence_authenticated or supplied is not None, 'Authenticated science requires pre-held cal freeze')
  held = rows[128:]
  high = np.array([r['signals']['D'] >= frozen['D75'] for r in held])
  draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 24, (DRAWS, 24))
  label_counts = {label: int(sum(r['label'] == label and keep for r, keep in zip(held, high))) for label in LABELS}
  families = sorted({r['family_id'] for r, keep in zip(held, high) if keep})
  covered = min(label_counts.values()) >= 8 and len(families) >= 12
  scopes = {}
  for scope, mask in (('highD', high), ('all_held_auxiliary', np.ones(384, bool))):
    responses = {}
    for response in ('U_obs', 'U_act', 'U_memory', 'sham', 'Y'):
      y = np.array([r[response] for r in held])
      report = dict(mean=cluster_interval(held, y, mask, draws))
      if response in ('U_obs', 'U_act'):
        predictions = {name: predict(frozen['models'][response][name],
            [[r['signals'][key] for key in names] for r in held]) for name, names in FEATURES.items()}
        report['comparisons'] = {}
        for name, (base, candidate) in COMPARISONS.items():
          risk0, risk1 = (y-predictions[base])**2, (y-predictions[candidate])**2
          entry = cluster_interval(held, risk0-risk1, mask, draws)
          entry.update(baseline_mse=cluster_interval(held, risk0, mask, draws)['mean'],
              candidate_mse=cluster_interval(held, risk1, mask, draws)['mean'])
          report['comparisons'][name] = entry
      responses[response] = report
    scopes[scope] = responses
  primary = scopes['highD']['U_obs']['comparisons']['DKe_vs_DK']
  sham = scopes['highD']['sham']['mean']['interval95']
  negative_control = sham_test(held, high)
  positive = primary['interval95'] is not None and primary['interval95'][0] > 0
  supporting = evidence_authenticated and covered and positive
  control = frozen['aux_nonredundant_basis_control']
  base, extended = basis_predictions(held, control)
  y = np.array([r['U_obs'] for r in held])
  aux_delta = (y-base)**2-(y-extended)**2
  auxiliary = {scope: cluster_interval(held, aux_delta, mask, draws)
      for scope, mask in (('highD', high), ('all_held_auxiliary', np.ones(384, bool)))}
  aux_ci = auxiliary['highD']['interval95']
  auxiliary.update(support_pattern=evidence_authenticated and covered and
      aux_ci is not None and aux_ci[0] > 0, calibration=control,
      scope='U_obs increment beyond fixed quadratic DK basis; not arbitrary conditional information independence',
      residual_threshold_is_scientific_equivalence=False,
      disabled_column_cannot_reactivate_from_held=True)
  return dict(version=VERSION, primary_support_pattern=supporting,
      aux_nonredundant_basis_control=auxiliary,
      implementation_concern=negative_control['rejected'],
      status='exploratory-primary-support-pattern' if supporting else 'current-finite-procedure-inconclusive-or-not-supported',
      calibration_freeze=frozen, supplied_freeze_authenticated=supplied is not None,
      coverage=dict(passed=covered, highD_label_counts={k: int(v) for k, v in label_counts.items()},
          held_family_ids=families, n_highD=int(high.sum()),
          per_label_family_counts={label: len({r['family_id'] for r, keep in zip(held, high)
              if keep and r['label'] == label}) for label in LABELS},
          limitation='Counts are finite support, not power or positive/negative state truth'),
      sham_negative_control=negative_control,
      sham_precision_flag=sham is not None and sham[0] >= -DELTA/4 and sham[1] <= DELTA/4,
      sham_precision_threshold=DELTA/4, sham_precision_is_support_gate=False,
      external_evidence_authenticated=evidence_authenticated, scopes=scopes,
      bootstrap=dict(seed=BOOTSTRAP_SEED, draws=DRAWS, clusters=24,
          unit='whole held family, retain all16units', refit=False),
      model_protocol=dict(features={k: list(v) for k, v in FEATURES.items()},
          primary='held highD U_obs DKe_vs_DK', baseline='fixed pretreatment suite F64 r_t',
          delta_average_reward=DELTA, primary_gain_units='squared average reward',
          delta_is_primary_gain_threshold=False,
          target='average over selected units, not equal-family conditional means'),
      weighted_gate_effect='not_evaluated', method_effectiveness_established=False,
      conditional_information_independence_established=False,
      training_authorized=False, limitations=[
          'Caller independently authenticates full512 execution, true pre-treatment randomization/reveal, shadow source, physics, reward padding and cal seal.',
          'Schema/provenance declarations cannot establish causal independence or exclude hidden label access.',
          'Nominal24-family bootstrap conditions on cal fit/threshold/checkpoint and assumes suitable family exchangeability; not exact randomization CI or training-seed inference.',
          'Shared worker/GL history is not proved independent by distinct policy streams; hidden cross-unit interference can invalidate interpretation.',
          'HT responses are noisy signed pseudo-outcomes, never per-state expected-utility truth.',
          'Absolute MSE includes pseudo-outcome noise; shared-label risk differences cancel the common noise-squared term.',
          'HighD retains cal75 ties; it may cover far more than25 percent. Cal-all fitting can miss tail heterogeneity.',
          'Sham rejection is implementation concern, not proof of bug; nonrejection/equivalence do not prove conditional measurement validity.',
          'Primary support plus implementation concern must both be reported; no unqualified canonical-qq claim or protection/training follows.',
          'Observed r_t is a legitimate control variate but can increase variance; no baseline selection or sample expansion.',
          'Main linear-ridge increment can arise from duplicate-column penalty changes or nonlinear DK basis functions; absent auxiliary support those alternatives remain unresolved.',
          'The fixed quadratic residual control excludes only that finite basis explanation, not all functions of D/K; 1e-10 disables numerical redundancy, not scientific equivalence.',
          'Pre-treatment shadow e can share the first reward contributing1/100 of H100 return; lawful pre-treatment use does not establish long-term information utilization.',
          'Original H1 diagnostic and current sampled-actor effects differ; prediction increment does not establish information availability or compression protection.'])


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--input', required=True, type=Path)
  parser.add_argument('--output', required=True, type=Path)
  args = parser.parse_args()
  require(not args.output.exists(), 'Refuse existing output')
  result = analyze(decode(args.input.read_text(encoding='utf-8')))
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
  main()
