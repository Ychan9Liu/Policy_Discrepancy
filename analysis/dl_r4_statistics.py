"""DL-r4-utility-r1: fixed-procedure CPU statistics, never a physics verifier.

The input contains authenticated reward sums, not independent state truth.
The caller must separately authenticate branch records, hashes and physics.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

VERSION = 'DL-r4-utility-r1'
HORIZON, PAIRS, DELTA, RIDGE = 100, 16, .01, .1
BOOTSTRAP_SEED, BOOTSTRAP_DRAWS = 20261028, 10000
CALIBRATION = tuple(range(8))
POSITIONS = tuple(range(64, 897, 64))
ARMS = ('qq', 'pq', 'pp')
FEATURES = {'D': ('D', 'time'), 'DK': ('D', 'K', 'time'),
    'DKe': ('D', 'K', 'e', 'time'), 'e': ('e', 'time')}
COMPARISONS = {'DKe_vs_DK': ('DK', 'DKe'),
    'DK_vs_D': ('D', 'DK'), 'DKe_vs_D': ('D', 'DKe'),
    'e_vs_D': ('D', 'e')}
T15 = 2.131449545559323


def require(value, message):
  if not value:
    raise ValueError(message)


def finite(value, low=None, high=None):
  require(type(value) in (int, float) and math.isfinite(value),
      'Require finite JSON numeric scalar, not bool')
  value = float(value)
  require(low is None or value >= low, 'Scalar below allowed range')
  require(high is None or value <= high, 'Scalar above allowed range')
  return value


def keys(value, expected, name):
  require(type(value) is dict and set(value) == set(expected),
      f'Exact {name} schema required')


def decode(payload):
  def unique(items):
    out = {}
    for key, value in items:
      require(key not in out, 'Duplicate JSON key')
      out[key] = value
    return out
  return json.loads(payload, object_pairs_hook=unique,
      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def selected_positions(candidates):
  """D only, deterministic lower-position tie break; never read e or utility."""
  require(type(candidates) is list and len(candidates) == len(POSITIONS),
      'Exactly 14 position candidates required')
  for candidate in candidates:
    keys(candidate, ('position', 'D', 'K', 'e'), 'candidate')
    require(type(candidate['position']) is int, 'Integer position required')
    finite(candidate['D'], 0)
    finite(candidate['K'], 0)
    finite(candidate['e'], 0, 1)
  require(tuple(x['position'] for x in candidates) == POSITIONS,
      'Candidate grid must be exactly 64..896, sorted by position')
  top2 = tuple(sorted(x['position'] for x in sorted(candidates,
      key=lambda x: (-x['D'], x['position']))[:2]))
  return top2, tuple(sorted(set((*top2, 512))))


def validate(data, *, expected_episodes=32):
  expected_keys = {'version', 'horizon', 'mc_pairs', 'calibration_episode_ids',
      'bootstrap_seed', 'episodes'}
  require(type(data) is dict and set(data) in (expected_keys,
      expected_keys | {'frozen_models'}), 'Exact root schema required')
  require(data['version'] == VERSION, 'Version mismatch')
  require(type(data['horizon']) is int and data['horizon'] == HORIZON,
      'Require unchanged H100')
  require(type(data['mc_pairs']) is int and data['mc_pairs'] == PAIRS,
      'Require exactly 16 paired roots')
  require(type(data['calibration_episode_ids']) is list and
      all(type(x) is int for x in data['calibration_episode_ids']) and
      tuple(data['calibration_episode_ids']) == CALIBRATION,
      'Calibration IDs must be new episode IDs 0..7')
  require(type(data['bootstrap_seed']) is int and
      data['bootstrap_seed'] == BOOTSTRAP_SEED, 'Bootstrap seed mismatch')
  episodes = data['episodes']
  require(type(episodes) is list and len(episodes) == expected_episodes,
      f'Exactly {expected_episodes} complete episodes required')
  rows = []
  for episode_id, episode in enumerate(episodes):
    keys(episode, ('episode_id', 'env_seed', 'candidates', 'selected'), 'episode')
    require(type(episode['episode_id']) is int and
        episode['episode_id'] == episode_id, 'Sorted unique episode IDs 0..31 required')
    require(type(episode['env_seed']) is int and
        episode['env_seed'] == 4096+episode_id, 'New environment seed mismatch')
    top2, union = selected_positions(episode['candidates'])
    selected = episode['selected']
    require(type(selected) is list and len(selected) == len(union),
        'D-only top2 plus deduplicated fixed512 required')
    lookup = {x['position']: x for x in episode['candidates']}
    for position, state in zip(union, selected):
      keys(state, ('position', 'returns', 'qq_null_return',
          'null_max_abs_step_reward_diff'), 'selected state')
      require(type(state['position']) is int and state['position'] == position,
          'Selected positions must equal independently recomputed D-only union')
      keys(state['returns'], ARMS, 'returns')
      returns = {}
      for arm in ARMS:
        values = state['returns'][arm]
        require(type(values) is list and len(values) == PAIRS,
            'Each arm must have exactly 16 ordered paired returns')
        returns[arm] = np.array([finite(x, 0, HORIZON) for x in values], np.float64)
      null = finite(state['qq_null_return'], 0, HORIZON)
      null_step = finite(state['null_max_abs_step_reward_diff'], 0, 1)
      # Root must authenticate actual per-step [0,1] rewards and same-H padding.
      # A valid aggregate sum by itself cannot establish these facts.
      signals = {key: float(lookup[position][key]) for key in ('D', 'K', 'e')}
      signals['time'] = position / 1000
      rows.append(dict(episode_id=episode_id, env_seed=4096+episode_id,
          position=position, top2=position in top2, fixed512=position == 512,
          signals=signals, returns=returns,
          U_act=(returns['qq']-returns['pq'])/HORIZON,
          U_obs=(returns['qq']-returns['pp'])/HORIZON,
          U_memory=(returns['pq']-returns['pp'])/HORIZON,
          null_average_difference=float(abs(null-returns['qq'][0])/HORIZON),
          null_max_abs_step_reward_diff=null_step))
  return rows


def episode_weights(episode_ids):
  ids = np.asarray(episode_ids)
  require(ids.ndim == 1 and ids.size > 0 and np.issubdtype(ids.dtype, np.integer),
      'Nonempty integer episode vector required')
  unique, counts = np.unique(ids, return_counts=True)
  lookup = dict(zip(unique.tolist(), counts.tolist()))
  return np.array([1/(len(unique)*lookup[x]) for x in ids], np.float64)


def fit_predictor(features, targets, episode_ids):
  """Minimize EP/state-weighted mean loss + .1*||standardized slopes||².

  The intercept is unpenalized. All moments come only from these training rows.
  Duplicating every row of one episode leaves the fitted function unchanged.
  """
  x, y = np.asarray(features, np.float64), np.asarray(targets, np.float64)
  w = episode_weights(episode_ids)
  require(x.ndim == 2 and x.shape[1] > 0 and x.shape[0] == y.size == w.size and
      y.ndim == 1 and np.isfinite(x).all() and np.isfinite(y).all(),
      'Finite aligned feature/target rows required')
  with np.errstate(over='ignore', invalid='ignore'):
    center = w @ x
    scale = np.sqrt(w @ ((x-center)**2))
  require(np.isfinite(center).all() and np.isfinite(scale).all(),
      'Nonfinite training standardization')
  constant = scale == 0
  scale = np.where(constant, 1., scale)
  z = (x-center)/scale
  z[:, constant] = 0
  target_center = float(w @ y)
  # Center again numerically to preserve the unpenalized intercept explicitly.
  z_center = w @ z
  centered = z-z_center
  coefficient = np.linalg.solve(centered.T @ (w[:, None]*centered) +
      RIDGE*np.eye(x.shape[1]), centered.T @ (w*(y-target_center)))
  intercept = float(target_center-z_center @ coefficient)
  require(np.isfinite(coefficient).all() and math.isfinite(intercept),
      'Nonfinite fitted predictor')
  return dict(center=center.tolist(), scale=scale.tolist(),
      constant=constant.tolist(), coefficient=coefficient.tolist(),
      intercept=intercept, ridge=RIDGE,
      objective='episode/state weighted mean squared loss + .1*sum(slopes**2); intercept unpenalized',
      standardization='training episodes only; EP equal, states within EP equal',
      prediction_clipping=False)


def predict(model, features):
  x = np.asarray(features, np.float64)
  center, scale = np.array(model['center']), np.array(model['scale'])
  require(x.ndim == 2 and x.shape[1] == center.size and np.isfinite(x).all(),
      'Finite predictor input of frozen feature dimension required')
  with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
    z = (x-center)/scale
  z[:, np.array(model['constant'], bool)] = 0
  require(np.isfinite(z).all(), 'Nonfinite held standardized features')
  with np.errstate(over='ignore', invalid='ignore'):
    out = model['intercept'] + z @ np.array(model['coefficient'])
  require(np.isfinite(out).all(), 'Nonfinite prediction')
  return out


def freeze_calibration(episodes8):
  """Fit from cal8 alone; suitable for sealing before any held labels open.

  The canonical-input hash supplements, and does not replace, the caller's raw
  cal artifact bytes/hash and pre-held supervisor seal.
  """
  payload = dict(version=VERSION, horizon=HORIZON, mc_pairs=PAIRS,
      calibration_episode_ids=list(CALIBRATION), bootstrap_seed=BOOTSTRAP_SEED,
      episodes=episodes8)
  rows = validate(payload, expected_episodes=8)
  train = [i for i, row in enumerate(rows) if row['top2']]
  ids = [rows[i]['episode_id'] for i in train]
  models = {}
  for response in ('U_act', 'U_obs'):
    models[response] = {}
    y = np.array([row[response].mean() for row in rows])
    for name, names in FEATURES.items():
      x = np.array([[row['signals'][key] for key in names] for row in rows])
      model = fit_predictor(x[train], y[train], ids)
      model.update(features=list(names), training_episode_ids=list(CALIBRATION),
          training_scope='D-top2 only, exactly two states per episode')
      models[response][name] = model
  canonical = json.dumps(episodes8, sort_keys=True, separators=(',', ':'),
      allow_nan=False).encode('utf-8')
  return dict(version=VERSION, models=models,
      calibration_canonical_input_sha256=hashlib.sha256(canonical).hexdigest(),
      canonical_hash_encoding='UTF8 json.dumps(sort_keys=True,separators=(comma,colon),allow_nan=False)',
      training_weights=[dict(episode_id=rows[i]['episode_id'],
          position=rows[i]['position'], weight=float(w))
          for i, w in zip(train, episode_weights(ids))],
      null_passed=all(row['null_average_difference'] <= DELTA/4 for row in rows),
      scope='Only calibration IDs0..7; caller separately seals raw input SHA and pre-held timing')


def authenticate_frozen(actual, frozen):
  """Compare the independent cal-only refit, never refit or tune on held data."""
  if isinstance(actual, dict):
    keys(frozen, actual, 'frozen model')
    for name, value in actual.items():
      authenticate_frozen(value, frozen[name])
  elif isinstance(actual, list):
    require(type(frozen) is list and len(actual) == len(frozen),
        'Frozen model sequence mismatch')
    for value, other in zip(actual, frozen):
      authenticate_frozen(value, other)
  elif type(actual) is float:
    require(abs(actual-finite(frozen)) <= 1e-12,
        'Frozen cal-only model arithmetic mismatch')
  else:
    require(type(actual) is type(frozen) and actual == frozen,
        'Frozen model exact field mismatch')


def summary(values):
  values = np.asarray(values, np.float64)
  require(values.shape == (PAIRS,) and np.isfinite(values).all(),
      'Exactly 16 finite MC values required')
  mean = float(values.mean())
  se = float(values.std(ddof=1)/math.sqrt(PAIRS))
  lower, upper = mean-T15*se, mean+T15*se
  if lower > DELTA:
    status = 'exploratory_useful'
  elif upper < -DELTA:
    status = 'exploratory_harmful'
  elif lower >= -DELTA and upper <= DELTA:
    status = 'exploratory_practically_small'
  else:
    status = 'exploratory_unresolved'
  return dict(mean=mean, mc_se=se, interval95=[lower, upper], classification=status,
      half8_means=[float(values[:8].mean()), float(values[8:].mean())],
      min=float(values.min()), max=float(values.max()), values=values.tolist(),
      interval_scope='Nominal Student t15 description; not multiplicity corrected, not state truth')


def interval(episode_values, draws):
  values = np.asarray(episode_values, np.float64)
  require(values.shape == (24,) and np.isfinite(values).all(),
      'All 24 held episodes required')
  sampled = values[draws].mean(1)
  return dict(mean=float(values.mean()),
      interval95=np.quantile(sampled, [.025, .975], method='linear').tolist(),
      per_episode=values.tolist())


def group_episode(rows, values, *, scope, held=True):
  ids = range(8, 32) if held else range(8)
  out = []
  for episode_id in ids:
    require(scope in ('union', 'top2', 'fixed512'), 'Unknown evaluation scope')
    indices = [i for i, row in enumerate(rows) if row['episode_id'] == episode_id and
        (scope == 'union' or (scope == 'top2' and row['top2']) or
            (scope == 'fixed512' and row['fixed512']))]
    require(indices, 'Each episode must contribute to the declared scope')
    out.append(float(np.asarray(values)[indices].mean()))
  return out


def coverage(rows, scope='top2'):
  useful, nonuseful, classifications = set(), set(), {}
  for row in rows:
    if row['episode_id'] < 8 or (scope == 'top2' and not row['top2']):
      continue
    label = summary(row['U_obs'])['classification']
    classifications[label] = classifications.get(label, 0)+1
    if label == 'exploratory_useful':
      useful.add(row['episode_id'])
    if label in ('exploratory_harmful', 'exploratory_practically_small'):
      nonuseful.add(row['episode_id'])
  return dict(useful_episode_ids=sorted(useful), nonuseful_episode_ids=sorted(nonuseful),
      state_classification_counts=classifications,
      passed=len(useful) >= 4 and len(nonuseful) >= 4,
      required_distinct_episodes_each=4,
      scope='Held D-top2 U_obs only; finite coverage prerequisite, not power or truth')


def analyze(data, *, evidence_authenticated=False):
  require(type(evidence_authenticated) is bool, 'Explicit external-authentication bool required')
  rows = validate(data)
  frozen = freeze_calibration(data['episodes'][:8])
  provided = data.get('frozen_models')
  if provided is not None:
    authenticate_frozen(frozen, provided)
  require(not evidence_authenticated or provided is not None,
      'Scientific authentication requires the pre-held frozen_models receipt')
  draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 24, (BOOTSTRAP_DRAWS, 24))
  models, predictions, responses = {}, {}, {}
  for response in ('U_act', 'U_obs'):
    models[response], predictions[response] = {}, {}
    y = np.array([row[response].mean() for row in rows])
    for name, names in FEATURES.items():
      x = np.array([[row['signals'][key] for key in names] for row in rows])
      model = frozen['models'][response][name]
      models[response][name] = model
      predictions[response][name] = predict(model, x)
    scopes = {}
    for scope in ('top2', 'fixed512', 'union'):
      comparisons = {}
      for comparison, (baseline, candidate) in COMPARISONS.items():
        p0, p1 = predictions[response][baseline], predictions[response][candidate]
        with np.errstate(over='ignore', invalid='ignore'):
          risk0, risk1 = (y-p0)**2, (y-p1)**2
        require(np.isfinite(risk0).all() and np.isfinite(risk1).all(),
            'Nonfinite predictive risk')
        risk_delta = group_episode(rows, risk0-risk1, scope=scope)
        entry = interval(risk_delta, draws)
        entry.update(baseline=baseline, candidate=candidate,
            baseline_mse=float(np.mean(group_episode(rows, risk0, scope=scope))),
            candidate_mse=float(np.mean(group_episode(rows, risk1, scope=scope))))
        halves = []
        for section in (slice(0, 8), slice(8, 16)):
          yh = np.array([row[response][section].mean() for row in rows])
          with np.errstate(over='ignore', invalid='ignore'):
            delta = (yh-p0)**2-(yh-p1)**2
          require(np.isfinite(delta).all(), 'Nonfinite half-root predictive risk')
          halves.append(interval(group_episode(rows, delta, scope=scope), draws))
        entry['held_half8_fixed_predictor'] = halves
        entry['half8_positive_means_both'] = all(x['mean'] > 0 for x in halves)
        comparisons[comparison] = entry
      scopes[scope] = dict(utility=interval(group_episode(rows, y, scope=scope), draws),
          comparisons=comparisons)
    responses[response] = scopes
  memory = np.array([row['U_memory'].mean() for row in rows])
  responses['U_memory'] = {scope: dict(
      utility=interval(group_episode(rows, memory, scope=scope), draws),
      scope='Reference-path memory effect pq-pp, auxiliary; no separate predictor fit')
      for scope in ('top2', 'fixed512', 'union')}
  nulls = [row['null_average_difference'] for row in rows]
  null_passed = all(value <= DELTA/4 for value in nulls)
  covered = coverage(rows)
  primary = responses['U_obs']['top2']['comparisons']['DKe_vs_DK']
  measurement_passed = evidence_authenticated and null_passed
  if not measurement_passed:
    status = 'measurement-inconclusive'
  elif not covered['passed']:
    status = 'coverage-inconclusive'
  elif primary['interval95'][0] > 0:
    status = 'exploratory-supporting-finite-procedure'
  elif primary['interval95'][1] <= 0:
    status = 'current-finite-procedure-not-supported'
  else:
    status = 'increment-inconclusive'
  selected = []
  for i, row in enumerate(rows):
    selected.append(dict(episode_id=row['episode_id'], env_seed=row['env_seed'],
        position=row['position'], top2=row['top2'], fixed512=row['fixed512'],
        signals=row['signals'],
        U_act=summary(row['U_act']), U_obs=summary(row['U_obs']),
        U_memory=summary(row['U_memory']),
        null_average_difference=row['null_average_difference'],
        null_max_abs_step_reward_diff=row['null_max_abs_step_reward_diff'],
        null_passed=row['null_average_difference'] <= DELTA/4,
        predictions={r: {name: float(p[i]) for name, p in predictions[r].items()}
            for r in predictions}))
  per_episode = []
  for episode_id in range(32):
    members = [row for row in rows if row['episode_id'] == episode_id and row['top2']]
    report = dict(episode_id=episode_id, partition='calibration' if episode_id < 8 else 'held',
        positions=[row['position'] for row in members], responses={})
    for response in ('U_act', 'U_obs', 'U_memory'):
      conditional_var = sum(float(row[response].var(ddof=1))/PAIRS
          for row in members)/len(members)**2
      report['responses'][response] = dict(
          mean=float(np.mean([row[response].mean() for row in members])),
          conditional_mc_se=math.sqrt(conditional_var),
          half8_means=[float(np.mean([row[response][start:start+8].mean()
              for row in members])) for start in (0, 8)])
    per_episode.append(report)
  return dict(version=VERSION, status=status, horizon=HORIZON, mc_pairs=PAIRS,
      bootstrap=dict(seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS,
          unit='whole held episode', fit_refitted=False,
          scope='Descriptive fixed-procedure/fixed-training-model interval; not training-seed or MC-integration guarantee'),
      model_protocol=dict(ridge=RIDGE, delta_average_reward=DELTA,
          primary='U_obs / held D-top2 / DKe_vs_DK',
          primary_training='calibration0..7 D-top2, EP equal then state equal',
          auxiliary='fixed512 and selected union use the same top2-fitted predictors; no refit'),
      measurement=dict(external_evidence_authenticated=evidence_authenticated,
          authentication_scope='Caller must verify branch/hash/physics/raw/applied/per-step reward/padding; this module only checks compact statistics',
          null_threshold_average_reward=DELTA/4,
          null_max_average_difference=max(nulls), null_passed=null_passed,
          passed=measurement_passed,
          limitation='One repeated qq pair0 per state is not a global MC-error or renderer-distribution bound'),
      coverage=covered, predictors=models, responses=responses,
      calibration_freeze=frozen, supplied_freeze_authenticated=provided is not None,
      selected_states=selected, per_episode=per_episode,
      limitations=[
          '16-root MC means and nominal state intervals are not expected-utility truth.',
          'State classes are exploratory and not multiplicity corrected.',
          'D/H1 mode features and sampled-policy utilities are different estimands.',
          'High D here means the frozen D-top2 selected grid, not all high-D replay states.',
          'Null qualification/coverage are prerequisites, not power or classification-sufficiency proofs.',
          'Bootstrap holds eight calibration episodes, fits, checkpoint and model training seed fixed.',
          'Utility describes current actor/memory use, not information availability or protection benefit.',
          'No automatic additional episodes/roots, gate adoption or training follows any result.'],
      method_effectiveness_established=False, training_authorized=False,
      proceed_short_training=False)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--input', required=True, type=Path)
  parser.add_argument('--output', required=True, type=Path)
  args = parser.parse_args()
  require(not args.output.exists(), 'Refuse to overwrite existing output')
  data = decode(args.input.read_text(encoding='utf-8'))
  result = analyze(data)  # CLI cannot claim external physics authentication.
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False,
      allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
  main()
