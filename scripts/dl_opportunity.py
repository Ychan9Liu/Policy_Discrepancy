"""DL-opportunity-r1 CPU analysis of fixed H100 first-action consequences.

H100 cumulative effects are the only primary opportunity labels. Prespecified
H1/10/50 prefixes are descriptive. Coverage may support designing a revision;
it never validates the H1 gate or authorizes training. No environment, JAX,
checkpoint pickle, or GPU is loaded by this analyzer.
"""

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_diagnostic as diagnostic

VERSION = 'DL-opportunity-r1'
PREFIXES = (1, 10, 50, 100)
VARIANTS = ('clean', 'background_a', 'background_b', 'cue_deprivation')
SEED = 20261012
CONFIG_SHA = '11428512fdc9b9e04a5384e419c5abcef93e5429db0a3327d05ccb0d4bd20445'
SOURCES = {
    '2ab693c8ded4d8f2186bd489aa8b77498448ba2f2c331090159da4e5f08056b9':
        (300000, 74493, '14859c4e129b51aa488c10eab09f16d56fd110c76239b2c9072f415829d111c2'),
    'c9c893081485c8ad22468611d046bc5bf9fbe463d55996fda15cf85b2a0323fe':
        (1000000, 249493, 'fb8c429cb13861d90f2aeba32842eda8958bd17d1b80f2dfa15c0363aca7549d'),
}


def require(condition, message):
  if not condition:
    raise ValueError(message)


def exact(actual, expected, message):
  require(np.array_equal(actual, expected), message)


def load_scores(source):
  """Verify raw source proofs, unchanged parameters and common-path axes."""
  source = Path(source).resolve()
  plan_path, complete_path = source / 'score_plan.json', source / 'score_complete.json'
  collection_path = source / 'collection_plan.json'
  collection_complete_path = source / 'collection_complete.json'
  plan = json.loads(plan_path.read_text(encoding='utf-8'))
  complete = json.loads(complete_path.read_text(encoding='utf-8'))
  collection = json.loads(collection_path.read_text(encoding='utf-8'))
  collected = json.loads(collection_complete_path.read_text(encoding='utf-8'))
  require(diagnostic.sha256(plan_path) == complete['plan_sha256'], 'Score plan hash mismatch')
  require(diagnostic.sha256(collection_path) == plan['dataset_plan_sha256'],
      'Copied collection plan hash mismatch')
  require(diagnostic.sha256(collection_complete_path) == plan['dataset_complete_sha256'],
      'Copied collection completion hash mismatch')
  require(collected['plan_sha256'] == plan['dataset_plan_sha256'] and
      collected['params_unchanged'] is True, 'Collection completion plan or parameter proof differs')
  require(complete['params_unchanged'] is True, 'Scoring changed frozen parameters')
  require(plan['opportunity_version'] == VERSION and plan['horizon'] == 100 and
      plan['sample_count'] == 16 and plan['consequence_prefixes'] == list(PREFIXES),
      'Requires frozen opportunity version, H100, 16 positions and prescribed prefixes')
  require(plan['diagnostic_seed'] == SEED, 'Scoring diagnostic RNG seed differs')
  require(plan['variants'] == list(VARIANTS) and plan['task'] == 'dmc_quadruped_walk',
      'Unexpected task or variant order')
  require(plan['alpha'] == 20 and plan['rho'] == .5 and plan['free_nats'] == 1 and
      plan['restoration_tolerance'] == 1e-8 and plan['cross_model_behavior'] is False,
      'Original H1 signal or restoration protocol differs')
  require(plan['consequence'] == 'first q/p action; shared recorded continuation',
      'Unexpected consequence object; closed-loop or different continuation is not supported')
  checkpoint = plan['checkpoint_sha256']
  require(checkpoint in SOURCES, 'Checkpoint is not one of the two fixed audited sources')
  stage, updates, params = SOURCES[checkpoint]
  require(plan['collection_checkpoint_sha256'] == collection['checkpoint_sha256'] == checkpoint,
      'Collection/scoring checkpoint mismatch')
  require(plan['params_sha256'] == plan['collection_params_sha256'] ==
      collection['params_sha256'] == params, 'Collection/scoring parameter digest mismatch')
  require(plan['checkpoint_counters']['updates'] == updates,
      'Checkpoint update counter differs from fixed source stage')
  require(plan['config_source_sha256'] == collection['config_source_sha256'] == CONFIG_SHA,
      'Actual source configuration hash differs')
  require(collection['task'] == plan['task'] and collection['seed'] == SEED,
      'Collection task or independent seed differs')
  expected = diagnostic.episode_plan(count=32, calibration=8, seed=SEED)
  require(collection['episodes'] == expected, 'Collection episode seeds or 8/24 split differ')
  old = diagnostic.episode_plan(count=32, calibration=8, seed=20261010)
  require(not ({e['seed'] for e in expected} & {e['seed'] for e in old}) and
      not ({e['policy_seed'] for e in expected} & {e['policy_seed'] for e in old}),
      'New collection seed plan overlaps original r1 episodes')
  entries = complete['artifacts']
  require(len(entries) == 32 and {x['episode'] for x in entries} == set(range(32)),
      'Requires all 32 distinct planned episodes, without replacement')
  require(len({x['path'] for x in entries}) == 32, 'Score artifact paths must be distinct')
  expected_by_id = {entry['episode']: entry for entry in expected}
  source_entries = collected['artifacts']
  require(len(source_entries) == 32 and {x['episode'] for x in source_entries} == set(range(32))
      and len({x['path'] for x in source_entries}) == 32,
      'Requires all 32 distinct source collection artifacts')
  source_by_id = {}
  for entry in source_entries:
    require({key: entry.get(key) for key in expected_by_id[entry['episode']]} ==
        expected_by_id[entry['episode']], 'Collection artifact seeds or split differ')
    raw_path = Path(entry['path'])
    require(not raw_path.is_absolute() and '..' not in raw_path.parts and
        raw_path.suffix == '.npz' and re.fullmatch(r'[0-9a-f]{64}', entry['sha256']) is not None,
        'Invalid source collection artifact path or hash')
    require(type(entry['frames']) is int and entry['frames'] > 0,
        'Invalid source collection frame count')
    source_by_id[entry['episode']] = entry
  pieces, hashes = [], []
  signal_keys = ('D', 'K', 'm', 'v', 'e', 'f_D', 'signed_support', 'loss_gap', 'u_q', 'u_p')
  for entry in sorted(entries, key=lambda x: x['episode']):
    require(entry['split'] == expected_by_id[entry['episode']]['split'] and
        entry['positions'] == 16, 'Episode split or prespecified sample count differs')
    path = (source / entry['path']).resolve()
    require(path.is_relative_to(source) and path.is_file(), 'Unsafe or missing score artifact path')
    require(diagnostic.sha256(path) == entry['sha256'], 'Score artifact hash mismatch')
    with np.load(path, allow_pickle=False) as file:
      item = {k: file[k].copy() for k in file.files}
    n = entry['positions']
    require(np.issubdtype(item['consequence_prefixes'].dtype, np.integer),
        'Prefix axis must contain integer horizons')
    exact(item['consequence_prefixes'], np.broadcast_to(PREFIXES, (n, 4)),
        'Prefix ordering or repeated prefix axis differs')
    reward = item['true_reward_sequences']
    require(reward.shape == (n, 5, 100) and reward.dtype == np.float64 and
        np.isfinite(reward).all(), 'Requires float64 rewards [position,q4+p1,100]')
    reference = item['reference_reward_sequence']
    require(reference.shape == (n, 100) and reference.dtype == np.float64 and
        np.isfinite(reference).all(), 'Invalid full-window reference reward sequence')
    exact(item['reference_full_window_verified'], np.ones(n, bool),
        'Full recorded-action reference window was not verified')
    require(item['reference_full_window_verified'].dtype == np.bool_,
        'Full-window reference verification must be boolean')
    qsum = np.stack([reward[:, :4, :h].sum(-1) for h in PREFIXES], -1)
    psum = np.stack([reward[:, 4, :h].sum(-1) for h in PREFIXES], -1)
    refsum = np.stack([reference[:, :h].sum(-1) for h in PREFIXES], -1)
    for key, expected_array in dict(true_q_prefix_sum=qsum, true_p_prefix_sum=psum,
        true_q_prefix_mean=qsum / np.asarray(PREFIXES),
        true_p_prefix_mean=psum / np.asarray(PREFIXES), reference_prefix_sum=refsum,
        true_q_h1=reward[:, :4, 0], true_p_h1=reward[:, 4, 0],
        true_q_horizon=reward[:, :4].mean(-1), true_p_horizon=reward[:, 4].mean(-1)).items():
      exact(item[key], expected_array, f'{key}: common sequence or axes mismatch')
    for key in signal_keys:
      require(item[key].shape == (n, 4) and np.isfinite(item[key]).all(),
          f'{key}: invalid original H1 signal shape or values')
    require(np.isin(item['m'], [0, 1]).all() and (item['D'] >= 0).all() and
        (item['K'] >= 0).all() and ((item['v'] >= .5) & (item['v'] <= 1)).all() and
        ((item['e'] >= 0) & (item['e'] <= 1)).all(), 'Invalid H1 mask or signal range')
    require(item['position'].shape == (n,) and np.issubdtype(item['position'].dtype, np.integer)
        and (item['position'] >= 1).all() and (np.diff(item['position']) > 0).all(),
        'Positions must be distinct ordered preselected episode positions')
    source_entry = source_by_id[entry['episode']]
    predetermined = diagnostic.sample_positions(source_entry['frames'], maximum=16, horizon=100)
    require(len(predetermined) == n, 'Source episode lacks sixteen legal full H100 windows')
    exact(item['position'], predetermined, 'Positions differ from frozen source-frame selection')
    require(item['physics_repeat_error'].shape == (n,) and
        np.isfinite(item['physics_repeat_error']).all() and
        (item['physics_repeat_error'] >= 0).all() and
        (item['physics_repeat_error'] <= 1e-8).all(), 'Full consequence restoration error')
    item['episode'] = np.full(n, entry['episode'])
    item['blind'] = np.full(n, entry['split'] == 'blind')
    pieces.append(item)
    hashes.append(dict(episode=entry['episode'], split=entry['split'],
        path=entry['path'], sha256=entry['sha256'],
        collection_path=source_entry['path'], collection_sha256=source_entry['sha256'],
        source_frames=source_entry['frames'], expected_positions=predetermined.tolist()))
  common = set.intersection(*(set(piece) for piece in pieces))
  data = {key: np.concatenate([piece[key] for piece in pieces])
      for key in common if key != 'bins'}
  if 'bins' in common:
    for piece in pieces[1:]:
      exact(piece['bins'], pieces[0]['bins'], 'Reward bins changed across score artifacts')
    data['bins'] = pieces[0]['bins']
  proof = dict(stage=stage, checkpoint_sha256=checkpoint, params_sha256=params,
      config_source_sha256=CONFIG_SHA, score_plan_sha256=diagnostic.sha256(plan_path),
      score_complete_sha256=diagnostic.sha256(complete_path),
      collection_plan_sha256=diagnostic.sha256(collection_path), artifacts=hashes,
      collection_complete_sha256=diagnostic.sha256(collection_complete_path),
      scope='NPZ/provenance integrity and sequence arithmetic; does not re-run physics or hash remote checkpoint bytes')
  return data, plan, proof


def distribution(values, episodes, draws, seed):
  values = np.asarray(values, np.float64)
  if not len(values):
    return dict(positions=0, episodes=0, quantiles=None, mean_interval=None)
  return dict(positions=len(values), episodes=len(np.unique(episodes)),
      quantiles={str(q): float(np.quantile(values, q)) for q in (0, .1, .25, .5, .75, .9, 1)},
      mean_interval=diagnostic.cluster_interval(values, episodes, draws=draws, seed=seed),
      per_episode=[dict(episode=int(e), positions=int((episodes == e).sum()),
          mean=float(values[episodes == e].mean()), min=float(values[episodes == e].min()),
          max=float(values[episodes == e].max())) for e in np.unique(episodes)])


def analyze(scores, output, bootstrap=2000, seed=SEED):
  output = Path(output)
  if output.exists():
    raise FileExistsError(output)
  require(bootstrap == 2000 and seed == SEED, 'Requires frozen bootstrap=2000 and analysis seed=20261012')
  data, plan, proof = load_scores(scores)
  blind, episode = data['blind'], data['episode']
  qsum, psum = data['true_q_prefix_sum'], data['true_p_prefix_sum']
  gain = qsum[:, 0] - psum
  cue = qsum[:, 0] - qsum[:, 3]
  positive, negative, useful = gain[:, -1] >= .1, gain[:, -1] <= 0, cue[:, -1] >= .1
  gray, joint = ~(positive | negative), positive & useful
  C = 1 - data['v'][:, 0]
  result = dict(opportunity_version=VERSION, analysis_git_sha=diagnostic.git_sha(),
      analysis=dict(bootstrap=bootstrap, seed=seed, statistical_unit='independent episode',
          primary_prefix=100, other_prefixes='descriptive only; no winner selection'),
      source=proof, frozen_score_plan=plan, blind_positions=int(blind.sum()),
      blind_episodes=int(len(np.unique(episode[blind]))), calibration_positions=int((~blind).sum()),
      calibration_episodes=int(len(np.unique(episode[~blind]))),
      label=dict(positive='clean q H100 cumulative minus p >= .1',
          negative='clean q H100 cumulative minus p <= 0', gray='0 < gain < .1',
          useful_cue='clean q H100 cumulative minus deprived q >= .1',
          joint='positive and useful_cue'),
      causal_object='Clipped mean first-action intervention; 99 common recorded continuations; not closed-loop or policy value',
      per_prefix={}, coverage={}, signal_relationship={}, training_authorized=False,
      proceed_short_training=False, eligible_method_training=False,
      limitations=['Same baseline training seed at two stages; analyze stages separately.',
          'Cue deprivation may be out of distribution; joint does not prove posterior causal correctness.',
          'Enough opportunity is not H1 gate validity, representation protection benefit, or method readiness.',
          'Original r1 failed/inconclusive decisions and thresholds are unchanged.'])
  for j, h in enumerate(PREFIXES):
    result['per_prefix'][str(h)] = dict(primary=h == 100,
        clean_q_minus_p_sum=distribution(gain[blind, j], episode[blind], bootstrap, seed),
        clean_q_minus_p_mean=distribution(gain[blind, j] / h, episode[blind], bootstrap, seed),
        cue_cost_sum=distribution(cue[blind, j], episode[blind], bootstrap, seed),
        cue_cost_mean=distribution(cue[blind, j] / h, episode[blind], bootstrap, seed),
        q_variant_minus_p_sum={name: distribution((qsum[:, v, j] - psum[:, j])[blind],
            episode[blind], bootstrap, seed) for v, name in enumerate(VARIANTS)})
  for name, mask in dict(positive=positive, negative=negative, gray=gray,
      useful_cue=useful, joint=joint).items():
    count, episodes = int((blind & mask).sum()), int(len(np.unique(episode[blind & mask])))
    result['coverage'][name] = dict(positions=count, episodes=episodes,
        sufficient=count >= 128 and episodes >= 8,
        blind_rate_interval=diagnostic.cluster_interval(mask[blind].astype(float),
            episode[blind], draws=bootstrap, seed=seed))
  coverage = result['coverage']
  signal_coverage = coverage['positive']['sufficient'] and coverage['negative']['sufficient']
  sufficient = signal_coverage and coverage['useful_cue']['sufficient'] and coverage['joint']['sufficient']
  result.update(signal_label_coverage_sufficient=signal_coverage,
      target_opportunity_coverage_sufficient=sufficient,
      opportunity_guard='may_design_revision' if sufficient else 'limited_opportunity',
      next_step='May design a separately frozen revision with independent validation; no training'
          if sufficient else 'Stop this H100 opportunity branch; do not extend H, alter labels, or select favorable episodes')
  candidates = {key: data[key][:, 0] for key in
      ('D', 'f_D', 'e', 'signed_support', 'loss_gap', 'u_q', 'u_p', 'K')}
  candidates['C'] = C
  for name, mask in dict(all=np.ones(len(blind), bool), positive=positive,
      negative=negative, gray=gray, joint=joint).items():
    idx = blind & mask
    result['signal_relationship'][name] = {key: distribution(value[idx], episode[idx],
        bootstrap, seed) for key, value in candidates.items()}
  labelled = blind & (positive | negative)
  labels, eps = positive[labelled], episode[labelled]
  result['exploratory_auc'] = {}
  for key in ('C', 'D', 'e', 'K'):
    values = candidates[key][labelled]
    result['exploratory_auc'][key] = diagnostic.cluster_interval(values, eps,
        statistic=lambda idx, v=values: diagnostic.auc(labels[idx], v[idx]),
        draws=bootstrap, seed=seed)
  result['exploratory_auc']['paired_C_minus_D'] = diagnostic.cluster_interval(C[labelled], eps,
      statistic=lambda idx: None if not labels[idx].any() or labels[idx].all() else
          diagnostic.auc(labels[idx], C[labelled][idx]) -
          diagnostic.auc(labels[idx], candidates['D'][labelled][idx]), draws=bootstrap, seed=seed)
  result['exploratory_auc'].update(sufficient_label_coverage=signal_coverage,
      interpretation='Exploratory relation to this new external object; not original r1 acceptance or gate validation')
  excess = data['m'][:, 0] * np.maximum(data['K'][:, 0] - 1, 0)
  def release_stat(idx):
    weight = excess[blind][idx]
    return float(np.sum(C[blind][idx] * weight) / weight.sum()) if weight.sum() > 0 else None
  result['original_H1_release'] = dict(formula='sum(m*max(K-1,0)*C)/sum(m*max(K-1,0))',
      ratio_interval=diagnostic.cluster_interval(C[blind], episode[blind],
          statistic=release_stat, draws=bootstrap, seed=seed),
      active_positions=int((blind & (excess > 0)).sum()),
      active_episodes=int(len(np.unique(episode[blind & (excess > 0)]))),
      total_excess=float(excess[blind].sum()),
      by_primary_label={name: dict(excess=float(excess[blind & mask].sum()),
          released_excess=float((C * excess)[blind & mask].sum()))
          for name, mask in dict(positive=positive, negative=negative, gray=gray, joint=joint).items()},
      interpretation='Same H1 gate magnitude on fresh opportunity data; does not overwrite original r1 release')
  calibration = ~blind
  d_bins = np.unique(np.quantile(data['D'][calibration, 0], [1/3, 2/3]))
  active_calibration = calibration & (data['K'][:, 0] > 1)
  k_bins = (np.asarray([np.median(data['K'][active_calibration, 0])])
      if active_calibration.any() else np.asarray([]))
  db, kb = np.digitize(data['D'][:, 0], d_bins), np.digitize(data['K'][:, 0], k_bins)
  strata = []
  for d in range(len(d_bins) + 1):
    for k in range(len(k_bins) + 1):
      mask = blind & (db == d) & (kb == k)
      if not mask.any():
        continue
      idx = mask & (positive | negative)
      strata.append(dict(d_bin=d, k_bin=k, positions=int(mask.sum()),
          episodes=int(len(np.unique(episode[mask]))),
          positive_positions=int((mask & positive).sum()),
          negative_positions=int((mask & negative).sum()), joint_positions=int((mask & joint).sum()),
          mean_C=float(C[mask].mean()), mean_e=float(data['e'][mask, 0].mean()),
          descriptive_auc_C=diagnostic.auc(positive[idx], C[idx]),
          descriptive_auc_D=diagnostic.auc(positive[idx], data['D'][idx, 0]),
          descriptive_auc_e=diagnostic.auc(positive[idx], data['e'][idx, 0])))
  result['opportunity_calibration_strata'] = dict(d_bins=d_bins.tolist(),
      k_bins=k_bins.tolist(), calibration_episodes=8,
      source='Fresh opportunity 8 calibration episodes only; D tertiles and active-K median',
      interpretation='Descriptive relation only; not original r1 S/P/W control validity, gate parameters or revised method choices',
      strata=strata)
  # Describe counterexamples on the same data; do not rerun r1 pass/fail gates.
  result['H1_counterexamples'] = dict(nuisance_delta_C={VARIANTS[v]:
      distribution((1 - data['v'][:, v] - C)[blind], episode[blind], bootstrap, seed)
      for v in (1, 2)}, cue_delta_C=distribution(
          (C - (1 - data['v'][:, 3]))[blind], episode[blind], bootstrap, seed))
  for key in ('mode_perturb_C', 'mode_perturb_e', 'q_margin_min', 'fixed_e', 'fixed_signed_support'):
    if key in data:
      value = data[key][:, 0]
      if value.ndim > 1:
        value = value.mean(-1)
      result['H1_counterexamples'][key] = distribution(value[blind], episode[blind], bootstrap, seed)
  if all(key in data for key in ('raw_logits_q', 'raw_logits_p', 'next_reward', 'bins')):
    rewards, _ = diagnostic.different_episode_labels(data['next_reward'][blind], episode[blind], seed)
    shuffled = diagnostic.compatibility_numpy(data['raw_logits_q'][blind, 0],
        data['raw_logits_p'][blind, 0], rewards, data['bins'])
    result['H1_counterexamples']['label_shuffle_real_minus_shuffled_e'] = distribution(
        data['e'][blind, 0] - shuffled['e'], episode[blind], bootstrap, seed)
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--scores', required=True)
  parser.add_argument('--output', required=True, help='new immutable JSON result path')
  parser.add_argument('--bootstrap', type=int, default=2000)
  parser.add_argument('--seed', type=int, default=SEED)
  args = parser.parse_args()
  result = analyze(args.scores, args.output, args.bootstrap, args.seed)
  print(json.dumps(dict(opportunity_version=VERSION, stage=result['source']['stage'],
      opportunity_guard=result['opportunity_guard'], proceed_short_training=False)))


if __name__ == '__main__':
  main()
