"""Synthetic full artifact-path engineering tests, never real opportunity evidence."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('dl_opportunity', ROOT / 'scripts/dl_opportunity.py')
opportunity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(opportunity)
diagnostic = opportunity.diagnostic


def write(path, value):
  path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n', encoding='utf-8')


def synthetic_scores(source, adequate=True):
  """32 distinct files with five common H100 sequences and four prefix axes."""
  source.mkdir()
  episodes = diagnostic.episode_plan(32, 8, opportunity.SEED)
  checkpoint, (stage, updates, params) = next(iter(opportunity.SOURCES.items()))
  collection = dict(task='dmc_quadruped_walk', seed=opportunity.SEED, episodes=episodes,
      checkpoint_sha256=checkpoint, params_sha256=params,
      config_source_sha256=opportunity.CONFIG_SHA)
  write(source / 'collection_plan.json', collection)
  collected = dict(params_unchanged=True,
      plan_sha256=diagnostic.sha256(source / 'collection_plan.json'),
      artifacts=[dict(**entry, frames=1001, path=f'episode_{entry["episode"]:03}.npz',
          sha256=f'{entry["episode"]:064x}') for entry in episodes])
  write(source / 'collection_complete.json', collected)
  plan = dict(opportunity_version=opportunity.VERSION, horizon=100, sample_count=16,
      diagnostic_seed=opportunity.SEED,
      consequence_prefixes=list(opportunity.PREFIXES), variants=list(opportunity.VARIANTS),
      task=collection['task'], alpha=20., rho=.5, free_nats=1., restoration_tolerance=1e-8,
      cross_model_behavior=False, consequence='first q/p action; shared recorded continuation',
      checkpoint_sha256=checkpoint, collection_checkpoint_sha256=checkpoint,
      params_sha256=params, collection_params_sha256=params,
      checkpoint_counters=dict(updates=updates), config_source_sha256=opportunity.CONFIG_SHA,
      dataset_plan_sha256=diagnostic.sha256(source / 'collection_plan.json'),
      dataset_complete_sha256=diagnostic.sha256(source / 'collection_complete.json'),
      engineering_fixture=True)
  write(source / 'score_plan.json', plan)
  first_blind = next(e['episode'] for e in episodes if e['split'] == 'blind')
  artifacts = []
  for entry in episodes:
    positive = np.arange(16) < (8 if adequate else 2 if entry['episode'] == first_blind else 0)
    rewards = np.full((16, 5, 100), .5, np.float64)
    rewards[:, 0] += np.where(positive, .002, -.002)[:, None]
    rewards[:, 1] += .001  # Deliberately different variant axis.
    rewards[:, 2] -= .001
    rewards[:, 3] = rewards[:, 0] - .003
    reference = np.full((16, 100), .5, np.float64)
    qsum = np.stack([rewards[:, :4, :h].sum(-1) for h in opportunity.PREFIXES], -1)
    psum = np.stack([rewards[:, 4, :h].sum(-1) for h in opportunity.PREFIXES], -1)
    reference_sum = np.stack([reference[:, :h].sum(-1) for h in opportunity.PREFIXES], -1)
    bins = np.array([0, .5, 1], np.float32)
    q = np.zeros((16, 4, 3), np.float32)
    q[:, :, 1] = np.where(positive, .3, -.3)[:, None]
    p = np.zeros_like(q)
    support = diagnostic.compatibility_numpy(q, p, np.full((16, 4), .5, np.float32), bins)
    D = np.full((16, 4), .01, np.float32)
    f = 1 - 1 / (1 + 20 * D)
    value = dict(consequence_prefixes=np.broadcast_to(opportunity.PREFIXES, (16, 4)),
        true_reward_sequences=rewards, reference_reward_sequence=reference,
        reference_full_window_verified=np.ones(16, bool), reference_prefix_sum=reference_sum,
        true_q_prefix_sum=qsum, true_p_prefix_sum=psum,
        true_q_prefix_mean=qsum / np.asarray(opportunity.PREFIXES),
        true_p_prefix_mean=psum / np.asarray(opportunity.PREFIXES),
        true_q_h1=rewards[:, :4, 0], true_p_h1=rewards[:, 4, 0],
        true_q_horizon=rewards[:, :4].mean(-1), true_p_horizon=rewards[:, 4].mean(-1),
        D=D, K=np.full((16, 4), 2., np.float32), m=np.ones((16, 4), np.float32), f_D=f,
        v=1 - .5 * support['e'] * f,
        physics_repeat_error=np.zeros(16),
        position=diagnostic.sample_positions(1001, maximum=16, horizon=100),
        raw_logits_q=q, raw_logits_p=p, bins=bins, next_reward=np.full(16, .5, np.float32),
        q_margin_min=np.full((16, 4), .02, np.float32),
        mode_perturb_e=support['e'], mode_perturb_C=.5 * support['e'] * f,
        fixed_e=np.repeat(support['e'][..., None], 5, -1),
        fixed_signed_support=np.repeat(support['signed_support'][..., None], 5, -1),
        **{k: support[k] for k in ('e', 'signed_support', 'loss_gap')},
        u_q=np.exp(-(support['ell_q'] - support['H_y'])),
        u_p=np.exp(-(support['ell_p'] - support['H_y'])))
    path = source / f'episode_{entry["episode"]:03}.npz'
    np.savez_compressed(path, **value)
    artifacts.append(dict(episode=entry['episode'], split=entry['split'],
        positions=16, path=path.name, sha256=diagnostic.sha256(path)))
  write(source / 'score_complete.json', dict(params_unchanged=True,
      plan_sha256=diagnostic.sha256(source / 'score_plan.json'), artifacts=artifacts))
  return source


def rewrite_artifact(source, key, change):
  complete = json.loads((source / 'score_complete.json').read_text())
  entry = complete['artifacts'][0]
  path = source / entry['path']
  with np.load(path, allow_pickle=False) as file:
    data = {k: file[k] for k in file.files}
  data[key] = change(data[key])
  np.savez_compressed(path, **data)
  entry['sha256'] = diagnostic.sha256(path)
  write(source / 'score_complete.json', complete)


class OpportunityTest(unittest.TestCase):

  def test_full_synthetic_artifact_cli_primary_axes_coverage_and_immutable_output(self):
    with tempfile.TemporaryDirectory() as tmp:
      source = synthetic_scores(Path(tmp) / 'scores')
      output = Path(tmp) / 'result.json'
      run = subprocess.run([sys.executable, str(ROOT / 'scripts/dl_opportunity.py'),
          '--scores', str(source), '--output', str(output)], cwd=ROOT,
          capture_output=True, text=True)
      self.assertEqual(run.returncode, 0, run.stdout[-2000:] + run.stderr[-2000:])
      result = json.loads(output.read_text())
      self.assertEqual(result['analysis']['primary_prefix'], 100)
      self.assertEqual(result['blind_positions'], 384)
      self.assertEqual(result['calibration_positions'], 128)
      self.assertEqual(result['coverage']['positive']['positions'], 192)
      self.assertEqual(result['coverage']['negative']['positions'], 192)
      self.assertEqual(result['coverage']['joint']['positions'], 192)
      self.assertEqual(result['opportunity_guard'], 'may_design_revision')
      self.assertFalse(result['proceed_short_training'])
      self.assertFalse(result['eligible_method_training'])
      self.assertEqual(result['per_prefix']['10']['primary'], False)
      self.assertEqual(result['per_prefix']['100']['primary'], True)
      background = result['per_prefix']['100']['q_variant_minus_p_sum']['background_a']
      self.assertAlmostEqual(background['mean_interval']['point'], .1)
      before = output.read_bytes()
      with self.assertRaises(FileExistsError):
        opportunity.analyze(source, output)
      self.assertEqual(before, output.read_bytes())

  def test_common_sequence_axes_are_checked_beyond_artifact_hash(self):
    with tempfile.TemporaryDirectory() as tmp:
      source = synthetic_scores(Path(tmp) / 'scores')
      rewrite_artifact(source, 'true_q_prefix_sum', lambda x: x.swapaxes(1, 2))
      with self.assertRaisesRegex(ValueError, 'common sequence or axes mismatch'):
        opportunity.load_scores(source)
    with tempfile.TemporaryDirectory() as tmp:
      source = synthetic_scores(Path(tmp) / 'scores')
      rewrite_artifact(source, 'reference_full_window_verified', lambda x: np.zeros_like(x))
      with self.assertRaisesRegex(ValueError, 'Full recorded-action reference'):
        opportunity.load_scores(source)

  def test_hash_and_checkpoint_source_mismatches_are_rejected(self):
    with tempfile.TemporaryDirectory() as tmp:
      source = synthetic_scores(Path(tmp) / 'scores')
      with (source / 'collection_plan.json').open('a') as stream:
        stream.write(' ')
      with self.assertRaisesRegex(ValueError, 'collection plan hash mismatch'):
        opportunity.load_scores(source)
    with tempfile.TemporaryDirectory() as tmp:
      source = synthetic_scores(Path(tmp) / 'scores')
      plan = json.loads((source / 'score_plan.json').read_text())
      plan['collection_checkpoint_sha256'] = list(opportunity.SOURCES)[1]
      write(source / 'score_plan.json', plan)
      complete = json.loads((source / 'score_complete.json').read_text())
      complete['plan_sha256'] = diagnostic.sha256(source / 'score_plan.json')
      write(source / 'score_complete.json', complete)
      with self.assertRaisesRegex(ValueError, 'checkpoint mismatch'):
        opportunity.load_scores(source)

  def test_sparse_joint_coverage_cannot_borrow_calibration_or_authorize_training(self):
    with tempfile.TemporaryDirectory() as tmp:
      source = synthetic_scores(Path(tmp) / 'scores', adequate=False)
      result = opportunity.analyze(source, Path(tmp) / 'result.json')
      self.assertEqual(result['coverage']['positive']['positions'], 2)
      self.assertEqual(result['coverage']['positive']['episodes'], 1)
      self.assertEqual(result['coverage']['joint']['positions'], 2)
      self.assertEqual(result['coverage']['useful_cue']['positions'], 384)
      self.assertEqual(result['opportunity_guard'], 'limited_opportunity')
      self.assertFalse(result['signal_label_coverage_sufficient'])
      self.assertFalse(result['target_opportunity_coverage_sufficient'])
      self.assertFalse(result['training_authorized'])

  def test_scoring_rng_seed_cannot_be_inferred_from_collection_seed(self):
    with tempfile.TemporaryDirectory() as tmp:
      source = synthetic_scores(Path(tmp) / 'scores')
      plan = json.loads((source / 'score_plan.json').read_text())
      plan['diagnostic_seed'] = 20261010
      write(source / 'score_plan.json', plan)
      complete = json.loads((source / 'score_complete.json').read_text())
      complete['plan_sha256'] = diagnostic.sha256(source / 'score_plan.json')
      write(source / 'score_complete.json', complete)
      with self.assertRaisesRegex(ValueError, 'diagnostic RNG seed differs'):
        opportunity.load_scores(source)

  def test_rehashed_ordered_subset_cannot_replace_preselected_source_positions(self):
    with tempfile.TemporaryDirectory() as tmp:
      source = synthetic_scores(Path(tmp) / 'scores')
      def change(position):
        position = position.copy()
        position[4] += 1  # Still sorted, distinct and legal, but not prespecified.
        return position
      rewrite_artifact(source, 'position', change)
      with self.assertRaisesRegex(ValueError, 'frozen source-frame selection'):
        opportunity.load_scores(source)


if __name__ == '__main__':
  unittest.main()
