"""Synthetic CPU audit fixtures only; no actual B/source/model verification."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('independent_reward_audit', ROOT / 'analysis/dl_production_reward_audit.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class ProductionRewardAuditTest(unittest.TestCase):
  def test_offdiagonal_pairs_bias_correction_and_signed_negative(self):
    rng = np.random.default_rng(17)
    paths = rng.normal(size=(3, 5, 7)).astype(np.float64)
    target = rng.normal(size=(3, 7)).astype(np.float64)
    value = m.moments(paths, target)
    residual = (paths - paths.mean(-1, keepdims=True)) - (target - target.mean(-1, keepdims=True))[:, None]
    brute = np.stack([sum(residual[k, i] * residual[k, j]
        for i in range(5) for j in range(5) if i != j) / 20 for k in range(3)]).mean(-1)
    np.testing.assert_allclose(value['bc_mse'], brute, atol=1e-15, rtol=0.)
    np.testing.assert_allclose(value['bc_mse'], value['plugin_centered_mse'] - value['mc_correction'], atol=1e-15, rtol=0.)
    signed = m.moments(np.array([[1., 0.], [0., 1.]]), np.zeros(2))
    self.assertEqual(signed['bc_mse'], -.25)
    self.assertEqual(signed['plugin_centered_mse'], 0.)
    self.assertEqual(signed['mc_correction'], .25)
    for bad in (paths.astype(np.float32), paths[:, :1], np.full_like(paths, np.nan)):
      with self.assertRaises(ValueError): m.moments(bad, target)

  def test_perfect_translation_and_loss_before_initial_averaging(self):
    target = np.array([1., .2, -.3, .4, 0., 0., 0.], np.float64)
    result = m.moments(np.tile(target, (128, 1)), target)
    self.assertEqual(result['bc_mse'], 0.)
    self.assertEqual(result['plugin_centered_mse'], 0.)
    paths = np.array([[[1., -1.], [1., -1.]], [[-1., 1.], [-1., 1.]]])
    actual = m.moments(paths, np.zeros((2, 2)))['bc_mse'].mean()
    wrong = m.moments(paths.mean(0), np.zeros(2))['bc_mse']
    self.assertEqual(actual, 1.); self.assertEqual(wrong, 0.)

  def test_global_closed_tie_set_not_chained_pairwise(self):
    self.assertEqual(int(m.choose(np.array([0., 1e-10]))), 0)
    self.assertEqual(int(m.choose(np.array([0., np.nextafter(1e-10, np.inf)]))), 1)
    self.assertEqual(int(m.choose(np.array([0., .75e-10, 1.5e-10]))), 1)
    with self.assertRaises(ValueError): m.choose(np.array([np.nan, 1.]))

  def test_loeo_constant_never_uses_heldout_rewards(self):
    truth = np.zeros((8, 4, 7)); truth[0, :, 6] = 20.; truth[1:, :, 2] = 1.
    value, mse, selected = m.constant(truth, np.asarray(m.EPISODES, np.int32))
    self.assertEqual(value['control_by_episode'][0], 2)
    changed = truth.copy(); changed[0] *= 3.
    again, _, _ = m.constant(changed, np.asarray(m.EPISODES, np.int32))
    self.assertEqual(value['folds'][0], again['folds'][0])
    self.assertNotEqual(value['folds'][1], again['folds'][1])
    np.testing.assert_array_equal(selected[0], np.zeros(4))
    self.assertEqual(value['folds'][0]['training_states'], 28)

  def test_bootstrap_whole_episode_and_four_initial_not_independent_units(self):
    episodes = np.array([-1., -.5, 0., .125, .25, .375, .5, 3.], np.float64)
    values = np.broadcast_to(episodes[:, None, None], (8, 4, 4)).copy()
    draws = np.random.default_rng(20261025).integers(0, 8, (2000, 8))
    interval = m.gap(values, draws)
    expected = np.quantile(episodes[draws].mean(1), [.025, .975], method='linear')
    self.assertEqual(interval['episode_values'], episodes.tolist())
    np.testing.assert_allclose([interval['lower95'], interval['upper95']], expected, atol=1e-15, rtol=0.)
    duplicate = m.gap(np.repeat(values, 3, axis=2), draws)
    for key in ('state_values', 'episode_values', 'primary', 'lower95', 'upper95'):
      self.assertEqual(interval[key], duplicate[key])

  def test_reliability_priority_relative_margin_and_zero_boundary(self):
    subsets = {name: {'mse_gaps': {'a': dict(primary=.01, lower95=.009, upper95=.012),
                                  'b': dict(primary=.02, lower95=.019, upper95=.022)}} for name in m.SUBSETS}
    value = m.reliability(subsets, 'mse_gaps', .0004, 'mean')
    self.assertEqual(value['status'], 'finite_native_mean_exploratory_supported')
    changed = copy.deepcopy(subsets); changed['block0']['mse_gaps']['a']['primary'] += .001
    bad = m.reliability(changed, 'mse_gaps', .0004, 'mean')
    self.assertEqual(bad['status'], 'MC-inconclusive')
    self.assertTrue(bad['positive_margin_conditions_met']); self.assertFalse(bad['necessary_directions_robust'])
    for item in subsets.values(): item['mse_gaps']['a'] = dict(primary=0., lower95=0., upper95=0.)
    stopped = m.reliability(subsets, 'mse_gaps', .0004, 'mean')
    self.assertEqual(stopped['status'], 'finite_native_mean_not_supported')

  def test_mse_skill_does_not_imply_choice_utility(self):
    truth = np.array([1., .99, 0.])
    q = np.tile(np.array([.98, 1., 0.]), (4, 1))
    p = np.tile(np.array([1., .98, -1.]), (4, 1))
    self.assertLess(m.moments(q, truth)['bc_mse'], m.moments(p, truth)['bc_mse'])
    self.assertLess(truth[int(m.choose(q.mean(0)))], truth[int(m.choose(p.mean(0)))])

  def test_fulltree_shape_types_exact_classification_and_ci_zero_sides(self):
    a = {'primary': 0., 'lower95': 0., 'upper95': 1., 'choices': [1, 2], 'flag': False,
         'bc_mse': {'per_initial': [-1e-12, 1e-12]}, 'negative_bc_mse_estimates': 1}
    m.compare_tree(a, copy.deepcopy(a))
    close = copy.deepcopy(a); close['bc_mse']['per_initial'] = [1e-12, -1e-12]
    m.compare_tree(a, close)  # Frozen abs tolerance + aggregate signed count, no extra pointwise sign gate.
    for key, value in (('primary', 1e-12), ('lower95', -1e-12), ('choices', [1, 3]), ('flag', 0),
                       ('negative_bc_mse_estimates', 2)):
      bad = copy.deepcopy(a); bad[key] = value
      with self.assertRaises(ValueError): m.compare_tree(a, bad)
    with self.assertRaises(ValueError): m.compare_tree(a, dict(a, extra=True))

  def test_complete_numeric_report_axes_and_fixed_stop_without_model(self):
    # A broadcast numeric fixture, not a model/source/GPU or scientific run.
    prediction = np.broadcast_to(np.zeros((), np.float32), m.SHAPE)
    truth = np.zeros((8, 4, 7), np.float64)
    result = m.recompute(prediction, truth, np.asarray(m.EPISODES, np.int32))
    self.assertEqual(result['primary_status'], 'finite_native_mean_not_supported')
    self.assertEqual(result['utility_reliability']['status'], 'finite_native_choice_not_supported')
    self.assertEqual(result['constant']['control_by_episode'], [0] * 8)
    self.assertEqual(result['subsets']['last64']['path_indices'], list(range(64, 128)))
    for subset in result['subsets'].values():
      for name in ('q', 'p'):
        self.assertEqual(subset['interfaces'][name]['negative_bc_mse_estimates'], 0)
        self.assertEqual(subset['interfaces'][name]['bc_mse']['primary'], 0.)
        self.assertEqual(subset['interfaces'][name]['choices'], np.zeros((8, 4, 4), np.int32).tolist())
    comparison = m.compare_tree(result, copy.deepcopy(result))
    self.assertGreater(comparison['compared_numeric_leaves'], 50000)
    self.assertGreater(comparison['zero_side_checks'], 50)
    self.assertFalse(result['training_authorized'])

  def test_synthetic_full_tree_wiring_to_producer_zero_identity_and_fixed_random(self):
    # Only this integration fixture imports the producer's numeric stats.
    # The independent audit implementation never imports any producer module.
    sys.path.insert(0, str(ROOT))
    from scripts import dl_production_reward_stats as producer
    identities = np.asarray(m.EPISODES, np.int32)
    zero = np.zeros(m.SHAPE, np.float32)
    zero_truth = np.zeros((8, 4, 7), np.float64)
    rng = np.random.default_rng(20261026)  # One fixed engineering draw; never tuned or real data.
    random = rng.normal(size=m.SHAPE).astype(np.float32)
    truth = rng.normal(size=(8, 4, 7)).astype(np.float64)
    for prediction, target in ((zero, zero_truth), (random, truth)):
      independent = m.recompute(prediction, target, identities)
      published = producer.report(prediction, target, identities)
      forward = m.compare_tree(independent, published)
      backward = m.compare_tree(published, independent)
      self.assertGreater(forward['compared_numeric_leaves'], 50000)
      self.assertLessEqual(forward['max_abs_error'], 1e-10)
      self.assertEqual(forward['max_abs_error'], backward['max_abs_error'])
      self.assertEqual(independent['primary_status'], published['primary_status'])
      self.assertFalse(independent['training_authorized'])

  def test_scalar_hash_preserves_shape_and_source_whitelist_avoids_object(self):
    scalar = np.asarray(9, np.int32)
    self.assertEqual(m.raw_digest(scalar, True)['shape'], [])
    self.assertEqual(m.raw_digest(scalar, True)['dtype_name'], 'int32')
    with tempfile.TemporaryDirectory() as td:
      path = Path(td) / 'synthetic.npz'
      h = np.zeros((4, 4096), np.float64)
      latent = np.zeros((4, 4, 2, 32, 32), np.float64); latent[..., 0] = 1.
      controls = np.zeros((4, 15, 12), np.float32); controls[:, :7] = m.commands()
      fields = dict(episode=scalar, positions=np.array(m.POSITIONS, np.int32), h=h,
                    latent=latent, reward_f64=np.ones((4, 15, 20)), first_commands=controls,
                    opaque_never_read=np.array(['would require pickle'], object))
      np.savez(path, **fields)
      truth, library, proof = m.load_source_numeric(path, 9)
      self.assertEqual(proof['episode']['shape'], [])
      np.testing.assert_array_equal(truth, np.full((4, 7), 20.))
      self.assertEqual(library.tobytes(), np.broadcast_to(m.commands(), (4, 7, 12)).tobytes())
      fields['h'][0, 0] = .1; np.savez(path, **fields)
      with self.assertRaisesRegex(ValueError, 'BF16'): m.load_source_numeric(path, 9)

  def test_source_hash_and_path_rejection_before_numeric_decode(self):
    with tempfile.TemporaryDirectory() as td:
      directory = Path(td).resolve()
      (directory / 'plan.json').write_text('{}', encoding='utf-8')
      with self.assertRaisesRegex(ValueError, 'Unapproved source JSON'): m.source_authority(directory)
      with self.assertRaisesRegex(ValueError, 'Unsafe'): m.safe_file(directory, '../plan.json')
      (directory / 'duplicate.json').write_text('{"a":1,"a":2}', encoding='utf-8')
      with self.assertRaisesRegex(ValueError, 'Duplicate'): m.read_json(directory / 'duplicate.json')

  def test_symmetric_native_point_formula_and_fixed_threshold(self):
    bins = np.arange(-127, 128, dtype=np.float32)
    logits = np.full((2, 255), -30., np.float32); logits[0, 127] = 0.; logits[1, 128] = 0.
    mean = m.symmetric_mean(logits, bins)
    self.assertEqual(mean[0], 0.)
    self.assertAlmostEqual(mean[1], 1., places=9)
    with self.assertRaises(ValueError): m.symmetric_mean(logits.astype(np.float64), bins)
    with self.assertRaises(ValueError): m.symmetric_mean(logits, bins[::-1].copy())

  def test_first_chunk_saved_prefix_axes_and_count_receipt(self):
    with tempfile.TemporaryDirectory() as td:
      directory = Path(td).resolve()
      data = dict(reward=np.zeros((4, 2, 8, 7, 20), np.float32),
          logits=np.zeros((4, 2, 8, 7, 20, 255), np.float32),
          bins=np.arange(-127, 128, dtype=np.float32),
          root_keys=np.zeros((4, 8, 20, 2), np.uint32),
          sample_keys=np.ones((4, 8, 20, 2), np.uint32))
      path = directory / 'first_chunk_oracle.npz'
      np.savez_compressed(path, **data)
      receipt = dict(path=path.name, sha256=m.sha(path), max_abs_error=0., atol=1e-6, rtol=1e-5,
          finite_point_oracle_passed=True, full_model_independently_reproduced=False,
          compared_points=8960, failed_points=0)
      (directory / 'first_chunk_oracle_check.json').write_text(json.dumps(receipt), encoding='utf-8')
      first = dict(reward=np.zeros((4, 4, 2, 128, 7, 20), np.float32),
          bins=data['bins'].copy(), root_keys=np.zeros((4, 4, 128, 20, 2), np.uint32),
          sample_keys=np.ones((4, 4, 128, 20, 2), np.uint32))
      actual = m.oracle_check(directory, receipt, first)
      self.assertEqual(actual['compared_points'], 8960)
      first['reward'][0, 0, 0, 0, 0, 0] = 1.
      with self.assertRaisesRegex(ValueError, 'predictions'): m.oracle_check(directory, receipt, first)
      first['reward'][0, 0, 0, 0, 0, 0] = 0.
      bad = dict(receipt, compared_points=8961)
      with self.assertRaisesRegex(ValueError, 'Exact statistic'): m.oracle_check(directory, bad, first)

  def test_cpu_private_key_domain_with_no_producer_model_import(self):
    code = '''import importlib.util,sys,numpy as np
import jax
s=importlib.util.spec_from_file_location("audit", "analysis/dl_production_reward_audit.py")
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
roots,samples=m.independent_keys(9,1)
key=jax.random.fold_in(jax.random.PRNGKey(20261024),np.uint32(0x50525231))
for label in (9,1,0,0,0,1):key=jax.random.fold_in(key,np.uint32(label))
assert roots.shape==(4,128,20,2)
assert roots[0,0,0].tobytes()==np.asarray(key).tobytes()
assert samples[0,0,0].tobytes()==np.asarray(jax.random.split(key,16)[1]).tobytes()
assert roots[0,0,0].tobytes()!=roots[0,32,0].tobytes()
assert roots[0,0,0].tobytes()!=roots[0,0,1].tobytes()
assert not any(k.startswith(("scripts.dl_production", "dreamerv3", "dm_control", "ninjax")) for k in sys.modules)
print("CPU key domain synthetic fixture passed")'''
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=30,
                            env=dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES=''))
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_cli_small_selftest_is_separate_scope_and_no_overwrite(self):
    with tempfile.TemporaryDirectory() as td:
      path = Path(td) / 'audit.json'
      command = [sys.executable, str(ROOT / 'analysis/dl_production_reward_audit.py'), '--selftest', '--output', str(path)]
      result = subprocess.run(command, capture_output=True, text=True, timeout=30)
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      value = json.loads(path.read_text(encoding='utf-8'))
      self.assertEqual(value['status'], 'synthetic_selftest_passed')
      self.assertTrue(value['synthetic_fixture']); self.assertEqual(value['actual_numeric_episode_reads'], 0)
      self.assertFalse(value['training_authorized'])
      repeat = subprocess.run(command, capture_output=True, text=True, timeout=30)
      self.assertNotEqual(repeat.returncode, 0)


if __name__ == '__main__':
  unittest.main()
