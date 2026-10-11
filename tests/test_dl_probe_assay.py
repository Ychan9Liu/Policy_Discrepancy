"""Pure synthetic construction and authority guards; never a probe fit.

These tests do not call _readouts, analyze, ridge_predict, the assay worker, a
model or simulator. No scientific artifacts or real calibration arrays load.
Known decoders test only the deliberately supplied synthetic feature recipe.
"""
import copy
import hashlib
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MUJOCO_GL'] = 'disable'
os.environ['JAX_PLATFORMS'] = 'cpu'
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS'):
  os.environ[variable] = '1'

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import dl_probe_assay as assay


def clone(arrays):
  return {key: value.copy() for key, value in arrays.items()}


def hashes(arrays):
  result = {}
  for name, value in arrays.items():
    array = np.ascontiguousarray(value)
    result[name] = (str(array.dtype), array.shape,
        hashlib.sha256(array.tobytes()).hexdigest())
  return result


class ProbeAssayConstructionTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.inputs = assay.generate_inputs()

  def setUp(self):
    # Construction must remain independent of fitting or synthetic decisions.
    for name in ('analyze', 'ridge_predict', '_readouts', '_external', '_interval'):
      patcher = mock.patch.object(assay.stats, name,
          side_effect=AssertionError('Construction test attempted statistical analysis'))
      patcher.start()
      self.addCleanup(patcher.stop)

  def test_fixed_shapes_labels_rewards_and_onehot_contract(self):
    self.assertEqual(tuple(assay.CASES), ('sparse', 'replicated', 'history', 'identity'))
    self.assertEqual(assay.SEED, 20261022)
    self.assertEqual(set(self.inputs), set(assay.CASES))
    labels = (np.arange(4)[None] == np.arange(8)[:, None] % 4).astype(np.uint8)
    expected_rewards = np.zeros((8, 4, 7), np.float64)
    expected_rewards[..., 2] = 1 - labels
    expected_rewards[..., 3] = labels
    shapes = {'classes': ((8, 4), np.dtype(np.uint8)),
        'rewards': ((8, 4, 7), np.dtype(np.float64)),
        'h': ((8, 4, 4096), np.dtype(np.float64)),
        'latent': ((8, 4, 4, 2, 32, 32), np.dtype(np.float64)),
        'episodes': ((8,), np.dtype(np.int32))}
    for case, arrays in self.inputs.items():
      with self.subTest(case=case):
        self.assertEqual(set(arrays), set(shapes))
        for key, (shape, dtype) in shapes.items():
          self.assertIsInstance(arrays[key], np.ndarray)
          self.assertEqual(arrays[key].shape, shape)
          self.assertEqual(arrays[key].dtype, dtype)
          self.assertTrue(np.isfinite(arrays[key]).all())
        np.testing.assert_array_equal(arrays['classes'], labels)
        np.testing.assert_array_equal(arrays['episodes'], np.arange(8, dtype=np.int32))
        np.testing.assert_array_equal(arrays['rewards'], expected_rewards)
        z = arrays['latent']
        self.assertTrue(np.all((z == 0) | (z == 1)))
        np.testing.assert_array_equal(z.sum(-1), np.ones((8, 4, 4, 2, 32)))

  def test_q_p_share_nuisance_and_only_named_cue_changes(self):
    base = self.inputs['identity']['latent'][..., 1, :, :]
    labels = self.inputs['sparse']['classes']
    onehot = np.eye(32, dtype=np.float64)[labels]
    for case in assay.CASES:
      np.testing.assert_array_equal(self.inputs[case]['latent'][..., 1, :, :], base)
    sparse = self.inputs['sparse']['latent']
    np.testing.assert_array_equal(sparse[..., 0, 1:, :], base[..., 1:, :])
    np.testing.assert_array_equal(sparse[..., 0, 0, :],
        np.broadcast_to(onehot[:, :, None, :], (8, 4, 4, 32)))
    replicated = self.inputs['replicated']['latent']
    np.testing.assert_array_equal(replicated[..., 0, :, :],
        np.broadcast_to(onehot[:, :, None, None, :], (8, 4, 4, 32, 32)))
    # The intervention changes only its named factors. This does not claim a
    # fitted readout can overcome normalization/intercept/regularization.
    changed = sparse[..., 0, :, :] != sparse[..., 1, :, :]
    self.assertFalse(changed[..., 1:, :].any())
    for case in ('sparse', 'replicated', 'identity'):
      np.testing.assert_array_equal(self.inputs[case]['h'], self.inputs['identity']['h'])

  def test_history_and_identity_exact_shared_branches(self):
    for case in ('history', 'identity'):
      z = self.inputs[case]['latent']
      np.testing.assert_array_equal(z[..., 0, :, :], z[..., 1, :, :])
      self.assertEqual(z[..., 0, :, :].tobytes(), z[..., 1, :, :].tobytes())
    labels = self.inputs['history']['classes'].astype(np.float64)
    expected = np.broadcast_to((2 * labels - 1)[..., None], (8, 4, 4096))
    np.testing.assert_array_equal(self.inputs['history']['h'], expected)

  def test_known_feature_decoders_exact_without_readout(self):
    for case, branches in (('sparse', (0,)), ('replicated', (0,)),
                           ('history', (0, 1))):
      arrays = self.inputs[case]
      expected = np.broadcast_to(2 + arrays['classes'][..., None], (8, 4, 4)).astype(np.int32)
      for branch in branches:
        with self.subTest(case=case, branch=branch):
          decoded = assay.feature_decoder(case, arrays, branch)
          self.assertIsInstance(decoded, np.ndarray)
          self.assertEqual(decoded.dtype, np.int32)
          np.testing.assert_array_equal(decoded, expected)
          # Decoder returns a command index, not a latent class. Reward is
          # used only here to check the independently known construction.
          utility = np.broadcast_to(arrays['rewards'][:, :, None, :], (8, 4, 4, 7))
          chosen = np.take_along_axis(utility, decoded[..., None], -1)[..., 0]
          np.testing.assert_array_equal(chosen, np.ones((8, 4, 4), np.float64))
          features_only = {'h': arrays['h'], 'latent': arrays['latent']}
          np.testing.assert_array_equal(
              assay.feature_decoder(case, features_only, branch), expected)
    for case in ('sparse', 'replicated'):
      self.assertIsNone(assay.feature_decoder(case, self.inputs[case], 1))
    for branch in (0, 1):
      self.assertIsNone(assay.feature_decoder('identity', self.inputs['identity'], branch))

  def test_generation_independent_of_legacy_global_numpy_rng(self):
    legacy_state = np.random.get_state()
    try:
      np.random.seed(123)
      np.random.random(19)
      first = assay.generate_inputs()
      np.random.seed(901)
      np.random.random(101)
      second = assay.generate_inputs()
    finally:
      np.random.set_state(legacy_state)
    for case in assay.CASES:
      self.assertEqual(hashes(first[case]), hashes(second[case]))
      self.assertEqual(hashes(first[case]), hashes(self.inputs[case]))

  def test_validate_and_decoder_are_readonly(self):
    for case, original in self.inputs.items():
      arrays = clone(original)
      before = hashes(arrays)
      audit = assay.validate_input(case, arrays)
      self.assertIsInstance(audit, dict)
      for branch in (0, 1):
        assay.feature_decoder(case, arrays, branch)
      self.assertEqual(hashes(arrays), before)

  def test_reject_shape_dtype_nonfinite_and_episode_label_mismatch(self):
    base = self.inputs['sparse']
    for key in ('classes', 'rewards', 'h', 'latent', 'episodes'):
      with self.subTest(kind='dtype', key=key):
        arrays = clone(base)
        dtype = np.float32 if key in ('rewards', 'h', 'latent') else np.int64
        arrays[key] = arrays[key].astype(dtype)
        with self.assertRaises(ValueError):
          assay.validate_input('sparse', arrays)
      with self.subTest(kind='shape', key=key):
        arrays = clone(base)
        arrays[key] = arrays[key][:-1]
        with self.assertRaises(ValueError):
          assay.validate_input('sparse', arrays)
    for key in ('rewards', 'h', 'latent'):
      for value in (np.nan, np.inf, -np.inf):
        with self.subTest(kind='nonfinite', key=key, value=value):
          arrays = clone(base)
          arrays[key].flat[0] = value
          with self.assertRaises(ValueError):
            assay.validate_input('sparse', arrays)
    for key in ('classes', 'episodes'):
      with self.subTest(kind='label_alignment', key=key):
        arrays = clone(base)
        arrays[key] = np.roll(arrays[key], 1, axis=0)
        with self.assertRaises(ValueError):
          assay.validate_input('sparse', arrays)
    with self.assertRaises(ValueError):
      assay.validate_input('unknown_real_dataset', clone(base))

  def test_reject_non_onehot_even_with_unit_sum_and_reward_or_cue_tamper(self):
    base = self.inputs['sparse']
    malformed = []
    for values in ((0., 0.), (.5, .5), (1., 1.)):
      arrays = clone(base)
      arrays['latent'][0, 0, 0, 0, 0] = 0
      arrays['latent'][0, 0, 0, 0, 0, :2] = values
      malformed.append(arrays)
    reward = clone(base)
    reward['rewards'][0, 0, 2] = np.nextafter(reward['rewards'][0, 0, 2], np.inf)
    malformed.append(reward)
    wrong_cue = clone(base)
    wrong_cue['latent'][0, 0, 0, 0, 0] = np.roll(wrong_cue['latent'][0, 0, 0, 0, 0], 1)
    malformed.append(wrong_cue)
    for index, arrays in enumerate(malformed):
      with self.subTest(index=index):
        with self.assertRaises(ValueError):
          assay.validate_input('sparse', arrays)


class ProbeAssayBoundaryTest(unittest.TestCase):
  def fixture(self):
    boundary = dict(assay.BOUNDARY)
    expected = dict(synthetic_only=True, source_provenance_verified=False,
        actor_applicable=False, source_provenance_applicable=False,
        method_evidence=False, proceed_short_training=False,
        decision='stop_after_fixed_synthetic_assay',
        real_scientific_decision='not_evaluated', real_gate_authority=False,
        eligible_method_training=False, training_authorized=False,
        method_effectiveness_established=False, real_dataset_reads=0,
        blind_arrays_read=False, model_calls=0, physics_steps=0,
        optimizer_updates=0, gpu_devices_used=0,
        current_rgb_increment_over_complete_history_established=False,
        continuing_kl_compression_effect_established=False)
    self.assertEqual(boundary, expected)
    return boundary

  def test_minimal_synthetic_boundary_passes_without_mutation(self):
    result = self.fixture()
    before = copy.deepcopy(result)
    self.assertIsNone(assay.ensure_synthetic_boundary(result))
    self.assertEqual(result, before)
    result['decision'] = 'stop_after_fixed_synthetic_assay'
    self.assertIsNone(assay.ensure_synthetic_boundary(result))

  def test_every_authority_or_call_counter_tamper_and_missing_key_rejected(self):
    base = self.fixture()
    for key, expected in base.items():
      with self.subTest(key=key, kind='missing'):
        result = copy.deepcopy(base)
        del result[key]
        with self.assertRaises(ValueError):
          assay.ensure_synthetic_boundary(result)
      if type(expected) is bool:
        incorrect = (not expected, int(expected), str(expected))
      elif type(expected) is int:
        incorrect = (1, -1, True, '0', 0.)
      else:
        incorrect = ('interface_increment_exploratory_supported', True, None)
      for bad in incorrect:
        with self.subTest(key=key, bad=bad):
          result = copy.deepcopy(base)
          result[key] = bad
          with self.assertRaises(ValueError):
            assay.ensure_synthetic_boundary(result)

  def test_inherited_may_design_decision_never_authorizes_real_progress(self):
    for bad in ('may_design_independent_signal_and_causal_cue_validation',
                'may_design_finite_normal_loss_compression_mechanism',
                'do_not_advance_gate_or_training', 'not_evaluated', None, False):
      with self.subTest(decision=bad):
        result = self.fixture()
        result['decision'] = bad
        with self.assertRaises(ValueError):
          assay.ensure_synthetic_boundary(result)


if __name__ == '__main__':
  unittest.main()
