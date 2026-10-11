"""Synthetic numeric/source authority guards; no real arrays or probe fitting.

The NPZ fixtures match production dimensions but contain constructed numbers.
An opaque object envelope would execute a marker payload if unpickled. These
tests prove the loader never requests that envelope and always disables pickle.
No worker, statistical readout, model, simulator or source dataset is called.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS'):
  os.environ[variable] = '1'

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_nested_probe as loader


def poison_payload(marker):
  """Would run only on deserialization; never invoked by this test suite."""
  Path(marker).write_text('opaque records were decoded', encoding='utf-8')
  raise AssertionError('Opaque records must never be unpickled')


class PoisonRecord:
  def __init__(self, marker):
    self.marker = str(marker)

  def __reduce__(self):
    return poison_payload, (self.marker,)


def numeric_fields(episode=loader.EPISODES[0]):
  h = np.zeros((4, 4096), np.float64)
  h[:, :4] = np.eye(4)
  latent = np.zeros((4, 4, 2, 32, 32), np.float64)
  latent[..., 0] = 1.
  reward = np.arange(4 * 15 * 20, dtype=np.float64).reshape(4, 15, 20) / 100.
  return dict(episode=np.array(episode, np.int32),
      positions=np.array(loader.POSITIONS, np.int32), h=h, latent=latent,
      reward_f64=reward)


def numeric_npz(path, fields, marker):
  np.savez_compressed(path, **fields,
      records=np.array([PoisonRecord(marker)], dtype=object),
      unrelated_object=np.array([PoisonRecord(marker)], dtype=object))


class NumericLoaderTest(unittest.TestCase):
  def setUp(self):
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    self.directory = Path(directory.name)
    self.path = self.directory / 'episode_009.npz'
    self.marker = self.directory / 'poison-executed.txt'
    self.fields = numeric_fields()

  def load(self, fields=None):
    numeric_npz(self.path, self.fields if fields is None else fields, self.marker)
    return loader.load_numeric_episode(self.path, loader.EPISODES[0])

  def test_production_dimensions_lazy_whitelist_pickle_disabled(self):
    numeric_npz(self.path, self.fields, self.marker)
    original_load = np.load
    original_getitem = np.lib.npyio.NpzFile.__getitem__
    requested = []

    def numeric_only(archive, name):
      requested.append(name)
      if name not in ('episode', 'positions', 'h', 'latent', 'reward_f64'):
        raise AssertionError('Non-whitelisted NPZ member requested: ' + name)
      return original_getitem(archive, name)

    with (mock.patch.object(loader.np, 'load', wraps=original_load) as load_spy,
          mock.patch.object(np.lib.npyio.NpzFile, '__getitem__', numeric_only)):
      h, latent, cost = loader.load_numeric_episode(self.path, loader.EPISODES[0])
    self.assertEqual(load_spy.call_count, 1)
    self.assertEqual(load_spy.call_args.kwargs, {'allow_pickle': False})
    self.assertEqual(requested, ['episode', 'positions', 'h', 'latent', 'reward_f64'])
    self.assertEqual(h.dtype, np.float64)
    self.assertEqual(h.shape, (4, 4096))
    self.assertEqual(latent.dtype, np.float64)
    self.assertEqual(latent.shape, (4, 4, 2, 32, 32))
    np.testing.assert_array_equal(h, self.fields['h'])
    np.testing.assert_array_equal(latent, self.fields['latent'])
    expected = np.sum(self.fields['reward_f64'][:, :7], axis=-1, dtype=np.float64)
    self.assertEqual(cost.shape, (4, 7))
    self.assertEqual(cost.tobytes(), expected.tobytes())
    self.assertFalse(self.marker.exists())
    # Even an explicit attempted object read with this flag cannot execute it.
    with original_load(self.path, allow_pickle=False) as archive:
      self.assertIn('records', archive.files)
      with self.assertRaisesRegex(ValueError, 'Object arrays cannot be loaded'):
        archive['records']
    self.assertFalse(self.marker.exists())

  def test_required_fields_and_episode_positions_contract(self):
    cases = []
    missing = copy.deepcopy(self.fields)
    del missing['h']
    cases.append(('missing', missing))
    for name, value in (
        ('episode', np.array(loader.EPISODES[1], np.int32)),
        ('episode', np.array(loader.EPISODES[0], np.int64)),
        ('episode', np.array([loader.EPISODES[0]], np.int32)),
        ('positions', np.array((1, 300, 600, 901), np.int32)),
        ('positions', np.array(loader.POSITIONS, np.int64)),
        ('positions', np.array(loader.POSITIONS, np.int32).reshape(2, 2))):
      fields = copy.deepcopy(self.fields)
      fields[name] = value
      cases.append((name + ':' + str(value.dtype) + str(value.shape), fields))
    for label, fields in cases:
      with self.subTest(case=label):
        with self.assertRaises(ValueError):
          self.load(fields)
        self.assertFalse(self.marker.exists())

  def test_h_latent_reward_reject_shape_dtype_and_nonfinite(self):
    for name in ('h', 'latent', 'reward_f64'):
      variants = [self.fields[name][:-1], self.fields[name].astype(np.float32)]
      for invalid in (np.nan, np.inf, -np.inf):
        value = self.fields[name].copy()
        value.flat[0] = invalid
        variants.append(value)
      for number, value in enumerate(variants):
        with self.subTest(field=name, variant=number):
          fields = copy.deepcopy(self.fields)
          fields[name] = value
          with self.assertRaisesRegex(ValueError, 'Invalid finite F64 numeric field'):
            self.load(fields)
          self.assertFalse(self.marker.exists())

  def test_actual_latent_reject_fractional_empty_and_twohot(self):
    for kind in ('fractional', 'empty', 'twohot', 'negative'):
      with self.subTest(kind=kind):
        fields = copy.deepcopy(self.fields)
        row = fields['latent'][0, 0, 0, 0]
        if kind == 'fractional':
          row[:2] = .5
        elif kind == 'empty':
          row[:] = 0.
        elif kind == 'twohot':
          row[1] = 1.
        else:
          row[1] = -1.
        with self.assertRaisesRegex(ValueError, 'exactly onehot'):
          self.load(fields)
        self.assertFalse(self.marker.exists())

  def test_finite_reward_sum_overflow_rejected(self):
    fields = copy.deepcopy(self.fields)
    fields['reward_f64'][:] = np.finfo(np.float64).max
    self.assertTrue(np.isfinite(fields['reward_f64']).all())
    with np.errstate(over='ignore', invalid='ignore'):
      with self.assertRaisesRegex(ValueError, 'Cost sum overflow'):
        self.load(fields)
    self.assertFalse(self.marker.exists())


class SourceAuthorityTest(unittest.TestCase):
  def setUp(self):
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    self.directory = Path(directory.name)
    self.marker = self.directory / 'poison-executed.txt'
    identity = dict(actual_git_sha=loader.SOURCE_GIT, version='DL-action-information-r1')
    artifacts = []
    for episode in loader.EPISODES:
      path = self.directory / f'episode_{episode:03}.npz'
      numeric_npz(path, numeric_fields(episode), self.marker)
      artifacts.append(dict(episode=episode, path=path.name, sha256=loader.sha(path)))
    self.documents = {
        'plan.json': dict(identity, episodes=list(loader.EPISODES),
            positions=list(loader.POSITIONS), horizon=20, delta=.2),
        'result.json': dict(identity, evidence_artifacts=artifacts),
        'action_information_complete.json': dict(identity, complete_episodes=8,
            complete_states=32),
        'budget.json': dict(seconds=120, fixture_only=True),
        'supervisor_complete.json': dict(identity, exit_code=0, timed_out=False,
            worker_terminal=True),
        'independent-numeric-audit-v1.json': dict(status='passed',
            expected_git_sha=loader.SOURCE_GIT,
            source_h_sampled_z_and_actor_raw_byte_exact=True, blind_arrays_read=False)}

  def publish(self, refresh_chain=True):
    def save(name):
      path = self.directory / name
      path.write_text(json.dumps(self.documents[name], allow_nan=False), encoding='utf-8')
      return hashlib.sha256(path.read_bytes()).hexdigest()
    pins = {name: save(name) for name in ('plan.json', 'result.json', 'budget.json')}
    if refresh_chain:
      self.documents['action_information_complete.json'].update(
          plan_sha256=pins['plan.json'], result_sha256=pins['result.json'])
      self.documents['supervisor_complete.json']['budget_sha256'] = pins['budget.json']
    for name in ('action_information_complete.json', 'supervisor_complete.json',
                 'independent-numeric-audit-v1.json'):
      pins[name] = save(name)
    return pins

  def authorize(self, pins):
    with mock.patch.object(loader, 'SOURCE_PINS', pins):
      return loader.authority(self.directory)

  def test_valid_full_hash_chain_then_only_numeric_loads(self):
    pins = self.publish()
    plan, result, items = self.authorize(pins)
    self.assertEqual(plan, self.documents['plan.json'])
    self.assertEqual(result, self.documents['result.json'])
    self.assertEqual([item['episode'] for item in items], list(loader.EPISODES))
    for item in items:
      h, z, cost = loader.load_numeric_episode(
          self.directory / item['path'], item['episode'])
      self.assertEqual((h.shape, z.shape, cost.shape),
          ((4, 4096), (4, 4, 2, 32, 32), (4, 7)))
    self.assertFalse(self.marker.exists())

  def test_published_json_bad_hash_rejected_before_any_np_load(self):
    pins = self.publish()
    for name in pins:
      with self.subTest(document=name):
        invalid = dict(pins, **{name: '0' * 64})
        with mock.patch.object(loader.np, 'load',
            side_effect=AssertionError('Read arrays before source authority')) as load_spy:
          with self.assertRaisesRegex(ValueError, 'Published source identity differs'):
            self.authorize(invalid)
          load_spy.assert_not_called()

  def test_npz_bad_hash_rejected_before_any_np_load(self):
    self.documents['result.json']['evidence_artifacts'][0]['sha256'] = '0' * 64
    pins = self.publish()
    with mock.patch.object(loader.np, 'load',
        side_effect=AssertionError('Decoded unauthenticated array')) as load_spy:
      with self.assertRaisesRegex(ValueError, 'Numeric input bytes changed'):
        self.authorize(pins)
      load_spy.assert_not_called()

  def test_source_git_and_version_all_required_receipts(self):
    baseline = copy.deepcopy(self.documents)
    for name in ('plan.json', 'result.json', 'action_information_complete.json',
                 'supervisor_complete.json'):
      for field, value in (('actual_git_sha', '1' * 40), ('version', 'other-version')):
        with self.subTest(document=name, field=field):
          self.documents = copy.deepcopy(baseline)
          self.documents[name][field] = value
          with self.assertRaisesRegex(ValueError, 'Wrong source Git/version'):
            self.authorize(self.publish())

  def test_completion_and_worker_termination_required(self):
    baseline = copy.deepcopy(self.documents)
    for name, field, value in (
        ('action_information_complete.json', 'complete_episodes', 7),
        ('action_information_complete.json', 'complete_states', 31),
        ('supervisor_complete.json', 'exit_code', 1),
        ('supervisor_complete.json', 'timed_out', True),
        ('supervisor_complete.json', 'worker_terminal', False)):
      with self.subTest(document=name, field=field):
        self.documents = copy.deepcopy(baseline)
        self.documents[name][field] = value
        with self.assertRaisesRegex(ValueError, 'Source completion chain failed'):
          self.authorize(self.publish())

  def test_plan_result_budget_chain_hashes_required(self):
    baseline = copy.deepcopy(self.documents)
    for name, field in (
        ('action_information_complete.json', 'plan_sha256'),
        ('action_information_complete.json', 'result_sha256'),
        ('supervisor_complete.json', 'budget_sha256')):
      with self.subTest(document=name, field=field):
        self.documents = copy.deepcopy(baseline)
        self.publish()
        self.documents[name][field] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Source completion chain failed'):
          self.authorize(self.publish(refresh_chain=False))

  def test_frozen_source_plan_episode_positions_horizon_delta(self):
    baseline = copy.deepcopy(self.documents)
    for field, value in (
        ('episodes', list(reversed(loader.EPISODES))),
        ('positions', [1, 300, 600, 901]), ('horizon', 21), ('delta', .1)):
      with self.subTest(field=field):
        self.documents = copy.deepcopy(baseline)
        self.documents['plan.json'][field] = value
        with self.assertRaisesRegex(ValueError, 'Frozen source object changed'):
          self.authorize(self.publish())

  def test_independent_source_audit_must_pass_exact_proof_and_no_blind(self):
    baseline = copy.deepcopy(self.documents)
    for field, value in (
        ('status', 'failed'), ('expected_git_sha', '1' * 40),
        ('source_h_sampled_z_and_actor_raw_byte_exact', False),
        ('source_h_sampled_z_and_actor_raw_byte_exact', 1),
        ('blind_arrays_read', True), ('blind_arrays_read', 0)):
      with self.subTest(field=field, value=value):
        self.documents = copy.deepcopy(baseline)
        self.documents['independent-numeric-audit-v1.json'][field] = value
        with self.assertRaisesRegex(ValueError, 'Missing independent source feature proof'):
          self.authorize(self.publish())

  def test_episode_artifact_order_and_safe_exact_relative_path(self):
    baseline = copy.deepcopy(self.documents)
    self.documents['result.json']['evidence_artifacts'].reverse()
    with self.assertRaisesRegex(ValueError, 'Source episode order changed'):
      self.authorize(self.publish())
    for path in ('../episode_009.npz', '/episode_009.npz',
                 'C:\\episode_009.npz', 'sub/episode_009.npz', 'episode_010.npz'):
      with self.subTest(path=path):
        self.documents = copy.deepcopy(baseline)
        self.documents['result.json']['evidence_artifacts'][0]['path'] = path
        with self.assertRaisesRegex(ValueError, 'Unsafe/unexpected source path'):
          self.authorize(self.publish())


if __name__ == '__main__':
  unittest.main()
