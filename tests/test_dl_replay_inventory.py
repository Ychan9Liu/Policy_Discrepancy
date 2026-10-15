"""Small generated persistent chunks; no real replay, model or GPU acceptance."""
import copy
import json
import tempfile
from pathlib import Path
import sys
import unittest
from unittest import mock
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_replay_inventory as r


def uuid(value):
  text = ''
  while value:
    value, digit = divmod(value, 62)
    text = r.ALPHABET[digit] + text
  return text.rjust(22, '0')


def chunk(directory, identity, successor, *, first=True, length=3, change=None):
  arrays = {name: np.zeros((length, *tail), dtype) for name, (dtype, tail) in r.FIELDS.items()}
  arrays['dyn/stoch'][..., 0] = 1
  arrays['is_first'][0] = first
  arrays['stepid'][:, :16] = np.frombuffer(r.uuid_bytes(uuid(identity)), np.uint8)
  arrays['stepid'][:, 16:] = np.arange(length, dtype='>u4').view(np.uint8).reshape(length, 4)
  if change: change(arrays)
  path = directory / f'20261014T005815F054475-{uuid(identity)}-{uuid(successor)}-{length}.npz'
  np.savez_compressed(path, **arrays)
  return path


class ReplayInventoryTest(unittest.TestCase):
  def setUp(self):
    temp = tempfile.TemporaryDirectory()
    self.addCleanup(temp.cleanup)
    self.base = Path(temp.name)

  def test_real_zip_crc_npy_fields_ids_and_context(self):
    path = chunk(self.base, 1, 2)
    row = r.validate_chunk(path)
    self.assertTrue(row['crc_verified'])
    self.assertTrue(row['cached_context_validated'])
    self.assertEqual((row['length'], row['resets'], row['first'], row['last']), (3, 1, True, False))
    self.assertEqual(row['successor'], uuid(2))
    # Corrupt the central-directory CRC, retaining valid NPY schema/headers.
    data = bytearray(path.read_bytes())
    at = data.index(b'PK\x01\x02') + 16
    data[at] ^= 1
    path.write_bytes(data)
    with self.assertRaises((ValueError, zipfile.BadZipFile)):
      r.validate_chunk(path)

  def test_bad_dtype_shape_object_ids_flags_and_cache_refused_without_skip(self):
    changes = (
        lambda a: a.update(action=a['action'].astype(np.float64)),
        lambda a: a.update(image=a['image'][:, :, :63]),
        lambda a: a.update(reward=np.full(3, object(), object)),
        lambda a: a['stepid'].__setitem__((1, 19), 99),
        lambda a: a['stepid'].__setitem__((0, 0), 1),
        lambda a: a['dyn/deter'].__setitem__((0, 0), np.nan),
        lambda a: a['dyn/stoch'].__setitem__((0, 0, 0), .5),
        lambda a: a['is_terminal'].__setitem__(1, True),
        lambda a: a['is_first'].__setitem__(1, True),
        lambda a: a.pop('dyn/deter'),
        lambda a: a.update(consec=np.zeros(3, np.int32)),
    )
    for number, change in enumerate(changes):
      with self.subTest(number=number):
        path = chunk(self.base, number + 1, number + 100, change=change)
        with self.assertRaises((ValueError, zipfile.BadZipFile)):
          r.validate_chunk(path)

  def test_sixteen_complete_chains_with_legitimate_dangling_tail_successors(self):
    rows = []
    for stream in range(16):
      for offset, first in ((0, True), (1, False)):
        identity = 10 * stream + offset + 1
        rows.append(r.validate_chunk(chunk(self.base, identity, identity + 1, first=first)))
    chains = r.validate_chains(rows)
    self.assertEqual(len(chains), 16)
    self.assertEqual(sum(c['rows'] for c in chains), 96)
    self.assertTrue(all(len(c['paths']) == 2 for c in chains))
    for issue in ('missing', 'duplicate', 'merge', 'cycle', 'boundary', 'backtime'):
      bad = copy.deepcopy(rows)
      if issue == 'missing': bad.pop(0)
      elif issue == 'duplicate': bad.append(copy.deepcopy(bad[0]))
      elif issue == 'merge': bad[2]['successor'] = bad[0]['successor']
      elif issue == 'cycle': bad[1]['successor'] = bad[0]['uuid']
      elif issue == 'boundary': bad[1]['first'] = True
      else: bad[1]['time'] = '20261013T005815F054475'
      with self.subTest(issue=issue), self.assertRaises(ValueError): r.validate_chains(bad)

  def test_small_full_offline_capture_preserves_every_input_byte(self):
    run = self.base / 'run'
    replay = run / 'replay'
    replay.mkdir(parents=True)
    for stream in range(16): chunk(replay, stream + 1, stream + 100)
    manifest = dict(git_commit=r.SOURCE_GIT, group='N', root_seed=0,
        task='dmc_quadruped_walk', config={'agent.dyn.rssm.deter': 4096,
            'agent.dyn.rssm.stoch': 32, 'agent.dyn.rssm.classes': 32,
            'replay_context': 1, 'replay.chunksize': 1024, 'run.envs': 16,
            'batch_size': 16, 'batch_length': 64, 'consec_train': 1})
    final = dict(git_commit=r.SOURCE_GIT, group='N', seed=0,
        task='dmc_quadruped_walk', train_action_steps=32, train_resets=16,
        status='training_ended_pending_audit')
    for name, value in (('protocol_manifest.json', manifest), ('final_state.json', final)):
      (run / name).write_text(json.dumps(value), encoding='utf-8')
    output = self.base / 'inventory.json'
    with mock.patch.object(r, 'EXPECTED_FILES', 16), mock.patch.object(r, 'TRAIN_ACTIONS', 32), \
        mock.patch.object(r, 'binding', return_value={'fixture-source': 'a' * 64}):
      before = r.snapshot(replay)
      report = r.validate(run, replay, output, r.SOURCE_GIT, 'b' * 40)
      self.assertEqual(report['before_inventory'], before)
      self.assertEqual(report['after_inventory'], before)
      self.assertEqual(r.snapshot(replay), before)
      self.assertEqual(report['rows'], 48)
      self.assertEqual(report['train_actions'], 32)
      self.assertEqual(len(report['chains']), 16)
      for field in ('sampler_rng_restored', 'online_queue_restored', 'environment_restored',
          'seamless_resume_established', 'model_loaded', 'gpu_used'):
        self.assertFalse(report[field])
      self.assertEqual(report['corrupt_chunks_skipped'], 0)
      self.assertEqual(json.loads(output.read_text())['version'], r.VERSION)
      with self.assertRaisesRegex(ValueError, 'Fresh external'):
        r.validate(run, replay, output, r.SOURCE_GIT, 'b' * 40)
      with self.assertRaisesRegex(ValueError, 'outside original run'):
        r.validate(run, replay, run / 'inventory.json', r.SOURCE_GIT, 'b' * 40)

  def test_file_count_names_uuid_and_metadata_json_fail_closed(self):
    chunk(self.base, 1, 2)
    with self.assertRaisesRegex(ValueError, 'chunk count'): r.snapshot(self.base)
    for name in ('junk.npz', '20261014T005815F054475-' + uuid(1) + '-' + uuid(2) + '-1025.npz',
        '20261314T005815F054475-' + uuid(1) + '-' + uuid(2) + '-3.npz',
        '20261014T005815F054475-' + uuid(1) + '-' + uuid(1) + '-3.npz'):
      with self.subTest(name=name), self.assertRaises(ValueError): r.filename(name)
    with self.assertRaises(ValueError): r.uuid_bytes('Z' * 22)
    for text in ('{"a":1,"a":2}', '{"a":NaN}'):
      with self.assertRaises(ValueError): r.read_json(text)
    (self.base / 'unexpected.txt').write_text('not a chunk')
    with self.assertRaisesRegex(ValueError, 'only regular NPZ'): r.snapshot(self.base)


if __name__ == '__main__':
  unittest.main()
