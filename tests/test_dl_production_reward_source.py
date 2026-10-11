"""Constructed numeric/source fixtures only; no real episode or checkpoint reads."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import ml_dtypes
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_production_reward_source as source


def poison(marker):
  Path(marker).write_text('decoded')
  raise AssertionError('Opaque envelope decoded')


class Poison:
  def __init__(self, marker):
    self.marker = str(marker)

  def __reduce__(self):
    return poison, (self.marker,)


def fields(episode=9):
  h = np.zeros((4, 4096), np.float64)
  h[:, :4] = np.eye(4)
  z = np.zeros((4, 4, 2, 32, 32), np.float64)
  z[..., 0] = 1.
  reward = np.arange(1200, dtype=np.float64).reshape(4, 15, 20) / 100.
  commands = np.zeros((4, 15, 12), np.float32)
  commands[:, :7] = source.command_library()
  return dict(episode=np.array(episode, np.int32), positions=np.array(source.POSITIONS, np.int32),
              h=h, latent=z, reward_f64=reward, first_commands=commands)


def save_numeric(path, data, marker):
  np.savez_compressed(path, **data, records=np.array([Poison(marker)], object),
                      unrelated_object=np.array([Poison(marker)], object))


def fixture_documents(directory, marker):
  artifacts, costs = [], []
  for episode in source.EPISODES:
    data = fields(episode)
    path = directory / f'episode_{episode:03}.npz'
    save_numeric(path, data, marker)
    costs.append(np.sum(data['reward_f64'][:, :7], -1, dtype=np.float64))
    proofs = [dict(position=p, control=c, executed_steps=20, physics_steps=20,
        padding_steps=0, end_type='window', float64_reward_byte_exact=True,
        float32_reward_encoding_byte_exact=True, discount_byte_exact=True)
        for p in source.POSITIONS for c in range(15)]
    artifacts.append(dict(episode=episode, path=path.name, sha256=source.sha(path), records=60,
                         main_steps=1200, replay_steps=1200, verifications=proofs))
  identity = dict(actual_git_sha=source.SOURCE_GIT, version='DL-action-information-r1')
  config = {'task': 'dmc_quadruped_walk', 'env.dmc.proprio': False,
      'env.dmc.size': [64, 64], 'env.dmc.repeat': 1, 'env.dmc.camera': -1,
      'jax.compute_dtype': 'bfloat16', 'agent.dyn.typ': 'rssm',
      'agent.dyn.rssm.deter': 4096, 'agent.dyn.rssm.stoch': 32,
      'agent.dyn.rssm.classes': 32, 'agent.dyn.rssm.unimix': .01,
      'agent.rewhead.output': 'symexp_twohot', 'agent.dt_latch.mode': 'off',
      'agent.rep_probe.mode': 'off'}
  return {
      'plan.json': dict(identity, episodes=list(source.EPISODES), positions=list(source.POSITIONS),
          horizon=20, delta=.2, actor_command_order=[f'pair{i}{b}' for i in range(4) for b in ('q', 'p')],
          model_provenance=dict(source.MODEL_PINS, checkpoint_counters=source.COUNTERS, config=config)),
      'result.json': dict(identity, evidence_artifacts=artifacts,
          external_control=dict(rewards=np.stack(costs).tolist()), main_physics_steps=9600,
          replay_physics_steps=9600, fresh_model_calls=0, render_calls=0,
          fresh_actor_rng_calls=0, source_artifacts_unchanged=True),
      'budget.json': dict(version='DL-action-information-r1', git_sha=source.SOURCE_GIT, seconds=180),
      'action_information_complete.json': dict(identity, complete_episodes=8,
          complete_states=32, complete_records=480),
      'supervisor_complete.json': dict(identity, exit_code=0, timed_out=False, worker_terminal=True),
      'independent-numeric-audit-v1.json': dict(status='passed', expected_git_sha=source.SOURCE_GIT,
          source_h_sampled_z_and_actor_raw_byte_exact=True, source_feedback_bytes_verified=True,
          blind_arrays_read=False)}


def publish(directory, documents):
  def save(name):
    (directory / name).write_text(json.dumps(documents[name], allow_nan=False), encoding='utf-8')
    return source.sha(directory / name)
  pins = {name: save(name) for name in ('plan.json', 'result.json', 'budget.json')}
  documents['action_information_complete.json'].update(
      plan_sha256=pins['plan.json'], result_sha256=pins['result.json'])
  documents['supervisor_complete.json']['budget_sha256'] = pins['budget.json']
  pins.update({name: save(name) for name in ('action_information_complete.json',
      'supervisor_complete.json', 'independent-numeric-audit-v1.json')})
  return pins


class ProductionRewardSourceTest(unittest.TestCase):
  def setUp(self):
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    self.directory = Path(directory.name)
    self.marker = self.directory / 'poison'
    self.path = self.directory / 'episode_009.npz'

  def test_module_import_preserves_runtime_environment(self):
    script = ('import os,json,sys;before=dict(os.environ);'
        'from scripts import dl_production_reward_source;'
        'print(json.dumps(dict(unchanged=before==dict(os.environ),'
        'jax="jax" in sys.modules,worker="scripts.dl_nested_probe" in sys.modules)))')
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, check=True,
        capture_output=True, text=True, timeout=15,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='4', JAX_PLATFORMS='cuda', MUJOCO_GL='disable'))
    self.assertEqual(json.loads(result.stdout), dict(unchanged=True, jax=False, worker=False))

  def test_lazy_six_whitelist_native_bf16_and_poison_never_loaded(self):
    data = fields()
    data['h'][0, 0] = -.0
    save_numeric(self.path, data, self.marker)
    original_load, original_get = np.load, np.lib.npyio.NpzFile.__getitem__
    requested = []
    def numeric_only(archive, name):
      requested.append(name)
      if name not in source.NUMERIC_FIELDS:
        raise AssertionError('Non-whitelisted input requested')
      return original_get(archive, name)
    with (mock.patch.object(source.np, 'load', wraps=original_load) as load,
          mock.patch.object(np.lib.npyio.NpzFile, '__getitem__', numeric_only)):
      loaded = source.load_numeric_episode(self.path, 9)
    self.assertEqual(requested, list(source.NUMERIC_FIELDS))
    self.assertEqual(load.call_args.kwargs, {'allow_pickle': False})
    self.assertEqual(loaded['h_native'].dtype, ml_dtypes.bfloat16)
    self.assertEqual(loaded['latent_native'].dtype, ml_dtypes.bfloat16)
    self.assertEqual(loaded['h_native'].astype(np.float64).tobytes(), data['h'].tobytes())
    self.assertEqual(loaded['true_cost'].tobytes(),
                     data['reward_f64'][:, :7].sum(-1, dtype=np.float64).tobytes())
    self.assertFalse(self.marker.exists())

  def test_identity_shape_dtype_nonfinite_and_raw_control_guards(self):
    replacements = [('episode', np.array(10, np.int32)),
        ('episode', np.array(9, np.int64)), ('positions', np.array([1, 300, 600, 901], np.int32))]
    for name in ('h', 'latent', 'reward_f64', 'first_commands'):
      baseline = fields()[name]
      replacements.extend([(name, baseline[:-1]), (name, baseline.astype(np.int32))])
      invalid = baseline.copy(); invalid.flat[0] = np.nan
      replacements.append((name, invalid))
    invalid = fields()['first_commands']; invalid[0, 0, 0] = .01
    replacements.append(('first_commands', invalid))
    for index, (name, value) in enumerate(replacements):
      with self.subTest(index=index, field=name):
        data = fields(); data[name] = value
        save_numeric(self.path, data, self.marker)
        with self.assertRaises(ValueError):
          source.load_numeric_episode(self.path, 9)
    data = fields(); del data['first_commands']
    save_numeric(self.path, data, self.marker)
    with self.assertRaisesRegex(ValueError, 'Missing required numeric'):
      source.load_numeric_episode(self.path, 9)
    with self.assertRaisesRegex(ValueError, 'Unapproved calibration'):
      source.load_numeric_episode(self.path, 10)
    self.assertFalse(self.marker.exists())

  def test_sampled_onehot_bf16_roundtrip_and_cost_overflow_guards(self):
    for kind in ('fractional', 'empty', 'twohot', 'non_bf16', 'overflow'):
      with self.subTest(kind=kind):
        data = fields()
        if kind == 'fractional': data['latent'][0, 0, 0, 0, :2] = .5
        elif kind == 'empty': data['latent'][0, 0, 0, 0] = 0
        elif kind == 'twohot': data['latent'][0, 0, 0, 0, 1] = 1
        elif kind == 'non_bf16': data['h'][0, 0] = .1
        else: data['reward_f64'][:] = np.finfo(np.float64).max
        save_numeric(self.path, data, self.marker)
        with np.errstate(over='ignore', invalid='ignore'), self.assertRaises(ValueError):
          source.load_numeric_episode(self.path, 9)
    self.assertFalse(self.marker.exists())

  def test_complete_constructed_source_chain_and_cost_bytes(self):
    documents = fixture_documents(self.directory, self.marker)
    pins = publish(self.directory, documents)
    with mock.patch.object(source, 'SOURCE_PINS', pins):
      data = source.load_dataset(self.directory)
    self.assertEqual(data['h_native'].shape, (8, 4, 4096))
    self.assertEqual(data['latent_native'].shape, (8, 4, 4, 2, 32, 32))
    self.assertEqual(data['controls'].shape, (8, 4, 7, 12))
    self.assertEqual(data['true_cost'].shape, (8, 4, 7))
    self.assertIs(data['records_decoded'], False)
    self.assertIs(data['blind_arrays_read'], False)
    documents['result.json']['external_control']['rewards'][0][0][0] += 1e-10
    with mock.patch.object(source, 'SOURCE_PINS', publish(self.directory, documents)):
      with self.assertRaisesRegex(ValueError, 'True cost bytes differ'):
        source.load_dataset(self.directory)
    self.assertFalse(self.marker.exists())

  def test_source_authority_rejections_precede_numeric_reads(self):
    baseline = fixture_documents(self.directory, self.marker)
    for kind in ('json_hash', 'npz_hash', 'git', 'partial', 'model', 'counter', 'latent_group',
                 'blind', 'early_terminal', 'path', 'duplicate_episode'):
      with self.subTest(kind=kind):
        documents = copy.deepcopy(baseline)
        if kind == 'npz_hash': documents['result.json']['evidence_artifacts'][0]['sha256'] = '0' * 64
        elif kind == 'git': documents['plan.json']['actual_git_sha'] = '0' * 40
        elif kind == 'partial': documents['action_information_complete.json']['complete_records'] = 479
        elif kind == 'model': documents['plan.json']['model_provenance']['params_sha256'] = '0' * 64
        elif kind == 'counter': documents['plan.json']['model_provenance']['checkpoint_counters'] = {}
        elif kind == 'latent_group': documents['plan.json']['model_provenance']['config']['agent.dyn.rssm.stoch'] = 64
        elif kind == 'blind': documents['independent-numeric-audit-v1.json']['blind_arrays_read'] = True
        elif kind == 'early_terminal': documents['result.json']['evidence_artifacts'][0]['verifications'][0]['executed_steps'] = 19
        elif kind == 'path': documents['result.json']['evidence_artifacts'][0]['path'] = '../episode_009.npz'
        elif kind == 'duplicate_episode': documents['result.json']['evidence_artifacts'][1]['episode'] = 9
        pins = publish(self.directory, documents)
        if kind == 'json_hash': pins['plan.json'] = '0' * 64
        with (mock.patch.object(source, 'SOURCE_PINS', pins),
              mock.patch.object(source.np, 'load', side_effect=AssertionError('Numeric read before authority')) as load):
          with self.assertRaises(ValueError): source.authority(self.directory)
          load.assert_not_called()

  def test_source_is_rechecked_after_numeric_reads(self):
    documents = fixture_documents(self.directory, self.marker)
    pins = publish(self.directory, documents)
    original = source.load_numeric_episode
    def tamper_last(path, episode):
      result = original(path, episode)
      if episode == source.EPISODES[-1]:
        with (self.directory / 'plan.json').open('a') as stream: stream.write(' ')
      return result
    with (mock.patch.object(source, 'SOURCE_PINS', pins),
          mock.patch.object(source, 'load_numeric_episode', side_effect=tamper_last)):
      with self.assertRaisesRegex(ValueError, 'Published source identity differs'):
        source.load_dataset(self.directory)


if __name__ == '__main__':
  unittest.main()
