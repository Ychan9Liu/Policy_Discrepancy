"""CPU fixtures for DCS semantics, independent RNG and frozen-scope protection.

Fake rendering tests never establish real CUDA/EGL acceptance. The separate
probe_dcs_background entry point exercises actual MuJoCo rendering.
"""

import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import dm_env
import numpy as np
from PIL import Image

from embodied.envs.dcs_background import (
    BackgroundDataset, DistractingBackground, GROUND_ALPHA, TRAIN_VIDEOS,
    VAL_VIDEOS, resolve_options, seed_for_background)
from embodied.envs.dmc_background import DMCBackground as DMC
from dreamerv3.main import make_env, validate_environment_config
from scripts.prepare_m2_v1 import check, resolve


REPO = Path(__file__).resolve().parents[1]


def rng_state(rng):
  value = rng.get_state()
  return value[0], value[1].tolist(), value[2], value[3], value[4]


class FakeEnv:
  def __init__(self, seed=17):
    self.rng = np.random.RandomState(seed)
    self.counter = 0
    self.uploads = 0
    self.grid = {('grid', 'a'): 1.0}
    self.model = SimpleNamespace(tex_type=np.array([2]),
        tex_width=np.array([800]), tex_height=np.array([4800]),
        tex_nchannel=np.array([3]), tex_adr=np.array([0]),
        tex_data=np.full(800 * 800 * 3, 89, np.uint8), ptr=None)
    self.physics = SimpleNamespace(model=self.model,
        named=SimpleNamespace(model=SimpleNamespace(mat_rgba=self.grid)),
        contexts=SimpleNamespace(gl=self, mujoco=SimpleNamespace(ptr=None)))

  @contextlib.contextmanager
  def make_current(self):
    yield self

  def call(self, *_):
    self.uploads += 1

  def reset(self):
    self.counter = int(self.rng.randint(20))
    return dm_env.restart(dict(position=np.array([self.counter])))

  def step(self, action):
    self.counter += float(action)
    return dm_env.transition(self.counter, dict(position=np.array([self.counter])))

  def close(self):
    pass


class DcsBackgroundTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.temp = TemporaryDirectory()
    cls.root = Path(cls.temp.name)
    # Reverse creation order deliberately; filenames define chronological order.
    for name in ('fixture-a', 'fixture-b'):
      (cls.root / name).mkdir()
      for index in reversed(range(4)):
        Image.new('RGB', (800, 800), (index * 60, 20, 90)).save(
            cls.root / name / f'{index:05}.png')

  @classmethod
  def tearDownClass(cls):
    cls.temp.cleanup()

  def options(self, mode='dynamic', **kwargs):
    return resolve_options(dict(mode=mode, dataset=str(self.root),
        videos='fixture-a,fixture-b', **kwargs), 'walker', 7)

  def test_dataset_order_hash_selection_and_disjoint_official_splits(self):
    dataset = BackgroundDataset(self.root, 'fixture-b,fixture-a')
    self.assertEqual(tuple(dataset.frames), ('fixture-b', 'fixture-a'))
    self.assertEqual([p.name for p in dataset.frames['fixture-a']],
                     [f'{x:05}.png' for x in range(4)])
    self.assertEqual(len(TRAIN_VIDEOS), 60)
    self.assertEqual(len(VAL_VIDEOS), 30)
    self.assertFalse(set(TRAIN_VIDEOS) & set(VAL_VIDEOS))
    self.assertTrue(dataset.manifest(True)['content_hashed'])
    self.assertEqual(len(BackgroundDataset(self.root,
        'fixture-a,fixture-b', 1).frames), 1)
    for selection in ('../fixture-a', 'fixture-a,fixture-a', ''):
      with self.assertRaises(ValueError):
        BackgroundDataset(self.root, selection)
    with self.assertRaises(FileNotFoundError):
      BackgroundDataset(self.root, 'missing')
    with self.assertRaises(ValueError):
      BackgroundDataset(self.root, 'fixture-a', 2)

  def test_dynamic_requires_two_frames_and_rgb_decode_errors_are_fatal(self):
    with TemporaryDirectory() as temp:
      path = Path(temp) / 'mono'
      path.mkdir()
      Image.new('L', (4, 4), 7).save(path / '000.png')
      with self.assertRaises(ValueError):
        BackgroundDataset(temp, 'mono', dynamic=True)
      options = resolve_options(dict(mode='static', dataset=temp,
          videos='mono'), 'walker', 7)
      with self.assertRaisesRegex(ValueError, 'RGB frames'):
        DistractingBackground(FakeEnv(), options).reset()

  def test_configuration_rejects_invalid_values_and_undefined_ground_rules(self):
    self.assertEqual(self.options()['ground_plane_alpha'], 1)
    self.assertEqual(GROUND_ALPHA['reacher'], 0)
    for domain in ('hopper', 'quadruped', 'acrobot', 'pendulum'):
      with self.assertRaisesRegex(ValueError, 'no ground alpha rule'):
        resolve_options(dict(mode='static', dataset=str(self.root),
            videos='fixture-a'), domain, 7)
      result = resolve_options(dict(mode='static', dataset=str(self.root),
          videos='fixture-a', ground_plane_alpha=0.5), domain, 7)
      self.assertEqual(result['ground_plane_alpha'], 0.5)
    for kwargs in (dict(video_alpha=0), dict(video_alpha=float('nan')),
                   dict(ground_plane_alpha=2), dict(cache_size=0),
                   dict(num_videos=0), dict(seed=True), dict(typo=3)):
      with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
        self.options(**kwargs)
    with self.assertRaises(ValueError):
      seed_for_background(None)
    with self.assertRaises(ValueError):
      DMC('walker_walk', seed=7, proprio=False, background={'mode': 'typo'})

  def test_static_is_fixed_until_reset_dynamic_moves_and_bounces(self):
    for mode in ('static', 'dynamic'):
      env = DistractingBackground(FakeEnv(), self.options(mode))
      env.reset()
      initial = env.state()
      env._index, env._direction = 2, 1
      indices = []
      for _ in range(6):
        env.step(0)
        indices.append(env.state()['frame_index'])
      self.assertEqual(indices, [2]*6 if mode == 'static' else [3, 3, 2, 1, 0, 1])
      self.assertEqual(initial['rng'], env.state()['rng'])
      self.assertLessEqual(len(env._cache), 2)
      env.reset()
      self.assertEqual(env.state()['episode'], 1)
      self.assertEqual(env.state()['dm_steps'], 0)

  def test_rng_isolation_replay_and_read_only_diagnostic_restoration(self):
    global_before = rng_state(np.random)
    first = DistractingBackground(FakeEnv(), self.options())
    second = DistractingBackground(FakeEnv(), self.options())
    unrelated = DistractingBackground(FakeEnv(), self.options(seed=99))
    raw = FakeEnv()
    for episode in range(3):
      raw_ts = raw.reset()
      a, b = first.reset(), second.reset()
      np.testing.assert_array_equal(raw_ts.observation['position'], a.observation['position'])
      self.assertEqual(first.state(), second.state())
      for action in (0, 1, -1, 0.3):
        before, texture = first.state(), first.model.tex_data.copy()
        for _ in range(3):
          first.state()
        with first.frame_override('fixture-a', 3):
          self.assertEqual(first.state()['frame'], '00003.png')
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
          with first.frame_override('fixture-a', 0):
            raise RuntimeError('fixture')
        self.assertEqual(first.state(), before)
        np.testing.assert_array_equal(first.model.tex_data, texture)
        unrelated.reset()
        a, b, c = first.step(action), second.step(action), raw.step(action)
        self.assertEqual(a.reward, b.reward)
        self.assertEqual(a.reward, c.reward)
        self.assertEqual(first.state(), second.state())
        self.assertEqual(rng_state(first._env.rng), rng_state(raw.rng))
    self.assertEqual(global_before, rng_state(np.random))
    seeds = {seed_for_background(x, y) for x in (1, 2) for y in (-1, 0, 1)}
    self.assertEqual(len(seeds), 6)

  def test_alpha_blends_original_sky_across_resets_not_previous_background(self):
    env = DistractingBackground(FakeEnv(), self.options(video_alpha=0.5))
    for _ in range(3):
      env.reset()
      with env.frame_override('fixture-a', 0):
        np.testing.assert_array_equal(env.model.tex_data[:3], [44, 54, 89])

  def test_upstream_reference_frame_schedule_and_rng_match(self):
    reference = REPO / 'third_party/distracting_control/background_reference.py'
    self.assertEqual(hashlib.sha256(reference.read_bytes()).hexdigest(),
        '1a5f95fa3b53dabe7111549a5fec38fabab17256bab35f9f5e28cf756119e1d9')
    # Equal-size solid RGB fixtures avoid the deliberately different resizers.
    tensor = lambda x, shape: SimpleNamespace(numpy=lambda: np.reshape(x, shape))
    tf = SimpleNamespace(reshape=tensor, io=SimpleNamespace(gfile=SimpleNamespace(
        # gfile returns Unicode names; os.listdir(np.str_) can return bytes on
        # Linux/Python 3.11. Normalize the test double's input, not DCS behavior.
        listdir=lambda x: sorted(os.listdir(str(x))))))
    spec = importlib.util.spec_from_file_location('dcs_reference_fixture', reference)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, tensorflow=tf):
      spec.loader.exec_module(module)
    for mode in ('static', 'dynamic'):
      options = self.options(mode)
      local = DistractingBackground(FakeEnv(), options)
      original = module.DistractingBackgroundEnv(FakeEnv(),
          dataset_path=str(self.root), dataset_videos=['fixture-a', 'fixture-b'],
          dynamic=mode == 'dynamic', seed=options['derived_seed'])
      for _ in range(3):
        local.reset()
        original.reset()
        for _ in range(12):
          np.testing.assert_array_equal(local.model.tex_data, original.physics.model.tex_data)
          self.assertEqual(rng_state(local._rng), rng_state(original._random_state))
          local.step(0)
          original.step(0)

  def test_presets_clean_legacy_config_and_formal_scope_rejection(self):
    legacy = resolve(['--configs', 'm2_v1', '--task', 'dmc_hopper_hop', '--jax.prealloc', 'False',
                      '--logger.outputs', 'jsonl'])
    check(legacy)
    for mode in ('clean', 'static', 'dynamic'):
      config = resolve(['--configs', 'dcs_' + mode])
      self.assertEqual(config.env.dmc.background.mode, mode)
      self.assertFalse(config.env.dmc.proprio)
      self.assertTrue(config.env.dmc.use_seed)
    for mode in ('static', 'dynamic'):
      config = legacy.update({'env.dmc.background.mode': mode})
      with self.assertRaisesRegex(ValueError, 'clean-only'):
        check(config)
      with self.assertRaisesRegex(ValueError, 'clean-only'):
        validate_environment_config(config)

  def test_factory_dispatch_keeps_clean_constructor_arguments_and_seed(self):
    clean = resolve(['--configs', 'dcs_clean', '--task', 'dmc_walker_walk'])
    sentinel = object()
    with patch('embodied.envs.dmc.DMC', return_value=sentinel) as ctor, \
         patch('dreamerv3.main.wrap_env', side_effect=lambda env, config: env), \
         patch('embodied.envs.dcs_background.seed_for_background',
               side_effect=AssertionError('clean must not derive a background seed')):
      self.assertIs(make_env(clean, 3, seed=91), sentinel)
    self.assertNotIn('background', ctor.call_args.kwargs)
    self.assertEqual(ctor.call_args.kwargs['seed'], 91)
    self.assertEqual(ctor.call_args.args, ('walker_walk',))
    noisy = clean.update({'env.dmc.background.mode': 'dynamic'})
    with patch('embodied.envs.dmc_background.DMCBackground', return_value=sentinel) as ctor, \
         patch('dreamerv3.main.wrap_env', side_effect=lambda env, config: env):
      self.assertIs(make_env(noisy, 3, seed=92), sentinel)
    self.assertEqual(ctor.call_args.kwargs['seed'], 92)
    self.assertEqual(ctor.call_args.kwargs['background']['mode'], 'dynamic')
    with self.assertRaisesRegex(ValueError, 'explicit dmc_ task'):
      validate_environment_config(resolve(['--configs', 'dcs_static']))
    with self.assertRaisesRegex(ValueError, 'clean-only'):
      make_env(clean.update({'script': 'protocol_v1'}), 0,
               background=dict(mode='static'))


if __name__ == '__main__':
  unittest.main()
