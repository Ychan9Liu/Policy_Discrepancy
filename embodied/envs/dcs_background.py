# Copyright 2026 The Google Research Authors.
# Modifications Copyright 2026 Policy_Discrepancy contributors.
# Licensed under the Apache License, Version 2.0. See
# third_party/distracting_control/LICENSE and README.md for provenance.
"""Seeded, background-only DCS port; no color or camera wrappers.

The pinned upstream source is retained in third_party/distracting_control.
This port uses deterministic file order, Pillow bilinear resizing (not TF),
and a bounded texture cache. It is not pixel-identical to the TF benchmark.
"""

from collections import OrderedDict
import contextlib
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


UPSTREAM_REVISION = '5b09c22d73a9d35eb6c5d2a99b95677a45053466'
IMPLEMENTATION = 'pd-dcs-background-v1'
GROUND_ALPHA = dict(reacher=0.0, walker=1.0, cheetah=1.0,
                    cartpole=0.3, finger=0.3, ball_in_cup=0.3, humanoid=0.3)
TRAIN_VIDEOS = (
    'bear', 'bmx-bumps', 'boat', 'boxing-fisheye', 'breakdance-flare', 'bus',
    'car-turn', 'cat-girl', 'classic-car', 'color-run', 'crossing',
    'dance-jump', 'dancing', 'disc-jockey', 'dog-agility', 'dog-gooses',
    'dogs-scale', 'drift-turn', 'drone', 'elephant', 'flamingo', 'hike',
    'hockey', 'horsejump-low', 'kid-football', 'kite-walk', 'koala',
    'lady-running', 'lindy-hop', 'longboard', 'lucia', 'mallard-fly',
    'mallard-water', 'miami-surf', 'motocross-bumps', 'motorbike', 'night-race',
    'paragliding', 'planes-water', 'rallye', 'rhino', 'rollerblade',
    'schoolgirls', 'scooter-board', 'scooter-gray', 'sheep', 'skate-park',
    'snowboard', 'soccerball', 'stroller', 'stunt', 'surf', 'swing', 'tennis',
    'tractor-sand', 'train', 'tuk-tuk', 'upside-down', 'varanus-cage', 'walking')
VAL_VIDEOS = (
    'bike-packing', 'blackswan', 'bmx-trees', 'breakdance', 'camel',
    'car-roundabout', 'car-shadow', 'cows', 'dance-twirl', 'dog', 'dogs-jump',
    'drift-chicane', 'drift-straight', 'goat', 'gold-fish', 'horsejump-high',
    'india', 'judo', 'kite-surf', 'lab-coat', 'libby', 'loading', 'mbike-trick',
    'motocross-jump', 'paragliding-launch', 'parkour', 'pigs', 'scooter-black',
    'shooting', 'soapbox')


def seed_for_background(env_seed, root_seed=-1):
  """No RNG draws. Different train/eval environment seeds give separate streams."""
  if type(env_seed) is not int or not 0 <= env_seed < 2 ** 32:
    raise ValueError('DCS requires an explicit uint32 environment seed')
  if type(root_seed) is not int or not -1 <= root_seed < 2 ** 32:
    raise ValueError('background.seed must be -1 or a uint32 integer')
  parts = [IMPLEMENTATION, str(env_seed), str(root_seed)]
  return int.from_bytes(hashlib.sha256('\0'.join(parts).encode()).digest()[:4],
                        'big')


def resolve_options(options, domain, env_seed):
  defaults = dict(mode='clean', dataset='', videos='', num_videos=-1,
                  video_alpha=1.0, ground_plane_alpha=-1.0, seed=-1, cache_size=2)
  unknown = set(options) - defaults.keys()
  if unknown:
    raise ValueError(f'Unknown DCS background options: {sorted(unknown)}')
  values = dict(defaults, **options)
  if values['mode'] not in ('clean', 'static', 'dynamic'):
    raise ValueError('background.mode must be clean, static, or dynamic')
  if values['mode'] == 'clean':
    return values
  values['derived_seed'] = seed_for_background(env_seed, values['seed'])
  values['environment_seed'] = env_seed
  alpha = values['ground_plane_alpha']
  if alpha == -1:
    if domain not in GROUND_ALPHA:
      raise ValueError(f'DCS has no ground alpha rule for {domain}; specify '
                       'background.ground_plane_alpha explicitly')
    alpha = GROUND_ALPHA[domain]
  values['ground_plane_alpha'] = float(alpha)
  for key in ('video_alpha', 'ground_plane_alpha'):
    if not np.isfinite(values[key]) or not 0 <= values[key] <= 1:
      raise ValueError(f'background.{key} must be finite and in [0, 1]')
  if values['video_alpha'] == 0:
    raise ValueError('Zero video alpha is not a noisy condition; use clean')
  for key in ('num_videos', 'cache_size'):
    if type(values[key]) is not int:
      raise ValueError(f'background.{key} must be an integer')
  if values['num_videos'] != -1 and values['num_videos'] < 1:
    raise ValueError('background.num_videos must be -1 (all selected) or positive')
  if values['cache_size'] < 1:
    raise ValueError('background.cache_size must be positive')
  if not values['dataset'] or not values['videos']:
    raise ValueError('DCS requires an explicit dataset path and videos selection')
  return values


class BackgroundDataset:
  """Explicit video selection; no fallback from missing/invalid data to clean."""

  def __init__(self, directory, videos, num_videos=-1, dynamic=False):
    self.root = Path(directory).expanduser().resolve(strict=True)
    if videos == 'train':
      names = list(TRAIN_VIDEOS)
    elif videos == 'val':
      names = list(VAL_VIDEOS)
    elif isinstance(videos, str):
      names = videos.split(',')
    else:
      raise ValueError('videos must be train, val, or comma-separated names')
    if (not names or len(names) != len(set(names)) or
        any(not x or x in ('.', '..') or '/' in x or '\\' in x for x in names)):
      raise ValueError('Video names must be distinct directory names')
    if num_videos != -1:
      if type(num_videos) is not int or not 1 <= num_videos <= len(names):
        raise ValueError('num_videos exceeds selected videos or is invalid')
      names = names[:num_videos]
    self.frames = OrderedDict()
    for name in names:
      directory = (self.root / name).resolve(strict=True)
      if not directory.is_relative_to(self.root):
        raise ValueError('Video escapes the dataset directory')
      paths = sorted(p for p in directory.iterdir()
                     if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
      if len(paths) < (2 if dynamic else 1):
        raise ValueError(f'{name} needs {2 if dynamic else 1} or more frames')
      for path in paths:
        if not path.is_file() or not path.resolve().is_relative_to(self.root):
          raise ValueError(f'Invalid frame path: {path}')
      self.frames[name] = tuple(paths)
    listing = [[name, [p.name for p in paths]]
               for name, paths in self.frames.items()]
    self.listing_sha256 = hashlib.sha256(
        json.dumps(listing, separators=(',', ':')).encode()).hexdigest()

  def manifest(self, hash_frames=False):
    videos = []
    for name, paths in self.frames.items():
      frames = [dict(file=p.name, bytes=p.stat().st_size,
                     **({'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                        if hash_frames else {})) for p in paths]
      videos.append(dict(name=name, frames=frames))
    return dict(root=str(self.root), listing_sha256=self.listing_sha256,
                content_hashed=hash_frames, videos=videos)


class DistractingBackground:
  """dm_env wrapper inserted before FromDM and ActionRepeat.

Static chooses one frame at reset. Dynamic chooses a start/direction at reset
and advances one frame per dm_control step, bouncing at the video boundaries.
Extra renders or diagnostic reads never advance the background or its RNG.
"""

  def __init__(self, env, options):
    self._env = env
    self.options = dict(options)
    self.dataset = BackgroundDataset(options['dataset'], options['videos'],
        options['num_videos'], options['mode'] == 'dynamic')
    self._rng = np.random.RandomState(options['derived_seed'])
    self._cache = OrderedDict()
    self._original_sky = None
    self._episode = -1
    self._video = None
    self._index = 0
    self._direction = 1
    self._steps = 0
    self._original_ground_alpha = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)

  def reset(self):
    result = self._env.reset()
    self._reset_background()
    return result

  def step(self, action):
    result = self._env.step(action)
    if result.first():
      self._reset_background()
    else:
      self._steps += 1
      if self.options['mode'] == 'dynamic':
        self._index += self._direction
        if self._index <= 0:
          self._index = 0
          self._direction = abs(self._direction)
        if self._index >= len(self.dataset.frames[self._video]):
          self._index = len(self.dataset.frames[self._video]) - 1
          self._direction = -abs(self._direction)
        self._apply()
    return result

  def _prepare_texture(self):
    model = self._env.physics.model
    # DCS uses texture 0 and sets its height to 800. Check the layout before
    # changing anything; do not silently write a different texture or channel.
    if int(model.tex_type[0]) != 2 or int(model.tex_nchannel[0]) != 3:
      raise ValueError('Expected DCS RGB skybox at texture index 0')
    width, height = int(model.tex_width[0]), int(model.tex_height[0])
    address = int(model.tex_adr[0])
    if width != 800 or height not in (800, 4800):
      raise ValueError(f'Unsupported DCS sky texture dimensions: {width}x{height}')
    self._size = 800 * width * 3
    self._address = address
    self._shape = (800, width)
    if address < 0 or address + self._size > len(model.tex_data):
      raise ValueError('DCS sky texture does not fit the texture buffer')
    self._original_sky = model.tex_data[address:address + self._size].copy()
    self._original_height = height
    self._original_ground_alpha = float(
        self._env.physics.named.model.mat_rgba['grid', 'a'])
    model.tex_height[0] = 800

  def _reset_background(self):
    if self._original_sky is None:
      self._prepare_texture()
    self._env.physics.named.model.mat_rgba['grid', 'a'] = (
        self.options['ground_plane_alpha'])
    names = tuple(self.dataset.frames)
    self._video = names[int(self._rng.choice(len(names)))]
    self._index = int(self._rng.choice(len(self.dataset.frames[self._video])))
    self._direction = int(self._rng.choice([-1, 1]))
    self._episode += 1
    self._steps = 0
    self._apply()

  def _texture(self):
    path = self.dataset.frames[self._video][self._index]
    if path not in self._cache:
      with Image.open(path) as image:
        if image.mode != 'RGB':
          raise ValueError(f'DCS requires RGB frames: {path} ({image.mode})')
        image = image.resize(self._shape[::-1], Image.Resampling.BILINEAR)
        pixels = np.asarray(image, np.uint8).reshape(-1)
      alpha = self.options['video_alpha']
      if alpha != 1:
        pixels = (alpha * pixels.astype(np.float32) +
            (1 - alpha) * self._original_sky.astype(np.float32)).astype(np.uint8)
      self._cache[path] = pixels
      while len(self._cache) > self.options['cache_size']:
        self._cache.popitem(last=False)
    self._cache.move_to_end(path)
    return self._cache[path]

  def _apply(self):
    self._upload(self._texture())

  def _upload(self, texture):
    from dm_control.mujoco.wrapper import mjbindings
    physics = self._env.physics
    physics.model.tex_data[self._address:self._address + self._size] = texture
    with physics.contexts.gl.make_current() as context:
      context.call(mjbindings.mjlib.mjr_uploadTexture, physics.model.ptr,
                   physics.contexts.mujoco.ptr, 0)

  @contextlib.contextmanager
  def frame_override(self, video, frame_index):
    """Temporary paired render on an idle env; restores texture even on error."""
    if self._original_sky is None:
      raise RuntimeError('Reset the environment before a paired background render')
    if (video not in self.dataset.frames or type(frame_index) is not int or
        not 0 <= frame_index < len(self.dataset.frames[video])):
      raise ValueError('Unknown video or frame index')
    previous_video, previous_index = self._video, self._index
    previous = self._env.physics.model.tex_data[
        self._address:self._address + self._size].copy()
    try:
      self._video, self._index = video, frame_index
      self._apply()
      yield
    finally:
      self._video, self._index = previous_video, previous_index
      self._upload(previous)

  def state(self):
    """JSON-safe background-only state, not a resumable environment checkpoint."""
    rng = self._rng.get_state()
    return dict(implementation=IMPLEMENTATION, upstream_revision=UPSTREAM_REVISION,
        mode=self.options['mode'], seed=self.options['derived_seed'],
        environment_seed=self.options['environment_seed'],
        background_root_seed=self.options['seed'],
        video_alpha=self.options['video_alpha'],
        ground_plane_alpha=self.options['ground_plane_alpha'],
        dataset_listing_sha256=self.dataset.listing_sha256,
        episode=self._episode, video=self._video, frame_index=self._index,
        frame=(self.dataset.frames[self._video][self._index].name
               if self._video is not None else None),
        direction=self._direction, dm_steps=self._steps,
        rng=[rng[0], rng[1].tolist(), int(rng[2]), int(rng[3]), float(rng[4])])

  def close(self):
    self._cache.clear()
    self._env.close()
