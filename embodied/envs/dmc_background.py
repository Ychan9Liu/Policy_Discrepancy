"""Optional DCS adapter around the unchanged clean DMC implementation."""

import embodied

from .dmc import DMC
from . import from_dm
from .dcs_background import DistractingBackground, resolve_options


class DMCBackground(DMC):

  def __init__(self, env, repeat=1, size=(64, 64), proprio=False, image=True,
               camera=-1, seed=None, engineering_trace_physics=False,
               background=None):
    background = dict(background or {})
    mode = background.get('mode', 'clean')
    if mode not in ('clean', 'static', 'dynamic'):
      raise ValueError('background.mode must be clean, static, or dynamic')
    if mode != 'clean':
      if not isinstance(env, str) or env.startswith(('manip_', 'rodent_')):
        raise ValueError('DCS backgrounds require a named standard DMC task')
      if not image or proprio:
        raise ValueError('DCS background conditions require image=True, proprio=False')
      domain = env.split('_', 1)[0]
      domain = 'ball_in_cup' if domain == 'cup' else domain
      options = resolve_options(background, domain, seed)
    super().__init__(env, repeat, size, proprio, image, camera, seed,
                     engineering_trace_physics)
    if mode != 'clean':
      try:
        self._dmenv = DistractingBackground(self._dmenv, options)
      except Exception:
        self._dmenv.close()
        raise
      self._env = from_dm.FromDM(self._dmenv)
      self._env = embodied.wrappers.ActionRepeat(self._env, repeat)

  def background_state(self):
    """Diagnostic read only; never advances environment or background RNG."""
    if isinstance(self._dmenv, DistractingBackground):
      return self._dmenv.state()
    return dict(mode='clean')

  def render_background_frame(self, video, frame_index):
    """Paired image on an idle environment, with the original texture restored."""
    if not isinstance(self._dmenv, DistractingBackground):
      raise ValueError('Paired background rendering needs static or dynamic mode')
    with self._dmenv.frame_override(video, frame_index):
      return self._dmenv.physics.render(*self._size, camera_id=self._camera)

  def close(self):
    self._dmenv.close()
