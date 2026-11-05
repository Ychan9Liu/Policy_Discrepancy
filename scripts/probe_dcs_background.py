"""Real DMC fixed-action acceptance for three background conditions; no agent.

Requires an explicit rendering backend. On Windows --software-gl-dir loads
isolated Mesa binaries and requires llvmpipe, without a system-wide install.
All seeds, short steps and ground-alpha overrides here are engineering fixtures.
"""

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(value):
  return hashlib.sha256(value.tobytes()).hexdigest()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--dataset', required=True)
  parser.add_argument('--videos', required=True)
  parser.add_argument('--tasks', nargs='+', default=['dmc_walker_walk'])
  parser.add_argument('--all-vision20', action='store_true')
  parser.add_argument('--seed', type=int, default=7)
  parser.add_argument('--background-seed', type=int, default=19)
  parser.add_argument('--steps', type=int, default=16)
  parser.add_argument('--repeat', type=int, default=1)
  parser.add_argument('--reset-indices', nargs='+', type=int)
  parser.add_argument('--ground-alpha', type=float, default=-1)
  parser.add_argument('--unmapped-ground-alpha', type=float, default=-1,
                      help='Explicit fixture override only for domains absent from DCS rules')
  parser.add_argument('--backend', choices=['egl', 'osmesa', 'glfw'], required=True)
  parser.add_argument('--software-gl-dir')
  parser.add_argument('--reference', default='9c0a44b55c18866150a054090da2e978e74400e5')
  parser.add_argument('--output', required=True)
  args = parser.parse_args()
  if args.steps < 2 or args.repeat < 1:
    parser.error('Use at least two fixture steps and positive repeat')
  os.environ['MUJOCO_GL'] = args.backend
  os.environ['JAX_PLATFORMS'] = 'cpu'
  handles = []
  software_gl = None
  if args.software_gl_dir:
    if sys.platform != 'win32' or args.backend != 'glfw':
      parser.error('software-gl-dir requires Windows GLFW')
    os.environ['GALLIUM_DRIVER'] = 'llvmpipe'
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
    directory = Path(args.software_gl_dir).resolve(strict=True)
    handles.append(os.add_dll_directory(str(directory)))
    software_gl = ctypes.WinDLL(str(directory / 'opengl32.dll'))
    handles.append(software_gl)
    software_gl.glGetString.argtypes = [ctypes.c_uint]
    software_gl.glGetString.restype = ctypes.c_char_p
  import numpy as np
  from PIL import Image
  from OpenGL import GL
  from embodied.envs.dcs_background import GROUND_ALPHA
  from dreamerv3.protocol_scope import VISION20_TASKS
  from dreamerv3.main import make_env, wrap_env
  from scripts.prepare_m2_v1 import resolve
  output = Path(args.output)
  output.mkdir(parents=True, exist_ok=False)
  repo = Path(__file__).resolve().parents[1]
  reference_source = subprocess.check_output(
      ['git', 'show', args.reference + ':embodied/envs/dmc.py'], cwd=repo, text=True)
  namespace = dict(__package__='embodied.envs')
  exec(compile(reference_source, 'reference_dmc.py', 'exec'), namespace)
  ReferenceDMC = namespace['DMC']
  tasks = VISION20_TASKS if args.all_vision20 else args.tasks
  reset_indices = args.reset_indices or [0, args.steps // 2 + 1]
  report = dict(engineering_fixture=True, training_started=False,
      git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
          cwd=repo, text=True).strip(), reference_commit=args.reference,
      command=sys.argv, python=sys.version,
      packages={p: importlib.metadata.version(p) for p in
                ['numpy', 'pillow', 'dm-control', 'mujoco', 'jax']},
      backend=args.backend, seed=args.seed, background_seed=args.background_seed,
      steps=args.steps, repeat=args.repeat, ground_alpha=args.ground_alpha,
      reset_indices=reset_indices,
      unmapped_ground_alpha=args.unmapped_ground_alpha,
      tolerance='exact array equality; finite RGB uint8; positive image difference',
      tasks=[], accepted=False)
  start = time.monotonic()
  def rng_state(rng):
    x = rng.get_state()
    return [x[0], x[1].tolist(), int(x[2]), int(x[3]), float(x[4])]
  try:
    for task in tasks:
      name = task.removeprefix('dmc_')
      domain = name.split('_')[0]
      domain = 'ball_in_cup' if domain == 'cup' else domain
      ground_alpha = args.ground_alpha
      if ground_alpha == -1 and domain not in GROUND_ALPHA:
        ground_alpha = args.unmapped_ground_alpha
      results = {}
      for mode in ('reference', 'clean', 'static', 'dynamic', 'dynamic-diagnostics'):
        options = dict(mode=mode.split('-')[0], dataset=args.dataset,
            videos=args.videos, seed=args.background_seed,
            ground_plane_alpha=ground_alpha)
        condition = 'clean' if mode == 'reference' else mode.split('-')[0]
        config = resolve(['--configs', 'dcs_' + condition, '--task', task])
        config = config.update({'env.dmc.repeat': args.repeat,
            **{'env.dmc.background.' + k: v for k, v in options.items()
               if k != 'mode'}})
        before_global = rng_state(np.random)
        if mode == 'reference':
          env = wrap_env(ReferenceDMC(name, repeat=args.repeat, seed=args.seed,
                         image=True, proprio=False), config)
        else:
          env = make_env(config, 0, seed=args.seed)
        actions = np.random.default_rng(103).uniform(
            -0.3, 0.3, (args.steps + 2,) + env.act_space['action'].shape).astype(np.float32)
        frames, states, rewards, boundaries, backgrounds, task_rng = [], [], [], [], [], []
        try:
          # Exercise resets as well as action steps; the second reset must not
          # accumulate alpha blending or consume task RNG differently.
          for index, action in enumerate(actions):
            obs = env.step(dict(action=action, reset=index in reset_indices))
            frame = obs['image']
            assert frame.dtype == np.uint8 and frame.shape == (64, 64, 3)
            frames.append(frame.copy())
            states.append(env._dmenv.physics.get_state().copy())
            rewards.append(obs['reward'])
            boundaries.append([obs[k] for k in ('is_first', 'is_last', 'is_terminal')])
            task_rng.append(rng_state(env._dmenv.task.random))
            if mode != 'reference':
              backgrounds.append(dict(mode='clean') if mode == 'clean' else env.background_state())
            if index == 0:
              with env._dmenv.physics.contexts.gl.make_current() as context:
                # Query the explicitly loaded Mesa library on Windows. PyOpenGL
                # may bind Windows' system OpenGL instead of this isolated DLL.
                get_string = software_gl.glGetString if software_gl else GL.glGetString
                renderer = context.call(get_string, GL.GL_RENDERER).decode()
              report['renderer'] = renderer
              if args.software_gl_dir and 'llvmpipe' not in renderer.lower():
                raise RuntimeError(f'Expected CPU llvmpipe, got {renderer}')
            if mode == 'dynamic-diagnostics':
              original_state = env.background_state()
              paired = env.render_background_frame(next(iter(env._dmenv.dataset.frames)), 0)
              assert paired.shape == frame.shape
              assert original_state == env.background_state()
              np.testing.assert_array_equal(env._dmenv.physics.get_state(), states[-1])
              np.testing.assert_array_equal(
                  env._dmenv.physics.render(64, 64, camera_id=env._camera), frame)
          assert before_global == rng_state(np.random)
          results[mode] = dict(images=np.stack(frames), physics=np.stack(states),
              rewards=np.array(rewards), boundaries=np.array(boundaries),
              background=backgrounds, task_rng=task_rng)
        finally:
          env._dmenv.close()
        for key in ('physics', 'rewards', 'boundaries'):
          np.testing.assert_array_equal(results['reference'][key], results[mode][key])
        assert results['reference']['task_rng'] == results[mode]['task_rng']
      np.testing.assert_array_equal(results['reference']['images'], results['clean']['images'])
      for key in ('images', 'physics', 'rewards', 'boundaries'):
        np.testing.assert_array_equal(results['dynamic'][key], results['dynamic-diagnostics'][key])
      assert results['dynamic']['background'] == results['dynamic-diagnostics']['background']
      differences = {mode: float(np.mean(results[mode]['images'] !=
          results['clean']['images'])) for mode in ('static', 'dynamic')}
      assert all(x > 0 for x in differences.values()), differences
      for mode in ('static', 'dynamic'):
        trace = results[mode]['background']
        for previous, current in zip(trace, trace[1:]):
          if previous['episode'] == current['episode'] and mode == 'static':
            assert previous['frame_index'] == current['frame_index']
      assert len({x['frame_index'] for x in results['dynamic']['background']}) > 1
      row = dict(task=task, accepted=True, changed_pixel_coordinate_fraction=differences,
                 episode_end_count=int(results['clean']['boundaries'][:, 1].sum()),
                 effective_ground_alpha=(GROUND_ALPHA.get(domain)
                     if ground_alpha == -1 else ground_alpha),
                 clean_matches_reference=True, physics_rewards_boundaries_rng_equal=True,
                 diagnostics_do_not_change_future_images_or_rng=True,
                 hashes={mode: {k: digest(v) for k, v in values.items()
                               if isinstance(v, np.ndarray)} for mode, values in results.items()})
      report['tasks'].append(row)
      taskdir = output / task
      taskdir.mkdir()
      for mode, values in results.items():
        np.savez_compressed(taskdir / f'{mode}.npz', **{k: v for k, v in values.items()
                                                      if isinstance(v, np.ndarray)})
        (taskdir / f'{mode}-background.json').write_text(
            json.dumps(values['background'], indent=2), encoding='utf-8')
      # 3 columns x 3 time points, genuine rendered input frames.
      canvas = Image.new('RGB', (64 * 3, 64 * 3))
      for col, mode in enumerate(('clean', 'static', 'dynamic')):
        for row_index, index in enumerate((0, 1, args.steps // 2)):
          canvas.paste(Image.fromarray(results[mode]['images'][index]), (col*64, row_index*64))
      canvas.resize((768, 768), Image.Resampling.NEAREST).save(taskdir / 'conditions.png')
      print(json.dumps(row), flush=True)
    report['accepted'] = True
  finally:
    report['elapsed_seconds'] = time.monotonic() - start
    (output / 'acceptance.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
  main()
