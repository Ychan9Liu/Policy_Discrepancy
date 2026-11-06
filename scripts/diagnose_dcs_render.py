"""Collect exact render controls; never train or relax the strict DCS probe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--runtime-repo',type=Path,required=True)
  parser.add_argument('--dataset',required=True)
  parser.add_argument('--task',default='dmc_hopper_hop')
  parser.add_argument('--seed',type=int,default=0)
  parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args()
  runtime=args.runtime_repo.resolve(strict=True)
  sys.path.insert(0,str(runtime))
  os.environ.update(MUJOCO_GL='egl',JAX_PLATFORMS='cpu')
  import numpy as np
  from OpenGL import GL
  from dreamerv3.main import make_env
  from scripts.prepare_m2_v1 import resolve
  from embodied.envs.dcs_background import GROUND_ALPHA
  args.output.mkdir(parents=True,exist_ok=False)
  def git(repo,*command):
    return subprocess.check_output(['git',*command],cwd=repo,text=True).strip()
  assert not git(runtime,'status','--porcelain')
  source=Path(__file__).resolve().parents[1]
  report=dict(test_sha=git(source,'rev-parse','HEAD'),runtime_sha=git(runtime,'rev-parse','HEAD'),
      task=args.task,seed=args.seed,engineering_only=True,agent_updates=0,
      numerical_tolerance='exact array equality; differences retained, never waived',cases={})
  results={}
  domain=args.task.split('_')[1]
  ground=GROUND_ALPHA.get('ball_in_cup' if domain=='cup' else domain,.3)
  def diff(a,b):
    delta=np.asarray(a,dtype=np.int64)-np.asarray(b,dtype=np.int64)
    return dict(equal=bool(np.array_equal(a,b)),different=int(np.count_nonzero(delta)),
                total=int(delta.size),max_abs=int(np.abs(delta).max(initial=0)))
  def rng(r):
    a=r.get_state();return [a[0],a[1].tolist(),int(a[2]),int(a[3]),float(a[4])]
  for mode in ('clean','dynamic'):
    arms=('none','repeat') if mode=='clean' else ('none','repeat','reupload','override','no-dither')
    for arm in arms:
      config=resolve(['--configs','dcs_'+mode,'--task',args.task])
      config=config.update({'env.dmc.background.dataset':args.dataset,
          'env.dmc.background.videos':'train','env.dmc.background.seed':-1,
          'env.dmc.background.ground_plane_alpha':ground})
      env=make_env(config,0,seed=args.seed)
      dither=[]
      if arm=='no-dither':
        with env._dmenv.physics.contexts.gl.make_current() as context:
          context.call(GL.glDisable,GL.GL_DITHER)
      actions=np.random.default_rng(103).uniform(-.3,.3,(18,)+env.act_space['action'].shape).astype(np.float32)
      frames=[]; states=[]; rewards=[]; task_rng=[]; backgrounds=[]; comparisons=[]
      immediate=[]; restored=[]
      try:
        for index,action in enumerate(actions):
          obs=env.step(dict(action=action,reset=index in (0,9)))
          with env._dmenv.physics.contexts.gl.make_current() as context:
            dither.append(bool(context.call(GL.glIsEnabled,GL.GL_DITHER)))
          frames.append(obs['image'].copy())
          physics=env._dmenv.physics
          states.append(physics.get_state().copy())
          rewards.append(obs['reward'])
          task_rng.append(rng(env._dmenv.task.random))
          bg=env.background_state() if mode=='dynamic' else {}
          backgrounds.append(bg)
          if arm=='none': continue
          before=physics.render(64,64,camera_id=env._camera)
          immediate.append(before.copy())
          tex=physics.model.tex_data.copy()
          if arm=='reupload':
            wrapper=env._dmenv
            wrapper._upload(tex[wrapper._address:wrapper._address+wrapper._size])
          elif arm=='override':
            env.render_background_frame(next(iter(env._dmenv.dataset.frames)),0)
          after=physics.render(64,64,camera_id=env._camera)
          restored.append(after.copy())
          comparisons.append(dict(index=index,observation_to_rerender=diff(frames[-1],before),
              rerender_to_after=diff(before,after),texture_equal=bool(np.array_equal(tex,physics.model.tex_data)),
              physics_equal=bool(np.array_equal(states[-1],physics.get_state())),
              background_rng_equal=(bg==env.background_state()) if mode=='dynamic' else True))
        results[mode,arm]=dict(images=np.stack(frames),physics=np.stack(states),
            rewards=np.asarray(rewards),task_rng=task_rng,background=backgrounds)
        record=dict(comparisons=comparisons,dither=dither)
        if arm!='none':
          reference=results[mode,'none']
          record.update(future_images=diff(reference['images'],results[mode,arm]['images']),
              physics_equal=bool(np.array_equal(reference['physics'],results[mode,arm]['physics'])),
              rewards_equal=bool(np.array_equal(reference['rewards'],results[mode,arm]['rewards'])),
              task_rng_equal=reference['task_rng']==task_rng,
              background_equal=reference['background']==backgrounds)
        report['cases'][mode+'-'+arm]=record
        np.savez_compressed(args.output/(mode+'-'+arm+'.npz'),images=np.stack(frames),
            physics=np.stack(states),rewards=np.asarray(rewards),
            immediate=np.asarray(immediate),after=np.asarray(restored))
      finally:
        env._dmenv.close()
  (args.output/'diagnostic.json').write_text(json.dumps(report,indent=2))
  print(json.dumps({name:{k:v for k,v in row.items() if k!='comparisons'}
      for name,row in report['cases'].items()},indent=2))


if __name__=='__main__': main()
