"""Seven never-started seed1 slots moved to sv2; four serial queues."""
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from scripts.run_vision20_reallocated import TRAIN_SHA,AUDITED,next_row
from scripts.run_hopper_seeds123 import read,dump,clean_sha,require_free,child_environment,audit_run,now,sha
from scripts.run_missing_dt import approved_context
from scripts.prepare_m2_v1 import resolve
from scripts.prepare_vision20_seed1 import check_config
from dreamerv3.protocol_scope import VISION20_SEED1

ROOT=Path('/data/Policy_Discrepancy/runs/formal-m2-vision20-seed1-remaining-20261105-sv2')
ENGINEERING=Path('/data/Policy_Discrepancy/runs/engineering-vision20-seed1-remaining-20261105-sv2')
TRAIN_REPO=Path('/data/Policy_Discrepancy/checkouts/vision20-seed1-bef29480df12')
MATRIX=[(0,'dmc_pendulum_swingup','logging'),(0,'dmc_pendulum_swingup','dt'),
        (1,'dmc_walker_walk','logging'),(1,'dmc_walker_walk','dt'),
        (2,'dmc_walker_stand','logging'),(2,'dmc_walker_stand','dt'),
        (3,'dmc_reacher_easy','dt')]


def make_plan(source,proofs,controller_sha):
  rows=[]
  if len(source['runs'])!=7:raise ValueError('Exact seven sealed source slots required')
  for gpu,task,mode in MATRIX:
    old=next(r for r in source['runs'] if (r['task'],r['mode'])==(task,mode))
    if old['state']!='blocked' or old.get('pid') or old['code_sha']!=TRAIN_SHA:
      raise ValueError('Only never-started blocked slots')
    proof=proofs[str(gpu)]
    if (proof['code_sha']!=TRAIN_SHA or proof['physical_gpu']!=gpu or
        not proof['verified_compute_and_graphics_same_gpu']):raise ValueError('Wrong device proof')
    row=copy.deepcopy(old);d=str(ROOT/task/(mode+'-moved01'))
    argv=list(old['review_argv']);argv[argv.index('--logdir')+1]=d
    config=resolve(argv[4:]);check_config(config);flat=json.loads(json.dumps(dict(config.flat)))
    if flat!=dict(old['config'],logdir=d):raise ValueError('Only logdir may change')
    for k in ('error','exit_code','pid','run_audit'):row.pop(k,None)
    row.update(server='sv2',host='lyg0360',physical_gpu=gpu,gpu_uuid=proof['gpu_uuid'],
        egl_index=proof['egl_index'],directory=d,config=flat,review_argv=argv,launch_argv=argv,
        state='queued',source_run_id=old['run_id'],source_directory=old['directory'],
        run_id=old['run_id']+':moved-sv2-20261105',queue_position=1+sum(r['physical_gpu']==gpu for r in rows),
        verified_compute_and_graphics_same_gpu=True)
    row['environment_template'].update(CUDA_VISIBLE_DEVICES=proof['gpu_uuid'],
        MUJOCO_EGL_DEVICE_ID=str(proof['egl_index']),TMPDIR=d+'/tmp',WANDB_DIR=d,
        WANDB_RUN_ID=old['environment_template']['WANDB_RUN_ID']+'-moved-sv2-20261105')
    rows.append(row)
  return dict(root=str(ROOT),code_sha=controller_sha,training_sha=TRAIN_SHA,
      training_checkout=str(TRAIN_REPO),protocol=VISION20_SEED1,authorized_to_train=True,runs=rows)


def validate_seal(source,seal):
  ids={r['run_id'] for r in source['runs']}
  if (not seal['passed'] or set(seal['claimed_source_run_ids'])!=ids or len(ids)!=7 or
      seal['target_root']!=str(ROOT) or seal['source_sha256']!=sha(ENGINEERING/'source.json')):
    raise RuntimeError('Missing cross-server never-started source claims/hash')


def run(plan_path):
  p=read(plan_path);source=read(ENGINEERING/'source.json');proofs=read(ENGINEERING/'proofs.json')
  validate_seal(source,read(ENGINEERING/'source-claims.json'))
  if platform.node()!='lyg0360' or p!=make_plan(source,proofs,clean_sha()):raise ValueError('Host/SHA/plan differs')
  if (subprocess.check_output(['git','-C',str(TRAIN_REPO),'rev-parse','HEAD'],text=True).strip()!=TRAIN_SHA or
      subprocess.check_output(['git','-C',str(TRAIN_REPO),'status','--porcelain'])):raise RuntimeError('Training checkout changed')
  a=read(ENGINEERING/'integration/acceptance.json')
  first={(t,m,g) for g,t,m in MATRIX if m=='logging' or t=='dmc_reacher_easy'}
  if not a['passed'] or a['code_sha']!=TRAIN_SHA or len(a['runs'])!=4 or {(r['task'],r['mode'],r['physical_gpu']) for r in a['runs']}!=first:raise RuntimeError('Missing four-card actual CUDA acceptance')
  if ROOT.exists():raise FileExistsError('No overwrite/resume')
  checks=[require_free(g,proofs[str(g)]['gpu_uuid']) for g in range(4)]
  if shutil.disk_usage(ROOT.parent).free<240_000_000_000:raise RuntimeError('Initial four-card 240GB floor')
  ROOT.mkdir();dump(ROOT/'bound-plan.json',p);dump(ROOT/'source-claims.json',read(ENGINEERING/'source-claims.json'))
  dump(ROOT/'prelaunch.json',dict(utc=now(),checks=checks,supervisor_pid=os.getpid(),free_bytes=shutil.disk_usage(ROOT).free,initial_floor=240_000_000_000,per_run_floor=60_000_000_000))
  handles=[]
  while True:
    for row,proc,log in handles:
      if row['state']!='running' or proc.poll() is None:continue
      log.close();row.update(state='failed',exit_code=proc.returncode,end_utc=now())
      if proc.returncode==0:
        try:row['run_audit']=audit_run(row,VISION20_SEED1);row['state']=AUDITED
        except Exception as e:row.update(state='audit_failed',error=repr(e))
      dump(Path(row['directory'])/'execution-final.json',row)
    for g in range(4):
      row=next_row(p['runs'],g)
      if row is None:continue
      try:
        check=require_free(g,row['gpu_uuid'])
        if shutil.disk_usage(ROOT).free<60_000_000_000:raise RuntimeError('Below 60GB per-run floor')
        d=Path(row['directory']);d.mkdir(parents=True,exist_ok=False);(d/'tmp').mkdir()
        log=(d/'stdout-initial.log').open('x')
        try:proc=subprocess.Popen(row['launch_argv'],cwd=TRAIN_REPO,env=child_environment(row['environment_template'],inherited=approved_context()),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        except BaseException:log.close();raise
        row.update(state='running',pid=proc.pid,start_utc=now(),prelaunch_check=check)
        handles.append((row,proc,log));dump(d/'execution-initial.json',row)
      except Exception as e:row.update(state='blocked',error=repr(e))
    dump(ROOT/'actual-RUN_INDEX.json',dict(runs=p['runs']))
    if not any(r['state'] in ('queued','running') for r in p['runs']):break
    time.sleep(30)
  dump(ROOT/'supervisor-final.json',dict(cohort_complete=False,runs=p['runs'],utc=now()))


if __name__=='__main__':
  a=argparse.ArgumentParser(description=__doc__);a.add_argument('--plan',required=True);run(a.parse_args().plan)
