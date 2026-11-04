"""User-authorized six blocked seed1 runs on sv1 GPU0/1/4; pinned training."""
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

from dreamerv3.protocol_scope import VISION20_SEED1
from scripts.prepare_vision20_seed1 import check_config
from scripts.prepare_m2_v1 import resolve
from scripts.run_hopper_seeds123 import (
    read, dump, clean_sha, require_free, child_environment, audit_run, now)
from scripts.run_missing_dt import approved_context

TRAIN_SHA = 'bef29480df12d07659c640b22218804dd1d69e63'
TRAIN_REPO = Path('/data/Policy_Discrepancy/checkouts/vision20-seed1-bef29480df12')
SOURCE = Path('/data/Policy_Discrepancy/runs/formal-m2-vision20-seed1-20261103-sv1/actual-RUN_INDEX.json')
ROOT = Path('/data/Policy_Discrepancy/runs/formal-m2-vision20-seed1-reallocated-20261104-sv1')
ENGINEERING = Path('/data/Policy_Discrepancy/runs/engineering-vision20-seed1-reallocated-20261104-sv1')
MATRIX = [(0, 'dmc_quadruped_run', 'dt'), (0, 'dmc_reacher_hard', 'dt'),
          (1, 'dmc_cup_catch', 'dt'), (1, 'dmc_reacher_easy', 'logging'),
          (4, 'dmc_hopper_stand', 'logging'), (4, 'dmc_hopper_stand', 'dt')]
AUDITED = 'run_audited_pending_group_and_WB'


def make_plan(source, proofs, controller_sha):
  rows = []
  for gpu, task, mode in MATRIX:
    old = next(r for r in source['runs'] if (r['task'], r['mode']) == (task, mode))
    if old['state'] != 'blocked' or old.get('pid') or old['code_sha'] != TRAIN_SHA:
      raise ValueError('Only original never-started blocked slots may be replaced')
    proof = proofs[str(gpu)]
    if (proof['physical_gpu'] != gpu or proof['code_sha'] != TRAIN_SHA or
        not proof['verified_compute_and_graphics_same_gpu']):
      raise ValueError('Exact training SHA device proof required')
    row = copy.deepcopy(old)
    directory = str(ROOT/task/(mode+'-reallocated01'))
    argv = list(old['review_argv']); argv[argv.index('--logdir')+1] = directory
    config = json.loads(json.dumps(dict(resolve(argv[4:]).flat))); check_config(resolve(argv[4:]))
    expected = dict(old['config'], logdir=directory)
    if config != expected: raise ValueError('Only logdir may change in training configuration')
    for key in ('error', 'exit_code', 'pid', 'run_audit'): row.pop(key, None)
    row.update(directory=directory, config=config, review_argv=argv, launch_argv=argv,
        source_directory=old['directory'], source_run_id=old['run_id'],
        physical_gpu=gpu, gpu_uuid=proof['gpu_uuid'], egl_index=proof['egl_index'],
        queue_position=1+sum(r['physical_gpu']==gpu for r in rows), state='queued',
        run_id=old['run_id']+':reallocated01', verified_compute_and_graphics_same_gpu=True)
    row['environment_template'].update(CUDA_VISIBLE_DEVICES=proof['gpu_uuid'],
        MUJOCO_EGL_DEVICE_ID=str(proof['egl_index']), TMPDIR=directory+'/tmp', WANDB_DIR=directory,
        WANDB_RUN_ID=old['environment_template']['WANDB_RUN_ID']+'-reallocated01')
    rows.append(row)
  return dict(root=str(ROOT), source=str(SOURCE), code_sha=controller_sha,
      training_sha=TRAIN_SHA, training_checkout=str(TRAIN_REPO), protocol=VISION20_SEED1,
      authorized_to_train=True, runs=rows)


def next_row(rows, gpu):
  group=[r for r in rows if r['physical_gpu']==gpu]
  if any(r['state']=='running' for r in group): return None
  row=next((r for r in group if r['state']=='queued'),None)
  if row and any(r['state']!=AUDITED for r in group if r['queue_position']<row['queue_position']):
    row.update(state='blocked',error='Predecessor not audited'); return None
  return row


def verify_source(row):
  source=next(r for r in read(SOURCE)['runs'] if r['run_id']==row['source_run_id'])
  if (source['state']!='blocked' or source.get('pid') or
      source['directory']!=row['source_directory'] or Path(source['directory']).exists()):
    raise RuntimeError('Source started/changed or existing artifacts; do not duplicate')


def run(plan_path, proofs_path):
  p=read(plan_path);proofs=read(proofs_path)
  if p!=make_plan(read(SOURCE),proofs,clean_sha()) or platform.node()!='lyg2153':
    raise ValueError('Plan differs from exact six-slot authorization/clean controller SHA/host')
  actual=subprocess.check_output(['git','-C',str(TRAIN_REPO),'rev-parse','HEAD'],text=True).strip()
  if actual!=TRAIN_SHA or subprocess.check_output(['git','-C',str(TRAIN_REPO),'status','--porcelain']):
    raise RuntimeError('Training checkout changed')
  acceptance=read(ENGINEERING/'integration/acceptance.json')
  first={(t,m,g) for g,t,m in MATRIX[::2]}
  if (not acceptance['passed'] or acceptance['code_sha']!=TRAIN_SHA or
      len(acceptance['runs'])!=3 or
      {(r['task'],r['mode'],r['physical_gpu']) for r in acceptance['runs']}!=first):
    raise RuntimeError('Missing exact three-card real CUDA acceptance')
  root=Path(p['root'])
  if root.exists(): raise FileExistsError('No overwrite/resume')
  checks=[require_free(g,proofs[str(g)]['gpu_uuid']) for g in (0,1,4)]
  if shutil.disk_usage(root.parent).free<180_000_000_000:
    raise RuntimeError('Three-card initial 180GB reservation not satisfied')
  for row in p['runs']: verify_source(row)
  root.mkdir();dump(root/'bound-plan.json',p)
  dump(root/'prelaunch.json',dict(checks=checks,utc=now(),supervisor_pid=os.getpid(),
      free_bytes=shutil.disk_usage(root).free,initial_floor=180_000_000_000,per_run_floor=60_000_000_000))
  # Original supervisor never retries blocked slots. Keep its index/history unchanged.
  claims=SOURCE.parent/'reallocation-claims';claims.mkdir(exist_ok=True)
  for row in p['runs']:
    with (claims/(row['task']+'-'+row['mode']+'.json')).open('x') as f:
      json.dump(dict(source_run_id=row['source_run_id'],replacement_run_id=row['run_id'],
          replacement_root=str(root),controller_sha=p['code_sha'],training_sha=TRAIN_SHA),f)
  handles=[]
  while True:
    for row,proc,log in handles:
      if row['state']!='running' or proc.poll() is None: continue
      log.close();row.update(state='failed',exit_code=proc.returncode,end_utc=now())
      if proc.returncode==0:
        try:row['run_audit']=audit_run(row,VISION20_SEED1);row['state']=AUDITED
        except Exception as e:row.update(state='audit_failed',error=repr(e))
      dump(Path(row['directory'])/'execution-final.json',row)
    for gpu in (0,1,4):
      row=next_row(p['runs'],gpu)
      if row is None: continue
      try:
        verify_source(row);check=require_free(gpu,row['gpu_uuid'])
        if shutil.disk_usage(root).free<60_000_000_000: raise RuntimeError('Below 60GB per-new-run floor')
        d=Path(row['directory']);d.mkdir(parents=True,exist_ok=False);(d/'tmp').mkdir()
        log=(d/'stdout-initial.log').open('x')
        try:
          proc=subprocess.Popen(row['launch_argv'],cwd=TRAIN_REPO,
              env=child_environment(row['environment_template'],inherited=approved_context()),
              stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        except BaseException:log.close();raise
        row.update(state='running',pid=proc.pid,start_utc=now(),prelaunch_check=check)
        handles.append((row,proc,log));dump(d/'execution-initial.json',row)
      except Exception as e: row.update(state='blocked',error=repr(e))
    dump(root/'actual-RUN_INDEX.json',dict(runs=p['runs']))
    if not any(r['state'] in ('queued','running') for r in p['runs']): break
    time.sleep(30)
  dump(root/'supervisor-final.json',dict(cohort_complete=False,runs=p['runs'],utc=now()))


if __name__=='__main__':
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--plan',required=True);parser.add_argument('--proofs',required=True)
  args=parser.parse_args();run(args.plan,args.proofs)
