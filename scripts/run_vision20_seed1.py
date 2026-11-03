"""Approved seed1 task pairs; one job per assigned GPU, no resume or idle retry."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from dreamerv3.protocol_scope import VISION20_SEED1
from scripts.prepare_vision20_seed1 import CARDS,NEW_TASKS,HOSTS,check_config
from scripts.prepare_m2_v1 import resolve
from scripts.run_hopper_seeds123 import (
    REPO,read,dump,sha,now,clean_sha,require_free,audit_run,child_environment,probe)
from scripts.run_missing_dt import approved_context
from scripts.run_dmc8_serial import integration

AUDITED='run_audited_pending_group_and_WB'


def logger_check(output,server):
  output=Path(output);output.mkdir(parents=True,exist_ok=False);commit=clean_sha()
  os.environ.update(WANDB_PROJECT='PD_1',WANDB_ENTITY='ychan9liu-tsinghua-university',
      WANDB_MODE='online',WANDB_RESUME='never',WANDB_DIR=str(output),
      WANDB_RUN_GROUP='logger-engineering-validation',
      WANDB_RUN_ID='logger-vision20-seed1-20261103-'+server+'-'+commit[:7])
  from dreamerv3.main import make_logger
  import wandb
  c=resolve(['--configs','m2_vision20_seed1','--logdir',str(output),
             '--jax.prealloc','False','--logger.outputs','jsonl','wandb'])
  check_config(c);logger=make_logger(c);remote=wandb.run
  path=remote.entity+'/PD_1/'+remote.id
  if remote.config['seed']!=1:raise RuntimeError('Wrong logger seed')
  logger.add(dict(logger_validation=1),prefix='engineering');logger.write();logger.close();wandb.finish()
  remote=wandb.Api().run(path)
  if remote.state!='finished' or remote.summary['engineering/logger_validation']!=1:
    raise RuntimeError('Online readback failed')
  dump(output/'logger_validation.json',dict(code_sha=commit,remote_state=remote.state,
      seed=1,entity=remote.entity,project=remote.project,url=remote.url,training_actions=0,
      agent_updates=0,jsonl_sha256=sha(output/'metrics.jsonl')))


def validate(plan):
  server=plan['server'];rows=plan['runs']
  expected=[(t,m,g,2*(i//len(CARDS))+j+1) for i,t in enumerate(NEW_TASKS)
      for j,m in enumerate(('logging','dt')) for s,g in [CARDS[i%len(CARDS)]] if s==server]
  if (plan['protocol']!=VISION20_SEED1 or plan['host']!=HOSTS[server] or
      [(r['task'],r['mode'],r['physical_gpu'],r['queue_position']) for r in rows]!=expected or
      plan['root']!='/data/Policy_Discrepancy/runs/'+plan['campaign']+'-'+server):
    raise ValueError('Wrong fixed host/task/group/GPU queue')
  for r in rows:
    c=resolve(r['review_argv'][4:]);check_config(c)
    if (r['code_sha']!=plan['code_sha'] or r['host']!=plan['host'] or r['server']!=server or
        c.seed!=1 or r['seed']!=1 or c.task!=r['task'] or c.agent.rep_probe.mode!=r['mode'] or
        c.logdir!=r['directory'] or r['directory']!=plan['root']+'/'+r['task']+'/'+r['mode']+'-attempt01' or
        json.loads(json.dumps(dict(c.flat)))!=r['config']):raise ValueError('Config differs')
    e=r['environment_template']
    if (e['CUDA_VISIBLE_DEVICES']!=r['gpu_uuid'] or e['WANDB_MODE']!='online' or
        e['WANDB_PROJECT']!='PD_1' or e['WANDB_ENTITY']!='ychan9liu-tsinghua-university' or
        e['WANDB_RESUME']!='never' or e['WANDB_DIR']!=r['directory']):raise ValueError('Identity differs')
    if plan['authorized_to_train'] and (r['launch_argv']!=r['review_argv'] or
        not r['verified_compute_and_graphics_same_gpu'] or
        e['MUJOCO_EGL_DEVICE_ID']!=str(r['egl_index'])):raise ValueError('Missing binding')
  for k in ('directory','run_id'):
    if len({r[k] for r in rows})!=len(rows):raise ValueError('Duplicate identity')


def split(source,server,output):
  p=read(source)
  if Path(output).exists():raise FileExistsError(output)
  plan=dict(protocol=p['protocol'],code_sha=p['code_sha'],campaign=p['campaign'],
      root='/data/Policy_Discrepancy/runs/'+p['campaign']+'-'+server,server=server,
      host=HOSTS[server],authorized_to_train=False,runs=[r for r in p['runs'] if r['server']==server],
      source_plan_sha256=sha(source),reused_hopper_runs=p['reused_hopper_runs'])
  validate(plan);dump(output,plan)


def bind(plan_path,evidence,output):
  plan=read(plan_path);validate(plan);ev=Path(evidence)
  if (Path(output).exists() or clean_sha()!=plan['code_sha'] or
      platform.node()!=plan['host']):raise RuntimeError('Wrong SHA/host/existing output')
  a=read(ev/'integration/acceptance.json');l=read(ev/'logger-check/logger_validation.json')
  if (not a['passed'] or a['code_sha']!=plan['code_sha'] or
      {(r['task'],r['mode']) for r in a['runs']}!={(r['task'],r['mode']) for r in plan['runs']} or
      l['code_sha']!=plan['code_sha'] or l['remote_state']!='finished'):
    raise ValueError('Missing exact task CUDA/logger acceptance')
  for r in plan['runs']:
    p=read(ev/f'gpu{r["physical_gpu"]}-binding.json')
    if (p['code_sha']!=plan['code_sha'] or p['gpu_uuid']!=r['gpu_uuid'] or
        p['physical_gpu']!=r['physical_gpu'] or not p['verified_compute_and_graphics_same_gpu']):
      raise ValueError('Unverified device')
    r.update(launch_argv=r['review_argv'],egl_index=p['egl_index'],
        verified_compute_and_graphics_same_gpu=True,authorized_to_train=True,state='queued')
    r['environment_template']['MUJOCO_EGL_DEVICE_ID']=str(p['egl_index'])
  plan['authorized_to_train']=True;validate(plan);dump(output,plan)


def run(plan_path, accept_initial_capacity_deficit=False):
  p=read(plan_path);validate(p);root=Path(p['root'])
  if accept_initial_capacity_deficit and p['server']!='sv1':
    raise ValueError('Initial capacity exception is approved only for sv1')
  if root.exists():raise FileExistsError('No overwrite/resume')
  if not p['authorized_to_train'] or platform.node()!=p['host'] or clean_sha()!=p['code_sha']:
    raise RuntimeError('Wrong authorization/host/clean pinned SHA')
  cards=sorted({r['physical_gpu'] for r in p['runs']})
  checks=[require_free(g,next(r['gpu_uuid'] for r in p['runs'] if r['physical_gpu']==g)) for g in cards]
  free=shutil.disk_usage(root.parent).free;required=len(cards)*60_000_000_000
  deficit=free<required
  if deficit and (not accept_initial_capacity_deficit or free<60_000_000_000):
    raise RuntimeError('Insufficient initial capacity; no deletion or protocol changes')
  root.mkdir();dump(root/'bound-plan.json',p);dump(root/'prelaunch.json',dict(utc=now(),checks=checks,supervisor_pid=os.getpid()))
  dump(root/'capacity-at-start.json',dict(free_bytes=free,normal_reservation_bytes=required,
      reservation_satisfied=not deficit,explicit_sv1_capacity_exception=accept_initial_capacity_deficit,
      per_run_floor_bytes=60_000_000_000,
      authorization='User 2026-11-04: start directly; I will continue cleaning' if accept_initial_capacity_deficit else None))
  handles=[]
  while True:
    for r,proc,log in handles:
      if r['state']!='running' or proc.poll() is None:continue
      log.close();r.update(state='failed',exit_code=proc.returncode,end_utc=now())
      if proc.returncode==0:
        try:r['run_audit']=audit_run(r,VISION20_SEED1);r['state']=AUDITED
        except Exception as e:r.update(state='audit_failed',error=repr(e))
      dump(Path(r['directory'])/'execution-final.json',r)
    for g in cards:
      group=[r for r in p['runs'] if r['physical_gpu']==g]
      if any(r['state']=='running' for r in group):continue
      row=next((r for r in group if r['state']=='queued'),None)
      if row is None:continue
      prior=[r for r in group if r['queue_position']<row['queue_position']]
      if any(r['state']!=AUDITED for r in prior):
        row.update(state='blocked',error='Previous assigned job not audited');continue
      d=Path(row['directory'])
      try:
        require_free(g,row['gpu_uuid'])
        if shutil.disk_usage(root).free<60_000_000_000:raise RuntimeError('Capacity below per-run floor')
        d.mkdir(parents=True,exist_ok=False);(d/'tmp').mkdir();log=(d/'stdout-initial.log').open('x')
        try:proc=subprocess.Popen(row['launch_argv'],cwd=REPO,
            env=child_environment(row['environment_template'],inherited=approved_context()),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        except BaseException:log.close();raise
        row.update(state='running',pid=proc.pid,process_group_id=proc.pid,start_utc=now())
        handles.append((row,proc,log));dump(d/'execution-initial.json',row)
      except Exception as e:row.update(state='blocked',error=repr(e))
    dump(root/'actual-RUN_INDEX.json',dict(runs=p['runs']))
    if not any(r['state'] in ('running','queued') for r in p['runs']):break
    time.sleep(30)
  # Per-host metrics stay incomplete for the twenty-task cohort. No false completion.
  out=root/'return-audit-attempt01'
  rc=subprocess.run([sys.executable,'-m','analysis.return_metrics','--profile','vision20-seed1',
      '--index',str(root/'actual-RUN_INDEX.json'),'--output',str(out)],cwd=REPO).returncode
  dump(root/'supervisor-final.json',dict(utc=now(),all_training_exited=True,
      host_training_audited=all(r['state']==AUDITED for r in p['runs']),
      cohort_complete=False,return_audit_exit_code=rc,runs=p['runs']))


if __name__=='__main__':
  a=argparse.ArgumentParser(description=__doc__)
  a.add_argument('action',choices=('split','probe','logger','integration','bind','run'))
  for n in ('plan','evidence','output'):a.add_argument('--'+n,type=Path)
  a.add_argument('--server',choices=HOSTS);a.add_argument('--gpu',type=int);a.add_argument('--uuid')
  a.add_argument('--accept-initial-capacity-deficit',action='store_true')
  x=a.parse_args()
  if x.action=='split':split(x.plan,x.server,x.output)
  elif x.action=='probe':probe(x.gpu,x.uuid,x.output,host=HOSTS[x.server],allowed_gpus=[g for s,g in CARDS if s==x.server])
  elif x.action=='logger':logger_check(x.output,x.server)
  elif x.action=='integration':
    p=read(x.plan);integration(x.plan,x.evidence,validator=validate,host=p['host'],new_session=True)
  elif x.action=='bind':bind(x.plan,x.evidence,x.output)
  else:run(x.plan,x.accept_initial_capacity_deficit)
