"""Launch only the eleven never-started Dt slots on the 2026-10-30 allocation.

Old indices and completed baselines are immutable provenance. No resume, baseline
retraining, resource substitution, or scientific configuration override.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import time

from dreamerv3.protocol_scope import DMC16, DMC16_TASKS
from scripts.prepare_dmc16 import PYTHON, check_dmc16, source_state
from scripts.prepare_m2_v1 import resolve
from scripts import run_dmc8_serial as serial
from scripts.run_hopper_seeds123 import (
    REPO, ENTITY, read, dump, sha, now, clean_sha, require_free, child_environment, audit_run, probe)

TASKS = tuple(DMC16_TASKS[:6]) + ('dmc_hopper_stand', 'dmc_pendulum_swingup',
    'dmc_reacher_easy', 'dmc_walker_run', 'dmc_walker_walk')
HOSTS = {'sv2':'lyg0360', 'sv3':'lyg0326'}
MATRIX = dict(zip(TASKS, [('sv2',g) for g in (0,1,2,3,4,5,7)] +
                        [('sv3',g) for g in (4,5,6,7)]))
PREFIX = 'formal-m2-dmc16-missing-dt-'
AUDITED = 'run_audited_pending_group_and_WB'


def approved_context(inherited=None):
  env = dict(os.environ if inherited is None else inherited)
  for key in ('LD_LIBRARY_PATH','LD_PRELOAD','CUDA_HOME','CUDA_PATH','PYTHONPATH'):
    env.pop(key, None)
  env['PATH'] = '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
  return env


def validate_plan(plan):
  server = plan['server']
  if server not in HOSTS or plan['host'] != HOSTS[server] or plan['protocol'] != DMC16:
    raise ValueError('Wrong fixed host/protocol')
  expected = [t for t in TASKS if MATRIX[t][0] == server]
  root = PurePosixPath(plan['root'])
  if (not plan['campaign'].startswith(PREFIX) or '/' in plan['campaign'] or
      '..' in plan['campaign'] or root.parent != PurePosixPath('/data/Policy_Discrepancy/runs') or
      root.name != plan['campaign']+'-'+server or len(plan['runs']) != len(expected) or
      [r['task'] for r in plan['runs']] != expected):
    raise ValueError('Only the fixed eleven missing slots in a fresh root')
  for r in plan['runs']:
    task = r['task']; old = r['source_blocked']; base = r['source_baseline']
    if (MATRIX[task] != (server,r['physical_gpu']) or r['server'] != server or
        r['host'] != HOSTS[server] or r['mode'] != 'dt' or type(r['seed']) is not int or
        r['seed'] != 0 or r['code_sha'] != plan['code_sha'] or
        old['task'] != task or old['mode'] != 'dt' or old['state'] != 'blocked' or
        old.get('pid') is not None or old.get('start_utc') is not None or
        base['task'] != task or base['mode'] != 'logging' or base['seed'] != 0 or
        base['state'] not in (AUDITED,'accepted') or base['run_audit']['updates'] != 249493 or
        base['run_audit']['evaluations'] != 21 or
        not math.isfinite(base['run_audit']['c']) or not 0 < base['run_audit']['c'] <= 1):
      raise ValueError('Not an unstarted Dt with its completed own baseline')
    expected_dir = str(root/task/'dt-attempt02')
    argv = r['review_argv']; config = resolve(argv[4:]); check_dmc16(config)
    flat = json.loads(json.dumps(dict(config.flat)))
    if (argv[:4] != [PYTHON,'-u','-m','dreamerv3.main'] or r['directory'] != expected_dir or
        config.logdir != expected_dir or config.task != task or config.agent.rep_probe.mode != 'dt' or
        flat != r['config'] or {k for k in flat if flat[k] != old['config'][k]} != {'logdir'} or
        {k for k in old['config'] if old['config'][k] != base['config'][k]} !=
          {'logdir','agent.rep_probe.mode'}):
      raise ValueError('Scientific configuration changed from original slot/baseline')
    e = r['environment_template']
    if (e['CUDA_VISIBLE_DEVICES'] != r['gpu_uuid'] or e['WANDB_PROJECT'] != 'PD_1' or
        e['WANDB_ENTITY'] != ENTITY or e['WANDB_MODE'] != 'online' or e['WANDB_RESUME'] != 'never' or
        e['JAX_PLATFORMS'] != 'cuda' or e['WANDB_DIR'] != expected_dir or e['TMPDIR'] != expected_dir+'/tmp'):
      raise ValueError('Device, online destination or output mismatch')
    if plan['authorized_to_train'] and (r['launch_argv'] != argv or
        not r['verified_compute_and_graphics_same_gpu'] or
        e['MUJOCO_EGL_DEVICE_ID'] != str(r['egl_index'])):
      raise ValueError('Unverified launch binding')
  for key in ('directory','run_id','gpu_uuid'):
    if len({r[key] for r in plan['runs']}) != len(expected):
      raise ValueError('Duplicate run/output/device')
  if len({r['environment_template']['WANDB_RUN_ID'] for r in plan['runs']}) != len(expected):
    raise ValueError('Duplicate W&B identity')


def prepare(output, resources, sources, campaign):
  output = Path(output)
  if output.exists(): raise FileExistsError(output)
  commit,dirty = source_state(); inventory = read(resources)['servers']
  indices = [read(p) for p in sources]; rows = [r for s in indices for r in s['runs']]
  output.mkdir(parents=True)
  for server,host in HOSTS.items():
    actual = inventory[server]
    if actual['host'] != host: raise ValueError('Host inventory differs')
    devices = {int(line.split(',')[0]):line.split(',')[1].strip()
               for line in actual['gpus'].splitlines()}
    selected = []
    for task in [t for t in TASKS if MATRIX[t][0] == server]:
      old = next(r for r in rows if r['task']==task and r['mode']=='dt')
      baseline = next(r for r in rows if r['task']==task and r['mode']=='logging')
      r = copy.deepcopy(old); gpu = MATRIX[task][1]
      directory = '/data/Policy_Discrepancy/runs/'+campaign+'-'+server+'/'+task+'/dt-attempt02'
      argv = [PYTHON,'-u','-m','dreamerv3.main','--configs','m2_dmc16','--task',task,
          '--seed','0','--logdir',directory,'--agent.rep_probe.mode','dt','--jax.prealloc','False',
          '--logger.outputs','jsonl','wandb']
      config = resolve(argv[4:])
      env = copy.deepcopy(old['environment_template'])
      env.update(CUDA_VISIBLE_DEVICES=devices[gpu], MUJOCO_EGL_DEVICE_ID=None,
          TMPDIR=directory+'/tmp',WANDB_DIR=directory,WANDB_RUN_GROUP=campaign,
          WANDB_RUN_ID=campaign+'-'+task[4:]+'-seed0-dt-02',WANDB_RESUME='never')
      r = dict(task=task,mode='dt',group='dt',seed=0,server=server,host=host,
          physical_gpu=gpu,gpu_uuid=devices[gpu],protocol=DMC16,code_sha=commit,
          directory=directory,config=json.loads(json.dumps(dict(config.flat))),
          review_argv=argv,launch_argv=None,environment_template=env,
          run_id=f'{server}:{campaign}:{task}:seed0:dt:attempt02',state='planned',
          authorized_to_train=False,verified_compute_and_graphics_same_gpu=False,
          egl_index=None,source_blocked=old,source_baseline=baseline)
      selected.append(r)
    plan = dict(server=server,host=host,protocol=DMC16,campaign=campaign,
        root='/data/Policy_Discrepancy/runs/'+campaign+'-'+server,code_sha=commit,
        authorized_to_train=False,runs=selected,created_utc=now(),local_drafts=dirty,
        source_index_hashes={str(p):sha(p) for p in sources},resource_hash=sha(resources),
        authorization='User: sv2 seven GPU0-5/7, exclude6; sv3 four GPU4-7, remaining eleven Dt (2026-10-30)')
    validate_plan(plan); dump(output/(server+'-plan.json'),plan)


def bind(plan_path, evidence, output):
  plan = read(plan_path); evidence=Path(evidence); validate_plan(plan)
  if platform.node()!=plan['host'] or clean_sha()!=plan['code_sha'] or Path(output).exists():
    raise RuntimeError('Wrong host/SHA or existing output')
  acceptance=read(evidence/'integration/acceptance.json');logger=read(evidence/'logger-check/logger_validation.json')
  if (acceptance['code_sha']!=plan['code_sha'] or not acceptance['passed'] or
      {(r['task'],r['mode']) for r in acceptance['runs']} != {(r['task'],'dt') for r in plan['runs']} or
      logger['code_sha']!=plan['code_sha'] or logger['remote_state']!='finished' or logger['entity']!=ENTITY):
    raise ValueError('Real task or logger evidence missing')
  for r in plan['runs']:
    proof=read(evidence/f'gpu{r["physical_gpu"]}-binding.json')
    if (proof['code_sha']!=plan['code_sha'] or proof['gpu_uuid']!=r['gpu_uuid'] or
        proof['physical_gpu']!=r['physical_gpu'] or not proof['verified_compute_and_graphics_same_gpu']):
      raise ValueError('Device binding missing')
    r.update(egl_index=proof['egl_index'],launch_argv=r['review_argv'],state='queued',
        authorized_to_train=True,verified_compute_and_graphics_same_gpu=True)
    r['environment_template']['MUJOCO_EGL_DEVICE_ID']=str(proof['egl_index'])
  plan['authorized_to_train']=True; validate_plan(plan);dump(output,plan)


def finish(plan, output):
  output=Path(output);output.mkdir(parents=True,exist_ok=False)
  verified=[audit_run(r,DMC16) for r in plan['runs']]
  dump(output/'RUN_INDEX.json',dict(runs=plan['runs']))
  subprocess.run([sys.executable,'-m','analysis.return_metrics','--profile','dmc16',
      '--index',str(output/'RUN_INDEX.json'),'--output',str(output/'return-audit')],cwd=REPO,check=True)
  tables=read(output/'return-audit/tables.json')
  if (len(tables['seed_metrics'])!=len(verified) or
      not all(m['auc_complete'] and m['tail_complete'] for m in tables['seed_metrics']) or
      any(m['auc_complete'] or m['tail_complete'] for m in tables['overall_metrics'])):
    raise ValueError('Partial shard metrics or false cohort completion')
  from scripts.audit_formal_pd1 import publish
  for r in plan['runs']:
    m=next(m for m in tables['seed_metrics'] if m['run_id']==r['run_id'])
    values={'formal/auc_return_per_budget':m['auc_return_per_budget'],
        'formal/tail_mean_return':m['tail_mean_return'],'formal/return_metrics_complete':True,
        'formal/missing_dt_shard_complete':True,'formal/dmc16_cohort_complete':False}
    dump(output/('wandb-'+r['task']+'.json'),publish(ENTITY+'/PD_1/'+r['environment_template']['WANDB_RUN_ID'],values))
  dump(output/'completion.json',dict(code_sha=plan['code_sha'],verified_run_count=len(verified),
      shard_complete=True,cohort_complete=False,runs=verified,tables_sha256=sha(output/'return-audit/tables.json'),checked_utc=now()))


def run(plan_path):
  plan=read(plan_path);validate_plan(plan);root=Path(plan['root'])
  if (not plan['authorized_to_train'] or platform.node()!=plan['host'] or
      clean_sha()!=plan['code_sha']):raise RuntimeError('Authorization/host/SHA mismatch')
  if root.exists():raise FileExistsError('No resume/overwrite: '+str(root))
  if shutil.disk_usage(root.parent).free < len(plan['runs'])*60_000_000_000:
    raise RuntimeError('Insufficient space for allocated runs')
  checks=[require_free(r['physical_gpu'],r['gpu_uuid']) for r in plan['runs']]
  root.mkdir();dump(root/'bound-plan.json',plan);dump(root/'prelaunch.json',dict(checks=checks,supervisor_pid=os.getpid(),utc=now()))
  handles=[]
  for r in plan['runs']:
    d=Path(r['directory'])
    try:
      require_free(r['physical_gpu'],r['gpu_uuid']);d.mkdir(parents=True,exist_ok=False);(d/'tmp').mkdir()
      log=(d/'stdout-initial.log').open('x')
      try:
        p=subprocess.Popen(r['launch_argv'],cwd=REPO,
            env=child_environment(r['environment_template'],inherited=approved_context()),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
      except BaseException:log.close();raise
      r.update(state='running',pid=p.pid,process_group_id=p.pid,start_utc=now())
      handles.append((r,p,log));dump(d/'execution-initial.json',r)
    except Exception as e:r.update(state='failed' if d.exists() else 'blocked',error=repr(e))
    dump(root/'actual-RUN_INDEX.json',dict(runs=plan['runs']))
  while any(r['state']=='running' for r in plan['runs']):
    for r,p,log in handles:
      if r['state']!='running' or p.poll() is None:continue
      log.close();r.update(exit_code=p.returncode,end_utc=now(),state='failed')
      if p.returncode==0:
        try:r['run_audit']=audit_run(r,DMC16);r['state']=AUDITED
        except Exception as e:r.update(state='audit_failed',error=repr(e))
      dump(Path(r['directory'])/'execution-final.json',r)
    dump(root/'actual-RUN_INDEX.json',dict(runs=plan['runs']))
    with (root/'resources.jsonl').open('a') as f:
      f.write(json.dumps(dict(utc=now(),free_bytes=shutil.disk_usage(root).free,
          GPUs=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True)))+'\n')
    if any(r['state']=='running' for r in plan['runs']):time.sleep(30)
  result=dict(checked_utc=now(),all_training_exited=True,complete=False,cohort_complete=False)
  if all(r['state']==AUDITED for r in plan['runs']):
    try:
      finish(plan,root/'final-audit-attempt01');result['complete']=True
      for r in plan['runs']:r['state']='accepted'
    except Exception as e:result['audit_error']=repr(e)
  result['runs']=plan['runs'];dump(root/'actual-RUN_INDEX.json',dict(runs=plan['runs']));dump(root/'supervisor-final.json',result)


if __name__=='__main__':
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('action',choices=('prepare','probe','logger','integration','bind','run'))
  p.add_argument('--plan',type=Path);p.add_argument('--output',type=Path);p.add_argument('--evidence',type=Path)
  p.add_argument('--resources',type=Path);p.add_argument('--sources',nargs='+',type=Path)
  p.add_argument('--campaign');p.add_argument('--gpu',type=int);p.add_argument('--uuid');p.add_argument('--server',choices=HOSTS)
  a=p.parse_args()
  if a.action=='prepare':prepare(a.output,a.resources,a.sources,a.campaign)
  elif a.action=='probe':probe(a.gpu,a.uuid,a.output,host=HOSTS[a.server],allowed_gpus=[g for s,g in MATRIX.values() if s==a.server])
  elif a.action=='logger':serial.logger_check(a.output,run_tag='missing-dt-20261030-'+a.server,task=next(t for t in TASKS if MATRIX[t][0]==a.server))
  elif a.action=='integration':
    plan=read(a.plan);serial.integration(a.plan,a.evidence,validator=validate_plan,host=plan['host'],new_session=True)
  elif a.action=='bind':bind(a.plan,a.evidence,a.output)
  else:run(a.plan)
