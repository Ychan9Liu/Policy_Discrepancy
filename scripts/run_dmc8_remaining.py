"""Remaining eight DMC tasks on the user's exact sv1/sv2 allocation.

Each host owns a fixed shard. Host completion never implies eight-task cohort
completion. Training semantics, RNG derivation and protocol_dmc16 are unchanged.
"""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

from dreamerv3.protocol_scope import DMC16, DMC16_TASKS, DMC8_REMAINING
from scripts.prepare_dmc16 import check_dmc16, source_state, verify_task_inventory, PYTHON
from scripts.prepare_m2_v1 import resolve, seed32
from scripts import run_dmc8_serial as serial
from scripts.run_hopper_seeds123 import (
    REPO, ENTITY, read, dump, sha, now, clean_sha, require_free, audit_run, probe)

TASKS = DMC16_TASKS[8:]
HOSTS = {'sv1':'lyg2153', 'sv2':'lyg0360'}
DEVICES = {'sv1':(2,7), 'sv2':(0,1,2,3,4,5)}
MATRIX = dict(zip(TASKS,[('sv1',2),('sv1',7),*[('sv2',g) for g in range(6)]]))


def validate_plan(plan):
  server = plan['server']
  if server not in HOSTS:
    raise ValueError('Unknown server shard')
  tasks = [t for t in TASKS if MATRIX[t][0]==server]
  rows = plan['runs']
  if (plan['protocol']!=DMC16 or plan['cohort']!=DMC8_REMAINING or plan['host']!=HOSTS[server] or
      plan['tasks']!=tasks or plan['cohort_tasks']!=list(TASKS) or len(rows)!=2*len(tasks) or
      {(r['task'],r['mode']) for r in rows}!={(t,m) for t in tasks for m in ('logging','dt')}):
    raise ValueError('Shard differs from the approved remaining-eight matrix')
  root = PurePosixPath(plan['root'])
  if (root.parent!=PurePosixPath('/data/Policy_Discrepancy/runs') or
      not plan['campaign'].startswith('formal-m2-dmc16-remaining8-serial-') or
      root.name!=plan['campaign']+'-'+server):
    raise ValueError('Unexpected campaign root')
  for row in rows:
    mode = row['mode']
    expected = str(root/row['task']/(mode+'-attempt01'))
    if (MATRIX[row['task']]!=(server,row['physical_gpu']) or row['server']!=server or
        row['host']!=HOSTS[server] or row['code_sha']!=plan['code_sha'] or
        type(row['seed']) is not int or row['seed']!=0 or row['directory']!=expected or
        row['queue_position']!=(1 if mode=='logging' else 2)):
      raise ValueError('Run differs from user resources/seed')
    argv = row['review_argv']
    config = resolve(argv[4:])
    check_dmc16(config)
    if (argv[:4]!=[PYTHON,'-u','-m','dreamerv3.main'] or
        json.loads(json.dumps(dict(config.flat)))!=row['config'] or config.task!=row['task'] or
        config.agent.rep_probe.mode!=mode or config.logdir!=expected):
      raise ValueError('Command and frozen configuration differ')
    env = row['environment_template']
    if (env['CUDA_VISIBLE_DEVICES']!=row['gpu_uuid'] or env['JAX_PLATFORMS']!='cuda' or
        env['WANDB_ENTITY']!=ENTITY or env['WANDB_PROJECT']!='PD_1' or
        env['WANDB_MODE']!='online' or env['WANDB_RESUME']!='never' or
        env['WANDB_DIR']!=expected or env['TMPDIR']!=expected+'/tmp'):
      raise ValueError('Device/platform/log destination differs')
    if plan['authorized_to_train'] and (
        not row['authorized_to_train'] or row['launch_argv']!=argv or
        not row['verified_compute_and_graphics_same_gpu'] or
        env['MUJOCO_EGL_DEVICE_ID']!=str(row['egl_index'])):
      raise ValueError('Unbound launch')
  if (len({r['run_id'] for r in rows})!=len(rows) or
      len({r['environment_template']['WANDB_RUN_ID'] for r in rows})!=len(rows) or
      len({r['gpu_uuid'] for r in rows})!=len(tasks)):
    raise ValueError('Duplicate run/device identity')
  for task in tasks:
    a,b = [r for r in rows if r['task']==task]
    if a['gpu_uuid']!=b['gpu_uuid'] or {k for k in a['config'] if a['config'][k]!=b['config'][k]}!={'logdir','agent.rep_probe.mode'}:
      raise ValueError('Paired configurations or GPU differ')


def random_streams(task):
  result = dict(task=task,root_seed=0,derivation='SHA256 NUL-joined strings, first four big-endian bytes',
      train_env_seeds=[seed32(task,'train_env',i) for i in range(16)],
      replay_seed=seed32(task,'train_replay'),initialization_key=[0,0],
      training_policy_and_update='independent PCG64([root_seed, own_counter])',
      report='disabled; seed32(root_seed,task,report,id)',shuffle='not used',evaluations=[])
  for point,step in enumerate(range(0,1000001,50000)):
    episodes=[]
    for episode in range(10):
      seed=seed32(task,'eval_env',point,episode)
      key=np.random.default_rng([0,0x4556414C,seed,0]).integers(0,np.iinfo(np.uint32).max,(2,),np.uint32).tolist()
      episodes.append(dict(episode_id=episode,environment_seed=seed,first_eval_policy_key=key))
    result['evaluations'].append(dict(grid_index=point,action_step=step,episodes=episodes))
  return result


def prepare(output, campaign, resources):
  if (not campaign.startswith('formal-m2-dmc16-remaining8-serial-') or
      any(c in campaign for c in '/\\') or '..' in campaign):
    raise ValueError('Use a fresh remaining-eight serial campaign ID')
  output=Path(output)
  if output.exists():raise FileExistsError(output)
  mapping=read(resources)['servers']
  if set(mapping)!=set(HOSTS):raise ValueError('Need both assigned servers')
  for server,host in HOSTS.items():
    if (mapping[server]['host']!=host or set(mapping[server]['gpus'])!={str(g) for g in DEVICES[server]}):
      raise ValueError('Resource inventory differs from user allocation')
  all_uuids=[r['uuid'] for s in mapping.values() for r in s['gpus'].values()]
  if len(set(all_uuids))!=8:raise ValueError('Need eight distinct UUIDs')
  commit,dirty=source_state()
  output.mkdir(parents=True)
  dump(output/'task-inventory.json',verify_task_inventory())
  all_rows=[]
  for server,host in HOSTS.items():
    root='/data/Policy_Discrepancy/runs/'+campaign+'-'+server
    rows=[]
    for task in [t for t in TASKS if MATRIX[t][0]==server]:
      dump(output/(task+'-random-streams.json'),random_streams(task))
      gpu=MATRIX[task][1]
      for mode in ('logging','dt'):
        directory=f'{root}/{task}/{mode}-attempt01'
        argv=[PYTHON,'-u','-m','dreamerv3.main','--configs','m2_dmc16','--task',task,
            '--seed','0','--logdir',directory,'--agent.rep_probe.mode',mode,
            '--jax.prealloc','False','--logger.outputs','jsonl','wandb']
        config=resolve(argv[4:]);check_dmc16(config)
        name=task+'-'+mode+'-config';dump(output/(name+'.json'),dict(config.flat))
        config.save(str(output/(name+'.yaml')))
        uuid=mapping[server]['gpus'][str(gpu)]['uuid']
        env=dict(CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',MUJOCO_EGL_DEVICE_ID=None,
            MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',JAX_PLATFORMS='cuda',
            XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
            MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',TMPDIR=directory+'/tmp',
            WANDB_ENTITY=ENTITY,WANDB_PROJECT='PD_1',WANDB_MODE='online',WANDB_RESUME='never',
            WANDB_DIR=directory,WANDB_RUN_GROUP=campaign,
            WANDB_JOB_TYPE='formal-'+('baseline' if mode=='logging' else 'dt'),
            WANDB_RUN_ID=f'{campaign}-{task[4:]}-seed0-{mode}-01')
        rows.append(dict(task=task,mode=mode,group='baseline' if mode=='logging' else 'dt',seed=0,
            run_id=f'{server}:{campaign}:{task}:seed0:{mode}:attempt01',protocol=DMC16,
            server=server,host=host,physical_gpu=gpu,gpu_uuid=uuid,code_sha=commit,
            directory=directory,queue_position=1 if mode=='logging' else 2,state='queued_not_bound',
            config=dict(config.flat),config_file=name+'.json',config_sha256=sha(output/(name+'.json')),
            review_argv=argv,command_for_review=shlex.join(argv),launch_argv=None,
            authorized_to_train=False,verified_compute_and_graphics_same_gpu=False,egl_index=None,
            environment_template=env,depends_on_baseline_c=False,
            serial_predecessor='own audited logging baseline' if mode=='dt' else None))
    plan=dict(protocol=DMC16,cohort=DMC8_REMAINING,server=server,host=host,
        tasks=[t for t in TASKS if MATRIX[t][0]==server],cohort_tasks=list(TASKS),
        campaign=campaign,root=root,code_sha=commit,created_utc=now(),
        authorized_to_train=False,auto_start_when_idle=False,runs=rows,
        cohort_run_count=16,host_run_count=len(rows),kind='planned_not_results',
        total_train_actions=len(rows)*1000000,complete_episodes_per_point=10,
        evaluation_points_per_run=21,resource_sha256=sha(resources),
        local_worktree_status=dirty,host_completion_is_not_cohort_completion=True)
    plan=json.loads(json.dumps(plan));validate_plan(plan)
    dump(output/(server+'-plan.json'),plan);all_rows+=plan['runs']
  dump(output/'global-planned-index.json',dict(cohort=DMC8_REMAINING,code_sha=commit,
      authorized_to_train=False,runs=all_rows,kind='planned_not_results'))
  dump(output/'artifact-hashes.json',{p.name:sha(p) for p in output.iterdir() if p.is_file()})


def bind(plan_path, evidence, output, authorized):
  plan,evidence,output=read(plan_path),Path(evidence),Path(output)
  validate_plan(plan)
  if not authorized or platform.node()!=plan['host'] or clean_sha()!=plan['code_sha']:
    raise RuntimeError('Authorization, host or clean SHA differs')
  if output.exists():raise FileExistsError(output)
  logger=read(evidence/'logger-check/logger_validation.json')
  acceptance=read(evidence/'integration/acceptance.json')
  if (logger['code_sha']!=plan['code_sha'] or logger['remote_state']!='finished' or
      logger['entity']!=ENTITY or acceptance['code_sha']!=plan['code_sha'] or
      not acceptance['passed'] or acceptance['run_count']!=len(plan['runs']) or
      {(r['task'],r['mode']) for r in acceptance['runs']}!={(r['task'],r['mode']) for r in plan['runs']}):
    raise ValueError('Logger or real task acceptance differs')
  for row in plan['runs']:
    proof=read(evidence/f'gpu{row["physical_gpu"]}-binding.json')
    if (proof['code_sha']!=plan['code_sha'] or proof['gpu_uuid']!=row['gpu_uuid'] or
        proof['physical_gpu']!=row['physical_gpu'] or not proof['verified_compute_and_graphics_same_gpu']):
      raise ValueError('Device binding differs')
    row.update(egl_index=proof['egl_index'],verified_compute_and_graphics_same_gpu=True,
        authorized_to_train=True,launch_argv=row['review_argv'],state='queued')
    row['environment_template']['MUJOCO_EGL_DEVICE_ID']=str(proof['egl_index'])
  plan.update(authorized_to_train=True,bound_utc=now(),
      authorization='User: remaining tasks, sv1 physical GPU2/7, sv2 GPU0-5 (2026-10-24)')
  validate_plan(plan);dump(output,plan)


def before_start(row, plan, *, timeout=90):
  """Wait only for this predecessor's process group; reject foreign jobs.

  This closes the exit/context cleanup race without permitting sharing. A fresh
  require_free check still happens inside start immediately before Popen.
  """
  if clean_sha()!=plan['code_sha']:raise RuntimeError('Checkout changed since plan binding')
  predecessor=next((r for r in plan['runs'] if r['task']==row['task'] and r['mode']=='logging'),None)
  expected_group=predecessor.get('process_group_id') if row['mode']=='dt' and predecessor else None
  deadline=time.monotonic()+timeout
  observations=[]
  while True:
    try:
      result=require_free(row['physical_gpu'],row['gpu_uuid'])
      row['handoff_release_check']=dict(result,observations=observations)
      return
    except RuntimeError:
      xml=subprocess.check_output(['nvidia-smi','-i',str(row['physical_gpu']),'-q','-x'],text=True)
      device=ET.fromstring(xml).find('gpu')
      if device is None or device.findtext('uuid')!=row['gpu_uuid']:raise RuntimeError('GPU identity changed')
      pids=[int(r.findtext('pid')) for r in device.findall('processes/process_info')]
      # A process may disappear between the inventory and ownership lookup.
      groups=[]
      for pid in pids:
        try:groups.append(dict(pid=pid,group=os.getpgid(pid)))
        except ProcessLookupError:pass
      if not expected_group or any(r['group']!=expected_group for r in groups):
        row['handoff_busy_inventory']=dict(utc=now(),processes=groups,xml=xml)
        raise RuntimeError('Assigned GPU has foreign/unverified jobs; no waiting, sharing or replacement')
      observations.append(dict(utc=now(),owned_predecessor_group=expected_group,processes=groups))
      if time.monotonic()>=deadline:raise RuntimeError('Predecessor GPU cleanup did not finish within 90 seconds')
      time.sleep(2)


def finish_host(plan, output):
  """Per-host receipts, raw return and W&B; missing other host stays null."""
  validate_plan(plan)
  output=Path(output);output.mkdir(parents=True,exist_ok=False)
  verified=[audit_run(r,DMC16) for r in plan['runs']]
  dump(output/'RUN_INDEX.json',dict(runs=plan['runs']))
  subprocess.run([sys.executable,'-m','analysis.return_metrics','--profile','dmc8-remaining',
      '--index',str(output/'RUN_INDEX.json'),'--output',str(output/'return-audit')],cwd=REPO,check=True)
  tables=read(output/'return-audit/tables.json')
  metrics=tables['seed_metrics']
  if len(metrics)!=len(plan['runs']) or not all(r['auc_complete'] and r['tail_complete'] for r in metrics):
    raise ValueError('Host raw-return metrics incomplete')
  if any(r['auc_complete'] or r['tail_complete'] for r in tables['overall_metrics']):
    raise ValueError('A host shard cannot stand in for the full eight-task cohort')
  from scripts.audit_formal_pd1 import publish
  for row in plan['runs']:
    metric=next(r for r in metrics if r['run_id']==row['run_id'])
    values={'formal/auc_return_per_budget':metric['auc_return_per_budget'],
        'formal/tail_mean_return':metric['tail_mean_return'],'formal/return_metrics_complete':True,
        'formal/host_shard_complete':True,'formal/eight_task_cohort_complete':False}
    if row['mode']=='logging':
      c=read(Path(row['directory'])/'c_frozen.json')
      values.update({'formal/c':c['c'],'formal/match_sum':c['match_sum'],'formal/match_count':c['match_count']})
    else:
      pair=next(r for r in tables['dt_differences'] if r['level']=='task' and r['task']==row['task'])
      values.update({'formal/dt_minus_baseline_auc':pair['auc_return_per_budget'],
          'formal/dt_minus_baseline_tail':pair['tail_mean_return']})
    dump(output/f'wandb-{row["task"]}-{row["mode"]}.json',
        publish(f'{ENTITY}/PD_1/{row["environment_template"]["WANDB_RUN_ID"]}',values))
  dump(output/'completion.json',dict(code_sha=plan['code_sha'],server=plan['server'],
      host_complete=True,cohort_complete=False,verified_run_count=len(verified),runs=verified,
      tables_sha256=sha(output/'return-audit/tables.json'),checked_utc=now()))


def run(plan_path):
  plan=read(plan_path);validate_plan(plan)
  serial.run(plan_path,validator=validate_plan,host=plan['host'],gpus=DEVICES[plan['server']],
      finisher=finish_host,capacity_bytes=len(plan['runs'])*60_000_000_000,
      before_start=before_start,new_session=True)


if __name__=='__main__':
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('action',choices=('prepare','probe','logger','integration','bind','run','audit-host'))
  for name in ('output','resources','plan','evidence'):parser.add_argument('--'+name,type=Path)
  parser.add_argument('--campaign');parser.add_argument('--server',choices=tuple(HOSTS))
  parser.add_argument('--gpu',type=int);parser.add_argument('--uuid')
  parser.add_argument('--authorized',action='store_true');args=parser.parse_args()
  if args.action=='prepare':prepare(args.output,args.campaign,args.resources)
  elif args.action=='probe':probe(args.gpu,args.uuid,args.output,host=HOSTS[args.server],allowed_gpus=DEVICES[args.server])
  elif args.action=='logger':serial.logger_check(args.output,run_tag='remaining8-'+args.server+'-20261024',task=next(t for t in TASKS if MATRIX[t][0]==args.server))
  elif args.action=='integration':
    plan=read(args.plan)
    serial.integration(args.plan,args.evidence,validator=validate_plan,host=plan['host'],new_session=True)
  elif args.action=='bind':bind(args.plan,args.evidence,args.output,args.authorized)
  elif args.action=='run':run(args.plan)
  else:finish_host(read(args.plan),args.output)
