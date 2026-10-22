"""User-authorized sv1 first-eight tasks, baseline then Dt on each physical GPU.

The runtime remains protocol_dmc16. This controller restricts its cohort to eight
tasks and never resumes, replaces a failed run, changes cards, or shares a GPU.
"""
import argparse
import copy
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import time

from dreamerv3.protocol_scope import DMC16, DMC16_TASKS, DMC8_FIRST
from scripts.prepare_dmc16 import check_dmc16, prepare as catalog
from scripts.prepare_m2_v1 import resolve
from scripts.run_hopper_seeds123 import (
    REPO, ENTITY, now, read, dump, sha, clean_sha, require_free,
    child_environment, probe as device_probe, audit_run)

TASKS = DMC16_TASKS[:8]
HOST = 'lyg2153'
AUDITED = 'run_audited_pending_group_and_WB'


def validate_plan(plan):
  rows = plan['runs']
  if (plan['protocol'] != DMC16 or plan['cohort'] != DMC8_FIRST or len(rows) != 16
      or {(r['task'], r['mode']) for r in rows} != {(t,m) for t in TASKS for m in ('logging','dt')}):
    raise ValueError('Only the approved first eight tasks x baseline/Dt are allowed')
  root = PurePosixPath(plan['root'])
  if (str(root.parent) != '/data/Policy_Discrepancy/runs' or
      not root.name.startswith('formal-m2-dmc16-first8-serial-')):
    raise ValueError('Unexpected campaign root')
  for row in rows:
    gpu = TASKS.index(row['task'])
    mode = row['mode']
    expected = str(root/row['task']/(mode+'-attempt01'))
    if (row['server'] != 'sv1' or row['host'] != HOST or row['physical_gpu'] != gpu or
        type(row['seed']) is not int or row['seed'] != 0 or row['code_sha'] != plan['code_sha'] or
        row['directory'] != expected or row['queue_position'] != (1 if mode=='logging' else 2) or
        row['environment_template']['CUDA_VISIBLE_DEVICES'] != row['gpu_uuid']):
      raise ValueError('Run differs from the user allocation')
    config = resolve(row['review_argv'][4:])
    check_dmc16(config)
    if (json.loads(json.dumps(dict(config.flat))) != row['config'] or
        config.task != row['task'] or config.agent.rep_probe.mode != mode or config.logdir != expected or
        row['review_argv'][:4] != ['/data/Policy_Discrepancy/envs/dreamer/bin/python', '-u', '-m', 'dreamerv3.main']):
      raise ValueError('Command/configuration differs')
    env = row['environment_template']
    if (env['WANDB_ENTITY'] != ENTITY or env['WANDB_PROJECT'] != 'PD_1' or
        env['WANDB_MODE'] != 'online' or env['WANDB_RESUME'] != 'never' or
        env['JAX_PLATFORMS'] != 'cuda' or env['WANDB_DIR'] != expected):
      raise ValueError('Destination/platform differs')
    if plan['authorized_to_train'] and (
        row['launch_argv'] != row['review_argv'] or not row['authorized_to_train'] or
        not row['verified_compute_and_graphics_same_gpu'] or
        env['MUJOCO_EGL_DEVICE_ID'] != str(row['egl_index'])):
      raise ValueError('Invalid bound launch')
  if (len({r['run_id'] for r in rows}) != 16 or len({r['environment_template']['WANDB_RUN_ID'] for r in rows}) != 16
      or len({r['gpu_uuid'] for r in rows}) != 8):
    raise ValueError('Duplicate run/device identity')
  for task in TASKS:
    a,b = [r for r in rows if r['task']==task]
    if a['gpu_uuid'] != b['gpu_uuid']:
      raise ValueError('Serial pair must use the same physical device')


def prepare(output, campaign, resources):
  output = Path(output)
  if output.exists():
    raise FileExistsError(output)
  output.mkdir(parents=True)
  catalog(output/'unlaunched-dmc16-catalog', campaign, resources)
  plan = read(output/'unlaunched-dmc16-catalog/planned-run-index.json')
  plan['runs'] = [r for r in plan['runs'] if r['task'] in TASKS]
  mapping = read(resources)
  for row in plan['runs']:
    gpu = TASKS.index(row['task'])
    row.update(physical_gpu=gpu, gpu_uuid=mapping['gpus'][str(gpu)]['uuid'],
        queue_position=1 if row['mode']=='logging' else 2, state='queued_not_bound',
        blockers=['fresh_free_GPU','same_SHA_CUDA_EGL_and_real_task_acceptance','online_WB'],
        serial_predecessor='own audited logging baseline' if row['mode']=='dt' else None)
    row['environment_template']['CUDA_VISIBLE_DEVICES'] = row['gpu_uuid']
  plan.update(cohort=DMC8_FIRST, tasks=list(TASKS), run_count=16, total_train_actions=16000000,
      total_evaluation_episodes=3360, authorization_status='user selected first eight; pending engineering binding',
      schedule='one task per physical GPU0-7, logging then audited barrier then Dt',
      analysis_scope='separate first-eight task cohort; no automatic 16/20-task pooling')
  plan['result_audit_argv'][4] = 'dmc8-first8'
  validate_plan(plan)
  dump(output/'planned-run-index.json',plan)


def logger_check(output):
  output = Path(output)
  output.mkdir(parents=True, exist_ok=False)
  commit = clean_sha()
  os.environ.update(WANDB_PROJECT='PD_1', WANDB_ENTITY=ENTITY, WANDB_MODE='online',
      WANDB_RESUME='never', WANDB_RUN_GROUP='logger-engineering-validation',
      WANDB_RUN_ID='logger-dmc8-20261022-'+commit[:7], WANDB_DIR=str(output))
  from dreamerv3.main import make_logger
  import wandb
  config = resolve(['--configs','m2_dmc16','--task',TASKS[0],'--logdir',str(output),
                    '--logger.outputs','jsonl','wandb'])
  logger = make_logger(config)
  run = wandb.run
  record = dict(checked_utc=now(), code_sha=commit, entity=run.entity, project=run.project,
      id=run.id, url=run.url, training_actions=0, agent_updates=0)
  if run.entity != ENTITY or run.project != 'PD_1' or run.config['seed'] != 0:
    raise RuntimeError('W&B destination/configuration differs')
  logger.add(dict(logger_validation=1), prefix='engineering')
  logger.write()
  logger.close()
  wandb.finish()
  remote = wandb.Api(timeout=30).run(f'{ENTITY}/PD_1/{record["id"]}')
  if remote.state != 'finished' or remote.summary['engineering/logger_validation'] != 1:
    raise RuntimeError('Online W&B write/read validation failed')
  record.update(remote_state=remote.state, jsonl_sha256=sha(output/'metrics.jsonl'))
  dump(output/'logger_validation.json',record)


def bind(plan_path, evidence, output, authorized):
  if not authorized or platform.node() != HOST:
    raise RuntimeError('User authorization and assigned host required')
  plan, evidence, output = read(plan_path), Path(evidence), Path(output)
  validate_plan(plan)
  commit = clean_sha()
  if plan['code_sha'] != commit or output.exists():
    raise ValueError('Pinned SHA differs or output exists')
  logger = read(evidence/'logger-check/logger_validation.json')
  integration = read(evidence/'integration/acceptance.json')
  if (logger['code_sha'] != commit or logger['remote_state'] != 'finished' or logger['entity'] != ENTITY or
      integration['code_sha'] != commit or not integration['passed'] or integration['run_count'] != 16):
    raise ValueError('Logger or real CUDA task acceptance differs')
  for row in plan['runs']:
    path = evidence/f'gpu{row["physical_gpu"]}-binding.json'
    proof = read(path)
    if (proof['code_sha'] != commit or proof['gpu_uuid'] != row['gpu_uuid'] or
        proof['physical_gpu'] != row['physical_gpu'] or not proof['verified_compute_and_graphics_same_gpu']):
      raise ValueError('GPU binding evidence differs')
    row.update(egl_index=proof['egl_index'], verified_compute_and_graphics_same_gpu=True,
        launch_argv=row['review_argv'], authorized_to_train=True, binding_evidence=str(path), state='queued')
    row['environment_template']['MUJOCO_EGL_DEVICE_ID'] = str(proof['egl_index'])
  plan.update(authorized_to_train=True, bound_utc=now(),
      authorization='User sv1 GPU0-7, 16 total runs, first eight tasks, serial baseline then Dt (2026-10-22)',
      logger_evidence=str(evidence/'logger-check/logger_validation.json'), integration_evidence=str(evidence/'integration/acceptance.json'))
  validate_plan(plan)
  dump(output,plan)


def next_for_gpu(rows, gpu):
  pair = [r for r in rows if r['physical_gpu']==gpu]
  if any(r['state']=='running' for r in pair):
    return None
  base = next(r for r in pair if r['mode']=='logging')
  dt = next(r for r in pair if r['mode']=='dt')
  if base['state']=='queued':
    return base
  if dt['state']=='queued':
    if base['state']==AUDITED:
      return dt
    dt.update(state='blocked', error='Baseline failed or was not audited; no Dt launch')
  return None


def start(row):
  check = require_free(row['physical_gpu'],row['gpu_uuid'])
  directory = Path(row['directory'])
  directory.mkdir(parents=True, exist_ok=False)
  (directory/'tmp').mkdir()
  log = (directory/'stdout-initial.log').open('x')
  try:
    process = subprocess.Popen(row['launch_argv'],cwd=REPO,
        env=child_environment(row['environment_template']),stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT)
  except BaseException:
    log.close()
    raise
  row.update(state='running',pid=process.pid,start_utc=now(),start_epoch=time.time(),
      prelaunch_check=check,command=row['launch_argv'],
      wandb_url=f'https://wandb.ai/{ENTITY}/PD_1/runs/{row["environment_template"]["WANDB_RUN_ID"]}')
  dump(directory/'execution-initial.json',row)
  return process,log


def finish(plan, output):
  output = Path(output)
  output.mkdir(parents=True, exist_ok=False)
  verified = [audit_run(r,DMC16) for r in plan['runs']]
  dump(output/'RUN_INDEX.json',dict(runs=plan['runs']))
  subprocess.run([sys.executable,'-m','analysis.return_metrics','--profile','dmc8-first8',
      '--index',str(output/'RUN_INDEX.json'),'--output',str(output/'return-audit'),'--require-complete'],cwd=REPO,check=True)
  from scripts.audit_formal_pd1 import publish
  tables = read(output/'return-audit/tables.json')
  for row in plan['runs']:
    metric = next(r for r in tables['seed_metrics'] if r['run_id']==row['run_id'])
    values = {'formal/auc_return_per_budget':metric['auc_return_per_budget'],
        'formal/tail_mean_return':metric['tail_mean_return'],'formal/return_metrics_complete':True}
    if row['mode']=='logging':
      c = read(Path(row['directory'])/'c_frozen.json')
      values.update({'formal/c':c['c'],'formal/match_sum':c['match_sum'],'formal/match_count':c['match_count']})
    else:
      pair = next(r for r in tables['dt_differences'] if r['level']=='task' and r['task']==row['task'])
      values.update({'formal/dt_minus_baseline_auc':pair['auc_return_per_budget'],
          'formal/dt_minus_baseline_tail':pair['tail_mean_return']})
    transport = publish(f'{ENTITY}/PD_1/{row["environment_template"]["WANDB_RUN_ID"]}',values)
    dump(output/f'wandb-{row["task"]}-{row["mode"]}.json',transport)
  dump(output/'completion.json',dict(checked_utc=now(),code_sha=plan['code_sha'],
      verified_run_count=16,runs=verified,tables_sha256=sha(output/'return-audit/tables.json'),
      interpretation='First eight tasks seed0; two-group descriptive comparison, no strong selective M2 attribution.'))


def run(plan_path):
  plan = read(plan_path)
  validate_plan(plan)
  if not plan['authorized_to_train'] or platform.node()!=HOST or clean_sha()!=plan['code_sha']:
    raise RuntimeError('Authorization, host or exact clean SHA differs')
  root = Path(plan['root'])
  if root.exists():
    raise FileExistsError('No resume/overwrite: '+str(root))
  checks = [require_free(g, next(r['gpu_uuid'] for r in plan['runs'] if r['physical_gpu']==g)) for g in range(8)]
  # Engineering capacity floor, not a change to replay/checkpoint configuration.
  if shutil.disk_usage(root.parent).free < 960_000_000_000:
    raise RuntimeError('Less than 960GB free for 16 full runs; report capacity, do not change protocol')
  root.mkdir(parents=True)
  dump(root/'prelaunch.json',dict(checked_utc=now(),checks=checks,supervisor_pid=os.getpid()))
  dump(root/'bound-plan.json',plan)
  handles = []
  while True:
    for row,process,log in handles:
      if row['state']!='running' or process.poll() is None:
        continue
      log.close()
      row.update(state='training_finished_pending_audit' if process.returncode==0 else 'failed',
          exit_code=process.returncode,end_utc=now(),elapsed_seconds=time.time()-row['start_epoch'])
      if process.returncode==0:
        try:
          row['run_audit']=audit_run(row,DMC16)
          row['state']=AUDITED
        except Exception as error:
          row.update(state='audit_failed',audit_error=repr(error))
      dump(Path(row['directory'])/'execution-initial.json',row)
    for gpu in range(8):
      row = next_for_gpu(plan['runs'],gpu)
      if row is None:
        continue
      try:
        process,log = start(row)
        handles.append((row,process,log))
      except Exception as error:
        row.update(state='blocked',error=repr(error))
    dump(root/'actual-RUN_INDEX.json',dict(runs=plan['runs']))
    with (root/'resources.jsonl').open('a') as file:
      sample = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
          '--format=csv,noheader,nounits'],text=True)
      file.write(json.dumps(dict(utc=now(),gpus=sample,free_disk_bytes=shutil.disk_usage(root).free))+'\n')
    if not any(r['state'] in ('queued','running') for r in plan['runs']):
      break
    time.sleep(30)
  status = dict(checked_utc=now(),runs=plan['runs'],all_training_exited=True,complete=False)
  if all(r['state']==AUDITED for r in plan['runs']):
    try:
      finish(plan,root/'final-audit-attempt01')
      status['complete']=True
      for row in plan['runs']:
        row['state']='accepted'
    except Exception as error:
      status['audit_error']=repr(error)
  dump(root/'actual-RUN_INDEX.json',dict(runs=plan['runs']))
  dump(root/'supervisor-final.json',status)


def integration(plan_path, evidence):
  """Real size50m CUDA/bfloat16 acceptance, engineering budgets only."""
  plan, evidence = read(plan_path), Path(evidence)
  validate_plan(plan)
  commit = clean_sha()
  if plan['code_sha']!=commit or platform.node()!=HOST:
    raise RuntimeError('Engineering host/SHA differs')
  output = evidence/'integration'
  output.mkdir(parents=True,exist_ok=False)
  completed = []
  for mode in ('logging','dt'):
    children = []
    for source in [r for r in plan['runs'] if r['mode']==mode]:
      row = copy.deepcopy(source)
      gpu = row['physical_gpu']
      proof = read(evidence/f'gpu{gpu}-binding.json')
      if proof['code_sha']!=commit or proof['gpu_uuid']!=row['gpu_uuid']:
        raise ValueError('Engineering binding differs')
      directory = output/row['task']/mode
      argv = row['review_argv'] + ['--logdir',str(directory),'--logger.outputs','jsonl',
          '--run.engineering_fixture','True','--run.action_budget','4098',
          '--run.eval_every_actions','2049','--run.eval_eps','1',
          '--run.match_start','2200','--run.match_end','3600','--run.report_every_actions','2049']
      row.update(directory=str(directory),launch_argv=argv)
      row['environment_template'].update(MUJOCO_EGL_DEVICE_ID=str(proof['egl_index']),
          TMPDIR=str(directory/'tmp'),WANDB_DIR=str(directory))
      process,log = start(row)
      children.append((row,process,log))
    while any(p.poll() is None for _,p,_ in children):
      time.sleep(10)
    for row,process,log in children:
      log.close()
      directory = Path(row['directory'])
      if process.returncode:
        raise RuntimeError(f'Engineering CUDA run failed: {directory}, rc={process.returncode}')
      from embodied.run.protocol_v1 import ProtocolState, audit_persisted_history, audit_complete_outputs
      state = ProtocolState(4098,2049,1,2200,3600)
      for line in (directory/'update_receipts.jsonl').read_text().splitlines():
        receipt = json.loads(line)
        if not state.record_update(receipt['update_id'],receipt['start_action'],
            receipt['match_sum'],receipt['match_count'],receipt['invalid_count']):
          raise ValueError('Duplicate engineering update receipt')
      evaluations = [json.loads(l) for l in (directory/'evaluations.jsonl').read_text().splitlines()]
      if [r['action_step'] for r in evaluations] != list(state.grid):
        raise ValueError('Engineering evaluation grid differs')
      if any(len(r['episodes'])!=1 or not r['episodes'][0]['complete'] for r in evaluations):
        raise ValueError('Engineering evaluation incomplete')
      state.evaluations = {r['action_step']:{k:v for k,v in r.items() if k!='action_step'} for r in evaluations}
      audit_persisted_history(directory,state)
      audit_complete_outputs(directory,state,row['mode'],commit)
      reports = [json.loads(l) for l in (directory/'engineering_report_audit.jsonl').read_text().splitlines()]
      eval_audits = [json.loads(l) for l in (directory/'engineering_evaluation_audit.jsonl').read_text().splitlines()]
      if len(reports)!=2 or len(eval_audits)!=3:
        raise ValueError('CUDA report read-only audit missing')
      completed.append(dict(task=row['task'],mode=row['mode'],physical_gpu=row['physical_gpu'],
          directory=str(directory),updates=state.updates,match_count=state.match_count,
          reports=len(reports),evaluations=len(state.evaluations),exit_code=process.returncode))
    dump(output/('completed-'+mode+'.json'),completed)
  dump(output/'acceptance.json',dict(code_sha=commit,passed=True,run_count=16,
      engineering_only=True,budget=4098,evaluation_episodes=1,runs=completed,checked_utc=now()))


if __name__=='__main__':
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('action',choices=('prepare','probe','logger','integration','bind','run','audit'))
  for name in ('plan','evidence','output','resources'):
    parser.add_argument('--'+name,type=Path)
  parser.add_argument('--campaign')
  parser.add_argument('--gpu',type=int)
  parser.add_argument('--uuid')
  parser.add_argument('--authorized',action='store_true')
  args=parser.parse_args()
  if args.action=='prepare': prepare(args.output,args.campaign,args.resources)
  elif args.action=='probe': device_probe(args.gpu,args.uuid,args.output,host=HOST,allowed_gpus=range(8))
  elif args.action=='logger': logger_check(args.output)
  elif args.action=='integration': integration(args.plan,args.evidence)
  elif args.action=='bind': bind(args.plan,args.evidence,args.output,args.authorized)
  elif args.action=='run': run(args.plan)
  else: finish(read(args.plan),args.output)
