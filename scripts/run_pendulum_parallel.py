"""Approved GPU5/6 Dt relocation; keep GPU1/4 baselines and their parent alive.

Only orchestration changes. All four agents execute the previously tested,
immutable training checkout. Empty old Dt destinations become explicit guards;
the original parent retains waitpid and baseline terminal-audit responsibility.
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

from dreamerv3.protocol_scope import PENDULUM_23
from scripts.prepare_m2_v1 import resolve, PYTHON
from scripts.prepare_dmc16 import source_state
from scripts.prepare_pendulum_seeds23 import HOST, TASK, check_config
from scripts.run_pendulum_seeds23 import validate as validate_old, AUDITED, finish
from scripts.run_hopper_seeds123 import read, dump, sha, now, clean_sha, require_free, child_environment, audit_run
from scripts.run_missing_dt import approved_context

TRAIN_SHA = 'c5f86ca1a16648f57db3c3abfa2584ac560495ca'
TRAIN_REPO = PurePosixPath('/data/Policy_Discrepancy/checkouts/pendulum-seeds23-c5f86ca1a166')
OLD_ROOT = PurePosixPath('/data/Policy_Discrepancy/runs/formal-m2-pendulum-seeds23-20261103-sv3')
OLD_ENG = PurePosixPath('/data/Policy_Discrepancy/runs/engineering-pendulum-seeds23-20261103-sv3')
ROOT = PurePosixPath('/data/Policy_Discrepancy/runs/formal-m2-pendulum-seeds23-parallel-dt-20261104-sv3')
CARDS = {2: 5, 3: 6}
GUARD = 'SUPERSEDED_BY_PARALLEL.json'


def build(source, resources, controller_sha, bindings=None):
  validate_old(source)
  if (source['code_sha'] != TRAIN_SHA or source['root'] != str(OLD_ROOT) or
      not source['authorized_to_train'] or resources['host'] != HOST):
    raise ValueError('Wrong source cohort/commit/host')
  cards = {r['physical_gpu']: r for r in resources['cards']}
  if len({cards[g]['uuid'] for g in CARDS.values()}) != 2:
    raise ValueError('Two distinct assigned GPUs required')
  rows = []
  for seed, gpu in CARDS.items():
    old = next(r for r in source['runs'] if r['seed']==seed and r['mode']=='dt')
    r = copy.deepcopy(old); directory = str(ROOT/f'seed{seed}'/'dt-attempt01')
    wid = ROOT.name+f'-pendulum-seed{seed}-dt-01'
    argv = list(old['review_argv']); argv[argv.index('--logdir')+1] = directory
    config = json.loads(json.dumps(dict(resolve(argv[4:]).flat))); check_config(resolve(argv[4:]))
    if {k for k in config if config[k] != old['config'][k]} != {'logdir'}:
      raise ValueError('Relocation changed scientific configuration')
    proof = bindings[str(gpu)] if bindings else None
    if proof and (proof['code_sha']!=TRAIN_SHA or proof['physical_gpu']!=gpu or
        proof['gpu_uuid']!=cards[gpu]['uuid'] or not proof['verified_compute_and_graphics_same_gpu'] or
        type(proof['egl_index']) is not int):
      raise ValueError('Wrong training-SHA device proof')
    r.update(directory=directory, physical_gpu=gpu, gpu_uuid=cards[gpu]['uuid'], config=config,
        review_argv=argv, launch_argv=argv if proof else None, run_id='sv3:'+wid,
        queue_position=1, serial_predecessor=None, old_run_id=old['run_id'],
        old_pending_directory=old['directory'], controller_sha=controller_sha,
        egl_index=proof['egl_index'] if proof else None, authorized_to_train=bool(proof),
        verified_compute_and_graphics_same_gpu=bool(proof), state='queued' if proof else 'prepared_not_started')
    r['environment_template'].update(CUDA_VISIBLE_DEVICES=r['gpu_uuid'],
        MUJOCO_EGL_DEVICE_ID=str(proof['egl_index']) if proof else None,
        TMPDIR=directory+'/tmp', WANDB_DIR=directory, WANDB_RUN_ID=wid)
    original = source['references']['dt']['config']
    r['reference_config_differences'] = {k:[original.get(k),config.get(k)]
        for k in original.keys() | config.keys() if original.get(k)!=config.get(k)}
    rows.append(r)
  return dict(protocol=PENDULUM_23, code_sha=TRAIN_SHA, controller_sha=controller_sha,
      training_checkout=str(TRAIN_REPO), root=str(ROOT), server='sv3', host=HOST,
      source_plan=source, resource_snapshot=resources, device_bindings=bindings,
      authorized_to_train=bool(bindings), runs=rows, required_storage_bytes=240_000_000_000,
      source_baseline_index=str(OLD_ROOT/'actual-RUN_INDEX.json'),
      authorization='User explicitly allowed remaining assigned idle GPUs and parallel testing, 2026-11-04')


def validate(plan):
  expected = build(plan['source_plan'], plan['resource_snapshot'], plan['controller_sha'], plan['device_bindings'])
  if plan != expected:
    raise ValueError('Parallel plan differs from exact approved matrix/configuration')


def pinned_training():
  actual = subprocess.check_output(['git','-C',str(TRAIN_REPO),'rev-parse','HEAD'],text=True).strip()
  if actual != TRAIN_SHA or subprocess.check_output(['git','-C',str(TRAIN_REPO),'status','--porcelain']):
    raise RuntimeError('Training checkout is not the pinned clean tested version')


def source_rows(plan, *, before_launch=False):
  rows = read(plan['source_baseline_index'])['runs']
  for r in rows:
    original = next(x for x in plan['source_plan']['runs'] if x['run_id']==r['run_id'])
    for key in ('seed','mode','task','physical_gpu','gpu_uuid','config','directory','code_sha','launch_argv'):
      if r[key] != original[key]:
        raise ValueError('Original active cohort identity changed: '+key)
    if before_launch and ((r['mode']=='logging' and r['state'] not in ('running',AUDITED)) or
                          (r['mode']=='dt' and r['state']!='queued')):
      raise RuntimeError('Original pending run already changed state; do not duplicate it')
  if len(rows)!=4 or len({r['run_id'] for r in rows})!=4:
    raise ValueError('Original cohort incomplete/duplicate')
  return rows


def guard_payload(plan, row):
  return dict(reason='Pending serial Dt superseded by explicitly approved parallel run; no training occurred here.',
      controller_sha=plan['controller_sha'], training_sha=TRAIN_SHA,
      old_run_id=row['old_run_id'], replacement_run_id=row['run_id'],
      replacement_directory=row['directory'], canonical_cohort=str(ROOT/'actual-RUN_INDEX.json'))


def retire_pending(plan):
  source_rows(plan, before_launch=True)
  for r in plan['runs']:
    d = Path(r['old_pending_directory'])
    if d.as_posix() != str(OLD_ROOT/f'seed{r["seed"]}'/'dt-attempt01'):
      raise ValueError('Guard outside the exact unused pending slot')
    payload = guard_payload(plan, r)
    if d.exists():
      if {p.name for p in d.iterdir()}!={GUARD} or read(d/GUARD)!=payload:
        raise FileExistsError('Existing run/evidence must never be replaced: '+str(d))
    else:
      d.mkdir(parents=False, exist_ok=False)
      dump(d/GUARD, payload)


def verify_evidence(plan, evidence):
  evidence = Path(evidence)
  cpu = read(evidence/'controller-cpu-acceptance.json')
  old_cpu = read(OLD_ENG/'cpu-acceptance.json'); logger = read(OLD_ENG/'logger-check/logger_validation.json')
  a = read(evidence/'integration/acceptance.json')
  if (not cpu['passed'] or cpu['code_sha']!=plan['controller_sha'] or
      not old_cpu['passed'] or old_cpu['code_sha']!=TRAIN_SHA or
      logger['code_sha']!=TRAIN_SHA or logger['remote_state']!='finished' or
      logger['project']!='PD_1' or logger['entity']!='ychan9liu-tsinghua-university' or
      logger['seed']!=2 or logger['training_actions']!=0 or logger['agent_updates']!=0 or
      not a['passed'] or a['training_sha']!=TRAIN_SHA or not a['engineering_only'] or
      a['controller_sha']!=plan['controller_sha'] or a['run_count']!=2 or
      any((r['updates'],r['evaluations'],r['reports'],r['exit_code'])!=(517,3,2,0) for r in a['runs']) or
      {(r['seed'],r['mode'],r['physical_gpu']) for r in a['runs']} != {(2,'dt',5),(3,'dt',6)}):
    raise ValueError('Missing exact controller/training/device integration evidence')
  bindings = {str(g):read(evidence/f'gpu{g}-binding.json') for g in CARDS.values()}
  build(plan['source_plan'], plan['resource_snapshot'], plan['controller_sha'], bindings)
  if plan['device_bindings'] and plan['device_bindings']!=bindings:
    raise ValueError('Bound device proof changed')
  return bindings


def start(row):
  pinned_training()
  before = require_free(row['physical_gpu'], row['gpu_uuid'])
  d = Path(row['directory']); d.mkdir(parents=True,exist_ok=False); (d/'tmp').mkdir()
  log = (d/'stdout-initial.log').open('x')
  try:
    p = subprocess.Popen(row['launch_argv'],cwd=TRAIN_REPO,
        env=child_environment(row['environment_template'],inherited=approved_context()),
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  except BaseException:
    log.close(); raise
  import psutil
  row.update(state='running',pid=p.pid,process_group_id=p.pid,start_epoch=time.time(),start_utc=now(),
      process_create_time=psutil.Process(p.pid).create_time(),prelaunch_check=before,
      wandb_url='https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/'+row['environment_template']['WANDB_RUN_ID'])
  dump(d/'execution-initial.json',row)
  return p,log


def integration(plan_path, evidence):
  plan = read(plan_path); validate(plan); pinned_training(); evidence=Path(evidence)
  if clean_sha()!=plan['controller_sha'] or platform.node()!=HOST:
    raise RuntimeError('Wrong controller SHA/host')
  output=evidence/'integration'; output.mkdir(parents=True,exist_ok=False)
  children=[]
  for src in plan['runs']:
    row=copy.deepcopy(src); proof=read(evidence/f'gpu{row["physical_gpu"]}-binding.json')
    build(plan['source_plan'],plan['resource_snapshot'],plan['controller_sha'],
          {str(g):read(evidence/f'gpu{g}-binding.json') for g in CARDS.values()})
    d=output/f'seed{row["seed"]}'/'dt'
    row['launch_argv']=row['review_argv']+['--logdir',str(d),'--logger.outputs','jsonl',
        '--run.engineering_fixture','True','--run.action_budget','4098',
        '--run.eval_every_actions','2049','--run.eval_eps','1',
        '--run.match_start','2200','--run.match_end','3600','--run.report_every_actions','2049']
    row['directory']=str(d)
    row['environment_template'].update(MUJOCO_EGL_DEVICE_ID=str(proof['egl_index']),
        TMPDIR=str(d/'tmp'),WANDB_DIR=str(d))
    p,log=start(row); children.append((row,p,log))
  dump(output/'active.json',dict(runs=[r for r,_,_ in children]))
  while any(p.poll() is None for _,p,_ in children): time.sleep(10)
  completed=[]
  from embodied.run.protocol_v1 import ProtocolState,audit_persisted_history,audit_complete_outputs
  for row,p,log in children:
    log.close(); d=Path(row['directory'])
    if p.returncode: raise RuntimeError('CUDA engineering failed '+str(d))
    state=ProtocolState(4098,2049,1,2200,3600)
    for line in (d/'update_receipts.jsonl').read_text().splitlines():
      r=json.loads(line)
      if not state.record_update(r['update_id'],r['start_action'],r['match_sum'],r['match_count'],r['invalid_count']):
        raise ValueError('Duplicate engineering receipt')
    evaluations=[json.loads(s) for s in (d/'evaluations.jsonl').read_text().splitlines()]
    if [r['action_step'] for r in evaluations]!=list(state.grid) or any(
        len(r['episodes'])!=1 or not r['episodes'][0]['complete'] for r in evaluations):
      raise ValueError('Incomplete engineering evaluation')
    state.evaluations={r['action_step']:{k:v for k,v in r.items() if k!='action_step'} for r in evaluations}
    audit_persisted_history(d,state); audit_complete_outputs(d,state,'dt',TRAIN_SHA)
    m=read(d/'protocol_manifest.json')
    if (m['protocol']!=PENDULUM_23 or m['root_seed']!=row['seed'] or
        m['config']!=json.loads(json.dumps(dict(resolve(row['launch_argv'][4:]).flat)))):
      raise ValueError('Engineering actual configuration mismatch')
    if len((d/'engineering_report_audit.jsonl').read_text().splitlines())!=2 or len(
        (d/'engineering_evaluation_audit.jsonl').read_text().splitlines())!=3:
      raise ValueError('Missing report/evaluation isolation evidence')
    completed.append(dict(seed=row['seed'],mode='dt',physical_gpu=row['physical_gpu'],
        updates=state.updates,evaluations=3,reports=2,exit_code=p.returncode,directory=str(d)))
  dump(output/'acceptance.json',dict(passed=True,training_sha=TRAIN_SHA,controller_sha=clean_sha(),
      run_count=2,engineering_only=True,runs=completed,checked_utc=now()))


def merged_rows(plan, own):
  source=source_rows(plan)
  return [next(r for r in (source if mode=='logging' else own) if r['seed']==seed and r['mode']==mode)
          for seed in CARDS for mode in ('logging','dt')]


def run(plan_path, evidence):
  plan=read(plan_path); validate(plan); pinned_training(); verify_evidence(plan,evidence)
  if not plan['authorized_to_train'] or clean_sha()!=plan['controller_sha'] or platform.node()!=HOST:
    raise RuntimeError('Wrong bound authorization/controller/host')
  root=Path(plan['root'])
  if root.exists(): raise FileExistsError('No overwrite/resume')
  checks=[require_free(r['physical_gpu'],r['gpu_uuid']) for r in plan['runs']]
  if shutil.disk_usage(root.parent).free<plan['required_storage_bytes']:
    raise RuntimeError('Storage below conservative four-run reserve')
  source_rows(plan,before_launch=True)
  root.mkdir(); dump(root/'bound-plan.json',plan)
  retire_pending(plan)
  dump(root/'prelaunch.json',dict(checked_utc=now(),checks=checks,supervisor_pid=os.getpid(),
      original_baseline_supervisor_retained=True,training_sha=TRAIN_SHA,controller_sha=clean_sha()))
  own=copy.deepcopy(plan['runs']); handles=[]
  for row in own:
    try:
      p,log=start(row); handles.append((row,p,log))
    except Exception as e: row.update(state='blocked',error=repr(e))
  while True:
    for row,p,log in handles:
      if row['state']!='running' or p.poll() is None: continue
      log.close(); row.update(state='training_finished_pending_audit' if p.returncode==0 else 'failed',
          exit_code=p.returncode,end_utc=now(),elapsed_seconds=time.time()-row['start_epoch'])
      if p.returncode==0:
        try: row['run_audit']=audit_run(row,PENDULUM_23); row['state']=AUDITED
        except Exception as e: row.update(state='audit_failed',error=repr(e))
      dump(Path(row['directory'])/'execution-final.json',row)
    rows=merged_rows(plan,own)
    dump(root/'actual-RUN_INDEX.json',dict(runs=rows,controller_sha=plan['controller_sha'],
        supersedes_pending_only=True,checked_utc=now()))
    if not any(r['state'] in ('running','queued','training_finished_pending_audit') for r in rows): break
    time.sleep(30)
  if all(r['state']==AUDITED for r in rows):
    try:
      finish(dict(code_sha=TRAIN_SHA,runs=rows),root/'final-audit-attempt01')
      for r in rows: r['state']='accepted'
    except Exception as e: dump(root/'postprocess-failed.json',dict(error=repr(e),checked_utc=now()))
  dump(root/'actual-RUN_INDEX.json',dict(runs=rows,controller_sha=plan['controller_sha'],checked_utc=now()))
  dump(root/'supervisor-final.json',dict(checked_utc=now(),complete=all(r['state']=='accepted' for r in rows),
      training_sha=TRAIN_SHA,audit_controller_sha=plan['controller_sha'],runs=rows))


if __name__=='__main__':
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('action',choices=('prepare','integration','bind','run','audit'))
  for name in ('plan','source','resources','evidence','output'): p.add_argument('--'+name,type=Path)
  a=p.parse_args()
  if a.action=='prepare':
    if a.output.exists(): raise FileExistsError(a.output)
    commit,_=source_state(); dump(a.output,build(read(a.source),read(a.resources),commit))
  elif a.action=='integration': integration(a.plan,a.evidence)
  elif a.action=='bind':
    plan=read(a.plan); validate(plan); pinned_training()
    if clean_sha()!=plan['controller_sha'] or platform.node()!=HOST or a.output.exists():
      raise RuntimeError('Wrong bind SHA/host/existing output')
    proofs=verify_evidence(plan,a.evidence)
    dump(a.output,build(plan['source_plan'],plan['resource_snapshot'],plan['controller_sha'],proofs))
  elif a.action=='audit':
    plan=read(a.plan); rows=read(Path(plan['root'])/'actual-RUN_INDEX.json')['runs']
    finish(dict(code_sha=TRAIN_SHA,runs=rows),a.output)
  else: run(a.plan,a.evidence)
