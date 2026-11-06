"""Prepare and supervise a declared dynamic Vision20 baseline/Dt campaign.

Eight assigned sv2 GPUs, one serial lane per GPU. At most one ready successor
per lane; all other attempts remain backlog. No recovery or automatic retries.
"""
import argparse
import copy
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import signal
import subprocess
import time

from dreamerv3.protocol_scope import DCS_DYNAMIC, VISION20_TASKS
from dreamerv3.dcs_protocol import check_config, backgrounds
from scripts.prepare_m2_v1 import resolve
from scripts.run_hopper_seeds123 import (
    REPO, ENTITY, now, read, dump, sha, clean_sha, require_free,
    child_environment, probe as device_probe)
from scripts.run_missing_dt import approved_context
from scripts.run_dtl_three import stop_owned

HOST = 'lyg0360'
GPUS = tuple(range(8))
UUIDS = ('GPU-76344ce3-10a1-952d-68f5-7720a3cda667',
    'GPU-1e27fd5e-4231-926a-754f-e7d4c37d41ab',
    'GPU-24608f8d-a512-70b2-8578-45945b57bc35',
    'GPU-92630efa-be4e-7074-d29a-ccff0ad7f801',
    'GPU-7bf636ec-fa03-aa3f-7314-dd803c1658be',
    'GPU-ac733835-08ee-7a17-a2fa-1e49d62d385b',
    'GPU-548be1c8-86f0-b7a2-de92-5c455b12221b',
    'GPU-4cbf5b69-adb3-5210-2396-ca5d4e400f26')
PYTHON = '/data/Policy_Discrepancy/envs/dreamer/bin/python'
RUNS = PurePosixPath('/data/Policy_Discrepancy/runs')
INTEGRATION_TASKS = ('dmc_hopper_hop', 'dmc_quadruped_run',
                     'dmc_finger_spin', 'dmc_walker_walk')
FIXTURE = {'run.engineering_fixture': True, 'run.action_budget': 4098,
    'run.eval_every_actions': 2049, 'run.eval_eps': 1,
    'run.match_start': 2200, 'run.match_end': 3600, 'run.report_every_actions': 2049}


def command(declaration_path, task, mode, directory, engineering=False):
  declaration = read(declaration_path)
  background = declaration['background']
  argv = [PYTHON, '-u', '-m', 'dreamerv3.main', '--configs', 'm2_dcs_dynamic',
      '--task', task, '--seed', str(declaration['seed']), '--logdir', str(directory),
      '--agent.rep_probe.mode', mode, '--jax.prealloc', 'False',
      '--logger.outputs', 'jsonl', *([] if engineering else ['wandb']),
      '--dcs_protocol.declaration_file', str(declaration_path),
      '--dcs_protocol.declaration_sha256', sha(declaration_path),
      '--env.dmc.background.dataset', background['dataset'],
      '--env.dmc.background.videos', background['train_videos'],
      '--env.dmc.background.num_videos', str(background['num_videos']),
      '--env.dmc.background.video_alpha', str(background['video_alpha']),
      '--env.dmc.background.ground_plane_alpha', str(background['ground_by_task'][task]),
      '--env.dmc.background.seed', str(background['seed']),
      '--env.dmc.background.cache_size', str(background['cache_size'])]
  if engineering:
    for key,value in FIXTURE.items(): argv += ['--'+key,str(value)]
  check_config(resolve(argv[4:]))
  return argv


def initial_row(task, mode, gpu, position, declaration, root, campaign, commit, engineering=False):
  group = 'baseline' if mode == 'logging' else 'dt'
  directory = str(root/task/(group+'-attempt01'))
  argv = command(declaration, task, mode, directory, engineering)
  seed = resolve(argv[4:]).seed
  return dict(task=task, seed=seed, group=group, mode=mode, server='sv2', host=HOST,
      physical_gpu=gpu, gpu_uuid=UUIDS[gpu], lane_position=position,
      directory=directory, argv=argv, config=json.loads(json.dumps(dict(resolve(argv[4:]).flat))),
      run_id=f'{campaign}:{task}:seed{seed}:{group}:attempt01',
      code_sha=commit, state='backlog',
      environment=dict(CUDA_VISIBLE_DEVICES=UUIDS[gpu], CUDA_DEVICE_ORDER='PCI_BUS_ID',
          MUJOCO_EGL_DEVICE_ID=None, MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
          JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false',
          OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
          PYTHONUNBUFFERED='1', TMPDIR=directory+'/tmp', WANDB_DIR=directory,
          WANDB_ENTITY=ENTITY, WANDB_PROJECT='PD_1', WANDB_MODE='online',
          WANDB_RESUME='never', WANDB_RUN_GROUP=campaign,
          WANDB_RUN_ID=f'{campaign}-{task[4:]}-{group}-01'))


def prepare(output, declaration_path, campaign, rollout):
  if (not campaign.startswith('dcs-dynamic-vision20-') or
      any(x in campaign for x in ('/', '\\', '..'))):
    raise ValueError('Use a fresh dcs-dynamic-vision20-* campaign')
  if rollout not in ('all40', 'first16'):
    raise ValueError('Explicit all40 or first16 rollout required')
  declaration_path = Path(declaration_path).resolve(strict=True)
  output = Path(output)
  if output.exists(): raise FileExistsError(output)
  commit = clean_sha()
  root = RUNS/campaign
  rows=[]
  for index,task in enumerate(VISION20_TASKS):
    for offset,mode in enumerate(('logging','dt')):
      row = initial_row(task,mode,index%8,(index//8)*2+offset,
          declaration_path,root,campaign,commit)
      row['enabled'] = rollout == 'all40' or index < 8
      rows.append(row)
  plan=dict(protocol=DCS_DYNAMIC,code_sha=commit,campaign=campaign,root=str(root),
      declaration_file=str(declaration_path),declaration_sha256=sha(declaration_path),
      rollout=rollout,created_utc=now(),runs=rows)
  validate(plan)
  output.mkdir(parents=True)
  dump(output/'plan.json',plan)


def validate(plan):
  if (plan['protocol'] != DCS_DYNAMIC or len(plan['runs']) != 40 or
      plan['rollout'] not in ('all40','first16') or
      PurePosixPath(plan['root']) != RUNS/plan['campaign'] or
      not plan['campaign'].startswith('dcs-dynamic-vision20-') or
      any(x in plan['campaign'] for x in ('/','\\','..')) or
      sha(plan['declaration_file']) != plan['declaration_sha256']):
    raise ValueError('DCS plan scope, path or declaration differs')
  for index,row in enumerate(plan['runs']):
    task=VISION20_TASKS[index//2]
    mode=('logging','dt')[index%2]
    gpu=(index//2)%8
    position=(index//16)*2+index%2
    expected=initial_row(task,mode,gpu,position,plan['declaration_file'],
        PurePosixPath(plan['root']),plan['campaign'],plan['code_sha'])
    expected['enabled']=plan['rollout']=='all40' or index < 16
    if row != expected: raise ValueError('DCS row command, resources or scope differs')


def refresh_queue(rows):
  """One ready successor per lane. Failed lanes cannot silently advance."""
  for gpu in GPUS:
    lane=sorted((r for r in rows if r['physical_gpu']==gpu and r['enabled']),
                key=lambda r:r['lane_position'])
    pending=[r for r in lane if r['state']!='accepted']
    if not pending: continue
    first=pending[0]
    if first['state'] in ('backlog','queued'):
      first['state']='queued'
      successors=pending[1:]
    elif first['state']=='running':
      successors=pending[1:]
      if successors and successors[0]['state']=='backlog':
        successors[0]['state']='queued'
      successors=successors[1:]
    else:
      successors=pending[1:]
      for row in successors:
        if row['state'] in ('backlog','queued'): row['state']='blocked_predecessor'
      continue
    for row in successors:
      if row['state']=='queued': row['state']='backlog'


def launchable(rows):
  refresh_queue(rows)
  selected=[]
  for gpu in GPUS:
    lane=sorted((r for r in rows if r['physical_gpu']==gpu and r['enabled']),
                key=lambda r:r['lane_position'])
    pending=[r for r in lane if r['state']!='accepted']
    if pending and pending[0]['state']=='queued': selected.append(pending[0])
  return selected


def audit(row, engineering=False):
  from embodied.run.protocol_v1 import (ProtocolState, persisted_rows,
      audit_persisted_history, audit_complete_outputs, seed32)
  from analysis.return_metrics import trapezoid_auc
  config=resolve(row['argv'][4:])
  check_config(config)
  root=Path(row['directory'])
  manifest=read(root/'protocol_manifest.json')
  if (manifest['protocol']!=DCS_DYNAMIC or manifest['git_commit']!=row['code_sha'] or
      manifest['config']!=row['config'] or manifest['root_seed']!=row['seed']):
    raise ValueError('Run manifest differs from launch declaration')
  dcs=read(root/'dcs_environment.json')
  if (dcs['declaration_sha256']!=config.dcs_protocol.declaration_sha256 or
      dcs['backgrounds']!=backgrounds(config)):
    raise ValueError('Actual DCS environment differs')
  args=config.run
  state=ProtocolState(args.action_budget,args.eval_every_actions,args.eval_eps,
                      args.match_start,args.match_end)
  for item in persisted_rows(root/'update_receipts.jsonl'):
    if not state.record_update(item['update_id'],item['start_action'],
        item['match_sum'],item['match_count'],item['invalid_count']):
      raise ValueError('Duplicate successful update receipt')
  evaluations=persisted_rows(root/'evaluations.jsonl')
  if [x['action_step'] for x in evaluations] != list(state.grid):
    raise ValueError('Incomplete or repeated evaluation grid')
  for item in evaluations:
    point=item['action_step']
    if (len(item['episodes'])!=args.eval_eps or item['lengths']!=[1000]*args.eval_eps or
        not all(e['complete'] for e in item['episodes']) or
        item['seeds']!=[seed32(row['seed'],row['task'],'eval_env',
            point//int(args.eval_every_actions),i) for i in range(args.eval_eps)]):
      raise ValueError('Incomplete evaluation episodes or RNG mapping')
    saved=read(root/'eval_snapshots'/f'{point:07d}'/'evaluation_snapshot.json')
    if saved['actual_action_step']!=point or saved['snapshot_id']!=item['snapshot_id']:
      raise ValueError('Evaluation snapshot mismatch')
  state.evaluations={r['action_step']:{k:v for k,v in r.items() if k!='action_step'}
                     for r in evaluations}
  audit_persisted_history(root,state)
  audit_complete_outputs(root,state,row['mode'],row['code_sha'])
  final=read(root/'final_state.json')
  if (final['train_action_steps']!=args.action_budget or
      final['eval_action_steps']!=len(state.grid)*args.eval_eps*1000 or
      final['train_batch_keys']!=state.updates or
      state.updates!=(517 if engineering else 249493)):
    raise ValueError('Final counters differ')
  if engineering:
    if [x['action_step'] for x in persisted_rows(root/'engineering_evaluation_audit.jsonl')]!=list(state.grid):
      raise ValueError('Evaluation isolation evidence missing')
    if len(persisted_rows(root/'engineering_report_audit.jsonl'))!=2:
      raise ValueError('CUDA report isolation evidence missing')
  points={r['action_step']:r['mean'] for r in evaluations}
  return dict(task=row['task'],group=row['group'],physical_gpu=row['physical_gpu'],
      train_actions=state.budget,updates=state.updates,engineering_only=engineering,
      auc_raw_return=None if engineering else trapezoid_auc(list(points.items()),1000000),
      final_five_mean=None if engineering else sum(points[p] for p in (800000,850000,900000,950000,1000000))/5,
      matching_c=state.frozen_c() if row['mode']=='logging' else None)


def execute(plan_path,evidence,engineering=False,accept_capacity_deficit=False):
  plan=read(plan_path)
  validate(plan)
  commit=clean_sha()
  if commit!=plan['code_sha'] or platform.node()!=HOST:
    raise RuntimeError('Exact clean tested SHA on sv2 is required')
  evidence=Path(evidence)
  root=evidence/'integration' if engineering else Path(plan['root'])
  if root.exists(): raise FileExistsError('Fresh attempt required: '+str(root))
  if not engineering:
    proof=read(evidence/'integration/acceptance.json')
    expected={(t,g) for t in INTEGRATION_TASKS for g in ('baseline','dt')}
    if (proof['code_sha']!=commit or not proof['passed'] or
        {(r['task'],r['group']) for r in proof['runs']}!=expected):
      raise ValueError('Same-SHA real size50m CUDA acceptance is required')
  free=shutil.disk_usage('/data').free
  required=30_000_000_000 if engineering else 480_000_000_000
  if free<required and not (not engineering and accept_capacity_deficit):
    raise RuntimeError(f'Insufficient starting disk: {free} < {required}')
  if free<10_000_000_000: raise RuntimeError('Insufficient disk even with capacity exception')
  root.mkdir(parents=True)
  rows=copy.deepcopy(plan['runs'])
  if engineering:
    rows=[]
    for i,task in enumerate(INTEGRATION_TASKS):
      for j,mode in enumerate(('logging','dt')):
        row=initial_row(task,mode,2*i+j,0,plan['declaration_file'],
            PurePosixPath(str(root)),plan['campaign']+'-engineering',commit,True)
        row['enabled']=True
        rows.append(row)
  dump(root/'startup.json',dict(code_sha=commit,checked_utc=now(),free_bytes=free,
      required_bytes=required,accepted_capacity_deficit=accept_capacity_deficit,
      plan=plan,engineering_only=engineering))
  handles={}
  def publish():
    refresh_queue(rows)
    dump(root/'actual-RUN_INDEX.json',dict(code_sha=commit,checked_utc=now(),runs=rows))
  def terminated(signum,frame): raise SystemExit(f'Controller received signal {signum}')
  previous=signal.signal(signal.SIGTERM,terminated)
  try:
    while True:
      free=shutil.disk_usage('/data').free
      with (root/'resources.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(utc=now(),free_bytes=free))+'\n')
      if free<5_000_000_000: raise RuntimeError('Disk below 5GB; stopping only owned jobs')
      for row in rows:
        if row['state']!='running': continue
        process,log=handles[row['run_id']]
        if process.poll() is None:
          if engineering and time.time()-row['started_epoch']>3600:
            stop_owned(process)
            row['error']='Engineering timeout'
          else: continue
        log.close()
        row.update(exit_code=process.returncode,ended_utc=now(),state='training_finished_pending_audit')
        if process.returncode:
          row['state']='failed'
        else:
          try:
            row['audit']=audit(row,engineering)
            row['state']='accepted'
          except Exception as error:
            row.update(state='failed_audit',error=repr(error))
      for row in launchable(rows):
        gpu=row['physical_gpu']
        free=shutil.disk_usage('/data').free
        # An explicit initial-capacity exception covers only the first eight.
        # Later attempts wait for storage; never reinterpret it as permission to delete.
        if not engineering and row['lane_position']>0 and free<60_000_000_000:
          row['waiting_for_storage_bytes']=60_000_000_000
          continue
        row.pop('waiting_for_storage_bytes',None)
        binding=read(evidence/f'gpu{gpu}-binding.json')
        if (binding['code_sha']!=commit or binding['gpu_uuid']!=row['gpu_uuid'] or
            not binding['verified_compute_and_graphics_same_gpu']):
          raise ValueError('CUDA/EGL binding proof differs')
        try:
          row['resource_before_start']=require_free(gpu,row['gpu_uuid'])
        except RuntimeError as error:
          row.update(state='blocked_resource',error=str(error))
          continue
        directory=Path(row['directory'])
        directory.mkdir(parents=True,exist_ok=False)
        (directory/'tmp').mkdir()
        row['environment']['MUJOCO_EGL_DEVICE_ID']=str(binding['egl_index'])
        log=(directory/'worker.log').open('x',encoding='utf-8')
        try:
          process=subprocess.Popen(row['argv'],cwd=REPO,
              env=child_environment(row['environment'],inherited=approved_context()),
              stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        except BaseException:
          log.close()
          raise
        handles[row['run_id']]=(process,log)
        row.update(state='running',pid=process.pid,started_epoch=time.time(),start_utc=now())
        dump(directory/'execution-initial.json',row)
        publish()
      publish()
      if not any(r['state']=='running' for r in rows) and not any(
          r.get('waiting_for_storage_bytes') for r in rows): break
      time.sleep(15)
    passed=all(r['state']=='accepted' for r in rows if r['enabled'])
    dump(root/('acceptance.json' if engineering else 'supervisor-final.json'),
        dict(code_sha=commit,checked_utc=now(),passed=passed,
            runs=[r['audit'] for r in rows if r['state']=='accepted'],
            states=[r['state'] for r in rows],engineering_only=engineering))
    if not passed: raise RuntimeError('Failed/blocked attempts retained; no silent restart')
  except BaseException as error:
    dump(root/'supervisor-error.json',dict(utc=now(),error=repr(error),runs=rows))
    raise
  finally:
    for process,log in handles.values(): stop_owned(process); log.close()
    signal.signal(signal.SIGTERM,previous)


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('action',choices=('prepare','probe','integration','run'))
  for name in ('output','declaration','plan','evidence'): parser.add_argument('--'+name,type=Path)
  parser.add_argument('--campaign')
  parser.add_argument('--rollout',choices=('all40','first16'))
  parser.add_argument('--gpu',type=int,choices=GPUS)
  parser.add_argument('--accept-capacity-deficit',action='store_true')
  args=parser.parse_args()
  if args.action=='prepare': prepare(args.output,args.declaration,args.campaign,args.rollout)
  elif args.action=='probe':
    os.environ.update(CUDA_VISIBLE_DEVICES=UUIDS[args.gpu],CUDA_DEVICE_ORDER='PCI_BUS_ID',
        MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',JAX_PLATFORMS='cuda',
        XLA_PYTHON_CLIENT_PREALLOCATE='false')
    device_probe(args.gpu,UUIDS[args.gpu],args.output,host=HOST,allowed_gpus=GPUS)
  else: execute(args.plan,args.evidence,args.action=='integration',args.accept_capacity_deficit)


if __name__=='__main__': main()
