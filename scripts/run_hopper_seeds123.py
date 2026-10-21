"""Authorized six-run sv3 supervisor. No algorithm or protocol overrides.

All source edits are local Git commits. Probe/logger evidence precedes launch;
the supervisor uses CPU, training children each use one assigned CUDA/EGL GPU.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from dreamerv3.protocol_scope import HOPPER_123

REPO = Path(__file__).resolve().parents[1]
ENTITY = 'ychan9liu-tsinghua-university'
MATRIX = {(1, 'logging'): 1, (1, 'dt'): 3, (2, 'logging'): 4,
          (2, 'dt'): 5, (3, 'logging'): 6, (3, 'dt'): 7}


def now():
  return datetime.now(timezone.utc).isoformat()


def read(path):
  return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, data):
  path = Path(path)
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf-8')
  temporary.replace(path)


def sha(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_sha():
  if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO):
    raise RuntimeError('Dirty checkout')
  return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()


def validate_matrix(plan):
  rows = plan['runs']
  if (plan['protocol'] != HOPPER_123 or len(rows) != 6 or
      {(r['seed'], r['mode']) for r in rows} != set(MATRIX)):
    raise ValueError('Only the approved six Hopper runs are allowed')
  for row in rows:
    if (row['server'] != 'sv3' or row['host'] != 'lyg0326' or
        row['physical_gpu'] != MATRIX[(row['seed'], row['mode'])] or
        row['task'] != 'dmc_hopper_hop'):
      raise ValueError('Run differs from the user allocation')


def require_free(gpu, uuid):
  xml = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '-q', '-x'], text=True)
  device = ET.fromstring(xml).find('gpu')
  if device is None or device.findtext('uuid') != uuid:
    raise RuntimeError('Assigned physical GPU UUID differs')
  processes = device.find('processes')
  if processes is None or ((processes.text or '').strip() not in ('', 'None') and
                           not processes.findall('process_info')):
    raise RuntimeError('GPU process inventory unavailable')
  if processes.findall('process_info'):
    raise RuntimeError(f'Assigned GPU{gpu} has jobs; no replacement or sharing')
  pmon = subprocess.check_output(['nvidia-smi', 'pmon', '-c', '1'], text=True)
  active = [line for line in pmon.splitlines() if line.strip() and not line.startswith('#')
            and line.split()[0] == str(gpu) and line.split()[1] != '-']
  if active:
    raise RuntimeError(f'Assigned GPU{gpu} has C/G jobs: {active}')
  return dict(checked_utc=now(), physical_gpu=gpu, gpu_uuid=uuid,
              memory_used=device.findtext('fb_memory_usage/used'), no_jobs=True)


def egl_visible_device():
  """Enumerate without creating contexts; CUDA mask exposes one ordinal0."""
  lib = ctypes.CDLL('libEGL.so.1')
  lib.eglGetProcAddress.argtypes = [ctypes.c_char_p]
  lib.eglGetProcAddress.restype = ctypes.c_void_p
  enum = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
                         ctypes.POINTER(ctypes.c_int))(lib.eglGetProcAddress(b'eglQueryDevicesEXT'))
  attr = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int,
                         ctypes.POINTER(ctypes.c_ssize_t))(lib.eglGetProcAddress(b'eglQueryDeviceAttribEXT'))
  count = ctypes.c_int()
  if not enum(0, None, ctypes.byref(count)):
    raise RuntimeError('EGL enumeration failed')
  devices = (ctypes.c_void_p * count.value)()
  if not enum(count.value, devices, ctypes.byref(count)):
    raise RuntimeError('EGL enumeration failed')
  records = []
  for index, device in enumerate(devices):
    value = ctypes.c_ssize_t(-1)
    success = bool(attr(device, 0x323A, ctypes.byref(value)))  # EGL_CUDA_DEVICE_NV
    records.append(dict(egl_index=index, query_ok=success, cuda_ordinal=value.value))
  matches = [r['egl_index'] for r in records if r['query_ok'] and r['cuda_ordinal'] == 0]
  if len(matches) != 1:
    raise RuntimeError('EGL-to-visible-CUDA mapping is not unique')
  return matches[0], records


def probe(gpu, uuid, output):
  if platform.node() != 'lyg0326' or gpu not in MATRIX.values():
    raise RuntimeError('Probe outside user resources')
  output = Path(output)
  if output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
    raise ValueError('Probe output exists or CUDA mask differs')
  before = require_free(gpu, uuid)
  egl_index, enumeration = egl_visible_device()
  os.environ['MUJOCO_EGL_DEVICE_ID'] = str(egl_index)
  import jax
  import jax.numpy as jnp
  if len(jax.devices()) != 1 or jax.devices()[0].platform != 'gpu':
    raise RuntimeError('Probe must use the one assigned CUDA device')
  if int(jnp.arange(4).sum().block_until_ready()) != 6:
    raise RuntimeError('CUDA calculation failed')
  from dm_control import mujoco
  physics = mujoco.Physics.from_xml_string(
      '<mujoco><worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>')
  image = physics.render(height=64, width=64, camera_id=-1)
  compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                    '--format=csv,noheader'], text=True)
  pmon = subprocess.check_output(['nvidia-smi', 'pmon', '-c', '1'], text=True)
  own = [line for line in compute.splitlines() if line.split(',')[0].strip() == str(os.getpid())]
  locations = [line.split() for line in pmon.splitlines() if len(line.split()) >= 3
               and line.split()[1] == str(os.getpid())]
  if (len(own) != 1 or uuid not in own[0] or not locations or
      {int(r[0]) for r in locations} != {gpu} or not any('G' in r[2] for r in locations)):
    raise RuntimeError('Actual CUDA/graphics PID not on assigned physical GPU')
  dump(output, dict(code_sha=clean_sha(), checked_utc=now(), pid=os.getpid(),
      physical_gpu=gpu, gpu_uuid=uuid, egl_index=egl_index, logical_cuda_device=0,
      enumeration=enumeration, compute_apps=compute, pmon=pmon, before=before,
      verified_compute_and_graphics_same_gpu=True, training_actions=0, agent_updates=0,
      image_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
  physics.free()
  print(json.dumps(dict(gpu=gpu, egl_index=egl_index, passed=True)), flush=True)


def logger_check(output):
  output = Path(output)
  output.mkdir(parents=True, exist_ok=False)
  commit = clean_sha()
  os.environ.update(WANDB_PROJECT='PD_1', WANDB_ENTITY=ENTITY, WANDB_MODE='online',
      WANDB_RESUME='never', WANDB_RUN_GROUP='logger-engineering-validation',
      WANDB_RUN_ID='logger-hopper-seeds123-20261021-' + commit[:7], WANDB_DIR=str(output))
  from scripts.prepare_m2_v1 import resolve
  from dreamerv3.main import make_logger
  import wandb
  config = resolve(['--configs', 'm2_hopper_seeds123', '--seed', '1', '--logdir', str(output),
                    '--jax.prealloc', 'False', '--logger.outputs', 'jsonl', 'wandb'])
  logger = make_logger(config)
  run = wandb.run
  record = dict(checked_utc=now(), code_sha=commit, entity=run.entity, project=run.project,
                id=run.id, url=run.url, training_actions=0, agent_updates=0)
  if run.entity != ENTITY or run.project != 'PD_1' or run.config['seed'] != 1:
    raise RuntimeError('W&B destination/configuration differs')
  logger.add(dict(logger_validation=1), prefix='engineering')
  logger.write()
  logger.close()
  wandb.finish()
  remote = wandb.Api(timeout=30).run(f'{ENTITY}/PD_1/{record["id"]}')
  if remote.state != 'finished' or remote.summary['engineering/logger_validation'] != 1:
    raise RuntimeError('Online W&B write/read validation failed')
  record.update(remote_state=remote.state, jsonl_sha256=sha(output/'metrics.jsonl'))
  dump(output/'logger_validation.json', record)
  print(json.dumps(record), flush=True)


def bind(plan_path, evidence, output, authorized):
  if not authorized or platform.node() != 'lyg0326':
    raise RuntimeError('Explicit startup authorization and assigned host required')
  plan, evidence, output = read(plan_path), Path(evidence), Path(output)
  validate_matrix(plan)
  commit = clean_sha()
  if plan['code_sha'] != commit or output.exists():
    raise ValueError('Pinned plan differs or bound output already exists')
  logger = read(evidence/'logger-check/logger_validation.json')
  if logger['code_sha'] != commit or logger['remote_state'] != 'finished' or logger['entity'] != ENTITY:
    raise ValueError('Logger evidence differs')
  for row in plan['runs']:
    proof = read(evidence/f'gpu{row["physical_gpu"]}-binding.json')
    if (proof['code_sha'] != commit or proof['gpu_uuid'] != row['gpu_uuid'] or
        proof['physical_gpu'] != row['physical_gpu'] or not proof['verified_compute_and_graphics_same_gpu']):
      raise ValueError('GPU binding evidence differs')
    row.update(egl_index=proof['egl_index'], verified_compute_and_graphics_same_gpu=True,
               launch_argv=row['review_argv'], authorized_to_train=True,
               binding_evidence=str(evidence/f'gpu{row["physical_gpu"]}-binding.json'))
    row['environment_template'].update(MUJOCO_EGL_DEVICE_ID=str(proof['egl_index']), WANDB_ENTITY=ENTITY)
  plan.update(authorized_to_train=True, authorization='User: 启动实验; 挂上wandb (2026-10-21)',
              logger_evidence=str(evidence/'logger-check/logger_validation.json'), bound_utc=now())
  dump(output, plan)


def child_environment(template, inherited=None):
  env = dict(os.environ if inherited is None else inherited)
  for key in ('JAX_PLATFORM_NAME', 'JAX_PLATFORMS', 'XLA_FLAGS', 'JAX_MOCK_GPU_TOPOLOGY',
              'CUDA_MPS_PIPE_DIRECTORY', 'CUDA_MPS_LOG_DIRECTORY'):
    env.pop(key, None)
  if template['MUJOCO_EGL_DEVICE_ID'] is None:
    raise ValueError('Unbound EGL device')
  env.update(template)
  return env


def audit_run(row):
  """CPU audit of raw committed receipts/snapshots, not replay content dedupe."""
  from embodied.run.protocol_v1 import ProtocolState, audit_complete_outputs, audit_persisted_history
  from dreamerv3.freeze_c import freeze
  directory = Path(row['directory'])
  manifest, final = read(directory/'protocol_manifest.json'), read(directory/'final_state.json')
  if (manifest['protocol'] != HOPPER_123 or manifest['git_commit'] != row['code_sha'] or
      manifest['root_seed'] != row['seed'] or manifest['config'] != row['config']):
    raise ValueError('Actual manifest differs from bound configuration')
  state = ProtocolState(1000000, 50000, 10, 100000, 300000)
  for line in (directory/'update_receipts.jsonl').read_text().splitlines():
    r = json.loads(line)
    if not state.record_update(r['update_id'], r['start_action'], r['match_sum'],
                               r['match_count'], r['invalid_count']):
      raise ValueError('Duplicate persisted successful update')
  evaluations = [json.loads(line) for line in (directory/'evaluations.jsonl').read_text().splitlines()]
  if [r['action_step'] for r in evaluations] != list(state.grid):
    raise ValueError('Evaluation grid differs')
  if any(len(r['episodes']) != 10 or not all(e['complete'] for e in r['episodes']) for r in evaluations):
    raise ValueError('Incomplete evaluation episodes')
  if final['eval_action_steps'] != sum(sum(r['lengths']) for r in evaluations):
    raise ValueError('Evaluation action count differs')
  state.evaluations = {r['action_step']:{k:v for k,v in r.items() if k != 'action_step'} for r in evaluations}
  audit_persisted_history(directory, state)
  audit_complete_outputs(directory, state, row['mode'], row['code_sha'])
  if state.updates != final['train_batch_keys']:
    raise ValueError('Update/key count differs')
  artifact = freeze(directory, directory/'c_frozen.json') if row['mode'] == 'logging' else None
  return dict(run_id=row['run_id'], seed=row['seed'], group=row['group'],
              updates=state.updates, evaluations=21, c=artifact['c'] if artifact else None,
              c_artifact_sha256=sha(directory/'c_frozen.json') if artifact else None)


def finish(plan, output):
  output = Path(output)
  output.mkdir(parents=True, exist_ok=False)
  verified = [audit_run(row) for row in plan['runs']]
  index = output/'RUN_INDEX.json'
  dump(index, dict(runs=plan['runs']))
  subprocess.run([sys.executable, '-m', 'analysis.return_metrics', '--profile', 'hopper-seeds123',
                  '--index', str(index), '--output', str(output/'return-audit'), '--require-complete'],
                 cwd=REPO, check=True)
  from scripts.audit_formal_pd1 import publish
  tables = read(output/'return-audit/tables.json')
  for row in plan['runs']:
    metric = next(r for r in tables['seed_metrics'] if r['run_id'] == row['run_id'])
    values = {'formal/auc_return_per_budget':metric['auc_return_per_budget'],
              'formal/tail_mean_return':metric['tail_mean_return'], 'formal/return_metrics_complete':True}
    if row['mode'] == 'logging':
      c = read(Path(row['directory'])/'c_frozen.json')
      values.update({'formal/c':c['c'],'formal/match_sum':c['match_sum'],'formal/match_count':c['match_count']})
    else:
      pair = next(r for r in tables['paired_seed_differences'] if r['seed'] == row['seed'])
      values.update({'formal/dt_minus_baseline_auc':pair['auc_return_per_budget'],
                     'formal/dt_minus_baseline_tail':pair['tail_mean_return']})
    transport = publish(f'{ENTITY}/PD_1/{row["environment_template"]["WANDB_RUN_ID"]}',values)
    dump(output/f'wandb-seed{row["seed"]}-{row["mode"]}.json',transport)
  dump(output/'completion.json',dict(checked_utc=now(), code_sha=plan['code_sha'],
       verified_run_count=6, runs=verified, tables_sha256=sha(output/'return-audit/tables.json'),
       interpretation='Hopper seeds1/2/3 baseline-Dt repeat; no seed0 pooling or causal attribution.'))


def run(plan_path):
  plan = read(plan_path)
  validate_matrix(plan)
  if not plan['authorized_to_train'] or platform.node() != 'lyg0326' or clean_sha() != plan['code_sha']:
    raise RuntimeError('Authorization, host or pinned SHA mismatch')
  root = Path(plan['root'])
  if root.exists():
    raise FileExistsError('No resume or overwrite: ' + str(root))
  checks = [require_free(r['physical_gpu'],r['gpu_uuid']) for r in plan['runs']]
  root.mkdir(parents=True)
  dump(root/'prelaunch.json',dict(checked_utc=now(), checks=checks, supervisor_pid=os.getpid()))
  dump(root/'bound-plan.json',plan)
  handles = []
  for row in plan['runs']:
    directory = Path(row['directory'])
    try:
      check = require_free(row['physical_gpu'],row['gpu_uuid'])
      if not row['verified_compute_and_graphics_same_gpu'] or not row['launch_argv']:
        raise RuntimeError('Unverified launch')
      directory.mkdir(parents=True, exist_ok=False)
      (directory/'tmp').mkdir()
      log = (directory/'stdout-initial.log').open('w')
      process = subprocess.Popen(row['launch_argv'],cwd=REPO,
          env=child_environment(row['environment_template']),stdout=log,stderr=subprocess.STDOUT)
      row.update(state='running',pid=process.pid,start_utc=now(),start_epoch=time.time(),
                 prelaunch_check=check,command=row['launch_argv'], exit_code_available=True,
                 wandb_url=f'https://wandb.ai/{ENTITY}/PD_1/runs/{row["environment_template"]["WANDB_RUN_ID"]}')
      handles.append((row,process,log))
    except Exception as error:
      row.update(state='blocked' if not directory.exists() else 'failed',error=repr(error))
    if directory.exists():
      dump(directory/'execution-initial.json',row)
    dump(root/'actual-RUN_INDEX.json',dict(runs=plan['runs']))
  while any(r['state']=='running' for r in plan['runs']):
    sample = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
                                      '--format=csv,noheader,nounits'],text=True)
    with (root/'resources.jsonl').open('a') as file:
      file.write(json.dumps(dict(utc=now(),gpus=sample,free_disk_bytes=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize))+'\n')
    for row,process,log in handles:
      if row['state']!='running':
        continue
      result=process.poll()
      if result is None:
        continue
      log.close()
      row.update(state='training_finished_pending_audit' if result==0 else 'failed',
                 exit_code=result,end_utc=now(),elapsed_seconds=time.time()-row['start_epoch'])
      if result==0:
        try:
          row['run_audit']=audit_run(row)
          row['state']='run_audited_pending_group_and_WB'
        except Exception as error:
          row.update(state='audit_failed',audit_error=repr(error))
      dump(Path(row['directory'])/'execution-initial.json',row)
    dump(root/'actual-RUN_INDEX.json',dict(runs=plan['runs']))
    if any(r['state']=='running' for r in plan['runs']):
      time.sleep(30)
  status = dict(checked_utc=now(),runs=plan['runs'],all_training_exited=True,complete=False)
  if all(r['state']=='run_audited_pending_group_and_WB' for r in plan['runs']):
    try:
      finish(plan,root/'final-audit-attempt01')
      status['complete']=True
      for row in plan['runs']:
        row['state']='accepted'
    except Exception as error:
      status['audit_error']=repr(error)
  dump(root/'actual-RUN_INDEX.json',dict(runs=plan['runs']))
  dump(root/'supervisor-final.json',status)


if __name__ == '__main__':
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('action',choices=('probe','logger','bind','run','audit'))
  parser.add_argument('--plan',type=Path)
  parser.add_argument('--evidence',type=Path)
  parser.add_argument('--output',type=Path)
  parser.add_argument('--gpu',type=int)
  parser.add_argument('--uuid')
  parser.add_argument('--authorized',action='store_true')
  args=parser.parse_args()
  if args.action=='probe': probe(args.gpu,args.uuid,args.output)
  elif args.action=='logger': logger_check(args.output)
  elif args.action=='bind': bind(args.plan,args.evidence,args.output,args.authorized)
  elif args.action=='run': run(args.plan)
  else: finish(read(args.plan),args.output)
