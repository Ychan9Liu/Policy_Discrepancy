"""Explicit sv2 GPU0--3 baseline campaign; preparation never trains.

Run only under the user's allocation. No resume, other modes, seed changes,
resource substitution or retries. Raw failures and successful sources remain.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.prepare_m2_v1 import TASKS, resolve, check

UUIDS = (
    'GPU-76344ce3-10a1-952d-68f5-7720a3cda667',
    'GPU-1e27fd5e-4231-926a-754f-e7d4c37d41ab',
    'GPU-24608f8d-a512-70b2-8578-45945b57bc35',
    'GPU-92630efa-be4e-7074-d29a-ccff0ad7f801')

def now():
  return datetime.now(timezone.utc).isoformat()

def dump(path, value):
  path = Path(path)
  tmp = path.with_suffix(path.suffix + '.tmp')
  tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
  tmp.replace(path)

def sha(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def clean_sha():
  if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO):
    raise RuntimeError('Dirty checkout')
  return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                               cwd=REPO, text=True).strip()

def gpu_state():
  lines = subprocess.check_output(['nvidia-smi',
      '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'],
      text=True).splitlines()
  return {int(x[0]): (x[1], int(x[2])) for line in lines
          for x in [[s.strip() for s in line.split(',')]]}

def require_free(gpu):
  uuid, memory = gpu_state()[gpu]
  pmon = subprocess.check_output(['nvidia-smi', 'pmon', '-c', '1'], text=True)
  jobs = [line for line in pmon.splitlines() if line.strip()
          and not line.startswith('#') and line.split()[0] == str(gpu)
          and line.split()[1] != '-']
  if uuid != UUIDS[gpu] or memory != 0 or jobs:
    raise RuntimeError(f'Assigned physical GPU{gpu} unavailable; no substitution')
  return dict(checked_utc=now(), physical_gpu=gpu, uuid=uuid,
              memory_used_mib=memory, pmon=pmon)

def prepare(output, campaign, entity):
  commit = clean_sha()
  output = Path(output)
  if output.exists():
    raise FileExistsError(output)
  root = Path('/data/Policy_Discrepancy/runs') / campaign
  if root.exists() or not campaign.startswith('formal-m2-v1-') or '/' in campaign:
    raise ValueError('A new formal campaign directory is required')
  if platform.node() != 'lyg0360':
    raise RuntimeError('Only the user-assigned sv2 is allowed')
  output.mkdir(parents=True)
  rows = []
  for gpu, task in enumerate(TASKS):
    argv = ['--configs', 'm2_v1', '--task', task,
            '--logdir', str(root / task / 'logging-attempt01'),
            '--agent.rep_probe.mode', 'logging', '--jax.prealloc', 'False',
            '--logger.outputs', 'jsonl', 'wandb']
    config = resolve(argv)
    # Reuse the frozen scientific checker; only the approved logger differs.
    check(config.update({'logger.outputs': ['jsonl']}))
    assert tuple(config.logger.outputs) == ('jsonl', 'wandb')
    config.save(str(output / (task + '.yaml')))
    dump(output / (task + '.json'), dict(config.flat))
    rows.append(dict(run_id=campaign + ':' + task + ':logging:attempt01',
        task=task, group='baseline', seed=0, state='not_started',
        server='sv2', physical_gpu=gpu, uuid=UUIDS[gpu], cuda_logical_device=0,
        egl_device_id=gpu, directory=config.logdir,
        source_directory=config.logdir, config=dict(config.flat),
        command=[sys.executable, '-u', '-m', 'dreamerv3.main', *argv],
        wandb_id=campaign + '-' + task + '-logging-01',
        wandb_project='PD_1', wandb_entity=entity,
        config_sha256=sha(output / (task + '.yaml'))))
  dump(output / 'RUN_INDEX.json', dict(runs=rows))
  dump(output / 'plan.json', dict(campaign=campaign, code_sha=commit,
      created_utc=now(), host=platform.node(), root=str(root), runs=rows,
      output=str(output.resolve()), training_started=False,
      authorization='User: sv2 physical GPU0-3; four formal logging baselines; '
        'prealloc=False, JSONL+W&B PD_1, no scope'))
  print(json.dumps(dict(plan=str(output / 'plan.json'), code_sha=commit)))

def smoke(output):
  """Actual production logger, CPU scalar IO only, separate engineering run."""
  output = Path(output)
  output.mkdir(parents=True, exist_ok=False)
  os.environ.update(WANDB_PROJECT='PD_1', WANDB_MODE='online',
      WANDB_RUN_GROUP='logger-engineering-validation', WANDB_RESUME='never',
      WANDB_RUN_ID='logger-check-20261009-01', WANDB_DIR=str(output))
  config = resolve(['--configs', 'm2_v1', '--task', TASKS[0],
      '--logdir', str(output), '--agent.rep_probe.mode', 'logging',
      '--jax.prealloc', 'False', '--logger.outputs', 'jsonl', 'wandb'])
  from dreamerv3.main import make_logger
  import wandb
  logger = make_logger(config)
  run = wandb.run
  result = dict(checked_utc=now(), code_sha=clean_sha(), project=run.project,
      entity=run.entity, id=run.id, url=run.url,
      sdk=wandb.__version__, training_actions=0, agent_updates=0)
  assert run.project == 'PD_1'
  assert run.config['run.action_budget'] == 1000000
  assert run.config['logger.outputs'] == ['jsonl', 'wandb']
  logger.add(dict(logger_validation=1.0), prefix='engineering')
  logger.write()
  logger.close()
  wandb.finish()
  remote = wandb.Api().run(f'{result["entity"]}/PD_1/{result["id"]}')
  assert remote.state == 'finished'
  assert remote.summary['engineering/logger_validation'] == 1
  result['remote_state'] = remote.state
  result['jsonl_sha256'] = sha(output / 'metrics.jsonl')
  dump(output / 'logger_validation.json', result)
  print(json.dumps(result))

def run(plan_path):
  plan = json.loads(Path(plan_path).read_text())
  if clean_sha() != plan['code_sha'] or platform.node() != plan['host']:
    raise RuntimeError('Plan source/host differs')
  root = Path(plan['root'])
  if root.exists():
    raise FileExistsError('No continuation or overwrite: ' + str(root))
  checks = [require_free(i) for i in range(4)]
  root.mkdir(parents=True)
  dump(root / 'prelaunch.json', dict(utc=now(), checks=checks,
      code_sha=plan['code_sha'], supervisor_pid=os.getpid()))
  dump(root / 'plan.json', plan)
  handles = []
  for row in plan['runs']:
    require_free(row['physical_gpu'])
    directory = Path(row['directory'])
    directory.mkdir(parents=True)
    (directory / 'tmp').mkdir()
    env = dict(os.environ)
    env.pop('JAX_PLATFORMS', None)
    env.pop('JAX_PLATFORM_NAME', None)
    env.update(CUDA_VISIBLE_DEVICES=row['uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
        MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID=str(row['egl_device_id']),
        PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1', TMPDIR=str(directory / 'tmp'),
        WANDB_PROJECT='PD_1', WANDB_ENTITY=row['wandb_entity'],
        WANDB_MODE='online', WANDB_RESUME='never', WANDB_RUN_ID=row['wandb_id'],
        WANDB_RUN_GROUP=plan['campaign'], WANDB_JOB_TYPE='formal-baseline',
        WANDB_TAGS='M2-v1,formal,logging,seed0,' + row['task'],
        WANDB_DIR=str(directory))
    record_env = {k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
        'MUJOCO_GL', 'MUJOCO_EGL_DEVICE_ID', 'PYOPENGL_PLATFORM',
        'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'TMPDIR',
        'WANDB_PROJECT', 'WANDB_ENTITY', 'WANDB_MODE', 'WANDB_RESUME',
        'WANDB_RUN_ID', 'WANDB_RUN_GROUP', 'WANDB_JOB_TYPE', 'WANDB_TAGS', 'WANDB_DIR')}
    log = (directory / 'stdout-initial.log').open('w')
    process = subprocess.Popen(row['command'], cwd=REPO, env=env,
        stdout=log, stderr=subprocess.STDOUT)
    row.update(state='running', pid=process.pid, start_utc=now(),
        start_epoch=time.time(), code_sha=plan['code_sha'], environment=record_env,
        wandb_url=f'https://wandb.ai/{row["wandb_entity"]}/PD_1/runs/{row["wandb_id"]}')
    dump(directory / 'execution-initial.json', row)
    handles.append((row, process, log))
  dump(root / 'RUN_INDEX.json', dict(runs=plan['runs']))
  while any(row['state'] == 'running' for row, _, _ in handles):
    for row, process, log in handles:
      if row['state'] != 'running':
        continue
      directory = Path(row['directory'])
      rc = process.poll()
      resources = dict(utc=now(), gpu=gpu_state()[row['physical_gpu']],
          free_disk_bytes=os.statvfs(root).f_bavail * os.statvfs(root).f_frsize)
      with (directory / 'resources.jsonl').open('a') as f:
        f.write(json.dumps(resources) + '\n')
      if rc is None:
        continue
      log.close()
      row.update(exit_code=rc, end_utc=now(), elapsed_seconds=time.time()-row['start_epoch'],
                 state='training_finished_pending_audit' if rc == 0 else 'failed')
      dump(directory / 'execution-initial.json', row)
      if rc == 0:
        try:
          from dreamerv3.freeze_c import freeze
          frozen = freeze(directory, directory / 'c_frozen.json')
          row.update(c=frozen['c'], match_sum=frozen['match_sum'],
              match_count=frozen['match_count'], state='c_frozen_pending_return_audit')
        except Exception as e:
          row.update(state='audit_failed', audit_error=type(e).__name__ + ': ' + str(e))
      dump(directory / 'execution-initial.json', row)
    dump(root / 'RUN_INDEX.json', dict(runs=plan['runs']))
    if any(row['state'] == 'running' for row, _, _ in handles):
      time.sleep(30)
  output = Path(plan['output']) / 'return-audit'
  with (root / 'return-audit.log').open('w') as log:
    rc = subprocess.call([sys.executable, '-m', 'analysis.return_metrics',
        '--index', str(root / 'RUN_INDEX.json'), '--output', str(output)],
        cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
  dump(root / 'supervisor-final.json', dict(utc=now(), runs=plan['runs'],
      return_audit_exit_code=rc, output=str(output),
      note='Other groups intentionally absent. Final acceptance requires '
        'baseline metric/snapshot/receipt/device and W&B upload audit.'))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('action', choices=['prepare', 'smoke', 'run'])
  parser.add_argument('--output')
  parser.add_argument('--campaign')
  parser.add_argument('--entity')
  parser.add_argument('--plan')
  args = parser.parse_args()
  if args.action == 'prepare':
    prepare(args.output, args.campaign, args.entity)
  elif args.action == 'smoke':
    smoke(args.output)
  else:
    run(args.plan)
