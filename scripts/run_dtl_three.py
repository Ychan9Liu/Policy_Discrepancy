"""Closed clean DtL cohort authorized on 2026-11-05; never resumes a run."""
import argparse
import copy
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import signal
import subprocess
import sys
import time

from dreamerv3.protocol_scope import (
    DTL_THREE, DTL_THREE_TASKS, validate_scope, validate_dtl_scope)
from scripts.prepare_m2_v1 import resolve, check
from scripts.run_hopper_seeds123 import (
    REPO, ENTITY, now, read, dump, sha, clean_sha, require_free,
    child_environment, probe as device_probe)
from scripts.run_missing_dt import approved_context

HOST = 'lyg0360'
GPUS = (5, 6, 7)
UUIDS = ('GPU-ac733835-08ee-7a17-a2fa-1e49d62d385b',
         'GPU-548be1c8-86f0-b7a2-de92-5c455b12221b',
         'GPU-4cbf5b69-adb3-5210-2396-ca5d4e400f26')
PYTHON = '/data/Policy_Discrepancy/envs/dreamer/bin/python'
RUNS = PurePosixPath('/data/Policy_Discrepancy/runs')
SECONDS = 57600


def check_config(config):
  if validate_scope(config.script, config.task, config.seed,
                    config.agent.rep_probe.mode) != DTL_THREE:
    raise ValueError('Wrong DtL protocol')
  validate_dtl_scope(config)
  if tuple(config.logger.outputs) != ('jsonl', 'wandb'):
    raise ValueError('DtL cohort requires JSONL and online W&B')
  # Compare all inherited production settings without changing the actual config.
  check(config.update({'script': 'protocol_v1', 'task': 'dmc_hopper_hop',
      'agent.rep_probe.mode': 'logging', 'logger.outputs': ['jsonl']}))


def prepare(output, campaign):
  if (not campaign.startswith('dtl-three-clean-seed0-') or
      any(x in campaign for x in ('/', '\\', '..'))):
    raise ValueError('Use a fresh dtl-three-clean-seed0-* campaign')
  output = Path(output)
  output.mkdir(parents=True, exist_ok=False)
  commit = clean_sha()
  root = RUNS / campaign
  rows = []
  for task, gpu, uuid in zip(DTL_THREE_TASKS, GPUS, UUIDS):
    directory = str(root / task / 'dtl-attempt01')
    argv = [PYTHON, '-u', '-m', 'dreamerv3.main', '--configs', 'dtl_three',
        '--task', task, '--logdir', directory, '--jax.prealloc', 'False',
        '--logger.outputs', 'jsonl', 'wandb']
    config = resolve(argv[4:])
    check_config(config)
    rows.append(dict(task=task, seed=0, group='DtL', mode='off',
        protocol=DTL_THREE, code_sha=commit, server='sv2', host=HOST,
        physical_gpu=gpu, gpu_uuid=uuid, directory=directory,
        run_id=f'{campaign}:{task}:seed0:DtL:attempt01',
        state='prepared_not_started', argv=argv,
        config=json.loads(json.dumps(dict(config.flat))),
        environment=dict(CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
            MUJOCO_EGL_DEVICE_ID=None, MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
            JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
            PYTHONUNBUFFERED='1', TMPDIR=directory+'/tmp', WANDB_DIR=directory,
            WANDB_ENTITY=ENTITY, WANDB_PROJECT='PD_1', WANDB_MODE='online',
            WANDB_RESUME='never', WANDB_RUN_GROUP=campaign,
            WANDB_RUN_ID=f'{campaign}-{task[4:]}-dtl-01')))
  plan = dict(protocol=DTL_THREE, code_sha=commit, campaign=campaign,
      root=str(root), host=HOST, runs=rows, created_utc=now(),
      authorization='User: three clean DtL runs, seed0, sv2 physical GPU5/6/7',
      source_implementation='0b5d5a6c88d710dfaafd521dda177310a518d05b',
      interpretation='Exploratory online runs; no mechanism or method-effectiveness acceptance')
  validate(plan)
  dump(output/'plan.json', plan)


def validate(plan):
  if (plan['protocol'] != DTL_THREE or plan['host'] != HOST or
      len(plan['runs']) != 3 or
      PurePosixPath(plan['root']) != RUNS / plan['campaign'] or
      not plan['campaign'].startswith('dtl-three-clean-seed0-') or
      any(x in plan['campaign'] for x in ('/', '\\', '..'))):
    raise ValueError('DtL cohort root/host/scope differs')
  for row, task, gpu, uuid in zip(plan['runs'], DTL_THREE_TASKS, GPUS, UUIDS):
    directory = str(PurePosixPath(plan['root'])/task/'dtl-attempt01')
    if ((row['task'], row['seed'], row['physical_gpu'], row['gpu_uuid'],
         row['directory'], row['code_sha']) !=
        (task, 0, gpu, uuid, directory, plan['code_sha'])):
      raise ValueError('Task/seed/physical GPU/version differs')
    config = resolve(row['argv'][4:])
    check_config(config)
    if (row['argv'][:4] != [PYTHON, '-u', '-m', 'dreamerv3.main'] or
        config.task != task or config.logdir != directory or
        json.loads(json.dumps(dict(config.flat))) != row['config']):
      raise ValueError('Command and declared configuration differ')
    env = row['environment']
    if (env['CUDA_VISIBLE_DEVICES'] != uuid or env['WANDB_PROJECT'] != 'PD_1' or
        env['WANDB_ENTITY'] != ENTITY or env['WANDB_MODE'] != 'online' or
        env['WANDB_RESUME'] != 'never' or env['WANDB_DIR'] != directory or
        env['JAX_PLATFORMS'] != 'cuda'):
      raise ValueError('Device or logging configuration differs')


def audit(row, engineering=False):
  from embodied.run.protocol_v1 import (
      ProtocolState, audit_persisted_history, audit_complete_outputs,
      persisted_rows, seed32)
  from embodied.run.dtl_three import audit_dtl_receipts
  directory = Path(row['directory'])
  budget, interval, episodes = (2048, 2048, 1) if engineering else (1000000, 50000, 10)
  window = (2032, 2048) if engineering else (100000, 300000)
  state = ProtocolState(budget, interval, episodes, *window)
  manifest = read(directory/'protocol_manifest.json')
  if (manifest['protocol'] != DTL_THREE or manifest['git_commit'] != row['code_sha'] or
      manifest['config'] != row['config'] or manifest['task'] != row['task']):
    raise ValueError('Actual run manifest differs')
  for item in persisted_rows(directory/'update_receipts.jsonl'):
    if not state.record_update(item['update_id'], item['start_action'],
        item['match_sum'], item['match_count'], item['invalid_count']):
      raise ValueError('Duplicate successful update')
  evaluations = persisted_rows(directory/'evaluations.jsonl')
  if [r['action_step'] for r in evaluations] != list(state.grid):
    raise ValueError('Incomplete evaluation grid')
  for item in evaluations:
    if (len(item['episodes']) != episodes or item['lengths'] != [1000]*episodes or
        not all(e['complete'] for e in item['episodes']) or
        item['seeds'] != [seed32(0, row['task'], 'eval_env',
            item['action_step']//interval, i) for i in range(episodes)]):
      raise ValueError('Evaluation count, completeness or RNG mapping differs')
    point = item['action_step']
    saved = read(directory/'eval_snapshots'/f'{point:07d}'/'evaluation_snapshot.json')
    if saved['actual_action_step'] != point or saved['snapshot_id'] != item['snapshot_id']:
      raise ValueError('Evaluation snapshot identity differs')
  state.evaluations = {r['action_step']: {k:v for k,v in r.items() if k != 'action_step'}
                       for r in evaluations}
  audit_persisted_history(directory, state)
  audit_dtl_receipts(directory, state)
  audit_complete_outputs(directory, state, 'off', row['code_sha'])
  final = read(directory/'final_state.json')
  if (state.updates != (5 if engineering else 249493) or
      final['train_batch_keys'] != state.updates or
      final['eval_action_steps'] != len(state.grid)*episodes*1000 or
      final['match_count'] != 0):
    raise ValueError('Counters differ from the declared budget')
  eval_audits = persisted_rows(directory/'engineering_evaluation_audit.jsonl')
  if [x['action_step'] for x in eval_audits] != list(state.grid):
    raise ValueError('Evaluation training-state isolation audit missing')
  if engineering and len(persisted_rows(directory/'engineering_report_audit.jsonl')) != 1:
    raise ValueError('Actual CUDA report/read-only audit missing')
  return dict(task=row['task'], physical_gpu=row['physical_gpu'],
      directory=str(directory), updates=state.updates, evaluations=len(state.grid),
      eval_episodes=episodes, train_actions=budget,
      evaluation_isolation=True, dtl_receipts=state.updates, engineering=engineering)


def stop_owned(process):
  if process.poll() is None:
    os.killpg(process.pid, signal.SIGTERM)
    try:
      process.wait(timeout=20)
    except subprocess.TimeoutExpired:
      os.killpg(process.pid, signal.SIGKILL)
      process.wait(timeout=20)


def check_capacity(free, engineering=False, accept_initial_capacity_deficit=False):
  required = 10_000_000_000 if engineering else 180_000_000_000
  if free < required and not (not engineering and accept_initial_capacity_deficit):
    raise RuntimeError('Insufficient capacity for the declared three-card launch')
  if free < 5_000_000_000:
    raise RuntimeError('Less than 5GB free; unable to safely create run artifacts')
  return required


def execute(plan_path, evidence, engineering=False, accept_initial_capacity_deficit=False):
  plan = read(plan_path)
  validate(plan)
  commit = clean_sha()
  if commit != plan['code_sha'] or platform.node() != HOST:
    raise RuntimeError('Exact clean SHA and assigned sv2 required')
  evidence = Path(evidence)
  if not engineering:
    proof = read(evidence/'integration/acceptance.json')
    if (proof['code_sha'] != commit or not proof['passed'] or
        [(r['task'],r['physical_gpu']) for r in proof['runs']] != list(zip(DTL_THREE_TASKS,GPUS))):
      raise ValueError('Same-SHA three-task CUDA acceptance required')
  root = evidence/'integration' if engineering else Path(plan['root'])
  if root.exists():
    raise FileExistsError('No resume, silent overwrite, or automatic retry: '+str(root))
  free = shutil.disk_usage('/data').free
  required = check_capacity(free, engineering, accept_initial_capacity_deficit)
  root.mkdir(parents=True)
  handles = []
  rows = copy.deepcopy(plan['runs'])
  before = {g: require_free(g,u) for g,u in zip(GPUS,UUIDS)}
  dump(root/'resource-before.json', dict(utc=now(), gpus=before,
      free_bytes=shutil.disk_usage('/data').free, initial_required_bytes=required,
      accepted_initial_capacity_deficit=accept_initial_capacity_deficit,
      capacity_exception_authorization='User 2026-11-05: start now; user will clean storage'
          if accept_initial_capacity_deficit else None))
  def terminated(signum, frame):
    raise SystemExit(f'Supervisor received signal {signum}; stopping owned children')
  previous_handler = signal.signal(signal.SIGTERM, terminated)
  try:
    for row in rows:
      gpu = row['physical_gpu']
      proof = read(evidence/f'gpu{gpu}-binding.json')
      if (proof['code_sha'] != commit or proof['gpu_uuid'] != row['gpu_uuid'] or
          not proof['verified_compute_and_graphics_same_gpu']):
        raise ValueError('CUDA/EGL binding differs')
      row['resource_before_start'] = require_free(gpu, row['gpu_uuid'])
      directory = root/row['task']/'dtl-attempt01'
      argv = row['argv'][:]
      if engineering:
        argv += ['--logdir', str(directory), '--logger.outputs', 'jsonl',
            '--run.engineering_fixture', 'True', '--run.action_budget', '2048',
            '--run.eval_every_actions', '2048', '--run.eval_eps', '1',
            '--run.match_start', '2032', '--run.match_end', '2048',
            '--run.report_every_actions', '2048']
      row.update(directory=str(directory), argv=argv,
          config=json.loads(json.dumps(dict(resolve(argv[4:]).flat))))
      directory.mkdir(parents=True, exist_ok=False)
      (directory/'tmp').mkdir()
      env = row['environment']
      env.update(MUJOCO_EGL_DEVICE_ID=str(proof['egl_index']),
          TMPDIR=str(directory/'tmp'), WANDB_DIR=str(directory))
      log = (directory/'worker.log').open('x', encoding='utf-8')
      try:
        process = subprocess.Popen(argv, cwd=REPO,
            env=child_environment(env, inherited=approved_context()),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
      except BaseException:
        log.close()
        raise
      row.update(state='running', pid=process.pid, process_group_id=process.pid,
                 start_utc=now(), started_epoch=time.time())
      handles.append((row, process, log))
      dump(directory/'execution-initial.json', row)
    deadline = time.monotonic() + (2400 if engineering else SECONDS)
    while any(p.poll() is None for _,p,_ in handles):
      if time.monotonic() > deadline:
        raise TimeoutError('Attempt exceeded its fixed engineering wall-clock limit')
      free = shutil.disk_usage('/data').free
      with (root/'resources.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(utc=now(), free_bytes=free))+'\n')
      if free < 5_000_000_000:
        raise RuntimeError('Storage below 5GB; stopping only this supervisor owned jobs')
      for row, process, log in handles:
        if row['state'] != 'running' or process.poll() is None:
          continue
        log.close()
        row.update(exit_code=process.returncode, ended_utc=now(),
                   state='training_finished_pending_audit' if process.returncode == 0 else 'failed')
      dump(root/'actual-RUN_INDEX.json', dict(code_sha=commit, runs=rows))
      time.sleep(10)
    completed = []
    for row, process, log in handles:
      log.close()
      row.update(exit_code=process.returncode, ended_utc=now())
      if process.returncode:
        row['state'] = 'failed'
        continue
      row['state'] = 'training_finished_pending_audit'
      try:
        row['audit'] = audit(row, engineering)
        completed.append(row['audit'])
        row['state'] = 'accepted'
      except Exception as error:
        row.update(state='failed_audit', error=repr(error))
    dump(root/'actual-RUN_INDEX.json', dict(code_sha=commit, runs=rows))
    passed = len(completed) == 3
    dump(root/('acceptance.json' if engineering else 'supervisor-final.json'),
        dict(code_sha=commit, checked_utc=now(), passed=passed, runs=completed,
             engineering_only=engineering, final_states=[r['state'] for r in rows]))
    if not passed:
      raise RuntimeError('One or more DtL runs failed; original evidence retained')
  except BaseException as error:
    dump(root/'supervisor-error.json', dict(code_sha=commit, utc=now(), error=repr(error), runs=rows))
    raise
  finally:
    for _, process, log in handles:
      stop_owned(process)
      log.close()
    signal.signal(signal.SIGTERM, previous_handler)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('action', choices=('prepare', 'probe', 'integration', 'run'))
  for key in ('output', 'plan', 'evidence'):
    parser.add_argument('--'+key, type=Path)
  parser.add_argument('--campaign')
  parser.add_argument('--gpu', type=int, choices=GPUS)
  parser.add_argument('--accept-initial-capacity-deficit', action='store_true')
  args = parser.parse_args()
  if args.action == 'prepare':
    prepare(args.output, args.campaign)
  elif args.action == 'probe':
    uuid = UUIDS[GPUS.index(args.gpu)]
    os.environ.update(CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
        MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', JAX_PLATFORMS='cuda',
        XLA_PYTHON_CLIENT_PREALLOCATE='false')
    device_probe(args.gpu, uuid, args.output, host=HOST, allowed_gpus=GPUS)
  else:
    execute(args.plan, args.evidence, args.action == 'integration',
            args.accept_initial_capacity_deficit)


if __name__ == '__main__':
  main()
