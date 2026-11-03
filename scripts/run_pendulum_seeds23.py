"""Four approved repeats: sv3 GPU1 seed2, GPU4 seed3; audited serial pairs."""
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from dreamerv3.protocol_scope import PENDULUM_23
from scripts.prepare_pendulum_seeds23 import HOST, TASK, CARDS, PREFIX, check_config, prepare
from scripts.prepare_m2_v1 import resolve, PYTHON
from scripts.run_hopper_seeds123 import (
    REPO, ENTITY, read, dump, sha, now, clean_sha, require_free, audit_run,
    child_environment, probe)
from scripts.run_missing_dt import approved_context

AUDITED = 'run_audited_pending_group_and_WB'


def validate(plan):
  rows = plan['runs']
  expected = [(s, m, g, i) for s, g in CARDS.items() for i, m in enumerate(('logging', 'dt'), 1)]
  if (plan['protocol'] != PENDULUM_23 or plan['server'] != 'sv3' or plan['host'] != HOST or
      not plan['campaign'].startswith(PREFIX) or any(x in plan['campaign'] for x in ('/', '\\', '..')) or
      plan['root'] != '/data/Policy_Discrepancy/runs/'+plan['campaign'] or
      [(r['seed'], r['mode'], r['physical_gpu'], r['queue_position']) for r in rows] != expected or
      plan['required_storage_bytes'] != 240_000_000_000 or plan['auto_start_when_idle']):
    raise ValueError('Wrong exact host, scope, matrix, storage or queue')
  for r in rows:
    directory = f'{plan["root"]}/seed{r["seed"]}/{r["mode"]}-attempt01'
    wid = f'{plan["campaign"]}-pendulum-seed{r["seed"]}-{r["mode"]}-01'
    c = resolve(r['review_argv'][4:]); check_config(c)
    if (r['review_argv'][:4] != [PYTHON, '-u', '-m', 'dreamerv3.main'] or
        r['code_sha'] != plan['code_sha'] or r['protocol'] != PENDULUM_23 or
        r['host'] != HOST or r['server'] != 'sv3' or r['task'] != TASK or
        type(r['seed']) is not int or c.seed != r['seed'] or c.task != TASK or
        c.agent.rep_probe.mode != r['mode'] or c.logdir != directory or
        r['directory'] != directory or r['run_id'] != 'sv3:'+wid or
        r['group'] != ('baseline' if r['mode']=='logging' else 'dt') or
        json.loads(json.dumps(dict(c.flat))) != r['config']):
      raise ValueError('Declared identity/command/full config differs')
    original = plan['references'][r['group']]['config']
    differences = {k: [original.get(k), r['config'].get(k)]
        for k in original.keys() | r['config'].keys() if original.get(k) != r['config'].get(k)}
    if set(differences) != {'script', 'seed', 'logdir'} or differences != r['reference_config_differences']:
      raise ValueError('Scientific configuration differs from original pendulum')
    e = r['environment_template']
    expected_env = dict(CUDA_VISIBLE_DEVICES=r['gpu_uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
        MUJOCO_EGL_DEVICE_ID=str(r['egl_index']) if plan['authorized_to_train'] else None,
        MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', JAX_PLATFORMS='cuda',
        XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1', TMPDIR=directory+'/tmp', WANDB_DIR=directory,
        WANDB_ENTITY=ENTITY, WANDB_PROJECT='PD_1', WANDB_MODE='online', WANDB_RESUME='never',
        WANDB_RUN_GROUP=plan['campaign'], WANDB_RUN_ID=wid)
    if e != expected_env:
      raise ValueError('Execution environment differs')
    if plan['authorized_to_train']:
      if (not r['authorized_to_train'] or r['launch_argv'] != r['review_argv'] or
          not r['verified_compute_and_graphics_same_gpu'] or type(r['egl_index']) is not int):
        raise ValueError('Missing authorized binding')
    elif r['authorized_to_train'] or r['launch_argv'] is not None:
      raise ValueError('Unbound plan cannot contain launch commands')
  if len({r['gpu_uuid'] for r in rows}) != 2:
    raise ValueError('Expected two distinct devices')
  for seed in CARDS:
    pair = [r for r in rows if r['seed']==seed]
    if pair[0]['gpu_uuid'] != pair[1]['gpu_uuid']:
      raise ValueError('Same seed pair must share its assigned device serially')


def logger_check(output):
  output = Path(output); output.mkdir(parents=True, exist_ok=False)
  commit = clean_sha()
  os.environ.update(WANDB_PROJECT='PD_1', WANDB_ENTITY=ENTITY, WANDB_MODE='online',
      WANDB_RESUME='never', WANDB_DIR=str(output), WANDB_RUN_GROUP='logger-engineering-validation',
      WANDB_RUN_ID='logger-pendulum-seeds23-'+commit[:12])
  from dreamerv3.main import make_logger
  import wandb
  c = resolve(['--configs', 'm2_pendulum_seeds23', '--seed', '2', '--logdir', str(output),
               '--jax.prealloc', 'False', '--logger.outputs', 'jsonl', 'wandb'])
  check_config(c); logger = make_logger(c); remote = wandb.run
  path = remote.entity+'/PD_1/'+remote.id
  if remote.config['seed'] != 2 or remote.config['task'] != TASK:
    raise RuntimeError('Wrong online logger configuration')
  logger.add(dict(logger_validation=1), prefix='engineering'); logger.write(); logger.close(); wandb.finish()
  remote = wandb.Api(timeout=30).run(path)
  if remote.state != 'finished' or remote.summary['engineering/logger_validation'] != 1:
    raise RuntimeError('Online logger readback failed')
  dump(output/'logger_validation.json', dict(code_sha=commit, remote_state=remote.state,
      entity=remote.entity, project=remote.project, seed=2, task=TASK, url=remote.url,
      training_actions=0, agent_updates=0, jsonl_sha256=sha(output/'metrics.jsonl')))


def start(row):
  before = require_free(row['physical_gpu'], row['gpu_uuid'])
  directory = Path(row['directory'])
  directory.mkdir(parents=True, exist_ok=False); (directory/'tmp').mkdir()
  log = (directory/'stdout-initial.log').open('x')
  try:
    process = subprocess.Popen(row['launch_argv'], cwd=REPO,
        env=child_environment(row['environment_template'], inherited=approved_context()),
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
  except BaseException:
    log.close(); raise
  row.update(state='running', pid=process.pid, process_group_id=process.pid,
      start_utc=now(), start_epoch=time.time(), prelaunch_check=before,
      wandb_url=f'https://wandb.ai/{ENTITY}/PD_1/runs/{row["environment_template"]["WANDB_RUN_ID"]}')
  dump(directory/'execution-initial.json', row)
  return process, log


def integration(plan_path, evidence):
  plan, evidence = read(plan_path), Path(evidence); validate(plan)
  commit = clean_sha()
  if plan['code_sha'] != commit or platform.node() != HOST:
    raise RuntimeError('Engineering host/SHA differs')
  output = evidence/'integration'; output.mkdir(parents=True, exist_ok=False)
  completed = []
  for mode in ('logging', 'dt'):
    children = []
    for source in (r for r in plan['runs'] if r['mode']==mode):
      row = copy.deepcopy(source); proof = read(evidence/f'gpu{row["physical_gpu"]}-binding.json')
      if (proof['code_sha'] != commit or proof['gpu_uuid'] != row['gpu_uuid'] or
          proof['physical_gpu'] != row['physical_gpu'] or not proof['verified_compute_and_graphics_same_gpu']):
        raise ValueError('Engineering device binding differs')
      directory = output/f'seed{row["seed"]}'/mode
      argv = row['review_argv'] + ['--logdir', str(directory), '--logger.outputs', 'jsonl',
          '--run.engineering_fixture', 'True', '--run.action_budget', '4098',
          '--run.eval_every_actions', '2049', '--run.eval_eps', '1',
          '--run.match_start', '2200', '--run.match_end', '3600', '--run.report_every_actions', '2049']
      row.update(directory=str(directory), launch_argv=argv)
      row['environment_template'].update(MUJOCO_EGL_DEVICE_ID=str(proof['egl_index']),
          TMPDIR=str(directory/'tmp'), WANDB_DIR=str(directory))
      process, log = start(row); children.append((row, process, log))
    dump(output/f'active-{mode}.json', dict(runs=[r for r, _, _ in children]))
    while any(p.poll() is None for _, p, _ in children):
      time.sleep(10)
    for row, process, log in children:
      log.close(); directory = Path(row['directory'])
      if process.returncode:
        raise RuntimeError(f'Engineering CUDA run failed: {directory}, rc={process.returncode}')
      from embodied.run.protocol_v1 import ProtocolState, audit_persisted_history, audit_complete_outputs
      state = ProtocolState(4098, 2049, 1, 2200, 3600)
      for line in (directory/'update_receipts.jsonl').read_text().splitlines():
        r = json.loads(line)
        if not state.record_update(r['update_id'], r['start_action'], r['match_sum'], r['match_count'], r['invalid_count']):
          raise ValueError('Duplicate engineering update')
      evaluations = [json.loads(l) for l in (directory/'evaluations.jsonl').read_text().splitlines()]
      if ([r['action_step'] for r in evaluations] != list(state.grid) or
          any(len(r['episodes']) != 1 or not r['episodes'][0]['complete'] for r in evaluations)):
        raise ValueError('Engineering evaluation differs')
      state.evaluations = {r['action_step']: {k: v for k, v in r.items() if k!='action_step'} for r in evaluations}
      audit_persisted_history(directory, state); audit_complete_outputs(directory, state, row['mode'], commit)
      manifest = read(directory/'protocol_manifest.json')
      if (manifest['protocol'] != PENDULUM_23 or manifest['root_seed'] != row['seed'] or
          manifest['config'] != json.loads(json.dumps(dict(resolve(row['launch_argv'][4:]).flat)))):
        raise ValueError('Actual engineering seed/config differs')
      reports = (directory/'engineering_report_audit.jsonl').read_text().splitlines()
      eval_audits = (directory/'engineering_evaluation_audit.jsonl').read_text().splitlines()
      if len(reports) != 2 or len(eval_audits) != 3:
        raise ValueError('Read-only report/evaluation audits missing')
      completed.append(dict(task=TASK, seed=row['seed'], mode=row['mode'],
          physical_gpu=row['physical_gpu'], directory=str(directory), updates=state.updates,
          match_count=state.match_count, reports=2, evaluations=3, exit_code=process.returncode))
    dump(output/f'completed-{mode}.json', completed)
  dump(output/'acceptance.json', dict(code_sha=commit, passed=True, run_count=4,
      engineering_only=True, budget=4098, evaluation_episodes=1, runs=completed, checked_utc=now()))


def bind(plan_path, evidence, output):
  plan = read(plan_path); validate(plan); evidence = Path(evidence)
  if Path(output).exists() or clean_sha() != plan['code_sha'] or platform.node() != HOST:
    raise RuntimeError('Wrong clean SHA/host or existing output')
  cpu = read(evidence/'cpu-acceptance.json')
  a = read(evidence/'integration/acceptance.json'); logger = read(evidence/'logger-check/logger_validation.json')
  if (not cpu['passed'] or cpu['code_sha'] != plan['code_sha'] or
      not a['passed'] or a['code_sha'] != plan['code_sha'] or a['run_count'] != 4 or
      {(r['task'], r['seed'], r['mode'], r['physical_gpu']) for r in a['runs']} !=
      {(r['task'], r['seed'], r['mode'], r['physical_gpu']) for r in plan['runs']} or
      logger['code_sha'] != plan['code_sha'] or logger['remote_state'] != 'finished' or
      logger['entity'] != ENTITY or logger['project'] != 'PD_1' or logger['seed'] != 2):
    raise ValueError('Missing exact CPU / four CUDA / online logger acceptance')
  for r in plan['runs']:
    p = read(evidence/f'gpu{r["physical_gpu"]}-binding.json')
    if (p['code_sha'] != plan['code_sha'] or p['gpu_uuid'] != r['gpu_uuid'] or
        p['physical_gpu'] != r['physical_gpu'] or not p['verified_compute_and_graphics_same_gpu']):
      raise ValueError('Unverified device')
    r.update(launch_argv=r['review_argv'], egl_index=p['egl_index'],
        verified_compute_and_graphics_same_gpu=True, authorized_to_train=True, state='queued')
    r['environment_template']['MUJOCO_EGL_DEVICE_ID'] = str(p['egl_index'])
  plan.update(authorized_to_train=True, bound_utc=now(), engineering_evidence=str(evidence))
  validate(plan); dump(output, plan)


def next_for_gpu(rows, gpu):
  pair = [r for r in rows if r['physical_gpu']==gpu]
  if any(r['state']=='running' for r in pair):
    return None
  baseline = next(r for r in pair if r['mode']=='logging')
  dt = next(r for r in pair if r['mode']=='dt')
  if baseline['state']=='queued':
    return baseline
  if dt['state']=='queued':
    if baseline['state']==AUDITED:
      return dt
    dt.update(state='blocked', error='Baseline has not passed full run audit')
  return None


def finish(plan, output):
  output = Path(output); output.mkdir(parents=True, exist_ok=False)
  verified = [audit_run(r, PENDULUM_23) for r in plan['runs']]
  dump(output/'RUN_INDEX.json', dict(runs=plan['runs']))
  subprocess.run([sys.executable, '-m', 'analysis.return_metrics', '--profile', 'pendulum-seeds23',
      '--index', str(output/'RUN_INDEX.json'), '--output', str(output/'return-audit'),
      '--require-complete'], cwd=REPO, check=True)
  tables = read(output/'return-audit/tables.json')
  from scripts.audit_formal_pd1 import publish
  for row in plan['runs']:
    metric = next(r for r in tables['seed_metrics'] if r['run_id']==row['run_id'])
    values = {'formal/auc_return_per_budget': metric['auc_return_per_budget'],
        'formal/tail_mean_return': metric['tail_mean_return'], 'formal/return_metrics_complete': True}
    if row['mode']=='logging':
      c = read(Path(row['directory'])/'c_frozen.json')
      values.update({'formal/c': c['c'], 'formal/match_sum': c['match_sum'], 'formal/match_count': c['match_count']})
    else:
      pair = next(r for r in tables['paired_seed_differences'] if r['seed']==row['seed'])
      values.update({'formal/dt_minus_baseline_auc': pair['auc_return_per_budget'],
                    'formal/dt_minus_baseline_tail': pair['tail_mean_return']})
    transport = publish(f'{ENTITY}/PD_1/{row["environment_template"]["WANDB_RUN_ID"]}', values)
    dump(output/f'wandb-seed{row["seed"]}-{row["mode"]}.json', transport)
  dump(output/'completion.json', dict(checked_utc=now(), code_sha=plan['code_sha'],
      verified_run_count=4, runs=verified, tables_sha256=sha(output/'return-audit/tables.json'),
      interpretation='Pendulum seeds2/3 only, no seed0/1 pooling, causal or significance claim.'))


def run(plan_path):
  plan = read(plan_path); validate(plan); root = Path(plan['root'])
  if root.exists():
    raise FileExistsError('No overwrite/resume')
  if not plan['authorized_to_train'] or platform.node()!=HOST or clean_sha()!=plan['code_sha']:
    raise RuntimeError('Wrong authorization/host/pinned clean SHA')
  checks = [require_free(g, next(r['gpu_uuid'] for r in plan['runs'] if r['physical_gpu']==g)) for g in CARDS.values()]
  if shutil.disk_usage(root.parent).free < plan['required_storage_bytes']:
    raise RuntimeError('Insufficient four-run capacity; no deletion or protocol change')
  root.mkdir(); dump(root/'bound-plan.json', plan)
  dump(root/'prelaunch.json', dict(checked_utc=now(), checks=checks, supervisor_pid=os.getpid()))
  handles = []
  while True:
    for row, process, log in handles:
      if row['state']!='running' or process.poll() is None:
        continue
      log.close(); row.update(state='training_finished_pending_audit' if process.returncode==0 else 'failed',
          exit_code=process.returncode, end_utc=now(), elapsed_seconds=time.time()-row['start_epoch'])
      if process.returncode==0:
        try:
          row['run_audit'] = audit_run(row, PENDULUM_23); row['state'] = AUDITED
        except Exception as error:
          row.update(state='audit_failed', error=repr(error))
      dump(Path(row['directory'])/'execution-final.json', row)
    for gpu in CARDS.values():
      row = next_for_gpu(plan['runs'], gpu)
      if row is None:
        continue
      try:
        if shutil.disk_usage(root).free < 60_000_000_000:
          raise RuntimeError('Storage below per-run floor')
        process, log = start(row); handles.append((row, process, log))
      except Exception as error:
        row.update(state='blocked', error=repr(error))
    dump(root/'actual-RUN_INDEX.json', dict(runs=plan['runs']))
    if not any(r['state'] in ('running','queued') for r in plan['runs']):
      break
    time.sleep(30)
  if all(r['state']==AUDITED for r in plan['runs']):
    try:
      finish(plan, root/'final-audit-attempt01')
      for r in plan['runs']:
        r['state'] = 'accepted'
    except Exception as error:
      dump(root/'postprocess-failed.json', dict(error=repr(error), checked_utc=now()))
  dump(root/'actual-RUN_INDEX.json', dict(runs=plan['runs']))
  dump(root/'supervisor-final.json', dict(checked_utc=now(), all_training_exited=True,
      complete=all(r['state']=='accepted' for r in plan['runs']), runs=plan['runs']))


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('action', choices=('prepare','probe','logger','integration','bind','run','audit'))
  for name in ('plan','evidence','output','resources','references'):
    p.add_argument('--'+name, type=Path)
  p.add_argument('--campaign'); p.add_argument('--gpu', type=int); p.add_argument('--uuid')
  a = p.parse_args()
  if a.action=='prepare': prepare(a.output, a.resources, a.references, a.campaign)
  elif a.action=='probe': probe(a.gpu, a.uuid, a.output, host=HOST, allowed_gpus=tuple(CARDS.values()))
  elif a.action=='logger': logger_check(a.output)
  elif a.action=='integration': integration(a.plan, a.evidence)
  elif a.action=='bind': bind(a.plan, a.evidence, a.output)
  elif a.action=='audit': finish(read(a.plan), a.output)
  else: run(a.plan)
