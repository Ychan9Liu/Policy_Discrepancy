"""Run isolated real-DMC engineering checks; never starts a formal budget.

This external supervisor records resources without changing runner/agent code.
It is intended to execute from a clean, pinned server checkout.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
TASKS = ('dmc_hopper_hop', 'dmc_quadruped_run',
         'dmc_quadruped_walk', 'dmc_reacher_hard')


def write_json(path, value):
  Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def read_json(path):
  return json.loads(Path(path).read_text())


def rows(path):
  return [json.loads(x) for x in Path(path).read_text().splitlines() if x]


def sha(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def disk_bytes(path):
  return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def base_command(task, directory, mode, c=-1, episodes=2, report=0, stop=0):
  return [sys.executable, '-u', '-m', 'dreamerv3.main',
      '--configs', 'm2_v1', '--task', task, '--logdir', str(directory),
      '--run.engineering_fixture', 'True',
      '--run.action_budget', '4098', '--run.eval_every_actions', '2049',
      '--run.match_start', '2200', '--run.match_end', '3600',
      '--run.eval_eps', str(episodes),
      '--run.report_every_actions', str(report),
      '--run.engineering_stop_after_actions', str(stop),
      '--agent.rep_probe.mode', mode, '--agent.rep_probe.c', str(c),
      '--jax.prealloc', 'False', '--logger.outputs', 'jsonl']


def supervise(command, directory, label, gpu):
  directory.mkdir(parents=True, exist_ok=True)
  execution = directory / ('execution-' + label + '.json')
  log_path = directory / ('stdout-' + label + '.log')
  samples = directory / ('resources-' + label + '.jsonl')
  env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
      MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', PYTHONUNBUFFERED='1',
      OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
      TMPDIR=str(directory / 'tmp'))
  (directory / 'tmp').mkdir(exist_ok=True)
  start = time.time()
  info = dict(command=command, cwd=str(REPO), host=platform.node(),
      git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
          cwd=REPO, text=True).strip(), gpu_index=gpu,
      environment={k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'MUJOCO_GL',
          'PYOPENGL_PLATFORM', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
          'MKL_NUM_THREADS', 'TMPDIR')},
      start_utc=datetime.now(timezone.utc).isoformat(), start_epoch=start,
      disk_before_bytes=disk_bytes(directory))
  process = subprocess.Popen(command, cwd=REPO, env=env,
      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
      bufsize=1)
  info['pid'] = process.pid
  write_json(execution, info)
  stop = threading.Event()

  def monitor():
    with samples.open('w') as file:
      last_disk, size, receipt_count = 0, 0, 0
      while not stop.is_set():
        now = time.time()
        gpu_line = subprocess.run(['nvidia-smi', '-i', str(gpu),
            '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True).stdout.strip()
        if now - last_disk >= 10:
          size, last_disk = disk_bytes(directory), now
        receipt_path = directory / 'update_receipts.jsonl'
        last_receipt = None
        if receipt_path.exists():
          try:
            lines = receipt_path.read_text().splitlines()
            receipt_count = len(lines)
            last_receipt = json.loads(lines[-1]) if lines else None
          except (ValueError, OSError):
            pass
        evpath = directory / 'evaluations.jsonl'
        eval_count = len(evpath.read_text().splitlines()) if evpath.exists() else 0
        parts = gpu_line.split(',')
        row = dict(epoch=now, elapsed=now-start,
            gpu_memory_mib=int(parts[0]) if parts[0].strip().isdigit() else None,
            gpu_util_pct=int(parts[1]) if len(parts)>1 and parts[1].strip().isdigit() else None,
            disk_bytes=size, receipts=receipt_count, last_receipt=last_receipt,
            completed_evaluations=eval_count)
        file.write(json.dumps(row) + '\n')
        file.flush()
        stop.wait(1)

  thread = threading.Thread(target=monitor, daemon=True)
  thread.start()
  with log_path.open('w') as output:
    for line in process.stdout:
      output.write(f'{time.time():.6f} {line}')
      output.flush()
  code = process.wait()
  stop.set()
  thread.join()
  info.update(exit_code=code, end_epoch=time.time(),
      end_utc=datetime.now(timezone.utc).isoformat(),
      disk_after_bytes=disk_bytes(directory))
  info['wall_seconds'] = info['end_epoch'] - start
  resource_rows = rows(samples)
  info['sampled_gpu_peak_mib'] = max(
      (r['gpu_memory_mib'] or 0 for r in resource_rows), default=0)
  write_json(execution, info)
  print(json.dumps(dict(run=str(directory), label=label, exit_code=code,
      wall_seconds=info['wall_seconds'],
      sampled_gpu_peak_mib=info['sampled_gpu_peak_mib'])), flush=True)
  return code


def audit_run(directory, mode):
  manifest = read_json(directory / 'protocol_manifest.json')
  config = manifest['config']
  from embodied.run.protocol_v1 import ProtocolState
  state = ProtocolState(4098, 2049, 2, 2200, 3600)
  receipts = rows(directory / 'update_receipts.jsonl')
  for row in receipts:
    state.record_update(row['update_id'], row['start_action'],
        row['match_sum'], row['match_count'], row['invalid_count'])
  evaluations = rows(directory / 'evaluations.jsonl')
  final = read_json(directory / 'final_state.json')
  assert final['train_action_steps'] == 4098
  assert final['updates'] == len(receipts) and final['updates'] > 0
  assert [r['action_step'] for r in evaluations] == [0, 2049, 4098]
  assert all(len(r['scores']) == len(r['seeds']) == len(r['lengths']) == 2
             for r in evaluations)
  assert all(r['update_id'] <= final['updates'] for r in evaluations)
  assert evaluations[0]['update_id'] == 0
  assert final['eval_action_steps'] == sum(sum(r['lengths']) for r in evaluations)
  assert final['train_resets'] >= 16
  for point in (0, 2049, 4098):
    assert (directory / 'eval_snapshots' / f'{point:07d}' / 'latest').exists()
  assert config['seed'] == 0 and config['run.engineering_fixture']
  assert config['run.envs'] == 16 and config['run.train_ratio'] == 256
  assert config['batch_size'] == 16 and config['batch_length'] == 64
  assert config['agent.dyn.rssm.deter'] == 4096
  assert config['agent.dyn.rssm.hidden'] == 512
  assert config['agent.dyn.rssm.classes'] == 32
  assert config['agent.enc.simple.depth'] == config['agent.dec.simple.depth'] == 32
  assert config['agent.policy.units'] == config['agent.value.units'] == 512
  assert not config['agent.ac_grads'] and config['agent.reward_grad']
  assert config['agent.repval_grad'] and config['agent.repval_loss']
  assert config['agent.dyn.rssm.free_nats'] == 1
  assert config['agent.loss_scales.dyn'] == 1 and config['agent.loss_scales.rep'] == .1
  assert config['env.dmc.image'] and not config['env.dmc.proprio']
  assert config['env.dmc.repeat'] == 1 and config['env.dmc.use_seed']
  assert config['agent.rep_probe.alpha'] == 20
  assert manifest['git_commit'] == final['git_commit']
  assert all(r['invalid_count'] == 0 for r in receipts)
  metrics = rows(directory / 'metrics.jsonl')
  diagnostic = [{k: v for k, v in r.items() if k.startswith('train/dt/')
      or k.startswith('train/loss/') or k in ('train/opt/loss', 'train/opt/grad_norm')}
      for r in metrics if 'train/dt/rep_active_frac' in r]
  assert diagnostic, 'No real training diagnostics'
  import math
  nonfinite = [(i, k, v) for i, r in enumerate(diagnostic) for k, v in r.items()
      if isinstance(v, (int, float)) and not math.isfinite(v)]
  # Only explicitly unavailable active-set diagnostics are allowed to be NaN.
  unavailable = [x for x in nonfinite if 'active_weight_mean' in x[1]]
  assert len(nonfinite) == len(unavailable), nonfinite
  result = dict(task=manifest['task'], mode=mode, final=final,
      first_update_action=receipts[0]['start_action'],
      unique_update_ids=len(state.records), receipt_rows=len(receipts),
      engineering_c=state.frozen_c() if mode == 'logging' else None,
      diagnostic=diagnostic, unavailable_active_diagnostics=unavailable,
      passed=True)
  write_json(directory / 'audit.json', result)
  return result


def run_task(args):
  root = Path(args.root).resolve()
  root.mkdir(parents=True, exist_ok=True)
  results = []
  frozen = None
  for mode in ('logging', 'dt', 'constant', 'shuffle'):
    directory = root / mode
    if directory.exists():
      raise RuntimeError(f'Refusing to overwrite {directory}')
    c = frozen['c'] if mode == 'constant' else -1
    command = base_command(args.task, directory, mode, c)
    code = supervise(command, directory, 'initial', args.gpu)
    if code:
      write_json(root / 'task_status.json', dict(task=args.task,
          failed_mode=mode, exit_code=code, completed=results))
      return code
    results.append(audit_run(directory, mode))
    if mode == 'logging':
      from dreamerv3.freeze_c import freeze
      frozen = freeze(directory, directory / 'c_frozen_engineering.json', True)
      assert frozen['engineering_fixture'] and frozen['raw_rep_threshold'] == 1
      # Formal freeze must reject these isolated engineering data.
      try:
        freeze(directory, directory / 'FORBIDDEN-formal-c.json', False)
      except ValueError as error:
        write_json(directory / 'formal_freeze_rejection.json', dict(error=str(error)))
      else:
        raise AssertionError('Formal freeze accepted engineering data')
      assert not (directory / 'FORBIDDEN-formal-c.json').exists()
      before = {p: sha(directory / p) for p in ('evaluations.jsonl',
          'update_receipts.jsonl', 'final_state.json', 'matching_result.json')}
      code = supervise(command, directory, 'reopen', args.gpu)
      assert code == 0
      assert before == {p: sha(directory / p) for p in before}
      assert 'Completed protocol run already checkpointed' in (
          directory / 'stdout-reopen.log').read_text()
      write_json(directory / 'reopen_audit.json', dict(passed=True, unchanged=before))
  write_json(root / 'task_status.json', dict(task=args.task, passed=True,
      engineering_only=True, completed=results))
  return 0


def supplemental(args):
  root = Path(args.root).resolve()
  # Explicit supplement runs, never overwrite an existing run.
  directory = root / args.variant
  if directory.exists():
    raise RuntimeError(f'Refusing to overwrite {directory}')
  stop = 2049 if args.variant == 'interrupted' else 0
  episodes = 2 if stop else 1
  command = base_command(args.task, directory, 'logging',
      episodes=episodes, report=0 if stop else 2049, stop=stop)
  code = supervise(command, directory, 'initial', args.gpu)
  if stop:
    assert code != 0
    assert 'Intentional fixture stop after checkpoint' in (
        directory / 'stdout-initial.log').read_text()
    before = {p: sha(directory / p) for p in ('evaluations.jsonl', 'update_receipts.jsonl')}
    code = supervise(command, directory, 'restore', args.gpu)
    assert code != 0 and 'Training continuation refused' in (
        directory / 'stdout-restore.log').read_text()
    assert before == {p: sha(directory / p) for p in before}
    assert not (directory / 'final_state.json').exists()
    write_json(directory / 'interruption_audit.json', dict(passed=True, unchanged=before))
  else:
    assert code == 0
    comparison = subprocess.run([sys.executable, str(REPO / 'tests' /
        'compare_protocol_runs.py'), str(root / 'logging'), str(directory)],
        cwd=REPO, capture_output=True, text=True)
    (directory / 'isolation_comparison.txt').write_text(
        comparison.stdout + comparison.stderr)
    write_json(directory / 'isolation_comparison.json', dict(
        exit_code=comparison.returncode, passed=comparison.returncode == 0))
    assert comparison.returncode == 0, comparison.stdout + comparison.stderr
  return 0


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--task', required=True, choices=TASKS)
  parser.add_argument('--root', required=True)
  parser.add_argument('--gpu', required=True, type=int)
  parser.add_argument('--variant', choices=('interrupted', 'isolation'))
  args = parser.parse_args()
  if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip():
    raise RuntimeError('Supervisor requires a clean checkout')
  sys.exit(supplemental(args) if args.variant else run_task(args))


if __name__ == '__main__':
  main()
