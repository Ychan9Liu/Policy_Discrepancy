"""CPU-only queue for the assigned acrobot job after cup's completed audit.

The training checkout stays pinned to its independently GPU-tested commit.
An exited agent may still have a live supervisor auditing its outputs.
"""
import argparse
import os
from pathlib import Path
import platform
import subprocess
import time

from scripts.run_missing_dt import validate_plan, approved_context, PYTHON
from scripts.run_hopper_seeds123 import read, dump, clean_sha, now, sha

RUNTIME_SHA = 'c7689d50f4caed0fef89912c65280b8d2558b126'
RUNTIME = Path('/data/Policy_Discrepancy/checkouts/missing-dt-c7689d50f4ca')


def predecessor_stage(plan):
  wait = plan['wait_for']; directory = Path(wait['directory'])
  final = directory/'execution-final.json'
  if final.exists():
    row = read(final)
    if (row['state'] not in ('accepted','run_audited_pending_group_and_WB') or
        row['task'] != wait['task'] or row['code_sha'] != wait['code_sha'] or
        row['directory'] != wait['directory'] or row['run_audit']['updates'] != 249493 or
        row['run_audit']['evaluations'] != 21):
      raise RuntimeError('Predecessor failed or audit incomplete')
    return 'ready'
  root = directory.parent.parent
  row = next(r for r in read(root/'actual-RUN_INDEX.json')['runs'] if r['task']==wait['task'])
  if (row['state'] not in ('running','training_finished_pending_audit') or
      row['directory'] != wait['directory'] or row['code_sha'] != wait['code_sha']):
    raise RuntimeError('Incompatible predecessor; no idle-GPU retry')
  # Its agent can already have exited while the supervisor audits 249493 receipts.
  supervisor = read(root/'prelaunch.json')['supervisor_pid']
  os.kill(supervisor, 0)
  return 'waiting_for_own_predecessor_or_its_audit'


def run(plan_path):
  plan_path = Path(plan_path); plan = read(plan_path); validate_plan(plan)
  if (plan['server'] != 'sv2' or plan.get('shard') != 'remaining-three' or
      plan['code_sha'] != RUNTIME_SHA or not plan['authorized_to_train'] or
      platform.node() != plan['host'] or Path(plan['root']).exists()):
    raise RuntimeError('Only the unstarted authorized acrobot queue on sv2 GPU1')
  if (subprocess.check_output(['git','rev-parse','HEAD'],cwd=RUNTIME,text=True).strip()!=RUNTIME_SHA or
      subprocess.check_output(['git','status','--porcelain'],cwd=RUNTIME)):
    raise RuntimeError('Runtime checkout changed')
  evidence = plan_path.parent
  identity = dict(queue_code_sha=clean_sha(),runtime_code_sha=RUNTIME_SHA,
      plan_sha256=sha(plan_path),pid=os.getpid(),training_started=False)
  while True:
    stage = predecessor_stage(plan)
    dump(evidence/'queue-progress.json',dict(identity,stage=stage,checked_utc=now()))
    if stage == 'ready':break
    time.sleep(30)
  # The pinned runner re-audits the predecessor and refreshes the free-GPU gate.
  command = [PYTHON,'-u','-m','scripts.run_missing_dt','run','--plan',str(plan_path)]
  dump(evidence/'queue-released.json',dict(identity,argv=command,checked_utc=now()))
  env = approved_context();env.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',
      OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
  result = subprocess.run(command,cwd=RUNTIME,env=env,stdin=subprocess.DEVNULL)
  dump(evidence/'queue-final.json',dict(identity,exit_code=result.returncode,checked_utc=now()))
  if result.returncode:raise RuntimeError('Queued runtime failed; preserve evidence, no retry')


if __name__ == '__main__':
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--plan',type=Path,required=True)
  run(parser.parse_args().plan)
