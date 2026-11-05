"""Move the only remaining sv1 queued slot to sv2 GPU4 without duplication."""
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from scripts.run_vision20_sv2_remaining import TRAIN_SHA,TRAIN_REPO
from scripts.run_hopper_seeds123 import read,dump,clean_sha,require_free,child_environment,audit_run,now,sha
from scripts.run_missing_dt import approved_context
from scripts.prepare_m2_v1 import resolve
from scripts.prepare_vision20_seed1 import check_config
from dreamerv3.protocol_scope import VISION20_SEED1

ROOT=Path('/data/Policy_Discrepancy/runs/formal-m2-vision20-walker-dt-20261105-sv2')
ENGINEERING=Path('/data/Policy_Discrepancy/runs/engineering-vision20-walker-dt-20261105-sv2')


def make_plan(source,proof,controller_sha):
  if len(source['runs'])!=1:raise ValueError('Only one remaining queued source')
  old=source['runs'][0]
  if (old['task']!='dmc_walker_run' or old['mode']!='dt' or old['seed']!=1 or
      old['state']!='queued' or old.get('pid') or old['code_sha']!=TRAIN_SHA):raise ValueError('Source differs/started')
  if proof['physical_gpu']!=4 or proof['code_sha']!=TRAIN_SHA or not proof['verified_compute_and_graphics_same_gpu']:raise ValueError('Exact GPU4 training SHA proof required')
  row=copy.deepcopy(old);directory=str(ROOT/'dmc_walker_run/dt-moved01')
  argv=list(row['review_argv']);argv[argv.index('--logdir')+1]=directory
  c=resolve(argv[4:]);check_config(c);config=json.loads(json.dumps(dict(c.flat)))
  if config!=dict(old['config'],logdir=directory):raise ValueError('Only logdir may change')
  row.update(server='sv2',host='lyg0360',physical_gpu=4,gpu_uuid=proof['gpu_uuid'],egl_index=proof['egl_index'],
      directory=directory,review_argv=argv,launch_argv=argv,config=config,state='queued',queue_position=1,
      source_run_id=old['run_id'],source_directory=old['directory'],run_id=old['run_id']+':moved-sv2-walker-20261105',baseline_c_source=None)
  row['environment_template'].update(CUDA_VISIBLE_DEVICES=proof['gpu_uuid'],MUJOCO_EGL_DEVICE_ID=str(proof['egl_index']),TMPDIR=directory+'/tmp',WANDB_DIR=directory,WANDB_RUN_ID=old['environment_template']['WANDB_RUN_ID']+'-moved-sv2-walker-20261105')
  return dict(root=str(ROOT),code_sha=controller_sha,training_sha=TRAIN_SHA,authorized_to_train=True,protocol=VISION20_SEED1,runs=[row])


def validate_seal(source,seal):
  row=source['runs'][0]
  if (not seal['passed'] or not seal['source_directory_reserved_before_spawn'] or
      seal['source_sha256']!=sha(ENGINEERING/'source.json') or seal['replacement_root']!=str(ROOT) or
      seal['source_run_id']!=row['run_id'] or seal['source_reserved_directory']!=row['directory'] or
      seal['reservation_contents']!=['MOVED_TO_SV2.json']):raise ValueError('Missing exclusive source reservation/claim')


def run(plan_path):
  p=read(plan_path);source=read(ENGINEERING/'source.json');proof=read(ENGINEERING/'gpu4-binding.json')
  validate_seal(source,read(ENGINEERING/'source-claims.json'))
  if p!=make_plan(source,proof,clean_sha()) or platform.node()!='lyg0360':raise ValueError('Exact authorized plan/host/SHA differs')
  if subprocess.check_output(['git','-C',str(TRAIN_REPO),'rev-parse','HEAD'],text=True).strip()!=TRAIN_SHA or subprocess.check_output(['git','-C',str(TRAIN_REPO),'status','--porcelain']):raise RuntimeError('Training checkout changed')
  a=read(ENGINEERING/'integration/acceptance.json')
  if not a['passed'] or a['code_sha']!=TRAIN_SHA or [(r['task'],r['mode'],r['physical_gpu']) for r in a['runs']]!=[('dmc_walker_run','dt',4)]:raise RuntimeError('Actual GPU4 CUDA acceptance required')
  if ROOT.exists():raise FileExistsError('No overwrite/resume')
  check=require_free(4,proof['gpu_uuid'])
  if shutil.disk_usage(ROOT.parent).free<60_000_000_000:raise RuntimeError('Below 60GB floor')
  ROOT.mkdir();dump(ROOT/'bound-plan.json',p);dump(ROOT/'prelaunch.json',dict(check=check,utc=now(),free_bytes=shutil.disk_usage(ROOT).free,supervisor_pid=os.getpid()))
  row=p['runs'][0];d=Path(row['directory']);d.mkdir(parents=True,exist_ok=False);(d/'tmp').mkdir()
  with (d/'stdout-initial.log').open('x') as log:
    proc=subprocess.Popen(row['launch_argv'],cwd=TRAIN_REPO,env=child_environment(row['environment_template'],inherited=approved_context()),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    row.update(state='running',pid=proc.pid,start_utc=now());dump(d/'execution-initial.json',row);dump(ROOT/'actual-RUN_INDEX.json',dict(runs=[row]));rc=proc.wait()
  row.update(state='failed',exit_code=rc,end_utc=now())
  if rc==0:
    try:row['run_audit']=audit_run(row,VISION20_SEED1);row['state']='run_audited_pending_group_and_WB'
    except Exception as e:row.update(state='audit_failed',error=repr(e))
  dump(d/'execution-final.json',row);dump(ROOT/'actual-RUN_INDEX.json',dict(runs=[row]));dump(ROOT/'supervisor-final.json',dict(cohort_complete=False,runs=[row],utc=now()))


if __name__=='__main__':
  a=argparse.ArgumentParser(description=__doc__);a.add_argument('--plan',required=True);run(a.parse_args().plan)
