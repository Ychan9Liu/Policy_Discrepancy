"""Run an explicitly authorized single-slot Dt migration on sv2."""
import argparse
import copy
from dataclasses import dataclass
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

@dataclass(frozen=True)
class Migration:
  task: str
  gpu: int
  source_state: str
  slug: str

  @property
  def root(self):
    return Path(f'/data/Policy_Discrepancy/runs/formal-m2-vision20-{self.slug}-dt-20261105-sv2')

  @property
  def engineering(self):
    return Path(f'/data/Policy_Discrepancy/runs/engineering-vision20-{self.slug}-dt-20261105-sv2')

  @property
  def suffix(self):
    return f'moved-sv2-{self.slug}-20261105'


WALKER = Migration('dmc_walker_run', 4, 'queued', 'walker')
HOPPER_STAND = Migration('dmc_hopper_stand', 3, 'blocked', 'hopper-stand')
ROOT = WALKER.root
ENGINEERING = WALKER.engineering


def make_plan(source, proof, controller_sha, migration=WALKER):
  if len(source['runs'])!=1:raise ValueError('Expected one authorized source')
  old=source['runs'][0]
  if (old['task']!=migration.task or old['mode']!='dt' or old['seed']!=1 or
      old['state']!=migration.source_state or old.get('pid') or old['code_sha']!=TRAIN_SHA):raise ValueError('Source differs or has started')
  if (proof['physical_gpu']!=migration.gpu or proof['code_sha']!=TRAIN_SHA or
      not proof['verified_compute_and_graphics_same_gpu']):
    raise ValueError('Exact assigned GPU and training SHA proof required')
  row = copy.deepcopy(old)
  directory = str(migration.root / migration.task / 'dt-moved01')
  for key in ('error', 'exit_code', 'pid', 'run_audit'):
    row.pop(key, None)
  if old.get('source_run_id'):
    row['original_run_id'] = old['source_run_id']
  argv=list(row['review_argv']);argv[argv.index('--logdir')+1]=directory
  c=resolve(argv[4:]);check_config(c);config=json.loads(json.dumps(dict(c.flat)))
  if config!=dict(old['config'],logdir=directory):raise ValueError('Only logdir may change')
  row.update(server='sv2',host='lyg0360',physical_gpu=migration.gpu,gpu_uuid=proof['gpu_uuid'],egl_index=proof['egl_index'],
      directory=directory,review_argv=argv,launch_argv=argv,config=config,state='queued',queue_position=1,
      source_run_id=old['run_id'],source_directory=old['directory'],run_id=old['run_id']+':'+migration.suffix,baseline_c_source=None)
  row['environment_template'].update(CUDA_VISIBLE_DEVICES=proof['gpu_uuid'],MUJOCO_EGL_DEVICE_ID=str(proof['egl_index']),TMPDIR=directory+'/tmp',WANDB_DIR=directory,WANDB_RUN_ID=old['environment_template']['WANDB_RUN_ID']+'-'+migration.suffix)
  return dict(root=str(migration.root),code_sha=controller_sha,training_sha=TRAIN_SHA,authorized_to_train=True,protocol=VISION20_SEED1,runs=[row])


def validate_seal(source, seal, migration=WALKER):
  row=source['runs'][0]
  if (not seal['passed'] or not seal['source_directory_reserved_before_spawn'] or
      seal['source_sha256']!=sha(migration.engineering/'source.json') or seal['replacement_root']!=str(migration.root) or
      seal['source_run_id']!=row['run_id'] or seal['source_reserved_directory']!=row['directory'] or
      seal['reservation_contents']!=['MOVED_TO_SV2.json']):raise ValueError('Missing exclusive source reservation/claim')


def run(plan_path, migration=WALKER):
  root = migration.root
  engineering = migration.engineering
  p=read(plan_path);source=read(engineering/'source.json');proof=read(engineering/f'gpu{migration.gpu}-binding.json')
  validate_seal(source,read(engineering/'source-claims.json'),migration)
  if p!=make_plan(source,proof,clean_sha(),migration) or platform.node()!='lyg0360':raise ValueError('Exact authorized plan/host/SHA differs')
  if subprocess.check_output(['git','-C',str(TRAIN_REPO),'rev-parse','HEAD'],text=True).strip()!=TRAIN_SHA or subprocess.check_output(['git','-C',str(TRAIN_REPO),'status','--porcelain']):raise RuntimeError('Training checkout changed')
  a=read(engineering/'integration/acceptance.json')
  if not a['passed'] or a['code_sha']!=TRAIN_SHA or [(r['task'],r['mode'],r['physical_gpu']) for r in a['runs']]!=[(migration.task,'dt',migration.gpu)]:raise RuntimeError('Actual assigned GPU CUDA acceptance required')
  if root.exists():raise FileExistsError('No overwrite/resume')
  check=require_free(migration.gpu,proof['gpu_uuid'])
  if shutil.disk_usage(root.parent).free<60_000_000_000:raise RuntimeError('Below 60GB floor')
  root.mkdir();dump(root/'bound-plan.json',p);dump(root/'prelaunch.json',dict(check=check,utc=now(),free_bytes=shutil.disk_usage(root).free,supervisor_pid=os.getpid()))
  row=p['runs'][0];d=Path(row['directory']);d.mkdir(parents=True,exist_ok=False);(d/'tmp').mkdir()
  with (d/'stdout-initial.log').open('x') as log:
    proc=subprocess.Popen(row['launch_argv'],cwd=TRAIN_REPO,env=child_environment(row['environment_template'],inherited=approved_context()),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    row.update(state='running',pid=proc.pid,start_utc=now());dump(d/'execution-initial.json',row);dump(root/'actual-RUN_INDEX.json',dict(runs=[row]));rc=proc.wait()
  row.update(state='failed',exit_code=rc,end_utc=now())
  if rc==0:
    try:row['run_audit']=audit_run(row,VISION20_SEED1);row['state']='run_audited_pending_group_and_WB'
    except Exception as e:row.update(state='audit_failed',error=repr(e))
  dump(d/'execution-final.json',row);dump(root/'actual-RUN_INDEX.json',dict(runs=[row]));dump(root/'supervisor-final.json',dict(cohort_complete=False,runs=[row],utc=now()))


if __name__=='__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--plan', required=True)
  parser.add_argument('--hopper-stand', action='store_true')
  args = parser.parse_args()
  run(args.plan, HOPPER_STAND if args.hopper_stand else WALKER)
