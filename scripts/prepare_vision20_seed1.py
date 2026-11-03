"""CPU-only pinned plan for 38 new runs and two audited Hopper source runs."""
import argparse
import copy
import json
from pathlib import Path
import hashlib
from datetime import datetime, timezone

from dreamerv3.protocol_scope import (
    VISION20_SEED1, VISION20_SCRIPT, VISION20_TASKS, HOPPER_123, validate_scope)
from scripts.prepare_dmc16 import source_state, verify_task_inventory, PYTHON
from scripts.prepare_m2_v1 import resolve, check

CARDS = [('sv1',g) for g in range(2,8)] + [('sv2',g) for g in (0,1,2,3,4,5,7)]
HOSTS = {'sv1':'lyg2153','sv2':'lyg0360'}
NEW_TASKS = tuple(t for t in VISION20_TASKS if t != 'dmc_hopper_hop')


def check_config(c):
  if (validate_scope(c.script,c.task,c.seed,c.agent.rep_probe.mode)!=VISION20_SEED1 or
      c.script!=VISION20_SCRIPT or tuple(c.logger.outputs)!=('jsonl','wandb')):
    raise ValueError('Wrong vision20 seed1 entry/logger')
  check(c.update({'script':'protocol_v1','seed':0,'task':'dmc_hopper_hop',
                  'logger.outputs':['jsonl']}))


def validate_hopper(rows):
  if (len(rows)!=2 or {(r['task'],r['seed'],r['mode']) for r in rows} !=
      {('dmc_hopper_hop',1,m) for m in ('logging','dt')}):
    raise ValueError('Only the exact existing Hopper seed1 pair may be reused')
  for r in rows:
    if (type(r['seed']) is not int or r['state']!='accepted' or r['protocol']!=HOPPER_123 or
        r['exit_code']!=0 or r['run_audit']['updates']!=249493 or
        r['run_audit']['evaluations']!=21 or not r['code_sha']):
      raise ValueError('Hopper source not audited/accepted')
    c=resolve(r['review_argv'][4:]);check(c.update({'script':'protocol_v1','seed':0,
                                              'logger.outputs':['jsonl']}))
    if json.loads(json.dumps(dict(c.flat)))!=r['config']:
      raise ValueError('Hopper source command/config differs')


def prepare(output, resources, hopper_index, campaign):
  output=Path(output)
  if output.exists():raise FileExistsError(output)
  if not campaign.startswith('formal-m2-vision20-seed1-') or any(x in campaign for x in ('/','\\','..')):
    raise ValueError('Fresh vision20 seed1 campaign required')
  commit,dirty=source_state();inventory=verify_task_inventory()
  resource=json.loads(Path(resources).read_text(encoding='utf-8'))['servers']
  hopper=copy.deepcopy([r for r in json.loads(Path(hopper_index).read_text())['runs'] if r['seed']==1])
  validate_hopper(hopper)
  for s,h in HOSTS.items():
    if resource[s]['host']!=h:raise ValueError('Resource host differs')
  devices={s:{int(l.split(',')[0]):l.split(',')[1].strip() for l in v['gpus'].splitlines()}
           for s,v in resource.items()}
  output.mkdir(parents=True)
  rows=[]
  for i,task in enumerate(NEW_TASKS):
    server,gpu=CARDS[i%len(CARDS)]
    for offset,mode in enumerate(('logging','dt')):
      directory=f'/data/Policy_Discrepancy/runs/{campaign}-{server}/{task}/{mode}-attempt01'
      argv=[PYTHON,'-u','-m','dreamerv3.main','--configs','m2_vision20_seed1',
          '--task',task,'--seed','1','--logdir',directory,'--agent.rep_probe.mode',mode,
          '--jax.prealloc','False','--logger.outputs','jsonl','wandb']
      c=resolve(argv[4:]);check_config(c)
      flat=json.loads(json.dumps(dict(c.flat)))
      row=dict(task=task,mode=mode,group='baseline' if mode=='logging' else 'dt',
          seed=1,protocol=VISION20_SEED1,code_sha=commit,server=server,host=HOSTS[server],
          physical_gpu=gpu,gpu_uuid=devices[server][gpu],directory=directory,
          queue_position=2*(i//len(CARDS))+offset+1,
          run_id=f'{server}:{campaign}:{task}:seed1:{mode}:attempt01',
          state='prepared_not_started',authorized_to_train=False,launch_argv=None,
          review_argv=argv,config=flat,egl_index=None,
          verified_compute_and_graphics_same_gpu=False,
          baseline_c_source=directory if mode=='logging' else None,
          environment_template=dict(CUDA_VISIBLE_DEVICES=devices[server][gpu],
              CUDA_DEVICE_ORDER='PCI_BUS_ID',MUJOCO_EGL_DEVICE_ID=None,
              MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',JAX_PLATFORMS='cuda',
              XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',
              OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',
              TMPDIR=directory+'/tmp',WANDB_DIR=directory,WANDB_ENTITY='ychan9liu-tsinghua-university',
              WANDB_PROJECT='PD_1',WANDB_MODE='online',WANDB_RESUME='never',
              WANDB_RUN_GROUP=campaign,WANDB_RUN_ID=f'{campaign}-{task[4:]}-seed1-{mode}-01'))
      rows.append(row)
  for task in NEW_TASKS:
    a,b=[r['config'] for r in rows if r['task']==task]
    if {k for k in a if a[k]!=b[k]}!={'logdir','agent.rep_probe.mode'}:
      raise ValueError('Undeclared group difference')
  plan=dict(protocol=VISION20_SEED1,code_sha=commit,campaign=campaign,runs=rows,
      reused_hopper_runs=hopper,new_run_count=38,source_run_count=2,
      final_record_count=40,authorized_to_train=False,auto_start_when_idle=False,
      local_drafts=dirty,resource_snapshot=resource,task_inventory=inventory,
      created_utc=datetime.now(timezone.utc).isoformat(),
      hopper_index_sha256=hashlib.sha256(Path(hopper_index).read_bytes()).hexdigest(),
      pending=['storage_solution','fresh_CUDA_EGL_probes','committed_supervisor_and_tests',
               'real_CUDA_agent_acceptance','online_logger','fresh_resource_capacity_check'])
  (output/'planned-run-index.json').write_text(json.dumps(plan,indent=2)+'\n')
  for s in HOSTS:
    (output/(s+'-queues.json')).write_text(json.dumps({str(g):[r for r in rows
        if r['server']==s and r['physical_gpu']==g] for host,g in CARDS if host==s},indent=2)+'\n')
  print(json.dumps(dict(code_sha=commit,new_runs=len(rows),reused_runs=len(hopper),training_started=False)))


if __name__=='__main__':
  p=argparse.ArgumentParser(description=__doc__)
  for name in ('output','resources','hopper-index','campaign'):p.add_argument('--'+name,required=True)
  a=p.parse_args();prepare(a.output,a.resources,a.hopper_index,a.campaign)
