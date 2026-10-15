"""DL normal R5 source index: complete authority, then pretreatment prefixes.

No model, environment, physics replay or scientific verdict runs here. The old
full physics audit and both external producer authority pins are mandatory.
Published rows refer to RGB/raw bytes; no old latent/carry is a feature input.
"""
import time
ENTRY_STARTED = time.monotonic()
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import signal
import socket
import sys
from types import SimpleNamespace

SECONDS, DEADLINE, AUDITOR = 900, float('inf'), None


def expire(signum,frame):
  if AUDITOR is not None:AUDITOR.expire(signum,frame)
  os._exit(124)


if __name__ == '__main__':
  DEADLINE=ENTRY_STARTED+SECONDS
  os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',MUJOCO_GL='disable')
  for key in ('MUJOCO_EGL_DEVICE_ID','PYOPENGL_PLATFORM'):os.environ.pop(key,None)
  if os.name == 'posix':
    signal.signal(signal.SIGALRM,expire);signal.signal(signal.SIGTERM,expire)
    signal.setitimer(signal.ITIMER_REAL,max(.001,DEADLINE-time.monotonic()))

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.dl_reset_learning import tree_digest

VERSION = 'DL-normal-source-index-r1'
SNAPSHOT_HASH_FORMAT = 'DL-r5-snapshot-json-array-task-pickle5-sha256-r1'
RUN_GIT = '1584ef31c322fc3371ea9ce220c06fe7e654b23a'
AUDIT_LAUNCHER_SHA = '72d98c21161c4372b21981b20a59a1d040914e30ee40892cae7c128a49ec565f'
DEPENDENCIES = ('analysis/dl_live_audit.py','analysis/dl_feedback_physics_audit_v1.py',
    'analysis/dl_r4_audit.py','analysis/dl_live_statistics.py','analysis/dl_r4_statistics.py')
OBS = dict(image=(np.uint8,(64,64,3)),reward=(np.float32,()),
    is_first=(np.bool_,()),is_last=(np.bool_,()),is_terminal=(np.bool_,()))


def require(value, message):
  if not value:raise ValueError(message)


def check():
  require(time.monotonic() < DEADLINE,'Whole source-index900sec deadline exceeded')


def digest(body):return hashlib.sha256(body).hexdigest()


def canonical(value):
  return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def decode(body):
  def pairs(items):
    result={}
    for key,value in items:
      require(key not in result,'Duplicate JSON key');result[key]=value
    return result
  return json.loads(body,object_pairs_hook=pairs,parse_constant=lambda value:
      (_ for _ in ()).throw(ValueError('Nonfinite JSON number')))


def raw_pin(path, expected=None):
  path=Path(path).absolute()
  require(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)), 'Ordinary pinned file required')
  before=path.stat();h=hashlib.sha256()
  with path.open('rb') as stream:
    for part in iter(lambda:stream.read(1024*1024),b''):check();h.update(part)
  after=path.stat()
  require((before.st_size,before.st_mtime_ns,before.st_ino) ==
      (after.st_size,after.st_mtime_ns,after.st_ino),'File changed during raw authentication')
  result=dict(path=str(path.resolve()),bytes=after.st_size,sha256=h.hexdigest())
  if expected is not None:
    require(type(expected) is str and re.fullmatch('[0-9a-f]{64}',expected) and
        result['sha256'] == expected,'External raw pin differs')
  return result


def bound_file(item):
  require(type(item) is dict and set(item) == {'path','bytes','sha256'}, 'Exact bound raw file pin required')
  require(Path(item['path']).is_absolute() and raw_pin(item['path'],item['sha256']) == item,
      'Bound raw file bytes/path differ')


def fresh_output(path,run):
  requested=Path(path).absolute();root=Path(run).resolve(strict=True)
  require(not any(p.is_symlink() for p in (requested,*requested.parents)),
      'Source index output and parent paths must not be symlinks')
  parent=requested.parent.resolve(strict=True)
  output=(parent/requested.name).resolve(strict=False)
  require(parent.is_dir() and output.parent == parent and not output.exists() and
      not output.is_relative_to(root),'Fresh resolved external source index output required')
  return output


def plan(unit):
  require(type(unit) is int and 0 <= unit < 512,'Fixed source unit0..511 required')
  index=int(np.random.default_rng([20261030,0x54494D45,unit]).integers(0,14))
  return dict(unit=unit,family=unit//16,replicate=unit%16,env_seed=8192+unit//16,
      eval_seed=0x52350000+unit,t_index=index,t=64+64*index)


def root_key(unit,step):
  return np.random.default_rng([0,0x4556414C,plan(unit)['eval_seed'],step]).integers(
      0,np.iinfo(np.uint32).max,(2,),np.uint32)


def array(value,dtype,shape,label):
  require(type(value) is np.ndarray and value.dtype == dtype and value.shape == shape and
      np.isfinite(value).all(),'Exact finite source array differs: '+label)
  return value


def snapshot_sha256(value):
  """R5 snapshot wire hash of an already trusted snapshot, not a parser.

  Task dictionaries contain official RandomState objects, not tensor leaves.
  Serialize only this trusted task dictionary with protocol5, preserving its
  complete RNG state. This matches the original R5 snapshot_hash protocol.
  """
  require(type(value) is dict and set(value) == {'suite','from_dm_done'},'Full wrapper snapshot required')
  state=value['suite']
  require(type(state) is dict and set(state) == {'integration','step_count','reset_next_step','task'},
      'Full suite/task snapshot required')
  integration=state['integration']
  require(type(integration) is np.ndarray and integration.dtype == np.float64 and integration.ndim == 1 and
      len(integration) and np.isfinite(integration).all(),'Finite full F64 integration snapshot required')
  require(type(state['step_count']) is int and state['step_count'] >= 0 and type(state['task']) is dict and
      isinstance(state['reset_next_step'],(bool,np.bool_)) and isinstance(value['from_dm_done'],(bool,np.bool_)),
      'Exact task/clock/reset/wrapper snapshot types required')
  integration_sha=digest(str(integration.dtype).encode()+str(integration.shape).encode()+integration.tobytes())
  meta=[integration_sha,state['step_count'],bool(state['reset_next_step']),bool(value['from_dm_done']),
      digest(pickle.dumps(state['task'],protocol=5))]
  return digest(json.dumps(meta,separators=(',',':')).encode())


def project_prefix(record,unit,trajectory_pin,inventory_sha256):
  """Pure projection of an already trusted record; does NOT authenticate it.

  Excludes raw[t] and every postintervention field. Current t is an ordinary
  physical frame whose outgoing SOURCE raw is unavailable, not an episode end.
  """
  p=plan(unit);t=p['t']
  require(record['version'] == 'DL-r5-live-trajectory-r1' and record['unit'] == unit and record['plan'] == p,
      'Source unit/plan identity or order differs')
  n=record['executed_steps']
  require(type(n) is int and t < n <= t+100 and len(record['observations']) == len(record['snapshots']) == n+1,
      'Complete trajectory/current-prefix bounds differ')
  require(type(trajectory_pin) is dict and set(trajectory_pin) == {'path','bytes','sha256'} and
      trajectory_pin['path'] == f'family_{unit//16:02d}/unit_{unit%16:02d}/trajectory.pkl' and
      type(trajectory_pin['bytes']) is int and trajectory_pin['bytes'] > 0 and
      re.fullmatch('[0-9a-f]{64}',trajectory_pin['sha256']) and
      re.fullmatch('[0-9a-f]{64}',inventory_sha256),'Bound trajectory/source inventory pin required')
  raw=array(record['raw'],np.float32,(n,12),'raw')[:t]
  applied=array(record['applied'],np.float64,(n,12),'applied')[:t]
  keys=array(record['keys'],np.uint32,(n,2),'keys')[:t+1]
  require(keys.tobytes() == np.stack([root_key(unit,i) for i in range(t+1)]).tobytes(),
      'Actual continuous source EVAL root/counter differs')
  rewards={name:array(record[name],dtype,(n,),name)[:t] for name,dtype in
      (('reward_f64',np.float64),('reward_f32',np.float32),('discount',np.float64),
       ('is_first',np.bool_),('is_last',np.bool_),('is_terminal',np.bool_))}
  observed={name:[] for name in OBS};snapshots=[];frames=[]
  for step in range(t+1):
    check();obs=record['observations'][step]
    require(type(obs) is dict and set(obs) == set(OBS),'Exact prefix observation fields required')
    for name,(dtype,shape) in OBS.items():
      value=np.asarray(obs[name]);array(value,dtype,shape,'observation '+name);observed[name].append(value)
    require(bool(obs['is_first']) == (step == 0) and not obs['is_last'] and not obs['is_terminal'],
        'Pretreatment prefix crossed a physical reset/last/terminal')
    if step == 0:
      require(obs['reward'] == 0,'Actual reset reward encoding must be zero')
    else:
      require(np.asarray(rewards['reward_f64'][step-1],np.float32).tobytes() ==
          np.asarray(obs['reward']).tobytes() == np.asarray(rewards['reward_f32'][step-1]).tobytes() and
          all(bool(rewards[name][step-1]) == bool(obs[name]) for name in ('is_first','is_last','is_terminal')),
          'Source transition i must align with observation i+1')
    saved=record['snapshots'][step]
    require(type(saved) is dict and set(saved) == {'suite','from_dm_done'} and not saved['from_dm_done'],
        'Complete ordinary prefix wrapper snapshot required')
    suite=saved['suite']
    require(type(suite) is dict and set(suite) == {'integration','step_count','reset_next_step','task'} and
        type(suite['step_count']) is int and suite['step_count'] == step and not suite['reset_next_step'],
        'Source snapshot absolute clock/reset alignment differs')
    integration=suite['integration']
    require(type(integration) is np.ndarray and integration.dtype == np.float64 and integration.ndim == 1 and
        len(integration) and np.isfinite(integration).all(),'Complete finite F64 integration required')
    snapshot_hash=snapshot_sha256(saved);snapshots.append(snapshot_hash)
    frames.append(dict(counter=step,observation_index=step,snapshot_index=step,
        observation_sha256=tree_digest(obs),snapshot_sha256=snapshot_hash,snapshot_hash_format=SNAPSHOT_HASH_FORMAT,
        root_key=keys[step].tolist(),previous_raw_index=step-1 if step else None,
        previous_raw_sha256=tree_digest(raw[step-1] if step else np.zeros(12,np.float32)),
        source_raw_index=step if step < t else None,source_raw_available=step < t,
        source_action_in_prefix=step < t))
  observed={name:np.stack(values) for name,values in observed.items()}
  prefix_hash=tree_digest(dict(observations=observed,raw=raw,applied=applied,keys=keys,
      transition_fields=rewards,snapshot_hashes=snapshots))
  identity=dict(run_inventory_sha256=inventory_sha256,trajectory=trajectory_pin,plan=p)
  return dict(source_id=digest(canonical(identity)),global_unit=unit,family=unit//16,replicate=unit%16,
      split='fit' if unit//16 < 8 else 'held',plan=p,trajectory=dict(trajectory_pin),
      bounds=dict(observations=[0,t+1],snapshots=[0,t+1],raw=[0,t],applied=[0,t],
          transition_fields=[0,t],policy_keys=[0,t+1]),frames=frames,prefix_typed_sha256=prefix_hash,
      current=dict(observation_index=t,snapshot_index=t,preceding_reward_index=t-1,
          preceding_reward_f64_sha256=tree_digest(np.asarray(rewards['reward_f64'][t-1])),
          observation_sha256=frames[-1]['observation_sha256'],snapshot_sha256=snapshots[-1],
          snapshot_hash_format=SNAPSHOT_HASH_FORMAT,
          source_raw_available=False,source_raw_index=None,source_action_in_prefix=False,
          physical_is_last=False,physical_is_terminal=False),
      projection_authenticates_source=False,old_latent_used_as_feature=False)


def validate_order(rows,inventory_sha256):
  require(type(rows) is list and len(rows) == 512,'Exactly complete32x16 source rows required')
  ids=set()
  for i,row in enumerate(rows):
    p=plan(i)
    require(row['global_unit'] == i and row['family'] == i//16 and row['replicate'] == i%16 and
        row['plan'] == p and row['split'] == ('fit' if i//16 < 8 else 'held'),
        'Source identity/order/fit-held alignment differs despite matching counts')
    require(row['trajectory']['path'] == f'family_{i//16:02d}/unit_{i%16:02d}/trajectory.pkl',
        'Source trajectory identity differs despite matching counts')
    expected=digest(canonical(dict(run_inventory_sha256=inventory_sha256,trajectory=row['trajectory'],plan=p)))
    require(row['source_id'] == expected and expected not in ids,'Source ID is mismatched or duplicated')
    ids.add(expected)


def _load_auditor():
  global AUDITOR
  require(os.name == 'posix' and socket.gethostname().split('.')[0] == 'lyg0326',
      'Source process-retirement checks require actual sv3 Linux host')
  require('jax' not in sys.modules,'Source indexing must not run inside a model process')
  from analysis import dl_live_audit as a
  AUDITOR=a
  return a


def _verify_dependencies(a,prior_git):
  raw=a.command_output(['git','cat-file','--batch'],
      ''.join(prior_git+':'+name+'\n' for name in DEPENDENCIES).encode())
  offset=0;result={}
  for name in DEPENDENCIES:
    end=raw.index(b'\n',offset);header=raw[offset:end].split();offset=end+1
    require(len(header) == 3 and header[1] == b'blob','Original audit dependency blob required')
    size=int(header[2]);body=raw[offset:offset+size];offset+=size+1
    require(raw[offset-1:offset] == b'\n' and len(body) == size and (ROOT/name).read_bytes() == body,
        'Metadata audit/physics/primitive/statistics differs from successful prior audit: '+name)
    result[name]=dict(bytes=size,sha256=digest(body))
  require(offset == len(raw),'Unexpected prior dependency blob tail')
  modules=(a,a.physics,a.primitive,a.statistics,a.primitive.statistics)
  require(all(Path(module.__file__).resolve() == ROOT/name for module,name in zip(modules,DEPENDENCIES)),
      'Metadata audit imported a dependency from another checkout')
  return result


def _physics_authority(args,a,pins):
  report,terminal,launch=(decode(Path(pins[name]['path']).read_bytes()) for name in
      ('audit_report','audit_terminal','audit_launch'))
  run=str(Path(args.run).resolve())
  require(all(x['version'] == 'DL-r5-audit-launcher-r1' and x['run_directory'] == run and
      x['run_git'] == RUN_GIT and x['audit_git'] == args.prior_physics_audit_git and x['seconds'] == 1800
      for x in (terminal,launch)), 'Prior full physics audit launcher identity differs')
  require(type(terminal['returncode']) is int and terminal['returncode'] == 0 and
      terminal['external_failure'] is None and terminal['parent_identity_gone'] is True and
      terminal['owned_groups_cleaned'] is True and type(terminal['elapsed_seconds']) in (int,float) and
      0 < terminal['elapsed_seconds'] <= 1800 and terminal['audit_output'] == pins['audit_report'] and
      terminal['parent_identity'] == launch['parent_identity'],'True complete prior physics-audit exit required')
  require(terminal['launcher'] == launch['launcher'] and terminal['launcher']['sha256'] == AUDIT_LAUNCHER_SHA,
      'Actual prior audit controller differs')
  for field in ('launcher','log'):bound_file(terminal[field])
  for field,source in (('producer_launch','launcher_launch'),('producer_terminal','launcher_terminal')):
    require(terminal[field] == launch[field] == pins[source],'Prior audit did not use these producer authority pins')
  identity=terminal['parent_identity']
  require(all(type(identity[k]) is int and identity[k] > 0 for k in ('pid','pgid','sid','start_ticks')) and
      identity['pid'] == identity['pgid'] == identity['sid'],'Actual independent audit session identity required')
  a.original_inactive(identity['pid'],identity['start_ticks']);a.original_group_inactive(identity)
  command=[ '/data/Policy_Discrepancy/envs/dreamer/bin/python',
      str(Path(launch['cwd'])/'analysis/dl_live_audit.py'),'--run',run,'--expected-run-git',RUN_GIT,
      '--expected-audit-git',args.prior_physics_audit_git,'--output',pins['audit_report']['path'],
      '--launcher-terminal',pins['launcher_terminal']['path'],'--launcher-sha',pins['launcher_terminal']['sha256'],
      '--launcher-launch',pins['launcher_launch']['path'],'--launcher-launch-sha',pins['launcher_launch']['sha256']]
  require(Path(launch['cwd']).is_absolute() and launch['command'] == command and
      launch['environment'] == dict(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',MUJOCO_GL='disable',
          MUJOCO_EGL_DEVICE_ID=None,PYOPENGL_PLATFORM=None),'Actual prior audit argv/CPU environment differs')
  require(report['version'] == 'DL-r5-independent-audit-r1' and report['run_directory'] == run and
      report['run_git'] == RUN_GIT and report['audit_git'] == args.prior_physics_audit_git and
      report['complete_units'] == 512 and report['complete_families'] == 32 and report['seconds'] == 1800 and
      0 < report['elapsed_seconds'] <= 1800 and report['evidence_authenticated'] is True and
      report['all_physics_and_rewards_byte_exact'] is True,'Complete prior32x16 physics audit required')
  return report


def authenticate_and_index(args):
  """Production path; all five authority files have explicit caller raw pins.

  The four original producer CLI arguments are mandatory here as well as in
  main; the direct metadata fixture bypass in the old auditor is never used.
  """
  check()
  require(np.__version__ == '1.26.4','Frozen NumPy1.26.4 source plan required')
  for field in ('expected_git','prior_physics_audit_git'):
    value=getattr(args,field,None)
    require(type(value) is str and re.fullmatch('[0-9a-f]{40}',value), 'Full index and successful prior audit Git required')
  pin_fields=(('launcher_terminal','launcher_sha'),('launcher_launch','launcher_launch_sha'),
      ('audit_report','audit_report_sha'),('audit_terminal','audit_terminal_sha'),('audit_launch','audit_launch_sha'))
  require(all(type(getattr(args,sha_name,None)) is str and re.fullmatch('[0-9a-f]{64}',getattr(args,sha_name))
      for _,sha_name in pin_fields),'Every external authority path needs its explicit raw SHA; four producer pins cannot be omitted')
  pins={name:raw_pin(getattr(args,name),getattr(args,sha_name)) for name,sha_name in pin_fields}
  run=Path(args.run).resolve(strict=True)
  require(all(not Path(pin['path']).is_relative_to(run) for pin in pins.values()), 'External authority must stay outside run')
  a=_load_auditor();old_cli,old_deadline=a.CLI_AUTHORITY,a.DEADLINE
  a.DEADLINE=min(old_deadline,DEADLINE)
  authority=SimpleNamespace(run=str(run),expected_run_git=RUN_GIT,expected_audit_git=args.expected_git,
      launcher_terminal=pins['launcher_terminal']['path'],launcher_sha=args.launcher_sha,
      launcher_launch=pins['launcher_launch']['path'],launcher_launch_sha=args.launcher_launch_sha)
  try:
    dependencies=_verify_dependencies(a,args.prior_physics_audit_git)
    original_authority=a.external_authority(authority)
    report=_physics_authority(args,a,pins)
    a.CLI_AUTHORITY=authority
    auth=a.authenticate(run,RUN_GIT,args.expected_git)
    require(report['inventory'] == auth['inventory'] and a.inventory(run) == auth['inventory'],
        'Whole run differs from completed physics audit before trusted decode')
    require(len(auth['units']) == 512 and all(entry['unit'] == i and
        entry['prefix'] == f'family_{i//16:02d}/unit_{i%16:02d}/' for i,entry in enumerate(auth['units'])),
        'Complete fixed ordered32x16 metadata required before trusted decode')
    inventory_sha=digest(canonical(auth['inventory']));rows=[]
    for entry in auth['units']:
      check();relative=entry['prefix']+'trajectory.pkl'
      require(entry['complete']['plan'] == plan(entry['unit']),'Metadata source plan differs')
      # This helper rechecks the same raw bytes immediately before pickle.
      record=a.trusted_pickle(auth,relative)
      rows.append(project_prefix(record,entry['unit'],dict(auth['inventory']['files'][relative]),inventory_sha))
    validate_order(rows,inventory_sha)
    require(a.inventory(run) == auth['inventory'],'Whole run changed during source projection')
    for name,pin in pins.items():require(raw_pin(pin['path'],pin['sha256']) == pin,'External authority changed')
    require(a.external_authority(authority) == original_authority and
        _physics_authority(args,a,pins) == report,'Complete external authority graph changed')
    require(_verify_dependencies(a,args.prior_physics_audit_git) == dependencies,'Prior audit dependencies changed')
    require(a.command_output(['git','rev-parse','HEAD']).decode().strip() == args.expected_git and
        not a.command_output(['git','status','--porcelain','--untracked-files=all']) and not a.OWNED,
        'Index checkout changed or left metadata helpers')
    return dict(version=VERSION,source_kind='complete-audited-R5-pretreatment',run_directory=str(run),
        run_git=RUN_GIT,prior_physics_audit_git=args.prior_physics_audit_git,metadata_authentication_git=args.expected_git,
        original_authority=original_authority,external_pins=pins,prior_dependency_pins=dependencies,
        run_inventory_sha256=inventory_sha,complete_families=32,complete_units=512,
        fit_families=list(range(8)),held_families=list(range(8,32)),fit_rows=128,held_rows=384,rows=rows,
        source_graph_and_prior_physics_audit_authenticated=True,source_recomputed_latents=False,
        physics_reexecuted=False,model_executed=False,scientific_verdict_recomputed=False,
        source_selected_by_scientific_sign=False,GPU_used=False,subsequent_learning_authorized=False,
        elapsed_seconds=time.monotonic()-ENTRY_STARTED,whole_limit_seconds=SECONDS)
  finally:
    a.CLI_AUTHORITY,a.DEADLINE=old_cli,old_deadline


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  for name in ('run','expected-git','prior-physics-audit-git','launcher-terminal','launcher-sha','launcher-launch','launcher-launch-sha',
      'audit-report','audit-report-sha','audit-terminal','audit-terminal-sha','audit-launch','audit-launch-sha','output'):
    parser.add_argument('--'+name,required=True)
  args=parser.parse_args()
  require(os.name == 'posix','Real source-index entry requires Linux CPU')
  output=fresh_output(args.output,args.run)
  result=authenticate_and_index(args);check()
  with output.open('x',encoding='utf-8',newline='\n') as stream:
    json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
  check()


if __name__ == '__main__':main()
