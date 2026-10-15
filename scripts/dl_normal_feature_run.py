"""Reference0 normal-history features; implementation does not activate a run.

CPU parent authenticates complete512 source and successful labels before a
single owned CUDA worker. Every source frame keeps the real public/native
byte oracle. No default GPU budget, physics, rendering, learning or readout.
"""
import time
ENTRY_STARTED=time.monotonic()
import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import pickle
import re
import signal
import socket
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
VERSION='DL-normal-feature-run-r1'
PLAN_VERSION='DL-normal-feature-run-execution-r1'
CHECKPOINT='c6a7d92e24669a515bb44693faf73e442d9868f9fc814eec0e22e069aa594d24'
CONFIG='877f8470cb4f7f45d143e3099fbd3f55a828991afe33779b7b04c4264f6c8377'
FULL_STATE='4e0d1a0e1de806bf8e4863278cabd171ed0d208dddc1fea18b8b54d68ca6383f'
PARAMS='cffc13c0dcdfbd0c8b0e5c503c9d6251f4b1550c08b85a8742a638655d66d372'
COUNTERS=dict(updates=249493,batches=249493,actions=62563,eval_actions=1001,eval_seed=1289196729)
DEADLINE,AUDITOR=float('inf'),None
OUTPUT=None
NORMS=('docs/DL_NORMAL_READOUT.md','docs/DL_NORMAL_FEATURE_ENGINEERING.md',
    'docs/DL_NORMAL_FEATURE_RUN_ENGINEERING.md','docs/DL_NORMAL_LABEL_ENGINEERING.md')
COMMON=('run','expected_git','prior_physics_audit_git','launcher_terminal','launcher_sha','launcher_launch',
    'launcher_launch_sha','audit_report','audit_report_sha','audit_terminal','audit_terminal_sha','audit_launch',
    'audit_launch_sha','labels','labels_sha','labels_terminal','labels_terminal_sha','checkpoint','config',
    'execution_plan','execution_plan_sha','output')


def require(value,message):
  if not value:raise ValueError(message)


def check():
  require(time.monotonic() < DEADLINE,'Whole adopted feature execution deadline exceeded')


def sha_bytes(body):return hashlib.sha256(body).hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def decode(body):
  def pairs(items):
    result={}
    for key,value in items:
      require(key not in result,'Duplicate JSON key');result[key]=value
    return result
  return json.loads(body,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))


def pin(path,expected=None):
  check();path=Path(path).absolute()
  require(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)),'Ordinary pinned file required')
  before=path.stat();digest=hashlib.sha256()
  with path.open('rb') as stream:
    for block in iter(lambda:stream.read(1024*1024),b''):check();digest.update(block)
  after=path.stat()
  require((before.st_size,before.st_mtime_ns,before.st_ino) == (after.st_size,after.st_mtime_ns,after.st_ino),
      'File changed during hash')
  result=dict(path=str(path.resolve()),bytes=after.st_size,sha256=digest.hexdigest())
  if expected is not None:require(re.fullmatch('[0-9a-f]{64}',expected) and result['sha256'] == expected,'Raw file SHA differs')
  return result


def pinned_bytes(entry):
  require(pin(entry['path'],entry['sha256']) == entry,'Bound path/bytes differ')
  body=Path(entry['path']).read_bytes();check()
  require(len(body) == entry['bytes'] and sha_bytes(body) == entry['sha256'],'Actual same-buffer bytes differ')
  return body


def write_json(path,value):
  check()
  with Path(path).open('xb') as stream:
    stream.write(canonical(value)+b'\n');stream.flush();os.fsync(stream.fileno())
  check()


def typed_spec(value):
  import numpy as np
  from scripts.dl_reset_learning import tree_digest
  value=np.asarray(value)
  require((value.dtype.kind in 'buif' or str(value.dtype) == 'bfloat16') and np.isfinite(value).all(),
      'Finite typed numeric tensor required')
  return dict(logical_dtype=str(value.dtype),dtype_str=value.dtype.str,shape=list(value.shape),bytes=value.nbytes,
      typed_sha256=tree_digest(value),raw_sha256=sha_bytes(value.tobytes()))


def store_current(path,current):
  import numpy as np
  names=('h','q_prob','p_prob','qlogits','plogits','root_key','latent_key','actor_key','native_raw')
  schema={};arrays={}
  for name in names:
    value=np.asarray(current[name]);schema[name]=typed_spec(value)
    arrays[name]=value.view(np.uint16) if str(value.dtype) == 'bfloat16' else value
  check()
  with Path(path).open('xb') as stream:np.savez(stream,**arrays);stream.flush();os.fsync(stream.fileno())
  check();return schema


def extract_prefix(wrapper,record,row,inventory_sha256,*,fixture=False):
  """Reusable prefix core; no source, label or endpoint identity certification.

  Production reference0 supplies the authenticated rows and a production-geometry
  wrapper. Future endpoints need their own complete provenance and authorization.
  """
  import numpy as np
  from scripts import dl_normal_source_index as source
  from scripts.dl_reset_learning import tree_digest
  require(type(fixture) is bool and wrapper.fixture is fixture,'Explicit matching fixture boundary required')
  require(fixture or wrapper.shape == (4096,32,32),'Real production feature geometry required')
  require(source.project_prefix(record,row['global_unit'],row['trajectory'],inventory_sha256) == row,
      'Feature source ID/prefix/clock/order differs')
  t=row['plan']['t'];identity=row['source_id'];receipts=[]
  reset=wrapper.reset(record['observations'][0],eval_seed=row['plan']['eval_seed'],source_id=identity)
  reset=dict(reset,root_key=reset['root_key'].tolist())
  for step in range(t+1):
    check();previous=np.zeros(12,np.float32) if step == 0 else record['raw'][step-1]
    if step < t:frame=wrapper.step(record['observations'][step],record['raw'][step],step,source_prev_raw=previous)
    else:frame=wrapper.observe_current(record['observations'][step],step,source_prev_raw=previous)
    expected=row['frames'][step]
    require(frame['source_id'] == identity and frame['counter'] == step and
        frame['observation_sha256'] == expected['observation_sha256'] and
        frame['root_key'].tolist() == expected['root_key'] and
        tree_digest(frame['source_prev_raw']) == expected['previous_raw_sha256'],
        'Feature frame does not bind the exact source observation/root/previous raw')
    oracle=frame['public_oracle']
    require(oracle == dict(public_policy_calls=1,public_key_byte_exact=True,native_raw_byte_exact=True,
        native_full_after_carry_byte_exact=True,counters_restored=True,before_host_override=True),
        'Every frame requires the actual exact public/native oracle')
    require(frame['original_nj_seed_calls'] == 2 and frame['probability_nj_seed_calls'] == 0,
        'Original two-seed policy and RNG-free probability query required')
    require(frame['source_raw_available'] is (step < t) and frame['feature_only_final'] is (step == t) and
        frame['host_source_override_performed'] is (step < t) and frame['next_prev_raw_bound'] is (step < t),
        'Current source raw unavailable boundary differs')
    if step < t:
      require(frame['source_raw'].tobytes() == record['raw'][step].tobytes() and
          frame['history_after_sha256'] == tree_digest(frame['history_after']), 'Actual source raw/history override differs')
    else:
      require(frame['source_raw'] is None and frame['history_after'] is frame['history_after_sha256'] is None,
          'No fabricated current source raw/history allowed')
    receipts.append(dict(source_id=identity,counter=step,observation_sha256=frame['observation_sha256'],
        root_key=frame['root_key'].tolist(),latent_key=frame['latent_key'].tolist(),actor_key=frame['actor_key'].tolist(),
        before_carry_sha256=frame['before_carry_sha256'],native_after_sha256=tree_digest(frame['native_after']),
        native_raw=frame['native_raw'].tolist(),native_raw_sha256=tree_digest(frame['native_raw']),
        source_prev_raw_sha256=tree_digest(frame['source_prev_raw']),source_raw_sha256=tree_digest(frame['source_raw']) if step < t else None,
        history_after_sha256=frame['history_after_sha256'],source_history_sha256=frame['source_history_sha256'],
        source_raw_available=step < t,feature_only_final=step == t,public_oracle=oracle,
        original_nj_seed_calls=frame['original_nj_seed_calls'],probability_nj_seed_calls=frame['probability_nj_seed_calls']))
  current=frame;seal=wrapper.finish()
  require(seal['episode']['source_id'] == identity and seal['episode']['next_counter'] == t+1 and
      seal['episode']['source_command_frames'] == t and seal['feature_only_final'] is True and
      seal['final_carry_kind'] == 'feature_only_native_after' and seal['final_carry_sha256'] is None and
      seal['final_native_after_sha256'] == receipts[-1]['native_after_sha256'] and
      seal['full_state_policy_copy_counters_unchanged'] is True,'Complete own-history/reference state seal differs')
  return dict(current=current,reset=reset,frames=receipts,seal=seal,source_id=identity,
      prefix_typed_sha256=row['prefix_typed_sha256'],source_authenticated=False,labels_authenticated=False,
      real_N1M_authenticated=False,fixture_geometry=fixture)


def execution_plan(args):
  global DEADLINE
  item=pin(args.execution_plan,args.execution_plan_sha);plan=decode(pinned_bytes(item))
  require(plan['version'] == PLAN_VERSION and plan['git_sha'] == args.expected_git and plan['role'] == 'reference0' and
      plan['logical_server'] == 'sv3' and type(plan['physical_gpu']) is int and plan['physical_gpu'] in (4,5,6,7) and
      re.fullmatch(r'GPU-[0-9a-f-]+',plan['gpu_uuid']) and type(plan['whole_seconds']) is int and plan['whole_seconds'] > 0 and
      plan['complete_units'] == 512,'Explicit adopted reference0 execution plan required; no default budget/device')
  outer=plan.get('labels_outer_execution')
  require(type(outer) is dict and set(outer) == {'path','bytes','sha256'} and
      pin(outer['path'],outer['sha256']) == outer,'Mandatory successful labels actual-outer-exit raw pin required')
  require(os.name == 'posix' and socket.gethostname().split('.')[0] == 'lyg0326','Actual assigned sv3 Linux required')
  DEADLINE=ENTRY_STARTED+plan['whole_seconds'];check()
  return plan,item


def current_sources(a,args):
  require(a.command_output(['git','rev-parse','HEAD']).decode().strip() == args.expected_git and
      not a.command_output(['git','status','--porcelain','--untracked-files=all']),'Clean exact feature source required')
  manifest=a.source_manifest(args.expected_git)
  for name in NORMS:
    body=a.command_output(['git','cat-file','blob',args.expected_git+':'+name])
    manifest[name]=dict(bytes=len(body),sha256=sha_bytes(body))
  for name in ('scripts/dl_normal_feature_run.py','scripts/dl_normal_features.py','scripts/dl_normal_source_index.py',
      'scripts/dl_normal_labels.py','analysis/dl_normal_task_query.py'):
    require(name in manifest,'Complete tracked feature/source/labels implementation required')
  for name,entry in manifest.items():
    actual=pin(ROOT/name)
    require(all(actual[k] == entry[k] for k in ('bytes','sha256')),'Actual current feature source blob differs: '+name)
  return manifest


def labels_authority(args,index,a,plan):
  import numpy as np
  output=pin(args.labels,args.labels_sha);terminal=pin(args.labels_terminal,args.labels_terminal_sha)
  value=decode(pinned_bytes(output));authority=decode(pinned_bytes(terminal))
  require(value['version'] == 'DL-normal-labels-r1' and value['run_git'] == index['run_git'] and
      value['prior_physics_audit_git'] == index['prior_physics_audit_git'] and value['run_directory'] == index['run_directory'] and
      value['complete_units'] == 512 and value['complete_families'] == 32 and value['fit_families'] == list(range(8)) and
      value['held_families'] == list(range(8,32)) and value['source_complete_and_stable'] is True and
      value['current_labels_replayed'] is True and value['seconds'] == 900 and 0 < value['elapsed_seconds'] <= 900 and
      value['external_pins'] == index['external_pins'] and value['run_inventory_sha256'] == index['run_inventory_sha256'],
      'Complete successful source-aligned current labels required')
  require(authority['version'] == 'DL-normal-labels-launcher-r1' and authority['git_sha'] == value['git_sha'] and
      authority['run_directory'] == index['run_directory'] and authority['returncode'] == 0 and authority['seconds'] == 900 and
      0 < authority['elapsed_seconds'] <= 900 and authority['labels_output'] == output and authority['success'] is True and authority['parent_identity_gone'] is True and
      authority['owned_groups_cleaned'] is True and authority['external_failure'] is None,'Actual bounded labels exit authority required')
  execution_pin=plan['labels_outer_execution']
  execution=decode(pinned_bytes(execution_pin))
  controller='/data/Policy_Discrepancy/runs/dl-normal-labels-inputs-20261015-'+value['git_sha'][:7]+'/launch.py'
  expected_command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','sv3',
      '/data/Policy_Discrepancy/envs/dreamer/bin/python',controller,'--expected-git',value['git_sha']]
  require(execution['git_sha'] == value['git_sha'] and execution['command'] == expected_command and
      execution['returncode'] == 0 and execution['actual_outer_complete_within900'] is True and
      execution['GPU_used'] is False and 0 < authority['elapsed_seconds'] <= execution['elapsed_seconds'] <= 900 and
      re.fullmatch('[0-9a-f]{64}',execution['log_sha256']),
      'Actual observed labels-controller final exit0/whole900 authority required')
  identity=authority['parent_identity']
  require(identity['pid'] == identity['pgid'] == identity['sid'],'Actual labels owned session identity required')
  a.original_inactive(identity['pid'],identity['start_ticks']);a.original_group_inactive(identity)
  require(len(value['rows']) == 512,'Exactly512 successful label rows required')
  for source_row,row in zip(index['rows'],value['rows']):
    for key in ('source_id','global_unit','family','replicate','split','plan','trajectory','prefix_typed_sha256'):
      require(row[key] == source_row[key],'Labels source ID/order/prefix/fit-held differs')
    require(row['source_reward_index'] == source_row['plan']['t']-1 and row['current_observation_index'] == source_row['plan']['t'] and
        row['current_observation_sha256'] == source_row['current']['observation_sha256'] and
        row['current_source_snapshot_sha256'] == source_row['current']['snapshot_sha256'] and
        row['source_snapshot_hash_format'] == source_row['current']['snapshot_hash_format'] and
        row['query_snapshot_hash_format'] == 'DL-normal-task-query-snapshot-r1' and
        row['query_before_snapshot_sha256'] == row['query_after_snapshot_sha256'] and
        row['task_rng_before_sha256'] == row['task_rng_after_sha256'],'Labels current position/readonly source binding differs')
    for name in ('velocity_x','upright','upright_reward','move_reward','reward_component_product','current_reward_f64'):
      raw=bytes.fromhex(row['current_f64_hex'][name]);number=np.asarray(row['current_values'][name])
      require(len(raw) == 8 and number.dtype == np.float64 and number.shape == () and np.isfinite(number) and
          number.tobytes() == raw,'Original finite F64 labels bytes differ')
  # Bind exactly the labels/query/index source blobs at its successful Git, and
  # require their actual current copies to match; future endpoints are not inferred.
  names=sorted(value['current_source_pins'])
  require(set(names) == {'scripts/dl_normal_labels.py','scripts/dl_normal_source_index.py','analysis/dl_normal_task_query.py'} and
      re.fullmatch('[0-9a-f]{40}',value['git_sha']),'Exact successful labels implementation pins required')
  raw=a.command_output(['git','cat-file','--batch'],''.join(value['git_sha']+':'+n+'\n' for n in names).encode());offset=0
  for name in names:
    end=raw.index(b'\n',offset);head=raw[offset:end].split();offset=end+1
    require(len(head) == 3 and head[1] == b'blob','Original successful label source Git blob required')
    size=int(head[2]);data=raw[offset:offset+size];offset+=size+1
    require(raw[offset-1:offset] == b'\n' and value['current_source_pins'][name] == dict(bytes=size,sha256=sha_bytes(data)) and
        (ROOT/name).read_bytes() == data,'Successful labels source differs from current reused code')
  require(offset == len(raw),'Unexpected labels Git source blob tail')
  return dict(output=output,terminal=terminal,outer_execution=execution_pin),value


def identity(pid):
  fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
  return dict(pid=int(pid),pgid=int(fields[2]),sid=int(fields[3]),start_ticks=int(fields[19]))


def command_output(command,payload=None):
  """GPU-worker metadata helpers inherit its one fenced owned session."""
  check()
  result=subprocess.run(command,cwd=ROOT,input=payload,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
      timeout=min(30,max(.001,DEADLINE-time.monotonic())))
  require(result.returncode == 0,'Feature metadata command failed: '+result.stderr.decode(errors='replace')[:200])
  check();return result.stdout


def worker_source(expected,manifest):
  require(command_output(['git','rev-parse','HEAD']).decode().strip() == expected and
      not command_output(['git','status','--porcelain','--untracked-files=all']), 'Clean exact worker source required')
  # CPU parent derived these hashes from real Git blobs, not archive/autocrlf.
  for name,entry in manifest.items():
    actual=pin(ROOT/name)
    require(all(actual[k] == entry[k] for k in ('bytes','sha256')),'Worker source changed: '+name)


def child_environment(plan):
  result=dict(os.environ,CUDA_VISIBLE_DEVICES=plan['gpu_uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID',
      JAX_PLATFORMS='cuda',MUJOCO_GL='disable')
  for name in ('MUJOCO_EGL_DEVICE_ID','PYOPENGL_PLATFORM'):result.pop(name,None)
  return result


def compute_environment(plan):
  expected=dict(CUDA_VISIBLE_DEVICES=plan['gpu_uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID',JAX_PLATFORMS='cuda',MUJOCO_GL='disable')
  require(all(os.environ.get(k) == v for k,v in expected.items()) and
      all(k not in os.environ for k in ('MUJOCO_EGL_DEVICE_ID','PYOPENGL_PLATFORM')),
      'Actual single-GPU compute-only environment differs; no EGL initialization allowed')


def sample_resource(plan,path,allowed,*,idle=False):
  from scripts import dl_resources as resources
  state=resources.live_state(plan['physical_gpu'])
  # Persist complete available XML summary and raw pmon BEFORE any conflict guard.
  with Path(path).open('ab') as stream:
    stream.write(canonical(dict(monotonic=time.monotonic(),allowed_pids=sorted(allowed),live_state=state))+b'\n')
    stream.flush();os.fsync(stream.fileno())
  require(state['gpu_uuid'] == plan['gpu_uuid'],'Actual assigned physical UUID changed')
  if idle:resources.require_idle(state,plan['gpu_uuid'])
  for job in state['processes']+state['pmon_jobs']:
    require(job['pid'] in allowed and job['type'] == 'C','Unknown or graphics work appeared on assigned GPU')
  for line in state['raw_pmon'].splitlines():
    cells=line.split()
    if len(cells) >= 3 and cells[1].isdigit() and int(cells[1]) in allowed:
      require(cells[0] == str(plan['physical_gpu']) and cells[2] == 'C','Owned context on another GPU or graphics context')
  check();return state


def initialized_runtime(plan):
  compute_environment(plan)
  rows=command_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader']).decode()
  own=[x.split(',')[1].strip() for x in rows.splitlines() if len(x.split(',')) == 2 and x.split(',')[0].strip() == str(os.getpid())]
  require(own == [plan['gpu_uuid']],'Actual initialized compute UUID differs')
  state=sample_resource(plan,Path(OUTPUT)/'worker_resources.jsonl',{os.getpid()})
  return dict(pid=os.getpid(),compute_uuid=own,physical_gpu=plan['physical_gpu'],egl_uuid=None,
      graphics_verified=False,environment={k:os.environ.get(k) for k in
      ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','JAX_PLATFORMS','MUJOCO_GL','MUJOCO_EGL_DEVICE_ID','PYOPENGL_PLATFORM')},live_state=state)


def load_reference(inputs,output):
  """Single original N1M load; authenticate the actual decoded byte buffer."""
  import numpy as np
  import jax
  from scripts.dl_reset_learning import tree_digest
  from scripts.dl_diagnostic import load_config
  from scripts.dl_predictive_mc import agent_from_spaces
  from embodied.run.protocol_v1 import parameter_digest
  require(importlib.metadata.version('numpy') == '1.26.4' and importlib.metadata.version('jax') == '0.4.33' and
      str(jax.config.jax_default_prng_impl) == 'threefry2x32' and jax.config.jax_threefry_partitionable is False and
      jax.config.jax_enable_x64 is False,'Frozen numerical package/PRNG configuration differs')
  require(inputs['checkpoint']['sha256'] == CHECKPOINT and inputs['config']['sha256'] == CONFIG,'Original N1M file pins required')
  payload=pickle.loads(pinned_bytes(inputs['checkpoint']))  # Complete authority and raw pin precede trusted decode.
  require(set(payload) == {'params','counters'} and len(payload['params']) == 294 and payload['counters'] == COUNTERS and
      tree_digest(payload) == FULL_STATE,'Original complete N1M state/counters differ')
  pinned_bytes(inputs['config'])
  config=load_config(inputs['config']['path'],'dmc_quadruped_walk',output,'cuda')
  require(config.seed == 0 and str(config.jax.compute_dtype) == 'bfloat16','Original seed0/BF16 required')
  agent=agent_from_spaces(config);agent.load(payload)
  saved=agent.save()  # Preserve original Python counter types in full-state hash.
  require(tree_digest(saved) == FULL_STATE and saved['counters'] == COUNTERS and parameter_digest(agent) == PARAMS,
      'Actual loaded original N1M full state/parameter identity differs')
  del payload,saved
  require(len(jax.devices()) == 1 and jax.devices()[0].platform == 'gpu','Exactly one actual CUDA device required')
  metadata=dict(role='reference0',checkpoint=dict(inputs['checkpoint']),config_pin=dict(inputs['config']),
      full_state_sha256=FULL_STATE,params_sha256=PARAMS,counters=dict(COUNTERS),config=config.flat,
      packages={name:importlib.metadata.version(name) for name in ('numpy','jax','jaxlib','ninjax','elements','ml-dtypes')},
      jax_devices=[str(x) for x in jax.devices()],training_updates=0,environment_constructions=0,render_calls=0)
  return agent,metadata


def load_record(run,row):
  root=Path(run).resolve(strict=True);path=root/row['trajectory']['path']
  require(path.resolve(strict=True).is_relative_to(root),'Source trajectory escape refused')
  item=pin(path,row['trajectory']['sha256'])
  require(item['bytes'] == row['trajectory']['bytes'],'Source trajectory size differs')
  return pickle.loads(pinned_bytes(item))  # Complete source/labels authority is mandatory at caller.


def bind_current_label(record,row,label):
  import numpy as np
  from scripts.dl_reset_learning import tree_digest
  t=row['plan']['t']
  for name in ('source_id','global_unit','family','replicate','split','plan','trajectory','prefix_typed_sha256'):
    require(label[name] == row[name],'Current label/source ID or prefix differs')
  reward=np.asarray(record['reward_f64'][t-1]);encoded=np.asarray(record['observations'][t]['reward'])
  require(reward.dtype == np.float64 and reward.shape == () and encoded.dtype == np.float32 and encoded.shape == () and
      reward.tobytes().hex() == label['current_f64_hex']['current_reward_f64'] and
      np.asarray(label['current_values']['current_reward_f64'],np.float64).tobytes() == reward.tobytes() and
      np.asarray(label['current_values']['current_reward_f32'],np.float32).tobytes() == encoded.tobytes() and
      tree_digest(reward) == row['current']['preceding_reward_f64_sha256'] and label['static_model'] == record['static_model'] and
      label['velocity_frame'] == 'torso-local-velocimeter-x' and label['current_source_raw_used'] is False,
      'Current labels must bind reward[t-1]/obs[t], not the next intervention reward')


def produce_units(wrapper,index,labels,output):
  """All512 only. This helper accepts no fixture/partial-count production flag."""
  from scripts import dl_normal_source_index as source
  source.validate_order(index['rows'],index['run_inventory_sha256'])
  require(len(labels['rows']) == 512 and wrapper.fixture is False,'Complete actual reference0 extraction required')
  files=[];calls=0
  for row,label in zip(index['rows'],labels['rows']):
    check();unit=row['global_unit'];directory=output/f'unit_{unit:03d}';directory.mkdir()
    record=load_record(index['run_directory'],row);bind_current_label(record,row,label)
    result=extract_prefix(wrapper,record,row,index['run_inventory_sha256'])
    specs=store_current(directory/'current.npz',result['current'])
    write_json(directory/'history.json',dict(version=VERSION,source_id=row['source_id'],frames=result['frames']))
    complete=dict(version=VERSION,role='reference0',source_id=row['source_id'],global_unit=unit,family=row['family'],
        replicate=row['replicate'],split=row['split'],plan=row['plan'],trajectory=row['trajectory'],
        prefix_typed_sha256=row['prefix_typed_sha256'],current_label_sha256=sha_bytes(canonical(label)),
        feature_schema=specs,reset=result['reset'],seal=result['seal'],history_frames=row['plan']['t']+1,
        public_policy_calls=row['plan']['t']+1,current_source_raw_available=False,
        artifacts={name:pin(directory/name) for name in ('current.npz','history.json')})
    write_json(directory/'complete.json',complete);files.append(pin(directory/'complete.json'))
    calls+=complete['public_policy_calls']
    print(json.dumps(dict(unit_complete=unit,complete_units=len(files))),flush=True)
  require(len(files) == 512,'No partial feature publication')
  return files,calls


def verify_worker(output,index,labels):
  """CPU closure: all512 receipts/tensors; no independent NN reevaluation claim."""
  import numpy as np
  import ml_dtypes
  from scripts.dl_reset_learning import tree_digest
  before=output_inventory(output)
  complete_pin=pin(output/'worker_complete.json');complete=decode(pinned_bytes(complete_pin))
  require(complete['version'] == VERSION and complete['role'] == 'reference0' and complete['complete_units'] == 512 and
      complete['real_N1M_inputs_authenticated'] is True and complete['source_index_sha256'] == pin(output/'source_index.json')['sha256'] and
      complete['full_state_policy_copy_counters_unchanged'] is True,'Complete actual reference0 worker required')
  expected_files={'budget.json','source_index.json','resources.jsonl','worker.log','process.json','initial.json',
      'runtime_before.json','runtime_after.json','final.json','worker_resources.jsonl','worker_complete.json'}
  calls=0
  require(len(complete['units']) == 512,'Every512 unit closure required')
  for row,label,item in zip(index['rows'],labels['rows'],complete['units']):
    unit=row['global_unit'];directory=output/f'unit_{unit:03d}'
    require(item == pin(directory/'complete.json'),'Ordered unit artifact pin differs')
    receipt=decode(pinned_bytes(item));t=row['plan']['t']
    require(receipt['version'] == VERSION and receipt['role'] == 'reference0' and receipt['source_id'] == row['source_id'] and
        receipt['global_unit'] == unit and receipt['plan'] == row['plan'] and receipt['split'] == row['split'] and
        receipt['prefix_typed_sha256'] == row['prefix_typed_sha256'] and receipt['current_label_sha256'] == sha_bytes(canonical(label)) and
        receipt['history_frames'] == receipt['public_policy_calls'] == t+1 and receipt['current_source_raw_available'] is False,
        'Source-aligned current reference0 unit differs')
    require(set(receipt['artifacts']) == {'current.npz','history.json'},'Exact unit artifact set required')
    for name,item in receipt['artifacts'].items():require(item == pin(directory/name),'Unit file bytes changed')
    history=decode(pinned_bytes(receipt['artifacts']['history.json']))
    require(history['source_id'] == row['source_id'] and len(history['frames']) == t+1,'All public history receipts required')
    require(receipt['reset']['initial_carry_sha256'] == history['frames'][0]['before_carry_sha256'] and
        receipt['reset']['root_key'] == row['frames'][0]['root_key'],'Actual reset/first-frame carry or root differs')
    for i,(frame,source_frame) in enumerate(zip(history['frames'],row['frames'])):
      require(frame['source_id'] == row['source_id'] and frame['counter'] == i and
          frame['observation_sha256'] == source_frame['observation_sha256'] and frame['root_key'] == source_frame['root_key'] and
          frame['source_prev_raw_sha256'] == source_frame['previous_raw_sha256'] and
          frame['public_oracle'] == dict(public_policy_calls=1,public_key_byte_exact=True,native_raw_byte_exact=True,
              native_full_after_carry_byte_exact=True,counters_restored=True,before_host_override=True) and
          frame['source_raw_available'] is (i < t) and frame['feature_only_final'] is (i == t) and
          frame['original_nj_seed_calls'] == 2 and frame['probability_nj_seed_calls'] == 0,
          'Every actual public/native oracle must bind its original source frame')
      if i < t:
        require(frame['source_raw_sha256'] == row['frames'][i+1]['previous_raw_sha256'] and
            frame['history_after_sha256'] == history['frames'][i+1]['before_carry_sha256'],
            'Actual source raw or consecutive conditional carry chain differs')
      else:
        require(frame['source_raw_sha256'] is None and frame['history_after_sha256'] is None,
            'Current frame cannot fabricate outgoing source raw or history after')
    seal=receipt['seal']
    require(seal['full_state_policy_copy_counters_unchanged'] is True and seal['full_state_sha256'] == FULL_STATE and
        seal['counters'] == COUNTERS and seal['fixture_geometry'] is False and seal['feature_only_final'] is True and
        seal['final_carry_sha256'] is None and seal['final_native_after_sha256'] == history['frames'][-1]['native_after_sha256'],
        'Reference full state/policy/counters or current-final carry seal differs')
    require(seal['episode']['source_id'] == row['source_id'] and seal['episode']['next_counter'] == t+1 and
        seal['episode']['source_command_frames'] == t and
        seal['episode']['history_sha256'] == history['frames'][-1]['source_history_sha256'],
        'Final conditional source-history seal differs')
    with np.load(io.BytesIO(pinned_bytes(receipt['artifacts']['current.npz'])),allow_pickle=False) as archive:
      specs=receipt['feature_schema'];require(set(archive.files) == set(specs),'Current tensor schema/file set differs')
      for name,spec in specs.items():
        value=archive[name]
        if spec['logical_dtype'] == 'bfloat16':
          require(value.dtype == np.uint16,'BF16 current storage must preserve original uint16 bits')
          value=value.view(ml_dtypes.bfloat16)
        require(typed_spec(value) == spec,'Original current tensor dtype/shape/raw/typed hash differs')
      geometry={'h':('bfloat16',[4096]),'q_prob':('float32',[32,32]),'p_prob':('float32',[32,32]),
          'qlogits':('bfloat16',[32,32]),'plogits':('bfloat16',[32,32]),'native_raw':('float32',[12]),
          'root_key':('uint32',[2]),'latent_key':('uint32',[2]),'actor_key':('uint32',[2])}
      require(set(specs) == set(geometry) and all((specs[n]['logical_dtype'],specs[n]['shape']) == x for n,x in geometry.items()),
          'Actual production current feature geometry/dtype required')
      require(all(archive[name].tolist() == history['frames'][-1][name] for name in ('root_key','latent_key','actor_key')) and
          archive['root_key'].tolist() == row['frames'][-1]['root_key'] and
          tree_digest(archive['native_raw']) == history['frames'][-1]['native_raw_sha256'],'Current feature/oracle key and raw differ')
      for name in ('q_prob','p_prob'):
        value=archive[name]
        require(np.all((value>=0)&(value<=1)) and np.all(np.abs(value.sum(-1)-1)<=1e-6),'Actual distribution probability invalid')
    expected_files.update(f'unit_{unit:03d}/'+name for name in ('current.npz','history.json','complete.json'));calls+=t+1
  require(complete['public_policy_calls'] == calls and complete['native_policy_calls'] == calls,'Full history call coverage differs')
  require(set(complete['artifacts']) == {'initial.json','runtime_before.json','runtime_after.json','final.json','worker_resources.jsonl'},
      'Exact worker closure artifacts required')
  for name,item in complete['artifacts'].items():require(pin(output/name) == item,'Final worker closure artifact changed')
  initial=decode(pinned_bytes(complete['artifacts']['initial.json']));final=decode(pinned_bytes(complete['artifacts']['final.json']))
  require(initial['full_state_sha256'] == final['full_state_sha256'] == FULL_STATE and
      initial['model']['params_sha256'] == PARAMS and initial['model']['role'] == 'reference0' and
      initial['policy_copy_sha256'] == final['policy_copy_sha256'] and initial['counters'] == final['counters'] == COUNTERS and
      final['resets'] == 512 and final['steps'] == calls and final['training_updates'] == final['environment_constructions'] == final['render_calls'] == 0,
      'Actual complete model identity and readonly final seal differ')
  actual_files={str(path.relative_to(output)).replace('\\','/') for path in output.rglob('*') if path.is_file()}
  require(actual_files == expected_files and not any(path.is_symlink() for path in output.rglob('*')),'Unexpected/missing feature artifact')
  require(output_inventory(output) == before,'Complete output inventory changed during CPU closure')
  return complete_pin,complete


def output_inventory(root):
  result={}
  for path in sorted(root.rglob('*')):
    require(not path.is_symlink() and path.resolve().is_relative_to(root),'Output link/escape refused')
    if path.is_dir():continue
    require(path.is_file(),'Output special file refused')
    result[str(path.relative_to(root)).replace('\\','/')]=pin(path)
  return result


def expire(signum,frame):
  # Initial source authentication can already own independent metadata sessions
  # before authenticate_and_index returns and parent assigns our global. Do not
  # import while handling a signal; use that already-loaded module's registry.
  auditor=AUDITOR
  if auditor is None:
    source=sys.modules.get('scripts.dl_normal_source_index')
    auditor=getattr(source,'AUDITOR',None)
  if auditor is not None:auditor.expire(signum,frame)
  os._exit(124)


def arm_timer():
  signal.signal(signal.SIGALRM,expire);signal.signal(signal.SIGTERM,expire)
  signal.setitimer(signal.ITIMER_REAL,max(.001,DEADLINE-time.monotonic()))


def worker(args,plan):
  global DEADLINE,OUTPUT
  require('jax' not in sys.modules,'Worker must authenticate before model runtime import')
  output=Path(args.output).resolve(strict=True);OUTPUT=output
  budget_item=pin(output/'budget.json',args.budget_sha);budget=decode(pinned_bytes(budget_item))
  require(budget['version'] == VERSION and budget['execution_plan'] == pin(args.execution_plan,args.execution_plan_sha) and
      budget['plan'] == plan and budget['git_sha'] == args.expected_git and budget['complete_units'] == 512 and
      budget['role'] == 'reference0' and os.environ.get('DL_NORMAL_FEATURE_PARENT') == str(os.getppid()) and
      os.environ.get('DL_NORMAL_FEATURE_BUDGET_SHA') == budget_item['sha256'], 'Direct bound worker invocation required')
  require(identity(os.getppid()) == budget['parent_identity'] and os.getsid(0) == os.getpgrp() == os.getpid(),
      'Original live parent and single fresh owned worker session required')
  DEADLINE=budget['monotonic_deadline'];require(DEADLINE == budget['entry_started']+plan['whole_seconds'],
      'Shared whole-entry deadline differs');check();arm_timer();compute_environment(plan)
  worker_source(args.expected_git,budget['source_files'])
  index=decode(pinned_bytes(budget['source_index']));labels=decode(pinned_bytes(budget['labels']['output']))
  require(pin(args.labels,args.labels_sha) == budget['labels']['output'] and
      pin(args.labels_terminal,args.labels_terminal_sha) == budget['labels']['terminal'],'Mandatory successful label pins changed')
  from scripts import dl_normal_source_index as source
  source.validate_order(index['rows'],index['run_inventory_sha256'])
  require(index['metadata_authentication_git'] == args.expected_git and index['source_graph_and_prior_physics_audit_authenticated'] is True,
      'Complete CPU source authentication required before trusted decode')
  # Fresh idleness here, after CPU input validation and immediately before model allocation.
  sample_resource(plan,output/'worker_resources.jsonl',set(),idle=True)
  agent,metadata=load_reference(budget['inputs'],output)
  from scripts import dl_normal_features as features
  from scripts.dl_reset_learning import tree_digest
  wrapper=features.FrozenNormalFeatures(agent)
  initial=dict(version=VERSION,model=metadata,full_state_sha256=tree_digest(agent.save()),
      policy_copy_sha256=wrapper._policy_hash(),counters=features.counters(agent))
  write_json(output/'initial.json',initial)
  write_json(output/'runtime_before.json',initialized_runtime(plan))
  units,calls=produce_units(wrapper,index,labels,output)
  wrapper.fence(full=True)
  final=dict(version=VERSION,full_state_sha256=tree_digest(agent.save()),policy_copy_sha256=wrapper._policy_hash(),
      counters=features.counters(agent),resets=wrapper._resets,steps=wrapper._steps,
      accessed_keys=sorted(wrapper.accessed_keys),training_updates=0,environment_constructions=0,render_calls=0)
  require(final['full_state_sha256'] == initial['full_state_sha256'] == FULL_STATE and
      final['policy_copy_sha256'] == initial['policy_copy_sha256'] and final['counters'] == initial['counters'] == COUNTERS and
      final['resets'] == 512 and final['steps'] == calls,'Actual reference0 full state/policy/counters changed')
  write_json(output/'final.json',final)
  write_json(output/'runtime_after.json',initialized_runtime(plan))
  worker_source(args.expected_git,budget['source_files'])
  for item in (*budget['inputs'].values(),budget['execution_plan'],budget['source_index'],*budget['labels'].values()):pinned_bytes(item)
  write_json(output/'worker_complete.json',dict(version=VERSION,git_sha=args.expected_git,role='reference0',complete_units=512,
      source_index_sha256=budget['source_index']['sha256'],real_N1M_inputs_authenticated=True,
      full_state_policy_copy_counters_unchanged=True,units=units,public_policy_calls=calls,native_policy_calls=calls,
      full_state_sha256=FULL_STATE,params_sha256=PARAMS,labels=budget['labels'],model_calls_readonly=True,
      physics_reexecuted=False,render_calls=0,training_updates=0,readout_fitted=False,scientific_verdict=False,
      artifacts={name:pin(output/name) for name in ('initial.json','runtime_before.json','runtime_after.json','final.json','worker_resources.jsonl')}))


def spawn_worker(args,a,plan,budget):
  command=[sys.executable,str(Path(__file__).resolve())]
  for name in COMMON:command.extend(['--'+name.replace('_','-'),str(getattr(args,name))])
  command.extend(['--worker','--budget-sha',budget['sha256']])
  environment=child_environment(plan)
  environment.update(DL_NORMAL_FEATURE_PARENT=str(os.getpid()),DL_NORMAL_FEATURE_BUDGET_SHA=budget['sha256'])
  previous=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM,signal.SIGTERM})
  child=None
  try:
    stream=(Path(args.output)/'worker.log').open('xb')
    try:
      child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT,
          start_new_session=True,preexec_fn=lambda:signal.pthread_sigmask(signal.SIG_SETMASK,previous))
    finally:stream.close()
    owned=identity(child.pid)
    a.OWNED[child.pid]={key:owned[key] for key in ('pid','pgid','start_ticks')}
    require(owned['pid'] == owned['pgid'] == owned['sid'],'Fresh owned feature worker session required')
    write_json(Path(args.output)/'process.json',dict(version=VERSION,parent_identity=identity(os.getpid()),
        worker_identity=owned,command=command,environment={k:environment.get(k) for k in
        ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','JAX_PLATFORMS','MUJOCO_GL','MUJOCO_EGL_DEVICE_ID','PYOPENGL_PLATFORM')},
        budget=budget,spawned_monotonic=time.monotonic()))
    return child
  except BaseException:
    if child is not None and child.pid in a.OWNED:a.cleanup(child)
    raise
  finally:signal.pthread_sigmask(signal.SIG_SETMASK,previous)


def fresh_directory(path,immutable):
  output=Path(path).absolute()
  require(not any(p.is_symlink() for p in (output,*output.parents)),'Output symlink refused')
  output=output.parent.resolve()/output.name
  require(not output.exists() and all(not output.is_relative_to(Path(root).resolve()) for root in immutable),
      'Fresh output outside every immutable input directory required')
  output.parent.mkdir(parents=True,exist_ok=True)
  output.mkdir();return output


def parent(args,plan,plan_pin):
  global AUDITOR
  require('jax' not in sys.modules,'Source/labels CPU parent must never import JAX')
  os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',MUJOCO_GL='disable')
  for name in ('MUJOCO_EGL_DEVICE_ID','PYOPENGL_PLATFORM'):os.environ.pop(name,None)
  from scripts import dl_normal_source_index as source
  from scripts import dl_normal_labels as labels_module
  # No source artifact or original input directory can receive new output files.
  inputs=dict(checkpoint=pin(args.checkpoint,CHECKPOINT),config=pin(args.config,CONFIG))
  immutable=[args.run,ROOT,*[Path(getattr(args,name)).resolve().parent for name in
      ('labels','labels_terminal','checkpoint','config','execution_plan','audit_report','audit_terminal','audit_launch','launcher_terminal','launcher_launch')]]
  immutable.append(Path(plan['labels_outer_execution']['path']).resolve().parent)
  output=fresh_directory(args.output,immutable);args.output=str(output)
  child=None;failure=None;worker_pin=None;worker_complete=None;index=None;label_pins=None;budget_pin=None
  try:
    source.DEADLINE=min(DEADLINE,time.monotonic()+900)
    index=source.authenticate_and_index(args);labels_module._index(index,args)
    a=AUDITOR=source.AUDITOR;require(a is not None,'Actual CPU authenticated auditor required')
    a.DEADLINE=DEADLINE;source.DEADLINE=DEADLINE
    manifest=current_sources(a,args);label_pins,label_values=labels_authority(args,index,a,plan)
    require('jax' not in sys.modules,'CPU authentication imported model runtime')
    write_json(output/'source_index.json',index)
    budget=dict(version=VERSION,role='reference0',git_sha=args.expected_git,complete_units=512,plan=plan,
        execution_plan=plan_pin,source_files=manifest,source_index=pin(output/'source_index.json'),labels=label_pins,
        inputs=inputs,parent_identity=identity(os.getpid()),entry_started=ENTRY_STARTED,monotonic_deadline=DEADLINE)
    write_json(output/'budget.json',budget);budget_pin=pin(output/'budget.json')
    sample_resource(plan,output/'resources.jsonl',set(),idle=True)
    child=spawn_worker(args,a,plan,budget_pin)
    last=0
    while child.poll() is None:
      check()
      require(child.pid in a.group_alive(a.OWNED[child.pid]) or child.poll() is not None,
          'Original owned feature worker identity no longer live')
      if time.monotonic()-last>=15:
        sample_resource(plan,output/'resources.jsonl',{child.pid});last=time.monotonic()
      try:child.wait(timeout=min(1,max(.001,DEADLINE-time.monotonic())))
      except subprocess.TimeoutExpired:pass
    a.cleanup(child)
    require(child.returncode == 0,'Owned feature worker failed; preserve evidence, no retry/partial acceptance')
    # All source files/raw pins/current Git are authenticated again after actual child exit.
    source.DEADLINE=a.DEADLINE=min(DEADLINE,time.monotonic()+900)
    labels_module._reauthenticate(a,args,index)
    new_label_pins,new_labels=labels_authority(args,index,a,plan)
    require(new_label_pins == label_pins and new_labels == label_values and current_sources(a,args) == manifest,
        'Complete source/code/labels changed during extraction')
    for entry in (*inputs.values(),plan_pin,budget_pin):pinned_bytes(entry)
    worker_pin,worker_complete=verify_worker(output,index,label_values)
    require(not a.OWNED and 'jax' not in sys.modules,'CPU parent or owned process closure differs')
    a.DEADLINE=DEADLINE;source.DEADLINE=DEADLINE
    write_json(output/'generation_complete.json',dict(version=VERSION,git_sha=args.expected_git,role='reference0',
        complete_units=512,complete_families=32,fit_families=list(range(8)),held_families=list(range(8,32)),
        source_complete_and_stable=True,labels_complete_and_stable=True,real_N1M_inputs_authenticated=True,
        worker_complete=worker_pin,budget=budget_pin,labels=label_pins,source_index=pin(output/'source_index.json'),
        output_inventory=output_inventory(output),
        elapsed_seconds=time.monotonic()-ENTRY_STARTED,whole_seconds=plan['whole_seconds'],
        independent_NN_reexecution=False,physics_reexecuted=False,render_calls=0,training_updates=0,
        readout_fitted=False,scientific_verdict=False,subsequent_learning_authorized=False))
  except BaseException as error:
    failure=type(error).__name__+': '+str(error);raise
  finally:
    if child is not None and AUDITOR is not None and child.pid in AUDITOR.OWNED:
      try:AUDITOR.cleanup(child)
      except BaseException as error:failure=(failure+'; ' if failure else '')+'cleanup '+str(error)
    if AUDITOR is not None:AUDITOR.DEADLINE=DEADLINE
    if time.monotonic() < DEADLINE:
      generation=pin(output/'generation_complete.json') if (output/'generation_complete.json').is_file() else None
      write_json(output/'supervisor_terminal.json',dict(version=VERSION,git_sha=args.expected_git,plan=plan,
          parent_identity=identity(os.getpid()),worker_returncode=None if child is None else child.returncode,
          failure=failure,owned_groups_cleaned=AUDITOR is not None and not AUDITOR.OWNED,
          generation_complete=generation,elapsed_seconds=time.monotonic()-ENTRY_STARTED,
          parent_exit_still_requires_external_authority=True))
    require(failure is None,'Feature extraction failed: '+str(failure))
  return worker_complete


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  for name in COMMON:parser.add_argument('--'+name.replace('_','-'),required=True)
  parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
  parser.add_argument('--budget-sha',help=argparse.SUPPRESS)
  args=parser.parse_args();plan,plan_pin=execution_plan(args);arm_timer()
  if args.worker:worker(args,plan)
  else:
    require(not args.budget_sha,'Worker authority flag cannot be passed to parent')
    parent(args,plan,plan_pin)
  check()
  print(json.dumps(dict(version=VERSION,role='reference0',complete_units=512,
      output=str(Path(args.output).resolve()),elapsed_seconds=time.monotonic()-ENTRY_STARTED)),flush=True)


if __name__ == '__main__':main()
