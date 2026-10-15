"""DL-normal-labels-r1: authenticate all R5 sources, then replay current labels.

The production entry is fixed512/8 fit families/24 held families, CPU900sec
including imports, source authority, hashes, official replay and publication.
It never selects by D/e/Y, reconstructs a renderer/model, restores a snapshot,
or uses the intervention's outgoing raw[t]. Implementation is not permission
to execute a real dataset; the caller still needs the adopted execution budget.
"""
import time
ENTRY_STARTED=time.monotonic()
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import signal
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
SECONDS,DEADLINE,SOURCE=900,ENTRY_STARTED+900,None


def expire(signum,frame):
  if SOURCE is not None:SOURCE.expire(signum,frame)
  os._exit(124)


if __name__ == '__main__':
  os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',MUJOCO_GL='disable')
  for name in ('MUJOCO_EGL_DEVICE_ID','PYOPENGL_PLATFORM'):os.environ.pop(name,None)
  if os.name == 'posix':
    signal.signal(signal.SIGALRM,expire);signal.signal(signal.SIGTERM,expire)
    signal.setitimer(signal.ITIMER_REAL,max(.001,DEADLINE-time.monotonic()))

import numpy as np
from scripts import dl_normal_source_index as source
from analysis import dl_normal_task_query as query
SOURCE=source
VERSION='DL-normal-labels-r1'
OWN_SOURCES=('scripts/dl_normal_labels.py','scripts/dl_normal_source_index.py','analysis/dl_normal_task_query.py')
PIN_FIELDS=(('launcher_terminal','launcher_sha'),('launcher_launch','launcher_launch_sha'),
    ('audit_report','audit_report_sha'),('audit_terminal','audit_terminal_sha'),('audit_launch','audit_launch_sha'))


def require(value,message):
  if not value:raise ValueError(message)


def check():
  require(time.monotonic() < DEADLINE,'Whole normal-labels900sec deadline exceeded')


def label_record(record,row,inventory_sha256,a):
  """Replay one already-authenticated source; this alone authenticates no run.

  The complete production path obtains ``a`` and rows only from the mandatory
  source index. This narrow helper is also exposed for generated CPU fixtures.
  """
  check();unit=row['global_unit']
  require(source.project_prefix(record,unit,row['trajectory'],inventory_sha256) == row,
      'Source ID/order/clock/current boundary or prefix hash differs')
  p=row['plan'];t=p['t']
  from dm_control import suite
  env=suite.load('quadruped','walk',task_kwargs={'random':p['env_seed']})
  try:
    reset=env.reset()
    require(reset.first() and reset.reward is None and reset.discount is None and
        a.static_model(env) == record['static_model'],'Official seeded reset/static MJB differs')
    a.state_equal(a.physics.snapshot(env,False),record['snapshots'][0])
    spec=env.action_spec()
    require(spec.dtype == np.float64 and spec.shape == (12,),'Official action12 F64 spec differs')
    a.same(a.physics.mapping(record['raw'][:t],spec.minimum,spec.maximum),record['applied'][:t],
        'Independent ordinary raw/applied mapping')
    for step in range(t):
      check()
      a.official_transition(env,record['applied'][step],record['reward_f64'][step],record['reward_f32'][step],
          record['discount'][step],tuple(bool(record[name][step]) for name in ('is_first','is_last','is_terminal')),
          record['snapshots'][step+1])
    result=query.query_current(env,record['snapshots'][t],np.asarray(record['reward_f64'][t-1]),
        np.asarray(record['observations'][t]['reward']),t)
    require(source.snapshot_sha256(record['snapshots'][t]) == row['current']['snapshot_sha256'] and
        row['current']['snapshot_hash_format'] == source.SNAPSHOT_HASH_FORMAT,
        'Source/index current snapshot hash binding differs')
    require(result['snapshot_hash_format'] == 'DL-normal-task-query-snapshot-r1' and
        result['current_model_sha256'] == record['static_model']['mjb_sha256'] and
        result['current_model_bytes'] == record['static_model']['mjb_bytes'],
        'Readonly query model/hash format differs from the replayed source')
    # Preserve exact F64 target/reward bytes as well as JSON-friendly values.
    values={name:float(result[name]) for name in ('velocity_x','upright','upright_reward','move_reward',
        'reward_component_product','current_reward_f64','current_reward_f32','physics_time','control_timestep')}
    return dict(source_id=row['source_id'],global_unit=unit,family=row['family'],replicate=row['replicate'],
        split=row['split'],plan=dict(p),trajectory=dict(row['trajectory']),prefix_typed_sha256=row['prefix_typed_sha256'],
        current_observation_sha256=row['current']['observation_sha256'],
        current_source_snapshot_sha256=row['current']['snapshot_sha256'],
        source_snapshot_hash_format=source.SNAPSHOT_HASH_FORMAT,
        query_snapshot_hash_format=result['snapshot_hash_format'],
        query_snapshot_sha256=result['source_snapshot_sha256'],
        query_before_snapshot_sha256=result['before_snapshot_sha256'],
        query_after_snapshot_sha256=result['after_snapshot_sha256'],
        task_rng_before_sha256=result['task_rng_before_sha256'],task_rng_after_sha256=result['task_rng_after_sha256'],
        static_model=dict(record['static_model']),velocity_frame='torso-local-velocimeter-x',
        current_values=values,current_f64_hex={name:np.asarray(result[name]).tobytes().hex() for name in
            ('velocity_x','upright','upright_reward','move_reward','reward_component_product','current_reward_f64')},
        source_reward_index=t-1,current_observation_index=t,replayed_steps=t,official_resets=1,
        explicit_restores=0,explicit_forwards=0,render_calls=0,current_source_raw_used=False,
        replay_full_snapshot_byte_exact=True,current_reward_F64_byte_exact=True,
        current_observation_F32_byte_exact=True,query_readonly_snapshot_and_model_byte_exact=True)
  finally:env.close()


def _index(index,args):
  require(index['version'] == source.VERSION and index['source_kind'] == 'complete-audited-R5-pretreatment' and
      index['run_git'] == source.RUN_GIT and index['metadata_authentication_git'] == args.expected_git and
      index['prior_physics_audit_git'] == args.prior_physics_audit_git and
      index['run_directory'] == str(Path(args.run).resolve()) and index['complete_units'] == 512 and
      index['complete_families'] == 32 and index['fit_families'] == list(range(8)) and
      index['held_families'] == list(range(8,32)) and index['fit_rows'] == 128 and index['held_rows'] == 384 and
      index['source_graph_and_prior_physics_audit_authenticated'] is True,'Complete fixed32x16 source authentication required')
  source.validate_order(index['rows'],index['run_inventory_sha256'])


def _bind_current(a,args):
  require(a.command_output(['git','rev-parse','HEAD']).decode().strip() == args.expected_git and
      not a.command_output(['git','status','--porcelain','--untracked-files=all']),
      'Complete labels require the clean exact current checkout')
  manifest=a.source_manifest(args.expected_git);result={}
  modules=(sys.modules[__name__],source,query)
  for name,module in zip(OWN_SOURCES,modules):
    require(name in manifest and Path(module.__file__).resolve() == ROOT/name,'Current label/query/index module not Git-bound')
    actual=source.raw_pin(ROOT/name)
    require(all(actual[k] == manifest[name][k] for k in ('bytes','sha256')),'Current label/query/index source blob differs')
    result[name]=dict(manifest[name])
  return result


def _load_record(run,row):
  relative=row['trajectory']['path'];path=(run/relative).resolve(strict=True)
  require(path.is_relative_to(run),'Source trajectory escape refused')
  actual=source.raw_pin(path,row['trajectory']['sha256'])
  require(actual['bytes'] == row['trajectory']['bytes'],'Indexed source trajectory length differs')
  raw=path.read_bytes();check()
  require(len(raw) == actual['bytes'] and hashlib.sha256(raw).hexdigest() == actual['sha256'],
      'Actual trajectory bytes differ before trusted decode')
  return pickle.loads(raw)  # Complete authority first; same pinned byte buffer only.


def _reauthenticate(a,args,index):
  pins=index['external_pins']
  require(set(pins) == {name for name,_ in PIN_FIELDS},'Complete explicit source authority pin set required')
  require(all(type(getattr(args,hash_name,None)) is str and
      re.fullmatch('[0-9a-f]{64}',getattr(args,hash_name)) for _,hash_name in PIN_FIELDS),
      'Every original source authority raw SHA remains mandatory at reauthentication')
  for name,hash_name in PIN_FIELDS:
    actual=source.raw_pin(getattr(args,name),getattr(args,hash_name))
    require(actual == pins[name],'Original source authority pin changed')
  authority=SimpleNamespace(run=index['run_directory'],expected_run_git=source.RUN_GIT,expected_audit_git=args.expected_git,
      launcher_terminal=pins['launcher_terminal']['path'],launcher_sha=args.launcher_sha,
      launcher_launch=pins['launcher_launch']['path'],launcher_launch_sha=args.launcher_launch_sha)
  old=a.CLI_AUTHORITY
  try:
    a.CLI_AUTHORITY=authority  # Explicit four producer arguments; never the fixture bypass.
    require(a.external_authority(authority) == index['original_authority'],'Original producer authority changed')
    report=source._physics_authority(args,a,pins)
    auth=a.authenticate(args.run,source.RUN_GIT,args.expected_git)
    stock=auth['inventory']
    require(stock == report['inventory'] and source.digest(source.canonical(stock)) == index['run_inventory_sha256'] and
        a.inventory(Path(args.run)) == stock,'Whole source run changed during current-label replay')
    require(source._verify_dependencies(a,args.prior_physics_audit_git) == index['prior_dependency_pins'],
        'Prior exact physics/source audit dependencies changed')
    for name,hash_name in PIN_FIELDS:
      require(source.raw_pin(getattr(args,name),getattr(args,hash_name)) == pins[name],'Source authority changed at closure')
    require(not a.OWNED,'Label authentication left owned metadata helpers')
  finally:a.CLI_AUTHORITY=old


def run(args):
  check();output=source.fresh_output(args.output,args.run)
  previous=source.DEADLINE;source.DEADLINE=min(previous,DEADLINE)
  a=None;old_deadline=None
  try:
    index=source.authenticate_and_index(args);_index(index,args)
    a=source.AUDITOR;require(a is not None,'Actual source-bound auditor required')
    old_deadline=a.DEADLINE;a.DEADLINE=min(old_deadline,DEADLINE)
    dependencies=_bind_current(a,args);rows=[];root=Path(index['run_directory']).resolve(strict=True)
    for row in index['rows']:
      check();record=_load_record(root,row)
      rows.append(label_record(record,row,index['run_inventory_sha256'],a))
    require(len(rows) == 512 and [x['source_id'] for x in rows] == [x['source_id'] for x in index['rows']],
        'All512 source-aligned current labels required; no partial publication')
    _reauthenticate(a,args,index)
    require(_bind_current(a,args) == dependencies,'Current label/query/index source changed')
    require('jax' not in sys.modules,'CPU labels unexpectedly imported a model runtime')
    binding_fields=('source_id','global_unit','plan','trajectory','split','prefix_typed_sha256')
    binding=[{key:row[key] for key in binding_fields} for row in index['rows']]
    result=dict(version=VERSION,git_sha=args.expected_git,run_git=source.RUN_GIT,
        prior_physics_audit_git=args.prior_physics_audit_git,run_directory=str(root),
        source_index_row_binding_sha256=source.digest(source.canonical(binding)),source_index_row_binding_fields=list(binding_fields),
        run_inventory_sha256=index['run_inventory_sha256'],
        external_pins=index['external_pins'],current_source_pins=dependencies,prior_dependency_pins=index['prior_dependency_pins'],
        complete_units=512,complete_families=32,fit_families=list(range(8)),held_families=list(range(8,32)),
        fit_rows=128,held_rows=384,rows=rows,official_prefix_steps=sum(x['replayed_steps'] for x in rows),
        official_resets=512,source_complete_and_stable=True,current_labels_replayed=True,
        seconds=SECONDS,elapsed_seconds=time.monotonic()-ENTRY_STARTED,
        model_calls=0,render_calls=0,explicit_restores=0,explicit_forwards=0,GPU_used=False,
        D_e_Y_used_for_labels_or_selection=False,scientific_verdict=False,subsequent_learning_authorized=False,
        limitations=['Velocity x is torso-local velocimeter x, not world x.',
            'Queries depend on synchronized derived buffers retained by the continuous official CPU replay.',
            'No renderer/model was replayed; cached RGB remains source-authenticated evidence.',
            'Same-instance query byte hashes and original R5 source snapshot hashes use different formats.',
            'This implementation and fixture do not authorize actual dataset execution or learning.'])
    check()
    with output.open('x',encoding='utf-8',newline='\n') as stream:
      json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    check();return result
  finally:
    source.DEADLINE=previous
    if a is not None and old_deadline is not None:a.DEADLINE=old_deadline


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  for name in ('run','expected-git','prior-physics-audit-git','launcher-terminal','launcher-sha','launcher-launch',
      'launcher-launch-sha','audit-report','audit-report-sha','audit-terminal','audit-terminal-sha','audit-launch',
      'audit-launch-sha','output'):parser.add_argument('--'+name,required=True)
  args=parser.parse_args();require(os.name == 'posix','Real current-label entry requires Linux CPU')
  result=run(args)
  print(json.dumps(dict(version=VERSION,complete_units=512,output=str(Path(args.output).resolve()),
      sha256=source.raw_pin(args.output)['sha256'],elapsed_seconds=time.monotonic()-ENTRY_STARTED)),flush=True)


if __name__ == '__main__':main()
