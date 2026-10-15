"""DL-r4: independent CPU audit of the original calibration-null stop.

This is a failure-evidence audit, never a replacement successful 32-EP assay.
An external capture pins all original bytes before any trusted pickle decode.
Only planned pair0 qq/null controls are independently replayed. No model,
renderer, held statistics, fitting, gate, optimizer or new training is executed.
"""
import time
STARTED = time.monotonic()
import os
import signal
import sys
DEADLINE = STARTED+1800
OWNED = set()


def expire(signum, frame):
  for pid in tuple(OWNED):
    try: os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError: pass
  os._exit(124)


if __name__ == '__main__' and os.name == 'posix':
  signal.signal(signal.SIGALRM, expire)
  signal.signal(signal.SIGTERM, expire)
  signal.setitimer(signal.ITIMER_REAL, max(.001, DEADLINE-time.monotonic()))

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['MUJOCO_GL'] = 'disable'
import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import re

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import dl_r4_audit as a
from analysis import dl_r4_statistics as s
a.DEADLINE = DEADLINE
OWNED = a.OWNED

VERSION = 'DL-r4-calibration-null-audit-r1'
CAPTURE_VERSION = 'DL-r4-failure-capture-r1'
RUN_GIT = '5ce1d3724ee8c5d01f09c4d687100a2f42f6ac83'
IDENTITIES = {'parent':(247014,407058309),'worker':(247226,407058446)}
NULL_LIMIT = .0025
STOP_MESSAGE = 'ValueError: Calibration null measurement failed; held remains unopened'
MISSING = ['worker_complete.json','statistics_input.json','statistics_provisional.json'] + [
    f'episode_{i:03}' for i in range(8,32)]


def require(condition, message):
  if not condition: raise ValueError(message)


def check():
  require(time.monotonic() < DEADLINE, 'Calibration failure audit deadline exceeded')


def inventory(root):
  files = []
  for path in sorted(root.rglob('*')):
    check()
    require(not path.is_symlink(), 'Symlink artifacts are not accepted')
    require(path.resolve().is_relative_to(root), 'Artifact escapes original run')
    if path.is_file():
      files.append(dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size,
          sha256=a.sha(path)))
    else: require(path.is_dir(), 'Nonregular run artifact')
  return sorted(files,key=lambda item:item['path'])


def no_held(root):
  require(all(not (root/name).exists() for name in MISSING),
      'Held directory or successful full-assay output exists')
  require(sorted(path.name for path in root.glob('episode_*')) ==
      [f'episode_{i:03}' for i in range(8)], 'Exactly calibration episode directories0..7 required')


def directories(root):
  return sorted(path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_dir())


def authenticate(run, capture_path, capture_sha, run_git, audit_git):
  root = Path(run).resolve(strict=True)
  require(root.is_dir() and run_git == RUN_GIT, 'Original run/Git differs')
  require(re.fullmatch('[0-9a-f]{40}',audit_git or ''), 'Full audit Git SHA required')
  capture_bytes=Path(capture_path).read_bytes()
  require(re.fullmatch('[0-9a-f]{64}',capture_sha or '') and hashlib.sha256(capture_bytes).hexdigest() == capture_sha,
      'External capture SHA differs')
  capture = s.decode(capture_bytes)
  require(capture['version'] == CAPTURE_VERSION and capture['schema'] == 1 and
      capture['run_git'] == run_git and capture['audit_git'] == audit_git and
      Path(capture['run_directory']).resolve(strict=True) == root and capture['stable'] is True and
      capture['producer_closure_claimed'] is False, 'Capture identity/stability differs')
  require(capture['scientific_audit_complete'] is False and capture['pickle_decoded'] is False and
      capture['run_files_written'] is False and capture['calibration_null_passed'] is False and
      capture['terminal_class'] == 'calibration_null_measurement_failed_held_unopened' and
      capture['final_error'] == STOP_MESSAGE, 'External capture cannot claim scientific success')
  require(capture['expected_missing'] == MISSING and capture['completed_episode_ids'] == list(range(8)) and
      capture['held_episode_directories'] == [], 'Capture must describe only stopped cal8')
  require(capture['before_inventory'] == capture['after_inventory'], 'Capture changed during collection')
  require(capture['before_directories'] == capture['after_directories'] == directories(root),
      'Original directory inventory differs from external capture')
  files = capture['after_inventory']
  require(type(files) is list and files and len({x['path'] for x in files}) == len(files),
      'Unique complete captured file inventory required')
  for item in files:
    require(set(item) == {'path','bytes','sha256'} and type(item['bytes']) is int and item['bytes'] >= 0,
        'Malformed captured file inventory')
    path = a.assert_hash(root,item['path'],item['sha256'])
    require(path.stat().st_size == item['bytes'], 'Captured artifact size differs')
  require(files == inventory(root), 'Original file inventory changed after capture')
  no_held(root)
  budget, process, terminal, seal = [a.read_json(root/name) for name in
      ('budget.json','process.json','supervisor_terminal.json','calibration_frozen.json')]
  require(budget['version'] == terminal['version'] == seal['version'] == a.SCIENCE and
      budget['git_sha'] == terminal['git_sha'] == run_git and budget['seconds'] == 21000,
      'Original producer science/budget differs')
  require(terminal['worker_returncode'] == 1 and terminal['worker_terminal'] is True and
      terminal['main_generation_complete'] is False and terminal['worker_complete_sha256'] is None and
      terminal['parent_failure'] is None and terminal['scientific_support_claimed'] is False and
      terminal['owned_worker_group_cleaned'] is True and
      0 < terminal['elapsed_seconds'] <= 21000, 'Not the original bounded calibration-stop terminal')
  a.assert_hash(root,'worker.log',terminal['worker_log_sha256'])
  a.assert_hash(root,'resources.jsonl',terminal['resources_sha256'])
  file_map = {x['path']:x for x in files}
  require(capture['terminal'] == file_map['supervisor_terminal.json'] and
      capture['worker_log'] == file_map['worker.log'], 'Capture terminal/log alias differs')
  require(process['parent_pid'] == budget['parent_pid'] and process['worker_pid'] == process['worker_pgid'],
      'Original owned process chain differs')
  for stage in ('before','after'):
    for name,pid in (('parent',process['parent_pid']),('worker',process['worker_pid'])):
      node = capture['process_observations'][stage][name]
      require(node['pid'] == pid and node['present'] is False and node['alive'] is False and
          node['identity_status'] == 'absent' and
          (pid,node['expected_starttime_ticks']) == IDENTITIES[name],
          'Capture did not establish terminal original processes')
      if os.name == 'posix':
        require(not Path(f'/proc/{pid}').exists(), 'Recorded PID currently exists; do not guess PID reuse')
  require(a.owned_output(['git','rev-parse','HEAD']).decode().strip() == audit_git and
      not a.owned_output(['git','status','--porcelain','--untracked-files=all']), 'Clean independent audit Git required')
  pins = budget['source_files']
  require(capture['sourcepins'] == pins, 'Captured producer source pins differ')
  original = a.owned_output(['git','ls-tree','-r','--name-only',run_git]).decode().splitlines()
  expected = {name for name in original if name.endswith(('.py','.yaml','.yml','.lock')) or
      name in ('AGENTS.md','docs/DL_R4_UTILITY.md')}
  require(set(pins) == expected, 'Whole original executable/config source manifest required')
  for name,digest in pins.items():
    blob = a.owned_output(['git','show',run_git+':'+name])
    require(hashlib.sha256(blob).hexdigest() == digest and (ROOT/name).read_bytes() == blob,
        'Original producer source/blob differs: '+name)
  model, runtime = [a.read_json(root/name) for name in ('model_authenticated.json','runtime_cuda_egl.json')]
  meta, resource = model['metadata'], model['resource']
  require(meta['checkpoint_sha256'] == budget['input_pins']['checkpoint_sha256'] == a.CHECKPOINT and
      meta['config_sha256'] == budget['input_pins']['config_sha256'] == a.CONFIG and
      meta['params_sha256'] == a.PARAMS and resource['physical_gpu'] == 4 and
      resource['logical_server'] == 'sv3' and resource['gpu_uuid'] == resource['compute_uuid'] == resource['egl_uuid'],
      'Original input/resource binding differs')
  require(runtime['pid'] == process['worker_pid'] and runtime['physical_gpu'] == 4 and
      runtime['compute_uuid'] == [resource['gpu_uuid']] and runtime['graphics_verified'] is True,
      'Original initialized CUDA/EGL evidence differs')
  require(runtime['live_state']['gpu_uuid'] == resource['gpu_uuid'] and all(
      node['pid'] == process['worker_pid'] for node in runtime['live_state']['processes']),
      'Original runtime contains unknown process or UUID')
  telemetry=[s.decode(line) for line in (root/'resources.jsonl').read_text(encoding='utf-8').splitlines()]
  require(telemetry and all(row['physical_gpu'] == 4 and row['gpu_uuid'] == resource['gpu_uuid'] and
      all(node['pid'] == process['worker_pid'] for node in row['processes']) for row in telemetry),
      'Recorded telemetry contains unknown process or UUID')
  events=[]
  log=(root/'worker.log').read_text(encoding='utf-8',errors='replace')
  require(log.rstrip().endswith(STOP_MESSAGE) and 'During handling of the above exception' not in log,
      'Final failure is not exclusively the frozen calibration-null gate')
  for line in log.splitlines():
    if line.startswith('{'):
      try:event=s.decode(line)
      except ValueError:continue
      if event.get('event') == 'episode_complete':events.append(event['episode_id'])
  require(events == list(range(7)), 'Expected ep0..6 events: ep7 gate precedes its print')
  require(seal['held_episodes_generated'] == 0 and seal['frozen_models']['null_passed'] is False,
      'Seal does not record calibration qualification failure')
  data=dict(version=a.SCIENCE,horizon=100,mc_pairs=16,calibration_episode_ids=list(range(8)),
      bootstrap_seed=20261028,episodes=seal['calibration_episodes'])
  s.validate(data,expected_episodes=8)  # Structure only; no fitting or held statistics.
  require(len(seal['calibration_evidence']) == 8,'Exactly8 sealed calibration manifests required')
  artifacts={}; previous=None
  for ep,entry in enumerate(seal['calibration_evidence']):
    relative=f'episode_{ep:03}/complete.json'
    require(entry['path'] == relative,'Calibration manifest order differs')
    receipt=a.read_json(a.assert_hash(root,relative,entry['sha256']))
    require(receipt['version'] == a.SCIENCE and receipt['episode_id'] == ep and
        receipt['model_full_state_unchanged'] is True,'Calibration completion identity differs')
    start,end=receipt['episode_started_monotonic'],receipt['episode_completed_monotonic']
    require(budget['monotonic_deadline']-21000 <= start <= end < budget['monotonic_deadline'] and
        (previous is None or previous <= start),'Calibration chronology differs')
    previous=end
    prefix=f'episode_{ep:03}/'
    for field in ('source','signals'):
      item=receipt[field];require(item['path'] == field+'.pkl','Source/signal file differs')
      name=prefix+item['path'];artifacts[name]=(a.assert_hash(root,name,item['sha256']),item['sha256'])
    _,selected=s.selected_positions(data['episodes'][ep]['candidates'])
    expected={(t,p,arm) for t in selected for p in range(16) for arm in a.ARMS} | {(t,0,'qq_null') for t in selected}
    seen=set()
    for item in receipt['branches']:
      identity=(item['position'],item['pair'],item['arm'])
      require(identity in expected and identity not in seen,'Incomplete/duplicate branch manifest')
      seen.add(identity);t,p,arm=identity
      require(item['path'] == f't{t:03}_pair{p:02}_{arm}.pkl','Branch filename differs')
      name=prefix+item['path'];artifacts[name]=(a.assert_hash(root,name,item['sha256']),item['sha256'])
    require(seen == expected,'Incomplete planned branch files')
    require([x['path'] for x in receipt['branch_public_oracles']] ==
        [f't{t:03}_branch_public_oracle.pkl' for t in selected],'Private public-oracle coverage differs')
    for item in receipt['branch_public_oracles']:
      name=prefix+item['path'];artifacts[name]=(a.assert_hash(root,name,item['sha256']),item['sha256'])
  require(previous <= seal['sealed_monotonic'] < budget['monotonic_deadline'],'Cal seal chronology differs')
  check();return root,capture,data,artifacts


def null_difference(qq,null,stored):
  qsum=math.fsum(float(x) for x in qq['reward_f64'])
  nsum=math.fsum(float(x) for x in null['reward_f64'])
  require(abs(qsum-stored['returns']['qq'][0]) <= 1e-12 and
      abs(nsum-stored['qq_null_return']) <= 1e-12,'Original null sums differ from independent fsum')
  delta=abs(qsum-nsum)/100
  original=abs(stored['returns']['qq'][0]-stored['qq_null_return'])/100
  require((delta > NULL_LIMIT) == (original > NULL_LIMIT),
      'Null threshold side differs under independent arithmetic; numerical-inconclusive')
  maximum=max(abs(float(x)-float(y)) for x,y in zip(qq['reward_f64'],null['reward_f64']))
  require(abs(maximum-stored['null_max_abs_step_reward_diff']) <= 1e-12,'Null step difference differs')
  return dict(absolute_average_difference=delta,threshold=NULL_LIMIT,failed=delta > NULL_LIMIT,
      maximum_step_reward_difference=maximum,original_absolute_average_difference=original)


def snapshot_equal(actual,expected):
  a.independent.compare_state(actual,expected)
  a.same(actual['suite']['integration'],expected['suite']['integration'],'complete initial integration')
  require(pickle.dumps(actual,protocol=5) == pickle.dumps(expected,protocol=5),
      'Complete initial snapshot serialized tree differs')


def stored_difference_order(qq,null):
  """Describe stored sequence order only; do not execute model/render or infer cause."""
  def equal(x,y):
    if isinstance(x,(np.ndarray,np.generic)) and isinstance(y,(np.ndarray,np.generic)):
      return x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
    return type(x) is type(y) and x == y
  def first(xs,ys):
    for index,(x,y) in enumerate(zip(xs,ys)):
      if not equal(x,y):return index
    return min(len(xs),len(ys)) if len(xs) != len(ys) else None
  images=[[obs['image'] for obs in rec['observations']] for rec in (qq,null)]
  image_step=first(*images)
  fields={name:first(qq[name],null[name]) for name in
      ('raw','applied','reward_f64','pre_carry_sha256','after_carry_sha256','action_executed')}
  fields['image_frame']=image_step
  fields['observation_frame']=first(*[[a.digest_tree(obs) for obs in rec['observations']] for rec in (qq,null)])
  fields['physics_integration_frame']=first(*[[state['suite']['integration']
      for state in rec['snapshots']] for rec in (qq,null)])
  for name in ('h','latent','selected_logit','action_latent','action_logit'):
    fields[name+'_action']=first(*[[proof[name] for proof in rec['proofs']] for rec in (qq,null)])
  pixel=None
  if image_step is not None and image_step < min(map(len,images)):
    dx=np.abs(images[0][image_step].astype(np.int16)-images[1][image_step].astype(np.int16))
    pixel=dict(frame=image_step,unequal_pixels=int(np.any(dx != 0,axis=-1).sum()),
        unequal_channel_elements=int(np.count_nonzero(dx)),max_abs_integer_difference=int(dx.max()))
  return dict(indexing='All 0-based: frame0=current source observation/physics; action0=first executed action; proof/carry indices are action indices.',
      first_byte_difference=fields,first_image_difference=pixel,
      executed_steps=[qq['executed_steps'],null['executed_steps']],
      interpretation='Stored sequences only, includes declared absorbing masks; not independent model/render replay or causal root identification.')


def audit(run,capture_path,capture_sha,run_git,audit_git):
  root,capture,data,artifacts=authenticate(run,capture_path,capture_sha,run_git,audit_git)
  from dm_control import suite
  rows=[];steps=0
  for ep,episode in enumerate(data['episodes']):
    prefix=f'episode_{ep:03}/'
    source=a.trusted_pickle(artifacts,prefix+'source.pkl')
    signals=a.trusted_pickle(artifacts,prefix+'signals.pkl')
    require(source['episode_id'] == ep and source['env_seed'] == 4096+ep,'Source namespace differs')
    a.compare_compact(signals['candidates'],episode['candidates'])
    top,selected=s.selected_positions(signals['candidates'])
    require(sorted(signals['top2']) == list(top) and signals['selected'] == list(selected),'D-only selection differs')
    records={r['position']:r for r in source['records']}
    env=suite.load('quadruped','walk',task_kwargs={'random':4096+ep})
    try:
      env.reset();a.static_visual(env,signals['visual_identity'])
      for stored in episode['selected']:
        t=stored['position'];pair={}
        for arm in ('qq','qq_null'):
          record=a.trusted_pickle(artifacts,prefix+f't{t:03}_pair00_{arm}.pkl')
          require(record['visual_identity'] == signals['visual_identity'] == record['visual_identity_after'],
              'Source/null static identity differs')
          require(a.carry(record['initial_carry']) == records[t]['carry_sha256'], 'Null/source history differs')
          a.same(record['observations'][0]['image'],source['arrays']['image'][t],'null current source RGB')
          require(a.policy_digest(record['observations'][0]) == a.policy_digest({key:np.asarray(source['arrays'][key][t]).copy()
              for key in ('image','reward','is_first','is_last','is_terminal')}),'Null/source full current observation differs')
          snapshot_equal(record['snapshots'][0],source['snapshots'][t])
          steps+=a.replay_record(env,record,episode=ep,position=t,pair=0,arm=arm)
          pair[arm]=record
        q,n=pair['qq'],pair['qq_null']
        require(q['current_bundle_sha256'] == n['current_bundle_sha256'] and
            a.carry(q['initial_carry']) == a.carry(n['initial_carry']), 'Null inputs/bundle differ')
        snapshot_equal(q['snapshots'][0],n['snapshots'][0])
        for field in ('keys','raw','applied'):
          a.same(q[field] if field == 'keys' else q[field][0],n[field] if field == 'keys' else n[field][0],'null shared '+field)
        require(q['after_carry_sha256'][0] == n['after_carry_sha256'][0], 'Null first model after carry differs')
        rows.append(dict(episode_id=ep,position=t,**null_difference(q,n,stored),
            stored_difference_order=stored_difference_order(q,n)))
      a.static_visual(env,signals['visual_identity'])
    finally:env.close()
    print(json.dumps(dict(event='cal_null_audit',episode_id=ep,physics_steps=steps)),flush=True)
  require(any(row['failed'] for row in rows),'Independent null audit did not reproduce the stopping gate')
  require(inventory(root) == capture['after_inventory'],'Original run bytes changed during audit')
  require(directories(root) == capture['after_directories'],'Original directories changed during audit')
  require(a.sha(capture_path) == capture_sha,'External capture changed during audit')
  no_held(root);check()
  return dict(version=VERSION,status='measurement-inconclusive',integrity_passed=True,audit_integrity_passed=True,
      qualification_passed=False,run_git=run_git,audit_git=audit_git,capture_sha256=capture_sha,
      calibrated_episodes=list(range(8)),planned_null_controls=len(rows),null_controls=rows,
      independent_physics_steps=steps,full_assay_complete=False,held_generated=False,held_evaluated=False,held_increment_evaluated=False,
      source_policy_reexecuted=False,source_physics_replayed=False,non_qq_or_later_pair_branches_decoded=False,
      model_executed=False,renderer_executed=False,jax_imported='jax' in sys.modules,
      full_final_worker_seal_present=False,independent_statistical_formula=False,
      original_attempt_successful=False,producer_closure_claimed=False,
      limitations=['External capture is post-stop evidence, not the absent producer final seal.',
          'Only source linkage and all planned pair0 qq/null stored-control physics are audited.',
          'No independent model/head/D/K/NJ subkeys, cached-policy, renderer, fitting or held analysis.',
          'Same official simulator and reused independent restore/snapshot primitives, not a second simulator.',
          'One repeated pair0 per state is not a renderer-distribution or complete MC-error bound.'],
      elapsed_seconds=time.monotonic()-STARTED,deadline_seconds=1800,
      training_authorized=False,method_effectiveness_established=False)


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  for name in ('run','capture','expected-capture-sha','expected-run-git','expected-audit-git','output'):
    parser.add_argument('--'+name,required=True)
  args=parser.parse_args();output=Path(args.output)
  require(os.name == 'posix' and not output.exists(),'POSIX hard deadline and fresh output required')
  result=audit(args.run,args.capture,args.expected_capture_sha,args.expected_run_git,args.expected_audit_git)
  require('jax' not in sys.modules,'CPU failure audit imported JAX')
  result['auditor_sha256']=a.sha(__file__)
  partial=output.with_name(output.name+'.partial');output.parent.mkdir(parents=True,exist_ok=True)
  check()
  with partial.open('x',encoding='utf-8') as stream:
    json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
  check();partial.rename(output);check()


if __name__ == '__main__':main()
