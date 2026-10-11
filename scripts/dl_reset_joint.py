"""DL-reset-joint-assay-r1: fixed sealed 0/B128/R128 diagnostic.

The stdlib CLI parent owns all worker groups and a 3600s whole-entry limit.
Only its authenticated workers may read scientific inputs. CPU physics is
isolated from JAX; each GPU phase freshly checks the assigned device.
"""
import time
ENTRY_STARTED = time.monotonic()
import os
import signal
import sys
OWNED = set()


def expire(signum, frame):
  for pid in tuple(OWNED):
    try: os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError: pass
  os._exit(124)


if __name__ == '__main__' and os.name == 'posix':
  signal.signal(signal.SIGALRM, expire)
  signal.setitimer(signal.ITIMER_REAL, max(.001, 3600-(time.monotonic()-ENTRY_STARTED)))

import argparse
import hashlib
import io
import json
from pathlib import Path
import pickle
import socket
import subprocess
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'DL-reset-joint-assay-r1'
SECONDS = 3600
CAL = (9, 14, 15, 18, 20, 24, 30, 31)
BLIND = tuple(i for i in range(32) if i not in CAL)
PHASES = ('baseline', 'B', 'R', 'freeze', 'truth', 'eval0', 'evalB', 'evalR', 'statistics')
FILES = ('scripts/dl_reset_joint.py', 'scripts/dl_reset_features.py',
    'scripts/dl_reset_learning.py', 'scripts/dl_reset_readout.py',
    'scripts/dl_resources.py', 'scripts/dl_predictive_mc.py',
    'scripts/dl_diagnostic.py', 'scripts/dl_feedback_model.py',
    'scripts/dl_nested_probe_stats.py', 'scripts/dl_natural_reset.py',
    'scripts/dl_action_information_physics.py', 'analysis/dl_task_semantics.py',
    'analysis/dl_feedback_physics_audit_v1.py', 'dreamerv3/agent.py',
    'dreamerv3/rssm.py', 'dreamerv3/dt_latch.py', 'dreamerv3/rep_probe.py',
    'embodied/jax/agent.py', 'docs/DL_RESET_JOINT_ASSAY.md')


def require(condition, message):
  if not condition: raise ValueError(message)


def sha(path):
  h = hashlib.sha256()
  with Path(path).open('rb') as f:
    for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
  return h.hexdigest()


def read_json(path):
  return json.loads(Path(path).read_text(encoding='utf-8'),
      parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))


def jsonify(value):
  if isinstance(value, dict): return {str(k):jsonify(v) for k,v in value.items()}
  if isinstance(value, (tuple, list)): return [jsonify(v) for v in value]
  if hasattr(value, 'tolist'): return jsonify(value.tolist())
  if isinstance(value, Path): return str(value)
  return value


def write_json(path, value):
  with Path(path).open('x', encoding='utf-8', newline='\n') as f:
    json.dump(jsonify(value), f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')


def save_pickle(path, value):
  with Path(path).open('xb') as f: pickle.dump(value, f, protocol=5)
  return sha(path)


def load_generated(path, expected):
  """Only newly generated, explicitly hash-bound project payloads."""
  payload = Path(path).read_bytes()
  require(hashlib.sha256(payload).hexdigest() == expected, 'Generated payload changed')
  return pickle.loads(payload)


def git_binding(expected):
  require(len(expected) == 40 and all(c in '0123456789abcdef' for c in expected), 'Full SHA required')
  require(owned_output(['git','rev-parse','HEAD']).decode().strip()
      == expected, 'Execution Git differs')
  require(not owned_output(['git','status','--porcelain']),
      'Clean execution checkout required')
  out = {}
  for name in FILES:
    blob = owned_output(['git','show',expected+':'+name])
    require((ROOT/name).read_bytes() == blob, 'Dependency differs: '+name)
    out[name] = hashlib.sha256(blob).hexdigest()
  return out


def cleanup(child):
  if os.name == 'posix' and getattr(child,'_dl_group_owned',True):
    try: os.killpg(child.pid,signal.SIGKILL)
    except ProcessLookupError: pass
  elif child.poll() is None: child.kill()
  child.wait(timeout=5);OWNED.discard(child.pid)


def owned_output(command):
  # Worker metadata helpers inherit its already-owned process group. A nested
  # new session would escape the root parent's timeout cleanup.
  own_group=os.name == 'posix' and 'DL_RESET_JOINT_PARENT' not in os.environ
  child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
      start_new_session=own_group)
  child._dl_group_owned=own_group
  if own_group: OWNED.add(child.pid)
  try:
    stdout,stderr=child.communicate(timeout=10)
    if child.returncode: raise subprocess.CalledProcessError(child.returncode,command,stdout,stderr)
    return stdout
  finally: cleanup(child)


def run_owned(command, log, deadline, env):
  require(time.monotonic() < deadline, 'Late before owned worker creation')
  child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
      start_new_session=os.name == 'posix')
  OWNED.add(child.pid);start=time.monotonic();timed_out=False
  try: code=child.wait(timeout=max(.001,deadline-time.monotonic()))
  except subprocess.TimeoutExpired: code=124;timed_out=True
  finally: cleanup(child)
  return dict(worker_pid=child.pid,exit_code=code,timed_out=timed_out,
      worker_terminal=child.poll() is not None,owned_group_cleanup_requested=os.name == 'posix',
      elapsed_seconds=time.monotonic()-start)


def authenticated_phase(args):
  output = Path(args.output).resolve(strict=True)
  budget = read_json(output/'budget.json')
  require(budget['version'] == VERSION and budget['actual_git_sha'] == args.expected_git and
      budget['parent_pid'] == os.getppid() == int(os.environ.get('DL_RESET_JOINT_PARENT','-1')) and
      budget['seconds'] == SECONDS and budget['script_sha256'] == sha(__file__),
      'Authenticated live fixed parent required')
  os.kill(os.getppid(), 0)
  if args.worker_phase in ('truth','eval0','evalB','evalR','statistics'):
    require(isinstance(args.seal,str) and len(args.seal) == 64 and
        sha(output/'freeze.json') == args.seal, 'Parent sealed preblind digest differs')
  deadline = budget['monotonic_start'] + SECONDS
  def check(): require(time.monotonic() < deadline, 'Whole-entry budget exhausted')
  check(); git_binding(args.expected_git)
  return output, budget, check


def load_cal(args, check):
  import numpy as np
  from scripts import dl_natural_reset as n
  selected, binding = n.source_authority(args.dataset, args.artifact_audit, args.physics_audit, check)
  arrays = {}
  for row in selected:
    ep = row['definition']['episode']; item = row['artifact']
    payload = Path(binding['all_32_artifact_hashes'][ep]['path']).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == item['sha256'], 'Cal RGB source changed')
    with np.load(io.BytesIO(payload), allow_pickle=False) as f:
      image = f['image'].copy()
    require(image.dtype == np.uint8 and image.shape == (1001,64,64,3), 'Cal image contract differs')
    arrays[ep] = {**{k:row['data'][k] for k in
        ('reward','is_first','is_last','is_terminal','action','action_executed')}, 'image':image}
  binding.update(image_arrays_read=True, cal_image_frames_decoded=8008,
      before_carries_read=False, blind_arrays_read=False)
  return arrays, binding


def cal_returns(args):
  from scripts import dl_natural_reset as n
  path = Path(args.cal_reset).resolve(strict=True)
  require(sha(path/'result.json') == '9a11da123f9743f3815f6a36bf3da26641900217708990830ef15c092d49c8be' and
      sha(path/'natural_reset_complete.json') == '678451d9a2b94807016e1be9bd84469d74941a5ba1d68b3df9740e4007b4f774',
      'Pinned cal physics result/completion differs')
  result = read_json(path/'result.json')
  complete = read_json(path/'natural_reset_complete.json')
  supervisor = read_json(path/'supervisor_complete.json')
  terminal_path=path.parent/'engineering/primary-terminal01.json'
  require(sha(terminal_path) == '09339ceea9ef507b172f020a57058fe7ce99b616ff9b4c3ab6ad01ad36ed38da',
      'Pinned cal CLI external terminal differs')
  terminal=read_json(terminal_path)
  audit_path = path.parent/'independent-audit01/audit.json'
  require(sha(audit_path) == 'c76a6473617ecf10824efc0bcd8eebb4042075354cfcf74e4e0d65546d3ae7f0',
      'Pinned independent cal audit differs')
  audit=read_json(audit_path)
  require(audit['status'] == 'passed' and not (path/'late_supervisor_receipt.json').exists(),
      'Cal late receipt/audit failure')
  require(set(audit['primary_json_sha256']) == {'budget.json','plan.json','result.json',
      'natural_reset_complete.json','supervisor_complete.json'}, 'Cal primary chain incomplete')
  for name,pin in audit['primary_json_sha256'].items():
    require(sha(path/name) == pin, 'Audited cal original completion chain differs: '+name)
  require(supervisor['exit_code'] == 0 and supervisor['complete_verified'] is True and
      supervisor['timed_out'] is False and supervisor['worker_terminal'] is True and
      sha(path/'budget.json') == supervisor['budget_sha256'] == complete['budget_sha256'] and
      complete['actual_git_sha'] == terminal['actual_git_sha'] ==
          'aa9fb7f777a28e8a82d57c31f9e749b5912af316' and terminal['exit_code'] == 0,
      'Cal physics exit chain differs')
  import numpy as np
  rows = []
  for ep,item in zip(CAL,result['evidence_artifacts']):
    require(item['episode'] == ep and item['path'] == f'episode_{ep:03}.npz' and
        sha(path/item['path']) == item['sha256'], 'Cal cost artifact differs')
    payload=(path/item['path']).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == item['sha256'], 'Cal cost changed before decode')
    with np.load(io.BytesIO(payload), allow_pickle=False) as f:
      reward = f['reward_f64']
    require(reward.shape == (7,20) and reward.dtype == np.float64 and np.isfinite(reward).all(),
        'Cal F64 complete reward contract differs')
    rows.append(reward.sum(-1,dtype=np.float64))
  return np.stack(rows)


def observations(arrays, episodes):
  return [{k:arrays[e][k][0].copy() for k in
      ('image','reward','is_first','is_last','is_terminal')} for e in episodes]


def numeric_bundle(feature, episodes, returns):
  import numpy as np
  return dict(episode_ids=np.asarray(episodes,np.int32), returns=np.asarray(returns,np.float64),
      **{k:np.asarray(feature[k],np.float64) for k in ('h','zq','zp','D','K')})


def gpu_phase(args, output, check):
  import numpy as np
  from scripts import dl_resources as resource
  from scripts import dl_predictive_mc as pinned
  from scripts import dl_reset_features as features
  from scripts import dl_reset_learning as learning
  device = resource.require_resource(args.resource_record)
  require(device['physical_gpu'] == 4, 'First-choice fixed physical GPU4 required')
  phase = args.worker_phase
  if phase.startswith('eval'):
    frozen = verify_freeze(output)
    truth = read_json(output/'truth-complete.json')
    require(truth['freeze_sha256'] == sha(output/'freeze.json'), 'Truth predates freeze')
    saved = load_generated(output/'blind-observations.pkl',truth['observations_sha256'])
    require(tuple(saved) == BLIND, 'All blind reset observations required')
    agent, config, metadata = pinned.load_agent_no_env(args)
    group = phase[4:]
    if group != '0':
      agent.load(load_generated(output/f'{group}-checkpoint.pkl', frozen['files'][f'{group}-checkpoint.pkl']))
    from embodied.run.protocol_v1 import parameter_digest
    metadata.update(current_evaluation_group=group,
        current_params_sha256=parameter_digest(agent),baseline_loader_metadata=True,
        current_full_state_sha256=learning.tree_digest(agent.save()),
        current_group_checkpoint_sha256=(pinned.CHECKPOINT_SHA if group == '0' else
            frozen['files'][f'{group}-checkpoint.pkl']))
    if group != '0':
      completed = read_json(output/f'{group}-complete.json')['learning']
      require(metadata['current_full_state_sha256'] == completed['final_state_digest'],
          'Reloaded full evaluation state differs from completed learning state')
    device['runtime_compute_start'] = resource.verify_runtime(device, compute=True)
    cal_group = load_generated(output/f'{group}-cal.pkl',frozen['files'][f'{group}-cal.pkl'])
    adapter = features.ResetFeatureAdapter(agent,common_reference=cal_group['aux']['common_reference'])
    extracted = adapter.extract([saved[e] for e in BLIND],check=check)
    feature_sha = save_pickle(output/f'{group}-blind.pkl', extracted)
    device['runtime_compute_end'] = resource.verify_runtime(device,compute=True)
    write_json(output/f'{phase}-complete.json', dict(version=VERSION, phase=phase,
        actual_git_sha=args.expected_git, episodes=list(BLIND), feature_sha256=feature_sha,
        freeze_sha256=sha(output/'freeze.json'), device=device, model_metadata=metadata))
    return
  arrays, binding = load_cal(args,check)
  checkpoint_payload = Path(pinned.checkpoint_path(args.checkpoint)).read_bytes()
  require(hashlib.sha256(checkpoint_payload).hexdigest() == pinned.CHECKPOINT_SHA,
      'Checkpoint verification before unpickle required')
  checkpoint = pickle.loads(checkpoint_payload)
  if phase == 'baseline':
    agent, config, metadata = pinned.load_agent_no_env(args)
    device['runtime_compute_start'] = resource.verify_runtime(device,compute=True)
    adapter = features.ResetFeatureAdapter(agent)
    extracted = adapter.extract(observations(arrays,CAL),check=check)
    feature_sha = save_pickle(output/'0-cal.pkl', extracted)
    contexts, context_proofs = {}, {}
    for ep in CAL:
      check()
      key = np.random.default_rng([20261026,0x43545831,ep]).integers(0,2**32,(2,),dtype=np.uint32)
      annotated = adapter.annotate_episode_terminal([arrays[ep]],key,check=check)
      contexts[ep] = {k:v[0].astype(np.float32) for k,v in annotated['terminal_entry'].items()}
      for name,value in annotated['terminal_entry'].items():
        roundtrip=contexts[ep][name].astype(value.dtype)
        require(roundtrip.shape == value[0].shape and roundtrip.tobytes() == value[0].tobytes(),
            'Replay context F32 promotion did not roundtrip to actual BF16 bytes')
      context_proofs[ep] = annotated['aux']
      context_proofs[ep]['replay_context_promoted_to_float32_exact_roundtrip']=True
    context_sha = save_pickle(output/'contexts.pkl',contexts)
    device['runtime_compute_end'] = resource.verify_runtime(device,compute=True)
    write_json(output/'baseline-complete.json',dict(version=VERSION,phase=phase,
        actual_git_sha=args.expected_git, feature_sha256=feature_sha, context_sha256=context_sha,
        context_proofs=context_proofs, source_binding=binding, device=device, model_metadata=metadata,
        cal_image_frames_decoded=8008, blind_arrays_read=False))
    return
  require(phase in ('B','R'), 'Unknown GPU phase')
  baseline = read_json(output/'baseline-complete.json')
  contexts = load_generated(output/'contexts.pkl',baseline['context_sha256'])
  hashes = {e:binding['all_32_artifact_hashes'][e]['sha256'] for e in CAL}
  with learning.rssm_branch(phase):
    from scripts import dl_diagnostic as d
    from embodied.run.protocol_v1 import parameter_digest
    require(sha(args.config) == pinned.CONFIG_SHA, 'Config byte verification required')
    config = d.load_config(args.config,args.task,args.output,args.platform).update({
        'jax.enable_policy':False})
    agent = pinned.agent_from_spaces(config)
    agent.load(checkpoint)
    metadata = dict(config=config.flat,checkpoint_sha256=pinned.CHECKPOINT_SHA,
        config_source_sha256=pinned.CONFIG_SHA,params_sha256=parameter_digest(agent),
        checkpoint_counters=checkpoint['counters'],offline_policy_sync_disabled=True)
    require(metadata['params_sha256'] == pinned.PARAMS_SHA, 'Initial model parameter identity differs')
    device['runtime_compute_start'] = resource.verify_runtime(device,compute=True)
    def extract(current, stage):
      check()
      return features.ResetFeatureAdapter(current).extract(observations(arrays,CAL),check=check)
    outcome = learning.run_branch(agent,checkpoint,arrays,hashes,contexts,
        group=phase,reset_feature_callback=extract)
  check()
  checkpoint_sha = save_pickle(output/f'{phase}-checkpoint.pkl',outcome.pop('final_checkpoint'))
  extracted = outcome.pop('features')
  initial_sha = save_pickle(output/f'{phase}-initial-cal.pkl',extracted[0])
  feature_sha = save_pickle(output/f'{phase}-cal.pkl',extracted[128])
  device['runtime_compute_end'] = resource.verify_runtime(device,compute=True)
  write_json(output/f'{phase}-complete.json',dict(version=VERSION,phase=phase,
      actual_git_sha=args.expected_git,feature_sha256=feature_sha,initial_feature_sha256=initial_sha,
      checkpoint_sha256=checkpoint_sha, learning=outcome,source_binding=binding,
      context_sha256=baseline['context_sha256'],device=device,model_metadata=metadata,
      cal_image_frames_decoded=8008,blind_arrays_read=False))


def verify_freeze(output):
  frozen = read_json(output/'freeze.json')
  seal_path=Path(output)/'freeze-seal-parent.json'
  if seal_path.exists():
    require(read_json(seal_path)['freeze_sha256'] == sha(Path(output)/'freeze.json'),
        'Parent independently sealed preblind digest differs')
  require(frozen['version'] == VERSION and frozen['blind_arrays_read'] is False and
      frozen['cal_episodes'] == list(CAL) and frozen['blind_episodes'] == list(BLIND) and
      frozen['groups'] == ['0','B128','R128'] and frozen['updates_per_arm'] == 128 and
      frozen['statistics_seed'] == 20261026 and len(frozen['files']) >= 8, 'Incomplete sealed freeze')
  for name, pin in frozen['files'].items():
    path = (Path(output)/name).resolve(strict=True)
    require(path.parent == Path(output).resolve() and sha(path) == pin, 'Frozen file changed: '+name)
  require(git_binding(frozen['actual_git_sha']) == frozen['source_files_sha256'], 'Frozen code changed')
  return frozen


def freeze_phase(args, output, check):
  import ml_dtypes
  import numpy as np
  from scripts import dl_reset_readout as stats
  from scripts import dl_reset_learning as learning
  truth = cal_returns(args)
  bundles, fits = {}, {}
  names = ['0-cal.pkl','contexts.pkl','B-checkpoint.pkl','R-checkpoint.pkl',
      'B-cal.pkl','R-cal.pkl','baseline-complete.json','B-complete.json','R-complete.json','budget.json']
  for group in ('0','B','R'):
    name = 'baseline' if group == '0' else group
    receipt = read_json(output/f'{name}-complete.json')
    require(receipt['version'] == VERSION and receipt['blind_arrays_read'] is False,
        'Cal phase identity differs')
    bundles[group] = numeric_bundle(load_generated(output/f'{group}-cal.pkl',
        receipt['feature_sha256']), CAL,truth)
  b,r = [read_json(output/f'{g}-complete.json')['learning'] for g in ('B','R')]
  require(b['initial_state_digest'] == r['initial_state_digest'] and
      b['binding'] == r['binding'] and [x['train_seed'] for x in b['receipts']] ==
      [x['train_seed'] for x in r['receipts']] and b['updates'] == r['updates'] == 128,
      'Paired initial state/batch/keys/update identity differs')
  baseline_native=load_generated(output/'0-cal.pkl',read_json(output/'baseline-complete.json')['feature_sha256'])
  for group,receipt in (('B',b),('R',r)):
    record=read_json(output/f'{group}-complete.json')
    actual_state=load_generated(output/f'{group}-checkpoint.pkl',record['checkpoint_sha256'])
    require(learning.tree_digest(actual_state) == receipt['final_state_digest'],
        'Saved full state differs from actual final learning state')
    initial=load_generated(output/f'{group}-initial-cal.pkl',record['initial_feature_sha256'])
    for name in ('h','zq','zp','D','K'):
      a,c=np.ascontiguousarray(initial[name]),np.ascontiguousarray(baseline_native[name])
      require(a.dtype == c.dtype and a.shape == c.shape and a.tobytes() == c.tobytes(),
          'Initial B/R/cal0 feature byte identity differs: '+name)
    require(np.array_equal(initial['aux']['root_key'],baseline_native['aux']['root_key']),
        'Initial model roots differ')
  frozen0 = stats.fit_calibration(bundles['0'])
  refits = {g:stats.refit_calibration(bundles[g],lambda_value=frozen0['selected_lambda'])
      for g in ('B','R')}
  save_pickle(output/'readouts.pkl',dict(frozen=frozen0,refits=refits,cal=bundles))
  names.append('readouts.pkl')
  write_json(output/'freeze.json',dict(version=VERSION,actual_git_sha=args.expected_git,
      source_files_sha256=git_binding(args.expected_git),cal_episodes=list(CAL),blind_episodes=list(BLIND),
      groups=['0','B128','R128'],updates_per_arm=128,statistics_seed=20261026,
      selection='four-prediction-mean then exact argmax, lowest exact tie ID',
      files={name:sha(output/name) for name in names},blind_arrays_read=False,
      lambda_value=frozen0['selected_lambda'],monotonic_freeze=time.monotonic(),
      method_effectiveness_established=False,continuous_actor_supported=False))
  verify_freeze(output);check()


def truth_phase(args, output, check):
  import numpy as np
  from scripts import dl_natural_reset as n
  from scripts import dl_action_information_physics as physical
  from analysis import dl_task_semantics as semantics
  frozen = verify_freeze(output)
  selected,binding = n.source_authority(args.dataset,args.artifact_audit,args.physics_audit,check)
  audit,dependency = semantics.load_auditor(ROOT/'analysis/dl_feedback_physics_audit_v1.py')
  suite,_,_,official,packages = semantics.official_modules()
  plan = read_json(Path(args.dataset)/'collection_plan.json')
  complete = read_json(Path(args.dataset)/'collection_complete.json')
  strict = SimpleNamespace(**{k:getattr(audit,k) for k in
      ('validate_snapshot','array','byte_equal','restore','snapshot')},
      compare_state=lambda a,b:n.compare_exact(audit,a,b))
  receipts,observed,returns = [],{},[]
  for ep in BLIND:
    check(); item = complete['artifacts'][ep]
    payload = Path(binding['all_32_artifact_hashes'][ep]['path']).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == item['sha256'], 'Blind source changed before decode')
    with np.load(io.BytesIO(payload),allow_pickle=False) as f:
      data = {k:f[k].copy() for k in n.FIELDS}
      image = f['image'].copy()
    n.validate_source_arrays(data)
    require(image.dtype == np.uint8 and image.shape == (1001,64,64,3), 'Blind cached RGB differs')
    snapshots = pickle.loads(data.pop('snapshots').tobytes())
    require(len(snapshots) == 1001, 'Blind source snapshot coverage differs')
    observed[ep] = {k:(image[0].copy() if k == 'image' else data[k][0].copy())
        for k in ('image','reward','is_first','is_last','is_terminal')}
    source = dict(definition=plan['episodes'][ep],artifact=item,data=data,snapshots=snapshots)
    env = suite.load('quadruped','walk',task_kwargs={'random':source['definition']['seed']})
    records,replays = [],[]
    try:
      ref = n.reference_replay(env,source,audit,physical,check)
      for index,raw in enumerate(physical.controls(env)):
        record = physical.run_sequence(env,snapshots[0],raw,audit,check)
        record.update(episode=ep,position=0,control=index)
        n.complete_record(env,record,audit,physical)
        replays.append(physical.replay_sequence(env,record,strict,check))
        records.append(record)
    finally: env.close()
    name = f'blind-physics-{ep:03}.pkl'
    pin = save_pickle(output/name,records)
    returns.append(np.stack([x['reward_f64'] for x in records]).sum(-1,dtype=np.float64))
    receipts.append(dict(episode=ep,path=name,sha256=pin,reference=ref,replays=replays))
  labels = save_pickle(output/'blind-returns.pkl',np.stack(returns))
  images = save_pickle(output/'blind-observations.pkl',observed)
  n.verify_source_unchanged(binding,check);verify_freeze(output)
  write_json(output/'truth-complete.json',dict(version=VERSION,actual_git_sha=args.expected_git,
      episodes=list(BLIND),freeze_sha256=sha(output/'freeze.json'),labels_sha256=labels,
      observations_sha256=images,source_binding=binding,source_files_sha256=git_binding(args.expected_git),
      official_modules=official,packages=packages,derived_physics_auditor=dependency,
      reference_physics_steps=480,main_physics_steps=3360,replay_physics_steps=3360,
      total_physics_steps=7200,new_reset_calls=24,receipts=receipts,
      blind_image_frames_decoded=24024,blind_scientific_positions=[0],before_carries_read=False,
      new_renderer_calls=0,physics_execution='CPU official same simulator'))


def statistics_phase(args, output, check):
  import ml_dtypes
  from scripts import dl_reset_readout as stats
  frozen = verify_freeze(output)
  readouts = load_generated(output/'readouts.pkl',frozen['files']['readouts.pkl'])
  truth = read_json(output/'truth-complete.json')
  returns = load_generated(output/'blind-returns.pkl',truth['labels_sha256'])
  blind = {}
  for group in ('0','B','R'):
    receipt = read_json(output/f'eval{group}-complete.json')
    require(receipt['freeze_sha256'] == sha(output/'freeze.json'), 'Model evaluation predates freeze')
    feature = load_generated(output/f'{group}-blind.pkl',receipt['feature_sha256'])
    blind[group] = numeric_bundle(feature,BLIND,returns)
  result = stats.evaluate_joint(readouts['frozen'],readouts['refits'],blind)
  numerical_sha=save_pickle(output/'blind-numeric.pkl',blind)
  from scripts import dl_natural_reset as n
  n.verify_source_unchanged(truth['source_binding'],check)
  write_json(output/'result.json',dict(version=VERSION,actual_git_sha=args.expected_git,
      analysis=result,freeze_sha256=sha(output/'freeze.json'),truth_sha256=sha(output/'truth-complete.json'),
      blind_numeric_sha256=numerical_sha,source_inputs_unchanged_at_final_statistics=True,
      method_effectiveness_established=False,continuous_actor_supported=False,
      full_goal_complete=False,decision='stop_after_fixed_reset_joint_assay'))
  check()


def worker(args):
  sys.path.insert(0,str(ROOT))
  output,budget,check = authenticated_phase(args)
  functions = {'freeze':freeze_phase,'truth':truth_phase,'statistics':statistics_phase}
  if args.worker_phase in functions: functions[args.worker_phase](args,output,check)
  else: gpu_phase(args,output,check)
  check();git_binding(args.expected_git)


def parent(args):
  require(os.name == 'posix' and socket.gethostname() == 'lyg0326', 'Execution only on assigned sv3')
  sources = git_binding(args.expected_git)
  output = Path(args.output).resolve()
  require(not output.exists(), 'New attempt only');output.mkdir(parents=True)
  deadline = ENTRY_STARTED + SECONDS
  write_json(output/'budget.json',dict(version=VERSION,actual_git_sha=args.expected_git,
      parent_pid=os.getpid(),seconds=SECONDS,monotonic_start=ENTRY_STARTED,
      script_sha256=sha(__file__),planned_physical_gpu=4,max_gpu_devices=1,
      max_gpu_hours=1,source_files_sha256=sources))
  device = read_json(args.resource_record)
  phases=[];code=0;timed_out=False;seal=None
  for phase in PHASES:
    is_gpu = phase in ('baseline','B','R','eval0','evalB','evalR')
    env = dict(os.environ,DL_RESET_JOINT_PARENT=str(os.getpid()),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    resource_path = args.resource_record
    if is_gpu:
      # Fresh live occupancy and retained real CUDA/EGL mapping, not a refreshed timestamp.
      sys.path.insert(0,str(ROOT))
      from scripts import dl_resources
      old_command=dl_resources.command_output
      dl_resources.command_output=lambda command:owned_output(command).decode()
      try: fresh = dl_resources.refresh(device)
      finally: dl_resources.command_output=old_command
      resource_path = str(output/f'{phase}-resource.json');write_json(resource_path,fresh)
    else: env.update(CUDA_VISIBLE_DEVICES='',MUJOCO_GL='disable',JAX_PLATFORMS='cpu')
    command = [sys.executable,str(Path(__file__).resolve()),'--worker-phase',phase,
        '--resource-record',resource_path]
    if seal is not None: command += ['--seal',seal]
    for key in ('dataset','artifact_audit','physics_audit','cal_reset','checkpoint','config',
        'output','expected_git','task','platform'):
      command += ['--'+key.replace('_','-'),str(getattr(args,key))]
    with (output/f'{phase}.log').open('x',encoding='utf-8') as log:
      terminal=run_owned(command,log,deadline,env)
    code=terminal['exit_code'];timed_out=terminal['timed_out']
    phases.append(dict(phase=phase,is_gpu=is_gpu,**terminal))
    write_json(output/f'{phase}-exit.json',phases[-1]);print(json.dumps(phases[-1]),flush=True)
    if code or time.monotonic() >= deadline: break
    if phase == 'freeze':
      seal=sha(output/'freeze.json')
      write_json(output/'freeze-seal-parent.json',dict(version=VERSION,actual_git_sha=args.expected_git,
          freeze_sha256=seal,monotonic_sealed=time.monotonic(),blind_started=False))
  complete = code == 0 and len(phases) == len(PHASES) and time.monotonic() < deadline
  if complete:
    verify_freeze(output);require((output/'result.json').is_file(), 'Missing complete result')
    require(git_binding(args.expected_git) == sources, 'Final source binding differs')
  receipt = dict(version=VERSION,actual_git_sha=args.expected_git,parent_pid=os.getpid(),phases=phases,
      exit_code=code if not timed_out else 124,timed_out=timed_out,complete_verified=complete,
      elapsed_parent_entry_seconds=time.monotonic()-ENTRY_STARTED,budget_sha256=sha(output/'budget.json'),
      actual_gpu_devices_used=1 if complete else None,method_effectiveness_established=False,
      independent_audit_complete=False,result_sha256=sha(output/'result.json') if complete else None)
  pending=output/'supervisor-receipt-pending.json'
  write_json(pending,receipt)
  if time.monotonic() >= deadline:
    write_json(output/'late-supervisor.json',dict(exit_code=124,incomplete=True));return 124
  os.replace(pending,output/'supervisor_complete.json')
  if time.monotonic() >= deadline:
    os.replace(output/'supervisor_complete.json',output/'late-provisional-supervisor.json')
    write_json(output/'late-supervisor.json',dict(exit_code=124,incomplete=True));return 124
  return receipt['exit_code'] if complete or receipt['exit_code'] else 1


def main():
  p=argparse.ArgumentParser(description=__doc__)
  for name in ('dataset','artifact-audit','physics-audit','cal-reset','checkpoint','config',
      'output','expected-git','resource-record'): p.add_argument('--'+name,required=True)
  p.add_argument('--task',default='dmc_quadruped_walk');p.add_argument('--platform',default='cuda')
  p.add_argument('--worker-phase',choices=PHASES)
  p.add_argument('--seal')
  args=p.parse_args()
  if args.worker_phase: worker(args);return 0
  return parent(args)


if __name__ == '__main__': sys.exit(main())
