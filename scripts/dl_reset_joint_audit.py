"""Bounded independent CPU audit of the sealed DL reset joint diagnostic.

The stdlib parent owns a 300s whole-entry deadline. The worker authenticates
receipts and raw inputs, independently recalculates the numeric assay and
replays 3360 stored-applied physics steps. It never decodes original source
archives, RGB packets or model checkpoints, or repeats optimizer updates.
"""
import time
ENTRY_STARTED = time.monotonic()
import os
import signal
import sys
OWNED = set()
SECONDS = 300


def expire(signum, frame):
  for pid in tuple(OWNED):
    try: os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError: pass
  os._exit(124)


if __name__ == '__main__' and os.name == 'posix':
  signal.signal(signal.SIGALRM, expire)
  signal.setitimer(signal.ITIMER_REAL, max(.001, SECONDS-(time.monotonic()-ENTRY_STARTED)))

import argparse
import copy
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import pickle
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'DL-reset-joint-assay-r1'
IMPLEMENTATION = 'DL-reset-joint-audit-r1'
CAL = (9,14,15,18,20,24,30,31)
BLIND = tuple(i for i in range(32) if i not in CAL)
PHASES = ('baseline','B','R','freeze','truth','eval0','evalB','evalR','statistics')
FILES = ('scripts/dl_reset_joint.py','scripts/dl_reset_features.py',
    'scripts/dl_reset_learning.py','scripts/dl_reset_readout.py',
    'scripts/dl_resources.py','scripts/dl_predictive_mc.py',
    'scripts/dl_diagnostic.py','scripts/dl_feedback_model.py',
    'scripts/dl_nested_probe_stats.py','scripts/dl_natural_reset.py',
    'scripts/dl_action_information_physics.py','analysis/dl_task_semantics.py',
    'analysis/dl_feedback_physics_audit_v1.py','dreamerv3/agent.py',
    'dreamerv3/rssm.py','dreamerv3/dt_latch.py','dreamerv3/rep_probe.py',
    'embodied/jax/agent.py','docs/DL_RESET_JOINT_ASSAY.md')
AUDIT_FILES = ('scripts/dl_reset_joint_audit.py','tests/test_dl_reset_joint_audit.py',
    'analysis/dl_reset_joint_numeric_audit.py','tests/test_dl_reset_joint_numeric_audit.py',
    'analysis/dl_natural_reset_audit.py')
FREEZE_FILES = ('0-cal.pkl','contexts.pkl','B-checkpoint.pkl','R-checkpoint.pkl',
    'B-cal.pkl','R-cal.pkl','baseline-complete.json','B-complete.json',
    'R-complete.json','budget.json','readouts.pkl')
SOURCE_PINS = dict(
    plan='633efe28d551d5793ac8f0ed5f0e36e937e7f1af3c8edefc1d9f5fa278146fcc',
    complete='d155f577d19d2bbc051ec56b95f15e26afc774c130c3c9728a7fe599f295f3cf',
    artifact_audit='8f93788ca212118139fab135036fd47ccbad79f04a34139b5dfa7d2dd26f7516',
    physics_audit='8ed3528fae18dc7a3043c767da87f873642a40a1a028ce86247f0521c3264722')
SOURCE_GIT = 'f2d24da1831da2ffca22d313e7f534c96fc0d3eb'
MODEL_PINS = dict(
    checkpoint_sha256='2ab693c8ded4d8f2186bd489aa8b77498448ba2f2c331090159da4e5f08056b9',
    params_sha256='14859c4e129b51aa488c10eab09f16d56fd110c76239b2c9072f415829d111c2',
    config_source_sha256='11428512fdc9b9e04a5384e419c5abcef93e5429db0a3327d05ccb0d4bd20445')
CAL_ROOT = Path('/data/Policy_Discrepancy/runs/dl-natural-reset-20261012-aa9fb7f/calibration-attempt01')
CAL_PINS = {'result.json':'9a11da123f9743f3815f6a36bf3da26641900217708990830ef15c092d49c8be',
    'natural_reset_complete.json':'678451d9a2b94807016e1be9bd84469d74941a5ba1d68b3df9740e4007b4f774',
    'audit':'c76a6473617ecf10824efc0bcd8eebb4042075354cfcf74e4e0d65546d3ae7f0',
    'external':'09339ceea9ef507b172f020a57058fe7ce99b616ff9b4c3ab6ad01ad36ed38da'}


def require(condition, message):
  if not condition: raise ValueError(message)


def check_deadline():
  require(time.monotonic() < ENTRY_STARTED + SECONDS, 'Whole-entry CPU audit 300s exhausted')


def sha(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024*1024), b''):
      check_deadline(); digest.update(chunk)
  return digest.hexdigest()


def read_json(path):
  def pairs(items):
    result = {}
    for key,value in items:
      require(key not in result, 'Duplicate JSON key: '+key);result[key]=value
    return result
  def nonfinite(value): raise ValueError('Nonfinite JSON: '+value)
  return json.loads(Path(path).read_text(encoding='utf-8'),
      object_pairs_hook=pairs,parse_constant=nonfinite)


def write_json(path,value):
  with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
    json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def safe_file(root,name):
  root=Path(root).resolve(strict=True)
  require(type(name) is str and Path(name).name == name and not Path(name).is_absolute(),
      'Unsafe generated filename')
  path=(root/name).resolve(strict=True)
  require(path.parent == root and path.is_file(), 'Artifact escapes its directory')
  return path


def hash_payload(path,pin):
  require(type(pin) is str and re.fullmatch('[0-9a-f]{64}',pin), 'Invalid byte SHA256')
  check_deadline();payload=Path(path).read_bytes();check_deadline()
  require(hashlib.sha256(payload).hexdigest() == pin, 'Payload hash differs before decode')
  return payload


def load_new(root,name,pin):
  """Only newly generated, hash-authenticated project pickle bytes."""
  return pickle.loads(hash_payload(safe_file(root,name),pin))


def kill_owned(child):
  if os.name == 'posix':
    try: os.killpg(child.pid,signal.SIGKILL)
    except ProcessLookupError: pass
  elif child.poll() is None: child.kill()
  try: child.wait(timeout=1)
  except subprocess.TimeoutExpired: raise RuntimeError('Owned worker did not terminate')
  OWNED.discard(child.pid)


def git_output(arguments):
  # Inherit the worker group; the stdlib parent owns this helper too.
  check_deadline()
  with subprocess.Popen(['git',*arguments],cwd=ROOT,stdout=subprocess.PIPE,
      stderr=subprocess.PIPE) as child:
    try: out,err=child.communicate(timeout=max(.001,ENTRY_STARTED+SECONDS-time.monotonic()))
    except subprocess.TimeoutExpired:
      child.kill();child.wait();raise
  require(child.returncode == 0,'Git helper failed: '+err.decode(errors='replace')[:256])
  check_deadline();return out


def allowed_diff(name):
  return name in AUDIT_FILES or (name.startswith('docs/') and name.endswith('.md'))


def git_binding(expected,primary_git):
  for value in (expected,primary_git):
    require(type(value) is str and re.fullmatch('[0-9a-f]{40}',value),'Full Git SHA required')
  require(git_output(['rev-parse','HEAD']).decode().strip() == expected,'Auditor HEAD differs')
  require(git_output(['status','--porcelain','--untracked-files=all']) == b'', 'Auditor Git must be clean')
  changed=git_output(['diff','--name-only',primary_git,expected]).decode().splitlines()
  require(all(allowed_diff(name) for name in changed),'Producer/auditor scientific code differs')
  pins={}
  for name in (*FILES,*AUDIT_FILES):
    current=git_output(['show',expected+':'+name])
    require(hashlib.sha256(current).hexdigest() == sha(ROOT/name),'Current Git/file bytes differ: '+name)
    if name in FILES:
      old=git_output(['show',primary_git+':'+name])
      require(old == current,'Primary 19 frozen files differ: '+name)
    pins[name]=hashlib.sha256(current).hexdigest()
  return pins


def successful_exit(value,limit,elapsed_key):
  require(type(value.get('exit_code')) is int and value['exit_code'] == 0 and
      value.get('timed_out') is False,'Missing explicit successful terminal exit')
  elapsed=value.get(elapsed_key)
  require(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0 < elapsed <= limit,
      'Whole-entry elapsed budget differs')


def primary_authority(primary,terminal,code_pins):
  primary=Path(primary).resolve(strict=True)
  for name in ('late-supervisor.json','late-provisional-supervisor.json','supervisor-receipt-pending.json'):
    require(not (primary/name).exists(),'Late/incomplete primary receipt')
  budget=read_json(primary/'budget.json');supervisor=read_json(primary/'supervisor_complete.json')
  frozen=read_json(primary/'freeze.json');seal=read_json(primary/'freeze-seal-parent.json')
  result=read_json(primary/'result.json');external=read_json(terminal)
  primary_git=budget['actual_git_sha']
  require(all(x['version'] == VERSION and x['actual_git_sha'] == primary_git
      for x in (budget,supervisor,frozen,seal,result)), 'Primary version/Git chain differs')
  require(budget['seconds'] == 3600 and budget['planned_physical_gpu'] == 4 and
      budget['max_gpu_devices'] == 1 and budget['max_gpu_hours'] == 1 and
      type(budget['parent_pid']) is int and budget['parent_pid'] > 0 and
      supervisor['parent_pid'] == budget['parent_pid'], 'Primary fixed budget/resource differs')
  producer_pins={name:code_pins[name] for name in FILES}
  require(budget['source_files_sha256'] == frozen['source_files_sha256'] == producer_pins and
      budget['script_sha256'] == producer_pins['scripts/dl_reset_joint.py'], 'Primary code byte pins differ')
  successful_exit(supervisor,3600,'elapsed_parent_entry_seconds')
  require(supervisor['complete_verified'] is True and supervisor['actual_gpu_devices_used'] == 1 and
      supervisor['budget_sha256'] == sha(primary/'budget.json') and
      supervisor['result_sha256'] == sha(primary/'result.json') and
      supervisor['method_effectiveness_established'] is False and
      supervisor['independent_audit_complete'] is False,'Primary complete chain differs')
  successful_exit(external,3600,'elapsed_external_seconds')
  require(external['actual_git_sha'] == primary_git and
      Path(external['output']).resolve() == primary,'External CLI terminal does not bind primary')
  phases=supervisor['phases']
  require(len(phases) == 9 and [x['phase'] for x in phases] == list(PHASES),'Partial/reordered primary phases')
  for phase in phases:
    successful_exit(phase,3600,'elapsed_seconds')
    require(phase['worker_terminal'] is True and phase['owned_group_cleanup_requested'] is True and
        type(phase['worker_pid']) is int and phase['worker_pid'] > 0 and
        phase['is_gpu'] == (phase['phase'] in ('baseline','B','R','eval0','evalB','evalR')) and
        read_json(primary/(phase['phase']+'-exit.json')) == phase,'Primary worker exit chain differs')
  require(len({x['worker_pid'] for x in phases}) == 9,'Repeated worker identity')
  require(frozen['cal_episodes'] == list(CAL) and frozen['blind_episodes'] == list(BLIND) and
      frozen['groups'] == ['0','B128','R128'] and frozen['updates_per_arm'] == 128 and
      frozen['statistics_seed'] == 20261026 and frozen['blind_arrays_read'] is False and
      frozen['selection'] == 'four-prediction-mean then exact argmax, lowest exact tie ID' and
      set(frozen['files']) == set(FREEZE_FILES),'Frozen scope/split differs')
  require(seal['freeze_sha256'] == result['freeze_sha256'] == sha(primary/'freeze.json') and
      seal['blind_started'] is False and
      budget['monotonic_start'] <= frozen['monotonic_freeze'] <= seal['monotonic_sealed'] <=
      budget['monotonic_start']+3600,'Parent preblind seal/order differs')
  require(result['truth_sha256'] == sha(primary/'truth-complete.json') and
      result['source_inputs_unchanged_at_final_statistics'] is True and
      result['method_effectiveness_established'] is False and result['continuous_actor_supported'] is False and
      result['full_goal_complete'] is False and result['decision'] == 'stop_after_fixed_reset_joint_assay',
      'Primary scientific scope/result differs')
  pins={name:sha(primary/name) for name in ('budget.json','supervisor_complete.json',
      'freeze.json','freeze-seal-parent.json','result.json','truth-complete.json')}
  for name,pin in frozen['files'].items():
    require(sha(safe_file(primary,name)) == pin,'Frozen artifact changed: '+name);pins[name]=pin
  for name in PHASES:
    pins[name+'-exit.json']=sha(primary/(name+'-exit.json'))
  pins['external_cli_terminal']=sha(terminal)
  return dict(budget=budget,supervisor=supervisor,frozen=frozen,seal=seal,result=result,
      primary_git=primary_git,pins=pins)


def authenticate_source(binding):
  """Only raw archive bytes and small JSON are read, never np.load/pickle."""
  require(binding['source_git_sha'] == SOURCE_GIT and binding['source_json_sha256'] == SOURCE_PINS and
      binding['source_original_exit_chain_verified'] is True,'Source authority identity differs')
  source=Path(binding['source_dataset']).resolve(strict=True)
  paths=binding['source_json_paths']
  require(set(paths) == set(SOURCE_PINS) and
      Path(paths['plan']).resolve() == source/'collection_plan.json' and
      Path(paths['complete']).resolve() == source/'collection_complete.json', 'Source metadata paths differ')
  for name,pin in SOURCE_PINS.items(): require(sha(paths[name]) == pin,'Pinned source JSON bytes differ')
  plan,complete,artifact,physical=[read_json(paths[k]) for k in ('plan','complete','artifact_audit','physics_audit')]
  require(binding['source_model_metadata'] == plan['metadata'],'Source declared model metadata differs')
  require(artifact['source_contract_artifact_audit_passed'] is True and physical['status'] == 'passed' and
      artifact['expected_source_sha'] == physical['expected_source_sha'] == SOURCE_GIT and
      physical['blind_npz_arrays_opened'] is False and physical['physics_steps'] == 12800,
      'Fixed source audits incomplete')
  names=('budget.json','supervisor_complete.json','worker_exit_code.txt')
  require(set(binding['source_terminal_paths']) == set(binding['source_terminal_sha256']) == set(names),
      'Source original exit endpoints missing')
  for name in (*names,'collection_plan.json','collection_complete.json'):
    pin=sha(safe_file(source,name))
    require(pin == artifact['metadata_sha256'][name],'Original source exit/JSON hash differs')
    if name in names:
      require(Path(binding['source_terminal_paths'][name]).resolve() == source/name and
          binding['source_terminal_sha256'][name] == pin,'Source terminal binding differs')
  budget=read_json(source/'budget.json');supervisor=read_json(source/'supervisor_complete.json')
  successful_exit(supervisor,3600,'elapsed_seconds')
  require(budget['version'] == supervisor['version'] == 'DL-source-r1' and
      budget['git_sha'] == supervisor['git_sha'] == SOURCE_GIT and budget['seconds'] == 3600 and
      supervisor['worker_terminal'] is True and supervisor['budget_sha256'] == sha(source/'budget.json') and
      (source/'worker_exit_code.txt').read_bytes() == b'0\n','Original source exit chain differs')
  require(all(plan['metadata'][k] == v for k,v in MODEL_PINS.items()), 'Original model metadata pins differ')
  entries=binding['all_32_artifact_hashes']
  require(len(entries) == len(complete['artifacts']) == len(plan['episodes']) == 32,'Incomplete source archive binding')
  for ep,(bound,item,definition) in enumerate(zip(entries,complete['artifacts'],plan['episodes'])):
    path=safe_file(source,f'episode_{ep:03}.npz')
    require(bound == dict(episode=ep,path=str(path),sha256=item['sha256']) and
        item['episode'] == definition['episode'] == ep and item['path'] == path.name and
        definition['split'] == ('calibration' if ep in CAL else 'blind') and
        sha(path) == item['sha256'],'Source archive identity/bytes/split differ')
  return plan


def calibration_truth(readouts):
  """Reuse the fixed completed cal audit; independently sum its new rewards."""
  import numpy as np
  root=CAL_ROOT.resolve(strict=True)
  external_path=root.parent/'engineering/primary-terminal01.json'
  audit_path=root.parent/'independent-audit01/audit.json'
  paths={**{k:root/k for k in ('result.json','natural_reset_complete.json')},
      'external':external_path,'audit':audit_path}
  for name,path in paths.items():require(sha(path) == CAL_PINS[name],'Fixed previously audited cal evidence differs')
  require(not (root/'late_supervisor_receipt.json').exists(),'Previously audited cal late completion')
  report=read_json(audit_path);result=read_json(root/'result.json')
  complete=read_json(root/'natural_reset_complete.json');supervisor=read_json(root/'supervisor_complete.json')
  external=read_json(external_path)
  require(report['status'] == 'passed' and set(report['primary_json_sha256']) ==
      {'budget.json','plan.json','result.json','natural_reset_complete.json','supervisor_complete.json'},
      'Fixed original cal independent audit chain incomplete')
  hashes={str(path):CAL_PINS[name] for name,path in paths.items()}
  for name,pin in report['primary_json_sha256'].items():
    require(sha(root/name) == pin,'Previously audited cal original JSON changed');hashes[str(root/name)]=pin
  require(supervisor['exit_code'] == external['exit_code'] == 0 and
      type(supervisor['exit_code']) is int and type(external['exit_code']) is int and
      supervisor['complete_verified'] is True and supervisor['timed_out'] is False and
      supervisor['worker_terminal'] is True and complete['actual_git_sha'] == external['actual_git_sha'] ==
      'aa9fb7f777a28e8a82d57c31f9e749b5912af316' and
      supervisor['budget_sha256'] == complete['budget_sha256'] == sha(root/'budget.json'),
      'Previously audited cal complete/external exit chain differs')
  oracle=[];measured=[]
  require(len(result['evidence_artifacts']) == 8,'Eight fixed calibrated truth packets required')
  for ep,item in zip(CAL,result['evidence_artifacts']):
    require(item['episode'] == ep and item['path'] == f'episode_{ep:03}.npz','Fixed cal truth episode differs')
    path=safe_file(root,item['path']);payload=hash_payload(path,item['sha256']);hashes[str(path)]=item['sha256']
    # This is a new numeric calibration product, never an original source/RGB archive.
    with np.load(io.BytesIO(payload),allow_pickle=False) as archive:rewards=archive['reward_f64']
    require(rewards.shape == (7,20) and rewards.dtype == np.float64 and np.isfinite(rewards).all(),
        'Fixed cal complete F64 reward contract differs')
    oracle.append([math.fsum(float(x) for x in row) for row in rewards])
    measured.append(rewards.sum(-1,dtype=np.float64))
  oracle=np.asarray(oracle,np.float64);measured=np.asarray(measured,np.float64)
  require(np.max(np.abs(oracle-measured)) <= 1e-10,'Previously audited cal fsum differs')
  verify_reward_boundaries(oracle,measured)
  for batch in readouts['cal'].values():require(byte_equal(batch['returns'],measured),'Cal readout true costs differ from fixed physics')
  return dict(hashes=hashes,independent_physics_steps=0,prior_completed_cal_audit_authenticated=True,
      numeric_rewards_reduced_independently_with_fsum=True,
      maximum_absolute_sum_difference=float(np.max(np.abs(oracle-measured))))


def verify_reward_boundaries(oracle,measured):
  import numpy as np
  require(np.array_equal(oracle.max(1)[:,None]-oracle <= .2,
      measured.max(1)[:,None]-measured <= .2),'Fsum crosses closed delta-optimal boundary')
  # Preserve every possible signed choice gap, including exact zero/ties.
  actual=oracle[:,:,None]-oracle[:,None,:];expected=measured[:,:,None]-measured[:,None,:]
  require(np.array_equal(np.sign(actual),np.sign(expected)),'Fsum crosses signed external utility/tie boundary')


def validate_learning(receipt,group,source_binding):
  import numpy as np
  require(receipt['version'] == 'DL-reset-learning-r1' and receipt['group'] == group and
      receipt['updates'] == 128 and len(receipt['receipts']) == 128 and
      receipt['reset_rep_gradient_stopped'] == (group == 'R') and
      receipt['rep_forward_values_retained'] is True and receipt['historical_optimizer_momentum_retained'] is True and
      receipt['per_update_init_train'] is True and receipt['blind_arrays_read'] is False and
      receipt['simulator_calls'] == 0 and receipt['total_loss_positions'] == 131072 and
      receipt['total_reset_loss_positions'] == 2048,'Learning configuration/128 update scope differs')
  binding=receipt['binding']
  require(binding['source_hashes'] == {str(e):source_binding['all_32_artifact_hashes'][e]['sha256'] for e in CAL} and
      binding['offline_episode_concatenation'] is True and binding['original_continuous_replay_restored'] is False and
      binding['balanced_duplicates_per_episode'] == 2 and binding['reset_loss_positions_per_update'] == 16 and
      binding['loss_positions_per_update'] == 1024,'Learning fixed input/batch scope differs')
  for index,row in enumerate(binding['layout']):
    ep=(CAL*2)[index]
    require(row == dict(row=index,episode=ep,context_episode=CAL[(CAL.index(ep)-1)%8],
        context_position=1000,loss_positions=[0,63],terminal_action_executed=False,
        context_first_is_replay_annotation=True),'Learning terminal/context/reset rows differ')
  require(len(binding['layout']) == 16,'Incomplete batch layout')
  saved=receipt['saved_source_counters'];base=receipt['restored_counters']
  names={'updates','batches','actions','eval_actions','eval_seed'}
  require(set(saved) == set(base) == names and all(type(v) is int and v >= 0 for v in saved.values()) and
      base == {**saved,'batches':saved['updates']},'Standard Agent.load counters differ')
  optimizer=receipt['optimizer_counter']
  require(set(optimizer) == {'key','dtype','shape','initial','final','updates'} and
      optimizer['key'] == 'opt/step/value' and optimizer['dtype'] == 'int32' and optimizer['shape'] == [] and
      type(optimizer['initial']) is int and optimizer['initial'] >= 0 and
      type(optimizer['final']) is int and optimizer['final'] == optimizer['initial']+128 and
      optimizer['updates'] == 128,'Actual official optimizer variable counter receipt differs')
  active=0;excess=[]
  for index,row in enumerate(receipt['receipts'],1):
    require(row['update'] == index and row['seed_counter'] == base['batches']+index-1 and
        row['counters'] == {**base,'updates':base['updates']+index,'batches':base['batches']+index},
        'Actual per-update counter progression differs')
    require(type(row['train_seed']) is list and len(row['train_seed']) == 2 and
        all(type(x) is int and 0 <= x < 2**32 for x in row['train_seed']),'Actual train root malformed')
    pressure=row['reset_pressure'];metrics=row['metrics']
    require(set(pressure) == {'active_count','excess_sum','reset_count'} and
        type(pressure['active_count']) is int and 0 <= pressure['active_count'] <= 16 and
        pressure['reset_count'] == 16 and math.isfinite(pressure['excess_sum']) and pressure['excess_sum'] >= 0 and
        (pressure['active_count'] > 0) == (pressure['excess_sum'] > 0), 'Actual rep pressure invariant differs')
    require(metrics and all(np.isfinite(np.asarray(v)).all() for v in metrics.values()) and
        all(float(metrics['dl_reset/'+k]) == v for k,v in pressure.items()),'Pressure/actual train metrics differ')
    require(type(metrics.get('opt/updates')) is int and metrics['opt/updates'] == optimizer['initial']+index,
        'Actual per-update official optimizer counter metric differs')
    require(re.fullmatch('[0-9a-f]{64}',row['metrics_digest']), 'Missing original typed metrics digest')
    active+=pressure['active_count'];excess.append(pressure['excess_sum'])
  summary=receipt['reset_pressure']
  require(summary['active_count'] == active and summary['reset_count'] == 2048 and
      summary['excess_sum'] == math.fsum(excess) and summary['entire_128_update_metrics_observed'] is True,
      'All-128 actual pressure aggregation differs')
  return dict(actual_updates=128,active_count=active,reset_count=2048,excess_sum=math.fsum(excess),
      optimizer_counter=optimizer,
      optimizer_independently_reexecuted=False,typed_metrics_digest_recomputed_from_json=False)


def byte_equal(a,b):
  import numpy as np
  return type(a) is np.ndarray and type(b) is np.ndarray and a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def validate_device(primary,phase,device,authority):
  uuid=device['gpu_uuid'];worker=next(x for x in authority['supervisor']['phases'] if x['phase'] == phase)
  require(device['physical_gpu'] == 4 and device['logical_server'] == 'sv3' and
      device['actual_hostname'].split('.')[0] == 'lyg0326' and type(uuid) is str and uuid.startswith('GPU-') and
      device['compute_uuid'] == device['egl_uuid'] == device['cuda_visible_devices'] == uuid and
      device['mujoco_egl_device_id'] == '4' and device['exclusive'] is True,'Recorded single CUDA/EGL GPU4 binding differs')
  path=Path(device['resource_source_path']).resolve(strict=True)
  require(path == (primary/(phase+'-resource.json')).resolve() and sha(path) == device['resource_source_sha256'],
      'Actual fresh parent resource record differs')
  authority['pins'][phase+'-resource.json']=device['resource_source_sha256']
  for key in ('runtime_compute_start','runtime_compute_end'):
    actual=device[key]
    require(actual['pid'] == worker['worker_pid'] and actual['physical_gpu'] == 4 and
        actual['compute_uuid'] == [uuid] and actual['live_state']['gpu_uuid'] == uuid,
        'Actual worker start/end CUDA runtime UUID receipt differs')
  proof=device['device_mapping_evidence'];proof_path=Path(proof['path']).resolve(strict=True)
  require(sha(proof_path) == proof['sha256'] == device['device_probe_sha256'], 'Recorded actual EGL/compute probe hash differs')
  probe=read_json(proof_path)
  require(probe['physical_gpu'] == 4 and probe['uuid'] == uuid and
      probe['verified_compute_and_graphics_same_gpu'] is True and probe['training_actions'] == 0 and
      probe['agent_updates'] == 0 and probe['logical_cuda_device'] == 0,
      'Actual mapping probe does not certify one device')
  expected=dict(CUDA_VISIBLE_DEVICES=uuid,MUJOCO_EGL_DEVICE_ID='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',
      MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
  require(all(probe['environment'][k] == v for k,v in expected.items()), 'Actual mapping probe environment differs')
  payload=git_output(['show',authority['primary_git']+':scripts/probe_formal_devices.py'])
  require(probe['source_sha256'] == hashlib.sha256(payload).hexdigest() == sha(ROOT/'scripts/probe_formal_devices.py'),
      'Actual mapping probe code identity differs')
  return str(proof_path),proof['sha256']


def validate_features(feature,numeric,episodes,reference=None):
  import numpy as np
  aux=feature['aux'];n=len(episodes)
  for key in ('h','zq','zp','D','K'):
    original=feature[key]
    require(type(original) is np.ndarray and np.isfinite(original).all() and
        byte_equal(np.asarray(original,np.float64),numeric[key]),'Native/numeric feature promotion differs: '+key)
  require(feature['h'].shape[0] == n and str(feature['h'].dtype) == 'bfloat16' and
      feature['zq'].shape == feature['zp'].shape == (n,4,32,32) and
      str(feature['zq'].dtype) == str(feature['zp'].dtype) == 'bfloat16' and
      feature['D'].dtype == feature['K'].dtype == np.float32,'Actual feature shape/dtype differs')
  roots=np.stack([np.random.default_rng([20261026,0x52535231,k]).integers(
      0,np.iinfo(np.uint32).max,(2,),np.uint32) for k in range(4)])
  require(aux['common_h_byte_exact'] is True and aux['common_prior_byte_exact'] is True and
      aux['common_zp_per_key_byte_exact'] is True and aux['current_agent_params_used'] is True and
      aux['public_policy_copy_used'] is False and aux['sample_roots_have_episode_model_branch_id'] is False and
      aux['source_q_sample_exact_claimed'] is False and aux['learned_only'] is True and
      aux['optimizer_updates'] == 0 and aux['full_state_before_sha256'] == aux['full_state_after_sha256'] and
      aux['counters_before'] == aux['counters_after'],'Read-only actual feature receipt differs')
  common=aux['common_reference']
  require(common['verified_observations'] == (8 if n == 8 else 32) and
      common['model_state_sha256'] == aux['full_state_before_sha256'] and
      byte_equal(common['roots'],roots), 'Common reset reference/count differs')
  for ep in range(n):
    require(byte_equal(aux['root_key'][ep],roots) and byte_equal(aux['zp'][ep],feature['zp'][ep]) and
        byte_equal(aux['zq'][ep],feature['zq'][ep]),'Actual sample roots/latent payload differs')
    for draw in range(4):
      require(byte_equal(aux['h'][ep,draw],feature['h'][ep]) and
          byte_equal(feature['h'][ep],common['h']) and
          byte_equal(aux['prior_logit'][ep,draw],common['prior_logit']) and
          byte_equal(feature['zp'][ep,draw],common['zp'][draw]) and
          byte_equal(aux['post_logit'][ep,draw],aux['post_logit'][ep,0]) and
          aux['D'][ep,draw] == feature['D'][ep] and aux['K'][ep,draw] == feature['K'][ep] and
          bool(aux['rng_valid'][ep,draw]) and int(aux['rng_consumption'][ep,draw]) == 1,
          'Actual common h/prior/zp or mode-D/K bytes differ')
  if reference is not None:
    for key in ('h','prior_logit','zp','roots'):
      require(byte_equal(common[key],reference[key]),'All32 common reference differs: '+key)
    require(common['model_state_sha256'] == reference['model_state_sha256'], 'Cal/blind current full state differs')
  return common


def new_artifacts(primary,authority):
  """Authenticate generated packets before every possible pickle decode."""
  import numpy as np
  import ml_dtypes  # Register BF16 only; no model/JAX import.
  frozen=authority['frozen'];result=authority['result'];pins=authority['pins']
  readouts=load_new(primary,'readouts.pkl',frozen['files']['readouts.pkl'])
  blind=load_new(primary,'blind-numeric.pkl',result['blind_numeric_sha256'])
  pins['blind-numeric.pkl']=result['blind_numeric_sha256']
  receipts={g:read_json(primary/(('baseline' if g == '0' else g)+'-complete.json')) for g in ('0','B','R')}
  contexts=load_new(primary,'contexts.pkl',frozen['files']['contexts.pkl'])
  require(set(contexts) == set(CAL),'Incomplete frozen context entries')
  for ep in CAL:
    require(set(contexts[ep]) == {'dyn/deter','dyn/stoch'} and
        all(type(x) is np.ndarray and x.dtype == np.float32 and np.isfinite(x).all() for x in contexts[ep].values()) and
        contexts[ep]['dyn/stoch'].shape == (32,32) and
        np.logical_or(contexts[ep]['dyn/stoch'] == 0,contexts[ep]['dyn/stoch'] == 1).all() and
        np.all(contexts[ep]['dyn/stoch'].sum(-1) == 1),'Actual frozen terminal entries malformed')
    proof=receipts['0']['context_proofs'][str(ep)]
    key=np.random.default_rng([20261026,0x43545831,ep]).integers(0,2**32,(2,),dtype=np.uint32)
    require(proof['root_key'] == key.astype(int).tolist() and proof['frames'] == 1001 and proof['episodes'] == 1 and
        proof['annotation_is_new_private_representation'] is True and proof['source_original_carry_exact_claimed'] is False and
        proof['previous_terminal_action_is_raw999'] is True and proof['terminal_placeholder_executed'] is False and
        proof['actor_calls'] == proof['environment_actions'] == proof['optimizer_updates'] == 0 and
        proof['full_state_before_sha256'] == proof['full_state_after_sha256'] and
        proof['counters_before'] == proof['counters_after'] and
        proof['replay_context_promoted_to_float32_exact_roundtrip'] is True,'Full-history terminal annotation receipt differs')
  cal_features={};summaries={};external_device_pins={}
  for g in ('0','B','R'):
    receipt=receipts[g]
    require(receipt['version'] == VERSION and receipt['actual_git_sha'] == authority['primary_git'] and
        receipt['blind_arrays_read'] is False and receipt['cal_image_frames_decoded'] == 8008 and
        receipt['context_sha256'] == frozen['files']['contexts.pkl'],'Cal source/context receipts differ')
    require(all(receipt['model_metadata'][k] == v for k,v in MODEL_PINS.items()),'Baseline loader model metadata differs')
    path,pin=validate_device(primary,'baseline' if g == '0' else g,receipt['device'],authority)
    external_device_pins[path]=pin
    cal_features[g]=load_new(primary,g+'-cal.pkl',receipt['feature_sha256'])
    require(receipt['feature_sha256'] == frozen['files'][g+'-cal.pkl'],'Frozen cal feature pin differs')
    ref=validate_features(cal_features[g],readouts['cal'][g],CAL)
    eval_receipt=read_json(primary/('eval'+g+'-complete.json'))
    require(eval_receipt['version'] == VERSION and eval_receipt['actual_git_sha'] == authority['primary_git'] and
        eval_receipt['phase'] == 'eval'+g and eval_receipt['episodes'] == list(BLIND) and
        eval_receipt['freeze_sha256'] == authority['pins']['freeze.json'],'Blind evaluation receipt differs')
    feature=load_new(primary,g+'-blind.pkl',eval_receipt['feature_sha256'])
    path,pin=validate_device(primary,'eval'+g,eval_receipt['device'],authority)
    external_device_pins[path]=pin
    validate_features(feature,blind[g],BLIND,ref)
    metadata=eval_receipt['model_metadata']
    require(metadata['current_evaluation_group'] == g and metadata['baseline_loader_metadata'] is True and
        feature['aux']['counters_before'] == cal_features[g]['aux']['counters_before'],
        'Current model/counter eval identity differs')
    if g == '0':
      require(metadata['current_group_checkpoint_sha256'] == MODEL_PINS['checkpoint_sha256'] and
          metadata['current_params_sha256'] == MODEL_PINS['params_sha256'],'Baseline eval identity differs')
    else:
      learning=receipt['learning'];summaries[g]=validate_learning(learning,g,receipt['source_binding'])
      pin=receipt['checkpoint_sha256']
      require(pin == frozen['files'][g+'-checkpoint.pkl'] and sha(primary/(g+'-checkpoint.pkl')) == pin and
          metadata['current_group_checkpoint_sha256'] == pin and
          metadata['current_full_state_sha256'] == learning['final_state_digest'],
          'Current full checkpoint/final learning/eval receipt differs')
      seed=int(receipt['model_metadata']['config']['seed'])
      for row in learning['receipts']:
        expected=np.random.default_rng([seed,row['seed_counter']]).integers(
            0,np.iinfo(np.uint32).max,(2,),np.uint32).astype(int).tolist()
        require(row['train_seed'] == expected,'Actual normal counter train RNG differs')
      initial=load_new(primary,g+'-initial-cal.pkl',receipt['initial_feature_sha256'])
      validate_features(initial,readouts['cal']['0'],CAL)
      for key in ('h','zq','zp','D','K'):
        require(byte_equal(initial[key],cal_features['0'][key]),'Initial paired full model feature bytes differ')
      require(initial['aux']['full_state_before_sha256'] == cal_features['0']['aux']['full_state_before_sha256'] and
          initial['aux']['counters_before'] == learning['restored_counters'] and
          cal_features[g]['aux']['counters_before'] == learning['receipts'][-1]['counters'],
          'Initial/final full-state/counter feature receipt differs')
      pins[g+'-initial-cal.pkl']=receipt['initial_feature_sha256']
    for name in ('eval'+g+'-complete.json',g+'-blind.pkl'):
      pins[name]=sha(primary/name)
  b,r=[receipts[g]['learning'] for g in ('B','R')]
  for key in ('binding','source_checkpoint_digest','initial_state_digest','initial_params_digest',
      'calibration_arrays_digest','restored_counters','saved_source_counters'):
    require(b[key] == r[key],'B/R paired original full-state/input binding differs: '+key)
  require([x['train_seed'] for x in b['receipts']] == [x['train_seed'] for x in r['receipts']],
      'B/R common actual train roots differ')
  return readouts,blind,receipts,summaries,external_device_pins


def verify_assay_utility_boundaries(oracle,measured,readouts,result):
  import numpy as np
  indices=np.random.default_rng(20261026).integers(0,24,(2000,24))
  def summarize(values):
    draws=np.asarray([math.fsum(float(values[i]) for i in row)/24 for row in indices],np.float64)
    return np.asarray([math.fsum(float(x) for x in values)/24,*np.quantile(draws,[.025,.975])])
  utility={};rows=np.arange(24)
  for name in ('0','B','R'):
    choices=np.asarray(result['analysis']['models'][name]['choices'],np.int32)
    fit=readouts['frozen']['fit'] if name == '0' else readouts['refits'][name]
    const=fit['constant_choice']
    q,p=choices[:,0],choices[:,1]
    utility[name]=[(oracle[rows,q]-oracle[rows,p],measured[rows,q]-measured[rows,p]),
        (oracle[rows,q]-oracle[:,const],measured[rows,q]-measured[:,const])]
  pairs=[pair for values in utility.values() for pair in values]
  for left,right in (('B','0'),('R','B'),('R','0')):
    for metric in range(2):
      pairs.append((utility[left][metric][0]-utility[right][metric][0],
          utility[left][metric][1]-utility[right][metric][1]))
  for actual,expected in pairs:
    require(np.array_equal(np.sign(actual),np.sign(expected)), 'Fsum crosses paired episode utility zero')
    a,b=summarize(actual),summarize(expected)
    require(np.max(np.abs(a-b)) <= 1e-10 and np.array_equal(np.sign(a),np.sign(b)),
        'Fsum crosses utility mean/bootstrap CI zero')


def replay_truth(primary,truth,blind,pins,source_plan,readouts,result):
  import numpy as np
  from analysis import dl_natural_reset_audit as old
  old.check_deadline=check_deadline  # Every old loop/hash shares our complete 300s entry.
  restore=old.load_restore();suite,official=old.official_modules()
  require({k:v['sha256'] for k,v in truth['official_modules'].items()} ==
      {k:v['sha256'] for k,v in official.items()},'Primary/independent official module pins differ')
  packages={name:importlib.metadata.version(name) for name in ('numpy','ml_dtypes','dm-control','mujoco')}
  require(packages == truth['packages'] == {'numpy':'1.26.4','ml_dtypes':'0.5.4','dm-control':'1.0.48','mujoco':'3.15.0'},
      'Actual official CPU package pins differ')
  require(truth['episodes'] == list(BLIND) and len(truth['receipts']) == 24 and
      truth['reference_physics_steps'] == 480 and truth['main_physics_steps'] == 3360 and
      truth['replay_physics_steps'] == 3360 and truth['total_physics_steps'] == 7200 and
      truth['new_reset_calls'] == 24 and truth['new_renderer_calls'] == 0 and
      truth['before_carries_read'] is False and truth['blind_scientific_positions'] == [0],
      'Primary truth full execution scope differs')
  labels=load_new(primary,'blind-returns.pkl',truth['labels_sha256'])
  require(type(labels) is np.ndarray and labels.shape == (24,7) and labels.dtype == np.float64 and
      np.isfinite(labels).all(),'Truth labels malformed')
  pins['blind-returns.pkl']=truth['labels_sha256']
  # Authenticate but never decode cached RGB, including this generated packet.
  require(sha(primary/'blind-observations.pkl') == truth['observations_sha256'],'Cached blind RGB packet byte hash differs')
  pins['blind-observations.pkl']=truth['observations_sha256']
  all_returns=[];replays=[]
  for ep,item in zip(BLIND,truth['receipts']):
    check_deadline()
    require(item['episode'] == ep and item['path'] == f'blind-physics-{ep:03}.pkl', 'Truth record identity differs')
    ref=item['reference']
    require(ref['episode'] == ep and ref['position'] == 0 and ref['physics_steps'] == 20 and
        ref['fresh_official_seed_reset_verified'] is True and ref['fresh_reset_calls'] == 1 and
        ref['original_raw_applied_byte_exact'] is True and ref['source_reward_float32_byte_exact'] is True and
        ref['source_float64_reward_byte_exact_claimed'] is False and
        ref['source_complete_state_tolerance'] == 1e-8 and ref['flags_exact'] is True and
        ref['complete_task_rng_clock_reset_done_exact'] is True and
        0 <= ref['maximum_complete_integration_error'] <= 1e-8 and
        0 <= ref['fresh_reset_complete_integration_error'] <= 1e-8,
        'Authenticated freshreset/source reference receipt incomplete')
    records=load_new(primary,item['path'],item['sha256']);pins[item['path']]=item['sha256']
    require(type(records) is list and len(records) == 7 and len(item['replays']) == 7,'Incomplete seven-control truth')
    env=suite.load('quadruped','walk',task_kwargs={'random':source_plan['episodes'][ep]['seed']})
    try:
      old.verify_official_model(env)
      state_size=records[0]['snapshots'][0]['suite']['integration'].size
      initial=records[0]['snapshots'][0]
      ep_replays=[];values=[]
      for control,record in enumerate(records):
        require(record['episode'] == ep and record['position'] == 0 and record['control'] == control,
            'Truth record position/control differs')
        old.complete_state_exact(record['snapshots'][0],initial)
        old.validate_record(record,old.commands()[control],initial,state_size)
        claimed=item['replays'][control]
        require(claimed['physics_steps'] == 20 and claimed['padding_steps'] == 0,'Primary stored replay incomplete')
        ep_replays.append(dict(control=control,**old.replay_record(env,record,restore.restore,restore.snapshot)))
        values.append(math.fsum(float(x) for x in record['reward_f64']))
      all_returns.append(values);replays.append(dict(episode=ep,path=item['path'],sha256=item['sha256'],replays=ep_replays))
    finally: env.close()
  oracle=np.asarray(all_returns,np.float64)
  require(np.max(np.abs(oracle-labels)) <= 1e-10,'Independent F64 fsum/labels differ')
  verify_reward_boundaries(oracle,labels)
  verify_assay_utility_boundaries(oracle,labels,readouts,result)
  # The authentic original numeric audit has already passed. For this separate
  # reduction sensitivity check only, replace the derived input digest in a
  # private copy; never change original results, decisions, labels, fits or files.
  # This checks all reported CI/coverage/decision sides, including signal errors.
  from analysis import dl_reset_joint_numeric_audit as numeric
  oracle_blind={g:{**batch,'returns':oracle.copy()} for g,batch in blind.items()}
  sensitivity_result=copy.deepcopy(result)
  for g in ('0','B','R'):
    sensitivity_result['analysis']['input_digests'][g]['returns']=numeric.digest(oracle)
  sensitivity=numeric.audit_numeric(readouts,oracle_blind,sensitivity_result)
  require(sensitivity['passed'] is True,'Independent fsum numerical boundary sensitivity failed')
  # Tolerance applies only to sums; decision boundaries remain strict in numeric audit.
  for batch in blind.values(): require(byte_equal(batch['returns'],labels),'Numeric decision truth labels differ')
  require('jax' not in sys.modules,'CPU independent physics imported JAX')
  return dict(physics_steps=3360,complete_state_byte_exact=True,float64_reward_byte_exact=True,
      replay_receipts=replays,official_modules=official,packages=packages,
      maximum_return_absolute_difference=float(np.max(np.abs(oracle-labels))),
      fsum_closed_delta_signed_utility_and_bootstrap_zero_sides_exact=True,
      fsum_result_boundary_sensitivity_audit=sensitivity,
      sensitivity_is_original_artifact_hash_authentication=False,
      sensitivity_original_result_values_changed=False,
      fresh_seed_resets_independently_repeated=False,source_reference_steps_independently_repeated=0,
      source_reset_binding='authenticated primary freshreset/reference receipts; original archives never decoded')


def audit(args):
  check_deadline()
  sys.path.insert(0,str(ROOT))
  primary=Path(args.primary).resolve(strict=True);output=Path(args.output).resolve(strict=True)
  budget=read_json(output/'budget.json')
  require(os.environ.get('DL_RESET_JOINT_AUDIT_PARENT') == str(budget['parent_pid']) and
      os.getppid() == budget['parent_pid'] and
      budget['actual_git_sha'] == args.expected_git and budget['seconds'] == SECONDS and
      budget['script_sha256'] == sha(__file__) and budget['planned_gpu_devices'] == 0,
      'Worker requires its authenticated owned parent')
  global ENTRY_STARTED
  ENTRY_STARTED=budget['monotonic_start'];check_deadline()
  require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('MUJOCO_GL') == 'disable',
      'CPU/no-render process isolation differs')
  primary_git=read_json(primary/'budget.json')['actual_git_sha']
  code=git_binding(args.expected_git,primary_git)
  authority=primary_authority(primary,args.primary_cli_terminal,code)
  truth=read_json(primary/'truth-complete.json')
  require(truth['actual_git_sha'] == primary_git and truth['version'] == VERSION and
      truth['source_files_sha256'] == {k:code[k] for k in FILES} and
      truth['freeze_sha256'] == authority['pins']['freeze.json'],'Truth origin/freezing differs')
  source_plan=authenticate_source(truth['source_binding'])
  readouts,blind,receipts,learning,device_pins=new_artifacts(primary,authority)
  calibration=calibration_truth(readouts)
  for receipt in receipts.values():
    source=receipt['source_binding'];other=truth['source_binding']
    for key in ('source_git_sha','source_json_sha256','source_json_paths','source_terminal_sha256',
        'source_terminal_paths','all_32_artifact_hashes','source_model_metadata'):
      require(source[key] == other[key], 'Cal/truth raw source identity differs: '+key)
  from analysis.dl_reset_joint_numeric_audit import audit_numeric
  numerical=audit_numeric(readouts,blind,authority['result'])
  require(numerical['passed'] is True,'Independent numeric audit did not pass')
  physics=replay_truth(primary,truth,blind,authority['pins'],source_plan,readouts,authority['result'])
  # Close every raw-byte and Git binding again after actual numeric/physics work.
  authenticate_source(truth['source_binding'])
  for path,pin in calibration['hashes'].items():
    require(sha(path) == pin,'Previously audited cal truth chain changed at endpoint')
  for path,pin in device_pins.items():
    require(sha(path) == pin,'Actual CUDA/EGL mapping proof changed at endpoint')
  for name,pin in authority['pins'].items():
    path=Path(args.primary_cli_terminal) if name == 'external_cli_terminal' else primary/name
    require(sha(path) == pin,'Primary generated/exit artifact changed at endpoint: '+name)
  require(git_binding(args.expected_git,primary_git) == code,'Auditor/current primary code changed')
  check_deadline()
  report=dict(version=VERSION,implementation=IMPLEMENTATION,status='passed',
      auditor_git_sha=args.expected_git,primary_git_sha=primary_git,primary=str(primary),
      primary_cli_terminal=str(Path(args.primary_cli_terminal).resolve()),
      primary_cli_terminal_sha256=authority['pins']['external_cli_terminal'],
      primary_json_and_generated_sha256=authority['pins'],source_files_sha256=code,
      authenticated_recorded_device_proofs_sha256=device_pins,
      source_binding=truth['source_binding'],original_source_archives_decoded=0,original_blind_arrays_read=False,
      cached_rgb_packets_decoded=0,model_checkpoints_decoded=0,model_calls=0,optimizer_updates=0,
      optimizer_independently_reexecuted=False,actual_gpu_devices_used=0,new_renderer_calls=0,
      learning_receipt_audit=learning,calibration_truth_audit=calibration,numerical_audit=numerical,physics_audit=physics,
      actual_independent_physics_steps=3360,full32_source_bytes_unchanged=True,
      complete_primary_exit_and_parent_preblind_seal_verified=True,
      method_effectiveness_established=False,continuous_actor_supported=False,full_goal_complete=False,
      elapsed_worker_from_parent_entry_seconds=time.monotonic()-ENTRY_STARTED)
  write_json(output/'audit.json',report);check_deadline();return 0


def verify_receipt(output,expected):
  report=read_json(Path(output)/'audit.json')
  require(report['status'] == 'passed' and report['version'] == VERSION and
      report['implementation'] == IMPLEMENTATION and report['auditor_git_sha'] == expected and
      report['actual_gpu_devices_used'] == 0 and report['actual_independent_physics_steps'] == 3360 and
      report['numerical_audit']['passed'] is True and report['physics_audit']['physics_steps'] == 3360 and
      report['complete_primary_exit_and_parent_preblind_seal_verified'] is True and
      report['full32_source_bytes_unchanged'] is True and report['optimizer_independently_reexecuted'] is False,
      'Independent complete audit receipt missing')
  require(sum(x['physics_steps'] for e in report['physics_audit']['replay_receipts'] for x in e['replays']) == 3360,
      'Independent full 3360 receipts missing')
  return dict(complete_verified=True,audit_sha256=sha(Path(output)/'audit.json'))


def supervise(command,output,*,expected_git=None,seconds=SECONDS,started=None,verify=None):
  """Generic subprocess fixtures never gain scientific completion authority."""
  started=time.monotonic() if started is None else started;deadline=started+seconds
  output=Path(output).resolve();require(not output.exists(),'Fresh audit attempt required');output.mkdir(parents=True)
  write_json(output/'budget.json',dict(version=VERSION,implementation=IMPLEMENTATION,
      parent_pid=os.getpid(),seconds=seconds,monotonic_start=started,actual_git_sha=expected_git,
      script_sha256=sha(__file__),planned_gpu_devices=0,planned_physics_steps=3360))
  timed_out=False;verification=None
  with (output/'audit.log').open('x',encoding='utf-8') as log:
    child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=os.name == 'posix',
        env=dict(os.environ,DL_RESET_JOINT_AUDIT_PARENT=str(os.getpid()),CUDA_VISIBLE_DEVICES='',
          MUJOCO_GL='disable',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    OWNED.add(child.pid)
    try: code=child.wait(timeout=max(.001,deadline-time.monotonic()))
    except subprocess.TimeoutExpired: timed_out=True;code=124
    finally: kill_owned(child)
  (output/'worker_exit_code.txt').write_text(str(code)+'\n',encoding='utf-8')
  if code == 0 and not timed_out and verify is not None:
    try: verification=verify(output,expected_git)
    except Exception as error:
      write_json(output/'verification-failure.json',dict(error=repr(error)));code=1
  if time.monotonic() >= deadline: timed_out=True;code=124;verification=None
  real=verify is verify_receipt and bool(verification and verification.get('complete_verified') is True)
  receipt=dict(version=VERSION,implementation=IMPLEMENTATION,actual_git_sha=expected_git,
      parent_pid=os.getpid(),worker_pid=child.pid,exit_code=code,timed_out=timed_out,
      worker_terminal=child.poll() is not None,owned_group_cleanup_requested=os.name == 'posix',
      elapsed_parent_entry_seconds=time.monotonic()-started,complete_verified=real,
      verification=verification,verification_kind='actual_complete_audit_chain' if real else 'generic_unknown_actual_scope',
      actual_gpu_devices_used=0 if real else None,actual_physics_steps=3360 if real else None,
      budget_sha256=sha(output/'budget.json'),worker_exit_sha256=sha(output/'worker_exit_code.txt'))
  pending=output/'supervisor-receipt-pending.json';write_json(pending,receipt)
  late=time.monotonic() >= deadline
  if not late:
    os.replace(pending,output/'supervisor_complete.json');late=time.monotonic() >= deadline
  if late:
    if (output/'supervisor_complete.json').exists():
      os.replace(output/'supervisor_complete.json',output/'late-provisional-supervisor.json')
    receipt.update(exit_code=124,timed_out=True,complete_verified=False,verification=None,
        actual_gpu_devices_used=None,actual_physics_steps=None)
    write_json(output/'late-supervisor.json',dict(exit_code=124,incomplete=True))
    write_json(output/'supervisor_complete.json',receipt)
  return receipt


def main():
  p=argparse.ArgumentParser(description=__doc__)
  for name in ('primary','primary-cli-terminal','expected-git','output'): p.add_argument('--'+name,required=True)
  p.add_argument('--worker',action='store_true');args=p.parse_args()
  if args.worker: return audit(args)
  command=[sys.executable,str(Path(__file__).resolve()),'--worker']
  for name in ('primary','primary_cli_terminal','expected_git','output'):
    command.extend(['--'+name.replace('_','-'),str(getattr(args,name))])
  receipt=supervise(command,args.output,expected_git=args.expected_git,started=ENTRY_STARTED,verify=verify_receipt)
  return receipt['exit_code'] if receipt['exit_code'] or receipt['complete_verified'] else 1


if __name__ == '__main__': raise SystemExit(main())
