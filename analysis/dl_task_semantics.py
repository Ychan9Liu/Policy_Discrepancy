"""DL-task-semantics-r1: fixed calibration task queries, not critical labels.

Derived from the frozen independent feedback physics auditor through an
explicit, SHA256-pinned file path. This is not another independent physics
audit. Only hash-bound cal32 initial snapshots are queried: no environment
transition, control, reset, rendering, model, JAX, signal/utility computation,
state selection, or blind decoding. Full trusted BF16 records require
ml_dtypes dtype registration through the pinned auditor dependency.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import signal
import sys
import time

ENTRY_STARTED = time.monotonic()
os.environ['MUJOCO_GL'] = 'disable'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
import numpy as np

VERSION = 'DL-task-semantics-r1'
FEEDBACK_SHA = '0d49c589b525ba0c26f9f85bf52c23573dc19507'
AUDITOR_SHA = '54e5b5a24fc9971ef8f454979a1ee9a9bc47c536f52bcac42b4c50efd96ef1ab'
SECONDS = 120
EPISODES = [9,14,15,18,20,24,30,31]
POSITIONS = [1,300,600,900]
MODULE_PINS = {
    'dm_control.suite': '9b7002e4b70d7223376258173fff8ff7fa8ce53dd3535992f9e6dcb21a233760',
    'dm_control.suite.quadruped': '154662d7fb3ab576a013dbf3741ffb9d83009d791309fb50f2d33d9d5b04c91f',
    'dm_control.utils.rewards': '009f02389fd2af84b2955324707638e79a0e18b6a540dd7c254abf5096e157f3',
    'dm_control.rl.control': '3ac4b726242ea31f28235cfc94ff87ebbd7238bc71a673024281c74b9aef1606'}
PACKAGE_PINS = {'numpy':'1.26.4','ml_dtypes':'0.5.4','dm-control':'1.0.48','mujoco':'3.15.0'}


def require(condition,message):
  if not condition:
    raise ValueError(message)


def hash_bytes(payload):
  return hashlib.sha256(payload).hexdigest()


def load_auditor(path):
  path=Path(path).resolve(strict=True)
  # Bind the bytes actually executed, without a second pathname read between
  # hashing and execution. A wrong/modified dependency never executes.
  payload=path.read_bytes()
  require(hash_bytes(payload)==AUDITOR_SHA,'Frozen physics auditor SHA256 differs')
  spec=importlib.util.spec_from_file_location('_dl_semantics_pinned_physics',path)
  module=importlib.util.module_from_spec(spec)
  exec(compile(payload,str(path),'exec'),module.__dict__)
  require(module.VERSION=='DL-independent-feedback-physics-audit-v1',
      'Unexpected auditor identity')
  require('jax' not in sys.modules,'Pinned dependency unexpectedly imported JAX')
  return module,dict(path=str(path),sha256=AUDITOR_SHA,
      version=module.VERSION,derived_implementation=True,new_independent_physics_audit=False)


def official_modules():
  from dm_control import suite
  from dm_control.suite import quadruped
  from dm_control.utils import rewards
  from dm_control.rl import control
  receipts={}
  for module in (suite,quadruped,rewards,control):
    filename=Path(module.__file__).resolve(strict=True)
    digest=hash_bytes(filename.read_bytes())
    require(digest==MODULE_PINS[module.__name__],
        'Pinned official source module differs: '+module.__name__)
    receipts[module.__name__]=dict(path=str(filename),sha256=digest)
  versions={name:importlib.metadata.version(name) for name in PACKAGE_PINS}
  require(versions==PACKAGE_PINS,'Pinned official CPU package versions differ')
  return suite,quadruped,rewards,receipts,versions


def numeric_scalar(value,name):
  value=np.asarray(value)
  require(value.dtype==np.float64 and value.shape==() and np.isfinite(value),
      'Finite official float64 scalar required: '+name)
  return value.copy()


def query_state(env,saved,expected_reward,audit,quadruped,rewards):
  import mujoco
  require(type(env._task) is quadruped.Move and type(env.physics) is quadruped.Physics,
      'Unexpected official quadruped task/physics class')
  size=mujoco.mj_stateSize(env.physics.model.ptr,mujoco.mjtState.mjSTATE_INTEGRATION)
  audit.validate_snapshot(saved,size)
  require(not saved['from_dm_done'] and not saved['suite']['reset_next_step'],
      'Query source is inactive/terminal/reset')
  audit.array(expected_reward,np.float32,(),'initial reward encoding')
  audit.restore(env,saved)
  before=audit.snapshot(env,False)
  restore_error=audit.compare_state(before,saved)
  before_pickle=pickle.dumps(before,protocol=5)
  upright=numeric_scalar(env.physics.torso_upright(),'torso upright')
  velocity=np.asarray(env.physics.torso_velocity()).copy()
  audit.array(velocity,np.float64,(3,),'torso local velocity')
  speed=numeric_scalar(env._task._desired_speed,'actual desired speed')
  require(float(speed)==.5,'Actual frozen walk desired speed differs')
  upright_reward=numeric_scalar(quadruped._upright_reward(env.physics),'official upright reward')
  # The actual official Move.get_reward expression, with its actual task speed.
  # No threshold, label or independently invented reward is introduced.
  move_reward=numeric_scalar(rewards.tolerance(velocity[0],
      bounds=(float(speed),float('inf')),margin=float(speed),value_at_margin=.5,
      sigmoid='linear'),'official move reward')
  product=numeric_scalar(upright_reward*move_reward,'official component product')
  task_reward=numeric_scalar(env._task.get_reward(env.physics),'official task reward')
  require(product.tobytes()==task_reward.tobytes(),'Official task/component reward bytes differ')
  require(np.asarray(task_reward,np.float32).tobytes()==expected_reward.tobytes(),
      'Current task reward does not match original initial F32 observation encoding')
  physical_time=numeric_scalar(env.physics.data.time,'physical time')
  step_limit=numeric_scalar(env._step_limit,'actual time limit clock')
  control_timestep=numeric_scalar(env.control_timestep(),'actual control timestep')
  require(float(step_limit).is_integer() and float(step_limit)==1000 and
      0<=env._step_count<=step_limit and physical_time>=0 and control_timestep>0,
      'Invalid/frozen task time or clock')
  after=audit.snapshot(env,False)
  query_error=audit.compare_state(after,before)
  # A physics restore may use the existing 1e-8 contract. A pure query must
  # change no integration byte, even when a change is below that tolerance.
  require(audit.byte_equal(before['suite']['integration'],after['suite']['integration']) and
      pickle.dumps(after,protocol=5)==before_pickle,'Readonly query changed state/task/RNG/counter bytes')
  require('jax' not in sys.modules,'Query unexpectedly imported JAX')
  return dict(torso_upright=float(upright),torso_velocity_local=velocity.tolist(),
      desired_speed=float(speed),upright_reward=float(upright_reward),move_reward=float(move_reward),
      reward_component_product=float(product),task_reward=float(task_reward),
      initial_observation_reward_float32=float(expected_reward),
      physics_time=float(physical_time),step_count=int(env._step_count),step_limit=int(step_limit),
      remaining_clock=int(step_limit)-int(env._step_count),control_timestep=float(control_timestep),
      source_snapshot_sha256=hash_bytes(pickle.dumps(saved,protocol=5)),
      restored_snapshot_sha256=hash_bytes(before_pickle),post_query_snapshot_sha256=hash_bytes(pickle.dumps(after,protocol=5)),
      task_rng_before_sha256=hash_bytes(pickle.dumps(before['suite']['task'],protocol=5)),
      task_rng_after_sha256=hash_bytes(pickle.dumps(after['suite']['task'],protocol=5)),
      reward_product_bytes_exact=True,initial_reward_float32_bytes_exact=True,
      readonly_integration_bytes_exact=True,readonly_task_rng_counter_reset_bytes_exact=True,
      from_dm_expected_done=False,actual_from_dm_wrapper_executed=False,
      maximum_restore_error=restore_error,maximum_query_error=query_error)


def records_for_episode(root,item,ep,audit):
  payload=(root/item['path']).read_bytes()
  require(hash_bytes(payload)==item['sha256'],'Artifact changed after completed authority check')
  with np.load(io.BytesIO(payload),allow_pickle=False) as archive:
    require(set(archive.files)==set(audit.ARRAYS)|
        {'episode','positions','source_image','fresh_image','records'},'Complete artifact schema differs')
    audit.array(archive['episode'],np.int32,(),'episode')
    audit.array(archive['positions'],np.int32,(4,),'positions')
    require(int(archive['episode'])==ep and
        audit.byte_equal(archive['positions'],np.array(POSITIONS,np.int32)), 'Episode/position identity differs')
    serialized=archive['records']
    require(isinstance(serialized,np.ndarray) and serialized.dtype==np.uint8 and
        serialized.ndim==1 and serialized.size>0,'Trusted full-record envelope differs')
    records=pickle.loads(serialized.tobytes())
    require(isinstance(records,list) and len(records)==32,'Full record count differs')
    numeric={name:archive[name] for name in audit.ARRAYS}
    for name,(dtype,shape) in audit.ARRAYS.items():
      audit.array(numeric[name],dtype,(4,4,2)+shape,'NPZ '+name)
    for index,record in enumerate(records):
      i,pair,branch=np.unravel_index(index,(4,4,2))
      require((record['episode'],record['position'],record['pair'],record['branch'])==
          (ep,POSITIONS[i],int(pair),int(branch)) and record['prior'] is bool(branch) and
          record['version']==audit.SCIENCE and record['implementation']==audit.PHYSICAL and
          record['action_contract']==audit.CONTRACT and record['horizon']==100,
          'Record identity/version/branch contract differs')
      require(type(record['executed_steps']) is int and 1<=record['executed_steps']<=100 and
          len(record['snapshots'])==len(record['observations'])==record['executed_steps']+1,
          'Record genuine snapshot/observation coverage differs')
      for name,(dtype,shape) in audit.ARRAYS.items():
        audit.array(record[name],dtype,shape,name)
        require(audit.byte_equal(record[name],numeric[name][i,pair,branch]),
            'NPZ/trusted record byte binding differs: '+name)
      audit.observation(record['observations'][0])
      require(record['snapshots'][0]['suite']['step_count']==POSITIONS[i] and
          not bool(record['observations'][0]['is_first']) and
          not bool(record['observations'][0]['is_last']) and
          not bool(record['observations'][0]['is_terminal']),'Initial active source/absolute clock differs')
  return records


def run(dataset,expected_sha,audit,dependency,official,check):
  require(expected_sha==FEEDBACK_SHA,'This object is bound to the completed engineering-r2 run only')
  root=Path(dataset).resolve(strict=True)
  plan,result,json_hashes=audit.authority(root,expected_sha)
  suite,quadruped,rewards,module_receipts,versions=official
  require(all(plan['environment']['packages'][name]==versions[name] for name in
      ('numpy','dm-control','mujoco')),'Input/runtime official package version differs')
  require(plan['calibration'] and [x['definition']['episode'] for x in plan['calibration']]==EPISODES,
      'Fixed calibration set differs')
  budget=audit.json_file(root/'budget.json')
  require(budget['script_sha256']==plan['source_files_sha256']['scripts/dl_feedback.py'],
      'Producer source file/budget hash differs')
  rows=[];artifacts=[]
  for item,cal in zip(result['evidence_artifacts'],plan['calibration']):
    check()
    ep=item['episode']
    require(cal['definition']['seed']==audit.environment_seed(ep) and
        cal['definition']['split']=='calibration','Actual source seed/split differs')
    records=records_for_episode(root,item,ep,audit)
    env=suite.load('quadruped','walk',task_kwargs={'random':audit.environment_seed(ep)})
    try:
      for i,pos in enumerate(POSITIONS):
        check()
        reference=records[8*i]  # Fixed pair0/q, never chosen by an outcome.
        import mujoco
        size=mujoco.mj_stateSize(env.physics.model.ptr,mujoco.mjtState.mjSTATE_INTEGRATION)
        initial=reference['snapshots'][0]
        audit.validate_snapshot(initial,size)
        for other in records[8*i:8*i+8]:
          audit.validate_snapshot(other['snapshots'][0],size)
          audit.compare_state(other['snapshots'][0],initial)
          require(audit.byte_equal(other['snapshots'][0]['suite']['integration'],initial['suite']['integration']),
              'Eight branches do not share byte-identical initial integration')
          require(pickle.dumps(other['initial_carry'],protocol=5)==
              pickle.dumps(reference['initial_carry'],protocol=5),'Eight initial model carry receipts differ')
          for key in audit.OBS_KEYS:
            require(audit.byte_equal(other['observations'][0][key],reference['observations'][0][key]),
                'Eight branches do not share initial observation bytes')
        row=query_state(env,initial,reference['observations'][0]['reward'],audit,quadruped,rewards)
        rows.append(dict(episode=ep,position=pos,pair=0,branch=0,shared_initial_branches_verified=8,
            task_class=type(env._task).__module__+'.'+type(env._task).__name__,**row))
    finally:
      env.close()
    artifacts.append(dict(episode=ep,path=item['path'],sha256=item['sha256']))
    print(json.dumps(dict(stage='task_semantics',episode=ep,positions=4)),flush=True)
  check()
  require(len(rows)==32 and [(x['episode'],x['position']) for x in rows]==
      [(ep,pos) for ep in EPISODES for pos in POSITIONS],'All fixed32 query coverage required')
  return dict(version=VERSION,status='passed',dataset=str(root),feedback_git_sha=expected_sha,
      feedback_science=audit.SCIENCE,feedback_implementation=audit.ENGINEERING,
      derived_auditor_dependency=dependency,official_module_sha256=module_receipts,environment=versions,
      feedback_json_sha256=json_hashes,producer_source_files_sha256=plan['source_files_sha256'],
      source_binding=plan['source_binding'],artifacts=artifacts,rows=rows,complete_states=32,
      environment_steps=0,controls_executed=0,environment_resets=0,render_calls=0,model_calls=0,
      gpu_devices_used=0,jax_imported=('jax' in sys.modules),blind_arrays_decoded=False,
      utility_or_D_K_e_computed=False,critical_labels_generated=False,
      state_selection='all fixed eight calibration episodes x four positions; pair0/q initial only',
      ml_dtypes_dtype_registration_only=True,exploratory_after_feedback_result=True,
      action_order_or_information_increment_or_actor_use_established=False,
      training_authorized=False,method_effectiveness_established=False,
      limitations=[
          'Derived authority/restore/query implementation, not another independent physics audit.',
          'Official privileged task variables describe coverage; never inputs to the vision-only policy/gate.',
          'No actions, rollouts, renderer, model or causal task-cue intervention; no critical state labels.',
          'Current result-after exploration is not preregistered evidence or independent training-seed replication.',
          'Does not prove action ordering, current RGB information beyond history, actor utilization or method effect.'])


def self_test(audit,dependency,official):
  from unittest import mock
  suite,quadruped,rewards,module_receipts,versions=official
  env=suite.load('quadruped','walk',task_kwargs={'random':audit.environment_seed(9)})
  tests=[]
  try:
    # Factory-model synthetic active snapshot: no reset or physics transition.
    # This tests real official read-only queries, not a naturally collected state.
    env.physics.forward()
    saved=audit.snapshot(env,False)
    saved['suite']['step_count']=1
    saved['suite']['reset_next_step']=False
    audit.restore(env,saved)
    expected=np.asarray(env._task.get_reward(env.physics),np.float32)
    original_step=type(env).step
    original_reset=type(env).reset
    with (mock.patch.object(type(env),'step',side_effect=AssertionError('Forbidden env.step')),
         mock.patch.object(type(env),'reset',side_effect=AssertionError('Forbidden env.reset')),
         mock.patch.object(type(env.physics),'render',side_effect=AssertionError('Forbidden render'))):
      row=query_state(env,saved,expected,audit,quadruped,rewards)
    require(row['maximum_query_error']==0 and row['readonly_integration_bytes_exact'],
        'Actual official readonly query selftest failed')
    tests.append('official_quadruped_query_no_step_reset_render_and_all_readonly_bytes')
    def rejected(snapshot_value,expected_value,label):
      try:
        query_state(env,snapshot_value,expected_value,audit,quadruped,rewards)
      except (ValueError,AssertionError,TypeError):
        tests.append(label);return
      raise AssertionError('Counterexample accepted: '+label)
    changed=copy.deepcopy(saved);changed['suite']['integration'][0]=np.nan
    rejected(changed,expected,'nonfinite_snapshot_rejected')
    changed=copy.deepcopy(saved);changed['from_dm_done']=True
    rejected(changed,expected,'inactive_source_rejected')
    rejected(saved,np.array(np.nan,np.float32),'nonfinite_original_reward_rejected')
    rejected(saved,np.asarray(np.nextafter(expected,np.float32(np.inf))),
        'initial_F32_reward_tamper_rejected')
    original_reward=quadruped.Move.get_reward
    def mutating_reward(task,physics):
      value=original_reward(task,physics)
      physics.data.qvel[0]=np.nextafter(physics.data.qvel[0],np.float64(np.inf))
      return value
    with mock.patch.object(quadruped.Move,'get_reward',mutating_reward):
      rejected(saved,expected,'sub_1e8_single_ulp_query_state_write_rejected')
    with mock.patch.object(quadruped.Move,'get_reward',return_value=np.float64(np.nan)):
      rejected(saved,expected,'nonfinite_task_reward_rejected')
    def task_rng_write(task,physics):
      value=original_reward(task,physics)
      task.random.uniform()
      return value
    with mock.patch.object(quadruped.Move,'get_reward',task_rng_write):
      rejected(saved,expected,'query_task_RNG_consumption_rejected')
    require(type(env).step is original_step and type(env).reset is original_reset,
        'Selftest patches not restored')
    # Wrong dependency bytes must be rejected before any arbitrary code executes.
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
      wrong=Path(directory)/'wrong.py';wrong.write_text('raise AssertionError("executed")\n',encoding='utf-8')
      try:
        load_auditor(wrong)
      except ValueError:
        tests.append('modified_dependency_rejected_before_execution')
      else:
        raise AssertionError('Wrong dependency accepted')
    require('jax' not in sys.modules,'Selftest unexpectedly imported JAX')
    return dict(version=VERSION,status='selftest_passed',tests=tests,query_example=row,
        fixture='official factory model with synthetic active clock; not a collected trajectory',
        actual_artifacts_opened=False,derived_auditor_dependency=dependency,
        official_module_sha256=module_receipts,environment=versions,
        environment_steps=0,controls_executed=0,environment_resets=0,render_calls=0,model_calls=0,
        gpu_devices_used=0,jax_imported=False,training_authorized=False)
  finally:
    env.close()


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--dataset')
  parser.add_argument('--expected-feedback-sha',default=FEEDBACK_SHA)
  parser.add_argument('--physics-auditor',required=True)
  parser.add_argument('--output',required=True)
  parser.add_argument('--self-test',action='store_true')
  args=parser.parse_args()
  output=Path(args.output)
  if output.exists():
    raise FileExistsError('New task semantics output path required')
  timer_armed=False
  def check():
    if time.monotonic()-ENTRY_STARTED>=SECONDS:
      raise TimeoutError('DL-task-semantics-r1 hard deadline exhausted; incomplete')
  try:
    if hasattr(signal,'setitimer'):
      def interrupted(signum,frame):
        raise TimeoutError('DL-task-semantics-r1 POSIX hard wall deadline exhausted; incomplete')
      signal.signal(signal.SIGALRM,interrupted)
      signal.setitimer(signal.ITIMER_REAL,max(.001,SECONDS-(time.monotonic()-ENTRY_STARTED)))
      timer_armed=True
    check()
    audit,dependency=load_auditor(args.physics_auditor)
    official=official_modules()
    check()
    require(args.self_test or args.dataset,'Dataset required for actual fixed32 query')
    receipt=self_test(audit,dependency,official) if args.self_test else run(
        args.dataset,args.expected_feedback_sha,audit,dependency,official,check)
    check()
    code=0
  except Exception as exc:
    receipt=dict(version=VERSION,status='failed',complete=False,error_type=type(exc).__name__,
        error=str(exc),feedback_git_sha=args.expected_feedback_sha,training_authorized=False,
        method_effectiveness_established=False)
    code=1
  finally:
    if timer_armed:
      signal.setitimer(signal.ITIMER_REAL,0)
  receipt['script_sha256']=hash_bytes(Path(__file__).read_bytes())
  receipt['elapsed_seconds']=time.monotonic()-ENTRY_STARTED
  receipt['deadline_seconds']=SECONDS
  receipt['deadline_enforcement']='POSIX wall timer plus monotonic checks' if timer_armed else 'monotonic checks; Windows has no POSIX timer'
  output.parent.mkdir(parents=True,exist_ok=True)
  with output.open('x',encoding='utf-8') as stream:
    json.dump(receipt,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
  print(json.dumps(dict(status=receipt['status'],output=str(output),elapsed_seconds=receipt['elapsed_seconds'],
      complete_states=receipt.get('complete_states'),error=receipt.get('error')),allow_nan=False),flush=True)
  return code


if __name__=='__main__':
  raise SystemExit(main())
