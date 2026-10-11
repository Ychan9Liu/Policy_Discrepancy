"""DL-action-information-r1: first command, then 19 normalized-zero commands.

Official CPU suite only. The caller supplies the frozen complete-state auditor
and hash-bound initial states. No policy, model, rendering, or fresh RNG calls.
"""
import hashlib
from pathlib import Path

import numpy as np

VERSION = 'DL-action-information-r1'
IMPLEMENTATION = 'DL-action-information-physics-r1'
CONTRACT = 'production_clip_unify_normalize_v1'
HORIZON = 20
XML_SHA = 'c6b0ade5f3160b155bbe673aec3e0430eed3273ae13b60880855647e82bf8334'
NAMES = tuple(f'{joint}_{leg}' for leg in
    ('front_left','front_right','back_right','back_left')
    for joint in ('yaw','lift','extend'))
ARRAYS = dict(raw=(np.float32,(20,12)), applied=(np.float64,(20,12)),
    reward_f64=(np.float64,(20,)), reward_f32=(np.float32,(20,)),
    discount=(np.float64,(20,)), is_first=(np.bool_,(20,)),
    is_last=(np.bool_,(20,)), is_terminal=(np.bool_,(20,)),
    action_executed=(np.bool_,(20,)), snapshot_executed=(np.bool_,(21,)))


def require(condition, message):
  if not condition:
    raise ValueError(message)


def array(value, dtype, shape, name):
  require(isinstance(value,np.ndarray) and value.dtype==np.dtype(dtype) and
      value.shape==shape and np.isfinite(value).all(), 'Invalid finite array: '+name)


def verify_model(env):
  import mujoco
  from dm_control.suite import quadruped
  xml=Path(quadruped.__file__).with_suffix('.xml')
  digest=hashlib.sha256(xml.read_bytes()).hexdigest()
  require(digest==XML_SHA,'Official quadruped XML differs')
  require(type(env._task) is quadruped.Move and type(env.physics) is quadruped.Physics and
      env._task._desired_speed==.5,'Unreviewed task or physics')
  model=env.physics.model
  names=tuple(model.id2name(i,'actuator') for i in range(model.nu))
  require(names==NAMES,'Actuator identity/order differs')
  require(np.all(model.actuator_dyntype==int(mujoco.mjtDyn.mjDYN_FILTER)) and
      np.array_equal(model.actuator_dynprm[:,0],np.full(12,.1)), 'Actuator filter differs')
  require(model.opt.timestep==.005 and env.control_timestep()==.02 and
      env._step_limit==1000,'Time contract differs')
  spec=env.action_spec()
  low,high=spec.minimum.copy(),spec.maximum.copy()
  array(low,np.float64,(12,),'low'); array(high,np.float64,(12,),'high')
  require(np.array_equal(low,np.tile([-1.,-1.,-.8],4)) and
      np.array_equal(high,np.tile([1.,1.1,.8],4)) and
      np.array_equal(np.stack([low,high],-1),model.actuator_ctrlrange), 'Actual control bounds differ')
  return dict(xml_sha256=digest,actuator_names=list(names),filter_tau=.1,
      physics_timestep=.005,control_timestep=.02,step_limit=1000,
      low=low.tolist(),high=high.tolist(),action_contract=CONTRACT)


def mapping(raw, low, high):
  array(raw,np.float32,(12,),'raw command')
  array(low,np.float64,(12,),'low'); array(high,np.float64,(12,),'high')
  clipped=np.clip(raw,-1,1).astype(np.float32,copy=False)
  fraction=(clipped+1)/2
  applied=fraction*(high-low)+low
  array(applied,np.float64,(12,),'mapped command')
  return applied


def controls(env):
  verify_model(env)
  values=np.zeros((7,12),np.float32)
  values[1]=.25; values[2]=-.25
  values[3,[1,10]]=.25; values[3,[4,7]]=-.25
  values[4]=-values[3]
  values[5,[2,8]]=.25; values[5,[5,11]]=-.25
  values[6]=-values[5]
  return values


def timestep(ts, start_clock, index):
  require(not ts.first(),'Unexpected reset/first transition')
  reward=np.asarray(ts.reward); discount=np.asarray(ts.discount,np.float64)
  array(reward,np.float64,(),'official F64 reward')
  array(discount,np.float64,(),'official discount')
  require(float(discount) in (0.,1.),'Unreviewed discount')
  require(not (discount==0 and not ts.last()),'Terminal is not last')
  if ts.last() and discount==1:
    require(start_clock+index+1==1000 and index==19,'Early time limit cannot enter H20')
  return reward, discount


def run_sequence(env, saved, first_raw, audit, check=lambda:None):
  receipt=verify_model(env); check()
  import mujoco
  size=mujoco.mj_stateSize(env.physics.model.ptr,mujoco.mjtState.mjSTATE_INTEGRATION)
  audit.validate_snapshot(saved,size)
  require(not saved['from_dm_done'] and not saved['suite']['reset_next_step'] and
      saved['suite']['step_count']<=980,'Inactive state or insufficient clock')
  array(first_raw,np.float32,(12,),'first raw')
  audit.restore(env,saved)
  require(verify_model(env)==receipt,'Restored task/model differs from frozen contract')
  initial=audit.snapshot(env,False)
  restore_error=audit.compare_state(initial,saved)
  record={name:np.zeros(shape,dtype) for name,(dtype,shape) in ARRAYS.items()}
  record.update(version=VERSION,implementation=IMPLEMENTATION,action_contract=CONTRACT,
      horizon=HORIZON,first_raw=first_raw.copy(),model_receipt=receipt,
      snapshots=[initial],executed_steps=0,end_type='window',maximum_restore_error=restore_error)
  record['snapshot_executed'][0]=True
  low=np.asarray(receipt['low'],np.float64); high=np.asarray(receipt['high'],np.float64)
  clock=initial['suite']['step_count']
  for i in range(HORIZON):
    check()
    raw=first_raw.copy() if i==0 else np.zeros(12,np.float32)
    applied=mapping(raw,low,high)
    ts=env.step(applied)
    reward,discount=timestep(ts,clock,i)
    record['raw'][i]=raw; record['applied'][i]=applied
    record['reward_f64'][i]=reward; record['reward_f32'][i]=np.float32(reward)
    record['discount'][i]=discount
    record['is_first'][i]=ts.first(); record['is_last'][i]=ts.last()
    record['is_terminal'][i]=ts.last() and discount==0
    record['action_executed'][i]=True; record['snapshot_executed'][i+1]=True
    state=audit.snapshot(env,ts.last())
    require(state['suite']['step_count']==clock+i+1,'Absolute physics clock differs')
    record['snapshots'].append(state); record['executed_steps']=i+1
    if ts.last():
      record['end_type']='true_terminal' if discount==0 else 'time_limit'
      break
  return record


def validate_record(env, record, audit):
  require(record['version']==VERSION and record['implementation']==IMPLEMENTATION and
      record['action_contract']==CONTRACT and record['horizon']==20,'Unknown sequence contract')
  require(record['model_receipt']==verify_model(env),'Recorded model differs')
  count=record['executed_steps']
  require(type(count) is int and 1<=count<=20,'Malformed executed length')
  array(record['first_raw'],np.float32,(12,),'first raw')
  for name,(dtype,shape) in ARRAYS.items():
    array(record[name],dtype,shape,name)
  require(np.array_equal(record['action_executed'],np.arange(20)<count) and
      np.array_equal(record['snapshot_executed'],np.arange(21)<=count) and
      len(record['snapshots'])==count+1,'Executed prefix coverage differs')
  for name in set(ARRAYS)-{'action_executed','snapshot_executed'}:
    require(np.all(record[name][count:]==0),'Nonzero absorbing padding: '+name)
  require(audit.byte_equal(record['raw'][0],record['first_raw']) and
      np.all(record['raw'][1:count]==0),'First command or common tail differs')
  low=np.asarray(record['model_receipt']['low'],np.float64)
  high=np.asarray(record['model_receipt']['high'],np.float64)
  for i in range(count):
    require(audit.byte_equal(mapping(record['raw'][i],low,high),record['applied'][i]),'Stored mapping differs')
    require(np.asarray(record['reward_f64'][i],np.float32).tobytes()==record['reward_f32'][i].tobytes(),
        'F64/F32 reward encoding differs')
    require(not record['is_first'][i] and (i==count-1 or not record['is_last'][i]),'Reset/early last differs')
    require(record['discount'][i] in (0,1) and
        not (record['discount'][i]==0 and not record['is_last'][i]),'Invalid terminal discount')
    require(record['is_terminal'][i]==(record['is_last'][i] and record['discount'][i]==0),
        'Terminal flags differ')
  last=bool(record['is_last'][count-1]); terminal=bool(record['is_terminal'][count-1])
  require(count==20 or (last and terminal),'Partial branch lacks true terminal')
  require(record['end_type']==('true_terminal' if terminal else 'time_limit' if last else 'window'),
      'End type differs')
  initial=record['snapshots'][0]
  require(not initial['from_dm_done'] and not initial['suite']['reset_next_step'] and
      initial['suite']['step_count']<=980,'Inactive recorded initial state')
  import mujoco
  size=mujoco.mj_stateSize(env.physics.model.ptr,mujoco.mjtState.mjSTATE_INTEGRATION)
  for i,state in enumerate(record['snapshots']):
    audit.validate_snapshot(state,size)
    require(state['suite']['step_count']==initial['suite']['step_count']+i and
        state['from_dm_done']==(i>0 and bool(record['is_last'][i-1])) and
        state['suite']['reset_next_step']==(i>0 and bool(record['is_last'][i-1])),
        'Snapshot clock/done/reset differs')
  if last and not terminal:
    require(count==20 and initial['suite']['step_count']+count==1000,'Recorded early time limit')


def replay_sequence(env, record, audit, check=lambda:None):
  validate_record(env,record,audit); check()
  initial=record['snapshots'][0]
  audit.restore(env,initial)
  require(verify_model(env)==record['model_receipt'],'Restored replay task/model differs')
  restore_error=audit.compare_state(audit.snapshot(env,False),initial)
  maximum=restore_error; clock=initial['suite']['step_count']
  for i in range(record['executed_steps']):
    check()
    ts=env.step(record['applied'][i].copy())  # already mapped; never normalize again
    reward,discount=timestep(ts,clock,i)
    require(reward.tobytes()==record['reward_f64'][i].tobytes(), 'F64 reward byte replay differs')
    require(np.asarray(reward,np.float32).tobytes()==record['reward_f32'][i].tobytes() and
        discount.tobytes()==record['discount'][i].tobytes(),'Reward encoding/discount replay differs')
    require((ts.first(),ts.last(),ts.last() and discount==0)==
        tuple(bool(record[x][i]) for x in ('is_first','is_last','is_terminal')),'Replayed flags differ')
    maximum=max(maximum,audit.compare_state(audit.snapshot(env,ts.last()),record['snapshots'][i+1]))
  return dict(physics_steps=record['executed_steps'],padding_steps=20-record['executed_steps'],
      maximum_restore_error=restore_error,maximum_integration_error=maximum,
      float64_reward_byte_exact=True,float32_reward_encoding_byte_exact=True,
      discount_byte_exact=True,expected_from_dm_done_exact=True,actual_from_dm_wrapper_executed=False,
      rendering_executed=False,model_executed=False,conditional_stored_applied_replay=True)
