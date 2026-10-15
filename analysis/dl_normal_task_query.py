"""DL-normal-task-query-r1: read only the current official quadruped/walk state.

The caller must already have continuously replayed the authenticated prefix
from its declared seed and bound the source model. This helper reads no source
files and certifies neither that history nor cross-machine reproducibility.
It never restores, steps, resets, forwards, or renders a physics environment.

Exact integration/task/RNG/clock bytes apply to this same live CPU instance
immediately around a query. They do not assert that separately regenerated or
restored MuJoCo states will be byte identical. The current full MJB is checked
before/after and returned for the caller's separate source-model binding.
Integration state excludes derived xmat/sensordata buffers: the caller must
retain their synchronization from official live replay; this query does not
reconstruct them. ``velocity_x`` is torso-local velocimeter x, not world x.
The length-prefixed query snapshot hash uses DL-normal-task-query-snapshot-r1;
it differs from R5/source-index hashes. Bind them through the same byte-exact
saved snapshot and authenticated source_id, never by comparing those two SHAs.
"""
import copy
import hashlib
import pickle

import numpy as np

VERSION = 'DL-normal-task-query-r1'


def require(value,message):
  if not value:raise ValueError(message)


def _scalar(value,dtype,name,source=False):
  if source:
    require(isinstance(value,(np.ndarray,np.generic)),'Typed source scalar required: '+name)
  value=np.asarray(value)
  require(value.dtype == np.dtype(dtype) and value.shape == () and np.isfinite(value),
      'Finite exact '+str(np.dtype(dtype))+' scalar required: '+name)
  return value.copy()


def _snapshot(env,mujoco):
  bits=mujoco.mjtState.mjSTATE_INTEGRATION
  integration=np.empty(mujoco.mj_stateSize(env.physics.model.ptr,bits),np.float64)
  mujoco.mj_getState(env.physics.model.ptr,env.physics.data.ptr,integration,bits)
  return dict(suite=dict(integration=integration,step_count=env._step_count,
      reset_next_step=env._reset_next_step,task=copy.deepcopy(env._task.__dict__)),from_dm_done=False)


def _snapshot_bytes(value,size):
  require(type(value) is dict and set(value) == {'suite','from_dm_done'},'Complete wrapper snapshot required')
  state=value['suite']
  require(type(state) is dict and set(state) == {'integration','step_count','reset_next_step','task'},
      'Complete suite/task/RNG/counter/reset snapshot required')
  integration=state['integration']
  require(type(integration) is np.ndarray and integration.dtype == np.float64 and
      integration.shape == (size,) and np.isfinite(integration).all(),'Exact complete F64 integration state required')
  require(type(state['step_count']) is int and type(state['task']) is dict and
      type(state['reset_next_step']) is bool and type(value['from_dm_done']) is bool,
      'Exact task/counter/reset snapshot types required')
  require(not value['from_dm_done'] and not state['reset_next_step'],'Current query requires active nonreset state')
  # The integration payload is compared directly, never via a numeric tolerance.
  meta=pickle.dumps((state['step_count'],state['reset_next_step'],value['from_dm_done'],state['task']),protocol=5)
  return integration.tobytes(order='C'),meta


def _digest(parts):
  digest=hashlib.sha256()
  for part in parts:
    digest.update(len(part).to_bytes(8,'big'));digest.update(part)
  return digest.hexdigest()


def query_current(env,saved_snapshot,current_reward_f64,current_reward_f32,t):
  """Return typed current labels/components and exact read-only receipts.

  ``current_reward_f64`` is source reward[t-1] and ``current_reward_f32`` is
  observation[t].reward, both original typed scalar arrays/NumPy scalars. No
  next-state/shadow reward may substitute for this current source reward.
  """
  from dm_control.rl import control
  from dm_control.suite import quadruped
  from dm_control.utils import rewards
  import mujoco
  require(type(env) is control.Environment and type(env._task) is quadruped.Move and
      type(env.physics) is quadruped.Physics,'Exact official quadruped Move/Physics environment required')
  require(type(t) is int and 1 <= t < 1000,'Active ordinary absolute integer clock1..999 required')
  expected64=_scalar(current_reward_f64,np.float64,'source current reward',True)
  expected32=_scalar(current_reward_f32,np.float32,'source observation reward',True)
  require(np.asarray(expected64,np.float32).tobytes() == expected32.tobytes(),
      'Original source F64 reward/F32 observation encoding differs')
  speed=_scalar(env._task._desired_speed,np.float64,'actual desired speed')
  limit=_scalar(env._step_limit,np.float64,'actual time limit')
  require(speed.tobytes() == np.float64(.5).tobytes() and float(limit) == 1000,
      'Frozen walk desired speed .5/time limit1000 differs')
  require(type(env._step_count) is int and env._step_count == t and not env._reset_next_step and
      saved_snapshot['suite']['step_count'] == t,'Current/saved absolute active clock differs')
  before=_snapshot(env,mujoco);size=before['suite']['integration'].size
  before_parts=_snapshot_bytes(before,size);saved_parts=_snapshot_bytes(saved_snapshot,size)
  require(before_parts == saved_parts,'Current complete integration/task/RNG/counter/reset differs byte-exactly from source')
  model_before=env.physics.model.to_bytes()
  require(type(model_before) is bytes and bool(model_before),'Complete current MJB bytes required')
  upright=_scalar(env.physics.torso_upright(),np.float64,'torso upright')
  velocity=np.asarray(env.physics.torso_velocity()).copy()
  require(velocity.dtype == np.float64 and velocity.shape == (3,) and np.isfinite(velocity).all(),
      'Finite exact official F64 torso velocity3 required')
  upright_reward=_scalar(quadruped._upright_reward(env.physics),np.float64,'official upright reward')
  move_reward=_scalar(rewards.tolerance(velocity[0],bounds=(float(speed),float('inf')),
      margin=float(speed),value_at_margin=.5,sigmoid='linear'),np.float64,'official move reward')
  product=_scalar(upright_reward*move_reward,np.float64,'official reward component product')
  actual_reward=_scalar(env._task.get_reward(env.physics),np.float64,'official current Move reward')
  require(product.tobytes() == actual_reward.tobytes(),'Official Move reward/component product differs byte-exactly')
  require(actual_reward.tobytes() == expected64.tobytes(),'Official current F64 reward differs byte-exactly from source r_t')
  require(np.asarray(actual_reward,np.float32).tobytes() == expected32.tobytes(),
      'Official current reward F32 encoding differs from observation[t]')
  physical_time=_scalar(env.physics.data.time,np.float64,'current physical time')
  timestep=_scalar(env.control_timestep(),np.float64,'actual control timestep')
  require(physical_time >= 0 and timestep > 0,'Invalid official current physical clock')
  after=_snapshot(env,mujoco);after_parts=_snapshot_bytes(after,size)
  require(after_parts == before_parts,'Readonly query changed integration/task/RNG/counter/reset bytes')
  model_after=env.physics.model.to_bytes()
  require(model_after == model_before,'Readonly query changed complete model/MJB bytes')
  return dict(version=VERSION,t=t,velocity_x=velocity[0].copy(),upright=upright,
      torso_velocity_local=velocity,desired_speed=speed,upright_reward=upright_reward,
      move_reward=move_reward,reward_component_product=product,current_reward_f64=actual_reward,
      current_reward_f32=expected32,physics_time=physical_time,control_timestep=timestep,
      step_count=t,step_limit=1000,snapshot_hash_format='DL-normal-task-query-snapshot-r1',
      source_snapshot_sha256=_digest(saved_parts),
      before_snapshot_sha256=_digest(before_parts),after_snapshot_sha256=_digest(after_parts),
      task_rng_before_sha256=hashlib.sha256(pickle.dumps(before['suite']['task'],protocol=5)).hexdigest(),
      task_rng_after_sha256=hashlib.sha256(pickle.dumps(after['suite']['task'],protocol=5)).hexdigest(),
      current_model_sha256=hashlib.sha256(model_before).hexdigest(),current_model_bytes=len(model_before),
      current_reward_F64_bytes_exact=True,current_observation_F32_bytes_exact=True,
      readonly_full_snapshot_bytes_exact=True,readonly_model_bytes_exact=True,
      continuous_prefix_authenticated=False,source_model_binding_authenticated=False,
      environment_steps=0,resets=0,restores=0,forward_calls=0,render_calls=0)
