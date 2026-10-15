"""Real CPU suite, freshly reset/stepped generated prefix; no RGB or R5 files.

This verifies a same-instance readonly query, not any collected task/source
authentication, cross-machine regeneration, model inference or mechanism.
"""
import copy
import os
from pathlib import Path
import pickle
import subprocess
import sys
import unittest
from unittest import mock

os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('MUJOCO_GL','disable')
import numpy as np
from analysis import dl_normal_task_query as q

ROOT=Path(__file__).resolve().parents[1]


def snapshot(env):
  import mujoco
  bits=mujoco.mjtState.mjSTATE_INTEGRATION
  integration=np.empty(mujoco.mj_stateSize(env.physics.model.ptr,bits),np.float64)
  mujoco.mj_getState(env.physics.model.ptr,env.physics.data.ptr,integration,bits)
  return dict(suite=dict(integration=integration,step_count=env._step_count,
      reset_next_step=env._reset_next_step,task=copy.deepcopy(env._task.__dict__)),from_dm_done=False)


class NormalTaskQueryTest(unittest.TestCase):
  def setUp(self):
    from dm_control import suite
    self.env=suite.load('quadruped','walk',task_kwargs={'random':7919})
    self.addCleanup(self.env.close)
    self.env.reset();self.t=6
    for i in range(self.t):
      ts=self.env.step(np.linspace(-.2,.2,12,dtype=np.float64)*(i+1)/self.t)
    self.assertFalse(ts.first() or ts.last())
    self.reward64=np.asarray(ts.reward)
    self.assertEqual(self.reward64.dtype,np.float64)
    self.reward32=np.asarray(self.reward64,np.float32)
    self.saved=snapshot(self.env)

  def query(self,**changes):
    kwargs=dict(env=self.env,saved_snapshot=self.saved,current_reward_f64=self.reward64,
        current_reward_f32=self.reward32,t=self.t)
    kwargs.update(changes);return q.query_current(**kwargs)

  def test_true_cpu_current_query_is_readonly_and_forbids_state_advancement(self):
    from dm_control.rl import control
    from dm_control.suite import quadruped
    expected_velocity=self.env.physics.torso_velocity().copy()
    expected_upright=np.asarray(self.env.physics.torso_upright()).copy()
    before=pickle.dumps(snapshot(self.env),protocol=5)
    model=self.env.physics.model.to_bytes();saved=pickle.dumps(self.saved,protocol=5)
    with mock.patch.object(control.Environment,'step',side_effect=AssertionError('step forbidden')), \
        mock.patch.object(control.Environment,'reset',side_effect=AssertionError('reset forbidden')), \
        mock.patch.object(quadruped.Physics,'render',side_effect=AssertionError('render forbidden')), \
        mock.patch.object(quadruped.Physics,'forward',side_effect=AssertionError('forward forbidden')):
      result=self.query()
    self.assertEqual(result['version'],q.VERSION)
    self.assertEqual(result['snapshot_hash_format'],'DL-normal-task-query-snapshot-r1')
    self.assertEqual(result['velocity_x'].tobytes(),expected_velocity[0].tobytes())
    self.assertEqual(result['upright'].tobytes(),expected_upright.tobytes())
    self.assertEqual(result['current_reward_f64'].tobytes(),self.reward64.tobytes())
    self.assertEqual(result['current_reward_f32'].tobytes(),self.reward32.tobytes())
    self.assertEqual(result['reward_component_product'].tobytes(),self.reward64.tobytes())
    self.assertEqual(pickle.dumps(snapshot(self.env),protocol=5),before)
    self.assertEqual(self.env.physics.model.to_bytes(),model)
    self.assertEqual(pickle.dumps(self.saved,protocol=5),saved)
    self.assertEqual(result['source_snapshot_sha256'],result['before_snapshot_sha256'])
    self.assertEqual(result['before_snapshot_sha256'],result['after_snapshot_sha256'])
    self.assertEqual(result['task_rng_before_sha256'],result['task_rng_after_sha256'])
    self.assertTrue(result['readonly_full_snapshot_bytes_exact'] and result['readonly_model_bytes_exact'])
    self.assertFalse(result['continuous_prefix_authenticated'] or result['source_model_binding_authenticated'])
    for name in ('environment_steps','resets','restores','forward_calls','render_calls'):self.assertEqual(result[name],0)

  def test_wrong_absolute_clock_or_inactive_snapshot_rejected(self):
    for t in (self.t+1,0,1000,True,np.int64(self.t)):
      with self.subTest(t=t),self.assertRaises(ValueError):self.query(t=t)
    for field in ('from_dm_done','reset_next_step'):
      changed=copy.deepcopy(self.saved)
      if field == 'from_dm_done':changed[field]=True
      else:changed['suite'][field]=True
      with self.subTest(field=field),self.assertRaises(ValueError):self.query(saved_snapshot=changed)

  def test_single_ulp_source_integration_or_rng_mismatch_rejected(self):
    changed=copy.deepcopy(self.saved)
    value=changed['suite']['integration'][0]
    changed['suite']['integration'][0]=np.nextafter(value,np.float64(np.inf))
    self.assertLess(abs(changed['suite']['integration'][0]-value),1e-8)
    with self.assertRaisesRegex(ValueError,'differs byte-exactly'):self.query(saved_snapshot=changed)
    changed=copy.deepcopy(self.saved)
    changed['suite']['task']['_random'].uniform()
    with self.assertRaisesRegex(ValueError,'differs byte-exactly'):self.query(saved_snapshot=changed)

  def test_f64_one_ulp_reward_rejected_even_when_f32_is_identical(self):
    changed=np.asarray(np.nextafter(self.reward64,np.float64(np.inf)))
    self.assertEqual(np.asarray(changed,np.float32).tobytes(),self.reward32.tobytes())
    self.assertNotEqual(changed.tobytes(),self.reward64.tobytes())
    with self.assertRaisesRegex(ValueError,'current F64 reward differs'):self.query(current_reward_f64=changed)
    with self.assertRaises(ValueError):self.query(current_reward_f64=self.reward32)
    with self.assertRaises(ValueError):self.query(current_reward_f32=self.reward64)
    with self.assertRaisesRegex(ValueError,'Typed source'):self.query(current_reward_f64=float(self.reward64))

  def test_query_task_rng_consumption_or_integration_write_rejected(self):
    from dm_control.suite import quadruped
    original=quadruped.Move.get_reward
    def rng(task,physics):
      value=original(task,physics);task.random.uniform();return value
    with mock.patch.object(quadruped.Move,'get_reward',rng),self.assertRaisesRegex(ValueError,'Readonly query changed'):
      self.query()
    # Begin another official live prefix; helper never restores a failed query.
    self.env.reset()
    for _ in range(self.t):ts=self.env.step(np.zeros(12,np.float64))
    self.reward64=np.asarray(ts.reward);self.reward32=np.asarray(self.reward64,np.float32);self.saved=snapshot(self.env)
    def write(task,physics):
      value=original(task,physics)
      physics.data.qvel[0]=np.nextafter(physics.data.qvel[0],np.float64(np.inf));return value
    with mock.patch.object(quadruped.Move,'get_reward',write),self.assertRaisesRegex(ValueError,'Readonly query changed'):
      self.query()

  def test_query_model_write_rejected(self):
    from dm_control.suite import quadruped
    original=quadruped.Move.get_reward
    def write(task,physics):
      value=original(task,physics)
      physics.model.opt.gravity[0]=np.nextafter(physics.model.opt.gravity[0],np.float64(np.inf));return value
    with mock.patch.object(quadruped.Move,'get_reward',write),self.assertRaisesRegex(ValueError,'model/MJB'):
      self.query()

  def test_import_has_no_env_mutation_or_lazy_simulator_model_import(self):
    code="import os,sys; before=dict(os.environ); import analysis.dl_normal_task_query; assert dict(os.environ)==before; assert not any(n.split('.')[0] in ('jax','dm_control','mujoco','OpenGL') for n in sys.modules)"
    result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=30,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',MUJOCO_GL='disable'))
    self.assertEqual(result.returncode,0,result.stderr)


if __name__ == '__main__':unittest.main()
