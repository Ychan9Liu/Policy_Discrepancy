"""Official CPU quadruped engineering fixtures, never real calibration data.

The frozen independent auditor is loaded by its byte-pinned file interface.
No Dreamer, JAX, renderer, policy, or GPU is initialized. These tests establish
finite-control engineering contracts, not natural-state or scientific effects.
"""
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

os.environ['MUJOCO_GL'] = 'disable'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import dl_task_semantics
from scripts import dl_action_information_physics as physics


class ActionInformationPhysicsTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.audit, cls.dependency = dl_task_semantics.load_auditor(
        ROOT/'analysis/dl_feedback_physics_audit_v1.py')

  def setUp(self):
    from dm_control import suite
    from dm_control.mujoco.engine import Physics
    self.render_patch = mock.patch.object(Physics, 'render',
        side_effect=AssertionError('This CPU fixture must never render'))
    self.render_patch.start()
    self.addCleanup(self.render_patch.stop)
    self.env = suite.load('quadruped', 'walk', task_kwargs={'random': 77})
    self.addCleanup(self.env.close)
    # Genuine official reset/one transition construct fixture initial state;
    # reset is not part of the finite sequence or the conditional replay.
    self.env.reset()
    spec = self.env.action_spec()
    self.env.step(physics.mapping(np.zeros(12, np.float32),spec.minimum,spec.maximum))
    self.saved = self.audit.snapshot(self.env, False)

  def record(self, raw=None, saved=None):
    return physics.run_sequence(self.env, self.saved if saved is None else saved,
        physics.controls(self.env)[3] if raw is None else raw, self.audit)

  def test_frozen_named_controls_and_dtype_ordered_production_mapping(self):
    values = physics.controls(self.env)
    self.assertEqual(values.dtype, np.float32)
    self.assertEqual(values.shape, (7,12))
    expected = np.zeros((7,12), np.float32)
    expected[1] = .25; expected[2] = -.25
    expected[3,[1,10]] = .25; expected[3,[4,7]] = -.25
    expected[4] = -expected[3]
    expected[5,[2,8]] = .25; expected[5,[5,11]] = -.25
    expected[6] = -expected[5]
    self.assertTrue(self.audit.byte_equal(values, expected))
    receipt = physics.verify_model(self.env)
    self.assertEqual(receipt['xml_sha256'], physics.XML_SHA)
    self.assertEqual(receipt['actuator_names'], list(physics.NAMES))
    self.assertEqual(receipt['filter_tau'], .1)
    self.assertEqual(receipt['control_timestep'], .02)
    low, high = self.env.action_spec().minimum, self.env.action_spec().maximum
    raw = np.linspace(-2.3, 2.3, 12, dtype=np.float32)
    original = raw.copy()
    actual = physics.mapping(raw,low,high)
    ordered = (np.asarray(np.clip(raw,-1,1),np.float32)+1)/2*(high-low)+low
    self.assertTrue(self.audit.byte_equal(actual,ordered))
    self.assertTrue(self.audit.byte_equal(raw,original))
    zero = physics.mapping(values[0],low,high)
    self.assertTrue(np.allclose(zero[[1,4,7,10]],.05,rtol=0,atol=1e-15))
    self.assertTrue(np.all(zero[[0,2,3,5,6,8,9,11]] == 0))
    self.assertFalse(np.array_equal(zero,np.zeros(12,np.float64)))
    # Algebraically same all-F64 expression does not have production rounding.
    odd = np.full(12,.1,np.float32)
    ordered = physics.mapping(odd,low,high)
    all_double = (odd.astype(np.float64)+1)/2*(high-low)+low
    self.assertFalse(self.audit.byte_equal(ordered,all_double))

  def test_real_H20_and_complete_stored_applied_replay_preserve_activation(self):
    raw = np.linspace(-1.7,1.7,12,dtype=np.float32)
    original = copy.deepcopy(self.saved)
    original_raw = raw.copy()
    # mjSTATE_INTEGRATION includes actual nonzero filter activation from setup.
    self.assertTrue(np.any(self.env.physics.data.act != 0))
    with mock.patch.object(type(self.env), 'reset', side_effect=AssertionError('No sequence reset')):
      record = self.record(raw)
      self.assertEqual(record['executed_steps'],20)
      self.assertEqual(record['end_type'],'window')
      self.assertEqual(len(record['snapshots']),21)
      self.assertEqual(record['action_executed'].tolist(),[True]*20)
      self.assertEqual(record['snapshot_executed'].tolist(),[True]*21)
      self.assertEqual(record['reward_f64'].dtype,np.float64)
      self.assertEqual(record['reward_f32'].dtype,np.float32)
      self.assertTrue(self.audit.byte_equal(record['raw'][0],raw))
      self.assertTrue(np.all(record['raw'][1:] == 0))
      self.assertTrue(self.audit.byte_equal(raw,original_raw))
      self.assertEqual(self.audit.compare_state(record['snapshots'][0],self.saved),0)
      low, high = self.env.action_spec().minimum, self.env.action_spec().maximum
      self.assertTrue(self.audit.byte_equal(record['applied'][1],
          physics.mapping(np.zeros(12,np.float32),low,high)))
      with mock.patch.object(type(self.env), 'step', wraps=self.env.step) as step:
        receipt = physics.replay_sequence(self.env,record,self.audit)
      self.assertEqual(step.call_count,20)
      for index,call in enumerate(step.call_args_list):
        self.assertTrue(self.audit.byte_equal(call.args[0],record['applied'][index]))
      self.assertEqual(receipt['physics_steps'],20)
      self.assertEqual(receipt['padding_steps'],0)
      self.assertEqual(receipt['maximum_integration_error'],0)
      self.assertEqual(receipt['maximum_restore_error'],0)
      for key in ('float64_reward_byte_exact','float32_reward_encoding_byte_exact',
                  'discount_byte_exact','expected_from_dm_done_exact'):
        self.assertIs(receipt[key],True)
      for key in ('actual_from_dm_wrapper_executed','rendering_executed','model_executed'):
        self.assertIs(receipt[key],False)
    self.assertEqual(self.audit.compare_state(self.saved,original),0)
    self.assertNotIn('jax',sys.modules)

  def test_F64_single_ulp_same_F32_and_applied_state_discount_tamper_fail(self):
    record = self.record()
    damaged = copy.deepcopy(record)
    before = damaged['reward_f64'][3]
    damaged['reward_f64'][3] = np.nextafter(before,np.float64(np.inf))
    self.assertNotEqual(before.tobytes(),damaged['reward_f64'][3].tobytes())
    self.assertEqual(np.float32(before).tobytes(),np.float32(damaged['reward_f64'][3]).tobytes())
    with self.assertRaisesRegex(ValueError,'F64 reward byte replay'):
      physics.replay_sequence(self.env,damaged,self.audit)
    damaged = copy.deepcopy(record)
    damaged['applied'][0,0] = np.nextafter(damaged['applied'][0,0],np.inf)
    with self.assertRaisesRegex(ValueError,'mapping'):
      physics.replay_sequence(self.env,damaged,self.audit)
    damaged = copy.deepcopy(record)
    damaged['snapshots'][1]['suite']['integration'][0] += 1e-5
    with self.assertRaisesRegex(ValueError,'integration'):
      physics.replay_sequence(self.env,damaged,self.audit)
    damaged = copy.deepcopy(record); damaged['discount'][0] = .5
    with self.assertRaises(ValueError):
      physics.replay_sequence(self.env,damaged,self.audit)

  def test_true_terminal_absorbs_without_reset_and_no_fake_snapshots(self):
    with mock.patch.object(type(self.env._task),'get_termination',return_value=0.), \
        mock.patch.object(type(self.env),'reset',side_effect=AssertionError('No terminal reset')):
      with mock.patch.object(type(self.env),'step',wraps=self.env.step) as step:
        record = self.record()
      self.assertEqual(step.call_count,1)
      self.assertEqual(record['executed_steps'],1)
      self.assertEqual(record['end_type'],'true_terminal')
      self.assertEqual(record['action_executed'].tolist(),[True]+[False]*19)
      self.assertEqual(record['snapshot_executed'].tolist(),[True,True]+[False]*19)
      self.assertEqual(len(record['snapshots']),2)
      self.assertTrue(record['is_terminal'][0])
      self.assertTrue(record['is_last'][0])
      self.assertEqual(record['discount'][0],0)
      self.assertTrue(record['snapshots'][-1]['from_dm_done'])
      self.assertTrue(record['snapshots'][-1]['suite']['reset_next_step'])
      self.assertTrue(np.all(record['reward_f64'][1:] == 0))
      self.assertTrue(np.all(record['applied'][1:] == 0))
      with mock.patch.object(type(self.env),'step',wraps=self.env.step) as step:
        receipt = physics.replay_sequence(self.env,record,self.audit)
      self.assertEqual(step.call_count,1)
      self.assertEqual(receipt['physics_steps'],1)
      self.assertEqual(receipt['padding_steps'],19)
      self.assertEqual(receipt['maximum_integration_error'],0)
      damaged = copy.deepcopy(record); damaged['reward_f64'][2] = .1
      with self.assertRaisesRegex(ValueError,'padding'):
        physics.replay_sequence(self.env,damaged,self.audit)
      damaged = copy.deepcopy(record); damaged['snapshots'].append(copy.deepcopy(record['snapshots'][-1]))
      with self.assertRaisesRegex(ValueError,'coverage'):
        physics.replay_sequence(self.env,damaged,self.audit)

  def test_clock_final_time_limit_and_early_last_fail(self):
    start = copy.deepcopy(self.saved); start['suite']['step_count'] = 980
    record = self.record(saved=start)
    self.assertEqual(record['end_type'],'time_limit')
    self.assertTrue(record['is_last'][-1]);self.assertFalse(record['is_terminal'][-1])
    self.assertEqual(physics.replay_sequence(self.env,record,self.audit)['maximum_integration_error'],0)
    start['suite']['step_count'] = 981
    with mock.patch.object(type(self.env),'step',wraps=self.env.step) as step:
      with self.assertRaisesRegex(ValueError,'insufficient clock'):
        self.record(saved=start)
      self.assertEqual(step.call_count,0)
    import dm_env
    with mock.patch.object(type(self.env),'step',return_value=dm_env.TimeStep(
        dm_env.StepType.LAST,np.float64(.5),np.float64(1.),{})):
      with self.assertRaisesRegex(ValueError,'Early time limit'):
        self.record()
    record = self.record()
    damaged = copy.deepcopy(record); damaged['snapshots'][1]['suite']['step_count'] += 1
    with self.assertRaisesRegex(ValueError,'clock'):
      physics.replay_sequence(self.env,damaged,self.audit)
    damaged = copy.deepcopy(record); damaged['is_last'][0] = True
    with self.assertRaisesRegex(ValueError,'early last'):
      physics.replay_sequence(self.env,damaged,self.audit)

  def test_nonfinite_wrong_dtype_snapshot_and_deadline_fail_closed(self):
    record = self.record()
    for key in ('reward_f64','reward_f32','raw','applied','discount'):
      with self.subTest(key=key):
        damaged = copy.deepcopy(record); damaged[key].flat[0] = np.nan
        with self.assertRaises(ValueError):
          physics.replay_sequence(self.env,damaged,self.audit)
    damaged = copy.deepcopy(record); damaged['snapshots'][0]['suite']['integration'][0] = np.inf
    with self.assertRaises(ValueError):
      physics.replay_sequence(self.env,damaged,self.audit)
    for raw in (np.zeros(12,np.float64),np.zeros((1,12),np.float32),np.full(12,np.nan,np.float32)):
      with self.assertRaises(ValueError):
        self.record(raw)
    damaged = copy.deepcopy(self.saved); damaged['from_dm_done'] = True
    with self.assertRaises(ValueError):
      self.record(saved=damaged)
    def deadline():
      raise TimeoutError('fixture hard budget expired')
    with mock.patch.object(type(self.env),'step',wraps=self.env.step) as step:
      with self.assertRaises(TimeoutError):
        physics.run_sequence(self.env,self.saved,np.zeros(12,np.float32),self.audit,deadline)
      self.assertEqual(step.call_count,0)

  def test_actuator_names_filter_bounds_and_task_guards(self):
    model = self.env.physics.model
    original = model.actuator_dynprm.copy()
    model.actuator_dynprm[0,0] = .2
    with self.assertRaisesRegex(ValueError,'filter'):
      physics.controls(self.env)
    model.actuator_dynprm[:] = original
    original = model.actuator_dyntype.copy()
    model.actuator_dyntype[0] = 0
    with self.assertRaisesRegex(ValueError,'filter'):
      physics.verify_model(self.env)
    model.actuator_dyntype[:] = original
    original = model.actuator_ctrlrange.copy()
    model.actuator_ctrlrange[1,1] = 1
    with self.assertRaisesRegex(ValueError,'bounds'):
      physics.verify_model(self.env)
    model.actuator_ctrlrange[:] = original
    with mock.patch.object(type(model),'id2name',return_value='wrong_actuator'):
      with self.assertRaisesRegex(ValueError,'identity/order'):
        physics.controls(self.env)
    self.env._task._desired_speed = 1
    with self.assertRaisesRegex(ValueError,'task or physics'):
      physics.verify_model(self.env)
    self.env._task._desired_speed = .5
    model.opt.timestep = .01
    with self.assertRaisesRegex(ValueError,'Time contract'):
      physics.verify_model(self.env)
    model.opt.timestep = .005
    with mock.patch.object(physics,'XML_SHA','0'*64):
      with self.assertRaisesRegex(ValueError,'XML'):
        physics.verify_model(self.env)
    self.assertEqual(physics.verify_model(self.env)['xml_sha256'],physics.XML_SHA)

  def test_snapshot_restored_task_identity_must_match_before_any_step(self):
    damaged = copy.deepcopy(self.saved)
    damaged['suite']['task']['_desired_speed'] = 1.
    with mock.patch.object(type(self.env),'step',wraps=self.env.step) as step:
      with self.assertRaisesRegex(ValueError,'task|model'):
        self.record(saved=damaged)
      self.assertEqual(step.call_count,0)
    # Replaying an externally serialized record must apply the same guard
    # after restore, even when the ambient official task is currently valid.
    self.audit.restore(self.env,self.saved)
    record = self.record()
    damaged = copy.deepcopy(record)
    damaged['snapshots'][0]['suite']['task']['_desired_speed'] = 1.
    with mock.patch.object(type(self.env),'step',wraps=self.env.step) as step:
      with self.assertRaisesRegex(ValueError,'task|model'):
        physics.replay_sequence(self.env,damaged,self.audit)
      self.assertEqual(step.call_count,0)


if __name__ == '__main__':
  unittest.main()
