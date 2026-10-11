"""Exact endpoint/report and owned-timeout contracts, never GPU evidence."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

import numpy as np
from scripts import dl_rep_observation_runner as runner


class TypedEndpointContracts(unittest.TestCase):

  def test_exact_state_and_located_bit_dtype_shape_signedzero_differences(self):
    original = {'params': {'opt': np.array([1., 0.], np.float32)},
        'counters': {'updates': 128}}
    self.assertTrue(runner.typed_differences(original, copy.deepcopy(original))['equal'])
    for change, kind in ((lambda a:a.astype(np.float64), 'dtype'),
        (lambda a:a.reshape(1,2), 'shape'),
        (lambda a:np.array([1., -0.], np.float32), 'bytes'),
        (lambda a:np.nextafter(a, np.float32(np.inf)), 'bytes')):
      with self.subTest(kind=kind):
        changed = copy.deepcopy(original); changed['params']['opt'] = change(changed['params']['opt'])
        actual = runner.typed_differences(original, changed)
        self.assertFalse(actual['equal'])
        self.assertEqual(actual['first_differences'][0]['kind'], kind)
        self.assertEqual(actual['first_differences'][0]['path'], '/params/opt')
        detail = actual['first_differences'][0]
        self.assertEqual(detail['left']['dtype'], 'float32')
        self.assertEqual(len(detail['left']['sha256']), 64)
        if kind == 'bytes':
          self.assertGreater(detail['differing_bytes'], 0)
          self.assertIn(detail['first_flat_index'], (0,1))
          self.assertTrue(detail['floating_diagnostic_available'])
    for value in (-0., False, np.int32(128)):
      changed = copy.deepcopy(original); changed['counters']['updates'] = value
      self.assertFalse(runner.typed_differences(original,changed)['equal'])

  def test_signedzero_BF16_and_overflow_diagnostics_do_not_relax_bytes(self):
    import ml_dtypes
    for dtype in (np.float32, ml_dtypes.bfloat16):
      actual = runner.typed_differences(np.array([0.],dtype),np.array([-0.],dtype))
      self.assertFalse(actual['equal'])
      detail = actual['first_differences'][0]
      self.assertEqual((detail['first_flat_index'],detail['signed_zero_differences']),(0,1))
      self.assertEqual(detail['maximum_absolute_difference'],0.)
      json.dumps(detail,allow_nan=False)
    scalar=runner.typed_differences(0.,-0.)['first_differences'][0]
    self.assertEqual(scalar['signed_zero_differences'],1)
    self.assertNotEqual(scalar['left']['sha256'],scalar['right']['sha256'])
    value = np.finfo(np.float64).max
    detail = runner.typed_differences(np.array(value),np.array(-value))['first_differences'][0]
    self.assertEqual(detail['first_index'],[])
    self.assertIsNone(detail['maximum_absolute_difference'])
    self.assertTrue(detail['derived_diagnostic_nonfinite'])

  def test_container_types_and_bounded_details_are_not_silently_normalized(self):
    self.assertFalse(runner.typed_differences([np.ones(1)],(np.ones(1),))['equal'])
    self.assertFalse(runner.typed_differences({'x':np.ones(1)}, {})['equal'])
    actual = runner.typed_differences({str(i):i for i in range(20)},
        {str(i):i+1 for i in range(20)}, limit=3)
    self.assertEqual((actual['differing_nodes'],len(actual['first_differences'])),(20,3))


class CompletionContracts(unittest.TestCase):

  def fixture(self, folder, equivalent=True, common=True):
    directory = Path(folder)
    runner.write(directory/'budget.json', dict(version=runner.VERSION,
        seconds=runner.SECONDS,actual_git_sha='f'*40,parent_pid=123,max_gpu_devices=1,
        planned_physical_gpu=4,max_gpu_hours=.5))
    runner.write(directory/'guard-authority.json', {'fixture_only':True})
    phases=[]
    for phase in runner.PHASES:
      value=dict(phase=phase,exit_code=0,worker_terminal=True,timed_out=False,
          elapsed_seconds=1.,is_gpu=phase in ('Bwhole','Amin'))
      phases.append(value);runner.write(directory/(phase+'-exit.json'),value)
    checkpoint,complete={},{}
    for mode in ('Bwhole','Amin'):
      (directory/(mode+'-checkpoint.pkl')).write_bytes(b'fixture only; never decode')
      runner.write(directory/(mode+'-complete.json'), {'fixture_only':True})
      checkpoint[mode]=runner.sha(directory/(mode+'-checkpoint.pkl'))
      complete[mode]=runner.sha(directory/(mode+'-complete.json'))
    result=dict(version=runner.VERSION,actual_git_sha='f'*40,complete_paths=2,
        actual_training_updates=256,new_physics_steps=0,new_RGB_frames=0,
        legacy_per_update_full_state_comparison_performed=False,
        new_native_evaluation_performed=False,method_effectiveness_established=False,
        guard_authority_sha256=runner.sha(directory/'guard-authority.json'),
        checkpoint_sha256=checkpoint,complete_receipt_sha256=complete,
        comparisons={m:dict(final_typed_state_equal=equivalent,complete_128_updates=True,
            helper_equivalent=equivalent and common,written_checkpoints_standard_reload_byte_exact=True,
            common_metrics={'passed':common}) for m in ('Bwhole','Amin')},
        endpoint_observation_equivalent=equivalent,guard_passed=equivalent and common)
    runner.write(directory/'result.json',result)
    supervisor=dict(version=runner.VERSION,actual_git_sha='f'*40,parent_pid=123,
        complete_verified=True,exit_code=0,timed_out=False,actual_gpu_devices_used=1,
        elapsed_parent_entry_seconds=10.,budget_sha256=runner.sha(directory/'budget.json'),
        result_sha256=runner.sha(directory/'result.json'),phases=phases)
    runner.write(directory/'supervisor_complete.json',supervisor)
    terminal=directory/'external.json'
    runner.write(terminal,dict(actual_git_sha='f'*40,worker_pid=123,exit_code=0,
        worker_terminal=True,timed_out=False,elapsed_cli_seconds=10.1))
    return directory,terminal

  def test_completed_mismatch_is_retained_without_becoming_guard_pass(self):
    for equivalent,common in ((True,True),(False,True),(True,False)):
      with self.subTest(equivalent=equivalent,common=common), tempfile.TemporaryDirectory() as folder:
        directory,terminal=self.fixture(folder,equivalent,common)
        actual=runner.verify_completion(directory,terminal)
        self.assertTrue(actual['complete_external_exit_verified'])
        self.assertEqual(actual['guard_passed'],equivalent and common)
        self.assertFalse(actual['historical_per_update_full_state_verified'])
        self.assertFalse(actual['method_effectiveness_established'])

  def test_late_missing_exit_and_tampered_endpoint_rejected(self):
    for problem in ('late','exit','checkpoint','unbound_phase'):
      with self.subTest(problem=problem), tempfile.TemporaryDirectory() as folder:
        directory,terminal=self.fixture(folder)
        if problem=='late': runner.write(directory/'late-supervisor.json',{'exit_code':124})
        elif problem=='exit':
          value=runner.read_json(terminal);value['worker_terminal']=False
          terminal.write_text(json.dumps(value),encoding='utf-8')
        elif problem=='checkpoint':(directory/'Amin-checkpoint.pkl').write_bytes(b'changed')
        else:(directory/'Bwhole-exit.json').write_text('{}',encoding='utf-8')
        with self.assertRaises((ValueError,KeyError)):
          runner.verify_completion(directory,terminal)


@unittest.skipUnless(os.name=='posix','Actual owned process-group cleanup requires POSIX')
class OwnedTimeoutContract(unittest.TestCase):

  def test_timeout_kills_only_the_new_worker_group(self):
    actual_popen=subprocess.Popen
    unrelated=actual_popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
    created=[]
    def launch(command,**kwargs):
      process=actual_popen([sys.executable,'-c','import time;time.sleep(30)'],**kwargs)
      created.append(process);return process
    try:
      with tempfile.TemporaryDirectory() as folder:
        output=Path(folder)/'attempt'
        args=types.SimpleNamespace(**{key:'unused' for key in runner.KEYS},resource_record='unused')
        args.output=str(output);args.expected_git='f'*40
        device={'physical_gpu':4,'gpu_uuid':'GPU-fixture'}
        from scripts import dl_resources as resource
        with (mock.patch.object(runner,'clean_binding',return_value={}),
            mock.patch.object(resource,'require_resource',return_value=device),
            mock.patch.object(resource,'command_output'),
            mock.patch.object(runner.subprocess,'Popen',side_effect=launch),
            mock.patch.object(runner,'ENTRY_STARTED',time.monotonic()),
            mock.patch.object(runner,'SECONDS',.15)):
          self.assertEqual(runner.parent(args),124)
        self.assertEqual(len(created),1)
        self.assertIsNotNone(created[0].poll())
        self.assertIsNone(unrelated.poll())
        self.assertEqual(runner.OWNED,set())
        self.assertTrue(runner.read_json(output/'authenticate-exit.json')['timed_out'])
        self.assertFalse((output/'result.json').exists())
        if (output/'supervisor_complete.json').exists():
          self.assertFalse(runner.read_json(output/'supervisor_complete.json')['complete_verified'])
    finally:
      os.killpg(unrelated.pid,signal.SIGKILL);unrelated.wait(timeout=5)
      for process in created:
        if process.poll() is None:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)


if __name__=='__main__':unittest.main()
