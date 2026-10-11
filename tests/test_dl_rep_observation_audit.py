"""Constructor fixtures only: no scientific source, checkpoint or GPU access.

Tiny host trees and fabricated producer receipts test independent endpoint
comparison and receipt failure modes, never actual optimizer equivalence.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

from analysis import dl_rep_observation_audit as audit
from scripts import dl_rep_observation_runner as runner


def write(path, value):
  path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding='utf-8')
  return audit.sha(path)


def checkpoint(change=False):
  return dict(params={'opt/step/value': np.array(74621, np.int32),
      'posterior/kernel': np.array([1., 0. if change else -0.], np.float32),
      'opt/history': np.array([.25, .5], np.float32),
      'normalizer/state': (np.array([2.], np.float64),)},
      counters=dict(updates=74621, batches=74621, actions=18769,
          eval_actions=1001, eval_seed=3404107006))


class Envelope:
  """Fixed synthetic four-checkpoint envelope, bound through real verifier."""
  def __init__(self, directory, *, endpoint_negative=False, metric_negative=False):
    self.root = Path(directory)/'primary'; self.root.mkdir()
    self.original = Path(directory)/'synthetic-original'; self.original.mkdir()
    self.git = 'a'*40
    self.external = Path(directory)/'synthetic-external.json'
    self.references, self.learning, self.receipts = {}, {}, {}
    self.pins = {name:audit.sha(audit.ROOT/name) for name in (
        'scripts/dl_rep_observation_runner.py', 'scripts/dl_rep_observation_guard.py',
        'docs/DL_REP_OBSERVATION_GUARD.md')}
    self.own = {name:audit.sha(audit.ROOT/name) for name in audit.OWN_FILES}
    immutable, comparisons, cp_pins, complete_pins, anchor_fields = {}, {}, {}, {}, {}
    counters = dict(updates=74493, batches=74493, actions=18769,
        eval_actions=1001, eval_seed=3404107006)
    for mode, anchor, group in (('Bwhole', 'old_B', 'B'), ('Amin', 'native_A', 'A')):
      old = checkpoint(); new = checkpoint(endpoint_negative and mode == 'Amin')
      oldpath = self.original/(anchor+'-checkpoint.pkl')
      newpath = self.root/(mode+'-checkpoint.pkl')
      oldpath.write_bytes(pickle.dumps(old, protocol=5)); newpath.write_bytes(pickle.dumps(new, protocol=5))
      oldpin, newpin = audit.sha(oldpath), audit.sha(newpath)
      immutable[str(oldpath.resolve())] = oldpin
      anchor_fields[anchor+'_checkpoint_path'] = str(oldpath.resolve())
      anchor_fields[anchor+'_checkpoint_sha256'] = oldpin
      ref = dict(version='DL-reset-learning-r1' if group == 'B' else 'DL-reset-allrep-learning-r1',
          group=group, updates=128, source_checkpoint_digest='1'*64,
          initial_state_digest='2'*64, initial_params_digest='3'*64,
          calibration_arrays_digest='4'*64, binding={'batch_digest':'5'*64},
          restored_counters=counters.copy(), saved_source_counters=counters.copy(),
          optimizer_counter=dict(key='opt/step/value', dtype='int32', shape=[],
              initial=74493, final=74621, updates=128),
          final_state_digest=audit.tree_digest(old), final_params_digest=audit.tree_digest(old['params']),
          receipts=[dict(update=i+1, seed_counter=74493+i, train_seed=[i, i+1],
              counters={**counters, 'updates':74494+i, 'batches':74494+i},
              metrics={'opt/updates':74494+i}, metrics_digest='6'*64) for i in range(128)])
      learning = copy.deepcopy(ref)
      learning.update(mode=mode, layout_version='DL-reset-learning-r1',
          version='DL-rep-observation-guard-candidate-r0',
          all_fixed128_updates_completed=True, input_reference_values_matched=True,
          historical_per_update_fullstate_carry_outs_equivalence_claimed=False,
          rep_gradient_stopped=mode == 'Amin',
          standard_final_checkpoint_reload_byte_exact=True,
          historical_anchor_standard_reload_byte_exact=True,
          final_state_digest=audit.tree_digest(new), final_params_digest=audit.tree_digest(new['params']))
      for i, row in enumerate(learning['receipts']):
        equal = not (metric_negative and mode == 'Amin' and i == 9)
        typed = equal if mode == 'Bwhole' else None
        row.update(optimizer_step=74494+i, common_metrics={'opt/updates':74494+i},
            common_metrics_digest='7'*64, full_state_digest=audit.tree_digest(new),
            full_params_digest=audit.tree_digest(new['params']), carry_digest='8'*64, outs_digest='9'*64,
            common_metrics_equal=equal, common_metrics_json_exact=equal,
            common_metrics_typed_exact=typed,
            historical_per_update_fullstate_carry_outs_anchor_available=False)
      common = dict(passed=all(r['common_metrics_equal'] for r in learning['receipts']),
          per_update_equal=[r['common_metrics_equal'] for r in learning['receipts']],
          rows=[dict(update=r['update'], equal=r['common_metrics_equal'],
              json_numeric_exact=r['common_metrics_json_exact'], typed_exact=r['common_metrics_typed_exact'])
              for r in learning['receipts']])
      same = audit.compare_trees(new, old)['equal']
      learning.update(common_metrics_comparison=common, equivalent=same and common['passed'])
      self.references[mode], self.learning[mode] = ref, learning
      self.receipts[mode] = dict(version=audit.VERSION, actual_git_sha=self.git, phase=mode,
          learning=learning, checkpoint_sha256=newpin, guard_authority_sha256=None,
          new_native_evaluation_performed=False)
      comparisons[mode] = dict(final_typed_state_equal=same,
          endpoint_differences=runner.typed_differences(new, old),
          new_full_state_sha256=audit.tree_digest(new), historical_full_state_sha256=audit.tree_digest(old),
          common_metrics=common, helper_equivalent=learning['equivalent'],
          written_checkpoints_standard_reload_byte_exact=True, complete_128_updates=True)
      cp_pins[mode] = newpin
    # Arbitrary immutable bytes are authenticated without interpreting them as
    # arrays. They are deliberately not a valid NPZ and must never be decoded.
    opaque = self.original/'synthetic-source.npz'; opaque.write_bytes(b'not an NPZ: do not decode')
    immutable[str(opaque.resolve())] = audit.sha(opaque)
    legacy = dict(original_19_producer_files_unchanged=True,
        old_full_exit_and_independent_audit_authenticated=True,
        original_B_learning=self.references['Bwhole'], immutable_inputs_sha256=immutable)
    legacy_pin = write(self.root/'old-authenticated.json', legacy)
    authority = dict(version=audit.VERSION, actual_git_sha=self.git,
        endpoint_only=True, legacy_per_update_full_state_available=False,
        new_native_evaluation_performed=False, old_authentication_sha256=legacy_pin,
        old_B_learning=self.references['Bwhole'], current_A_learning=self.references['Amin'],
        native_independent_verification=dict(complete_published_parent_verified=True,
            complete_external_exit_verified=True),
        immutable_inputs_sha256=immutable, **anchor_fields)
    self.authority = authority
    auth_pin = write(self.root/'guard-authority.json', authority)
    for mode in audit.MODES:
      self.receipts[mode]['guard_authority_sha256'] = auth_pin
      complete_pins[mode] = write(self.root/(mode+'-complete.json'), self.receipts[mode])
    self.result = dict(version=audit.VERSION, actual_git_sha=self.git,
        comparisons=comparisons, complete_paths=2, actual_training_updates=256,
        new_physics_steps=0, new_RGB_frames=0,
        legacy_per_update_full_state_comparison_performed=False,
        new_native_evaluation_performed=False, method_effectiveness_established=False,
        endpoint_observation_equivalent=all(v['final_typed_state_equal'] for v in comparisons.values()),
        guard_passed=all(v['final_typed_state_equal'] and v['common_metrics']['passed'] and
            v['helper_equivalent'] for v in comparisons.values()), guard_authority_sha256=auth_pin,
        complete_receipt_sha256=complete_pins, checkpoint_sha256=cp_pins)
    self.result_pin = write(self.root/'result.json', self.result)
    self.budget = dict(version=audit.VERSION, actual_git_sha=self.git, seconds=runner.SECONDS,
        parent_pid=111, max_gpu_devices=1, planned_physical_gpu=4, max_gpu_hours=.5,
        source_files_sha256=self.pins)
    budget_pin = write(self.root/'budget.json', self.budget)
    phases = []
    for i, phase in enumerate(runner.PHASES):
      row = dict(phase=phase, worker_pid=222+i, worker_terminal=True, exit_code=0,
          timed_out=False, is_gpu=phase in audit.MODES, elapsed_seconds=1.)
      phases.append(row); write(self.root/(phase+'-exit.json'), row)
    self.supervisor = dict(version=audit.VERSION, actual_git_sha=self.git,
        parent_pid=111, complete_verified=True, exit_code=0, timed_out=False,
        actual_gpu_devices_used=1, elapsed_parent_entry_seconds=10., phases=phases,
        budget_sha256=budget_pin, result_sha256=self.result_pin)
    write(self.root/'supervisor_complete.json', self.supervisor)
    write(self.external, dict(actual_git_sha=self.git, worker_pid=111, exit_code=0,
        worker_terminal=True, timed_out=False, elapsed_cli_seconds=11.))

  def authenticate(self):
    # Git commands are the only substituted authentication primitive. All
    # endpoint hashes, producer verifier, receipt bindings and four payload
    # comparisons remain actual constructor-file operations.
    with mock.patch.object(audit, 'code_binding', return_value=self.own):
      return audit.authenticate_primary(self.root, self.external, self.git)

  def perform(self):
    with mock.patch.object(audit, 'no_model_imports'):
      return audit.audit_authenticated(self.authenticate())


class AuditTests(unittest.TestCase):
  def test_independent_rawbytes_signedzero_container_dtype_and_bf16(self):
    import ml_dtypes
    value = {'x':np.array([0., -0.], np.float32), 'b':np.array([1], ml_dtypes.bfloat16), 'v':(1,)}
    self.assertTrue(audit.compare_trees(value, copy.deepcopy(value))['equal'])
    for changed in ({**value, 'x':np.array([0., 0.], np.float32)},
        {**value, 'x':np.array([0., -0.], np.float64)}, {**value, 'v':[1]},
        {**value, 'x':np.array([[0., -0.]], np.float32)}, {**value, 'extra':1}):
      self.assertFalse(audit.compare_trees(value, changed)['equal'])
      self.assertNotEqual(audit.tree_digest(value), audit.tree_digest(changed))
    for bad in (np.array([b'1234'], dtype='V4'), np.array([object()], object), np.array([np.nan])):
      with self.assertRaises(ValueError): audit.numeric(bad)
    self.assertFalse(audit.exact_json({'x':-0.}, {'x':0.}))
    self.assertFalse(audit.exact_json({'x':True}, {'x':1}))

  def test_payload_pin_precedes_pickle_and_official_opt_contract(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)/'fake.pkl'; path.write_bytes(pickle.dumps(checkpoint(), protocol=5))
      with mock.patch.object(audit.pickle, 'load') as decode:
        with self.assertRaisesRegex(ValueError, 'before decode'): audit.load_checkpoint(path, 'f'*64)
        decode.assert_not_called()
      for mutate in (lambda cp:cp['params'].update({'opt/step/value':np.array(74621., np.float32)}),
          lambda cp:cp['params'].update({'opt/step/value':np.array([74621], np.int32)}),
          lambda cp:cp['counters'].update(updates=True)):
        cp = checkpoint(); mutate(cp); path.write_bytes(pickle.dumps(cp, protocol=5))
        with mock.patch.object(audit, 'no_model_imports'):
          with self.assertRaises(ValueError): audit.load_checkpoint(path, audit.sha(path))

  def test_positive_and_negative_endpoint_audits_decode_only_four_host_trees(self):
    for negative in (False, True):
      with self.subTest(negative=negative), tempfile.TemporaryDirectory() as directory:
        env = Envelope(directory, endpoint_negative=negative)
        original = audit.pickle.load
        with mock.patch.object(audit.pickle, 'load', wraps=original) as decode:
          result = env.perform()
        self.assertEqual(decode.call_count, 4)
        self.assertEqual(result['status'], 'passed')
        self.assertTrue(result['passed'])
        self.assertIs(result['endpoint_observation_equivalent'], not negative)
        self.assertIs(result['guard_passed'], not negative)
        self.assertEqual(result['actual_auditor_optimizer_updates'], 0)
        self.assertFalse(result['source_arrays_decoded'])
        self.assertFalse(result['independent_optimizer_or_historical_per_update_replay'])
        if negative:
          diff = result['comparisons']['Amin']['independent_difference']
          self.assertEqual(diff['first_differences'][0]['kind'], 'bytes')

  def test_common_metric_negative_is_valid_completion_not_equivalence(self):
    with tempfile.TemporaryDirectory() as directory:
      result = Envelope(directory, metric_negative=True).perform()
      self.assertTrue(result['endpoint_observation_equivalent'])
      self.assertFalse(result['guard_passed'])
      self.assertEqual(result['status'], 'passed')

  def test_bound_keys_counters_summary_reload_and_boolean_spoofs_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      env = Envelope(directory)
      mutations = (
          lambda l:l['receipts'][12]['train_seed'].__setitem__(0, 44),
          lambda l:l['receipts'][0].update(optimizer_step=74494.),
          lambda l:l['receipts'][0].update(seed_counter=True),
          lambda l:l['receipts'][0]['counters'].update(actions=0),
          lambda l:l['common_metrics_comparison']['rows'][3].update(equal=False),
          lambda l:l.update(historical_anchor_standard_reload_byte_exact=False),
          lambda l:l['receipts'][0].update(historical_per_update_fullstate_carry_outs_anchor_available=True),
          lambda l:l['optimizer_counter'].update(initial=74493.))
      for mutation in mutations:
        receipt = copy.deepcopy(env.receipts['Bwhole']); mutation(receipt['learning'])
        with self.subTest(mutation=mutation), self.assertRaises(ValueError):
          audit.validate_receipt('Bwhole', receipt, env.authority, env.result['comparisons']['Bwhole'])

  def test_hash_tamper_and_end_of_audit_immutable_race_are_not_negative_results(self):
    with tempfile.TemporaryDirectory() as directory:
      env = Envelope(directory); bound = env.authenticate()
      path = env.root/'Amin-checkpoint.pkl'; path.write_bytes(path.read_bytes()+b'tamper')
      with mock.patch.object(audit, 'no_model_imports'):
        with self.assertRaisesRegex(ValueError, 'before decode'): audit.audit_authenticated(bound)
    with tempfile.TemporaryDirectory() as directory:
      env = Envelope(directory); bound = env.authenticate(); original = audit.compare_trees
      def change_after_comparison(*args, **kwargs):
        compared = original(*args, **kwargs)
        (env.original/'synthetic-source.npz').write_bytes(b'changed during audit')
        return compared
      with mock.patch.object(audit, 'no_model_imports'), mock.patch.object(audit, 'compare_trees', side_effect=change_after_comparison):
        with self.assertRaisesRegex(ValueError, 'Original input changed'): audit.audit_authenticated(bound)

  def test_authentication_rejects_failed_primary_late_exit_and_legacy_tamper(self):
    for kind in ('failed', 'late', 'external', 'legacy'):
      with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
        env = Envelope(directory)
        if kind == 'failed':
          value = copy.deepcopy(env.supervisor); value['complete_verified'] = False
          write(env.root/'supervisor_complete.json', value)
        elif kind == 'late': (env.root/'late-supervisor.json').write_text('{}')
        elif kind == 'external':
          value = audit.read_json(env.external); value['exit_code'] = 1; write(env.external, value)
        else: (env.root/'old-authenticated.json').write_text('{}')
        with self.assertRaises(ValueError): env.authenticate()

  def test_published_cpu_completion_authenticates_external_exit_even_for_negative_guard(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory); result = dict(version=audit.IMPLEMENTATION, status='passed', passed=True,
          auditor_git_sha='a'*40, complete_paths=2, actual_primary_training_updates=256,
          checkpoint_payloads_decoded=4, actual_gpu_devices_used=0, actual_auditor_optimizer_updates=0,
          endpoint_observation_equivalent=False, guard_passed=False)
      budget = dict(version=audit.VERSION, implementation=audit.IMPLEMENTATION, actual_git_sha='a'*40,
          parent_pid=111, seconds=audit.SECONDS, actual_gpu_devices_used=0)
      supervisor = dict(version=audit.VERSION, implementation=audit.IMPLEMENTATION,
          actual_git_sha='a'*40, parent_pid=111, complete_verified=True, exit_code=0,
          worker_terminal=True, timed_out=False, actual_gpu_devices_used=0,
          elapsed_parent_entry_seconds=2., budget_sha256=write(root/'budget.json', budget),
          worker_exit_sha256=write(root/'worker_exit_code.txt', 0), audit_sha256=write(root/'audit.json', result))
      write(root/'supervisor_complete.json', supervisor)
      external = root/'external.json'; terminal = dict(actual_git_sha='a'*40, worker_pid=111,
          exit_code=0, worker_terminal=True, timed_out=False, elapsed_cli_seconds=3.)
      write(external, terminal)
      verified = audit.verify_completion(root, external)
      self.assertTrue(verified['complete_external_exit_verified']); self.assertFalse(verified['guard_passed'])
      terminal['exit_code'] = 124; write(external, terminal)
      with self.assertRaises(ValueError): audit.verify_completion(root, external)
      write(root/'supervisor-receipt-pending.json', {})
      with self.assertRaises(ValueError): audit.verify_completion(root)

  def test_late_receipt_before_publish_or_after_rename_cannot_complete(self):
    for clock in ((11.,), (9.,11.)):
      with self.subTest(clock=clock), tempfile.TemporaryDirectory() as directory:
        with mock.patch.object(audit.time, 'monotonic', side_effect=clock):
          self.assertEqual(audit.publish_receipt(directory, {'exit_code':0}, 10.), 124)
        self.assertFalse((Path(directory)/'supervisor_complete.json').exists())
        self.assertTrue((Path(directory)/'late-supervisor.json').exists())

  def test_standalone_audit_imports_no_jax_model_or_simulator(self):
    code = ('from analysis import dl_rep_observation_audit as a; '
        'from scripts import dl_rep_observation_runner; '
        'a.no_model_imports(); import sys; '
        'assert not any(n == "mujoco" or n.startswith("dm_control") for n in sys.modules)')
    result = subprocess.run([sys.executable, '-c', code], cwd=audit.ROOT, timeout=30,
        capture_output=True, text=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES':''})
    self.assertEqual(result.returncode, 0, result.stderr)

  @unittest.skipUnless(os.name == 'posix', 'Meaningful owned process groups require POSIX')
  def test_posix_timeout_and_signals_kill_owned_descendants_not_unrelated_processes(self):
    # The real auditor's spawn/cleanup/ALRM/TERM functions run in an isolated
    # tiny CPU parent. An unrelated sentinel must remain alive in every case.
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory); sentinel = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'], start_new_session=True)
      try:
        for mode in ('timeout', 'SIGALRM', 'SIGTERM'):
          pidfile = root/(mode+'.pid')
          childcode = ('import subprocess,sys,time,pathlib; '
              'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]); '
              'pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(30)')
          if mode == 'timeout':
            parentcode = ('import os,sys,time; from analysis import dl_rep_observation_audit as a; '
                'cmd=[sys.executable,"-c",sys.argv[1],sys.argv[2]]; '
                'log=open(sys.argv[2]+".log","w"); '
                'result=a.run_owned(cmd,log,time.monotonic()+.8,os.environ.copy()); '
                'assert result["timed_out"] and result["worker_terminal"] and result["exit_code"]==124; '
                'assert os.path.exists(sys.argv[2]); sys.exit(124)')
          else:
            parentcode = ('import os,signal,sys,time; from analysis import dl_rep_observation_audit as a; '
                'cmd=[sys.executable,"-c",sys.argv[1],sys.argv[2]]; '
                'child=a.spawn(cmd,time.monotonic()+5); deadline=time.monotonic()+5; '
                'exec("while not os.path.exists(sys.argv[2]) and time.monotonic()<deadline: time.sleep(.01)"); '
                'assert os.path.exists(sys.argv[2]); '
                'signal.signal(getattr(signal,sys.argv[3]),a.expire); os.kill(os.getpid(),getattr(signal,sys.argv[3]))')
          proc = subprocess.run([sys.executable, '-c', parentcode, childcode, str(pidfile), mode],
              cwd=audit.ROOT, timeout=10, capture_output=True, text=True)
          self.assertEqual(proc.returncode, 124, proc.stderr)
          pid = int(pidfile.read_text()); deadline = time.monotonic()+2
          alive = True
          while time.monotonic() < deadline:
            try:
              os.kill(pid, 0)
              stat = Path(f'/proc/{pid}/stat')
              alive = not (stat.exists() and stat.read_text().split()[2] == 'Z')
            except ProcessLookupError: alive = False
            if not alive: break
            time.sleep(.02)
          self.assertFalse(alive, 'Owned descendant survived cleanup '+mode)
          self.assertIsNone(sentinel.poll(), 'Unrelated sentinel was disturbed')
      finally:
        os.killpg(sentinel.pid, __import__('signal').SIGKILL); sentinel.wait(timeout=5)


if __name__ == '__main__': unittest.main()
