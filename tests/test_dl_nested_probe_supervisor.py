"""Stdlib subprocess/JSON-chain engineering fixtures for the nested parent.

These tests read no research episode arrays and perform no readout fits.
Fabricated JSON chains exercise rejection/structure only. A generic verified
callback is never reported as real provenance or eight real input reads.
POSIX tests cover whole-entry alarms and only-owned descendant cleanup;
Windows certifies a directly owned simple child, not a Linux process tree.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_nested_probe_supervisor as supervisor

FIXTURE_GIT = '0' * 40


def shaped(dimensions, value=0.):
  if not dimensions:
    return value
  return [shaped(dimensions[1:], value) for _ in range(dimensions[0])]


def fit(rows, rule):
  return dict(regularization_lambda=rule, training_rows=rows, heldout_rows=16,
      eigen=dict(all_eigenvalues_retained=True, cutoff_applied=False),
      ridge_equation_max_abs_error=0.)


def fixture_nested():
  """Fabricated complete schema only; no numeric fitting or science conclusion."""
  folds = []
  for episode in supervisor.EPISODES:
    training = [e for e in supervisor.EPISODES if e != episode]
    inner = []
    for heldout in training:
      candidates = []
      for rule in supervisor.LAMBDAS:
        branches = {name: dict(choices=shaped([4, 4], 0),
            predicted_centered_reward=shaped([4, 4, 7]), fit=fit(96, rule))
            for name in ('q', 'p')}
        candidates.append(dict(lambda_value=rule, interfaces=branches))
      inner.append(dict(validation_episode=heldout,
          training_episodes=[e for e in training if e != heldout], training_rows=96,
          validation_rows=16, preprocess=dict(q={}, p={}), candidates=candidates))
    folds.append(dict(heldout_episode=episode, training_episodes=training,
        training_rows=112, heldout_rows=16, inner_folds=inner,
        outer_candidate_grid_evaluated=False, selected_lambda=1.,
        common_lambda_selection=dict(selected_lambda=1., q_p_selection_shared=True,
            branch_weights=[.5, .5]),
        candidate_scores=[dict(lambda_value=rule, **{'lambda': rule})
                          for rule in supervisor.LAMBDAS],
        outer_fits={name: fit(112, 1.) for name in ('q', 'p')},
        outer_interfaces=dict(q={}, p={})))
  nested = dict(version=supervisor.VERSION, episodes=list(supervisor.EPISODES),
      states_per_episode=4, samples_per_state=4, controls=7, h_dimensions=4096,
      latent_shape=[32, 32], lambda_grid=list(supervisor.LAMBDAS),
      tie_tolerance=1e-10, numeric_absolute_tolerance=1e-10,
      eigen_cutoff_applied=False, eigendecompositions=128, lambda_evaluations=464,
      q_p_selection_shared=True, equal_effective_degrees_of_freedom_claimed=False,
      bootstrap_draws=2000, statistical_seed=20261021, percentile_method='linear',
      input_arrays_unchanged=True, calibration_only=True, readout_fitted=True,
      source_provenance_verified_by_stats=False, real_gate_authority=False,
      eligible_method_training=False, training_authorized=False,
      proceed_short_training=False, method_effectiveness_established=False,
      dreamer_optimizer_updates=0, blind_arrays_read=False, model_calls=0,
      environment_calls=0, current_rgb_increment_over_complete_history_established=False,
      continuing_kl_compression_effect_established=False,
      outer_folds=folds, selected_common_lambdas=[1.] * 8,
      interfaces={name: dict(choices=shaped([8, 4, 4], 0),
          predicted_centered_reward=shaped([8, 4, 4, 7]), selected_reward=shaped([8, 4, 4]),
          finite_library_regret=shaped([8, 4, 4]), episode_selected_reward=shaped([8]))
          for name in ('q', 'p')},
      constant=dict(control_by_episode=[0] * 8, state_selected_reward=shaped([8, 4])),
      comparisons={name: dict(lower95=0., upper95=0., episode_values=[0.] * 8)
                   for name in ('q_minus_p', 'q_minus_constant')},
      selected_procedure_status='selected_procedure_exploratory_not_supported')
  return nested


def fixture_chain(directory, transform=None):
  """Write one invented chain, then optionally break one bound assertion."""
  output = Path(directory)
  sources = {name: supervisor.sha(ROOT / name) for name in supervisor.SOURCE_FILES}
  budget = dict(version=supervisor.VERSION, actual_git_sha=FIXTURE_GIT, seconds=120,
      planned_numeric_episode_reads=8, actual_numeric_episode_reads=None,
      planned_worker_boundary=supervisor.PLANNED_BOUNDARY,
      supervisor_sha256=sources['scripts/dl_nested_probe_supervisor.py'])
  plan = dict(version=supervisor.VERSION, actual_git_sha=FIXTURE_GIT,
      **supervisor.BOUNDARY, seconds=120, hostname='lyg0326', platform='linux',
      numpy_version='1.26.4', threads=1, episodes=list(supervisor.EPISODES),
      positions=list(supervisor.POSITIONS), lambdas=list(supervisor.LAMBDAS),
      fixed_inner_lambda_selection_permitted=True,
      additional_parameter_search_permitted=False, outer_selection_permitted=False,
      current_rgb_independent_cue_claim_permitted=False, source_git_sha=supervisor.SOURCE_GIT,
      source_json_sha256=supervisor.SOURCE_PINS, source_files_sha256=sources,
      source_external_control_sufficient=True, source_external_delta=.2,
      input_artifacts=[dict(episode=e, path=f'episode_{e:03}.npz', sha256='0' * 64,
          h_sha256='0' * 64, latent_sha256='0' * 64, cost_sha256='0' * 64)
          for e in supervisor.EPISODES])
  result = dict(version=supervisor.VERSION, actual_git_sha=FIXTURE_GIT,
      **supervisor.BOUNDARY, source_inputs_unchanged=True, cost_bytes_match_original=True,
      no_prior_real_result_rejudged=True, no_parameters_or_checkpoints_loaded=True,
      source_external_control_sufficient=True, source_external_delta=.2,
      elapsed_parent_entry_seconds=.1, nested=fixture_nested())
  complete = dict(version=supervisor.VERSION, actual_git_sha=FIXTURE_GIT,
      **supervisor.BOUNDARY, source_files_sha256=sources, complete_outer_folds=8,
      complete_inner_folds=56, eigendecompositions=128, lambda_evaluations=464)
  if transform:
    transform(budget, plan, result, complete)
  supervisor.write_json(output / 'budget.json', budget)
  plan['budget_sha256'] = supervisor.sha(output / 'budget.json')
  supervisor.write_json(output / 'plan.json', plan)
  result.update(budget_sha256=plan['budget_sha256'], plan_sha256=supervisor.sha(output / 'plan.json'))
  supervisor.write_json(output / 'result.json', result)
  complete.update(budget_sha256=plan['budget_sha256'], plan_sha256=result['plan_sha256'],
      result_sha256=supervisor.sha(output / 'result.json'))
  supervisor.write_json(output / 'nested_probe_complete.json', complete)
  return output


class NestedProbeSupervisorTest(unittest.TestCase):

  def test_real_verifier_requires_fixed_worker_before_launch(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'rejected-fake-worker'
      marker = Path(directory) / 'must-not-execute'
      command = self.command(f'from pathlib import Path; Path({str(marker)!r}).write_text("BAD")')
      with self.assertRaisesRegex(ValueError, 'fixed nested worker command'):
        supervisor.supervise(command, output, FIXTURE_GIT, verify=supervisor.verify_completion)
      self.assertFalse(output.exists())
      self.assertFalse(marker.exists())

  @unittest.skipIf(os.name == 'nt', 'POSIX CLI bootstrap alarm')
  def test_cli_alarm_interrupts_blocked_argument_parsing(self):
    # This only shortens the native timer in a fixture process. The research
    # CLI still has its fixed 120s; no real worker or dataset is launched.
    with tempfile.TemporaryDirectory() as directory:
      marker = Path(directory) / 'late-parser-return'
      script = (
          'import argparse,runpy,signal,time; from pathlib import Path; '
          'native_timer=signal.setitimer; '
          'signal.setitimer=lambda which,seconds,interval=0: native_timer(which,min(seconds,.2),interval); '
          f'argparse.ArgumentParser.parse_args=lambda *a,**k: (time.sleep(3),Path({str(marker)!r}).write_text("BAD"))[0]; '
          f'runpy.run_path({str(ROOT / "scripts/dl_nested_probe_supervisor.py")!r},run_name="__main__")')
      started = time.monotonic()
      process = subprocess.run(self.command(script), stdout=subprocess.PIPE,
          stderr=subprocess.PIPE, timeout=3)
      self.assertEqual(process.returncode, 124)
      self.assertLess(time.monotonic() - started, 2)
      self.assertFalse(marker.exists())
  def command(self, script):
    executable = getattr(sys, '_base_executable', sys.executable) if os.name == 'nt' else sys.executable
    return [executable, '-c', script]

  def invoke(self, script, output, seconds=3, verify=None):
    return supervisor.supervise(self.command(script), output, FIXTURE_GIT,
        seconds=seconds, verify=verify)

  def assert_no_real_authority(self, receipt):
    self.assertIsNone(receipt['real_dataset_reads'])
    self.assertIsNone(receipt['actual_numeric_episode_reads'])
    self.assertIs(receipt['source_provenance_verified'], False)
    self.assertIs(receipt['real_artifact_chain_verified'], False)
    self.assertIsNone(receipt['verified_worker_boundary'])
    for key in ('training_authorized', 'method_evidence', 'real_gate_authority',
                'eligible_method_training', 'proceed_short_training',
                'method_effectiveness_established'):
      self.assertIs(receipt[key], False)

  def test_generic_success_has_unknown_actual_reads_and_planning_only(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'success'
      receipt = self.invoke('import os; print(os.environ["DL_NESTED_PROBE_PARENT"],flush=True)', output)
      self.assertEqual(receipt['exit_code'], 0)
      self.assertIs(receipt['worker_terminal'], True)
      self.assertIs(receipt['complete_verified'], False)
      self.assert_no_real_authority(receipt)
      budget = json.loads((output / 'budget.json').read_text())
      self.assertEqual(budget['planned_numeric_episode_reads'], 8)
      self.assertIsNone(budget['actual_numeric_episode_reads'])
      self.assertNotIn('real_dataset_reads', budget)
      self.assertNotIn('source_provenance_verified', budget)
      self.assertIn(str(os.getpid()), (output / 'nested_probe.log').read_text())

  def test_generic_verified_callback_does_not_become_real_provenance(self):
    with tempfile.TemporaryDirectory() as directory:
      receipt = self.invoke('raise SystemExit(0)', Path(directory) / 'verified',
          verify=lambda *_: dict(fixture_only=True, worker_boundary=supervisor.BOUNDARY))
      self.assertEqual(receipt['exit_code'], 0)
      self.assertIs(receipt['complete_verified'], True)
      self.assert_no_real_authority(receipt)
      # A real-chain verifier cannot grant eight-read provenance to a generic
      # subprocess, even if it could fabricate structurally complete JSON.
      with mock.patch.object(supervisor.subprocess, 'Popen') as start:
        with self.assertRaisesRegex(ValueError, 'fixed nested worker command'):
          self.invoke('raise SystemExit(0)', Path(directory) / 'forged-command',
              verify=supervisor.verify_completion)
        start.assert_not_called()

  def test_failed_worker_and_rejected_chain_keep_logs(self):
    with tempfile.TemporaryDirectory() as directory:
      verify = mock.Mock(side_effect=AssertionError('Nonzero worker reached verify'))
      failed = Path(directory) / 'worker-failed'
      receipt = self.invoke('print("FAIL-EVIDENCE",flush=True); raise SystemExit(7)', failed, verify=verify)
      self.assertEqual(receipt['exit_code'], 7)
      verify.assert_not_called()
      self.assertIn('FAIL-EVIDENCE', (failed / 'nested_probe.log').read_text())
      self.assert_no_real_authority(receipt)
      rejected = self.invoke('raise SystemExit(0)', Path(directory) / 'rejected',
          verify=lambda *_: (_ for _ in ()).throw(ValueError('hash rejected')))
      self.assertEqual(rejected['exit_code'], 1)
      self.assertIn('hash rejected', rejected['failure']['message'])
      self.assert_no_real_authority(rejected)

  def test_existing_output_is_not_reused_or_overwritten(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'existing'
      output.mkdir()
      original = output / 'evidence'
      original.write_bytes(b'KEEP')
      marker = Path(directory) / 'not-started'
      with self.assertRaises(FileExistsError):
        self.invoke(f'from pathlib import Path;Path({str(marker)!r}).touch()', output)
      self.assertEqual(original.read_bytes(), b'KEEP')
      self.assertFalse(marker.exists())
      self.assertEqual(list(output.iterdir()), [original])

  def test_parent_wait_timeout_cleans_owned_direct_worker(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'timeout'
      marker = Path(directory) / 'late'
      receipt = self.invoke('import time;from pathlib import Path;time.sleep(10);'
          f'Path({str(marker)!r}).touch()', output, seconds=.15)
      self.assertEqual(receipt['exit_code'], 124)
      self.assertIs(receipt['timed_out'], True)
      self.assertIs(receipt['worker_terminal'], True)
      self.assertFalse(marker.exists())
      self.assertLess(receipt['terminal_elapsed_seconds'], 3)
      self.assert_no_real_authority(receipt)

  def test_stdlib_verifier_schema_and_hash_rejections_only(self):
    with tempfile.TemporaryDirectory() as directory:
      output = fixture_chain(directory)
      # This is a fabricated JSON structural fixture, never a real worker run.
      result = supervisor.verify_completion(output, FIXTURE_GIT)
      self.assertEqual(result['verified_outer_folds'], 8)
      with (output / 'result.json').open('a') as stream:
        stream.write(' ')
      with self.assertRaisesRegex(ValueError, 'hash mismatch'):
        supervisor.verify_completion(output, FIXTURE_GIT)
    for change in ('old_synthetic', 'partial', 'read_budget', 'wrong_split', 'bad_counts', 'bad_leaf'):
      def transform(budget, plan, result, complete):
        if change == 'old_synthetic':
          result['synthetic_only'] = True
        elif change == 'partial':
          result['nested']['outer_folds'].pop()
        elif change == 'read_budget':
          budget['real_dataset_reads'] = 8
        elif change == 'wrong_split':
          result['nested']['outer_folds'][0]['inner_folds'][0]['training_episodes'].append(9)
        elif change == 'bad_counts':
          complete['lambda_evaluations'] = 463
        elif change == 'bad_leaf':
          result['nested']['interfaces']['q']['choices'][0][0][0] = True
      with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
        fixture_chain(directory, transform)
        with self.assertRaises(ValueError):
          supervisor.verify_completion(directory, FIXTURE_GIT)

  @unittest.skipIf(os.name == 'nt', 'POSIX whole-entry verification alarm')
  def test_alarm_includes_post_worker_verification(self):
    with tempfile.TemporaryDirectory() as directory:
      receipt = self.invoke('print("FINISHED",flush=True)', Path(directory) / 'slow-verify',
          seconds=.2, verify=lambda *_: time.sleep(10))
      self.assertEqual(receipt['exit_code'], 124)
      self.assertIs(receipt['timed_out'], True)
      self.assertLess(receipt['terminal_elapsed_seconds'], 3)
      self.assert_no_real_authority(receipt)

  @unittest.skipIf(os.name == 'nt', 'POSIX whole-entry final publication alarm')
  def test_alarm_includes_final_success_write(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'slow-final'
      original = supervisor.write_json
      delayed = []
      def slow_write(path, value):
        if Path(path).name == 'supervisor_complete.json' and value['complete_verified'] and not delayed:
          delayed.append(True)
          time.sleep(10)
        return original(path, value)
      with mock.patch.object(supervisor, 'write_json', side_effect=slow_write):
        receipt = self.invoke('raise SystemExit(0)', output, seconds=.2,
            verify=lambda *_: dict(fixture_only=True))
      stored = json.loads((output / 'supervisor_complete.json').read_text())
      self.assertEqual(delayed, [True])
      self.assertEqual(receipt['exit_code'], 124)
      self.assertEqual(stored['exit_code'], 124)
      self.assertIs(stored['complete_verified'], False)
      self.assertLess(receipt['terminal_elapsed_seconds'], 3)
      self.assert_no_real_authority(stored)

  @unittest.skipIf(os.name == 'nt', 'POSIX owned descendants and unrelated sentinel')
  def test_group_cleanup_after_failed_or_timeout_leader_preserves_sentinel(self):
    for mode in ('failed', 'timeout'):
      with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
        marker = Path(directory) / 'unrelated-sentinel'
        sentinel = subprocess.Popen(self.command('import time;from pathlib import Path;'
            f'time.sleep(.5);Path({str(marker)!r}).touch();time.sleep(10)'),
            start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        late = Path(directory) / 'owned-descendant-survived'
        script = 'import time;from pathlib import Path;time.sleep(.5);' + f'Path({str(late)!r}).touch();time.sleep(10)'
        leader = ('import subprocess,sys,time;'
            f'subprocess.Popen([sys.executable,"-c",{script!r}]);' +
            ('raise SystemExit(7)' if mode == 'failed' else 'time.sleep(10)'))
        try:
          receipt = self.invoke(leader, Path(directory) / mode, seconds=.25 if mode == 'timeout' else 3)
          self.assertEqual(receipt['exit_code'], 124 if mode == 'timeout' else 7)
          self.assertIs(receipt['owned_group_cleanup_requested'], True)
          self.assertIs(receipt['worker_terminal'], True)
          time.sleep(.7)
          self.assertFalse(late.exists())
          self.assertIsNone(sentinel.poll())
          self.assertTrue(marker.exists())
          self.assert_no_real_authority(receipt)
        finally:
          if sentinel.poll() is None:
            sentinel.terminate()
          sentinel.wait(timeout=3)


if __name__ == '__main__':
  unittest.main()
