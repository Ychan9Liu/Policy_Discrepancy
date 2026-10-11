"""Stdlib subprocess and fabricated B-chain engineering tests, no science run."""
import copy
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
from scripts import dl_production_reward_supervisor as parent

FIXTURE_GIT = '0' * 40


def shaped(dimensions, value=0.):
  if not dimensions: return value
  return [shaped(dimensions[1:], value) for _ in range(dimensions[0])]


def metric():
  return dict(per_initial=shaped([8, 4, 4]), state_values=shaped([8, 4]),
              episode_values=shaped([8]), primary=0., lower95=0., upper95=0.)


def analysis_fixture():
  blocks = dict(first32=[0], first64=[0, 1], pooled128=[0, 1, 2, 3], last64=[2, 3],
                block0=[0], block1=[1], block2=[2], block3=[3])
  subsets = {}
  for name, indices in blocks.items():
    subset = dict(blocks=indices, paths=32 * len(indices),
        path_indices=[32 * b + i for b in indices for i in range(32)],
        interfaces={branch: dict(choices=shaped([8, 4, 4], 0),
            **{k: metric() for k in ('bc_mse', 'plugin_centered_mse', 'mc_correction',
                'plugin_level_mse', 'selected_reward', 'finite_library_regret')}) for branch in ('q', 'p')},
        mse_gaps={k: metric() for k in ('p_minus_q', 'constant_minus_q')},
        utility_gaps={k: metric() for k in ('q_minus_p', 'q_minus_constant')})
    subset.update({k: shaped([8, 4, 4, 2, 7]) for k in ('cumulative_mean',
        'centered_cumulative_mean', 'centered_path_sample_variance', 'centered_variance_over_n')})
    subsets[name] = subset
  def reliable(keys, limit, family):
    values = {k: dict(epsilon=0., components=dict(nested32_64=0., nested64_128=0.,
        independent_halves=0., block_t_margin=0.), stable=True, point=0., lower95=0., upper95=0.,
        lower95_minus_epsilon=0., upper95_plus_epsilon=0., positive_margin_robust=False,
        negative_margin_robust=True, relative_positive_margin_limit=.2) for k in keys}
    return dict(metrics=values, epsilon_limit=limit, mc_stable=True,
        necessary_directions_robust=False, status='finite_native_' + family + '_not_supported')
  return dict(version=parent.VERSION, episodes=list(parent.EPISODES), positions=list(parent.POSITIONS),
      prediction_shape=[8, 4, 4, 2, 128, 7, 20], truth_shape=[8, 4, 7],
      controls=['zero', 'all+', 'all-', 'lift-LR+', 'lift-LR-', 'extend-diag+', 'extend-diag-'],
      branches=['q', 'p'], horizon=20, blocks=4, paths_per_block=32, future_paths=128,
      initial_samples=4, axes=['episode', 'state', 'initial_sample', 'branch', 'future_path', 'control', 'step'],
      primary_subset='pooled128', negative_bc_mse_clipped=False, initial_samples_averaged_before_loss=False,
      future_paths_independence_verified_by_stats=False, source_provenance_verified_by_stats=False,
      source_or_model_execution_verified_by_stats=False, tie_tolerance=1e-10,
      bootstrap_draws=2000, statistical_seed=20261025, percentile_method='linear',
      mc_error_bound_claimed=False, inputs_unchanged=True, calibration_only=True,
      model_calls=0, physics_steps=0, optimizer_updates=0, gpu_devices_used=0,
      blind_arrays_read=False, records_decoded=False, current_rgb_increment_established=False,
      active_rep_compression_effect_established=False, method_effectiveness_established=False,
      real_gate_authority=False, eligible_method_training=False, training_authorized=False,
      proceed_short_training=False, decision='stop_after_fixed_production_reward_diagnostic',
      constant=dict(folds=[dict(heldout_episode=e, training_states=28,
          training_episodes=[x for x in parent.EPISODES if x != e], centered_cost_vector=shaped([7]))
          for e in parent.EPISODES], centered_cost_vectors=shaped([8, 7]), control_by_episode=[0] * 8,
          state_mse=shaped([8, 4]), state_selected_reward=shaped([8, 4]), correction=0.),
      subsets=subsets,
      mse_reliability=reliable(('p_minus_q', 'constant_minus_q'), .0004, 'mean'),
      utility_reliability=reliable(('q_minus_p', 'q_minus_constant'), .002, 'choice'),
      primary_status='finite_native_mean_not_supported')


def fixture_chain(directory, change=None):
  """Only a metadata/hash fixture. It never becomes a real-worker receipt."""
  output, dataset = directory / 'output', directory / 'source'
  output.mkdir(); dataset.mkdir()
  pins = {}
  for name in parent.SOURCE_PINS:
    (dataset / name).write_bytes(b'{}')
    pins[name] = parent.sha(dataset / name)
  resource_file = directory / 'resource.json'; resource_file.write_bytes(b'{}')
  dimensions = dict(episode=([], 'int32'), positions=([4], 'int32'), h=([4, 4096], 'float64'),
      latent=([4, 4, 2, 32, 32], 'float64'), reward_f64=([4, 15, 20], 'float64'),
      first_commands=([4, 15, 12], 'float32'), h_native=([4, 4096], 'bfloat16'),
      latent_native=([4, 4, 2, 32, 32], 'bfloat16'), controls=([4, 7, 12], 'float32'),
      true_cost=([4, 7], 'float64'))
  inputs, outputs = [], []
  for episode in parent.EPISODES:
    name = f'episode_{episode:03}.npz'
    (dataset / name).write_bytes(b'FAKE-NUMERIC-ARCHIVE')
    (output / name).write_bytes(b'FAKE-PREDICTION-ARCHIVE')
    inputs.append(dict(episode=episode, path=name, sha256=parent.sha(dataset / name),
        typed_hashes={k: dict(shape=d, dtype_name=t, sha256='0' * 64) for k, (d, t) in dimensions.items()}))
    outputs.append(dict(episode=episode, path=name, sha256=parent.sha(output / name),
        reward_typed_sha256='0' * 64, root_keys_typed_sha256='0' * 64, sample_keys_typed_sha256='0' * 64))
  oracle_path = output / 'first_chunk_oracle.npz'; oracle_path.write_bytes(b'FAKE-ORACLE')
  sources = {name: '0' * 64 for name in parent.SOURCE_FILES}
  boundary = dict(parent.NO_AUTHORITY, source_provenance_verified=True, real_dataset_reads=8,
      records_decoded=False, gpu_devices_used=1, physics_steps=0, render_calls=0, optimizer_updates=0,
      exploratory_calibration_only=True, actor_calls=0, initial_latents_resampled=False)
  identity = dict(version=parent.VERSION, actual_git_sha=FIXTURE_GIT)
  budget = dict(identity, seconds=1800, planned_numeric_episode_reads=8,
      actual_numeric_episode_reads=None, planned_physical_gpu=4, planned_gpu_devices=1,
      supervisor_sha256=sources['scripts/dl_production_reward_supervisor.py'])
  plan = dict(identity, **boundary, source_files_sha256=sources, seconds=1800,
      episodes=list(parent.EPISODES), positions=list(parent.POSITIONS), hostname='lyg0326', platform='linux',
      horizon=20, initial_samples=4, branches=['q', 'p'], controls=7, future_paths=128,
      blocks=4, paths_per_block=32, sample_chunk=8, latent_reward_steps=4587520,
      private_seed=20261024, statistical_seed=20261025, oracle_atol=1e-6, oracle_rtol=1e-5,
      stats_audit_atol=1e-10, source_dataset=str(dataset), source_git_sha=parent.SOURCE_GIT,
      source_json_sha256=pins, input_artifacts=inputs,
      model_metadata=dict(parent.MODEL_PINS, checkpoint_counters=dict(updates=74493, batches=74493,
          actions=18769, eval_actions=1001, eval_seed=3404107006)),
      source_model_provenance=dict(parent.MODEL_PINS, checkpoint_counters=dict(updates=74493, batches=74493,
          actions=18769, eval_actions=1001, eval_seed=3404107006)), checkpoint_state_sha256='0' * 64,
      params_unchanged=True, gate_changed=False, physics_provenance_reused=True,
      resource=dict(physical_gpu=4, logical_server='sv3', exclusive=True, gpu_uuid='GPU-FIXTURE',
          compute_uuid='GPU-FIXTURE', egl_uuid='GPU-FIXTURE', cuda_visible_devices='GPU-FIXTURE',
          resource_source_path=str(resource_file), resource_source_sha256=parent.sha(resource_file),
          runtime_compute=dict(physical_gpu=4, compute_uuid=['GPU-FIXTURE'], graphics_verified=False, pid=1)))
  result = dict(identity, **boundary, source_inputs_unchanged=True, state_unchanged=True,
      checkpoint_state_unchanged=True, params_unchanged=True, counters_unchanged=True,
      stopped_after_fixed_object=True, checkpoint_state_sha256='0' * 64, source_files_sha256=sources,
      evidence_artifacts=outputs, analysis=analysis_fixture(), elapsed_parent_entry_seconds=1.,
      first_chunk_oracle=dict(path=oracle_path.name, sha256=parent.sha(oracle_path),
          atol=1e-6, rtol=1e-5, compared_points=8960, failed_points=0,
          finite_point_oracle_passed=True, full_model_independently_reproduced=False))
  complete = dict(identity, **boundary, source_files_sha256=sources, complete_episodes=8,
      complete_states=32, initial_samples=4, branches=2, controls=7, future_paths=128,
      horizon=20, complete_chunks=512, latent_reward_steps=4587520,
      primary_status='finite_native_mean_not_supported')
  if change: change(budget, plan, result, complete)
  parent.write_json(output / 'first_chunk_oracle_check.json', result['first_chunk_oracle'])
  parent.write_json(output / 'budget.json', budget)
  plan['budget_sha256'] = parent.sha(output / 'budget.json'); parent.write_json(output / 'plan.json', plan)
  result.update(budget_sha256=plan['budget_sha256'], plan_sha256=parent.sha(output / 'plan.json'))
  parent.write_json(output / 'result.json', result)
  complete.update(budget_sha256=plan['budget_sha256'], plan_sha256=result['plan_sha256'],
                  result_sha256=parent.sha(output / 'result.json'))
  parent.write_json(output / 'production_reward_complete.json', complete)
  return output, pins


class ProductionRewardSupervisorTest(unittest.TestCase):
  def command(self, script):
    executable = getattr(sys, '_base_executable', sys.executable) if os.name == 'nt' else sys.executable
    return [executable, '-c', script]

  def invoke(self, script, output, seconds=3, verify=None):
    return parent.supervise(self.command(script), output, FIXTURE_GIT, seconds=seconds, verify=verify)

  def unknown(self, receipt):
    self.assertIsNone(receipt['real_dataset_reads'])
    self.assertIsNone(receipt['actual_numeric_episode_reads'])
    self.assertIsNone(receipt['actual_gpu_devices_used'])
    self.assertIs(receipt['source_provenance_verified'], False)
    self.assertIs(receipt['real_artifact_chain_verified'], False)
    self.assertIsNone(receipt['verified_worker_boundary'])
    for key in parent.NO_AUTHORITY: self.assertIs(receipt[key], False)

  def test_generic_success_preserves_cuda_env_but_has_no_real_authority(self):
    with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-FIXTURE'):
      output = Path(directory) / 'success'
      receipt = self.invoke('import os;print(os.environ["CUDA_VISIBLE_DEVICES"]);'
          'print(os.environ["DL_PRODUCTION_REWARD_PARENT"])', output,
          verify=lambda *_: dict(worker_boundary=dict(source_provenance_verified=True, real_dataset_reads=8)))
      self.assertEqual(receipt['exit_code'], 0); self.assertIs(receipt['worker_terminal'], True)
      self.assertIs(receipt['complete_verified'], True); self.unknown(receipt)
      log = (output / 'production_reward.log').read_text()
      self.assertIn('GPU-FIXTURE', log); self.assertIn(str(os.getpid()), log)
      budget = json.loads((output / 'budget.json').read_text())
      self.assertEqual(budget['planned_physical_gpu'], 4); self.assertEqual(budget['planned_gpu_devices'], 1)
      self.assertNotIn('real_dataset_reads', budget); self.assertNotIn('source_provenance_verified', budget)
      with mock.patch.object(parent.subprocess, 'Popen') as start:
        with self.assertRaisesRegex(ValueError, 'fixed production worker'):
          self.invoke('pass', Path(directory) / 'forged', verify=parent.verify_completion)
        start.assert_not_called()

  def test_failure_rejection_and_existing_output_are_preserved(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'failed'
      verify = mock.Mock(side_effect=AssertionError('Nonzero worker verified'))
      receipt = self.invoke('print("FAILURE-EVIDENCE");raise SystemExit(7)', output, verify=verify)
      self.assertEqual(receipt['exit_code'], 7); verify.assert_not_called(); self.unknown(receipt)
      self.assertIn('FAILURE-EVIDENCE', (output / 'production_reward.log').read_text())
      before = {p.name: p.read_bytes() for p in output.iterdir()}
      with self.assertRaises(FileExistsError): self.invoke('pass', output)
      self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})
      reject = self.invoke('pass', Path(directory) / 'rejected', verify=lambda *_: False)
      self.assertEqual(reject['exit_code'], 1); self.unknown(reject)

  def test_timeout_direct_child_terminal_no_late_work(self):
    with tempfile.TemporaryDirectory() as directory:
      marker = Path(directory) / 'late'
      receipt = self.invoke('import time;from pathlib import Path;time.sleep(10);'
          f'Path({str(marker)!r}).touch()', Path(directory) / 'timeout', seconds=.15)
      self.assertEqual(receipt['exit_code'], 124); self.assertIs(receipt['worker_terminal'], True)
      self.assertLess(receipt['terminal_elapsed_seconds'], 3); self.assertFalse(marker.exists()); self.unknown(receipt)

  def test_fixed_command_admission(self):
    output = (ROOT / 'fixture-output').resolve()
    command = [sys.executable, str(ROOT / 'scripts/dl_production_reward.py'), '--dataset', str(ROOT),
        '--checkpoint', str(ROOT / 'fixture.pkl'), '--config', str(ROOT / 'fixture.yaml'),
        '--resource-record', str(ROOT / 'fixture.json'), '--output', str(output),
        '--expected-git', FIXTURE_GIT, '--worker']
    parent.real_worker_command(command, output, FIXTURE_GIT)
    for index, replacement in ((1, str(ROOT / 'scripts/dl_nested_probe.py')), (2, '--other'),
                               (11, str(ROOT)), (13, '1' * 40), (14, '--fake')):
      bad = command.copy(); bad[index] = replacement
      with self.assertRaises(ValueError): parent.real_worker_command(bad, output, FIXTURE_GIT)

  def test_b_chain_hash_schema_and_partial_rejections(self):
    with tempfile.TemporaryDirectory() as directory:
      output, pins = fixture_chain(Path(directory))
      with mock.patch.object(parent, 'SOURCE_PINS', pins), mock.patch.object(parent, 'git_binding') as binding:
        verified = parent.verify_completion(output, FIXTURE_GIT)
        self.assertEqual(verified['verification_kind'], 'production_reward_artifact_chain_schema_and_hashes')
        binding.assert_called_once()
        with (output / 'episode_009.npz').open('ab') as stream: stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'prediction artifact'):
          parent.verify_completion(output, FIXTURE_GIT)
    for kind in ('nested_version', 'claimed_budget', 'partial_source', 'partial_prediction',
                 'wrong_gpu', 'partial_steps', 'missing_dependency', 'changed_state', 'wrong_status',
                 'partial_subset', 'duplicate_output', 'wrong_oracle'):
      def change(budget, plan, result, complete):
        if kind == 'nested_version': result['version'] = 'DL-nested-probe-r1'
        elif kind == 'claimed_budget': budget['real_dataset_reads'] = 8
        elif kind == 'partial_source': plan['input_artifacts'].pop()
        elif kind == 'partial_prediction': result['evidence_artifacts'].pop()
        elif kind == 'wrong_gpu': plan['resource']['runtime_compute']['physical_gpu'] = 5
        elif kind == 'partial_steps': complete['complete_chunks'] = 511
        elif kind == 'missing_dependency': del plan['source_files_sha256']['scripts/dl_predictive_mc.py']
        elif kind == 'changed_state': result['state_unchanged'] = False
        elif kind == 'wrong_status': result['analysis']['primary_status'] = 'finite_native_mean_exploratory_supported'
        elif kind == 'partial_subset': del result['analysis']['subsets']['last64']
        elif kind == 'duplicate_output': result['evidence_artifacts'][1] = copy.deepcopy(result['evidence_artifacts'][0])
        elif kind == 'wrong_oracle': result['first_chunk_oracle']['full_model_independently_reproduced'] = True
      with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
        output, pins = fixture_chain(Path(directory), change)
        with mock.patch.object(parent, 'SOURCE_PINS', pins), mock.patch.object(parent, 'git_binding'):
          with self.assertRaises(ValueError): parent.verify_completion(output, FIXTURE_GIT)

  def test_positive_directions_do_not_override_absolute_mc_limit(self):
    analysis = analysis_fixture()
    reliability = analysis['mse_reliability']
    for name, value in reliability['metrics'].items():
      analysis['subsets']['pooled128']['mse_gaps'][name].update(primary=.01, lower95=.005, upper95=.015)
      value.update(point=.01, lower95=.005, upper95=.015, epsilon=.001,
          components=dict(nested32_64=.001, nested64_128=0., independent_halves=0., block_t_margin=0.),
          stable=False, positive_margin_robust=True, negative_margin_robust=False,
          lower95_minus_epsilon=.005 - .001, upper95_plus_epsilon=.015 + .001)
    reliability.update(mc_stable=False, necessary_directions_robust=False, status='MC-inconclusive')
    analysis['primary_status'] = 'MC-inconclusive'
    # Positive point/CI and 20% margins still cannot establish the necessary
    # conjunction while absolute MC epsilon exceeds the fixed .0004 gate.
    parent.verify_analysis(analysis)
    reliability['necessary_directions_robust'] = True
    with self.assertRaisesRegex(ValueError, 'finite status changed'):
      parent.verify_analysis(analysis)

  @unittest.skipIf(os.name == 'nt', 'POSIX whole-entry verification/finalwrite alarm')
  def test_verification_and_finalwrite_are_bounded(self):
    for mode in ('verify', 'finalwrite'):
      with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / mode
        original, delayed = parent.write_json, []
        def slow_write(path, value):
          if Path(path).name == 'supervisor_complete.json' and value['complete_verified'] and not delayed:
            delayed.append(True); time.sleep(10)
          original(path, value)
        verify = (lambda *_: time.sleep(10)) if mode == 'verify' else (lambda *_: dict(fixture_only=True))
        with mock.patch.object(parent, 'write_json', side_effect=slow_write if mode == 'finalwrite' else original):
          receipt = self.invoke('pass', output, seconds=.2, verify=verify)
        self.assertEqual(receipt['exit_code'], 124); self.assertLess(receipt['terminal_elapsed_seconds'], 3)
        stored = json.loads((output / 'supervisor_complete.json').read_text())
        self.assertIs(stored['complete_verified'], False); self.unknown(stored)

  @unittest.skipIf(os.name == 'nt', 'POSIX owned descendant cleanup/unrelated sentinel')
  def test_cleanup_owns_only_its_group_after_failure_and_timeout(self):
    for mode in ('failed', 'timeout'):
      with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
        marker, late = Path(directory) / 'sentinel', Path(directory) / 'owned-descendant'
        sentinel = subprocess.Popen(self.command('import time;from pathlib import Path;'
            f'time.sleep(.5);Path({str(marker)!r}).touch();time.sleep(10)'), start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        descendant = ('import time;from pathlib import Path;time.sleep(.5);'
                      f'Path({str(late)!r}).touch();time.sleep(10)')
        leader = ('import subprocess,sys,time;' + f'subprocess.Popen([sys.executable,"-c",{descendant!r}]);' +
                  ('raise SystemExit(7)' if mode == 'failed' else 'time.sleep(10)'))
        try:
          receipt = self.invoke(leader, Path(directory) / mode, seconds=.25 if mode == 'timeout' else 3)
          self.assertEqual(receipt['exit_code'], 7 if mode == 'failed' else 124)
          time.sleep(.7)
          self.assertFalse(late.exists()); self.assertIsNone(sentinel.poll()); self.assertTrue(marker.exists())
          self.unknown(receipt)
        finally:
          if sentinel.poll() is None: sentinel.terminate()
          sentinel.wait(timeout=3)

  @unittest.skipIf(os.name == 'nt', 'POSIX bootstrap parse-time alarm')
  def test_entry_parse_operation_is_under_alarm_before_supervise(self):
    code = (ROOT / 'scripts/dl_production_reward_supervisor.py').read_text()
    code = code.replace('1800 - (time.monotonic() - ENTRY_STARTED)', '.15 - (time.monotonic() - ENTRY_STARTED)')
    # Inject a stalled parser before main reaches host or any real worker.
    code = code.replace('args = parser.parse_args()', 'time.sleep(10); args = parser.parse_args()')
    script = ('exec(compile(' + repr(code) + ', ' + repr(str(ROOT / 'scripts/dl_production_reward_supervisor.py')) +
              ', "exec"), {"__name__":"__main__","__file__":' +
              repr(str(ROOT / 'scripts/dl_production_reward_supervisor.py')) + '})')
    started = time.monotonic()
    result = subprocess.run(self.command(script), capture_output=True, timeout=3)
    self.assertEqual(result.returncode, 124); self.assertLess(time.monotonic() - started, 3)

  @unittest.skipIf(os.name == 'nt', 'POSIX entry-phase metadata helper group cleanup')
  def test_entry_alarm_cleans_metadata_helper_before_model_worker_exists(self):
    with tempfile.TemporaryDirectory() as directory:
      marker = Path(directory) / 'helper-survived'
      helper = ('import time;from pathlib import Path;time.sleep(.5);' +
                f'Path({str(marker)!r}).touch();time.sleep(10)')
      code = ('import signal,sys;from scripts import dl_production_reward_supervisor as p;'
          'signal.signal(signal.SIGALRM,p._entry_expire);signal.setitimer(signal.ITIMER_REAL,.15);'
          f'p._owned_output([sys.executable,"-c",{helper!r}])')
      result = subprocess.run(self.command(code), cwd=ROOT, capture_output=True, timeout=3)
      self.assertEqual(result.returncode, 124)
      time.sleep(.7)
      self.assertFalse(marker.exists())


if __name__ == '__main__':
  unittest.main()
