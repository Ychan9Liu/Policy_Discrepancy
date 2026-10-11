"""DL-nested-probe-r1: whole-entry bounded stdlib parent for real cal8.

Only the authenticated numerical worker may report eight real input reads.
Planning metadata and generic subprocess fixtures carry no provenance or
scientific completion authority. No model, simulator, numerical library or
dataset array is imported by this parent. Cleanup owns only its new POSIX
process group; Windows fixtures certify the directly owned child only.
"""
import time
ENTRY_STARTED = time.monotonic()
import os
import signal


def _entry_expire(signum, frame):
  # Before a worker is owned there is no cleanup obligation. Interrupt even a
  # blocked argument/help/path operation without depending on stdout/stderr.
  os._exit(124)


_CLI_ENTRY_ALARM = __name__ == '__main__' and os.name == 'posix'
if _CLI_ENTRY_ALARM:
  signal.signal(signal.SIGALRM, _entry_expire)
  signal.setitimer(signal.ITIMER_REAL, max(1e-6, 120 - (time.monotonic() - ENTRY_STARTED)))

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import socket
import subprocess
import sys

VERSION = 'DL-nested-probe-r1'
ROOT = Path(__file__).resolve().parents[1]
EPISODES = (9, 14, 15, 18, 20, 24, 30, 31)
POSITIONS = (1, 300, 600, 900)
LAMBDAS = (1., .1, .01, .001)
SOURCE_GIT = 'f1ae4d6513e81ed680992b64ba8ebade7d256d52'
SOURCE_FILES = ('scripts/dl_nested_probe.py', 'scripts/dl_nested_probe_stats.py',
                'scripts/dl_nested_probe_supervisor.py', 'docs/DL_NESTED_PROBE.md')
SOURCE_PINS = {
    'plan.json': '474bad6adf6558b7a47c40711546dc4cf2adc870684b8cff2766a802605f44d6',
    'result.json': '06df6efa5f9ce84b428921b59fa508df8badd1dd54e30335bf2d3a273e585fad',
    'action_information_complete.json': '28481a749476be0b3e8c05f28e719cb22e4cfffb638e56c04d8f86128026d37e',
    'budget.json': 'ddc8f43a3f4b57c11858be61f4c31315484c8c42b1e887f48391e71d077e2a68',
    'supervisor_complete.json': 'a920a0191429c7ffd05af5a897a30839da47d046d9126a1002a4997928dc1ea3',
    'independent-numeric-audit-v1.json': 'cbb99e43ad72a6960d86e5044455a18fecc38f068172defe60deb0d82f8678e5'}
BOUNDARY = dict(synthetic_only=False, exploratory_calibration_only=True,
    source_provenance_verified=True, actor_applicable=False, method_evidence=False,
    proceed_short_training=False, decision='stop_after_fixed_nested_probe',
    real_gate_authority=False, eligible_method_training=False, training_authorized=False,
    method_effectiveness_established=False, real_dataset_reads=8, blind_arrays_read=False,
    model_calls=0, physics_steps=0, optimizer_updates=0, gpu_devices_used=0,
    current_rgb_increment_over_complete_history_established=False,
    continuing_kl_compression_effect_established=False)
PLANNED_BOUNDARY = {k: v for k, v in BOUNDARY.items()
                    if k not in ('source_provenance_verified', 'real_dataset_reads')}
UNKNOWN_ACTUAL_BOUNDARY = dict(BOUNDARY,
    source_provenance_verified=False, real_dataset_reads=None)


def require(condition, message):
  if not condition:
    raise ValueError(message)


def sha(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
  with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(value, stream, indent=2, allow_nan=False)
    stream.write('\n')


def boundary(value):
  for key, expected in BOUNDARY.items():
    require(key in value and type(value[key]) is type(expected) and value[key] == expected,
            f'Nested real-cal8 authority boundary changed: {key}')


def finite_tree(value):
  """Reject non-standard JSON NaN/Infinity anywhere without loading NumPy."""
  if isinstance(value, dict):
    for item in value.values():
      finite_tree(item)
  elif isinstance(value, list):
    for item in value:
      finite_tree(item)
  elif type(value) is float:
    require(math.isfinite(value), 'Nonfinite numeric result')


def shape(value, dimensions, leaf=None):
  """Small structured result shape audit, not an independent numerical fit."""
  if not dimensions:
    if leaf is not None:
      require(leaf(value), 'Invalid result leaf')
    return
  require(type(value) is list and len(value) == dimensions[0], 'Incomplete result shape')
  for child in value:
    shape(child, dimensions[1:], leaf)


def verify_nested(nested):
  require(nested['version'] == VERSION and nested['episodes'] == list(EPISODES),
          'Wrong nested statistics identity')
  for key, value in dict(states_per_episode=4, samples_per_state=4, controls=7,
      h_dimensions=4096, latent_shape=[32, 32], lambda_grid=list(LAMBDAS),
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
      continuing_kl_compression_effect_established=False).items():
    require(key in nested and type(nested[key]) is type(value) and nested[key] == value,
            'Incomplete/changed nested statistics: ' + key)
  require(set(nested['interfaces']) == {'q', 'p'} and set(nested['comparisons']) == {
      'q_minus_p', 'q_minus_constant'}, 'Partial nested comparison')
  choices = lambda x: type(x) is int and 0 <= x < 7
  numeric = lambda x: type(x) in (int, float) and math.isfinite(x)
  for name in ('q', 'p'):
    interface = nested['interfaces'][name]
    shape(interface['choices'], [8, 4, 4], choices)
    shape(interface['predicted_centered_reward'], [8, 4, 4, 7], numeric)
    shape(interface['selected_reward'], [8, 4, 4], numeric)
    shape(interface['finite_library_regret'], [8, 4, 4], numeric)
    shape(interface['episode_selected_reward'], [8], numeric)
  shape(nested['constant']['control_by_episode'], [8], choices)
  shape(nested['constant']['state_selected_reward'], [8, 4], numeric)
  folds = nested['outer_folds']
  require(type(folds) is list and len(folds) == 8, 'Partial outer folds')
  require(nested['selected_common_lambdas'] == [x['selected_lambda'] for x in folds],
          'Selected lambda summary differs from actual outer folds')
  for episode, fold in zip(EPISODES, folds):
    training = [e for e in EPISODES if e != episode]
    require(fold['heldout_episode'] == episode and fold['training_episodes'] == training and
            fold['training_rows'] == 112 and fold['heldout_rows'] == 16 and
            fold['outer_candidate_grid_evaluated'] is False and
            type(fold['selected_lambda']) is float and fold['selected_lambda'] in LAMBDAS,
            'Outer split/selected-rule boundary changed')
    selection = fold['common_lambda_selection']
    require(selection['selected_lambda'] == fold['selected_lambda'] and
            selection['q_p_selection_shared'] is True and selection['branch_weights'] == [.5, .5],
            'q/p lambda selection is not common')
    candidates = fold['candidate_scores']
    require(len(candidates) == 4 and [c['lambda'] for c in candidates] == list(LAMBDAS),
            'Incomplete/reordered inner candidate grid')
    inner = fold['inner_folds']
    require(len(inner) == 7 and [x['validation_episode'] for x in inner] == training,
            'Partial inner folds')
    for heldout, inner_fold in zip(training, inner):
      require(inner_fold['training_episodes'] == [e for e in training if e != heldout] and
              inner_fold['training_rows'] == 96 and inner_fold['validation_rows'] == 16,
              'Inner episode isolation changed')
      require(set(inner_fold['preprocess']) == {'q', 'p'} and
              len(inner_fold['candidates']) == 4 and
              [x['lambda_value'] for x in inner_fold['candidates']] == list(LAMBDAS),
              'Partial inner candidate evaluations')
      for candidate in inner_fold['candidates']:
        require(set(candidate['interfaces']) == {'q', 'p'}, 'Partial inner interface')
        for interface in candidate['interfaces'].values():
          shape(interface['choices'], [4, 4], choices)
          shape(interface['predicted_centered_reward'], [4, 4, 7], numeric)
          fit = interface['fit']
          require(fit['regularization_lambda'] == candidate['lambda_value'] and
                  fit['training_rows'] == 96 and fit['heldout_rows'] == 16 and
                  fit['eigen']['all_eigenvalues_retained'] is True and
                  fit['eigen']['cutoff_applied'] is False and
                  fit['ridge_equation_max_abs_error'] <= 1e-10,
                  'Incomplete inner ridge fit')
    require(set(fold['outer_fits']) == set(fold['outer_interfaces']) == {'q', 'p'},
            'Partial outer interface fit')
    for fit in fold['outer_fits'].values():
      require(fit['regularization_lambda'] == fold['selected_lambda'] and
              fit['training_rows'] == 112 and fit['heldout_rows'] == 16 and
              fit['eigen']['all_eigenvalues_retained'] is True and
              fit['eigen']['cutoff_applied'] is False and
              fit['ridge_equation_max_abs_error'] <= 1e-10,
              'Incomplete outer ridge fit')
  for comparison in nested['comparisons'].values():
    shape(comparison['episode_values'], [8], numeric)
    require(comparison['lower95'] <= comparison['upper95'], 'Reversed bootstrap interval')
  supported = all(x['lower95'] > 0 for x in nested['comparisons'].values())
  negative = any(x['upper95'] <= 0 for x in nested['comparisons'].values())
  expected = ('selected_procedure_exploratory_supported' if supported else
              'selected_procedure_exploratory_not_supported' if negative else
              'selected_procedure_exploratory_inconclusive')
  require(nested['selected_procedure_status'] == expected,
          'Wrong descriptive status for the fixed nested procedure')


def verify_completion(output, expected_git):
  """Check four JSON/source hash chains; no array reads or real fits by parent."""
  output = Path(output)
  values = {}
  for name in ('budget.json', 'plan.json', 'result.json', 'nested_probe_complete.json'):
    values[name] = json.loads((output / name).read_text(encoding='utf-8'))
    finite_tree(values[name])
  budget, plan, result, complete = [values[name] for name in (
      'budget.json', 'plan.json', 'result.json', 'nested_probe_complete.json')]
  for value in (budget, plan, result, complete):
    require(value['version'] == VERSION and value['actual_git_sha'] == expected_git,
            'Wrong artifact version or execution commit')
  for value in (plan, result, complete):
    boundary(value)
  # Planning has no completed-read/provenance assertions, even in real runs.
  require(budget['seconds'] == plan['seconds'] == 120 and
          budget['planned_numeric_episode_reads'] == 8 and
          budget['actual_numeric_episode_reads'] is None and
          budget['planned_worker_boundary'] == PLANNED_BOUNDARY,
          'Changed fixed parent budget/planning boundary')
  require('real_dataset_reads' not in budget and 'source_provenance_verified' not in budget,
          'Parent budget fabricated completed worker provenance')
  require(plan['hostname'] == 'lyg0326' and plan['platform'].startswith('linux') and
          plan['numpy_version'] == '1.26.4' and plan['threads'] == 1,
          'Changed actual host/platform/numerical environment')
  require(plan['episodes'] == list(EPISODES) and plan['positions'] == list(POSITIONS) and
          plan['lambdas'] == list(LAMBDAS) and
          plan['fixed_inner_lambda_selection_permitted'] is True and
          plan['additional_parameter_search_permitted'] is False and
          plan['outer_selection_permitted'] is False and
          plan['current_rgb_independent_cue_claim_permitted'] is False,
          'Changed fixed object or selection authority')
  require(plan['source_git_sha'] == SOURCE_GIT and plan['source_json_sha256'] == SOURCE_PINS,
          'Wrong authenticated source identity')
  artifacts = plan['input_artifacts']
  require(len(artifacts) == 8 and [x['episode'] for x in artifacts] == list(EPISODES),
          'Incomplete source input identity')
  for item in artifacts:
    require(item['path'] == f"episode_{item['episode']:03}.npz", 'Unexpected input artifact path')
    for key in ('sha256', 'h_sha256', 'latent_sha256', 'cost_sha256'):
      require(type(item[key]) is str and re.fullmatch('[0-9a-f]{64}', item[key]) is not None,
              'Incomplete input byte proof')
  sources = plan['source_files_sha256']
  require(set(sources) == set(SOURCE_FILES) and complete['source_files_sha256'] == sources,
          'Incomplete/differing execution source hash chain')
  for name, digest in sources.items():
    require(digest == sha(ROOT / name), 'Changed execution source bytes: ' + name)
  require(budget['supervisor_sha256'] == sources['scripts/dl_nested_probe_supervisor.py'],
          'Parent source differs from worker binding')
  budget_hash, plan_hash = sha(output / 'budget.json'), sha(output / 'plan.json')
  require(plan['budget_sha256'] == result['budget_sha256'] == complete['budget_sha256'] == budget_hash,
          'Budget hash mismatch')
  require(result['plan_sha256'] == complete['plan_sha256'] == plan_hash and
          complete['result_sha256'] == sha(output / 'result.json'), 'Result/plan hash mismatch')
  require(result['source_inputs_unchanged'] is True and result['cost_bytes_match_original'] is True and
          result['no_prior_real_result_rejudged'] is True and
          result['no_parameters_or_checkpoints_loaded'] is True,
          'Incomplete source reuse/interpretation boundary')
  require(result['source_external_control_sufficient'] is True and
          plan['source_external_control_sufficient'] is True and
          result['source_external_delta'] == plan['source_external_delta'] == .2,
          'Changed source external control scope')
  require(result['elapsed_parent_entry_seconds'] <= 120, 'Worker exceeded parent-entry budget')
  for key, expected in dict(complete_outer_folds=8, complete_inner_folds=56,
      eigendecompositions=128, lambda_evaluations=464).items():
    require(type(complete[key]) is int and complete[key] == expected,
            'Partial complete count: ' + key)
  verify_nested(result['nested'])
  return dict(verification_kind='nested_artifact_chain_schema_and_hashes',
      worker_boundary={k: complete[k] for k in BOUNDARY},
      verified_outer_folds=8, verified_inner_folds=56, verified_eigendecompositions=128,
      verified_lambda_evaluations=464, plan_sha256=plan_hash,
      result_sha256=sha(output / 'result.json'), complete_sha256=sha(output / 'nested_probe_complete.json'))


def supervise(command, output, expected_git, seconds=120, verify=None, started=None):
  """Own only the newly started worker group, with all-entry publication limit."""
  started = time.monotonic() if started is None else started
  require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0 and
          math.isfinite(started) and started <= time.monotonic(), 'Invalid budget')
  output = Path(output)
  child = None
  alarm_active = os.name != 'nt'
  old_handler = None
  timed_out, verified, cleanup_requested = False, False, False
  failure, verification, code = None, None, 1
  def expire(signum, frame):
    raise TimeoutError('Whole-entry nested budget expired')
  def remaining():
    value = started + seconds - time.monotonic()
    if value <= 0:
      raise TimeoutError('Whole-entry nested budget expired')
    return value
  if alarm_active:
    inherited_entry_alarm = (_CLI_ENTRY_ALARM and started == ENTRY_STARTED and seconds == 120 and
                            signal.getsignal(signal.SIGALRM) is _entry_expire)
    require(inherited_entry_alarm or signal.getitimer(signal.ITIMER_REAL) == (0., 0.),
            'Cannot replace an existing process alarm')
    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, remaining())
  try:
    output = output.resolve()
    if verify is verify_completion:
      # The verifier alone cannot authenticate an arbitrary subprocess. Real
      # read receipts require this exact immutable worker entry, which itself
      # verifies its clean Git/source binding and the actual dataset inputs.
      require(type(command) in (list, tuple) and len(command) == 8 and
              Path(command[0]).resolve() == Path(sys.executable).resolve() and
              Path(command[1]).resolve() == ROOT / 'scripts/dl_nested_probe.py' and
              command[2] == '--dataset' and Path(command[3]).is_absolute() and
              command[4] == '--output' and Path(command[5]).resolve() == output and
              command[6] == '--expected-git' and command[7] == expected_git,
              'Real provenance requires the fixed nested worker command')
    # Existing attempts, including failed ones, are not reused or overwritten.
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'budget.json', dict(version=VERSION,
        actual_git_sha=expected_git, seconds=seconds, parent_pid=os.getpid(),
        parent_entry_monotonic=started, supervisor_sha256=sha(__file__),
        planned_numeric_episode_reads=8, actual_numeric_episode_reads=None,
        planned_worker_boundary=PLANNED_BOUNDARY))
    with (output / 'nested_probe.log').open('x', encoding='utf-8') as log:
      try:
        remaining()
        env = dict(os.environ, DL_NESTED_PROBE_PARENT=str(os.getpid()), CUDA_VISIBLE_DEVICES='')
        for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
          env[name] = '1'
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=os.name != 'nt', env=env)
        code = child.wait(timeout=remaining())
        if code == 0 and verify is not None:
          verification = verify(output, expected_git)
          require(verification is not False, 'Completion verification rejected artifacts')
          remaining()
          verified = True
      except (subprocess.TimeoutExpired, TimeoutError) as error:
        timed_out, verified, code = True, False, 124
        failure = dict(type=type(error).__name__, message=str(error))
      except Exception as error:
        verified, code = False, 1
        failure = dict(type=type(error).__name__, message=str(error))
      finally:
        if alarm_active:
          signal.setitimer(signal.ITIMER_REAL, 0)
        if child is not None:
          cleanup_requested = os.name != 'nt'
          if os.name == 'nt':
            if child.poll() is None:
              child.kill()
          else:
            try:
              os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
              pass
          try:
            child.wait(timeout=2)
          except subprocess.TimeoutExpired:
            verified, code = False, 1
            failure = dict(type='CleanupFailure', message='Owned worker did not terminate')
  finally:
    if alarm_active:
      signal.setitimer(signal.ITIMER_REAL, 0)
      signal.signal(signal.SIGALRM, old_handler)
  elapsed = time.monotonic() - started
  if elapsed > seconds:
    timed_out, verified, code = True, False, 124
  if alarm_active and not timed_out:
    # Cleanup has finished. Keep receipt construction (including hostname)
    # bounded too; no live owned group remains if this entry alarm fires.
    signal.signal(signal.SIGALRM, _entry_expire)
    signal.setitimer(signal.ITIMER_REAL, max(1e-6, started + seconds - time.monotonic()))
  # An arbitrary fixture callback never verifies a real source chain. Its own
  # returned object remains visible, while actual reads/provenance stay unknown.
  real_chain = verified and verify is verify_completion
  actual = BOUNDARY if real_chain else UNKNOWN_ACTUAL_BOUNDARY
  receipt = dict(version=VERSION, actual_git_sha=expected_git, **actual,
      planned_numeric_episode_reads=8,
      actual_numeric_episode_reads=8 if real_chain else None,
      planned_worker_boundary=PLANNED_BOUNDARY,
      verified_worker_boundary=verification['worker_boundary'] if real_chain else None,
      exit_code=code, timed_out=timed_out, worker_pid=child.pid if child else None,
      worker_terminal=child is not None and child.poll() is not None,
      owned_group_cleanup_requested=cleanup_requested,
      complete_verified=verified, real_artifact_chain_verified=real_chain,
      verification=verification, failure=failure, elapsed_seconds=elapsed,
      elapsed_parent_entry_seconds=elapsed, host=socket.gethostname(), platform=sys.platform,
      generic_fixture_without_scientific_verification=verify is not verify_completion,
      windows_cleanup_scope='known direct worker only' if os.name == 'nt' else None)
  completion_path = output / 'supervisor_complete.json'
  try:
    if not timed_out:
      if alarm_active:
        signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, remaining())
      remaining()
    write_json(completion_path, receipt)
    if not timed_out:
      remaining()
  except TimeoutError as error:
    if alarm_active:
      signal.setitimer(signal.ITIMER_REAL, 0)
    if completion_path.exists():
      completion_path.rename(output / 'supervisor_complete.incomplete.json')
    receipt.update(**UNKNOWN_ACTUAL_BOUNDARY, actual_numeric_episode_reads=None,
        verified_worker_boundary=None, real_artifact_chain_verified=False,
        exit_code=124, timed_out=True, complete_verified=False,
        failure=dict(type=type(error).__name__, message=str(error)),
        elapsed_seconds=time.monotonic() - started,
        elapsed_parent_entry_seconds=time.monotonic() - started)
    # Failure evidence/owned cleanup may finish after 120 seconds; that never
    # gives a timed-out run scientific completion or continuation authority.
    write_json(completion_path, receipt)
  finally:
    if alarm_active:
      signal.setitimer(signal.ITIMER_REAL, 0)
      signal.signal(signal.SIGALRM, old_handler)
  receipt['terminal_elapsed_seconds'] = time.monotonic() - started
  return receipt


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--dataset', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--expected-git', required=True)
  args = parser.parse_args()
  require(os.name == 'posix' and socket.gethostname() == 'lyg0326' and sys.platform.startswith('linux'),
          'Frozen nested execution requires sv3 Linux CPU')
  require(re.fullmatch('[0-9a-f]{40}', args.expected_git) is not None,
          'Require full expected Git SHA')
  command = [sys.executable, str(ROOT / 'scripts/dl_nested_probe.py'),
      '--dataset', str(args.dataset.resolve(strict=True)), '--output', str(args.output.resolve()),
      '--expected-git', args.expected_git]
  receipt = supervise(command, args.output, args.expected_git, seconds=120,
                      verify=verify_completion, started=ENTRY_STARTED)
  # The durable receipt is the output. A potentially blocked final stdout
  # flush would extend the full-entry limit after successful publication.
  return receipt['exit_code']


if __name__ == '__main__':
  try:
    raise SystemExit(main())
  except TimeoutError:
    # Covers initialization before a child is started. Owned-child timeouts
    # are handled inside supervise, which preserves failure evidence.
    os._exit(124)
