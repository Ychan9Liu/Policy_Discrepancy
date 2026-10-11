"""Bounded stdlib entry for the fixed CPU-only synthetic probe assay."""
import time
ENTRY_STARTED = time.monotonic()
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys

VERSION = 'DL-probe-assay-r1'
CASES = ('sparse', 'replicated', 'history', 'identity')
ROOT = Path(__file__).resolve().parents[1]
BOUNDARY = dict(
    synthetic_only=True, source_provenance_verified=False,
    actor_applicable=False, source_provenance_applicable=False, method_evidence=False,
    proceed_short_training=False, decision='stop_after_fixed_synthetic_assay',
    real_scientific_decision='not_evaluated', real_gate_authority=False,
    eligible_method_training=False, training_authorized=False,
    method_effectiveness_established=False, real_dataset_reads=0,
    blind_arrays_read=False, model_calls=0, physics_steps=0,
    optimizer_updates=0, gpu_devices_used=0,
    current_rgb_increment_over_complete_history_established=False,
    continuing_kl_compression_effect_established=False)


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
            f'Synthetic authority boundary changed: {key}')


def verify_completion(output, expected_git):
  """Verify the complete artifact chain, without importing the numerical worker."""
  output = Path(output)
  def read(name):
    return json.loads((output / name).read_text(encoding='utf-8'))
  budget, plan, summary, complete = [read(name) for name in
      ('budget.json', 'plan.json', 'summary.json', 'assay_complete.json')]
  for value in (budget, plan, summary, complete):
    boundary(value)
    require(value['version'] == VERSION and value['actual_git_sha'] == expected_git,
            'Wrong artifact version or execution commit')
  require(budget['seconds'] == plan['seconds'] == 120, 'Changed fixed budget')
  require(plan['hostname'] == 'lyg0326' and plan['platform'].startswith('linux'),
          'Wrong actual execution host/platform')
  require(plan['cases'] == summary['completed_cases'] == list(CASES) and
          complete['complete_cases'] == 4, 'Partial or reordered assay')
  require(summary['no_parameter_search_performed'] is True and
          summary['no_prior_real_result_rejudged'] is True and
          summary['synthetic_construction_and_identity_valid'] is True,
          'Incomplete fixed assay boundary')
  require(plan['numpy_version'] == '1.26.4' and plan['threads'] == 1 and
          plan['parameter_search_permitted'] is False, 'Changed numerical environment')
  require(summary['plan_sha256'] == complete['plan_sha256'] == sha(output / 'plan.json'),
          'Plan hash mismatch')
  require(complete['summary_sha256'] == sha(output / 'summary.json'), 'Summary hash mismatch')
  require(set(summary['case_result_sha256']) == set(CASES), 'Incomplete result hash chain')
  for case in CASES:
    name = f'{case}_result.json'
    require(summary['case_result_sha256'][case] == sha(output / name), 'Case hash mismatch')
    result = read(name)
    boundary(result)
    require(result['case'] == case and result['proof']['construction_verified'] is True,
            'Case construction not verified')
    artifact = result['input_artifact']
    require(artifact['path'] == f'{case}_inputs.npz' and
            artifact['sha256'] == sha(output / artifact['path']), 'Input hash mismatch')
  return dict(verified_cases=list(CASES), summary_sha256=sha(output / 'summary.json'))


def supervise(command, output, expected_git, seconds=120, verify=None, started=None):
  """Own only the new child's group; generic fixtures are not scientific evidence."""
  started = time.monotonic() if started is None else started
  require(seconds > 0 and started <= time.monotonic(), 'Invalid budget')
  output = Path(output).resolve()
  child = None
  alarm_active = os.name != 'nt'
  old_handler = None
  timed_out, verified, cleanup_requested = False, False, False
  failure, verification, code = None, None, 1
  def expire(signum, frame):
    raise TimeoutError('Whole-entry budget expired')
  def remaining():
    value = started + seconds - time.monotonic()
    if value <= 0:
      raise TimeoutError('Whole-entry budget expired')
    return value
  if alarm_active:
    require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.),
            'Cannot replace an existing process alarm')
    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, remaining())
  try:
    # Existing outputs are never reused, including failed attempts.
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'budget.json', dict(version=VERSION, **BOUNDARY,
        actual_git_sha=expected_git, seconds=seconds, parent_pid=os.getpid(),
        parent_entry_monotonic=started, supervisor_sha256=sha(__file__)))
    with (output / 'assay.log').open('x', encoding='utf-8') as log:
      try:
        remaining()
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=os.name != 'nt',
            env=dict(os.environ, DL_PROBE_ASSAY_PARENT=str(os.getpid())))
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
  receipt = dict(version=VERSION, actual_git_sha=expected_git, **BOUNDARY,
      exit_code=code, timed_out=timed_out, worker_pid=child.pid if child else None,
      worker_terminal=child is not None and child.poll() is not None,
      owned_group_cleanup_requested=cleanup_requested,
      complete_verified=verified, verification=verification, failure=failure,
      elapsed_seconds=elapsed, elapsed_parent_entry_seconds=elapsed,
      host=socket.gethostname(), platform=sys.platform,
      generic_fixture_without_scientific_verification=verify is None,
      windows_cleanup_scope='known direct worker only' if os.name == 'nt' else None)
  # Successful receipt publication is also inside the entry budget. A partial
  # or stale success is preserved separately if the write crosses the deadline.
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
    receipt.update(exit_code=124, timed_out=True, complete_verified=False,
        failure=dict(type=type(error).__name__, message=str(error)),
        elapsed_seconds=time.monotonic() - started,
        elapsed_parent_entry_seconds=time.monotonic() - started)
    # Failure reporting and owned-process cleanup may finish after the budget;
    # their existence never gives a timed-out assay completion authority.
    write_json(completion_path, receipt)
  finally:
    if alarm_active:
      signal.setitimer(signal.ITIMER_REAL, 0)
      signal.signal(signal.SIGALRM, old_handler)
  receipt['terminal_elapsed_seconds'] = time.monotonic() - started
  return receipt


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--expected-git', required=True)
  args = parser.parse_args()
  require(os.name != 'nt' and socket.gethostname() == 'lyg0326',
          'Frozen assay execution requires sv3 Linux CPU')
  require(re.fullmatch('[0-9a-f]{40}', args.expected_git) is not None,
          'Require full expected Git SHA')
  args.output = args.output.resolve()
  command = [sys.executable, str(ROOT / 'analysis/dl_probe_assay.py'),
             '--output', str(args.output), '--expected-git', args.expected_git]
  receipt = supervise(command, args.output, args.expected_git, seconds=120,
                      verify=verify_completion, started=ENTRY_STARTED)
  print(json.dumps(receipt), flush=True)
  return receipt['exit_code']


if __name__ == '__main__':
  raise SystemExit(main())
