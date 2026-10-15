"""DL-r5 bounded serial live-RCT supervisor; imports never initialize a GPU.

This is a diagnostic of a frozen checkpoint, not a training runner. Family
workers and their CPU shadows share an owned process group. A timeout cannot
extend, resume, refill, or splice this fixed object.
"""
import time
ENTRY_STARTED = time.monotonic()
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))
VERSION, SECONDS = 'DL-live-utility-r1', 12600
ENGINEERING_VERSION = 'DL-r5-engineering-r2'
ENGINEERING_NORM = 'docs/DL_R5_LIVE_RETRY.md'
HOST, GPU = 'lyg0326', 5
GPU_UUID = 'GPU-79704918-e096-92b4-252c-e3244497a8b3'
CHECKPOINT_SHA = 'c6a7d92e24669a515bb44693faf73e442d9868f9fc814eec0e22e069aa594d24'
CONFIG_SHA = '877f8470cb4f7f45d143e3099fbd3f55a828991afe33779b7b04c4264f6c8377'
DEADLINE = ENTRY_STARTED + SECONDS
OWNED = set()


def require(value, message):
  if not value:
    raise ValueError(message)


def require_host():
  require(os.name == 'posix' and socket.gethostname().split('.')[0] == HOST, 'sv3 Linux required')


def check():
  require(time.monotonic() < DEADLINE, 'Whole DL-r5 entry deadline exceeded')


def sha(path):
  result = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for block in iter(lambda: stream.read(1024*1024), b''):
      result.update(block)
  return result.hexdigest()


def read_json(path):
  def pairs(items):
    result = {}
    for key, value in items:
      require(key not in result, 'Duplicate JSON key')
      result[key] = value
    return result
  return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs,
      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def write_json(path, value):
  check()
  with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
    stream.write('\n')
    stream.flush()
    os.fsync(stream.fileno())
  check()


def file_pin(path):
  require(not Path(path).is_symlink(), 'Regular pinned file required')
  path = Path(path).resolve(strict=True)
  require(path.is_file(), 'Regular pinned file required')
  return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def process_identity(pid):
  # Split at the final ')' because comm may itself contain spaces/parentheses.
  fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
  return dict(pid=int(pid), parent_pid=int(fields[1]), pgid=int(fields[2]),
      start_ticks=int(fields[19]), state=fields[0])


def spawn_owned(command, *, deadline=None, **kwargs):
  deadline = DEADLINE if deadline is None else deadline
  require(time.monotonic() < deadline, 'Late process creation')
  if os.name != 'posix':
    child = subprocess.Popen(command, **kwargs)
    OWNED.add(child.pid)
    return child
  previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM, signal.SIGTERM})
  try:
    child = subprocess.Popen(command, start_new_session=True,
        preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, previous), **kwargs)
    OWNED.add(child.pid)
    return child
  finally:
    signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def cleanup(child):
  if os.name == 'posix':
    try:
      os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
      pass
  elif child.poll() is None:
    child.kill()
  child.wait(timeout=5)
  OWNED.discard(child.pid)


def expire(signum, frame):
  for pid in tuple(OWNED):
    try:
      os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
      pass
  os._exit(124)


def command_output(command, input_data=None):
  check()
  # Worker metadata helpers remain in the family's owned group, so a parent
  # timeout also kills a stuck git process; the idle parent owns its helpers.
  own = 'DL_R5_PARENT' not in os.environ
  if own:
    child = spawn_owned(command, cwd=ROOT, stdin=subprocess.PIPE if input_data is not None else None,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  else:
    child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE if input_data is not None else None,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  try:
    output, error = child.communicate(input=input_data,
        timeout=min(30, max(.001, DEADLINE-time.monotonic())))
    require(child.returncode == 0, 'Metadata command failed: '+error.decode(errors='replace')[:256])
    check()
    return output
  finally:
    if own:
      cleanup(child)
    elif child.poll() is None:
      child.kill()
      child.wait(timeout=5)
    for pipe in (child.stdin, child.stdout, child.stderr):
      if pipe is not None:
        pipe.close()


def source_binding(expected):
  require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{40}', expected), 'Full Git SHA required')
  require(command_output(['git', 'rev-parse', 'HEAD']).decode().strip() == expected, 'Git HEAD differs')
  require(not command_output(['git', 'status', '--porcelain', '--untracked-files=all']),
      'Clean checkout required')
  names = command_output(['git', 'ls-files', '-z']).decode().split('\0')
  selected = {name for name in names if name.endswith(('.py', '.yaml', '.yml', '.lock')) or
      name in ('AGENTS.md', 'docs/DL_R5_LIVE_TRIAL.md', ENGINEERING_NORM)}
  required = {'scripts/dl_live_trial.py', 'scripts/dl_live_family.py', 'scripts/dl_live_policy.py',
      'scripts/dl_live_shadow.py', 'analysis/dl_live_statistics.py', 'analysis/dl_live_audit.py',
      'docs/DL_R5_LIVE_TRIAL.md', ENGINEERING_NORM}
  require(required <= selected, 'Tracked DL-r5 implementation incomplete')
  # cat-file yields actual Git blobs. git archive can apply checkout newline
  # conversion, which must not silently redefine the authenticated source.
  ordered = sorted(selected)
  packed = command_output(['git', 'cat-file', '--batch'],
      ''.join(expected+':'+name+'\n' for name in ordered).encode())
  pins = {}
  offset = 0
  for name in ordered:
    end = packed.index(b'\n', offset)
    header = packed[offset:end].split()
    require(len(header) == 3 and header[1] == b'blob', 'Regular Git source blob required')
    size = int(header[2])
    offset = end+1
    blob = packed[offset:offset+size]
    require(len(blob) == size and packed[offset+size:offset+size+1] == b'\n', 'Truncated source batch')
    offset += size+1
    path = ROOT/name
    require(not path.is_symlink() and path.read_bytes() == blob, 'Checkout/blob mismatch: '+name)
    pins[name] = dict(sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob))
  require(offset == len(packed) and set(pins) == selected, 'Git/source inventory incomplete')
  return pins


def authenticate_inputs(checkpoint, config):
  result = dict(checkpoint=file_pin(checkpoint), config=file_pin(config))
  require(result['checkpoint']['sha256'] == CHECKPOINT_SHA, 'N1M checkpoint bytes differ')
  require(result['config']['sha256'] == CONFIG_SHA, 'N1M config bytes differ')
  return result


def engineering_pin(source):
  require(isinstance(source, dict) and ENGINEERING_NORM in source,
      'Tracked r2 engineering norm required')
  entry = source[ENGINEERING_NORM]
  require(isinstance(entry, dict) and set(entry) == {'sha256', 'bytes'} and
      type(entry['bytes']) is int and entry['bytes'] > 0 and
      isinstance(entry['sha256'], str) and re.fullmatch('[0-9a-f]{64}', entry['sha256']),
      'Engineering norm Git blob pin differs')
  return dict(path=ENGINEERING_NORM, **entry)


def validate_engineering_budget(budget):
  require(budget.get('engineering_version') == ENGINEERING_VERSION and
      type(budget.get('seconds')) is int and budget['seconds'] == SECONDS and
      type(budget.get('physical_gpu')) is int and budget['physical_gpu'] == GPU and
      budget.get('gpu_uuid') == GPU_UUID, 'R2 engineering budget/resource identity differs')
  require(budget.get('engineering_norm') == engineering_pin(budget.get('source_files')),
      'R2 engineering norm/source binding differs')


def verify_budget(path, expected_git, checkpoint, config, resource_record):
  """Direct exec-child authentication before GPU imports; return cooperative check."""
  global DEADLINE
  require_host()
  path = Path(path).resolve(strict=True)
  budget = read_json(path)
  validate_engineering_budget(budget)
  require(budget['version'] == VERSION and budget['seconds'] == SECONDS and
      budget['git_sha'] == expected_git and budget['logical_server'] == 'sv3' and
      budget['hostname'] == socket.gethostname() and budget['physical_gpu'] == GPU and
      budget['gpu_uuid'] == GPU_UUID, 'Live trial budget/identity differs')
  require(type(budget['monotonic_deadline']) is float and math.isfinite(budget['monotonic_deadline']),
      'Finite whole-entry deadline required')
  DEADLINE = budget['monotonic_deadline']
  check()
  parent_pid = budget['parent_pid']
  require(type(parent_pid) is int and parent_pid == os.getppid() == int(os.environ.get('DL_R5_PARENT', '-1')),
      'Direct live parent required')
  expected_parent = dict(pid=parent_pid, start_ticks=budget['parent_start_ticks'], pgid=budget['parent_pgid'])
  def checked():
    check()
    actual = process_identity(parent_pid)
    require(all(actual[k] == v for k, v in expected_parent.items()) and actual['state'] != 'Z',
        'Original live parent identity changed')
  checked()
  require(os.getpgid(0) == os.getpid(), 'Family must lead its owned process group')
  require(os.environ.get('DL_R5_BUDGET_SHA256') == sha(path), 'Parent budget raw hash differs')
  require(source_binding(expected_git) == budget['source_files'], 'Parent/family source bytes differ')
  require(authenticate_inputs(checkpoint, config) == budget['input_pins'], 'Parent/family input bytes differ')
  require(file_pin(resource_record) == budget['resource_record'], 'Parent/family original resource bytes differ')
  return budget, checked


def sample_resource(worker_pid, telemetry):
  from scripts.dl_resources import live_state
  state = live_state(GPU)
  # processes is the actual XML query's parsed process summary, not raw XML.
  sample = dict(state, monotonic=time.monotonic(), worker_pid=worker_pid,
      xml_process_summary=state['processes'])
  telemetry.write(json.dumps(sample, sort_keys=True, allow_nan=False)+'\n')
  telemetry.flush()
  os.fsync(telemetry.fileno())
  require(state['gpu_uuid'] == GPU_UUID, 'Actual assigned GPU UUID differs')
  require(all(item['pid'] == worker_pid for item in state['processes']+state['pmon_jobs']),
      'Unknown work appeared on assigned GPU5; stop only this owned family')
  return sample


def run_family(args, output, family, budget_path):
  budget = read_json(budget_path)
  validate_engineering_budget(budget)
  directory = output/f'family_{family:02}'
  directory.mkdir()
  command = [sys.executable, str(ROOT/'scripts/dl_live_family.py'), '--family', str(family),
      '--output', str(directory), '--budget', str(budget_path), '--expected-git', args.expected_git,
      '--checkpoint', args.checkpoint, '--config', args.config, '--resource-record', args.resource_record]
  env = dict(os.environ, DL_R5_PARENT=str(os.getpid()), DL_R5_BUDGET_SHA256=sha(budget_path))
  started = time.monotonic()
  failure, result = None, None
  with (directory/'worker.log').open('xb') as log:
    child = spawn_owned(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    try:
      write_json(directory/'process.json', dict(parent_pid=os.getpid(), worker_pid=child.pid,
          worker_pgid=child.pid, command=command, started_monotonic=started,
          started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
      with (directory/'resources.jsonl').open('x', encoding='utf-8') as telemetry:
        while child.poll() is None:
          check()
          sample_resource(child.pid, telemetry)
          try:
            child.wait(timeout=min(15, max(.001, DEADLINE-time.monotonic())))
          except subprocess.TimeoutExpired:
            pass
    except Exception as error:
      failure = repr(error)
    finally:
      cleanup(child)
      result = child.returncode
  complete_path = directory/'family_complete.json'
  terminal = dict(version=VERSION, engineering_version=ENGINEERING_VERSION,
      engineering_norm=budget['engineering_norm'],
      family=family, worker_returncode=result,
      failure=failure, worker_terminal=child.poll() is not None, owned_group_cleaned=True,
      started_monotonic=started, ended_monotonic=time.monotonic(),
      complete=file_pin(complete_path) if complete_path.is_file() else None,
      process=file_pin(directory/'process.json'),
      log=file_pin(directory/'worker.log'),
      resources=file_pin(directory/'resources.jsonl') if (directory/'resources.jsonl').is_file() else None)
  write_json(directory/'terminal.json', terminal)
  require(result == 0 and failure is None and terminal['complete'] is not None,
      f'Family {family} failed/incomplete; no refill or continuation')
  return read_json(complete_path), dict(path=f'{directory.name}/terminal.json',
      sha256=sha(directory/'terminal.json'))


def family_rows(output, family, complete):
  """Authenticate each compact row from its actual complete artifact, in order."""
  require(complete['version'] == 'DL-r5-live-family-r1' and
      complete['engineering_version'] == ENGINEERING_VERSION and complete['family'] == family,
      'Complete family identity differs')
  require(len(complete['units']) == 16, 'All16 unit artifact entries required')
  rows = []
  base = output/f'family_{family:02}'
  for k, entry in enumerate(complete['units']):
    path = (base/entry['path']).resolve(strict=True)
    require(path.is_relative_to(base.resolve()) and path.is_file() and
        sha(path) == entry['sha256'], 'Family unit artifact path/hash differs')
    record = read_json(path)
    row = record['unit_row']
    require(record['version'] == 'DL-r5-live-family-r1' and record['unit'] == 16*family+k and
        entry['unit'] == 16*family+k and entry['unit_row'] == row and
        row['family_id'] == family and row['unit_id'] == k, 'Unit compact row identity differs')
    rows.append(row)
  return rows


def parent(args):
  require_host()
  output = Path(args.output).resolve()
  require(not output.exists(), 'Fresh output directory required')
  output.mkdir(parents=True)
  failure, receipts, rows, frozen, norm_pin = None, [], [], None, None
  try:
    source = source_binding(args.expected_git)
    norm_pin = engineering_pin(source)
    inputs = authenticate_inputs(args.checkpoint, args.config)
    from scripts.dl_resources import require_resource
    resource = require_resource(args.resource_record)
    require(resource['physical_gpu'] == GPU and resource['gpu_uuid'] == GPU_UUID,
        'This fixed object uses only sv3 physical GPU5')
    identity = process_identity(os.getpid())
    budget = dict(version=VERSION, engineering_version=ENGINEERING_VERSION,
        engineering_norm=norm_pin, git_sha=args.expected_git, parent_pid=os.getpid(),
        parent_start_ticks=identity['start_ticks'], parent_pgid=identity['pgid'],
        hostname=socket.gethostname(), logical_server='sv3', seconds=SECONDS,
        monotonic_deadline=float(DEADLINE), source_files=source, input_pins=inputs,
        resource_record=file_pin(args.resource_record), physical_gpu=GPU, gpu_uuid=GPU_UUID)
    write_json(output/'budget.json', budget)
    from analysis.dl_live_statistics import analyze, freeze_calibration, PROVENANCE
    for family in range(32):
      check()
      if family >= 8:
        require(frozen is not None and (output/'calibration_frozen.json').is_file(),
            'Held family must not start before calibration seal')
      complete, receipt = run_family(args, output, family, output/'budget.json')
      rows.extend(family_rows(output, family, complete))
      receipts.append(receipt)
      if family == 7:
        frozen = freeze_calibration(rows)
        write_json(output/'calibration_frozen.json', dict(version=VERSION, frozen_models=frozen,
            calibration_units=rows.copy(), calibration_evidence=receipts.copy(),
            sealed_event=384, sealed_monotonic=time.monotonic(), held_families_started=0))
      print(json.dumps(dict(event='family_complete', family=family, units=len(rows))), flush=True)
    payload = dict(version=VERSION, horizon=100, bootstrap_seed=20261030,
        calibration_sealed_event=384, units=rows, frozen_models=frozen, **PROVENANCE)
    write_json(output/'statistics_input.json', payload)
    write_json(output/'statistics_provisional.json', analyze(payload, evidence_authenticated=False))
    require(authenticate_inputs(args.checkpoint, args.config) == inputs, 'Inputs changed during run')
    require(source_binding(args.expected_git) == source, 'Source changed during run')
    require(file_pin(args.resource_record) == budget['resource_record'], 'Original resource record changed')
    write_json(output/'generation_complete.json', dict(version=VERSION, git_sha=args.expected_git,
        engineering_version=ENGINEERING_VERSION, engineering_norm=norm_pin,
        complete_families=32, complete_units=512, family_evidence=receipts,
        fixed_artifact_hashes={name: sha(output/name) for name in ('budget.json',
            'calibration_frozen.json', 'statistics_input.json', 'statistics_provisional.json')},
        optimizer_updates=0, training_actions=0, independent_audit_complete=False,
        scientific_support_claimed=False))
  except Exception as error:
    failure = repr(error)
  complete_path = output/'generation_complete.json'
  write_json(output/'supervisor_terminal.json', dict(version=VERSION, git_sha=args.expected_git,
      engineering_version=ENGINEERING_VERSION, engineering_norm=norm_pin,
      elapsed_seconds=time.monotonic()-ENTRY_STARTED, parent_failure=failure,
      completed_families=len(receipts), completed_units=len(rows),
      main_generation_complete=complete_path.is_file() and failure is None,
      generation_complete=file_pin(complete_path) if complete_path.is_file() else None,
      owned_groups_cleaned=not OWNED, independent_audit_complete=False,
      scientific_support_claimed=False))
  require(failure is None and complete_path.is_file(), 'DL-r5 failed/incomplete; preserve evidence: '+str(failure))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('expected-git', 'checkpoint', 'config', 'output', 'resource-record'):
    parser.add_argument('--'+name, required=True)
  args = parser.parse_args()
  if os.name == 'posix':
    signal.signal(signal.SIGALRM, expire)
    signal.signal(signal.SIGTERM, expire)
    signal.setitimer(signal.ITIMER_REAL, max(.001, DEADLINE-time.monotonic()))
  parent(args)


if __name__ == '__main__':
  main()
