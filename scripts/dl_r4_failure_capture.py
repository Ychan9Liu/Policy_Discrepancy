"""DL-r4: external, stable inventory of the one cal-null failure attempt.

Only standard-library metadata/bytes are read. No pickle is decoded, no run
file is written, and the report is explicitly not a producer closure or a
scientific/physics audit. The fresh report must live outside the failed run.
"""
import argparse
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'DL-r4-failure-capture-r1'
SCHEMA = 1
RUN_VERSION = 'DL-r4-utility-r1'
RUN_GIT = '5ce1d3724ee8c5d01f09c4d687100a2f42f6ac83'
IDENTITIES = {'parent': (247014, 407058309), 'worker': (247226, 407058446)}
ERROR = 'ValueError: Calibration null measurement failed; held remains unopened'
EXPECTED_MISSING = ('worker_complete.json', 'statistics_input.json',
                    'statistics_provisional.json',
                    *(f'episode_{i:03}' for i in range(8, 32)))
SELF = 'scripts/dl_r4_failure_capture.py'
SECONDS = 180


def require(condition, message):
  if not condition:
    raise ValueError(message)


def utc():
  return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _sha(data):
  return hashlib.sha256(data).hexdigest()


def _full_sha(value, length):
  return isinstance(value, str) and re.fullmatch('[0-9a-f]{%d}' % length, value)


def _relative(value):
  require(isinstance(value, str) and '\\' not in value and '\x00' not in value,
      'Relative paths must use canonical POSIX names')
  path = PurePosixPath(value)
  require(value and not path.is_absolute() and str(path) == value
      and all(p not in ('', '.', '..') for p in path.parts), 'Unsafe relative path')
  return value


def _no_symlink_components(path):
  path = Path(path).absolute()
  for part in (*reversed(path.parents), path):
    try:
      info = part.lstat()
    except FileNotFoundError:
      continue
    require(not stat.S_ISLNK(info.st_mode), 'Symlink path component refused: ' + str(part))
    if hasattr(os.path, 'isjunction'):
      require(not os.path.isjunction(part), 'Junction path component refused')
  return path


def _inside(path, root):
  try:
    Path(path).relative_to(root)
    return True
  except ValueError:
    return False


def _regular_read(path, root=None, *, keep=False):
  """Do not silently follow a last-component symlink or accept a changing file."""
  path = _no_symlink_components(path)
  before = path.lstat()
  require(stat.S_ISREG(before.st_mode), 'Only regular files are eligible: ' + str(path))
  resolved = path.resolve(strict=True)
  require(root is None or _inside(resolved, root), 'File escaped the declared run')
  flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
  descriptor = os.open(path, flags)
  try:
    opened = os.fstat(descriptor)
    require(stat.S_ISREG(opened.st_mode) and (opened.st_dev, opened.st_ino)
        == (before.st_dev, before.st_ino), 'File identity changed before read')
    pieces, digest, length = [], hashlib.sha256(), 0
    with os.fdopen(descriptor, 'rb', closefd=False) as stream:
      for piece in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(piece)
        length += len(piece)
        if keep:
          pieces.append(piece)
    after = os.fstat(descriptor)
  finally:
    os.close(descriptor)
  signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
  # Compare each API with itself: on Windows lstat ctime is creation time,
  # while fstat can expose change time. No numerical tolerance is introduced.
  require(signature(opened) == signature(after), 'File changed while being read')
  final = path.lstat()
  require(not stat.S_ISLNK(final.st_mode) and signature(final) == signature(before),
      'File was replaced during read')
  require(root is None or _inside(path.resolve(strict=True), root), 'File escaped after read')
  require(length == before.st_size, 'File length changed during read')
  return (b''.join(pieces) if keep else None), {'bytes': length, 'sha256': digest.hexdigest()}


def _regular_bytes(path, root=None):
  return _regular_read(path, root, keep=True)[0]


def _file_pin(path, root=None):
  return _regular_read(path, root)[1]


def inventory(root):
  """All existing regular files, including opaque binary artifacts; no decoding."""
  rows, directories = [], []
  def visit(directory):
    require(_inside(directory.resolve(strict=True), root), 'Directory escaped run')
    for path in sorted(directory.iterdir(), key=lambda p: p.name):
      info = path.lstat()
      require(not stat.S_ISLNK(info.st_mode), 'Run symlink refused: ' + str(path))
      relative = _relative(path.relative_to(root).as_posix())
      if stat.S_ISDIR(info.st_mode):
        if hasattr(os.path, 'isjunction'):
          require(not os.path.isjunction(path), 'Run junction refused')
        directories.append(relative)
        visit(path)
      elif stat.S_ISREG(info.st_mode):
        rows.append({'path': relative, **_file_pin(path, root)})
      else:
        raise ValueError('Run contains a nonregular filesystem entry: ' + relative)
  visit(root)
  return sorted(rows, key=lambda r: r['path']), sorted(directories)


def _json_bytes(data):
  def pairs(values):
    result = {}
    for key, value in values:
      require(key not in result, 'Duplicate JSON key')
      result[key] = value
    return result
  def number(value):
    result = float(value)
    require(math.isfinite(result), 'Nonfinite JSON number')
    return result
  return json.loads(data.decode('utf-8'), object_pairs_hook=pairs,
      parse_float=number,
      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON constant')))


def _read_pinned_json(root, relative, pins):
  require(relative in pins, 'Required failure metadata is missing: ' + relative)
  data = _regular_bytes(root / relative, root)
  require(_sha(data) == pins[relative]['sha256'] and len(data) == pins[relative]['bytes'],
      'Metadata changed since inventory: ' + relative)
  value = _json_bytes(data)
  require(isinstance(value, dict), 'Failure metadata must be a JSON object')
  return value


def git_binding(run_git, audit_git):
  """Authenticate clean audit code and every frozen producer-source pin."""
  require(run_git == RUN_GIT and _full_sha(audit_git, 40), 'Wrong run/audit Git identity')
  def command(*args):
    result = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, timeout=30)
    require(result.returncode == 0, 'Git authentication command failed')
    return result.stdout
  require(Path(command('rev-parse', '--show-toplevel').decode().strip()).resolve() == ROOT,
      'Audit script is outside its own repository root')
  require(command('rev-parse', 'HEAD').decode().strip() == audit_git, 'Audit HEAD differs')
  require(not command('status', '--porcelain=v1', '--untracked-files=all'), 'Clean audit checkout required')
  own = command('show', audit_git + ':' + SELF)
  require(Path(__file__).read_bytes() == own, 'Audit checkout/script Git blob differs')
  pins = {}
  archive = command('archive', '--format=tar', run_git)
  with tarfile.open(fileobj=io.BytesIO(archive), mode='r:') as stream:
    for member in stream:
      name = _relative(member.name.rstrip('/'))
      selected = name.endswith(('.py', '.yaml', '.yml', '.lock')) or name in (
          'AGENTS.md', 'docs/DL_R4_UTILITY.md')
      if selected:
        require(member.isfile() and name not in pins, 'Invalid/duplicate frozen source entry')
        data = stream.extractfile(member).read()
        pins[name] = _sha(data)
  require('scripts/dl_r4_utility.py' in pins and 'scripts/dl_r4_physics.py' in pins
      and 'scripts/dl_r4_policy.py' in pins, 'Frozen producer sources are incomplete')
  return {'sourcepins': dict(sorted(pins.items())), 'audit_sourcepins': {SELF: _sha(own)}}


def process_observation(pid, expected_starttime_ticks):
  """Any present/reused/unreadable PID is refused, without acting on it."""
  root = Path('/proc') / str(pid)
  result = dict(pid=pid, expected_starttime_ticks=expected_starttime_ticks,
      checked_utc=utc())
  try:
    data = (root / 'stat').read_text()
    tail = data[data.rfind(')') + 2:].split()
    command = (root / 'cmdline').read_bytes()
  except FileNotFoundError:
    # A disappearing stat/cmdline is rechecked; an observational race is not
    # converted into a terminal claim while the PID directory remains present.
    require(not root.exists(), 'Process observation raced a still-present PID')
    try:
      os.kill(pid, 0)  # Permission/existence probe only; never a stop signal.
    except ProcessLookupError:
      pass
    except PermissionError as error:
      raise ValueError('Original PID existence cannot be distinguished from access denial') from error
    else:
      raise ValueError('PID still exists despite missing proc identity; capture refused')
    return dict(result, present=False, alive=False, identity_status='absent')
  except PermissionError as error:
    raise ValueError('Original process identity cannot be inspected') from error
  result.update(present=True, alive=tail[0] not in ('Z', 'X'), state=tail[0],
      observed_starttime_ticks=int(tail[19]),
      expected_identity_matches=int(tail[19]) == expected_starttime_ticks,
      cmdline_sha256=_sha(command))
  raise ValueError('Original PID still present; terminal capture refused: ' + json.dumps(result))


def process_set():
  require(os.name == 'posix' and Path('/proc').is_dir(),
      'Actual Linux /proc observation required; CPU tests use an explicit test double')
  return {name: process_observation(pid, start) for name, (pid, start) in IDENTITIES.items()}


def _dead_processes():
  observed = process_set()
  require(isinstance(observed, dict) and set(observed) == set(IDENTITIES),
      'Original process observations are incomplete')
  for name, (pid, start) in IDENTITIES.items():
    row = observed[name]
    require(isinstance(row, dict) and row.get('pid') == pid
        and row.get('expected_starttime_ticks') == start
        and row.get('present') is False and row.get('alive') is False
        and row.get('identity_status') == 'absent', 'Original process is not verifiably absent')
  return observed


def validate_failure(root, rows, directories, run_git, sourcepins):
  pins = {row['path']: row for row in rows}
  require(len(pins) == len(rows), 'Duplicate inventory paths')
  episode_dirs = sorted(p for p in directories if '/' not in p and p.startswith('episode_'))
  require(episode_dirs == [f'episode_{i:03}' for i in range(8)],
      'Exactly eight calibration episode directories, with no held directories, required')
  for relative in EXPECTED_MISSING:
    require(relative not in pins and relative not in directories and not (root / relative).exists(),
        'Required failure absence violated: ' + relative)
  require(not any(PurePosixPath(p).name.startswith('statistics') for p in pins),
      'Statistics artifacts are forbidden for a cal-null failure capture')
  require(not any(PurePosixPath(p).name in {f'episode_{i:03}' for i in range(8, 32)}
      for p in (*pins, *directories)), 'Nested held path is forbidden')
  budget = _read_pinned_json(root, 'budget.json', pins)
  process = _read_pinned_json(root, 'process.json', pins)
  terminal = _read_pinned_json(root, 'supervisor_terminal.json', pins)
  require(budget.get('version') == RUN_VERSION and budget.get('git_sha') == run_git
      and type(budget.get('parent_pid')) is int and budget['parent_pid'] == IDENTITIES['parent'][0]
      and budget.get('source_files') == sourcepins,
      'Run budget/source Git pins differ')
  require(process.get('parent_pid') == IDENTITIES['parent'][0]
      and process.get('worker_pid') == IDENTITIES['worker'][0]
      and process.get('worker_pgid') == IDENTITIES['worker'][0],
      'Original parent/worker process receipt differs')
  command = process.get('command')
  require(isinstance(command, list) and len(command) >= 3
      and all(isinstance(p, str) for p in command) and command[2] == 'worker',
      'Original worker command is missing')
  for flag, expected in (('--expected-git', run_git), ('--output', str(root))):
    require(command.count(flag) == 1 and command.index(flag) + 1 < len(command)
        and command[command.index(flag) + 1] == expected, 'Original worker command differs')
  require(terminal.get('version') == RUN_VERSION and terminal.get('git_sha') == run_git
      and type(terminal.get('worker_returncode')) is int and terminal['worker_returncode'] == 1
      and terminal.get('worker_terminal') is True
      and terminal.get('main_generation_complete') is False
      and terminal.get('independent_audit_complete') is False
      and terminal.get('scientific_support_claimed') is False
      and terminal.get('owned_worker_group_cleaned') is True
      and terminal.get('parent_failure') is None
      and terminal.get('worker_complete_sha256') is None,
      'Not the exact known cal-null terminal class')
  for name in ('elapsed_seconds', 'worker_seconds'):
    require(type(terminal.get(name)) in (int, float)
        and math.isfinite(terminal[name]) and 0 < terminal[name] < 21000,
        'Terminal elapsed/budget evidence differs')
  require('worker.log' in pins and terminal.get('worker_log_sha256') == pins['worker.log']['sha256']
      and 'resources.jsonl' in pins
      and terminal.get('resources_sha256') == pins['resources.jsonl']['sha256'],
      'Terminal log/resource hashes differ from inventory')
  log = _regular_bytes(root / 'worker.log', root)
  require(_sha(log) == pins['worker.log']['sha256'], 'Worker log changed since inventory')
  lines = log.decode('utf-8').splitlines()
  nonempty = [line for line in lines if line.strip()]
  require(nonempty and nonempty[-1] == ERROR and lines.count(ERROR) == 1
      and 'Traceback (most recent call last):' in lines,
      'Worker log does not end in the exact cal-null exception')
  event_ids = []
  for line in lines:
    try:
      event = _json_bytes(line.encode())
    except (ValueError, UnicodeError):
      continue
    if isinstance(event, dict) and event.get('event') == 'episode_complete':
      event_ids.append(event.get('episode_id'))
  require(event_ids == list(range(7)), 'Cal-null log completion-event sequence differs')
  for episode in range(8):
    record = _read_pinned_json(root, f'episode_{episode:03}/complete.json', pins)
    require(record.get('version') == RUN_VERSION and type(record.get('episode_id')) is int
        and record['episode_id'] == episode and record.get('model_full_state_unchanged') is True,
        'Calibration completion metadata differs')
  frozen = _read_pinned_json(root, 'calibration_frozen.json', pins)
  require(frozen.get('version') == RUN_VERSION
      and frozen.get('frozen_models', {}).get('null_passed') is False
      and type(frozen.get('held_episodes_generated')) is int
      and frozen['held_episodes_generated'] == 0, 'Cal-null/held gate metadata differs')
  evidence = frozen.get('calibration_evidence')
  require(isinstance(evidence, list) and len(evidence) == 8, 'Calibration evidence coverage differs')
  for episode, row in enumerate(evidence):
    path = f'episode_{episode:03}/complete.json'
    require(isinstance(row, dict) and row.get('path') == path
        and row.get('sha256') == pins[path]['sha256'], 'Calibration evidence hash differs')
  return dict(completed_episode_ids=list(range(8)), held_episode_directories=[],
      expected_missing=list(EXPECTED_MISSING), calibration_null_passed=False,
      terminal={'path': 'supervisor_terminal.json', **{k: pins['supervisor_terminal.json'][k]
          for k in ('bytes', 'sha256')}},
      worker_log={'path': 'worker.log', **{k: pins['worker.log'][k] for k in ('bytes', 'sha256')}},
      terminal_class='calibration_null_measurement_failed_held_unopened', final_error=ERROR)


def capture(run, output, run_git, audit_git, external_pins=()):
  require(run_git == RUN_GIT and _full_sha(audit_git, 40), 'Wrong run/audit Git SHA')
  run = _no_symlink_components(run).resolve(strict=True)
  require(run.is_dir(), 'Failed run must be an existing directory')
  output = _no_symlink_components(output)
  require(not output.exists() and not output.is_symlink(), 'Fresh external capture file required')
  parent = _no_symlink_components(output.parent).resolve(strict=True)
  require(parent.is_dir(), 'Capture parent must be an existing directory')
  output = parent / output.name
  require(not _inside(output, run), 'Capture output must be outside failed run')
  binding = git_binding(run_git, audit_git)
  before_processes = _dead_processes()
  before, before_directories = inventory(run)
  failure = validate_failure(run, before, before_directories, run_git, binding['sourcepins'])
  external = []
  for path in external_pins:
    path = _no_symlink_components(path).resolve(strict=True)
    require(not _inside(path, run) and path != output, 'External pin is not external')
    require(str(path) not in [p['path'] for p in external], 'Duplicate external pin')
    external.append({'path': str(path), **_file_pin(path)})
  after, after_directories = inventory(run)
  after_processes = _dead_processes()
  require(before == after and before_directories == after_directories,
      'Failed-run inventory changed between the two complete passes')
  for pin in external:
    require(_file_pin(Path(pin['path'])) == {k: pin[k] for k in ('bytes', 'sha256')},
        'External pin changed during capture')
  # Recheck every expected absence after the second complete inventory, rather
  # than infer absence from a stale first list.
  for path in EXPECTED_MISSING:
    require(not (run / path).exists(), 'Expected absence changed after inventory')
  require(git_binding(run_git, audit_git) == binding,
      'Audit source/Git binding changed during capture')
  report = dict(version=VERSION, schema=SCHEMA, run_git=run_git, audit_git=audit_git,
      run_directory=str(run), captured_utc=utc(), stable=True,
      producer_closure_claimed=False, scientific_audit_complete=False,
      pickle_decoded=False, run_files_written=False,
      before_inventory=before, after_inventory=after,
      before_directories=before_directories, after_directories=after_directories,
      process_observations={'before': before_processes, 'after': after_processes},
      process_scope='only the original recorded parent and worker PIDs',
      sourcepins=binding['sourcepins'], audit_sourcepins=binding['audit_sourcepins'],
      external_pins=external, **failure)
  with output.open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
    stream.write('\n')
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--run', required=True)
  parser.add_argument('--output', required=True)
  parser.add_argument('--run-git', required=True)
  parser.add_argument('--audit-git', required=True)
  parser.add_argument('--external-pin', action='append', default=[])
  args = parser.parse_args()
  started = time.monotonic()
  if os.name == 'posix':
    def expired(signum, frame):
      raise TimeoutError('Whole DL-r4 external failure capture exceeded 180 seconds')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, SECONDS)
  report = capture(args.run, args.output, args.run_git, args.audit_git, args.external_pin)
  if os.name == 'posix':
    signal.setitimer(signal.ITIMER_REAL, 0)
  require(time.monotonic() - started < SECONDS, 'Late capture cannot claim completion')
  print(json.dumps(dict(version=VERSION, output=str(Path(args.output).resolve()),
      capture_sha256=_file_pin(Path(args.output))['sha256'],
      files=len(report['before_inventory']), bytes=sum(x['bytes'] for x in report['before_inventory']),
      stable=True, producer_closure_claimed=False, terminal_class=report['terminal_class']),
      sort_keys=True))


if __name__ == '__main__':
  main()
