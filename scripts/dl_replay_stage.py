"""DL-NM1 byte-exact sv2 -> sv3 replay staging, never a new inventory.

Only standard-library code is imported. CLI source/materialize each has a
900-second complete entry and an owned CPU child; SCP transport is external.
Original schema/CRC/chain validation still belongs to the unchanged loader.
JSON host/process claims bind actual executed receipts; they are not independent
proof of a remote SSH session. The caller preserves that external exit evidence.
"""
import time
ENTRY_STARTED = time.monotonic()
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import signal
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VERSION, SOURCE_VERSION = 'DL-replay-stage-r1', 'DL-replay-source-capture-r1'
AUTHORITY_VERSION = 'DL-replay-stage-process-r1'
AUTHORITY_SCOPE = ('Entry start through owned child cleanup, excluding final publication; '
    'external complete CLI exit0 and full wall <=900s are additionally required')
INVENTORY_SHA = '3339b71ca033d5fc065130751fe163aca05a684b2f14f570ab2018219a3a706d'
SOURCE_GIT = '13ec72a214b0c03b886b2c7fac52c8f089966dec'
SOURCE_HOST, TARGET_HOST = 'lyg0360', 'lyg0326'
COUNTS, SECONDS = (1280, 16, 1001008, 1008, 1000000), 900
CPU_ENV = dict(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', MUJOCO_GL='disable')
SOURCES = ('scripts/dl_replay_stage.py', 'scripts/dl_normal_replay.py',
    'scripts/dl_replay_inventory.py')
DEADLINE = float('inf')  # Library authentication shares its caller's budget.
OWNED = {}


def require(value, message):
  if not value:
    raise ValueError(message)


def read_json(body):
  def pairs(items):
    result = {}
    for key, value in items:
      require(key not in result, 'Duplicate staging JSON key')
      result[key] = value
    return result
  return json.loads(body, object_pairs_hook=pairs,
      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite staging JSON')))


def plain(path):
  path = Path(path).absolute()
  require(not any(p.is_symlink() for p in (path, *path.parents)), 'Staging symlink refused')
  return path.resolve(strict=True)


def fresh(path):
  path = Path(path).absolute()
  require(not path.exists() and not path.is_symlink(), 'Fresh staging output required')
  return plain(path.parent)/path.name


def external_output(args, report):
  output = fresh(args.output)
  roots = [Path(report['run_directory']), ROOT]
  if args.action == 'materialize':
    roots.extend(Path(value).absolute() for value in (args.incoming, args.replay_directory))
  require(all(output != root and root not in output.parents for root in roots),
      'Staging receipts must stay outside original run/source/incoming/materialized replay')
  return output


def sha(path):
  digest = hashlib.sha256()
  with path.open('rb') as stream:
    for body in iter(lambda: stream.read(1024*1024), b''):
      check()
      digest.update(body)
  return digest.hexdigest()


def pin(path):
  path = plain(path)
  require(path.is_file(), 'Regular staging input required')
  return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def raw_pinned(path, expected):
  path = plain(path)
  require(path.is_file(), 'Regular pinned staging file required')
  body = path.read_bytes()
  require(hashlib.sha256(body).hexdigest() == expected, 'Staging raw SHA differs before JSON decode')
  return path, body


def write_json(path, value):
  path = fresh(path)
  with path.open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
    stream.write('\n');stream.flush();os.fsync(stream.fileno())


def check():
  require(time.monotonic() < DEADLINE, 'Whole staging900s deadline exceeded')


def host(role):
  actual = socket.gethostname()
  expected = SOURCE_HOST if role == 'source' else TARGET_HOST
  require(actual.split('.')[0] == expected, 'Actual staging hostname differs: '+role)
  return actual


def identity(pid=None):
  pid = os.getpid() if pid is None else pid
  fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
  return dict(pid=pid, pgid=int(fields[2]), start_ticks=int(fields[19]), state=fields[0])


def binding(expected_git):
  require(type(expected_git) is str and re.fullmatch('[0-9a-f]{40}', expected_git),
      'Full staging Git SHA required')
  git = lambda *args: subprocess.check_output(['git', '-C', str(ROOT), *args], timeout=30)
  require(git('rev-parse', 'HEAD').decode().strip() == expected_git and
      not git('status', '--porcelain=v1', '--untracked-files=all'), 'Clean expected staging HEAD required')
  result = {}
  for name in SOURCES:
    body = git('cat-file', 'blob', expected_git+':'+name)
    require(plain(ROOT/name).read_bytes() == body, 'Staging checkout/Git blob differs')
    result[name] = dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
  return result


def absolute_label(value):
  require(type(value) is str and (PurePosixPath(value).is_absolute() or
      PureWindowsPath(value).is_absolute()), 'Original absolute source path required')
  return value


def inventory(path):
  path, body = raw_pinned(path, INVENTORY_SHA)
  report = read_json(body)
  files, chains, rows, resets, actions = COUNTS
  require(report['version'] == 'DL-replay-inventory-r1' and report['schema'] == 1 and
      report['stable'] is True and report['source_git'] == SOURCE_GIT and
      report['rows'] == rows and report['resets'] == resets and report['train_actions'] == actions and
      len(report['chunks']) == files and len(report['chains']) == chains and
      report['before_inventory'] == report['after_inventory'] and
      report['corrupt_chunks_skipped'] == 0 and report['gpu_used'] is report['model_loaded'] is False,
      'Frozen staging inventory/counts differ')
  entries = report['before_inventory']
  require(type(entries) is list and len(entries) == files and
      [entry['path'] for entry in entries] == sorted({entry['path'] for entry in entries}),
      'Complete sorted staging manifest required')
  for entry in entries:
    require(set(entry) == {'path', 'bytes', 'sha256'} and
        type(entry['path']) is str and re.fullmatch(r'[0-9A-Za-z-]+\.npz', entry['path']) and
        type(entry['bytes']) is int and entry['bytes'] > 0 and
        type(entry['sha256']) is str and re.fullmatch('[0-9a-f]{64}', entry['sha256']),
        'Unsafe staging manifest entry')
  absolute_label(report['run_directory']);absolute_label(report['replay_directory'])
  require(report['replay_directory'] == str(Path(report['run_directory'])/'replay'),
      'Original run/replay relationship differs')
  require(set(report['source_metadata']) == {'protocol_manifest.json', 'final_state.json'},
      'Original source metadata names differ')
  return report, dict(path=str(path), bytes=len(body), sha256=INVENTORY_SHA)


def snapshot(directory):
  directory = plain(directory)
  require(directory.is_dir(), 'Regular staging replay directory required')
  result = []
  for path in sorted(directory.iterdir()):
    require(path.is_file() and not path.is_symlink() and path.suffix == '.npz',
        'Only regular NPZ staging chunks allowed')
    before = path.stat()
    digest = sha(path)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
        (after.st_size, after.st_mtime_ns, after.st_ino), 'Staged chunk changed during hash')
    result.append(dict(path=path.name, bytes=after.st_size, sha256=digest))
  require(len(result) == COUNTS[0], 'All original staging chunks required')
  return result


def validate_authority(value, action, actual_hostname):
  require(value['version'] == AUTHORITY_VERSION and value['action'] == action and
      value['actual_hostname'] == actual_hostname and value['returncode'] == 0 and
      value['worker_terminal'] is value['owned_group_cleaned'] is True and
      type(value['seconds']) is int and value['seconds'] == SECONDS and
      type(value['elapsed_seconds']) in (int, float) and
      math.isfinite(value['elapsed_seconds']) and 0 < value['elapsed_seconds'] <= SECONDS and
      value['cpu_environment'] == CPU_ENV and value['failure'] is None,
      'Completed bounded owned staging authority required')
  require(value.get('elapsed_scope') == AUTHORITY_SCOPE,
      'Staging child authority cannot self-certify final CLI exit')
  for key in ('parent_identity', 'worker_identity'):
    ident = value[key]
    require(all(type(ident[name]) is int and ident[name] > 0 for name in
        ('pid', 'pgid', 'start_ticks')), 'Actual staging process identity required')
  require(value['worker_identity']['pid'] == value['worker_identity']['pgid'] and
      value['worker_identity']['pid'] != value['parent_identity']['pid'],
      'Separate owned staging child required')
  require(type(value['started_utc']) is str and type(value['ended_utc']) is str and
      all(type(value[key]) in (int, float) and math.isfinite(value[key]) for key in
          ('started_monotonic', 'ended_monotonic')) and
      value['started_monotonic'] < value['ended_monotonic'], 'Staging authority clock differs')
  for key in ('started_utc', 'ended_utc'):
    require(datetime.datetime.fromisoformat(value[key]).tzinfo is not None,
        'Timezone-aware staging authority timestamp required')
  for key in ('worker_log', 'worker_output'):
    item = value[key]
    require(type(item) is dict and set(item) == {'path', 'bytes', 'sha256'} and
        type(item['bytes']) is int and item['bytes'] >= 0 and
        type(item['sha256']) is str and re.fullmatch('[0-9a-f]{64}', item['sha256']),
        'Pinned staging worker output/log required')
    absolute_label(item['path'])


def validate_source(record, report):
  require(record['version'] == SOURCE_VERSION and record['source_git'] == SOURCE_GIT and
      record['gpu_used'] is record['model_loaded'] is record['transport_performed'] is False and
      type(record['staging_git']) is str and re.fullmatch('[0-9a-f]{40}', record['staging_git']),
      'Source capture version/Git differs')
  source = record['source']
  require(source['logical_server'] == 'sv2' and
      source['actual_hostname'].split('.')[0] == SOURCE_HOST and
      source['run_directory'] == report['run_directory'] and
      source['replay_directory'] == report['replay_directory'] and
      source['before_inventory'] == source['after_inventory'] == report['before_inventory'] and
      source['source_metadata'] == report['source_metadata'], 'Source capture original bytes/path differs')
  original = record['original_inventory']
  require(original['sha256'] == INVENTORY_SHA and type(original['bytes']) is int and
      original['bytes'] > 0, 'Source capture original inventory pin differs')
  absolute_label(original['path'])
  validate_authority(record['authority'], 'source', source['actual_hostname'])
  validate_command(record['authority']['command'], 'source', {
      '--inventory':original['path'], '--expected-git':record['staging_git']})
  return source


def validate_command(command, action, flags):
  require(type(command) is list and len(command) >= 3 and all(type(v) is str for v in command) and
      command[2] == action and command.count('--worker') == 1,
      'Recorded actual staging command differs')
  for flag, value in flags.items():
    require(command.count(flag) == 1 and command[command.index(flag)+1] == value,
        'Recorded actual staging argument differs: '+flag)


def capture(inventory_path, expected_git):
  actual = host('source');before_code = binding(expected_git)
  report, original = inventory(inventory_path)
  directory = plain(report['replay_directory']);run = plain(report['run_directory'])
  require(directory.parent == run, 'Source capture must use the actual original run')
  before = snapshot(directory)
  require(before == report['before_inventory'], 'Source original bytes differ from pinned inventory')
  for name, entry in report['source_metadata'].items():
    require(pin(run/name) == dict(path=str(run/name), **entry), 'Original source metadata differs')
  after = snapshot(directory)
  require(before == after, 'Original source changed between complete SHA passes')
  for name, entry in report['source_metadata'].items():
    require(pin(run/name) == dict(path=str(run/name), **entry), 'Original source metadata changed')
  require(binding(expected_git) == before_code and pin(inventory_path) == original,
      'Source capture code/inventory changed')
  return dict(version=SOURCE_VERSION, source_git=SOURCE_GIT, staging_git=expected_git,
      source_files=before_code, original_inventory=original,
      source=dict(logical_server='sv2', actual_hostname=actual,
          run_directory=str(run), replay_directory=str(directory),
          source_metadata=report['source_metadata'], before_inventory=before, after_inventory=after),
      gpu_used=False, model_loaded=False, transport_performed=False,
      scope='Original byte capture only; no NPZ decode, CRC or online restoration')


def materialize(inventory_path, source_record_path, source_sha, incoming, directory, expected_git):
  actual = host('materialize');before_code = binding(expected_git)
  report, current_inventory = inventory(inventory_path)
  source_path, raw = raw_pinned(source_record_path, source_sha)
  source_record = read_json(raw);source = validate_source(source_record, report)
  require(source_record['original_inventory']['bytes'] == current_inventory['bytes'] and
      source_record['source_files'] == before_code, 'Source/current inventory or staging code differs')
  incoming, directory = plain(incoming), fresh(directory)
  original_run = Path(source['run_directory'])
  require(incoming.is_dir() and directory != incoming and incoming not in directory.parents and
      directory not in incoming.parents and str(directory) != source['replay_directory'] and
      original_run not in directory.parents and directory != original_run and
      original_run not in incoming.parents and incoming != original_run and
      directory != ROOT and ROOT not in directory.parents,
      'Distinct fresh materialized directory required')
  initial = snapshot(incoming)
  require(initial == report['before_inventory'], 'Incoming original bytes differ')
  directory.mkdir()
  for entry in initial:
    check();original = plain(incoming/entry['path']);target = directory/entry['path']
    digest, copied = hashlib.sha256(), 0
    with original.open('rb') as src, target.open('xb') as dst:
      for body in iter(lambda: src.read(1024*1024), b''):
        check();dst.write(body);digest.update(body);copied += len(body)
      dst.flush();os.fsync(dst.fileno())
    require(copied == entry['bytes'] and digest.hexdigest() == entry['sha256'],
        'Transport input changed during exact copy')
  first = snapshot(directory);incoming_after = snapshot(incoming);second = snapshot(directory)
  require(first == second == incoming_after == initial, 'Materialized/incoming complete SHA passes differ')
  require(pin(inventory_path) == current_inventory and pin(source_path) ==
      dict(path=str(source_path), bytes=len(raw), sha256=source_sha) and binding(expected_git) == before_code,
      'Materialization inputs/code changed')
  return dict(version=VERSION, source_git=SOURCE_GIT, staging_git=expected_git,
      source_files=before_code, original_inventory=source_record['original_inventory'],
      current_inventory=current_inventory,
      source_capture=dict(path=str(source_path), bytes=len(raw), sha256=source_sha),
      source=source, source_authority=source_record['authority'],
      materialized=dict(logical_server='sv3', actual_hostname=actual,
          replay_directory=str(directory), incoming_directory=str(incoming),
          before_inventory=first, after_inventory=second,
          incoming_before_inventory=initial, incoming_after_inventory=incoming_after),
      gpu_used=False, model_loaded=False, original_files_written=False,
      inventory_rewritten=False, crc_or_schema_revalidated=False,
      scope='Byte-exact relocation; loader must still decode/CRC/schema/chain-check all original chunks')


def authenticate_stage(stage_path, expected_stage_sha256, inventory_path, report, directory):
  """Pure read-only library path; never changes the caller's CUDA/JAX env."""
  require(type(expected_stage_sha256) is str and re.fullmatch('[0-9a-f]{64}', expected_stage_sha256),
      'Explicit full stage raw SHA required')
  path, body = raw_pinned(stage_path, expected_stage_sha256)
  record = read_json(body)
  require(record['version'] == VERSION and record['source_git'] == SOURCE_GIT and
      type(record['staging_git']) is str and re.fullmatch('[0-9a-f]{40}', record['staging_git']) and
      record['gpu_used'] is record['model_loaded'] is record['original_files_written'] is
      record['inventory_rewritten'] is record['crc_or_schema_revalidated'] is False,
      'Stage version/source/scope differs')
  actual_report, current = inventory(inventory_path)
  require(actual_report == report, 'Caller report differs from the raw-pinned original inventory')
  require(record['current_inventory'] == current, 'Stage current original-inventory bytes/path differ')
  captured = record['source_capture']
  source_path, source_raw = raw_pinned(captured['path'], captured['sha256'])
  require(pin(source_path) == captured, 'Source capture bytes/size differ')
  source_record = read_json(source_raw);source = validate_source(source_record, report)
  require(record['source'] == source and record['source_authority'] == source_record['authority'] and
      record['original_inventory'] == source_record['original_inventory'] and
      record['original_inventory']['bytes'] == current['bytes'] and
      record['source_files'] == source_record['source_files'], 'Stage/source capture binding differs')
  target = record['materialized'];actual = host('materialize')
  require(target['logical_server'] == 'sv3' and target['actual_hostname'] == actual and
      target['replay_directory'] == str(directory) and
      target['before_inventory'] == target['after_inventory'] ==
      target['incoming_before_inventory'] == target['incoming_after_inventory'] == report['before_inventory'],
      'Explicit materialized path/hostname/complete byte manifest differs')
  require(str(directory) != source['replay_directory'] and
      Path(source['run_directory']) not in directory.parents and str(directory) != source['run_directory'],
      'Relocation cannot impersonate the original run')
  validate_authority(record['authority'], 'materialize', actual)
  validate_command(record['authority']['command'], 'materialize', {
      '--inventory':current['path'], '--expected-git':record['staging_git'],
      '--source-record':captured['path'], '--expected-source-record-sha256':captured['sha256'],
      '--incoming':target['incoming_directory'], '--replay-directory':str(directory)})
  worker_output = record['authority']['worker_output']
  output_path, output_raw = raw_pinned(worker_output['path'], worker_output['sha256'])
  require(pin(output_path) == worker_output and read_json(output_raw) ==
      {key:value for key,value in record.items() if key != 'authority'},
      'Completed materialization worker raw output differs from stage record')
  require(pin(record['authority']['worker_log']['path']) == record['authority']['worker_log'],
      'Completed materialization worker log differs')
  # Code bytes are transport provenance; the enclosing producer/auditor binds
  # their own checkout/Git and the complete stage raw pin separately.
  require(set(record['source_files']) == set(SOURCES), 'Complete staging source blob pins required')
  for name in SOURCES:
    actual_pin = pin(ROOT/name)
    require(record['source_files'][name] == {key:actual_pin[key] for key in ('bytes', 'sha256')},
        'Actual staging source blob bytes differ')
  return dict(version=VERSION, record=dict(path=str(path), bytes=len(body), sha256=expected_stage_sha256),
      source_capture=captured, original_inventory=record['original_inventory'],
      source_logical_server='sv2', source_actual_hostname=source['actual_hostname'],
      original_run_directory=source['run_directory'], original_replay_directory=source['replay_directory'],
      materialized_logical_server='sv3', materialized_actual_hostname=actual,
      materialized_replay_directory=str(directory), staging_git=record['staging_git'],
      source_staging_git=source_record['staging_git'], copied_bytes_verified=True,
      remote_ssh_execution_independently_proven=False, external_full_entry_exit_required=True)


def group_alive(owned):
  try:
    os.killpg(owned['pgid'], 0)
  except ProcessLookupError:
    return []
  current = Path(f"/proc/{owned['pid']}/stat")
  if current.exists():
    try:
      now = identity(owned['pid'])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
      now = None
    require(now is None or now['start_ticks'] == owned['start_ticks'],
        'Owned staging leader PID was reused')
  result = []
  for path in Path('/proc').iterdir():
    if not path.name.isdigit():
      continue
    try:
      fields = (path/'stat').read_text().rsplit(') ', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
      continue
    if int(fields[2]) == owned['pgid'] and fields[0] not in ('Z', 'X'):
      require(int(fields[3]) == owned['pgid'] and int(fields[19]) >= owned['start_ticks'],
          'Staging process group lost original session ownership')
      result.append(int(path.name))
  return result


def cleanup(child, owned):
  if child.poll() is None:
    actual = identity(child.pid)
    require(all(actual[key] == owned[key] for key in ('pid', 'pgid', 'start_ticks')),
        'Owned staging child identity changed')
    os.killpg(child.pid, signal.SIGTERM)
    try:
      child.wait(timeout=3)
    except subprocess.TimeoutExpired:
      os.killpg(child.pid, signal.SIGKILL);child.wait(timeout=3)
  if group_alive(owned):
    os.killpg(owned['pgid'], signal.SIGKILL)
    until = time.monotonic()+3
    while group_alive(owned) and time.monotonic() < until:
      time.sleep(.01)
  require(not group_alive(owned), 'Owned staging process group remains live')
  OWNED.pop(child.pid, None)


def expire(signum, frame):
  for owned in tuple(OWNED.values()):
    try:
      if group_alive(owned):
        os.killpg(owned['pgid'], signal.SIGKILL)
    except (ProcessLookupError, FileNotFoundError):
      pass
  os._exit(124)


def supervise(args):
  require(sys.platform == 'linux', 'Actual staging CLI requires Linux')
  os.environ.update(CPU_ENV)
  require(not any(name == 'jax' or name.startswith('jax.') for name in sys.modules),
      'Staging CLI must not import JAX')
  actual = host(args.action);code = binding(args.expected_git)
  report, _ = inventory(args.inventory)
  output = external_output(args, report);work = fresh(str(output)+'.work')
  work.mkdir()
  started, utc = time.monotonic(), datetime.datetime.now(datetime.timezone.utc).isoformat()
  parent = identity();child = None;owned = None;failure = None
  command = [sys.executable, str(Path(__file__).resolve()), args.action,
      '--inventory', args.inventory, '--output', str(work/'capture.json'),
      '--expected-git', args.expected_git, '--worker']
  if args.action == 'materialize':
    command += ['--source-record', args.source_record,
        '--expected-source-record-sha256', args.expected_source_record_sha256,
        '--incoming', args.incoming, '--replay-directory', args.replay_directory]
  try:
    with (work/'worker.log').open('xb') as log:
      child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
          env=dict(os.environ, DL_REPLAY_STAGE_DEADLINE=str(DEADLINE),
              DL_REPLAY_STAGE_PARENT=str(os.getpid())), start_new_session=True)
      owned = identity(child.pid)
      require(owned['pid'] == owned['pgid'], 'Owned staging child must lead its process group')
      OWNED[child.pid] = owned
      child.wait(timeout=max(.001, DEADLINE-time.monotonic()))
      require(child.returncode == 0, 'Staging worker failed')
  except Exception as error:
    failure = repr(error)
  finally:
    if child is not None and owned is not None:
      cleanup(child, owned)
  ended = time.monotonic()
  authority = dict(version=AUTHORITY_VERSION, action=args.action, actual_hostname=actual,
      parent_identity=parent, worker_identity=owned, command=command,
      seconds=SECONDS, elapsed_seconds=ended-ENTRY_STARTED, elapsed_scope=AUTHORITY_SCOPE,
      started_utc=utc, ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      started_monotonic=started, ended_monotonic=ended,
      returncode=child.returncode if child is not None else None,
      worker_terminal=child is not None and child.poll() is not None,
      owned_group_cleaned=child is not None and owned is not None and not group_alive(owned),
      cpu_environment=CPU_ENV, failure=failure)
  authority.update(worker_log=pin(work/'worker.log'),
      worker_output=pin(work/'capture.json') if (work/'capture.json').is_file() else None)
  write_json(work/'terminal.json', authority)
  require(failure is None and ended < DEADLINE, 'Staging incomplete; partial directory retained')
  validate_authority(authority, args.action, actual)
  require(binding(args.expected_git) == code, 'Staging code changed before publication')
  check()
  payload = read_json((work/'capture.json').read_bytes());payload['authority'] = authority
  write_json(output, payload)
  print(json.dumps(dict(version=payload['version'], output=str(output), sha256=sha(output),
      elapsed_seconds=time.monotonic()-ENTRY_STARTED, GPU_used=False)), flush=True)
  check()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('action', choices=('source', 'materialize'))
  for name in ('inventory', 'output', 'expected-git'):
    parser.add_argument('--'+name, required=True)
  for name in ('source-record', 'expected-source-record-sha256', 'incoming', 'replay-directory'):
    parser.add_argument('--'+name)
  parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
  args = parser.parse_args()
  extra = (args.source_record, args.expected_source_record_sha256, args.incoming, args.replay_directory)
  require(all(value is None for value in extra) if args.action == 'source' else all(extra),
      'Source/materialize arguments differ')
  global DEADLINE
  if not args.worker:
    DEADLINE = ENTRY_STARTED+SECONDS
    require(sys.platform == 'linux', 'Actual staging CLI requires Linux')
    signal.signal(signal.SIGALRM, expire)
    signal.signal(signal.SIGTERM, expire)
    signal.setitimer(signal.ITIMER_REAL, max(.001, DEADLINE-time.monotonic()))
    supervise(args);return
  DEADLINE = float(os.environ['DL_REPLAY_STAGE_DEADLINE'])
  require(sys.platform == 'linux' and int(os.environ['DL_REPLAY_STAGE_PARENT']) == os.getppid() and
      os.getpgrp() == os.getpid(), 'Direct owned Linux staging child required')
  os.environ.update(CPU_ENV)
  signal.signal(signal.SIGALRM, lambda *args: os._exit(124))
  signal.setitimer(signal.ITIMER_REAL, max(.001, DEADLINE-time.monotonic()))
  check()
  report, _ = inventory(args.inventory)
  external_output(args, report)
  if args.action == 'source':
    result = capture(args.inventory, args.expected_git)
  else:
    result = materialize(args.inventory, args.source_record,
        args.expected_source_record_sha256, args.incoming, args.replay_directory, args.expected_git)
  check();write_json(args.output, result)


if __name__ == '__main__':
  main()
