"""DL: strict offline N1M persistent replay inventory; never Replay.load()."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import string
import subprocess
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'DL-replay-inventory-r1'
SOURCE_GIT = '13ec72a214b0c03b886b2c7fac52c8f089966dec'
EXPECTED_FILES, CHAINS, TRAIN_ACTIONS = 1280, 16, 1000000
FIELDS = {'image': ('uint8', (64, 64, 3)), 'reward': ('float32', ()),
    'is_first': ('bool', ()), 'is_last': ('bool', ()), 'is_terminal': ('bool', ()),
    'action': ('float32', (12,)), 'stepid': ('uint8', (20,)),
    'dyn/deter': ('float32', (4096,)), 'dyn/stoch': ('float32', (32, 32))}
SOURCES = ('embodied/core/chunk.py', 'embodied/core/replay.py', 'embodied/core/driver.py',
    'embodied/jax/agent.py', 'dreamerv3/agent.py', 'dreamerv3/rssm.py',
    'dreamerv3/main.py', 'scripts/dl_pilot_train.py')
SELF = 'scripts/dl_replay_inventory.py'
ALPHABET = string.digits + string.ascii_letters


def require(value, message):
  if not value:
    raise ValueError(message)


def read_json(raw):
  def pairs(values):
    result = {}
    for key, value in values:
      require(key not in result, 'Duplicate metadata JSON key')
      result[key] = value
    return result
  return json.loads(raw, object_pairs_hook=pairs,
      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite metadata JSON')))


def sha(path):
  digest = hashlib.sha256()
  with path.open('rb') as stream:
    for value in iter(lambda: stream.read(1024 * 1024), b''):
      digest.update(value)
  return digest.hexdigest()


def plain(path):
  path = Path(path).absolute()
  require(not any(p.is_symlink() for p in (path, *path.parents)), 'Symlink path refused')
  return path.resolve(strict=True)


def snapshot(directory):
  result = []
  for path in sorted(directory.iterdir()):
    require(not path.is_symlink() and path.is_file() and path.suffix == '.npz',
        'Replay must contain only regular NPZ chunks: ' + path.name)
    before = path.stat()
    value = sha(path)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino)
        == (after.st_size, after.st_mtime_ns, after.st_ino), 'Chunk changed during hash')
    result.append(dict(path=path.name, bytes=after.st_size, sha256=value))
  require(len(result) == EXPECTED_FILES, 'Persistent chunk count differs from frozen N1M inventory')
  return result


def uuid_bytes(text):
  require(re.fullmatch('[0-9a-zA-Z]{22}', text), 'Noncanonical official base62 UUID')
  number = 0
  for char in text:
    number = number * 62 + ALPHABET.index(char)
  require(number < 2 ** 128, 'UUID exceeds 16 bytes')
  return number.to_bytes(16, 'big')


def filename(name):
  match = re.fullmatch(r'(\d{8}T\d{6}F\d{6})-([0-9a-zA-Z]{22})-([0-9a-zA-Z]{22})-([1-9]\d*)\.npz', name)
  require(match is not None, 'Malformed official chunk filename: ' + name)
  timestamp, uuid, successor, length = match.groups()
  datetime.datetime.strptime(timestamp, '%Y%m%dT%H%M%SF%f')
  require(0 < int(length) <= 1024, 'Chunk length outside configured capacity')
  require(uuid_bytes(uuid) != bytes(16) and uuid_bytes(successor) != bytes(16)
      and uuid != successor, 'Invalid completed replay UUID/successor')
  return dict(path=name, time=timestamp, uuid=uuid, successor=successor, length=int(length))


def validate_chunk(path):
  row = filename(path.name)
  length = row['length']
  with path.open('rb') as stream:
    with zipfile.ZipFile(stream) as archive:
      members = archive.infolist()
      require(len(members) == len(FIELDS) and {m.filename for m in members}
          == {name + '.npy' for name in FIELDS}, 'Duplicate/missing/extra NPZ members')
      for member in members:
        name = member.filename[:-4]
        dtype, tail = FIELDS[name]
        expected = length * int(np.prod(tail, dtype=np.int64)) * np.dtype(dtype).itemsize
        require(member.compress_type == zipfile.ZIP_DEFLATED and not member.flag_bits & 1
            and member.file_size <= expected + 10000, 'Unexpected NPZ compression/size')
        with archive.open(member) as array:
          version = np.lib.format.read_magic(array)
          require(version in ((1, 0), (2, 0)), 'Unexpected official NPY version')
          reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
              else np.lib.format.read_array_header_2_0)
          shape, fortran, actual = reader(array, max_header_size=10000)
          require(shape == (length, *tail) and actual == np.dtype(dtype) and not fortran
              and member.file_size - array.tell() == expected, 'NPY dtype/shape/length/header differs')
      require(archive.testzip() is None, 'NPZ CRC verification failed')
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as data:
      require(set(data.files) == set(FIELDS), 'NPZ field names differ')
      arrays = {name: data[name] for name in FIELDS}
  for name, (dtype, tail) in FIELDS.items():
    value = arrays[name]
    require(value.dtype == np.dtype(dtype) and value.shape == (length, *tail)
        and np.isfinite(value).all(), 'Nonfinite/incorrect loaded chunk field: ' + name)
  identifiers = arrays['stepid']
  require(np.all(identifiers[:, :16] == np.frombuffer(uuid_bytes(row['uuid']), np.uint8)[None])
      and np.array_equal(identifiers[:, 16:], np.arange(length, dtype='>u4').view(np.uint8).reshape(length, 4)),
      'stepid is not filename UUID plus the exact big-endian row index')
  first, last, terminal = (arrays[k] for k in ('is_first', 'is_last', 'is_terminal'))
  require(np.all(~terminal | last) and np.array_equal(first[1:], last[:-1])
      and np.all(arrays['reward'][first] == 0) and np.all(arrays['action'][last] == 0),
      'Persistent episode flags/reset reward/last action semantics differ')
  latent = arrays['dyn/stoch']
  require(np.all((latent == 0) | (latent == 1)) and np.all(latent.sum(-1) == 1),
      'Cached stochastic context is not actual categorical onehot')
  return dict(row, first=bool(first[0]), last=bool(last[-1]),
      resets=int(first.sum()), terminals=int(terminal.sum()), ends=int(last.sum()),
      crc_verified=True, cached_context_validated=True)


def validate_chains(rows):
  nodes = {row['uuid']: row for row in rows}
  require(len(nodes) == len(rows), 'Duplicate chunk UUID')
  incoming = {}
  for row in rows:
    target = row['successor']
    require(target not in incoming, 'Replay chains merge or share a dangling successor')
    incoming[target] = row['uuid']
    if target in nodes:
      child = nodes[target]
      require(child['time'] >= row['time'] and child['first'] == row['last'],
          'Successor timestamp or cross-chunk episode boundary differs')
  heads = sorted(set(nodes) - set(incoming))
  require(len(heads) == CHAINS, 'Expected exactly sixteen persistent stream chains')
  seen, result = set(), []
  for head in heads:
    require(nodes[head]['first'], 'A full persistent stream must begin at its actual reset')
    current, paths, count = head, [], 0
    while current in nodes:
      require(current not in seen, 'Cycle or duplicate chain coverage')
      seen.add(current)
      row = nodes[current]
      paths.append(row['path'])
      count += row['length']
      current = row['successor']
    result.append(dict(head_uuid=head, dangling_successor=current, paths=paths,
        rows=count, length65_windows=max(0, count - 64)))
  require(len(seen) == len(nodes), 'Unreachable cycle/chunk outside the sixteen chains')
  return result


def binding(source_git, audit_git):
  require(source_git == SOURCE_GIT and re.fullmatch('[0-9a-f]{40}', audit_git), 'Full expected Git SHAs required')
  def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], timeout=30)
  require(git('rev-parse', 'HEAD').decode().strip() == audit_git
      and not git('status', '--porcelain=v1', '--untracked-files=all'), 'Clean audit HEAD required')
  require(git('show', audit_git + ':' + SELF) == Path(__file__).read_bytes(), 'Audit script/blob bytes differ')
  return {name: hashlib.sha256(git('show', source_git + ':' + name)).hexdigest() for name in SOURCES}


def validate(run, replay, output, source_git, audit_git):
  run, replay = plain(run), plain(replay)
  require(replay == run / 'replay' and replay.is_dir(), 'Exact run/replay directory required')
  output = Path(output).absolute()
  require(not output.exists() and not output.is_symlink(), 'Fresh external output required')
  output = plain(output.parent) / output.name
  require(run not in output.parents and output != run, 'Output must stay outside original run')
  sourcepins = binding(source_git, audit_git)
  metadata = {}
  for name in ('protocol_manifest.json', 'final_state.json'):
    path = plain(run / name)
    require(path.is_file(), 'Run metadata is not a regular file')
    raw = path.read_bytes()
    metadata[name] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), value=read_json(raw))
  manifest, final = (metadata[n]['value'] for n in ('protocol_manifest.json', 'final_state.json'))
  require(manifest['git_commit'] == final['git_commit'] == source_git
      and manifest['group'] == final['group'] == 'N' and manifest['root_seed'] == final['seed'] == 0
      and manifest['task'] == final['task'] == 'dmc_quadruped_walk', 'Declared N1M source identity differs')
  expected = {'agent.dyn.rssm.deter': 4096, 'agent.dyn.rssm.stoch': 32,
      'agent.dyn.rssm.classes': 32, 'replay_context': 1, 'replay.chunksize': 1024,
      'run.envs': 16, 'batch_size': 16, 'batch_length': 64, 'consec_train': 1}
  require(all(type(manifest['config'].get(k)) is int and manifest['config'][k] == v
      for k, v in expected.items()), 'Frozen replay/context schema config differs')
  require(type(manifest['root_seed']) is int and type(final['seed']) is int
      and final['train_action_steps'] == TRAIN_ACTIONS and type(final['train_resets']) is int
      and final['status'] == 'training_ended_pending_audit', 'Frozen completed-run counters differ')
  before = snapshot(replay)
  chunks = []
  for row in before:
    try:
      chunks.append(validate_chunk(replay / row['path']))
    except Exception as error:
      raise ValueError('Replay chunk rejected: ' + row['path']) from error
  chains = validate_chains(chunks)
  resets, total = sum(x['resets'] for x in chunks), sum(x['length'] for x in chunks)
  require(resets == final['train_resets'] and total - resets == final['train_action_steps'],
      'Persistent replay rows/resets do not account for all recorded train transitions')
  after = snapshot(replay)
  require(before == after, 'Replay inventory changed between complete SHA passes')
  for name, pin in metadata.items():
    require(sha(run / name) == pin['sha256'], 'Source metadata changed during validation')
  require(binding(source_git, audit_git) == sourcepins, 'Audit/source code binding changed')
  report = dict(version=VERSION, schema=1, source_git=source_git, audit_git=audit_git,
      run_directory=str(run), replay_directory=str(replay), captured_utc=datetime.datetime.now(
          datetime.timezone.utc).isoformat(), numpy_version=np.__version__,
      before_inventory=before, after_inventory=after,
      stable=True, sourcepins=sourcepins, source_metadata={n: {k: v for k, v in p.items()
          if k != 'value'} for n, p in metadata.items()}, fields={k: [v[0], list(v[1])] for k, v in FIELDS.items()},
      chunks=chunks, chains=chains, rows=total, resets=resets, train_actions=total - resets,
      object_scope='offline persistent buffer; cached contexts are historical and may mix update ages',
      sampler_rng_restored=False, online_queue_restored=False, environment_restored=False,
      seamless_resume_established=False, model_loaded=False, gpu_used=False, corrupt_chunks_skipped=0)
  with output.open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
    stream.write('\n')
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('run', 'replay', 'output', 'source-git', 'audit-git'):
    parser.add_argument('--' + name, required=True)
  args = parser.parse_args()
  report = validate(args.run, args.replay, args.output, args.source_git, args.audit_git)
  print(json.dumps(dict(version=VERSION, output=str(Path(args.output).resolve()),
      sha256=sha(Path(args.output)), chunks=len(report['chunks']), chains=len(report['chains']),
      rows=report['rows'], gpu_used=False), sort_keys=True))


if __name__ == '__main__':
  main()
