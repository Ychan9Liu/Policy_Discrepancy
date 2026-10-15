"""Full N1M private-replay CPU engineering probe; no model or learning.

The production CLI has no fixture, count, inventory-pin or budget override.
Synthetic cache entries exercise writeback, never represent network output.
"""
import time
ENTRY_STARTED = time.monotonic()
import os
import signal
import sys
SECONDS, RAM_BYTES = 900, 96*1024**3
DEADLINE, OWNED = ENTRY_STARTED+SECONDS, set()


def expire(signum, frame):
  for pid in tuple(OWNED):
    try:
      os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
      pass
  os._exit(124)


if __name__ == '__main__' and sys.platform == 'linux':
  signal.signal(signal.SIGALRM, expire)
  signal.signal(signal.SIGTERM, expire)
  signal.setitimer(signal.ITIMER_REAL, max(.001, DEADLINE-time.monotonic()))

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))
VERSION = 'DL-normal-private-replay-probe-r1'
INVENTORY_SHA = '3339b71ca033d5fc065130751fe163aca05a684b2f14f570ab2018219a3a706d'
ENGINEERING_SEED, LEGAL_STARTS = 20261035, 999984
CPU_ENV = dict(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', MUJOCO_GL='disable')


def require(value, message):
  if not value:
    raise ValueError(message)


def check():
  require(time.monotonic() < DEADLINE, 'Whole CPU probe900s deadline exceeded')
  require(all(os.environ.get(k) == v for k, v in CPU_ENV.items()), 'CPU environment changed')


def configure_cpu():
  os.environ.update(CPU_ENV)
  for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'


def limit_memory():
  import resource
  _, old_hard = resource.getrlimit(resource.RLIMIT_AS)
  cap = RAM_BYTES if old_hard == resource.RLIM_INFINITY else min(RAM_BYTES, old_hard)
  resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
  return dict(kind='Linux RLIMIT_AS; virtual address upper bound also bounds RSS',
      requested_bytes=RAM_BYTES, actual_soft_bytes=cap, actual_hard_bytes=cap)


def authenticated_inventory(path):
  """Wrong pins fail before JSON or NPZ decoding/importing the replay helper."""
  check()
  path = Path(path)
  require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
      'Regular inventory path required')
  body = path.read_bytes()
  require(hashlib.sha256(body).hexdigest() == INVENTORY_SHA,
      'Fixed inventory raw SHA differs before decode')
  return body


def exercise_cache(cache):
  """One engineering plan/batch, with independent stepid last-writer checks."""
  import numpy as np
  from scripts import dl_normal_replay as helper
  check()
  starts = cache.make_plan(ENGINEERING_SEED, 1)[0]
  before = cache.batch(starts)
  check()
  out = dict(stepid=before['stepid'][:, 1:].copy(), **{
      'dyn/deter': before['dyn/deter'][:, 1:]+np.float32(1),
      'dyn/stoch': np.roll(before['dyn/stoch'][:, 1:], 1, axis=-1).copy()})
  writers = {out['stepid'][r, t].tobytes(): (r, t) for r in range(16) for t in range(64)}
  before_digest = helper._digest(before)
  writeback = cache.strict_update(out, before)
  after = cache.batch(starts)
  for key in set(before)-set(helper.CACHE):
    require(before[key].dtype == after[key].dtype and before[key].shape == after[key].shape and
        before[key].tobytes() == after[key].tobytes(), 'Synthetic writeback polluted raw field '+key)
  for r in range(16):
    for t in range(65):
      writer = writers.get(after['stepid'][r, t].tobytes())
      for key in helper.CACHE:
        expected = before[key][r, t] if writer is None else out[key][writer]
        require(after[key][r, t].tobytes() == expected.tobytes(),
            'Independent stepid last-writer cache check failed')
  require(not np.array_equal(before['dyn/stoch'], after['dyn/stoch']), 'Synthetic cache did not change')
  # Consume this inspection batch without another transform or any model call.
  drained = cache.strict_update(dict(stepid=after['stepid'][:, 1:].copy(),
      **{key: after[key][:, 1:].copy() for key in helper.CACHE}), after)
  check()
  return dict(plan=starts, plan_seed=ENGINEERING_SEED, planned_batches=1,
      batch_shape=[16, 65], loss_shape=[16, 64], before_sha256=before_digest,
      after_sha256=helper._digest(after), writeback=writeback, inspection_drain=drained,
      unique_written_stepids=len(writers), raw_fields_bytes_unchanged=True,
      all65_cache_positions_checked=True, pending_cleared=cache._pending is None,
      synthetic_transform='F32 deter+1; onehot stoch roll1 on final axis',
      network_output_used=False, learning_performed=False, real_1m_import_established=False)


def self_gpu_receipt(lifecycle):
  """Read-only XML+pmon evidence, restricted to our PID; no backend calls."""
  commands = [('xml', ['nvidia-smi', '-q', '-x']),
      ('pmon', ['nvidia-smi', 'pmon', '-c', '1', '-s', 'um'])]
  receipts, limitations = {}, []
  for name, command in commands:
    try:
      raw = lifecycle.command_output(command)
    except Exception as error:
      check()
      limitations.append(name+': '+repr(error))
      continue
    if name == 'xml':
      pids = [int(node.findtext('pid')) for node in ET.fromstring(raw).findall('.//process_info')]
    else:
      pids = [int(parts[1]) for line in raw.decode().splitlines()
          if (parts := line.split()) and not line.lstrip().startswith('#') and
          len(parts) > 1 and parts[1].isdigit()]
    require(os.getpid() not in pids, 'Probe PID unexpectedly owns a CUDA/EGL GPU process')
    receipts[name] = dict(command=command, sha256=hashlib.sha256(raw).hexdigest(),
        bytes=len(raw), own_pid_present=False)
  return dict(pid=os.getpid(), both_queries_verified=len(receipts) == 2,
      queries=receipts, limitations=limitations,
      scope='Process inventory snapshots only; no independent backend/driver proof')


def run(args):
  require(sys.platform == 'linux', 'Production probe requires Linux CPU')
  check()
  memory_limit = limit_memory()
  require('DL_R5_PARENT' not in os.environ, 'CPU probe must own its metadata helpers')
  from scripts import dl_live_trial as lifecycle
  lifecycle.DEADLINE, lifecycle.OWNED = DEADLINE, OWNED
  sources = lifecycle.source_binding(args.expected_git)
  require({'scripts/dl_normal_replay_probe.py', 'scripts/dl_normal_replay.py',
      'scripts/dl_replay_inventory.py'} <= sources.keys(), 'Tracked replay probe source missing')
  inventory_body = authenticated_inventory(args.inventory)
  from scripts import dl_normal_replay as helper
  inventory = helper.inventory
  report = inventory.read_json(inventory_body)
  directory = inventory.plain(args.replay_directory)
  output = Path(args.output).absolute()
  require(not output.exists() and not output.is_symlink(), 'Fresh output JSON required')
  output = inventory.plain(output.parent)/output.name
  require(directory.parent not in output.parents and output != directory.parent,
      'Output must stay outside original run')
  metadata = report['source_metadata']
  def source_metadata():
    for name, pin in metadata.items():
      path = inventory.plain(directory.parent/name)
      require(path.is_file() and path.stat().st_size == pin['bytes'] and
          inventory.sha(path) == pin['sha256'], 'Original run metadata changed')
  source_metadata()
  gpu_before = self_gpu_receipt(lifecycle)
  loaded_at = time.monotonic()
  cache = helper.load_private_cache(directory, args.inventory)
  load_seconds = time.monotonic()-loaded_at
  check()
  receipt = cache.receipt()
  require(receipt['original_inventory_authenticated'] and receipt['chunks'] == 1280 and
      receipt['legal65_starts'] == LEGAL_STARTS, 'Full real1M official registration/count differs')
  synthetic = exercise_cache(cache)
  after = inventory.snapshot(directory)
  require(after == report['before_inventory'] == report['after_inventory'],
      'Original full1280 raw SHA inventory changed')
  source_metadata()
  require(lifecycle.source_binding(args.expected_git) == sources, 'Source binding changed during probe')
  gpu_after = self_gpu_receipt(lifecycle)
  import resource
  check()
  peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
  require(peak_rss <= RAM_BYTES, 'Measured process RSS exceeded96GiB cap')
  result = dict(version=VERSION, status='complete-real1M-cpu-engineering-probe',
      git_sha=args.expected_git, sources=sources,
      inputs=dict(replay_directory=str(directory), inventory=str(Path(args.inventory).resolve()),
          inventory_sha256=INVENTORY_SHA, source_metadata=metadata,
          original_files=len(after), original_bytes=sum(x['bytes'] for x in after),
          source_rows=report['rows'], source_reset_rows=report['resets'],
          source_train_actions=report['train_actions'], chains=len(report['chains'])),
      replay=receipt, synthetic_writeback=synthetic, load_seconds=load_seconds,
      elapsed_seconds=time.monotonic()-ENTRY_STARTED, whole_entry_limit_seconds=SECONDS,
      memory_limit=memory_limit,
      maximum_rss_bytes=peak_rss,
      after_inventory_canonical_sha256=hashlib.sha256(json.dumps(after,
          sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
      original_before_after_raw_hashes_equal=True, original_files_written=False,
      real_1m_import_established=True, model_loaded=False, optimizer_updates=0,
      training_performed=False, method_effectiveness_established=False,
      backend_creation_requested=False, jax_imported='jax' in sys.modules,
      self_gpu_before=gpu_before, self_gpu_after=gpu_after,
      owned_helpers_remaining=sorted(OWNED), limitations=[
          'Synthetic F32/+1 and onehot roll are engineering writes, never NN outputs or a learning result.',
          'Historical saved contexts mix update ages; sampler RNG, online queue and environment are not restored.',
          'Official Replay/Consec/_getseq/update are reused, not independently reimplemented.',
          'A complete JSON alone is not a successful terminal receipt; require outer exit0 and bound/cleanup evidence.',
          'One fixed engineering plan16 does not validate all future learning plans or define calibration/update budgets.'])
  require(not OWNED, 'Owned metadata helper remains active')
  with output.open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
    stream.write('\n')
    stream.flush()
    os.fsync(stream.fileno())
  check()
  return result


def main():
  configure_cpu()
  require(sys.platform == 'linux', 'Production probe requires Linux CPU')
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('expected-git', 'replay-directory', 'inventory', 'output'):
    parser.add_argument('--'+name, required=True)
  args = parser.parse_args()
  result = run(args)
  # Also print limits, outside the immutable result, for the outer launcher log.
  print(json.dumps(dict(status=result['status'], memory_limit=result['memory_limit'],
      elapsed_seconds=time.monotonic()-ENTRY_STARTED)), flush=True)


if __name__ == '__main__':
  main()
