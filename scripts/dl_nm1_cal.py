"""Bounded DL-NM1 C producer. Importing this entry initializes no GPU/model.

This is128 independent scratch counterfactuals, never cumulative learning.
The stdlib parent supervises one owned worker and persists resource conflicts.
All scientific qualifications, including negative ones, require complete128.
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
VERSION, SCHEMA, SECONDS = 'DL-NM1-cal-r1', 1, 3600
HOST, GPU = 'lyg0326', 5
GPU_UUID = 'GPU-79704918-e096-92b4-252c-e3244497a8b3'
RESOURCE_VERSION = 'DL-NM1-compute-resource-r1'
COMPUTE_ENV = dict(CUDA_VISIBLE_DEVICES=GPU_UUID, CUDA_DEVICE_ORDER='PCI_BUS_ID',
    JAX_PLATFORMS='cuda', MUJOCO_GL='disable', MUJOCO_EGL_DEVICE_ID=None, PYOPENGL_PLATFORM=None)
CHECKPOINT = 'c6a7d92e24669a515bb44693faf73e442d9868f9fc814eec0e22e069aa594d24'
CONFIG = '877f8470cb4f7f45d143e3099fbd3f55a828991afe33779b7b04c4264f6c8377'
INVENTORY = '3339b71ca033d5fc065130751fe163aca05a684b2f14f570ab2018219a3a706d'
FULL_STATE = '4e0d1a0e1de806bf8e4863278cabd171ed0d208dddc1fea18b8b54d68ca6383f'
PARAMS = 'cffc13c0dcdfbd0c8b0e5c503c9d6251f4b1550c08b85a8742a638655d66d372'
BATCHES, BATCH, LENGTH, CONTEXT, STEP = 128, 16, 64, 1, 249493
COUNTERS = dict(updates=STEP, batches=STEP, actions=62563, eval_actions=1001, eval_seed=1289196729)
CONSTANTS = dict(batches=BATCHES, batch_size=BATCH, loss_length=LENGTH, context=CONTEXT,
    plan_root=20261101, plan_tag=0x4E4D4341, model_seed=0, opt_step=STEP)
NORMS = {'docs/DL_NM1_TRIAL.md', 'docs/DL_NM1_CAL_ARTIFACTS.md'}
TOP_FILES = {'budget.json', 'plan.json', 'initial.json', 'initial.npz', 'resource_refreshed.json',
    'runtime_before.json', 'runtime_after.json', 'frozen.json', 'summary.json', 'final.json'}
BATCH_FILES = {'evidence.npz', 'record.json', 'controls.npz', 'controls.json'}
DEADLINE, OWNED = ENTRY_STARTED + SECONDS, {}


def require(value, message):
  if not value:
    raise ValueError(message)


def check():
  require(time.monotonic() < DEADLINE, 'Whole NM1 C3600sec deadline exceeded')


def require_host():
  require(os.name == 'posix' and socket.gethostname().split('.')[0] == HOST, 'Assigned sv3 Linux required')


def sha(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for block in iter(lambda: stream.read(1024*1024), b''):
      digest.update(block)
  return digest.hexdigest()


def canonical(value):
  return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


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
    stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
  check()


def pin(path, relative=False):
  path = Path(path)
  require(not path.is_symlink() and path.is_file(), 'Pinned regular file required')
  return dict(path=path.name if relative else str(path.resolve()), bytes=path.stat().st_size, sha256=sha(path))


def identity(pid):
  fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
  return dict(pid=int(pid), parent_pid=int(fields[1]), pgid=int(fields[2]), start_ticks=int(fields[19]))


def spawn(command, **kwargs):
  check()
  previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM, signal.SIGTERM}) if os.name == 'posix' else None
  try:
    child = subprocess.Popen(command, start_new_session=os.name == 'posix',
        preexec_fn=(lambda: signal.pthread_sigmask(signal.SIG_SETMASK, previous)) if previous is not None else None,
        **kwargs)
    owned = identity(child.pid) if os.name == 'posix' else dict(pid=child.pid)
    require(os.name != 'posix' or owned['pgid'] == child.pid, 'Owned child did not create its own group')
    OWNED[child.pid] = owned
    return child
  finally:
    if previous is not None:
      signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def group_alive(owned):
  """Fence a still-owned Linux session before any group signal."""
  try:
    os.killpg(owned['pgid'], 0)
  except ProcessLookupError:
    return []
  current = Path(f"/proc/{owned['pid']}/stat")
  if current.exists():
    try:
      actual = identity(owned['pid'])
    except (FileNotFoundError, ProcessLookupError):
      actual = None
    require(actual is None or all(actual[key] == owned[key] for key in ('pid','pgid','start_ticks')),
        'Owned C leader PID was reused or changed group')
  result = []
  for path in Path('/proc').iterdir():
    if not path.name.isdigit():
      continue
    try:
      fields = (path/'stat').read_text().rsplit(') ',1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
      continue
    if int(fields[2]) == owned['pgid'] and fields[0] not in ('Z','X'):
      require(int(fields[3]) == owned['pgid'] and int(fields[19]) >= owned['start_ticks'],
          'C process group lost original session ownership')
      result.append(int(path.name))
  return result


def cleanup(child):
  require(child.pid in OWNED, 'Cannot clean an unregistered child')
  owned = OWNED[child.pid]
  if os.name == 'posix':
    if group_alive(owned):
      os.killpg(owned['pgid'], signal.SIGKILL)
    child.wait(timeout=5)
    until = time.monotonic()+3
    while group_alive(owned) and time.monotonic() < until:
      time.sleep(.01)
    require(not group_alive(owned), 'Owned C process group remains live')
  else:
    if child.poll() is None: child.kill()
    child.wait(timeout=5)
  OWNED.pop(child.pid)


def expire(signum, frame):
  for owned in tuple(OWNED.values()):
    try:
      if group_alive(owned): os.killpg(owned['pgid'], signal.SIGKILL)
    except (OSError, ValueError):
      pass  # Lost ownership never authorizes a signal.
  os._exit(124)


def command_output(command, payload=None):
  check()
  own = 'DL_NM1_PARENT' not in os.environ
  kwargs = dict(cwd=ROOT, stdin=subprocess.PIPE if payload is not None else None,
      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  child = spawn(command, **kwargs) if own else subprocess.Popen(command, **kwargs)
  try:
    out, error = child.communicate(input=payload, timeout=min(30, max(.001, DEADLINE-time.monotonic())))
    require(child.returncode == 0, 'Source command failed: '+error.decode(errors='replace')[:256])
    check(); return out
  finally:
    if own: cleanup(child)
    elif child.poll() is None:
      child.kill(); child.wait(timeout=5)
    for pipe in (child.stdin, child.stdout, child.stderr):
      if pipe is not None: pipe.close()


def source_binding(expected):
  require(type(expected) is str and re.fullmatch('[0-9a-f]{40}', expected), 'Full Git SHA required')
  require(command_output(['git', 'rev-parse', 'HEAD']).decode().strip() == expected and
      not command_output(['git', 'status', '--porcelain', '--untracked-files=all']), 'Clean exact source required')
  names = command_output(['git', 'ls-files', '-z']).decode().split('\0')
  selected = sorted(name for name in names if name.endswith(('.py', '.yaml', '.yml', '.lock')) or name in NORMS | {'AGENTS.md'})
  require(NORMS | {'scripts/dl_nm1_cal.py', 'scripts/dl_nm1_counterfactual.py',
      'scripts/dl_normal_replay.py', 'analysis/dl_nm1_calibration.py', 'analysis/dl_nm1_cal_audit.py'} <= set(selected),
      'Complete tracked NM1 source/norm required')
  packed = command_output(['git', 'cat-file', '--batch'], ''.join(expected+':'+n+'\n' for n in selected).encode())
  result, offset = {}, 0
  for name in selected:
    end = packed.index(b'\n', offset); header = packed[offset:end].split()
    require(len(header) == 3 and header[1] == b'blob', 'Regular Git blob required')
    size = int(header[2]); offset = end+1; body = packed[offset:offset+size]
    require(len(body) == size and packed[offset+size:offset+size+1] == b'\n', 'Truncated Git blob')
    offset += size+1
    path = ROOT/name
    require(not path.is_symlink() and path.read_bytes() == body, 'Actual source/Git blob differs: '+name)
    result[name] = dict(bytes=size, sha256=hashlib.sha256(body).hexdigest())
  require(offset == len(packed), 'Git blob tail differs')
  return result


def input_pins(args):
  values = {name: pin(getattr(args, name)) for name in ('checkpoint', 'config', 'inventory', 'resource_record')}
  require(values['checkpoint']['sha256'] == CHECKPOINT and values['config']['sha256'] == CONFIG and
      values['inventory']['sha256'] == INVENTORY, 'Pinned N1M inputs differ before decoding')
  require(bool(args.stage_record) == bool(args.expected_stage_sha256), 'Stage path/SHA must be supplied together')
  values['stage_record'] = pin(args.stage_record) if args.stage_record else None
  if args.stage_record:
    require(re.fullmatch('[0-9a-f]{64}', args.expected_stage_sha256 or '') and
        values['stage_record']['sha256'] == args.expected_stage_sha256, 'Stage raw bytes differ')
  return values


def validate_budget(budget, expected_git):
  require(budget['version'] == VERSION and budget['schema'] == SCHEMA and budget['seconds'] == SECONDS and
      budget['git_sha'] == expected_git and budget['logical_server'] == 'sv3' and budget['physical_gpu'] == GPU and
      budget['gpu_uuid'] == GPU_UUID and budget['constants'] == CONSTANTS, 'Fixed C budget/identity differs')
  require(type(budget['monotonic_deadline']) is float and math.isfinite(budget['monotonic_deadline']), 'Finite deadline required')


def validate_compute_binding(record):
  require_host()
  require(record['version'] == RESOURCE_VERSION and record['logical_server'] == 'sv3' and
      record['actual_hostname'] == socket.gethostname() and record['physical_gpu'] == GPU and
      record['gpu_uuid'] == record['compute_uuid'] == GPU_UUID and record['egl_uuid'] is None and
      record['graphics_verified'] is False and record['environment'] == COMPUTE_ENV and
      record['source_mapping'] == 'not-applicable-no-environment-or-EGL-initialization',
      'Compute-only resource identity differs; EGL mapping is not applicable')
  require({key:os.environ.get(key) for key in COMPUTE_ENV} == COMPUTE_ENV,
      'Actual compute-only CUDA/disabled-EGL environment differs')


def compute_resource_record():
  """Read idle hardware using stdlib queries; never create a CUDA/EGL context."""
  from scripts.dl_resources import live_state, require_idle
  require_host()
  state = live_state(GPU); require_idle(state, GPU_UUID)
  record = dict(version=RESOURCE_VERSION, logical_server='sv3', actual_hostname=socket.gethostname(),
      physical_gpu=GPU, gpu_uuid=GPU_UUID, compute_uuid=GPU_UUID, egl_uuid=None, graphics_verified=False,
      environment={key:os.environ.get(key) for key in COMPUTE_ENV},
      checked_at_utc=state['checked_at_utc'], occupancy='idle', exclusive=True, live_check=state,
      occupancy_basis='available complete XML process inventory and assigned-device pmon',
      source_mapping='not-applicable-no-environment-or-EGL-initialization')
  validate_compute_binding(record)
  return record


def refresh_compute_resource(record):
  from scripts.dl_resources import live_state, require_idle
  validate_compute_binding(record)
  state = live_state(GPU); require_idle(state, GPU_UUID)
  return dict(record, checked_at_utc=state['checked_at_utc'], live_check=state)


def require_compute_resource(path):
  from scripts.dl_resources import validate_age
  record = read_json(path); validate_age(record)
  return dict(refresh_compute_resource(record), resource_source_path=str(Path(path).resolve()),
      resource_source_sha256=sha(path))


def require_compute_context(state, worker_pid):
  require(state['physical_gpu'] == GPU and state['gpu_uuid'] == GPU_UUID, 'Assigned GPU5 UUID differs')
  require(all(job['pid'] == worker_pid for job in state['processes']+state['pmon_jobs']),
      'Unknown work appeared on assigned GPU5; stop only this owned C worker')
  require(all(job['type'] == 'C' for job in state['processes']+state['pmon_jobs']),
      'C compute-only worker unexpectedly has a graphics context')
  for line in state['raw_pmon'].splitlines():
    fields = line.split()
    if len(fields) >= 3 and fields[1] == str(worker_pid):
      require(fields[0] == str(GPU) and fields[2] == 'C', 'Actual C worker context appears on another GPU or graphics device')


def verify_compute_runtime(record):
  from scripts.dl_resources import live_state, now
  validate_compute_binding(record)
  pid = os.getpid()
  text = command_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader']).decode()
  own = []
  for line in text.splitlines():
    fields = [x.strip() for x in line.split(',')]
    if len(fields) == 2 and fields[0] == str(pid): own.append(fields[1])
  require(own == [GPU_UUID], 'Initialized C CUDA context is not exclusively on the assigned UUID')
  state = live_state(GPU); require_compute_context(state, pid)
  return dict(checked_at_utc=now().isoformat(), pid=pid, compute_uuid=own,
      physical_gpu=GPU, pmon=state['raw_pmon'], graphics_verified=False, live_state=state)


def sample_resource(worker_pid, stream):
  from scripts.dl_resources import live_state
  state = live_state(GPU)
  sample = dict(state, worker_pid=worker_pid, monotonic=time.monotonic(), xml_process_summary=state['processes'])
  stream.write(json.dumps(sample, sort_keys=True, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
  require_compute_context(state, worker_pid)


def parent(args):
  require_host()
  output = Path(args.output).resolve()
  require(not output.exists(), 'Fresh C output required'); output.mkdir(parents=True)
  failure, child, original = None, None, None
  parent_identity = identity(os.getpid())
  try:
    sources, inputs = source_binding(args.expected_git), input_pins(args)
    resource = require_compute_resource(args.resource_record)
    require(resource['physical_gpu'] == GPU and resource['gpu_uuid'] == GPU_UUID, 'C uses only assigned GPU5')
    budget = dict(version=VERSION, schema=SCHEMA, git_sha=args.expected_git, seconds=SECONDS,
        monotonic_deadline=float(DEADLINE), hostname=socket.gethostname(), logical_server='sv3',
        physical_gpu=GPU, gpu_uuid=GPU_UUID, parent_identity=parent_identity, source_files=sources,
        inputs=inputs, stage_expected_sha256=args.expected_stage_sha256, replay_directory=str(Path(args.replay_directory).resolve()),
        constants=CONSTANTS)
    write_json(output/'budget.json', budget)
    command = [sys.executable, str(ROOT/'scripts/dl_nm1_cal.py'), '--worker']
    for name in ('expected_git','checkpoint','config','inventory','replay_directory','resource_record','output',
        'stage_record','expected_stage_sha256'):
      if getattr(args,name): command.extend(('--'+name.replace('_','-'),str(getattr(args,name))))
    env = dict(os.environ, DL_NM1_PARENT=str(os.getpid()), DL_NM1_BUDGET_SHA256=sha(output/'budget.json'))
    with (output/'worker.log').open('xb') as log:
      child = spawn(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
      original = identity(child.pid)
      write_json(output/'process.json', dict(version=VERSION, schema=SCHEMA, parent_identity=parent_identity,
          worker_identity=original, command=command, started_monotonic=time.monotonic()))
      try:
        with (output/'resources.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
          while child.poll() is None:
            check(); sample_resource(child.pid, stream)
            try: child.wait(timeout=min(15, max(.001, DEADLINE-time.monotonic())))
            except subprocess.TimeoutExpired: pass
      finally:
        cleanup(child)
      require(child.returncode == 0 and (output/'generation_complete.json').is_file(), 'C worker failed/incomplete')
  except Exception as error:
    failure = repr(error)
    if child is not None and child.pid in OWNED:
      try: cleanup(child)
      except Exception as error: failure += '; cleanup: '+repr(error)
  cleaned = child is not None and child.poll() is not None and child.pid not in OWNED
  if cleaned and os.name == 'posix':
    try: cleaned = not group_alive(original)
    except (OSError, ValueError): cleaned = False
  artifacts = {name: pin(output/name, True) if (output/name).is_file() else None for name in
      ('budget.json', 'process.json', 'worker.log', 'resources.jsonl', 'generation_complete.json')}
  terminal = dict(version=VERSION, schema=SCHEMA, git_sha=args.expected_git, parent_identity=parent_identity,
      worker_identity=original, worker_returncode=child.returncode if child is not None else None,
      failure=failure, worker_terminal=child is not None and child.poll() is not None,
      owned_group_cleaned=cleaned, elapsed_seconds=time.monotonic()-ENTRY_STARTED, artifacts=artifacts)
  write_json(output/'supervisor_terminal.json', terminal)
  print(json.dumps(dict(event='terminal', **terminal), sort_keys=True), flush=True)
  require(failure is None, 'C incomplete; preserve failure: '+str(failure))


def array_spec(value, allow_diagnostic=False):
  import numpy as np
  value = np.asarray(value)
  require(value.dtype.kind in 'buif' or str(value.dtype) == 'bfloat16', 'Numeric exact dtype required')
  require(allow_diagnostic or np.isfinite(value).all(), 'Nonfinite tensor')
  digest = hashlib.sha256()
  digest.update(json.dumps([value.dtype.str, str(value.dtype)]).encode())
  digest.update(json.dumps(value.shape).encode()); digest.update(value.tobytes(order='C'))
  return dict(logical_dtype=str(value.dtype), dtype_str=value.dtype.str, shape=list(value.shape),
      typed_sha256=digest.hexdigest(), bytes=value.nbytes)


def flatten(tree, prefix=''):
  result = {}
  if isinstance(tree, dict):
    for key, value in tree.items(): result.update(flatten(value, prefix+str(key)+'/'))
  elif isinstance(tree, (tuple, list)):
    for key, value in enumerate(tree): result.update(flatten(value, prefix+str(key)+'/'))
  else:
    result[prefix[:-1]] = tree
  return result


def write_arrays(path, values):
  import numpy as np
  arrays, schema = {}, {}
  for i, (name, value) in enumerate(sorted(values.items())):
    value = np.asarray(value); spec = array_spec(value); storage = f'a{i:05d}'
    arrays[storage] = value.view(np.uint16) if str(value.dtype) == 'bfloat16' else value
    schema[name] = dict(spec, storage_key=storage)
  check()
  with Path(path).open('xb') as stream:
    np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
  check(); return schema


def metric_specs(metrics):
  import numpy as np
  result = {}
  for name, value in metrics.items():
    valid = bool(np.isfinite(value).all())
    require(valid or (name.startswith('dl/') and name.endswith('_valid_mean')), 'Unexpected nonfinite metric: '+name)
    result[name] = array_spec(value, not valid)
  return result


def production_receipt(value, counters):
  from scripts.dl_reset_learning import tree_digest
  return dict(state_sha256=tree_digest(dict(params=value['params'], counters=counters)),
      params={k: array_spec(v) for k, v in sorted(value['params'].items())},
      carry_sha256=tree_digest(value['carry']), replay_sha256=tree_digest(value['outs']['replay']),
      metrics_sha256=hashlib.sha256(canonical(metric_specs(value['metrics']))).hexdigest())


def exact_production(a, b):
  from scripts.dl_reset_learning import tree_digest
  return tree_digest((a['params'], a['carry'], a['outs'])) == tree_digest((b['params'], b['carry'], b['outs'])) and \
      metric_specs(a['metrics']) == metric_specs(b['metrics'])


def selected(tree):
  return {k: v for k, v in tree.items() if k.startswith(('enc/', 'dyn/obs'))}


def cache_seal(cache):
  from scripts.dl_reset_learning import tree_digest
  digest = hashlib.sha256()
  for uuid, chunk in sorted(cache._replay.chunks.items(), key=lambda item: str(item[0])):
    digest.update(str(uuid).encode())
    digest.update(tree_digest(chunk.data).encode())
  return dict(sha256=digest.hexdigest(), receipt=cache.receipt(),
      metrics_sha256=tree_digest(cache._replay.metrics), pending=cache._pending)


def verified_state(agent):
  """Keep official save's NumPy parameters and Python integer counters intact."""
  from embodied.run.protocol_v1 import parameter_digest
  from scripts.dl_reset_learning import tree_digest
  saved = agent.save()
  require(tree_digest(saved) == FULL_STATE and saved['counters'] == COUNTERS and parameter_digest(agent) == PARAMS and
      int(saved['params']['opt/step/value']) == STEP, 'Loaded full state/params/counters differ')
  return saved


def load_variants(args):
  import importlib.metadata
  import pickle
  import elements
  import jax
  from scripts.dl_diagnostic import load_config
  from scripts.dl_predictive_mc import agent_from_spaces
  from embodied.run.protocol_v1 import parameter_digest
  from scripts.dl_nm1_counterfactual import Counterfactual, host
  from scripts.dl_reset_learning import tree_digest
  require(importlib.metadata.version('jax') == '0.4.33' and jax.config.jax_default_prng_impl == 'threefry2x32' and
      jax.config.jax_threefry_partitionable is False and jax.config.jax_enable_x64 is False, 'Frozen JAX PRNG implementation required')
  checkpoint_raw = Path(args.checkpoint).read_bytes()
  require(hashlib.sha256(checkpoint_raw).hexdigest() == CHECKPOINT and sha(args.config) == CONFIG,
      'Original model inputs changed before decoding')
  payload = pickle.loads(checkpoint_raw)
  del checkpoint_raw
  require(set(payload) == {'params', 'counters'} and len(payload['params']) == 294 and
      tree_digest(payload) == FULL_STATE, 'Original complete N1M state differs')
  config = load_config(args.config, 'dmc_quadruped_walk', args.output, 'cuda')
  variants, configs, seals, carry = {}, {}, {}, None
  for name, mode, rho in (('B', 'off', .5), ('rho0', 'dtlatch', 0.), ('N', 'dtlatch', .5)):
    actual = elements.Config({**config.flat, **{'jax.enable_policy':False,
        'agent.dt_latch.mode':mode, 'agent.dt_latch.rho':rho, 'agent.dt_latch.normal_only':True}})
    require(actual.seed == 0 and actual.jax.compute_dtype == 'bfloat16' and actual.consec_train == 1 and
        actual.batch_size == BATCH and actual.batch_length == LENGTH and actual.replay_context == CONTEXT,
        'Actual normal training config differs')
    agent = agent_from_spaces(actual); agent.load(payload)
    saved = verified_state(agent)
    ca = agent.init_train(BATCH)
    if carry is None: carry = ca
    else: require(tree_digest(host(ca)) == tree_digest(host(carry)), 'Configured initial carries differ')
    variants[name] = Counterfactual(agent); configs[name] = dict(actual.flat)
    seals[name] = dict(full_state_sha256=tree_digest(saved), params_sha256=parameter_digest(agent),
        counters=saved['counters'], opt_step=STEP, carry_sha256=tree_digest(host(ca)))
  require(all(device.platform == 'gpu' for device in jax.devices()) and len(jax.devices()) == 1, 'Single CUDA device required')
  return variants, carry, payload, configs, seals


def chain_map(report):
  from scripts.dl_replay_inventory import filename, uuid_bytes
  return {uuid_bytes(filename(path)['uuid']): i for i, chain in enumerate(report['chains']) for path in chain['paths']}


def collect_batch(raw, variants, carry, batch, chains, directory):
  import numpy as np
  from scripts.dl_nm1_counterfactual import prepare_readonly, host, neural_difference, neural_displacements
  from scripts.dl_reset_learning import tree_digest
  directory = Path(directory); directory.mkdir()
  values = {'raw/'+k: v for k, v in raw.items()}
  values.update(flatten(host(carry), 'initial_carry/'))
  counters = variants['B'].agent.save()['counters']
  prepared = {name: prepare_readonly(probe.agent, raw, STEP+batch) for name, probe in variants.items()}
  digest = tree_digest(host(prepared['B']))
  require(all(tree_digest(host(value)) == digest for value in prepared.values()), 'Prepared source/key differs across modes')
  contexts = {name: tree_digest(host(probe._context(carry,
      {key:value for key,value in prepared[name].items() if key != 'seed'})))
      for name,probe in variants.items()}
  require(len(set(contexts.values())) == 1, 'Actual replay context differs across modes')
  values['prepared_key'] = host(prepared['B']['seed'])
  b = variants['B'].production(carry, prepared['B'])
  repeated = variants['B'].production(carry, prepared['B'])
  repeat_receipt = production_receipt(repeated, counters)
  require(exact_production(b, repeated), 'B/B production repeat not byte-identical')
  del repeated
  zero = variants['rho0'].production(carry, prepared['rho0'])
  zero_receipt = production_receipt(zero, counters)
  require(exact_production(b, zero), 'rho0/B production not byte-identical')
  del zero
  n = variants['N'].production(carry, prepared['N'])
  refs = {name: variants[name].reference(carry, prepared[name]) for name in ('B', 'N')}
  forward = {name: host(refs[name]['aux'][1]) for name in refs}
  diagnostics = {name: host(refs[name]['aux'][2]) for name in refs}
  nr, dl = diagnostics['N']['nm1_reference'], diagnostics['N']['dl']
  ids = raw['stepid'][:, 1:].copy()
  signal = dict(K=nr['K'], E=nr['E'], D=dl['D'], e=dl['e'], f=dl['f_D'], v_N=nr['weight'],
      m=dl['m'], A=dl['m_intervention'], stepid=ids,
      chain=np.asarray([[chains[x[:16].tobytes()] for x in row] for row in ids], np.int32))
  expected_m = np.concatenate([(~raw['is_last'][:, 1:-1]) & (~raw['is_first'][:, 2:]),
      np.zeros((BATCH, 1), bool)], 1)
  require(np.array_equal(signal['m'], expected_m) and np.array_equal(signal['A'], expected_m & ~raw['is_first'][:, 1:]),
      'Actual reference H1/intervention masks differ from full raw input')
  for key, value in signal.items(): values['signal/'+key] = value
  for name in ('B', 'N'):
    for key, value in selected(refs[name]['grads']).items(): values['grad_'+name+'/'+key] = value
    for key, value in selected((b if name == 'B' else n)['params']).items(): values['step_'+name+'/'+key] = value
    for key, value in (b if name == 'B' else n)['outs']['replay'].items(): values['replay_'+name+'/'+key] = value
    require(not forward[name][0] and not forward[name][2], 'Original encoder/decoder cache schema differs')
    for key, value in forward[name][1].items(): values['reference_replay_'+name+'/dyn/'+key] = value
  for key in ('K', 'E', 'rep_before', 'weight'): values['reference_B/'+key] = diagnostics['B']['nm1_reference'][key]
  values['reference_N/rep_before'] = nr['rep_before']
  gradients = neural_difference(refs['B']['grads'], refs['N']['grads'])
  steps = neural_difference(b['params'], n['params'])
  effects = {group+'_'+kind+'_changed_elements': int(report[group]['changed_elements'])
      for kind, report in (('gradient', gradients), ('step', steps)) for group in ('encoder', 'posterior')}
  core_replay = lambda item: {k: v for k, v in item['outs']['replay'].items() if k.startswith('dyn/')}
  ref_replay = lambda name: {'dyn/'+k: v for k, v in forward[name][1].items()}
  record = dict(version=VERSION, schema=SCHEMA, batch=batch, counter=STEP+batch, opt_step_before=STEP,
      arrays=write_arrays(directory/'evidence.npz', values), raw_sha256=tree_digest(raw), prepared_sha256=digest,
      carry_sha256=tree_digest(host(carry)), context_sha256=contexts['B'], production=dict(B=production_receipt(b,counters),
          B_repeat=repeat_receipt, rho0=zero_receipt, N=production_receipt(n,counters)),
      identity_checks=dict(B_repeat_equal=True, rho0_equal=True),
      reference={name:dict(state_scope=refs[name]['gradient_scope'], accessed=list(variants[name].reference_accessed),
          modified=list(variants[name].reference_modified), gradient_sha256=tree_digest(refs[name]['grads']),
          forward_sha256=tree_digest(forward[name])) for name in ('B','N')}, effects=effects,
      gradient_differences=gradients, parameter_differences=steps,
      actual_displacements=neural_displacements(variants['B'].agent.save()['params'],b['params'],n['params']),
      forward=dict(production_replay_equal=tree_digest(core_replay(b)) == tree_digest(core_replay(n)),
          reference_replay_equal=tree_digest(ref_replay('B')) == tree_digest(ref_replay('N')),
          production_vs_reference_B_equal=tree_digest(core_replay(b)) == tree_digest(ref_replay('B')),
          production_vs_reference_N_equal=tree_digest(core_replay(n)) == tree_digest(ref_replay('N'))))
  write_json(directory/'record.json', record)
  return signal, effects, record['forward']


def control_arrays(signal, frozen):
  import jax
  import jax.numpy as jnp
  from dreamerv3.dt_latch import conditional_shuffle
  from scripts.dl_nm1_counterfactual import host
  # Diagnostic inputs and compiler-captured device constants need local transfers.
  with jax.transfer_guard('allow'):
    a, e, f = (jnp.asarray(signal[k]) for k in ('A','e','f'))
    key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(0),0x444C),jnp.uint32(STEP))
    arrays = dict(P_key=host(key), P_priorities=host(jax.random.uniform(key,(BATCH*LENGTH,))))
    controls, availability = {}, {}
    for name, factor in (('S',f), ('W',e)):
      entry = frozen[name]; value = None
      if entry['calibrated']:
        coefficient = jnp.float32(entry['coefficient_f32'])
        value = host(jax.jit(lambda mask,x: jnp.float32(1)-mask*coefficient*x)(a,factor))
        arrays['v_'+name] = value
      controls['v_'+name] = value
      availability['v_'+name] = dict(available=entry['calibrated'], reason=entry['reason'])
    entry = frozen['bins']; value = None
    if entry['calibrated']:
      d_bins, k_bins = tuple(entry['D_f32']), tuple(entry['K_f32'])
      def permute(w, mask, k, d):
        v, _, perm = conditional_shuffle(w,mask,k,d,1.,d_bins,k_bins,0,STEP)
        db=jnp.searchsorted(jnp.asarray(d_bins,jnp.float32),d,side='right')
        kb=jnp.searchsorted(jnp.asarray(k_bins,jnp.float32),k,side='right')
        group=(((mask.astype(jnp.int32)*2+(k>1))*(len(d_bins)+1)+db)*(len(k_bins)+1)+kb)
        counts=jnp.bincount(group.reshape(-1),length=4*(len(d_bins)+1)*(len(k_bins)+1))
        return v,perm,group,counts
      value, perm, group, counts = host(jax.jit(permute)(jnp.asarray(signal['v_N']),a,
          jnp.asarray(signal['K']),jnp.asarray(signal['D'])))
      arrays.update(v_P=value,P_perm=perm,P_group=group,P_counts=counts)
    controls['v_P'] = value
    availability['v_P'] = dict(available=entry['calibrated'], reason=entry['reason'])
    return arrays, controls, availability


def verify_worker(args):
  global DEADLINE
  require_host()
  root = Path(args.output).resolve(strict=True); path = root/'budget.json'; budget = read_json(path)
  validate_budget(budget,args.expected_git); DEADLINE=budget['monotonic_deadline']; check()
  parent_pid=int(os.environ.get('DL_NM1_PARENT','-1'))
  require(parent_pid == os.getppid() == budget['parent_identity']['pid'] and
      identity(parent_pid) == budget['parent_identity'] and os.getpgid(0) == os.getpid(), 'Original direct parent/owned worker required')
  require(os.environ.get('DL_NM1_BUDGET_SHA256') == sha(path) and
      budget['hostname'] == socket.gethostname() and source_binding(args.expected_git) == budget['source_files'] and
      input_pins(args) == budget['inputs'] and str(Path(args.replay_directory).resolve()) == budget['replay_directory'],
      'Parent/worker source/input binding differs')
  def checked():
    check(); require(os.getppid() == parent_pid and identity(parent_pid) == budget['parent_identity'],
        'Original live direct parent identity changed')
  return root,budget,checked


def worker(args):
  root,budget,checked = verify_worker(args)
  signal.setitimer(signal.ITIMER_REAL,max(.001,DEADLINE-time.monotonic()))
  resource = require_compute_resource(args.resource_record)
  require(resource['physical_gpu'] == GPU and resource['gpu_uuid'] == GPU_UUID, 'Actual GPU5 binding differs')
  import importlib.metadata
  import numpy as np
  require(importlib.metadata.version('numpy') == '1.26.4', 'Frozen NumPy1.26.4 required before replay/plan')
  from scripts import dl_normal_replay as replay
  from scripts.dl_reset_learning import tree_digest
  from scripts.dl_nm1_counterfactual import host
  from analysis import dl_nm1_calibration as arithmetic
  cache = replay.load_private_cache(args.replay_directory,args.inventory,
      stage_record_path=args.stage_record,expected_stage_record_sha256=args.expected_stage_sha256)
  checked(); before=cache_seal(cache)
  seed=int(np.random.SeedSequence([20261101,0x4E4D4341]).generate_state(1,dtype=np.uint64)[0])
  starts=cache.make_plan(seed,BATCHES)
  write_json(root/'plan.json',dict(version=VERSION,schema=SCHEMA,batches=BATCHES,seed=seed,starts=starts))
  resource=refresh_compute_resource(resource); write_json(root/'resource_refreshed.json',resource)
  variants,carry,payload,configs,seals=load_variants(args)
  write_json(root/'runtime_before.json',verify_compute_runtime(resource))
  import jax
  initial=dict(version=VERSION,schema=SCHEMA,checkpoint_counters=payload['counters'],postload_counters=COUNTERS,
      full_state_sha256=FULL_STATE,params_sha256=PARAMS,
      state_leaves={k:array_spec(v) for k,v in sorted(payload['params'].items())},
      arrays=write_arrays(root/'initial.npz',{'initial/'+k:v for k,v in selected(payload['params']).items()}),
      configs=configs,variant_seals=seals,cache_before=before,worker_identity=identity(os.getpid()),
      packages={name:importlib.metadata.version(name) for name in ('jax','jaxlib','numpy','ninjax','elements','ml-dtypes')},
      jax_prng=dict(default_prng_impl=jax.config.jax_default_prng_impl,
          threefry_partitionable=jax.config.jax_threefry_partitionable,enable_x64=jax.config.jax_enable_x64))
  write_json(root/'initial.json',initial); del payload
  chains=chain_map(cache._report); signals,effects,forwards=[],[],[]
  for batch,plan in enumerate(starts):
    checked(); raw=cache.read_batch(plan)
    sig,eff,fwd=collect_batch(raw,variants,carry,batch,chains,root/f'batch_{batch:03d}')
    signals.append(sig);effects.append(eff);forwards.append(fwd)
    print(json.dumps(dict(event='batch_collected',batch=batch)),flush=True)
  frozen=arithmetic.freeze(signals);write_json(root/'frozen.json',frozen)
  controls,receipts=[],[]
  for batch,sig in enumerate(signals):
    checked(); directory=root/f'batch_{batch:03d}'
    arrays,control,availability=control_arrays(sig,frozen);controls.append(control)
    record=dict(version=VERSION,schema=SCHEMA,batch=batch,arrays=write_arrays(directory/'controls.npz',arrays),
        availability=availability,frozen_sha256=sha(root/'frozen.json'),p_step=STEP,p_seed=0)
    write_json(directory/'controls.json',record)
    complete=dict(version=VERSION,schema=SCHEMA,batch=batch,
        artifacts={name:pin(directory/name,True) for name in sorted(BATCH_FILES)},
        record_sha256=sha(directory/'record.json'),controls_sha256=sha(directory/'controls.json'))
    write_json(directory/'complete.json',complete)
    receipts.append(dict(pin(directory/'complete.json',True),path=f'{directory.name}/complete.json'))
  summary=arithmetic.summarize(signals,effects,controls,frozen)
  forward_summary=dict(production_replay_equal_batches=sum(x['production_replay_equal'] for x in forwards),
      reference_replay_equal_batches=sum(x['reference_replay_equal'] for x in forwards),
      interpretation='Operational total update effect only; replay equality does not isolate every forward path',
      only_rep_pressure_isolated=False)
  write_json(root/'summary.json',summary)
  after=cache_seal(cache);require(after == before and cache._pending is None,'Read-only private cache changed')
  final_seals={}
  for name,probe in variants.items():
    probe.fence(full=True);verified_state(probe.agent)
    final_seals[name]=dict(seals[name],pending_clean=True,source_buffers_unchanged=True)
  source_after=replay.inventory.snapshot(Path(args.replay_directory))
  require(source_after == cache._report['before_inventory'],'Original full source files changed')
  inputs_after=input_pins(args);sources_after=source_binding(args.expected_git)
  require(inputs_after == budget['inputs'] and sources_after == budget['source_files'],'Inputs/source changed')
  write_json(root/'runtime_after.json',verify_compute_runtime(resource))
  write_json(root/'final.json',dict(version=VERSION,schema=SCHEMA,agent_seals=final_seals,cache_after=after,
      original_source_after=source_after,inputs_after=inputs_after,source_files_after=sources_after,
      complete_batches=BATCHES,optimizer_updates_committed=0,replay_updates_committed=0,
      forward_summary=forward_summary))
  checked()
  write_json(root/'generation_complete.json',dict(version=VERSION,schema=SCHEMA,git_sha=args.expected_git,
      complete_batches=BATCHES,top_artifacts={name:pin(root/name,True) for name in sorted(TOP_FILES)},
      batches=receipts,optimizer_updates_committed=0,replay_updates_committed=0,
      independent_audit_complete=False,subsequent_learning_authorized=False))


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  for name in ('expected-git','checkpoint','config','inventory','replay-directory','resource-record','output'):
    parser.add_argument('--'+name,required=True)
  parser.add_argument('--stage-record');parser.add_argument('--expected-stage-sha256')
  parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
  args=parser.parse_args()
  if os.name == 'posix':
    signal.signal(signal.SIGALRM,expire);signal.signal(signal.SIGTERM,expire)
    signal.setitimer(signal.ITIMER_REAL,max(.001,DEADLINE-time.monotonic()))
  (worker if args.worker else parent)(args)


if __name__ == '__main__': main()
