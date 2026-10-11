"""Owned endpoint-only GPU observation guard; does not extend native r1.

The new two-path diagnostic is usable only after its separate specification
is adopted, the native assay and its independent audit have fully terminated,
and the exact implementation passes clean-checkout engineering validation.
No new environments, renderings, evaluations, fits or selection gates occur.
"""
import time
ENTRY_STARTED = time.monotonic()
import os
import signal
import sys
OWNED = set()
SECONDS = 1800


def expire(signum, frame):
  for pid in tuple(OWNED):
    try: os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError: pass
  os._exit(124)


if __name__ == '__main__' and os.name == 'posix':
  signal.signal(signal.SIGALRM, expire)
  signal.signal(signal.SIGTERM, expire)
  signal.setitimer(signal.ITIMER_REAL, max(.001, SECONDS-(time.monotonic()-ENTRY_STARTED)))

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'DL-rep-observation-guard-r1'
PHASES = ('authenticate', 'Bwhole', 'Amin', 'seal')
KEYS = ('dataset', 'artifact_audit', 'physics_audit', 'checkpoint', 'config',
    'old_primary', 'old_primary_cli_terminal', 'old_audit', 'old_audit_cli_terminal',
    'native_primary', 'native_cli_terminal', 'native_audit', 'native_audit_cli_terminal',
    'output', 'expected_git', 'task')


def require(value, message):
  if not value: raise ValueError(message)


def sha(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
  return digest.hexdigest()


def read_json(path):
  def pairs(items):
    value = {}
    for key, item in items:
      require(key not in value, 'Duplicate JSON key: '+key); value[key] = item
    return value
  return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs,
      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def jsonify(value):
  if isinstance(value, dict): return {str(k): jsonify(v) for k, v in value.items()}
  if isinstance(value, (tuple, list)): return [jsonify(v) for v in value]
  if hasattr(value, 'tolist'): return jsonify(value.tolist())
  if isinstance(value, Path): return str(value)
  return value


def write(path, value):
  with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(jsonify(value), stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write('\n')


def spawn_owned(command, deadline, **kwargs):
  require(time.monotonic() < deadline, 'Late before owned process creation')
  previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM, signal.SIGTERM})
  try:
    child = subprocess.Popen(command, start_new_session=True,
        preexec_fn=lambda:signal.pthread_sigmask(signal.SIG_SETMASK, previous), **kwargs)
    OWNED.add(child.pid)
    return child
  finally:
    signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def cleanup_owned(child):
  try: os.killpg(child.pid, signal.SIGKILL)
  except ProcessLookupError: pass
  child.wait(timeout=5); OWNED.discard(child.pid)


def owned_metadata(command):
  deadline = ENTRY_STARTED+SECONDS
  child = spawn_owned(command, deadline, cwd=ROOT,
      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  try:
    out,err = child.communicate(timeout=min(30,max(.001,deadline-time.monotonic())))
    require(child.returncode == 0, 'Owned metadata failed: '+err.decode(errors='replace')[:256])
    return out
  finally: cleanup_owned(child)


def clean_binding(expected):
  from scripts import dl_rep_native as native
  pins = native.git_binding(expected)
  for name in ('scripts/dl_rep_observation_guard.py',
      'scripts/dl_rep_observation_runner.py', 'docs/DL_REP_OBSERVATION_GUARD.md'):
    blob = native.owned_output(['git', 'show', expected+':'+name])
    require((ROOT/name).read_bytes() == blob, 'Guard Git/file bytes differ: '+name)
    pins[name] = hashlib.sha256(blob).hexdigest()
  return pins


def input_authentication(args, output, check):
  from scripts import dl_rep_native as native
  from analysis import dl_rep_native_audit as audit
  native.authenticate_old(args, output, check)
  verified = audit.verify_completion(args.native_audit, args.native_audit_cli_terminal)
  authority = audit.authenticate_primary(args.native_primary,
      args.native_cli_terminal, args.expected_git, check)
  independent = read_json(Path(args.native_audit)/'audit.json')
  require(independent['primary'] == str(authority['root']) and
      independent['primary_git_sha'] == authority['primary_git'] and
      independent['primary_external_cli_terminal_sha256'] ==
          authority['external_cli_terminal_sha256'] and
      independent['primary_endpoint_sha256'] == authority['endpoint_sha256'] and
      independent['status'] == 'passed', 'Native audit does not bind this primary')
  legacy = read_json(output/'old-authenticated.json')
  current_a = read_json(authority['root']/'A-complete.json')
  old_b_receipt = read_json(Path(args.old_primary)/'B-complete.json')
  require(current_a['learning']['source_checkpoint_digest'] ==
      legacy['original_B_learning']['source_checkpoint_digest'] and
      current_a['learning']['binding'] == legacy['original_B_learning']['binding'] and
      current_a['learning']['initial_state_digest'] ==
          legacy['original_B_learning']['initial_state_digest'],
      'A/B original full initialization or common batch differs')
  immutable = dict(legacy['immutable_inputs_sha256'])
  immutable.update({str(authority['root']/name): pin
      for name, pin in authority['endpoint_sha256'].items()})
  immutable[str(Path(args.native_cli_terminal).resolve())] = authority['external_cli_terminal_sha256']
  for name in ('audit.json', 'budget.json', 'supervisor_complete.json', 'worker_exit_code.txt'):
    path = (Path(args.native_audit)/name).resolve(); immutable[str(path)] = sha(path)
  path = Path(args.native_audit_cli_terminal).resolve(); immutable[str(path)] = sha(path)
  current_checkpoint = authority['root']/'A-checkpoint.pkl'
  immutable[str(current_checkpoint)] = authority['freeze']['complete_checkpoint_sha256']['A']
  for path, pin in immutable.items(): check(); require(sha(path) == pin, 'Scientific input bytes differ')
  write(output/'guard-authority.json', dict(version=VERSION,
      actual_git_sha=args.expected_git, old_authentication_sha256=sha(output/'old-authenticated.json'),
      immutable_inputs_sha256=immutable, old_B_learning=legacy['original_B_learning'],
      current_A_learning=current_a['learning'],
      historical_configs={'B':old_b_receipt['model_metadata']['config'], 'A':current_a['config']},
      historical_A_package_versions=authority['result']['environment'],
      historical_B_package_versions=None,
      historical_unrecorded_ambient_compiler_flags_verified=False,
      native_primary_git_sha=authority['primary_git'],
      native_independent_verification=verified,
      native_A_checkpoint_path=str(current_checkpoint),
      native_A_checkpoint_sha256=immutable[str(current_checkpoint)],
      old_B_checkpoint_path=str(Path(args.old_primary).resolve()/'B-checkpoint.pkl'),
      old_B_checkpoint_sha256=legacy['old_files_sha256']['B-checkpoint.pkl'],
      legacy_per_update_full_state_available=False,
      new_native_evaluation_performed=False, endpoint_only=True))


def learn(args, output, check):
  from scripts import dl_rep_native as native
  from scripts import dl_reset_joint as old
  from scripts import dl_resources as resource
  from scripts import dl_rep_observation_guard as guard
  from scripts import dl_predictive_mc as original
  auth = read_json(output/'guard-authority.json')
  legacy = read_json(output/'old-authenticated.json')
  arrays, binding = old.load_cal(args, check)
  require(binding['all_32_artifact_hashes'] == legacy['source_binding']['all_32_artifact_hashes'],
      'Authenticated calibration identity differs')
  contexts = native.load_payload(Path(args.old_primary)/'contexts.pkl',
      legacy['old_files_sha256']['contexts.pkl'])
  checkpoint = native.load_payload(args.checkpoint, original.CHECKPOINT_SHA)
  device = resource.require_resource(args.resource_record)
  require(device['physical_gpu'] == 4, 'Only physical4 for this guard')
  with guard.observation_branch(args.worker_phase):
    agent, config = native.load_agent(args, checkpoint, policy=False)
    actual_versions = {name:importlib.metadata.version(name) for name in
        ('numpy','jax','jaxlib','ninjax','mujoco','dm-control','ml-dtypes')}
    for name,version in auth['historical_A_package_versions'].items():
      require(actual_versions[name] == version, 'Recorded native package version changed: '+name)
    for group,historical in auth['historical_configs'].items():
      require({k:v for k,v in jsonify(config.flat).items() if k != 'logdir'} ==
          {k:v for k,v in historical.items() if k != 'logdir'},
          'Recorded '+group+' full model/JAX/config fields changed')
    import jax
    runtime = dict(python=sys.version, package_versions=actual_versions,
        jax_enable_x64=bool(jax.config.jax_enable_x64),
        jax_disable_jit=bool(jax.config.jax_disable_jit),
        devices=[dict(platform=d.platform, kind=d.device_kind) for d in jax.devices()],
        ambient_flags={name:os.environ.get(name) for name in
            ('XLA_FLAGS','JAX_DISABLE_JIT','JAX_ENABLE_X64','NVIDIA_TF32_OVERRIDE',
             'JAX_DEFAULT_MATMUL_PRECISION')},
        historical_B_package_versions_unavailable=True,
        historical_unrecorded_ambient_flags_unavailable=True,
        known_configs_and_A_versions_verified=True)
    device['runtime_compute_start'] = resource.verify_runtime(device, compute=True)
    outcome = guard.run_guard(agent, checkpoint, arrays,
        {k: binding['all_32_artifact_hashes'][k]['sha256'] for k in old.CAL},
        contexts, args.worker_phase, reference_b=auth['old_B_learning'],
        reference_a=auth['current_A_learning'], check=check)
    from scripts import dl_reset_learning as learning
    final = outcome.pop('final_checkpoint')
    path = output/(args.worker_phase+'-checkpoint.pkl')
    pin = native.save_pickle(path, final)
    anchor = 'old_B' if args.worker_phase == 'Bwhole' else 'native_A'
    files = ((path,pin,'new'), (Path(auth[anchor+'_checkpoint_path']),
        auth[anchor+'_checkpoint_sha256'],'historical_anchor'))
    reload_receipts = {}
    for filename, filepin, label in files:
      payload = native.load_payload(filename,filepin)
      before_reload = learning.tree_digest(payload)
      agent.load(payload)
      reloaded = agent.save()
      difference = typed_differences(payload,reloaded)
      require(learning.tree_digest(reloaded) == before_reload and difference['equal'],
          'Written '+label+' complete endpoint fails standard Agent.load/save roundtrip')
      reload_receipts[label] = dict(file_sha256=filepin,
          full_state_sha256=before_reload,exact_typed_comparison=difference)
    agent.load(final)
    restored = agent.save()
    require(learning.tree_digest(restored) == learning.tree_digest(final) and
        typed_differences(final,restored)['equal'], 'New endpoint restore differs')
    outcome['standard_written_reload_receipts'] = reload_receipts
    outcome['standard_final_checkpoint_reload_byte_exact'] = True
    outcome['historical_anchor_standard_reload_byte_exact'] = True
    device['runtime_compute_end'] = resource.verify_runtime(device, compute=True)
  check()
  write(output/(args.worker_phase+'-complete.json'), dict(version=VERSION,
      actual_git_sha=args.expected_git, phase=args.worker_phase, learning=outcome,
      checkpoint_sha256=pin, device=device, config=config.flat, source_binding=binding,
      runtime_environment=runtime,
      guard_authority_sha256=sha(output/'guard-authority.json'),
      new_native_evaluation_performed=False))


def typed_differences(left, right, limit=16):
  """Locate exact container/shape/dtype/byte differences, with no tolerance."""
  import numpy as np
  count, first = 0, []
  def schema(value):
    return dict(dtype=str(value.dtype), dtype_str=value.dtype.str,
        shape=list(value.shape), sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
  def finite_number(value):
    value = float(value)
    return value if np.isfinite(value) else None
  def record(path, kind, **extra):
    nonlocal count
    count += 1
    if len(first) < limit: first.append(dict(path=path, kind=kind, **extra))
  def visit(a, b, path):
    if isinstance(a, dict) or isinstance(b, dict):
      if type(a) is not dict or type(b) is not dict:
        record(path, 'container_type'); return
      if set(a) != set(b): record(path, 'dict_keys', left=sorted(a), right=sorted(b))
      for key in sorted(set(a)&set(b)): visit(a[key], b[key], path+'/'+key)
    elif isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
      if type(a) is not type(b): record(path, 'container_type'); return
      if len(a) != len(b): record(path, 'container_length'); return
      for index,(x,y) in enumerate(zip(a,b)): visit(x,y,path+'/'+str(index))
    elif a is None or b is None or type(a) in (bool,int,float,str) or type(b) in (bool,int,float,str):
      if type(a) is not type(b) or json.dumps(a) != json.dumps(b):
        def scalar_schema(value):
          if value is None or type(value) in (bool,int,float,str):
            payload=json.dumps(value,allow_nan=False).encode()
            return dict(type=type(value).__name__,sha256=hashlib.sha256(payload).hexdigest())
          return dict(type=type(value).__name__,**schema(np.asarray(value)))
        detail=dict(left=scalar_schema(a),right=scalar_schema(b),
            maximum_absolute_difference=None,maximum_relative_difference=None,
            signed_zero_differences=None,floating_diagnostic_available=False)
        if type(a) is float and type(b) is float:
          with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            delta=np.abs(np.float64(a)-np.float64(b)); scale=max(abs(a),abs(b))
            relative=delta/scale if scale else np.float64(0.)
          detail.update(floating_diagnostic_available=True,
              maximum_absolute_difference=finite_number(delta),
              maximum_relative_difference=finite_number(relative),
              signed_zero_differences=int(a==0 and b==0 and np.signbit(a)!=np.signbit(b)),
              derived_diagnostic_nonfinite=not (np.isfinite(delta) and np.isfinite(relative)))
        record(path, 'scalar_type_or_value',**detail)
    else:
      a,b = np.asarray(a),np.asarray(b)
      if a.dtype != b.dtype or a.dtype.str != b.dtype.str:
        record(path, 'dtype', left=schema(a), right=schema(b)); return
      if a.shape != b.shape:
        record(path, 'shape', left=schema(a), right=schema(b)); return
      aa,bb = a.tobytes(order='C'),b.tobytes(order='C')
      if aa != bb:
        changed = np.frombuffer(aa,np.uint8) != np.frombuffer(bb,np.uint8)
        byte_index = int(changed.argmax())
        elements = changed.reshape(-1,a.dtype.itemsize).any(axis=1)
        element_index = int(elements.argmax())
        detail = dict(left=schema(a), right=schema(b),
            differing_bytes=int(np.count_nonzero(changed)), first_byte_index=byte_index,
            differing_elements=int(np.count_nonzero(elements)), first_flat_index=element_index,
            first_index=[int(x) for x in np.unravel_index(element_index,a.shape)] if a.shape else [],
            maximum_absolute_difference=None, maximum_relative_difference=None,
            signed_zero_differences=None, floating_diagnostic_available=False)
        if a.dtype.kind == 'f' or str(a.dtype) == 'bfloat16':
          av,bv = a.astype(np.float64),b.astype(np.float64)
          with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            delta = np.abs(av-bv)
            scale = np.maximum(np.abs(av),np.abs(bv))
            relative = np.divide(delta,scale,out=np.zeros_like(delta),where=scale>0)
          detail.update(floating_diagnostic_available=True,
              maximum_absolute_difference=finite_number(delta.max()),
              maximum_relative_difference=finite_number(relative.max()),
              signed_zero_differences=int(np.count_nonzero(
                  (av==0)&(bv==0)&(np.signbit(av)!=np.signbit(bv)))),
              promoted_float64_diagnostic_only=True,
              derived_diagnostic_nonfinite=not (np.isfinite(delta).all() and np.isfinite(relative).all()))
        record(path, 'bytes', **detail)
  visit(left,right,'')
  return dict(equal=count == 0, differing_nodes=count, first_differences=first,
      detail_limit=limit, comparison='exact container/type/shape/dtype/bytes')


def seal(args, output, check):
  from scripts import dl_reset_learning as learning
  from scripts import dl_rep_native as native
  auth = read_json(output/'guard-authority.json')
  comparison = {}
  for mode, anchor in (('Bwhole', 'old_B'), ('Amin', 'native_A')):
    receipt = read_json(output/(mode+'-complete.json'))
    newpath = output/(mode+'-checkpoint.pkl')
    original = Path(auth[anchor+'_checkpoint_path'])
    new = native.load_payload(newpath, receipt['checkpoint_sha256'])
    old = native.load_payload(original, auth[anchor+'_checkpoint_sha256'])
    newhash, oldhash = learning.tree_digest(new), learning.tree_digest(old)
    require(newhash == receipt['learning']['final_state_digest'], 'Guard checkpoint receipt differs')
    difference = typed_differences(new, old)
    require(difference['equal'] is (newhash == oldhash), 'Independent typed comparison/digest disagrees')
    comparison[mode] = dict(final_typed_state_equal=newhash == oldhash,
        endpoint_differences=difference,
        new_full_state_sha256=newhash, historical_full_state_sha256=oldhash,
        common_metrics=receipt['learning']['common_metrics_comparison'],
        helper_equivalent=receipt['learning']['equivalent'],
        written_checkpoints_standard_reload_byte_exact=(
            receipt['learning']['standard_final_checkpoint_reload_byte_exact'] and
            receipt['learning']['historical_anchor_standard_reload_byte_exact']),
        complete_128_updates=receipt['learning']['updates'] == 128)
    check()
  for path, pin in auth['immutable_inputs_sha256'].items():
    check(); require(sha(path) == pin, 'Scientific input changed during guard')
  write(output/'result.json', dict(version=VERSION, actual_git_sha=args.expected_git,
      comparisons=comparison, complete_paths=2, actual_training_updates=256,
      endpoint_observation_equivalent=all(r['final_typed_state_equal'] for r in comparison.values()),
      guard_passed=all(r['final_typed_state_equal'] and r['complete_128_updates'] and
          r['common_metrics']['passed'] and r['helper_equivalent'] and
          r['written_checkpoints_standard_reload_byte_exact'] for r in comparison.values()),
      legacy_per_update_full_state_comparison_performed=False,
      new_native_evaluation_performed=False, independent_second_signal_validated=False,
      method_effectiveness_established=False, new_physics_steps=0, new_RGB_frames=0,
      guard_authority_sha256=sha(output/'guard-authority.json'),
      checkpoint_sha256={mode:sha(output/(mode+'-checkpoint.pkl')) for mode in ('Bwhole', 'Amin')},
      complete_receipt_sha256={mode:sha(output/(mode+'-complete.json')) for mode in ('Bwhole', 'Amin')},
      decision='endpoint-only fixed observation-graph guard; exact mismatches retained'))


def worker(args):
  sys.path.insert(0, str(ROOT))
  output = Path(args.output).resolve(strict=True)
  budget = read_json(output/'budget.json')
  require(os.environ.get('DL_REP_GUARD_PARENT') == str(budget['parent_pid']) and
      os.getppid() == budget['parent_pid'], 'Actual live owned guard parent required')
  os.kill(os.getppid(), 0)
  require(budget['actual_git_sha'] == args.expected_git and budget['seconds'] == SECONDS and
      budget['version'] == VERSION, 'Parent guard budget differs')
  def check(): require(time.monotonic() < budget['monotonic_start']+SECONDS, 'Whole-entry guard timeout')
  check(); require(clean_binding(args.expected_git) == budget['source_files_sha256'], 'Worker code differs')
  if args.worker_phase == 'authenticate': input_authentication(args, output, check)
  elif args.worker_phase in ('Bwhole', 'Amin'): learn(args, output, check)
  else: seal(args, output, check)
  check(); require(clean_binding(args.expected_git) == budget['source_files_sha256'], 'Final worker code differs')


def parent(args):
  sys.path.insert(0, str(ROOT))
  from scripts import dl_rep_native as native
  from scripts import dl_resources as resource
  require(os.name == 'posix', 'Whole-entry owned GPU guard requires POSIX')
  output = Path(args.output).resolve()
  require(not output.exists(), 'New guard attempt directory required')
  output.mkdir(parents=True)
  # Parent helpers have separate owned groups; worker helpers share the owned
  # phase group. Both paths are covered by this parent's ALRM/TERM cleanup.
  os.environ.pop('DL_REP_NATIVE_PARENT', None)
  native.OWNED = OWNED
  native.owned_output = owned_metadata
  resource.command_output = lambda command:native.owned_output(command).decode()
  sources = clean_binding(args.expected_git)
  device = resource.require_resource(args.resource_record)
  require(device['physical_gpu'] == 4, 'No resource substitution permitted')
  write(output/'budget.json', dict(version=VERSION, actual_git_sha=args.expected_git,
      parent_pid=os.getpid(), monotonic_start=ENTRY_STARTED, seconds=SECONDS,
      planned_physical_gpu=4, max_gpu_devices=1, max_gpu_hours=.5,
      source_files_sha256=sources, new_physics_steps=0, new_native_evaluation=False))
  deadline, phases = ENTRY_STARTED+SECONDS, []
  complete = False
  for phase in PHASES:
    env = dict(os.environ, DL_REP_GUARD_PARENT=str(os.getpid()),
        DL_REP_NATIVE_PARENT=str(os.getpid()), DL_NATIVE_AUDIT_PARENT=str(os.getpid()),
        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    is_gpu = phase in ('Bwhole', 'Amin')
    resource_path = args.resource_record
    if is_gpu:
      env.update(CUDA_VISIBLE_DEVICES=device['gpu_uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
          MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='4', PYOPENGL_PLATFORM='egl')
      original_command = resource.command_output
      resource.command_output = lambda command:native.owned_output(command).decode()
      try: fresh = resource.refresh(device)
      finally: resource.command_output = original_command
      require(fresh['physical_gpu'] == 4, 'No resource substitution')
      resource_path = str(output/(phase+'-resource.json')); write(resource_path, fresh)
    else: env.update(CUDA_VISIBLE_DEVICES='', MUJOCO_GL='disable', JAX_PLATFORMS='cpu')
    command = [sys.executable, str(Path(__file__).resolve()), '--worker-phase', phase,
        '--resource-record', resource_path]
    for key in KEYS: command += ['--'+key.replace('_','-'), str(getattr(args,key))]
    with (output/(phase+'.log')).open('x', encoding='utf-8') as log:
      child = spawn_owned(command, deadline, cwd=ROOT, env=env, stdout=log,
          stderr=subprocess.STDOUT)
      started = time.monotonic(); timed_out = False
      try: code = child.wait(timeout=max(.001, deadline-time.monotonic()))
      except subprocess.TimeoutExpired: code = 124; timed_out = True
      finally:
        cleanup_owned(child)
    terminal = dict(phase=phase, worker_pid=child.pid, worker_terminal=True,
        owned_group_cleanup_requested=True, elapsed_seconds=time.monotonic()-started,
        exit_code=code, timed_out=timed_out, is_gpu=is_gpu)
    phases.append(terminal); write(output/(phase+'-exit.json'), terminal)
    print(json.dumps(terminal), flush=True)
    if code or time.monotonic() >= deadline: break
  complete = len(phases) == len(PHASES) and all(p['exit_code'] == 0 for p in phases) and time.monotonic() < deadline
  if complete: require(clean_binding(args.expected_git) == sources, 'Final parent code differs')
  code = phases[-1]['exit_code'] if phases else 1
  receipt = dict(version=VERSION, actual_git_sha=args.expected_git,
      parent_pid=os.getpid(), phases=phases, complete_verified=complete,
      exit_code=code if code else (0 if complete else 1),
      timed_out=any(p['timed_out'] for p in phases),
      elapsed_parent_entry_seconds=time.monotonic()-ENTRY_STARTED,
      budget_sha256=sha(output/'budget.json'), result_sha256=sha(output/'result.json') if complete else None,
      actual_gpu_devices_used=1 if complete else None, method_effectiveness_established=False)
  pending = output/'supervisor-receipt-pending.json'; write(pending, receipt)
  if time.monotonic() >= deadline:
    write(output/'late-supervisor.json', dict(exit_code=124, incomplete=True)); return 124
  os.replace(pending, output/'supervisor_complete.json')
  if time.monotonic() >= deadline:
    os.replace(output/'supervisor_complete.json', output/'late-provisional-supervisor.json')
    write(output/'late-supervisor.json', dict(exit_code=124, incomplete=True)); return 124
  return receipt['exit_code']


def verify_completion(directory, external_cli_terminal):
  """Verify a completed guard, keeping endpoint mismatch a valid result."""
  output = Path(directory).resolve(strict=True)
  require(not any((output/name).exists() for name in ('late-supervisor.json',
      'late-provisional-supervisor.json', 'supervisor-receipt-pending.json')),
      'Late/provisional guard cannot complete')
  supervisor = read_json(output/'supervisor_complete.json')
  terminal = read_json(external_cli_terminal)
  require(supervisor['version'] == VERSION and supervisor['complete_verified'] is True and
      supervisor['exit_code'] == 0 and supervisor['timed_out'] is False and
      supervisor['actual_gpu_devices_used'] == 1 and
      0 < supervisor['elapsed_parent_entry_seconds'] <= SECONDS and
      terminal['actual_git_sha'] == supervisor['actual_git_sha'] and
      terminal['worker_pid'] == supervisor['parent_pid'] and
      terminal['exit_code'] == 0 and terminal['worker_terminal'] is True and
      terminal['timed_out'] is False and 0 < terminal['elapsed_cli_seconds'] <= SECONDS,
      'Full guard parent and external terminal required')
  budget = read_json(output/'budget.json')
  require(sha(output/'budget.json') == supervisor['budget_sha256'] and
      budget['version'] == VERSION and budget['seconds'] == SECONDS and
      budget['actual_git_sha'] == supervisor['actual_git_sha'] and
      budget['parent_pid'] == supervisor['parent_pid'] and
      budget['max_gpu_devices'] == 1 and budget['planned_physical_gpu'] == 4 and
      budget['max_gpu_hours'] == .5, 'Guard budget authority differs')
  phases = supervisor['phases']
  require([p['phase'] for p in phases] == list(PHASES) and
      all(p['exit_code'] == 0 and p['worker_terminal'] is True and p['timed_out'] is False
          and p['is_gpu'] is (p['phase'] in ('Bwhole', 'Amin')) and
          0 < p['elapsed_seconds'] <= SECONDS for p in phases),
      'Four fully terminated fixed guard phases required')
  for phase in phases:
    require(read_json(output/(phase['phase']+'-exit.json')) == phase, 'Unbound phase terminal')
  result = read_json(output/'result.json')
  require(sha(output/'result.json') == supervisor['result_sha256'] and
      result['version'] == VERSION and result['actual_git_sha'] == supervisor['actual_git_sha'] and
      result['complete_paths'] == 2 and result['actual_training_updates'] == 256 and
      result['new_physics_steps'] == result['new_RGB_frames'] == 0 and
      result['legacy_per_update_full_state_comparison_performed'] is False and
      result['new_native_evaluation_performed'] is False and
      result['method_effectiveness_established'] is False and
      sha(output/'guard-authority.json') == result['guard_authority_sha256'],
      'Complete endpoint-only guard scope differs')
  for mode in ('Bwhole', 'Amin'):
    require(sha(output/(mode+'-checkpoint.pkl')) == result['checkpoint_sha256'][mode] and
        sha(output/(mode+'-complete.json')) == result['complete_receipt_sha256'][mode],
        'Guard generated endpoint bytes changed')
  require(result['endpoint_observation_equivalent'] is
      all(v['final_typed_state_equal'] for v in result['comparisons'].values()),
      'Guard equivalence decision differs from both exact endpoint comparisons')
  require(result['guard_passed'] is all(v['final_typed_state_equal'] and
      v['complete_128_updates'] and v['common_metrics']['passed'] and v['helper_equivalent'] and
      v['written_checkpoints_standard_reload_byte_exact']
      for v in result['comparisons'].values()), 'Guard pass decision does not include all fixed gates')
  return dict(complete_external_exit_verified=True, result_sha256=supervisor['result_sha256'],
      endpoint_observation_equivalent=result['endpoint_observation_equivalent'],
      guard_passed=result['guard_passed'],
      historical_per_update_full_state_verified=False, method_effectiveness_established=False)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for key in KEYS:
    parser.add_argument('--'+key.replace('_','-'), required=key != 'task',
        **({'default':'dmc_quadruped_walk'} if key == 'task' else {}))
  parser.add_argument('--resource-record', required=True)
  parser.add_argument('--worker-phase', choices=PHASES)
  args = parser.parse_args()
  if args.worker_phase: worker(args); return 0
  return parent(args)


if __name__ == '__main__': sys.exit(main())
