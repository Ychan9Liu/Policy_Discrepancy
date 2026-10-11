"""Candidate CPU endpoint audit of the fixed two-path observation guard.

Engineering preparation does not authorize the real guard. Its caller must
first obtain adoption and authenticate a successful native primary/audit.

The four authenticated checkpoint payloads are decoded as host numeric trees.
No JAX/model/optimizer, environment, renderer or scientific source array is
run or decoded. Common metrics and update/random-stream evidence are bound
producer receipts, not an independent replay of historical learning.
"""
import time
ENTRY_STARTED = time.monotonic()
import os
import signal
import sys
SECONDS = 180
OWNED = set()
IN_WORKER = False


def expire(signum, frame):
  for pid in tuple(OWNED):
    try: os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError: pass
  os._exit(124)


if __name__ == '__main__' and os.name == 'posix':
  signal.signal(signal.SIGALRM, expire)
  signal.signal(signal.SIGTERM, expire)
  signal.setitimer(signal.ITIMER_REAL,
      max(.001, SECONDS-(time.monotonic()-ENTRY_STARTED)))

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import re
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'DL-rep-observation-guard-r1'
IMPLEMENTATION = 'DL-rep-observation-independent-audit-r1'
MODES = ('Bwhole', 'Amin')
OWN_FILES = ('analysis/dl_rep_observation_audit.py',
    'tests/test_dl_rep_observation_audit.py')
WHOLE_FIELDS = frozenset('dl_allrep/'+name for name in (
    'active_count', 'excess_sum', 'position_count', 'nonreset_active_count',
    'nonreset_excess_sum'))


def require(value, message):
  if not value: raise ValueError(message)


def exact_json(left, right):
  """Retain JSON container/scalar types and signed zero in receipt bindings."""
  if type(left) is not type(right): return False
  if type(left) is dict:
    return set(left) == set(right) and all(exact_json(left[k], right[k]) for k in left)
  if type(left) is list:
    return len(left) == len(right) and all(exact_json(a, b) for a, b in zip(left, right))
  require(left is None or type(left) in (bool, int, float, str), 'Exact JSON leaf required')
  require(type(left) is not float or math.isfinite(left) and math.isfinite(right), 'Finite JSON leaf required')
  return json.dumps(left) == json.dumps(right)


def pin(value):
  require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'SHA256 required')
  return value


def sha(path, check=lambda: None):
  digest = hashlib.sha256(); check()
  with Path(path).open('rb') as stream:
    for part in iter(lambda: stream.read(1024*1024), b''):
      digest.update(part); check()
  return digest.hexdigest()


def read_json(path, expected=None, check=lambda: None):
  check(); payload = Path(path).read_bytes(); check()
  if expected is not None:
    require(hashlib.sha256(payload).hexdigest() == pin(expected), 'JSON payload hash differs')
  def pairs(items):
    result = {}
    for key, value in items:
      require(key not in result, 'Duplicate JSON key: '+key); result[key] = value
    return result
  def finite(text):
    value = float(text); require(math.isfinite(value), 'Nonfinite JSON float'); return value
  return json.loads(payload, object_pairs_hook=pairs, parse_float=finite,
      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def write_json(path, value):
  with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')


def no_model_imports():
  forbidden = ('jax', 'jaxlib', 'ninjax', 'dreamerv3', 'embodied.jax', 'mujoco', 'dm_control', 'OpenGL')
  require(not any(name == prefix or name.startswith(prefix+'.')
      for name in sys.modules for prefix in forbidden), 'CPU audit imported model/JAX code')


def numeric(value):
  array = np.asarray(value)
  valid = array.dtype.kind in 'buifc'
  if array.dtype.kind == 'V':
    import ml_dtypes
    valid = array.dtype == np.dtype(ml_dtypes.bfloat16)
  require(valid and not array.dtype.hasobject and np.isfinite(array).all(),
      'Finite numeric host leaf required; arbitrary void/object forbidden')
  return array


def tree_digest(tree):
  """Independent implementation of the frozen typed-tree wire encoding."""
  result = hashlib.sha256()
  def visit(value):
    if type(value) is dict:
      require(all(type(k) is str for k in value), 'String state keys required')
      result.update(b'dict')
      for key in sorted(value): result.update(json.dumps(key).encode()); visit(value[key])
    elif type(value) in (tuple, list):
      result.update(type(value).__name__.encode()); result.update(str(len(value)).encode())
      for child in value: visit(child)
    elif value is None or type(value) in (bool, int, float, str):
      require(type(value) is not float or math.isfinite(value), 'Nonfinite scalar')
      result.update(type(value).__name__.encode()); result.update(json.dumps(value).encode())
    else:
      array = numeric(value)
      result.update(json.dumps([array.dtype.str, str(array.dtype)]).encode())
      result.update(json.dumps(array.shape).encode()); result.update(array.tobytes(order='C'))
  visit(tree)
  return result.hexdigest()


def compare_trees(left, right, limit=16):
  """Compare containers and every raw numeric byte without a tolerance."""
  require(type(limit) is int and limit > 0, 'Positive detail limit required')
  differences, count, leaves = [], 0, 0
  def note(path, kind, **values):
    nonlocal count
    count += 1
    if len(differences) < limit: differences.append(dict(path=path, kind=kind, **values))
  def visit(a, b, path):
    nonlocal leaves
    if type(a) is dict or type(b) is dict:
      if type(a) is not dict or type(b) is not dict: note(path, 'container_type'); return
      require(all(type(k) is str for k in a) and all(type(k) is str for k in b), 'String keys required')
      if set(a) != set(b): note(path, 'dict_keys', left=sorted(a), right=sorted(b))
      for key in sorted(set(a)&set(b)): visit(a[key], b[key], path+'/'+key)
    elif type(a) in (tuple, list) or type(b) in (tuple, list):
      if type(a) is not type(b): note(path, 'container_type'); return
      if len(a) != len(b): note(path, 'container_length', left=len(a), right=len(b)); return
      for index, (aa, bb) in enumerate(zip(a, b)): visit(aa, bb, path+'/'+str(index))
    elif a is None or b is None or type(a) in (bool, int, float, str) or type(b) in (bool, int, float, str):
      leaves += 1
      for value in (a, b):
        require(type(value) is not float or math.isfinite(value), 'Nonfinite scalar')
      if type(a) is not type(b) or json.dumps(a) != json.dumps(b):
        note(path, 'scalar_type_or_value', left_type=type(a).__name__, right_type=type(b).__name__)
    else:
      leaves += 1; aa, bb = numeric(a), numeric(b)
      if (aa.dtype.str, str(aa.dtype)) != (bb.dtype.str, str(bb.dtype)):
        note(path, 'dtype', left=str(aa.dtype), right=str(bb.dtype)); return
      if aa.shape != bb.shape: note(path, 'shape', left=list(aa.shape), right=list(bb.shape)); return
      abytes, bbytes = aa.tobytes(order='C'), bb.tobytes(order='C')
      if abytes != bbytes:
        delta = np.frombuffer(abytes, np.uint8) != np.frombuffer(bbytes, np.uint8)
        note(path, 'bytes', first_byte_index=int(delta.argmax()), differing_bytes=int(np.count_nonzero(delta)),
            left_sha256=hashlib.sha256(abytes).hexdigest(), right_sha256=hashlib.sha256(bbytes).hexdigest())
  visit(left, right, '')
  return dict(equal=count == 0, differing_nodes=count, compared_leaves=leaves,
      first_differences=differences, detail_limit=limit,
      comparison='independent exact container/key/dtype/shape/rawbytes')


def load_checkpoint(path, expected, check=lambda: None):
  """Hash the exact trusted payload before decoding, never reopen for pickle."""
  check(); payload = Path(path).read_bytes(); check()
  require(hashlib.sha256(payload).hexdigest() == pin(expected), 'Checkpoint payload hash differs before decode')
  value = pickle.load(io.BytesIO(payload)); no_model_imports(); check()
  require(type(value) is dict and set(value) == {'params', 'counters'} and
      type(value['params']) is dict and value['params'], 'Complete host Agent checkpoint required')
  counters = value['counters']
  require(type(counters) is dict and set(counters) == {'updates', 'batches', 'actions', 'eval_actions', 'eval_seed'}
      and all(type(v) is int and v >= 0 for v in counters.values()), 'Complete exact Agent counters required')
  step = numeric(value['params'].get('opt/step/value'))
  require(step.dtype == np.int32 and step.shape == () and int(step) >= 0, 'Official int32 optimizer leaf required')
  tree_digest(value)
  return value


def cleanup(child, own_group=True):
  if own_group and os.name == 'posix':
    try: os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError: pass
  elif child.poll() is None: child.kill()
  child.wait(timeout=5); OWNED.discard(child.pid)


def spawn(command, deadline, *, own_group=True, **kwargs):
  require(time.monotonic() < deadline, 'Late before owned process creation')
  if os.name == 'posix':
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM, signal.SIGTERM})
    try:
      child = subprocess.Popen(command, start_new_session=own_group,
          preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, previous), **kwargs)
      if own_group: OWNED.add(child.pid)
      return child
    finally: signal.pthread_sigmask(signal.SIG_SETMASK, previous)
  child = subprocess.Popen(command, **kwargs); OWNED.add(child.pid); return child


def metadata(command, check=lambda: None):
  check(); deadline = ENTRY_STARTED+SECONDS; own_group = not IN_WORKER
  child = spawn(command, deadline, own_group=own_group, cwd=ROOT,
      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  try:
    out, err = child.communicate(timeout=min(30, max(.001, deadline-time.monotonic())))
    require(child.returncode == 0, 'Owned metadata command failed: '+err.decode(errors='replace')[:256])
    check(); return out
  finally: cleanup(child, own_group)


def code_binding(current_git, primary_git, source_pins, check=lambda: None):
  require(all(type(v) is str and re.fullmatch('[0-9a-f]{40}', v) for v in (current_git, primary_git)),
      'Full current/primary Git SHA required')
  require(metadata(['git', 'rev-parse', 'HEAD'], check).decode().strip() == current_git and
      not metadata(['git', 'status', '--porcelain', '--untracked-files=all'], check), 'Clean audit checkout required')
  require(type(source_pins) is dict and {'scripts/dl_rep_observation_runner.py',
      'scripts/dl_rep_observation_guard.py', 'docs/DL_REP_OBSERVATION_GUARD.md'} <= set(source_pins),
      'Missing guard source Git pins')
  for name, expected in source_pins.items():
    require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts,
        'Unsafe source name')
    blob = metadata(['git', 'show', primary_git+':'+name], check)
    require(hashlib.sha256(blob).hexdigest() == pin(expected) and (ROOT/name).read_bytes() == blob,
        'Producer Git/source binding differs: '+name)
  own = {}
  for name in OWN_FILES:
    blob = metadata(['git', 'show', current_git+':'+name], check)
    require((ROOT/name).read_bytes() == blob, 'Audit Git/source binding differs: '+name)
    own[name] = hashlib.sha256(blob).hexdigest()
  return own


def authenticate_primary(primary, external_cli_terminal, current_git, check=lambda: None):
  """Reuse production completion verifier, independently bind its exact files."""
  from scripts import dl_rep_observation_runner as runner
  root = Path(primary).resolve(strict=True); check()
  verification = runner.verify_completion(root, external_cli_terminal)
  supervisor = read_json(root/'supervisor_complete.json', check=check)
  budget = read_json(root/'budget.json', supervisor['budget_sha256'], check)
  result = read_json(root/'result.json', supervisor['result_sha256'], check)
  require(type(result['comparisons']) is dict and set(result['comparisons']) == set(MODES),
      'Exactly two fixed endpoint comparisons required')
  own = code_binding(current_git, supervisor['actual_git_sha'], budget['source_files_sha256'], check)
  auth = read_json(root/'guard-authority.json', result['guard_authority_sha256'], check)
  require(auth.get('version') == VERSION and auth.get('actual_git_sha') == supervisor['actual_git_sha'] and
      auth.get('endpoint_only') is True and auth.get('legacy_per_update_full_state_available') is False and
      auth.get('new_native_evaluation_performed') is False, 'Endpoint authority scope differs')
  legacy = read_json(root/'old-authenticated.json', auth['old_authentication_sha256'], check)
  require(legacy.get('old_full_exit_and_independent_audit_authenticated') is True and
      legacy.get('original_19_producer_files_unchanged') is True and
      exact_json(legacy['original_B_learning'], auth['old_B_learning']),
      'Old B authentication/learning authority binding differs')
  immutable = auth['immutable_inputs_sha256']
  require(type(immutable) is dict and immutable, 'Original immutable input pins required')
  require(type(legacy['immutable_inputs_sha256']) is dict and legacy['immutable_inputs_sha256'] and
      all(immutable.get(path) == expected for path, expected in legacy['immutable_inputs_sha256'].items()),
      'Original complete immutable authentication pins missing from guard authority')
  native = auth['native_independent_verification']
  require(native.get('complete_published_parent_verified') is True and
      native.get('complete_external_exit_verified') is True,
      'Successful completed native independent audit receipt required')
  for label in ('old_B', 'native_A'):
    path = str(Path(auth[label+'_checkpoint_path']).resolve(strict=True))
    require(immutable.get(path) == pin(auth[label+'_checkpoint_sha256']), 'Anchor absent from immutable authority')
  for path, expected in immutable.items():
    require(Path(path).is_absolute() and sha(path, check) == pin(expected), 'Original input byte binding differs')
  endpoints = {name:sha(root/name, check) for name in ('budget.json', 'supervisor_complete.json',
      'result.json', 'guard-authority.json', 'old-authenticated.json')}
  for phase in runner.PHASES: endpoints[phase+'-exit.json'] = sha(root/(phase+'-exit.json'), check)
  for mode in MODES:
    for suffix in ('-complete.json', '-checkpoint.pkl'):
      endpoints[mode+suffix] = sha(root/(mode+suffix), check)
  terminal = str(Path(external_cli_terminal).resolve(strict=True))
  return dict(root=root, result=result, authority=auth, budget=budget,
      verification=verification, primary_git=supervisor['actual_git_sha'], current_git=current_git,
      immutable=immutable, endpoints=endpoints, external_cli_terminal=terminal,
      external_cli_terminal_sha256=sha(terminal, check), audit_sources=own)


def validate_receipt(mode, receipt, authority, comparison):
  """Authenticate 128 bound receipts; this is not an optimizer/model replay."""
  require(receipt.get('version') == VERSION and receipt.get('phase') == mode and
      receipt.get('new_native_evaluation_performed') is False, 'Guard path receipt scope differs')
  learning = receipt['learning']; reference = authority['old_B_learning' if mode == 'Bwhole' else 'current_A_learning']
  require(learning.get('mode') == mode and learning.get('updates') == 128 and
      learning.get('all_fixed128_updates_completed') is True and
      learning.get('input_reference_values_matched') is True and
      learning.get('historical_per_update_fullstate_carry_outs_equivalence_claimed') is False and
      learning.get('rep_gradient_stopped') is (mode == 'Amin'), 'Complete fixed path/gradient scope differs')
  require(type(learning['receipts']) is type(reference['receipts']) is list and
      len(learning['receipts']) == len(reference['receipts']) == 128 and
      exact_json(learning['optimizer_counter'], reference['optimizer_counter']), 'Complete128 optimizer receipt binding differs')
  for name in ('source_checkpoint_digest', 'initial_state_digest', 'initial_params_digest', 'binding',
      'calibration_arrays_digest', 'restored_counters', 'saved_source_counters'):
    require(exact_json(learning[name], reference[name]), 'Original path input receipt differs: '+name)
  initial_step = learning['optimizer_counter']['initial']
  require(type(initial_step) is int and initial_step >= 0 and exact_json(learning['optimizer_counter'], dict(key='opt/step/value', dtype='int32', shape=[],
      initial=initial_step, final=initial_step+128, updates=128)), 'Official optimizer format differs')
  counters = learning['restored_counters']
  require(type(counters) is dict and set(counters) == {'updates', 'batches', 'actions', 'eval_actions', 'eval_seed'} and
      all(type(v) is int and v >= 0 for v in counters.values()), 'Exact restored counters required')
  for index, row in enumerate(learning['receipts']):
    target = reference['receipts'][index]
    require(type(row['update']) is type(target['update']) is type(row['optimizer_step']) is int and
        row['update'] == target['update'] == index+1 and row['optimizer_step'] == initial_step+index+1 and
        all(exact_json(row[name], target[name]) for name in ('seed_counter', 'train_seed', 'counters')) and
        type(row['seed_counter']) is int and row['seed_counter'] == counters['batches']+index and
        type(row['train_seed']) is list and len(row['train_seed']) == 2 and
        all(type(v) is int and 0 <= v < 2**32 for v in row['train_seed']) and
        exact_json(row['counters'], {**counters, 'updates':counters['updates']+index+1,
            'batches':counters['batches']+index+1}),
        'Bound normal key/counter/update sequence differs')
    require(row['historical_per_update_fullstate_carry_outs_anchor_available'] is False,
        'Historical per-update anchors cannot be invented')
    for name in ('metrics_digest', 'common_metrics_digest', 'full_state_digest',
        'full_params_digest', 'carry_digest', 'outs_digest'): pin(row[name])
    require(type(row['metrics']['opt/updates']) is int and
        row['metrics']['opt/updates'] == row['optimizer_step'], 'Official optimizer metric receipt differs')
  common = learning['common_metrics_comparison']
  require(type(learning['equivalent']) is bool and type(common['passed']) is bool and
      type(common['rows']) is type(common['per_update_equal']) is list and len(common['rows']) == 128 and
      len(common['per_update_equal']) == 128 and exact_json(comparison['common_metrics'], common) and
      comparison['helper_equivalent'] is learning['equivalent'] and
      comparison['complete_128_updates'] is True and
      comparison['written_checkpoints_standard_reload_byte_exact'] is True and
      learning['standard_final_checkpoint_reload_byte_exact'] is True and
      learning['historical_anchor_standard_reload_byte_exact'] is True, 'Common metric/reload receipt binding differs')
  for index, row in enumerate(learning['receipts']):
    brief = common['rows'][index]
    require(type(row['common_metrics_equal']) is type(row['common_metrics_json_exact']) is bool and
        row['common_metrics_typed_exact'] in (True, False, None) and
        (row['common_metrics_typed_exact'] is None or type(row['common_metrics_typed_exact']) is bool) and
        type(common['per_update_equal'][index]) is bool and type(brief['update']) is int and
        brief['update'] == index+1 and brief['equal'] is row['common_metrics_equal'] and
        brief['json_numeric_exact'] is row['common_metrics_json_exact'] and
        brief['typed_exact'] is row['common_metrics_typed_exact'], 'Bound per-update common metric summary differs')
  require(common['per_update_equal'] == [row['common_metrics_equal'] for row in learning['receipts']] and
      common['passed'] is (all(common['per_update_equal']) and
          all(row['common_metrics_typed_exact'] is not False for row in learning['receipts'])),
      'Common metric pass status differs from bound128 evidence')
  return learning


def audit_authenticated(bound, check=lambda: None):
  root, result, authority = bound['root'], bound['result'], bound['authority']
  comparisons = {}
  for mode, anchor in (('Bwhole', 'old_B'), ('Amin', 'native_A')):
    check()
    receipt = read_json(root/(mode+'-complete.json'), result['complete_receipt_sha256'][mode], check)
    require(receipt['actual_git_sha'] == bound['primary_git'] and
        receipt['guard_authority_sha256'] == result['guard_authority_sha256'] and
        receipt['checkpoint_sha256'] == result['checkpoint_sha256'][mode], 'Unbound path authority/checkpoint')
    learning = validate_receipt(mode, receipt, authority, result['comparisons'][mode])
    new = load_checkpoint(root/(mode+'-checkpoint.pkl'), result['checkpoint_sha256'][mode], check)
    old = load_checkpoint(authority[anchor+'_checkpoint_path'], authority[anchor+'_checkpoint_sha256'], check)
    expected_counters = {**learning['restored_counters'],
        'updates':learning['restored_counters']['updates']+128,
        'batches':learning['restored_counters']['batches']+128}
    require(all(exact_json(value['counters'], expected_counters) and
        int(value['params']['opt/step/value']) == learning['optimizer_counter']['final'] for value in (new, old)),
        'Actual complete endpoint counters/optimizer do not match fixed128 receipts')
    newhash, oldhash = tree_digest(new), tree_digest(old)
    require(newhash == learning['final_state_digest'] and tree_digest(new['params']) == learning['final_params_digest'],
        'New complete endpoint differs from bound typed receipt')
    reference = authority['old_B_learning' if mode == 'Bwhole' else 'current_A_learning']
    require(oldhash == reference['final_state_digest'] and tree_digest(old['params']) == reference['final_params_digest'],
        'Actual historical anchor differs from bound typed endpoint')
    difference = compare_trees(new, old)
    claimed = result['comparisons'][mode]
    require(difference['equal'] is (newhash == oldhash) and
        claimed['final_typed_state_equal'] is difference['equal'] and
        claimed['new_full_state_sha256'] == newhash and claimed['historical_full_state_sha256'] == oldhash and
        claimed['endpoint_differences']['equal'] is difference['equal'], 'Producer endpoint comparison/status differs')
    guard = difference['equal'] and claimed['common_metrics']['passed'] and claimed['helper_equivalent']
    comparisons[mode] = dict(equal=difference['equal'], independent_difference=difference,
        new_full_state_sha256=newhash, historical_full_state_sha256=oldhash,
        new_checkpoint_sha256=result['checkpoint_sha256'][mode],
        historical_checkpoint_sha256=authority[anchor+'_checkpoint_sha256'],
        guard_path_passed=guard, common_metrics_passed=claimed['common_metrics']['passed'],
        common_metrics_and_random_stream_receipts_only_authenticated=True)
    del new, old
  equivalent = all(row['equal'] for row in comparisons.values())
  passed = all(row['guard_path_passed'] for row in comparisons.values())
  require(result['endpoint_observation_equivalent'] is equivalent and result['guard_passed'] is passed,
      'Guard outcome differs from independent endpoint/bound-metric recomputation')
  for path, expected in bound['immutable'].items(): require(sha(path, check) == expected, 'Original input changed during audit')
  for name, expected in bound['endpoints'].items(): require(sha(root/name, check) == expected, 'Primary endpoint changed during audit')
  require(sha(bound['external_cli_terminal'], check) == bound['external_cli_terminal_sha256'], 'Primary CLI endpoint changed')
  for name, expected in {**bound['budget']['source_files_sha256'], **bound['audit_sources']}.items():
    require(sha(ROOT/name, check) == expected, 'Source changed during audit')
  no_model_imports(); check()
  return dict(version=IMPLEMENTATION, status='passed', passed=True,
      primary=str(root), primary_git_sha=bound['primary_git'], auditor_git_sha=bound['current_git'],
      comparisons=comparisons, endpoint_observation_equivalent=equivalent, guard_passed=passed,
      primary_endpoint_sha256=bound['endpoints'],
      primary_external_cli_terminal_sha256=bound['external_cli_terminal_sha256'],
      audit_source_sha256=bound['audit_sources'], source_immutable_sha256=bound['immutable'],
      complete_primary_exit_verified=True, complete_paths=2, actual_primary_training_updates=256,
      actual_auditor_optimizer_updates=0, actual_gpu_devices_used=0,
      checkpoint_payloads_decoded=4, source_arrays_decoded=False, native_records_decoded=False,
      new_physics_steps=0, new_RGB_frames=0, model_or_JAX_imported=False,
      independent_endpoint_container_key_dtype_shape_bytes_comparison=True,
      independent_optimizer_or_historical_per_update_replay=False,
      method_effectiveness_established=False,
      conditional_on='four authenticated host checkpoints; bound producer common-metric/key/counter/reload receipts')


def run_owned(command, log, deadline, env):
  child = spawn(command, deadline, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
  start, timed_out = time.monotonic(), False
  try: code = child.wait(timeout=max(.001, deadline-time.monotonic()))
  except subprocess.TimeoutExpired: code, timed_out = 124, True
  finally: cleanup(child)
  return dict(worker_pid=child.pid, worker_terminal=child.poll() is not None,
      exit_code=code, timed_out=timed_out, elapsed_seconds=time.monotonic()-start,
      owned_group_cleanup_requested=os.name == 'posix')


def worker(args):
  global IN_WORKER
  output = Path(args.output).resolve(strict=True); budget = read_json(output/'budget.json')
  require(budget['version'] == VERSION and budget['implementation'] == IMPLEMENTATION and
      budget['seconds'] == SECONDS and budget['actual_git_sha'] == args.expected_git and
      os.environ.get('DL_REP_OBSERVATION_AUDIT_PARENT') == str(budget['parent_pid']) and
      os.getppid() == budget['parent_pid'], 'Actual live owned CPU audit parent required')
  os.kill(os.getppid(), 0); IN_WORKER = True
  require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('MUJOCO_GL') == 'disable' and
      os.environ.get('JAX_PLATFORMS') == 'cpu', 'CPU/no-render environment required')
  deadline = budget['monotonic_start']+SECONDS
  def check(): require(time.monotonic() < deadline, 'Whole-entry CPU180s audit exhausted')
  check(); bound = authenticate_primary(args.primary, args.external_cli_terminal, args.expected_git, check)
  result = audit_authenticated(bound, check); no_model_imports()
  require(metadata(['git', 'rev-parse', 'HEAD'], check).decode().strip() == args.expected_git and
      not metadata(['git', 'status', '--porcelain', '--untracked-files=all'], check), 'Final audit Git binding differs')
  check(); write_json(output/'audit.json', result); check()


def publish_receipt(output, receipt, deadline):
  pending = Path(output)/'supervisor-receipt-pending.json'; write_json(pending, receipt)
  if time.monotonic() >= deadline:
    write_json(Path(output)/'late-supervisor.json', dict(exit_code=124, incomplete=True)); return 124
  os.replace(pending, Path(output)/'supervisor_complete.json')
  if time.monotonic() >= deadline:
    os.replace(Path(output)/'supervisor_complete.json', Path(output)/'late-provisional-supervisor.json')
    write_json(Path(output)/'late-supervisor.json', dict(exit_code=124, incomplete=True)); return 124
  return receipt['exit_code']


def parent(args):
  require(os.name == 'posix', 'Owned complete CPU audit requires POSIX')
  output = Path(args.output).resolve(); require(not output.exists(), 'New audit attempt directory required')
  output.mkdir(parents=True); deadline = ENTRY_STARTED+SECONDS
  write_json(output/'budget.json', dict(version=VERSION, implementation=IMPLEMENTATION,
      actual_git_sha=args.expected_git, parent_pid=os.getpid(), monotonic_start=ENTRY_STARTED,
      seconds=SECONDS, actual_gpu_devices_used=0,
      scope='imports/authentication/four-host-checkpoint-comparisons/hash/write/full-exit'))
  command = [sys.executable, '-m', 'analysis.dl_rep_observation_audit', '--worker']
  for name in ('primary', 'external_cli_terminal', 'expected_git', 'output'):
    command += ['--'+name.replace('_', '-'), str(getattr(args, name))]
  env = dict(os.environ, DL_REP_OBSERVATION_AUDIT_PARENT=str(os.getpid()), CUDA_VISIBLE_DEVICES='',
      MUJOCO_GL='disable', JAX_PLATFORMS='cpu', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
  with (output/'worker.log').open('x', encoding='utf-8') as log: terminal = run_owned(command, log, deadline, env)
  write_json(output/'worker_exit_code.txt', terminal['exit_code'])
  complete = terminal['exit_code'] == 0 and terminal['timed_out'] is False and time.monotonic() < deadline
  audit_pin = None
  if complete:
    result = read_json(output/'audit.json')
    require(result['version'] == IMPLEMENTATION and result['status'] == 'passed' and result['passed'] is True and
        result['auditor_git_sha'] == args.expected_git and result['complete_paths'] == 2 and
        result['actual_primary_training_updates'] == 256 and result['checkpoint_payloads_decoded'] == 4 and
        result['actual_gpu_devices_used'] == 0, 'Complete independent two-path endpoint audit required')
    audit_pin = sha(output/'audit.json')
  receipt = dict(version=VERSION, implementation=IMPLEMENTATION, actual_git_sha=args.expected_git,
      parent_pid=os.getpid(), **{**terminal, 'exit_code': terminal['exit_code'] if complete or terminal['exit_code'] else 1},
      complete_verified=complete, elapsed_parent_entry_seconds=time.monotonic()-ENTRY_STARTED,
      budget_sha256=sha(output/'budget.json'), worker_exit_sha256=sha(output/'worker_exit_code.txt'),
      audit_sha256=audit_pin, actual_gpu_devices_used=0 if complete else None,
      method_effectiveness_established=False)
  return publish_receipt(output, receipt, deadline)


def verify_completion(directory, external_cli_terminal=None):
  root = Path(directory).resolve(strict=True)
  require(not any((root/name).exists() for name in ('late-supervisor.json',
      'late-provisional-supervisor.json', 'supervisor-receipt-pending.json')), 'Late/provisional CPU audit rejected')
  supervisor = read_json(root/'supervisor_complete.json')
  require(supervisor['version'] == VERSION and supervisor['implementation'] == IMPLEMENTATION and
      supervisor['complete_verified'] is True and supervisor['exit_code'] == 0 and
      supervisor['worker_terminal'] is True and supervisor['timed_out'] is False and
      supervisor['actual_gpu_devices_used'] == 0 and
      0 < supervisor['elapsed_parent_entry_seconds'] <= SECONDS, 'Complete CPU audit supervisor required')
  budget = read_json(root/'budget.json', supervisor['budget_sha256'])
  require(budget['version'] == VERSION and budget['implementation'] == IMPLEMENTATION and
      budget['actual_git_sha'] == supervisor['actual_git_sha'] and budget['parent_pid'] == supervisor['parent_pid'] and
      budget['seconds'] == SECONDS and budget['actual_gpu_devices_used'] == 0 and
      sha(root/'worker_exit_code.txt') == supervisor['worker_exit_sha256'] and
      read_json(root/'worker_exit_code.txt') == 0, 'Complete audit budget/worker exit required')
  result = read_json(root/'audit.json', supervisor['audit_sha256'])
  require(result['version'] == IMPLEMENTATION and result['status'] == 'passed' and result['passed'] is True and
      result['auditor_git_sha'] == supervisor['actual_git_sha'] and result['complete_paths'] == 2 and
      result['actual_primary_training_updates'] == 256 and result['checkpoint_payloads_decoded'] == 4 and
      result['actual_gpu_devices_used'] == result['actual_auditor_optimizer_updates'] == 0,
      'Complete CPU independent result required')
  if external_cli_terminal is not None:
    external = read_json(external_cli_terminal)
    require(external['actual_git_sha'] == supervisor['actual_git_sha'] and
        external['worker_pid'] == supervisor['parent_pid'] and external['exit_code'] == 0 and
        external['worker_terminal'] is True and external['timed_out'] is False and
        0 < external['elapsed_cli_seconds'] <= SECONDS, 'Complete external CPU audit terminal required')
  return dict(audit_sha256=supervisor['audit_sha256'], complete_published_parent_verified=True,
      complete_external_exit_verified=external_cli_terminal is not None,
      endpoint_observation_equivalent=result['endpoint_observation_equivalent'],
      guard_passed=result['guard_passed'], method_effectiveness_established=False)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('primary', 'external-cli-terminal', 'expected-git', 'output'): parser.add_argument('--'+name, required=True)
  parser.add_argument('--worker', action='store_true'); args = parser.parse_args(argv)
  if args.worker: worker(args); return 0
  return parent(args)


if __name__ == '__main__': sys.exit(main())
