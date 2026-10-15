"""DL-r4 independent CPU artifact/physics audit, not an independent model run.

Authenticate the entire successful run/hash chain before trusted pickle decode.
Uses the same official simulator, with old independent restore/snapshot
primitives; never imports the main r4 runner/policy/physics implementation.
"""
import time
ENTRY_STARTED = time.monotonic()
import os
import signal
import sys
SECONDS = 1800
DEADLINE = ENTRY_STARTED+SECONDS
OWNED = set()


def expire(signum, frame):
  for pid in tuple(OWNED):
    try: os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError: pass
  os._exit(124)


if __name__ == '__main__' and os.name == 'posix':
  signal.signal(signal.SIGALRM, expire)
  signal.signal(signal.SIGTERM, expire)
  signal.setitimer(signal.ITIMER_REAL, max(.001, DEADLINE-time.monotonic()))

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['MUJOCO_GL'] = 'disable'

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
import subprocess

import numpy as np
import ml_dtypes  # Trusted BF16 pickle registration only, no JAX.

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import dl_feedback_physics_audit_v1 as independent
from analysis import dl_r4_statistics as statistics

VERSION = 'DL-r4-independent-audit-r1'
SCIENCE = 'DL-r4-utility-r1'
PHYSICAL = 'DL-r4-action-memory-r1'
CHECKPOINT = 'c6a7d92e24669a515bb44693faf73e442d9868f9fc814eec0e22e069aa594d24'
CONFIG = '877f8470cb4f7f45d143e3099fbd3f55a828991afe33779b7b04c4264f6c8377'
PARAMS = 'cffc13c0dcdfbd0c8b0e5c503c9d6251f4b1550c08b85a8742a638655d66d372'
ARMS = ('qq', 'pq', 'pp')
BUNDLE_FIELDS = ('q_after','p_after','q_raw','p_raw','h','qlogits','plogits',
    'qsample','psample','root_key','latent_key','actor_key','original_nj_seed_calls',
    'before_carry_sha256','observation_sha256')
PROOF_FIELDS = {'root_key','latent_key','actor_key','h','selected_logit','latent',
    'action_logit','action_latent','raw','rng_consumption','memory_source','action_source'}


def require(condition, message):
  if not condition: raise ValueError(message)


def check():
  require(time.monotonic() < DEADLINE, 'Independent whole-entry deadline exceeded')


def sha(path):
  h = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for block in iter(lambda: stream.read(1024*1024), b''):
      check(); h.update(block)
  return h.hexdigest()


def read_json(path):
  return statistics.decode(Path(path).read_text(encoding='utf-8'))


def owned_output(command):
  check()
  oldmask = None
  if os.name == 'posix':
    oldmask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM, signal.SIGTERM})
  try:
    child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, start_new_session=os.name == 'posix',
        preexec_fn=(lambda: signal.pthread_sigmask(signal.SIG_SETMASK, oldmask))
            if os.name == 'posix' else None)
    OWNED.add(child.pid)
  finally:
    if oldmask is not None: signal.pthread_sigmask(signal.SIG_SETMASK, oldmask)
  try:
    out, error = child.communicate(timeout=min(30, max(.001, DEADLINE-time.monotonic())))
    require(child.returncode == 0, 'Audit metadata command failed: '+error.decode(errors='replace')[:200])
    check(); return out
  finally:
    if os.name == 'posix':
      try: os.killpg(child.pid, signal.SIGKILL)
      except ProcessLookupError: pass
    elif child.poll() is None: child.kill()
    child.wait(timeout=5); OWNED.discard(child.pid)


def contained(root, relative):
  require(type(relative) is str and relative and not Path(relative).is_absolute(),
      'Run artifact must use a relative path')
  path = (root/relative).resolve(strict=True)
  require(path.is_relative_to(root) and path.is_file(), 'Artifact escapes run root')
  return path


def assert_hash(root, relative, expected):
  require(type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected), 'SHA256 required')
  path = contained(root, relative)
  require(sha(path) == expected, 'Artifact hash differs: '+relative)
  return path


def authenticate(run, expected_git):
  """All metadata/source/branch hashes are checked before any pickle decode."""
  root = Path(run).resolve(strict=True)
  require(re.fullmatch('[0-9a-f]{40}', expected_git or ''), 'Full expected Git SHA required')
  require(owned_output(['git','rev-parse','HEAD']).decode().strip() == expected_git,
      'Auditor checkout must be the expected complete Git version')
  require(not owned_output(['git','status','--porcelain','--untracked-files=all']),
      'Auditor requires clean checkout')
  budget, terminal, worker = (read_json(root/name) for name in
      ('budget.json','supervisor_terminal.json','worker_complete.json'))
  require(all(x['version'] == SCIENCE and x['git_sha'] == expected_git
      for x in (budget, terminal, worker)), 'Run science/Git differs')
  require(budget['seconds'] == 21000 and terminal['worker_returncode'] == 0 and
      terminal['worker_terminal'] is True and terminal['main_generation_complete'] is True and
      terminal['owned_worker_group_cleaned'] is True and
      0 < terminal['elapsed_seconds'] <= 21000 and
      terminal['scientific_support_claimed'] is False, 'No complete successful bounded supervisor')
  assert_hash(root, 'worker_complete.json', terminal['worker_complete_sha256'])
  assert_hash(root, 'worker.log', terminal['worker_log_sha256'])
  assert_hash(root, 'resources.jsonl', terminal['resources_sha256'])
  require(terminal['parent_failure'] is None, 'Parent telemetry/process supervision failed')
  require(worker['complete_episodes'] == 32 and worker['optimizer_updates'] == 0 and
      worker['training_actions'] == 0 and worker['full_model_state_unchanged'] is True and
      worker['source_seconds'] < 1800 and worker['labels_seconds'] < 18000 and
      worker['qualification_seconds'] < 1200, 'Complete finite workload/timing differs')
  for filename, key in (
      ('statistics_input.json','statistics_input_sha256'),
      ('statistics_provisional.json','provisional_statistics_sha256')):
    assert_hash(root, filename, worker[key])
  fixed = worker['fixed_artifact_hashes']
  require(set(fixed) == {'calibration_frozen.json','model_authenticated.json',
      'runtime_cuda_egl.json','budget.json','process.json'}, 'Fixed artifact manifest differs')
  for filename, digest in fixed.items(): assert_hash(root,filename,digest)
  source_files = budget['source_files']
  required = {'analysis/dl_r4_statistics.py','scripts/dl_r4_utility.py',
      'scripts/dl_r4_policy.py','scripts/dl_r4_physics.py','docs/DL_R4_UTILITY.md'}
  require(required <= source_files.keys(), 'Missing source file authentication')
  names = owned_output(['git','ls-files']).decode().splitlines()
  expected_names = {name for name in names if name.endswith(('.py','.yaml','.yml','.lock')) or
      name in ('AGENTS.md','docs/DL_R4_UTILITY.md')}
  require(set(source_files) == expected_names, 'Complete executable/config source manifest required')
  for name, digest in source_files.items():
    require(type(name) is str and not Path(name).is_absolute() and
        (ROOT/name).resolve().is_relative_to(ROOT), 'Unsafe source path')
    blob = owned_output(['git','show',expected_git+':'+name])
    require(hashlib.sha256(blob).hexdigest() == digest and (ROOT/name).read_bytes() == blob,
        'Actual source/blob binding differs: '+name)
  model = read_json(root/'model_authenticated.json')
  metadata = model['metadata']
  require(metadata['checkpoint_sha256'] == CHECKPOINT and metadata['config_sha256'] == CONFIG and
      metadata['params_sha256'] == PARAMS and
      budget['input_pins']['checkpoint_sha256'] == CHECKPOINT and
      budget['input_pins']['config_sha256'] == CONFIG, 'N1M provenance differs')
  resource = model['resource']
  require(resource['physical_gpu'] == 4 and resource['logical_server'] == 'sv3' and
      resource['compute_uuid'] == resource['egl_uuid'] == resource['gpu_uuid'],
      'Recorded assigned CUDA/EGL mapping differs')
  process = read_json(root/'process.json')
  require(process['parent_pid'] == budget['parent_pid'] and
      process['worker_pid'] == process['worker_pgid'], 'Owned parent/worker process chain differs')
  for runtime in (read_json(root/'runtime_cuda_egl.json'),worker['runtime_final']):
    require(runtime['physical_gpu'] == 4 and runtime['pid'] == process['worker_pid'] and
        runtime['compute_uuid'] == [resource['gpu_uuid']] and runtime['graphics_verified'] is True and
        runtime['live_state']['gpu_uuid'] == resource['gpu_uuid'], 'Runtime CUDA/EGL evidence differs')
  telemetry = [read_json_line for read_json_line in
      (statistics.decode(line) for line in (root/'resources.jsonl').read_text(encoding='utf-8').splitlines())]
  require(telemetry and all(x['physical_gpu'] == 4 and x['gpu_uuid'] == resource['gpu_uuid'] and
      all(p['pid'] == process['worker_pid'] for p in x['processes']) for x in telemetry),
      'Recorded telemetry includes missing mapping or unknown process')
  policy_receipt = worker['policy_receipt']
  require(policy_receipt['version'] == SCIENCE and policy_receipt['key_namespace'] == 'DL_R4_U1' and
      policy_receipt['scientific_reads_current_params'] is True and
      policy_receipt['full_state_and_counters_unchanged'] is True and
      policy_receipt['public_rng_not_consumed_by_private_calls'] is True and
      policy_receipt['current_original_nj_seed_calls'] == 2 and
      policy_receipt['derived_nj_seed_calls'] == policy_receipt['h1_nj_seed_calls'] == 0 and
      policy_receipt['first_is_host_assembly_no_policy_or_random_call'] is True and
      policy_receipt['current_full_state_sha256'] == metadata['full_state_sha256'],
      'Actual frozen policy receipt differs')
  payload, seal = read_json(root/'statistics_input.json'), read_json(root/'calibration_frozen.json')
  statistics.validate(payload)
  require(seal['version'] == SCIENCE and seal['held_episodes_generated'] == 0 and
      seal['calibration_episodes'] == payload['episodes'][:8] and
      seal['frozen_models'] == payload['frozen_models'], 'Calibration seal payload differs')
  entries = worker['evidence']
  require(type(entries) is list and len(entries) == 32 and seal['calibration_evidence'] == entries[:8],
      'Exactly32 episode evidence and pre-held cal8 seal required')
  manifests, artifacts = [], {}
  for episode_id, entry in enumerate(entries):
    expected = f'episode_{episode_id:03}/complete.json'
    require(entry['path'] == expected, 'Episode evidence ordering differs')
    path = assert_hash(root, expected, entry['sha256'])
    receipt = read_json(path)
    require(receipt['version'] == SCIENCE and receipt['episode_id'] == episode_id and
        receipt['model_full_state_unchanged'] is True, 'Episode contract differs')
    require(receipt['episode_started_monotonic'] <= receipt['episode_completed_monotonic'],
        'Episode monotonic times differ')
    require(budget['monotonic_deadline']-21000 <= receipt['episode_started_monotonic'] <=
        receipt['episode_completed_monotonic'] < budget['monotonic_deadline'],
        'Episode falls outside the whole-entry deadline')
    if episode_id:
      require(manifests[-1]['episode_completed_monotonic'] <= receipt['episode_started_monotonic'],
          'Episode execution order differs')
    epdir = f'episode_{episode_id:03}/'
    for field, expected_name in (('source','source.pkl'), ('signals','signals.pkl')):
      record = receipt[field]
      require(record['path'] == expected_name, 'Source/signal artifact naming differs')
      relative = epdir+expected_name
      artifacts[relative] = (assert_hash(root, relative, record['sha256']), record['sha256'])
    top2, selected = statistics.selected_positions(payload['episodes'][episode_id]['candidates'])
    expected_branches = {(t, pair, arm) for t in selected for pair in range(16) for arm in ARMS}
    expected_branches |= {(t, 0, 'qq_null') for t in selected}
    oracles = receipt['branch_public_oracles']
    require([x['path'] for x in oracles] == [f't{t:03}_branch_public_oracle.pkl' for t in selected],
        'Actual selected private-key public oracle coverage differs')
    for oracle in oracles:
      relative = epdir+oracle['path']
      artifacts[relative] = (assert_hash(root,relative,oracle['sha256']),oracle['sha256'])
    seen = set()
    for branch in receipt['branches']:
      identity = (branch['position'], branch['pair'], branch['arm'])
      require(identity in expected_branches and identity not in seen, 'Branch coverage/duplicate differs')
      seen.add(identity)
      t, pair, arm = identity
      expected_name = f't{t:03}_pair{pair:02}_{arm}.pkl'
      require(branch['path'] == expected_name, 'Branch artifact name differs')
      relative = epdir+expected_name
      artifacts[relative] = (assert_hash(root, relative, branch['sha256']), branch['sha256'])
    require(seen == expected_branches, 'Incomplete three-arm/null coverage')
    manifests.append(receipt)
  require(manifests[7]['episode_completed_monotonic'] <= seal['sealed_monotonic'] <=
      manifests[8]['episode_started_monotonic'], 'Calibration was not sealed before held source')
  events = []
  for line in (root/'worker.log').read_text(encoding='utf-8', errors='replace').splitlines():
    if line.startswith('{'):
      try: event = statistics.decode(line)
      except ValueError: continue
      if event.get('event') == 'episode_complete': events.append(event['episode_id'])
  require(events == list(range(32)), 'Complete chronological episode log required')
  check()
  return root, payload, manifests, artifacts


def trusted_pickle(artifacts, relative):
  path, expected = artifacts[relative]
  check()
  data = path.read_bytes()
  require(hashlib.sha256(data).hexdigest() == expected, 'Artifact changed before trusted decode')
  result = pickle.loads(data)
  check(); return result


def digest_tree(tree):
  """Independent spelling of original source numeric hash protocol."""
  digest = hashlib.sha256()
  def visit(node):
    if isinstance(node, dict):
      digest.update(b'dict')
      for key in sorted(node): digest.update(str(key).encode()); visit(node[key])
    elif isinstance(node, (list, tuple)):
      digest.update(type(node).__name__.encode()+str(len(node)).encode())
      for child in node: visit(child)
    else:
      value = np.ascontiguousarray(node)
      require(not value.dtype.hasobject and (value.dtype.kind in 'SU' or np.isfinite(value).all()),
          'Finite typed hash leaves required')
      digest.update(str(value.dtype).encode()+str(value.shape).encode()); digest.update(value.tobytes())
  visit(tree)
  return digest.hexdigest()


def policy_digest(tree):
  """Independent encoding of the separate public-policy input hash protocol."""
  h = hashlib.sha256()
  def visit(value):
    if isinstance(value,dict):
      h.update(b'dict')
      for key in sorted(value):
        encoded = key.encode(); h.update(str(len(encoded)).encode()+b':'+encoded); visit(value[key])
    elif isinstance(value,(tuple,list)):
      h.update(type(value).__name__.encode()+str(len(value)).encode()+b':')
      for child in value: visit(child)
    else:
      leaf = np.asarray(value)
      require(not leaf.dtype.hasobject and leaf.dtype.kind not in 'SU' and np.isfinite(leaf).all(),
          'Policy input hash requires finite numeric leaves')
      h.update(str(leaf.dtype).encode()+b':'+str(leaf.shape).encode()+b':')
      h.update(np.ascontiguousarray(leaf).tobytes())
  visit(tree); return h.hexdigest()


def same(actual, expected, name):
  require(isinstance(actual, (np.ndarray, np.generic)) and
      isinstance(expected, (np.ndarray, np.generic)) and
      actual.dtype == expected.dtype and actual.shape == expected.shape and
      actual.tobytes() == expected.tobytes(), 'Byte mismatch: '+name)


def branch_keys(episode, position, pair):
  seed = 0x52345531 + ((episode*833+position-64)*16+pair)
  return np.stack([np.random.default_rng([0,0x4556414C,seed,offset]).integers(
      0,4294967295,(2,),np.uint32) for offset in range(100)])


def static_visual(env, expected):
  model = env.physics.model
  blob = model.to_bytes()
  require(isinstance(blob,bytes) and blob and int(model.ncam) > 2,
      'Actual official CPU visual model/camera missing')
  actual = dict(mjb_sha256=hashlib.sha256(blob).hexdigest(),mjb_bytes=len(blob),
      camera=2,camera_name=model.id2name(2,'camera'))
  require(all(expected[name] == value for name,value in actual.items()),
      'Independent static MJB/camera identity differs')
  require(expected['renderer_byte_exact_established'] is False and
      expected['complete_effective_render_inputs_authenticated'] is False,
      'Static model identity cannot establish GL/renderer equivalence')
  return actual


def carry(value, raw=None):
  require(type(value) is tuple and len(value) == 4 and value[0] == {} and value[2] == {} and
      set(value[1]) == {'deter','stoch'} and set(value[3]) == {'action'}, 'Malformed original carry')
  for name in ('deter','stoch'):
    require(type(value[1][name]) is list and len(value[1][name]) == 1 and
        isinstance(value[1][name][0], np.ndarray) and np.isfinite(value[1][name][0]).all(),
        'Malformed finite memory carry')
  require(type(value[3]['action']) is list and len(value[3]['action']) == 1, 'Malformed prevraw')
  independent.array(value[3]['action'][0], np.float32, (12,), 'actual prevraw')
  if raw is not None: same(value[3]['action'][0], raw, 'prevraw equals executed raw')
  return digest_tree(value)


def validate_record(record, *, episode, position, pair, arm):
  require(record['version'] == PHYSICAL and record['implementation'] == 'DL-r4-physics-r1' and
      record['schema'] == 1 and record['action_contract'] == 'production_clip_unify_normalize_v1' and
      record['horizon'] == 100 and record['arm'] == ('qq' if arm == 'qq_null' else arm),
      'Unknown r4 action/memory branch')
  count = record['executed_steps']
  require(type(count) is int and 1 <= count <= 100, 'Complete actual prefix required')
  for name, (dtype, shape) in independent.ARRAYS.items():
    independent.array(record[name], dtype, shape, name)
  same(record['keys'], branch_keys(episode, position, pair), 'actual named roots')
  require(np.array_equal(record['action_executed'], np.arange(100) < count) and
      np.array_equal(record['observation_executed'], np.arange(101) <= count), 'Execution mask gap')
  require(len(record['observations']) == len(record['snapshots']) == count+1 and
      len(record['proofs']) == len(record['after_carries']) == count, 'Missing genuine branch evidence')
  for name in set(independent.ARRAYS)-{'keys','action_executed','observation_executed'}:
    same(record[name][count:], np.zeros_like(record[name][count:]), 'absorbing zeros '+name)
  require(np.all((record['reward_f64'] >= 0) & (record['reward_f64'] <= 1)), 'Reward outside0..1')
  require(record['snapshots'][0]['suite']['step_count'] == position and
      not record['snapshots'][0]['from_dm_done'] and
      not record['snapshots'][0]['suite']['reset_next_step'], 'Nonreset active initial snapshot required')
  bundle = record['current_bundle']
  require(record['current_bundle_sha256'] == digest_tree({k:bundle[k] for k in BUNDLE_FIELDS}),
      'Current bundle typed hash differs')
  require(bundle['before_carry_sha256'] == policy_digest(record['initial_carry']) and
      bundle['observation_sha256'] == policy_digest(record['observations'][0]),
      'Actual current bundle input provenance differs')
  previous = carry(record['initial_carry'])
  for step, (obs, state) in enumerate(zip(record['observations'], record['snapshots'])):
    independent.observation(obs)
    if step == 0: require(not bool(obs['is_first']) and not bool(obs['is_last']), 'Reset initial frame')
    require(state['suite']['step_count'] == position+step, 'Stored absolute clock differs')
  for step in range(count):
    require(previous == record['pre_carry_sha256'][step], 'Broken initial/previous carry chain')
    after, proof, raw = record['after_carries'][step], record['proofs'][step], record['raw'][step]
    require(set(proof) == PROOF_FIELDS, 'Incomplete action/memory proof')
    for name,value in proof.items():
      require(isinstance(value,(np.ndarray,np.generic)) and
          not value.dtype.hasobject and np.isfinite(value).all(), 'Nonfinite numeric proof: '+name)
    for name in ('deter','stoch'):
      same_shape = record['initial_carry'][1][name][0]
      require(after[1][name][0].shape == same_shape.shape and
          after[1][name][0].dtype == same_shape.dtype, 'Memory dtype/shape changed during policy')
    sources = (int(record['arm'][1] == 'p'), int(record['arm'][0] == 'p')) if step == 0 else (0,0)
    for name, value in zip(('memory_source','action_source'), sources):
      independent.array(proof[name], np.int8, (), name)
      require(int(proof[name]) == value, 'Future nonposterior or wrong initial source')
    independent.array(proof['rng_consumption'], np.int32, (), 'NJ call count')
    require(int(proof['rng_consumption']) == 2, 'Normal NJ sample count differs')
    same(proof['root_key'], record['keys'][step], 'proof root')
    same(proof['raw'], raw, 'proof raw')
    same(proof['h'], after[1]['deter'][0], 'outgoing h')
    same(proof['latent'], after[1]['stoch'][0], 'outgoing latent')
    previous = carry(after, raw)
    require(previous == record['after_carry_sha256'][step], 'Broken after carry chain')
    if step == 0:
      a, m = record['arm']
      expected = copy.deepcopy(bundle[m+'_after']); expected[3]['action'][0] = bundle[a+'_raw'].copy()
      require(carry(expected, raw) == previous, 'Wrong crossed memory/actual action assembly')
      for name in ('h','root_key','latent_key','actor_key'):
        same(proof[name], bundle[name], 'first shared '+name)
      for name, field in (('selected_logit',m+'logits'), ('latent',m+'sample'),
          ('action_logit',a+'logits'), ('action_latent',a+'sample'), ('raw',a+'_raw')):
        same(proof[name], bundle[field], 'first source '+name)
    else:
      same(proof['selected_logit'], proof['action_logit'], 'normal same-source logits')
      same(proof['latent'], proof['action_latent'], 'normal same-source latent')
    next_obs = record['observations'][step+1]
    same(record['reward_f32'][step], next_obs['reward'], 'cached reward')
    require(np.float32(record['reward_f64'][step]).tobytes() == next_obs['reward'].tobytes(),
        'F64/F32 reward encoding differs')
    flags = tuple(bool(record[k][step]) for k in ('is_first','is_last','is_terminal'))
    require(flags == tuple(bool(next_obs[k]) for k in ('is_first','is_last','is_terminal')) and
        not flags[0] and record['discount'][step] in (0,1) and
        flags[2] == (record['discount'][step] == 0) and
        (not flags[1] or step == count-1), 'Flags/discount/terminal differ')
  require(carry(record['final_carry'], record['raw'][count-1]) == previous, 'Final carry differs')
  terminal = bool(record['is_terminal'][count-1]); last = bool(record['is_last'][count-1])
  require(record['end_type'] == ('true_terminal' if terminal else 'time_limit' if last else 'window') and
      (count == 100 or terminal), 'Incomplete window not true terminal')
  require(record['future_normal_policy_calls'] == count-1 and
      record['original_current_nj_seed_calls'] == 2 and record['first_assembly_model_calls'] == 0 and
      record['first_assembly_nj_seed_calls'] == 0, 'Model call receipt differs')
  return count


def replay_record(env, record, *, episode, position, pair, arm):
  count = validate_record(record, episode=episode, position=position, pair=pair, arm=arm)
  spec = env.action_spec()
  same(independent.mapping(record['raw'][:count], spec.minimum, spec.maximum),
      record['applied'][:count], 'independent physical action mapping')
  independent.restore(env, record['snapshots'][0])
  # Main r4 acceptance is byte exact; do not fall back to the old1e-8 tolerance.
  def state_equal(saved, done):
    current = independent.snapshot(env, done)
    independent.compare_state(current, saved)  # Finite/task/counter guards.
    same(current['suite']['integration'], saved['suite']['integration'], 'integration exact')
  state_equal(record['snapshots'][0], False)
  for step in range(count):
    check()
    require(not env._reset_next_step, 'Replay crossed reset')
    ts = env.step(record['applied'][step].copy())
    same(np.asarray(ts.reward, np.float64), record['reward_f64'][step], 'official F64 reward')
    same(np.asarray(ts.reward, np.float32), record['reward_f32'][step], 'official F32 encoding')
    same(np.asarray(ts.discount, np.float64), record['discount'][step], 'official discount')
    require(tuple(bool(record[k][step]) for k in ('is_first','is_last','is_terminal')) ==
        (bool(ts.first()), bool(ts.last()), bool(ts.last() and ts.discount == 0)), 'Official flags differ')
    state_equal(record['snapshots'][step+1], bool(ts.last()))
    require(not ts.last() or step == count-1, 'Replay crossed terminal')
    if ts.last() and ts.discount == 1:
      require(step == 99 and env._step_count == env._step_limit, 'Unexpected early time limit')
  return count


def pair_equal(records):
  qq, pq, pp = (records[a] for a in ARMS)
  for other in (pq, pp):
    require(digest_tree(other['initial_carry']) == digest_tree(qq['initial_carry']) and
        digest_tree(other['observations'][0]) == digest_tree(qq['observations'][0]) and
        other['current_bundle_sha256'] == qq['current_bundle_sha256'], 'Unpaired initial model inputs')
    same(other['keys'], qq['keys'], 'paired planned keys')
    require(pickle.dumps(other['snapshots'][0],protocol=5) ==
        pickle.dumps(qq['snapshots'][0],protocol=5), 'Unpaired initial physics')
  require(digest_tree(qq['after_carries'][0][1]) == digest_tree(pq['after_carries'][0][1]),
      'qq/pq current q memory differs')
  for name in ('raw','applied','reward_f64','reward_f32','discount'):
    same(pq[name][0], pp[name][0], 'pq/pp first physical '+name)
  require(pickle.dumps(pq['snapshots'][1],protocol=5) == pickle.dumps(pp['snapshots'][1],protocol=5),
      'pq/pp equal first action gives different physics')


def compare_compact(actual, expected, path='compact'):
  if isinstance(expected, dict):
    require(type(actual) is dict and actual.keys() == expected.keys(), path+' fields differ')
    for key in expected: compare_compact(actual[key], expected[key], path+'.'+key)
  elif isinstance(expected, list):
    require(type(actual) is list and len(actual) == len(expected), path+' count differs')
    for i, (a,b) in enumerate(zip(actual,expected)): compare_compact(a,b,path+f'[{i}]')
  elif type(expected) is float:
    # Independent fsum vs production np.sum: fixed1e-12 absolute arithmetic,
    # not a relaxation of real reward/physics byte acceptance.
    require(type(actual) in (int,float) and math.isfinite(actual) and
        abs(actual-expected) <= 1e-12, path+' numeric differs')
  else:
    require(type(actual) is type(expected) and actual == expected, path+' exact value differs')


def oracle_receipt(oracle, *, seed, counter, root_key):
  for field in ('public_original_policy_called','public_q_raw_byte_exact',
      'public_q_full_after_carry_byte_exact','original_public_root_byte_exact',
      'eval_counters_restored','full_state_and_policycopy_unchanged'):
    require(oracle[field] is True, 'Public policy oracle failed: '+field)
  require(oracle['eval_seed'] == seed and oracle['eval_counter'] == counter and
      oracle['public_policy_calls'] == 1 and
      oracle['source_physics_or_renderer_authenticated'] is False, 'Public oracle domain differs')
  same(oracle['root_key'],root_key,'public oracle actual root')


def audit(run, expected_git):
  root, payload, manifests, artifacts = authenticate(run, expected_git)
  from dm_control import suite
  reconstructed = copy.deepcopy(payload)
  physics_steps, source_steps, branches = 0, 0, 0
  for episode_id, manifest in enumerate(manifests):
    check()
    prefix = f'episode_{episode_id:03}/'
    source = trusted_pickle(artifacts, prefix+'source.pkl')
    signals = trusted_pickle(artifacts, prefix+'signals.pkl')
    require(source['episode_id'] == episode_id and source['env_seed'] == 4096+episode_id,
        'New source namespace differs')
    episode = reconstructed['episodes'][episode_id]
    compare_compact(signals['candidates'], episode['candidates'])
    top, union = statistics.selected_positions(signals['candidates'])
    require(signals['selected'] == list(union) and sorted(signals['top2']) == list(top),
        'Source D-only selection differs')
    a = source['arrays']; snapshots = source['snapshots']
    require(len(snapshots) == 1001 and [x['position'] for x in source['records']] == list(statistics.POSITIONS),
        'Source full episode/candidate coverage differs')
    records_by_position = {x['position']:x for x in source['records']}
    env = suite.load('quadruped','walk',task_kwargs={'random':4096+episode_id})
    try:
      env.reset(); independent.compare_state(independent.snapshot(env,False),snapshots[0])
      static_visual(env,signals['visual_identity'])
      independent.restore(env,snapshots[0])
      spec = env.action_spec()
      independent.array(a['action'],np.float32,(1000,12),'source raw')
      independent.array(a['action_applied'],np.float64,(1000,12),'source applied')
      independent.array(a['reward_f64'],np.float64,(1001,),'source actual F64 reward')
      independent.array(a['reward'],np.float32,(1001,),'source encoded F32 reward')
      require(np.all((a['reward_f64'] >= 0) & (a['reward_f64'] <= 1)), 'Source reward outside0..1')
      same(independent.mapping(a['action'],spec.minimum,spec.maximum),a['action_applied'],'source mapping')
      for step in range(1000):
        check(); ts = env.step(a['action_applied'][step].copy())
        same(np.asarray(ts.reward,np.float32),a['reward'][step+1],'source F32 reward')
        same(np.asarray(ts.reward,np.float64),a['reward_f64'][step+1],'source F64 reward')
        state = independent.snapshot(env,bool(ts.last()))
        independent.compare_state(state,snapshots[step+1])
        same(state['suite']['integration'],snapshots[step+1]['suite']['integration'],'source integration')
        source_steps += 1
      for candidate in episode['candidates']:
        t = candidate['position']; record = records_by_position[t]
        bundle, h1 = signals['bundles'][t]['bundle'], signals['bundles'][t]['h1']
        root_key = np.random.default_rng([0,0x4556414C,4096+episode_id,t]).integers(
            0,4294967295,(2,),np.uint32)
        oracle_receipt(signals['public_oracles'][list(statistics.POSITIONS).index(t)],
            seed=4096+episode_id,counter=t,root_key=root_key)
        same(bundle['root_key'],root_key,'source bundle actual public key')
        same(bundle['q_raw'],a['action'][t],'source actual public q raw')
        require(carry(bundle['q_after']) == record['after_carry_sha256'],
            'Source actual public q after carry differs')
        require(record['eval_seed'] == 4096+episode_id and record['eval_counter'] == t,
            'Source actual eval namespace differs')
        require(carry(record['carry']) == record['carry_sha256'], 'Source carry typed hash differs')
        for name,shape in (('deter',(4096,)),('stoch',(32,32))):
          leaf = record['carry'][1][name][0]
          require(leaf.shape == shape and str(leaf.dtype) == 'bfloat16',
              'Actual N1M BF16 source memory shape/dtype differs')
        for key in ('D','K','e'):
          signal_key = 'Dmode' if key == 'D' else key
          require(float(h1[signal_key]) == candidate[key], 'Candidate model signal differs')
        same(h1['recorded_raw'],a['action'][t],'H1 common recorded raw')
        same(h1['reward_f32'],a['reward'][t+1],'H1 actual next reward')
        same(h1['reward_f64'],a['reward_f64'][t+1],'H1 actual F64 next reward')
        # H1 head/model forwards are not independently executed here.
        require(bool(h1['valid']) and h1['mode_probe'] is True and
            h1['native_policy_sample_is_distinct'] is True, 'H1 target semantics differ')
        require(abs(float(h1['e'])-max(float(h1['u_q'])-float(h1['u_p']),0.)) <= 1e-7,
            'H1 positive support arithmetic differs')
      for row in episode['selected']:
        t = row['position']; row['returns'] = {arm:[] for arm in ARMS}
        public = trusted_pickle(artifacts,prefix+f't{t:03}_branch_public_oracle.pkl')
        seed = 0x52345531+((episode_id*833+t-64)*16)
        oracle_receipt(public,seed=seed,counter=0,root_key=branch_keys(episode_id,t,0)[0])
        for pair in range(16):
          group = {}
          for arm in ARMS:
            branch = trusted_pickle(artifacts,prefix+f't{t:03}_pair{pair:02}_{arm}.pkl')
            require(branch['visual_identity'] == signals['visual_identity'] ==
                branch['visual_identity_after'], 'Source/branch static model identity differs')
            static_visual(env,branch['visual_identity'])
            physics_steps += replay_record(env,branch,episode=episode_id,position=t,pair=pair,arm=arm)
            require(carry(branch['initial_carry']) == records_by_position[t]['carry_sha256'],
                'Branch history differs from actual source')
            require(pickle.dumps(branch['snapshots'][0],protocol=5) == pickle.dumps(snapshots[t],protocol=5),
                'Branch initial snapshot differs from source')
            same(branch['observations'][0]['image'],a['image'][t],'shared source current RGB')
            row['returns'][arm].append(math.fsum(float(x) for x in branch['reward_f64']))
            group[arm] = branch; branches += 1
          pair_equal(group)
          if pair == 0:
            null = trusted_pickle(artifacts,prefix+f't{t:03}_pair00_qq_null.pkl')
            require(null['visual_identity'] == signals['visual_identity'] ==
                null['visual_identity_after'], 'Null/source static model identity differs')
            static_visual(env,null['visual_identity'])
            physics_steps += replay_record(env,null,episode=episode_id,position=t,pair=0,arm='qq_null')
            same(null['raw'][0],group['qq']['raw'][0],'null initial action')
            require(null['after_carry_sha256'][0] == group['qq']['after_carry_sha256'][0],
                'Null initial model carry differs')
            row['qq_null_return'] = math.fsum(float(x) for x in null['reward_f64'])
            row['null_max_abs_step_reward_diff'] = float(max(abs(float(x)-float(y))
                for x,y in zip(null['reward_f64'],group['qq']['reward_f64'])))
            branches += 1
      static_visual(env,signals['visual_identity'])
    finally: env.close()
    print(json.dumps(dict(event='independent_episode',episode_id=episode_id,
        branch_physics_steps=physics_steps,source_physics_steps=source_steps)),flush=True)
  compare_compact(payload, reconstructed)
  selected_count = sum(len(episode['selected']) for episode in payload['episodes'])
  expected_calls = dict(current=448+16*selected_count+branches,
      derived=448+16*selected_count+branches,
      normal=2*(physics_steps-branches), first=2*branches,
      h1=448,oracle=448+selected_count,initial=0)
  actual_calls = read_json(root/'worker_complete.json')['policy_receipt']['calls']
  require(set(actual_calls) == set(expected_calls) and all(
      type(actual_calls[name]) is int and actual_calls[name] == value
      for name,value in expected_calls.items()), 'Whole-object policy call counts differ')
  # All authority/physics checks completed. This reuses the frozen statistics
  # engine; its separate unit checks are not an independent statistics formula.
  result = statistics.analyze(payload,evidence_authenticated=True)
  check()
  return dict(version=VERSION,status='passed',expected_git=expected_git,
      source_physics_steps=source_steps,branch_physics_steps=physics_steps,branches=branches,
      full_integration_byte_exact=True,task_rng_counter_exact=True,
      reward_f64_byte_exact=True,applied_mapping_byte_exact=True,
      compact_fsum_comparison_absolute_tolerance=1e-12,
      trusted_artifacts=len(artifacts),statistics=result,
      policy_call_counts_independently_counted=expected_calls,
      static_model_mjb_camera_identity_verified=True,gl_identity_verified=False,
      model_executed=False,renderer_executed=False,jax_imported='jax' in sys.modules,
      independent_statistical_formula=False,actual_fromdm_wrapper_executed=False,
      same_official_physics_backend=True,
      limitations=['No independent model, D/K/head forward, NJ subkey, rendering or training replay.',
          'Reuses old independently written official restore/snapshot primitives, not a second simulator.',
          'Statistical analysis calls the separately tested main fixed-procedure implementation.',
          'Metadata/hash authentication is provenance checking, not cryptographic authorship or a second experiment.'],
      elapsed_seconds=time.monotonic()-ENTRY_STARTED,deadline_seconds=SECONDS,
      method_effectiveness_established=False,training_authorized=False)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--run',required=True)
  parser.add_argument('--expected-git',required=True)
  parser.add_argument('--output',type=Path,required=True)
  args = parser.parse_args()
  require(os.name == 'posix', 'Actual audit needs POSIX whole-entry deadline')
  require(not args.output.exists(), 'New audit output required')
  result = audit(args.run,args.expected_git)
  require('jax' not in sys.modules,'Independent CPU audit imported JAX')
  result['auditor_sha256'] = sha(__file__)
  result['input_sha256'] = sha(Path(args.run)/'statistics_input.json')
  check()
  args.output.parent.mkdir(parents=True,exist_ok=True)
  temporary = args.output.with_name(args.output.name+'.partial')
  with temporary.open('x',encoding='utf-8') as stream:
    json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')
  check(); temporary.rename(args.output)
  check()


if __name__ == '__main__': main()
