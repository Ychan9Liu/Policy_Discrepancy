"""Independent CPU audit of the actual DL-r5 producer, never a model/render run.

Authenticate the complete successful producer graph before trusted pickle5
decoding. Replay every genuine source/outcome transition and baseline-q shadow
using the official CPU simulator. Cached RGB, NN values and public-policy
receipts are authenticated evidence, not independently recomputed predictions.
"""
import time
ENTRY_STARTED = time.monotonic()
import os
import signal
import sys
SECONDS, OWNED = 1800, set()
DEADLINE = ENTRY_STARTED + SECONDS


def expire(signum, frame):
  for pid in tuple(OWNED):
    try:
      os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
      pass
  os._exit(124)


# Includes imports, metadata helpers, hashing, decoding, physics and output.
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
import importlib.metadata
import io
import json
import math
from pathlib import Path
import pickle
import re
import stat
import subprocess

import numpy as np
import ml_dtypes  # Trusted BF16 dtype registration; no JAX import.

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import dl_feedback_physics_audit_v1 as physics
from analysis import dl_r4_audit as primitive
from analysis import dl_live_statistics as statistics

VERSION = 'DL-r5-independent-audit-r1'
SCIENCE, LIVE, FAMILY = 'DL-live-utility-r1', 'DL-r5-live-r1', 'DL-r5-live-family-r1'
TRAJECTORY, SHADOW = 'DL-r5-live-trajectory-r1', 'DL-r5-live-shadow-r1'
NAMESPACE = 'DL_R5_LIVE_U1'
GPU_UUID = 'GPU-9c070d5a-ac68-4c26-9522-a790ad874b23'
CHECKPOINT, CONFIG, PARAMS = primitive.CHECKPOINT, primitive.CONFIG, primitive.PARAMS
UNIT_FILES = {'shadow_request.pkl', 'shadow_response.pkl', 'pretreatment.pkl',
    'signals_sealed.json', 'assignment.json', 'trajectory.pkl'}
FAMILY_FILES = {'shadow_process.json', 'shadow_closed.json', 'model_authenticated.json',
    'runtime_cuda_egl.json', 'runtime_after.json', 'shadow.stderr.log'}
PROOF_FIELDS = primitive.PROOF_FIELDS
BF16 = np.dtype(ml_dtypes.bfloat16)
require, same = primitive.require, primitive.same


def check():
  require(time.monotonic() < DEADLINE, 'Independent whole-entry1800s deadline exceeded')


def sha_bytes(value):
  return hashlib.sha256(value).hexdigest()


def sha(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for block in iter(lambda: stream.read(1024*1024), b''):
      check()
      digest.update(block)
  return digest.hexdigest()


def decode(value):
  return statistics.decode(value)


def read_json(path):
  return decode(Path(path).read_text(encoding='utf-8'))


def command_output(command, payload=None):
  check()
  previous = None
  if os.name == 'posix':
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM, signal.SIGTERM})
  try:
    child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE if payload is not None else None,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=os.name == 'posix', preexec_fn=(
            lambda: signal.pthread_sigmask(signal.SIG_SETMASK, previous)) if previous is not None else None)
    OWNED.add(child.pid)
  finally:
    if previous is not None:
      signal.pthread_sigmask(signal.SIG_SETMASK, previous)
  try:
    out, error = child.communicate(input=payload, timeout=min(30, max(.001, DEADLINE-time.monotonic())))
    require(child.returncode == 0, 'Audit metadata command failed: '+error.decode(errors='replace')[:256])
    check()
    return out
  finally:
    if os.name == 'posix':
      try:
        os.killpg(child.pid, signal.SIGKILL)
      except ProcessLookupError:
        pass
    elif child.poll() is None:
      child.kill()
    child.wait(timeout=5)
    OWNED.discard(child.pid)
    child.stdout.close()
    child.stderr.close()
    if child.stdin is not None:
      child.stdin.close()


def inventory(root):
  """All regular run files, with no symlink/escape/special-file exceptions."""
  root = Path(root)
  require(root.is_dir() and not root.is_symlink(), 'Regular run directory required')
  files, dirs = {}, []
  for path in sorted(root.rglob('*')):
    check()
    require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), 'Run link/escape forbidden')
    mode = path.stat().st_mode
    relative = path.relative_to(root).as_posix()
    if stat.S_ISDIR(mode):
      dirs.append(relative)
    else:
      require(stat.S_ISREG(mode), 'Run special file forbidden')
      files[relative] = dict(path=relative, bytes=path.stat().st_size, sha256=sha(path))
  return dict(files=files, directories=dirs)


def pin(root, relative, entry, stock, *, absolute=False):
  require(type(entry) is dict and type(relative) is str and relative in stock['files'],
      'Missing artifact: '+str(relative))
  expected_path = str(root/relative) if absolute else Path(relative).name
  require(entry.get('path') == expected_path, 'Artifact path differs: '+relative)
  actual = stock['files'][relative]
  require(entry.get('sha256') == actual['sha256'] and
      ('bytes' not in entry or type(entry['bytes']) is int and entry['bytes'] == actual['bytes']),
      'Artifact hash/length differs: '+relative)
  return root/relative


def full_digest(value):
  """Independent implementation of the producer's typed full-tree protocol."""
  digest = hashlib.sha256()
  def visit(item):
    if type(item) is dict:
      require(all(type(k) is str for k in item), 'String typed-tree keys required')
      digest.update(b'dict')
      for key in sorted(item):
        digest.update(json.dumps(key).encode())
        visit(item[key])
    elif type(item) in (tuple, list):
      digest.update(type(item).__name__.encode()+str(len(item)).encode())
      for child in item:
        visit(child)
    elif item is None or type(item) in (bool, int, float, str):
      require(type(item) is not float or math.isfinite(item), 'Nonfinite typed scalar')
      digest.update(type(item).__name__.encode()+json.dumps(item).encode())
    else:
      array = np.asarray(item)
      require((array.dtype.kind in 'buifc' or array.dtype == BF16) and np.isfinite(array).all(),
          'Invalid typed-tree array')
      digest.update(json.dumps([array.dtype.str, str(array.dtype)]).encode())
      digest.update(json.dumps(array.shape).encode()+array.tobytes(order='C'))
  visit(value)
  return digest.hexdigest()


def sealed_digest(value, field):
  digest = full_digest({k: v for k, v in value.items() if k != field})
  require(value.get(field) == digest, 'Typed seal/bundle hash differs: '+field)
  return digest


def array_hash(value):
  value = np.asarray(value)
  return sha_bytes(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes())


def snapshot_hash(value):
  require(type(value) is dict and set(value) == {'suite', 'from_dm_done'}, 'Full wrapper snapshot required')
  state = value['suite']
  require(type(state) is dict and set(state) == {'integration','step_count','reset_next_step','task'},
      'Full suite/task snapshot required')
  physics.validate_snapshot(value, len(state['integration']))
  meta = [array_hash(state['integration']), state['step_count'], bool(state['reset_next_step']),
      bool(value['from_dm_done']), sha_bytes(pickle.dumps(state['task'], protocol=5))]
  return sha_bytes(json.dumps(meta, separators=(',', ':')).encode())


def state_equal(actual, expected):
  physics.compare_state(actual, expected)
  same(actual['suite']['integration'], expected['suite']['integration'], 'Exact full integration')
  require(snapshot_hash(actual) == snapshot_hash(expected), 'Exact complete snapshot differs')


def plan(unit):
  require(type(unit) is int and 0 <= unit < 512, 'Global unit0..511 required')
  j = int(np.random.default_rng([20261030, 0x54494D45, unit]).integers(0, 14))
  return dict(unit=unit, family=unit//16, replicate=unit%16, env_seed=8192+unit//16,
      eval_seed=0x52350000+unit, t_index=j, t=64+64*j)


def root_key(unit, step):
  return np.random.default_rng([0, 0x4556414C, 0x52350000+unit, step]).integers(
      0, np.iinfo(np.uint32).max, (2,), np.uint32)


def label_receipt(unit, seal):
  index = int(np.random.default_rng([20261030, 0x41524D53, unit]).integers(0, 4))
  return dict(version=LIVE,key_namespace=NAMESPACE,unit=unit,label_index=index,
      label=statistics.LABELS[index],pretreatment_sha256=seal)


def carry(value, raw=None, *, initial=False):
  digest = primitive.carry(value, raw)
  physics.array(value[1]['deter'][0], BF16, (4096,), 'N1M deter')
  z = value[1]['stoch'][0]
  physics.array(z, BF16, (32,32), 'N1M stoch')
  if initial:
    require(not np.any(value[1]['deter'][0]) and not np.any(z) and
        not np.any(value[3]['action'][0]), 'Genuine zero initial carry required')
  else:
    require(np.all((z == 0) | (z == 1)) and np.all(np.asarray(z,np.float32).sum(-1) == 1),
        'Actual posterior/prior categorical sample is not one-hot')
  return digest


def proof(value, unit, step, raw, after, arm):
  require(type(value) is dict and set(value) == PROOF_FIELDS, 'Exact numeric proof schema required')
  for name in ('root_key','latent_key','actor_key'):
    physics.array(value[name], np.uint32, (2,), name)
  same(value['root_key'],root_key(unit,step),'Continuous absolute policy key')
  carry(after,raw)
  for name,dtype,shape in (('h',BF16,(4096,)),('raw',np.float32,(12,)),
      ('rng_consumption',np.int32,()),('memory_source',np.int8,()),('action_source',np.int8,())):
    physics.array(value[name],dtype,shape,name)
  require(int(value['rng_consumption']) == 2 and
      (int(value['memory_source']),int(value['action_source'])) ==
      {'qq':(0,0),'pq':(0,1),'pp':(1,1)}[arm], 'Proof action/memory/NJ trace differs')
  same(value['h'],after[1]['deter'][0],'Proof/actual h')
  same(value['raw'],raw,'Proof/actual raw')
  same(value['latent'],after[1]['stoch'][0],'Proof/actual memory sample')
  for name in ('selected_logit','latent','action_logit','action_latent'):
    physics.array(value[name],BF16,(32,32),name)
  if arm != 'pq':
    same(value['selected_logit'],value['action_logit'],'Same-source logits')
    same(value['latent'],value['action_latent'],'Same-source samples')


def validate_bundle(record, unit, model_state):
  t = plan(unit)['t']
  outer, inner = record['bundle'], record['bundle']['kernel_bundle']
  sealed_digest(outer,'bundle_sha256'); sealed_digest(inner,'bundle_sha256')
  require(outer['version'] == LIVE and outer['key_namespace'] == NAMESPACE and
      outer['unit'] == unit and outer['step'] == t and outer['eval_seed'] == 0x52350000+unit and
      outer['kernel_version'] == inner['version'] == 'DL-r4-utility-r1' and
      outer['kernel_namespace'] == inner['key_namespace'] == 'DL_R4_U1' and
      inner['current_full_state_sha256'] == model_state, 'Current/kernel provenance differs')
  before = record['intervention_before_carry']
  require(inner['before_carry_sha256'] == primitive.policy_digest(before) == outer['before_carry_sha256'] and
      inner['observation_sha256'] == primitive.policy_digest(record['observations'][t]) ==
      outer['observation_sha256'], 'Current actual input provenance differs')
  for key,value in dict(original_nj_seed_calls=2,normal_policy_calls=1,derived_nj_seed_calls=0,
      extra_observe_calls=0).items():
    require(type(inner[key]) is int and inner[key] == value, 'Current call count differs')
  for name in ('Dmode','K','q_raw','q_after','root_key'):
    require(full_digest(outer[name]) == full_digest(inner[name]), 'Outer/kernel copy differs: '+name)
  same(outer['r_t_f32'],record['observations'][t]['reward'],'Current observed reward')
  same(inner['root_key'],root_key(unit,t),'Current absolute key')
  for key in ('root_key','latent_key','actor_key'):
    physics.array(inner[key],np.uint32,(2,),key)
  physics.array(inner['h'],BF16,(4096,),'Current h')
  for source in ('q','p'):
    physics.array(inner[source+'_raw'],np.float32,(12,),'Current raw')
    carry(inner[source+'_after'],inner[source+'_raw'])
    same(inner['h'],inner[source+'_after'][1]['deter'][0],'Shared current h')
    same(inner[source+'sample'],inner[source+'_after'][1]['stoch'][0],'Current latent')
    for suffix in ('logits','sample','mode'):
      physics.array(inner[source+suffix],BF16,(32,32),'Current distribution/sample/mode')
  for key in ('D','Dmode','K','sample_D'):
    value=np.asarray(inner[key]); require(value.shape == () and np.isfinite(value) and value >= 0,
        'Invalid current signal')
  same(inner['D'],inner['Dmode'],'D is frozen mode signal')
  return outer,inner


def request_hash(request):
  state_sha = snapshot_hash(request['actual_full_snapshot'])
  require(state_sha == request['source_snapshot_sha256'], 'Shadow request snapshot hash differs')
  meta=[SHADOW,request['unit'],request['env_seed'],request['request_id'],request['bundle_sha256'],
      state_sha,request['static_mjb'],array_hash(request['raw_q'])]
  return sha_bytes(json.dumps(meta,sort_keys=True,separators=(',',':')).encode())


def response_hash(response):
  fields=('version','unit','env_seed','request_id','request_sha256','bundle_sha256',
      'source_snapshot_sha256','static_mjb_sha256','official_steps','render_calls','model_calls',
      'live_environment_touched')
  meta={key:response[key] for key in fields}
  for key in ('raw_q','applied','reward_f64','reward_f32','discount','is_first','is_last','is_terminal'):
    meta[key]=array_hash(response[key])
  meta['next_snapshot_sha256']=snapshot_hash(response['next_snapshot'])
  require(meta['next_snapshot_sha256'] == response['next_snapshot_sha256'],'Shadow poststate hash differs')
  return sha_bytes(json.dumps(meta,sort_keys=True,separators=(',',':')).encode())


def validate_pretreatment(record, request, response, seal, signals, assignment, unit):
  outer,inner=record['bundle'],record['bundle']['kernel_bundle']; t=plan(unit)['t']
  require(request['version'] == response['version'] == SHADOW and
      request['unit'] == response['unit'] == unit and request['env_seed'] == response['env_seed'] == 8192+unit//16 and
      request['request_id'] == response['request_id'] == f'unit{unit:03d}' and
      request['bundle_sha256'] == response['bundle_sha256'] == outer['bundle_sha256'] and
      request['static_mjb'] == record['static_model'] and
      response['static_mjb_sha256'] == record['static_model']['mjb_sha256'], 'Shadow source identity differs')
  require(request_hash(request) == request['request_sha256'] == response['request_sha256'] and
      response_hash(response) == response['response_sha256'], 'Shadow wire object hashes differ')
  state_equal(request['actual_full_snapshot'],record['snapshots'][t])
  require(request['source_snapshot_sha256'] == response['source_snapshot_sha256'] ==
      snapshot_hash(record['snapshots'][t]), 'Shadow source linkage differs')
  same(request['raw_q'],inner['q_raw'],'Shadow current q raw')
  same(response['raw_q'],request['raw_q'],'Shadow request/response raw')
  require(response['official_steps'] == 1 and response['render_calls'] == response['model_calls'] == 0 and
      response['live_environment_touched'] is False, 'Shadow bounded CPU-only evidence differs')
  for key,dtype,shape in (('raw_q',np.float32,(12,)),('applied',np.float64,(12,)),
      ('reward_f64',np.float64,()),('reward_f32',np.float32,()),('discount',np.float64,()),
      ('is_first',np.bool_,()),('is_last',np.bool_,()),('is_terminal',np.bool_,())):
    value=response[key]
    require(isinstance(value,(np.ndarray,np.generic)) and value.dtype == np.dtype(dtype) and
        value.shape == shape and np.isfinite(value).all(),'Shadow numeric field differs: '+key)
  same(response['reward_f32'],np.asarray(response['reward_f64'],np.float32),'Shadow F64/F32 encoding')
  sealed_digest(seal,'pretreatment_sha256')
  require(full_digest(record['seal']) == full_digest(seal) and seal['version'] == LIVE and
      seal['key_namespace'] == NAMESPACE and seal['unit'] == unit and seal['label_revealed'] is False and
      seal['bundle_sha256'] == outer['bundle_sha256'] and
      seal['source_snapshot_sha256'] == request['source_snapshot_sha256'] and
      seal['shadow_request_sha256'] == request['request_sha256'], 'Typed pretreatment identity differs')
  same(seal['r_t'],record['reward_f64'][t-1],'Actual preceding F64 baseline')
  same(seal['r_t_f32'],record['observations'][t]['reward'],'Observed baseline F32')
  same(np.asarray(seal['r_t'],np.float32),seal['r_t_f32'],'Baseline F64/F32 encoding')
  for key in ('Dmode','K'):
    same(seal[key],inner[key],'Pretreatment/current signal '+key)
  score=seal['score_h1']
  require(score['source_bundle_sha256'] == inner['bundle_sha256'] and score['mode_probe'] is True and
      score['native_policy_sample_is_distinct'] is True and score['nj_seed_calls'] == score['actual_policy_calls'] == 0 and
      np.asarray(score['valid']).shape == () and bool(score['valid']), 'Original H1 provenance differs')
  for key in ('D','Dmode','K','sample_D'):
    same(score[key],inner[key],'H1/current signal')
  same(score['recorded_raw'],inner['q_raw'],'H1 common recorded baseline-q raw')
  same(score['reward_f64'],response['reward_f64'],'H1 shadow F64 next reward')
  same(score['reward_f32'],response['reward_f32'],'H1 shadow F32 target')
  same(score['e'],seal['e'],'H1/sealed e')
  require(np.asarray(seal['e']).shape == () and np.isfinite(seal['e']) and 0 <= seal['e'] <= 1,
      'Invalid finite clipped H1 support')
  expected=label_receipt(unit,seal['pretreatment_sha256'])
  require(record['label_receipt'] == assignment['label_receipt'] == expected and
      signals['pretreatment_sha256'] == assignment['pretreatment_sha256'] == seal['pretreatment_sha256'] and
      signals['bundle_sha256'] == outer['bundle_sha256'] and
      signals['source_snapshot_sha256'] == request['source_snapshot_sha256'], 'Seal-before-label hash binding differs')
  for key,source in (('D','Dmode'),('K','K'),('e','e'),('r_t','r_t'),('r_t_f32','r_t_f32')):
    require(type(signals[key]) is float and signals[key].hex() == float(seal[source]).hex(),
        'JSON/typed pretreatment value differs: '+key)
  return expected['label']


def validate_record(record, complete, request, response, seal, signals, assignment, unit, model_state):
  p=plan(unit); t=p['t']; n=record['executed_steps']
  require(record['version'] == TRAJECTORY and record['unit'] == unit and record['plan'] == p and
      type(n) is int and t+1 <= n <= t+100 and record['outcome_steps'] == n-t,
      'Actual trajectory identity/count differs')
  shapes={'raw':(np.float32,(n,12)),'applied':(np.float64,(n,12)),
      'keys':(np.uint32,(n,2)),'reward_f64':(np.float64,(n,)),
      'reward_f32':(np.float32,(n,)),'discount':(np.float64,(n,))}
  shapes.update({k:(np.bool_,(n,)) for k in ('is_first','is_last','is_terminal')})
  for key,(dtype,shape) in shapes.items(): physics.array(record[key],dtype,shape,key)
  require(len(record['observations']) == len(record['snapshots']) == n+1 and
      len(record['outcome_proofs']) == len(record['after_carries']) == n-t and
      len(record['pre_carry_sha256']) == len(record['after_carry_sha256']) == n,
      'Actual source/outcome/carry coverage differs')
  for key in ('pre_carry_sha256','after_carry_sha256'):
    require(all(type(v) is str and re.fullmatch('[0-9a-f]{64}',v) for v in record[key]),'Invalid carry digest')
  require(record['pre_carry_sha256'][1:] == record['after_carry_sha256'][:-1], 'Carry hash chain discontinuity')
  require(carry(record['initial_carry'],initial=True) == record['pre_carry_sha256'][0] and
      carry(record['intervention_before_carry'],record['raw'][t-1]) == record['pre_carry_sha256'][t] and
      carry(record['final_carry'],record['raw'][-1]) == record['after_carry_sha256'][-1],
      'Saved full carry/hash chain linkage differs')
  initial=record['observations'][0]
  physics.observation(initial)
  require(bool(initial['is_first']) and not bool(initial['is_last']) and not bool(initial['is_terminal']) and
      initial['reward'] == 0, 'Genuine reset observation required')
  for index,(obs,saved) in enumerate(zip(record['observations'],record['snapshots'])):
    physics.observation(obs); snapshot_hash(saved)
    require(saved['suite']['step_count'] == index and saved['from_dm_done'] == bool(obs['is_last']) and
        saved['suite']['reset_next_step'] == bool(obs['is_last']), 'Absolute source clock/reset mismatch')
    if index:
      same(obs['reward'],record['reward_f32'][index-1],'Recorded genuine observation reward')
      same(np.asarray(record['reward_f64'][index-1],np.float32),obs['reward'],'Real F64/F32 reward encoding')
      require(not bool(obs['is_first']) and bool(obs['is_terminal']) == (record['discount'][index-1] == 0) and
          record['discount'][index-1] in (0,1) and
          (not bool(obs['is_last']) or index == n and index > t and bool(obs['is_terminal'])),
          'Premature prefix/reset/time-limit/terminal differs')
      for name in ('is_first','is_last','is_terminal'):
        require(bool(obs[name]) == bool(record[name][index-1]),'Stored flags/observation differ')
  terminal=bool(record['is_terminal'][-1])
  require((n == t+100 or terminal) and record['end_type'] == ('true_terminal' if terminal else 'window') and
      record['live_restore_calls'] == record['extra_live_render_calls'] == 0,
      'Incomplete/restore-based live outcome')
  outer,inner=validate_bundle(record,unit,model_state)
  label=validate_pretreatment(record,request,response,seal,signals,assignment,unit)
  arm='qq' if label in ('qqA','qqB') else label
  for step in range(n):
    same(record['keys'][step],root_key(unit,step),'Actual absolute policy root')
    if step >= t:
      after=record['after_carries'][step-t]; receipt=record['outcome_proofs'][step-t]
      proof(receipt,unit,step,record['raw'][step],after,arm if step == t else 'qq')
      require(carry(after,record['raw'][step]) == record['after_carry_sha256'][step],
          'Actual full outgoing carry hash differs')
  first=record['after_carries'][0]; fp=record['outcome_proofs'][0]
  memory,action=('p' if arm[1] == 'p' else 'q'),arm[0]
  expected=copy.deepcopy(inner[memory+'_after']); expected[3]['action'][0]=inner[action+'_raw'].copy()
  require(carry(first,record['raw'][t]) == carry(expected,inner[action+'_raw']), 'First action/memory assembly differs')
  same(record['raw'][t],inner[action+'_raw'],'First actual action source')
  for key in ('root_key','latent_key','actor_key','h'): same(fp[key],inner[key],'Current/first proof '+key)
  for key,field in (('selected_logit',memory+'logits'),('latent',memory+'sample'),
      ('action_logit',action+'logits'),('action_latent',action+'sample')):
    same(fp[key],inner[field],'Current/first latent/logit '+key)
  if arm == 'qq':
    for key in ('applied','reward_f64','reward_f32','discount','is_first','is_last','is_terminal'):
      same(record[key][t],response[key],'Actual qq/shadow first transition '+key)
    state_equal(record['snapshots'][t+1],response['next_snapshot'])
  last_normal=n-1 if n-1 != t else t-1
  require(record['last_normal_oracle_step'] == last_normal and len(record['oracles']) == 3,
      'Actual public oracle coverage differs')
  for oracle,step in zip(record['oracles'],(0,t,last_normal)):
    require(oracle['version'] == LIVE and oracle['key_namespace'] == NAMESPACE and oracle['unit'] == unit and
        oracle['step'] == step and oracle['eval_seed'] == p['eval_seed'] and oracle['public_policy_calls'] == 1 and
        all(oracle[key] is True for key in ('public_q_raw_byte_exact','public_q_full_after_carry_byte_exact',
            'original_counters_restored')), 'Actual public policy receipt differs')
    same(oracle['root_key'],root_key(unit,step),'Public/private absolute root')
    require(oracle['observation_sha256'] == primitive.policy_digest(record['observations'][step]),
        'Public oracle actual RGB/observation provenance differs')
    before=record['initial_carry'] if step == 0 else record['intervention_before_carry'] if step == t else (
        record['after_carries'][step-t-1] if step > t else None)
    after=inner['q_after'] if step == t else (record['after_carries'][step-t] if step > t else
        record['intervention_before_carry'] if step == t-1 else None)
    if before is not None: require(oracle['before_carry_sha256'] == primitive.policy_digest(before),'Oracle before carry differs')
    if after is not None: require(oracle['after_carry_sha256'] == primitive.policy_digest(after),'Oracle after carry differs')
  event=3*unit+int(unit >= 128)
  require(signals['event'] == event and assignment['event'] == event+1 and complete['event'] == event+2 and
      signals['version'] == assignment['version'] == complete['version'] == FAMILY and
      signals['unit'] == assignment['unit'] == complete['unit'] == unit and complete['plan'] == p and
      complete['started_monotonic'] <= signals['monotonic'] < assignment['monotonic'] < complete['monotonic'],
      'Actual seal/assignment/outcome event or monotonic ordering differs')
  row=dict(family_id=unit//16,unit_id=unit%16,env_seed=p['env_seed'],policy_seed=p['eval_seed'],position=t,label=label,
      pretreatment={key:signals[key] for key in ('D','K','e','r_t','r_t_f32')},
      outcome=dict(reward_sum=math.fsum(float(x) for x in record['reward_f64'][t:])),
      chronology=dict(signals_sealed=event,assignment=event+1,outcome_complete=event+2))
  require(complete['unit_row'] == row and complete['executed_steps'] == n and complete['outcome_steps'] == n-t and
      complete['policy_calls'] == n and complete['original_nj_seed_calls'] == 2*n and
      complete['derived_nj_seed_calls'] == complete['first_assembly_model_calls'] == complete['live_restore_calls'] ==
      complete['extra_live_render_calls'] == 0 and complete['public_oracle_calls'] == 3 and
      complete['oracle_steps'] == [0,t,last_normal] and complete['renderer_byte_exact_claimed'] is False,
      'Compact statistics/counts differ from full actual evidence')
  return row


def static_model(env):
  binary=env.physics.model.to_bytes()
  return dict(mjb_sha256=sha_bytes(binary),mjb_bytes=len(binary),camera=2,
      camera_name=env.physics.model.id2name(2,'camera'))


def official_transition(env, applied, reward, encoded, discount, flags, saved):
  require(not env._reset_next_step,'Physics replay crossed last/reset')
  ts=env.step(applied.copy())
  actual_reward,actual_discount=np.asarray(ts.reward),np.asarray(ts.discount)
  # Deliberately no cast hiding the official reward's native dtype.
  require(actual_reward.dtype == actual_discount.dtype == np.float64 and
      actual_reward.shape == actual_discount.shape == (), 'Official native F64 reward/discount required')
  same(actual_reward,reward,'Official F64 reward byte replay')
  same(np.asarray(actual_reward,np.float32),encoded,'Official F32 reward encoding replay')
  same(actual_discount,discount,'Official F64 discount replay')
  require((bool(ts.first()),bool(ts.last()),bool(ts.last() and ts.discount == 0)) == flags,
      'Official transition flags differ')
  state_equal(physics.snapshot(env,ts.last()),saved)


def replay(record, request, response):
  """Actual seeded source reset/prefix/outcome plus separate q-shadow transition."""
  from dm_control import suite
  check()
  env=suite.load('quadruped','walk',task_kwargs={'random':record['plan']['env_seed']})
  try:
    ts=env.reset()
    require(ts.first() and static_model(env) == record['static_model'], 'Fresh official environment/MJB differs')
    state_equal(physics.snapshot(env,False),record['snapshots'][0])
    spec=env.action_spec()
    require(spec.dtype == np.float64 and spec.shape == (12,), 'Official action spec differs')
    same(physics.mapping(record['raw'],spec.minimum,spec.maximum),record['applied'],'Independent complete raw/applied mapping')
    for step in range(record['executed_steps']):
      check()
      official_transition(env,record['applied'][step],record['reward_f64'][step],record['reward_f32'][step],
          record['discount'][step],tuple(bool(record[k][step]) for k in ('is_first','is_last','is_terminal')),
          record['snapshots'][step+1])
    # This is an offline independent CPU reconstruction, never a live restore.
    physics.restore(env,request['actual_full_snapshot'])
    state_equal(physics.snapshot(env,False),request['actual_full_snapshot'])
    same(physics.mapping(request['raw_q'],spec.minimum,spec.maximum),response['applied'],
        'Independent shadow raw/applied mapping')
    official_transition(env,response['applied'],response['reward_f64'],response['reward_f32'],response['discount'],
        tuple(bool(response[k]) for k in ('is_first','is_last','is_terminal')),response['next_snapshot'])
    require(static_model(env) == record['static_model'],'Replay static MJB changed')
  finally:
    env.close()
  return dict(source_steps=record['plan']['t'],outcome_steps=record['outcome_steps'],shadow_steps=1,
      terminal_absorbing_steps=100-record['outcome_steps'])


def source_manifest(git):
  require(type(git) is str and re.fullmatch('[0-9a-f]{40}',git),'Full Git SHA required')
  names=command_output(['git','ls-tree','-rz','--name-only',git]).decode().split('\0')
  selected={name for name in names if name.endswith(('.py','.yaml','.yml','.lock')) or
      name in ('AGENTS.md','docs/DL_R5_LIVE_TRIAL.md')}
  ordered=sorted(selected)
  # Raw blobs, not git archive's possible export/worktree conversions.
  source=io.BytesIO(command_output(['git','cat-file','--batch'],
      ''.join(git+':'+name+'\n' for name in ordered).encode()))
  result={}
  for name in ordered:
    header=source.readline().decode().strip().split()
    require(len(header) == 3 and header[1] == 'blob' and header[2].isdigit(), 'Regular Git source blob required')
    size=int(header[2]); blob=source.read(size)
    require(len(blob) == size and source.read(1) == b'\n','Truncated Git blob batch')
    result[name]=dict(sha256=sha_bytes(blob),bytes=size)
  require(not source.read(),'Unexpected Git blob batch tail')
  require(set(result) == selected and {'analysis/dl_live_audit.py','scripts/dl_live_family.py',
      'scripts/dl_live_trial.py','scripts/dl_live_policy.py','scripts/dl_live_shadow.py',
      'analysis/dl_live_statistics.py','docs/DL_R5_LIVE_TRIAL.md'} <= selected,'Complete tracked R5 source required')
  return result


def external_pin(value, digest=None):
  path=Path(value['path'])
  require(path.is_absolute() and path.is_file() and not path.is_symlink(),'Pinned external regular file required')
  require(type(value['bytes']) is int and value['bytes'] == path.stat().st_size and sha(path) == value['sha256'] and
      (digest is None or value['sha256'] == digest),'External original input bytes differ')
  return path


def original_inactive(pid, start_ticks):
  """A reused PID is distinct; an original live identity forbids final audit."""
  require(os.name == 'posix' and type(pid) is int and pid > 0 and
      type(start_ticks) is int and start_ticks > 0,'Actual Linux process identity required')
  path=Path(f'/proc/{pid}/stat')
  try:
    fields=path.read_text().rsplit(') ',1)[1].split()
  except FileNotFoundError:
    return
  require(int(fields[19]) != start_ticks or fields[0] == 'Z',
      'Original producer process remains active: '+str(pid))


def resource_evidence(directory, family, budget, model):
  process=read_json(directory/'process.json'); spawned=read_json(directory/'shadow_process.json')
  closed=read_json(directory/'shadow_closed.json')
  pid=process['worker_pid']; resource=model['resource']; uuid=budget['gpu_uuid']
  require(process['parent_pid'] == budget['parent_pid'] and process['worker_pgid'] == pid and
      spawned['version'] == FAMILY and spawned['parent_pid'] == budget['parent_pid'] and
      spawned['worker_identity']['pid'] == spawned['worker_identity']['pgid'] == pid and
      spawned['shadow_identity']['pgid'] == pid and spawned['shadow_identity']['pid'] != pid and
      spawned['shadow_before_gpu_import'] is True and
      spawned['spawned_monotonic'] < model['authenticated_monotonic'] and
      closed == family['shadow'] and closed['identity'] == spawned['shadow_identity'] and
      closed['returncode'] == 0 and closed['exchanges'] == 16 and closed['eof_sent'] is True,
      'Fresh exec/shadow-before-model/owned closure evidence differs')
  require(resource['logical_server'] == 'sv3' and resource['physical_gpu'] == 4 and
      resource['gpu_uuid'] == resource['compute_uuid'] == resource['egl_uuid'] == uuid and
      resource['cuda_visible_devices'] == uuid and str(resource['mujoco_egl_device_id']) == '4',
      'Recorded assigned CUDA/EGL resource differs')
  for name in ('runtime_cuda_egl.json','runtime_after.json'):
    runtime=read_json(directory/name)
    require(runtime['physical_gpu'] == 4 and runtime['pid'] == pid and runtime['compute_uuid'] == [uuid] and
        runtime['graphics_verified'] is True and runtime['live_state']['gpu_uuid'] == uuid,
        'Runtime actual CUDA/EGL mapping evidence differs')
  telemetry=[decode(line) for line in (directory/'resources.jsonl').read_text().splitlines()]
  require(telemetry and all(sample['physical_gpu'] == 4 and sample['gpu_uuid'] == uuid and
      sample['worker_pid'] == pid and all(job['pid'] == pid for job in sample['processes']+sample['pmon_jobs'])
      for sample in telemetry),'Resource telemetry incomplete or unknown GPU job')
  original_inactive(pid,spawned['worker_identity']['starttime_ticks'])
  original_inactive(spawned['shadow_identity']['pid'],spawned['shadow_identity']['starttime_ticks'])
  return process


def policy_receipt(receipt, metadata, steps):
  require(receipt['version'] == LIVE and receipt['key_namespace'] == NAMESPACE and
      receipt['public_oracle_calls'] == 48 and receipt['sealed_current_bundles'] == 16 and
      receipt['label_not_in_policy_root'] is True and receipt['public_rng_not_consumed_by_private_calls'] is True,
      'R5 frozen policy receipt differs')
  kernel=receipt['kernel']
  require(kernel['version'] == 'DL-r4-utility-r1' and kernel['key_namespace'] == 'DL_R4_U1' and
      kernel['current_full_state_sha256'] == metadata['full_state_sha256'] and
      all(kernel[key] is True for key in ('scientific_reads_current_params','full_state_and_counters_unchanged',
          'public_rng_not_consumed_by_private_calls','first_is_host_assembly_no_policy_or_random_call')) and
      kernel['current_original_nj_seed_calls'] == 2 and kernel['derived_nj_seed_calls'] == kernel['h1_nj_seed_calls'] == 0,
      'Frozen kernel state/trace evidence differs')
  require(kernel['calls'] == dict(initial=16,normal=steps-16,current=16,derived=16,first=16,h1=16,oracle=0),
      'Family actual policy call accounting differs')
  require(kernel['accessed_parameter_keys'] and all(type(k) is str for k in kernel['accessed_parameter_keys']),
      'Missing actual frozen parameter read inventory')


def authenticate(run, run_git, audit_git):
  """No pickle is decoded until this complete metadata/artifact pass returns."""
  root=Path(run).resolve(strict=True)
  require(command_output(['git','rev-parse','HEAD']).decode().strip() == audit_git and
      not command_output(['git','status','--porcelain','--untracked-files=all']),'Clean exact auditor checkout required')
  audit_source=source_manifest(audit_git)
  for name,entry in audit_source.items():
    path=ROOT/name
    require(not path.is_symlink() and path.is_file() and path.stat().st_size == entry['bytes'] and
        sha(path) == entry['sha256'],'Auditor actual source/blob differs: '+name)
  stock=inventory(root)
  budget,terminal,generation=(read_json(root/name) for name in
      ('budget.json','supervisor_terminal.json','generation_complete.json'))
  require(all(x['version'] == SCIENCE and x['git_sha'] == run_git for x in (budget,terminal,generation)),
      'Producer version/Git differs')
  require(budget['seconds'] == 7200 and budget['logical_server'] == 'sv3' and
      budget['hostname'].split('.')[0] == 'lyg0326' and budget['physical_gpu'] == 4 and
      budget['gpu_uuid'] == GPU_UUID and budget['source_files'] == source_manifest(run_git),
      'Complete producer source/resource/budget binding differs')
  original_inactive(budget['parent_pid'],budget['parent_start_ticks'])
  require(terminal['parent_failure'] is None and terminal['main_generation_complete'] is True and
      terminal['completed_families'] == 32 and terminal['completed_units'] == 512 and
      terminal['owned_groups_cleaned'] is True and 0 < terminal['elapsed_seconds'] <= 7200 and
      terminal['independent_audit_complete'] is terminal['scientific_support_claimed'] is False,
      'Successful bounded producer closure required')
  pin(root,'generation_complete.json',terminal['generation_complete'],stock,absolute=True)
  require(generation['complete_families'] == 32 and generation['complete_units'] == 512 and
      generation['optimizer_updates'] == generation['training_actions'] == 0 and
      generation['independent_audit_complete'] is generation['scientific_support_claimed'] is False,
      'Producer workload/status differs')
  external_pin(budget['input_pins']['checkpoint'],CHECKPOINT)
  external_pin(budget['input_pins']['config'],CONFIG)
  external_pin(budget['resource_record'])
  required={'budget.json','calibration_frozen.json','statistics_input.json','statistics_provisional.json'}
  require(set(generation['fixed_artifact_hashes']) == required,'Fixed producer artifact graph differs')
  for name,digest in generation['fixed_artifact_hashes'].items():
    require(stock['files'][name]['sha256'] == digest,'Fixed producer artifact hash differs')
  payload,cal=(read_json(root/name) for name in ('statistics_input.json','calibration_frozen.json'))
  statistics.validate_units(payload['units'])
  require(cal['version'] == SCIENCE and cal['sealed_event'] == payload['calibration_sealed_event'] == 384 and
      cal['held_families_started'] == 0 and cal['calibration_units'] == payload['units'][:128] and
      cal['frozen_models'] == payload['frozen_models'] and
      cal['calibration_evidence'] == generation['family_evidence'][:8], 'Cal128 pre-held freeze graph differs')
  statistics.authenticate_frozen(statistics.freeze_calibration(cal['calibration_units']),cal['frozen_models'])
  require(len(generation['family_evidence']) == 32,'Exactly32 family receipts required')
  expected_files=required | {'generation_complete.json','supervisor_terminal.json'}
  families,units,previous=[],[],budget['monotonic_deadline']-7200
  for f,entry in enumerate(generation['family_evidence']):
    check(); prefix=f'family_{f:02d}'; directory=root/prefix
    require(entry['path'] == prefix+'/terminal.json','Ordered family evidence path differs')
    terminal_path=pin(root,entry['path'],dict(entry,path='terminal.json'),stock)
    ft=read_json(terminal_path)
    require(ft['version'] == SCIENCE and ft['family'] == f and ft['worker_returncode'] == 0 and
        ft['failure'] is None and ft['worker_terminal'] is ft['owned_group_cleaned'] is True and
        previous <= ft['started_monotonic'] < ft['ended_monotonic'] < budget['monotonic_deadline'],
        'Family successful serial bounded closure differs')
    if f == 8: require(previous <= cal['sealed_monotonic'] < ft['started_monotonic'],'Cal seal did not precede held exec')
    previous=ft['ended_monotonic']
    for key,name in (('complete','family_complete.json'),('log','worker.log'),('resources','resources.jsonl'),('process','process.json')):
      pin(root,prefix+'/'+name,ft[key],stock,absolute=True)
    family=read_json(directory/'family_complete.json')
    require(family['version'] == FAMILY and family['family'] == f and len(family['units']) == 16 and
        set(family['artifacts']) == FAMILY_FILES and
        ft['started_monotonic'] <= family['completed_monotonic'] <= ft['ended_monotonic'],
        'Fresh complete family graph differs')
    for name,artifact in family['artifacts'].items(): pin(root,prefix+'/'+name,artifact,stock)
    model=read_json(directory/'model_authenticated.json'); metadata=model['metadata']
    require(model['version'] == FAMILY and metadata['checkpoint_sha256'] == CHECKPOINT and
        metadata['config_sha256'] == CONFIG and metadata['params_sha256'] == PARAMS and
        metadata['checkpoint_path'] == budget['input_pins']['checkpoint']['path'] and
        metadata['config_path'] == budget['input_pins']['config']['path'], 'Actual N1M model pins differ')
    require(metadata['counters'] == dict(actions=62563,batches=249493,updates=249493,
        eval_actions=1001,eval_seed=1289196729) and
        type(metadata['full_state_sha256']) is str and re.fullmatch('[0-9a-f]{64}',metadata['full_state_sha256']) and
        (not units or metadata['full_state_sha256'] == units[0]['model_state']),
        'Frozen N1M full counters/state identity differs between families')
    for package in ('numpy','mujoco','dm-control'):
      require(importlib.metadata.version(package) == metadata['packages'][package],'Official CPU replay package version differs: '+package)
    process=resource_evidence(directory,family,budget,model)
    command=process['command']
    for flag,value in (('--family',str(f)),('--expected-git',run_git),('--output',str(directory)),
        ('--budget',str(root/'budget.json')),('--checkpoint',metadata['checkpoint_path']),('--config',metadata['config_path']),
        ('--resource-record',budget['resource_record']['path'])):
      require(command.count(flag) == 1 and command[command.index(flag)+1] == value,'Actual family command differs: '+flag)
    expected_files |= {prefix+'/'+name for name in FAMILY_FILES | {'terminal.json','family_complete.json',
        'worker.log','resources.jsonl','process.json'}}
    prior=model['authenticated_monotonic']; total_steps=0
    for k,ue in enumerate(family['units']):
      u=16*f+k; relative=prefix+f'/unit_{k:02d}/complete.json'
      require(ue['unit'] == u and ue['path'] == f'unit_{k:02d}/complete.json','Ordered global/local unit entry differs')
      uc=read_json(pin(root,relative,dict(ue,path='complete.json'),stock))
      require(uc['version'] == FAMILY and uc['unit'] == u and uc['plan'] == plan(u) and
          ue['unit_row'] == uc['unit_row'] == payload['units'][u] and
          set(uc['artifacts']) == UNIT_FILES and prior <= uc['started_monotonic'] < uc['monotonic'] <
          family['completed_monotonic'],'Actual serial unit identity/row/chronology differs')
      prior=uc['monotonic']; total_steps+=uc['executed_steps']
      unitprefix=prefix+f'/unit_{k:02d}/'
      for name,artifact in uc['artifacts'].items(): pin(root,unitprefix+name,artifact,stock)
      signals,assignment=(read_json(root/(unitprefix+name)) for name in ('signals_sealed.json','assignment.json'))
      for key,name in (('pretreatment_file','pretreatment.pkl'),('shadow_request','shadow_request.pkl'),('shadow_response','shadow_response.pkl')):
        require(signals[key] == uc['artifacts'][name],'Durable signal artifact binding differs')
      require(assignment['pretreatment_file'] == uc['artifacts']['pretreatment.pkl'] and
          assignment['signals_file'] == uc['artifacts']['signals_sealed.json'],'Assignment durable raw seal binding differs')
      expected_files |= {unitprefix+name for name in UNIT_FILES | {'complete.json'}}
      units.append(dict(unit=u,prefix=unitprefix,complete=uc,signals=signals,assignment=assignment,model_state=metadata['full_state_sha256']))
    policy_receipt(family['policy_receipt'],metadata,total_steps)
    families.append(family)
  require(set(stock['files']) == expected_files,'Unexpected/missing run artifact inventory')
  expected_dirs={f'family_{f:02d}' for f in range(32)} | {
      f'family_{f:02d}/unit_{k:02d}' for f in range(32) for k in range(16)}
  require(set(stock['directories']) == expected_dirs,'Unexpected/missing run directory inventory')
  return dict(root=root,inventory=stock,payload=payload,units=units,calibration=cal,budget=budget,
      audit_source=audit_source)


def trusted_pickle(auth, relative):
  check()
  body=(auth['root']/relative).read_bytes()
  entry=auth['inventory']['files'][relative]
  require(len(body) == entry['bytes'] and sha_bytes(body) == entry['sha256'],'Artifact changed before trusted decode')
  return pickle.loads(body)


def audit(args):
  output=Path(args.output).resolve(); run=Path(args.run).resolve(strict=True)
  require(not output.exists() and not output.is_relative_to(run),'Fresh external audit output required')
  auth=authenticate(run,args.expected_run_git,args.expected_audit_git)
  counts=dict(source_steps=0,outcome_steps=0,shadow_steps=0,terminal_absorbing_steps=0)
  for entry in auth['units']:
    check(); prefix=entry['prefix']
    record,request,response,seal=(trusted_pickle(auth,prefix+name) for name in
        ('trajectory.pkl','shadow_request.pkl','shadow_response.pkl','pretreatment.pkl'))
    row=validate_record(record,entry['complete'],request,response,seal,entry['signals'],entry['assignment'],
        entry['unit'],entry['model_state'])
    require(row == auth['payload']['units'][entry['unit']],'Audited statistics row differs')
    for key,value in replay(record,request,response).items(): counts[key]+=value
  require(inventory(run) == auth['inventory'],'Run changed during complete independent audit')
  require('jax' not in sys.modules,'Audit unexpectedly imported JAX/model runtime')
  provisional=statistics.analyze(auth['payload'],evidence_authenticated=False)
  statistics.authenticate_frozen(provisional,read_json(run/'statistics_provisional.json'))
  result=statistics.analyze(auth['payload'],evidence_authenticated=True)
  require(inventory(run) == auth['inventory'],'Run changed before audit publication')
  check()
  report=dict(version=VERSION,run_git=args.expected_run_git,audit_git=args.expected_audit_git,
      run_directory=str(run),complete_units=512,complete_families=32,counts=counts,
      seconds=SECONDS,elapsed_seconds=time.monotonic()-ENTRY_STARTED,
      evidence_authenticated=True,all_physics_and_rewards_byte_exact=True,
      model_reexecuted=False,renderer_reexecuted=False,subkey_sampling_reexecuted=False,
      prefix_full_carry_reconstructed=False,public_policy_reexecuted_by_auditor=False,
      inventory=auth['inventory'],statistics=result,limitations=[
          'Official CPU physics is replayed independently from the live producer, using the same simulator implementation.',
          'RGB, NN outputs including D/K/e, actor/latent subkeys and public-policy calls are authenticated stored evidence; no NN, RNG sampler or renderer is reexecuted.',
          'Prefix carry digests form a continuous recorded chain; full carries are available only at initial/intervention/outcome/final boundaries.',
          'Source code, wire bodies, hash-linked durable seal/assignment artifacts and monotonic receipts support recorded chronology; they cannot exclude hidden label access or hardware/driver interference.',
          'A successful engineering audit is not evidence that posterior protection improves training or model performance.'])
  output.parent.mkdir(parents=True,exist_ok=True)
  with output.open('x',encoding='utf-8',newline='\n') as stream:
    json.dump(report,stream,sort_keys=True,indent=2,allow_nan=False); stream.write('\n')
    stream.flush(); os.fsync(stream.fileno())
  check()
  print(json.dumps(dict(version=VERSION,complete_units=512,output=str(output),sha256=sha(output),
      elapsed_seconds=time.monotonic()-ENTRY_STARTED),sort_keys=True),flush=True)
  return report


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  for name in ('run','expected-run-git','expected-audit-git','output'): parser.add_argument('--'+name,required=True)
  audit(parser.parse_args())


if __name__ == '__main__':
  main()
