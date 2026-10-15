"""Independent finite DL-NM1 C audit; no JAX, NN, backward or Adam replay.

The fixed Threefry implementation below follows installed official JAX0.4.33
prng.py (nonpartitionable 2x32) and random.py's float32 uniform conversion.
It independently reconstructs the actual fixed permutation RNG, not GPU math.
"""
import time
ENTRY_STARTED = time.monotonic()
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import re
import signal
import subprocess
import sys
import zipfile

SECONDS, DEADLINE, OWNED = 900, float('inf'), {}
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
VERSION = 'DL-NM1-cal-independent-audit-r1'
SCIENCE = 'DL-NM1-cal-r1'
ORIGINAL_CHECKPOINT = 'c6a7d92e24669a515bb44693faf73e442d9868f9fc814eec0e22e069aa594d24'
CHECKPOINT = ORIGINAL_CHECKPOINT
CONFIG = '877f8470cb4f7f45d143e3099fbd3f55a828991afe33779b7b04c4264f6c8377'
INVENTORY = '3339b71ca033d5fc065130751fe163aca05a684b2f14f570ab2018219a3a706d'
FULLSTATE = '4e0d1a0e1de806bf8e4863278cabd171ed0d208dddc1fea18b8b54d68ca6383f'
PARAMS = 'cffc13c0dcdfbd0c8b0e5c503c9d6251f4b1550c08b85a8742a638655d66d372'
COUNTERS = dict(updates=249493,batches=249493,actions=62563,eval_actions=1001,eval_seed=1289196729)
RAW_FIELDS = dict(image=('uint8',(64,64,3)),reward=('float32',()),is_first=('bool',()),
    is_last=('bool',()),is_terminal=('bool',()),action=('float32',(12,)),stepid=('uint8',(20,)),
    **{'dyn/deter':('float32',(4096,)),'dyn/stoch':('float32',(32,32))})
COUNTS = (1280,16,1001008,1008,1000000)
GPU_UUID = 'GPU-79704918-e096-92b4-252c-e3244497a8b3'


def expire(signum, frame):
  for owned in tuple(OWNED.values()):
    try:
      if group_alive(owned):os.killpg(owned['pgid'],signal.SIGKILL)
    except (OSError,ValueError):pass  # Lost identity never authorizes a signal.
  os._exit(124)


if __name__ == '__main__':
  os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',MUJOCO_GL='disable')
  DEADLINE = ENTRY_STARTED+SECONDS
  if os.name == 'posix':
    signal.signal(signal.SIGALRM,expire);signal.signal(signal.SIGTERM,expire)
    signal.setitimer(signal.ITIMER_REAL,max(.001,DEADLINE-time.monotonic()))

import numpy as np


def require(condition, message):
  if not condition:
    raise ValueError(message)


def check():
  require(time.monotonic() < DEADLINE,'Whole independent CPU900s deadline exceeded')


def decode(body):
  def pairs(items):
    value={}
    for key,item in items:
      require(key not in value,'Duplicate JSON key');value[key]=item
    return value
  return json.loads(body,object_pairs_hook=pairs,parse_constant=lambda v:
      (_ for _ in ()).throw(ValueError('Nonfinite JSON number')))


def typed_hash(value):
  """Declared tree_digest wire encoding, independently parsed, no producer import."""
  digest=hashlib.sha256()
  def visit(item):
    if type(item) is dict:
      require(all(type(k) is str for k in item),'String tree keys required');digest.update(b'dict')
      for k in sorted(item):digest.update(json.dumps(k).encode());visit(item[k])
    elif type(item) in (tuple,list):
      digest.update(type(item).__name__.encode());digest.update(str(len(item)).encode())
      for child in item:visit(child)
    elif item is None or type(item) in (bool,int,float,str):
      require(type(item) is not float or math.isfinite(item),'Nonfinite scalar')
      digest.update(type(item).__name__.encode());digest.update(json.dumps(item).encode())
    else:
      a=np.asarray(item)
      require(a.dtype.kind in 'buifc' and np.isfinite(a).all(),'Finite numeric array required')
      digest.update(json.dumps([a.dtype.str,str(a.dtype)]).encode())
      digest.update(json.dumps(a.shape).encode());digest.update(a.tobytes())
  visit(value)
  return digest.hexdigest()


class NumpyCheckpoint(pickle.Unpickler):
  def find_class(self,module,name):
    allowed={('numpy','ndarray'):np.ndarray,('numpy','dtype'):np.dtype,
        ('numpy.core.multiarray','_reconstruct'):np.core.multiarray._reconstruct,
        ('numpy.core.multiarray','scalar'):np.core.multiarray.scalar,
        ('numpy.core.numeric','_frombuffer'):np.core.numeric._frombuffer}
    require((module,name) in allowed,'Checkpoint non-NumPy class refused: '+module+'.'+name)
    return allowed[module,name]


def checkpoint(body):
  require(hashlib.sha256(body).hexdigest() == CHECKPOINT,'Original checkpoint raw hash differs before decode')
  stream=io.BytesIO(body);value=NumpyCheckpoint(stream).load()
  require(not stream.read() and type(value) is dict and set(value) == {'params','counters'},
      'Exact original checkpoint dictionary required')
  require(type(value['params']) is dict and len(value['params']) == 294 and
      all(type(k) is str and type(v) is np.ndarray and v.dtype.kind in 'buif' and
          np.isfinite(v).all() for k,v in value['params'].items()),'Complete294 finite NumPy leaves required')
  require(type(value['counters']) is dict and set(value['counters']) == set(COUNTERS) and
      all(type(v) is int for v in value['counters'].values()),'Original integer counters required')
  loaded=dict(value['counters'],batches=value['counters']['updates'])
  require(loaded == COUNTERS and int(value['params']['opt/step/value']) == 249493,
      'Actual post-load counters/optstep differ')
  require(parameter_hash(value['params']) == PARAMS and
      typed_hash(dict(params=value['params'],counters=loaded)) == FULLSTATE,'Original complete state hash differs')
  return value,dict(params=value['params'],counters=loaded)


def parameter_hash(params):
  digest=hashlib.sha256()
  for key,value in sorted(params.items()):
    digest.update(key.encode());digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode());digest.update(value.tobytes())
  return digest.hexdigest()


def plain(path):
  path=Path(path).absolute()
  require(not any(p.is_symlink() for p in (path,*path.parents)),'Symlink path refused')
  return path.resolve(strict=True)


def raw_hash(path):
  digest=hashlib.sha256();before=path.stat()
  with path.open('rb') as stream:
    for part in iter(lambda:stream.read(1024*1024),b''):check();digest.update(part)
  after=path.stat()
  require((before.st_size,before.st_mtime_ns,before.st_ino) ==
      (after.st_size,after.st_mtime_ns,after.st_ino),'File changed during raw SHA')
  return dict(bytes=after.st_size,sha256=digest.hexdigest())


def pinned(root,item):
  require(type(item) is dict and set(item) == {'path','bytes','sha256'} and
      type(item['bytes']) is int and item['bytes'] >= 0 and
      re.fullmatch('[0-9a-f]{64}',item['sha256']) is not None,'Exact raw file pin required')
  label=Path(item['path']);path=plain(label if label.is_absolute() else root/label)
  require(label.is_absolute() or (path.is_relative_to(root) and '..' not in label.parts),'Unsafe relative file pin')
  require(path.is_file() and raw_hash(path) == {k:item[k] for k in ('bytes','sha256')},'Raw file pin differs: '+item['path'])
  return path


def read_json(path):
  return decode(path.read_bytes())


def inventory(root):
  files={}
  for path in sorted(root.rglob('*')):
    check();require(not path.is_symlink(),'Run symlink refused')
    if path.is_file():files[path.relative_to(root).as_posix()]=raw_hash(path)
    else:require(path.is_dir(),'Nonregular run entry refused')
  return files


def command_output(command,payload=None):
  check();previous=None
  if os.name == 'posix':previous=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM,signal.SIGTERM})
  try:
    child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.PIPE if payload is not None else None,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=os.name == 'posix',
        **({'preexec_fn':lambda:signal.pthread_sigmask(signal.SIG_SETMASK,previous)} if previous is not None else {}))
    owned=process_identity(child.pid) if os.name == 'posix' else dict(pid=child.pid)
    require(os.name != 'posix' or owned['pgid'] == child.pid,'Owned helper session required')
    OWNED[child.pid]=owned
  finally:
    if previous is not None:signal.pthread_sigmask(signal.SIG_SETMASK,previous)
  try:
    out,error=child.communicate(input=payload,timeout=min(30,max(.001,DEADLINE-time.monotonic())))
    require(child.returncode == 0,'Git metadata helper failed: '+error.decode(errors='replace')[:200]);return out
  finally:
    cleanup(child)
    for pipe in (child.stdin,child.stdout,child.stderr):
      if pipe is not None:pipe.close()


def process_identity(pid):
  fields=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(') ',1)[1].split()
  return dict(pid=int(pid),pgid=int(fields[2]),start_ticks=int(fields[19]))


def group_alive(owned):
  """Only the original independent session can authorize a group signal."""
  try:os.killpg(owned['pgid'],0)
  except ProcessLookupError:return []
  try:actual=process_identity(owned['pid'])
  except (FileNotFoundError,ProcessLookupError):actual=None
  require(actual is None or identity_equal(actual,owned),'Owned helper PID reused or group changed')
  result=[]
  for path in Path('/proc').iterdir():
    if not path.name.isdigit():continue
    try:fields=(path/'stat').read_text().rsplit(') ',1)[1].split()
    except (FileNotFoundError,ProcessLookupError):continue
    if int(fields[2]) == owned['pgid'] and fields[0] not in ('Z','X'):
      require(int(fields[3]) == owned['pgid'] and int(fields[19]) >= owned['start_ticks'],
          'Original helper group/session ownership lost')
      result.append(int(path.name))
  return result


def cleanup(child):
  require(child.pid in OWNED,'Unregistered helper cleanup refused');owned=OWNED[child.pid]
  if os.name == 'posix':
    if group_alive(owned):os.killpg(owned['pgid'],signal.SIGKILL)
    child.wait(timeout=5);until=time.monotonic()+3
    while group_alive(owned) and time.monotonic() < until:time.sleep(.01)
    require(not group_alive(owned),'Original owned helper descendants remain live')
  else:
    if child.poll() is None:child.kill()
    child.wait(timeout=5)
  OWNED.pop(child.pid)


def git_binding(run_git,audit_git,declared):
  require(all(type(s) is str and re.fullmatch('[0-9a-f]{40}',s) for s in (run_git,audit_git)),'Full Git SHAs required')
  require(command_output(['git','rev-parse','HEAD']).decode().strip() == audit_git and
      not command_output(['git','status','--porcelain=v1','--untracked-files=all']),'Clean exact audit HEAD required')
  own=Path(__file__).resolve().relative_to(ROOT).as_posix()
  require(command_output(['git','cat-file','blob',audit_git+':'+own]) == Path(__file__).read_bytes(),'Audit script/Git bytes differ')
  required={'dreamerv3/agent.py','dreamerv3/rssm.py','dreamerv3/dt_latch.py','embodied/jax/agent.py',
      'scripts/dl_nm1_counterfactual.py','scripts/dl_normal_replay.py','analysis/dl_nm1_calibration.py',
      'docs/DL_NM1_TRIAL.md','docs/DL_NM1_CAL_ARTIFACTS.md'}
  require(required <= set(declared),'Complete C source/norm bindings required')
  names=command_output(['git','ls-tree','-rz','--name-only',run_git]).decode().split('\0')
  expected={name for name in names if name.endswith(('.py','.yaml','.yml','.lock')) or
      name in ('AGENTS.md','docs/DL_NM1_TRIAL.md','docs/DL_NM1_CAL_ARTIFACTS.md')}
  require(set(declared) == expected,'Complete original C Git source set differs')
  packed=io.BytesIO(command_output(['git','cat-file','--batch'],
      ''.join(run_git+':'+name+'\n' for name in sorted(declared)).encode()))
  for name in sorted(declared):
    header=packed.readline().split();require(len(header) == 3 and header[1] == b'blob','Regular source Git blob required')
    body=packed.read(int(header[2]));require(packed.read(1) == b'\n','Truncated source Git blob')
    require(declared[name] == dict(bytes=len(body),sha256=hashlib.sha256(body).hexdigest()),'Original source blob differs: '+name)
  require(not packed.read(),'Unexpected source Git batch tail')


def inactive(identity):
  require(all(type(identity[k]) is int and identity[k] > 0 for k in ('pid','pgid','start_ticks')),'Real process identity required')
  try:
    fields=Path('/proc/'+str(identity['pid'])+'/stat').read_text().rsplit(') ',1)[1].split()
  except (FileNotFoundError,ProcessLookupError):return
  require(int(fields[19]) != identity['start_ticks'] or fields[0] in ('Z','X'),
      'Original parent/worker is still active')


def inactive_worker_group(identity):
  require(identity['pgid'] == identity['pid'],'Original worker independent group required')
  require(not group_alive(identity),'Original worker session still has live descendants')


def tensor_hash(array,logical,dtype_str):
  digest=hashlib.sha256()
  digest.update(json.dumps([dtype_str,logical]).encode());digest.update(json.dumps(array.shape).encode())
  digest.update(array.tobytes());return digest.hexdigest()


def specs_for(array):
  return dict(logical_dtype=str(array.dtype),dtype_str=array.dtype.str,shape=list(array.shape),
      typed_sha256=typed_hash(array),bytes=array.nbytes)


def load_pack(path,specs):
  """Decode only previously closed/raw-authenticated NPZ; CRC each member once."""
  require(type(specs) is dict and specs,'Nonempty exact array schema required')
  storage={}
  for name,s in specs.items():
    require(type(name) is str and set(s) == {'storage_key','logical_dtype','dtype_str','shape','typed_sha256','bytes'},
        'Exact stored array specification required')
    require(type(s['storage_key']) is str and re.fullmatch('a[0-9]{5}',s['storage_key']) and
        s['storage_key'] not in storage,'Unique safe NPZ storage key required')
    storage[s['storage_key']]=name
  result={}
  with zipfile.ZipFile(path) as archive:
    members=archive.infolist()
    require(len(members) == len(storage) and {m.filename for m in members} ==
        {k+'.npy' for k in storage},'NPZ missing/extra/duplicate members')
    for member in members:
      check();name=storage[member.filename[:-4]];s=specs[name]
      require(type(s['shape']) is list and all(type(x) is int and 0 <= x <= 10**8 for x in s['shape']),
          'Bounded exact tensor shape required')
      dtype=np.dtype(np.uint16 if s['logical_dtype'] == 'bfloat16' else s['logical_dtype'])
      size=math.prod(s['shape'])*dtype.itemsize
      require(dtype.kind in 'buif' and size == s['bytes'] and member.file_size <= size+10000 and
          member.compress_type in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED) and not member.flag_bits & 1,
          'Numeric bounded unencrypted NPZ member required')
      # ZipFile.read checks the actual member CRC. No extraction or pickle.
      value=np.lib.format.read_array(io.BytesIO(archive.read(member)),allow_pickle=False,max_header_size=10000)
      require(type(value) is np.ndarray and value.shape == tuple(s['shape']) and value.dtype == dtype,
          'NPZ dtype/shape differs: '+name)
      if s['logical_dtype'] == 'bfloat16':
        bf16_values(value);require(s['dtype_str'] == '<V2','Declared BF16 dtype differs')
      else:require(value.dtype.str == s['dtype_str'] and np.isfinite(value).all(),'Original array dtype/finite differs')
      require(tensor_hash(value,s['logical_dtype'],s['dtype_str']) == s['typed_sha256'],'Original typed tensor SHA differs')
      result[name]=value
  return result


def replay_source(directory,report):
  """Independent NPZ/UUID/chain decoder and original insertion-order starts."""
  require(report['version'] == 'DL-replay-inventory-r1' and report['schema'] == 1 and
      report['source_git'] == '13ec72a214b0c03b886b2c7fac52c8f089966dec' and report['stable'] is True,
      'Original replay identity differs')
  files,chains,rows,resets,actions=COUNTS
  require(len(report['before_inventory']) == files and report['before_inventory'] == report['after_inventory'] and
      (report['rows'],report['resets'],report['train_actions']) == (rows,resets,actions),
      'Original complete replay counts differ')
  require(report['fields'] == {k:[dtype,list(shape)] for k,(dtype,shape) in RAW_FIELDS.items()},'Original replay schema differs')
  expected={e['path']:dict(bytes=e['bytes'],sha256=e['sha256']) for e in report['before_inventory']}
  require(len(expected) == files and set(p.name for p in directory.iterdir()) == set(expected),
      'Replay missing/extra/duplicate original chunks')
  # Close ALL raw input hashes before decoding a single original NPZ.
  for name,pin in expected.items():require(raw_hash(plain(directory/name)) == pin,'Original replay raw SHA differs')
  alphabet='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
  def uid(text):
    require(re.fullmatch('[0-9a-zA-Z]{22}',text) is not None,'Original canonical UUID required')
    value=0
    for char in text:value=62*value+alphabet.index(char)
    require(0 < value < 2**128,'Nonzero128bit original UUID required');return value.to_bytes(16,'big')
  chunks={};summaries=[]
  for name in expected:
    check();match=re.fullmatch(r'(\d{8}T\d{6}F\d{6})-([0-9a-zA-Z]{22})-([0-9a-zA-Z]{22})-([1-9]\d*)\.npz',name)
    require(match is not None,'Original replay filename differs')
    timestamp,identity,successor,length=match.groups();length=int(length)
    require(identity != successor and length <= 1024,'Original chunk bounds differ')
    ids,succ=uid(identity),uid(successor)
    arrays={}
    with zipfile.ZipFile(directory/name) as archive:
      require(len(archive.infolist()) == len(RAW_FIELDS) and set(archive.namelist()) ==
          {k+'.npy' for k in RAW_FIELDS},'Original NPZ member schema/duplicates differ')
      for key,(dtype,shape) in RAW_FIELDS.items():
        size=length*math.prod(shape)*np.dtype(dtype).itemsize
        require(archive.getinfo(key+'.npy').file_size <= size+10000,'Original oversized NPZ member')
        value=np.lib.format.read_array(io.BytesIO(archive.read(key+'.npy')),allow_pickle=False,max_header_size=10000)
        require(value.dtype == np.dtype(dtype) and value.shape == (length,*shape) and
            np.isfinite(value).all(),'Original replay dtype/shape/finite differs')
        arrays[key]=value
    require(np.all(arrays['stepid'][:,:16] == np.frombuffer(ids,np.uint8)) and
        arrays['stepid'][:,16:].tobytes() == np.arange(length,dtype='>u4').view(np.uint8).reshape(-1,4).tobytes(),
        'Original filename UUID/index and full stepid differ')
    first,last,terminal=(arrays[k] for k in ('is_first','is_last','is_terminal'))
    require(np.all(~terminal | last) and np.array_equal(first[1:],last[:-1]) and
        not np.any(arrays['reward'][first]) and not np.any(arrays['action'][last]),'Original replay reset/action semantics differ')
    stoch=arrays['dyn/stoch']
    require(np.all((stoch == 0) | (stoch == 1)) and np.all(stoch.sum(-1) == 1),'Actual onehot cached state required')
    summary=dict(path=name,time=timestamp,uuid=identity,successor=successor,length=length,
        first=bool(first[0]),last=bool(last[-1]),resets=int(first.sum()),terminals=int(terminal.sum()),
        ends=int(last.sum()),crc_verified=True,cached_context_validated=True)
    require(identity not in chunks,'Duplicate original UUID')
    chunks[identity]=dict(summary,data=arrays,uuid_bytes=ids,successor_bytes=succ);summaries.append(summary)
  require(summaries == report['chunks'],'Decoded original chunk summaries differ')
  incoming={}
  for identity,c in chunks.items():
    require(c['successor'] not in incoming,'Original replay chains merge');incoming[c['successor']]=identity
    if c['successor'] in chunks:
      child=chunks[c['successor']]
      require(child['time'] >= c['time'] and child['first'] == c['last'],'Original chain timestamp/reset boundary differs')
  heads=sorted(set(chunks)-set(incoming));require(len(heads) == chains,'Original chain count differs')
  seen=set();actual_chains=[];chain_by_uuid={};counts={}
  for chain,head in enumerate(heads):
    require(chunks[head]['first'],'Full original chain begins without reset')
    identity=head;names=[];chain_chunks=[];length=0
    while identity in chunks:
      require(identity not in seen,'Original replay graph cycle');seen.add(identity)
      c=chunks[identity];chain_by_uuid[identity]=chain;names.append(c['path']);length+=c['length']
      chain_chunks.append(identity);identity=c['successor']
    remaining=0
    for current in reversed(chain_chunks):
      remaining+=chunks[current]['length'];counts[current]=min(chunks[current]['length'],max(0,remaining-64))
    actual_chains.append(dict(head_uuid=head,dangling_successor=identity,paths=names,rows=length,length65_windows=max(0,length-64)))
  require(len(seen) == files and actual_chains == report['chains'] and
      sum(c['length'] for c in chunks.values()) == rows and sum(c['resets'] for c in chunks.values()) == resets,
      'Full original replay graph/counts differ')
  order=sorted(chunks,key=lambda x:chunks[x]['path']);amounts=np.array([counts[x] for x in order],np.int64)
  require(int(amounts.sum()) == rows-64*chains,'Legal65 start total differs')
  seed=int(np.random.SeedSequence([20261101,0x4E4D4341]).generate_state(1,dtype=np.uint64)[0])
  draws=np.random.default_rng(seed).integers(0,int(amounts.sum()),(128,16));ends=np.cumsum(amounts)
  plan=[]
  for batch in draws:
    starts=[]
    for draw in batch:
      index=int(np.searchsorted(ends,draw,side='right'));base=int(ends[index-1]) if index else 0
      starts.append([order[index],int(draw)-base])
    plan.append(starts)
  return dict(chunks=chunks,chain_by_uuid=chain_by_uuid,plan=plan,seed=seed,
      inventory=expected,legal65_starts=int(amounts.sum()))


def reconstruct_batch(source,starts):
  rows=[];chains=[]
  for identity,index in starts:
    length=65;parts={k:[] for k in RAW_FIELDS};chain=source['chain_by_uuid'][identity]
    while length:
      require(identity in source['chunks'],'Original65 span outside full replay')
      chunk=source['chunks'][identity];used=min(length,chunk['length']-index)
      require(used > 0,'Illegal original65 start/index')
      for key in parts:parts[key].append(chunk['data'][key][index:index+used])
      length-=used;identity,index=chunk['successor'],0
    rows.append({key:np.concatenate(value) for key,value in parts.items()});chains.append(chain)
  data={key:np.stack([row[key] for row in rows]) for key in RAW_FIELDS}
  data['is_first'][:,0]=True
  nxt=np.roll(data['is_first'],-1,axis=1);nxt[:,-1]=False
  data['is_last']|=nxt;data['consec']=np.zeros((16,65),np.int32)
  return data,np.broadcast_to(np.array(chains,np.int32)[:,None],(16,64)).copy()


def identity_equal(a,b):
  return all(a[k] == b[k] for k in ('pid','pgid','start_ticks'))


def authenticate(run,run_git,audit_git,launcher_path,launcher_sha):
  """Close every original run/input raw byte and true exit before ANY decode."""
  run=plain(run);require(run.is_dir(),'Original run directory required')
  launch_path=plain(launcher_path)
  require(raw_hash(launch_path)['sha256'] == launcher_sha,'External launcher raw SHA differs')
  launcher=read_json(launch_path);terminal=read_json(run/'supervisor_terminal.json')
  require(launcher['version'] == 'DL-NM1-launcher-r1' and launcher['git_sha'] == run_git and
      launcher['parent_returncode'] == 0 and launcher['parent_terminal'] is True and
      launcher['parent_identity_gone'] is True and launcher['watchdog_seconds'] == 3610 and
      type(launcher['elapsed_seconds']) in (int,float) and 0 < launcher['elapsed_seconds'] <= 3600 and
      launcher['run_directory'] == str(run) and launcher['supervisor_terminal_sha256'] == raw_hash(run/'supervisor_terminal.json')['sha256'],
      'True external parent exit/budget required before array decode')
  require(terminal['version'] == SCIENCE and terminal['schema'] == 1 and terminal['git_sha'] == run_git and
      terminal['worker_returncode'] == 0 and terminal['failure'] is None and terminal['worker_terminal'] is True and
      terminal['owned_group_cleaned'] is True and 0 < terminal['elapsed_seconds'] <= 3600 and
      identity_equal(launcher['parent_identity'],terminal['parent_identity']),'Successful original parent/worker terminal required')
  inactive(terminal['parent_identity']);inactive(terminal['worker_identity'])
  inactive_worker_group(terminal['worker_identity'])
  files=inventory(run);required_top={'budget.json','plan.json','initial.json','initial.npz','resource_refreshed.json',
      'runtime_before.json','runtime_after.json','frozen.json','summary.json','final.json'}
  require(set(terminal['artifacts']) == {'budget.json','process.json','worker.log','resources.jsonl','generation_complete.json'},
      'Exact parent artifact set required')
  for name,pin in terminal['artifacts'].items():require(pinned(run,pin) == run/name,'Original parent artifact path differs')
  budget=read_json(run/'budget.json');generation=read_json(run/'generation_complete.json')
  require(budget['version'] == SCIENCE and budget['schema'] == 1 and budget['git_sha'] == run_git and
      budget['seconds'] == 3600 and budget['hostname'].split('.')[0] == 'lyg0326' and
      budget['logical_server'] == 'sv3' and budget['physical_gpu'] == 5 and budget['gpu_uuid'] == GPU_UUID and
      budget['constants'] == dict(batches=128,batch_size=16,loss_length=64,context=1,plan_root=20261101,
          plan_tag=0x4E4D4341,model_seed=0,opt_step=249493) and
      identity_equal(budget['parent_identity'],terminal['parent_identity']),'Frozen C source/budget differs')
  git_binding(run_git,audit_git,budget['source_files'])
  process=read_json(run/'process.json')
  require(process['version'] == SCIENCE and process['schema'] == 1 and
      process['parent_identity'] == terminal['parent_identity'] and process['worker_identity'] == terminal['worker_identity'] and
      process['worker_identity']['pgid'] == process['worker_identity']['pid'] and
      process['worker_identity']['parent_pid'] == process['parent_identity']['pid'],'Direct owned worker identity differs')
  require(generation['version'] == SCIENCE and generation['schema'] == 1 and generation['git_sha'] == run_git and
      generation['complete_batches'] == 128 and generation['optimizer_updates_committed'] == 0 and
      generation['replay_updates_committed'] == 0 and generation['independent_audit_complete'] is False and
      generation['subsequent_learning_authorized'] is False and set(generation['top_artifacts']) == required_top,
      'Complete fixed128 generation required')
  expected_files=set(required_top)|{'process.json','worker.log','resources.jsonl','generation_complete.json','supervisor_terminal.json'}
  for name,pin in generation['top_artifacts'].items():require(pinned(run,pin) == run/name,'Top artifact path differs')
  require(type(generation['batches']) is list and len(generation['batches']) == 128,'All128 ordered batch closures required')
  for i,pin in enumerate(generation['batches']):
    directory=run/f'batch_{i:03d}';path=pinned(run,pin)
    require(path == directory/'complete.json','Fixed ordered batch closure path differs')
    complete=read_json(path)
    require(complete['version'] == SCIENCE and complete['schema'] == 1 and complete['batch'] == i and
        set(complete['artifacts']) == {'evidence.npz','record.json','controls.npz','controls.json'},'Complete batch graph differs')
    for name,entry in complete['artifacts'].items():
      require(pinned(directory,entry) == directory/name,'Contained batch artifact differs')
      expected_files.add(f'batch_{i:03d}/'+name)
    expected_files.add(f'batch_{i:03d}/complete.json')
    require(complete['record_sha256'] == complete['artifacts']['record.json']['sha256'] and
        complete['controls_sha256'] == complete['artifacts']['controls.json']['sha256'],'Batch record/control pins differ')
  require(set(files) == expected_files,'Run closure missing/extra files')
  require({p.relative_to(run).as_posix() for p in run.rglob('*') if p.is_dir()} ==
      {f'batch_{i:03d}' for i in range(128)},'Run closure extra/missing directories')
  require(set(budget['inputs']) == {'checkpoint','config','inventory','resource_record','stage_record'},'Exact original input set required')
  paths={name:pinned(run,pin) if pin is not None else None for name,pin in budget['inputs'].items()}
  for name,expected in (('checkpoint',CHECKPOINT),('config',CONFIG),('inventory',INVENTORY)):
    require(budget['inputs'][name]['sha256'] == expected,'Published original input pin differs: '+name)
  report=read_json(paths['inventory']);directory=plain(budget['replay_directory'])
  stage=None
  if paths['stage_record'] is not None:
    require(budget['stage_expected_sha256'] == budget['inputs']['stage_record']['sha256'],'Original stage SHA differs')
    from scripts import dl_replay_stage
    dl_replay_stage.DEADLINE=DEADLINE
    stage=dl_replay_stage.authenticate_stage(paths['stage_record'],budget['stage_expected_sha256'],
        paths['inventory'],report,directory)
  else:
    require(budget['stage_expected_sha256'] is None and report['replay_directory'] == str(directory),
        'Original replay path differs without explicit stage binding')
  require(directory not in run.parents and run not in directory.parents and directory != run,
      'C output must stay outside immutable replay')
  return dict(run=run,budget=budget,terminal=terminal,process=process,launcher=launcher,files=files,
      paths=paths,report=report,directory=directory,stage=stage)


def resources(auth):
  run,budget=auth['run'],auth['budget'];worker=auth['terminal']['worker_identity']['pid']
  command=auth['process']['command']
  require(type(command) is list and len(command) > 2 and Path(command[1]).name == 'dl_nm1_cal.py' and
      command.count('--worker') == 1,'Actual owned C worker command required')
  flags={'--expected-git':budget['git_sha'],'--output':str(run),'--replay-directory':str(auth['directory'])}
  flags.update({'--'+k.replace('_','-'):str(auth['paths'][k]) for k in
      ('checkpoint','config','inventory','resource_record')})
  if auth['paths']['stage_record'] is not None:
    flags.update({'--stage-record':str(auth['paths']['stage_record']),
        '--expected-stage-sha256':budget['stage_expected_sha256']})
  for flag,value in flags.items():
    require(command.count(flag) == 1 and command[command.index(flag)+1] == value,'Recorded original worker argument differs: '+flag)
  start=auth['process']['started_monotonic'];deadline=budget['monotonic_deadline']
  require(type(start) in (int,float) and type(deadline) is float and math.isfinite(deadline) and
      0 < deadline-start <= 3600,'Original3600 monotonic budget differs')
  def live(value,idle=False):
    require(value['physical_gpu'] == 5 and value['gpu_uuid'] == GPU_UUID,'Assigned compute UUID differs')
    jobs=value['processes']+value['pmon_jobs']
    require(not jobs if idle else all(x['pid'] == worker and x['type'] == 'C' for x in jobs),
        'Foreign or graphics work in completed compute-only C resource records')
    parsed=[];observed=False
    for line in value['raw_pmon'].splitlines():
      fields=line.split()
      if len(fields) >= 3 and fields[0] == '5':observed=True
      if len(fields) >= 3 and fields[0] == '5' and fields[1] != '-':
        require(fields[1].isdigit(),'Unavailable actual assigned pmon inventory');parsed.append((int(fields[1]),fields[2]))
    require(observed and parsed == [(x['pid'],x['type']) for x in value['pmon_jobs']],'Raw assigned-device pmon differs')
    for line in value['raw_pmon'].splitlines():
      fields=line.split()
      if len(fields) >= 3 and fields[0].isdigit() and fields[1] == str(worker):
        require(fields[0] == '5' and fields[2] == 'C','Worker pmon reports another GPU or graphics')
  original=read_json(auth['paths']['resource_record']);refreshed=read_json(run/'resource_refreshed.json')
  for value in (original,refreshed):
    require(value['version'] == 'DL-NM1-compute-resource-r1' and
        value['logical_server'] == 'sv3' and value['actual_hostname'].split('.')[0] == 'lyg0326' and
        value['physical_gpu'] == 5 and value['gpu_uuid'] == value['compute_uuid'] == GPU_UUID and
        value['egl_uuid'] is None and value['graphics_verified'] is False and
        value['environment'] == dict(CUDA_VISIBLE_DEVICES=GPU_UUID,CUDA_DEVICE_ORDER='PCI_BUS_ID',
            JAX_PLATFORMS='cuda',MUJOCO_GL='disable',MUJOCO_EGL_DEVICE_ID=None,PYOPENGL_PLATFORM=None) and
        value['occupancy_basis'] == 'available complete XML process inventory and assigned-device pmon' and
        value['source_mapping'] == 'not-applicable-no-environment-or-EGL-initialization' and
        value['occupancy'] == 'idle' and value['exclusive'] is True,'Initial/refreshed physical resource binding differs')
    live(value['live_check'],True)
  require(refreshed['resource_source_path'] == str(auth['paths']['resource_record']) and
      refreshed['resource_source_sha256'] == budget['inputs']['resource_record']['sha256'],'Resource refresh rewrote original source')
  samples=[decode(line) for line in (run/'resources.jsonl').read_text().splitlines()]
  require(samples,'Complete live resource telemetry required')
  last=start
  for sample in samples:
    require(sample['worker_pid'] == worker and last <= sample['monotonic'] <= deadline and
        sample['xml_process_summary'] == sample['processes'],'Original resource chronology/XML summary differs')
    live(sample);last=sample['monotonic']
  for name in ('runtime_before.json','runtime_after.json'):
    value=read_json(run/name)
    require(value['pid'] == worker and value['compute_uuid'] == [GPU_UUID] and
        value['physical_gpu'] == 5 and value['graphics_verified'] is False,'Actual compute-only runtime identity differs')
    live(value['live_state']);require(value['pmon'] == value['live_state']['raw_pmon'],'Runtime raw pmon differs')
  return len(samples)


def selected_initial(auth,original,loaded,source):
  value=read_json(auth['run']/'initial.json');params=loaded['params']
  require(value['version'] == SCIENCE and value['schema'] == 1 and
      value['checkpoint_counters'] == original['counters'] and value['postload_counters'] == COUNTERS and
      value['full_state_sha256'] == FULLSTATE and value['params_sha256'] == PARAMS and
      value['state_leaves'] == {k:specs_for(v) for k,v in params.items()},'Original294 leaf/counter seals differ')
  require(value['worker_identity'] == auth['terminal']['worker_identity'] and
      value['packages']['jax'] == '0.4.33' and value['packages']['numpy'] == '1.26.4' and
      value['jax_prng'] == dict(default_prng_impl='threefry2x32',threefry_partitionable=False,enable_x64=False),
      'Frozen actual RNG/runtime packages differ')
  expected={k:v for k,v in params.items() if k.startswith(('enc/','dyn/obs'))}
  arrays=load_pack(auth['run']/'initial.npz',value['arrays'])
  require(set(arrays) == {'initial/'+k for k in expected},'Complete retained initial tensor set differs')
  for key,a in expected.items():
    require(arrays['initial/'+key].dtype == a.dtype and arrays['initial/'+key].shape == a.shape and
        arrays['initial/'+key].tobytes() == a.tobytes(), 'Retained initial tensor differs from original checkpoint')
  from ruamel.yaml import YAML
  configured=YAML(typ='safe').load(auth['paths']['config'].read_text(encoding='utf-8'))
  def flatten(value,prefix=''):
    require(type(value) is dict,'Original full config dictionary required');result={}
    for key,item in value.items():
      if type(item) is dict:result.update(flatten(item,prefix+key+'.'))
      else:result[prefix+key]=item
    return result
  configured=flatten(configured)
  configured.update(task='dmc_quadruped_walk',logdir=str(auth['run']),**{
      'jax.platform':'cuda','jax.policy_devices':[0],'jax.train_devices':[0],
      'jax.prealloc':False,'jax.precompile':False,'jax.profiler':False,
      'jax.enable_policy':False,'agent.rep_probe.mode':'off','agent.dt_latch.normal_only':True})
  for name,mode,rho in (('B','off',.5),('rho0','dtlatch',0.),('N','dtlatch',.5)):
    config=value['configs'][name]
    require(config == dict(configured,**{'agent.dt_latch.mode':mode,'agent.dt_latch.rho':rho}),
        'Complete original loss/optimizer/model/norm configuration differs')
    require(config['seed'] == 0 and config['task'] == 'dmc_quadruped_walk' and config['jax.platform'] == 'cuda' and
        config['jax.compute_dtype'] == 'bfloat16' and config['jax.enable_policy'] is False and
        config['batch_size'] == 16 and config['batch_length'] == 64 and config['replay_context'] == config['consec_train'] == 1 and
        config['agent.dyn.rssm.free_nats'] == 1 and config['agent.dt_latch.alpha'] == 20 and
        config['agent.dt_latch.rho'] == rho and config['agent.dt_latch.mode'] == mode and
        config['agent.dt_latch.normal_only'] is True and config['agent.rep_probe.mode'] == 'off',
        'Actual full configured branch differs')
    seal=value['variant_seals'][name]
    require(seal['full_state_sha256'] == FULLSTATE and seal['params_sha256'] == PARAMS and
        seal['counters'] == COUNTERS and seal['opt_step'] == 249493,'Common full initial variant differs')
  require(value['variant_seals']['B']['carry_sha256'] == value['variant_seals']['rho0']['carry_sha256'] ==
      value['variant_seals']['N']['carry_sha256'],'Common original carry differs across variants')
  cache=hashlib.sha256()
  for uid,chunk in sorted(source['chunks'].items()):cache.update(uid.encode());cache.update(typed_hash(chunk['data']).encode())
  require(value['cache_before']['sha256'] == cache.hexdigest() and value['cache_before']['pending'] is None,
      'Full historical source cache bytes differ')
  require(value['cache_before']['metrics_sha256'] == typed_hash(dict(samples=0,inserts=0,updates=0)),
      'Readonly replay metrics are not the original unsampled counters')
  receipt=value['cache_before']['receipt']
  require(receipt['original_inventory_authenticated'] is True and receipt['chunks'] == COUNTS[0] and
      receipt['legal65_starts'] == source['legal65_starts'] and receipt['batch_size'] == 16 and receipt['length'] == 64 and
      receipt['prefix'] == receipt['consec'] == 1 and receipt['private_cache'] is True and
      receipt['original_files_written'] is receipt['prefetch'] is receipt['online'] is False,
      'Authenticated original private replay receipt differs')
  return value,expected


def bf16_values(bits):
  """Lossless BF16-bit decoding for finite stored forward values, no ml/JAX."""
  require(type(bits) is np.ndarray and bits.dtype == np.uint16, 'BF16 raw uint16 bits required')
  require(not np.any((bits & np.uint16(0x7F80)) == np.uint16(0x7F80)),
      'Nonfinite original BF16 values')
  return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def threefry(key, count):
  """Twenty Threefry2x32 rounds, modulo2**32; even count arrays only."""
  require(type(key) is np.ndarray and key.dtype == np.uint32 and key.shape == (2,),
      'Threefry uint32 key pair required')
  require(type(count) is np.ndarray and count.dtype == np.uint32 and count.size % 2 == 0,
      'Threefry even uint32 counter array required')
  left, right = np.split(count.reshape(-1), 2)
  keys = (key[0], key[1], key[0] ^ key[1] ^ np.uint32(0x1BD11BDA))
  left, right = left + keys[0], right + keys[1]
  rotations = ((13, 15, 26, 6), (17, 29, 16, 24))
  for block in range(5):
    for rotation in rotations[block % 2]:
      left = left + right
      right = ((right << np.uint32(rotation)) |
          (right >> np.uint32(32 - rotation))) ^ left
    left = left + keys[(block + 1) % 3]
    right = right + keys[(block + 2) % 3] + np.uint32(block + 1)
  return np.concatenate((left, right)).reshape(count.shape)


def fold_in(key, value):
  require(type(value) is int and 0 <= value < 2 ** 32, 'Fold data must be uint32-valued int')
  return threefry(key, np.array([0, value], np.uint32))


def permutation_randomness():
  """Actual C P key: root0, tag0x444C, original fixed optstep249493."""
  key = fold_in(fold_in(np.array([0, 0], np.uint32), 0x444C), 249493)
  bits = threefry(key, np.arange(1024, dtype=np.uint32))
  floats = ((bits >> np.uint32(9)) | np.uint32(0x3F800000)).view(np.float32)
  return key, floats - np.float32(1)


def permutation(mask, raw_kl, discrepancy, d_bins, k_bins):
  """Independent actual global1024 strata/priority sort, invalid rows fixed."""
  require(mask.dtype == np.bool_ and mask.shape == raw_kl.shape == discrepancy.shape == (16, 64),
      'Actual16x64 P fields required')
  require(raw_kl.dtype == discrepancy.dtype == np.float32 and
      np.isfinite(raw_kl).all() and np.isfinite(discrepancy).all(), 'Finite F32 P fields required')
  d_bins, k_bins = np.asarray(d_bins, np.float32), np.asarray(k_bins, np.float32)
  require(d_bins.ndim == k_bins.ndim == 1 and np.isfinite(d_bins).all() and
      np.isfinite(k_bins).all(), 'Finite frozen one-dimensional P boundaries required')
  require(np.all(np.diff(d_bins) > 0) and np.all(np.diff(k_bins) > 0), 'Unique ordered P boundaries required')
  valid, kl, d = mask.reshape(-1), raw_kl.reshape(-1), discrepancy.reshape(-1)
  nd, nk = len(d_bins) + 1, len(k_bins) + 1
  group = (((valid.astype(np.int32) * 2 + (kl > 1)) * nd +
      np.searchsorted(d_bins, d, side='right')) * nk + np.searchsorted(k_bins, kl, side='right'))
  group = group.astype(np.int32)
  key, priorities = permutation_randomness()
  targets = np.argsort(group, kind='stable')
  sources = np.lexsort((priorities, group))
  perm = np.empty(1024, np.int32)
  perm[targets] = sources
  perm = np.where(valid, perm, np.arange(1024, dtype=np.int32))
  return dict(key=key, priorities=priorities, perm=perm.reshape(16, 64),
      group=group.reshape(16, 64), counts=np.bincount(group, minlength=4 * nd * nk).astype(np.int32))


def independent_freeze(signals):
  """One pooled F64 fsum, never a sum of rounded batch aggregates."""
  require(len(signals) == 128,'Complete128 signals required')
  a=np.concatenate([s['A'].reshape(-1) for s in signals])
  flat={k:np.concatenate([s[k].reshape(-1) for s in signals]) for k in ('E','e','f','D','K')}
  e,q,f=(flat[k][a] for k in ('E','e','f'))
  numerator=.5*math.fsum(float(x)*float(y)*float(z) for x,y,z in zip(e,q,f))
  ds=math.fsum(float(x)*float(y) for x,y in zip(e,f))
  dw=math.fsum(float(x)*float(y) for x,y in zip(e,q))
  def coefficient(den):
    value=numerator/den if den > 0 and math.isfinite(den) else None
    valid=value is not None and math.isfinite(value) and 0 <= value <= .5
    return dict(calibrated=valid,reason=None if valid else
        ('zero_denominator' if den == 0 else 'invalid_coefficient'),
        coefficient_f64=value if valid else None,
        coefficient_f32=float(np.float32(value)) if valid else None,denominator_f64=den)
  active=a & (flat['K'] > 1);count=int(active.sum())
  if count:
    dq=np.quantile(flat['D'][active].astype(np.float64),[1/3,2/3],method='linear')
    kq=np.quantile(flat['K'][active].astype(np.float64),[.5],method='linear')
    bins=dict(calibrated=True,reason=None,eligible_visits=count,
        D_quantiles_f64=dq.tolist(),K_quantiles_f64=kq.tolist(),
        D_f32=np.unique(dq.astype(np.float32)).tolist(),K_f32=np.unique(kq.astype(np.float32)).tolist())
  else:
    bins=dict(calibrated=False,reason='no_active_positions',eligible_visits=0,
        D_quantiles_f64=[],K_quantiles_f64=[],D_f32=[],K_f32=[])
  return dict(version='DL-NM1-calibration-r1',batches=128,positions=128*1024,
      aggregate64=dict(N_analytic_release=numerator,S_denominator=ds,W_denominator=dw),
      S=coefficient(ds),W=coefficient(dw),bins=bins)


def independent_summary(signals,effects,controls,frozen):
  require(len(signals) == len(effects) == len(controls) == 128,'Complete128 qualification object required')
  positions={};effect_batches=protected=changed=moved=0
  for s,e,c in zip(signals,effects,controls):
    effect_batches+=int((e['encoder_gradient_changed_elements'] > 0 and e['encoder_step_changed_elements'] > 0) or
        (e['posterior_gradient_changed_elements'] > 0 and e['posterior_step_changed_elements'] > 0))
    chosen=s['A'] & (s['K'] > 1) & (s['v_N'] < 1)
    for identifier,chain in zip(s['stepid'][chosen],s['chain'][chosen]):
      key=identifier.tobytes();chain=int(chain)
      require(key not in positions or positions[key] == chain,'One stepid belongs to multiple chains')
      positions[key]=chain
    p=s['A'] & (s['K'] > 1) & (s['v_N'] < np.float32(1-1e-6))
    protected+=int(p.sum())
    if c['v_P'] is not None:
      change=np.abs(c['v_P']-s['v_N']) > np.float32(1e-6)
      changed+=int((p & change).sum());moved+=int((s['A'] & change).sum())
  totals={}
  for name in ('N','S','W','P'):
    weights=[s['v_N'] for s in signals] if name == 'N' else [c['v_'+name] for c in controls]
    totals[name]=None if any(w is None for w in weights) else math.fsum(
        (1-float(v))*float(x) for s,w in zip(signals,weights) for v,x in
        zip(w.reshape(-1),s['E'].reshape(-1)))
  drifts={name:abs(totals[name]-totals['N'])/totals['N']
      if totals[name] is not None and totals['N'] > 0 else None for name in ('S','W','P')}
  fraction=changed/protected if protected else None
  coverage=len(positions) >= 128 and len(set(positions.values())) >= 8
  qualification=dict(C_N_coverage=coverage,C_N_effect=effect_batches >= 16,
      C_N=coverage and effect_batches >= 16,C_S=drifts['S'] is not None and drifts['S'] <= .01,
      C_W=drifts['W'] is not None and drifts['W'] <= .01,
      C_P=fraction is not None and fraction >= .5 and drifts['P'] is not None and drifts['P'] <= .10)
  return dict(version='DL-NM1-calibration-r1',batches=128,distinct_protected_stepids=len(positions),
      distinct_protected_chains=len(set(positions.values())),numerical_effect_batches=effect_batches,
      release_actual_f64=totals,relative_release_drift=drifts,P_protected_visits=protected,
      P_changed_protected_visits=changed,P_changed_protected_fraction=fraction,
      P_changed_intervention_visits=moved,qualification=qualification,
      T1_necessary_qualification=bool(qualification['C_N'] and qualification['C_S']),
      T2_control_qualification=bool(qualification['C_P'] and qualification['C_W']),
      subsequent_learning_authorized=False,method_efficacy_established=False)


def effect_tensors(initial,arrays):
  """Independent retained enc/post value arithmetic; never infer unstored leaves."""
  output={};stats={}
  for group,prefix in (('encoder','enc/'),('posterior','dyn/obs')):
    keys=sorted(k for k in initial if k.startswith(prefix));require(keys,'Retained '+group+' tensors required')
    grad_count=step_count=0;g2=[];u2=[];b2=[];n2=[];dots=[]
    detail={name:dict(elements=0,reference_zero_elements=0,max_abs_difference=0.,
        max_reference_ulp_difference=0.,max_nonzero_reference_ulp_difference=0.,byte_identical=True)
        for name in ('gradient','parameter')}
    for key in keys:
      values=[initial[key],*(arrays[name+'/'+key] for name in ('grad_B','grad_N','step_B','step_N'))]
      require(all(type(v) is np.ndarray and v.dtype == np.float32 and
          v.shape == values[0].shape and np.isfinite(v).all() for v in values),'Finite matched F32 effect tensors required')
      start,gb,gn,b,n=[v.astype(np.float64) for v in values]
      grad_count+=int(np.count_nonzero(gb != gn));step_count+=int(np.count_nonzero(b != n))
      for name,left32,right32 in (('gradient',values[1],values[2]),('parameter',values[3],values[4])):
        info=detail[name];difference=np.abs(right32.astype(np.float64)-left32.astype(np.float64))
        exponent=((left32.view(np.uint32) >> 23) & 255).astype(np.int32)
        ulp=difference/np.exp2(np.maximum(exponent-150,-149).astype(np.float64))
        info['elements']+=left32.size;info['reference_zero_elements']+=int(np.count_nonzero(left32 == 0))
        info['byte_identical'] &= left32.tobytes() == right32.tobytes()
        for key,value in (('max_abs_difference',difference),('max_reference_ulp_difference',ulp),
            ('max_nonzero_reference_ulp_difference',ulp[left32 != 0])):
          info[key]=max(info[key],float(np.max(value,initial=0)))
      dg,db,dn=gn-gb,b-start,n-start;du=dn-db
      g2.append(float(np.dot(dg.reshape(-1),dg.reshape(-1))))
      u2.append(float(np.dot(du.reshape(-1),du.reshape(-1))))
      b2.append(float(np.dot(db.reshape(-1),db.reshape(-1))))
      n2.append(float(np.dot(dn.reshape(-1),dn.reshape(-1))))
      dots.append(float(np.dot(db.reshape(-1),dn.reshape(-1))))
    left,right=math.fsum(b2),math.fsum(n2)
    output[group+'_gradient_changed_elements']=grad_count;output[group+'_step_changed_elements']=step_count
    stats[group]=dict(gradient_difference_l2=math.sqrt(math.fsum(g2)),update_difference_l2=math.sqrt(math.fsum(u2)),
        B_update_l2=math.sqrt(left),N_update_l2=math.sqrt(right),
        update_cosine=math.fsum(dots)/math.sqrt(left*right) if left and right else None,
        retained_tensors=len(keys),gradient=detail['gradient'],parameter=detail['parameter'])
  return output,stats


class Stored:
  def __init__(self,name):self.name=name


def wire_hash(tree,arrays,specs):
  """Original dtype/bytes of stored BF16 and numeric trees, without ml_dtypes."""
  digest=hashlib.sha256()
  def visit(value):
    if type(value) is dict:
      digest.update(b'dict')
      for key in sorted(value):digest.update(json.dumps(key).encode());visit(value[key])
    elif type(value) in (tuple,list):
      digest.update(type(value).__name__.encode());digest.update(str(len(value)).encode())
      for child in value:visit(child)
    else:
      require(type(value) is Stored,'Stored typed leaf required');s=specs[value.name];a=arrays[value.name]
      digest.update(json.dumps([s['dtype_str'],s['logical_dtype']]).encode())
      digest.update(json.dumps(a.shape).encode());digest.update(a.tobytes())
  visit(tree);return digest.hexdigest()


def flat_stored(specs,prefix):
  return {k[len(prefix):]:Stored(k) for k in specs if k.startswith(prefix)}


def same_stored(arrays,specs,left,right,fields):
  return all({k:v for k,v in specs[left+k].items() if k != 'storage_key'} ==
      {k:v for k,v in specs[right+k].items() if k != 'storage_key'} and
      arrays[left+k].tobytes() == arrays[right+k].tobytes() for k in fields)


def audit_batch(auth,source,initial,selected,index):
  directory=auth['run']/f'batch_{index:03d}';record=read_json(directory/'record.json')
  require(record['version'] == SCIENCE and record['schema'] == 1 and record['batch'] == index and
      record['counter'] == 249493+index and record['opt_step_before'] == 249493,'Fixed independent calibration counter differs')
  specs=record['arrays'];arrays=load_pack(directory/'evidence.npz',specs)
  raw,chain=reconstruct_batch(source,source['plan'][index])
  expected={'raw/'+k for k in raw}|{'prepared_key'}|{'initial_carry/'+k for k in ('1/deter','1/stoch','3/action')}|\
      {'signal/'+k for k in ('K','E','D','e','f','v_N','m','A','stepid','chain')}|\
      {prefix+'/'+k for prefix in ('grad_B','grad_N','step_B','step_N') for k in selected}|\
      {prefix+'/'+k for prefix in ('replay_B','replay_N') for k in ('stepid','dyn/deter','dyn/stoch')}|\
      {prefix+'/'+k for prefix in ('reference_replay_B','reference_replay_N') for k in ('dyn/deter','dyn/stoch')}|\
      {'reference_B/'+k for k in ('K','E','rep_before','weight')}|{'reference_N/rep_before'}
  require(set(arrays) == expected,'Exact complete batch logical tensor set differs')
  for key,value in raw.items():
    require(specs['raw/'+key]['logical_dtype'] == str(value.dtype) and
        arrays['raw/'+key].shape == value.shape and arrays['raw/'+key].tobytes() == value.tobytes(),
        'Independent source65/context/flags/action differs: '+key)
  seed=np.random.default_rng([0,249493+index]).integers(0,np.iinfo(np.uint32).max,(2,),np.uint32)
  require(arrays['prepared_key'].dtype == np.uint32 and arrays['prepared_key'].shape == (2,) and
      arrays['prepared_key'].tobytes() == seed.tobytes(),
      'Original public training key differs')
  require(record['raw_sha256'] == typed_hash(raw) and record['prepared_sha256'] == typed_hash(dict(raw,seed=seed)),
      'Actual raw/prepared source hashes differ')
  carry=({},dict(deter=Stored('initial_carry/1/deter'),stoch=Stored('initial_carry/1/stoch')),{},
      dict(action=Stored('initial_carry/3/action')))
  require(record['carry_sha256'] == wire_hash(carry,arrays,specs) == initial['variant_seals']['B']['carry_sha256'],
      'Common input carry differs')
  for key,tail in (('1/deter',RAW_FIELDS['dyn/deter'][1]),('1/stoch',RAW_FIELDS['dyn/stoch'][1]),('3/action',RAW_FIELDS['action'][1])):
    require(arrays['initial_carry/'+key].shape == (16,*tail) and
        not arrays['initial_carry/'+key].any(),'Original public init carry shape/zero differs')
    require(specs['initial_carry/'+key]['logical_dtype'] == ('float32' if key == '3/action' else 'bfloat16'),
        'Original configured initial carry logical dtype differs')
  context=({},dict(deter=raw['dyn/deter'][:,0],stoch=raw['dyn/stoch'][:,0]),{})
  obs={k:raw[k][:,1:] for k in ('image','reward','is_first','is_last','is_terminal')}
  require(record['context_sha256'] == typed_hash((context,obs,dict(action=raw['action'][:,:-1]),raw['stepid'][:,1:])),
      'Actual applied consec0 context/previous raw/target alignment differs')
  signal={k:arrays['signal/'+k] for k in ('K','E','D','e','f','v_N','m','A','stepid','chain')}
  require(all(signal[k].dtype == np.bool_ and signal[k].shape == (16,64) for k in ('m','A')),
      'Actual boolean m/A shape/dtype differs')
  require(signal['stepid'].dtype == np.uint8 and signal['stepid'].shape == (16,64,20) and
      signal['chain'].dtype == np.int32 and signal['chain'].shape == (16,64),
      'Full64 typed stepid/source chain shape differs')
  m=np.concatenate(((~raw['is_last'][:,1:-1]) & (~raw['is_first'][:,2:]),np.zeros((16,1),bool)),1)
  require(np.array_equal(signal['m'],m) and np.array_equal(signal['A'],m & ~raw['is_first'][:,1:]),'Actual m/A source semantics differ')
  require(signal['stepid'].tobytes() == raw['stepid'][:,1:].tobytes() and
      signal['chain'].dtype == np.int32 and signal['chain'].tobytes() == chain.tobytes(),'Full64 stepid/source chain differs')
  for key in ('K','E','D','e','f','v_N'):
    require(signal[key].dtype == np.float32 and signal[key].shape == (16,64),'Actual F32 signal shape/dtype differs')
  require(np.all(signal['K'] >= -1e-5) and np.all(signal['D'] >= -1e-5) and np.all(signal['D'][m] >= 0) and
      np.array_equal(signal['E'],np.maximum(signal['K']-np.float32(1),np.float32(0))),
      'Actual raw KL tolerance/E relation differs')
  for key in ('e','f','v_N'):require(np.all((signal[key] >= 0) & (signal[key] <= 1)),'Actual unit signal range differs')
  for key in ('reference_B/K','reference_B/E','reference_B/rep_before','reference_B/weight','reference_N/rep_before'):
    require(arrays[key].dtype == np.float32 and arrays[key].shape == (16,64),'Original reference outlet dtype/shape differs')
  require(np.all(signal['v_N'][~signal['A']] == 1),'N changed a position outside A')
  require(np.all(arrays['reference_B/weight'] == 1) and
      np.array_equal(arrays['reference_B/E'],np.maximum(arrays['reference_B/K']-np.float32(1),0)) and
      np.array_equal(arrays['reference_B/rep_before'],np.maximum(arrays['reference_B/K'],np.float32(1))) and
      np.array_equal(arrays['reference_N/rep_before'],np.maximum(signal['K'],np.float32(1))),
      'Reference-only original raw/floor/weight outlet differs')
  effects,statistics=effect_tensors(selected,arrays)
  require(effects == record['effects'] and all(type(x) is int for x in record['effects'].values()),
      'Independent retained gradient/step numerical effects differ')
  require(set(record['production']) == {'B','B_repeat','rho0','N'},'Four complete production receipts required')
  base=record['production']['B']
  require(base == record['production']['B_repeat'] == record['production']['rho0'] and
      record['identity_checks'] == dict(B_repeat_equal=True,rho0_equal=True),'Original B/B or rho0 full byte identity failed')
  for name in ('B','N'):
    receipt=record['production'][name]
    require(all(type(receipt[k]) is str and re.fullmatch('[0-9a-f]{64}',receipt[k]) for k in
        ('state_sha256','carry_sha256','replay_sha256','metrics_sha256')),'Complete production typed hashes required')
    require(set(receipt['params']) == set(initial['state_leaves']),'Complete294 scratch leaf receipt required')
    for key,s in receipt['params'].items():
      require({k:v for k,v in s.items() if k != 'typed_sha256'} ==
          {k:v for k,v in initial['state_leaves'][key].items() if k != 'typed_sha256'},'Scratch original leaf schema changed')
      if key in selected:
        require(s == {k:v for k,v in specs['step_'+name+'/'+key].items() if k != 'storage_key'},'Stored actual step tensor/receipt differs')
    step=receipt['params']['opt/step/value'];original_step=auth['original_params']['opt/step/value']
    require(step['typed_sha256'] == typed_hash(np.array(249494,dtype=original_step.dtype)),
        'Scratch optimizer was not exactly one update')
    replay=flat_stored(specs,'replay_'+name+'/')
    require(receipt['replay_sha256'] == wire_hash(replay,arrays,specs),'Actual full production replay hash differs')
    require(arrays['replay_'+name+'/stepid'].dtype == np.uint8 and
        arrays['replay_'+name+'/stepid'].shape == (16,64,20) and
        arrays['replay_'+name+'/stepid'].tobytes() == raw['stepid'][:,1:].tobytes(),'Actual replay64 targets differ')
    for key in ('dyn/deter','dyn/stoch'):
      for prefix in ('replay_'+name,'reference_replay_'+name):
        require(arrays[prefix+'/'+key].shape == (16,64,*RAW_FIELDS[key][1]) and
            specs[prefix+'/'+key]['logical_dtype'] == 'bfloat16','Actual full forward tensor shape/dtype differs')
    ref=record['reference'][name]
    require(ref['state_scope'] == 'separate-complete-reference-JIT-not-production-intermediate' and
        all(k in ref['accessed'] for k in selected) and
        all(k in initial['state_leaves'] and k.startswith(('retnorm/','valnorm/','advnorm/')) for k in ref['modified']),
        'Independent reference accessed/written state scope differs')
    forward=({},dict(deter=Stored('reference_replay_'+name+'/dyn/deter'),stoch=Stored('reference_replay_'+name+'/dyn/stoch')),{})
    require(ref['forward_sha256'] == wire_hash(forward,arrays,specs),'Original reference forward hash differs')
  fields=('dyn/deter','dyn/stoch')
  forward=dict(production_replay_equal=same_stored(arrays,specs,'replay_B/','replay_N/',fields),
      reference_replay_equal=same_stored(arrays,specs,'reference_replay_B/','reference_replay_N/',fields),
      production_vs_reference_B_equal=same_stored(arrays,specs,'replay_B/','reference_replay_B/',fields),
      production_vs_reference_N_equal=same_stored(arrays,specs,'replay_N/','reference_replay_N/',fields))
  require(forward == record['forward'],'Actual stored forward alignment differs')
  return signal,effects,statistics,forward


def audit_control(auth,index,signal,frozen):
  directory=auth['run']/f'batch_{index:03d}';record=read_json(directory/'controls.json')
  require(record['version'] == SCIENCE and record['schema'] == 1 and record['batch'] == index and
      record['p_step'] == 249493 and record['p_seed'] == 0 and
      record['frozen_sha256'] == auth['files']['frozen.json']['sha256'],'Original fixed control freeze/key differs')
  arrays=load_pack(directory/'controls.npz',record['arrays']);result={};expected={'P_key','P_priorities'}
  key,priorities=permutation_randomness()
  require(arrays['P_key'].dtype == np.uint32 and arrays['P_key'].shape == (2,) and
      arrays['P_key'].tobytes() == key.tobytes() and arrays['P_priorities'].dtype == np.float32 and
      arrays['P_priorities'].shape == (1024,) and arrays['P_priorities'].tobytes() == priorities.tobytes(),
      'Independent Threefry key/uniform priorities differ')
  for name in ('S','W','P'):
    entry=frozen['bins' if name == 'P' else name];available=record['availability']['v_'+name]
    require(available == dict(available=entry['calibrated'],reason=entry['reason']),'Uncalibrated control availability differs')
    result['v_'+name]=None
    if not entry['calibrated']:continue
    expected.add('v_'+name);value=arrays['v_'+name]
    require(value.dtype == np.float32 and value.shape == (16,64) and np.all((value >= 0) & (value <= 1)) and
        np.all(value[~signal['A']] == 1),'Actual F32 control range/eligibility differs')
    if name in ('S','W'):
      factor=signal['f' if name == 'S' else 'e'];coefficient=np.float32(entry['coefficient_f32'])
      unfused=np.float32(1)-signal['A']*coefficient*factor
      fused=(1-signal['A']*float(coefficient)*factor.astype(np.float64)).astype(np.float32)
      # This checks the two legal F32 contraction forms, not GPU bit identity.
      require(np.all((value == unfused) | (value == fused)),
          'Stored S/W control is neither legal fused nor unfused F32 formula')
    result['v_'+name]=value
  if frozen['bins']['calibrated']:
    expected|={'P_perm','P_group','P_counts'}
    p=permutation(signal['A'],signal['K'],signal['D'],frozen['bins']['D_f32'],frozen['bins']['K_f32'])
    for key,name in (('perm','P_perm'),('group','P_group'),('counts','P_counts')):
      require(arrays[name].dtype == np.int32 and arrays[name].shape == p[key].shape and arrays[name].tobytes() == p[key].tobytes(),
          'Independent original global P '+key+' differs')
    require(np.array_equal(result['v_P'],signal['v_N'].reshape(-1)[p['perm']]),'Actual P is not the true fixed-strata weight gather')
  require(set(arrays) == expected,'Exact optional control tensor set differs')
  return result


def analyze(run,run_git,audit_git,launcher_path,launcher_sha):
  auth=authenticate(run,run_git,audit_git,launcher_path,launcher_sha)
  sample_count=resources(auth)
  source=replay_source(auth['directory'],auth['report'])
  original,loaded=checkpoint(auth['paths']['checkpoint'].read_bytes())
  auth['original_params']=loaded['params']
  initial,selected=selected_initial(auth,original,loaded,source)
  plan=read_json(auth['run']/'plan.json')
  require(plan == dict(version=SCIENCE,schema=1,batches=128,seed=source['seed'],starts=source['plan']),
      'Independent fixed128 PCG64/legal65 plan differs')
  signals=[];effects=[];statistics=[];forwards=[]
  for i in range(128):
    check();s,e,stat,forward=audit_batch(auth,source,initial,selected,i)
    signals.append(s);effects.append(e);statistics.append(stat);forwards.append(forward)
  frozen=independent_freeze(signals)
  require(frozen == read_json(auth['run']/'frozen.json'),'Independent pooled F64 freeze/boundaries differ')
  controls=[audit_control(auth,i,s,frozen) for i,s in enumerate(signals)]
  summary=independent_summary(signals,effects,controls,frozen)
  require(summary == read_json(auth['run']/'summary.json'),'Independent complete C qualifications/pooled measurements differ')
  final=read_json(auth['run']/'final.json')
  require(final['version'] == SCIENCE and final['schema'] == 1 and final['complete_batches'] == 128 and
      final['optimizer_updates_committed'] == final['replay_updates_committed'] == 0 and
      final['cache_after'] == initial['cache_before'] and final['inputs_after'] == auth['budget']['inputs'] and
      final['source_files_after'] == auth['budget']['source_files'] and
      final['original_source_after'] == auth['report']['before_inventory'],'Original complete readonly final closure differs')
  require(set(final['agent_seals']) == {'B','rho0','N'},'Three complete original final variants required')
  for name,value in final['agent_seals'].items():
    require(value == dict(initial['variant_seals'][name],pending_clean=True,source_buffers_unchanged=True),
        'Original full model/counters/buffers/pending changed')
  fwd=final['forward_summary']
  require(fwd['production_replay_equal_batches'] == sum(x['production_replay_equal'] for x in forwards) and
      fwd['reference_replay_equal_batches'] == sum(x['reference_replay_equal'] for x in forwards) and
      fwd['only_rep_pressure_isolated'] is False and type(fwd['interpretation']) is str,
      'Stored forward alignment interpretation/counts differ')
  # Everything above is conditional on one complete fixed procedure. Close all
  # source/run bytes again, including unselected original replay chunks.
  for name,expected in source['inventory'].items():
    require(raw_hash(auth['directory']/name) == expected,'Immutable original replay changed during audit')
  for pin in auth['budget']['inputs'].values():
    if pin is not None:pinned(auth['run'],pin)
  require(inventory(auth['run']) == auth['files'],'Original complete run changed during CPU audit')
  git_binding(run_git,audit_git,auth['budget']['source_files']);check()
  require(not OWNED,'Independent audit left owned helpers')
  return dict(version=VERSION,run_git=run_git,audit_git=audit_git,audit_integrity_passed=True,
      scientific_qualification=all(summary['qualification'][k] for k in ('C_N','C_S','C_W','C_P')),
      status='complete-finite-C-audit',batches=128,legal65_starts=source['legal65_starts'],
      resource_samples=sample_count,independent_frozen=frozen,independent_summary=summary,
      retained_tensor_statistics=statistics,stored_forward=forwards,
      actual_inputs_authenticated=(auth['budget']['inputs']['checkpoint']['sha256'] == ORIGINAL_CHECKPOINT),
      subsequent_learning_authorized=False,method_efficacy_established=False,NN_reexecuted=False,
      backward_reexecuted=False,Adam_reexecuted=False,GPU_used=False,
      elapsed_seconds=time.monotonic()-ENTRY_STARTED,whole_limit_seconds=900,
      remaining_owned=[],limitations=[
        'Full B/B/rho0 state/carry/common metrics without stored full arrays are authenticated producer receipts, not independently recomputed NN outputs.',
        'Only retained encoder/posterior gradients and parameters have independent value arithmetic; unstored Adam/full-state updates are not replayed.',
        'Original GPU KL/exp/f/N gate values are authenticated arrays, not CPU reconstructions or claims of GPU operator bit identity.',
        'S/W weights satisfy fused-or-unfused F32 formula membership; this is not proof of the GPU compiler contraction choice.',
        'Resource telemetry/transport host and execution receipts are authenticated records; this CPU audit does not query a GPU or independently prove remote SSH history.',
        'Stored replay equality does not isolate all forward paths or establish a pure compression-information mechanism.',
        'C qualification is finite engineering/operational evidence and does not authorize learning or demonstrate method efficacy.'])


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  for name in ('run','expected-git','run-git','launcher-terminal','launcher-sha','output'):
    parser.add_argument('--'+name,required=True)
  args=parser.parse_args();requested=Path(args.output).absolute();output=plain(requested.parent)/requested.name
  require(not output.exists() and not output.is_symlink(),'Fresh independent audit output required')
  require(plain(args.run) not in output.parents and plain(args.run) != output,'Output must stay outside original run')
  try:
    require(os.name == 'posix','Real independent CLI requires Linux CPU')
    require(np.__version__ == '1.26.4','Frozen NumPy1.26.4 required')
    value=analyze(args.run,args.run_git,args.expected_git,args.launcher_terminal,args.launcher_sha)
  except Exception as error:
    value=dict(version=VERSION,audit_integrity_passed=False,status='integrity-rejected',error=repr(error),
        elapsed_seconds=time.monotonic()-ENTRY_STARTED,remaining_owned=sorted(OWNED))
    with output.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
    raise
  with output.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
  print(json.dumps(dict(status=value['status'],audit_integrity_passed=True,
      scientific_qualification=value['scientific_qualification'],elapsed_seconds=value['elapsed_seconds'])),flush=True)


if __name__ == '__main__':main()
