"""Strict private adaptation of the authenticated offline persistent replay.

No Replay.load(), original-file writes, prefetch, model, environment or training.
The caller freezes a common explicit (original chunk UUID, index) plan across
arms and follows prepare_batch -> train -> take_train_result -> strict_update.
Historical saved context is retained; this does not restore the online replay,
sampler RNG, queue, environment or an exact end-of-training context cache.
"""
import hashlib
import io
from pathlib import Path
import zipfile

import numpy as np

from scripts import dl_replay_inventory as inventory

VERSION = 'DL-normal-private-replay-r1'
INVENTORY_SHA = '3339b71ca033d5fc065130751fe163aca05a684b2f14f570ab2018219a3a706d'
EXPECTED_COUNTS = (1280, 16, 1001008, 1008, 1000000)
BATCH, LENGTH, CONTEXT = 16, 64, 1
CACHE = {'dyn/deter': (4096,), 'dyn/stoch': (32,32)}
_VERIFIED = object()
require = inventory.require


def _digest(data):
  digest = hashlib.sha256()
  for key,value in sorted(data.items()):
    digest.update(key.encode()+str(value.dtype).encode()+str(value.shape).encode())
    digest.update(value.tobytes())
  return digest.hexdigest()


def _array(value,dtype,shape,name):
  require(type(value) is np.ndarray and value.dtype == np.dtype(dtype) and
      value.shape == shape and np.isfinite(value).all(),'Invalid '+name+' dtype/shape/finite')


def _context(value):
  require(np.all((value == 0) | (value == 1)) and np.all(value.sum(-1) == 1),
      'Cached stochastic context must be actual onehot, including reset rows')


def _decode(body,row):
  """Decode precisely the already raw-SHA-authenticated bytes, never reopen."""
  length=row['length']
  with zipfile.ZipFile(io.BytesIO(body)) as archive:
    names=archive.namelist()
    require(len(names) == len(inventory.FIELDS) and set(names) ==
        {name+'.npy' for name in inventory.FIELDS},'NPZ member schema/duplicates differ')
    require(archive.testzip() is None,'NPZ CRC failure')
  with np.load(io.BytesIO(body),allow_pickle=False) as source:
    require(set(source.files) == set(inventory.FIELDS),'NPZ source schema differs')
    arrays={key:source[key] for key in inventory.FIELDS}
  for key,(dtype,tail) in inventory.FIELDS.items():
    _array(arrays[key],dtype,(length,*tail),'source '+key)
  ids=arrays['stepid']
  require(np.all(ids[:,:16] == np.frombuffer(inventory.uuid_bytes(row['uuid']),np.uint8)[None]) and
      np.array_equal(ids[:,16:],np.arange(length,dtype='>u4').view(np.uint8).reshape(length,4)),
      'Original filename UUID/index and full stepid differ')
  first,last,terminal=(arrays[key] for key in ('is_first','is_last','is_terminal'))
  require(np.all(~terminal | last) and np.array_equal(first[1:],last[:-1]) and
      np.all(arrays['reward'][first] == 0) and np.all(arrays['action'][last] == 0),
      'Original episode/reset semantics differ')
  _context(arrays['dyn/stoch'])
  summary=dict(row,first=bool(first[0]),last=bool(last[-1]),resets=int(first.sum()),
      terminals=int(terminal.sum()),ends=int(last.sum()),crc_verified=True,cached_context_validated=True)
  return arrays,summary


def load_private_cache(replay_directory, inventory_path, *, expected_inventory_sha256=INVENTORY_SHA,
                       stage_record_path=None, expected_stage_record_sha256=None):
  """Fresh complete private cache; a nondefault SHA is only an explicit fixture pin.

  Production counts/schema/source remain fixed. Small tests explicitly patch
  counts and graph expectations; their receipts cannot authenticate N1M replay.
  All fields are separately decoded for each invocation/arm, without sharing
  mutable caches or retaining an open NPZ handle.
  """
  directory=inventory.plain(replay_directory); path=inventory.plain(inventory_path)
  body=path.read_bytes()
  require(hashlib.sha256(body).hexdigest() == expected_inventory_sha256,'Inventory raw SHA differs before JSON decode')
  report=inventory.read_json(body)
  require((stage_record_path is None) == (expected_stage_record_sha256 is None),
      'Explicit stage record path and raw SHA must be supplied together')
  stage_provenance=None
  if stage_record_path is not None:
    from scripts import dl_replay_stage as stage
    stage_provenance=stage.authenticate_stage(stage_record_path,expected_stage_record_sha256,
        path,report,directory)
  original_directory=(str(directory) if stage_provenance is None else
      stage_provenance['original_replay_directory'])
  count,chains,rows,resets,actions=EXPECTED_COUNTS
  require(report['version'] == inventory.VERSION and report['schema'] == 1 and report['stable'] is True and
      report['source_git'] == inventory.SOURCE_GIT and report['replay_directory'] == original_directory and
      report['run_directory'] == str(Path(original_directory).parent) and report['before_inventory'] == report['after_inventory'] and
      report['rows'] == rows and report['resets'] == resets and report['train_actions'] == actions and
      report['corrupt_chunks_skipped'] == 0 and report['model_loaded'] is report['gpu_used'] is False and
      all(report[key] is False for key in ('sampler_rng_restored','online_queue_restored',
          'environment_restored','seamless_resume_established')),'Frozen offline inventory provenance/counts differ')
  require(report['fields'] == {k:[v[0],list(v[1])] for k,v in inventory.FIELDS.items()},'Frozen field schema differs')
  entries=report['before_inventory']
  require(type(entries) is list and len(entries) == count and
      [x['path'] for x in entries] == sorted({x['path'] for x in entries}) and
      len(report['chunks']) == count and len(report['chains']) == chains,'Complete unique sorted inventory required')
  require({p.name for p in directory.iterdir()} == {e['path'] for e in entries},'Missing/extra persistent source file')
  from embodied.core.chunk import Chunk
  import elements
  chunks,actual=[],[]
  for entry in entries:
    require(set(entry) == {'path','bytes','sha256'} and Path(entry['path']).name == entry['path'],
        'Unsafe chunk inventory entry')
    row=inventory.filename(entry['path']); source=inventory.plain(directory/entry['path'])
    require(source.is_file() and source.parent == directory,'Regular contained source chunk required')
    raw=source.read_bytes()
    require(len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256'],
        'Source chunk raw hash differs before NPZ decode: '+entry['path'])
    data,summary=_decode(raw,row)
    chunk=Chunk(row['length'])
    chunk.time=row['time'];chunk.uuid=elements.UUID(row['uuid']);chunk.succ=elements.UUID(row['successor'])
    chunk.length=row['length'];chunk.data=data;chunk.saved=True
    for key in set(data)-set(CACHE): data[key].setflags(write=False)
    chunks.append(chunk);actual.append(summary)
  require(actual == report['chunks'] and inventory.validate_chains(actual) == report['chains'],
      'Decoded complete chunk/chain metadata differs from pinned inventory')
  require(sum(x['length'] for x in actual) == rows and sum(x['resets'] for x in actual) == resets and
      rows-resets == actions,'Decoded original replay totals differ')
  return PrivateReplay(chunks,report,expected_inventory_sha256,_verified=_VERIFIED,
      _stage_provenance=stage_provenance)


class PrivateReplay:
  """Single-threaded cache; one outstanding submitted raw batch at a time."""
  def __init__(self,chunks,report,inventory_sha256,*,_verified=None,_stage_provenance=None):
    require(_verified is _VERIFIED,'Use the strict inventory/byte-authenticated loader')
    from embodied.core.replay import Replay
    self._replay=Replay(length=LENGTH+CONTEXT,capacity=None,directory=None,online=False,seed=0)
    self._report=report;self._inventory_sha=inventory_sha256
    self._stage_provenance=_stage_provenance
    self._pending=None;self._poisoned=False
    ordered=sorted(chunks,key=lambda x:x.filename,reverse=True)
    counts=self._replay._numitems(ordered)
    require(len(counts) == len(chunks),'Duplicate original chunk UUID')
    self._replay.chunks={chunk.uuid:chunk for chunk in ordered}
    self._replay.refs={chunk.uuid:0 for chunk in ordered}
    self._counts={}
    # Independently check official filename-based availability against the
    # authenticated graph before registering any start. No zero context/tail.
    for chain in report['chains']:
      remaining=0
      for name in reversed(chain['paths']):
        row=inventory.filename(name);remaining+=row['length']
        import elements
        uuid=elements.UUID(row['uuid'])
        expected=min(row['length'],max(0,remaining-(LENGTH+CONTEXT)+1))
        require(int(counts[uuid]) == expected,'Official _numitems and complete legal65-start graph differ')
        self._counts[uuid]=expected
    for chunk in reversed(ordered):
      amount=self._counts[chunk.uuid];self._replay.refs[chunk.uuid]+=amount
      if chunk.succ in self._replay.refs: self._replay.refs[chunk.succ]+=1
      for index in range(amount): self._replay._insert(chunk.uuid,index)
    require(len(self._replay) == sum(c['length65_windows'] for c in report['chains']) and len(self._replay) > 0,
        'Official registered legal65 starts differ')

  def receipt(self):
    result=dict(version=VERSION,inventory_sha256=self._inventory_sha,
        original_inventory_authenticated=self._inventory_sha == INVENTORY_SHA,
        chunks=len(self._replay.chunks),legal65_starts=len(self._replay),
        batch_size=BATCH,length=LENGTH,prefix=CONTEXT,consec=1,prefetch=False,
        directory=None,online=False,private_cache=True,original_files_written=False,
        sampler_rng_restored=False,online_queue_restored=False,environment_restored=False,
        exact_endpoint_cache_restored=False,object_scope=self._report['object_scope'])
    if self._stage_provenance is not None:
      # Extra provenance is explicit only for relocation. The original path's
      # receipt remains byte/field compatible, including its version string.
      import copy
      result['stage_provenance']=copy.deepcopy(self._stage_provenance)
    return result

  def make_plan(self,seed,updates):
    """Explicit uniform plan in official insertion order, not restored RNG."""
    require(type(seed) is int and seed >= 0 and type(updates) is int and updates > 0,
        'Explicit nonnegative plan seed and positive updates required')
    draws=np.random.default_rng(seed).integers(0,len(self._replay),(updates,BATCH))
    return tuple(tuple((str(self._replay.items[int(i)][0]),self._replay.items[int(i)][1]) for i in row) for row in draws)

  def _start(self,value):
    import elements
    require(type(value) in (tuple,list) and len(value) == 2 and type(value[0]) is str and
        type(value[1]) is int,'Original (UUID string,index) start required')
    inventory.uuid_bytes(value[0]);uuid=elements.UUID(value[0]);index=value[1]
    require(uuid in self._counts and 0 <= index < self._counts[uuid],'Unknown/illegal original65 start')
    return uuid,index

  def _assemble_starts(self,starts):
    from embodied.core import streams
    require(not self._poisoned and self._pending is None,'Pending batch must be updated before another batch')
    require(type(starts) in (tuple,list) and len(starts) == BATCH,'Exactly16 planned starts required')
    starts=tuple(self._start(value) for value in starts)
    seqs=[self._replay._getseq(uuid,index,concat=False) for uuid,index in starts]
    raw=self._replay._assemble_batch(seqs,0,LENGTH+CONTEXT)
    annotated=self._replay._annotate_batch(raw,(False,)*BATCH,True)
    stream=iter(streams.Consec(streams.Stateless(lambda:annotated),length=LENGTH,
        consec=1,prefix=CONTEXT,strict=True,contiguous=True))
    result=next(stream)
    return result,starts

  def read_batch(self,starts):
    """DL-NM1: normal annotated batch without sampling counts or pending update.

    Calibration reads the frozen historical cache. It neither pretends to
    consume a training batch nor writes a synthetic network output back.
    """
    result,_=self._assemble_starts(starts)
    return result

  def batch(self,starts):
    result,starts=self._assemble_starts(starts)
    self._replay.metrics['samples']+=BATCH
    self._pending=dict(starts=starts,ids=result['stepid'].copy(),batch_sha256=_digest(result))
    return result

  def _span(self,uuid,index,length):
    targets=[]
    while length:
      require(uuid in self._replay.chunks,'Update would silently lose unknown chunk')
      chunk=self._replay.chunks[uuid]
      require(0 <= index < chunk.length,'Update index outside original chunk')
      used=min(length,chunk.length-index)
      targets.extend((uuid,index+i) for i in range(used))
      length-=used;uuid,index=chunk.succ,0
    return targets

  def strict_update(self,out,submitted_raw_batch):
    """Prevalidate full65 input / exact64 outputs, then official last-writer order.

    No mutation occurs for a validation error. Unexpected application or
    verification failure poisons this arm instead of continuing a partial cache.
    The original Replay.update swallows KeyError; all routes are prevalidated
    here and the final value at every written position is checked afterwards.
    """
    require(not self._poisoned and self._pending is not None,'No outstanding unconsumed batch')
    pending=self._pending;size=LENGTH+CONTEXT
    require(type(submitted_raw_batch) is dict and set(submitted_raw_batch) == set(inventory.FIELDS)|{'consec'},
        'Submitted raw batch schema differs')
    for key,(dtype,tail) in inventory.FIELDS.items():
      _array(submitted_raw_batch[key],dtype,(BATCH,size,*tail),'submitted '+key)
    _array(submitted_raw_batch['consec'],np.int32,(BATCH,size),'submitted consec')
    require(not np.any(submitted_raw_batch['consec']) and _digest(submitted_raw_batch) == pending['batch_sha256'],
        'Submitted raw batch changed since the planned draw')
    require(np.array_equal(submitted_raw_batch['stepid'],pending['ids']),'Submitted full65 stepid changed')
    for row,(uuid,index) in enumerate(pending['starts']):
      expected=self._replay._getseq(uuid,index,keys=['stepid'])['stepid']
      require(np.array_equal(submitted_raw_batch['stepid'][row],expected),'Submitted full65 span differs from official source')
    require(type(out) is dict and set(out) == {'stepid',*CACHE},'Only actual context-cache replay output fields allowed')
    _array(out['stepid'],np.uint8,(BATCH,LENGTH,20),'update stepid')
    require(np.array_equal(out['stepid'],pending['ids'][:,CONTEXT:]),
        'Output stepid full64 span differs from submitted replay_context1 batch')
    for key,tail in CACHE.items(): _array(out[key],np.float32,(BATCH,LENGTH,*tail),'update '+key)
    _context(out['dyn/stoch'])
    import elements
    final_writers={}
    for row in range(BATCH):
      first=out['stepid'][row,0].tobytes();uuid=elements.UUID(first[:16]);index=int.from_bytes(first[16:],'big')
      for offset,target in enumerate(self._span(uuid,index,LENGTH)):
        expected=bytes(target[0])+target[1].to_bytes(4,'big')
        require(out['stepid'][row,offset].tobytes() == expected,'Unknown/nonconsecutive output stepid')
        final_writers[target]=(row,offset)
    try:
      self._replay.update(dict(out))
      for (uuid,index),(row,offset) in final_writers.items():
        for key in CACHE:
          require(self._replay.chunks[uuid].data[key][index].tobytes() == out[key][row,offset].tobytes(),
              'Official cache write missing/different after overlap last-writer application')
    except Exception:
      self._poisoned=True
      raise
    self._pending=None
    return dict(updated_rows=BATCH*LENGTH,unique_written_rows=len(final_writers),
        overlap_last_writer='official batch-row order; later rows overwrite earlier rows')
