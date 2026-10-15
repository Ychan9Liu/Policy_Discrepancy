"""Generated connected chunks only; no real1M import or learning acceptance."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('JAX_PLATFORMS','cpu')
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import dl_normal_replay as helper
from scripts import dl_replay_inventory as inventory
from tests.test_dl_replay_inventory import uuid


class NormalReplayTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
    cls.runpath=cls.root/'run';cls.directory=cls.runpath/'replay';cls.directory.mkdir(parents=True)
    for c in range(3):
      length=40;start=40*c
      arrays={key:np.zeros((length,*tail),dtype) for key,(dtype,tail) in inventory.FIELDS.items()}
      for i in range(length):
        at=start+i;first=at % 50 == 0;last=at % 50 == 49
        arrays['image'][i]=at;arrays['reward'][i]=0 if first else np.float32((at % 10)/10)
        arrays['action'][i]=0 if last else np.linspace(-.3,.3,12,dtype=np.float32)
        arrays['is_first'][i]=first;arrays['is_last'][i]=last
        arrays['dyn/deter'][i]=np.float32(100+at)
        arrays['dyn/stoch'][i,:,at % 32]=1
      arrays['stepid'][:,:16]=np.frombuffer(inventory.uuid_bytes(uuid(c+1)),np.uint8)
      arrays['stepid'][:,16:]=np.arange(length,dtype='>u4').view(np.uint8).reshape(length,4)
      name=f'20261014T005815F{c*1000:06d}-{uuid(c+1)}-{uuid(c+2)}-40.npz'
      np.savez_compressed(cls.directory/name,**arrays)
    manifest=dict(git_commit=inventory.SOURCE_GIT,group='N',root_seed=0,task='dmc_quadruped_walk',
        config={'agent.dyn.rssm.deter':4096,'agent.dyn.rssm.stoch':32,'agent.dyn.rssm.classes':32,
          'replay_context':1,'replay.chunksize':1024,'run.envs':16,'batch_size':16,'batch_length':64,'consec_train':1})
    final=dict(git_commit=inventory.SOURCE_GIT,group='N',seed=0,task='dmc_quadruped_walk',
        train_action_steps=117,train_resets=3,status='training_ended_pending_audit')
    for name,value in (('protocol_manifest.json',manifest),('final_state.json',final)):
      (cls.runpath/name).write_text(json.dumps(value),encoding='utf-8')
    cls.report=cls.root/'fixture_inventory.json'
    with mock.patch.object(inventory,'EXPECTED_FILES',3),mock.patch.object(inventory,'CHAINS',1),\
        mock.patch.object(inventory,'TRAIN_ACTIONS',117),mock.patch.object(inventory,'binding',return_value={'fixture':True}):
      inventory.validate(cls.runpath,cls.directory,cls.report,inventory.SOURCE_GIT,'b'*40)
    cls.pin=hashlib.sha256(cls.report.read_bytes()).hexdigest()
    cls.sourcebytes={path.name:path.read_bytes() for path in cls.directory.iterdir()}

  @classmethod
  def tearDownClass(cls): cls.temp.cleanup()

  def load(self):
    # Explicit fixture SHA and scope patches; production remains1280/16/1M.
    with mock.patch.object(helper,'EXPECTED_COUNTS',(3,1,120,3,117)),mock.patch.object(inventory,'CHAINS',1):
      return helper.load_private_cache(self.directory,self.report,expected_inventory_sha256=self.pin)

  def output(self,batch,value):
    stoch=np.zeros((16,64,32,32),np.float32);stoch[...,int(value) % 32]=1
    return dict(stepid=batch['stepid'][:,1:].copy(),
        **{'dyn/deter':np.full((16,64,4096),value,np.float32),'dyn/stoch':stoch})

  def starts(self): return tuple((uuid(1),30+(i % 2)) for i in range(16))

  def test_official_three_chunk_full65_annotation_consec_and_original_context(self):
    cache=self.load();batch=cache.batch(self.starts());buffer=cache._replay
    self.assertEqual(len(buffer),56)
    self.assertEqual((buffer.directory,buffer.online),(None,False))
    self.assertTrue(np.all(batch['consec'] == 0));self.assertTrue(np.all(batch['is_first'][:,0]))
    self.assertEqual(batch['image'].shape,(16,65,64,64,3))
    # row0 is original global30..94, across40+40+40 storage chunks.
    self.assertTrue(np.array_equal(batch['image'][0,:,0,0,0],np.arange(30,95)))
    self.assertTrue(np.array_equal(batch['dyn/deter'][0,:,0],np.arange(130,195)))
    self.assertTrue(np.all(batch['is_last'][0,[19]]));self.assertTrue(batch['is_first'][0,20])
    self.assertEqual(batch['stepid'][0,10,:16].tobytes(),inventory.uuid_bytes(uuid(2)))
    self.assertEqual(batch['stepid'][0,50,:16].tobytes(),inventory.uuid_bytes(uuid(3)))
    self.assertFalse(cache.receipt()['original_inventory_authenticated'])
    cache.strict_update(self.output(batch,6),batch)
    self.assertEqual(buffer.metrics['updates'],16*64)

  def test_two_arms_share_fixed_raw_plan_but_only_private_context_diverges(self):
    left,right=self.load(),self.load();plan=left.make_plan(91,2)
    self.assertEqual(plan,right.make_plan(91,2))
    l,r=left.batch(plan[0]),right.batch(plan[0])
    for key in l: self.assertEqual(l[key].tobytes(),r[key].tobytes())
    left.strict_update(self.output(l,4),l);right.strict_update(self.output(r,9),r)
    l,r=left.batch(plan[1]),right.batch(plan[1])
    for key in set(l)-set(helper.CACHE): self.assertEqual(l[key].tobytes(),r[key].tobytes())
    self.assertFalse(np.array_equal(l['dyn/deter'],r['dyn/deter']))
    for uid,chunk in left._replay.chunks.items():
      for key in chunk.data:
        self.assertFalse(np.shares_memory(chunk.data[key],right._replay.chunks[uid].data[key]))
    for name,body in self.sourcebytes.items(): self.assertEqual((self.directory/name).read_bytes(),body)

  def test_calibration_read_has_no_pending_counts_or_cache_write(self):
    cache=self.load();starts=self.starts()
    before={uid:{k:a.tobytes() for k,a in c.data.items()}
        for uid,c in cache._replay.chunks.items()}
    metrics=dict(cache._replay.metrics)
    first,second=cache.read_batch(starts),cache.read_batch(starts)
    self.assertIsNone(cache._pending)
    self.assertEqual(dict(cache._replay.metrics),metrics)
    for key in first:self.assertEqual(first[key].tobytes(),second[key].tobytes())
    self.assertTrue(np.all(first['consec'] == 0))
    self.assertEqual(first['dyn/deter'][0,0,0],130)
    for uid,chunk in cache._replay.chunks.items():
      for key in chunk.data:self.assertEqual(chunk.data[key].tobytes(),before[uid][key])
    # Normal training still requires actual output feedback; a read cannot be
    # presented as an outstanding train/update, nor bypass a pending update.
    with self.assertRaisesRegex(ValueError,'outstanding'):
      cache.strict_update(self.output(first,3),first)
    submitted=cache.batch(starts)
    with self.assertRaisesRegex(ValueError,'Pending batch'):cache.read_batch(starts)
    cache.strict_update(self.output(submitted,8),submitted)
    later=cache.read_batch(((uuid(1),31),)*16)
    self.assertEqual(later['dyn/deter'][0,0,0],8)
    for name,body in self.sourcebytes.items():self.assertEqual((self.directory/name).read_bytes(),body)

  def test_original_path_receipt_has_no_implicit_stage_provenance(self):
    cache=self.load()
    expected=dict(version=helper.VERSION,inventory_sha256=self.pin,
        original_inventory_authenticated=False,chunks=3,legal65_starts=56,
        batch_size=16,length=64,prefix=1,consec=1,prefetch=False,directory=None,
        online=False,private_cache=True,original_files_written=False,
        sampler_rng_restored=False,online_queue_restored=False,environment_restored=False,
        exact_endpoint_cache_restored=False,object_scope=json.loads(self.report.read_bytes())['object_scope'])
    self.assertEqual(cache.receipt(),expected)

  def test_full_span_unknown_id_and_bad_cache_refused_before_any_write(self):
    cache=self.load();batch=cache.batch(self.starts());valid=self.output(batch,5)
    baseline={uid:chunk.data['dyn/deter'].tobytes() for uid,chunk in cache._replay.chunks.items()}
    for issue in ('middle-id','unknown-first','float64','nan','stoch','shape','rawfield'):
      out=copy.deepcopy(valid)
      if issue == 'middle-id': out['stepid'][7,31,19]^=np.uint8(1)
      elif issue == 'unknown-first': out['stepid'][0,0,:16]=255
      elif issue == 'float64': out['dyn/deter']=out['dyn/deter'].astype(np.float64)
      elif issue == 'nan': out['dyn/deter'][0,0,0]=np.nan
      elif issue == 'stoch': out['dyn/stoch'][0,0,0,0]=.5
      elif issue == 'shape': out['dyn/deter']=out['dyn/deter'][:,:63]
      else: out['reward']=batch['reward'][:,1:]
      with self.subTest(issue=issue),self.assertRaises(ValueError): cache.strict_update(out,batch)
      for uid,chunk in cache._replay.chunks.items(): self.assertEqual(chunk.data['dyn/deter'].tobytes(),baseline[uid])
    altered={key:value.copy() for key,value in batch.items()};altered['stepid'][0,0,19]^=np.uint8(1)
    with self.assertRaisesRegex(ValueError,'raw batch changed'): cache.strict_update(valid,altered)
    with self.assertRaisesRegex(ValueError,'Pending batch'): cache.batch(self.starts())
    cache.strict_update(valid,batch)
    with self.assertRaisesRegex(ValueError,'outstanding'): cache.strict_update(valid,batch)

  def test_official_overlap_later_rows_win_and_missing_write_is_not_silent(self):
    cache=self.load();starts=((uuid(1),30),)*16;batch=cache.batch(starts);out=self.output(batch,2)
    for row in range(16): out['dyn/deter'][row]=row
    result=cache.strict_update(out,batch)
    self.assertEqual(result['unique_written_rows'],64)
    targets=cache._span(cache._start(starts[0])[0],31,64)
    for uid,index in targets: self.assertTrue(np.all(cache._replay.chunks[uid].data['dyn/deter'][index] == 15))
    cache=self.load();batch=cache.batch(starts);out=self.output(batch,3)
    with mock.patch.object(cache._replay,'_setseq',side_effect=KeyError('old update would skip')):
      with self.assertRaisesRegex(ValueError,'write missing'): cache.strict_update(out,batch)
    with self.assertRaisesRegex(ValueError,'Pending batch'): cache.batch(starts)

  def test_metadata_and_raw_source_hash_before_decode_corruption_never_skipped(self):
    with mock.patch.object(inventory,'read_json',side_effect=AssertionError('must not decode bad hash')):
      with self.assertRaisesRegex(ValueError,'Inventory raw SHA'): helper.load_private_cache(self.directory,self.report)
    name=next(iter(self.sourcebytes));path=self.directory/name;original=path.read_bytes()
    path.write_bytes(original[:-1]+bytes([original[-1]^1]))
    try:
      with mock.patch.object(helper,'_decode',side_effect=AssertionError('bad source must not decode')):
        with self.assertRaisesRegex(ValueError,'raw hash differs'): self.load()
    finally: path.write_bytes(original)
    row=inventory.filename(name)
    with self.assertRaises((ValueError,zipfile.BadZipFile)):
      helper._decode(b'not a zip',row)
    with self.assertRaisesRegex(ValueError,'Frozen offline inventory provenance/counts'):
      helper.load_private_cache(self.directory,self.report,expected_inventory_sha256=self.pin)


if __name__ == '__main__': unittest.main()
