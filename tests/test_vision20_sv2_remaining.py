import copy
import json
import unittest
from unittest.mock import patch
from scripts import run_vision20_sv2_remaining as r
from scripts.prepare_m2_v1 import resolve


class RemainingTest(unittest.TestCase):
  def fixture(self):
    source={'runs':[]};proofs={str(g):dict(physical_gpu=g,code_sha=r.TRAIN_SHA,gpu_uuid='uuid'+str(g),egl_index=g,verified_compute_and_graphics_same_gpu=True) for g in range(4)}
    for g,t,m in r.MATRIX:
      argv=['python','-u','-m','dreamerv3.main','--configs','m2_vision20_seed1','--task',t,'--logdir','/old/'+t+m,'--agent.rep_probe.mode',m,'--jax.prealloc','False','--logger.outputs','jsonl','wandb']
      source['runs'].append(dict(task=t,mode=m,state='blocked',code_sha=r.TRAIN_SHA,run_id=t+m,directory='/old/'+t+m,review_argv=argv,config=json.loads(json.dumps(dict(resolve(argv[4:]).flat))),environment_template={'WANDB_RUN_ID':t+m}))
    return source,proofs

  def test_exact_seven_four_queues_only_logdir_changes(self):
    s,p=self.fixture();out=r.make_plan(s,p,'controller')
    self.assertEqual([(x['physical_gpu'],x['task'],x['mode']) for x in out['runs']],r.MATRIX)
    self.assertEqual([x['queue_position'] for x in out['runs']],[1,2,1,2,1,2,1])
    for a,b in zip(s['runs'],out['runs']):
      self.assertEqual(b['config'],dict(a['config'],logdir=b['directory']))
      self.assertEqual(b['server'],'sv2');self.assertEqual(b['host'],'lyg0360')
      self.assertNotEqual(a['run_id'],b['run_id'])

  def test_source_live_missing_or_numeric_tampering_rejected(self):
    s,p=self.fixture()
    for changes in ({'pid':123},{'state':'running'},{'code_sha':'untested'}):
      bad=copy.deepcopy(s);bad['runs'][0].update(changes)
      with self.assertRaises(ValueError):r.make_plan(bad,p,'controller')
    bad=copy.deepcopy(s);bad['runs'][0]['config']['seed']=2
    with self.assertRaises(ValueError):r.make_plan(bad,p,'controller')
    with self.assertRaises(ValueError):r.make_plan({'runs':s['runs'][:-1]},p,'controller')

  def test_seal_must_match_exact_source_ids_hash_and_target(self):
    s,_=self.fixture();seal=dict(passed=True,claimed_source_run_ids=[x['run_id'] for x in s['runs']],target_root=str(r.ROOT),source_sha256='hash')
    with patch.object(r,'sha',return_value='hash'):
      r.validate_seal(s,seal)
      for changes in ({'claimed_source_run_ids':seal['claimed_source_run_ids'][:-1]},{'source_sha256':'wrong'},{'target_root':'/other'},{'passed':False}):
        with self.assertRaises(RuntimeError):r.validate_seal(s,dict(seal,**changes))
