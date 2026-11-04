import copy
import json
import unittest
from unittest.mock import patch
from scripts import run_vision20_reallocated as r


class ReallocatedTest(unittest.TestCase):
  def test_serial_gate_audit_failure_blocks_and_running_prevents_overlap(self):
    rows=[dict(physical_gpu=0,queue_position=i+1,state='queued') for i in range(2)]
    self.assertIs(r.next_row(rows,0),rows[0])
    rows[0]['state']='running';self.assertIsNone(r.next_row(rows,0))
    rows[0]['state']='failed';self.assertIsNone(r.next_row(rows,0))
    self.assertEqual(rows[1]['state'],'blocked')
    rows[1]['state']='queued';rows[0]['state']=r.AUDITED
    self.assertIs(r.next_row(rows,0),rows[1])

  def test_source_guard_rejects_started_changed_or_existing_artifacts(self):
    row=dict(source_run_id='old',source_directory='/old')
    old=dict(run_id='old',state='blocked',directory='/old')
    with patch.object(r,'read',return_value={'runs':[old]}),patch.object(r.Path,'exists',return_value=False):
      r.verify_source(row)
      for changes in ({'pid':123},{'state':'running'},{'directory':'/other'}):
        bad=dict(old,**changes)
        with patch.object(r,'read',return_value={'runs':[bad]}),self.assertRaises(RuntimeError):r.verify_source(row)
    with patch.object(r,'read',return_value={'runs':[old]}),patch.object(r.Path,'exists',return_value=True),self.assertRaises(RuntimeError):r.verify_source(row)

  def test_exact_six_slots_only_logdir_and_device_identity_change(self):
    from scripts.prepare_m2_v1 import resolve
    source={'runs':[]}
    proofs={str(g):dict(physical_gpu=g,gpu_uuid='uuid'+str(g),egl_index=g,code_sha=r.TRAIN_SHA,verified_compute_and_graphics_same_gpu=True) for g in (0,1,4)}
    for gpu,task,mode in r.MATRIX:
      argv=['python','-u','-m','dreamerv3.main','--configs','m2_vision20_seed1','--task',task,'--logdir','/old/'+task+mode,'--agent.rep_probe.mode',mode,'--jax.prealloc','False','--logger.outputs','jsonl','wandb']
      source['runs'].append(dict(task=task,mode=mode,state='blocked',code_sha=r.TRAIN_SHA,run_id=task+mode,directory='/old/'+task+mode,review_argv=argv,config=json.loads(json.dumps(dict(resolve(argv[4:]).flat))),environment_template={'WANDB_RUN_ID':task+mode}))
    plan=r.make_plan(source,proofs,'controller')
    self.assertEqual([(x['physical_gpu'],x['task'],x['mode']) for x in plan['runs']],r.MATRIX)
    for a,b in zip(source['runs'],plan['runs']):
      self.assertEqual(b['config'],dict(a['config'],logdir=b['directory']))
      self.assertNotEqual(a['run_id'],b['run_id'])
    bad=copy.deepcopy(source);bad['runs'][0]['config']['seed']=2
    with self.assertRaises(ValueError):r.make_plan(bad,proofs,'controller')
    bad=copy.deepcopy(proofs);bad['0']['code_sha']='untested'
    with self.assertRaises(ValueError):r.make_plan(source,bad,'controller')
