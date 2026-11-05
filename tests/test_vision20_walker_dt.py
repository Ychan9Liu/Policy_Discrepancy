import json
import copy
import unittest
from unittest.mock import patch
from scripts import run_vision20_walker_dt as r
from scripts.prepare_m2_v1 import resolve


class WalkerDtTest(unittest.TestCase):
  def fixture(self):
    argv=['python','-u','-m','dreamerv3.main','--configs','m2_vision20_seed1','--task','dmc_walker_run','--logdir','/old/dt','--agent.rep_probe.mode','dt','--jax.prealloc','False','--logger.outputs','jsonl','wandb']
    row=dict(task='dmc_walker_run',mode='dt',seed=1,state='queued',code_sha=r.TRAIN_SHA,run_id='old',directory='/old/dt',review_argv=argv,config=json.loads(json.dumps(dict(resolve(argv[4:]).flat))),environment_template={'WANDB_RUN_ID':'old'})
    return {'runs':[row]},dict(physical_gpu=4,code_sha=r.TRAIN_SHA,gpu_uuid='uuid4',egl_index=4,verified_compute_and_graphics_same_gpu=True)

  def test_exact_queued_slot_only_logdir_changes_and_unique_identity(self):
    s,p=self.fixture();out=r.make_plan(s,p,'controller');row=out['runs'][0]
    self.assertEqual(row['config'],dict(s['runs'][0]['config'],logdir=row['directory']));self.assertEqual(row['physical_gpu'],4);self.assertIsNone(row['baseline_c_source']);self.assertNotEqual(row['run_id'],'old')
    for changes in ({'pid':123},{'state':'running'},{'task':'dmc_hopper_stand'},{'seed':2}):
      bad=copy.deepcopy(s);bad['runs'][0].update(changes)
      with self.assertRaises(ValueError):r.make_plan(bad,p,'controller')

  def test_exclusive_reservation_and_hash_mandatory(self):
    s,_=self.fixture();seal=dict(passed=True,source_directory_reserved_before_spawn=True,source_sha256='hash',replacement_root=str(r.ROOT),source_run_id='old',source_reserved_directory='/old/dt',reservation_contents=['MOVED_TO_SV2.json'])
    with patch.object(r,'sha',return_value='hash'):
      r.validate_seal(s,seal)
      for changes in ({'source_directory_reserved_before_spawn':False},{'source_sha256':'wrong'},{'reservation_contents':[]},{'source_run_id':'other'}):
        with self.assertRaises(ValueError):r.validate_seal(s,dict(seal,**changes))
