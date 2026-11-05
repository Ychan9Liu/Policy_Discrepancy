import json
import copy
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from scripts import run_vision20_walker_dt as r
from scripts.prepare_m2_v1 import resolve


class WalkerDtTest(unittest.TestCase):
  def test_original_supervisor_cannot_spawn_reserved_source_slot(self):
    from scripts import run_vision20_seed1 as old
    with TemporaryDirectory() as temp:
      root=Path(temp)/'root';root.mkdir();reserved=root/'dt';reserved.mkdir();(reserved/'MOVED_TO_SV2.json').write_text('{}')
      row=dict(state='queued',physical_gpu=5,gpu_uuid='uuid',queue_position=2,directory=str(reserved))
      plan=dict(root=str(root),server='sv1',host='fixture',code_sha='fixture',authorized_to_train=True,runs=[dict(state=old.AUDITED,physical_gpu=5,gpu_uuid='uuid',queue_position=1),row])
      exists=Path.exists;mkdir=Path.mkdir
      def guarded_exists(p):return False if p==root else exists(p)
      def guarded_mkdir(p,*a,**k):return None if p==root else mkdir(p,*a,**k)
      with patch.object(old,'read',return_value=plan),patch.object(old,'validate'),patch.object(old,'clean_sha',return_value='fixture'),patch.object(old.platform,'node',return_value='fixture'),patch.object(old,'require_free',return_value={}),patch.object(old.shutil,'disk_usage') as disk,patch.object(Path,'exists',guarded_exists),patch.object(Path,'mkdir',guarded_mkdir),patch.object(old.subprocess,'Popen') as spawn,patch.object(old.subprocess,'run') as metrics:
        disk.return_value.free=10**12;metrics.return_value.returncode=0;old.run('fixture.json')
        spawn.assert_not_called();self.assertEqual(row['state'],'blocked');self.assertIn('FileExistsError',row['error'])
      self.assertEqual(list(reserved.iterdir()),[reserved/'MOVED_TO_SV2.json'])

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

  def test_hopper_blocked_source_keeps_lineage_and_requires_gpu3(self):
    source, proof = self.fixture()
    old = source['runs'][0]
    old.update(task='dmc_hopper_stand', state='blocked', error='GPU occupied',
               source_run_id='original', run_id='reallocated')
    old['review_argv'][old['review_argv'].index('--task') + 1] = old['task']
    old['config'] = json.loads(json.dumps(dict(resolve(old['review_argv'][4:]).flat)))
    with self.assertRaises(ValueError):
      r.make_plan(source, proof, 'controller', r.HOPPER_STAND)
    proof.update(physical_gpu=3, gpu_uuid='uuid3', egl_index=3)
    row = r.make_plan(source, proof, 'controller', r.HOPPER_STAND)['runs'][0]
    self.assertEqual(row['config'], dict(old['config'], logdir=row['directory']))
    self.assertEqual((row['source_run_id'], row['original_run_id']), ('reallocated', 'original'))
    self.assertNotIn('error', row)
    self.assertIsNone(row['baseline_c_source'])
    old['state'] = 'queued'
    with self.assertRaises(ValueError):
      r.make_plan(source, proof, 'controller', r.HOPPER_STAND)
