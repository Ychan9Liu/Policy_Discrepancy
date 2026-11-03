"""Control-plane relocation only: no agent/physics or completed-run claims."""
import copy
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from scripts import run_pendulum_parallel as parallel
from scripts import run_pendulum_seeds23 as serial
from tests import test_pendulum_seeds23 as fixtures


class PendulumParallelTest(unittest.TestCase):
  def setUp(self):
    temp=TemporaryDirectory(); self.addCleanup(temp.cleanup); self.temp=Path(temp.name)
    helper=fixtures.PendulumRepeatsTest(); self.source=helper.bound(helper.make_plan(self.temp))
    for name,value in [('TRAIN_SHA',self.source['code_sha']),
                       ('OLD_ROOT',PurePosixPath(self.source['root']))]:
      p=patch.object(parallel,name,value); p.start(); self.addCleanup(p.stop)
    self.resources=dict(host=parallel.HOST,cards=[dict(physical_gpu=g,uuid=f'new-gpu-{g}') for g in (5,6,7)])
    self.plan=parallel.build(self.source,self.resources,'controller-fixture')

  def bound(self):
    proofs={str(g):dict(code_sha=self.source['code_sha'],physical_gpu=g,gpu_uuid=f'new-gpu-{g}',
        verified_compute_and_graphics_same_gpu=True,egl_index=g) for g in (5,6)}
    return parallel.build(self.source,self.resources,'controller-fixture',proofs)

  def test_only_two_pending_dt_relocated_with_unchanged_training_commit_and_science(self):
    before=copy.deepcopy(self.source); parallel.validate(self.plan)
    self.assertEqual(self.source,before)
    self.assertEqual([(r['seed'],r['mode'],r['physical_gpu']) for r in self.plan['runs']],[(2,'dt',5),(3,'dt',6)])
    for r in self.plan['runs']:
      old=next(x for x in self.source['runs'] if x['run_id']==r['old_run_id'])
      self.assertEqual({k for k in r['config'] if r['config'][k]!=old['config'][k]},{'logdir'})
      self.assertEqual(r['code_sha'],old['code_sha'])
      self.assertEqual(r['baseline_c_source'],old['baseline_c_source'])
      self.assertEqual(r['environment_template']['WANDB_RUN_GROUP'],old['environment_template']['WANDB_RUN_GROUP'])
      self.assertIsNone(r['launch_argv'])
    self.assertFalse(self.plan['authorized_to_train'])

  def test_wrong_gpu_seed_mode_config_environment_target_or_version_refused(self):
    for key,value in [('physical_gpu',7),('seed',1),('mode','logging'),('directory','/overlap'),
                      ('code_sha','wrong'),('launch_argv',['unsafe'])]:
      p=copy.deepcopy(self.plan);p['runs'][0][key]=value
      with self.assertRaises(ValueError):parallel.validate(p)
    for key,value in [('agent.rep_probe.alpha',10),('run.eval_eps',1),('agent.opt.lr',.5)]:
      p=copy.deepcopy(self.plan);p['runs'][0]['config'][key]=value
      with self.assertRaises(ValueError):parallel.validate(p)
    p=copy.deepcopy(self.plan);p['runs'][0]['environment_template']['WANDB_MODE']='offline'
    with self.assertRaises(ValueError):parallel.validate(p)
    p=copy.deepcopy(self.plan);p['training_checkout']='/wrong'
    with self.assertRaises(ValueError):parallel.validate(p)

  def test_binding_requires_pinned_training_sha_and_exact_new_devices(self):
    p=self.bound();parallel.validate(p)
    self.assertTrue(all(r['authorized_to_train'] and r['launch_argv']==r['review_argv'] for r in p['runs']))
    for key,value in [('code_sha','controller-fixture'),('physical_gpu',1),('gpu_uuid','new-gpu-7'),
                      ('verified_compute_and_graphics_same_gpu',False),('egl_index',True)]:
      proofs=copy.deepcopy(p['device_bindings']);proofs['5'][key]=value
      with self.assertRaises(ValueError):parallel.build(self.source,self.resources,'controller-fixture',proofs)

  def guard_plan(self):
    old=self.temp/'old';old.mkdir()
    plan=copy.deepcopy(self.plan)
    for r in plan['runs']:
      d=old/f'seed{r["seed"]}';d.mkdir()
      r['old_pending_directory']=(d/'dt-attempt01').as_posix()
    return old,plan

  def test_guard_preserves_baselines_and_blocks_original_launcher_before_popen(self):
    old,p=self.guard_plan()
    for seed in (2,3):
      b=old/f'seed{seed}'/'logging-attempt01';b.mkdir();(b/'original').write_text('untouched')
    with patch.object(parallel,'OLD_ROOT',PurePosixPath(old.as_posix())),patch.object(parallel,'source_rows'):
      parallel.retire_pending(p);parallel.retire_pending(p)
    for r in p['runs']:
      d=Path(r['old_pending_directory'])
      self.assertEqual({x.name for x in d.iterdir()},{parallel.GUARD})
      row=dict(directory=str(d),physical_gpu=1,gpu_uuid='fixture')
      with patch.object(serial,'require_free'),patch.object(serial.subprocess,'Popen') as spawn:
        with self.assertRaises(FileExistsError):serial.start(row)
        spawn.assert_not_called()
      self.assertEqual((d.parent/'logging-attempt01'/'original').read_text(),'untouched')

  def test_existing_attempt_or_mismatched_guard_never_overwritten(self):
    old,p=self.guard_plan();d=Path(p['runs'][0]['old_pending_directory']);d.mkdir()
    (d/'protocol_manifest.json').write_text('existing run')
    with patch.object(parallel,'OLD_ROOT',PurePosixPath(old.as_posix())),patch.object(parallel,'source_rows'):
      with self.assertRaises(FileExistsError):parallel.retire_pending(p)
    self.assertEqual((d/'protocol_manifest.json').read_text(),'existing run')
    self.assertFalse(Path(p['runs'][1]['old_pending_directory']).exists())

  def test_combined_index_uses_original_baselines_and_only_new_dt_not_superseded_slots(self):
    source=copy.deepcopy(self.source['runs']);own=copy.deepcopy(self.bound()['runs'])
    for r in source:r['state']=parallel.AUDITED if r['mode']=='logging' else 'blocked'
    for r in own:r['state']='running'
    with patch.object(parallel,'source_rows',return_value=source):rows=parallel.merged_rows(self.plan,own)
    self.assertEqual([(r['seed'],r['mode']) for r in rows],[(2,'logging'),(2,'dt'),(3,'logging'),(3,'dt')])
    self.assertEqual([r['state'] for r in rows],[parallel.AUDITED,'running',parallel.AUDITED,'running'])
    self.assertEqual(len({r['run_id'] for r in rows}),4)
    self.assertTrue(all(r['run_id'] not in {s['run_id'] for s in source if s['mode']=='dt'} for r in rows))

  def test_source_identity_or_already_started_pending_run_refused(self):
    rows=copy.deepcopy(self.source['runs'])
    for r in rows:r['state']='running' if r['mode']=='logging' else 'queued'
    with patch.object(parallel,'read',return_value=dict(runs=rows)):
      self.assertEqual(parallel.source_rows(self.plan,before_launch=True),rows)
      rows[1]['state']='running'
      with self.assertRaises(RuntimeError):parallel.source_rows(self.plan,before_launch=True)
      rows[1]['state']='queued';rows[0]['config']['seed']=7
      with self.assertRaises(ValueError):parallel.source_rows(self.plan)

  def test_existing_root_or_low_storage_never_guards_or_starts(self):
    p=self.bound()
    for exists,free in ((True,10**15),(False,239_999_999_999)):
      with (patch.object(parallel,'read',return_value=p),patch.object(parallel,'pinned_training'),
          patch.object(parallel,'verify_evidence'),patch.object(parallel,'clean_sha',return_value=p['controller_sha']),
          patch.object(parallel.platform,'node',return_value=parallel.HOST),patch.object(Path,'exists',return_value=exists),
          patch.object(parallel,'require_free') as resource,patch.object(parallel.shutil,'disk_usage',return_value=MagicMock(free=free)),
          patch.object(parallel,'retire_pending') as guards,patch.object(parallel,'start') as start):
        with self.assertRaises((FileExistsError,RuntimeError)):parallel.run('plan','evidence')
        guards.assert_not_called();start.assert_not_called()
        self.assertEqual([c.args[0] for c in resource.call_args_list],[] if exists else [5,6])

  def test_acceptance_requires_both_exact_seeds_devices_and_completed_engineering(self):
    bound=self.bound()
    files={'controller-cpu-acceptance.json':dict(passed=True,code_sha='controller-fixture'),
        'cpu-acceptance.json':dict(passed=True,code_sha=self.source['code_sha']),
        'logger_validation.json':dict(code_sha=self.source['code_sha'],remote_state='finished',project='PD_1',
            entity='ychan9liu-tsinghua-university',seed=2,training_actions=0,agent_updates=0),
        'acceptance.json':dict(passed=True,training_sha=self.source['code_sha'],controller_sha='controller-fixture',
            run_count=2,engineering_only=True,runs=[dict(seed=s,mode='dt',physical_gpu=g,
            updates=517,evaluations=3,reports=2,exit_code=0) for s,g in ((2,5),(3,6))])}
    files.update({f'gpu{g}-binding.json':v for g,v in bound['device_bindings'].items()})
    with patch.object(parallel,'read',side_effect=lambda path:files[Path(path).name]):
      self.assertEqual(parallel.verify_evidence(self.plan,'evidence'),bound['device_bindings'])
      for key,value in [('seed',3),('physical_gpu',7),('updates',516),('exit_code',1)]:
        original=files['acceptance.json']['runs'][0][key]
        files['acceptance.json']['runs'][0][key]=value
        with self.assertRaises(ValueError):parallel.verify_evidence(self.plan,'evidence')
        files['acceptance.json']['runs'][0][key]=original


if __name__=='__main__':unittest.main()
