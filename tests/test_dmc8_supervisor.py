"""First-eight selection and serial barriers; CPU only, no training launches."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from analysis.return_metrics import MetricSpec, aggregate
from dreamerv3.protocol_scope import DMC16, DMC16_TASKS, DMC8_FIRST
from scripts import prepare_dmc16 as prep
from scripts import run_dmc8_serial as runner


class Dmc8SupervisorTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.temporary=TemporaryDirectory()
    cls.root=Path(cls.temporary.name)
    resources=cls.root/'resources.json'
    resources.write_text(json.dumps(dict(server='sv1',host=runner.HOST,
        gpus={str(i):dict(uuid=f'fixture-UUID-{i}') for i in range(8)})))
    with patch.object(prep,'source_state',return_value=('fixture-not-training-SHA','docs draft retained')):
      runner.prepare(cls.root/'plan','formal-m2-dmc16-first8-serial-fixture',resources)
    cls.plan=json.loads((cls.root/'plan/planned-run-index.json').read_text())

  @classmethod
  def tearDownClass(cls):
    cls.temporary.cleanup()

  def test_exact_first_eight_and_same_card_pairs(self):
    runner.validate_plan(self.plan)
    self.assertEqual(self.plan['tasks'],list(DMC16_TASKS[:8]))
    self.assertEqual(self.plan['result_audit_argv'][4],'dmc8-first8')
    for gpu,task in enumerate(DMC16_TASKS[:8]):
      pair=[r for r in self.plan['runs'] if r['task']==task]
      self.assertEqual([r['physical_gpu'] for r in pair],[gpu,gpu])
      self.assertEqual([r['queue_position'] for r in pair],[1,2])
      a,b=[r['config'] for r in pair]
      self.assertEqual({k for k in a if a[k]!=b[k]},{'logdir','agent.rep_probe.mode'})
    self.assertFalse(self.plan['authorized_to_train'])

  def test_modified_resources_modes_seed_and_config_refused(self):
    for field,value in [('physical_gpu',7),('host','other'),('server','sv3'),('seed',1),
        ('mode','constant'),('task',DMC16_TASKS[8]),('directory','/elsewhere')]:
      plan=copy.deepcopy(self.plan);plan['runs'][0][field]=value
      with self.subTest(field=field),self.assertRaises(ValueError):runner.validate_plan(plan)
    plan=copy.deepcopy(self.plan);plan['runs'][0]['config']['agent.rep_probe.alpha']=19
    with self.assertRaises(ValueError):runner.validate_plan(plan)
    plan=copy.deepcopy(self.plan);plan['runs'].append(copy.deepcopy(plan['runs'][0]))
    with self.assertRaises(ValueError):runner.validate_plan(plan)

  def test_baseline_must_finish_audit_before_Dt_not_just_exit(self):
    rows=copy.deepcopy(self.plan['runs'])
    for row in rows:row['state']='queued'
    base=runner.next_for_gpu(rows,0)
    self.assertEqual(base['mode'],'logging')
    base['state']='running'
    self.assertIsNone(runner.next_for_gpu(rows,0))
    base['state']=runner.AUDITED
    dt=runner.next_for_gpu(rows,0)
    self.assertEqual(dt['mode'],'dt')
    dt['state']='running'
    self.assertIsNone(runner.next_for_gpu(rows,0))

  def test_failed_baseline_blocks_only_own_Dt(self):
    for failure in ('failed','audit_failed','blocked','training_finished_pending_audit'):
      rows=copy.deepcopy(self.plan['runs'])
      for row in rows:row['state']='queued'
      rows[0]['state']=failure
      self.assertIsNone(runner.next_for_gpu(rows,0))
      self.assertEqual(rows[1]['state'],'blocked')
      self.assertEqual(runner.next_for_gpu(rows,1)['mode'],'logging')

  def test_unauthorized_and_existing_root_never_launch(self):
    plan=copy.deepcopy(self.plan)
    path=self.root/'run-plan.json';path.write_text(json.dumps(plan))
    with patch.object(runner.subprocess,'Popen') as launch:
      with self.assertRaises(RuntimeError):runner.run(path)
      launch.assert_not_called()
    plan['authorized_to_train']=True
    path.write_text(json.dumps(plan))
    # Validation rejects unbound launches before any filesystem writes.
    with patch.object(runner.subprocess,'Popen') as launch:
      with self.assertRaises(ValueError):runner.run(path)
      launch.assert_not_called()

  def test_occupied_card_is_refused_before_creating_output(self):
    row=copy.deepcopy(self.plan['runs'][0]);row['directory']=str(self.root/'must-not-exist')
    with patch.object(runner,'require_free',side_effect=RuntimeError('unknown G process')):
      with patch.object(runner.subprocess,'Popen') as launch:
        with self.assertRaises(RuntimeError):runner.start(row)
        launch.assert_not_called()
    self.assertFalse(Path(row['directory']).exists())

  def test_existing_campaign_refused_without_resume_or_overwrite(self):
    plan=copy.deepcopy(self.plan);plan['authorized_to_train']=True
    for row in plan['runs']:
      row.update(authorized_to_train=True,launch_argv=row['review_argv'],
          egl_index=row['physical_gpu'],verified_compute_and_graphics_same_gpu=True)
      row['environment_template']['MUJOCO_EGL_DEVICE_ID']=str(row['egl_index'])
    with patch.object(runner,'read',return_value=plan), patch.object(runner.platform,'node',return_value=runner.HOST),\
        patch.object(runner,'clean_sha',return_value=plan['code_sha']),patch.object(runner,'Path') as path,\
        patch.object(runner.subprocess,'Popen') as launch:
      path.return_value.exists.return_value=True
      with self.assertRaises(FileExistsError):runner.run('unused')
      launch.assert_not_called()

  def test_first8_metric_scope_missing_task_never_reweights(self):
    spec=MetricSpec(tasks=DMC16_TASKS[:8],groups=('baseline','dt'),scope=DMC8_FIRST)
    rows=[dict(task=t,group=g,seed=0,run_id=t+g,auc_complete=True,tail_complete=True,
        auc_return_per_budget=float(i)+(2 if g=='dt' else 0),tail_mean_return=float(i)+1+(2 if g=='dt' else 0))
        for i,t in enumerate(spec.tasks) for g in spec.groups]
    _,overall,diffs=aggregate(rows,spec)
    self.assertEqual([r['auc_return_per_budget'] for r in overall],[3.5,5.5])
    self.assertTrue(all(r['auc_return_per_budget']==2 for r in diffs))
    _,overall,_=aggregate(rows[:-1],spec)
    self.assertIsNone(overall[1]['auc_return_per_budget'])
    with self.assertRaises(ValueError):MetricSpec(tasks=DMC16_TASKS,groups=spec.groups,scope=DMC8_FIRST)
    with self.assertRaises(ValueError):MetricSpec(tasks=spec.tasks,groups=spec.groups,seeds=(1,),scope=DMC8_FIRST)


if __name__=='__main__':unittest.main()
