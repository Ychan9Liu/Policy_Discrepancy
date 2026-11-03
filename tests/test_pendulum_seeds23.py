"""Approved scope, unchanged science, serial gate and hand-computed metrics."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from dreamerv3.protocol_scope import PENDULUM_23, PENDULUM_SCRIPT, tasks_for_protocol, validate_scope
from analysis.return_metrics import MetricSpec, aggregate
from scripts.prepare_m2_v1 import resolve
from scripts import prepare_pendulum_seeds23 as prepare
from scripts import run_pendulum_seeds23 as runner
from tests.test_return_metrics import point


class PendulumRepeatsTest(unittest.TestCase):
  def make_plan(self, root):
    refs = {}
    for group, mode in (('baseline','logging'), ('dt','dt')):
      c = resolve(['--configs','m2_dmc16','--task',prepare.TASK,'--seed','0',
          '--logdir','/original/'+mode,'--agent.rep_probe.mode',mode,
          '--jax.prealloc','False','--logger.outputs','jsonl','wandb'])
      refs[group] = dict(training_sha='original', config=json.loads(json.dumps(dict(c.flat))))
    ref = root/'refs.json'; ref.write_text(json.dumps(refs))
    resource = root/'resources.json'
    resource.write_text(json.dumps(dict(host=prepare.HOST,
        cards=[dict(physical_gpu=g, uuid='fixture-gpu-'+str(g)) for g in (1,4,5,6,7)])))
    with patch.object(prepare,'source_state',return_value=('fixture-sha','retained drafts')):
      prepare.prepare(root/'plan', resource, ref, 'formal-m2-pendulum-seeds23-fixture')
    return json.loads((root/'plan/planned-run-index.json').read_text())

  def bound(self, plan):
    plan['authorized_to_train'] = True
    for r in plan['runs']:
      r.update(authorized_to_train=True, launch_argv=r['review_argv'],
          egl_index=r['physical_gpu'], verified_compute_and_graphics_same_gpu=True, state='queued')
      r['environment_template']['MUJOCO_EGL_DEVICE_ID'] = str(r['egl_index'])
    runner.validate(plan)
    return plan

  def test_scope_accepts_only_two_seeds_task_and_modes_without_relaxing_older_entries(self):
    self.assertEqual(tasks_for_protocol(PENDULUM_23), (prepare.TASK,))
    for seed in (2,3):
      for mode in ('logging','dt'):
        self.assertEqual(validate_scope(PENDULUM_SCRIPT,prepare.TASK,seed,mode),PENDULUM_23)
        for script in ('protocol_v1','protocol_dmc16','protocol_vision20_seed1','protocol_hopper_seeds123'):
          with self.assertRaises(ValueError): validate_scope(script,prepare.TASK,seed,mode)
    for task, seed, mode in [(prepare.TASK,0,'dt'),(prepare.TASK,1,'dt'),(prepare.TASK,4,'dt'),
        (prepare.TASK,2.0,'dt'),(prepare.TASK,True,'dt'),('dmc_hopper_hop',2,'dt'),
        (prepare.TASK,2,'constant'),(prepare.TASK,3,'shuffle')]:
      with self.assertRaises(ValueError): validate_scope(PENDULUM_SCRIPT,task,seed,mode)

  def test_full_prepared_configs_match_reference_except_scope_seed_and_path(self):
    with TemporaryDirectory() as temp:
      root = Path(temp); p = self.make_plan(root); runner.validate(p)
      self.assertEqual([(r['seed'],r['mode'],r['physical_gpu']) for r in p['runs']],
                       [(2,'logging',1),(2,'dt',1),(3,'logging',4),(3,'dt',4)])
      self.assertTrue(all(set(r['reference_config_differences'])=={'script','seed','logdir'} for r in p['runs']))
      self.assertTrue(all(r['launch_argv'] is None and not r['authorized_to_train'] for r in p['runs']))
      with self.assertRaises(FileExistsError):
        prepare.prepare(root/'plan',root/'resources.json',root/'refs.json','formal-m2-pendulum-seeds23-fixture')
      for field, value in [('physical_gpu',5),('seed',1),('mode','constant'),('directory','/old')]:
        bad = copy.deepcopy(p); bad['runs'][0][field] = value
        with self.assertRaises(ValueError): runner.validate(bad)
      for field, value in [('WANDB_PROJECT','other'),('CUDA_VISIBLE_DEVICES','other'),('JAX_PLATFORMS','cpu')]:
        bad = copy.deepcopy(p); bad['runs'][0]['environment_template'][field] = value
        with self.assertRaises(ValueError): runner.validate(bad)
      for field,value in [('agent.rep_probe.alpha',10),('agent.opt.lr',1e-3),('run.eval_eps',1)]:
        bad = copy.deepcopy(p); r = bad['runs'][0]
        r['review_argv'] += ['--'+field,str(value)]
        r['config'] = json.loads(json.dumps(dict(resolve(r['review_argv'][4:]).flat)))
        with self.assertRaises(ValueError): runner.validate(bad)

  def test_binding_requires_every_seed_mode_engineering_result(self):
    with TemporaryDirectory() as temp:
      p = self.make_plan(Path(temp)); rows = p['runs']
      files = {'plan.json':p, 'cpu-acceptance.json':dict(passed=True,code_sha=p['code_sha']),
          'acceptance.json':dict(passed=True,code_sha=p['code_sha'],run_count=4,runs=rows[:2]),
          'logger_validation.json':dict(code_sha=p['code_sha'],remote_state='finished',entity=runner.ENTITY,project='PD_1',seed=2)}
      with (patch.object(runner,'read',side_effect=lambda path:files[Path(path).name]),
          patch.object(Path,'exists',return_value=False), patch.object(runner,'clean_sha',return_value=p['code_sha']),
          patch.object(runner.platform,'node',return_value=prepare.HOST)):
        with self.assertRaises(ValueError): runner.bind('plan.json','evidence','bound.json')

  def test_existing_output_and_insufficient_capacity_cannot_start_jobs(self):
    with TemporaryDirectory() as temp:
      p = self.bound(self.make_plan(Path(temp)))
      with (patch.object(runner,'read',return_value=p), patch.object(Path,'exists',return_value=True),
          patch.object(runner,'start') as start):
        with self.assertRaises(FileExistsError): runner.run('plan')
        start.assert_not_called()
      with (patch.object(runner,'read',return_value=p), patch.object(Path,'exists',return_value=False),
          patch.object(runner.platform,'node',return_value=prepare.HOST), patch.object(runner,'clean_sha',return_value=p['code_sha']),
          patch.object(runner,'require_free'), patch.object(runner.shutil,'disk_usage',return_value=MagicMock(free=239_999_999_999)),
          patch.object(runner,'start') as start):
        with self.assertRaises(RuntimeError): runner.run('plan')
        start.assert_not_called()

  def test_baseline_audit_barrier_blocks_dt_on_exit_or_audit_failure(self):
    for exit_code, audit_fails in ((0,False),(1,False),(0,True)):
      with TemporaryDirectory() as temp:
        p = self.bound(self.make_plan(Path(temp))); audited=set(); started=[]
        process=MagicMock(returncode=exit_code); process.poll.return_value=exit_code
        def start(row):
          if row['mode']=='dt': self.assertIn(row['seed'],audited)
          started.append((row['seed'],row['mode'])); row.update(state='running',start_epoch=0)
          return process, MagicMock()
        def audit(row, scope):
          self.assertEqual(scope,PENDULUM_23)
          if audit_fails: raise ValueError('fixture bad receipt')
          audited.add(row['seed']); return dict(passed=True)
        with (patch.object(runner,'read',return_value=p), patch.object(Path,'exists',return_value=False),
            patch.object(Path,'mkdir'), patch.object(runner,'dump'), patch.object(runner.platform,'node',return_value=prepare.HOST),
            patch.object(runner,'clean_sha',return_value=p['code_sha']), patch.object(runner,'require_free'),
            patch.object(runner.shutil,'disk_usage',return_value=MagicMock(free=10**15)),
            patch.object(runner,'start',side_effect=start), patch.object(runner,'audit_run',side_effect=audit),
            patch.object(runner.time,'sleep'), patch.object(runner,'finish') as finish):
          runner.run('plan')
        good = exit_code==0 and not audit_fails
        self.assertEqual(len(started),4 if good else 2)
        if good: finish.assert_called_once()
        else:
          finish.assert_not_called()
          self.assertTrue(all(r['state']=='blocked' for r in p['runs'] if r['mode']=='dt'))

  def test_metric_scope_requires_exact_pair_seeds_and_preserves_missing_data(self):
    spec=MetricSpec(seeds=(2,3),tasks=(prepare.TASK,),groups=('baseline','dt'),scope=PENDULUM_23)
    for field,value in [('seeds',(0,1,2,3)),('seeds',(2.0,3)),('tasks',('dmc_hopper_hop',)),('episodes',1)]:
      with self.assertRaises(ValueError): replace(spec,**{field:value})
    self.assertTrue(all(not r['auc_complete'] for r in aggregate([],spec)[1]))

  def test_real_freezer_preserves_half_open_window_and_exact_seed_identity(self):
    from dreamerv3.freeze_c import freeze
    from tests.test_hopper_repeat_preparation import HopperPreparationTest
    for seed in (2,3):
      with TemporaryDirectory() as temp:
        root=Path(temp); HopperPreparationTest().source(root,seed)
        path=root/'protocol_manifest.json'; m=json.loads(path.read_text())
        m.update(protocol=PENDULUM_23,task=prepare.TASK)
        m['config']['script']=PENDULUM_SCRIPT; path.write_text(json.dumps(m))
        result=freeze(root,root/'c.json')
        self.assertEqual((result['root_seed'],result['protocol'],result['c']), (seed,PENDULUM_23,.5))
        self.assertEqual(result['included_update_ids'],[1,2])
        self.assertEqual(freeze(root,root/'c.json'),result)
        m['root_seed']=1; path.write_text(json.dumps(m))
        with self.assertRaises(ValueError): freeze(root,root/'wrong.json')

  def test_complete_hand_cli_and_missing_seed_and_wrong_profile(self):
    with TemporaryDirectory() as temp:
      root=Path(temp); runs=[]
      for seed, baseline, dt in ((2,10,13),(3,20,18)):
        for mode,value in (('logging',baseline),('dt',dt)):
          d=root/f'{seed}-{mode}'; d.mkdir(); runs.append(dict(directory=str(d),run_id=d.name))
          config={'seed':seed,'agent.rep_probe.mode':mode,'run.engineering_fixture':False,
                  'run.action_budget':1000000,'run.eval_eps':10}
          (d/'protocol_manifest.json').write_text(json.dumps(dict(protocol=PENDULUM_23,task=prepare.TASK,
              root_seed=seed,git_commit='hand-fixture-only',config=config,eval_policy='fixture')))
          records=[]
          for step in range(0,1000001,50000):
            r=point(step,(value,)*10); r['lengths']=[1]*10; r['seeds']=list(range(10))
            r['episodes']=[dict(r['episodes'][0],episode_id=e,environment_seed=e,length=1) for e in range(10)]
            records.append(r); snap=d/'eval_snapshots'/f'{step:07d}'; (snap/'fixture').mkdir(parents=True)
            (snap/'latest').write_text('fixture'); (snap/'fixture'/'done').touch()
            (snap/'fixture'/'agent.pkl').write_bytes(b'not-an-executable-checkpoint')
            (snap/'evaluation_snapshot.json').write_text(json.dumps(dict(snapshot_id=r['snapshot_id'],
                actual_action_step=step,update_id=step,git_commit='hand-fixture-only')))
          (d/'evaluations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
      index=root/'index.json'; index.write_text(json.dumps(dict(runs=runs)))
      cmd=[sys.executable,'-m','analysis.return_metrics','--index',str(index),'--require-complete']
      passed=subprocess.run(cmd+['--profile','pendulum-seeds23','--output',str(root/'good')],capture_output=True,text=True)
      self.assertEqual(passed.returncode,0,passed.stderr)
      tables=json.loads((root/'good/tables.json').read_text())
      self.assertEqual(len(tables['episodes']),840)
      self.assertEqual([r['auc_return_per_budget'] for r in tables['paired_seed_differences']],[3,-2])
      self.assertEqual([r['auc_return_per_budget'] for r in tables['overall_metrics']],[15,15.5])
      wrong=subprocess.run(cmd+['--profile','dmc16','--output',str(root/'wrong')],capture_output=True,text=True)
      self.assertEqual(wrong.returncode,2,wrong.stderr)
      index.write_text(json.dumps(dict(runs=runs[:-1])))
      missing=subprocess.run(cmd+['--profile','pendulum-seeds23','--output',str(root/'missing')],capture_output=True,text=True)
      self.assertEqual(missing.returncode,2,missing.stderr)
      self.assertIsNone(next(r for r in json.loads((root/'missing/tables.json').read_text())['overall_metrics'] if r['group']=='dt')['auc_return_per_budget'])


if __name__=='__main__': unittest.main()
