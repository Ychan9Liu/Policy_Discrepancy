import unittest
import json
import sys
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, MagicMock
from dataclasses import replace
from dreamerv3.protocol_scope import (
    VISION20_SEED1,VISION20_SCRIPT,VISION20_TASKS,V1_TASKS,DMC16_TASKS,
    validate_scope,tasks_for_protocol)
from scripts.prepare_vision20_seed1 import check_config,NEW_TASKS,CARDS
from scripts.prepare_m2_v1 import resolve
from analysis.return_metrics import MetricSpec,aggregate
from tests.test_return_metrics import point
from scripts import prepare_vision20_seed1 as prepare
from scripts import run_vision20_seed1 as runner


class Vision20Seed1Test(unittest.TestCase):
  def make_plan(self,root):
    hopper=[]
    for mode in ('logging','dt'):
      argv=['python','-u','-m','dreamerv3.main','--configs','m2_hopper_seeds123',
          '--seed','1','--logdir','/old/'+mode,'--agent.rep_probe.mode',mode,
          '--jax.prealloc','False','--logger.outputs','jsonl','wandb']
      hopper.append(dict(task='dmc_hopper_hop',seed=1,mode=mode,protocol='M2-v1-hopper-seeds123-baseline-dt',
          state='accepted',exit_code=0,code_sha='old-sha',run_audit=dict(updates=249493,evaluations=21),
          review_argv=argv,config=json.loads(json.dumps(dict(resolve(argv[4:]).flat)))))
    hp=root/'hopper.json';hp.write_text(json.dumps(dict(runs=hopper)))
    res=root/'res.json';res.write_text(json.dumps(dict(servers={s:dict(host=h,
        gpus='\n'.join(f'{g}, fixture-{s}-{g}, 0 MiB' for g in range(8))) for s,h in prepare.HOSTS.items()})))
    with patch.object(prepare,'source_state',return_value=('fixture-sha','retained drafts')):
      prepare.prepare(root/'out',res,hp,'formal-m2-vision20-seed1-fixture')
    runner.split(root/'out/planned-run-index.json','sv2',root/'sv2.json')
    return json.loads((root/'sv2.json').read_text())

  def test_preparation_and_queue_refuse_config_card_or_seed_tampering(self):
    with TemporaryDirectory() as temp:
      p=self.make_plan(Path(temp));runner.validate(p)
      self.assertEqual(len(p['runs']),14);self.assertEqual({r['physical_gpu'] for r in p['runs']},{0,1,2,3,4,5,7})
      self.assertTrue(all(r['seed']==1 and r['launch_argv'] is None for r in p['runs']))
      for key,value in [('physical_gpu',6),('seed',0),('mode','constant'),('directory','/old')]:
        bad=json.loads(json.dumps(p));bad['runs'][0][key]=value
        with self.assertRaises(ValueError):runner.validate(bad)

  def test_existing_root_is_refused_without_GPU_or_spawn(self):
    with TemporaryDirectory() as temp:
      p=self.make_plan(Path(temp))
      with patch.object(runner,'read',return_value=p),patch.object(runner,'validate'),patch.object(Path,'exists',return_value=True),patch.object(runner,'require_free') as guard,patch.object(runner.subprocess,'Popen') as spawn:
        with self.assertRaises(FileExistsError):runner.run('fixture.json')
        guard.assert_not_called();spawn.assert_not_called()

  def test_sv1_initial_capacity_exception_is_explicit_and_keeps_per_run_floor(self):
    with TemporaryDirectory() as temp:
      root=Path(temp);self.make_plan(root)
      runner.split(root/'out/planned-run-index.json','sv1',root/'sv1.json')
      p=json.loads((root/'sv1.json').read_text());p['authorized_to_train']=True
      for r in p['runs']:
        r.update(launch_argv=r['review_argv'],egl_index=r['physical_gpu'],state='queued',
            verified_compute_and_graphics_same_gpu=True)
        r['environment_template']['MUJOCO_EGL_DEVICE_ID']=str(r['egl_index'])
      runner.validate(p)
      for free,exception in ((340_000_000_000,False),(59_999_999_999,True)):
        with patch.object(runner,'read',return_value=p),patch.object(runner,'validate'),patch.object(Path,'exists',return_value=False),patch.object(runner.platform,'node',return_value=p['host']),patch.object(runner,'clean_sha',return_value=p['code_sha']),patch.object(runner,'require_free',return_value={}),patch.object(runner.shutil,'disk_usage',return_value=MagicMock(free=free)),patch.object(Path,'mkdir') as mkdir,patch.object(runner.subprocess,'Popen') as spawn:
          with self.assertRaises(RuntimeError):runner.run('fixture.json',exception)
          mkdir.assert_not_called();spawn.assert_not_called()
      proc=MagicMock(pid=123,returncode=0);proc.poll.return_value=0
      with patch.object(runner,'read',return_value=p),patch.object(runner,'validate'),patch.object(Path,'exists',return_value=False),patch.object(Path,'mkdir'),patch.object(Path,'open',return_value=MagicMock()),patch.object(runner,'dump') as saved,patch.object(runner.platform,'node',return_value=p['host']),patch.object(runner,'clean_sha',return_value=p['code_sha']),patch.object(runner,'require_free',return_value={}),patch.object(runner.shutil,'disk_usage',return_value=MagicMock(free=340_000_000_000)),patch.object(runner.subprocess,'Popen',return_value=proc) as launched,patch.object(runner,'audit_run',return_value={}),patch.object(runner.time,'sleep'),patch.object(runner.subprocess,'run',return_value=MagicMock(returncode=0)):
        runner.run('fixture.json',True)
      self.assertEqual(launched.call_count,24)
      record=next(c.args[1] for c in saved.call_args_list if c.args[0].name=='capacity-at-start.json')
      self.assertFalse(record['reservation_satisfied']);self.assertTrue(record['explicit_sv1_capacity_exception'])
      self.assertEqual(record['per_run_floor_bytes'],60_000_000_000)

  def test_queue_audits_baseline_before_dt_and_blocks_failed_predecessor(self):
    for exit_code in (0,1):
      with TemporaryDirectory() as temp:
        p=self.make_plan(Path(temp));p['authorized_to_train']=True
        for r in p['runs']:
          r.update(launch_argv=r['review_argv'],egl_index=r['physical_gpu'],state='queued',
              verified_compute_and_graphics_same_gpu=True)
          r['environment_template']['MUJOCO_EGL_DEVICE_ID']=str(r['egl_index'])
        runner.validate(p)
        proc=MagicMock(pid=123,returncode=exit_code);proc.poll.return_value=exit_code
        audited=set()
        def audit(row,scope):
          self.assertEqual(scope,VISION20_SEED1);audited.add((row['task'],row['mode']));return dict(passed=True)
        def spawn(argv,**kwargs):
          task=argv[argv.index('--task')+1];mode=argv[argv.index('--agent.rep_probe.mode')+1]
          if mode=='dt':self.assertIn((task,'logging'),audited)
          return proc
        disk=MagicMock(free=10**15)
        with patch.object(runner,'read',return_value=p),patch.object(runner,'validate'),patch.object(Path,'exists',return_value=False),patch.object(Path,'mkdir'),patch.object(Path,'open',return_value=MagicMock()),patch.object(runner,'dump'),patch.object(runner.platform,'node',return_value=p['host']),patch.object(runner,'clean_sha',return_value=p['code_sha']),patch.object(runner,'require_free',return_value={}),patch.object(runner.shutil,'disk_usage',return_value=disk),patch.object(runner.subprocess,'Popen',side_effect=spawn) as launched,patch.object(runner,'audit_run',side_effect=audit),patch.object(runner.time,'sleep'),patch.object(runner.subprocess,'run',return_value=MagicMock(returncode=0)):
          runner.run('fixture.json')
        self.assertEqual(launched.call_count,14 if exit_code==0 else 7)
        self.assertTrue(all(r['state']==runner.AUDITED for r in p['runs']) if exit_code==0 else
            all(r['state']=='blocked' for r in p['runs'] if r['mode']=='dt'))

  def test_exact_tasks_seed_modes_and_original_scopes_remain_closed(self):
    self.assertEqual(len(set(VISION20_TASKS)),20)
    self.assertEqual(tasks_for_protocol(VISION20_SEED1),V1_TASKS+DMC16_TASKS)
    for task in VISION20_TASKS:
      for mode in ('logging','dt'):
        self.assertEqual(validate_scope(VISION20_SCRIPT,task,1,mode),VISION20_SEED1)
    for seed in (0,2,True):
      with self.assertRaises(ValueError):validate_scope(VISION20_SCRIPT,VISION20_TASKS[0],seed,'dt')
    for mode in ('constant','shuffle'):
      with self.assertRaises(ValueError):validate_scope(VISION20_SCRIPT,VISION20_TASKS[0],1,mode)
    with self.assertRaises(ValueError):validate_scope('protocol_dmc16',DMC16_TASKS[0],1,'dt')
    with self.assertRaises(ValueError):validate_scope('protocol_v1',V1_TASKS[0],1,'dt')

  def test_forty_full_configs_only_approved_scope_seed_task_changes(self):
    for task in VISION20_TASKS:
      for mode in ('logging','dt'):
        c=resolve(['--configs','m2_vision20_seed1','--task',task,'--seed','1',
            '--agent.rep_probe.mode',mode,'--jax.prealloc','False','--logger.outputs','jsonl','wandb'])
        check_config(c)
    for field,value in [('agent.rep_probe.alpha',10),('run.eval_eps',1),
                        ('run.action_budget',4098),('agent.ac_grads',True)]:
      with self.assertRaises(ValueError):check_config(c.update({field:value}))

  def test_new_scope_metrics_require_all_twenty_seed1_pairs(self):
    spec=MetricSpec(seeds=(1,),tasks=VISION20_TASKS,groups=('baseline','dt'),scope=VISION20_SEED1)
    for field,value in [('seeds',(0,)),('tasks',DMC16_TASKS),('episodes',2),('groups',('baseline','dt','constant'))]:
      with self.assertRaises(ValueError):replace(spec,**{field:value})
    _,overall,_=aggregate([],spec)
    self.assertTrue(all(not r['auc_complete'] and not r['tail_complete'] for r in overall))
    self.assertEqual(len(NEW_TASKS),19);self.assertNotIn('dmc_hopper_hop',NEW_TASKS)
    self.assertEqual(len(CARDS),13);self.assertNotIn(('sv2',6),CARDS)

  def test_complete_cli_with_reused_hopper_and_missing_members_refused(self):
    from dreamerv3.protocol_scope import HOPPER_123
    with TemporaryDirectory() as temp:
      root=Path(temp);runs=[]
      for i,task in enumerate(VISION20_TASKS):
        for mode,delta in [('logging',0),('dt',2)]:
          d=root/(task+'-'+mode);d.mkdir();runs.append(dict(directory=str(d),run_id=d.name))
          protocol=HOPPER_123 if task=='dmc_hopper_hop' else VISION20_SEED1
          config={'seed':1,'agent.rep_probe.mode':mode,'run.engineering_fixture':False,
                  'run.action_budget':1000000,'run.eval_eps':10}
          (d/'protocol_manifest.json').write_text(json.dumps(dict(protocol=protocol,task=task,
              root_seed=1,git_commit='hand-fixture-only',config=config,eval_policy='fixture')))
          records=[]
          for step in range(0,1000001,50000):
            r=point(step,(i+delta,)*10);r['lengths']=[1]*10;r['seeds']=list(range(10))
            template=r['episodes'][0];r['episodes']=[dict(template,episode_id=e,
                environment_seed=e,length=1) for e in range(10)];records.append(r)
            snap=d/'eval_snapshots'/f'{step:07d}';(snap/'fixture').mkdir(parents=True)
            (snap/'latest').write_text('fixture');(snap/'fixture'/'done').touch()
            (snap/'fixture'/'agent.pkl').write_bytes(b'not-an-executable-checkpoint')
            (snap/'evaluation_snapshot.json').write_text(json.dumps(dict(snapshot_id=r['snapshot_id'],
                actual_action_step=step,update_id=step,git_commit='hand-fixture-only')))
          (d/'evaluations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
      index=root/'index.json';index.write_text(json.dumps(dict(runs=runs)))
      cmd=[sys.executable,'-m','analysis.return_metrics','--index',str(index),
          '--profile','vision20-seed1','--require-complete']
      out=root/'complete';result=subprocess.run(cmd+['--output',str(out)],capture_output=True,text=True)
      self.assertEqual(result.returncode,0,result.stderr)
      t=json.loads((out/'tables.json').read_text())
      self.assertEqual(len(t['seed_metrics']),40);self.assertEqual(len(t['episodes']),8400)
      self.assertEqual([r['auc_return_per_budget'] for r in t['overall_metrics']],[9.5,11.5])
      index.write_text(json.dumps(dict(runs=runs[2:])))
      missing=subprocess.run(cmd+['--output',str(root/'missing')],capture_output=True,text=True)
      self.assertEqual(missing.returncode,2,missing.stderr)
      t=json.loads((root/'missing/tables.json').read_text())
      self.assertTrue(all(r['auc_return_per_budget'] is None for r in t['overall_metrics']))


if __name__=='__main__':unittest.main()
