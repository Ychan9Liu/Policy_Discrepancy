"""CPU preparation fixtures, never a real environment, agent, or GPU test.

AST extraction executes the production validation/ledger/freeze functions while
avoiding JAX imports. This does not establish full-agent or CUDA acceptance.
"""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analysis.return_metrics import MetricSpec, aggregate, analyze_run
from dreamerv3 import protocol_scope as scope
from scripts import prepare_dmc16 as prep
from scripts.prepare_m2_v1 import resolve
from tests.test_return_metrics import point

REPO = Path(__file__).resolve().parents[1]
REFERENCE = '133888f4116db1898784757269fce7ae0b629c67'


def extracted(path, name, namespace=None):
  node = next(x for x in ast.parse((REPO/path).read_text()).body
              if isinstance(x, (ast.FunctionDef,ast.ClassDef)) and x.name == name)
  namespace = {} if namespace is None else namespace
  exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
  return namespace[name]


def frozen_function():
  state = extracted('embodied/run/protocol_v1.py','ProtocolState',dict(math=math))
  namespace = dict(vars(scope), Path=Path, json=json, hashlib=hashlib,
      datetime=datetime, timezone=timezone, ProtocolState=state)
  return extracted('dreamerv3/freeze_c.py','freeze',namespace)


class Dmc16PreparationTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.temporary = TemporaryDirectory()
    cls.root = Path(cls.temporary.name)
    resources = cls.root/'resources.json'
    resources.write_text(json.dumps(dict(server='sv1',host='fixture-host-not-a-server',
        gpus={str(i):dict(uuid=f'fixture-UUID-{i}') for i in range(8)})))
    cls.output = cls.root/'preparation'
    with patch.object(prep,'source_state',return_value=('fixture-not-a-training-SHA','unrelated docs dirty')):
      prep.prepare(cls.output,'formal-m2-dmc16-fixture',resources)
    cls.plan=json.loads((cls.output/'planned-run-index.json').read_text())

  @classmethod
  def tearDownClass(cls):
    cls.temporary.cleanup()

  def config(self,task=None,mode='logging'):
    return resolve(['--configs','m2_dmc16','--task',task or scope.DMC16_TASKS[0],
        '--agent.rep_probe.mode',mode,'--jax.prealloc','False','--logger.outputs','jsonl','wandb'])

  def test_exact_twenty_minus_four_and_explicit_cup_alias(self):
    inventory=prep.verify_task_inventory()
    self.assertEqual(len(inventory['visual_tasks']),20)
    self.assertEqual(len(scope.DMC16_TASKS),16)
    self.assertFalse(set(scope.V1_TASKS)&set(scope.DMC16_TASKS))
    self.assertEqual(prep.SCORE_ALIASES,{'dmc_cup_catch':'dmc_ball_in_cup_catch'})

  def test_closed_extension_original_and_hopper_scopes(self):
    for task in scope.DMC16_TASKS:
      for mode in ('logging','dt'):
        self.assertEqual(scope.validate_scope(scope.DMC16_SCRIPT,task,0,mode),scope.DMC16)
    for task,seed,mode in [(t,0,'logging') for t in scope.V1_TASKS]+[
        (scope.DMC16_TASKS[0],1,'dt'),(scope.DMC16_TASKS[0],False,'dt'),
        (scope.DMC16_TASKS[0],0,'constant'),(scope.DMC16_TASKS[0],0,'shuffle'),
        ('dmc_ball_in_cup_catch',0,'dt')]:
      with self.subTest(task=task,seed=seed,mode=mode),self.assertRaises(ValueError):
        scope.validate_scope(scope.DMC16_SCRIPT,task,seed,mode)
    self.assertEqual(scope.tasks_for_protocol(scope.V1),scope.V1_TASKS)
    self.assertEqual(scope.tasks_for_protocol(scope.HOPPER_123),scope.V1_TASKS)
    self.assertEqual(MetricSpec().tasks,scope.V1_TASKS)
    with self.assertRaises(ValueError): scope.validate_scope('protocol_v1',scope.V1_TASKS[0],1,'dt')

  def test_32_full_configs_keep_original_scientific_settings(self):
    original=dict(resolve(['--configs','m2_v1','--task','dmc_hopper_hop',
        '--jax.prealloc','False','--logger.outputs','jsonl','wandb']).flat)
    for row in self.plan['runs']:
      config=self.config(row['task'],row['mode'])
      prep.check_dmc16(config)
      self.assertLessEqual({k for k in original if original[k]!=dict(config.flat)[k]},
                           {'script','task','agent.rep_probe.mode'})
      self.assertEqual(row['config']['run.action_budget'],1000000)
      self.assertEqual(row['config']['run.eval_every_actions'],50000)
      self.assertEqual(row['config']['run.eval_eps'],10)
    for changes in ({'agent.rep_probe.alpha':19},{'seed':1},{'run.action_budget':1000},
        {'agent.ac_grads':True},{'agent.loss_scales.rep':.2},{'agent.dyn.rssm.free_nats':2},
        {'logger.outputs':['jsonl']},{'agent.rep_probe.mode':'constant'}):
      with self.subTest(changes=changes),self.assertRaises(ValueError):
        prep.check_dmc16(self.config().update(changes))

  def test_actual_runner_pre_agent_guards_accept_new_scope_reject_original_expansion(self):
    node=next(n for n in ast.parse((REPO/'embodied/run/protocol_v1.py').read_text()).body
              if isinstance(n,ast.FunctionDef) and n.name=='run')
    boundary=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and
                  any(isinstance(t,ast.Name) and t.id=='logdir' for t in n.targets))
    node.body=node.body[:boundary]+[ast.Return(ast.Constant(True))]
    namespace=dict(vars(scope));exec(compile(ast.fix_missing_locations(
        ast.Module(body=[node],type_ignores=[])),'<actual pre-agent guards>','exec'),namespace)
    def invoke(config):
      return namespace['run'](None,None,None,None,None,config.run,config)
    for task in scope.DMC16_TASKS:
      self.assertTrue(invoke(self.config(task)))
      with self.assertRaisesRegex(ValueError,'task set'):
        invoke(self.config(task).update({'script':'protocol_v1'}))
    for changes in ({'run.eval_eps':2},{'env.dmc.repeat':2},{'agent.ac_grads':True},
                    {'agent.loss_scales.rep':.2}):
      with self.subTest(changes=changes),self.assertRaises(ValueError):
        invoke(self.config().update(changes))

  def test_actual_wrapper_prefix_cup_mapping_without_creating_an_environment(self):
    tree=ast.parse((REPO/'embodied/envs/dmc.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DMC')
    ctor=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    boundary=next(i for i,n in enumerate(ctor.body) if isinstance(n,ast.Assign) and
        any(isinstance(t,ast.Attribute) and t.attr=='_dmenv' for t in n.targets))
    ctor.body=ctor.body[:boundary]
    observed=[]
    fake_suite=SimpleNamespace(load=lambda domain,task,**kw: observed.append((domain,task,kw)))
    namespace=dict(os=SimpleNamespace(environ={}),suite=fake_suite)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ctor],type_ignores=[])),
                 '<wrapper dispatch only; fake suite>','exec'),namespace)
    for task in scope.DMC16_TASKS:
      namespace['__init__'](SimpleNamespace(DEFAULT_CAMERAS={}),task[4:],seed=123)
    self.assertEqual(observed[1],('ball_in_cup','catch',{'task_kwargs':{'random':123}}))
    self.assertEqual(len(observed),16)

  def test_exact_original_rng_derivation_and_fixed_evaluation_grid(self):
    actual=extracted('embodied/run/protocol_v1.py','seed32',dict(hashlib=hashlib))
    for task in scope.DMC16_TASKS:
      rng=json.loads((self.output/(task+'-random-streams.json')).read_text())
      self.assertEqual(rng['train_env_seeds'],[actual(0,task,'train_env',i) for i in range(16)])
      self.assertEqual(rng['replay_seed'],actual(0,task,'train_replay'))
      self.assertEqual([r['action_step'] for r in rng['evaluations']],list(range(0,1000001,50000)))
      for j,row in enumerate(rng['evaluations']):
        self.assertEqual(len(row['episodes']),10)
        self.assertEqual([e['environment_seed'] for e in row['episodes']],
                         [actual(0,task,'eval_env',j,i) for i in range(10)])

  def test_fixed_queues_no_sharing_and_paused_plan_cannot_be_authorized(self):
    prep.validate_plan(self.plan)
    for gpu in range(8):
      rows=[r for r in self.plan['runs'] if r['physical_gpu']==gpu]
      self.assertEqual([r['queue_position'] for r in rows],[1,2,3,4])
      self.assertEqual({r['mode'] for r in rows},{'logging' if gpu%2==0 else 'dt'})
    for field,value in [('authorized_to_train',True),('auto_start_when_idle',True)]:
      plan=copy.deepcopy(self.plan);plan[field]=value
      with self.assertRaises(ValueError):prep.validate_plan(plan)
    for field,value in [('physical_gpu',8),('seed',1),('mode','shuffle'),
                        ('launch_argv',['not-authorized']),('state','running')]:
      plan=copy.deepcopy(self.plan);plan['runs'][0][field]=value
      with self.subTest(field=field),self.assertRaises(ValueError):prep.validate_plan(plan)
    self.assertEqual(self.plan['total_evaluation_episodes'],6720)
    self.assertEqual(self.plan['total_train_actions'],32000000)
    self.assertTrue(all(r['auc_return_per_budget'] is None for r in self.plan['runs']))

  def test_all_generated_artifact_hashes_and_no_overwrite(self):
    manifest=json.loads((self.output/'artifact-hashes.json').read_text())
    for name,digest in manifest.items():
      self.assertEqual(hashlib.sha256((self.output/name).read_bytes()).hexdigest(),digest,name)
    with self.assertRaises(FileExistsError):
      prep.prepare(self.output,'formal-m2-dmc16-fixture',self.root/'resources.json')

  def test_runtime_dirty_source_refused_without_touching_user_drafts(self):
    for dirty,untracked in [(b'changed runtime',b''),(b'',b'untracked runtime')]:
      responses=[b'fixture-SHA\n',dirty] + ([untracked] if not dirty else [])
      with patch.object(prep.subprocess,'check_output',side_effect=responses):
        with self.assertRaises(ValueError):prep.source_state()

  def test_freeze_actual_production_function_all_16_tasks_half_open_window(self):
    freeze=frozen_function()
    for task in scope.DMC16_TASKS:
      with TemporaryDirectory() as temporary:
        root=Path(temporary)
        config=dict(self.config(task).flat)
        def write(name,value): (root/name).write_text(json.dumps(value))
        write('protocol_manifest.json',dict(protocol=scope.DMC16,task=task,root_seed=0,
            git_commit='artifact-fixture-only',config=config))
        receipts=[dict(update_id=i,start_action=step,match_sum=s,match_count=n,invalid_count=0)
            for i,(step,s,n) in enumerate(((99999,.4,1),(100000,.5,1),(299999,1.5,3),(300000,.2,1)))]
        (root/'update_receipts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in receipts))
        (root/'evaluations.jsonl').write_text(''.join(json.dumps(dict(action_step=s))+'\n'
            for s in range(0,1000001,50000)))
        write('final_state.json',dict(git_commit='artifact-fixture-only',train_action_steps=1000000,
            updates=4,evaluations=21,match_sum=2.,match_count=4))
        write('matching_result.json',dict(match_sum=2.,match_count=4,c=.5,start=100000,end=300000))
        result=freeze(root,root/'c.json')
        self.assertEqual((result['c'],result['included_update_ids'],result['root_seed']),(.5,[1,2],0))
        self.assertEqual((root/'c.json.sha256').read_text().strip(),
                         hashlib.sha256((root/'c.json').read_bytes()).hexdigest())
        self.assertEqual(freeze(root,root/'c.json'),result)
        config['seed']=1
        write('protocol_manifest.json',dict(protocol=scope.DMC16,task=task,root_seed=1,
            git_commit='artifact-fixture-only',config=config))
        with self.assertRaises(ValueError):freeze(root,root/'bad-c.json')

  def test_16_task_metrics_strict_composition_and_original_profile_unchanged(self):
    spec=MetricSpec(tasks=scope.DMC16_TASKS,groups=('baseline','dt'),scope=scope.DMC16)
    rows=[dict(task=task,group=group,seed=0,run_id=task+group,
        auc_complete=True,tail_complete=True,auc_return_per_budget=i+delta,tail_mean_return=i+delta)
        for i,task in enumerate(scope.DMC16_TASKS) for group,delta in [('baseline',0),('dt',2)]]
    tasks,overall,differences=aggregate(rows,spec)
    self.assertEqual([r['auc_return_per_budget'] for r in overall],[7.5,9.5])
    self.assertEqual(len(differences),17)
    self.assertTrue(all(r['auc_return_per_budget']==2 for r in differences))
    self.assertIsNone(aggregate(rows[:-1],spec)[1][1]['auc_return_per_budget'])
    self.assertEqual(MetricSpec().groups,('baseline','dt','constant','shuffle'))
    for changes in [dict(seeds=(1,)),dict(tasks=scope.DMC16_TASKS[:-1]),
                    dict(groups=('baseline','dt','constant')),dict(budget=500000)]:
      with self.assertRaises(ValueError):
        MetricSpec(**(dict(tasks=scope.DMC16_TASKS,groups=('baseline','dt'),scope=scope.DMC16)|changes))

  def test_original_v1_and_hopper_c_artifact_semantics_match_pre_extension(self):
    state=extracted('embodied/run/protocol_v1.py','ProtocolState',dict(math=math))
    namespace=dict(vars(scope),Path=Path,json=json,hashlib=hashlib,
                   datetime=datetime,timezone=timezone,ProtocolState=state)
    old_source=subprocess.check_output(['git','show',REFERENCE+':dreamerv3/freeze_c.py'],cwd=REPO)
    old_node=next(n for n in ast.parse(old_source).body if isinstance(n,ast.FunctionDef) and n.name=='freeze')
    exec(compile(ast.Module(body=[old_node],type_ignores=[]),'<pre-extension freeze>','exec'),namespace)
    for protocol,script,seed in [(scope.V1,'protocol_v1',0),(scope.HOPPER_123,scope.HOPPER_SCRIPT,1)]:
      with TemporaryDirectory() as temporary:
        root=Path(temporary)
        config=dict(self.config().flat);config.update(script=script,seed=seed)
        def write(name,value):(root/name).write_text(json.dumps(value))
        write('protocol_manifest.json',dict(protocol=protocol,task='dmc_hopper_hop',root_seed=seed,
              git_commit='artifact-fixture-only',config=config))
        receipts=[dict(update_id=i,start_action=step,match_sum=s,match_count=n,invalid_count=0)
            for i,(step,s,n) in enumerate(((100000,.5,1),(299999,1.5,3)))]
        (root/'update_receipts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in receipts))
        (root/'evaluations.jsonl').write_text(''.join(json.dumps(dict(action_step=s))+'\n'
            for s in range(0,1000001,50000)))
        write('final_state.json',dict(git_commit='artifact-fixture-only',train_action_steps=1000000,
            updates=2,evaluations=21,match_sum=2.,match_count=4))
        write('matching_result.json',dict(match_sum=2.,match_count=4,c=.5,start=100000,end=300000))
        old=namespace['freeze'](root,root/'old.json');new=frozen_function()(root,root/'new.json')
        old.pop('frozen_at_utc');new.pop('frozen_at_utc')
        self.assertEqual(old,new)

  def test_protocol_ledger_evaluation_and_training_computation_source_unchanged(self):
    for path in ('dreamerv3/agent.py','dreamerv3/rssm.py','dreamerv3/rep_probe.py',
                 'embodied/jax/agent.py','embodied/envs/dmc.py'):
      old=subprocess.check_output(['git','show',REFERENCE+':'+path],cwd=REPO)
      self.assertEqual((REPO/path).read_bytes().replace(b'\r\n',b'\n'),old,path)
    path='embodied/run/protocol_v1.py'
    old=ast.parse(subprocess.check_output(['git','show',REFERENCE+':'+path],cwd=REPO))
    new=ast.parse((REPO/path).read_text())
    old_defs={n.name:ast.dump(n) for n in old.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    new_defs={n.name:ast.dump(n) for n in new.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    for name in old_defs:
      if name!='run':self.assertEqual(old_defs[name],new_defs[name],name)
    def normalize_run(tree):
      node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
      class Guard(ast.NodeTransformer):
        def visit_If(self,n):
          if (isinstance(n.test,ast.Compare) and ast.dump(n.test.left)==ast.dump(
              ast.Attribute(value=ast.Name(id='config',ctx=ast.Load()),attr='task',ctx=ast.Load()))):
            return ast.Pass()
          return self.generic_visit(n)
      return ast.dump(Guard().visit(node))
    self.assertEqual(normalize_run(old),normalize_run(new))

  def test_complete_32_run_cli_fixture_and_missing_grid_never_reweighted(self):
    # Files/markers only. No checkpoint is executed or environment created.
    with TemporaryDirectory() as temporary:
      root=Path(temporary);runs=[]
      for i,task in enumerate(scope.DMC16_TASKS):
        for mode,delta in [('logging',0),('dt',2)]:
          folder=root/(task+'-'+mode);folder.mkdir();runs.append(dict(directory=str(folder),run_id=folder.name))
          config={'seed':0,'agent.rep_probe.mode':mode,'run.engineering_fixture':False,
                  'run.action_budget':1000000,'run.eval_eps':10}
          (folder/'protocol_manifest.json').write_text(json.dumps(dict(protocol=scope.DMC16,
              task=task,root_seed=0,git_commit='hand-fixture-only',config=config,eval_policy='fixture')))
          records=[]
          for step in range(0,1000001,50000):
            row=point(step,(i+delta,)*10);row['lengths']=[1]*10;row['seeds']=list(range(10))
            template=row['episodes'][0]
            row['episodes']=[dict(template,episode_id=e,environment_seed=e,length=1) for e in range(10)]
            records.append(row)
            snap=folder/'eval_snapshots'/f'{step:07d}';(snap/'fixture').mkdir(parents=True)
            (snap/'latest').write_text('fixture');(snap/'fixture'/'done').touch()
            (snap/'fixture'/'agent.pkl').write_bytes(b'not-an-executable-checkpoint')
            (snap/'evaluation_snapshot.json').write_text(json.dumps(dict(snapshot_id=row['snapshot_id'],
                actual_action_step=step,update_id=step,git_commit='hand-fixture-only')))
          (folder/'evaluations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
      index=root/'index.json';index.write_text(json.dumps(dict(runs=runs)))
      command=[sys.executable,'-m','analysis.return_metrics','--index',str(index),'--require-complete']
      passed=subprocess.run(command+['--profile','dmc16','--output',str(root/'complete')],capture_output=True,text=True)
      self.assertEqual(passed.returncode,0,passed.stderr)
      tables=json.loads((root/'complete/tables.json').read_text())
      self.assertEqual((len(tables['episodes']),len(tables['evaluation_points']),len(tables['seed_metrics'])),(6720,672,32))
      self.assertEqual([r['auc_return_per_budget'] for r in tables['overall_metrics']],[7.5,9.5])
      wrong=subprocess.run(command+['--output',str(root/'old-v1')],capture_output=True,text=True)
      self.assertEqual(wrong.returncode,2,wrong.stderr)
      damaged=Path(runs[0]['directory'])/'evaluations.jsonl'
      records=[json.loads(line) for line in damaged.read_text().splitlines()]
      damaged.write_text(''.join(json.dumps(r)+'\n' for r in records if r['action_step']!=50000))
      missing=subprocess.run(command+['--profile','dmc16','--output',str(root/'missing')],capture_output=True,text=True)
      self.assertEqual(missing.returncode,2,missing.stderr)
      result=json.loads((root/'missing/tables.json').read_text())['overall_metrics'][0]
      self.assertIsNone(result['auc_return_per_budget'])
      self.assertTrue(result['tail_complete'])


if __name__=='__main__':unittest.main()
