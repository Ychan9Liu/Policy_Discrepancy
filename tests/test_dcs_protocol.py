"""Engineering fixtures only; no approval of a noise strength or training seed."""
import copy
import hashlib
import json
import pickle
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from dreamerv3.protocol_scope import DCS_DYNAMIC, VISION20_TASKS
from dreamerv3.dcs_protocol import check_config, backgrounds
from embodied.run.dcs_dynamic import SplitFactory, seed_routes, environment_manifest
from embodied.run import protocol_v1
from scripts.prepare_m2_v1 import resolve


def capture(index, **kwargs):
  return index, kwargs


class DcsProtocolTest(unittest.TestCase):
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.root = Path(self.tmp.name)
    for name in ('train-fixture', 'eval-fixture'):
      (self.root/name).mkdir()
      for i in range(2):
        Image.new('RGB', (4,4), (i*30,40,50)).save(self.root/name/f'{i:05d}.png')
    self.declaration = dict(protocol=DCS_DYNAMIC, tasks=list(VISION20_TASKS),
        groups=['baseline','dt'], seed=0, engineering_only=True,
        background=dict(dataset=str(self.root), train_videos='train-fixture',
            eval_videos='eval-fixture', num_videos=-1, video_alpha=1.,
            ground_by_task={task:.3 for task in VISION20_TASKS}, seed=-1, cache_size=2))
    self.path = self.root/'declaration.json'
    self.write_declaration()

  def write_declaration(self):
    self.path.write_text(json.dumps(self.declaration))
    self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()

  def config(self, task=VISION20_TASKS[0], mode='logging'):
    return resolve(['--configs','m2_dcs_dynamic','--task',task,'--seed','0',
        '--agent.rep_probe.mode',mode,'--jax.prealloc','False',
        '--logger.outputs','jsonl','--dcs_protocol.declaration_file',str(self.path),
        '--dcs_protocol.declaration_sha256',self.digest,
        '--env.dmc.background.dataset',str(self.root),
        '--env.dmc.background.videos','train-fixture',
        '--env.dmc.background.ground_plane_alpha','0.3'])

  def test_all_forty_configs_keep_algorithm_and_production_settings(self):
    for task in VISION20_TASKS:
      base = self.config(task)
      gate = self.config(task,'dt')
      check_config(base)
      check_config(gate)
      self.assertEqual({k for k in base.flat if base.flat[k] != gate.flat[k]},
                       {'agent.rep_probe.mode'})
      self.assertEqual((base.run.action_budget,base.run.eval_every_actions,base.run.eval_eps),
                       (1000000,50000,10))
      self.assertEqual(base.agent.dt_latch.mode,'off')

  def test_declaration_and_legacy_scopes_are_fail_closed(self):
    from dreamerv3.main import validate_environment_config, make_env
    config = self.config()
    validate_environment_config(config)
    for change in ({'seed':1},{'env.dmc.background.mode':'clean'},
        {'env.dmc.background.video_alpha':.5},{'agent.rep_probe.alpha':10},
        {'agent.dt_latch.mode':'dtlatch'},{'run.action_budget':10000},
        {'run.report_every_actions':1},{'agent.ac_grads':True},
        {'dcs_protocol.declaration_sha256':'wrong'}):
      with self.assertRaises(ValueError): check_config(config.update(change))
    for script in ('protocol_v1','protocol_vision20_seed1','protocol_dtl_three'):
      with self.assertRaisesRegex(ValueError,'clean-only'):
        validate_environment_config(config.update(script=script))
    with self.assertRaisesRegex(ValueError,'override differs'):
      make_env(config,0,seed=0,background=dict(backgrounds(config)['eval'],video_alpha=.5))
    with self.assertRaisesRegex(ValueError,'override differs'):
      make_env(config,0,seed=0,background={'mode':'clean'})

  def test_factory_exact_train_eval_seeds_and_no_random_draws(self):
    config = self.config()
    before = np.random.get_state()
    factory = SplitFactory(capture,config)
    factory = pickle.loads(pickle.dumps(factory))
    for seed, route in seed_routes(config).items():
      index, kw = factory(route['index'],seed=seed)
      self.assertEqual(kw['background']['videos'],route['role']+'-fixture')
      self.assertEqual(kw['seed'],seed)
      if route['role'] == 'train':
        expected = protocol_v1.seed32(0,config.task,'train_env',index)
      else:
        expected = protocol_v1.seed32(0,config.task,'eval_env',
            route['action_step']//50000,route['episode'])
      self.assertEqual(seed,expected)
    after = np.random.get_state()
    for a,b in zip(before,after): np.testing.assert_array_equal(a,b)
    with self.assertRaises(ValueError): factory(0,seed=-1)
    with self.assertRaises(ValueError): factory(500,seed=next(iter(factory.routes)))
    self.assertEqual(len(factory.routes),16+21*10)
    with patch.object(protocol_v1,'seed32',return_value=7):
      with self.assertRaisesRegex(ValueError,'collision'): seed_routes(config)
    with patch('embodied.run.dcs_dynamic.seed_for_background',return_value=8):
      with self.assertRaisesRegex(ValueError,'background seed collision'): seed_routes(config)

  def test_dynamic_logs_cannot_be_classified_as_original_clean_metrics(self):
    from analysis.return_metrics import analyze_run
    config=self.config()
    root=self.root/'metric-input'
    root.mkdir()
    (root/'protocol_manifest.json').write_text(json.dumps(dict(
        protocol=DCS_DYNAMIC,task=config.task,root_seed=0,git_commit='fixture-sha',
        config=dict(config.flat),eval_policy='fixture')))
    (root/'evaluations.jsonl').write_text('')
    _,_,summary,audit=analyze_run(dict(directory=str(root)))
    self.assertIn('dcs_dynamic_requires_separate_metric_profile',audit['parse_issues'])

  def test_assets_hashed_and_disjoint_no_hidden_fallback(self):
    manifest = environment_manifest(self.config())
    self.assertEqual(len(manifest['routes']),226)
    for role in ('train','eval'):
      asset = manifest['assets'][role]
      self.assertTrue(asset['content_hashed'])
      self.assertEqual(len(asset['videos'][0]['frames']),2)
    self.declaration['background']['eval_videos']='train-fixture'
    self.write_declaration()
    with self.assertRaisesRegex(ValueError,'overlap'): environment_manifest(self.config())

  def test_actual_runner_delegates_without_changing_algorithm(self):
    from embodied.run.dcs_dynamic import run
    config = self.config().update(logdir=str(self.root/'run'))
    from types import SimpleNamespace
    args = SimpleNamespace(logdir=config.logdir)
    functions = [object(),object(),capture,object(),object()]
    with patch.object(protocol_v1,'run') as legacy:
      run(*functions,args,config)
      values=legacy.call_args.args
      self.assertIsInstance(values[2],SplitFactory)
      for index in (0,1,3,4): self.assertIs(values[index],functions[index])
      self.assertIs(values[-1],config)
      self.assertIs(values[-2],args)
    payload=(self.root/'run/dcs_environment.json').read_bytes()
    with patch.object(protocol_v1,'run'):
      run(*functions,args,config)
    self.assertEqual(payload,(self.root/'run/dcs_environment.json').read_bytes())
    (self.root/'run/dcs_environment.json').write_text('{}')
    with self.assertRaisesRegex(ValueError,'differs'): run(*functions,args,config)

  def test_core_source_is_byte_identical_to_pre_noise_training(self):
    repo=Path(__file__).resolve().parents[1]
    reference='34c7a8570569791d791c3fd94b170dde47ab8d79'
    for path in ('dreamerv3/agent.py','dreamerv3/rssm.py','dreamerv3/rep_probe.py',
        'embodied/jax/agent.py','embodied/envs/dmc.py','embodied/run/protocol_v1.py'):
      old=subprocess.check_output(['git','show',reference+':'+path],cwd=repo)
      self.assertEqual(old.replace(b'\r\n',b'\n'),
          (repo/path).read_bytes().replace(b'\r\n',b'\n'),path)

  def test_queue_eight_running_eight_ready_and_fixed_serial_lanes(self):
    from scripts import run_dcs_dynamic as queue
    with patch.object(queue,'clean_sha',return_value='fixture-commit'):
      queue.prepare(self.root/'prepared',self.path,'dcs-dynamic-vision20-fixture','all40')
    plan=json.loads((self.root/'prepared/plan.json').read_text())
    queue.validate(plan)
    rows=plan['runs']
    first=queue.launchable(rows)
    self.assertEqual([x['task'] for x in first],list(VISION20_TASKS[:8]))
    self.assertEqual([x['physical_gpu'] for x in first],list(range(8)))
    self.assertTrue(all(x['mode']=='logging' for x in first))
    for row in first: row['state']='running'
    self.assertEqual(queue.launchable(rows),[])
    self.assertEqual(sum(r['state']=='running' for r in rows),8)
    self.assertEqual(sum(r['state']=='queued' for r in rows),8)
    self.assertEqual(sum(r['state']=='backlog' for r in rows),24)
    for row in first: row['state']='accepted'
    second=queue.launchable(rows)
    self.assertTrue(all(x['mode']=='dt' for x in second))
    second[0]['state']='failed'
    queue.refresh_queue(rows)
    self.assertTrue(all(r['state']=='blocked_predecessor' for r in rows
        if r['physical_gpu']==0 and r['lane_position']>1))
    for row in second[1:]: row['state']='accepted'
    self.assertEqual(len(queue.launchable(rows)),7)

  def test_first16_rollout_and_tampering_are_not_silent_expansion(self):
    from scripts import run_dcs_dynamic as queue
    with patch.object(queue,'clean_sha',return_value='fixture-commit'):
      queue.prepare(self.root/'prepared',self.path,'dcs-dynamic-vision20-fixture','first16')
      with self.assertRaises(FileExistsError):
        queue.prepare(self.root/'prepared',self.path,'dcs-dynamic-vision20-fixture','first16')
    plan=json.loads((self.root/'prepared/plan.json').read_text())
    self.assertEqual(sum(r['enabled'] for r in plan['runs']),16)
    for row in plan['runs']:
      if row['enabled']: row['state']='accepted'
    self.assertEqual(queue.launchable(plan['runs']),[])
    original=json.loads((self.root/'prepared/plan.json').read_text())
    for key,value in (('enabled',True),('physical_gpu',7),('mode','constant')):
      altered=copy.deepcopy(original)
      altered['runs'][16][key]=value
      with self.assertRaises(ValueError): queue.validate(altered)

  def test_formal_launch_requires_same_declaration_and_eight_cuda_acceptances(self):
    from scripts import run_dcs_dynamic as queue
    plan=dict(code_sha='fixture-code',declaration_sha256='fixture-declaration')
    proof=dict(plan,passed=True,engineering_only=True,
        runs=[dict(task=t,group=g) for t in queue.INTEGRATION_TASKS for g in ('baseline','dt')])
    queue.validate_acceptance(proof,plan)
    for field,value in (('code_sha','other'),('declaration_sha256','changed-strength'),
        ('engineering_only',False),('passed',False),('runs',proof['runs'][:-1])):
      with self.assertRaises(ValueError):
        queue.validate_acceptance(dict(proof,**{field:value}),plan)


if __name__ == '__main__':
  unittest.main()
