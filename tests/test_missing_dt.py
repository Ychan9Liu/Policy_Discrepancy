"""CPU boundary fixtures; no scientific conclusions or new training seeds."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch, MagicMock

from scripts import run_missing_dt as runner
from scripts.prepare_m2_v1 import resolve


class MissingDtTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.temp=TemporaryDirectory();cls.base=Path(cls.temp.name)
    rows=[]
    for task in runner.TASKS:
      for mode in ('logging','dt'):
        argv=[runner.PYTHON,'-u','-m','dreamerv3.main','--configs','m2_dmc16',
            '--task',task,'--seed','0','--logdir','/old/'+task+'/'+mode,
            '--agent.rep_probe.mode',mode,'--jax.prealloc','False','--logger.outputs','jsonl','wandb']
        rows.append(dict(task=task,seed=0,mode=mode,state=runner.AUDITED if mode=='logging' else 'blocked',
            pid=100 if mode=='logging' else None,config=json.loads(json.dumps(dict(resolve(argv[4:]).flat))),
            environment_template=dict(WANDB_ENTITY=runner.ENTITY,WANDB_PROJECT='PD_1',WANDB_MODE='online',
                WANDB_RESUME='never',JAX_PLATFORMS='cuda',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl'),
            run_audit=dict(updates=249493,evaluations=21,c=0.83) if mode=='logging' else None))
    source=cls.base/'source.json';source.write_text(json.dumps(dict(runs=rows)))
    resources=cls.base/'resources.json';resources.write_text(json.dumps(dict(servers={
        s:dict(host=h,gpus='\n'.join(f'{g}, fixture-{s}-{g}, 0 MiB, 0 %' for g in range(8)))
        for s,h in runner.HOSTS.items()})))
    with patch.object(runner,'source_state',return_value=('fixture-sha','retained drafts')):
      runner.prepare(cls.base/'plan',resources,[source],runner.PREFIX+'fixture')
    cls.plans={s:json.loads((cls.base/'plan'/(s+'-plan.json')).read_text()) for s in runner.HOSTS}

  @classmethod
  def tearDownClass(cls):cls.temp.cleanup()

  def bound(self,server):
    p=copy.deepcopy(self.plans[server]);p['authorized_to_train']=True
    for r in p['runs']:
      r.update(launch_argv=r['review_argv'],verified_compute_and_graphics_same_gpu=True,
          egl_index=r['physical_gpu'],authorized_to_train=True,state='queued')
      r['environment_template']['MUJOCO_EGL_DEVICE_ID']=str(r['egl_index'])
    return p

  def test_exact_eleven_only_Dt_on_seven_and_four_cards(self):
    for s,p in self.plans.items():
      runner.validate_plan(p);runner.validate_plan(self.bound(s))
      self.assertEqual(len(p['runs']),7 if s=='sv2' else 4)
      self.assertTrue(all(r['mode']=='dt' and r['seed']==0 for r in p['runs']))
      self.assertEqual({r['physical_gpu'] for r in p['runs']},set((0,1,2,3,4,5,7) if s=='sv2' else (4,5,6,7)))
      for r in p['runs']:
        self.assertEqual({k for k in r['config'] if r['config'][k]!=r['source_blocked']['config'][k]},{'logdir'})

  def test_completed_live_source_or_invalid_calibration_refused(self):
    for field,value in [('state','running'),('state',runner.AUDITED),('pid',123),('start_utc','already-started')]:
      p=copy.deepcopy(self.plans['sv2']);p['runs'][0]['source_blocked'][field]=value
      with self.subTest(field=field,value=value),self.assertRaises(ValueError):runner.validate_plan(p)
    for c in (0,float('nan'),1.1):
      p=copy.deepcopy(self.plans['sv2']);p['runs'][0]['source_baseline']['run_audit']['c']=c
      with self.assertRaises(ValueError):runner.validate_plan(p)

  def test_gpu6_server_seed_scope_or_output_tampering_refused(self):
    for field,value in [('physical_gpu',6),('seed',False),('seed',1),('mode','logging'),
        ('host','wrong'),('task','dmc_finger_turn_hard'),('directory','/old/completed')]:
      p=copy.deepcopy(self.plans['sv2']);p['runs'][0][field]=value
      with self.subTest(field=field),self.assertRaises(ValueError):runner.validate_plan(p)

  def test_numerical_override_rejected_even_if_stored_config_rewritten(self):
    for flag,value in [('--agent.rep_probe.alpha','10'),('--run.action_budget','4098'),
                       ('--agent.ac_grads','True'),('--run.eval_eps','1')]:
      p=copy.deepcopy(self.plans['sv2']);r=p['runs'][0];r['review_argv'] += [flag,value]
      r['config']=json.loads(json.dumps(dict(resolve(r['review_argv'][4:]).flat)))
      with self.subTest(flag=flag),self.assertRaises(ValueError):runner.validate_plan(p)

  def test_unbound_identity_collision_and_offline_logging_refused(self):
    p=self.bound('sv3');p['runs'][0]['environment_template']['MUJOCO_EGL_DEVICE_ID']='wrong'
    with self.assertRaises(ValueError):runner.validate_plan(p)
    p=copy.deepcopy(self.plans['sv3']);p['runs'][1]['environment_template']['WANDB_RUN_ID']=p['runs'][0]['environment_template']['WANDB_RUN_ID']
    with self.assertRaises(ValueError):runner.validate_plan(p)
    p=copy.deepcopy(self.plans['sv3']);p['runs'][0]['environment_template']['WANDB_MODE']='offline'
    with self.assertRaises(ValueError):runner.validate_plan(p)

  def test_clean_context_preserves_input_and_target_rng_configuration(self):
    polluted=dict(PATH='/usr/local/cuda-12.4/bin',CUDA_HOME='/usr/local/cuda-12.4',
        LD_LIBRARY_PATH=':/usr/local/cuda-12.4/lib64',JAX_PLATFORMS='cpu',XLA_FLAGS='fake')
    original=dict(polluted);r=self.bound('sv3')['runs'][0]
    env=runner.child_environment(r['environment_template'],inherited=runner.approved_context(polluted))
    self.assertEqual(polluted,original)
    for k in ('LD_LIBRARY_PATH','CUDA_HOME','XLA_FLAGS'):self.assertNotIn(k,env)
    self.assertEqual(env['CUDA_VISIBLE_DEVICES'],r['gpu_uuid'])
    self.assertEqual(env['JAX_PLATFORMS'],'cuda')
    self.assertEqual(r['config']['seed'],0)

  def test_existing_root_is_refused_before_GPU_or_spawn(self):
    p=self.bound('sv3');file=self.base/'bound-test.json';file.write_text(json.dumps(p))
    with patch.object(runner.platform,'node',return_value=p['host']),patch.object(runner,'clean_sha',return_value=p['code_sha']),patch.object(Path,'exists',return_value=True),patch.object(runner,'require_free') as free,patch.object(runner.subprocess,'Popen') as spawn:
      with self.assertRaises(FileExistsError):runner.run(file)
      free.assert_not_called();spawn.assert_not_called()

  def make_subset(self):
    source=self.base/'parent-subset.json';source.write_text(json.dumps(self.plans['sv2']))
    with TemporaryDirectory() as temp,patch.object(runner,'source_state',return_value=('new-sha','drafts')):
      output=Path(temp)/'plan.json';runner.subset(source,output)
      return json.loads(output.read_text())

  def test_partial_exact_four_cards_and_three_deferred_unchanged(self):
    parent=copy.deepcopy(self.plans['sv2']);p=self.make_subset();runner.validate_plan(p)
    self.assertEqual([r['physical_gpu'] for r in p['runs']],[1,3,5,7])
    self.assertEqual(p['deferred_tasks'],['dmc_acrobot_swingup','dmc_cartpole_balance','dmc_cartpole_swingup'])
    self.assertEqual(self.plans['sv2'],parent)
    self.assertFalse(p['authorized_to_train'])
    for r in p['runs']:
      old=next(x for x in parent['runs'] if x['task']==r['task'])
      self.assertEqual({k for k in r['config'] if r['config'][k]!=old['config'][k]},{'logdir'})
      self.assertEqual(r['environment_template']['CUDA_VISIBLE_DEVICES'],old['gpu_uuid'])
      self.assertNotEqual(r['environment_template']['WANDB_RUN_ID'],old['environment_template']['WANDB_RUN_ID'])
      self.assertIsNone(r['launch_argv']);self.assertIsNone(r['egl_index'])

  def test_partial_cannot_expand_or_drop_card_or_change_server(self):
    original=self.make_subset()
    for change in ('drop','expand','server','shard','campaign'):
      p=copy.deepcopy(original)
      if change=='drop':p['runs'].pop()
      elif change=='expand':p['runs'].append(copy.deepcopy(self.plans['sv2']['runs'][0]))
      elif change=='server':p.update(server='sv3',host=runner.HOSTS['sv3'])
      elif change=='shard':p['shard']='arbitrary'
      else:p['campaign']='formal-m2-dmc16-missing-dt-wrong'
      with self.subTest(change=change),self.assertRaises(ValueError):runner.validate_plan(p)

  def test_subset_refuses_other_host_already_split_or_existing_output(self):
    for p in (self.plans['sv3'],self.make_subset()):
      source=self.base/'bad-parent.json';source.write_text(json.dumps(p))
      with self.assertRaises(ValueError):runner.subset(source,self.base/'unused-subset.json')
    source=self.base/'full-parent.json';source.write_text(json.dumps(self.plans['sv2']))
    with self.assertRaises(ValueError):runner.subset(source,source)

  def test_partial_execution_never_checks_or_spawns_deferred_cards(self):
    p=self.make_subset();p['authorized_to_train']=True
    for r in p['runs']:
      r.update(launch_argv=r['review_argv'],verified_compute_and_graphics_same_gpu=True,
          egl_index=r['physical_gpu'],authorized_to_train=True,state='queued')
      r['environment_template']['MUJOCO_EGL_DEVICE_ID']=str(r['egl_index'])
    process=MagicMock(returncode=0,pid=12345);process.poll.return_value=0
    disk=MagicMock(free=10**15)
    def free(gpu,uuid):
      self.assertIn(gpu,(1,3,5,7));return dict(physical_gpu=gpu)
    runner.validate_plan(p)
    with patch.object(runner,'validate_plan'),patch.object(runner,'read',return_value=p),patch.object(runner,'dump'),patch.object(runner.platform,'node',return_value=p['host']),patch.object(runner,'clean_sha',return_value=p['code_sha']),patch.object(Path,'exists',return_value=False),patch.object(Path,'mkdir'),patch.object(Path,'open',return_value=MagicMock()),patch.object(runner.shutil,'disk_usage',return_value=disk),patch.object(runner,'require_free',side_effect=free) as guard,patch.object(runner.subprocess,'Popen',return_value=process) as spawn,patch.object(runner.subprocess,'check_output',return_value='fixture GPUs'),patch.object(runner,'audit_run',return_value={}),patch.object(runner,'finish') as finish,patch.object(runner.time,'sleep') as sleep:
      runner.run('fixture.json')
    self.assertEqual([x.args[0] for x in guard.call_args_list],[1,3,5,7]*2)
    self.assertEqual(spawn.call_count,4);finish.assert_called_once();sleep.assert_not_called()
    self.assertEqual({c.kwargs['env']['CUDA_VISIBLE_DEVICES'] for c in spawn.call_args_list},
        {r['gpu_uuid'] for r in p['runs']})
    self.assertTrue(all(r['state']=='accepted' for r in p['runs']))

  def make_remainder(self):
    with TemporaryDirectory() as temp:
      base=Path(temp);parent=base/'parent.json';parent.write_text(json.dumps(self.plans['sv2']))
      partial=base/'partial.json';partial.write_text(json.dumps(self.make_subset()))
      inventory=base/'resources.json';inventory.write_text(json.dumps(dict(servers={
          s:dict(host=h,gpus='\n'.join(f'{g}, fixture-{s}-{g}, 0 MiB' for g in range(8)))
          for s,h in runner.HOSTS.items()})))
      with patch.object(runner,'source_state',return_value=('new-sha','drafts')):
        runner.remainder(parent,partial,inventory,base/'out')
      return {s:json.loads((base/'out'/(s+'-plan.json')).read_text()) for s in runner.HOSTS}

  def test_remaining_three_exact_new_mapping_and_mandatory_dependency(self):
    plans=self.make_remainder()
    self.assertEqual([(r['task'],r['physical_gpu']) for r in plans['sv2']['runs']],
        [('dmc_acrobot_swingup',1)])
    self.assertEqual([(r['task'],r['physical_gpu']) for r in plans['sv3']['runs']],
        [('dmc_cartpole_balance',1),('dmc_cartpole_swingup',3)])
    for p in plans.values():runner.validate_plan(p)
    p=copy.deepcopy(plans['sv2']);p.pop('wait_for')
    with self.assertRaises(ValueError):runner.validate_plan(p)
    p=copy.deepcopy(plans['sv3']);p['runs'][0]['physical_gpu']=4
    with self.assertRaises(ValueError):runner.validate_plan(p)

  def test_waiter_releases_only_after_successful_own_predecessor_audit(self):
    p=self.make_remainder()['sv2'];wait=p['wait_for']
    prior=dict(state=runner.AUDITED,task=wait['task'],code_sha=wait['code_sha'],directory=wait['directory'])
    with patch.object(Path,'exists',return_value=True),patch.object(runner,'read',return_value=prior),patch.object(runner,'audit_run',return_value={'passed':True}) as audit,patch.object(runner,'dump'),patch.object(runner.time,'sleep') as sleep:
      runner.await_predecessor(p,'fixture')
      audit.assert_called_once_with(prior,runner.DMC16);sleep.assert_not_called()
    for field,value in [('state','failed'),('code_sha','wrong'),('task','other'),('directory','wrong')]:
      bad=dict(prior,**{field:value})
      with patch.object(Path,'exists',return_value=True),patch.object(runner,'read',return_value=bad),patch.object(runner,'audit_run') as audit:
        with self.assertRaises(RuntimeError):runner.await_predecessor(p,'fixture')
        audit.assert_not_called()

  def test_waiter_does_not_poll_foreign_or_failed_jobs(self):
    p=self.make_remainder()['sv2'];wait=p['wait_for']
    bad=dict(task=wait['task'],directory=wait['directory'],code_sha=wait['code_sha'],state='failed')
    with patch.object(Path,'exists',return_value=False),patch.object(runner,'read',return_value=dict(runs=[bad])),patch.object(runner.time,'sleep') as sleep:
      with self.assertRaises(RuntimeError):runner.await_predecessor(p,'fixture')
      sleep.assert_not_called()


if __name__=='__main__':unittest.main()
