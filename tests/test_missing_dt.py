"""CPU boundary fixtures; no scientific conclusions or new training seeds."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

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


if __name__=='__main__':unittest.main()
