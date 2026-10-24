"""CPU fixtures for the fixed two-host allocation; no network or GPU jobs."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from analysis.return_metrics import MetricSpec, aggregate
from dreamerv3.protocol_scope import DMC16_TASKS, DMC8_REMAINING
from scripts import run_dmc8_remaining as runner
from scripts import run_dmc8_serial as serial


class RemainingTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.tmp=TemporaryDirectory();cls.root=Path(cls.tmp.name)
    mapping={s:dict(host=h,gpus={str(g):dict(uuid=f'fixture-{s}-{g}') for g in runner.DEVICES[s]}) for s,h in runner.HOSTS.items()}
    resources=cls.root/'resources.json';resources.write_text(json.dumps(dict(servers=mapping)))
    with patch.object(runner,'source_state',return_value=('fixture-only','retained docs drafts')):
      runner.prepare(cls.root/'plan','formal-m2-dmc16-remaining8-serial-fixture',resources)
    cls.plans={s:json.loads((cls.root/'plan'/(s+'-plan.json')).read_text()) for s in runner.HOSTS}

  @classmethod
  def tearDownClass(cls):cls.tmp.cleanup()

  def bound(self,server):
    p=copy.deepcopy(self.plans[server]);p['authorized_to_train']=True
    for r in p['runs']:
      r.update(authorized_to_train=True,launch_argv=r['review_argv'],
          verified_compute_and_graphics_same_gpu=True,egl_index=r['physical_gpu'],state='queued')
      r['environment_template']['MUJOCO_EGL_DEVICE_ID']=str(r['egl_index'])
    return p

  def test_exact_remaining_eight_two_hosts_and_serial_pairs(self):
    rows=[]
    for s,p in self.plans.items():
      runner.validate_plan(p);rows+=p['runs']
      self.assertEqual(len(p['runs']),4 if s=='sv1' else 12)
      self.assertFalse(p['authorized_to_train'])
      self.assertTrue(p['host_completion_is_not_cohort_completion'])
      self.assertEqual({r['physical_gpu'] for r in p['runs']},set(runner.DEVICES[s]))
    self.assertEqual({r['task'] for r in rows},set(DMC16_TASKS[8:]))
    self.assertEqual(len(rows),16)
    for task in runner.TASKS:
      a,b=[r for r in rows if r['task']==task]
      self.assertEqual(a['gpu_uuid'],b['gpu_uuid'])
      self.assertEqual([a['queue_position'],b['queue_position']],[1,2])
      self.assertEqual({k for k in a['config'] if a['config'][k]!=b['config'][k]},{'logdir','agent.rep_probe.mode'})
    for p in (self.bound('sv1'),self.bound('sv2')):runner.validate_plan(p)

  def test_resource_config_seed_and_scope_tampering_refused(self):
    for field,value in [('physical_gpu',0),('host','elsewhere'),('server','sv2'),('seed',False),('mode','shuffle'),('task',DMC16_TASKS[0])]:
      p=copy.deepcopy(self.plans['sv1']);p['runs'][0][field]=value
      with self.subTest(field=field),self.assertRaises(ValueError):runner.validate_plan(p)
    p=copy.deepcopy(self.plans['sv2']);p['runs'][0]['config']['run.action_budget']=4098
    with self.assertRaises(ValueError):runner.validate_plan(p)
    p=self.bound('sv2');p['runs'][0]['environment_template']['WANDB_MODE']='offline'
    with self.assertRaises(ValueError):runner.validate_plan(p)

  def test_original_rng_derivation_and_grid(self):
    from scripts.prepare_m2_v1 import seed32
    for task in runner.TASKS:
      r=runner.random_streams(task)
      self.assertEqual(r['train_env_seeds'],[seed32(task,'train_env',i) for i in range(16)])
      self.assertEqual(r['replay_seed'],seed32(task,'train_replay'))
      self.assertEqual([e['action_step'] for e in r['evaluations']],list(range(0,1000001,50000)))
      self.assertTrue(all(len(e['episodes'])==10 for e in r['evaluations']))

  def test_split_card_never_uses_blocked_other_card_or_changes_science(self):
    source=self.root/'host-plan.json';source.write_text(json.dumps(self.plans['sv1']))
    output=self.root/'card7-plan.json'
    runner.split_card(source,7,output);p=json.loads(output.read_text());runner.validate_plan(p)
    self.assertEqual(p['tasks'],['dmc_finger_turn_hard'])
    self.assertEqual({r['physical_gpu'] for r in p['runs']},{7})
    self.assertEqual(len(p['runs']),2)
    self.assertTrue(p['root'].endswith('-sv1-gpu7'))
    for r in p['runs']:
      old=next(x for x in self.plans['sv1']['runs'] if x['task']==r['task'] and x['mode']==r['mode'])
      self.assertEqual({k for k in r['config'] if r['config'][k]!=old['config'][k]},{'logdir'})
    with self.assertRaises(ValueError):runner.split_card(source,0,self.root/'unallocated.json')
    with self.assertRaises(ValueError):runner.split_card(source,7,output)
    bad=copy.deepcopy(p);bad['card_gpu']=2
    with self.assertRaises(ValueError):runner.validate_plan(bad)

  def test_each_shard_metrics_remain_incomplete_until_all_eight_present(self):
    spec=MetricSpec(tasks=runner.TASKS,groups=('baseline','dt'),scope=DMC8_REMAINING)
    rows=[dict(task=t,group=g,seed=0,run_id=t+g,auc_complete=True,tail_complete=True,
        auc_return_per_budget=float(i)+(2 if g=='dt' else 0),tail_mean_return=float(i)+1+(2 if g=='dt' else 0))
        for i,t in enumerate(runner.TASKS) for g in spec.groups]
    for shard in (rows[:4],rows[4:]):
      _,overall,_=aggregate(shard,spec)
      self.assertTrue(all(not r['auc_complete'] and r['auc_return_per_budget'] is None for r in overall))
      self.assertTrue(all(set(r['task_weights'])==set(runner.TASKS) for r in overall))
    _,overall,diffs=aggregate(rows,spec)
    self.assertEqual([r['auc_return_per_budget'] for r in overall],[3.5,5.5])
    self.assertTrue(all(r['auc_return_per_budget']==2 for r in diffs))
    for fields in (dict(tasks=DMC16_TASKS[:8]),dict(episodes=1),dict(seeds=(1,)),dict(budget=50000)):
      with self.assertRaises(ValueError):MetricSpec(**(dict(tasks=runner.TASKS,groups=spec.groups,scope=DMC8_REMAINING)|fields))

  def test_serial_audit_barrier_and_other_card_independence(self):
    p=self.bound('sv1');rows=p['runs'];base=serial.next_for_gpu(rows,2)
    self.assertEqual(base['mode'],'logging');base['state']='running'
    self.assertIsNone(serial.next_for_gpu(rows,2))
    base['state']=serial.AUDITED
    self.assertEqual(serial.next_for_gpu(rows,2)['mode'],'dt')
    rows[2]['state']='audit_failed'
    self.assertIsNone(serial.next_for_gpu(rows,7));self.assertEqual(rows[3]['state'],'blocked')

  def test_handoff_waits_only_owned_group_and_never_shares(self):
    p=self.bound('sv1');base,dt=p['runs'][:2];base['process_group_id']=77
    xml=f'<nvidia_smi_log><gpu><uuid>{dt["gpu_uuid"]}</uuid><processes><process_info><pid>99</pid></process_info></processes></gpu></nvidia_smi_log>'
    with patch.object(runner,'clean_sha',return_value=p['code_sha']),\
        patch.object(runner,'require_free',side_effect=[RuntimeError('busy'),dict(no_jobs=True)]),\
        patch.object(runner.subprocess,'check_output',return_value=xml),patch.object(runner.os,'getpgid',return_value=77,create=True),\
        patch.object(runner.time,'sleep') as sleep:
      runner.before_start(dt,p);sleep.assert_called_once_with(2)
      self.assertTrue(dt['handoff_release_check']['no_jobs'])
      self.assertEqual(dt['handoff_release_check']['observations'][0]['owned_predecessor_group'],77)
    with patch.object(runner,'clean_sha',return_value=p['code_sha']),\
        patch.object(runner,'require_free',side_effect=RuntimeError('busy')),\
        patch.object(runner.subprocess,'check_output',return_value=xml),patch.object(runner.os,'getpgid',return_value=123,create=True),\
        patch.object(runner.time,'sleep') as sleep:
      with self.assertRaisesRegex(RuntimeError,'foreign'):runner.before_start(dt,p)
      sleep.assert_not_called()

  def test_unauthorized_or_existing_root_never_launches(self):
    path=self.root/'run.json';path.write_text(json.dumps(self.plans['sv1']))
    with patch.object(serial.subprocess,'Popen') as launch:
      with self.assertRaises(RuntimeError):runner.run(path)
      launch.assert_not_called()
    p=self.bound('sv1');path.write_text(json.dumps(p))
    with patch.object(serial.platform,'node',return_value=runner.HOSTS['sv1']),\
        patch.object(serial,'clean_sha',return_value=p['code_sha']),patch.object(serial,'Path') as directory,\
        patch.object(serial.subprocess,'Popen') as launch:
      directory.return_value.exists.return_value=True
      with self.assertRaises(FileExistsError):runner.run(path)
      launch.assert_not_called()

  def test_new_session_is_explicit_for_owned_cleanup_provenance(self):
    row=copy.deepcopy(self.plans['sv1']['runs'][0]);row['directory']=str(self.root/'session-run')
    row['launch_argv']=row['review_argv'];row['environment_template']['MUJOCO_EGL_DEVICE_ID']='2'
    with patch.object(serial,'require_free',return_value={'no_jobs':True}),patch.object(serial.subprocess,'Popen') as launch:
      launch.return_value.pid=77
      _,log=serial.start(row,new_session=True);log.close()
      self.assertTrue(launch.call_args.kwargs['start_new_session'])


if __name__=='__main__':unittest.main()
