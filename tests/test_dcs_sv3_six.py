"""Engineering controller fixtures; no additional scientific runs."""
import copy
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import run_dcs_dynamic as base
from scripts import run_dcs_sv3_six as six
from scripts import dcs_process_adoption as adoption
from dreamerv3.protocol_scope import DCS_DYNAMIC, VISION20_TASKS


class SixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        declaration = root/'declaration.json'
        declaration.write_text(json.dumps(dict(protocol=DCS_DYNAMIC, tasks=list(VISION20_TASKS),
            groups=['baseline','dt'], seed=0, engineering_only=True,
            background=dict(dataset=str(root),train_videos='train-fixture',eval_videos='eval-fixture',
                num_videos=-1,video_alpha=1.,ground_by_task={t:.3 for t in VISION20_TASKS},seed=-1,cache_size=2))))
        with patch.object(base,'clean_sha',return_value=six.TRAIN_SHA):
            base.prepare(root/'plan',declaration,six.SOURCE.name,'all40')
        cls.original = json.loads((root/'plan/plan.json').read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def source(self):
        rows=copy.deepcopy(self.original['runs'])
        for r in rows:
            if r['lane_position']==0:
                r.update(state='running',pid=100+r['physical_gpu'])
            elif r['lane_position']==1:
                r['state']='queued'
        return rows

    def test_twelve_targets_only_logdir_config_changes_and_exact_ids(self):
        rows=six.target_rows(self.original)
        self.assertEqual(len(rows),12)
        for old,new in zip([r for r in self.original['runs'] if r['task'] in six.TASKS],rows):
            self.assertEqual({k for k in old['config'] if old['config'][k]!=new['config'][k]}, {'logdir'})
            self.assertEqual(old['run_id'],new['run_id'])
            self.assertEqual(old['environment']['WANDB_RUN_ID'],new['environment']['WANDB_RUN_ID'])
            self.assertEqual(new['code_sha'],six.TRAIN_SHA)
        self.assertEqual([r['physical_gpu'] for r in rows], [g for g in six.GPUS for _ in range(2)])
        self.assertEqual([r['gpu_uuid'] for r in rows], [u for u in six.UUIDS for _ in range(2)])

    def test_matrix_is_partitioned_28_plus_12_without_duplicates(self):
        old=self.source();before=copy.deepcopy(old)
        six.validate_live_source(self.original,old,check_paths=False)
        retained=[r for r in old if r['task'] not in six.TASKS]
        target=six.target_rows(self.original)
        self.assertEqual((len(retained),len(target)),(28,12))
        self.assertEqual(len({r['run_id'] for r in retained+target}),40)
        self.assertEqual(old,before)
        self.assertEqual(len(base.launchable(target)),6)
        for r in base.launchable(target):r['state']='running'
        base.refresh_queue(target)
        self.assertEqual([r['state'] for r in target],['running','queued']*6)

    def test_started_drifted_missing_and_duplicate_sources_rejected(self):
        source=self.source()
        for key,value in [('state','running'),('pid',777),('start_utc','fixture'),('exit_code',0),('mode','constant')]:
            rows=copy.deepcopy(source);row=next(r for r in rows if r['task'] in six.TASKS);row[key]=value
            with self.assertRaises(ValueError):six.validate_live_source(self.original,rows,check_paths=False)
        for rows in (source[:39],source[:39]+source[:1]):
            with self.assertRaises(ValueError):six.validate_live_source(self.original,rows,check_paths=False)
        with patch.object(six.Path,'exists',return_value=True):
            with self.assertRaises(FileExistsError):six.validate_live_source(self.original,source)

    def test_retained_lanes_skip_only_transferred_pairs(self):
        retained=[r for r in self.source() if r['task'] not in six.TASKS]
        self.assertFalse(base.launchable(retained))
        for r in retained:
            if r['lane_position']<=1:r['state']='accepted'
        tasks={r['task'] for r in base.launchable(retained)}
        self.assertEqual(tasks,set(VISION20_TASKS[14:]))
        self.assertFalse(tasks & set(six.TASKS))

    def test_engineering_fixture_never_looks_like_formal_metrics(self):
        rows=six.target_rows(self.original,True)
        for r in rows:
            self.assertTrue(r['config']['run.engineering_fixture'])
            self.assertEqual(r['config']['run.action_budget'],4098)
            self.assertEqual(r['config']['run.eval_eps'],1)
            self.assertEqual(r['config']['logger.outputs'],['jsonl'])
            self.assertTrue(r['run_id'].endswith(':sv3-engineering'))
        plan=dict(controller_sha='fixture',original_plan=self.original)
        proof=dict(passed=True,engineering_only=True,code_sha=six.TRAIN_SHA,controller_sha='fixture',
            declaration_sha256=self.original['declaration_sha256'],
            runs=[dict(task=r['task'],group=r['group']) for r in rows])
        six.validate_engineering(plan,proof)
        for key,value in [('passed',False),('engineering_only',False),('controller_sha','other'),('declaration_sha256','other')]:
            with self.assertRaises(ValueError):six.validate_engineering(plan,dict(proof,**{key:value}))

    def test_target_requires_complete_handoff_seal(self):
        rows=self.source();retained=[r for r in rows if r['task'] not in six.TASKS]
        plan=dict(controller_sha='fixture',original_plan=self.original)
        seal=dict(controller_sha='fixture',code_sha=six.TRAIN_SHA,plan_sha256='hash',manager_pid=22,
            source_snapshot=rows,retained=retained,transferred=[r for r in rows if r['task'] in six.TASKS],
            adopted={r['run_id']:{'pid':r['pid']} for r in retained if r['state']=='running'})
        with patch.object(six,'sha',return_value='hash'):
            six.validate_seal(plan,seal)
            for key,value in [('manager_pid',None),('plan_sha256','wrong'),('retained',retained[:27]),('transferred',[]),('adopted',{})]:
                with self.assertRaises(ValueError):six.validate_seal(plan,dict(seal,**{key:value}))


class AdoptionTest(unittest.TestCase):
    @unittest.skipUnless(sys.platform=='linux','Linux process-group storage guard')
    def test_emergency_storage_stop_requires_verified_own_process_group(self):
        identity=dict(pid=8,start_ticks=10,ppid=5)
        with patch.object(adoption,'adopted_poll',return_value=None),patch.object(adoption.os,'getpgid',return_value=9),patch.object(adoption.os,'killpg') as kill:
            with self.assertRaises(RuntimeError):adoption.stop_worker_for_storage(identity)
            kill.assert_not_called()
        with patch.object(adoption,'adopted_poll',side_effect=[None,-15]),patch.object(adoption,'same_process'),patch.object(adoption.os,'getpgid',return_value=8),patch.object(adoption.os,'killpg') as kill:
            self.assertEqual(adoption.stop_worker_for_storage(identity),-15)
            kill.assert_called_once_with(8,signal.SIGTERM)

    def test_pid_reuse_and_live_child_retirement_are_refused(self):
        identity=dict(pid=8,start_ticks=10,ppid=5)
        with patch.object(adoption,'stat',return_value=dict(identity,start_ticks=11)):
            with self.assertRaises(RuntimeError):adoption.same_process(identity)
        with patch.object(adoption,'same_process',return_value=dict(identity,state='S',ppid=1)):
            with self.assertRaises(RuntimeError):adoption.adopted_poll(identity)
        with patch.object(adoption,'assert_frozen'),patch.object(adoption,'adopted_poll',return_value=None),patch.object(adoption.os,'kill') as kill:
            with self.assertRaises(RuntimeError):adoption.retire(identity,[identity])
            kill.assert_not_called()

    @unittest.skipUnless(sys.platform=='linux','Real /proc wait-status and supervisor signals require Linux')
    def test_real_worker_keeps_progress_rng_and_exit_status_while_parent_stopped(self):
        worker="""import random,time,pathlib,sys
r=random.Random(0)
with pathlib.Path(sys.argv[1]).open('w',buffering=1) as f:
 for i in range(30):
  f.write(str(r.random())+'\\n');time.sleep(.05)
raise SystemExit(7)
"""
        parent="""import subprocess,sys,pathlib,json,time
p=subprocess.Popen([sys.executable,'-u','-c',sys.argv[1],sys.argv[2]],start_new_session=True)
pathlib.Path(sys.argv[3]).write_text(str(p.pid))
p.wait()
"""
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);progress=root/'progress';pidfile=root/'pid'
            proc=subprocess.Popen([sys.executable,'-u','-c',parent,worker,str(progress),str(pidfile)],start_new_session=True)
            owner=adoption.stat(proc.pid);child=None
            try:
                for _ in range(200):
                    if pidfile.exists() and progress.exists() and len(progress.read_text().splitlines())>=2:break
                    time.sleep(.01)
                child=adoption.stat(int(pidfile.read_text()))
                self.assertEqual(child['ppid'],owner['pid'])
                adoption.freeze(owner)
                before=len(progress.read_text().splitlines());time.sleep(.2)
                self.assertGreater(len(progress.read_text().splitlines()),before)
                self.assertIsNone(adoption.adopted_poll(child))
                for _ in range(300):
                    if adoption.adopted_poll(child) is not None:break
                    time.sleep(.01)
                self.assertEqual(adoption.adopted_poll(child),7)
                expected=random.Random(0)
                self.assertEqual([float(x) for x in progress.read_text().splitlines()],[expected.random() for _ in range(30)])
                adoption.retire(owner,[child]);proc.wait(timeout=5)
                self.assertEqual(proc.returncode,-signal.SIGKILL)
            finally:
                if proc.poll() is None:
                    if child is not None and adoption.adopted_poll(child) is None:os.kill(child['pid'],signal.SIGKILL)
                    os.kill(proc.pid,signal.SIGKILL);proc.wait(timeout=5)


if __name__=='__main__':unittest.main()
