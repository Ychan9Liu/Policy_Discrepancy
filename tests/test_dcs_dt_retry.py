"""Operational fixtures only; never authorize or simulate scientific results."""
import copy
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import run_dcs_dt_retry as retry
from scripts import run_dcs_dynamic as base
from tests import test_dcs_sv3_six as existing


class RetryPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        existing.SixTest.setUpClass()
        cls.original = existing.SixTest.original

    @classmethod
    def tearDownClass(cls):
        existing.SixTest.tearDownClass()

    def test_exact_eight_dt_science_unchanged_unique_attempt02_and_wandb(self):
        before = copy.deepcopy(self.original)
        rows = retry.retry_rows(self.original)
        self.assertEqual(len(rows), 8)
        self.assertEqual([r['task'] for r in rows], list(retry.TASKS))
        self.assertEqual([r['physical_gpu'] for r in rows], list(range(8)))
        for new in rows:
            old = next(r for r in before['runs'] if r['run_id'] == new['source_run_id'])
            self.assertEqual({k for k in new['config'] if new['config'][k] != old['config'][k]}, {'logdir'})
            self.assertEqual(new['code_sha'], retry.TRAIN_SHA)
            self.assertEqual(new['mode'], 'dt')
            self.assertEqual(new['environment']['WANDB_RESUME'], 'never')
            self.assertTrue(new['run_id'].endswith('attempt02'))
            self.assertTrue(new['environment']['WANDB_RUN_ID'].endswith('-02'))
            self.assertNotEqual(new['directory'], old['directory'])
            self.assertEqual({k for k in new['environment'] if new['environment'][k] != old['environment'][k]},
                {'TMPDIR', 'WANDB_DIR', 'WANDB_RUN_ID'})
        self.assertEqual(len({r['run_id'] for r in rows}), 8)
        self.assertEqual(len({r['environment']['WANDB_RUN_ID'] for r in rows}), 8)
        self.assertEqual(self.original, before)

    def test_drifted_source_rejected(self):
        for key, value in [('code_sha', 'wrong'), ('root', '/tmp/wrong'), ('rollout', 'first16')]:
            bad = copy.deepcopy(self.original); bad[key] = value
            with self.assertRaises(ValueError):
                retry.retry_rows(bad)
        bad = copy.deepcopy(self.original)
        bad['runs'][1]['config']['agent.rep_probe.alpha'] = 0
        with self.assertRaises(ValueError):
            retry.retry_rows(bad)

    def test_engineering_scope_and_acceptance_cannot_be_substituted(self):
        rows = retry.retry_rows(self.original, True)
        for r in rows:
            self.assertTrue(r['config']['run.engineering_fixture'])
            self.assertEqual(r['config']['run.action_budget'], 4098)
            self.assertEqual(r['config']['logger.outputs'], ['jsonl'])
            self.assertTrue(r['run_id'].endswith(':engineering'))
        plan = dict(controller_sha='fixture', original_plan=self.original)
        proof = dict(passed=True, engineering_only=True, code_sha=retry.TRAIN_SHA,
            controller_sha='fixture', declaration_sha256=self.original['declaration_sha256'], runs=rows)
        retry.validate_acceptance(proof, plan)
        for key, value in [('passed', False), ('engineering_only', False), ('code_sha', 'wrong'),
                           ('controller_sha', 'wrong'), ('declaration_sha256', 'wrong'), ('runs', rows[:7])]:
            with self.assertRaises(ValueError):
                retry.validate_acceptance(dict(proof, **{key: value}), plan)


class CapacityAndCleanupTest(unittest.TestCase):
    def test_aggregate_capacity_inclusive_boundary_and_original_failure(self):
        self.assertEqual(retry.required_capacity(8), 530_000_000_000)
        for count in range(1, 9):
            needed = retry.required_capacity(count)
            self.assertEqual(retry.require_capacity(needed, count), needed)
            with self.assertRaises(RuntimeError):
                retry.require_capacity(needed-1, count)
        for free in (189_407_000_000, 176_286_000_000, 4_654_387_200):
            with self.assertRaises(RuntimeError):
                retry.require_capacity(free, 8)
        for count in (0, 9):
            with self.assertRaises(ValueError):
                retry.required_capacity(count)

    def test_deletion_exact_allowlist_completed_and_path_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'old'; root.mkdir()
            target = root/retry.TASKS[0]/'dt-attempt01'; target.mkdir(parents=True)
            baseline = target.parent/'baseline-attempt01'; baseline.mkdir()
            (baseline/'canary').write_text('keep')
            with patch.object(retry, 'SOURCE', root):
                self.assertEqual(retry.old_target(retry.TASKS[0]), target)
                for task in ('../escape', 'dmc_walker_walk', 'dmc_hopper_hop/baseline-attempt01'):
                    with self.assertRaises(ValueError):
                        retry.old_target(task)
                (target/'final_state.json').write_text('{}')
                with self.assertRaises(ValueError):
                    retry.old_target(retry.TASKS[0])
            self.assertEqual((baseline/'canary').read_text(), 'keep')

    @unittest.skipUnless(sys.platform == 'linux', 'Server symlink and open-file semantics')
    def test_symlink_target_and_live_file_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'old'; root.mkdir()
            parent = root/retry.TASKS[0]; parent.mkdir()
            outside = Path(folder)/'outside'; outside.mkdir()
            (parent/'dt-attempt01').symlink_to(outside, target_is_directory=True)
            with patch.object(retry, 'SOURCE', root):
                with self.assertRaises(ValueError):
                    retry.old_target(retry.TASKS[0])
            proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)', str(outside)])
            try:
                with self.assertRaises(RuntimeError):
                    retry.no_users([outside])
            finally:
                proc.terminate(); proc.wait(timeout=5)

    def test_archive_round_trip_keeps_raw_evidence_not_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'old'; root.mkdir()
            for name in ('protocol_manifest.json', 'execution-initial.json', 'update_receipts.jsonl',
                         'evaluations.jsonl', 'worker.log'):
                (root/name).write_text(name+'\n')
            for name in ('replay', 'checkpoint', 'wandb'):
                (root/name).mkdir(); (root/name/'payload.bin').write_bytes(b'payload')
            snap = root/'eval_snapshots/0000000'; snap.mkdir(parents=True)
            (snap/'evaluation_snapshot.json').write_text('{}')
            before = retry.inventory(root)
            proof = retry.archive_attempt(root, Path(folder)/'saved.tar.gz')
            self.assertEqual(len(proof['files']), 6)
            self.assertEqual(retry.inventory(root), before)
            self.assertNotIn('replay/payload.bin', proof['files'])
            with self.assertRaises(FileExistsError):
                retry.archive_attempt(root, Path(folder)/'saved.tar.gz')

    @unittest.skipUnless(sys.platform == 'linux', 'Linux UID access boundaries')
    def test_same_account_inaccessible_fds_fail_other_account_is_explicit_limit(self):
        p = Mock(pid=1234, info=dict(cwd='/unrelated', cmdline=['system-service'], uids=Mock(real=os.getuid()+1)))
        p.open_files.side_effect = retry.psutil.AccessDenied(1234)
        with patch.object(retry.psutil, 'process_iter', return_value=[p]):
            proof = retry.no_users([Path('/fixture')])
            self.assertEqual(proof['other_account_fds_inaccessible'], [1234])
            p.info['uids'] = Mock(real=os.getuid())
            with self.assertRaises(retry.psutil.AccessDenied):
                retry.no_users([Path('/fixture')])
            p.info['cmdline'] = ['(sd-pam)']
            proof = retry.no_users([Path('/fixture')])
            self.assertEqual(proof['systemd_pam_helpers_inaccessible'], [1234])
            p.info['cmdline'] = ['(sd-pam)', '/fixture']
            with self.assertRaises(RuntimeError):
                retry.no_users([Path('/fixture')])

    def test_cleanup_final_index_contains_exit_and_keeps_accepted(self):
        rows = [dict(run_id='done', state='accepted', exit_code=0),
                dict(run_id='live', state='running'), dict(run_id='wait', state='queued')]
        done = Mock(returncode=0); done.poll.return_value = 0
        live = Mock(returncode=None); live.poll.side_effect = lambda: live.returncode
        handles = {'done': (done, io.StringIO()), 'live': (live, io.StringIO())}
        snapshots = []
        def stop(proc):
            if proc is live:
                proc.returncode = -15
        with patch.object(retry, 'stop_owned', side_effect=stop):
            retry.finish_owned(rows, handles, 'storage_below_5GB', lambda: snapshots.append(copy.deepcopy(rows)))
        self.assertEqual(rows[0], dict(run_id='done', state='accepted', exit_code=0))
        self.assertEqual(rows[1]['state'], 'failed_storage')
        self.assertEqual(rows[1]['exit_code'], -15)
        self.assertTrue(rows[1]['stopped_by_supervisor'])
        self.assertEqual(rows[2]['state'], 'blocked_supervisor')
        self.assertEqual(snapshots[-1], rows)

    def test_one_cleanup_failure_still_stops_other_owned_workers(self):
        rows = [dict(run_id=str(i), state='running') for i in range(2)]
        procs = [Mock(returncode=None), Mock(returncode=None)]
        for p in procs:
            p.poll.return_value = None
        calls = []
        def stop(proc):
            calls.append(proc)
            if proc is procs[0]:
                raise RuntimeError('fixture stop failure')
            proc.returncode = -9
        with patch.object(retry, 'stop_owned', side_effect=stop):
            with self.assertRaises(RuntimeError):
                retry.finish_owned(rows, {str(i):(p,io.StringIO()) for i,p in enumerate(procs)}, 'fixture', lambda: None)
        self.assertEqual(len(calls), 2)
        self.assertEqual([r['state'] for r in rows], ['cleanup_failed', 'failed_supervisor'])
        self.assertEqual(rows[1]['exit_code'], -9)

    def test_controller_disk_exception_persists_post_cleanup_and_rejects_reopen(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.ExitStack() as stack:
            prep = Path(folder)/'prep'; prep.mkdir()
            root = prep/'integration'
            row = dict(run_id='fixture', state='queued', physical_gpu=0, gpu_uuid='fixture-uuid',
                directory=str(root/'task/dt-attempt02'), argv=['fixture'], environment={})
            plan = dict(controller_sha='fixture', original_plan={},
                runs=[dict(source_directory=str(Path(folder)/'absent-old'))])
            retry.dump(prep/'plan.json', plan)
            retry.dump(prep/'deletion.json', dict(completed=True, baselines_unchanged=True))
            (prep/'bindings').mkdir()
            retry.dump(prep/'bindings/gpu0-binding.json', dict(code_sha=retry.TRAIN_SHA,
                gpu_uuid='fixture-uuid', physical_gpu=0, verified_compute_and_graphics_same_gpu=True, egl_index=0))
            proc = Mock(returncode=None, pid=123); proc.poll.side_effect = lambda: proc.returncode
            def stop(p): p.returncode = -15
            for target, name, value in ((retry, 'PREP', prep), (retry, 'check_plan', Mock(return_value=plan)),
                    (retry, 'retry_rows', Mock(return_value=[row])), (retry, 'training_checkout', Mock()),
                    (retry, 'stop_owned', stop), (retry.platform, 'node', Mock(return_value=base.HOST)),
                    (retry.shutil, 'disk_usage', Mock(side_effect=[Mock(free=f) for f in [10**12]*3+[4_999_999_999]])),
                    (retry.subprocess, 'Popen', Mock(return_value=proc)), (retry.time, 'sleep', Mock()),
                    (base, 'require_free', Mock(return_value={'no_jobs':True})),
                    (base, 'child_environment', Mock(return_value={}))):
                stack.enter_context(patch.object(target, name, value))
            with self.assertRaisesRegex(RuntimeError, 'storage_below_5GB'):
                retry.execute(True)
            final = retry.read(root/'actual-RUN_INDEX.json')['runs'][0]
            self.assertEqual(final['state'], 'failed_storage')
            self.assertEqual(final['exit_code'], -15)
            self.assertIn('ended_utc', final)
            self.assertEqual(retry.read(root/'supervisor-error.json')['reason'], 'storage_below_5GB')
            with self.assertRaises(FileExistsError):
                retry.execute(True)
            retry.subprocess.Popen.assert_called_once()

    def test_full_filesystem_does_not_skip_cleanup_of_later_children(self):
        rows = [dict(run_id=str(i), state='running') for i in range(2)]
        procs = [Mock(returncode=None), Mock(returncode=None)]
        for p in procs: p.poll.return_value = None
        def stop(p): p.returncode = -15
        publisher = Mock(side_effect=OSError('No space left on device'))
        with patch.object(retry, 'stop_owned', side_effect=stop) as stopped:
            with self.assertRaises(RuntimeError):
                retry.finish_owned(rows, {str(i):(p,io.StringIO()) for i,p in enumerate(procs)},
                    'storage_below_5GB', publisher)
            self.assertEqual(stopped.call_count, 2)
        self.assertEqual([r['exit_code'] for r in rows], [-15, -15])

    @unittest.skipUnless(sys.platform == 'linux', 'Linux child process groups')
    def test_real_owned_worker_terminated_and_unrelated_process_untouched(self):
        with tempfile.TemporaryDirectory() as folder:
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
            unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
            rows = [dict(run_id='owned', state='running')]
            output = Path(folder)/'index.json'
            try:
                retry.finish_owned(rows, {'owned':(child, io.StringIO())}, 'storage_below_5GB',
                    lambda: retry.dump(output, rows))
                final = retry.read(output)[0]
                self.assertEqual(final['state'], 'failed_storage')
                self.assertEqual(final['exit_code'], -signal.SIGTERM)
                self.assertIsNone(unrelated.poll())
            finally:
                for p in (child, unrelated):
                    if p.poll() is None:
                        p.terminate()
                    p.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
