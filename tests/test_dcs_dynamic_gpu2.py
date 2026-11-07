"""CPU-only controller fixtures, not additional training authorization."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from scripts import run_dcs_dynamic as queue
from scripts import run_dcs_dynamic_gpu2 as continuation
from dreamerv3.protocol_scope import DCS_DYNAMIC, VISION20_TASKS


class ContinuationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory()
        root = Path(cls.workspace.name)
        declaration = root / 'declaration.json'
        declaration.write_text(json.dumps(dict(protocol=DCS_DYNAMIC, tasks=list(VISION20_TASKS),
            groups=['baseline', 'dt'], seed=0, engineering_only=True,
            background=dict(dataset=str(root), train_videos='train-fixture',
                eval_videos='eval-fixture', num_videos=-1, video_alpha=1,
                ground_by_task={task: .3 for task in VISION20_TASKS}, seed=-1, cache_size=2))))
        with patch.object(queue, 'clean_sha', return_value=continuation.TRAIN_SHA):
            queue.prepare(root / 'plan', declaration, continuation.SOURCE.name, 'all40')
        cls.plan = json.loads((root / 'plan/plan.json').read_text())

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.index = dict(code_sha=continuation.TRAIN_SHA, runs=copy.deepcopy(self.plan['runs']))
        for r in self.index['runs']:
            if r['physical_gpu'] == 2:
                r['state'] = 'blocked_predecessor'
                if r['lane_position'] == 0:
                    r.update(state='blocked_resource', error='Assigned GPU2 has jobs; no replacement or sharing')

    def test_exact_six_original_commands_configurations_rng_and_ids(self):
        rows = continuation.blocked_rows(self.plan, self.index)
        self.assertEqual(rows, [r for r in self.plan['runs'] if r['physical_gpu'] == 2])
        self.assertEqual([r['task'] for r in rows], [
            'dmc_quadruped_walk'] * 2 + ['dmc_cheetah_run'] * 2 + ['dmc_walker_stand'] * 2)
        self.assertEqual([r['lane_position'] for r in rows], list(range(6)))
        for row in rows:
            self.assertEqual(row['code_sha'], continuation.TRAIN_SHA)
            self.assertEqual(row['config']['run.action_budget'], 1000000)
            self.assertEqual(row['config']['seed'], 0)

    def test_reject_started_changed_or_missing_source_slots(self):
        for key, value in (('pid', 42), ('state', 'running'), ('mode', 'dt'),
                           ('start_utc', 'fixture'), ('directory', '/different')):
            altered = copy.deepcopy(self.index)
            row = next(r for r in altered['runs'] if r['physical_gpu'] == 2)
            row[key] = value
            with self.assertRaises(ValueError):
                continuation.blocked_rows(self.plan, altered)
        altered = copy.deepcopy(self.index)
        altered['runs'] = [r for r in altered['runs'] if not (r['physical_gpu'] == 2 and r['lane_position'] == 5)]
        with self.assertRaises(ValueError):
            continuation.blocked_rows(self.plan, altered)

    def test_old_controller_cannot_advance_the_blocked_lane(self):
        rows = copy.deepcopy(self.index['runs'])
        self.assertFalse(any(r['physical_gpu'] == 2 for r in queue.launchable(rows)))
        self.assertEqual([r['state'] for r in rows if r['physical_gpu'] == 2],
                         ['blocked_resource'] + ['blocked_predecessor'] * 5)

    def test_claim_refuses_second_controller_and_existing_output(self):
        rows = continuation.blocked_rows(self.plan, self.index)
        for i, row in enumerate(rows):
            row['directory'] = str(self.root / f'output{i}')
        source = self.root / 'source'
        source.mkdir()
        (source / 'actual-RUN_INDEX.json').write_text(json.dumps(self.index))
        claim_root = self.root / 'claimed'
        continuation.claim(claim_root, source, rows, 'controller-fixture')
        before = (claim_root / 'claim.json').read_bytes()
        with self.assertRaises(FileExistsError):
            continuation.claim(claim_root, source, rows, 'another-controller')
        self.assertEqual(before, (claim_root / 'claim.json').read_bytes())
        Path(rows[0]['directory']).mkdir()
        with self.assertRaises(FileExistsError):
            continuation.claim(self.root / 'second-claim', source, rows, 'controller-fixture')
        self.assertFalse((self.root / 'second-claim').exists())

    def test_audit_required_for_each_successor_and_failure_blocks_rest(self):
        rows = continuation.blocked_rows(self.plan, self.index)
        first = continuation.first_pending(rows)
        self.assertEqual((first['lane_position'], first['state']), (0, 'queued'))
        first['state'] = 'running'
        self.assertIs(continuation.first_pending(rows), first)
        first['state'] = 'training_finished_pending_audit'
        self.assertIs(continuation.first_pending(rows), first)
        self.assertEqual(rows[1]['state'], 'blocked_predecessor')
        rows = continuation.blocked_rows(self.plan, self.index)
        rows[0]['state'] = 'accepted'
        self.assertEqual(continuation.first_pending(rows)['lane_position'], 1)
        rows[1]['state'] = 'failed_audit'
        self.assertEqual(continuation.first_pending(rows)['state'], 'failed_audit')
        self.assertTrue(all(r['state'] == 'blocked_predecessor' for r in rows[2:]))

    def test_combined_index_replaces_six_without_mutating_or_duplicating(self):
        before = copy.deepcopy(self.index)
        rows = continuation.blocked_rows(self.plan, self.index)
        rows[0]['state'] = 'running'
        merged = continuation.merge_index(self.index, rows)
        self.assertEqual(self.index, before)
        self.assertEqual(len(merged['runs']), 40)
        self.assertEqual(len({r['run_id'] for r in merged['runs']}), 40)
        self.assertEqual([r for r in merged['runs'] if r['physical_gpu'] != 2],
                         [r for r in self.index['runs'] if r['physical_gpu'] != 2])
        for invalid in (rows[:5], rows[:5] + rows[:1], rows[:5] + [self.plan['runs'][0]]):
            with self.assertRaises(ValueError):
                continuation.merge_index(self.index, invalid)

    def test_training_checkout_cannot_drift_or_be_dirty(self):
        for responses in ((continuation.TRAIN_SHA + '\n', b''),
                          ('wrong\n', b''), (continuation.TRAIN_SHA + '\n', b' M model.py')):
            with patch.object(continuation.subprocess, 'check_output', side_effect=responses):
                if responses[0].strip() == continuation.TRAIN_SHA and not responses[1]:
                    continuation.validate_training_checkout()
                else:
                    with self.assertRaises(RuntimeError):
                        continuation.validate_training_checkout()

    def test_controller_audits_baseline_then_stops_after_failed_dt(self):
        rows = continuation.blocked_rows(self.plan, self.index)
        for i, row in enumerate(rows):
            row['directory'] = str(self.root / f'run{i}')
        source = self.root / 'source'
        evidence = self.root / 'proof'
        output = self.root / 'controller'
        source.mkdir()
        (evidence / 'integration').mkdir(parents=True)
        (source / 'startup.json').write_text(json.dumps({'plan': self.plan}))
        (source / 'actual-RUN_INDEX.json').write_text(json.dumps(self.index))
        (evidence / 'integration/acceptance.json').write_text('{}')
        (evidence / 'gpu2-binding.json').write_text(json.dumps(dict(code_sha=continuation.TRAIN_SHA,
            gpu_uuid=queue.UUIDS[2], verified_compute_and_graphics_same_gpu=True, egl_index=2)))
        processes = [SimpleNamespace(pid=123+i, returncode=code, poll=lambda code=code: code)
                     for i, code in enumerate((0, 1))]
        with ExitStack() as stack:
            for key, value in (('SOURCE', source), ('EVIDENCE', evidence), ('ROOT', output)):
                stack.enter_context(patch.object(continuation, key, value))
            stack.enter_context(patch.object(continuation, 'clean_sha', return_value='controller-fixture'))
            stack.enter_context(patch.object(continuation.platform, 'node', return_value=queue.HOST))
            stack.enter_context(patch.object(continuation, 'blocked_rows', return_value=copy.deepcopy(rows)))
            stack.enter_context(patch.object(continuation, 'validate_training_checkout'))
            stack.enter_context(patch.object(queue, 'validate_acceptance'))
            free = stack.enter_context(patch.object(queue, 'require_free', return_value={'no_jobs': True}))
            stack.enter_context(patch.object(continuation.shutil, 'disk_usage', return_value=SimpleNamespace(free=100_000_000_000)))
            popen = stack.enter_context(patch.object(continuation.subprocess, 'Popen', side_effect=processes))
            audit = stack.enter_context(patch.object(queue, 'audit', return_value={'fixture': True}))
            stop = stack.enter_context(patch.object(continuation, 'stop_owned'))
            continuation.run()
        self.assertEqual(popen.call_count, 2)
        self.assertEqual(audit.call_count, 1)
        self.assertEqual(free.call_count, 3)  # initial claim, baseline, then Dt
        for call, row in zip(popen.call_args_list, rows):
            self.assertEqual(call.args[0], row['argv'])
            self.assertEqual(call.kwargs['cwd'], continuation.TRAIN_REPO)
            self.assertEqual(call.kwargs['env']['CUDA_VISIBLE_DEVICES'], queue.UUIDS[2])
            self.assertEqual(call.kwargs['env']['MUJOCO_EGL_DEVICE_ID'], '2')
        final = json.loads((output / 'supervisor-final.json').read_text())
        self.assertFalse(final['passed'])
        self.assertEqual([r['state'] for r in final['runs']],
                         ['accepted', 'failed'] + ['blocked_predecessor'] * 4)
        self.assertEqual(json.loads((source / 'actual-RUN_INDEX.json').read_text()), self.index)
        self.assertEqual(stop.call_args.args[0].pid, processes[-1].pid)


if __name__ == '__main__':
    unittest.main()
