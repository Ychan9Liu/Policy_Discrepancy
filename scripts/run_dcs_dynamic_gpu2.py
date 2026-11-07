"""Claim the six never-started GPU2 slots after their external job exits.

The original controller leaves resource-blocked lanes terminal. This controller
does not edit its index or restart it. Training still uses the exact accepted
5245096 checkout, commands, output paths, W&B IDs and scientific configuration.
"""
import copy
import json
import platform
from pathlib import Path
import shutil
import signal
import subprocess
import time

from scripts import run_dcs_dynamic as queue
from scripts.run_hopper_seeds123 import clean_sha, dump, now, read, sha
from scripts.run_missing_dt import approved_context
from scripts.run_dtl_three import stop_owned

TRAIN_SHA = '5245096ee52de5ba1ff6bcafb96d19298b74dcad'
TRAIN_REPO = Path('/data/Policy_Discrepancy/checkouts/dcs-dynamic-5245096ee52d')
SOURCE = Path('/data/Policy_Discrepancy/runs/dcs-dynamic-vision20-seed0-20261106-5245096')
EVIDENCE = Path('/data/Policy_Discrepancy/runs/engineering-dcs-dynamic-preparation-20261106/device-binding-5245096')
ROOT = Path(str(SOURCE) + '-gpu2-continuation-20261107')


def blocked_rows(plan, index):
    queue.validate(plan)
    if (plan['code_sha'] != TRAIN_SHA or plan['root'] != SOURCE.as_posix() or
            plan['rollout'] != 'all40' or index['code_sha'] != TRAIN_SHA):
        raise ValueError('Wrong source campaign or training SHA')
    planned = [r for r in plan['runs'] if r['physical_gpu'] == 2]
    actual = [r for r in index['runs'] if r['physical_gpu'] == 2]
    if len(planned) != 6 or len(actual) != 6:
        raise ValueError('Exactly the original six GPU2 slots are required')
    for original, row in zip(planned, actual):
        # Dynamic launch metadata is forbidden: these slots never started.
        expected = copy.deepcopy(original)
        expected['state'] = ('blocked_resource' if original['lane_position'] == 0
                             else 'blocked_predecessor')
        if original['lane_position'] == 0:
            expected['error'] = 'Assigned GPU2 has jobs; no replacement or sharing'
        if row != expected:
            raise ValueError('Source slot is not an unchanged never-started resource block')
    return copy.deepcopy(planned)


def merge_index(source, continuation):
    output = copy.deepcopy(source)
    replacements = {r['run_id']: r for r in continuation}
    if len(replacements) != 6:
        raise ValueError('Six unique continuation slots required')
    if set(replacements) != {r['run_id'] for r in source['runs'] if r['physical_gpu'] == 2}:
        raise ValueError('Continuation must replace only the original GPU2 slots')
    output['runs'] = [copy.deepcopy(replacements.get(r['run_id'], r)) for r in source['runs']]
    output['checked_utc'] = now()
    output['continuation_root'] = str(ROOT)
    return output


def validate_training_checkout():
    commit = subprocess.check_output(['git', '-C', str(TRAIN_REPO), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(TRAIN_REPO), 'status', '--porcelain'])
    if commit != TRAIN_SHA or dirty:
        raise RuntimeError('Training checkout must remain clean at the accepted SHA')


def claim(root, source, rows, controller_sha):
    # A unique exclusive root and fresh original output paths jointly prevent
    # duplicate controllers, including a second invocation after a crash.
    if any(Path(r['directory']).exists() for r in rows):
        raise FileExistsError('A never-started output already exists; refuse overwrite')
    root.mkdir(parents=True, exist_ok=False)
    dump(root / 'claim.json', dict(created_utc=now(), controller_sha=controller_sha,
        training_sha=TRAIN_SHA, source_index=str(source / 'actual-RUN_INDEX.json'),
        source_sha256=sha(source / 'actual-RUN_INDEX.json'),
        run_ids=[r['run_id'] for r in rows], physical_gpu=2,
        reason='User asked to retain blocked GPU2 slots until the external job exits'))


def first_pending(rows):
    queue.refresh_queue(rows)
    pending = [r for r in sorted(rows, key=lambda r: r['lane_position']) if r['state'] != 'accepted']
    return pending[0] if pending else None


def run():
    controller_sha = clean_sha()
    if platform.node() != queue.HOST:
        raise RuntimeError('Only the assigned sv2 host is allowed')
    plan = read(SOURCE / 'startup.json')['plan']
    rows = blocked_rows(plan, read(SOURCE / 'actual-RUN_INDEX.json'))
    queue.validate_acceptance(read(EVIDENCE / 'integration/acceptance.json'), plan)
    validate_training_checkout()
    binding = read(EVIDENCE / 'gpu2-binding.json')
    if (binding['code_sha'] != TRAIN_SHA or binding['gpu_uuid'] != queue.UUIDS[2] or
            not binding['verified_compute_and_graphics_same_gpu']):
        raise ValueError('Existing same-SHA CUDA/EGL binding must match GPU2')
    before = queue.require_free(2, queue.UUIDS[2])
    if shutil.disk_usage('/data').free < 60_000_000_000:
        raise RuntimeError('GPU2 continuation requires 60GB available before claim')
    claim(ROOT, SOURCE, rows, controller_sha)
    dump(ROOT / 'startup.json', dict(controller_sha=controller_sha, code_sha=TRAIN_SHA,
        resource_before_claim=before, free_bytes=shutil.disk_usage('/data').free,
        plan=plan, binding=binding))
    process = log = None

    def publish():
        # Source GPU2 rows must remain blocked. The old controller cannot launch
        # them; its original immutable per-slot evidence stays intact.
        source = read(SOURCE / 'actual-RUN_INDEX.json')
        blocked_rows(plan, source)
        queue.refresh_queue(rows)
        dump(ROOT / 'actual-RUN_INDEX.json', dict(code_sha=TRAIN_SHA,
            controller_sha=controller_sha, checked_utc=now(), runs=rows))
        dump(ROOT / 'combined-RUN_INDEX.json', merge_index(source, rows))

    def terminate(signum, frame):
        raise SystemExit(f'Continuation controller received signal {signum}')

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        publish()
        while True:
            free = shutil.disk_usage('/data').free
            with (ROOT / 'resources.jsonl').open('a') as f:
                f.write(json.dumps(dict(utc=now(), free_bytes=free)) + '\n')
            if free < 5_000_000_000:
                raise RuntimeError('Disk below 5GB; stopping only the owned GPU2 job')
            row = first_pending(rows)
            if row is None:
                break
            if row['state'] == 'running':
                if process.poll() is None:
                    publish()
                    time.sleep(15)
                    continue
                log.close()
                row.update(exit_code=process.returncode, ended_utc=now(),
                           state='training_finished_pending_audit')
                if process.returncode:
                    row['state'] = 'failed'
                else:
                    try:
                        row['audit'] = queue.audit(row)
                        row['state'] = 'accepted'
                    except Exception as error:
                        row.update(state='failed_audit', error=repr(error))
                publish()
                continue
            if row['state'] != 'queued':
                break
            if free < 60_000_000_000:
                row['waiting_for_storage_bytes'] = 60_000_000_000
                publish()
                time.sleep(15)
                continue
            row.pop('waiting_for_storage_bytes', None)
            validate_training_checkout()
            try:
                row['resource_before_start'] = queue.require_free(2, queue.UUIDS[2])
            except RuntimeError as error:
                row.update(state='blocked_resource', error=str(error))
                publish()
                break
            directory = Path(row['directory'])
            directory.mkdir(parents=True, exist_ok=False)
            (directory / 'tmp').mkdir()
            row['environment']['MUJOCO_EGL_DEVICE_ID'] = str(binding['egl_index'])
            log = (directory / 'worker.log').open('x', encoding='utf-8')
            process = subprocess.Popen(row['argv'], cwd=TRAIN_REPO,
                env=queue.child_environment(row['environment'], inherited=approved_context()),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            row.update(state='running', pid=process.pid, started_epoch=time.time(), start_utc=now(),
                       controller_sha=controller_sha, continuation_root=str(ROOT))
            dump(directory / 'execution-initial.json', row)
            publish()
        publish()
        dump(ROOT / 'supervisor-final.json', dict(code_sha=TRAIN_SHA, controller_sha=controller_sha,
            checked_utc=now(), passed=all(r['state'] == 'accepted' for r in rows), runs=rows))
    except BaseException as error:
        dump(ROOT / 'supervisor-error.json', dict(utc=now(), error=repr(error), runs=rows))
        raise
    finally:
        if process is not None:
            stop_owned(process)
        if log is not None:
            log.close()
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    run()
