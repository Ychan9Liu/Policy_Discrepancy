"""Explicit attempt02 of the eight sv2 Dt runs interrupted on 2026-11-08.

Only the supervisor changes; training executes the accepted 5245096 checkout.
No recovery, automatic retries, new seeds, deletion of W&B runs, or new lanes.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import tarfile
import time

import psutil

from scripts import run_dcs_dynamic as base
from scripts.run_dcs_dynamic_gpu2 import TRAIN_SHA, TRAIN_REPO, SOURCE
from scripts.run_hopper_seeds123 import clean_sha, dump, now, read, sha
from scripts.run_missing_dt import approved_context
from scripts.run_dtl_three import stop_owned
from scripts.run_dcs_sv3_six import training_checkout

PREP = Path('/data/Policy_Discrepancy/runs/engineering-dcs-dt-retry-20261112')
ROOT = Path('/data/Policy_Discrepancy/runs/dcs-dynamic-dt-retry-20261112-5245096')
OLD_PREP = Path('/data/Policy_Discrepancy/runs/engineering-dcs-dynamic-preparation-20261106')
OLD_MANAGER = Path('/data/Policy_Discrepancy/runs/dcs-dynamic-vision20-seed0-20261107-sv2-remaining28')
TASKS = tuple(base.VISION20_TASKS[:8])
RESERVE = 60_000_000_000
HEADROOM = 50_000_000_000
EMERGENCY = 5_000_000_000


def required_capacity(count, engineering=False):
    if not 1 <= count <= 8:
        raise ValueError('Only the assigned eight lanes are supported')
    return count * (5_000_000_000 if engineering else RESERVE) + HEADROOM


def require_capacity(free, count, engineering=False):
    required = required_capacity(count, engineering)
    if free < required:
        raise RuntimeError(f'Aggregate reservation not met: {free} < {required}')
    return required


def retry_rows(original, engineering=False):
    base.validate(original)
    if (original['code_sha'] != TRAIN_SHA or original['root'] != SOURCE.as_posix()
            or original['rollout'] != 'all40'):
        raise ValueError('Wrong original campaign')
    rows = []
    for task in TASKS:
        old = next(r for r in original['runs'] if r['task'] == task and r['mode'] == 'dt')
        row = copy.deepcopy(old)
        directory = str((PREP/'integration' if engineering else ROOT)/task/'dt-attempt02')
        argv = base.command(original['declaration_file'], task, 'dt', directory, engineering)
        config = json.loads(json.dumps(dict(base.resolve(argv[4:]).flat)))
        if not engineering and {k for k in config if config[k] != old['config'][k]} != {'logdir'}:
            raise ValueError('Scientific configuration changed')
        row.update(directory=directory, argv=argv, config=config,
            run_id=old['run_id'].removesuffix('attempt01')+'attempt02',
            source_run_id=old['run_id'], source_directory=old['directory'],
            lane_position=0, state='queued', enabled=True)
        row['environment'].update(TMPDIR=directory+'/tmp', WANDB_DIR=directory,
            WANDB_RUN_ID=old['environment']['WANDB_RUN_ID'].removesuffix('-01')+'-02')
        if engineering:
            row['run_id'] += ':engineering'
            row['environment']['WANDB_RUN_ID'] += '-engineering'
        rows.append(row)
    return rows


def no_users(paths):
    """Check visible commands plus all same-account fds; record visibility limits."""
    prefixes = tuple(str(p) for p in paths)
    inaccessible = []
    login_helpers = []
    checked = 0
    def refers(value):
        return any(value == p or value.startswith(p+'/') for p in prefixes)
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cwd', 'uids']):
        if proc.pid == os.getpid():
            continue
        try:
            values = [proc.info['cwd'] or '', *(proc.info['cmdline'] or [])]
            if any(refers(value) for value in values):
                raise RuntimeError(f'Protected attempt in use by PID {proc.pid}')
            try:
                files = proc.open_files()
            except psutil.AccessDenied:
                # The project uses this account. System/other-user descriptors
                # are normally hidden by Linux; do not claim they were inspected.
                uids = proc.info['uids']
                argv = [arg for arg in proc.info['cmdline'] or [] if arg]
                login_helper = argv == ['(sd-pam)'] or (
                    proc.info.get('name') == 'sshd' and len(argv) == 1
                    and argv[0].startswith('sshd: ') and '@' in argv[0])
                if uids is not None and uids.real == os.getuid() and login_helper:
                    # Nondumpable PAM/SSH session helpers use our UID, but are
                    # not user-launched workers. Preserve the exact exceptions.
                    login_helpers.append(proc.pid)
                    continue
                if uids is None or uids.real == os.getuid():
                    raise
                inaccessible.append(proc.pid)
                continue
            if any(refers(f.path) for f in files):
                raise RuntimeError(f'Protected attempt open by PID {proc.pid}')
            checked += 1
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
    return dict(utc=now(), same_account_worker_fds_checked=True, fd_processes_checked=checked,
        other_account_fds_inaccessible=inaccessible, login_helpers_inaccessible=login_helpers,
        limitation='Other-account and nondumpable PAM/SSH session helper cwd/fds not inspectable')


def old_target(task):
    if task not in TASKS:
        raise ValueError('Deletion task outside exact eight failed attempts')
    root = SOURCE.resolve(strict=True)
    target = SOURCE/task/'dt-attempt01'
    if (SOURCE.is_symlink() or target.is_symlink() or target.parent.is_symlink()
            or target.resolve(strict=True) != root/task/'dt-attempt01'
            or not target.is_dir()):
        raise ValueError('Deletion path is not the exact non-symlink failed directory')
    if (target/'final_state.json').exists():
        raise ValueError('Completed run cannot be deleted by this operation')
    return target


def inventory(path):
    """Content metadata, without reading multi-GB model/replay payloads."""
    records = []
    for folder, dirs, files in os.walk(path, followlinks=False):
        for name in sorted(dirs + files):
            item = Path(folder)/name
            stat = item.lstat()
            record = dict(path=str(item.relative_to(path)), size=stat.st_size,
                mtime_ns=stat.st_mtime_ns, inode=stat.st_ino, device=stat.st_dev,
                mode=stat.st_mode)
            if item.is_symlink():
                record['link'] = os.readlink(item)
            records.append(record)
    return sorted(records, key=lambda x: x['path'])


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def archive_attempt(target, archive):
    """Preserve raw small evidence, excluding checkpoints, replay and W&B binaries."""
    selected = [p for p in target.iterdir() if p.is_file() and not p.is_symlink()
                and p.suffix in ('.json', '.jsonl', '.yaml', '.log')]
    selected += [p for p in (target/'eval_snapshots').glob('*/*.json')
                 if p.resolve() == p.absolute() and p.is_file()]
    records = {p.relative_to(target).as_posix(): file_hash(p) for p in sorted(selected)}
    if not {'protocol_manifest.json', 'execution-initial.json', 'update_receipts.jsonl',
            'evaluations.jsonl', 'worker.log'} <= records.keys():
        raise ValueError('Incomplete failure evidence; do not delete')
    with tarfile.open(archive, 'x:gz') as output:
        for p in selected:
            output.add(p, arcname=p.relative_to(target).as_posix(), recursive=False)
    with tarfile.open(archive) as saved:
        actual = {m.name: hashlib.sha256(saved.extractfile(m).read()).hexdigest()
                  for m in saved.getmembers()}
    if actual != records:
        raise RuntimeError('Archived evidence failed round-trip verification')
    return dict(files=records, sha256=file_hash(archive), bytes=archive.stat().st_size)


def prepare():
    if platform.node() != base.HOST:
        raise RuntimeError('Assigned sv2 only')
    training_checkout()
    original = read(OLD_PREP/'formal-plan-all40/plan.json')
    rows = retry_rows(original)
    index = read(OLD_MANAGER/'actual-RUN_INDEX.json')
    failure = read(OLD_MANAGER/'supervisor-error.json')
    if failure['error'] != "RuntimeError('Disk below 5GB; stopping only this campaign workers')":
        raise ValueError('Unexpected source failure')
    paths = [old_target(task) for task in TASKS]
    process_check = no_users(paths)
    baselines = {}
    for row in rows:
        old = next(r for r in index['runs'] if r['run_id'] == row['source_run_id'])
        expected = next(r for r in original['runs'] if r['run_id'] == row['source_run_id'])
        if any(old[k] != expected[k] for k in ('directory', 'argv', 'config', 'code_sha')):
            raise ValueError('Source attempt configuration differs')
        manifest = read(Path(old['directory'])/'protocol_manifest.json')
        if manifest['git_commit'] != TRAIN_SHA or manifest['config'] != old['config']:
            raise ValueError('Actual failed manifest differs')
        baseline = next(r for r in index['runs'] if r['task'] == row['task'] and r['mode'] == 'logging')
        if baseline['state'] != 'accepted' or baseline['exit_code'] != 0:
            raise ValueError('Keep only already accepted baseline partners')
        path = SOURCE/row['task']/'baseline-attempt01'
        final = read(path/'final_state.json')
        if final['train_action_steps'] != 1000000 or final['train_batch_keys'] != 249493:
            raise ValueError('Baseline terminal counters differ')
        baselines[row['task']] = dict(path=str(path), inventory=inventory(path),
            final_sha256=sha(path/'final_state.json'))
    PREP.mkdir(parents=True, exist_ok=False)
    failure_dir = PREP/'failure-evidence'
    failure_dir.mkdir()
    for name in ('actual-RUN_INDEX.json', 'supervisor-error.json', 'resources.jsonl', 'manager.log'):
        shutil.copyfile(OLD_MANAGER/name, failure_dir/name)
    dump(PREP/'plan.json', dict(controller_sha=clean_sha(), code_sha=TRAIN_SHA,
        original_plan=original, runs=rows, created_utc=now(),
        source_error_sha256=sha(OLD_MANAGER/'supervisor-error.json'),
        process_check=process_check,
        deletion_authorization='User 2026-11-12: delete only eight failed Dt directories; keep W&B',
        baselines=baselines, reservation_per_run=RESERVE, headroom=HEADROOM,
        emergency_bytes=EMERGENCY, root=str(ROOT)))


def check_plan():
    plan = read(PREP/'plan.json')
    if (plan['controller_sha'] != clean_sha() or plan['code_sha'] != TRAIN_SHA
            or plan['runs'] != retry_rows(plan['original_plan']) or plan['root'] != str(ROOT)
            or plan['reservation_per_run'] != RESERVE or plan['headroom'] != HEADROOM
            or plan['emergency_bytes'] != EMERGENCY):
        raise ValueError('Pinned retry plan differs')
    training_checkout()
    return plan


def cleanup():
    if platform.node() != base.HOST:
        raise RuntimeError('Assigned sv2 only')
    plan = check_plan()
    receipt = PREP/'deletion.json'
    if receipt.exists():
        raise FileExistsError('No repeated deletion or silent partial retry')
    paths = [old_target(task) for task in TASKS]
    process_check = no_users(paths)
    proof = dict(controller_sha=clean_sha(), started_utc=now(), free_before=shutil.disk_usage('/data').free,
        wandb_operation='none; all eight failed cloud runs retained', completed=False, runs=[],
        process_check=process_check)
    dump(receipt, proof)
    for task, path in zip(TASKS, paths):
        before = inventory(path)
        saved = archive_attempt(path, PREP/'failure-evidence'/f'{task}.tar.gz')
        dump(PREP/'failure-evidence'/f'{task}-inventory.json', before)
        if path != old_target(task) or inventory(path) != before:
            raise RuntimeError('Deletion target changed during evidence preservation')
        process_check = no_users([path])
        shutil.rmtree(path)
        proof['runs'].append(dict(task=task, path=str(path), absent=not path.exists(), evidence=saved,
            process_check=process_check))
        dump(receipt, proof)
    for baseline in plan['baselines'].values():
        path = Path(baseline['path'])
        if inventory(path) != baseline['inventory'] or sha(path/'final_state.json') != baseline['final_sha256']:
            raise RuntimeError('Protected baseline changed')
    proof.update(completed=True, ended_utc=now(), baselines_unchanged=True,
        free_after=shutil.disk_usage('/data').free)
    dump(receipt, proof)


def finish_owned(rows, handles, reason, publish):
    """Persist each owned child's real exit after cleanup, even on exceptions."""
    errors = []
    def persist():
        try:
            publish()
        except BaseException as error:
            # A full filesystem must not prevent stopping the remaining children.
            errors.append('publish: '+repr(error))
    for row in rows:
        handle = handles.get(row['run_id'])
        if handle is None:
            if row['state'] == 'queued':
                row.update(state='blocked_supervisor', error=reason)
            continue
        proc, log = handle
        try:
            was_live = proc.poll() is None
            stop_owned(proc)
            if row['state'] in ('queued', 'running'):
                row.update(state='failed_storage' if reason == 'storage_below_5GB' else 'failed_supervisor',
                    error=reason, exit_code=proc.returncode, ended_utc=now(),
                    stopped_by_supervisor=was_live)
        except BaseException as error:
            row.update(state='cleanup_failed', error=repr(error), cleanup_reason=reason)
            errors.append(repr(error))
        finally:
            log.close()
            persist()
    persist()
    if errors:
        raise RuntimeError('Owned child cleanup failed: '+repr(errors))


def validate_acceptance(proof, plan):
    if (not proof['passed'] or not proof['engineering_only'] or proof['code_sha'] != TRAIN_SHA
            or proof['controller_sha'] != plan['controller_sha']
            or proof['declaration_sha256'] != plan['original_plan']['declaration_sha256']
            or len(proof['runs']) != 8
            or {(r['task'], r['group'], r['physical_gpu']) for r in proof['runs']}
               != {(task, 'dt', gpu) for gpu, task in enumerate(TASKS)}):
        raise ValueError('Eight same-controller real CUDA engineering acceptances required')


def execute(engineering=False):
    if platform.node() != base.HOST:
        raise RuntimeError('Assigned sv2 only')
    plan = check_plan()
    deletion = read(PREP/'deletion.json')
    if not deletion['completed'] or not deletion['baselines_unchanged'] or any(
            Path(r['source_directory']).exists() for r in plan['runs']):
        raise ValueError('Deletion/protection audit incomplete')
    if not engineering:
        validate_acceptance(read(PREP/'integration/acceptance.json'), plan)
    rows = retry_rows(plan['original_plan'], engineering)
    root = PREP/'integration' if engineering else ROOT
    # An exclusive root is a durable claim: a second supervisor cannot relaunch.
    root.mkdir(parents=True, exist_ok=False)
    handles = {}
    reason = 'supervisor_exception'
    def publish():
        dump(root/'actual-RUN_INDEX.json', dict(code_sha=TRAIN_SHA, controller_sha=plan['controller_sha'],
            checked_utc=now(), engineering_only=engineering, runs=rows))
    def terminated(signum, frame):
        raise SystemExit(f'Supervisor signal {signum}')
    previous = signal.signal(signal.SIGTERM, terminated)
    try:
        free = shutil.disk_usage('/data').free
        required = require_capacity(free, len(rows), engineering)
        dump(root/'startup.json', dict(code_sha=TRAIN_SHA, controller_sha=plan['controller_sha'],
            utc=now(), engineering_only=engineering, free_bytes=free, required_bytes=required,
            plan_sha256=sha(PREP/'plan.json'), environment_python=base.PYTHON))
        publish()
        while True:
            free = shutil.disk_usage('/data').free
            with (root/'resources.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(utc=now(), free_bytes=free))+'\n')
            if free < EMERGENCY:
                reason = 'storage_below_5GB'
                raise RuntimeError(reason)
            for row in rows:
                if row['state'] != 'running':
                    continue
                proc, log = handles[row['run_id']]
                rc = proc.poll()
                if rc is None:
                    if engineering and time.time()-row['started_epoch'] > 3600:
                        stop_owned(proc)
                        rc = proc.returncode
                        row['error'] = 'Engineering timeout'
                    else:
                        continue
                log.close()
                row.update(exit_code=rc, ended_utc=now(), state='training_finished_pending_audit')
                publish()
                if rc:
                    row['state'] = 'failed'
                else:
                    try:
                        row['audit'] = base.audit(row, engineering)
                        row['state'] = 'accepted'
                    except Exception as error:
                        row.update(state='failed_audit', error=repr(error))
                publish()
            for row in rows:
                if row['state'] != 'queued':
                    continue
                # Reserve for ALL not-yet-finished attempts, not this worker alone.
                unfinished = sum(r['state'] in ('queued', 'running') for r in rows)
                require_capacity(shutil.disk_usage('/data').free, unfinished, engineering)
                training_checkout()
                binding = read(PREP/'bindings'/f'gpu{row["physical_gpu"]}-binding.json')
                if (binding['code_sha'] != TRAIN_SHA or binding['gpu_uuid'] != row['gpu_uuid']
                        or binding['physical_gpu'] != row['physical_gpu']
                        or not binding['verified_compute_and_graphics_same_gpu']):
                    raise ValueError('CUDA/EGL binding differs')
                try:
                    row['resource_before_start'] = base.require_free(row['physical_gpu'], row['gpu_uuid'])
                except RuntimeError as error:
                    row.update(state='blocked_resource', error=str(error))
                    publish()
                    continue
                directory = Path(row['directory'])
                directory.mkdir(parents=True, exist_ok=False)
                (directory/'tmp').mkdir()
                row['environment']['MUJOCO_EGL_DEVICE_ID'] = str(binding['egl_index'])
                log = (directory/'worker.log').open('x')
                try:
                    proc = subprocess.Popen(row['argv'], cwd=TRAIN_REPO,
                        env=base.child_environment(row['environment'], inherited=approved_context()),
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                handles[row['run_id']] = proc, log
                row.update(state='running', pid=proc.pid, started_epoch=time.time(), start_utc=now(),
                    controller_sha=plan['controller_sha'])
                dump(directory/'execution-initial.json', row)
                publish()
            if not any(r['state'] == 'running' for r in rows):
                break
            time.sleep(15)
        proof = dict(passed=all(r['state'] == 'accepted' for r in rows), engineering_only=engineering,
            code_sha=TRAIN_SHA, controller_sha=plan['controller_sha'], checked_utc=now(),
            declaration_sha256=plan['original_plan']['declaration_sha256'],
            runs=[r['audit'] for r in rows if r['state'] == 'accepted'], states=[r['state'] for r in rows])
        dump(root/('acceptance.json' if engineering else 'supervisor-final.json'), proof)
        if not proof['passed']:
            raise RuntimeError('Failed/blocked attempts retained; no automatic retry')
    except BaseException as error:
        dump(root/'supervisor-error.json', dict(utc=now(), error=repr(error), reason=reason))
        raise
    finally:
        # Suppress repeated TERM during the bounded child-group cleanup itself.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            finish_owned(rows, handles, reason, publish)
        finally:
            signal.signal(signal.SIGTERM, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'cleanup', 'integration', 'run'))
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'cleanup':
        cleanup()
    else:
        execute(args.action == 'integration')


if __name__ == '__main__':
    main()
