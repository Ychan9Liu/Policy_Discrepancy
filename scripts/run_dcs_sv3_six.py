"""Move six untouched DCS pairs to sv3; retain and supervise sv2's live workers."""
import argparse
import copy
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import signal
import subprocess
import time

import psutil

from scripts import run_dcs_dynamic as base
from scripts import dcs_process_adoption as adoption
from scripts.run_dcs_dynamic_gpu2 import TRAIN_SHA, TRAIN_REPO, SOURCE
from scripts.run_hopper_seeds123 import clean_sha, dump, now, read, sha
from scripts.run_missing_dt import approved_context
from scripts.run_dtl_three import stop_owned

PREP = Path('/data/Policy_Discrepancy/runs/engineering-dcs-sv3-six-20261107')
OLD_PREP = Path('/data/Policy_Discrepancy/runs/engineering-dcs-dynamic-preparation-20261106')
OLD_AUX = Path(str(SOURCE) + '-gpu2-continuation-20261107')
SOURCE_ROOT = Path('/data/Policy_Discrepancy/runs/dcs-dynamic-vision20-seed0-20261107-sv2-remaining28')
TARGET_ROOT = Path('/data/Policy_Discrepancy/runs/dcs-dynamic-vision20-seed0-20261107-sv3-six')
TASKS = tuple(base.VISION20_TASKS[8:14])
GPUS = (1, 3, 4, 5, 6, 7)
UUIDS = ('GPU-5b1ebede-99ec-d080-29b0-3c04011dc663',
         'GPU-4725722b-bd64-ec77-2fcd-9692cf3ee36e',
         'GPU-9c070d5a-ac68-4c26-9522-a790ad874b23',
         'GPU-79704918-e096-92b4-252c-e3244497a8b3',
         'GPU-70db6c66-9917-e0e2-44db-0fc80fcf41b9',
         'GPU-c9a4f076-5f8a-8136-9535-7e74e14e895f')


def training_checkout():
    commit = subprocess.check_output(['git', '-C', str(TRAIN_REPO), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != TRAIN_SHA or subprocess.check_output(['git', '-C', str(TRAIN_REPO), 'status', '--porcelain']):
        raise RuntimeError('Exact clean accepted training checkout required')


def target_rows(plan, engineering=False):
    base.validate(plan)
    if plan['code_sha'] != TRAIN_SHA or plan['rollout'] != 'all40' or plan['root'] != SOURCE.as_posix():
        raise ValueError('Wrong original forty-run campaign')
    rows = []
    for task, gpu, uuid in zip(TASKS, GPUS, UUIDS):
        for position, mode in enumerate(('logging', 'dt')):
            old = next(r for r in plan['runs'] if r['task'] == task and r['mode'] == mode)
            row = copy.deepcopy(old)
            root = PREP / 'integration' if engineering else TARGET_ROOT
            directory = str(root / task / (row['group'] + '-attempt01'))
            argv = base.command(plan['declaration_file'], task, mode, directory, engineering)
            row.update(directory=directory, argv=argv,
                config=json.loads(json.dumps(dict(base.resolve(argv[4:]).flat))),
                server='sv3', host='lyg0326', physical_gpu=gpu, gpu_uuid=uuid,
                lane_position=position, state='backlog', source_run_id=old['run_id'],
                source_directory=old['directory'])
            row['environment'].update(CUDA_VISIBLE_DEVICES=uuid, MUJOCO_EGL_DEVICE_ID=None,
                TMPDIR=directory+'/tmp', WANDB_DIR=directory)
            if engineering:
                row['run_id'] += ':sv3-engineering'
                row['environment']['WANDB_RUN_ID'] += '-sv3-engineering'
            elif {k for k in row['config'] if row['config'][k] != old['config'][k]} != {'logdir'}:
                raise ValueError('Scientific configuration changed during relocation')
            rows.append(row)
    return rows


def merge_sources():
    a, b = read(SOURCE/'actual-RUN_INDEX.json'), read(OLD_AUX/'actual-RUN_INDEX.json')
    replacement = {r['run_id']: r for r in b['runs']}
    if len(replacement) != 6:
        raise ValueError('Original GPU2 continuation must have exactly six slots')
    return [copy.deepcopy(replacement.get(r['run_id'], r)) for r in a['runs']]


def validate_live_source(plan, rows, check_paths=True):
    if len(rows) != 40 or len({r['run_id'] for r in rows}) != 40:
        raise ValueError('Source must contain forty unique logical runs')
    expected = {r['run_id']: r for r in plan['runs']}
    for row in rows:
        old = expected[row['run_id']]
        if any(row[k] != old[k] for k in ('argv', 'config', 'directory', 'code_sha',
                'task', 'seed', 'mode', 'group', 'physical_gpu', 'gpu_uuid', 'lane_position', 'enabled')):
            raise ValueError('Source launch identity/configuration changed')
        if row['task'] in TASKS:
            if row['state'] != 'backlog' or any(k in row for k in ('pid', 'start_utc', 'exit_code')):
                raise ValueError('Transferred slot is not untouched backlog')
            if check_paths and Path(row['directory']).exists():
                raise FileExistsError('Transferred source output already exists')


def prepare():
    if (PREP/'plan.json').exists():
        raise FileExistsError('Prepared plan already exists')
    plan = read(PREP/'original-plan.json')
    rows = target_rows(plan)
    dump(PREP/'plan.json', dict(controller_sha=clean_sha(), code_sha=TRAIN_SHA,
        original_plan=plan, target_runs=rows, created_utc=now(), target_root=str(TARGET_ROOT),
        source_root=str(SOURCE_ROOT)))


def check_plan():
    plan = read(PREP/'plan.json')
    if (plan['controller_sha'] != clean_sha() or plan['code_sha'] != TRAIN_SHA or
            plan['target_runs'] != target_rows(plan['original_plan']) or
            plan['target_root'] != str(TARGET_ROOT) or plan['source_root'] != str(SOURCE_ROOT)):
        raise ValueError('Pinned six-pair plan differs')
    training_checkout()
    return plan


def source_controllers():
    result = []
    for receipt, module, repo in (
        ('launch-receipt.json', 'scripts.run_dcs_dynamic', TRAIN_REPO),
        ('gpu2-launch-receipt.json', 'scripts.run_dcs_dynamic_gpu2',
         Path('/data/Policy_Discrepancy/checkouts/dcs-gpu2-e04b1e8f517e'))):
        saved = read(OLD_PREP/'launch-20261107'/receipt)
        pid = saved['controller_pid']
        proc = psutil.Process(pid)
        if (proc.uids().real != os.getuid() or proc.cmdline() != saved['argv'] or
                proc.cwd() != str(repo) or module not in proc.cmdline()):
            raise RuntimeError('Source supervisor identity differs')
        result.append(dict(**adoption.stat(pid), argv=proc.cmdline(), cwd=proc.cwd()))
    return result


def takeover():
    if platform.node() != 'lyg0360':
        raise RuntimeError('Takeover only on sv2')
    plan = check_plan()
    validate_engineering(plan, read(PREP/'sv3-acceptance.json'))
    if SOURCE_ROOT.exists():
        raise FileExistsError('Takeover root already exists; no silent retry')
    owners = source_controllers()
    stopped, published, manager = [], False, None
    try:
        for owner in owners:
            adoption.freeze(owner)
            stopped.append(owner)
        rows = merge_sources()
        validate_live_source(plan['original_plan'], rows)
        adopted = {}
        for row in rows:
            if row['state'] != 'running':
                continue
            proc = psutil.Process(row['pid'])
            identity = adoption.stat(row['pid'])
            if (identity['ppid'] not in {o['pid'] for o in owners} or
                    proc.cmdline() != row['argv'] or proc.cwd() != str(TRAIN_REPO)):
                raise RuntimeError('Live training process identity differs')
            adopted[row['run_id']] = identity
        if len(adopted) != 8:
            raise RuntimeError('Expected the eight live original workers at this handoff')
        for owner in owners:
            actual = {p.pid for p in psutil.Process(owner['pid']).children()}
            wanted = {a['pid'] for a in adopted.values() if a['ppid'] == owner['pid']}
            if actual != wanted:
                raise RuntimeError('Unaccounted source child at suspension; rollback')
        SOURCE_ROOT.mkdir(parents=True, exist_ok=False)
        claim = dict(controller_sha=plan['controller_sha'], code_sha=TRAIN_SHA,
            plan_sha256=sha(PREP/'plan.json'), utc=now(), owners=owners, adopted=adopted,
            transferred=[r for r in rows if r['task'] in TASKS],
            retained=[r for r in rows if r['task'] not in TASKS], source_snapshot=rows,
            source_plan_sha256=sha(PREP/'original-plan.json'))
        dump(SOURCE_ROOT/'claim-pending.json', claim)
        log = (SOURCE_ROOT/'manager.log').open('x')
        manager = subprocess.Popen([base.PYTHON, '-u', '-m', 'scripts.run_dcs_sv3_six', 'source-run'],
            cwd=Path(__file__).resolve().parents[1], env=approved_context(),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        dump(SOURCE_ROOT/'manager-process.json', dict(pid=manager.pid, utc=now()))
        for _ in range(300):
            if manager.poll() is not None:
                raise RuntimeError('Replacement supervisor failed before handoff')
            if (SOURCE_ROOT/'ready.json').exists():
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Replacement supervisor did not become ready')
        claim['manager_pid'] = manager.pid
        dump(SOURCE_ROOT/'source-claim.json', claim)
        published = True
        print(json.dumps(dict(passed=True, manager_pid=manager.pid, retained=28, transferred=12)), flush=True)
    finally:
        # A source claim is published only after the replacement is ready.
        # Before that boundary, no target is authorized and rollback is safe.
        if not published:
            if manager is not None:
                stop_owned(manager)
            for owner in stopped:
                adoption.rollback_freeze(owner)


def validate_engineering(plan, proof):
    expected = {(task, group) for task in TASKS for group in ('baseline', 'dt')}
    if (not proof['passed'] or not proof['engineering_only'] or proof['code_sha'] != TRAIN_SHA or
            proof['controller_sha'] != plan['controller_sha'] or len(proof['runs']) != 12 or
            {(r['task'], r['group']) for r in proof['runs']} != expected or
            proof['declaration_sha256'] != plan['original_plan']['declaration_sha256']):
        raise ValueError('Same-declaration twelve real CUDA acceptances required')


def validate_seal(plan, claim):
    if (claim['controller_sha'] != plan['controller_sha'] or claim['code_sha'] != TRAIN_SHA or
            claim['plan_sha256'] != sha(PREP/'plan.json') or
            len(claim['adopted']) != 8 or len(claim['retained']) != 28 or
            len(claim['transferred']) != 12 or not claim.get('manager_pid')):
        raise ValueError('Missing live-source takeover seal')
    original = plan['original_plan']
    validate_live_source(original, claim['source_snapshot'], check_paths=False)
    if claim['transferred'] != [r for r in claim['source_snapshot'] if r['task'] in TASKS]:
        raise ValueError('Transfer list differs from the sealed source snapshot')
    if claim['retained'] != [r for r in claim['source_snapshot'] if r['task'] not in TASKS]:
        raise ValueError('Retained list differs from the sealed source snapshot')
    if set(claim['adopted']) != {r['run_id'] for r in claim['retained'] if r['state']=='running'}:
        raise ValueError('Adoption set differs from the live retained workers')


def execute(role):
    plan = check_plan()
    engineering = role == 'integration'
    source = role == 'source-run'
    if platform.node() != ('lyg0360' if source else 'lyg0326'):
        raise RuntimeError('Wrong assigned host')
    handles, adopted, owners = {}, {}, []
    retired = set()
    if source:
        root = SOURCE_ROOT
        (root/'observer-lock').mkdir(exist_ok=False)
        claim = read(root/'claim-pending.json')
        if claim['controller_sha'] != clean_sha() or claim['plan_sha256'] != sha(PREP/'plan.json'):
            raise ValueError('Replacement claim differs')
        rows, adopted, owners = copy.deepcopy(claim['retained']), claim['adopted'], claim['owners']
        for owner in owners:
            adoption.assert_frozen(owner)
        dump(root/'ready.json', dict(utc=now(), pid=os.getpid()))
        # Parent publishes the seal only after observing this readiness.
        for _ in range(600):
            if (root/'source-claim.json').exists():
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Source handoff was not committed')
    else:
        root = PREP/'integration' if engineering else TARGET_ROOT
        if not engineering:
            validate_engineering(plan, read(PREP/'integration/acceptance.json'))
            validate_seal(plan, read(PREP/'source-claim.json'))
        if shutil.disk_usage('/data').free < (30_000_000_000 if engineering else 360_000_000_000):
            raise RuntimeError('Insufficient six-card starting capacity')
        root.mkdir(parents=True, exist_ok=False)
        rows = target_rows(plan['original_plan'], engineering)
    dump(root/'startup.json', dict(code_sha=TRAIN_SHA, controller_sha=clean_sha(),
        utc=now(), engineering_only=engineering, plan=plan, role=role))

    def publish():
        base.refresh_queue(rows)
        dump(root/'actual-RUN_INDEX.json', dict(code_sha=TRAIN_SHA,
            controller_sha=plan['controller_sha'], checked_utc=now(), runs=rows))

    def terminated(signum, frame):
        raise SystemExit(f'Supervisor received signal {signum}')

    previous = signal.signal(signal.SIGTERM, terminated)
    try:
        while True:
            free = shutil.disk_usage('/data').free
            with (root/'resources.jsonl').open('a') as f:
                f.write(json.dumps(dict(utc=now(), free_bytes=free))+'\n')
            if free < 5_000_000_000:
                for row in rows:
                    if row['state']=='running' and row['run_id'] in adopted:
                        row.update(state='failed_storage',
                            exit_code=adoption.stop_worker_for_storage(adopted[row['run_id']]))
                publish()
                raise RuntimeError('Disk below 5GB; stopping only this campaign workers')
            for owner in owners:
                if owner['pid'] not in retired:
                    adoption.assert_frozen(owner)
            for row in rows:
                if row['state'] != 'running':
                    continue
                identity = adopted.get(row['run_id'])
                if identity:
                    rc = adoption.adopted_poll(identity)
                else:
                    proc, log = handles[row['run_id']]
                    rc = proc.poll()
                    if rc is None and engineering and time.time()-row['started_epoch'] > 3600:
                        stop_owned(proc)
                        rc = proc.returncode
                        row['error'] = 'Engineering timeout'
                if rc is None:
                    continue
                if not identity:
                    log.close()
                row.update(exit_code=rc, ended_utc=now(), state='training_finished_pending_audit')
                if identity:
                    row['adopted_exit_evidence'] = adoption.stat(identity['pid'])
                if rc:
                    row['state'] = 'failed'
                else:
                    try:
                        row['audit'] = base.audit(row, engineering)
                        row['state'] = 'accepted'
                    except Exception as error:
                        row.update(state='failed_audit', error=repr(error))
            # Preserve parent-held zombie exit evidence until it is persisted.
            publish()
            for owner in owners:
                child_rows = [r for r in rows if r['run_id'] in adopted and
                              adopted[r['run_id']]['ppid'] == owner['pid']]
                if owner['pid'] not in retired and all('adopted_exit_evidence' in r for r in child_rows):
                    adoption.retire(owner, [adopted[r['run_id']] for r in child_rows])
                    retired.add(owner['pid'])
                    dump(root/f'retired-{owner["pid"]}.json', dict(utc=now(), owner=owner, children=child_rows))
            for row in base.launchable(rows):
                free = shutil.disk_usage('/data').free
                if not engineering and free < 60_000_000_000:
                    row['waiting_for_storage_bytes'] = 60_000_000_000
                    continue
                row.pop('waiting_for_storage_bytes', None)
                training_checkout()
                evidence = OLD_PREP/'device-binding-5245096' if source else PREP/'bindings'
                binding = read(evidence/f'gpu{row["physical_gpu"]}-binding.json')
                if (binding['code_sha'] != TRAIN_SHA or binding['gpu_uuid'] != row['gpu_uuid'] or
                        not binding['verified_compute_and_graphics_same_gpu']):
                    raise ValueError('Accepted training CUDA/EGL binding differs')
                try:
                    row['resource_before_start'] = base.require_free(row['physical_gpu'], row['gpu_uuid'])
                except RuntimeError as error:
                    row.update(state='blocked_resource', error=str(error))
                    continue
                directory = Path(row['directory'])
                directory.mkdir(parents=True, exist_ok=False)
                (directory/'tmp').mkdir()
                row['environment']['MUJOCO_EGL_DEVICE_ID'] = str(binding['egl_index'])
                log = (directory/'worker.log').open('x')
                proc = subprocess.Popen(row['argv'], cwd=TRAIN_REPO,
                    env=base.child_environment(row['environment'], inherited=approved_context()),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                handles[row['run_id']] = (proc, log)
                row.update(pid=proc.pid, state='running', start_utc=now(), started_epoch=time.time(),
                           controller_sha=plan['controller_sha'])
                dump(directory/'execution-initial.json', row)
                publish()
            publish()
            if not any(r['state']=='running' or r.get('waiting_for_storage_bytes') for r in rows):
                break
            time.sleep(15)
        proof = dict(passed=all(r['state']=='accepted' for r in rows), engineering_only=engineering,
            code_sha=TRAIN_SHA, controller_sha=plan['controller_sha'], checked_utc=now(),
            declaration_sha256=plan['original_plan']['declaration_sha256'],
            runs=[r.get('audit', {}) for r in rows], states=[r['state'] for r in rows])
        dump(root/('acceptance.json' if engineering else 'supervisor-final.json'), proof)
    except BaseException as error:
        dump(root/'supervisor-error.json', dict(utc=now(), error=repr(error), runs=rows))
        raise
    finally:
        for proc, log in handles.values():
            stop_owned(proc)
            log.close()
        # Adopted workers are deliberately not signalled on observer failure.
        signal.signal(signal.SIGTERM, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'integration', 'takeover', 'source-run', 'target-run'))
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'takeover':
        takeover()
    else:
        execute(args.action)


if __name__ == '__main__':
    main()
