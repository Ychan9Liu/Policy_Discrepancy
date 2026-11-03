"""Read-only initial checks and terminal analysis of four approved pendulum repeats.

python -m analysis.pendulum_seeds23_analysis collect --output NEW_INPUT
python -m analysis.pendulum_seeds23_analysis analyze --input NEW_INPUT --output NEW_RESULT
python -m analysis.pendulum_seeds23_analysis initial --output NEW_INITIAL_CHECK
python -m analysis.pendulum_seeds23_analysis replay --input TERMINAL_INPUT --output NEW_REPLAY_CHECK

Reads original JSONL and existing audits. Never trains, writes W&B, downloads
checkpoint/replay payloads, or substitutes a partial curve for formal metrics.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

TRAIN_SHA = 'c5f86ca1a16648f57db3c3abfa2584ac560495ca'
CONTROLLER_SHA = 'b7f42d46d16d4eda5cff5a798ec7b60cb27a3790'
ROOT = '/data/Policy_Discrepancy/runs/formal-m2-pendulum-seeds23-parallel-dt-20261104-sv3'
OLD_ROOT = '/data/Policy_Discrepancy/runs/formal-m2-pendulum-seeds23-20261103-sv3'
SCOPE = 'M2-v1-pendulum-seeds23-baseline-dt'
TASK = 'dmc_pendulum_swingup'
MATRIX = {(2, 'baseline'): (1, 'logging'), (3, 'baseline'): (4, 'logging'),
          (2, 'dt'): (5, 'dt'), (3, 'dt'): (6, 'dt')}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False)+'\n', encoding='utf-8')


def cohort(index, completion, supervisor):
    """Reject superseded slots, duplicate seeds and incomplete terminal claims."""
    rows = index['runs']
    require(len(rows) == 4 and {(r['seed'], r['group']) for r in rows} == set(MATRIX),
            'Expected exactly seed2/3 x baseline/PPD-KL')
    require(supervisor.get('complete') is True and completion['verified_run_count'] == 4,
            'Four-run terminal acceptance is not complete')
    require(completion['code_sha'] == supervisor['training_sha'] == TRAIN_SHA,
            'Training SHA differs from this approved cohort')
    require(index['controller_sha'] == supervisor['audit_controller_sha'] == CONTROLLER_SHA,
            'Actual audit controller identity differs')
    require(supervisor['runs'] == rows, 'Terminal index and supervisor disagree')
    require(len({r['run_id'] for r in rows}) == 4, 'Duplicate run ID')
    audit = {r['run_id']: r for r in completion['runs']}
    require(len(audit) == 4, 'Duplicate or missing completion audit')
    for row in rows:
        seed, group = row['seed'], row['group']
        gpu, mode = MATRIX[seed, group]
        root = OLD_ROOT if group == 'baseline' else ROOT
        expected = f'{root}/seed{seed}/{mode}-attempt01'
        require(row['directory'] == expected and row['physical_gpu'] == gpu and
                row['mode'] == mode and row['task'] == TASK and row['code_sha'] == TRAIN_SHA,
                'Unexpected attempt, source directory, device or training identity')
        require(row['state'] == 'accepted' and row['exit_code'] == 0,
                'Run has not passed terminal acceptance')
        require(row['run_audit'] == audit[row['run_id']] and
                audit[row['run_id']]['updates'] == 249493 and
                audit[row['run_id']]['evaluations'] == 21, 'Terminal receipt disagrees')
        config = row['config']
        require(type(seed) is int and config['seed'] == seed and config['task'] == TASK and
                config['agent.rep_probe.mode'] == mode and not config['run.engineering_fixture'],
                'Wrong seed/config or engineering run')
    for seed in (2, 3):
        pair = {r['group']: r['config'] for r in rows if r['seed'] == seed}
        changed = {k for k in pair['baseline'].keys() | pair['dt'].keys()
                   if pair['baseline'].get(k) != pair['dt'].get(k)}
        require(changed == {'logdir', 'agent.rep_probe.mode'}, 'Paired scientific configuration differs')
    return rows


def remote_collect():
    """Executed through SSH stdin, stdlib only, no server file writes."""
    root = Path(ROOT)
    index_path = root/'actual-RUN_INDEX.json'
    final = root/'final-audit-attempt01'
    index, completion, supervisor = (read(p) for p in
        (index_path, final/'completion.json', root/'supervisor-final.json'))
    rows = cohort(index, completion, supervisor)
    index_hash = sha(index_path)
    require(sha(final/'return-audit/tables.json') == completion['tables_sha256'],
            'Controller metric table hash changed')
    files = {index_path, root/'supervisor-final.json', root/'bound-plan.json',
             final/'completion.json', final/'return-audit/tables.json', final/'return-audit/audit.json'}
    proof = dict(collected_utc=datetime.now(timezone.utc).isoformat(), server='sv3',
                 index_path=str(index_path), files={}, runs=[])
    for row in rows:
        directory = Path(row['directory'])
        names = ['protocol_manifest.json', 'evaluations.jsonl', 'metrics.jsonl',
                 'update_receipts.jsonl', 'config.yaml', 'execution-initial.json',
                 'execution-final.json', 'final_state.json']
        if row['group'] == 'baseline':
            names += ['matching_result.json', 'c_frozen.json', 'c_frozen.json.sha256']
        files.update(directory/name for name in names)
        transport = final/f"wandb-seed{row['seed']}-{row['mode']}.json"
        require(read(transport)['exact_transport_verified'] is True, 'W&B exact transport not verified')
        files.add(transport)
        manifest = read(directory/'protocol_manifest.json')
        require(manifest['git_commit'] == row['code_sha'] and manifest['protocol'] == SCOPE and
                manifest['root_seed'] == row['seed'] and manifest['config'] == row['config'],
                'Actual manifest differs from accepted configuration')
        run = dict(directory=str(directory), git_commit=manifest['git_commit'],
            artifact_sha256={'evaluations.jsonl': sha(directory/'evaluations.jsonl')}, snapshots=[])
        evaluations = [json.loads(line) for line in (directory/'evaluations.jsonl').read_text().splitlines()]
        require([e['action_step'] for e in evaluations] == list(range(0,1000001,50000)),
                'Missing, duplicate or reordered evaluation point')
        for e in evaluations:
            point = e['action_step']; folder = directory/'eval_snapshots'/f'{point:07d}'
            tag = (folder/'latest').read_text().strip()
            require(tag and PurePosixPath(tag).name == tag and tag not in ('.', '..'), 'Invalid snapshot tag')
            meta = read(folder/'evaluation_snapshot.json')
            payload = folder/tag/'agent.pkl'
            require(meta['snapshot_id'] == e['snapshot_id'] == f'eval_snapshots/{point:07d}/{tag}' and
                    meta['git_commit'] == manifest['git_commit'] and meta['update_id'] == e['update_id'] and
                    meta['actual_action_step'] == point and payload.is_file() and
                    (folder/tag/'done').is_file(), 'Snapshot metadata/marker/presence differs')
            files.update((folder/'latest', folder/'evaluation_snapshot.json', folder/tag/'done'))
            run['snapshots'].append(dict(action_step=point, actual_action_step=point,
                counters={'updates': meta['update_id']}, snapshot_id=meta['snapshot_id'],
                params_sha256=None, checkpoint_bytes=payload.stat().st_size,
                basis='remote-runner-metadata-and-completion-marker; payload not independently decoded'))
        proof['runs'].append(run)
    require(sha(index_path) == index_hash, 'Index changed during collection')
    # Explicit filenames only. No recursive payload, environment or credential collection.
    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
        for path in sorted(files):
            require(path.is_file() and path.stat().st_size < 128*1024**2, 'Missing/oversized named artifact')
            data = path.read_bytes(); relative = str(path).lstrip('/')
            proof['files'][relative] = dict(source=str(path), bytes=len(data),
                                           sha256=hashlib.sha256(data).hexdigest())
            info = tarfile.TarInfo(relative); info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        data = json.dumps(proof, indent=2, allow_nan=False).encode()
        info = tarfile.TarInfo('COLLECTION.json'); info.size = len(data)
        archive.addfile(info, io.BytesIO(data))


def collect(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).read_text(encoding='utf-8')
    program = "namespace={'__name__':'readonly_collection'}\nexec("+repr(code)+",namespace)\nnamespace['remote_collect']()\n"
    (output/'collector.py').write_text(program, encoding='utf-8')
    with (output/'inputs.tar.gz').open('xb') as stream, (output/'stderr.txt').open('xb') as errors:
        subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', 'sv3',
            '/data/Policy_Discrepancy/envs/dreamer/bin/python -'], input=program.encode(),
            stdout=stream, stderr=errors, check=True)
    mirror = output/'inputs'
    with tarfile.open(output/'inputs.tar.gz') as archive:
        archive.extractall(mirror, filter='data')
    proof = verify_mirror(output)
    dump(output/'mirror_validation.json', dict(files=len(proof['files']), all_hashes_match=True,
         archive_sha256=sha(output/'inputs.tar.gz'), collector_sha256=sha(output/'collector.py'),
         collection_sha256=sha(mirror/'COLLECTION.json')))
    print(json.dumps(dict(files=len(proof['files']), runs=len(proof['runs']))))


def verify_mirror(source):
    mirror = source/'inputs'; proof = read(mirror/'COLLECTION.json')
    if (source/'mirror_validation.json').exists():
        validation = read(source/'mirror_validation.json')
        require(sha(source/'inputs.tar.gz') == validation['archive_sha256'] and
                sha(mirror/'COLLECTION.json') == validation['collection_sha256'],
                'Archive/collection changed after initial mirror verification')
    for rel, entry in proof['files'].items():
        path = mirror.joinpath(*PurePosixPath(rel).parts)
        require(path.resolve().is_relative_to(mirror.resolve()), 'Source path escapes mirror')
        require(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'],
                'Mirrored source size/hash changed: '+rel)
    return proof


def compare_prefix_arrays(a, b):
    """Rows are [worker, callback, ...]; report the first chronological difference."""
    import numpy as np
    require(a.shape == b.shape and a.dtype == b.dtype and a.ndim >= 2,
            'Prefix array shape/dtype differs')
    unequal = a != b
    chronological = np.swapaxes(unequal, 0, 1)
    first = [int(x) for x in np.unravel_index(np.argmax(chronological), chronological.shape)] if unequal.any() else None
    difference = np.abs(a.astype(np.float64)-b.astype(np.float64))
    return dict(unequal_elements=int(unequal.sum()), total_elements=int(a.size),
        max_abs_difference=float(difference.max()), mean_abs_difference=float(difference.mean()),
        first_difference_row_worker_coordinate=first,
        first_difference_callback_index=first[0]*a.shape[0]+first[1] if first else None,
        initial_reset_unequal_elements=int(unequal[:,0].sum()))


def remote_initial():
    """CPU-only read of trusted initial checkpoints and immutable replay prefixes."""
    import gc
    import pickle
    import numpy as np
    def array_sha(value):
        value = np.asarray(value)
        return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()
    index = read(Path(ROOT)/'actual-RUN_INDEX.json')
    rows = index['runs']
    require(len(rows) == 4 and {(r['seed'],r['group']) for r in rows} == set(MATRIX), 'Unexpected repeat matrix')
    result = dict(checked_utc=datetime.now(timezone.utc).isoformat(), training_sha=TRAIN_SHA,
        read_only=True, new_agent_or_environment_execution=False, terminal_acceptance=False, pairs=[])
    for seed in (2,3):
        pair, arrays = {}, {}
        for group in ('baseline','dt'):
            row = next(r for r in rows if r['seed'] == seed and r['group'] == group)
            gpu, mode = MATRIX[seed,group]
            root = OLD_ROOT if group == 'baseline' else ROOT
            require(row['directory'] == f'{root}/seed{seed}/{mode}-attempt01' and
                    row['physical_gpu'] == gpu and row['code_sha'] == TRAIN_SHA, 'Wrong initial-check source')
            directory = Path(row['directory']); manifest = read(directory/'protocol_manifest.json')
            require(manifest['protocol'] == SCOPE and manifest['root_seed'] == seed and
                    manifest['git_commit'] == TRAIN_SHA and manifest['config'] == row['config'] and
                    not manifest['config']['run.engineering_fixture'], 'Wrong initial manifest/config')
            folder = directory/'eval_snapshots/0000000'; tag = (folder/'latest').read_text().strip()
            require(tag and Path(tag).name == tag and tag not in ('.','..'), 'Invalid initial checkpoint tag')
            meta = read(folder/'evaluation_snapshot.json'); payload = folder/tag/'agent.pkl'
            require(meta['actual_action_step'] == meta['update_id'] == 0 and meta['git_commit'] == TRAIN_SHA and
                    meta['snapshot_id'] == f'eval_snapshots/0000000/{tag}' and (folder/tag/'done').is_file(),
                    'Initial snapshot identity/marker differs')
            before = payload.stat()
            with payload.open('rb') as file: saved = pickle.load(file)
            require(saved['counters']['updates'] == 0, 'Checkpoint is not before optimization')
            require(all(np.isfinite(np.asarray(v)).all() for v in saved['params'].values()),
                    'Initial parameter contains a nonfinite value')
            params = {k:array_sha(v) for k,v in sorted(saved['params'].items())}
            item = dict(directory=str(directory),run_id=row['run_id'],physical_gpu=gpu,
                manifest_sha256=sha(directory/'protocol_manifest.json'), metadata=meta,
                checkpoint_sha256=sha(payload), checkpoint_bytes=before.st_size,
                initial_counters=saved['counters'], parameter_array_sha256=params, initial_parameters_finite=True)
            del saved; gc.collect()
            after = payload.stat()
            require((before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns), 'Initial payload changed')
            with (directory/'update_receipts.jsonl').open('rb') as file: first_raw = file.readline()
            first = json.loads(first_raw)
            require(first['update_id'] == 0 and first['start_action'] == 2032 and first['invalid_count'] == 0,
                    'Prefix boundary does not precede the first successful update')
            paths = {p.stem.split('-')[1]:p for p in (directory/'replay').glob('*.npz')}
            successors = {p.stem.split('-')[2] for p in paths.values()}
            roots = sorted((p for key,p in paths.items() if key not in successors), key=lambda p:p.name)
            require(len(roots) == 16 and len({p.stem.split('-')[0] for p in roots}) == 16, 'Replay roots ambiguous')
            fields = ('reward','action','is_first','is_last','is_terminal','image')
            parts, sources = [], []
            for path in roots:
                require(int(path.stem.split('-')[-1]) >= 128, 'Root chunk shorter than pre-update prefix')
                with np.load(path,allow_pickle=False) as data:
                    part = {k:data[k][:128].copy() for k in fields}
                require(all(len(v) == 128 for v in part.values()) and part['is_first'][0] and
                        not part['is_first'][1:].any(), 'Prefix reset/count differs')
                require(np.isfinite(part['reward']).all() and np.isfinite(part['action']).all(),
                        'Nonfinite pre-update reward/action')
                sources.append(dict(path=str(path),prefix_rows=128,
                                    prefix_array_sha256={k:array_sha(v) for k,v in part.items()}))
                parts.append(part)
            arrays[group] = {k:np.stack([p[k] for p in parts]) for k in fields}
            item.update(first_update_receipt=first,first_update_receipt_sha256=hashlib.sha256(first_raw).hexdigest(),
                        prefix_sources=sources,prefix_actions=2032,prefix_resets=16)
            pair[group] = item
        ka, kb = pair['baseline']['parameter_array_sha256'], pair['dt']['parameter_array_sha256']
        require(ka.keys() == kb.keys(), 'Initial parameter name set differs')
        result['pairs'].append(dict(seed=seed,runs=pair,parameter_arrays_identical=ka == kb,
            parameter_array_count=len(ka),changed_parameter_names=[k for k in ka if ka[k] != kb[k]],
            initial_counters_identical=pair['baseline']['initial_counters'] == pair['dt']['initial_counters'],
            prefix_comparison={k:compare_prefix_arrays(arrays['baseline'][k],arrays['dt'][k]) for k in fields}))
    print(json.dumps(result,indent=2,allow_nan=False))


def collect_initial(output):
    output = output.resolve(); output.mkdir(parents=True,exist_ok=False)
    code = Path(__file__).read_text(encoding='utf-8')
    program = "namespace={'__name__':'readonly_initial'}\nexec("+repr(code)+",namespace)\nnamespace['remote_initial']()\n"
    (output/'collector.py').write_text(program,encoding='utf-8')
    with (output/'inspection.json').open('xb') as stream, (output/'stderr.txt').open('xb') as errors:
        subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','sv3',
            "CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/Policy_Discrepancy/envs/dreamer/bin/python -"],
            input=program.encode(),stdout=stream,stderr=errors,check=True)
    result = read(output/'inspection.json')
    dump(output/'verification.json',dict(collector_sha256=sha(output/'collector.py'),
        inspection_sha256=sha(output/'inspection.json'),script_sha256=sha(Path(__file__)),
        analysis_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        interpretation='Read-only actual initial/prefill comparison; not a terminal audit or numerical determinism guarantee.'))
    print(json.dumps([dict(seed=p['seed'],parameter_arrays_identical=p['parameter_arrays_identical'],
        prefix_comparison=p['prefix_comparison']) for p in result['pairs']],indent=2))


def reward_entry_summary(episodes):
    """Describe entry/holding without classifying an arbitrary return as success."""
    rows = []
    for initial_positive in (False, True):
        selected = [e for e in episodes if (e['first_action_reward'] > 0) == initial_positive]
        count = len(selected)
        ever = sum(e['positive_steps'] > 0 for e in selected)
        rows.append(dict(first_action_has_reward=initial_positive, episodes=count,
            ever_reward_episodes=ever, ever_reward_fraction=ever/count if count else None,
            mean_return_raw=math.fsum(e['return_raw'] for e in selected)/count if count else None))
    return rows


def remote_training_rewards(specs, replay_reader):
    """Read complete training replay after the same four-run terminal gate."""
    require(sys.flags.optimize == 0, 'Replay checks require Python assertions enabled')
    root = Path(ROOT)
    rows = cohort(read(root/'actual-RUN_INDEX.json'), read(root/'final-audit-attempt01/completion.json'),
                  read(root/'supervisor-final.json'))
    require(len(specs) == 4 and [s['row'] for s in specs] == rows,
            'Replay sources differ from the accepted terminal cohort')
    result = dict(checked_utc=datetime.now(timezone.utc).isoformat(),
        actual_training_sha=TRAIN_SHA, actual_audit_controller_sha=CONTROLLER_SHA,
        read_only=True, source_kind='training replay, not fixed-snapshot evaluation', runs=[])
    for spec in specs:
        row = spec['row']; directory = Path(row['directory'])
        def verify_sources():
            for name, expected in spec['expected_hashes'].items():
                require(sha(directory/name) == expected, 'Terminal source changed: '+name)
        verify_sources()
        manifest = read(directory/'protocol_manifest.json')
        require(manifest['config'] == row['config'] and manifest['git_commit'] == TRAIN_SHA and
                manifest['root_seed'] == row['seed'] and manifest['protocol'] == SCOPE,
                'Replay manifest/config differs from terminal acceptance')
        final = read(directory/'final_state.json')
        require(final['train_action_steps'] == 1000000 and final['updates'] == 249493,
                'Replay source is not a full terminal run')
        def inventory():
            return {str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in (directory/'replay').glob('*.npz')}
        before = inventory()
        evidence = replay_reader(directory, row['config'], final, prefix_images=False)
        require(before == inventory(), 'Replay inventory changed during read')
        verify_sources()
        summary = evidence['replay_summary']
        require(summary['actions'] == 1000000 and summary['complete_episodes'] == 992 and
                summary['resets'] == 1008 and summary['terminals'] == 0 and
                set(summary['reward_values']).issubset({0.,1.}) and
                all(e['length'] == 1000 for e in evidence['training_episodes']),
                'Pendulum raw reward or full replay counts differ')
        result['runs'].append(dict(seed=row['seed'], group=row['group'], run_id=row['run_id'],
            directory=row['directory'], source_hashes=spec['expected_hashes'],
            reward_entry=reward_entry_summary(evidence['training_episodes']), **evidence))
    print(json.dumps(result, allow_nan=False))


def collect_training_rewards(source, output):
    """Reuse terminal-mirror identities, read replay on sv3 CPU, retain scalar evidence."""
    from analysis.analyze_m2_v1 import csv_write
    source, output = source.resolve(), output.resolve()
    proof = verify_mirror(source); mirror = source/'inputs'
    locate = lambda remote: mirror.joinpath(*PurePosixPath(remote).parts[1:])
    root = locate(ROOT)
    rows = cohort(read(root/'actual-RUN_INDEX.json'), read(root/'final-audit-attempt01/completion.json'),
                  read(root/'supervisor-final.json'))
    names = ('protocol_manifest.json', 'final_state.json', 'evaluations.jsonl', 'update_receipts.jsonl')
    specs = [dict(row=row, expected_hashes={name:proof['files'][
        row['directory'].lstrip('/')+'/'+name]['sha256'] for name in names}) for row in rows]
    code = Path(__file__).read_text(encoding='utf-8')
    helper_path = Path(__file__).with_name('diagnose_pendulum_v1.py')
    helper = helper_path.read_text(encoding='utf-8')
    program = ("helper={'__name__':'readonly_replay_helper'}\nexec("+repr(helper)+",helper)\n"
        "namespace={'__name__':'readonly_training_rewards'}\nexec("+repr(code)+",namespace)\n"
        "namespace['remote_training_rewards']("+repr(specs)+",helper['training_replay'])\n")
    output.mkdir(parents=True,exist_ok=False)
    (output/'collector.py').write_text(program,encoding='utf-8')
    with (output/'inspection.json').open('xb') as stream, (output/'stderr.txt').open('xb') as errors:
        subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','sv3',
            "CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/Policy_Discrepancy/envs/dreamer/bin/python -"],
            input=program.encode(),stdout=stream,stderr=errors,check=True)
    result = read(output/'inspection.json')
    require([(r['seed'],r['group'],r['run_id']) for r in result['runs']] ==
            [(r['seed'],r['group'],r['run_id']) for r in rows], 'Collected replay identities differ')
    summary_rows, entry_rows = [], []
    for run in result['runs']:
        s = run['replay_summary']; identity = dict(seed=run['seed'],group=run['group'],run_id=run['run_id'])
        summary_rows.append(dict(**identity, **{k:s[k] for k in ('actions','resets','complete_episodes',
            'zero_return_episodes','positive_actions','first_positive_action','total_reward_sum')},
            complete_episode_mean_return=s['episode_returns']['mean'],
            first_100k_positive_actions=sum(p['positive_reward_count'] for p in run['early_5k'])))
        entry_rows.extend(dict(**identity,**e) for e in run['reward_entry'])
    csv_write(output/'training_reward_summary.csv',summary_rows)
    csv_write(output/'training_reward_entry.csv',entry_rows)
    dump(output/'verification.json',dict(collector_sha256=sha(output/'collector.py'),
        inspection_sha256=sha(output/'inspection.json'),script_sha256=sha(Path(__file__)),
        helper_sha256=sha(helper_path), input_archive_sha256=sha(source/'inputs.tar.gz'),
        analysis_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        training_actions=4000000, formal_evaluation_metrics_changed=False,
        limitations=['Training replay reward sums checked independently; evaluation transition rewards not replayed.',
                     'Array hashes/ZIP CRC for named scalar fields; not complete image/replay file hashes.',
                     'First-action reward is after the first action, not a measured reset angle.',
                     'Training episodes share a changing policy and are not independent training seeds.']))
    print(json.dumps(summary_rows,indent=2))


def summarize(rows):
    """Per-seed update-weighted summaries; quantiles remain means of batch quantiles."""
    from analysis.analyze_m2_v1 import FIELDS
    out = []
    for seed, group in sorted({(r['seed'], r['group']) for r in rows}):
        members = [r for r in rows if (r['seed'], r['group']) == (seed, group)]
        for stage, lo, hi in (('full',0,1000000), ('early',0,100000), ('match',100000,300000),
                              ('middle',300000,800000), ('late',800000,1000000)):
            bins = [r for r in members if lo <= r['start_action'] and r['end_action'] <= hi]
            total = sum(r['updates'] for r in bins)
            require(total > 0, 'Missing diagnostic stage')
            value = dict(task=TASK, seed=seed, group=group, stage=stage, updates=total)
            for field in FIELDS + ('D_zero_frac', 'pooled_active_frac'):
                value[field] = math.fsum(r[field]*r['updates'] for r in bins)/total
            raw = value['weighted_excess_mean']+value['excess_reduction_mean']
            value['excess_reduction_ratio'] = value['excess_reduction_mean']/raw if raw else None
            active = math.fsum(r['pooled_active_frac']*r['updates'] for r in bins)
            value['pooled_source_active_weight'] = math.fsum(
                r['pooled_source_active_weight']*r['pooled_active_frac']*r['updates'] for r in bins)/active if active else None
            value['rep_value_reduction_ratio'] = 1-value['rep_after_mean']/value['rep_before_mean']
            out.append(value)
    return out


def analyze(source, output):
    from analysis import analyze_m2_v1 as m2
    source, output = source.resolve(), output.resolve()
    proof = verify_mirror(source); mirror = source/'inputs'
    locate = lambda remote: mirror.joinpath(*PurePosixPath(remote).parts[1:])
    root = locate(ROOT); final_audit = root/'final-audit-attempt01'
    rows = cohort(read(root/'actual-RUN_INDEX.json'), read(final_audit/'completion.json'),
                  read(root/'supervisor-final.json'))
    output.mkdir(parents=True, exist_ok=False)
    local_index = [dict(r, directory=str(locate(r['directory']).resolve()),
        source_directory=r['directory'], snapshot_audit_file=str((mirror/'COLLECTION.json').resolve()),
        source_archive=str((source/'inputs.tar.gz').resolve())) for r in rows]
    dump(output/'RUN_INDEX.json', dict(runs=local_index))
    subprocess.run([sys.executable, '-m', 'analysis.return_metrics', '--profile', 'pendulum-seeds23',
        '--index', str(output/'RUN_INDEX.json'), '--output', str(output/'return-audit'),
        '--require-complete'], cwd=Path(__file__).resolve().parents[1], check=True)
    tables = read(output/'return-audit/tables.json'); server_tables = read(final_audit/'return-audit/tables.json')
    diagnostics, receipt_audits, constants = [], [], []
    for row in rows:
        directory = locate(row['directory']); final = read(directory/'final_state.json')
        ev = [json.loads(line) for line in (directory/'evaluations.jsonl').read_text().splitlines()]
        require(final['train_action_steps'] == 1000000 and final['updates'] == 249493 and
                final['eval_action_steps'] == sum(sum(e['lengths']) for e in ev), 'Final action/update count differs')
        receipt = m2.receipts(directory/'update_receipts.jsonl', final, ev)
        receipt_audits.append(dict(run_id=row['run_id'], seed=row['seed'], group=row['group'], **receipt))
        diag = m2.diagnostics(directory, row, receipt, ev)
        # Historical helper labels seed0; this cohort always carries its actual seed.
        diagnostics.extend(dict(r, seed=row['seed']) for r in diag)
        metric = next(r for r in tables['seed_metrics'] if r['run_id'] == row['run_id'])
        old = next(r for r in server_tables['seed_metrics'] if r['run_id'] == row['run_id'])
        require(all(metric[k] == old[k] for k in ('auc_return_per_budget', 'tail_mean_return',
                                                 'auc_complete', 'tail_complete')), 'Controller/independent metric mismatch')
        if row['group'] == 'baseline':
            frozen = read(directory/'c_frozen.json')
            require(frozen['root_seed'] == row['seed'] and frozen['protocol'] == SCOPE and
                    frozen['baseline_dir'] == row['directory'] and frozen['baseline_git_commit'] == TRAIN_SHA and
                    frozen['baseline_config_sha256'] == hashlib.sha256(json.dumps(row['config'], sort_keys=True).encode()).hexdigest() and
                    frozen['window'] == [100000,300000] and not frozen['engineering_fixture'] and
                    frozen['alpha'] == 20 and frozen['raw_rep_threshold'] == 1 and frozen['invalid_count'] == 0 and
                    frozen['receipt_sha256'] == sha(directory/'update_receipts.jsonl') and
                    (directory/'c_frozen.json.sha256').read_text().strip() == sha(directory/'c_frozen.json') and
                    row['run_audit']['c_artifact_sha256'] == sha(directory/'c_frozen.json') and
                    frozen['included_update_ids'] == receipt['included_update_ids'] and
                    frozen['match_sum'] == receipt['match_sum'] and frozen['match_count'] == receipt['match_count'] and
                    frozen['c'] == receipt['match_sum']/receipt['match_count'] == row['run_audit']['c'],
                    'Frozen baseline c does not match full original receipts')
            constants.append(dict(seed=row['seed'], c=frozen['c'], match_sum=frozen['match_sum'],
                                  match_count=frozen['match_count'], source_directory=row['directory']))
    for name, values in [('gate_intervals',diagnostics), ('gate_summary',summarize(diagnostics)),
                         ('receipt_audits',receipt_audits), ('constants',constants), ('run_manifest',rows)]:
        dump(output/(name+'.json'), values)
        if name in ('gate_intervals','gate_summary','constants'):
            m2.csv_write(output/(name+'.csv'), values)
    plots(output, tables, diagnostics)
    dump(output/'verification.json', dict(input_archive_sha256=sha(source/'inputs.tar.gz'),
        collection_sha256=sha(mirror/'COLLECTION.json'), script_sha256=sha(Path(__file__)),
        analysis_sha=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        helper_sha256={name:sha(Path(__file__).parent/name) for name in ('analyze_m2_v1.py','return_metrics.py')},
        actual_training_sha=TRAIN_SHA, actual_audit_controller_sha=CONTROLLER_SHA,
        raw_metrics_match_controller=True, runs=4, evaluation_points=84, episodes=840,
        receipts=4*249493, checkpoint_basis=proof['runs'][0]['snapshots'][0]['basis'],
        limitations=['Evaluation per-transition reward not independently replayed; recorded episode sums only.',
                     'No automatic seed0/1 pooling, significance or position-selection causal claim.',
                     'Gate quantiles are update means of logged batch quantiles, not a pooled distribution.']))
    print(json.dumps(tables['paired_seed_differences'], indent=2))


def plots(output, tables, diagnostics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.spines.top':False, 'axes.spines.right':False, 'svg.fonttype':'none'})
    styles = [('baseline','#475569','baseline'), ('dt','#c2410c','PPD-KL')]
    fig, axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for seed, ax in zip((2,3),axes):
        for group,color,label in styles:
            points = [r for r in tables['evaluation_points'] if r['seed'] == seed and r['group'] == group]
            ax.plot([r['target_action_step']/1e6 for r in points], [r['mean_return_raw'] for r in points],
                    color=color,label=label,marker='o',ms=3)
        ax.set(title=f'seed{seed}',xlabel='Training actions (millions)',ylabel='Raw episode return',ylim=(-10,1010))
        ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Pendulum swingup | 21 unsmoothed points | 10 episodes / snapshot')
    for suffix in ('png','svg'): fig.savefig(output/f'learning_curves.{suffix}',dpi=170)
    plt.close(fig)
    fields = [('D_mean','Mean Dt'), ('D_zero_frac','Exact Dt=0 fraction'),
              ('rep_active_frac','Free-nats active fraction'), ('active_weight_mean','Actual active weight')]
    fig, axes = plt.subplots(2,4,figsize=(15,7),layout='constrained')
    for seed, row_axes in zip((2,3),axes):
        for ax,(field,title) in zip(row_axes,fields):
            for group,color,label in styles:
                bins = [r for r in diagnostics if r['seed'] == seed and r['group'] == group]
                ax.plot([r['end_action']/1e6 for r in bins], [r[field] for r in bins],color=color,label=label)
            ax.set(title=f'seed{seed} | {title}',xlabel='Training actions (millions)')
            if field != 'D_mean': ax.set_ylim(-.02,1.02)
            ax.grid(alpha=.2)
    axes[0,0].legend(); fig.suptitle('Pendulum gate diagnostics | update means in each 50k-action interval')
    for suffix in ('png','svg'): fig.savefig(output/f'gate_diagnostics.{suffix}',dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('collect','analyze','initial','replay'))
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'collect': collect(args.output)
    elif args.mode == 'initial': collect_initial(args.output)
    else:
        if args.input is None: parser.error(args.mode+' requires --input')
        if args.mode == 'replay': collect_training_rewards(args.input,args.output)
        else: analyze(args.input, args.output)
