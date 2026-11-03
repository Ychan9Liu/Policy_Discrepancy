"""Terminal-only collection/analysis of the four approved pendulum repeats.

python -m analysis.pendulum_seeds23_analysis collect --output NEW_INPUT
python -m analysis.pendulum_seeds23_analysis analyze --input NEW_INPUT --output NEW_RESULT

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
    parser.add_argument('mode', choices=('collect','analyze'))
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'collect': collect(args.output)
    else:
        if args.input is None: parser.error('analyze requires --input')
        analyze(args.input, args.output)
