"""Read-only, CPU-only pendulum seed0 replay forensics; never runs an agent.

collect --source DMC15_INPUT_DIR --output NEW_DIR
analyze --input COLLECTED_DIR --output NEW_DIR
Only trusted project checkpoints are deserialized, on their original server.
"""
import argparse
import concurrent.futures
import csv
import datetime
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import pickle
import subprocess
import sys
import zipfile

import numpy as np


FIELDS = ('reward', 'action', 'is_first', 'is_last', 'is_terminal')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def array_digest(value):
    value = np.asarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   allow_nan=False) + '\n', encoding='utf-8')


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(count=int(a.size), mean=float(a.mean()), min=float(a.min()),
                max=float(a.max()), p10=float(np.percentile(a, 10)),
                p50=float(np.percentile(a, 50)), p90=float(np.percentile(a, 90))) if a.size else None


def replay_order(chains, budget, interval):
    """Replay the historical Driver's ordered callbacks, counting no resets."""
    order = [np.empty(len(c['reward']), dtype=np.int64) for c in chains]
    stamps = [np.empty(len(c['reward']), dtype=np.int64) for c in chains]
    cursors = [0] * len(chains)
    actions = event = 0
    while actions < budget:
        boundary = min((actions // interval + 1) * interval, budget)
        for worker in range(min(len(chains), boundary - actions)):
            i = cursors[worker]
            if i >= len(chains[worker]['reward']):
                raise ValueError('Replay ends before requested action budget')
            actions += not bool(chains[worker]['is_first'][i])
            order[worker][i] = event
            stamps[worker][i] = actions
            event += 1
            cursors[worker] += 1
    if cursors != [len(c['reward']) for c in chains]:
        raise ValueError('Unconsumed replay rows / worker ordering mismatch')
    indices = np.argsort(np.concatenate(order))
    merged = {k: np.concatenate([c[k] for c in chains])[indices] for k in FIELDS}
    # reward at row t resulted from action at t-1 in this worker's chain.
    applied = [np.concatenate([np.zeros_like(c['action'][:1]), c['action'][:-1]]) for c in chains]
    merged['applied_raw_action'] = np.concatenate(applied)[indices]
    return merged, np.concatenate(stamps)[indices], stamps


def episodes_from_chain(chain, stamps, worker):
    first, last = chain['is_first'], chain['is_last']
    if not first[0]:
        raise ValueError('Missing initial reset')
    if not np.array_equal(np.flatnonzero(first)[1:], np.flatnonzero(last) + 1):
        raise ValueError('Reset and complete episode boundaries disagree')
    episodes, start = [], 0
    for end in np.flatnonzero(last):
        reward = chain['reward'][start + 1:end + 1]
        episodes.append(dict(worker=worker, start_index=int(start), end_index=int(end),
                             start_action=int(stamps[start + 1]), end_action=int(stamps[end]),
                             length=int(end - start), return_raw=float(reward.sum(dtype=np.float64)),
                             positive_steps=int((reward > 0).sum())))
        start = int(end + 1)
    tail = chain['reward'][start + 1:]
    return episodes, dict(worker=worker, length=len(tail), return_raw=float(tail.sum(dtype=np.float64)))


def phase(fields, stamps, lo, hi):
    use = (stamps > lo) & (stamps <= hi) & ~fields['is_first']
    reward, action = fields['reward'][use], fields['applied_raw_action'][use]
    return dict(start_action=lo, end_action=hi, action_count=int(use.sum()),
                positive_reward_count=int((reward > 0).sum()),
                reward_sum=float(reward.sum(dtype=np.float64)), reward=stats(reward),
                action_mean=action.mean(0).tolist(), action_std=action.std(0).tolist(),
                raw_action_abs_ge1_fraction=float((abs(action) >= 1).mean()),
                clipped_action_mean=np.clip(action, -1, 1).mean(0).tolist(),
                clipped_action_std=np.clip(action, -1, 1).std(0).tolist())


def inspect(spec):
    directory = Path(spec['directory'])
    assert directory.is_absolute() and str(directory).startswith('/data/Policy_Discrepancy/runs/formal-')
    result = dict(spec=spec, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  kind='CPU numpy/pickle only, no agent/environment/GPU execution')
    for name, expected in spec['expected_hashes'].items():
        assert digest(directory / name) == expected, ('Source changed', name)
    manifest = read(directory / 'protocol_manifest.json')
    config = manifest['config']
    assert manifest['git_commit'] == spec['training_sha']
    assert config['task'] == 'dmc_pendulum_swingup' and config['seed'] == 0
    assert not config['run.engineering_fixture']
    result.update(manifest=manifest, final=read(directory / 'final_state.json'))
    for name, key in [('metrics.jsonl', 'metrics'), ('evaluations.jsonl', 'evaluations')]:
        result[key] = [json.loads(line) for line in (directory / name).read_text().splitlines()]
    folder = directory / 'eval_snapshots/0000000'
    tag = (folder / 'latest').read_text().strip()
    assert Path(tag).name == tag
    checkpoint = folder / tag / 'agent.pkl'
    with checkpoint.open('rb') as file:
        saved = pickle.load(file)
    hashes = {k: array_digest(v) for k, v in sorted(saved['params'].items())}
    result['initial_checkpoint'] = dict(path=str(checkpoint), file_sha256=digest(checkpoint),
        bytes=checkpoint.stat().st_size, counters=saved['counters'], array_sha256=hashes,
        combined_array_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest())
    del saved
    gc.collect()
    paths = {p.stem.split('-')[1]: p for p in (directory / 'replay').glob('*.npz')}
    successors = {p.stem.split('-')[2] for p in paths.values()}
    roots = sorted((p for key, p in paths.items() if key not in successors), key=lambda p: p.name)
    assert len(roots) == config['run.envs'] == 16
    assert len({p.stem.split('-')[0] for p in roots}) == len(roots)
    chains, chunks, prefixes, visited = [], [], [], set()
    for worker, root in enumerate(roots):
        parts, path = [], root
        while path is not None:
            _, uuid, successor, length = path.stem.split('-')
            assert uuid not in visited
            visited.add(uuid)
            with np.load(path, allow_pickle=False) as file:
                fields = {k: file[k] for k in FIELDS}
                assert all(len(v) == int(length) for v in fields.values())
                if not parts:
                    assert int(length) >= 128
                    prefix = {k: v[:128] for k, v in fields.items()}
                    prefix['image'] = file['image'][:128]
                    prefixes.append({k: array_digest(v) for k, v in prefix.items()})
                with zipfile.ZipFile(path) as archive:
                    members = {k: dict(array_sha256=array_digest(v),
                        crc=archive.getinfo(k+'.npy').CRC,
                        bytes=archive.getinfo(k+'.npy').file_size) for k, v in fields.items()}
            chunks.append(dict(path=str(path), worker=worker, uuid=uuid,
                               successor=successor, rows=int(length), members=members))
            parts.append(fields)
            path = paths.get(successor)
        chains.append({k: np.concatenate([p[k] for p in parts]) for k in FIELDS})
    assert len(visited) == len(paths)
    final = result['final']
    fields, stamps, chain_stamps = replay_order(chains, final['train_action_steps'], config['run.eval_every_actions'])
    assert final['train_action_steps'] == 1000000
    assert len(stamps) == 1000000 + final['train_resets']
    assert fields['is_first'].sum() == final['train_resets']
    assert np.isfinite(fields['reward']).all() and np.isfinite(fields['action']).all()
    assert (fields['reward'][fields['is_first']] == 0).all()
    episodes, tails = [], []
    for worker, (chain, s) in enumerate(zip(chains, chain_stamps)):
        complete, tail = episodes_from_chain(chain, s, worker)
        episodes.extend(complete)
        tails.append(tail)
    all_reward = fields['reward'][~fields['is_first']]
    assert sum(e['length'] for e in episodes + tails) == 1000000
    assert math.isclose(sum(e['return_raw'] for e in episodes + tails), float(all_reward.sum(dtype=np.float64)), abs_tol=1e-10)
    positives = stamps[(fields['reward'] > 0) & ~fields['is_first']]
    values, counts = np.unique(all_reward, return_counts=True)
    result.update(replay_chunks=chunks, root_prefix_hashes=prefixes, training_episodes=episodes,
        incomplete_tails=tails,
        replay_summary=dict(chains=16, rows=len(stamps), actions=1000000,
            resets=int(fields['is_first'].sum()), terminals=int(fields['is_terminal'].sum()),
            complete_episodes=len(episodes), episode_returns=stats([e['return_raw'] for e in episodes]),
            zero_return_episodes=sum(e['return_raw'] == 0 for e in episodes),
            first_positive_action=int(positives[0]) if len(positives) else None,
            reward_values=values.tolist(), reward_value_counts=counts.tolist(),
            finite_reward_and_action=True, scalar_array_sha256={k: array_digest(v) for k, v in fields.items()},
            total_reward_sum=float(all_reward.sum(dtype=np.float64)), positive_actions=len(positives)),
        phases=[phase(fields, stamps, lo, lo+50000) for lo in range(0, 1000000, 50000)],
        early_5k=[phase(fields, stamps, lo, lo+5000) for lo in range(0, 100000, 5000)],
        prefill_2032=phase(fields, stamps, 0, 2032))
    dist = importlib.metadata.distribution('dm-control')
    result['installed_source'] = {}
    for name in ('suite/pendulum.py', 'suite/pendulum.xml', 'utils/rewards.py'):
        source = Path(dist.locate_file('dm_control/' + name))
        result['installed_source'][name] = dict(path=str(source), sha256=digest(source), source=source.read_text())
    result['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(json.dumps(result, allow_nan=False))


def collect(source, output):
    output.mkdir(parents=True, exist_ok=False)
    specs, input_hashes = [], {}
    for server in ('sv1', 'sv2', 'sv3'):
        path = source / (server + '.json')
        input_hashes[str(path.resolve())] = digest(path)
        for run in read(path)['runs']:
            row = run['row']
            if row['task'] != 'dmc_pendulum_swingup':
                continue
            specs.append(dict(server=server, directory=row['directory'], group=row['group'],
                run_id=row['run_id'], physical_gpu=row['physical_gpu'], state_at_previous_audit=row['state'],
                training_sha=row['code_sha'], expected_hashes=run['hashes']))
    assert len(specs) == 2 and {r['group'] for r in specs} == {'baseline', 'dt'}
    dump(output / 'collection_inputs.json', dict(sources=input_hashes, runs=specs))
    source_code = Path(__file__).read_text(encoding='utf-8')
    def one(spec):
        program = "namespace={'__name__':'readonly_inspection'}\nexec(" + repr(source_code) + ",namespace)\nnamespace['inspect'](" + repr(spec) + ")\n"
        (output / (spec['group']+'-remote.py')).write_text(program, encoding='utf-8')
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', spec['server'],
                   "CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/Policy_Discrepancy/envs/dreamer/bin/python -"]
        proc = subprocess.run(command, input=program, text=True, capture_output=True)
        (output / (spec['group']+'-stderr.txt')).write_text(proc.stderr, encoding='utf-8')
        proc.check_returncode()
        result = json.loads(proc.stdout)
        result['collector_sha256'] = hashlib.sha256(source_code.encode()).hexdigest()
        dump(output / (spec['group']+'.json'), result)
        print(spec['group'], result['replay_summary'], flush=True)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        list(pool.map(one, specs))


def compare_prefix(source, output):
    """Copy only 16 x 128 pre-update rows, never a full replay/checkpoint."""
    output.mkdir(parents=True, exist_ok=False)
    proofs = {g: read(source / (g+'.json')) for g in ('baseline', 'dt')}
    remote = '''from pathlib import Path
import numpy as np, sys
parts=[]
for path in paths:
 with np.load(path,allow_pickle=False) as z:
  parts.append({k:z[k][:128] for k in ('reward','action','is_first','is_last','is_terminal','image')})
np.savez_compressed(sys.stdout.buffer,**{k:np.stack([p[k] for p in parts]) for k in parts[0]})
'''
    def one(group):
        proof = proofs[group]
        paths = [next(c['path'] for c in proof['replay_chunks'] if c['worker'] == i) for i in range(16)]
        program = 'paths=' + repr(paths) + '\n' + remote
        (output / (group+'-remote.py')).write_text(program, encoding='utf-8')
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', proof['spec']['server'],
            "CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/Policy_Discrepancy/envs/dreamer/bin/python -"]
        proc = subprocess.run(command, input=program.encode(), capture_output=True)
        (output / (group+'-stderr.txt')).write_bytes(proc.stderr)
        proc.check_returncode()
        path = output / (group+'.npz')
        path.write_bytes(proc.stdout)
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        for k, values in arrays.items():
            assert all(array_digest(v) == old[k] for v, old in zip(values, proof['root_prefix_hashes']))
        return group, arrays
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        arrays = dict(pool.map(one, proofs))
    result = dict(source_hashes={g: digest(source / (g+'.json')) for g in proofs},
                  copied_npz_hashes={g: digest(output / (g+'.npz')) for g in proofs}, fields={})
    for k, a in arrays['baseline'].items():
        b = arrays['dt'][k]
        unequal = a != b
        # First chronological callback: row first, then ordered worker.
        where = np.argwhere(np.swapaxes(unequal, 0, 1))
        first = where[0].tolist() if len(where) else None
        difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
        result['fields'][k] = dict(unequal_elements=int(unequal.sum()), total_elements=a.size,
            max_abs_difference=float(difference.max()), mean_abs_difference=float(difference.mean()),
            first_difference_row_worker_coordinate=first,
            first_difference_callback_index=first[0]*16+first[1] if first else None,
            initial_reset_unequal_elements=int(unequal[:, 0].sum()))
    dump(output / 'comparison.json', result)
    print(json.dumps(result))


def analyze(source, output):
    output.mkdir(parents=True, exist_ok=False)
    runs = {group: read(source / (group+'.json')) for group in ('baseline', 'dt')}
    checks = dict(source_hashes={group: digest(source / (group+'.json')) for group in runs})
    a, b = runs.values()
    checks['initial_arrays_identical'] = a['initial_checkpoint']['array_sha256'] == b['initial_checkpoint']['array_sha256']
    checks['initial_counters_identical'] = a['initial_checkpoint']['counters'] == b['initial_checkpoint']['counters']
    checks['prefill_identical_by_field'] = {k: all(x[k] == y[k] for x, y in zip(a['root_prefix_hashes'], b['root_prefix_hashes'])) for k in a['root_prefix_hashes'][0]}
    ca, cb = a['manifest']['config'], b['manifest']['config']
    checks['config_differences'] = {k: [ca.get(k), cb.get(k)] for k in ca.keys() | cb.keys() if ca.get(k) != cb.get(k)}
    assert set(checks['config_differences']) == {'logdir', 'agent.rep_probe.mode'}
    rows, summaries, curves = [], [], []
    fields = dict(imagined_reward='train/rew', imagined_normalized_return='train/ret',
        advantage_magnitude='train/adv_mag', policy_entropy='train/ent/action',
        reward_model_loss='train/loss/rew', image_loss='train/loss/image',
        gradient_norm='train/opt/grad_norm', update_rms='train/opt/update_rms',
        active_weight='train/dt/active_weight_mean', active_fraction='train/dt/rep_active_frac',
        D_mean='train/dt/D_mean', mean_abs_actor_mean_diff='train/dt/mean_abs_diff',
        mean_abs_actor_std_diff='train/dt/std_abs_diff')
    for group, run in runs.items():
        ev = run['evaluations']
        assert [e['action_step'] for e in ev] == list(range(0, 1000001, 50000))
        means = []
        for e in ev:
            assert len(e['episodes']) == 10 and all(p['complete'] and p['length'] == 1000 for p in e['episodes'])
            mean = math.fsum(p['return_raw'] for p in e['episodes']) / 10
            assert math.isclose(mean, e['mean'], abs_tol=1e-10)
            means.append(mean)
            curves.append(dict(group=group, action_step=e['action_step'], mean_return=mean))
        logs = [r for r in run['metrics'] if 'train/dt/D_mean' in r]
        assert len(logs) == len(run['phases']) == 20
        scales = {k.removeprefix('agent.loss_scales.'): v for k, v in run['manifest']['config'].items() if k.startswith('agent.loss_scales.')}
        scales['image'] = scales.pop('rec')
        loss_errors = []
        for i, (phase_row, log) in enumerate(zip(run['phases'], logs)):
            assert log['step'] == phase_row['end_action'] and phase_row['action_count'] == 50000
            assert all(math.isfinite(v) for v in log.values() if isinstance(v, (float, int)))
            total = math.fsum(log['train/loss/'+k]*v for k, v in scales.items())
            loss_errors.append(abs(total-log['train/opt/loss']))
            assert math.isclose(total, log['train/opt/loss'], rel_tol=1e-7, abs_tol=1e-6)
            row = dict(group=group, **phase_row, **{k: log[v] for k, v in fields.items()},
                       update_count=ev[i+1]['update_id']-ev[i]['update_id'], evaluation_return=means[i+1],
                       positive_reward_fraction=phase_row['positive_reward_count']/50000,
                       excess_reduction=log['train/dt/excess_reduction_mean'],
                       weighted_excess=log['train/dt/weighted_excess_mean'],
                       D_zero_fraction=log['train/dt/D_zero_count']/log['train/dt/loss_position_count'])
            rows.append(row)
        summaries.append(dict(group=group, **run['replay_summary'],
            auc=math.fsum((x+y)*.025 for x, y in zip(means, means[1:])),
            tail_five_mean=math.fsum(means[-5:])/5,
            first_100k_positive_steps=sum(p['positive_reward_count'] for p in run['phases'][:2]),
            first_100k_reward=sum(p['reward_sum'] for p in run['phases'][:2]),
            max_loss_reconstruction_abs_error=max(loss_errors)))
    for name, data in [('phases', rows), ('summary', summaries), ('learning_points', curves)]:
        dump(output / (name+'.json'), data)
        flat = [{k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in data]
        with (output / (name+'.csv')).open('w', encoding='utf-8-sig', newline='') as file:
            writer = csv.DictWriter(file, list(dict.fromkeys(k for r in flat for k in r)))
            writer.writeheader()
            writer.writerows(flat)
    dump(output / 'verification.json', checks)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), layout='constrained')
    series = [('evaluation_return', 'Unsmoothed evaluation return'),
              ('positive_reward_fraction', 'Real positive-reward fraction / 50k actions'),
              ('imagined_reward', 'Model imagined reward / update mean'),
              ('policy_entropy', 'Imagined Gaussian policy entropy'),
              ('advantage_magnitude', 'Imagined advantage magnitude / update mean'),
              ('active_weight', 'Actual active rep weight / update mean')]
    for ax, (key, title) in zip(axes.flat, series):
        for group, color, label in [('baseline', '#475569', 'baseline'), ('dt', '#c2410c', 'PPD-KL')]:
            subset = [r for r in rows if r['group'] == group]
            x, y = [r['end_action']/1e6 for r in subset], [r[key] for r in subset]
            if key == 'evaluation_return':
                point0 = next(c['mean_return'] for c in curves if c['group'] == group and c['action_step'] == 0)
                x, y = [0]+x, [point0]+y
            ax.plot(x, y, color=color, label=label)
        ax.set(title=title, xlabel='Training actions (millions)')
        ax.grid(alpha=.2)
        if key in ('imagined_reward', 'advantage_magnitude'):
            ax.set_yscale('log')
        if key == 'policy_entropy':
            ax.axhline(.5*math.log(2*math.pi*math.e), color='black', ls='--', lw=.8)
    axes[0, 0].legend()
    fig.suptitle('Pendulum swingup | seed0 | original replay and JSONL')
    fig.savefig(output / 'diagnostics.png', dpi=160)
    fig.savefig(output / 'diagnostics.svg')
    plt.close(fig)
    print(json.dumps(dict(checks=checks, summary=summaries), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('collect', 'analyze', 'prefix'))
    parser.add_argument('--source', '--input', dest='source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'collect': collect, 'analyze': analyze, 'prefix': compare_prefix}[args.mode](args.source, args.output)


if __name__ == '__main__':
    main()
