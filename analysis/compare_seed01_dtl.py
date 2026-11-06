"""Read-only seed1, seed0/1 equal means, and three-task seed0 DtL analysis.

collect --inventory INVENTORY_DIR --output NEW_INPUT
analyze --input NEW_INPUT --output NEW_RESULT
Acrobot is excluded by default for the earlier analysis; --include-acrobot
selects the user's subsequent full 20-task request without dropping old inputs.
No agent, environment, GPU, W&B writes or server source changes.
"""
import argparse
import concurrent.futures
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

GRID = tuple(range(0, 1000001, 50000))
DTL_TASKS = ('dmc_quadruped_run', 'dmc_quadruped_walk', 'dmc_finger_spin')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def csv_dump(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def evaluation_metrics(evaluations):
    require([e['action_step'] for e in evaluations] == list(GRID), 'Incomplete/duplicate evaluation grid')
    means = []
    for e in evaluations:
        episodes = e['episodes']
        require(len(episodes) == 10 and [p['episode_id'] for p in episodes] == list(range(10)), 'Not ten unique episodes')
        require(all(p['complete'] and p['length'] == 1000 for p in episodes), 'Incomplete episode')
        require(all(p['return_definition'] == 'undiscounted_sum_non_reset_rewards_repeat_already_summed' for p in episodes), 'Wrong return definition')
        require(all(math.isfinite(p['return_raw']) for p in episodes), 'Nonfinite return')
        mean = math.fsum(p['return_raw'] for p in episodes)/10
        require(math.isclose(mean, e['mean'], rel_tol=1e-12, abs_tol=1e-9), 'Episode mean disagrees')
        means.append(mean)
    return dict(auc=math.fsum((a+b)*.025 for a, b in zip(means, means[1:])),
                tail=math.fsum(means[-5:])/5, curve=means)


def inspect_one(spec):
    """Run on source host (stdlib); only named small files and receipt streams."""
    d = Path(spec.get('local_directory', spec['row']['directory']))
    row = spec['row']
    m, f = read(d/'protocol_manifest.json'), read(d/'final_state.json')
    c = m['config']
    require(m['git_commit'] == row['code_sha'] and c == row['config'], 'Manifest/config/SHA differs')
    require(not c['run.engineering_fixture'] and c['seed'] == row['seed'] and c['task'] == row['task'], 'Wrong scope')
    require(f['train_action_steps'] == 1000000 and f['updates'] == f['train_batch_keys'] == 249493, 'Budget/update mismatch')
    require(f['eval_action_steps'] == 210000 and f['evaluations'] == 21, 'Wrong evaluation budget')
    ev = [json.loads(line) for line in (d/'evaluations.jsonl').read_text().splitlines()]
    evaluation_metrics(ev)
    snapshots = []
    for e in ev:
        folder = d/'eval_snapshots'/f"{e['action_step']:07d}"
        tag = (folder/'latest').read_text().strip()
        require(Path(tag).name == tag and tag not in ('.', '..'), 'Unsafe snapshot tag')
        s = read(folder/'evaluation_snapshot.json')
        require(s['actual_action_step'] == e['action_step'] and s['update_id'] == e['update_id'] and
                s['snapshot_id'] == e['snapshot_id'] and s['git_commit'] == m['git_commit'], 'Snapshot differs')
        payload = folder/tag/'agent.pkl'
        require((folder/tag/'done').is_file(), 'Snapshot not completed')
        # Old small mirrors intentionally exclude large parameter payloads.
        if not spec.get('local_directory'):
            require(payload.is_file(), 'Snapshot payload absent')
        snapshots.append(dict(metadata=s, metadata_sha256=sha(folder/'evaluation_snapshot.json'),
                              payload_present=payload.exists(), payload_independently_decoded=False))
    names = ['protocol_manifest.json','final_state.json','evaluations.jsonl','metrics.jsonl','update_receipts.jsonl']
    hashes = {n: sha(d/n) for n in names}
    for n, expected in spec.get('expected_hashes', {}).items():
        require(sha(d/n) == expected, 'Previously audited artifact changed: '+n)
    count, total, n, last, interval = 0, 0., 0, -1, 0
    bins = [dict(updates=0, active_sum=0, match_sum=0.) for _ in range(20)]
    starts = []
    with (d/'update_receipts.jsonl').open() as file:
        for line in file:
            r = json.loads(line)
            require(r['update_id'] == count and r['invalid_count'] == 0, 'Missing/duplicate/invalid update')
            stamp = r['start_action']
            require(last <= stamp <= 1000000 and isinstance(stamp, int), 'Update stamp invalid')
            while count >= ev[interval+1]['update_id']:
                interval += 1
                require(interval < 20, 'Extra update beyond endpoint snapshot')
            require(GRID[interval] < stamp <= GRID[interval+1], 'Receipt/snapshot interval mismatch')
            require(0 <= r['match_count'] <= 1024 and math.isfinite(r['match_sum']) and
                    0 <= r['match_sum'] <= r['match_count']+1e-4, 'Invalid matching S/N')
            b = bins[interval]
            b['updates'] += 1; b['active_sum'] += r['match_count']; b['match_sum'] += r['match_sum']
            if 100000 <= stamp < 300000:
                total += r['match_sum']; n += r['match_count']
            starts.append(stamp); count += 1; last = stamp
    require(count == 249493 == ev[-1]['update_id'] and ev[0]['update_id'] == 0 and last == 1000000,
            'Final receipt/snapshot count differs')
    require(total == f['match_sum'] and n == f['match_count'], 'Final matching S/N differs')
    frozen = None
    if row['group'] == 'baseline':
        frozen = read(d/'c_frozen.json')
        hashes['c_frozen.json'] = sha(d/'c_frozen.json')
        require(frozen['window'] == [100000,300000] and not frozen['engineering_fixture'] and
                frozen['receipt_sha256'] == hashes['update_receipts.jsonl'] and n > 0 and
                frozen['match_sum'] == total and frozen['match_count'] == n and frozen['c'] == total/n,
                'Frozen c provenance/S/N differs')
    result = dict(**spec, collected_utc=datetime.now(timezone.utc).isoformat(), manifest=m, final=f,
        evaluations=ev, metrics=[json.loads(x) for x in (d/'metrics.jsonl').read_text().splitlines()],
        snapshots=snapshots, hashes=hashes, frozen=frozen,
        receipts=dict(count=count, match_sum=total, match_count=n, intervals=bins))
    if row['group'].lower() == 'dtl':
        dbins = [dict(updates=0, sums={}, min_weight=1., max_weight=0.) for _ in range(20)]
        count, interval = 0, 0
        with (d/'dtl_updates.jsonl').open() as file:
            for line in file:
                r = json.loads(line)
                require(r['update_id'] == count and r['start_action'] == starts[count], 'DtL receipt alignment differs')
                require(all(math.isfinite(v) for v in r.values()), 'Nonfinite DtL receipt')
                require(.5 <= r['actual_v_min'] <= r['actual_v_mean']+1e-6 <= r['actual_v_max']+1e-6 <= 1+1e-6,
                        'DtL weight outside rho bounds')
                while count >= ev[interval+1]['update_id']:
                    interval += 1
                b = dbins[interval]; b['updates'] += 1
                for k, v in r.items():
                    if k not in ('update_id','start_action'):
                        b['sums'][k] = b['sums'].get(k, 0.)+v
                b['min_weight'] = min(b['min_weight'], r['actual_v_min'])
                b['max_weight'] = max(b['max_weight'], r['actual_v_max'])
                count += 1
        require(count == 249493, 'Incomplete DtL receipts')
        isolation = [json.loads(x) for x in (d/'engineering_evaluation_audit.jsonl').read_text().splitlines()]
        require([r['action_step'] for r in isolation] == list(GRID), 'Missing DtL evaluation isolation audit')
        hashes.update({n: sha(d/n) for n in ('dtl_updates.jsonl','engineering_evaluation_audit.jsonl')})
        result.update(dtl_receipt_intervals=dbins, evaluation_isolation=isolation)
    return result


def select_runs(inventory, include_acrobot=False):
    selected, excluded = {}, []
    for server in ('sv1','sv2','sv3'):
        for index in read(inventory/(server+'-inventory.json'))['indices']:
            by_dir = {p['directory']:p for p in index['probes']}
            for row in index['data']['runs']:
                if (row['task'] == 'dmc_acrobot_swingup' and not include_acrobot) or row['seed'] not in (0,1):
                    continue
                is_dtl = row['group'].lower() == 'dtl'
                if not is_dtl and row['seed'] != 1:
                    continue
                final = by_dir[row['directory']]['final']
                audit = row.get('run_audit', row.get('audit'))
                if not final or not audit or row['state'] not in ('accepted','run_audited_pending_group_and_WB'):
                    excluded.append(dict(server=server, task=row['task'], group=row['group'],
                                         directory=row['directory'], state=row['state'], reason='No accepted per-run final audit'))
                    continue
                key = (row['task'], row['seed'], row['group'].lower())
                if key in selected:
                    require(selected[key]['row']['directory'] == row['directory'], 'Multiple completed attempts for same slot')
                    continue
                selected[key] = dict(server=server, index_path=index['path'], index_sha256=index['sha256'], row=row)
    require(len(selected) == (43 if include_acrobot else 41), 'Incomplete seed1/DtL source set')
    require({k[0] for k in selected if k[1] == 0} == set(DTL_TASKS), 'Unexpected DtL tasks')
    return list(selected.values()), excluded


def collect(inventory, output, include_acrobot=False):
    output.mkdir(parents=True, exist_ok=False)
    selected, excluded = select_runs(inventory, include_acrobot)
    inventory_hashes = {s:sha(inventory/(s+'-inventory.json')) for s in ('sv1','sv2','sv3')}
    if include_acrobot:
        extra_path = inventory/'acrobot-seed0-inventory.json'
        extra = read(extra_path)['runs']
        require(len(extra) == 2 and
                {(r['row']['task'],r['row']['seed'],r['row']['group']) for r in extra} ==
                {('dmc_acrobot_swingup',0,g) for g in ('baseline','dt')}, 'Incomplete acrobot seed0 sources')
        require(all(r['row'].get('run_audit') and r['row']['state'] in
                    ('accepted','run_audited_pending_group_and_WB') for r in extra), 'Unaudited acrobot seed0')
        selected += extra
        inventory_hashes[extra_path.name] = sha(extra_path)
    dump(output/'selection.json', dict(runs=selected, excluded=excluded,
        inventory_hashes=inventory_hashes,include_acrobot=include_acrobot))
    code = Path(__file__).read_text(encoding='utf-8')
    def one(server):
        specs = [r for r in selected if r['server'] == server]
        program = "namespace={'__name__':'readonly_collect'}\nexec("+repr(code)+",namespace)\n"+\
            'import json,sys\nresults=[]\nfor spec in '+repr(specs)+":\n results.append(namespace['inspect_one'](spec))\n print(spec['row']['task']+' '+spec['row']['group'],file=sys.stderr,flush=True)\nprint(json.dumps(results,allow_nan=False))\n"
        (output/(server+'-remote.py')).write_text(program, encoding='utf-8')
        command = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',server,
            "CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/Policy_Discrepancy/envs/dreamer/bin/python -"]
        proc = subprocess.run(command, input=program, text=True, capture_output=True)
        (output/(server+'-stderr.txt')).write_text(proc.stderr, encoding='utf-8')
        proc.check_returncode()
        result = json.loads(proc.stdout)
        dump(output/(server+'.json'), result)
        print(server, len(result), 'complete sources', flush=True)
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        list(pool.map(one, ('sv1','sv2','sv3')))
    old = []
    source = Path('analysis/outputs/dmc15-analysis-20261102-final')
    for server in ('sv1','sv2','sv3'):
        path = source/(server+'.json')
        for r in read(path)['runs']:
            old.append(dict(**r, server=server, local_input=str(path), local_input_sha256=sha(path)))
    mirror = Path('analysis/outputs/m2-v1-seed0-20261012/inputs')
    index = mirror/'data/Policy_Discrepancy/repo/analysis/outputs/formal-four-groups-pd1-20261010-final-attempt01/RUN_INDEX.json'
    for row in read(index)['runs']:
        if row['group'] not in ('baseline','dt'):
            continue
        d = mirror/row['directory'].lstrip('/')
        spec = dict(server=row['server'], row=row, local_directory=str(d),
                    index_path=str(index), index_sha256=sha(index), reused_previous_audit=True)
        old.append(inspect_one(spec))
    require(len(old) == 38, 'Expected full 19-task seed0 evidence')
    dump(output/'seed0.json', old)


def gate_rows(run):
    row, ev = run['row'], run['evaluations']
    is_dtl = row['group'].lower() == 'dtl'
    prefix = 'train/dl/' if is_dtl else 'train/dt/'
    key = prefix+('actual_v_mean' if is_dtl else 'D_mean')
    logs = [r for r in run['metrics'] if key in r]
    require([r['step'] for r in logs] == list(GRID[1:]), 'Missing gate interval')
    rows = []
    for i, log in enumerate(logs):
        n = ev[i+1]['update_id']-ev[i]['update_id']
        fields = dict(task=row['task'], group=row['group'].lower(), seed=row['seed'],
                      end_action=log['step'], updates=n)
        if is_dtl:
            b = run['dtl_receipt_intervals'][i]
            require(n == b['updates'], 'DtL aggregation count mismatch')
            for k, total in b['sums'].items():
                if prefix+k in log:
                    require(math.isclose(total/n, log[prefix+k], rel_tol=1e-10, abs_tol=1e-9), 'DtL log/receipt scalar differs')
            mapping = dict(D_mean='D_valid_mean', active_frac='active_frac', actual_weight_mean='actual_v_mean',
                active_protected_frac='active_protected_frac', valid_frac='valid_frac', support_mean='e_valid_mean',
                excess_release='excess_release_mean', weighted_excess='weighted_excess_mean')
            fields.update({k:log[prefix+v] for k,v in mapping.items()})
            fields.update(active_weight_mean=None, zero_D_fraction=None,
                          min_weight=b['min_weight'], max_weight=b['max_weight'])
        else:
            mapping = dict(D_mean='D_mean',active_frac='rep_active_frac',active_weight_mean='active_weight_mean',
                actual_weight_mean='actual_weight_mean',excess_release='excess_reduction_mean',weighted_excess='weighted_excess_mean')
            fields.update({k:log[prefix+v] for k,v in mapping.items()})
            fields['zero_D_fraction'] = log[prefix+'D_zero_count']/log[prefix+'loss_position_count']
            if row['group'] == 'baseline':
                require(fields['actual_weight_mean'] == 1 and fields['excess_release'] == 0, 'Baseline not logging-only')
        fields['excess_release_ratio'] = fields['excess_release']/(fields['excess_release']+fields['weighted_excess'])
        rows.append(fields)
    return rows


def paired_means(per_seed, tasks):
    """Equal training-seed means; never pool evaluation episodes as seeds."""
    result = []
    for task in tasks:
        members = [r for r in per_seed if r['task'] == task]
        require(len(members) == 4 and {(r['seed'],r['group']) for r in members} ==
                {(s,g) for s in (0,1) for g in ('baseline','dt')}, 'Missing/duplicate seed pair')
        r = dict(task=task)
        for group in ('baseline','dt'):
            for metric in ('auc','tail'):
                v = {x['seed']:x[metric] for x in members if x['group'] == group}
                r[group+'_'+metric] = (v[0]+v[1])/2
        for metric in ('auc','tail'):
            r['delta_'+metric] = r['dt_'+metric]-r['baseline_'+metric]
            for seed in (0,1):
                pair = {x['group']:x[metric] for x in members if x['seed'] == seed}
                r[f'seed{seed}_delta_{metric}'] = pair['dt']-pair['baseline']
        result.append(r)
    return result


def analysis_tasks(keys, include_acrobot=False):
    count = 20 if include_acrobot else 19
    require(len(keys) == len(set(keys)) == 4*count+3, 'Wrong task/two-seed/DtL analysis scope')
    tasks = sorted({t for t,s,g in keys if g != 'dtl'})
    require(len(tasks) == count and ('dmc_acrobot_swingup' in tasks) == include_acrobot, 'Wrong acrobot scope')
    return tasks


def analyze(source, output, include_acrobot=False):
    output.mkdir(parents=True, exist_ok=False)
    runs = read(source/'seed0.json')
    for server in ('sv1','sv2','sv3'):
        runs += read(source/(server+'.json'))
    keys = [(r['row']['task'],r['row']['seed'],r['row']['group'].lower()) for r in runs]
    tasks = analysis_tasks(keys, include_acrobot)
    metrics, curves, phases, manifest = [], [], [], []
    for r in runs:
        row = r['row']; group = row['group'].lower(); c = r['manifest']['config']
        environment = row.get('environment') or row.get('environment_template', {})
        wandb_url = row.get('wandb_url')
        if not wandb_url:
            require(all(environment.get(k) for k in ('WANDB_ENTITY','WANDB_PROJECT','WANDB_RUN_ID')),
                    'Missing original W&B identity')
            wandb_url = ('https://wandb.ai/'+environment['WANDB_ENTITY']+'/'+
                         environment['WANDB_PROJECT']+'/runs/'+environment['WANDB_RUN_ID'])
        require(c['seed'] == row['seed'] and c['env.dmc.repeat'] == 1 and c['run.action_budget'] == 1000000,
                'Nonmatching scientific configuration')
        result = evaluation_metrics(r['evaluations'])
        logs = {m['step']:m for m in r['metrics'] if 'protocol/eval_return' in m}
        require(set(logs) == set(GRID), 'JSONL evaluation points differ')
        for step, mean in zip(GRID, result['curve']):
            require(math.isclose(mean, logs[step]['protocol/eval_return'], rel_tol=1e-12, abs_tol=1e-9), 'JSONL return differs')
            curves.append(dict(task=row['task'],group=group,seed=row['seed'],step=step,mean=mean))
        metrics.append(dict(task=row['task'],group=group,seed=row['seed'],auc=result['auc'],tail=result['tail']))
        phases += gate_rows(r)
        manifest.append(dict(task=row['task'],group=group,seed=row['seed'],server=r['server'],
            physical_gpu=row.get('physical_gpu'),directory=row['directory'],run_id=row['run_id'],
            training_sha=r['manifest']['git_commit'],state=row['state'],
            audit=row.get('run_audit',row.get('audit')),source_hashes=r['hashes'],
            wandb_url=wandb_url, config=c, gpu_uuid=row.get('gpu_uuid'),
            cuda_visible_devices=environment.get('CUDA_VISIBLE_DEVICES'),
            egl_device_id=environment.get('MUJOCO_EGL_DEVICE_ID'),
            source_index=r.get('index_path'), source_index_sha256=r.get('index_sha256'),
            prior_input=r.get('local_input'), prior_input_sha256=r.get('local_input_sha256'),
            source_kind='prior raw input mirror' if r.get('local_input') or r.get('local_directory') else '2026-11-06 original server collection'))
    pure = [r for r in metrics if r['group'] != 'dtl']
    paired = paired_means(pure, tasks)
    summaries, seed1, dtl = [], [], []
    for seed in (0,1):
        for task in tasks:
            members = {r['group']:r for r in pure if r['task'] == task and r['seed'] == seed}
            r = dict(task=task,seed=seed)
            for g in ('baseline','dt'):
                r.update({g+'_'+k:members[g][k] for k in ('auc','tail')})
            r.update({f'delta_{k}':r[f'dt_{k}']-r[f'baseline_{k}'] for k in ('auc','tail')})
            seed1.append(r)
    for label, members in [('seed0',[r for r in seed1 if r['seed']==0]),
                           ('seed1',[r for r in seed1 if r['seed']==1]),('mean_seed0_seed1',paired)]:
        r = dict(scope=label,tasks=len(tasks),training_seeds=2 if label.startswith('mean') else 1)
        for key in ('baseline_auc','dt_auc','delta_auc','baseline_tail','dt_tail','delta_tail'):
            r[key] = math.fsum(x[key] for x in members)/len(tasks)
        r['positive_auc_tasks'] = sum(x['delta_auc']>0 for x in members)
        r['positive_tail_tasks'] = sum(x['delta_tail']>0 for x in members)
        summaries.append(r)
    for task in DTL_TASKS:
        pair = {r['group']:r for r in metrics if r['task']==task and r['seed']==0}
        require(set(pair)=={'baseline','dt','dtl'}, 'DtL comparisons must use seed0 controls')
        r = dict(task=task,seed=0)
        for g in pair:
            r.update({g+'_'+k:pair[g][k] for k in ('auc','tail')})
        for control in ('baseline','dt'):
            r.update({f'dtl_minus_{control}_{k}':r['dtl_'+k]-r[control+'_'+k] for k in ('auc','tail')})
        dtl.append(r)
    gates = []
    for t,s,g in keys:
        parts = [r for r in phases if (r['task'],r['seed'],r['group']) == (t,s,g)]
        n = sum(r['updates'] for r in parts)
        r = dict(task=t,seed=s,group=g,updates=n)
        for k in ('D_mean','active_frac','active_weight_mean','actual_weight_mean','zero_D_fraction',
                  'active_protected_frac','valid_frac','support_mean','excess_release','weighted_excess'):
            r[k] = math.fsum(p[k]*p['updates'] for p in parts)/n if all(p.get(k) is not None for p in parts) else None
        r['excess_release_ratio'] = r['excess_release']/(r['excess_release']+r['weighted_excess'])
        r['active_protected_given_active'] = r['active_protected_frac']/r['active_frac'] if g=='dtl' else None
        if g=='dtl':
            r['min_weight'] = min(p['min_weight'] for p in parts)
        gates.append(r)
    config_diff = {}
    for task in tasks:
        base = next(r['manifest']['config'] for r in runs if r['row']['task']==task and r['row']['seed']==0 and r['row']['group']=='baseline')
        for r in runs:
            row=r['row']
            if row['task'] != task: continue
            c=r['manifest']['config'];config_diff[f"{task}/{row['seed']}/{row['group']}"] = {
                k:[base.get(k),c.get(k)] for k in base.keys()|c.keys() if base.get(k)!=c.get(k)}
    for name, rows in [('run_metrics',metrics),('seed_metrics',seed1),('two_seed_means',paired),
                       ('overall',summaries),('dtl_comparison',dtl),('learning_points',curves),
                       ('gate_intervals',phases),('gate_summary',gates),('run_manifest',manifest)]:
        dump(output/(name+'.json'),rows);csv_dump(output/(name+'.csv'),rows)
    dump(output/'config_differences.json',config_diff)
    dump(output/'verification.json',dict(input_hashes={p.name:sha(p) for p in source.glob('*.json')},
        run_count=len(runs),evaluation_points=len(curves),evaluation_episodes=len(curves)*10,
        old_seed0_inputs_reused=True,include_acrobot=include_acrobot,
        mean_basis=f'equal mean of seed0 and seed1 within task then equal mean of {len(tasks)} tasks',
        normalized_scores=False,significance_claims=False,missing_points_imputed=False))
    plots(output,tasks,curves,paired,dtl,phases)
    print(json.dumps(dict(overall=summaries,dtl=dtl),ensure_ascii=False,indent=2))


def plots(output,tasks,curves,paired,dtl,phases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'baseline':'#475569','dt':'#d97706','dtl':'#2563eb'}
    labels={'baseline':'baseline','dt':'PPD-KL','dtl':'DtL'}
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    for name,mean in [('seed1_learning_curves',False),('two_seed_mean_learning_curves',True)]:
        fig,axes=plt.subplots(5,4,figsize=(15,15),layout='constrained')
        for task,ax in zip(tasks,axes.flat):
            for group in ('baseline','dt'):
                for seed in ((0,1) if mean else (1,)):
                    members=[r for r in curves if (r['task'],r['group'],r['seed'])==(task,group,seed)]
                    if mean:ax.plot([r['step']/1e6 for r in members],[r['mean'] for r in members],color=colors[group],lw=.7,alpha=.23)
                members=[sum(r['mean'] for r in curves if r['task']==task and r['group']==group and r['step']==step and (mean or r['seed']==1))/(2 if mean else 1) for step in GRID]
                ax.plot([s/1e6 for s in GRID],members,color=colors[group],label=labels[group],lw=1.7)
            ax.set(title=task.removeprefix('dmc_'),xlabel='Training actions (millions)',ylabel='Raw return');ax.grid(alpha=.18)
        for ax in list(axes.flat)[len(tasks):]: ax.axis('off')
        axes.flat[0].legend()
        fig.suptitle('Equal mean of seed0 and seed1 (faint: individual seeds)' if mean else 'Seed1 | unsmoothed evaluation returns')
        fig.savefig(output/(name+'.png'),dpi=160);fig.savefig(output/(name+'.svg'));plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,8),layout='constrained')
    for ax,metric in zip(axes,('auc','tail')):
        for offset,(name,color) in enumerate([('seed0','#94a3b8'),('seed1','#f59e0b'),('mean','#2563eb')]):
            values=[r['delta_'+metric] if name=='mean' else r[name+'_delta_'+metric] for r in paired]
            ax.barh([i+(offset-1)*.24 for i in range(len(tasks))],values,height=.22,color=color,label=name)
        ax.set_yticks(range(len(tasks)),[t.removeprefix('dmc_') for t in tasks]);ax.invert_yaxis();ax.axvline(0,color='black',lw=.7)
        ax.set(title='PPD-KL minus baseline: '+('AUC / 1M' if metric=='auc' else 'last-five mean'),xlabel='Raw return difference');ax.grid(axis='x',alpha=.2)
    axes[0].legend();fig.savefig(output/'seed_differences.png',dpi=170);fig.savefig(output/'seed_differences.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,task in zip(axes,DTL_TASKS):
        for group in ('baseline','dt','dtl'):
            members=[r for r in curves if (r['task'],r['seed'],r['group'])==(task,0,group)]
            ax.plot([r['step']/1e6 for r in members],[r['mean'] for r in members],label=labels[group],color=colors[group])
        ax.set(title=task.removeprefix('dmc_'),xlabel='Training actions (millions)',ylabel='Raw return');ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle('Seed0 | baseline, PPD-KL and DtL | unsmoothed evaluation returns')
    fig.savefig(output/'dtl_learning_curves.png',dpi=180);fig.savefig(output/'dtl_learning_curves.svg');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for column,task in enumerate(DTL_TASKS):
        for group in ('baseline','dt','dtl'):
            members=[r for r in phases if (r['task'],r['seed'],r['group'])==(task,0,group)]
            for ax,key in [(axes[0,column],'actual_weight_mean'),(axes[1,column],'excess_release_ratio')]:
                ax.plot([r['end_action']/1e6 for r in members],[r[key] for r in members],color=colors[group],label=labels[group]);ax.grid(alpha=.2)
        axes[0,column].set(title=task.removeprefix('dmc_'),ylabel='Actual weight (all positions)')
        axes[1,column].set(xlabel='Training actions (millions)',ylabel='Excess KL release fraction')
    axes[0,0].legend();fig.suptitle('Seed0 | 50k intervals; weights and loss diagnostics, not gradient norms')
    fig.savefig(output/'dtl_gate_diagnostics.png',dpi=180);fig.savefig(output/'dtl_gate_diagnostics.svg');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('collect','analyze'))
    parser.add_argument('--inventory',type=Path)
    parser.add_argument('--input',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--include-acrobot',action='store_true')
    args=parser.parse_args()
    if args.mode=='collect':collect(args.inventory,args.output,args.include_acrobot)
    else:analyze(args.input,args.output,args.include_acrobot)


if __name__=='__main__':main()
