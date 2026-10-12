"""Recompute frozen seed0 M2 metrics, receipts, diagnostics and figures from a mirror.

No training imports, GPU use, normalization, missing-point interpolation or W&B writes.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from analysis import return_metrics as rm

AUDIT_REL = 'data/Policy_Discrepancy/repo/analysis/outputs/formal-four-groups-pd1-20261010-final-attempt01'
FIELDS = ('D_mean', 'D_p10', 'D_p50', 'D_p90', 'w_mean',
          'actual_weight_mean', 'rep_active_frac', 'active_weight_available',
          'active_weight_mean', 'source_active_weight_mean',
          'weighted_excess_mean', 'excess_reduction_mean', 'rep_before_mean',
          'rep_after_mean', 'actual_weight_raw_cov', 'source_weight_raw_cov',
          'shuffle_moved_frac', 'mode_vector_diff_frac', 'mean_abs_diff',
          'std_abs_diff', 'raw_active_D_near_zero_frac', 'active_effective_frac')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as out:
        writer = csv.DictWriter(out, keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict))
                             else v for k, v in row.items()})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def receipts(path, final, evaluations):
    """Stream all successful receipts, retaining normal replay resampling."""
    count = 0
    window_sum, window_count, included = 0.0, 0, []
    intervals = [dict(updates=0, active_sum=0, source_weight_sum=0.0)
                 for _ in range(20)]
    limits = [row['update_id'] for row in evaluations[1:]]
    interval = 0
    last_action = -1
    first_action = None
    with path.open(encoding='utf-8') as file:
        for line in file:
            r = json.loads(line)
            require(r['update_id'] == count, 'Missing/duplicate/conflicting receipt ID')
            s, n, w = r['start_action'], r['match_count'], r['match_sum']
            require(isinstance(s, int) and last_action <= s <= 1000000,
                    'Nonmonotonic/out-of-budget update stamp')
            require(r['invalid_count'] == 0 and isinstance(n, int) and 0 <= n <= 1024
                    and math.isfinite(w) and 0 <= w <= n + 1e-4, 'Invalid matching receipt')
            if first_action is None:
                first_action = s
            while count >= limits[interval]:
                interval += 1
                require(interval < 20, 'Receipts after final evaluation')
            require(50000*interval < s <= 50000*(interval+1),
                    'Update action stamp disagrees with snapshot interval')
            b = intervals[interval]
            b['updates'] += 1
            b['active_sum'] += n
            b['source_weight_sum'] += w
            if 100000 <= s < 300000:
                window_sum += w
                window_count += n
                included.append(count)
            last_action = s
            count += 1
    require(count == final['updates'] == final['train_batch_keys'], 'Final receipt/key count mismatch')
    require(window_sum == final['match_sum'] and window_count == final['match_count'],
            'Final matching S/N mismatch')
    require(count == evaluations[-1]['update_id'] and evaluations[0]['update_id'] == 0,
            'Initial/final snapshot update mismatch')
    return dict(updates=count, first_update_start_action=first_action,
                match_sum=window_sum, match_count=window_count, included_update_ids=included,
                intervals=intervals)


def diagnostics(directory, meta, receipt, evaluations):
    lines = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
    bins = [r for r in lines if 'train/dt/D_mean' in r]
    require([r['step'] for r in bins] == list(range(50000, 1000001, 50000)),
            'Missing/duplicate diagnostic interval')
    logged_eval = [r for r in lines if 'protocol/eval_return' in r]
    require([r['step'] for r in logged_eval] == list(rm.MetricSpec().grid), 'JSONL eval grid mismatch')
    for log, ev in zip(logged_eval, evaluations):
        require(log['protocol/eval_return'] == ev['mean'] and
                log['protocol/updates'] == ev['update_id'], 'JSONL evaluation differs')
    result = []
    for row, counts in zip(bins, receipt['intervals']):
        r = dict(task=meta['task'], group=meta['group'], seed=0,
                 run_id=meta['run_id'], start_action=row['step']-50000,
                 end_action=row['step'], updates=counts['updates'])
        for field in FIELDS:
            val = row.get('train/dt/' + field)
            r[field] = val if val is not None and math.isfinite(val) else None
        require(all(r[f] is not None for f in FIELDS), 'Missing/nonfinite diagnostic field')
        require(row['train/dt/invalid_D_frac'] == 0 and row['train/dt/match_invalid_count'] == 0,
                'Invalid gate diagnostics')
        require(r['active_weight_available'] == 1, 'Undefined active mean was skipped by logger')
        # Mean counters divided by constant positions is the pooled exact-zero proportion.
        # Stored scalar means can have final float64 aggregation roundoff.
        positions = row['train/dt/loss_position_count']
        require(positions == 1024, 'Unexpected effective batch/time count')
        r['D_zero_frac'] = row['train/dt/D_zero_count'] / positions
        # Only logging baseline receipts persist S/N. Other groups store zero
        # matching fields by design; use their actual diagnostic scalar means.
        r['pooled_active_frac'] = row['train/dt/match_count'] / positions
        r['pooled_source_active_weight'] = row['train/dt/match_sum'] / row['train/dt/match_count']
        r['pooled_basis'] = 'logged scalar means'
        require(math.isclose(r['rep_active_frac'], r['pooled_active_frac'], abs_tol=1e-9, rel_tol=1e-12),
                'Logged active fraction/count differs')
        if meta['group'] == 'baseline':
            receipt_frac = counts['active_sum'] / (positions * counts['updates'])
            require(math.isclose(r['rep_active_frac'], receipt_frac, abs_tol=1e-9, rel_tol=1e-12),
                    'Baseline diagnostic/receipt active fraction differs')
            require(math.isclose(r['pooled_source_active_weight'],
                                counts['source_weight_sum']/counts['active_sum'],
                                abs_tol=1e-9, rel_tol=1e-12), 'Baseline diagnostic/receipt weight differs')
            r['pooled_basis'] = 'logged means crosschecked against full baseline receipts'
        expected_update_mean = (evaluations[len(result)]['update_id'] +
                                evaluations[len(result)+1]['update_id'] + 1) / 2
        require(row['train/opt/updates'] == expected_update_mean, 'Diagnostic update aggregation mismatch')
        denom = r['weighted_excess_mean'] + r['excess_reduction_mean']
        r['excess_reduction_ratio'] = r['excess_reduction_mean'] / denom if denom > 0 else None
        r['active_D_near_zero_conditional'] = r['raw_active_D_near_zero_frac'] / r['rep_active_frac']
        r['rep_value_reduction_ratio'] = 1-r['rep_after_mean']/r['rep_before_mean']
        if meta['group'] == 'baseline':
            require(row['train/dt/actual_weight_mean'] == 1 and r['excess_reduction_mean'] == 0,
                    'Baseline is not logging-only')
        if meta['group'] == 'constant':
            # The forward pass is float32; authoritative c remains the frozen double value.
            import struct
            c32 = struct.unpack('f', struct.pack('f', meta['c']))[0]
            require(row['train/dt/actual_weight_mean'] == c32, 'Constant weight does not equal frozen float32 c')
        result.append(r)
    return result


def summarize_diag(rows):
    result = []
    for task in rm.TASKS:
        for group in rm.GROUPS:
            members = [r for r in rows if r['task'] == task and r['group'] == group]
            for stage, lo, hi in (('full', 0, 1000000), ('early', 0, 100000),
                                  ('match', 100000, 300000), ('middle', 300000, 800000),
                                  ('late', 800000, 1000000)):
                b = [r for r in members if lo <= r['start_action'] and r['end_action'] <= hi]
                total = sum(r['updates'] for r in b)
                out = dict(task=task, group=group, seed=0, stage=stage,
                           start_action=lo, end_action=hi, updates=total)
                for field in FIELDS + ('D_zero_frac', 'pooled_active_frac',
                                      'active_D_near_zero_conditional', 'rep_value_reduction_ratio'):
                    out[field] = math.fsum(r[field]*r['updates'] for r in b)/total
                # Ratios of accumulated means, rather than mean of interval ratios.
                e = out['weighted_excess_mean'] + out['excess_reduction_mean']
                out['excess_reduction_ratio'] = out['excess_reduction_mean']/e if e else None
                out['active_D_near_zero_conditional'] = out['raw_active_D_near_zero_frac']/out['rep_active_frac']
                active = math.fsum(r['pooled_active_frac']*r['updates'] for r in b)
                out['pooled_source_active_weight'] = math.fsum(
                    r['pooled_source_active_weight']*r['pooled_active_frac']*r['updates'] for r in b)/active
                result.append(out)
    return result


def plots(output, tables, diagnostic_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    colors = dict(baseline='#4b5563', dt='#c2410c', constant='#047857', shuffle='#2563eb')
    labels = dict(baseline='Baseline (v=1)', dt='Dt (alpha=20)', constant='Frozen constant', shuffle='Global shuffle')
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for task, ax in zip(rm.TASKS, axs.flat):
        for group in rm.GROUPS:
            points = [r for r in tables['evaluation_points'] if r['task'] == task and r['group'] == group]
            ax.plot([p['actual_snapshot_action_step']/1e6 for p in points],
                    [p['mean_return_raw'] for p in points], marker='o', ms=3,
                    lw=1.7, color=colors[group], label=labels[group])
        ax.set(title=task.removeprefix('dmc_'), xlabel='Training actions (millions)', ylabel='Raw episode return')
        ax.grid(alpha=.18)
    axs[0, 0].legend(fontsize=9, loc='upper left')
    fig.suptitle('M2 v1 | training seed0 | 21 unsmoothed points, 10 episodes per snapshot', fontsize=13)
    fig.savefig(output/'learning_curves.png', dpi=180)
    fig.savefig(output/'learning_curves.svg')
    plt.close(fig)
    fig, axs = plt.subplots(4, 4, figsize=(15, 11), constrained_layout=True)
    fields = (('D_mean', 'Mean Dt (log scale)'), ('D_zero_frac', 'Exact D=0 fraction'),
              ('rep_active_frac', 'Free-nats active fraction'), ('active_weight_mean', 'Actual active weight'))
    for i, task in enumerate(rm.TASKS):
        for j, (field, label) in enumerate(fields):
            ax = axs[i, j]
            for group in rm.GROUPS:
                b = [r for r in diagnostic_rows if r['task'] == task and r['group'] == group]
                ax.plot([r['end_action']/1e6 for r in b], [r[field] for r in b],
                        color=colors[group], label=labels[group], lw=1.5)
            if j == 0:
                ax.set_yscale('log')
                ax.set_ylabel(task.removeprefix('dmc_'))
            else:
                ax.set_ylim(-.03, 1.03)
            if i == 0:
                ax.set_title(label)
            ax.grid(alpha=.18)
            if i == 3:
                ax.set_xlabel('Training actions (millions)')
    axs[0, 0].legend(fontsize=8)
    fig.suptitle('Seed0 gate diagnostics | update means within each 50k-action interval', fontsize=14)
    fig.savefig(output/'gate_diagnostics.png', dpi=170)
    fig.savefig(output/'gate_diagnostics.svg')
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, metric, title in zip(axs, ('auc_return_per_budget', 'tail_mean_return'),
                                ('Primary: raw return AUC / 1M', 'Secondary: mean of five 800k-1M points')):
        x = list(range(4))
        for j, ref in enumerate(('baseline', 'constant', 'shuffle')):
            vals = [next(r[metric] for r in tables['dt_differences'] if r['task']==task and
                         r['comparison']=='dt-'+ref) for task in rm.TASKS]
            ax.bar([a+(j-1)*.24 for a in x], vals, width=.23, color=colors[ref], label='Dt - '+ref)
        ax.axhline(0, color='black', lw=.7)
        ax.set_xticks(x, ('hopper', 'quad run', 'quad walk', 'reacher'))
        ax.set_title(title)
        ax.set_ylabel('Raw return difference')
        ax.legend(fontsize=9)
    fig.suptitle('M2 v1 | single training seed0 | descriptive differences')
    fig.savefig(output/'dt_differences.png', dpi=180)
    fig.savefig(output/'dt_differences.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs, output = args.inputs.resolve(), args.output.resolve()
    proof = read(inputs/'COLLECTION.json')
    for rel, evidence in proof['files'].items():
        require(rm.sha(inputs/rel) == evidence['sha256'], 'Mirror hash mismatch: '+rel)
    index = read(inputs/AUDIT_REL/'RUN_INDEX.json')
    completion = read(inputs/AUDIT_REL/'completion.json')
    require(completion['verified_run_count'] == 16 and completion['full_four_groups_complete'],
            '04 final audit is not complete')
    require(rm.sha(inputs/AUDIT_REL/'return-audit/tables.json') == completion['return_tables_sha256'],
            '04 table hash mismatch')
    require(all(v['matches_deployed'] and v['sha256'] == completion['source_sha256']
                for v in proof['audit_sources'].values()), 'Deployed audit source unidentified')
    require(len(index['runs']) == 16 and len({(r['task'],r['group'],r['seed']) for r in index['runs']})==16,
            'Run matrix conflict')
    source04 = read(inputs/AUDIT_REL/'return-audit/audit.json')
    for audit in source04['runs']:
        for path, digest in audit['input_sha256'].items():
            rel = path.lstrip('/')
            require(rel in proof['files'] and rm.sha(inputs/rel)==digest, '04 audited source changed: '+path)
    local_index = []
    run_checks, frozen, diagnostic_rows = [], [], []
    configs, random_evidence, initial_evaluations = {}, {}, {}
    initial_comparisons = []
    for entry in index['runs']:
        d = inputs/entry['directory'].lstrip('/')
        manifest = read(d/'protocol_manifest.json')
        config = manifest['config']
        final = read(d/'final_state.json')
        require(manifest['git_commit'] == entry['code_sha'] == final['git_commit'], 'Training SHA conflict')
        require(config == entry['config'], 'Config/index mismatch')
        fixed = {'seed':0, 'run.action_budget':1000000, 'run.eval_eps':10,
                 'run.eval_every_actions':50000, 'run.match_start':100000,'run.match_end':300000,
                 'run.engineering_fixture':False, 'env.dmc.image':True,'env.dmc.proprio':False,
                 'env.dmc.repeat':1, 'agent.rep_probe.alpha':20.0,
                 'agent.dyn.rssm.free_nats':1.0,'agent.loss_scales.dyn':1.0,
                 'agent.loss_scales.rep':0.1,'agent.ac_grads':False,
                 'agent.reward_grad':True,'agent.repval_grad':True,
                 'jax.prealloc':False,'logger.outputs':['jsonl','wandb'],
                 'batch_size':16,'batch_length':64,'run.from_checkpoint':''}
        require(all(config[k]==v for k,v in fixed.items()), 'Frozen config mismatch')
        require(entry['environment']['WANDB_PROJECT']=='PD_1', 'W&B project mismatch')
        common={k:v for k,v in config.items() if k not in ('logdir','agent.rep_probe.mode',
                 'agent.rep_probe.c','run.frozen_c_file')}
        if entry['task'] in configs:
            require(common==configs[entry['task']], 'Undeclared cross-group config difference')
        configs[entry['task']]=common
        require(final['train_action_steps']==1000000 and final['evaluations']==21,
                'Final budget/evaluation mismatch')
        evs=[json.loads(line) for line in (d/'evaluations.jsonl').read_text().splitlines()]
        require([r['action_step'] for r in evs]==list(rm.MetricSpec().grid),'Evaluation grid conflict')
        random_fields = ('eval_policy','evaluation','replay_seed','train_env_seeds',
                         'train_policy','train_update','report','shuffle','packages',
                         'dependency_lock_sha256','device_kinds','python','jax')
        streams = dict(manifest={k:manifest[k] for k in random_fields},
                       eval_seeds=[r['seeds'] for r in evs])
        if entry['task'] in random_evidence:
            require(streams == random_evidence[entry['task']], 'Cross-group RNG/environment declaration differs')
        random_evidence[entry['task']] = streams
        if entry['task'] not in initial_evaluations:
            initial_evaluations[entry['task']] = evs[0]
        base = initial_evaluations[entry['task']]
        delta = [a-b for a,b in zip(evs[0]['returns_raw'],base['returns_raw'])]
        initial_comparisons.append(dict(task=entry['task'],group=entry['group'],
            equal_returns=evs[0]['returns_raw']==base['returns_raw'],
            equal_lengths=evs[0]['lengths']==base['lengths'],
            mean_return=math.fsum(evs[0]['returns_raw'])/10,
            mean_difference_from_baseline=math.fsum(delta)/10,
            max_abs_episode_difference=max(map(abs,delta)),
            interpretation='Observed difference retained; exact trajectory reproducibility is not a frozen v1 requirement.'))
        require(sum(sum(r['lengths']) for r in evs)==final['eval_action_steps'], 'Evaluation action counter mismatch')
        receipt=receipts(d/'update_receipts.jsonl',final,evs)
        local_index.append(dict(entry, directory=str(d), source_directory=entry['directory'],
                                snapshot_audit_file=str(inputs/'COLLECTION.json')))
        verified=next(r for r in completion['runs'] if r['run_id']==entry['run_id'])
        run_checks.append(dict(task=entry['task'],group=entry['group'],seed=entry['seed'],
            run_id=entry['run_id'],attempt_id=d.name,training_sha=manifest['git_commit'],
            audit_sha=next(iter(proof['audit_sources'])),audit_source_sha256=completion['source_sha256'],
            server=entry['server'],physical_gpu=entry['physical_gpu'],cuda_uuid=entry['uuid'],
            cuda_logical_device=entry['cuda_logical_device'],egl_device_id=entry['egl_device_id'],
            source_directory=entry['directory'],local_directory=str(d),wandb_url=entry['wandb_url'],
            original_index_state=entry['state'],original_exit_code=entry.get('exit_code'),
            final_audit_status='accepted',audit_time=completion['checked_utc'],
            train_actions=final['train_action_steps'],train_resets=final['train_resets'],
            eval_actions=final['eval_action_steps'],updates=receipt['updates'],
            first_update_start_action=receipt['first_update_start_action'],
            receipt_duplicate_count=0,evaluation_points=len(evs),episodes=sum(len(r['episodes']) for r in evs),
            config_sha256=rm.sha(d/'config.yaml'),manifest_sha256=rm.sha(d/'protocol_manifest.json'),
            evaluation_sha256=rm.sha(d/'evaluations.jsonl'),receipt_sha256=rm.sha(d/'update_receipts.jsonl'),
            params_sha256=final['params_sha256'],environment=manifest,
            wandb_exact_transport_verified=verified['exact_transport_verified'],
            numeric_display_differences=verified['numeric_differences']))
        if entry['group']=='baseline':
            c=read(d/'c_frozen.json')
            require(c['included_update_ids']==receipt['included_update_ids'] and
                    c['match_sum']==receipt['match_sum'] and c['match_count']==receipt['match_count'] and
                    c['c']==receipt['match_sum']/receipt['match_count'], 'Frozen c/S/N/ID mismatch')
            require(c['receipt_sha256']==rm.sha(d/'update_receipts.jsonl') and
                    c['baseline_dir']==entry['directory'] and c['baseline_git_commit']==manifest['git_commit']
                    and c['baseline_config_sha256']==hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
                    and c['window']==[100000,300000] and c['alpha']==20 and c['root_seed']==0 and
                    c['raw_rep_threshold']==1 and not c['engineering_fixture'] and c['invalid_count']==0,
                    'Frozen c source metadata mismatch')
            require(rm.sha(d/'c_frozen.json')==(d/'c_frozen.json.sha256').read_text().strip(),'Frozen c hash mismatch')
            frozen.append(dict(task=entry['task'],c=c['c'],S=c['match_sum'],N=c['match_count'],
                matching_updates=len(c['included_update_ids']),baseline_run_id=entry['run_id'],
                baseline_sha=c['baseline_git_commit'],frozen_at=c['frozen_at_utc'],
                receipt_sha256=c['receipt_sha256'],artifact_sha256=rm.sha(d/'c_frozen.json')))
        if entry['group']=='constant':
            c=read(inputs/config['run.frozen_c_file'].lstrip('/'))
            require(config['agent.rep_probe.c']==c['c'] and c['task']==entry['task'], 'Wrong constant source')
            require(datetime.fromisoformat(c['frozen_at_utc']) < datetime.fromisoformat(entry['start_utc']),
                    'Constant started before freeze')
        diagnostic_rows.extend(diagnostics(d,dict(entry,c=config['agent.rep_probe.c']),receipt,evs))
    dump(output/'RUN_INDEX.json',dict(runs=local_index))
    command=[sys.executable,'-m','analysis.return_metrics','--index',str(output/'RUN_INDEX.json'),
             '--output',str(output/'return-audit'),'--require-complete']
    subprocess.run(command,check=True)
    tables=read(output/'return-audit/tables.json')
    old=read(inputs/AUDIT_REL/'return-audit/tables.json')
    for name in ('task_metrics','overall_metrics','dt_differences'):
        require(tables[name]==old[name], 'Independent metric/04 table disagreement: '+name)
    # Second arithmetic implementation uses the 21 episode arrays directly.
    for row in tables['seed_metrics']:
        entry=next(r for r in local_index if r['run_id']==row['run_id'])
        evs=[json.loads(line) for line in (Path(entry['directory'])/'evaluations.jsonl').read_text().splitlines()]
        means=[math.fsum(r['returns_raw'])/10 for r in evs]
        independent_auc=(means[0]/2+math.fsum(means[1:-1])+means[-1]/2)/20
        require(math.isclose(independent_auc,row['auc_return_per_budget'],abs_tol=1e-9,rel_tol=1e-12),
                'Direct 21-point arithmetic disagreement')
        require(math.fsum(means[16:])/5==row['tail_mean_return'],'Direct tail arithmetic disagreement')
    dump(output/'run_manifest.json',run_checks)
    csv_write(output/'run_manifest.csv',[{k:v for k,v in row.items() if k!='environment'} for row in run_checks])
    dump(output/'frozen_c.json',frozen)
    dump(output/'initial_evaluation_comparison.json',initial_comparisons)
    csv_write(output/'frozen_c.csv',frozen)
    csv_write(output/'gate_intervals.csv',diagnostic_rows)
    summary=summarize_diag(diagnostic_rows)
    dump(output/'gate_summary.json',summary)
    csv_write(output/'gate_summary.csv',summary)
    plots(output,tables,diagnostic_rows)
    repo=Path(__file__).resolve().parents[1]
    scripts={name:rm.sha(repo/name) for name in ('analysis/analyze_m2_v1.py',
             'analysis/collect_m2_v1.py','analysis/return_metrics.py','dreamerv3/return_stats.py')}
    validation=dict(all_16_accepted=True,runs=16,points=336,episodes=3360,
        unique_successful_updates=sum(r['updates'] for r in run_checks),
        formal_metrics_complete=True,exact_04_metric_tables_match=True,
        direct_arithmetic_crosscheck=True,input_files=len(proof['files']),all_input_hashes_match=True,
        training_shas=sorted({r['training_sha'] for r in run_checks}),
        audit_shas=sorted({r['audit_sha'] for r in run_checks}),
        analysis_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        analysis_script_sha256=scripts,collection_sha256=rm.sha(inputs/'COLLECTION.json'),
        snapshot_basis='remote metadata, completion markers and agent.pkl presence; reused 04 complete audit; no pickle reread',
        reward_basis='schema v2 episode returns and source-reviewed non-reset reward accumulation; no transition reward traces',
        diagnostic_basis='50k action intervals, exact update counts from snapshot IDs; update-weighted scalar means; batch quantiles averaged',
        wandb_current=proof.get('wandb_readonly_crosscheck',proof.get('wandb_readonly_crosscheck_error')))
    dump(output/'validation.json',validation)
    dump(output/'output_hashes.json',{str(p.relative_to(output)):rm.sha(p)
                                    for p in output.rglob('*') if p.is_file()})
    print(json.dumps(dict(accepted=16,output=str(output),overall=tables['overall_metrics'])))


if __name__=='__main__':
    main()
