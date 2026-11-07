"""Read-only decomposition of the audited clean seed0 DtL scalar logs.

The no-e calculation is a fixed-model, fixed-data loss counterfactual, not a
training experiment or a parameter recommendation. No agent/JAX/SSH imports.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess


TASKS = ('dmc_quadruped_run', 'dmc_quadruped_walk', 'dmc_finger_spin')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return Path(path).read_text(encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False)+'\n', encoding='utf-8')


def diagnostic(run):
    config = run['manifest']['config']
    require(config['seed'] == 0 and config['agent.dt_latch.mode'] == 'dtlatch'
            and config['agent.dt_latch.alpha'] == 20 and config['agent.dt_latch.rho'] == .5
            and config['agent.dt_latch.normal_only'] is False, 'Different DtL configuration')
    logs = [r for r in run['metrics'] if 'train/dl/actual_v_mean' in r]
    receipts = run['dtl_receipt_intervals']
    require([r['step'] for r in logs] == list(range(50000, 1000001, 50000)), 'Incomplete diagnostic grid')
    require(len(receipts) == 20 and sum(r['updates'] for r in receipts) == 249493,
            'Incomplete audited receipt intervals')
    size = config['batch_size'] * config['batch_length']
    require(size == 1024, 'Different loss-position denominator')
    rows = []
    for log, interval in zip(logs, receipts):
        values = {k.removeprefix('train/dl/'):v for k,v in log.items() if k.startswith('train/dl/')}
        require(all(math.isfinite(v) for v in values.values()), 'Nonfinite log scalar')
        rows.append(dict(task=run['row']['task'], step=log['step'], updates=interval['updates'], **values))

    def summarize(parts):
        total = sum(p['updates'] for p in parts)
        means = {k:math.fsum(p[k]*p['updates'] for p in parts)/total
                 for k in rows[0] if k not in ('task','step','updates')}
        excess = means['excess_release_mean'] + means['weighted_excess_mean']
        require(excess > 0 and means['calibration_den'] > 0, 'Undefined loss counterfactual')
        rho = config['agent.dt_latch.rho']
        # calibration_den = sum(m * max(KL-tau, 0) * f_D), per update.
        no_e_release = rho * means['calibration_den'] / size
        from_num = rho * means['calibration_num'] / size
        require(0 <= means['calibration_num'] <= means['calibration_den'], 'Support outside [0,1]')
        require(0 <= no_e_release <= excess, 'Invalid counterfactual release')
        # At any alpha, 1-v <= rho*m*e. At each update mean(m*e)
        # <= mean(e | valid). This uses the logged conditional means as an
        # upper bound; it does not multiply separately averaged m and e.
        alpha_upper = rho * means['e_valid_mean']
        actual_drop = 1-means['actual_v_mean']
        require(0 <= actual_drop <= alpha_upper, 'Observed weight violates conservative support bound')
        return dict(updates=total, mean_u_q=means['u_q_valid_mean'], mean_u_p=means['u_p_valid_mean'],
            mean_e=means['e_valid_mean'], mean_D_valid=means['D_valid_mean'],
            mean_actual_weight=means['actual_v_mean'], mean_actual_weight_drop=actual_drop,
            valid_fraction=means['valid_frac'], active_fraction=means['active_frac'],
            actual_excess_release_fraction=means['excess_release_mean']/excess,
            same_data_no_e_excess_release_fraction=no_e_release/excess,
            support_weighted_by_excess_and_D=means['calibration_num']/means['calibration_den'],
            alpha_unbounded_mean_weight_drop_upper=alpha_upper,
            observed_release=means['excess_release_mean'], reconstructed_release_from_num=from_num,
            reconstruction_absolute_residual=abs(from_num-means['excess_release_mean']),
            reward_outside_bins_count_mean=means['reward_outside_bins_count'])

    aggregate = dict(task=run['row']['task'], **summarize(rows))
    phases = [dict(task=run['row']['task'], end_action=p['step'], **summarize([p])) for p in rows]
    return aggregate, phases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    originals = json.loads(read(args.source/'artifact_hashes.json'))['files']
    names = ('input20/sv2.json', 'result20-final/dtl_comparison.json',
             'result20-final/gate_summary.json', 'result20-final/learning_points.json')
    for name in names:
        require(sha(args.source/name) == originals[name], 'Audited source changed: '+name)
    runs = [r for r in json.loads(read(args.source/names[0])) if r['row']['group'].lower() == 'dtl']
    require(len(runs) == 3 and {r['row']['task'] for r in runs} == set(TASKS), 'Wrong three-task scope')
    summary, phases = [], []
    for task in TASKS:
        r = next(r for r in runs if r['row']['task'] == task)
        s, p = diagnostic(r)
        summary.append(s); phases += p
    # Hash actual training source, rather than assume current HEAD is identical.
    git_sources = {}
    for name in ('dreamerv3/dt_latch.py', 'dreamerv3/dtl_agent.py'):
        commit = runs[0]['manifest']['git_commit']
        require(all(r['manifest']['git_commit'] == commit for r in runs), 'Different DtL training source')
        blob = subprocess.check_output(['git','show',commit+':'+name])
        git_sources[name] = dict(training_sha=commit, sha256=hashlib.sha256(blob).hexdigest(),
            current_source_lf_equal=blob == Path(name).read_bytes().replace(b'\r\n',b'\n'))
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'diagnostics.json', dict(summary=summary, phases=phases))
    write_json(args.output/'sources.json', dict(source_directory=str(args.source),
        input_hashes={name:sha(args.source/name) for name in names},training_sources=git_sources,
        script_sha256=sha(__file__),git_head=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        inference='No-e and unbounded-alpha values describe only the recorded model/data, not retrained returns',
        diagnostic_basis='Update-weighted scalar means; no per-position quantiles or gradient-norm counterfactual',
        reconstruction_note='The recorded float32 weight subtraction and separate calibration sums differ by roundoff; residuals are reported, not forced to zero'))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
