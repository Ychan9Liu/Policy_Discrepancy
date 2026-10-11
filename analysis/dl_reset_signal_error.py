"""DL exploratory error decomposition of authenticated, frozen reset outputs."""
import argparse
import datetime
import hashlib
import json
import math
import pickle
import signal
import subprocess
import sys
import time
from pathlib import Path

VERSION = 'DL-reset-signal-error-r1'
PRIMARY = Path('/data/Policy_Discrepancy/runs/dl-reset-joint-20261012-c030e26/primary-attempt02')
AUDIT = PRIMARY.parent / 'independent-audit03/audit.json'
PINS = {
    'result.json': 'e8f56b834d85f94b22a293c96bb5c52923f23317f8eada3a914253302cfde037',
    'freeze.json': '3901903f41c8694bd9c9047e7764eb129706fb40473a2da8475fce3bb50e2afb',
    'audit.json': 'ba4c7d9c5d47baeb83f48eb2cde8eca5637186c1abe6ae42a961d13fe4fc8a22',
}
NAMES = ('D-only', 'D+K', 'D+S', 'S-only')


def avg(x):
    return math.fsum(x) / len(x)


def cov(x, y):
    a, b = avg(x), avg(y)
    return avg([(v-a)*(w-b) for v, w in zip(x, y)])


def scalar(x):
    return dict(mean=avg(x), minimum=min(x), maximum=max(x), variance=cov(x, x))


def errors(g, u):
    e = [x-y for x, y in zip(g, u)]
    vg, vu, c = cov(g, g), cov(u, u), cov(g, u)
    mse = avg([v*v for v in e])
    bias = avg(e)
    assert abs(mse-(bias*bias+vg+vu-2*c)) < 1e-12
    return dict(mse=mse, bias=bias, bias_squared=bias*bias,
                prediction_variance=vg, target_variance=vu, covariance=c,
                correlation=c/math.sqrt(vg*vu) if vg*vu > 0 else None)


def displacement(g, d, u):
    r = [y-x for y, x in zip(u, d)]
    v = [x-y for x, y in zip(g, d)]
    align = 2*avg([x*y for x, y in zip(r, v)])
    cost = avg([x*x for x in v])
    direct = errors(d, u)['mse']-errors(g, u)['mse']
    mean_part = 2*avg(r)*avg(v)-avg(v)**2
    centered = 2*cov(r, v)-cov(v, v)
    assert abs(direct-(align-cost)) < 1e-12
    assert abs(direct-(mean_part+centered)) < 1e-12
    return dict(increment=direct, residual_alignment_term=align,
                displacement_squared_cost=cost, mean_component=mean_part,
                centered_component=centered, centered_alignment_term=2*cov(r,v),
                displacement_mean=avg(v), residual_mean=avg(r),
                per_episode_increment=[a*a-(a-b)**2 for a,b in zip(r,v)])


def cohort(x):
    n = len(x['U'])
    assert n in (8, 24) and len(set(x['episode_ids'])) == n
    assert all(len(x[k]) == n for k in ('D', 'K', 'S', 'choices'))
    assert all(math.isfinite(v) for k in ('D','K','S','U') for v in x[k])
    assert all(len(x['predicted_U'][k]) == n for k in NAMES)
    return dict(count=n, scalars={k:scalar(x[k]) for k in ('D','K','S','U')},
                S_to_U=errors(x['S'],x['U']),
                regressor_errors={k:errors(x['predicted_U'][k],x['U']) for k in NAMES},
                increments={k:displacement(x['predicted_U'][k],x['predicted_U']['D-only'],x['U'])
                            for k in NAMES if k != 'D-only'},
                practical_positive=sum(v >= .2 for v in x['U']),
                nonpositive=sum(v <= 0 for v in x['U']),
                prior_choices=[v[1] for v in x['choices']],
                rows=[dict(episode=e,D=x['D'][i],K=x['K'][i],S=x['S'][i],U=x['U'][i],
                           choices=x['choices'][i],S_minus_U=x['S'][i]-x['U'][i],
                           predictions={k:x['predicted_U'][k][i] for k in NAMES})
                      for i,e in enumerate(x['episode_ids'])])


def read_pinned(path, expected):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected, str(path)
    return raw


def main():
    start = time.monotonic()
    started_at = datetime.datetime.now().astimezone().isoformat()
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected-git', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('120s CPU limit')))
        signal.alarm(120)
    sha = subprocess.check_output(['git','rev-parse','HEAD'], text=True, timeout=5).strip()
    assert sha == args.expected_git
    assert not subprocess.check_output(['git','status','--porcelain'], text=True, timeout=5)
    args.output.mkdir(parents=True, exist_ok=False)
    result = json.loads(read_pinned(PRIMARY/'result.json', PINS['result.json']))
    freeze = json.loads(read_pinned(PRIMARY/'freeze.json', PINS['freeze.json']))
    audit = json.loads(read_pinned(AUDIT, PINS['audit.json']))
    assert audit['status'] == 'passed' and freeze['lambda_value'] == .01
    pins = {k:PINS[k] for k in ('result.json','freeze.json','audit.json')}
    pins['readouts.pkl'] = freeze['files']['readouts.pkl']
    pins['blind-numeric.pkl'] = result['blind_numeric_sha256']
    assert all(audit['primary_json_and_generated_sha256'][k] == pins[k]
               for k in ('readouts.pkl','blind-numeric.pkl'))
    readouts = pickle.loads(read_pinned(PRIMARY/'readouts.pkl',pins['readouts.pkl']))
    blind = pickle.loads(read_pinned(PRIMARY/'blind-numeric.pkl',pins['blind-numeric.pkl']))
    assert not any(k == 'jax' or k.startswith('jax.') for k in sys.modules)
    frozen = readouts['frozen']
    cal = {k:frozen['oof'][k] for k in ('episode_ids','D','K','S','U','choices')}
    cal['predicted_U'] = {k:frozen['regressors'][k]['training_predictions'] for k in NAMES}
    cal['returns'] = readouts['cal']['0']['returns'].tolist()
    assert len(cal['returns']) == 8 and all(len(v)==7 for v in cal['returns'])
    m = result['analysis']['models']['0']
    test = dict(episode_ids=m['episode_ids'],D=blind['0']['D'].tolist(),K=blind['0']['K'].tolist(),
                S=m['S_decision'],U=m['true_U'],choices=m['choices'],
                predicted_U={k:result['analysis']['discrimination']['regressors'][k]['predicted_U'] for k in NAMES})
    assert test['episode_ids'] == blind['0']['episode_ids'].tolist()
    exported = dict(version=VERSION,inputs_sha256=pins,cal=cal,already_opened_holdout=test,
                    regressors=frozen['regressors'],constant_choice=frozen['fit']['constant_choice'],
                    calibration_F_global_lambda_selection_not_fully_nested=True,
                    signal_cal_predictions_in_sample=True,
                    decoded_existing_pickles_include_unused_cal_and_blind_derived_features=True,
                    source_npz_rgb_checkpoint_replay_decoded=False)
    out = dict(version=VERSION,actual_git_sha=sha,cal=cohort(cal),already_opened_holdout=cohort(test),
               no_refit_no_new_threshold_no_independent_confirmation=True)
    out['support_ranges'] = {k:dict(cal=scalar(cal[k]),holdout=scalar(test[k]),
        holdout_below_cal=sum(v<min(cal[k]) for v in test[k]),
        holdout_above_cal=sum(v>max(cal[k]) for v in test[k])) for k in ('D','K','S')}
    c = exported['constant_choice']
    fixed = [row[a]-row[c] for row,(a,p) in zip(cal['returns'],cal['choices'])]
    shift = [row[c]-row[p] for row,(a,p) in zip(cal['returns'],cal['choices'])]
    assert all(abs(u-a-b)<1e-12 for u,a,b in zip(cal['U'],fixed,shift))
    out['cal_reference_decomposition'] = dict(frozen_constant_choice=c, original_U=cal['U'],
        U_same_q_choice_fixed_global_prior=fixed, prior_reference_shift=shift,
        original_mean=avg(cal['U']),fixed_prior_mean=avg(fixed),shift_mean=avg(shift),
        role='exploratory identity only; never replace original target or refit')
    for k in NAMES[1:]:
        old = result['analysis']['discrimination']['increments_over_D_only'][k]['mean']
        assert abs(out['already_opened_holdout']['increments'][k]['increment']-old)<1e-12
    for name,data in (('whitelist.json',exported),('diagnostic.json',out)):
        (args.output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    for name,pin in pins.items():
        read_pinned(AUDIT if name=='audit.json' else PRIMARY/name,pin)
    receipt = dict(version=VERSION,actual_git_sha=sha,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   started_at=started_at,finished_at=datetime.datetime.now().astimezone().isoformat(),elapsed_seconds=time.monotonic()-start,
                   input_sha256=pins,files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir()},
                   model_calls=0,optimizer_updates=0,physics_steps=0,actual_gpu_devices_used=0,
                   status='complete_exploratory_fixed_output_diagnostic')
    (args.output/'complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
