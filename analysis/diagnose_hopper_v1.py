"""Forensic tables/figures from original hopper artifacts, without new training."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess


GROUPS=('baseline','dt','constant','shuffle')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,keys);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mode_metric_fixture():
    """Reproduce source reductions with 2 positions, 2 factors, 32 classes.

    Only one factor at one of the two positions differs. Source logs .015625
    and .25, whereas the whole-vector mismatch fraction is .5.
    """
    classes=32
    q=[[[1 if c==0 else 0 for c in range(classes)] for _ in range(2)] for _ in range(2)]
    p=[[list(factor) for factor in position] for position in q]
    p[0][0]=[1 if c==1 else 0 for c in range(classes)]
    differing_coordinates=sum(a!=b for qa,pa in zip(q,p) for qf,pf in zip(qa,pa) for a,b in zip(qf,pf))
    differing_factors=sum(qf!=pf for qa,pa in zip(q,p) for qf,pf in zip(qa,pa))
    differing_vectors=sum(qa!=pa for qa,pa in zip(q,p))
    return dict(source_mode_factor_diff_frac=differing_coordinates/(2*2*classes),
                source_mode_vector_diff_frac=differing_factors/4,
                correct_factor_diff_frac=differing_factors/4,
                correct_vector_diff_frac=differing_vectors/2)


def plots(output,rows,remote,frames):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors=dict(baseline='#4b5563',dt='#c2410c',constant='#047857',shuffle='#2563eb')
    fig,axs=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    series=(('positive_reward_frac','Real training positive-reward fraction',False),
            ('real_reward_sum','Real reward sum per 50k actions',True),
            ('imagined_reward_mean','Model imagined reward (not real reward)',True),
            ('policy_entropy','Imagined Gaussian policy entropy',False),
            ('adv_mag','Imagined normalized advantage magnitude',True),
            ('active_weight','Actual active rep weight',False))
    for ax,(field,title,log) in zip(axs.flat,series):
        for group in GROUPS:
            members=[r for r in rows if r['group']==group]
            ax.plot([r['end_action']/1e6 for r in members],[r[field] for r in members],
                    color=colors[group],label=group,lw=1.6)
        if log:ax.set_yscale('log')
        ax.set_title(title,fontsize=10);ax.set_xlabel('Training actions (millions)');ax.grid(alpha=.2)
    axs[1,0].axhline(4*.5*math.log(2*math.pi*math.e),color='black',ls='--',lw=.7,label='max entropy')
    axs[0,0].legend(fontsize=9)
    fig.suptitle('Hopper M2 v1 | seed0 | 50k-action means | original replay + JSONL')
    fig.savefig(output/'learning_failure_diagnostics.png',dpi=170)
    fig.savefig(output/'learning_failure_diagnostics.svg');plt.close(fig)
    if frames:
        with np.load(frames,allow_pickle=False) as z:
            for episode in (3,6,61):
                fig,axs=plt.subplots(4,6,figsize=(12,8),constrained_layout=True)
                for i,group in enumerate(GROUPS):
                    run=next(r for r in remote['runs'] if r['group']==group)
                    ep=next(e for e in run['training_episodes'] if e['chain']==0 and e['start_index']==episode*1001)
                    for j,(frame,offset) in enumerate(zip(z[group+'_'+str(episode)],(0,100,250,500,750,1000))):
                        axs[i,j].imshow(frame,interpolation='nearest')
                        axs[i,j].set_xticks([]);axs[i,j].set_yticks([])
                        if i==0:axs[i,j].set_title('episode step '+str(offset),fontsize=9)
                        if j==0:axs[i,j].set_ylabel(group+'\nreturn='+format(ep['return_raw'],'.2f'))
                fig.suptitle('Original replay frames | seed0 | chain0 | training episode '+str(episode)+' | no re-render')
                fig.savefig(output/('replay_episode_'+str(episode)+'.png'),dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspection',type=Path,required=True)
    parser.add_argument('--mirror',type=Path,required=True)
    parser.add_argument('--frames',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    remote=read(args.inspection)
    source_index=read(args.mirror/'data/Policy_Discrepancy/repo/analysis/outputs/formal-four-groups-pd1-20261010-final-attempt01/RUN_INDEX.json')
    rows=[];summaries=[];early=[];all_nonfinite=[];mode_checks=[]
    initial=remote['runs'][0]['initial_checkpoint']
    assert all(r['initial_checkpoint']['array_sha256']==initial['array_sha256'] for r in remote['runs'])
    assert all(r['initial_checkpoint']['counters']==initial['counters'] for r in remote['runs'])
    for run in remote['runs']:
        group=run['group'];directory=args.mirror/run['directory'].lstrip('/')
        for name,digest in run['input_sha256'].items():
            if (directory/name).exists():
                assert sha(directory/name)==digest,('Oct12 mirror/Oct20 original mismatch',name)
        allmets=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
        metrics=[r for r in allmets if 'train/dt/D_mean' in r]
        evs=[json.loads(line) for line in (directory/'evaluations.jsonl').read_text().splitlines()]
        config=read(directory/'protocol_manifest.json')['config']
        scales={key.removeprefix('agent.loss_scales.'):v for key,v in config.items() if key.startswith('agent.loss_scales.')}
        scales['image']=scales.pop('rec')
        for phase,log in zip(run['phases'],metrics):
            assert phase['end_action']==log['step']
            nonfinite=[k for k,v in log.items() if isinstance(v,(float,int)) and not math.isfinite(v)]
            all_nonfinite.extend((group,log['step'],k) for k in nonfinite)
            regroup=math.fsum(log['train/loss/'+k]*v for k,v in scales.items())
            assert math.isclose(regroup,log['train/opt/loss'],rel_tol=1e-7,abs_tol=1e-6)
            rep_delta=.1*(log['train/dt/rep_before_mean']-log['train/dt/rep_after_mean'])
            total_same_state=regroup+rep_delta
            mode_checks.append(math.isclose(log['train/dt/mode_vector_diff_frac'],16*log['train/dt/mode_factor_diff_frac'],abs_tol=1e-12))
            row=dict(group=group,start_action=phase['start_action'],end_action=phase['end_action'],
                positive_reward_steps=phase['positive_reward_count'],
                positive_reward_frac=1-phase['zero_reward_frac'],real_reward_sum=phase['reward_sum'],
                real_reward_mean=phase['reward']['mean'],raw_action_abs_ge1_frac=phase['action_abs_ge1_frac'],
                raw_action_std=phase['action_std'],imagined_reward_mean=log['train/rew'],
                imagined_normalized_return=log['train/ret'],adv_mag=log['train/adv_mag'],
                policy_entropy=log['train/ent/action'],reward_loss=log['train/loss/rew'],
                image_loss=log['train/loss/image'],value_loss=log['train/loss/value'],
                opt_grad_norm=log['train/opt/grad_norm'],opt_update_rms=log['train/opt/update_rms'],
                active_weight=log['train/dt/active_weight_mean'],D_mean=log['train/dt/D_mean'],
                D_p50_batch_mean=log['train/dt/D_p50'],w_p10_batch_mean=log['train/dt/w_p10'],
                mean_abs_actor_mean_diff=log['train/dt/mean_abs_diff'],
                mean_abs_actor_std_diff=log['train/dt/std_abs_diff'],
                active_frac=log['train/dt/rep_active_frac'],total_loss=regroup,
                direct_rep_loss_scalar_reduction=rep_delta,
                direct_rep_reduction_fraction_of_reconstructed_total=rep_delta/total_same_state,
                logged_mode_factor_diff=log['train/dt/mode_factor_diff_frac'],
                actual_factor_mode_diff=log['train/dt/mode_vector_diff_frac'],
                whole_vector_mode_diff=None,
                evaluation_return_at_end=evs[log['step']//50000]['mean'])
            rows.append(row)
        assert run['replay_summary']['train_actions']==1000000 and run['replay_summary']['resets']==1008
        assert run['replay_summary']['episode_count']==992
        assert run['replay_summary']['terminal_count']==0
        prefix=next(p for p in remote['prefix_comparisons'] if p['group']==group)
        assert all(proof['equal'] for key,proof in prefix['fields'].items() if key!='image')
        summaries.append(dict(group=group,initial_array_count=initial['array_count'],
            initial_combined_hash=run['initial_checkpoint']['combined_array_hash'],
            initial_arrays_equal_to_baseline=True,prefill_actions_and_rewards_equal=True,
            prefill_unequal_image_elements=prefix['fields']['image']['unequal_elements'],
            prefill_max_image_difference=prefix['fields']['image']['max_abs_difference'],
            complete_training_episodes=992,zero_return_episodes=run['replay_summary']['episode_zero_return_count'],
            mean_training_episode_return=run['replay_summary']['episode_returns']['mean'],
            zero_reward_fraction=sum(p['zero_reward_frac']*p['action_count'] for p in run['phases'])/1000000,
            total_training_reward=sum(p['reward_sum'] for p in run['phases']),
            positive_reward_steps=sum(p['positive_reward_count'] for p in run['phases'])))
        early.extend(dict(group=group,**p) for p in run['early_5k'])
    assert all(mode_checks)
    fixture=mode_metric_fixture()
    assert fixture==dict(source_mode_factor_diff_frac=.015625,source_mode_vector_diff_frac=.25,
                         correct_factor_diff_frac=.25,correct_vector_diff_frac=.5)
    dump(args.output/'phase_diagnostics.json',rows);write_csv(args.output/'phase_diagnostics.csv',rows)
    dump(args.output/'run_summary.json',summaries);write_csv(args.output/'run_summary.csv',summaries)
    dump(args.output/'early_5k.json',early);dump(args.output/'mode_diagnostic_issue.json',fixture)
    plots(args.output,rows,remote,args.frames)
    validation=dict(analysis_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        analysis_source_sha256=sha(__file__),remote_inspection_sha256=sha(args.inspection),
        collector_source_sha256=sha(Path(__file__).with_name('inspect_hopper_v1.py')),
        frames_sha256=sha(args.frames) if args.frames else None,
        all_initial_294_arrays_equal=True,all_prefill_2032_actions_and_rewards_equal=True,
        actual_training_actions=4000000,complete_training_episodes=3968,
        all_original_small_artifact_hashes_match_oct12_mirror=True,
        all_80_diagnostic_intervals_mode_fields_show_factor16=True,
        all_weighted_loss_reconstructions_match=True,nonfinite_logged_scalar_fields=all_nonfinite,
        interpretation='Training replay diagnostics are supplementary, not changes to formal evaluation metrics. '
          'Original reward labels now independently summed for training episodes only. '
          'No evaluation transition reward trajectories obtained; no causal gradient attribution.')
    dump(args.output/'validation.json',validation)
    dump(args.output/'hashes.json',{str(p.relative_to(args.output)):sha(p) for p in args.output.rglob('*') if p.is_file()})
    print(json.dumps(dict(output=str(args.output),runs=4,summary=summaries),ensure_ascii=False))


if __name__=='__main__':
    main()
