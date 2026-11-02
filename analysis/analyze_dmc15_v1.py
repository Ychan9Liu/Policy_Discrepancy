"""Read-only collection and descriptive analysis of DMC16 excluding acrobot.

Usage: python -m analysis.analyze_dmc15_v1 collect|analyze --output NEW_DIR
Analyze uses the immutable server JSON files previously collected into NEW_DIR.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

REMOTE = r'''
from pathlib import Path
import json, hashlib, datetime
root=Path('/data/Policy_Discrepancy/runs')
def read(p): return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
out=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runs=[],excluded=[])
for index in sorted(root.glob('formal-m2-dmc16-*/actual-RUN_INDEX.json')):
 for row in read(index)['runs']:
  if row['task']=='dmc_acrobot_swingup': continue
  d=Path(row['directory'])
  if not (d/'final_state.json').exists() or not row.get('run_audit'):
   out['excluded'].append(dict(index=str(index),task=row['task'],group=row['group'],state=row['state'],directory=str(d)));continue
  m=read(d/'protocol_manifest.json'); f=read(d/'final_state.json')
  assert m['git_commit']==row['code_sha'] and m['config']==row['config']
  assert f['train_action_steps']==1000000 and f['updates']==249493
  names=['protocol_manifest.json','final_state.json','evaluations.jsonl','metrics.jsonl','update_receipts.jsonl']
  hashes={n:sha(d/n) for n in names}
  ev=[json.loads(x) for x in (d/'evaluations.jsonl').read_text().splitlines()]
  assert [e['action_step'] for e in ev]==list(range(0,1000001,50000))
  snapshots=[]
  for e in ev:
   assert len(e['episodes'])==10 and all(x['complete'] for x in e['episodes'])
   folder=d/'eval_snapshots'/f"{e['action_step']:07d}"
   tag=(folder/'latest').read_text().strip(); assert Path(tag).name==tag
   s=read(folder/'evaluation_snapshot.json')
   assert s['snapshot_id']==e['snapshot_id'] and s['update_id']==e['update_id'] and s['actual_action_step']==e['action_step']
   assert s['git_commit']==m['git_commit'] and (folder/tag/'done').exists() and (folder/tag/'agent.pkl').exists()
   snapshots.append(dict(metadata=s,sha256=sha(folder/'evaluation_snapshot.json'),checkpoint_bytes=(folder/tag/'agent.pkl').stat().st_size))
  count=0; total=0.; n=0; last=-1
  with (d/'update_receipts.jsonl').open() as file:
   for line in file:
    r=json.loads(line); assert r['update_id']==count and r['invalid_count']==0 and last<=r['start_action']<=1000000
    last=r['start_action'];count+=1
    if 100000<=last<300000: total+=r['match_sum'];n+=r['match_count']
  assert count==249493 and last==1000000
  frozen=read(d/'c_frozen.json') if (d/'c_frozen.json').exists() else None
  if row['group']=='baseline':
   assert n>0 and abs(total/n-row['run_audit']['c'])<1e-12
   hashes['c_frozen.json']=sha(d/'c_frozen.json')
  metrics=[json.loads(x) for x in (d/'metrics.jsonl').read_text().splitlines()]
  out['runs'].append(dict(index_path=str(index),index_sha256=sha(index),row=row,manifest=m,final=f,evaluations=ev,snapshots=snapshots,metrics=metrics,hashes=hashes,receipts=dict(count=count,match_sum=total,match_count=n),frozen=frozen))
print(json.dumps(out,allow_nan=False))
'''

def dump(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n',encoding='utf-8')

def collect(out):
    out.mkdir(parents=True,exist_ok=False)
    def one(server):
        p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',server,
                          '/data/Policy_Discrepancy/envs/dreamer/bin/python -'],
                         input=REMOTE,text=True,capture_output=True,check=True)
        value=json.loads(p.stdout);value['server']=server
        value['collector_remote_sha256']=hashlib.sha256(REMOTE.encode()).hexdigest()
        dump(out/(server+'.json'),value)
        return server,len(value['runs'])
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        for result in pool.map(one,('sv1','sv2','sv3')): print(result,flush=True)

def analyze(out):
    runs=[]; sources={}
    for host in ('sv1','sv2','sv3'):
        p=out/(host+'.json');sources[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        for r in json.loads(p.read_text(encoding='utf-8'))['runs']:
            r['server']=host;runs.append(r)
    assert len(runs)==30 and len({(r['row']['task'],r['row']['group']) for r in runs})==30
    tasks=sorted({r['row']['task'] for r in runs});assert len(tasks)==15
    results=[]; curves=[]; diag=[]; manifest=[]; config_diffs={}
    for task in tasks:
        pair={r['row']['group']:r for r in runs if r['row']['task']==task}
        assert set(pair)=={'baseline','dt'}
        bc=pair['baseline']['manifest']['config'];dc=pair['dt']['manifest']['config']
        config_diffs[task]={k:[bc.get(k),dc.get(k)] for k in set(bc)|set(dc) if bc.get(k)!=dc.get(k)}
        assert set(config_diffs[task])=={'logdir','agent.rep_probe.mode'}
        summary=dict(task=task)
        for group,r in pair.items():
            ev=r['evaluations'];means=[];conf=r['manifest']['config']
            assert conf['seed']==0 and conf['env.dmc.repeat']==1 and conf['agent.rep_probe.alpha']==20
            assert r['final']['train_batch_keys']==249493 and r['final']['eval_action_steps']==sum(sum(e['lengths']) for e in ev)
            if group=='baseline':
                frozen=r['frozen'];rec=r['receipts']
                assert frozen['window']==[100000,300000] and not frozen['engineering_fixture']
                assert frozen['receipt_sha256']==r['hashes']['update_receipts.jsonl']
                assert frozen['match_sum']==rec['match_sum'] and frozen['match_count']==rec['match_count']
                assert frozen['c']==rec['match_sum']/rec['match_count']
            logs={m['step']:m for m in r['metrics'] if 'protocol/eval_return' in m}
            for e in ev:
                assert [x['episode_id'] for x in e['episodes']]==list(range(10))
                assert all(x['return_definition']=='undiscounted_sum_non_reset_rewards_repeat_already_summed' for x in e['episodes'])
                vals=[x['return_raw'] for x in e['episodes']];mean=math.fsum(vals)/10
                assert math.isclose(mean,logs[e['action_step']]['protocol/eval_return'],abs_tol=1e-9,rel_tol=1e-12)
                means.append(mean);curves.append(dict(task=task,group=group,step=e['action_step'],mean=mean))
            summary[group+'_auc']=math.fsum((a+b)*.025 for a,b in zip(means,means[1:]))
            summary[group+'_tail']=math.fsum(means[-5:])/5
            bins=[m for m in r['metrics'] if 'train/dt/D_mean' in m]
            assert [b['step'] for b in bins]==list(range(50000,1000001,50000))
            weights=[ev[i+1]['update_id']-ev[i]['update_id'] for i in range(20)]
            d=dict(task=task,group=group)
            for key in ('D_mean','rep_active_frac','active_weight_mean','actual_weight_mean','excess_reduction_mean','weighted_excess_mean'):
                d[key]=math.fsum(b['train/dt/'+key]*w for b,w in zip(bins,weights))/sum(weights)
            d['excess_reduction_ratio']=d['excess_reduction_mean']/(d['weighted_excess_mean']+d['excess_reduction_mean'])
            d['zero_D_fraction']=math.fsum(b['train/dt/D_zero_count']/b['train/dt/loss_position_count']*w for b,w in zip(bins,weights))/sum(weights)
            for label,lo,hi in [('early',0,100000),('middle',100000,800000),('late',800000,1000000)]:
                chosen=[(b,w) for b,w in zip(bins,weights) if lo<b['step']<=hi]
                for key in ('D_mean','active_weight_mean','rep_active_frac'):
                    d[label+'_'+key]=math.fsum(b['train/dt/'+key]*w for b,w in chosen)/sum(w for b,w in chosen)
            d['c']=r['row']['run_audit'].get('c');diag.append(d)
            manifest.append(dict(task=task,group=group,seed=0,server=r['server'],directory=r['row']['directory'],
                                 sha=r['manifest']['git_commit'],state=r['row']['state'],run_id=r['row']['run_id'],
                                 physical_gpu=r['row'].get('physical_gpu'),hashes=r['hashes'],audit=r['row']['run_audit']))
        for metric in ('auc','tail'):summary['delta_'+metric]=summary['dt_'+metric]-summary['baseline_'+metric]
        results.append(summary)
    for name,rows in [('task_metrics',results),('gate_summary',diag),('learning_points',curves)]:
        dump(out/(name+'.json'),rows)
        with (out/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    dump(out/'run_manifest.json',manifest);dump(out/'config_differences.json',config_diffs)
    proof={}
    for file in ('dreamerv3/agent.py','dreamerv3/rep_probe.py','dreamerv3/rssm.py','embodied/run/protocol_v1.py','embodied/envs/dmc.py'):
        proof[file]={s:hashlib.sha256(subprocess.check_output(['git','show',s+':'+file])).hexdigest() for s in sorted({r['sha'] for r in manifest})}
        assert len(set(proof[file].values()))==1
    dump(out/'training_source_comparison.json',proof)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axs=plt.subplots(5,3,figsize=(14,15),layout='constrained')
    for task,ax in zip(tasks,axs.flat):
        for group,color in [('baseline','#4b5563'),('dt','#c2410c')]:
            p=[x for x in curves if x['task']==task and x['group']==group]
            ax.plot([x['step']/1e6 for x in p],[x['mean'] for x in p],label='PPD-KL' if group=='dt' else group,color=color)
        ax.set(title=task.removeprefix('dmc_'),xlabel='Training actions (millions)',ylabel='Raw return');ax.grid(alpha=.2)
    axs[0,0].legend();fig.suptitle('15 new tasks | seed0 | 10 episodes / point | unsmoothed')
    fig.savefig(out/'learning_curves.png',dpi=160);fig.savefig(out/'learning_curves.svg');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,8),layout='constrained')
    for ax,key,title in zip(axs,('delta_auc','delta_tail'),('AUC / 1M','Mean of last five points')):
        vals=[r[key] for r in results];ax.barh(range(15),vals,color=['#047857' if v>0 else '#b91c1c' for v in vals])
        ax.set_yticks(range(15),[t.removeprefix('dmc_') for t in tasks]);ax.invert_yaxis();ax.axvline(0,color='black',lw=.7);ax.set(title=title,xlabel='PPD-KL minus baseline (raw return)')
    fig.suptitle('seed0 descriptive differences | acrobot excluded')
    fig.savefig(out/'differences.png',dpi=160);fig.savefig(out/'differences.svg');plt.close(fig)
    dump(out/'verification.json',dict(input_sha256=sources,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),runs=30,evaluation_points=630,episodes=6300,
         receipts=30*249493,config_difference_keys=sorted({k for d in config_diffs.values() for k in d}),
         positive_auc=sum(r['delta_auc']>0 for r in results),positive_tail=sum(r['delta_tail']>0 for r in results)))
    print(json.dumps(results,ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['collect','analyze']);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();(collect if args.mode=='collect' else analyze)(args.output)
