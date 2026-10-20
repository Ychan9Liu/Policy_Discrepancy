"""CPU-only read of trusted original hopper checkpoints/replay; no model/env execution.

Run locally: python -m analysis.inspect_hopper_v1 --output NEW_DIRECTORY
The remote program is piped over SSH and never writes the server filesystem.
"""
import argparse
import json
from pathlib import Path
import subprocess


REMOTE = r'''
import datetime, gc, hashlib, importlib.metadata, json, pathlib, pickle, sys, zipfile
import numpy as np
P = pathlib.Path
INDEX = P('/data/Policy_Discrepancy/repo/analysis/outputs/formal-four-groups-pd1-20261010-final-attempt01/RUN_INDEX.json')
index=json.loads(INDEX.read_text())
rows=[r for r in index['runs'] if r['task']=='dmc_hopper_hop']
result=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    kind='read-only original artifacts; CPU numpy/pickle; no agent/environment/GPU initialization',
    training_sha='48d71f1541721208322442f5a1d5977124f3e888',
    input_index=str(INDEX),input_index_sha256=hashlib.sha256(INDEX.read_bytes()).hexdigest(),runs=[],prefix_comparisons=[])
def ahash(a):
    h=hashlib.sha256(); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()
def param_hashes(params):
    return {k:ahash(np.asarray(v)) for k,v in sorted(params.items())}
def summary(values):
    values=np.asarray(values,dtype=np.float64)
    return dict(count=int(values.size),mean=float(values.mean()),min=float(values.min()),max=float(values.max()),
                p10=float(np.percentile(values,10)),p50=float(np.percentile(values,50)),p90=float(np.percentile(values,90))) if values.size else None
def phase_summary(reward,action,first,last,stamp,lo,hi):
    use=(stamp>lo)&(stamp<=hi)&~first
    a=action[use]; r=reward[use]
    return dict(start_action=lo,end_action=hi,action_count=int(use.sum()),
        reward=summary(r),positive_reward_count=int((r>0).sum()),zero_reward_frac=float((r==0).mean()),
        reward_sum=float(r.sum(dtype=np.float64)),action_mean=a.mean(0).tolist(),
        action_std=a.std(0).tolist(),action_abs_ge1_frac=float((abs(a)>=1).mean()),
        action_abs_ge1_per_dim=(abs(a)>=1).mean(0).tolist(),action_finite=bool(np.isfinite(a).all()))
baseline_prefix=None
for row in rows:
    directory=P(row['directory']); group=row['group']
    proof=dict(group=group,directory=str(directory),run_id=row['run_id'],physical_gpu=row['physical_gpu'],
               input_sha256={},initial_checkpoint={},replay_chunks=[])
    for name in ('protocol_manifest.json','evaluations.jsonl','metrics.jsonl','final_state.json','execution-initial.json','stdout-initial.log'):
        proof['input_sha256'][name]=hashlib.sha256((directory/name).read_bytes()).hexdigest()
    manifest=json.loads((directory/'protocol_manifest.json').read_text())
    assert manifest['git_commit']==row['code_sha']==result['training_sha']
    assert not manifest['config']['run.engineering_fixture']
    folder=directory/'eval_snapshots/0000000'; tag=(folder/'latest').read_text().strip()
    path=folder/tag/'agent.pkl'
    with path.open('rb') as f: saved=pickle.load(f)
    hashes=param_hashes(saved['params'])
    proof['initial_checkpoint']=dict(path=str(path),bytes=path.stat().st_size,
        file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),counters=saved['counters'],
        array_count=len(hashes),array_sha256=hashes,
        combined_array_hash=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
        parameter_keys=[k for k in saved['params'] if not k.startswith('opt/')])
    del saved; gc.collect()
    paths={p.stem.split('-')[1]:p for p in (directory/'replay').glob('*.npz')}
    successors={p.stem.split('-')[2] for p in paths.values()}
    roots=sorted((p for key,p in paths.items() if key not in successors),key=lambda p:p.name)
    assert len(roots)==16, ('Unexpected replay chain count',group,len(roots))
    chains=[]; prefix=[]; visited=set()
    for chain_index, root in enumerate(roots):
        parts=[]; path=root
        while path is not None:
            stem=path.stem.split('-'); uuid,succ,n=stem[1],stem[2],int(stem[3]); assert uuid not in visited
            visited.add(uuid)
            with np.load(path,allow_pickle=False) as z:
                fields={k:z[k] for k in ('reward','action','is_first','is_last','is_terminal')}
                assert all(len(v)==n for v in fields.values())
                if not parts:
                    assert n>=128
                    initial={k:v[:128].copy() for k,v in fields.items()}
                    initial['image']=z['image'][:128].copy()
                    prefix.append(initial)
                with zipfile.ZipFile(path) as archive:
                    members={k:dict(crc=archive.getinfo(k+'.npy').CRC,
                        bytes=archive.getinfo(k+'.npy').file_size,sha256=ahash(v)) for k,v in fields.items()}
                proof['replay_chunks'].append(dict(file=str(path),chain=chain_index,
                    length=n,uuid=uuid,successor_uuid=succ,members=members))
            parts.append(fields)
            path=paths.get(succ)
        chain={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
        chains.append(chain)
    assert len(visited)==len(paths), 'Unvisited replay chunks/cycle'
    final=json.loads((directory/'final_state.json').read_text())
    assert sum(len(c['reward']) for c in chains)==final['train_action_steps']+final['train_resets']
    assert sum(c['is_first'].sum() for c in chains)==final['train_resets']
    cursors=[0]*16; actions=0; global_fields={k:[] for k in chains[0]}; stamps=[]
    # Reconstruct the exact selected-worker callback order of protocol_v1/Driver.
    while actions<1000000:
        boundary=(actions//50000+1)*50000
        selected=range(min(16,boundary-actions))
        for worker in selected:
            i=cursors[worker]; c=chains[worker]
            first=bool(c['is_first'][i]); actions+=not first
            for k in global_fields:global_fields[k].append(c[k][i])
            stamps.append(actions); cursors[worker]+=1
    assert cursors==[len(c['reward']) for c in chains], 'Replay order/worker assignment mismatch'
    fields={k:np.asarray(v) for k,v in global_fields.items()}; stamp=np.asarray(stamps)
    assert actions==final['train_action_steps'] and int(fields['is_first'].sum())==final['train_resets']
    episodes=[]
    for worker,c in enumerate(chains):
        # Independent per-chain complete episodes, including reset exclusion.
        start=0
        for i in np.flatnonzero(c['is_last']):
            assert c['is_first'][start] and not c['is_first'][start+1:i+1].any()
            episodes.append(dict(chain=worker,start_index=int(start),end_index=int(i),
                return_raw=float(c['reward'][start+1:i+1].sum(dtype=np.float64)),
                length=int(i-start),positive_steps=int((c['reward'][start+1:i+1]>0).sum())))
            start=i+1
    proof['replay_summary']=dict(chains=16,transitions=len(stamp),train_actions=actions,
        resets=int(fields['is_first'].sum()),terminal_count=int(fields['is_terminal'].sum()),
        reward_finite=bool(np.isfinite(fields['reward']).all()),
        episode_count=len(episodes),episode_returns=summary([e['return_raw'] for e in episodes]),
        episode_zero_return_count=sum(e['return_raw']==0 for e in episodes),
        prefix_worker_order_basis='chronological first chunk creation, same ordered Driver callbacks; validated complete counts',
        scalar_arrays_sha256={k:ahash(v) for k,v in fields.items()})
    proof['training_episodes']=episodes
    proof['phases']=[phase_summary(fields['reward'],fields['action'],fields['is_first'],fields['is_last'],stamp,lo,hi)
        for lo,hi in zip(range(0,1000000,50000),range(50000,1000001,50000))]
    proof['early_5k']=[phase_summary(fields['reward'],fields['action'],fields['is_first'],fields['is_last'],stamp,lo,lo+5000)
        for lo in range(0,100000,5000)]
    proof['initial_2032_actions']=phase_summary(fields['reward'],fields['action'],fields['is_first'],fields['is_last'],stamp,0,2032)
    # Prefix arrays are all generated before the first optimizer update at 2032 actions.
    if baseline_prefix is None:baseline_prefix=prefix
    compare=dict(group=group,root_prefix_rows=128,non_reset_actions=2032,fields={})
    for key in prefix[0]:
        a=np.stack([p[key] for p in baseline_prefix]); b=np.stack([p[key] for p in prefix])
        unequal=a!=b
        loc=np.argwhere(unequal)
        compare['fields'][key]=dict(equal=bool(not unequal.any()),baseline_sha256=ahash(a),current_sha256=ahash(b),
            unequal_elements=int(unequal.sum()),first_unequal_index=loc[0].tolist() if len(loc) else None,
            max_abs_difference=float(abs(a.astype(np.float64)-b.astype(np.float64)).max()))
    result['prefix_comparisons'].append(compare)
    result['runs'].append(proof)
    print('Read '+group+': '+str(len(paths))+' chunks; '+str(len(episodes))+' complete training episodes',file=sys.stderr,flush=True)
    del fields,global_fields,chains,prefix; gc.collect()
source=importlib.metadata.distribution('dm-control').locate_file('dm_control/suite/hopper.py')
result['hopper_installed_source']=dict(path=str(source),sha256=hashlib.sha256(P(source).read_bytes()).hexdigest(),source=P(source).read_text())
result['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
print(json.dumps(result,indent=2,allow_nan=False))
'''

FRAMES_REMOTE = r'''
import numpy as np, pathlib, json, sys
index=json.loads(pathlib.Path('/data/Policy_Discrepancy/repo/analysis/outputs/formal-four-groups-pd1-20261010-final-attempt01/RUN_INDEX.json').read_text())
arrays={}
for row in index['runs']:
    if row['task']!='dmc_hopper_hop':continue
    paths={p.stem.split('-')[1]:p for p in (pathlib.Path(row['directory'])/'replay').glob('*.npz')}
    succ={p.stem.split('-')[2] for p in paths.values()}
    roots=sorted([p for k,p in paths.items() if k not in succ],key=lambda p:p.name)
    chain=[]; start=0; p=roots[0]
    while p:
        n=int(p.stem.split('-')[3]);chain.append((start,start+n,p))
        start+=n;p=paths.get(p.stem.split('-')[2])
    for episode in (3,6,61):
        frames=[]
        for offset in (0,100,250,500,750,1000):
            index=episode*1001+offset
            a,b,p=next(z for z in chain if z[0]<=index<z[1])
            with np.load(p,allow_pickle=False) as z:frames.append(z['image'][index-a])
        arrays[row['group']+'_'+str(episode)]=np.stack(frames)
np.savez_compressed(sys.stdout.buffer,**arrays)
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',action='store_true',help='Also extract stored replay images; no rendering')
    parser.add_argument('--frames-only',action='store_true',help='Extract only stored replay images into a new directory')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    if args.frames or args.frames_only:
        (args.output/'frames_remote_source.py').write_text(FRAMES_REMOTE,encoding='utf-8')
        with (args.output/'replay_frames.npz').open('xb') as out:
            subprocess.run(['ssh','-o','BatchMode=yes','sv2',
                'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= /data/Policy_Discrepancy/envs/dreamer/bin/python -'],
                input=FRAMES_REMOTE.encode(),stdout=out,check=True)
    if args.frames_only:
        print(json.dumps(dict(output=str(args.output),frames_only=True)))
        return
    (args.output/'remote_source.py').write_text(REMOTE,encoding='utf-8')
    with (args.output/'remote_inspection.json').open('xb') as out, (args.output/'ssh.log').open('xb') as log:
        result=subprocess.run(['ssh','-o','BatchMode=yes','sv2',
            'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu /data/Policy_Discrepancy/envs/dreamer/bin/python -'],
            input=REMOTE.encode(),stdout=out,stderr=log)
    if result.returncode:
        raise RuntimeError('Remote read failed; see ssh.log; old directory retained')
    payload=json.loads((args.output/'remote_inspection.json').read_text())
    print(json.dumps(dict(output=str(args.output),runs=len(payload['runs']))))


if __name__=='__main__':
    main()
