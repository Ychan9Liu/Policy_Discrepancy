"""Read-only SSH mirror of the completed M2 v1 artifacts; no checkpoint/replay copy."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

REMOTE = r'''
import datetime, hashlib, io, json, pathlib, subprocess, sys, tarfile
P = pathlib.Path
repo = P('/data/Policy_Discrepancy/repo')
audit = repo / 'analysis/outputs/formal-four-groups-pd1-20261010-final-attempt01'
index = json.loads((audit / 'RUN_INDEX.json').read_text())
files = set(p for p in audit.rglob('*') if p.is_file())
proof = dict(collected_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    server='sv2', repo_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
    repo_status=subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True),
    files={}, runs=[], audit_sources={})
ops = P('/data/Policy_Discrepancy/runs/formal-remaining-preflight-20261010')
for name in ('audit_formal_pd1_with_config.py','all-sixteen-postprocess-launch.json',
             'eight-startup-audit.json','initial-evaluation-audit.json','FINAL_DELIVERY.json'):
    if (ops/name).exists(): files.add(ops/name)
script = ops/'audit_formal_pd1_with_config.py'
digest = hashlib.sha256(script.read_bytes()).hexdigest()
for commit in ('d239c3fb7e5b77ddf205fe994c46c2e5a6acce6a',):
    blob=subprocess.check_output(['git','show',commit+':scripts/audit_formal_pd1.py'],cwd=repo)
    proof['audit_sources'][commit]=dict(sha256=hashlib.sha256(blob).hexdigest(),matches_deployed=hashlib.sha256(blob).hexdigest()==digest)
for row in index['runs']:
    d=P(row['directory'])
    for name in ('protocol_manifest.json','evaluations.jsonl','config.yaml',
                 'execution-initial.json','final_state.json','matching_result.json',
                 'metrics.jsonl','update_receipts.jsonl','c_frozen.json','c_frozen.json.sha256'):
        if (d/name).exists(): files.add(d/name)
    manifest=json.loads((d/'protocol_manifest.json').read_text())
    run=dict(directory=str(d),git_commit=manifest['git_commit'],
        artifact_sha256={'evaluations.jsonl':hashlib.sha256((d/'evaluations.jsonl').read_bytes()).hexdigest()},snapshots=[])
    for ev in map(json.loads,(d/'evaluations.jsonl').read_text().splitlines()):
        point=ev['action_step']; folder=d/'eval_snapshots'/f'{point:07d}'
        tag=(folder/'latest').read_text().strip()
        assert P(tag).name==tag
        meta=json.loads((folder/'evaluation_snapshot.json').read_text())
        assert (folder/tag/'done').is_file() and (folder/tag/'agent.pkl').is_file()
        assert meta['snapshot_id']==ev['snapshot_id'] and meta['git_commit']==manifest['git_commit']
        assert meta['actual_action_step']==point and meta['update_id']==ev['update_id']
        for p in (folder/'latest',folder/'evaluation_snapshot.json',folder/tag/'done'): files.add(p)
        run['snapshots'].append(dict(action_step=point,actual_action_step=point,
            counters={'updates':meta['update_id']},snapshot_id=meta['snapshot_id'],
            params_sha256=None,checkpoint_bytes=(folder/tag/'agent.pkl').stat().st_size,
            basis='05-remote-metadata-markers-and-checkpoint-presence;04-complete-return-audit-reused'))
    proof['runs'].append(run)
    root=d.parent.parent
    for name in ('supervisor-final.json','RUN_INDEX.json','baseline-completion-audit-failed.json',
                 'comparison-postprocess-failure.json'):
        if (root/name).exists(): files.add(root/name)
for rel in ('analysis/outputs/formal-baseline-dt-pd1-20261010-audit-attempt02/completion.json',):
    if (repo/rel).exists(): files.add(repo/rel)
try:
    import wandb
    api=wandb.Api(); checks=[]
    completion=json.loads((audit/'completion.json').read_text())
    for row in index['runs']:
        remote=api.run(row['wandb_entity']+'/PD_1/'+row['wandb_id'])
        summary=dict(remote.summary)
        receipt=next(r for r in completion['runs'] if r['run_id']==row['run_id'])
        encoded=json.dumps(receipt['values'],sort_keys=True,separators=(',',':'),allow_nan=False)
        checks.append(dict(run_id=row['run_id'],state=remote.state,
            exact_json_matches=summary.get('formal/exact_values_json')==encoded,
            exact_hash_matches=summary.get('formal/exact_values_sha256')==hashlib.sha256(encoded.encode()).hexdigest()))
    proof['wandb_readonly_crosscheck']=checks
except Exception as error:
    proof['wandb_readonly_crosscheck_error']=type(error).__name__
proof['gpu_check_time']=datetime.datetime.now(datetime.timezone.utc).isoformat()
proof['gpu_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv'],text=True)
proof['gpu_pmon']=subprocess.check_output(['nvidia-smi','pmon','-c','1'],text=True)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
    for p in sorted(files):
        data=p.read_bytes(); rel=str(p).lstrip('/')
        proof['files'][rel]=dict(source=str(p),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
        info=tarfile.TarInfo(rel); info.size=len(data); tar.addfile(info,io.BytesIO(data))
    data=json.dumps(proof,indent=2,sort_keys=True).encode()
    info=tarfile.TarInfo('COLLECTION.json'); info.size=len(data); tar.addfile(info,io.BytesIO(data))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / 'inputs.tar.gz'
    if archive.exists():
        raise FileExistsError('Do not replace an existing input collection')
    with archive.open('xb') as out:
        subprocess.run(['ssh', '-o', 'BatchMode=yes', 'sv2',
                        '/data/Policy_Discrepancy/envs/dreamer/bin/python -'],
                       input=REMOTE.encode(), stdout=out, check=True)
    mirror = args.output / 'inputs'
    with tarfile.open(archive) as tar:
        tar.extractall(mirror, filter='data')
    proof = json.loads((mirror / 'COLLECTION.json').read_text())
    for rel, item in proof['files'].items():
        assert hashlib.sha256((mirror / rel).read_bytes()).hexdigest() == item['sha256'], rel
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    (args.output / 'mirror_validation.json').write_text(json.dumps(dict(
        file_count=len(proof['files']), all_hashes_match=True,
        archive_sha256=archive_hash), indent=2) + '\n')
    print(json.dumps(dict(files=len(proof['files']), archive_sha256=archive_hash,
                         snapshots=sum(len(r['snapshots']) for r in proof['runs']))))


if __name__ == '__main__':
    main()
