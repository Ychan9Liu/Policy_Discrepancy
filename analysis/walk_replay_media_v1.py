"""Read existing quadruped replay on sv2 and encode a labelled illustration.

No policy inference, environment creation, training, or server writes. The clip
uses the first saved replay chain's last complete episode before a target step,
chosen before looking at its return or frames. It is NOT a fixed-checkpoint eval.
The original walk/250k defaults are retained; run/500k is also supported.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[1]
REMOTE = r'''
import datetime, hashlib, io, json, pathlib, subprocess, sys, tarfile
import numpy as np
P = pathlib.Path
repo = P('/data/Policy_Discrepancy/repo')
index_path = repo/'analysis/outputs/formal-four-groups-pd1-20261010-final-attempt01/RUN_INDEX.json'
index = json.loads(index_path.read_text())
task, limit = sys.argv[1], int(sys.argv[2])
assert (task,limit) in (('dmc_quadruped_walk',250000),('dmc_quadruped_run',500000))
proof = dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    server='sv2', kind='existing replay images only; no environment, agent, or GPU',
    source_index=str(index_path), source_index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
    training_sha='48d71f1541721208322442f5a1d5977124f3e888', train_seed=0,
    task=task, target_action_step=limit,
    selection=f'First chronologically saved root chain; last complete episode ending at or before {limit} global training actions; no selection on reward or appearance.',
    clip_kind=f'training replay, changing train policy; NOT a frozen {limit}-action checkpoint evaluation',
    chain_order_basis='Chronological first-chunk creation in ordered Driver callbacks, as in the existing hopper inspection; stored chain order, not an independently stored worker ID.',
    runs=[])
arrays = {}

def digest_array(a):
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()

for group in ('baseline', 'dt'):
    row = next(r for r in index['runs'] if r['task']==task and r['group']==group)
    directory = P(row['directory'])
    manifest = json.loads((directory/'protocol_manifest.json').read_text())
    assert manifest['git_commit']==proof['training_sha'] and not manifest['config']['run.engineering_fixture']
    evaluations = list(map(json.loads, (directory/'evaluations.jsonl').read_text().splitlines()))
    evaluation = next(r for r in evaluations if r['action_step']==limit)
    folder = directory/'eval_snapshots'/f'{limit:07d}'
    tag = (folder/'latest').read_text().strip()
    assert P(tag).name==tag and (folder/tag/'done').is_file()
    checkpoint = folder/tag/'agent.pkl'
    snapshot = json.loads((folder/'evaluation_snapshot.json').read_text())
    assert snapshot['snapshot_id']==evaluation['snapshot_id'] and snapshot['actual_action_step']==limit
    files = {p.stem.split('-')[1]:p for p in (directory/'replay').glob('*.npz')}
    successors = {p.stem.split('-')[2] for p in files.values()}
    roots = sorted((p for key,p in files.items() if key not in successors), key=lambda p:p.name)
    assert len(roots)==16
    chains, path_chains, proof_chunks = [], [], []
    for chain_id, root in enumerate(roots):
        parts, paths, path, cursor = [], [], root, 0
        # Allow room for reset observations and the final partial chunk.
        while path is not None and cursor<limit//16+4096:
            with np.load(path, allow_pickle=False) as data:
                fields = {k:data[k] for k in ('reward','is_first','is_last')}
            n = int(path.stem.split('-')[3]); assert all(len(a)==n for a in fields.values())
            proof_chunks.append(dict(chain=chain_id, path=str(path), start_index=cursor,
                length=n, array_sha256={k:digest_array(v) for k,v in fields.items()}))
            paths.append((cursor, cursor+n, path)); cursor += n; parts.append(fields)
            path = files.get(path.stem.split('-')[2])
        chains.append({k:np.concatenate([part[k] for part in parts]) for k in parts[0]})
        path_chains.append(paths)
    cursors, actions = [0]*16, 0
    stamps = [[] for _ in chains]
    while actions<limit:
        boundary = min((actions//50000+1)*50000, limit)
        for worker in range(min(16, boundary-actions)):
            first = bool(chains[worker]['is_first'][cursors[worker]])
            actions += not first
            stamps[worker].append(actions); cursors[worker] += 1
    assert actions==limit
    candidates = []
    for chain_id, chain in enumerate(chains):
        start = 0
        for ordinal, end in enumerate(np.flatnonzero(chain['is_last'][:cursors[chain_id]])):
            end = int(end)
            assert chain['is_first'][start] and not chain['is_first'][start+1:end+1].any()
            candidates.append(dict(chain=chain_id, episode_ordinal=ordinal,
                start_index=start, end_index=end, length=end-start,
                first_global_action=stamps[chain_id][start+1],
                last_global_action=stamps[chain_id][end],
                return_raw=float(chain['reward'][start+1:end+1].sum(dtype=np.float64))))
            start = end+1
    selected = max((c for c in candidates if c['chain']==0), key=lambda c:c['end_index'])
    assert selected['length']==1000
    parts, source_images = [], []
    for start,end,path in path_chains[0]:
        lo,hi = max(start, selected['start_index']), min(end, selected['end_index']+1)
        if lo>=hi: continue
        with np.load(path, allow_pickle=False) as data:
            images = data['image']
            part = images[lo-start:hi-start].copy()
        parts.append(part)
        source_images.append(dict(path=str(path), local_slice=[lo-start,hi-start],
            image_array_sha256=digest_array(images), selected_frames_sha256=digest_array(part)))
    frames = np.concatenate(parts)
    assert frames.shape==(1001,64,64,3) and frames.dtype==np.uint8
    arrays[group+'_frames'] = frames
    arrays[group+'_reward'] = chains[0]['reward'][selected['start_index']:selected['end_index']+1]
    media = [dict(path=str(p), bytes=p.stat().st_size) for p in directory.rglob('*')
             if p.is_file() and p.suffix.lower() in ('.mp4','.gif','.webm','.avi','.mov')]
    run = dict(group=group, directory=str(directory), run_id=row['run_id'],
        wandb_id=row['wandb_id'], input_sha256={name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
            for name in ('protocol_manifest.json','evaluations.jsonl','final_state.json')},
        evaluation=evaluation, snapshot=snapshot,
        checkpoint=dict(path=str(checkpoint), bytes=checkpoint.stat().st_size,
                        done=True, content_read=False),
        existing_video_files=media, replay_chunks_total=len(files), replay_scalar_inputs=proof_chunks,
        selected_episode=selected, selected_frames_sha256=digest_array(frames), image_sources=source_images,
        latest_complete_episodes=[max((c for c in candidates if c['chain']==i), key=lambda c:c['end_index']) for i in range(16)])
    try:
        import wandb
        remote = wandb.Api(timeout=30).run('ychan9liu-tsinghua-university/PD_1/'+row['wandb_id'])
        run['wandb_files']=[dict(name=f.name, bytes=f.size) for f in remote.files()]
    except Exception as error:
        run['wandb_file_inventory_error']=type(error).__name__
    if task=='dmc_quadruped_walk' and limit==250000:
        run['evaluation_250k']=evaluation  # Compatibility with the original walk inspection.
    proof['runs'].append(run)

source = subprocess.check_output(['git','show',proof['training_sha']+':embodied/run/protocol_v1.py'],cwd=repo)
proof['original_runner_sha256']=hashlib.sha256(source).hexdigest()
text=source.decode(); proof['original_evaluate_source']=text[text.index('  def evaluate(point):'):text.index('  try:',text.index('  def evaluate(point):'))]
proof['original_runner_has_video_code']='video' in text.lower()
frames_io=io.BytesIO(); np.savez_compressed(frames_io, **arrays)
records={'inspection.json':json.dumps(proof, ensure_ascii=False, indent=2).encode(),
         'replay_frames.npz':frames_io.getvalue()}
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
    for name,data in records.items():
        item=tarfile.TarInfo(name); item.size=len(data); archive.addfile(item,io.BytesIO(data))
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(output, task='dmc_quadruped_walk', action_step=250000):
    output.mkdir(parents=True, exist_ok=False)
    archive = output / 'inputs.tar.gz'
    with archive.open('xb') as stream:
        subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'sv2',
                        f'/data/Policy_Discrepancy/envs/dreamer/bin/python - {task} {action_step}'],
                       input=REMOTE.encode(), stdout=stream, check=True)
    with tarfile.open(archive) as stream:
        stream.extractall(output, filter='data')
    info = json.loads((output / 'inspection.json').read_text(encoding='utf-8'))
    print(json.dumps(dict(output=str(output), runs=[
        dict(group=r['group'], selected_episode=r['selected_episode'],
             evaluation_mean=r['evaluation']['mean_return_raw']) for r in info['runs']]),
        ensure_ascii=False))


def verify_evaluations(info):
    """Cross-check freshly read sources against the original accepted audit."""
    original = ROOT / 'analysis/outputs/m2-v1-seed0-20261012/analysis-final'
    hashes = json.loads((original / 'output_hashes.json').read_text(encoding='utf-8'))
    for name in ('return-audit/tables.json', 'run_manifest.json'):
        expected = next(v for k, v in hashes.items() if k.replace('\\', '/') == name)
        assert sha(original / name) == expected
    tables = json.loads((original / 'return-audit/tables.json').read_text(encoding='utf-8'))
    runs = json.loads((original / 'run_manifest.json').read_text(encoding='utf-8'))
    checks = []
    for run in info['runs']:
        audited = next(r for r in runs if r['run_id'] == run['run_id'])
        point = next(r for r in tables['evaluation_points'] if
                     r['run_id'] == run['run_id'] and
                     r['target_action_step'] == info['target_action_step'])
        assert run['input_sha256']['protocol_manifest.json'] == audited['manifest_sha256']
        assert run['input_sha256']['evaluations.jsonl'] == audited['evaluation_sha256']
        for name, digest in run['input_sha256'].items():
            assert sha(Path(audited['local_directory']) / name) == digest
        ev = run.get('evaluation', run.get('evaluation_250k'))
        assert ev['returns_raw'] == point['returns_raw'] and ev['snapshot_id'] == point['snapshot_id']
        assert ev['update_id'] == point['update_id'] == run['snapshot']['update_id']
        assert len(ev['returns_raw']) == len(ev['episodes']) == 10
        assert all(r['complete'] and r['length'] == 1000 for r in ev['episodes'])
        assert math.isclose(math.fsum(ev['returns_raw']) / 10,
                            point['mean_return_raw'], rel_tol=0, abs_tol=1e-9)
        checks.append(dict(group=run['group'], mean_return_raw=point['mean_return_raw'],
                           returns_raw=point['returns_raw'], snapshot_id=point['snapshot_id'],
                           update_id=point['update_id'], source_hashes_match=True))
    evaluations = [r.get('evaluation', r.get('evaluation_250k')) for r in info['runs']]
    assert evaluations[0]['seeds'] == evaluations[1]['seeds']
    return dict(runs=checks, paired_eval_environment_seeds=evaluations[0]['seeds'],
                dt_minus_baseline=checks[1]['mean_return_raw'] - checks[0]['mean_return_raw'],
                audited_tables_sha256=sha(original / 'return-audit/tables.json'))


def render(source, output):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import imageio_ffmpeg

    output.mkdir(parents=True, exist_ok=False)
    info = json.loads((source / 'inspection.json').read_text(encoding='utf-8'))
    task = info.get('task', 'dmc_quadruped_walk')
    target = info['target_action_step']
    assert (task, target) in (('dmc_quadruped_walk', 250000), ('dmc_quadruped_run', 500000))
    verification = verify_evaluations(info)
    data = np.load(source / 'replay_frames.npz', allow_pickle=False)
    frames = [data[g + '_frames'] for g in ('baseline', 'dt')]
    for group, values in zip(('baseline', 'dt'), frames):
        run = next(r for r in info['runs'] if r['group'] == group)
        episode = run['selected_episode']
        assert 0 < episode['first_global_action'] <= episode['last_global_action'] <= target
        digest = hashlib.sha256()
        digest.update(str(values.dtype).encode()); digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
        assert digest.hexdigest() == run['selected_frames_sha256']
        assert values.shape == (1001, 64, 64, 3)
        assert abs(float(data[group+'_reward'][1:].sum(dtype=np.float64)) -
                   episode['return_raw']) < 1e-9
    fontpath = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(fontpath), 19)
    small = ImageFont.truetype(str(fontpath), 15)
    heading = ImageFont.truetype(str(fontpath), 23)
    colors = ('#4b5563', '#c2410c')
    width, height, panel = 820, 558, 384
    ranges = [(r['selected_episode']['first_global_action'],
               r['selected_episode']['last_global_action']) for r in info['runs']]
    assert ranges[0] == ranges[1]
    lo, hi = ranges[0]
    title = (f'{task.removeprefix("dmc_")} | seed0 | '
             f'{lo/1000:.0f}k-{hi/1000:.0f}k training actions')
    prefix = f'{task.rsplit("_", 1)[1]}_before{target//1000}k_training'
    full_name, excerpt_name = prefix + '_full.mp4', prefix + '_8s_to18s.mp4'

    def compose(index):
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        draw.text((width/2, 16), title,
                  font=heading, fill='#111111', anchor='mt')
        draw.text((width/2, 48), 'Original training replay | changing policy | NOT checkpoint evaluation',
                  font=small, fill='#555555', anchor='mt')
        for i, (values, label) in enumerate(zip(frames, ('baseline', 'PPD-KL'))):
            x = 18 + i*400
            draw.text((x+panel/2, 79), label, font=heading, fill=colors[i], anchor='mt')
            image.paste(Image.fromarray(values[index]).resize((panel,panel), Image.Resampling.NEAREST), (x,111))
            episode = info['runs'][i]['selected_episode']
            draw.text((x+panel/2, 502), f"Recorded episode return: {episode['return_raw']:.2f}",
                      font=font, fill=colors[i], anchor='mt')
        draw.text((width/2, 539), f'Time in recorded episode: {index*.02:.2f} s | original 64 x 64 pixels | 1x speed',
                  font=small, fill='#555555', anchor='mt')
        return image

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    def encode(name, indices):
        # Keep every second frame and encode at 25 fps: original .02s/action, 1x speed.
        command = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
                   '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
                   '-framerate', '25', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18',
                   '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output/name)]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
            for index in indices:
                process.stdin.write(compose(index).tobytes())
            process.stdin.close()
            assert process.wait()==0

    encode(full_name, range(0,1000,2))
    encode(excerpt_name, range(400,900,2))
    stills = [100,250,400,550,700,850]
    sheet = Image.new('RGB', (width*3,height*2), '#eeeeee')
    for i,index in enumerate(stills):
        sheet.paste(compose(index), ((i%3)*width, (i//3)*height))
    sheet.save(output / 'contact_sheet.png')
    command = [ffmpeg, '-nostdin', '-hide_banner', '-i', str(output/full_name),
               '-f', 'null', '-']
    decoded = subprocess.run(command, capture_output=True, text=True, check=True)
    excerpt_decoded = subprocess.run(
        [ffmpeg, '-nostdin', '-hide_banner', '-i',
         str(output/excerpt_name), '-f', 'null', '-'],
        capture_output=True, text=True, check=True)
    (output / 'decode_check.txt').write_text(decoded.stderr + '\n' + excerpt_decoded.stderr,
                                            encoding='utf-8')
    assert 'frame=  500' in decoded.stderr and 'frame=  250' in excerpt_decoded.stderr
    verification.update(source_frames_and_reward_verified=True,
                        both_mp4_fully_decoded=True, decoded_frames=[500,250],
                        task=task, target_action_step=target,
                        actual_training_action_range=[lo,hi],
                        inspection_sha256=sha(source / 'inspection.json'))
    (output/'verification.json').write_text(json.dumps(verification, ensure_ascii=False,
                                                     indent=2)+'\n', encoding='utf-8')
    record = dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        script_sha256=sha(Path(__file__)), source_hashes={p.name:sha(p) for p in sorted(source.iterdir()) if p.is_file()},
        display_alias={'dt':'PPD-KL'}, evidence_kind=info['clip_kind'],
        selection=info['selection'], source_resolution=[64,64], control_timestep_seconds=.02,
        encoded_fps=25, frame_stride=2, full_seconds=20, excerpt_seconds=[8,18], playback_speed=1,
        task=task, target_action_step=target, actual_training_action_range=[lo,hi],
        excerpt_selection='Fixed 8-18 seconds, matching the previous walk display; full 20 seconds also delivered. No episode selection on reward or appearance.',
        ffmpeg_version=imageio_ffmpeg.get_ffmpeg_version(), ffmpeg_exe_sha256=sha(Path(ffmpeg)),
        output_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(output), videos=2, original_frames='1001/group', training_only=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('collect','render'))
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', choices=('dmc_quadruped_walk', 'dmc_quadruped_run'),
                        default='dmc_quadruped_walk')
    parser.add_argument('--action-step', type=int, default=250000)
    args = parser.parse_args()
    if args.mode == 'collect':
        if (args.task, args.action_step) not in (('dmc_quadruped_walk',250000),
                                                ('dmc_quadruped_run',500000)):
            parser.error('Only walk/250000 and run/500000 are supported')
        collect(args.output.resolve(), args.task, args.action_step)
    else:
        if args.source is None:
            parser.error('--source is required for render')
        render(args.source.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
