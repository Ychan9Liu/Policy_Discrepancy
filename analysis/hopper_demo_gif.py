"""Six real hopper_hop replay clips in a seamless 3-row, 2-column task GIF.

Read existing seed0 baseline artifacts only. No policy/environment execution,
training, GPU use, server writes, or method comparison.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[1]
REMOTE = r'''
import datetime,hashlib,io,json,pathlib,sys,tarfile
import numpy as np
P=pathlib.Path
index_path=P('/data/Policy_Discrepancy/repo/analysis/outputs/formal-four-groups-pd1-20261010-final-attempt01/RUN_INDEX.json')
index=json.loads(index_path.read_text())
row=next(r for r in index['runs'] if r['task']=='dmc_hopper_hop' and r['group']=='baseline')
directory=P(row['directory'])
manifest=json.loads((directory/'protocol_manifest.json').read_text())
final=json.loads((directory/'final_state.json').read_text())
assert manifest['git_commit']==row['code_sha']=='48d71f1541721208322442f5a1d5977124f3e888'
assert manifest['config']['seed']==0 and manifest['config']['task']=='dmc_hopper_hop'
assert not manifest['config']['run.engineering_fixture']
assert final['train_action_steps']==1000000 and final['evaluations']==21

def ahash(a):
 h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()

proof=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 task='dmc_hopper_hop',kind='task illustration from existing train replay; not a method comparison or fixed-checkpoint evaluation',
 server='sv2',directory=str(directory),run_id=row['run_id'],training_sha=manifest['git_commit'],
 seed=0,group='baseline',source_index=str(index_path),source_index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
 input_sha256={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in ('protocol_manifest.json','final_state.json','evaluations.jsonl')},
 selection='First six chronologically saved root chains; last complete episode per chain at or before 1M actions; fixed 2-8 second excerpt. No selection on reward or appearance.',
 chain_order_basis='Chronological first-chunk creation in ordered Driver callbacks, matching the prior hopper inspection; not an independently stored worker ID.',
 source_resolution=[64,64],control_timestep_seconds=.02,source_frame_stride=2,encoded_fps=25,
 excerpt_seconds=[2,8],source_frame_indices=list(range(100,400,2)),replay_scalar_inputs=[],clips=[])
paths={p.stem.split('-')[1]:p for p in (directory/'replay').glob('*.npz')}
successors={p.stem.split('-')[2] for p in paths.values()}
roots=sorted((p for key,p in paths.items() if key not in successors),key=lambda p:p.name)
assert len(roots)==16
chains=[];path_chains=[];visited=set()
for chain_id,root in enumerate(roots):
 parts=[];chunks=[];path=root;cursor=0
 while path is not None:
  identity,successor=path.stem.split('-')[1:3];assert identity not in visited;visited.add(identity)
  with np.load(path,allow_pickle=False) as z:fields={key:z[key] for key in ('reward','is_first','is_last')}
  n=int(path.stem.split('-')[3]);assert all(len(v)==n for v in fields.values())
  proof['replay_scalar_inputs'].append(dict(chain=chain_id,path=str(path),start_index=cursor,length=n,array_sha256={k:ahash(v) for k,v in fields.items()}))
  chunks.append((cursor,cursor+n,path));cursor+=n;parts.append(fields);path=paths.get(successor)
 chain={k:np.concatenate([part[k] for part in parts]) for k in parts[0]}
 chains.append(chain);path_chains.append(chunks)
assert len(visited)==len(paths)
assert sum(len(c['reward']) for c in chains)==final['train_action_steps']+final['train_resets']
assert sum(int(c['is_first'].sum()) for c in chains)==final['train_resets']
cursors=[0]*16;stamps=[[] for _ in chains];actions=0
while actions<1000000:
 boundary=(actions//50000+1)*50000
 for worker in range(min(16,boundary-actions)):
  actions+=not bool(chains[worker]['is_first'][cursors[worker]])
  stamps[worker].append(actions);cursors[worker]+=1
assert cursors==[len(c['reward']) for c in chains]
arrays={}
for chain_id in range(6):
 chain=chains[chain_id];start=0;episodes=[]
 for ordinal,end in enumerate(np.flatnonzero(chain['is_last'])):
  end=int(end);assert chain['is_first'][start] and not chain['is_first'][start+1:end+1].any()
  episodes.append(dict(ordinal=ordinal,start_index=start,end_index=end,length=end-start,
   first_global_action=stamps[chain_id][start+1],last_global_action=stamps[chain_id][end],
   return_raw=float(chain['reward'][start+1:end+1].sum(dtype=np.float64))))
  start=end+1
 episode=episodes[-1];assert episode['length']==1000
 indices=episode['start_index']+np.arange(100,400,2)
 frames=[];image_sources=[]
 for lo,hi,path in path_chains[chain_id]:
  use=indices[(indices>=lo)&(indices<hi)]
  if not len(use):continue
  with np.load(path,allow_pickle=False) as z:images=z['image'];part=images[use-lo].copy()
  frames.append(part);image_sources.append(dict(path=str(path),local_indices=(use-lo).tolist(),
   full_image_array_sha256=ahash(images),selected_frames_sha256=ahash(part)))
 frames=np.concatenate(frames);assert frames.shape==(150,64,64,3) and frames.dtype==np.uint8
 rewards=chain['reward'][episode['start_index']:episode['end_index']+1]
 arrays[f'clip{chain_id}']=frames;arrays[f'clip{chain_id}_episode_rewards']=rewards
 proof['clips'].append(dict(clip_id=f'clip{chain_id}',row=chain_id//2+1,column=chain_id%2+1,
  chain=chain_id,episode=episode,frames_sha256=ahash(frames),reward_array_sha256=ahash(rewards),
  excerpt_first_global_action=stamps[chain_id][int(indices[0])],excerpt_last_global_action=stamps[chain_id][int(indices[-1])],
  image_sources=image_sources))
buffer=io.BytesIO();np.savez_compressed(buffer,**arrays)
records={'inspection.json':json.dumps(proof,ensure_ascii=False,indent=2).encode(),
 'replay_clips.npz':buffer.getvalue()}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for name,data in records.items():
  item=tarfile.TarInfo(name);item.size=len(data);tar.addfile(item,io.BytesIO(data))
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ahash(array):
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def collect(output):
    output.mkdir(parents=True, exist_ok=False)
    (output/'remote_source.py').write_text(REMOTE, encoding='utf-8')
    archive = output/'inputs.tar.gz'
    with archive.open('xb') as stream, (output/'ssh.log').open('xb') as log:
        subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'sv2',
            "CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            '/data/Policy_Discrepancy/envs/dreamer/bin/python -'],
            input=REMOTE.encode(), stdout=stream, stderr=log, check=True)
    with tarfile.open(archive) as stream:
        stream.extractall(output, filter='data')
    info = json.loads((output/'inspection.json').read_text(encoding='utf-8'))
    print(json.dumps(dict(output=str(output), clips=len(info['clips']),
                         episodes=[r['episode'] for r in info['clips']]), ensure_ascii=False))


def render(source, output):
    import numpy as np
    from PIL import Image, ImageSequence, __version__ as pillow_version

    info = json.loads((source/'inspection.json').read_text(encoding='utf-8'))
    assert info['task'] == 'dmc_hopper_hop' and info['group'] == 'baseline' and info['seed'] == 0
    assert len(info['clips']) == 6
    # Cross-check source receipts with the already accepted original inspection.
    original = ROOT/'analysis/outputs/hopper-diagnosis-20261020-inspection01/remote_inspection.json'
    old = json.loads(original.read_text(encoding='utf-8'))
    old_run = next(r for r in old['runs'] if r['group'] == 'baseline')
    assert info['run_id'] == old_run['run_id']
    for name, digest in info['input_sha256'].items():
        assert digest == old_run['input_sha256'][name]
    data = np.load(source/'replay_clips.npz', allow_pickle=False)
    clips = []
    for item in info['clips']:
        frames = data[item['clip_id']]
        rewards = data[item['clip_id']+'_episode_rewards']
        assert frames.shape == (150,64,64,3) and ahash(frames) == item['frames_sha256']
        assert ahash(rewards) == item['reward_array_sha256']
        assert abs(float(rewards[1:].sum(dtype=np.float64))-item['episode']['return_raw']) < 1e-9
        clips.append(frames)
    output.mkdir(parents=True, exist_ok=False)
    frames = []
    for index in range(150):
        mosaic = np.concatenate([np.concatenate([clips[2*row][index], clips[2*row+1][index]], axis=1)
                                  for row in range(3)], axis=0)
        assert mosaic.shape == (192,128,3)
        for item in info['clips']:
            y, x = (item['row']-1)*64, (item['column']-1)*64
            assert np.array_equal(mosaic[y:y+64,x:x+64], clips[item['chain']][index])
        frames.append(Image.fromarray(mosaic).resize((512,768), Image.Resampling.NEAREST))
    # One fixed 256-color palette across the animation avoids palette flicker.
    palette_sample = np.concatenate([np.asarray(frames[i])[::4,::4] for i in range(0,150,5)], axis=0)
    palette = Image.fromarray(palette_sample).quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    quantized = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    gif = output/'hopper_hop_任务示意_3x2.gif'
    quantized[0].save(gif, save_all=True, append_images=quantized[1:],
                      duration=40, loop=0, optimize=False, disposal=2)
    with Image.open(gif) as decoded:
        assert decoded.size == (512,768) and decoded.n_frames == 150 and decoded.info['loop'] == 0
        duration = 0
        for i, frame in enumerate(ImageSequence.Iterator(decoded)):
            duration += frame.info['duration']
            assert np.array_equal(np.asarray(frame.convert('RGB')), np.asarray(quantized[i].convert('RGB')))
        assert duration == 6000
    error_means, error_maxima = [], []
    for a, b in zip(frames,quantized):
        error = np.abs(np.asarray(a,dtype=np.int16)-np.asarray(b.convert('RGB'),dtype=np.int16))
        error_means.append(float(error.mean())); error_maxima.append(int(error.max()))
    still_indices = (0,50,100)
    sheet = Image.new('RGB',(512*3,768))
    for column,index in enumerate(still_indices):
        still = quantized[index].convert('RGB')
        still.save(output/f'preview_{2+index/25:.2f}s.png')
        sheet.paste(still,(column*512,0))
    sheet.save(output/'contact_sheet.png')
    for i, frame in enumerate(frames):
        # Lossless references retain colors before GIF's required quantization.
        if i in still_indices:
            frame.save(output/f'lossless_reference_{2+i/25:.2f}s.png')
    record = dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        script_sha256=sha(Path(__file__)), source_hashes={p.name:sha(p) for p in source.iterdir() if p.is_file()},
        original_inspection_sha256=sha(original), source_input_hashes_match_prior_inspection=True,
        kind=info['kind'], selection=info['selection'], task=info['task'],
        grid=dict(rows=3,columns=2,cell_pixels=[256,256],output_pixels=[512,768],
                  horizontal_gap=0,vertical_gap=0,border=0,labels=False),
        playback=dict(excerpt_seconds=[2,8],frames=150,frame_duration_ms=40,fps=25,
                      total_ms=6000,speed=1,loop=0,synchronization='same episode-relative times'),
        no_training_or_inference_or_environment_or_gpu=True,
        checks=dict(six_source_array_hashes=True,episode_reward_sums=True,
                    every_unquantized_tile_matches_source_pixels=True,
                    all_gif_frames_decode_to_expected_quantized_pixels=True,
                    frame_durations_and_loop=True),
        quantization=dict(colors=256,palette='fixed global median-cut, no dithering',
                          mean_absolute_rgb_error=float(np.mean(error_means)),
                          max_absolute_rgb_error=max(error_maxima)),
        runtime=dict(numpy=np.__version__,pillow=pillow_version),
        clips=info['clips'],output_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(output),gif=str(gif),grid='3 rows x 2 columns',frames=150,seconds=6,gaps=0),ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('collect','render'))
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'collect':
        collect(args.output.resolve())
    else:
        if args.source is None:
            parser.error('--source is required for render')
        render(args.source.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
