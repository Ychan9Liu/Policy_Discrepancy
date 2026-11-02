"""Four requested comparisons from existing replay; CPU read-only, no inference.

Select the first saved chain's last complete episode before the requested step.
Reuse the previously verified replay-chain reconstruction, with per-task timing.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

from analysis.walk_replay_media_v1 import REMOTE as PREVIOUS_REMOTE

TASKS = {'dmc_cartpole_swingup': (300000, .01), 'dmc_finger_spin': (1000000, .02),
         'dmc_pendulum_swingup': (1000000, .02), 'dmc_walker_run': (1000000, .025)}
ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ahash(a):
    h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()


def dump(p,v):
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def remote_source(row, target):
    # Reuse exactly the existing scalar-chain and global action reconstruction.
    start=PREVIOUS_REMOTE.index('    files = {p.stem')
    end=PREVIOUS_REMOTE.index('    media = ',start)
    core='\n'.join(line[4:] if line.startswith('    ') else line for line in PREVIOUS_REMOTE[start:end].splitlines())
    return '''import json,hashlib,pathlib,io,sys,tarfile,datetime
import numpy as np
P=pathlib.Path
''' + 'row='+repr(row)+'\nlimit='+str(target)+'''\n
directory=P(row['directory']);group=row['group']
manifest=json.loads((directory/'protocol_manifest.json').read_text())
assert manifest['git_commit']==row['sha'] and not manifest['config']['run.engineering_fixture']
assert manifest['config']['batch_size']>0 and manifest['config']['env.dmc.repeat']==1
for name,digest in row['hashes'].items():
 if name in ('protocol_manifest.json','evaluations.jsonl','final_state.json'):
  assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest
evaluation=next(json.loads(line) for line in (directory/'evaluations.jsonl').read_text().splitlines() if json.loads(line)['action_step']==limit)
arrays={}
def digest_array(a):
 h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()
''' + core + '''
proof=dict(row=row,target_action_step=limit,collected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 selected_episode=selected,selected_frames_sha256=digest_array(frames),image_sources=source_images,
 replay_scalar_inputs=proof_chunks,evaluation=evaluation,
 selection='First chronological root chain; last complete episode before target; no reward/appearance selection',
 kind='Existing training replay; changing policy; not checkpoint evaluation')
buf=io.BytesIO();np.savez_compressed(buf,**arrays)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for name,data in [('inspection.json',json.dumps(proof).encode()),('frames.npz',buf.getvalue())]:
  item=tarfile.TarInfo(name);item.size=len(data);tar.addfile(item,io.BytesIO(data))
'''


def collect(source,output):
    output.mkdir(parents=True,exist_ok=False)
    rows=json.loads((source/'run_manifest.json').read_text(encoding='utf-8'))
    selected=[r for r in rows if r['task'] in TASKS];assert len(selected)==8
    dump(output/'source.json',dict(path=str(source.resolve()),manifest_sha256=sha(source/'run_manifest.json')))
    def one(row):
        folder=output/(row['task']+'-'+row['group']);folder.mkdir()
        code=remote_source(row,TASKS[row['task']][0]);(folder/'remote_source.py').write_text(code,encoding='utf-8')
        with (folder/'inputs.tar.gz').open('xb') as out,(folder/'ssh.log').open('xb') as log:
            subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',row['server'],
                            "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/Policy_Discrepancy/envs/dreamer/bin/python -"],
                           input=code.encode(),stdout=out,stderr=log,check=True)
        with tarfile.open(folder/'inputs.tar.gz') as tar:tar.extractall(folder,filter='data')
        print(row['task'],row['group'],'collected',flush=True)
    with concurrent.futures.ThreadPoolExecutor(3) as pool:list(pool.map(one,selected))


def render(source,output):
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont,ImageSequence
    import imageio_ffmpeg
    output.mkdir(parents=True,exist_ok=False)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
    reports=[]
    for task,(target,dt) in TASKS.items():
        pair=[];proof=[]
        for group in ('baseline','dt'):
            folder=source/(task+'-'+group);info=json.loads((folder/'inspection.json').read_text())
            z=np.load(folder/'frames.npz',allow_pickle=False);frames=z[group+'_frames'];reward=z[group+'_reward']
            assert frames.shape==(1001,64,64,3) and ahash(frames)==info['selected_frames_sha256']
            assert abs(float(reward[1:].sum(dtype=np.float64))-info['selected_episode']['return_raw'])<1e-9
            assert info['selected_episode']['last_global_action']<=target
            pair.append(frames);proof.append(info)
        # 20 fps at real speed. Nearest existing frame; no interpolation.
        indices=np.rint(np.arange(round(1000*dt*20))/(20*dt)).astype(int)
        rendered=[]
        for idx in indices:
            im=Image.new('RGB',(640,414),'white');draw=ImageDraw.Draw(im)
            draw.text((320,8),f'{task.removeprefix("dmc_")} | {target//1000}k target',font=font,anchor='mt',fill='#111111')
            draw.text((320,33),'Training replay | seed0 | 1x speed',font=small,anchor='mt',fill='#555555')
            for i,(label,frames,info) in enumerate(zip(('baseline','PPD-KL'),pair,proof)):
                x=i*320
                draw.text((x+160,56),label,font=font,anchor='mt',fill=('#4b5563','#c2410c')[i])
                im.paste(Image.fromarray(frames[idx]).resize((320,320),Image.Resampling.NEAREST),(x,80))
                e=info['selected_episode']
                draw.text((x+160,400),f"{e['first_global_action']:,}-{e['last_global_action']:,} actions",font=small,anchor='mt',fill='#333333')
            rendered.append(im)
        palette=Image.fromarray(np.concatenate([np.asarray(f)[::5,::5] for f in rendered[::10]],axis=0)).quantize(colors=256)
        quantized=[im.quantize(palette=palette,dither=Image.Dither.NONE) for im in rendered]
        stem=task.removeprefix('dmc_')+'_'+str(target//1000)+'k'
        gif=output/(stem+'.gif')
        quantized[0].save(gif,save_all=True,append_images=quantized[1:],duration=50,loop=0,optimize=False,disposal=2)
        with Image.open(gif) as decoded:
            duration=0
            for i,f in enumerate(ImageSequence.Iterator(decoded)):
                assert np.array_equal(np.asarray(f.convert('RGB')),np.asarray(quantized[i].convert('RGB')))
                duration+=f.info['duration']
            assert i+1==len(indices) and duration==round(1000*dt*1000)
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();mp4=output/(stem+'.mp4')
        cmd=[ffmpeg,'-nostdin','-loglevel','error','-n','-f','rawvideo','-pixel_format','rgb24','-video_size','640x414','-framerate','20','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(mp4)]
        with subprocess.Popen(cmd,stdin=subprocess.PIPE) as proc:
            for f in rendered:proc.stdin.write(f.tobytes())
            proc.stdin.close();assert proc.wait()==0
        subprocess.run([ffmpeg,'-v','error','-i',str(mp4),'-f','null','-'],check=True)
        sheet=Image.new('RGB',(640*3,414))
        for j,index in enumerate((0,len(rendered)//2,len(rendered)-1)):sheet.paste(rendered[index],(j*640,0))
        sheet.save(output/(stem+'_contact.png'))
        reports.append(dict(task=task,target=target,control_timestep=dt,seconds=1000*dt,frames=len(indices),indices=indices.tolist(),
                            selected=[r['selected_episode'] for r in proof],evaluation_means=[r['evaluation']['mean_return_raw'] for r in proof],
                            source_inspection_sha256=[sha(source/(task+'-'+g)/'inspection.json') for g in ('baseline','dt')],
                            gif_sha256=sha(gif),mp4_sha256=sha(mp4),gif_pixel_decode_verified=True,mp4_full_decode_passed=True))
        print(stem,'rendered',flush=True)
    dump(output/'manifest.json',dict(git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         script_sha256=sha(Path(__file__)),source=str(source.resolve()),runs=reports))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['collect','render']);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();(collect if a.mode=='collect' else render)(a.source,a.output)
