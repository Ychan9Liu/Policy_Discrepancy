"""No-agent CUDA/EGL probe for the allocated physical GPU; zero actions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, required=True)
parser.add_argument('--uuid', required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
assert not args.output.exists()
assert os.environ['CUDA_VISIBLE_DEVICES'] == args.uuid
assert os.environ['MUJOCO_EGL_DEVICE_ID'] == str(args.gpu)
xml = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '-q', '-x'], text=True)
device = ET.fromstring(xml).find('gpu')
assert device is not None and device.findtext('uuid') == args.uuid
assert device.find('processes') is not None
assert not device.findall('processes/process_info')
before = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
    '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'], text=True).strip()
record = dict(start_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
    physical_gpu=args.gpu, uuid=args.uuid, memory_before=before,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    environment={k: os.environ[k] for k in ('CUDA_VISIBLE_DEVICES',
        'MUJOCO_EGL_DEVICE_ID', 'MUJOCO_GL', 'PYOPENGL_PLATFORM', 'CUDA_DEVICE_ORDER')})
import jax
import jax.numpy as jnp
assert len(jax.devices()) == 1 and jax.devices()[0].platform == 'gpu'
assert int(jnp.arange(4).sum().block_until_ready()) == 6
from dm_control import mujoco
physics = mujoco.Physics.from_xml_string(
    '<mujoco><worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>')
image = physics.render(height=64, width=64, camera_id=-1)
compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
    '--format=csv,noheader'], text=True)
pmon = subprocess.check_output(['nvidia-smi', 'pmon', '-c', '1'], text=True)
own_compute = [x for x in compute.splitlines() if x.split(',')[0].strip() == str(os.getpid())]
own_pmon = [x.split() for x in pmon.splitlines() if len(x.split()) >= 3
           and x.split()[1] == str(os.getpid())]
assert len(own_compute) == 1 and args.uuid in own_compute[0]
assert own_pmon and {int(x[0]) for x in own_pmon} == {args.gpu}
assert any('G' in x[2] for x in own_pmon)
record.update(end_utc=datetime.now(timezone.utc).isoformat(),
    training_actions=0, agent_updates=0, logical_cuda_device=0,
    compute_apps=compute, pmon=pmon,
    image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
    verified_compute_and_graphics_same_gpu=True)
args.output.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(dict(gpu=args.gpu, uuid=args.uuid, passed=True, training_actions=0)))
physics.free()
