"""DL-code-r1: sv3 GPU4--7 identity, occupancy and CUDA/EGL guards.

Only standard-library imports. The probe command runs the existing zero-action
CUDA/EGL helper in a separate process; importing this module never uses a GPU.
An isolated paired check may hold a parent session, but every child still reads
fresh hardware occupancy before constructing its model. No process is stopped.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'DL-code-r1'
HOST = 'lyg0326'  # sv3 identity read live by root, 2026-10-10.
GPUS = (4, 5, 6, 7)


def now():
  return dt.datetime.now(dt.timezone.utc)


def digest(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('x', encoding='utf-8') as stream:
    json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write('\n')


def command_output(command):
  return subprocess.check_output(command, text=True, timeout=30)


def validate_binding(record, environ=None, hostname=None, system=None):
  environ = os.environ if environ is None else environ
  hostname = platform.node() if hostname is None else hostname
  system = platform.system() if system is None else system
  physical = record.get('physical_gpu')
  if system != 'Linux' or hostname.split('.')[0] != HOST:
    raise RuntimeError('DL GPU/render work is limited to the assigned sv3 Linux host')
  if record.get('actual_hostname') != hostname or record.get('logical_server') != 'sv3':
    raise ValueError('Resource record is not bound to this actual sv3 hostname')
  if type(physical) is not int or physical not in GPUS:
    raise ValueError('Only assigned sv3 physical GPU4--7 may be used')
  uuid = record.get('gpu_uuid')
  if not isinstance(uuid, str) or not uuid.startswith('GPU-') or ',' in uuid:
    raise ValueError('Resource record requires one GPU UUID')
  if record.get('compute_uuid') != uuid or record.get('egl_uuid') != uuid:
    raise ValueError('Resource record lacks verified equal CUDA/EGL UUIDs')
  expected = dict(CUDA_VISIBLE_DEVICES=uuid, MUJOCO_EGL_DEVICE_ID=str(physical),
      CUDA_DEVICE_ORDER='PCI_BUS_ID', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl')
  for key, value in expected.items():
    if environ.get(key) != value:
      raise ValueError(f'DL physical binding differs: {key}')
  if str(record.get('cuda_visible_devices')) != uuid or str(
      record.get('mujoco_egl_device_id')) != str(physical):
    raise ValueError('Record environment differs from the single physical device binding')


def validate_age(record, instant=None):
  instant = now() if instant is None else instant
  checked = dt.datetime.fromisoformat(record['checked_at_utc'].replace('Z', '+00:00'))
  if checked.tzinfo is None or not -5 <= (instant - checked).total_seconds() <= 120:
    raise ValueError('Initial resource record must be checked within 120 seconds')
  if record.get('occupancy') != 'idle' or record.get('exclusive') is not True:
    raise ValueError('Initial assigned device must be idle and exclusive')


def live_state(physical):
  if type(physical) is not int or physical not in GPUS:
    raise ValueError('Physical GPU outside assigned sv3 GPU4--7')
  csv = command_output(['nvidia-smi', '-i', str(physical),
      '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits']).strip()
  pieces = [x.strip() for x in csv.split(',')]
  if len(pieces) != 2 or not pieces[1].isdigit():
    raise RuntimeError('Actual GPU UUID/memory query is unavailable')
  uuid, memory = pieces[0], int(pieces[1])
  xml = command_output(['nvidia-smi', '-i', str(physical), '-q', '-x'])
  device = ET.fromstring(xml).find('gpu')
  if device is None or device.findtext('uuid') != uuid:
    raise RuntimeError('Actual XML GPU UUID differs from the queried physical device')
  section = device.find('processes')
  if section is None or (not section.findall('process_info') and
      (section.text or '').strip() not in ('', 'None')):
    raise RuntimeError('Actual compute/graphics process inventory is unavailable')
  processes = [dict(pid=int(item.findtext('pid')), type=item.findtext('type'))
      for item in section.findall('process_info')]
  pmon = command_output(['nvidia-smi', 'pmon', '-c', '1'])
  jobs, observed = [], False
  for line in pmon.splitlines():
    fields = line.split()
    if not fields or fields[0].startswith('#'):
      continue
    if fields[0] == str(physical) and len(fields) >= 3:
      observed = True
    if fields[0] == str(physical) and len(fields) >= 3 and fields[1] != '-':
      if not fields[1].isdigit():
        raise RuntimeError('Actual assigned-device pmon process inventory is unavailable')
      jobs.append(dict(pid=int(fields[1]), type=fields[2], row=line))
  if not observed:
    raise RuntimeError('Assigned GPU is absent from the actual pmon process inventory')
  return dict(checked_at_utc=now().isoformat(), physical_gpu=physical,
      gpu_uuid=uuid, memory_used_mib=memory, processes=processes,
      pmon_jobs=jobs, raw_pmon=pmon)


def require_idle(state, expected_uuid):
  if state['gpu_uuid'] != expected_uuid:
    raise RuntimeError('Live physical GPU UUID differs from the assigned device')
  if state['processes'] or state['pmon_jobs'] or state['memory_used_mib'] != 0:
    raise RuntimeError(f"Assigned GPU{state['physical_gpu']} has occupied/unknown work; "
        'no sharing, stopping, or device substitution is permitted')


def mapping_evidence(record):
  path = Path(record['device_probe_path']).resolve()
  if digest(path) != record.get('device_probe_sha256'):
    raise ValueError('Actual zero-action CUDA/EGL probe evidence hash differs')
  probe = json.loads(path.read_text(encoding='utf-8'))
  if (probe.get('physical_gpu') != record['physical_gpu'] or
      probe.get('uuid') != record['gpu_uuid'] or
      probe.get('verified_compute_and_graphics_same_gpu') is not True or
      probe.get('training_actions') != 0 or probe.get('agent_updates') != 0 or
      probe.get('logical_cuda_device') != 0):
    raise ValueError('CUDA/EGL evidence does not prove this physical binding')
  expected = dict(CUDA_VISIBLE_DEVICES=record['gpu_uuid'],
      MUJOCO_EGL_DEVICE_ID=str(record['physical_gpu']), CUDA_DEVICE_ORDER='PCI_BUS_ID',
      MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl')
  if any(probe.get('environment', {}).get(k) != v for k, v in expected.items()):
    raise ValueError('Probe ran with a different actual CUDA/EGL environment')
  if probe.get('source_sha256') != digest(ROOT / 'scripts/probe_formal_devices.py'):
    raise ValueError('Mapping probe code differs from the actual reviewed helper')
  return dict(path=str(path), sha256=digest(path), probe_pid=probe.get('pid'),
      start_utc=probe.get('start_utc'), end_utc=probe.get('end_utc'))


def refresh(record):
  """New live observation, never a timestamp-only refresh of stale evidence."""
  validate_binding(record)
  mapping = mapping_evidence(record)
  state = live_state(record['physical_gpu'])
  require_idle(state, record['gpu_uuid'])
  return dict(record, checked_at_utc=state['checked_at_utc'], occupancy='idle',
      exclusive=True, live_check=state, device_mapping_evidence=mapping)


def require_resource(path):
  record = json.loads(Path(path).read_text(encoding='utf-8'))
  validate_age(record)
  result = refresh(record)
  return dict(result, resource_source_path=str(Path(path).resolve()),
      resource_source_sha256=digest(path))


def verify_runtime(record, compute=True, graphics=False):
  """Read the initialized process's actual UUID; no allocation or rendering."""
  validate_binding(record)
  pid = os.getpid()
  output = command_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
      '--format=csv,noheader'])
  own_compute = []
  for line in output.splitlines():
    fields = [x.strip() for x in line.split(',')]
    if len(fields) == 2 and fields[0] == str(pid):
      own_compute.append(fields[1])
  if compute and own_compute != [record['gpu_uuid']]:
    raise RuntimeError('Initialized process CUDA context is not exclusively on the assigned UUID')
  if not compute and own_compute:
    raise RuntimeError('CPU diagnostic unexpectedly allocated a CUDA compute context')
  state = live_state(record['physical_gpu'])
  if state['gpu_uuid'] != record['gpu_uuid']:
    raise RuntimeError('Runtime physical UUID changed')
  if any(item['pid'] != pid for item in state['processes'] + state['pmon_jobs']):
    raise RuntimeError('Another process occupied the assigned GPU during initialization')
  own_pmon = []
  for line in state['raw_pmon'].splitlines():
    fields = line.split()
    if len(fields) >= 3 and fields[1] == str(pid):
      own_pmon.append(fields)
  if any(int(row[0]) != record['physical_gpu'] for row in own_pmon):
    raise RuntimeError('Runtime CUDA/EGL context appears on another physical GPU')
  if graphics and not any('G' in row[2] for row in own_pmon):
    raise RuntimeError('Actual initialized EGL graphics mapping is not visible')
  return dict(checked_at_utc=now().isoformat(), pid=pid, compute_uuid=own_compute,
      physical_gpu=record['physical_gpu'], pmon=state['raw_pmon'],
      graphics_verified=bool(graphics), live_state=state)


def start_session(resource_path, session_path):
  """Idle parent orchestrator; it must never initialize a CUDA/EGL context."""
  resource = require_resource(resource_path)
  session = dict(code_version=VERSION, type='DL-paired-resource-session',
      parent_pid=os.getpid(), actual_hostname=platform.node(),
      started_at_utc=now().isoformat(), resource=resource)
  write_json(session_path, session)
  return resource


def require_session(path):
  session = json.loads(Path(path).read_text(encoding='utf-8'))
  if (session.get('type') != 'DL-paired-resource-session' or
      session.get('actual_hostname') != platform.node() or
      session.get('parent_pid') != os.getppid()):
    raise ValueError('CUDA worker must be a direct child of its live paired session')
  os.kill(session['parent_pid'], 0)  # Only an existence check; no signal is sent.
  # The parent validated the fresh initial record. Each sequential child now
  # obtains a new real idle snapshot instead of reusing the old 120s check.
  result = refresh(session['resource'])
  return dict(result, session_source_path=str(Path(path).resolve()),
      session_source_sha256=digest(path), session_parent_pid=session['parent_pid'])


def probe(args):
  """Produce verified physical mapping then a fresh idle startup record."""
  if platform.system() != 'Linux' or platform.node().split('.')[0] != HOST:
    raise RuntimeError('No GPU/render probe is allowed outside assigned sv3')
  state = live_state(args.physical_gpu)
  require_idle(state, state['gpu_uuid'])
  record = dict(logical_server='sv3', actual_hostname=platform.node(),
      physical_gpu=args.physical_gpu, gpu_uuid=state['gpu_uuid'],
      compute_uuid=state['gpu_uuid'], egl_uuid=state['gpu_uuid'],
      cuda_visible_devices=state['gpu_uuid'], mujoco_egl_device_id=str(args.physical_gpu),
      checked_at_utc=state['checked_at_utc'], occupancy='idle', exclusive=True)
  validate_binding(record)
  output = Path(args.output).resolve()
  if output.exists():
    raise FileExistsError(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  evidence = output.with_name(output.stem + '-device-probe.json')
  subprocess.run([sys.executable, str(ROOT / 'scripts/probe_formal_devices.py'),
      '--gpu', str(args.physical_gpu), '--uuid', state['gpu_uuid'],
      '--output', str(evidence)], check=True, timeout=180)
  record.update(device_probe_path=str(evidence), device_probe_sha256=digest(evidence),
      code_version=VERSION)
  record = refresh(record)
  write_json(output, record)
  print(json.dumps(record, sort_keys=True), flush=True)


def record(args):
  """Reuse an actual zero-action proof, but query fresh real idleness now."""
  if platform.system() != 'Linux' or platform.node().split('.')[0] != HOST:
    raise RuntimeError('No GPU resource check is allowed outside assigned sv3')
  state = live_state(args.physical_gpu)
  require_idle(state, state['gpu_uuid'])
  evidence = Path(args.device_probe).resolve()
  value = dict(code_version=VERSION, logical_server='sv3', actual_hostname=platform.node(),
      physical_gpu=args.physical_gpu, gpu_uuid=state['gpu_uuid'],
      compute_uuid=state['gpu_uuid'], egl_uuid=state['gpu_uuid'],
      cuda_visible_devices=state['gpu_uuid'], mujoco_egl_device_id=str(args.physical_gpu),
      checked_at_utc=state['checked_at_utc'], occupancy='idle', exclusive=True,
      device_probe_path=str(evidence), device_probe_sha256=digest(evidence))
  value = refresh(value)
  write_json(args.output, value)
  print(json.dumps(value, sort_keys=True), flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest='command', required=True)
  create = sub.add_parser('probe')
  create.add_argument('--physical-gpu', type=int, choices=GPUS, required=True)
  create.add_argument('--output', required=True)
  create.set_defaults(function=probe)
  check = sub.add_parser('record')
  check.add_argument('--physical-gpu', type=int, choices=GPUS, required=True)
  check.add_argument('--device-probe', required=True)
  check.add_argument('--output', required=True)
  check.set_defaults(function=record)
  args = parser.parse_args()
  args.function(args)


if __name__ == '__main__':
  main()
