"""DL-native-render-identity-r1: four fresh resets, twelve RGB, zero actions.

The parent imports only the standard library. Actual environment/render calls
exist only in authenticated owned workers on the assigned sv3 GPU4. This is an
engineering diagnostic, never a policy or method effectiveness experiment.
"""
import time
ENTRY_STARTED = time.monotonic()
import os
import signal
import sys
OWNED = set()
SECONDS = 180
DEADLINE = ENTRY_STARTED + SECONDS


def expire(signum, frame):
  for pid in tuple(OWNED):
    try: os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError: pass
  os._exit(124)


if __name__ == '__main__' and os.name == 'posix':
  signal.signal(signal.SIGALRM, expire)
  signal.signal(signal.SIGTERM, expire)
  signal.setitimer(signal.ITIMER_REAL, max(.001, DEADLINE-time.monotonic()))

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import re
import socket
import struct
import subprocess
import zlib

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'DL-native-render-identity-r1'
UUID = 'GPU-9c070d5a-ac68-4c26-9522-a790ad874b23'
SEED, CAMERA, WORKERS, MODELS, RENDERS = 1024, 2, 2, 2, 3
PINNED = {
  'dm_control/suite/quadruped.py': '154662d7fb3ab576a013dbf3741ffb9d83009d791309fb50f2d33d9d5b04c91f',
  'dm_control/suite/quadruped.xml': 'c6b0ade5f3160b155bbe673aec3e0430eed3273ae13b60880855647e82bf8334',
  'dm_control/suite/common/skybox.xml': '71db77c7c88dd4e1b54eb394deecfba673bbf539d110de68aa229dee00abf107',
  'dm_control/suite/common/materials.xml': '83ee4f7d3f0c9b3003ad121c2c35169a32a922c437321278f27db60d577ccfac',
  'dm_control/suite/common/visual.xml': '1e338ca511de35b902c4ca944d857c23fe4e7a5aea46a475cdc317dd454245b0',
  'dm_control/mujoco/wrapper/core.py': '4fd370e8ba9e4e8a871c65e7a19c091a7efecdb6e8f569131ef1ed3dc0422de6',
  'dm_control/mujoco/engine.py': 'ea0f71b4d79d802d2eda50a7320fa876be856798231d755f04966adc3c1e8e3d',
  'dm_control/_render/pyopengl/egl_renderer.py': '3fa55e7628b21be590dcac56cc811b5577af4b812c24c862c987514daed59be0',
  'mujoco/libmujoco.so.3.15.0': 'd3a1476dd52e220031aca40b2b00f69d1febce2ee85c57d5015f68b0cc72afa9',
}
DERIVED = ('xpos', 'xmat', 'geom_xpos', 'geom_xmat', 'site_xpos',
           'site_xmat', 'cam_xpos', 'cam_xmat', 'light_xpos', 'light_xdir')


def require(condition, message):
  if not condition: raise ValueError(message)


def check():
  require(time.monotonic() < DEADLINE, 'Whole 180-second entry expired')


def sha(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
  return digest.hexdigest()


def read_json(path):
  def pairs(items):
    result = {}
    for key, value in items:
      require(key not in result, 'Duplicate JSON key'); result[key] = value
    return result
  return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs,
      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def write_json(path, value):
  with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write('\n')


def cleanup(child):
  if os.name == 'posix':
    try: os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError: pass
  elif child.poll() is None: child.kill()
  child.wait(timeout=5); OWNED.discard(child.pid)


def spawn_owned(command, deadline, own_group=True, **kwargs):
  require(time.monotonic()<deadline,'Late before owned process creation')
  if os.name!='posix':
    child=subprocess.Popen(command,**kwargs)
    if own_group: OWNED.add(child.pid)
    return child
  previous=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM,signal.SIGTERM})
  try:
    child=subprocess.Popen(command,start_new_session=own_group,
        preexec_fn=lambda:signal.pthread_sigmask(signal.SIG_SETMASK,previous),**kwargs)
    if own_group: OWNED.add(child.pid)
    return child
  finally: signal.pthread_sigmask(signal.SIG_SETMASK,previous)


def owned_output(command):
  check()
  own_group = os.name == 'posix' and 'DL_NATIVE_RENDER_PARENT' not in os.environ
  child = spawn_owned(command,DEADLINE,own_group=own_group,cwd=ROOT,stdout=subprocess.PIPE,
      stderr=subprocess.PIPE)
  try:
    out, err = child.communicate(timeout=min(30, max(.001, DEADLINE-time.monotonic())))
    require(child.returncode == 0, 'Metadata helper failed: '+err.decode(errors='replace')[:256])
    check(); return out
  finally:
    if own_group: cleanup(child)
    elif child.poll() is None: child.kill(); child.wait(timeout=5)


def git_binding(expected):
  require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{40}', expected), 'Full SHA required')
  require(owned_output(['git', 'rev-parse', 'HEAD']).decode().strip() == expected, 'HEAD differs')
  require(not owned_output(['git', 'status', '--porcelain', '--untracked-files=all']), 'Clean checkout required')
  names = owned_output(['git', 'ls-files']).decode().splitlines()
  names = [x for x in names if x.endswith(('.py','.yaml','.yml','.lock')) or
      x in ('AGENTS.md','docs/DL_NATIVE_RENDER_IDENTITY.md','docs/DL_REP_NATIVE_ASSAY.md')]
  require('scripts/dl_native_render_identity.py' in names and
      'tests/test_dl_native_render_identity.py' in names, 'Tracked diagnostic incomplete')
  pins = {}
  for name in names:
    blob = owned_output(['git', 'show', expected+':'+name])
    require((ROOT/name).read_bytes() == blob, 'Checkout/blob differs: '+name)
    pins[name] = hashlib.sha256(blob).hexdigest()
  return pins


def installed_binding():
  require(importlib.metadata.version('dm_control') == '1.0.48' and
      importlib.metadata.version('mujoco') == '3.15.0', 'Installed official versions differ')
  result = {}
  for name, expected in PINNED.items():
    package = 'dm_control' if name.startswith('dm_control/') else 'mujoco'
    path = Path(importlib.metadata.distribution(package).locate_file(name)).resolve(strict=True)
    require(sha(path) == expected, 'Installed official bytes differ: '+name)
    result[name] = dict(path=str(path), sha256=expected)
  return result


def array_record(value):
  import numpy as np
  require(isinstance(value, (np.ndarray,np.generic,bool,int,float)), 'Numeric leaf required')
  value = np.asarray(value)
  require(value.dtype.kind in 'bifu' and np.isfinite(value).all(), 'Finite numeric dtype required')
  value = value.copy()
  return value, dict(dtype=value.dtype.str, shape=list(value.shape),
      sha256=hashlib.sha256(value.tobytes()).hexdigest())


def capture_object(obj, prefix, leaves, unknown, depth=0, names=None):
  """Read public numeric fields only; never call a discovered method."""
  import numpy as np
  for name in sorted(names if names is not None else [x for x in dir(obj) if not x.startswith('_')]):
    key = prefix+'/'+name
    try: value = getattr(obj, name)
    except Exception as error:
      unknown.append(dict(field=key, reason='getattr:'+type(error).__name__)); continue
    if callable(value): continue
    if isinstance(value, (np.ndarray,np.generic,bool,int,float)):
      try: leaves[key] = array_record(value)[0]
      except ValueError:
        unknown.append(dict(field=key,reason='unmeasured_numeric:'+str(np.asarray(value).dtype)))
    elif isinstance(value, bytes): leaves[key] = np.frombuffer(value, np.uint8).copy()
    elif isinstance(value, str): unknown.append(dict(field=key, reason='text:'+value))
    elif depth < 4 and type(value).__module__.startswith(('mujoco','dm_control')):
      capture_object(value, key, leaves, unknown, depth+1)
    else: unknown.append(dict(field=key, reason='unmeasured:'+type(value).__name__))


def leaf_manifest(leaves):
  return {key: array_record(value)[1] for key,value in sorted(leaves.items())}


def manifest_digest(manifest):
  return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',',':'),
      allow_nan=False).encode()).hexdigest()


def capture_model(physics):
  leaves, unknown = {}, []
  capture_object(physics.model.ptr, 'model', leaves, unknown)
  for prefix in ('tex_', 'mat_', 'cam_', 'light_'):
    require(any(key.startswith('model/'+prefix) for key in leaves), 'Missing model '+prefix)
  require(any(key.startswith('model/vis/') for key in leaves), 'Missing visual numeric fields')
  binary = physics.model.to_bytes()
  require(type(binary) is bytes and len(binary) > 0, 'Official MJB bytes required')
  return dict(mjb=binary, leaves=leaves, manifest=leaf_manifest(leaves), unknown=unknown,
      camera_id=CAMERA, camera_name=physics.model.id2name(CAMERA, 'camera'))


def capture_derived(physics):
  leaves,unknown={},[]
  capture_object(physics.data,'derived',leaves,unknown,names=DERIVED)
  return dict(leaves=leaves,manifest=leaf_manifest(leaves),unknown=unknown)


def capture_scene(camera):
  scene, leaves, unknown = camera._scene, {}, []
  require(camera._width==64 and camera._height==64 and camera._render_camera.fixedcamid==CAMERA,
      'Actual Camera dimensions/binding differ')
  excluded = {'geoms','lights','camera','flexedge','flexvert','flexface','skinvert','skinnormal'}
  capture_object(scene, 'scene', leaves, unknown,
      names=[x for x in dir(scene) if not x.startswith('_') and x not in excluded])
  require(type(int(scene.ngeom)) is int and 0 <= int(scene.ngeom) <= int(scene.maxgeom), 'Bad active scene geometry')
  for i in range(int(scene.ngeom)): capture_object(scene.geoms[i], f'geom/{i}', leaves, unknown)
  for i in range(int(scene.nlight)): capture_object(scene.lights[i], f'light/{i}', leaves, unknown)
  for i in range(2): capture_object(scene.camera[i], f'camera/{i}', leaves, unknown)
  capture_object(camera._render_camera, 'render_camera', leaves, unknown)
  capture_object(camera._scene_option, 'scene_option', leaves, unknown)
  capture_object(camera._perturb, 'perturb', leaves, unknown)
  unknown.append(dict(field='GPU/context/upload/framebuffer/draw/readback', reason='not authenticated as full rendering input'))
  return dict(leaves=leaves, manifest=leaf_manifest(leaves), unknown=unknown,
      complete_scene_authenticated=False)


@contextlib.contextmanager
def camera_capture(engine, captures):
  """Only read actual Camera inputs/buffers; flags and renderer calls unchanged."""
  original_update, original_render = engine.Camera.update, engine.Camera.render
  def update(camera, *args, **kwargs):
    result = original_update(camera, *args, **kwargs)
    camera._dl_identity_scene = capture_scene(camera)
    camera._dl_identity_before_buffer = hashlib.sha256(camera._rgb_buffer.tobytes()).hexdigest()
    return result
  def render(camera, *args, **kwargs):
    import numpy as np
    result = original_render(camera, *args, **kwargs)
    require(hasattr(camera, '_dl_identity_scene'), 'Actual scene hook was not reached')
    captures.append(dict(scene=camera._dl_identity_scene,
        old_buffer_sha256=camera._dl_identity_before_buffer,
        post_buffer_sha256=hashlib.sha256(camera._rgb_buffer.tobytes()).hexdigest(),
        returned_sha256=hashlib.sha256(result.tobytes()).hexdigest(),
        returned_shares_buffer=bool(np.shares_memory(result,camera._rgb_buffer))))
    return result
  require(not hasattr(engine.Camera.update,'_dl_identity_patch'), 'Nested render hook forbidden')
  update._dl_identity_patch = True
  engine.Camera.update, engine.Camera.render = update, render
  try: yield
  finally: engine.Camera.update, engine.Camera.render = original_update, original_render


def snapshot_bytes(snapshot):
  import numpy as np
  require(isinstance(snapshot,dict) and set(snapshot)=={'suite','from_dm_done'}, 'Snapshot schema differs')
  suite = snapshot['suite']
  require(isinstance(suite,dict) and set(suite)=={'integration','task','step_count','reset_next_step'}, 'Suite snapshot schema differs')
  integration = suite['integration']
  require(isinstance(integration,np.ndarray) and integration.dtype==np.float64 and
      integration.ndim==1 and integration.size and np.isfinite(integration).all(), 'Full F64 integration required')
  require(isinstance(suite['task'],dict) and type(suite['step_count']) is int and suite['step_count']>=0 and
      type(suite['reset_next_step']) is bool and type(snapshot['from_dm_done']) is bool, 'Snapshot counters/flags differ')
  return pickle.dumps(snapshot, protocol=5)


def rgb_copy(image):
  import numpy as np
  require(isinstance(image,np.ndarray) and image.dtype==np.uint8 and image.shape==(64,64,3), 'Actual uint8 64 RGB required')
  return image.copy()


def reset_observation(observation):
  import numpy as np
  require(isinstance(observation,dict) and set(observation)=={'image','reward','is_first','is_last','is_terminal'},
      'Actual reset observation schema differs')
  rgb_copy(observation['image'])
  reward=np.asarray(observation['reward'])
  require(reward.dtype==np.float32 and reward.shape==() and np.isfinite(reward) and reward==0,
      'Actual scalar F32 reset reward required')
  for name,expected in (('is_first',True),('is_last',False),('is_terminal',False)):
    value=np.asarray(observation[name])
    require(value.dtype==np.bool_ and value.shape==() and bool(value)==expected,'Actual reset flag differs')


def rgb_difference(left, right):
  import numpy as np
  left, right = rgb_copy(left), rgb_copy(right)
  delta = np.abs(left.astype(np.int16)-right.astype(np.int16))
  return dict(byte_equal=left.tobytes()==right.tobytes(), changed_channels=int(np.count_nonzero(delta)),
      changed_pixels=int(np.count_nonzero(np.any(delta,axis=-1))), maximum_absolute_difference=int(delta.max()))


def png_bytes(image):
  image = rgb_copy(image)
  def chunk(name, payload):
    return struct.pack('!I',len(payload))+name+payload+struct.pack('!I',zlib.crc32(name+payload)&0xffffffff)
  rows = b''.join(b'\0'+row.tobytes() for row in image)
  return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!IIBBBBB',64,64,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(rows))+chunk(b'IEND',b'')


def compare_recorded(left, right):
  differences = []
  for key in ('mjb_sha256','model_manifest','model_unknown','derived_manifest','derived_unknown','scene_manifest','scene_unknown'):
    if left[key] != right[key]: differences.append(key)
  same = not differences
  rgb_same = left['rgb_sha256'] == right['rgb_sha256']
  mjb_equal=left['mjb_sha256']==right['mjb_sha256']
  model_equal=left['model_manifest']==right['model_manifest']
  derived_equal=left['derived_manifest']==right['derived_manifest']
  scene_equal=left['scene_manifest']==right['scene_manifest']
  if not (model_equal and derived_equal and scene_equal): classification='measured_CPU_numeric_input_difference'
  elif not mjb_equal: classification='MJB_byte_difference_without_measured_numeric_difference_unresolved'
  elif not same: classification='measured_input_coverage_difference_unresolved'
  else: classification='same_measured_inputs_and_RGB' if rgb_same else 'same_measured_CPU_inputs_RGB_difference_unresolved'
  return dict(recorded_cpu_inputs_equal=same, rgb_byte_equal=rgb_same, differing_inputs=differences,
      mjb_equal=mjb_equal,typed_model_equal=model_equal,typed_derived_equal=derived_equal,typed_scene_equal=scene_equal,
      classification=classification,
      full_render_input_equivalence_proven=False, graphical_noise_proven=False)


def collect_model(adapter, output, worker_index, model_index, captures, check_callback=check):
  """Exactly one reset and two normal renders; dependency injection is fixture-only."""
  import numpy as np
  prefix = f'worker{worker_index}-model{model_index}'
  physics = adapter.suite.physics
  original_render = physics.render
  live_images, copies, records, numeric = [], [], [], {}
  render_count = [0]
  def guarded(*args, **kwargs):
    check_callback()
    require(args==(64,64) and kwargs=={'camera_id':CAMERA}, 'Normal camera/shape call differs')
    before = snapshot_bytes(adapter.snapshot())
    model_before, derived_before = capture_model(physics), capture_derived(physics)
    start = len(captures)
    image = original_render(*args,**kwargs)
    require(len(captures)==start+1, 'Exactly one actual Camera render required')
    event = captures[-1]
    require(snapshot_bytes(adapter.snapshot())==before, 'Render changed full physics/task/flags')
    model_after, derived_after = capture_model(physics), capture_derived(physics)
    require(model_before['mjb']==model_after['mjb'] and model_before['manifest']==model_after['manifest'] and
        model_before['unknown']==model_after['unknown'] and derived_before['manifest']==derived_after['manifest'] and
        derived_before['unknown']==derived_after['unknown'],
        'Render changed recorded CPU visual/derived input')
    copied = rgb_copy(image)
    require(all(hashlib.sha256(saved.tobytes()).hexdigest()==row['rgb_sha256']
        for saved,row in zip(copies,records)),'Previously copied RGB bytes changed')
    previous_changed = [live.tobytes()!=saved.tobytes() for live,saved in zip(live_images,copies)]
    live_images.append(image); copies.append(copied)
    index = render_count[0]; render_count[0] += 1
    row = dict(render=index, rgb_sha256=hashlib.sha256(copied.tobytes()).hexdigest(),
        mjb_sha256=hashlib.sha256(model_before['mjb']).hexdigest(),
        model_manifest=model_before['manifest'], model_unknown=model_before['unknown'],
        derived_manifest=derived_before['manifest'],derived_unknown=derived_before['unknown'],scene_manifest=event['scene']['manifest'],
        scene_unknown=event['scene']['unknown'], complete_scene_authenticated=False,
        old_buffer_sha256=event['old_buffer_sha256'], post_buffer_sha256=event['post_buffer_sha256'],
        returned_shares_buffer=event['returned_shares_buffer'],
        actual_returned_sha256=event['returned_sha256'], old_live_views_changed=previous_changed,
        copied_bytes_stable=True, physics_and_recorded_CPU_inputs_unchanged=True,
        actual_camera_name=model_before['camera_name'])
    require(row['rgb_sha256']==row['actual_returned_sha256'], 'Copied RGB differs from actual return')
    name = prefix+f'-rgb{index}.npy'
    with (output/name).open('xb') as stream: np.save(stream,copied,allow_pickle=False)
    png = prefix+f'-rgb{index}.png'; (output/png).write_bytes(png_bytes(copied))
    row.update(file=name,file_sha256=sha(output/name),debug_png=png,debug_png_sha256=sha(output/png))
    records.append(row)
    for label,data in (('model',model_before['leaves']),('derived',derived_before['leaves']),('scene',event['scene']['leaves'])):
      numeric.update({f'render{index}/{label}/{key}':value for key,value in data.items()})
    if index==0:
      (output/(prefix+'.mjb')).write_bytes(model_before['mjb'])
    check_callback(); return image
  physics.render = guarded
  try:
    observation = adapter.reset()
    reset_observation(observation)
    require(render_count[0]==1 and observation['image'].tobytes()==copies[0].tobytes(), 'First RGB must be reset return')
    initial = adapter.snapshot(); payload = snapshot_bytes(initial)
    require(initial['suite']['step_count']==0 and not initial['suite']['reset_next_step'] and
        not initial['from_dm_done'] and adapter.recorder.steps==0 and
        adapter.recorder.last_action is None, 'No suite transition/action is allowed')
    snapshot_name = prefix+'-snapshot.pkl'; (output/snapshot_name).write_bytes(payload)
    for _ in range(2): physics.render(64,64,camera_id=CAMERA)
    require(render_count[0]==3 and adapter.recorder.steps==0 and adapter.recorder.last_action is None and
        snapshot_bytes(adapter.snapshot())==payload, 'Exactly three unchanged reset renders required')
    data_name = prefix+'-CPU-inputs.npz'
    with (output/data_name).open('xb') as stream: np.savez(stream,**numeric)
    return dict(worker=worker_index, model=model_index, seed=SEED, camera_id=CAMERA,
        reset_count=1, render_count=3, suite_transitions=0, actor_calls=0,
        snapshot_file=snapshot_name,snapshot_sha256=sha(output/snapshot_name),
        mjb_file=prefix+'.mjb',mjb_file_sha256=sha(output/(prefix+'.mjb')),
        CPU_inputs_file=data_name,CPU_inputs_file_sha256=sha(output/data_name), renders=records,
        within_model=[dict(left=0,right=i,**compare_recorded(records[0],records[i]),
            pixels=rgb_difference(copies[0],copies[i])) for i in (1,2)],
        snapshot_unchanged=True, complete_render_input_equivalence_proven=False)
  finally: physics.render = original_render


def run_owned(command, log, deadline, env):
  require(time.monotonic()<deadline, 'Late before child creation')
  child = spawn_owned(command,deadline,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  start=time.monotonic(); timed_out=False
  try: code=child.wait(timeout=max(.001,deadline-time.monotonic()))
  except subprocess.TimeoutExpired: code=124; timed_out=True
  finally: cleanup(child)
  return dict(worker_pid=child.pid,worker_terminal=child.poll() is not None,
      exit_code=code,timed_out=timed_out,elapsed_seconds=time.monotonic()-start)


def publish_final(output, receipt, deadline):
  pending=output/'supervisor-receipt-pending.json'; write_json(pending,receipt)
  if time.monotonic()>=deadline:
    write_json(output/'late-supervisor.json',dict(exit_code=124,incomplete=True)); return 124
  os.replace(pending,output/'supervisor_complete.json')
  if time.monotonic()>=deadline:
    os.replace(output/'supervisor_complete.json',output/'late-provisional-supervisor.json')
    write_json(output/'late-supervisor.json',dict(exit_code=124,incomplete=True)); return 124
  return receipt['exit_code']


def worker(args):
  global DEADLINE
  require(os.name=='posix' and socket.gethostname()=='lyg0326','Only assigned sv3')
  output=Path(args.output).resolve(strict=True); budget=read_json(output/'budget.json')
  require(type(args.worker_index) is int and args.worker_index in (0,1) and
      os.getppid()==budget['parent_pid'] and os.environ.get('DL_NATIVE_RENDER_PARENT')==str(os.getppid()) and
      os.getpgid(0)==os.getpid() and budget['version']==VERSION and budget['seconds']==SECONDS and
      budget['actual_git_sha']==args.expected_git,'Direct owned worker authentication failed')
  DEADLINE=budget['monotonic_start']+SECONDS; check()
  signal.setitimer(signal.ITIMER_REAL,max(.001,DEADLINE-time.monotonic()))
  require(git_binding(args.expected_git)==budget['source_files_sha256'],'Worker sources changed')
  official=installed_binding()
  sys.path.insert(0,str(ROOT)); from scripts import dl_resources as resource
  previous=resource.command_output; resource.command_output=lambda command:owned_output(command).decode()
  try: device=resource.require_resource(args.resource_record)
  finally: resource.command_output=previous
  require(device['physical_gpu']==4 and device['gpu_uuid']==UUID,'No device substitution')
  # This worker allocates EGL only; the external pinned probe supplies CUDA mapping.
  os.environ['JAX_PLATFORMS']='cpu'
  from dm_control import suite
  from dm_control.mujoco import engine
  from scripts.dl_feedback_physics import FeedbackEnvironment
  models=[]; captured=[]; runtime=[]
  for index in range(MODELS):
    check(); env=suite.load('quadruped','walk',task_kwargs={'random':SEED})
    wrapped=FeedbackEnvironment(env)
    try:
      with camera_capture(engine,captured): row=collect_model(wrapped,output,args.worker_index,index,captured)
      previous=resource.command_output; resource.command_output=lambda command:owned_output(command).decode()
      try: runtime.append(resource.verify_runtime(device,compute=False,graphics=True))
      finally: resource.command_output=previous
      models.append(row)
    finally: wrapped.close()
  check(); require(installed_binding()==official,'Installed official sources changed')
  require(git_binding(args.expected_git)==budget['source_files_sha256'],'Worker end sources changed')
  write_json(output/f'worker{args.worker_index}-complete.json',dict(version=VERSION,
      actual_git_sha=args.expected_git,worker_pid=os.getpid(),parent_pid=os.getppid(),worker=args.worker_index,
      complete_verified=True,models=models,device=device,runtime_graphics=runtime,official_source_binding=official,
      this_worker_cuda_compute_context_created=False,CUDA_mapping_authenticated_by_external_probe=True,
      JAX_platform='cpu',actual_agent_instances=0,actual_agent_calls=0,checkpoint_reads=0,
      old_episode_reads=0,actual_resets=2,actual_RGB=6,actual_suite_transitions=0,
      elapsed_parent_entry_seconds=time.monotonic()-budget['monotonic_start']))
  check()


def parent(args):
  require(os.name=='posix' and socket.gethostname()=='lyg0326','Only assigned sv3')
  require('DL_NATIVE_RENDER_PARENT' not in os.environ,'Parent must not inherit a worker identity token')
  sources=git_binding(args.expected_git); output=Path(args.output).resolve()
  require(output.is_relative_to(Path('/data/Policy_Discrepancy/runs')) and not output.exists(),'New runs directory required')
  output.mkdir(parents=True); check()
  write_json(output/'budget.json',dict(version=VERSION,actual_git_sha=args.expected_git,parent_pid=os.getpid(),
      seconds=SECONDS,monotonic_start=ENTRY_STARTED,planned_physical_gpu=4,max_gpu_devices=1,
      max_gpu_hours=.05,source_files_sha256=sources))
  write_json(output/'plan.json',dict(version=VERSION,workers=2,models_per_worker=2,seed=SEED,camera=CAMERA,
      size=[64,64],resets=4,actual_RGB=12,suite_transitions=0,actor_calls=0,
      full_render_input_equivalence_claimed=False,r1_reinterpreted=False))
  sys.path.insert(0,str(ROOT)); from scripts import dl_resources as resource
  device=read_json(args.resource_record); phases=[]; code=1; timed_out=False
  for index in range(WORKERS):
    check(); previous=resource.command_output; resource.command_output=lambda command:owned_output(command).decode()
    try: fresh=resource.refresh(device)
    finally: resource.command_output=previous
    require(fresh['physical_gpu']==4 and fresh['gpu_uuid']==UUID,'No device substitution')
    resource_path=output/f'worker{index}-resource.json'; write_json(resource_path,fresh)
    env=dict(os.environ,DL_NATIVE_RENDER_PARENT=str(os.getpid()),JAX_PLATFORMS='cpu',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=[sys.executable,str(Path(__file__).resolve()),'--worker-index',str(index),
        '--output',str(output),'--expected-git',args.expected_git,'--resource-record',str(resource_path)]
    with (output/f'worker{index}.log').open('x',encoding='utf-8') as log: terminal=run_owned(command,log,DEADLINE,env)
    phase=dict(worker=index,**terminal); phases.append(phase); write_json(output/f'worker{index}-exit.json',phase)
    print(json.dumps(phase),flush=True); code,timed_out=terminal['exit_code'],terminal['timed_out']
    if code or time.monotonic()>=DEADLINE: break
  complete=code==0 and len(phases)==WORKERS and time.monotonic()<DEADLINE
  result_sha=None
  if complete:
    rows=[read_json(output/f'worker{i}-complete.json') for i in range(WORKERS)]
    models=[model for row in rows for model in row['models']]
    require(len(models)==4 and all(row['complete_verified'] is True and row['actual_git_sha']==args.expected_git and
        row['worker_pid']==phases[i]['worker_pid'] and row['parent_pid']==os.getpid() and
        row['actual_resets']==2 and row['actual_RGB']==6 and row['actual_suite_transitions']==0 and
        row['actual_agent_calls']==row['checkpoint_reads']==row['old_episode_reads']==0
        for i,row in enumerate(rows)),'Incomplete zero-action workers')
    pairs=[]
    for i,left in enumerate(models):
      for j,right in enumerate(models[i+1:],i+1):
        pairs.append(dict(left=[left['worker'],left['model']],right=[right['worker'],right['model']],
            same_worker=left['worker']==right['worker'],
            initial_snapshot_equal=left['snapshot_sha256']==right['snapshot_sha256'],
            **compare_recorded(left['renders'][0],right['renders'][0])))
    files={}
    for model in models:
      for field in ('snapshot_file','mjb_file','CPU_inputs_file'):
        name=model[field]; pin=model[field.replace('_file','_sha256') if field=='snapshot_file' else field+'_sha256']
        require(sha(output/name)==pin,'Generated model/input file changed'); files[name]=pin
      for row in model['renders']:
        for field,pinfield in (('file','file_sha256'),('debug_png','debug_png_sha256')):
          require(sha(output/row[field])==row[pinfield],'Generated RGB bytes changed'); files[row[field]]=row[pinfield]
    for i in range(WORKERS): files[f'worker{i}-complete.json']=sha(output/f'worker{i}-complete.json')
    require(git_binding(args.expected_git)==sources,'Final parent sources differ')
    result=dict(version=VERSION,actual_git_sha=args.expected_git,models=models,pairs=pairs,
        actual_resets=4,actual_RGB=12,actual_suite_transitions=0,actual_actor_calls=0,
        agent_instances=0,checkpoint_reads=0,old_episode_reads=0,files_sha256=files,
        same_measured_CPU_inputs_implies_full_render_equivalence=False,graphical_noise_proven=False,
        root_cause_fully_identified=False,method_effectiveness_established=False,r1_reinterpreted=False,
        worker_receipts_sha256=[sha(output/f'worker{i}-complete.json') for i in range(WORKERS)])
    write_json(output/'result.json',result); result_sha=sha(output/'result.json')
  receipt=dict(version=VERSION,actual_git_sha=args.expected_git,parent_pid=os.getpid(),phases=phases,
      complete_verified=complete,timed_out=timed_out,exit_code=0 if complete else (124 if timed_out else code or 1),
      elapsed_parent_entry_seconds=time.monotonic()-ENTRY_STARTED,budget_sha256=sha(output/'budget.json'),
      result_sha256=result_sha,actual_gpu_devices_used=1 if complete else None,
      method_effectiveness_established=False,independent_audit_complete=False)
  return publish_final(output,receipt,DEADLINE)


def verify_completion(directory, external_cli_terminal):
  """Stdlib receipt/hash verification; neither RGB nor snapshot is decoded."""
  output=Path(directory).resolve(strict=True)
  require(not any((output/name).exists() for name in ('late-supervisor.json',
      'late-provisional-supervisor.json','supervisor-receipt-pending.json')),'Late/provisional completion forbidden')
  supervisor=read_json(output/'supervisor_complete.json'); external=read_json(external_cli_terminal)
  require(supervisor['version']==VERSION and supervisor['complete_verified'] is True and
      supervisor['exit_code']==0 and supervisor['timed_out'] is False and
      supervisor['actual_gpu_devices_used']==1 and
      0<supervisor['elapsed_parent_entry_seconds']<=SECONDS and
      external['actual_git_sha']==supervisor['actual_git_sha'] and
      external['worker_pid']==supervisor['parent_pid'] and external['worker_terminal'] is True and
      external['exit_code']==0 and external['timed_out'] is False and
      0<external['elapsed_cli_seconds']<=SECONDS,'Complete parent plus bounded external exit required')
  require(sha(output/'budget.json')==supervisor['budget_sha256'],'Budget bytes changed')
  budget=read_json(output/'budget.json')
  require(budget['version']==VERSION and budget['actual_git_sha']==supervisor['actual_git_sha'] and
      budget['parent_pid']==supervisor['parent_pid'] and budget['seconds']==SECONDS and
      budget['max_gpu_devices']==1 and budget['planned_physical_gpu']==4 and budget['max_gpu_hours']==.05,
      'Fixed whole-entry budget required')
  phases=supervisor['phases']
  require([phase['worker'] for phase in phases]==[0,1] and all(phase['exit_code']==0 and
      phase['timed_out'] is False and phase['worker_terminal'] is True and
      0<phase['elapsed_seconds']<=SECONDS for phase in phases),'Two complete terminated workers required')
  for phase in phases:
    require(read_json(output/f"worker{phase['worker']}-exit.json")==phase,'Worker exit receipt changed')
  require(sha(output/'result.json')==supervisor['result_sha256'],'Result bytes changed')
  result=read_json(output/'result.json')
  require(result['version']==VERSION and result['actual_git_sha']==supervisor['actual_git_sha'] and
      result['actual_resets']==4 and result['actual_RGB']==12 and
      result['actual_suite_transitions']==result['actual_actor_calls']==result['agent_instances']==
      result['checkpoint_reads']==result['old_episode_reads']==0 and len(result['models'])==4 and
      len(result['pairs'])==6 and result['root_cause_fully_identified'] is False and
      result['graphical_noise_proven'] is False and result['r1_reinterpreted'] is False,
      'Exact zero-action engineering scope required')
  for name,pin in result['files_sha256'].items():
    require(Path(name).name==name and sha(output/name)==pin,'Generated artifact bytes changed')
  for i,phase in enumerate(phases):
    row=read_json(output/f'worker{i}-complete.json')
    require(row['worker_pid']==phase['worker_pid'] and row['parent_pid']==supervisor['parent_pid'] and
        row['worker']==i and row['version']==VERSION and row['actual_git_sha']==supervisor['actual_git_sha'] and
        row['complete_verified'] is True and row['actual_resets']==2 and row['actual_RGB']==6 and
        row['this_worker_cuda_compute_context_created'] is False and
        row['models']==[model for model in result['models'] if model['worker']==i],
        'Actual graphics-only worker receipt required')
  return dict(diagnostic_complete_external_exit_verified=True,result_sha256=supervisor['result_sha256'],
      full_render_input_equivalence_proven=False,graphical_noise_proven=False,
      model_or_renderer_independently_replayed=False,method_effectiveness_established=False)


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  for name in ('output','expected-git','resource-record'): parser.add_argument('--'+name,required=True)
  parser.add_argument('--worker-index',type=int,choices=(0,1))
  args=parser.parse_args()
  if args.worker_index is not None: worker(args); return 0
  return parent(args)


if __name__=='__main__': sys.exit(main())
