"""Readonly reference0 artifact authentication and fixed NumPy readouts.

This entry accepts only the adopted complete reference0 object, not future B/N
endpoints. External feature-controller and actual SSH-exit pins are mandatory.
It performs no model, physics, learning, GPU work or scientific pass decision.
"""
import time
ENTRY_STARTED = time.monotonic()
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
VERSION = 'DL-normal-readout-run-r2'
PRODUCER_GIT = '2a15c91aad16c0e3151fde3943e356dad1fcb69d'
LABELS_GIT = '47db6265311bc04dbdcc79439842eb8b6e707097'
PRIOR_GIT = '277c18001d353d9101a3622959351d3994699adb'
RUN_GIT = '1584ef31c322fc3371ea9ce220c06fe7e654b23a'
BASE = Path('/data/Policy_Discrepancy')
FEATURE_RUN = BASE/'runs/dl-normal-reference0-features-20261016-2a15c91/attempt02'
FEATURE_INPUTS = BASE/'runs/dl-normal-reference0-inputs-20261016-2a15c91'
PRODUCER_CODE = BASE/'dl-normal-feature-run-code-20261015-2a15c91'
FEATURE_SECONDS, GPU = 12600, 6
GPU_UUID = 'GPU-70db6c66-9917-e0e2-44db-0fc80fcf41b9'
CHECKPOINT = 'c6a7d92e24669a515bb44693faf73e442d9868f9fc814eec0e22e069aa594d24'
CONFIG = '877f8470cb4f7f45d143e3099fbd3f55a828991afe33779b7b04c4264f6c8377'
OUTER_COMMAND = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','sv3',
    str(BASE/'envs/dreamer/bin/python'),str(FEATURE_INPUTS/'launch.py')]
STAGE_ONLY = frozenset(('generation_complete.json','supervisor_terminal.json'))
REUSED = ('scripts/dl_normal_feature_run.py','scripts/dl_normal_source_index.py',
    'scripts/dl_normal_features.py','scripts/dl_reset_learning.py','analysis/dl_normal_readout.py')
NORMS = frozenset(('AGENTS.md','docs/DL_NORMAL_READOUT.md',
    'docs/DL_NORMAL_FEATURE_RUN_ENGINEERING.md','docs/DL_NORMAL_FEATURE_ENGINEERING.md',
    'docs/DL_NORMAL_LABEL_ENGINEERING.md','docs/DL_NORMAL_READOUT_RUN_ENGINEERING.md'))
TARGETS = ('velocity_x','upright')
DEADLINE = float('inf')


def require(value, message):
  if not value: raise ValueError(message)


def check():
  require(time.monotonic() < DEADLINE, 'Whole explicit readout CPU deadline exceeded')


def cpu_only():
  forbidden = ('jax','jaxlib','mujoco','dm_control','OpenGL','torch','tensorflow')
  require(not any(name.split('.')[0] in forbidden for name in sys.modules),
      'Readonly analysis must not import model/GPU/physics runtimes')
  require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('JAX_PLATFORMS') == 'cpu' and
      os.environ.get('MUJOCO_GL') == 'disable' and
      all(name not in os.environ for name in ('MUJOCO_EGL_DEVICE_ID','PYOPENGL_PLATFORM')),
      'Explicit zero-GPU/no-EGL CPU environment required')


def digest(body): return hashlib.sha256(body).hexdigest()


def canonical(value):
  return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def decode(body):
  def pairs(items):
    result = {}
    for key, value in items:
      require(key not in result, 'Duplicate JSON key'); result[key] = value
    return result
  return json.loads(body, object_pairs_hook=pairs,
      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def pin(path, expected=None):
  check(); path = Path(path).absolute()
  require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Ordinary pinned file required')
  before = path.stat(); h = hashlib.sha256()
  with path.open('rb') as stream:
    for block in iter(lambda: stream.read(1024*1024), b''): check(); h.update(block)
  after = path.stat()
  require((before.st_size,before.st_mtime_ns,before.st_ino) ==
      (after.st_size,after.st_mtime_ns,after.st_ino), 'Input changed during hash')
  result = dict(path=str(path.resolve()), bytes=after.st_size, sha256=h.hexdigest())
  if expected is not None:
    require(re.fullmatch('[0-9a-f]{64}', expected) and result['sha256'] == expected, 'Input raw SHA differs')
  return result


def bound(item):
  require(type(item) is dict and set(item) == {'path','bytes','sha256'} and pin(item['path'],item['sha256']) == item,
      'Input path/bytes/raw pin differs')
  body = Path(item['path']).read_bytes(); check()
  require(len(body) == item['bytes'] and digest(body) == item['sha256'], 'Same-buffer input SHA differs')
  return body


def inventory(root):
  root = Path(root).absolute()
  require(root.is_dir() and not any(p.is_symlink() for p in (root,*root.parents)), 'Ordinary complete run required')
  files, directories = {}, []
  for path in sorted(root.rglob('*')):
    check(); require(not path.is_symlink() and path.resolve().is_relative_to(root), 'Run link/escape refused')
    name = path.relative_to(root).as_posix()
    if path.is_dir(): directories.append(name)
    else: files[name] = pin(path)
  return dict(files=files, directories=directories)


def git(command, payload=None):
  check()
  # These local read-only Git commands have no hooks/fsmonitor/network helpers.
  result = subprocess.run(['git','-c','core.fsmonitor=false','-c','core.hooksPath=/dev/null',*command],
      cwd=ROOT, input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
      timeout=max(.001,DEADLINE-time.monotonic()) if DEADLINE != float('inf') else 30,
      env=dict(os.environ,GIT_OPTIONAL_LOCKS='0'))
  require(result.returncode == 0, 'Read-only Git command failed'); check(); return result.stdout


def blobs(commit, names):
  require(re.fullmatch('[0-9a-f]{40}',commit), 'Full Git SHA required')
  names = sorted(names)
  raw = git(['cat-file','--batch'], ''.join(commit+':'+name+'\n' for name in names).encode())
  offset = 0; result = {}
  for name in names:
    end = raw.index(b'\n',offset); head = raw[offset:end].split(); offset = end+1
    require(len(head) == 3 and head[1] == b'blob', 'Actual Git blob required: '+name)
    size = int(head[2]); body = raw[offset:offset+size]; offset += size+1
    require(len(body) == size and raw[offset-1:offset] == b'\n', 'Truncated Git batch')
    result[name] = body
  require(offset == len(raw), 'Extra Git batch tail'); return result


def analysis_binding(expected):
  require(git(['rev-parse','HEAD']).decode().strip() == expected and
      not git(['status','--porcelain','--untracked-files=all']), 'Exact clean analysis Git required')
  names = git(['ls-files','-z']).decode().split('\0')
  selected = {name for name in names if name.endswith(('.py','.yaml','.yml','.lock')) or name in NORMS}
  require(NORMS | {'analysis/dl_normal_readout_run.py','analysis/dl_normal_readout.py',
      'tests/test_dl_normal_readout_run.py'} <= selected, 'Tracked complete analysis source/norms required')
  result = {}
  for name, body in blobs(expected,selected).items():
    actual = pin(ROOT/name)
    require((actual['bytes'],actual['sha256']) == (len(body),digest(body)), 'Actual analysis Git blob differs: '+name)
    result[name] = dict(bytes=len(body),sha256=digest(body))
  return result


def producer_binding(declared):
  require(type(declared) is dict and set(REUSED) <= set(declared), 'Complete producer source manifest required')
  names = git(['ls-tree','-r','--name-only','-z',PRODUCER_GIT]).decode().split('\0')
  from scripts import dl_normal_feature_run as f
  selected = {n for n in names if n.endswith(('.py','.yaml','.yml','.lock')) or n in
      {'AGENTS.md','docs/DL_R5_LIVE_TRIAL.md','docs/DL_R5_LIVE_RETRY.md','docs/DL_R5_AUDIT_ENGINEERING.md',*f.NORMS}}
  require(set(declared) == selected, 'Exact producer manifest file set differs')
  raw = blobs(PRODUCER_GIT,selected)
  for name, body in raw.items():
    require(declared[name] == dict(bytes=len(body),sha256=digest(body)), 'Original producer Git blob differs: '+name)
  for name in REUSED:
    require((ROOT/name).read_bytes() == raw[name], 'Reused CPU verifier dependency differs from producer Git: '+name)
  cpu_only()
  return {name:declared[name] for name in REUSED}


def identity(item):
  require(type(item) is dict and set(item) in ({'pid','pgid','sid','start_ticks'},
      {'pid','parent_pid','pgid','sid','start_ticks'}) and
      all(type(v) is int and v > 0 for v in item.values()) and item['pid'] == item['pgid'] == item['sid'],
      'Original complete owned session identity required')
  return {name:item[name] for name in ('pid','pgid','sid','start_ticks')}


def inactive(item):
  original = identity(item)
  require(sys.platform == 'linux', 'Actual process retirement requires Linux /proc')
  require(not Path('/proc',str(original['pid'])).exists(), 'Original PID exists or was reused')
  for path in Path('/proc').iterdir():
    if not path.name.isdigit(): continue
    try: fields = (path/'stat').read_text().rsplit(') ',1)[1].split()
    except (FileNotFoundError,ProcessLookupError): continue
    require(int(fields[2]) != original['pgid'] or fields[0] in ('Z','X'), 'Original owned group still active')


def authority(args):
  require(Path(args.features_run).absolute() == FEATURE_RUN, 'Only the fixed reference0 run is supported')
  terminal_pin = pin(args.feature_controller_terminal,args.feature_controller_sha)
  outer_pin = pin(args.feature_outer_execution,args.feature_outer_sha)
  require(terminal_pin['path'] == str(FEATURE_INPUTS/'terminal.json'), 'Fixed feature-controller terminal required')
  terminal, outer = decode(bound(terminal_pin)), decode(bound(outer_pin))
  require(outer['git_sha'] == PRODUCER_GIT and outer['command'] == OUTER_COMMAND and
      type(outer['returncode']) is int and outer['returncode'] == 0 and
      type(outer['physical_gpu']) is int and outer['physical_gpu'] == GPU and
      outer['external_failure'] is None and outer['actual_outer_complete_within12600'] is True and
      0 < outer['elapsed_seconds'] <= FEATURE_SECONDS and
      re.fullmatch('[0-9a-f]{64}',outer['log_sha256']) and outer['controller_terminal'] == terminal_pin,
      'Actual bounded feature-controller SSH exit0 authority required')
  require(terminal['version'] == 'DL-normal-reference0-launcher-r1' and terminal['git_sha'] == PRODUCER_GIT and
      terminal['run_directory'] == str(FEATURE_RUN) and terminal['role'] == 'reference0' and
      terminal['whole_seconds'] == FEATURE_SECONDS and type(terminal['parent_returncode']) is int and
      terminal['parent_returncode'] == 0 and terminal['parent_terminal'] is True and terminal['success'] is True and
      terminal['parent_identity_gone'] is True and terminal['owned_groups_cleaned'] is True and
      terminal['external_failure'] is None and terminal['cleanup_errors'] == [] and terminal['remaining_helper_registry'] == [] and
      0 < terminal['elapsed_seconds'] <= outer['elapsed_seconds'] and
      terminal['physical_gpu'] == GPU and terminal['gpu_uuid'] == GPU_UUID,
      'Complete reference0 controller terminal required')
  expected_scope = dict(complete_units=512,fit_families=8,held_families=24,model_role='reference0',
      independent_NN_reexecution=False,training_updates=0,readout_fitted=False,scientific_verdict=False,
      subsequent_learning_authorized=False)
  require(terminal['scope'] == expected_scope, 'Reference0 controller scope differs')
  parent = identity(terminal['parent_identity']); inactive(terminal['parent_identity'])
  for child in terminal['owned_children']:
    identity(child); require(child['parent_pid'] == parent['pid'], 'Original direct worker parent differs'); inactive(child)
  return terminal, dict(controller_terminal=terminal_pin,outer_execution=outer_pin)


def validate_index(index):
  from scripts import dl_normal_source_index as s
  require(index['version'] == s.VERSION and index['metadata_authentication_git'] == PRODUCER_GIT and
      index['run_git'] == RUN_GIT and index['prior_physics_audit_git'] == PRIOR_GIT and
      index['complete_units'] == 512 and index['complete_families'] == 32 and index['fit_rows'] == 128 and
      index['held_rows'] == 384 and index['fit_families'] == list(range(8)) and
      index['held_families'] == list(range(8,32)) and
      index['run_directory'] == str(BASE/'runs/dl-r5-live-r2-20261015-1584ef3/attempt01') and
      index['source_kind'] == 'complete-audited-R5-pretreatment' and
      index['source_graph_and_prior_physics_audit_authenticated'] is True and
      all(index[name] is False for name in ('source_recomputed_latents','physics_reexecuted','model_executed',
          'scientific_verdict_recomputed','source_selected_by_scientific_sign','GPU_used','subsequent_learning_authorized')),
      'Fixed complete reference0 source index required')
  s.validate_order(index['rows'],index['run_inventory_sha256'])
  for row in index['rows']:
    t = row['plan']['t']; current = row['current']
    require(len(row['frames']) == t+1 and current['observation_index'] == current['snapshot_index'] == t and
        current['preceding_reward_index'] == t-1 and current['source_raw_available'] is False and
        current['source_raw_index'] is None and current['physical_is_last'] is current['physical_is_terminal'] is False and
        row['bounds']['observations'] == [0,t+1] and row['bounds']['raw'] == [0,t] and
        row['old_latent_used_as_feature'] is False, 'Source prefix/current nonterminal boundary differs')


class WorkerView:
  """Producer phase-one view of an already whole-graph authenticated run.

  Only these two exact top-level paths were written after verify_worker. No
  basename/subdirectory filtering is allowed; all exposed files are originals.
  """
  def __init__(self, root): self.root = Path(root)
  def __fspath__(self): return str(self.root)
  def __truediv__(self, name): return self.root/name
  def rglob(self, pattern):
    return (path for path in self.root.rglob(pattern) if path.relative_to(self.root).as_posix() not in STAGE_ONLY)


def validate_graph(root, terminal):
  first = inventory(root); proof = terminal['output_proof']
  for name, key in (('generation_complete.json','generation'),('supervisor_terminal.json','supervisor')):
    require(proof[key] == first['files'].get(name), 'Authenticated top-level terminal artifact differs')
  generation = decode(bound(proof['generation'])); supervisor = decode(bound(proof['supervisor']))
  require(generation['version'] == 'DL-normal-feature-run-r1' and generation['git_sha'] == PRODUCER_GIT and
      generation['role'] == 'reference0' and generation['complete_units'] == 512 and generation['complete_families'] == 32 and
      generation['fit_families'] == list(range(8)) and generation['held_families'] == list(range(8,32)) and
      generation['source_complete_and_stable'] is generation['labels_complete_and_stable'] is
      generation['real_N1M_inputs_authenticated'] is True and generation['whole_seconds'] == FEATURE_SECONDS and
      0 < generation['elapsed_seconds'] <= FEATURE_SECONDS and
      all(generation[n] is False for n in ('independent_NN_reexecution','physics_reexecuted','readout_fitted',
          'scientific_verdict','subsequent_learning_authorized')) and generation['render_calls'] == generation['training_updates'] == 0,
      'Complete fixed reference0 generation required')
  expected = dict(generation['output_inventory'])
  require(not (set(expected) & STAGE_ONLY), 'Worker-stage inventory cannot hide final terminal names')
  expected.update({name:proof[key] for name,key in
      (('generation_complete.json','generation'),('supervisor_terminal.json','supervisor'))})
  require(first['files'] == expected and first['directories'] == [f'unit_{u:03d}' for u in range(512)],
      'Whole real run inventory has missing/extra/changed files or directories')
  require(proof['inventory_sha256'] == digest(canonical(first)) and proof['files'] == len(first['files']) and
      proof['directories'] == 512, 'Actual controller whole inventory seal differs')
  require(supervisor['version'] == 'DL-normal-feature-run-r1' and supervisor['git_sha'] == PRODUCER_GIT and
      type(supervisor['worker_returncode']) is int and supervisor['worker_returncode'] == 0 and
      supervisor['failure'] is None and supervisor['owned_groups_cleaned'] is True and
      supervisor['generation_complete'] == proof['generation'] and
      0 < supervisor['elapsed_seconds'] <= terminal['elapsed_seconds'] and
      identity(supervisor['parent_identity']) == identity(terminal['parent_identity']), 'Successful feature supervisor differs')
  for name in ('budget','source_index','worker_complete'):
    require(generation[name] == proof[name] == first['files'].get(Path(proof[name]['path']).name),
        'Complete generation/proof artifact pin differs: '+name)
  return first, generation, supervisor


def labels_for(index, generation, budget):
  from scripts import dl_normal_feature_run as f
  labels = generation['labels']
  require(labels == budget['labels'] and set(labels) == {'output','terminal','outer_execution'}, 'Successful labels pins differ')
  # Adapter supplies only CPU Git and readonly /proc checks, never a physics auditor.
  def original_inactive(pid, start):
    require(not Path('/proc',str(pid)).exists(), 'Original labels PID exists or was reused')
  def command_output(command,payload=None):
    require(command[0] == 'git', 'Only local readonly Git metadata command allowed')
    return git(command[1:],payload)
  adapter = SimpleNamespace(command_output=command_output,original_inactive=original_inactive,original_group_inactive=inactive)
  args = SimpleNamespace(labels=labels['output']['path'],labels_sha=labels['output']['sha256'],
      labels_terminal=labels['terminal']['path'],labels_terminal_sha=labels['terminal']['sha256'])
  previous = f.DEADLINE; f.DEADLINE = DEADLINE
  try: actual, report = f.labels_authority(args,index,adapter,budget['plan'])
  finally: f.DEADLINE = previous
  require(actual == labels, 'Successful label pins differ')
  require(report['git_sha'] == LABELS_GIT and all(report[name] is False for name in
      ('GPU_used','D_e_Y_used_for_labels_or_selection','scientific_verdict','subsequent_learning_authorized')) and
      report['model_calls'] == report['render_calls'] == 0, 'Original CPU task-label scope differs')
  import numpy as np
  for row,label in zip(index['rows'],report['rows']):
    require(label['velocity_frame'] == 'torso-local-velocimeter-x' and label['current_source_raw_used'] is False and
        label['replayed_steps'] == row['plan']['t'] and label['official_resets'] == 1 and
        label['render_calls'] == 0 and label['query_readonly_snapshot_and_model_byte_exact'] is True,
        'Physical target/current-frame/readonly declaration differs')
    require(all(type(label['current_values'][name]) in (int,float) and
        np.float64(label['current_values'][name]).tobytes().hex() == label['current_f64_hex'][name] for name in TARGETS),
        'Physical target F64 hex differs')
  return report


def authenticate(args):
  cpu_only(); terminal, external = authority(args); root = Path(args.features_run)
  first, generation, supervisor = validate_graph(root,terminal)
  budget = decode(bound(generation['budget'])); index = decode(bound(generation['source_index']))
  reused = producer_binding(budget['source_files']); validate_index(index)
  plan = budget['plan']
  require(plan['version'] == 'DL-normal-feature-run-execution-r1' and plan['git_sha'] == PRODUCER_GIT and
      plan['role'] == 'reference0' and plan['logical_server'] == 'sv3' and plan['physical_gpu'] == GPU and
      plan['gpu_uuid'] == GPU_UUID and plan['whole_seconds'] == FEATURE_SECONDS and plan['complete_units'] == 512 and
      supervisor['plan'] == plan and budget['version'] == 'DL-normal-feature-run-r1' and budget['git_sha'] == PRODUCER_GIT and
      budget['role'] == 'reference0' and budget['complete_units'] == 512 and
      identity(budget['parent_identity']) == identity(terminal['parent_identity']) and
      budget['source_index'] == generation['source_index'] and
      budget['monotonic_deadline'] == budget['entry_started']+FEATURE_SECONDS,
      'Original reference0 budget/plan/parent differs')
  inputs = terminal['inputs']
  require(budget['execution_plan'] == inputs['execution_plan'] and decode(bound(budget['execution_plan'])) == plan and
      budget['inputs'] == {n:inputs[n] for n in ('checkpoint','config')}, 'Actual original input/plan pins differ')
  for item in inputs.values(): bound(item)
  require(index['external_pins'] == {name:inputs[name] for name in
      ('launcher_terminal','launcher_launch','audit_report','audit_terminal','audit_launch')} and
      generation['labels'] == dict(output=inputs['labels'],terminal=inputs['labels_terminal'],
          outer_execution=inputs['labels_outer_execution']) and
      inputs['checkpoint']['sha256'] == CHECKPOINT and inputs['config']['sha256'] == CONFIG,
      'Fixed original source/labels/checkpoint input graph differs')
  require(all(terminal['source_files'].get(n) == entry for n,entry in budget['source_files'].items()),
      'Actual controller/producer source manifest differs')
  process = decode(bound(first['files']['process.json'])); worker = identity(process['worker_identity'])
  require(identity(process['parent_identity']) == identity(terminal['parent_identity']) and
      process['budget'] == generation['budget'] and worker == identity(terminal['output_proof']['worker_identity']),
      'Actual worker/parent/budget differs')
  values = dict(run=index['run_directory'],expected_git=PRODUCER_GIT,prior_physics_audit_git=PRIOR_GIT,
      launcher_terminal=inputs['launcher_terminal']['path'],launcher_sha=inputs['launcher_terminal']['sha256'],
      launcher_launch=inputs['launcher_launch']['path'],launcher_launch_sha=inputs['launcher_launch']['sha256'],
      audit_report=inputs['audit_report']['path'],audit_report_sha=inputs['audit_report']['sha256'],
      audit_terminal=inputs['audit_terminal']['path'],audit_terminal_sha=inputs['audit_terminal']['sha256'],
      audit_launch=inputs['audit_launch']['path'],audit_launch_sha=inputs['audit_launch']['sha256'],
      labels=inputs['labels']['path'],labels_sha=inputs['labels']['sha256'],
      labels_terminal=inputs['labels_terminal']['path'],labels_terminal_sha=inputs['labels_terminal']['sha256'],
      checkpoint=inputs['checkpoint']['path'],config=inputs['config']['path'],
      execution_plan=inputs['execution_plan']['path'],execution_plan_sha=inputs['execution_plan']['sha256'],output=str(root))
  from scripts import dl_normal_feature_run as f
  command = [str(BASE/'envs/dreamer/bin/python'),str(PRODUCER_CODE/'scripts/dl_normal_feature_run.py')]
  for name in f.COMMON: command += ['--'+name.replace('_','-'),values[name]]
  require(process['command'] == command+['--worker','--budget-sha',generation['budget']['sha256']],
      'Actual original worker argv/keys/source pins differ')
  matches = [item for item in terminal['owned_children'] if identity(item) == worker]
  require(len(matches) == 1 and matches[0]['parent_pid'] == terminal['parent_identity']['pid'], 'Unique actual direct feature worker required')
  require(process['environment'] == dict(CUDA_VISIBLE_DEVICES=GPU_UUID,CUDA_DEVICE_ORDER='PCI_BUS_ID',
      JAX_PLATFORMS='cuda',MUJOCO_GL='disable',MUJOCO_EGL_DEVICE_ID=None,PYOPENGL_PLATFORM=None),
      'Actual original worker compute-only environment differs')
  for name in ('runtime_before.json','runtime_after.json'):
    runtime = decode(bound(first['files'][name]))
    require(runtime['pid'] == worker['pid'] and runtime['compute_uuid'] == [GPU_UUID] and runtime['physical_gpu'] == GPU and
        runtime['egl_uuid'] is None and runtime['graphics_verified'] is False, 'Actual original compute-only runtime differs')
  labels = labels_for(index,generation,budget)
  initial = decode(bound(first['files']['initial.json']))
  require(initial['model']['checkpoint'] == inputs['checkpoint'] and initial['model']['config_pin'] == inputs['config'],
      'Original loaded model file provenance differs')
  from scripts import dl_normal_feature_run as f
  complete_pin, complete = verify_worker_stage(root,first,index,labels)
  require(complete_pin == generation['worker_complete'] and complete['git_sha'] == PRODUCER_GIT and
      complete['labels'] == generation['labels'] and complete['full_state_sha256'] == f.FULL_STATE and
      complete['params_sha256'] == f.PARAMS and complete['model_calls_readonly'] is True and
      complete['readout_fitted'] is complete['scientific_verdict'] is False and complete['training_updates'] == 0,
      'Complete reference0 worker/probability-source role differs')
  require(complete['public_policy_calls'] == terminal['output_proof']['public_policy_calls'] and
      complete['physics_reexecuted'] is False and complete['render_calls'] == 0,
      'Full actual history/no-physics scope differs')
  require(inventory(root) == first, 'Whole real inventory changed across worker-stage view')
  cpu_only()
  return dict(terminal=terminal,external=external,inventory=first,generation=generation,budget=budget,
      index=index,labels=labels,reused_verifier_sources=reused,
      probability_source='jax.nn.softmax(model.dyn._dist(logits).output.dist.logits, -1); official unimix .01',
      source_declaration_scope='Producer code/typed bytes/history oracle authenticated; no independent NN reexecution')


def verify_worker_stage(root, whole, index, labels):
  """Reuse phase one only after the caller authenticates the whole real graph."""
  from scripts import dl_normal_feature_run as f
  require(inventory(root) == whole, 'Whole real inventory changed before worker-stage view')
  previous = f.DEADLINE; f.DEADLINE = DEADLINE
  try: result = f.verify_worker(WorkerView(root),index,labels)
  finally: f.DEADLINE = previous
  require(inventory(root) == whole, 'Whole real inventory changed across worker-stage view')
  cpu_only(); return result


def feature_arrays(auth):
  import numpy as np
  import ml_dtypes
  rows = auth['index']['rows']; require(len(rows) == 512, 'All512 aligned feature rows required')
  h, q, p, targets = [], [], [], []
  for row,label in zip(rows,auth['labels']['rows']):
    check(); unit = row['global_unit']; item = auth['inventory']['files'][f'unit_{unit:03d}/current.npz']
    receipt = decode(bound(auth['inventory']['files'][f'unit_{unit:03d}/complete.json']))
    require(receipt['source_id'] == row['source_id'] == label['source_id'] and
        receipt['current_label_sha256'] == digest(canonical(label)), 'Feature/physical target source/order differs')
    with np.load(io.BytesIO(bound(item)),allow_pickle=False) as archive:
      bits = archive['h']; qrow, prow = archive['q_prob'], archive['p_prob']
      require(bits.dtype == np.uint16 and bits.shape == (4096,) and
          all(x.dtype == np.float32 and x.shape == (32,32) for x in (qrow,prow)), 'Actual BF16 bits/F32 probability geometry required')
      for name,value in (('h',bits.view(ml_dtypes.bfloat16)),('q_prob',qrow),('p_prob',prow)):
        from scripts.dl_normal_feature_run import typed_spec
        require(typed_spec(value) == receipt['feature_schema'][name], 'Feature bits/schema changed before analysis')
      h.append(bits.view(ml_dtypes.bfloat16).astype(np.float64)); q.append(qrow.astype(np.float64)); p.append(prow.astype(np.float64))
    targets.append([label['current_values'][name] for name in TARGETS])
  return dict(h=np.stack(h),q_prob=np.stack(q),p_prob=np.stack(p),targets=np.asarray(targets,np.float64),
      families=np.array([row['family'] for row in rows],np.int64))


def calculate(features):
  import numpy as np
  from analysis import dl_normal_readout as math
  h, q, p, y, families = (features[n] for n in ('h','q_prob','p_prob','targets','families'))
  require(h.shape == (512,4096) and q.shape == p.shape == (512,32,32) and y.shape == (512,2) and
      np.array_equal(families,np.repeat(np.arange(32),16)), 'Fixed complete ordered reference0 readout geometry required')
  beliefs = dict(h=None,hq=math.belief_features(q),hp=math.belief_features(p)); heads = {}; risks = {}
  for name, belief in beliefs.items():
    check(); model = math.fit(h[:128],None if belief is None else belief[:128],y[:128])
    prediction = math.predict(model,h,belief)
    risk = {split:math.family_risks(prediction[part],y[part],families[part]) for split,part in
        (('fit',slice(0,128)),('held',slice(128,512)))}
    heads[name] = dict(fitted=model,predictions=prediction,residuals=prediction-y,
        total_effective_df_including_intercept=model['effective_df']+1,
        degenerate_history=model['history']['rms'] == 0,
        degenerate_belief=None if model['belief'] is None else model['belief']['rms'] == 0)
    risks[name] = risk
  comparisons = {}
  for name, reference, alternative in (('hq_over_h','h','hq'),('hq_over_hp','hp','hq'),('hp_over_h','h','hp')):
    comparisons[name] = dict(reference=reference,alternative=alternative,
        fitting_descriptive_difference=risks[reference]['fit']['mean']-risks[alternative]['fit']['mean'],
        held=math.improvement(risks[reference]['held'],risks[alternative]['held']))
  return dict(math_version=math.VERSION,ridge=math.RIDGE,target_names=list(TARGETS),
      targets_original_units=y,families=families,heads=heads,absolute_risks=risks,comparisons=comparisons,
      bootstrap_scope='Pointwise paired24 held-family resampling, conditional on fixed8 fit families; no multiplicity correction',
      scientific_verdict=False,actor_use_established=False,causal_mediation_established=False,method_effectiveness_established=False)


def write_result(output, result):
  import numpy as np
  arrays = {}
  def pack(value, path='result'):
    if type(value) is np.ndarray:
      require(value.dtype.kind in 'buif' and np.isfinite(value).all(), 'Finite numeric analysis arrays required')
      key = 'a'+str(len(arrays)); arrays[key] = value
      return dict(array=key,path=path,dtype=str(value.dtype),shape=list(value.shape),raw_sha256=digest(value.tobytes()))
    if isinstance(value, np.generic): return value.item()
    if type(value) is dict: return {key:pack(child,path+'/'+key) for key,child in value.items()}
    if type(value) in (tuple,list): return [pack(child,path+'/'+str(i)) for i,child in enumerate(value)]
    return value
  metadata = pack(result)
  with (output/'readout.npz').open('xb') as stream:
    np.savez(stream,**arrays); stream.flush(); os.fsync(stream.fileno())
  metadata['numeric_artifact'] = pin(output/'readout.npz')
  with (output/'readout.json').open('xb') as stream:
    stream.write(canonical(metadata)+b'\n'); stream.flush(); os.fsync(stream.fileno())
  check(); return pin(output/'readout.json')


def fresh_output(path, immutable):
  raw = Path(path).absolute()
  require(not any(p.is_symlink() for p in (raw,*raw.parents)), 'Output path aliases refused')
  output = raw.parent.resolve()/raw.name
  require(not output.exists() and all(not output.is_relative_to(Path(item).resolve()) for item in immutable),
      'Fresh output outside every immutable input/code directory required')
  output.mkdir(parents=True,exist_ok=False); return output


def run(args):
  cpu_only(); before = analysis_binding(args.expected_git); auth = authenticate(args)
  immutable = [ROOT,args.features_run,*[Path(auth['external'][name]['path']).parent for name in auth['external']],
      *[Path(item['path']).parent for item in auth['terminal']['inputs'].values()]]
  output = fresh_output(args.output,immutable)
  result = calculate(feature_arrays(auth))
  require(inventory(args.features_run) == auth['inventory'], 'Complete feature input changed during readout')
  for item in (*auth['external'].values(),*auth['terminal']['inputs'].values()): bound(item)
  require(analysis_binding(args.expected_git) == before, 'Analysis Git/working bytes changed during readout')
  cpu_only(); check()
  result.update(version=VERSION,role='reference0',producer_git=PRODUCER_GIT,analysis_git=args.expected_git,
      source_rows=[{n:row[n] for n in ('source_id','global_unit','family','replicate','split','plan','prefix_typed_sha256')}
          for row in auth['index']['rows']],external_authority=auth['external'],source_files=before,
      producer_verifier_sources=auth['reused_verifier_sources'],probability_source=auth['probability_source'],
      input_inventory=auth['inventory'],whole_limit_seconds=args.seconds,elapsed_seconds=time.monotonic()-ENTRY_STARTED,
      GPU_used=False,physics_reexecuted=False,NN_reexecuted=False,training_updates=0,
      actual_outer_exit0_and_full_wall_within_explicit_budget_still_required=True)
  item = write_result(output,result); check()
  print(json.dumps(dict(version=VERSION,report=item,complete_units=512,analysis_git=args.expected_git,
      producer_git=PRODUCER_GIT,elapsed_seconds=time.monotonic()-ENTRY_STARTED,
      actual_outer_exit_still_required=True)),flush=True)
  return item


def main():
  global DEADLINE
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('features-run','feature-controller-terminal','feature-controller-sha','feature-outer-execution',
      'feature-outer-sha','expected-git','output'): parser.add_argument('--'+name,required=True)
  parser.add_argument('--seconds',type=int,required=True,help='Explicit adopted whole-entry CPU budget; no default')
  args = parser.parse_args(); require(args.seconds > 0, 'Positive explicit CPU budget required')
  DEADLINE = ENTRY_STARTED+args.seconds
  if os.name == 'posix':
    def expired(signum,frame): raise TimeoutError('Whole explicit readout CPU deadline exceeded')
    signal.signal(signal.SIGALRM,expired); signal.setitimer(signal.ITIMER_REAL,max(.001,DEADLINE-time.monotonic()))
  os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',MUJOCO_GL='disable')
  for name in ('MUJOCO_EGL_DEVICE_ID','PYOPENGL_PLATFORM'): os.environ.pop(name,None)
  check(); run(args)


if __name__ == '__main__': main()
