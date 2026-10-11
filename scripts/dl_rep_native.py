"""DL-rep-native-assay-r1: authenticated A learning and fresh native evaluation.

The stdlib parent owns all workers and the entire 7200-second entry. CPU
fixtures never authorize scientific execution. All actual GPU work is limited
to the assigned sv3 physical GPU4. No original production file is modified.
"""
import time
ENTRY_STARTED = time.monotonic()
import os
import signal
import sys
OWNED = set()


def expire(signum, frame):
  for pid in tuple(OWNED):
    try: os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError: pass
  os._exit(124)


if __name__ == '__main__' and os.name == 'posix':
  signal.signal(signal.SIGALRM, expire)
  signal.signal(signal.SIGTERM, expire)
  signal.setitimer(signal.ITIMER_REAL, max(.001, 7200-(time.monotonic()-ENTRY_STARTED)))

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import pickle
import re
import socket
import subprocess

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'DL-rep-native-assay-r1'
SECONDS = 7200
SEEDS = tuple(range(1024, 1056))
GROUPS = ('0', 'B', 'A')
PHASES = ('authenticate', 'A', 'freeze', 'eval0', 'evalB', 'evalA', 'statistics')
PRIMARY_GIT = 'c030e26f688ed227aa43866bd68dcf4352a073e3'
AUDITOR_GIT = 'd987650c749b8917e2ea80a719d586232088f501'
OLD_AUDIT_SHA = 'ba4c7d9c5d47baeb83f48eb2cde8eca5637186c1abe6ae42a961d13fe4fc8a22'
OLD_AUDIT_SUPERVISOR_SHA = '3496eadb986937e2d04c061863e5a065c08fca0d302b4a231b4c4fc4ba4e2732'
OLD_AUDIT_TERMINAL_SHA = '6a97200e5511212ae28481615dbe733e89e7c03af0d75ee8392365b043bc2b83'


def require(condition, message):
  if not condition: raise ValueError(message)


def sha(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
  return digest.hexdigest()


def read_json(path):
  def pairs(items):
    result = {}
    for key, value in items:
      require(key not in result, 'Duplicate JSON key: '+key); result[key] = value
    return result
  return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs,
      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def jsonify(value):
  if isinstance(value, dict): return {str(k): jsonify(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)): return [jsonify(v) for v in value]
  if hasattr(value, 'tolist'): return jsonify(value.tolist())
  if isinstance(value, Path): return str(value)
  return value


def write_json(path, value):
  with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(jsonify(value), stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write('\n')


def save_pickle(path, value):
  with Path(path).open('xb') as stream: pickle.dump(value, stream, protocol=5)
  return sha(path)


def load_payload(path, pin):
  payload = Path(path).read_bytes()
  require(hashlib.sha256(payload).hexdigest() == pin, 'Bytes changed before decode: '+str(path))
  return pickle.loads(payload)


def cleanup(child):
  if os.name == 'posix':
    try: os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError: pass
  elif child.poll() is None: child.kill()
  child.wait(timeout=5); OWNED.discard(child.pid)


def owned_output(command):
  own_group = os.name == 'posix' and 'DL_REP_NATIVE_PARENT' not in os.environ
  child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
      stderr=subprocess.PIPE, start_new_session=own_group)
  if own_group: OWNED.add(child.pid)
  try:
    out, err = child.communicate(timeout=30)
    require(child.returncode == 0, 'Metadata helper failed: '+err.decode(errors='replace')[:256])
    return out
  finally:
    if own_group: cleanup(child)
    elif child.poll() is None: child.kill(); child.wait(timeout=5)


def git_binding(expected):
  require(type(expected) is str and re.fullmatch('[0-9a-f]{40}', expected), 'Full SHA required')
  require(owned_output(['git', 'rev-parse', 'HEAD']).decode().strip() == expected, 'Actual HEAD differs')
  require(not owned_output(['git', 'status', '--porcelain', '--untracked-files=all']), 'Clean checkout required')
  names = owned_output(['git', 'ls-files']).decode().splitlines()
  names = [x for x in names if x.endswith(('.py', '.yaml', '.yml', '.lock')) or
      x in ('docs/DL_REP_NATIVE_ASSAY.md', 'docs/DL_RESET_JOINT_ASSAY.md')]
  require('scripts/dl_rep_native.py' in names and 'scripts/dl_native_actor.py' in names and
      'analysis/dl_rep_native_audit.py' in names, 'Incomplete tracked implementation')
  pins = {}
  for name in names:
    blob = owned_output(['git', 'show', expected+':'+name])
    require((ROOT/name).read_bytes() == blob, 'Git/file bytes differ: '+name)
    pins[name] = hashlib.sha256(blob).hexdigest()
  return pins


def run_owned(command, log, deadline, env):
  require(time.monotonic() < deadline, 'Late before worker creation')
  child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
      stderr=subprocess.STDOUT, start_new_session=os.name == 'posix')
  OWNED.add(child.pid); started = time.monotonic(); timed_out = False
  try: code = child.wait(timeout=max(.001, deadline-time.monotonic()))
  except subprocess.TimeoutExpired: code = 124; timed_out = True
  finally: cleanup(child)
  return dict(worker_pid=child.pid, worker_terminal=child.poll() is not None,
      owned_group_cleanup_requested=os.name == 'posix', exit_code=code,
      timed_out=timed_out, elapsed_seconds=time.monotonic()-started)


def successful(value, key, limit):
  require(type(value.get('exit_code')) is int and value['exit_code'] == 0 and
      value.get('timed_out') is False and value.get('worker_terminal') is True,
      'Missing explicit successful terminal')
  elapsed = value.get(key)
  require(type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 < elapsed <= limit,
      'Whole-entry elapsed bound differs')


def authenticate_old(args, output, check):
  """Reuse an authenticated completed object; never decode old blind arrays."""
  from scripts import dl_reset_joint_audit as audit
  from scripts import dl_reset_learning as learning
  check()
  old = Path(args.old_primary).resolve(strict=True)
  auditor = Path(args.old_audit).resolve(strict=True)
  require(sha(auditor/'audit.json') == OLD_AUDIT_SHA and
      sha(auditor/'supervisor_complete.json') == OLD_AUDIT_SUPERVISOR_SHA and
      sha(args.old_audit_cli_terminal) == OLD_AUDIT_TERMINAL_SHA, 'Prior audit endpoint bytes differ')
  independently = read_json(auditor/'audit.json')
  supervisor = read_json(auditor/'supervisor_complete.json')
  terminal = read_json(args.old_audit_cli_terminal)
  require(independently['status'] == 'passed' and
      independently['primary_git_sha'] == PRIMARY_GIT and
      independently['auditor_git_sha'] == AUDITOR_GIT and
      independently['complete_primary_exit_and_parent_preblind_seal_verified'] is True and
      independently['actual_gpu_devices_used'] == 0 and
      independently['actual_independent_physics_steps'] == 3360, 'Prior audit scope differs')
  successful(supervisor, 'elapsed_parent_entry_seconds', 300)
  require(supervisor['complete_verified'] is True and supervisor['actual_git_sha'] == AUDITOR_GIT and
      supervisor['verification']['audit_sha256'] == OLD_AUDIT_SHA and
      supervisor['budget_sha256'] == sha(auditor/'budget.json') and
      supervisor['worker_exit_sha256'] == sha(auditor/'worker_exit_code.txt'), 'Prior audit complete chain differs')
  require(terminal['actual_git_sha'] == AUDITOR_GIT and terminal['exit_code'] == 0 and
      terminal['worker_terminal'] is True and terminal['worker_pid'] == supervisor['parent_pid'] and
      0 < terminal['elapsed_cli_seconds'] <= 300, 'Prior independent CLI terminal differs')
  code = git_binding(args.expected_git)
  for name in audit.FILES:
    old_blob = owned_output(['git', 'show', PRIMARY_GIT+':'+name])
    require(hashlib.sha256(old_blob).hexdigest() == code[name], 'Original 19 producer files changed: '+name)
  authority = audit.primary_authority(old, args.old_primary_cli_terminal, code)
  require(authority['primary_git'] == PRIMARY_GIT and
      str(old) == independently['primary'], 'Prior producer/auditor path binding differs')
  for name, pin in authority['pins'].items():
    require(independently['primary_json_and_generated_sha256'][name] == pin, 'Prior audit does not bind '+name)
  b = read_json(old/'B-complete.json'); baseline = read_json(old/'baseline-complete.json')
  audit.validate_learning(b['learning'], 'B', b['source_binding'])
  audit.authenticate_source(b['source_binding'])
  require(baseline['context_sha256'] == authority['frozen']['files']['contexts.pkl'] == b['context_sha256'],
      'Prior common annotation context differs')
  immutable = {str((old/name).resolve()): pin for name, pin in authority['pins'].items()
      if name != 'external_cli_terminal'}
  immutable[str(Path(args.old_primary_cli_terminal).resolve())] = authority['pins']['external_cli_terminal']
  for path in (auditor/'audit.json', auditor/'supervisor_complete.json', auditor/'budget.json',
      auditor/'worker_exit_code.txt', Path(args.old_audit_cli_terminal)):
    immutable[str(path.resolve())] = sha(path)
  for row in b['source_binding']['all_32_artifact_hashes']:
    immutable[row['path']] = row['sha256']
  for paths_key, pins_key in (('source_json_paths', 'source_json_sha256'),
      ('source_terminal_paths', 'source_terminal_sha256')):
    for name, path in b['source_binding'][paths_key].items():
      immutable[str(Path(path).resolve())] = b['source_binding'][pins_key][name]
  immutable[str(Path(args.checkpoint).resolve())] = audit.MODEL_PINS['checkpoint_sha256']
  immutable[str(Path(args.config).resolve())] = audit.MODEL_PINS['config_source_sha256']
  for path, pin in immutable.items(): check(); require(sha(path) == pin, 'Input pin differs: '+path)
  require(b['learning']['version'] == learning.VERSION, 'Original layout version changed')
  write_json(output/'old-authenticated.json', dict(version=VERSION, actual_git_sha=args.expected_git,
      primary=str(old), primary_git_sha=PRIMARY_GIT, auditor_git_sha=AUDITOR_GIT,
      old_audit_sha256=OLD_AUDIT_SHA, immutable_inputs_sha256=immutable,
      source_binding=b['source_binding'], original_B_learning=b['learning'],
      old_files_sha256=authority['frozen']['files'],
      original_19_producer_files_unchanged=True, old_B_retrained=False,
      original_blind_arrays_decoded=False, old_full_exit_and_independent_audit_authenticated=True))


def authenticate_worker(args):
  output = Path(args.output).resolve(strict=True)
  budget = read_json(output/'budget.json')
  require(os.environ.get('DL_REP_NATIVE_PARENT') == str(budget['parent_pid']) and
      os.getppid() == budget['parent_pid'], 'Worker must have live authentic parent')
  os.kill(os.getppid(), 0)
  require(budget['actual_git_sha'] == args.expected_git and budget['seconds'] == SECONDS and
      budget['version'] == VERSION, 'Parent budget/version differs')
  deadline = budget['monotonic_start']+SECONDS
  def check(): require(time.monotonic() < deadline, 'Whole-entry 7200-second budget exhausted')
  check(); require(git_binding(args.expected_git) == budget['source_files_sha256'], 'Worker code differs')
  if args.worker_phase in ('eval0', 'evalB', 'evalA', 'statistics'):
    require(type(args.seal) is str and sha(output/'freeze.json') == args.seal,
        'Evaluation predates parent frozen digest')
  return output, budget, check


def load_agent(args, payload, *, policy):
  from scripts import dl_diagnostic as diagnostic
  from scripts import dl_predictive_mc as original
  require(sha(args.config) == original.CONFIG_SHA, 'Original config bytes differ')
  config = diagnostic.load_config(args.config, args.task, args.output, 'cuda').update({'jax.enable_policy': policy})
  require(config.task == 'dmc_quadruped_walk' and config.seed == 0, 'Task/training seed differs')
  agent = original.agent_from_spaces(config)
  agent.load(payload)
  return agent, config


def learning_phase(args, output, check):
  from scripts import dl_resources as resource
  from scripts import dl_reset_joint as old
  from scripts import dl_reset_all_rep as intervention
  from scripts import dl_predictive_mc as original
  auth = read_json(output/'old-authenticated.json')
  arrays, binding = old.load_cal(args, check)
  require(binding['all_32_artifact_hashes'] == auth['source_binding']['all_32_artifact_hashes'],
      'Actual eight calibration source identity differs')
  contexts = load_payload(Path(args.old_primary)/'contexts.pkl', auth['old_files_sha256']['contexts.pkl'])
  checkpoint = load_payload(args.checkpoint, original.CHECKPOINT_SHA)
  device = resource.require_resource(args.resource_record)
  require(device['physical_gpu'] == 4, 'Only physical GPU4 allowed for this object')
  with intervention.allrep_branch():
    agent, config = load_agent(args, checkpoint, policy=False)
    device['runtime_compute_start'] = resource.verify_runtime(device, compute=True)
    outcome = intervention.run_a(agent, checkpoint, arrays,
        {k: binding['all_32_artifact_hashes'][k]['sha256'] for k in old.CAL}, contexts,
        check=check, reference_b=auth['original_B_learning'])
  final = outcome.pop('final_checkpoint'); outcome.pop('features')
  pin = save_pickle(output/'A-checkpoint.pkl', final)
  device['runtime_compute_end'] = resource.verify_runtime(device, compute=True)
  check()
  write_json(output/'A-complete.json', dict(version=VERSION, actual_git_sha=args.expected_git,
      phase='A', learning=outcome, checkpoint_sha256=pin, device=device, config=config.flat,
      source_binding=binding, original_B_learning_reference_authenticated=True,
      old_authentication_sha256=sha(output/'old-authenticated.json'),
      cal_image_frames_decoded=8008, original_blind_arrays_decoded=False,
      additional_F_measurements_performed=False))


def verify_freeze(output):
  freeze = read_json(output/'freeze.json')
  require(freeze['version'] == VERSION and freeze['groups'] == list(GROUPS) and
      freeze['evaluation_seeds'] == list(SEEDS) and freeze['transitions_per_episode'] == 1000 and
      freeze['primary'] == 'A_minus_B' and freeze['bootstrap_seed'] == 20261026 and
      freeze['bootstrap_draws'] == 2000 and freeze['new_evaluation_started'] is False and
      freeze['additional_F_measurements_performed'] is False, 'Frozen object differs')
  for name, pin in freeze['files_sha256'].items(): require(sha(output/name) == pin, 'Frozen local bytes changed')
  return freeze


def freeze_phase(args, output, check):
  auth = read_json(output/'old-authenticated.json')
  a = read_json(output/'A-complete.json')
  require(a['old_authentication_sha256'] == sha(output/'old-authenticated.json') and
      a['original_B_learning_reference_authenticated'] is True and
      a['learning']['entire_window_rep_gradient_stopped'] is True and
      a['learning']['optimizer_counter']['final'] == 74621, 'A completion scope differs')
  check()
  write_json(output/'freeze.json', dict(version=VERSION, actual_git_sha=args.expected_git,
      groups=list(GROUPS), evaluation_seeds=list(SEEDS), transitions_per_episode=1000,
      expected_physics_transitions=96000, expected_actual_RGB_frames=96096,
      primary='A_minus_B', bootstrap_seed=20261026, bootstrap_draws=2000,
      root_definition='default_rng([0,0x4556414C,environment_seed,step]).uint32_key2',
      new_evaluation_started=False, additional_F_measurements_performed=False,
      monotonic_frozen=time.monotonic(), old_authentication_sha256=sha(output/'old-authenticated.json'),
      files_sha256={name:sha(output/name) for name in
          ('budget.json', 'plan.json', 'old-authenticated.json', 'A-complete.json', 'A-checkpoint.pkl')},
      full_state_digests={'0':auth['original_B_learning']['initial_state_digest'],
          'B':auth['original_B_learning']['final_state_digest'], 'A':a['learning']['final_state_digest']},
      complete_checkpoint_sha256={'0':auth['immutable_inputs_sha256'][str(Path(args.checkpoint).resolve())],
          'B':auth['old_files_sha256']['B-checkpoint.pkl'], 'A':a['checkpoint_sha256']},
      old_B_authenticated_reuse=True, old_B_retrained=False, native_actor_action='official_sample',
      method_effectiveness_established=False))


def snapshot_digest(snapshot):
  return hashlib.sha256(pickle.dumps(snapshot, protocol=5)).hexdigest()


def assert_snapshot_exact(actual, expected):
  import numpy as np
  a, b = actual['suite'], expected['suite']
  require(a['integration'].dtype == b['integration'].dtype == np.float64 and
      a['integration'].shape == b['integration'].shape and
      a['integration'].ndim == 1 and a['integration'].size > 0 and
      np.isfinite(a['integration']).all() and np.isfinite(b['integration']).all() and
      a['integration'].tobytes() == b['integration'].tobytes() and
      pickle.dumps(a['task'], protocol=5) == pickle.dumps(b['task'], protocol=5) and
      a['step_count'] == b['step_count'] and a['reset_next_step'] == b['reset_next_step'] and
      actual['from_dm_done'] == expected['from_dm_done'], 'Full physical snapshot bytes differ')


def evaluate_phase(args, output, check):
  import numpy as np
  from dm_control import suite
  from scripts import dl_resources as resource
  from scripts import dl_reset_learning as learning
  from scripts import dl_native_actor as native
  from scripts import dl_feedback_physics as physical
  from scripts import dl_action_contract as actions
  group = args.worker_phase[4:]
  freeze = verify_freeze(output)
  checkpoint_path = (Path(args.checkpoint) if group == '0' else
      Path(args.old_primary)/'B-checkpoint.pkl' if group == 'B' else output/'A-checkpoint.pkl')
  device = resource.require_resource(args.resource_record)
  require(device['physical_gpu'] == 4, 'This object only uses physical GPU4')
  payload = load_payload(checkpoint_path, freeze['complete_checkpoint_sha256'][group])
  agent, config = load_agent(args, payload, policy=True)
  require(learning.tree_digest(agent.save()) == freeze['full_state_digests'][group], 'Actual full reloaded model differs')
  # A fresh exact policy copy supports the public production oracle. Private
  # native inference itself reads current agent.params.
  agent.sync_policy()
  adapter = native.NativeActor(agent)
  device['runtime_compute_start'] = resource.verify_runtime(device, compute=True)
  records, reset_receipts = [], []
  runtime_render_verified = False
  for seed in SEEDS:
    check(); env = suite.load('quadruped', 'walk', task_kwargs={'random': seed})
    wrapped = physical.FeedbackEnvironment(env)
    original_render = env.physics.render
    render_count = [0]
    def guarded_render(*a, **kw):
      before = wrapped.snapshot()
      image = original_render(*a, **kw)
      assert_snapshot_exact(wrapped.snapshot(), before)
      render_count[0] += 1
      return image
    env.physics.render = guarded_render
    try:
      observation = wrapped.reset()
      initial = wrapped.snapshot()
      require(initial['suite']['step_count'] == 0 and not initial['from_dm_done'] and
          not initial['suite']['reset_next_step'] and bool(observation['is_first']), 'Fresh official reset differs')
      if not runtime_render_verified:
        device['runtime_compute_and_graphics_start'] = resource.verify_runtime(device, compute=True, graphics=True)
        runtime_render_verified = True
      snapshots, stages = [initial], []
      reset_receipt = dict(seed=seed, initial_snapshot_sha256=snapshot_digest(initial),
          initial_rgb_sha256=hashlib.sha256(observation['image'].tobytes()).hexdigest(),
          rendering_full_state_byte_exact=True)
      def transition(raw):
        check()
        stage = actions.production_action_stages(raw, wrapped.low, wrapped.high)
        next_obs, applied = wrapped.step(raw)
        ts = wrapped.recorder.last_timestep
        require(ts is not None and not ts.first() and np.asarray(ts.reward).dtype == np.float64 and
            applied.dtype == np.float64 and applied.tobytes() == stage['applied'].tobytes(),
            'Production physical action/timestep differs')
        snapshots.append(wrapped.snapshot()); stages.append(stage)
        return dict(observation=next_obs, action_applied=applied.copy(),
            reward_f64=np.asarray(ts.reward).copy(), discount_f64=np.asarray(ts.discount, np.float64))
      trajectory = native.evaluate_episode(adapter, seed, observation, transition, check=check,
          oracle=adapter.oracle)
      require(len(snapshots) == render_count[0] == 1001 and len(stages) == 1000 and
          snapshots[-1]['suite']['step_count'] == 1000 and wrapped.recorder.steps == 1000,
          'Full episode physics/render count differs')
      for index, saved in enumerate(snapshots):
        require(saved['suite']['step_count'] == index and
            bool(saved['from_dm_done']) == (index == 1000) and
            bool(saved['suite']['reset_next_step']) == (index == 1000), 'Physical episode clock/reset differs')
      record = dict(version=VERSION, group=group, seed=seed, trajectory=trajectory,
          snapshots=snapshots, stages=stages, reset_receipt=reset_receipt,
          camera=2, size=[64,64], proprio=False, action_repeat=1,
          actual_physics_transitions=1000, actual_RGB_frames=1001,
          production_chain=wrapped.chain)
      name = f'{group}-episode-{seed}.pkl'
      pin = save_pickle(output/name, record)
      rewards = np.asarray(trajectory['reward_f64'])
      require(rewards.dtype == np.float64 and rewards.shape == (1000,), 'Actual full F64 reward array required')
      records.append(dict(group=group, seed=seed, file=name, sha256=pin,
          return_f64=float(rewards.sum(dtype=np.float64))))
      reset_receipts.append(reset_receipt)
      print(json.dumps(dict(group=group, seed=seed, completed_episodes=len(records),
          actual_steps=1000, return_f64=records[-1]['return_f64'])), flush=True)
    finally:
      env.physics.render = original_render
      wrapped.close()
  adapter.assert_frozen(full=True)
  require(learning.tree_digest(agent.save()) == freeze['full_state_digests'][group], 'Evaluation changed full state')
  device['runtime_compute_and_graphics_end'] = resource.verify_runtime(device, compute=True, graphics=True)
  check()
  write_json(output/(args.worker_phase+'-complete.json'), dict(version=VERSION,
      actual_git_sha=args.expected_git, phase=args.worker_phase, group=group, records=records,
      reset_receipts=reset_receipts, actual_physics_transitions=32000, actual_RGB_frames=32032,
      native_model_calls=32000, production_oracle_calls=96, additional_F_measurements_performed=False,
      freeze_sha256=sha(output/'freeze.json'), device=device, config=config.flat,
      full_state_digest_before_and_after=freeze['full_state_digests'][group],
      native_interface_receipt=adapter.receipt(), complete_normal_posterior_feedback=True))


def statistics(returns):
  import numpy as np
  require(type(returns) is dict and set(returns) == set(GROUPS), 'Exactly three frozen model groups required')
  values = {g:np.asarray(returns[g], np.float64) for g in GROUPS}
  require(all(v.shape == (32,) and np.isfinite(v).all() for v in values.values()), 'All 32 returns per arm required')
  draws = np.random.default_rng(20261026).integers(0,32,(2000,32))
  def summary(array):
    interval = np.quantile(array[draws].mean(-1), (.025,.975), method='linear')
    return dict(values=array.tolist(), mean=float(array.mean()), std=float(array.std()),
        minimum=float(array.min()), maximum=float(array.max()), ci95=interval.tolist())
  models = {g:summary(v) for g,v in values.items()}
  comparisons = {}
  for name, x, y in (('A_minus_B','A','B'), ('A_minus_0','A','0'), ('B_minus_0','B','0')):
    measured = summary(values[x]-values[y])
    lo, hi = measured['ci95']
    measured['ci_zero_side'] = [int(np.sign(lo)), int(np.sign(hi))]
    measured['classification'] = 'positive_finite_net_effect' if lo > 0 else (
        'positive_effect_not_supported' if hi <= 0 else 'inconclusive')
    comparisons[name] = measured
  return dict(version=VERSION, seeds=list(SEEDS), groups=list(GROUPS), horizon=1000,
      bootstrap=dict(seed=20261026, samples=2000, unit='whole_episode',
          method='percentile_linear', paired=True), models=models,
      comparisons=comparisons, primary='A_minus_B', method_effectiveness_established=False,
      independent_second_signal_validated=False)


def statistics_phase(args, output, check):
  freeze = verify_freeze(output)
  receipts = {g:read_json(output/('eval'+g+'-complete.json')) for g in GROUPS}
  reference = receipts['0']['reset_receipts']
  returns, manifest = {}, []
  for group, receipt in receipts.items():
    require(receipt['reset_receipts'] == reference and receipt['freeze_sha256'] == sha(output/'freeze.json') and
        receipt['actual_physics_transitions'] == 32000 and receipt['actual_RGB_frames'] == 32032 and
        receipt['full_state_digest_before_and_after'] == freeze['full_state_digests'][group],
        'New initial reset/image, state or count pairing differs')
    rows = receipt['records']
    require([r['seed'] for r in rows] == list(SEEDS) and all(r['group'] == group for r in rows),
        'Fixed complete evaluation seed order differs')
    for row in rows: check(); require(sha(output/row['file']) == row['sha256'], 'New trajectory bytes changed')
    returns[group] = [r['return_f64'] for r in rows]; manifest.extend(rows)
  auth = read_json(output/'old-authenticated.json')
  for path, pin in auth['immutable_inputs_sha256'].items():
    check(); require(sha(path) == pin, 'Original scientific input changed during new experiment')
  environment = {name:importlib.metadata.version(name) for name in
      ('numpy','jax','ninjax','mujoco','dm-control')}
  check()
  write_json(output/'result.json', dict(version=VERSION, actual_git_sha=args.expected_git,
      statistics=statistics(returns), record_manifest=manifest, environment=environment,
      freeze_sha256=sha(output/'freeze.json'),
      freeze_seal_parent_sha256=sha(output/'freeze-seal-parent.json'),
      old_authentication_sha256=sha(output/'old-authenticated.json'),
      evaluation_complete_sha256={g:sha(output/('eval'+g+'-complete.json')) for g in GROUPS},
      actual_physics_transitions=96000, actual_RGB_frames=96096, actual_training_updates=128,
      original_B_retrained=False, original_inputs_unchanged=True,
      independent_signal_identified=False, selective_gate_validated=False,
      ordinary_replay_training_validated=False, method_effectiveness_established=False,
      conditional_on='one training seed/checkpoint, fixed repeated offline cal128 updates and frozen actor/eval RNG',
      minimal_practical_effect_declared=False,
      full_goal_complete=False, decision='stop_after_fixed_rep_native_assay'))


def worker(args):
  sys.path.insert(0, str(ROOT))
  output, budget, check = authenticate_worker(args)
  if args.worker_phase == 'authenticate': authenticate_old(args, output, check)
  elif args.worker_phase == 'A': learning_phase(args, output, check)
  elif args.worker_phase == 'freeze': freeze_phase(args, output, check)
  elif args.worker_phase == 'statistics': statistics_phase(args, output, check)
  else: evaluate_phase(args, output, check)
  check(); require(git_binding(args.expected_git) == budget['source_files_sha256'], 'Final worker code changed')


def parent(args):
  require(os.name == 'posix' and socket.gethostname() == 'lyg0326', 'Only assigned sv3 may execute')
  sources = git_binding(args.expected_git)
  output = Path(args.output).resolve()
  require(not output.exists(), 'New attempt directory required'); output.mkdir(parents=True)
  deadline = ENTRY_STARTED+SECONDS
  write_json(output/'budget.json', dict(version=VERSION, actual_git_sha=args.expected_git,
      parent_pid=os.getpid(), seconds=SECONDS, monotonic_start=ENTRY_STARTED,
      planned_physical_gpu=4, max_gpu_devices=1, max_gpu_hours=2, source_files_sha256=sources))
  write_json(output/'plan.json', dict(version=VERSION, groups=list(GROUPS), evaluation_seeds=list(SEEDS),
      primary='A_minus_B', transitions_per_episode=1000, actual_training_updates_per_new_arm=128,
      old_B_retrained=False, additional_F_measurements_performed=False,
      independent_signal_identified=False, selective_gate_validated=False))
  device = read_json(args.resource_record)
  phases, seal, code, timed_out = [], None, 0, False
  for phase in PHASES:
    is_gpu = phase in ('A','eval0','evalB','evalA')
    env = dict(os.environ, DL_REP_NATIVE_PARENT=str(os.getpid()),
        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    resource_path = args.resource_record
    if is_gpu:
      sys.path.insert(0, str(ROOT))
      from scripts import dl_resources as resource
      original_command = resource.command_output
      resource.command_output = lambda command:owned_output(command).decode()
      try: fresh = resource.refresh(device)
      finally: resource.command_output = original_command
      require(fresh['physical_gpu'] == 4, 'No resource substitution')
      resource_path = str(output/(phase+'-resource.json')); write_json(resource_path, fresh)
    else: env.update(CUDA_VISIBLE_DEVICES='', MUJOCO_GL='disable', JAX_PLATFORMS='cpu')
    command = [sys.executable, str(Path(__file__).resolve()), '--worker-phase', phase,
        '--resource-record', resource_path]
    if seal: command += ['--seal', seal]
    for key in ('dataset','artifact_audit','physics_audit','checkpoint','config','old_primary',
        'old_primary_cli_terminal','old_audit','old_audit_cli_terminal','output','expected_git','task'):
      command += ['--'+key.replace('_','-'), str(getattr(args,key))]
    with (output/(phase+'.log')).open('x', encoding='utf-8') as log:
      terminal = run_owned(command, log, deadline, env)
    phases.append(dict(phase=phase, is_gpu=is_gpu, **terminal))
    write_json(output/(phase+'-exit.json'), phases[-1]); print(json.dumps(phases[-1]), flush=True)
    code, timed_out = terminal['exit_code'], terminal['timed_out']
    if code or time.monotonic() >= deadline: break
    if phase == 'freeze':
      seal = sha(output/'freeze.json')
      write_json(output/'freeze-seal-parent.json', dict(version=VERSION, actual_git_sha=args.expected_git,
          freeze_sha256=seal, monotonic_sealed=time.monotonic(), new_evaluation_started=False))
  complete = code == 0 and len(phases) == len(PHASES) and time.monotonic() < deadline
  if complete:
    verify_freeze(output)
    result = read_json(output/'result.json')
    require(len(result['record_manifest']) == 96 and result['actual_physics_transitions'] == 96000,
        'Partial evaluation cannot complete')
    require(git_binding(args.expected_git) == sources, 'Final parent code differs')
  receipt = dict(version=VERSION, actual_git_sha=args.expected_git, parent_pid=os.getpid(), phases=phases,
      exit_code=124 if timed_out else (code if complete or code else 1),
      timed_out=timed_out, complete_verified=complete,
      elapsed_parent_entry_seconds=time.monotonic()-ENTRY_STARTED, budget_sha256=sha(output/'budget.json'),
      actual_gpu_devices_used=1 if complete else None, independent_audit_complete=False,
      method_effectiveness_established=False, result_sha256=sha(output/'result.json') if complete else None)
  pending = output/'supervisor-receipt-pending.json'; write_json(pending, receipt)
  if time.monotonic() >= deadline:
    write_json(output/'late-supervisor.json', dict(exit_code=124, incomplete=True)); return 124
  os.replace(pending, output/'supervisor_complete.json')
  if time.monotonic() >= deadline:
    os.replace(output/'supervisor_complete.json', output/'late-provisional-supervisor.json')
    write_json(output/'late-supervisor.json', dict(exit_code=124, incomplete=True)); return 124
  return receipt['exit_code'] if complete or receipt['exit_code'] else 1


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('dataset','artifact-audit','physics-audit','checkpoint','config','old-primary',
      'old-primary-cli-terminal','old-audit','old-audit-cli-terminal','output','expected-git','resource-record'):
    parser.add_argument('--'+name, required=True)
  parser.add_argument('--task', default='dmc_quadruped_walk')
  parser.add_argument('--worker-phase', choices=PHASES)
  parser.add_argument('--seal')
  args = parser.parse_args()
  if args.worker_phase: worker(args); return 0
  return parent(args)


if __name__ == '__main__': sys.exit(main())
