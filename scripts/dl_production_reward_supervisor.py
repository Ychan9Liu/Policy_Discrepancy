"""DL-production-reward-r1 stdlib parent, fixed 1800s and one owned worker.

Importing this module changes no CUDA/JAX environment. The real CLI arms its
entry timer before other imports/parsing. Cleanup/failure evidence may finish
after the deadline, but a late attempt never publishes scientific completion.
"""
import time
ENTRY_STARTED = time.monotonic()
import os
import signal
_OWNED_HELPER_GROUPS = set()


def _entry_expire(signum, frame):
  # The entry phase may own short-lived Git helpers before the model group
  # exists. They have separate sessions; never signal the caller's group.
  for pid in tuple(_OWNED_HELPER_GROUPS):
    try: os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError: pass
  os._exit(124)


_CLI_ENTRY_ALARM = __name__ == '__main__' and os.name == 'posix'
if _CLI_ENTRY_ALARM:
  signal.signal(signal.SIGALRM, _entry_expire)
  signal.setitimer(signal.ITIMER_REAL, max(1e-6, 1800 - (time.monotonic() - ENTRY_STARTED)))

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import socket
import subprocess
import sys

VERSION = 'DL-production-reward-r1'
IMPLEMENTATION = 'DL-production-reward-supervisor-engineering-r1'
SECONDS = 1800
ROOT = Path(__file__).resolve().parents[1]
EPISODES = (9, 14, 15, 18, 20, 24, 30, 31)
POSITIONS = (1, 300, 600, 900)
SOURCE_GIT = 'f1ae4d6513e81ed680992b64ba8ebade7d256d52'
SOURCE_PINS = {
    'plan.json': '474bad6adf6558b7a47c40711546dc4cf2adc870684b8cff2766a802605f44d6',
    'result.json': '06df6efa5f9ce84b428921b59fa508df8badd1dd54e30335bf2d3a273e585fad',
    'action_information_complete.json': '28481a749476be0b3e8c05f28e719cb22e4cfffb638e56c04d8f86128026d37e',
    'budget.json': 'ddc8f43a3f4b57c11858be61f4c31315484c8c42b1e887f48391e71d077e2a68',
    'supervisor_complete.json': 'a920a0191429c7ffd05af5a897a30839da47d046d9126a1002a4997928dc1ea3',
    'independent-numeric-audit-v1.json': 'cbb99e43ad72a6960d86e5044455a18fecc38f068172defe60deb0d82f8678e5'}
MODEL_PINS = dict(
    checkpoint_sha256='2ab693c8ded4d8f2186bd489aa8b77498448ba2f2c331090159da4e5f08056b9',
    params_sha256='14859c4e129b51aa488c10eab09f16d56fd110c76239b2c9072f415829d111c2',
    config_source_sha256='11428512fdc9b9e04a5384e419c5abcef93e5429db0a3327d05ccb0d4bd20445')
SOURCE_FILES = (
    'scripts/dl_production_reward.py', 'scripts/dl_production_reward_source.py',
    'scripts/dl_production_reward_model.py', 'scripts/dl_production_reward_stats.py',
    'scripts/dl_production_reward_supervisor.py', 'docs/DL_PRODUCTION_REWARD.md',
    'scripts/dl_predictive_mc.py', 'scripts/dl_predictive.py',
    'scripts/dl_diagnostic.py', 'scripts/dl_resources.py',
    'dreamerv3/agent.py', 'dreamerv3/rssm.py', 'embodied/jax/agent.py',
    'embodied/jax/heads.py', 'embodied/jax/outs.py', 'embodied/jax/nets.py',
    'scripts/probe_formal_devices.py',
    'tests/test_dl_production_reward_source.py', 'tests/test_dl_production_reward_model.py',
    'tests/test_dl_production_reward_stats.py', 'tests/test_dl_production_reward_supervisor.py',
    'tests/test_dl_production_reward_worker.py', 'analysis/dl_production_reward_audit.py',
    'tests/test_dl_production_reward_audit.py')
NO_AUTHORITY = dict(training_authorized=False, method_effectiveness_established=False,
    method_evidence=False, real_gate_authority=False, eligible_method_training=False,
    proceed_short_training=False, blind_arrays_read=False,
    current_rgb_increment_over_complete_history_established=False,
    continuing_kl_compression_effect_established=False)
UNKNOWN_ACTUAL = dict(source_provenance_verified=False, real_dataset_reads=None,
    actual_numeric_episode_reads=None, actual_gpu_devices_used=None,
    gpu_devices_used=None, records_decoded=None, physics_steps=None, render_calls=None,
    optimizer_updates=None, actor_calls=None, initial_latents_resampled=None)


def require(condition, message):
  if not condition:
    raise ValueError(message)


def sha(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b''):
      digest.update(block)
  return digest.hexdigest()


def read_json(path):
  def reject(value):
    raise ValueError('Nonfinite JSON constant: ' + value)
  return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)


def finite_tree(value):
  if isinstance(value, dict):
    for child in value.values(): finite_tree(child)
  elif isinstance(value, list):
    for child in value: finite_tree(child)
  elif type(value) is float:
    require(math.isfinite(value), 'Nonfinite B result')


def shape(value, dimensions, leaf=None):
  if not dimensions:
    require(leaf(value) if leaf else type(value) in (int, float) and math.isfinite(value),
            'Invalid finite B result leaf')
    return
  require(type(value) is list and len(value) == dimensions[0], 'Partial B result shape')
  for item in value: shape(item, dimensions[1:], leaf)


def verify_analysis(value):
  """Require the complete fixed statistic tree; arithmetic audit is separate."""
  expected = dict(version=VERSION, episodes=list(EPISODES), positions=list(POSITIONS),
      prediction_shape=[8, 4, 4, 2, 128, 7, 20], truth_shape=[8, 4, 7],
      controls=['zero', 'all+', 'all-', 'lift-LR+', 'lift-LR-', 'extend-diag+', 'extend-diag-'],
      branches=['q', 'p'], horizon=20, blocks=4, paths_per_block=32, future_paths=128,
      initial_samples=4, axes=['episode', 'state', 'initial_sample', 'branch', 'future_path', 'control', 'step'],
      primary_subset='pooled128', negative_bc_mse_clipped=False, initial_samples_averaged_before_loss=False,
      future_paths_independence_verified_by_stats=False, source_provenance_verified_by_stats=False,
      source_or_model_execution_verified_by_stats=False, tie_tolerance=1e-10,
      bootstrap_draws=2000, statistical_seed=20261025, percentile_method='linear',
      mc_error_bound_claimed=False, inputs_unchanged=True, calibration_only=True,
      model_calls=0, physics_steps=0, optimizer_updates=0, gpu_devices_used=0,
      blind_arrays_read=False, records_decoded=False, current_rgb_increment_established=False,
      active_rep_compression_effect_established=False, method_effectiveness_established=False,
      real_gate_authority=False, eligible_method_training=False, training_authorized=False,
      proceed_short_training=False, decision='stop_after_fixed_production_reward_diagnostic')
  for key, target in expected.items():
    require(type(value[key]) is type(target) and value[key] == target, 'Changed B statistics scope: ' + key)
  constant = value['constant']
  require(len(constant['folds']) == 8, 'Partial B LOEO constant')
  for episode, fold in zip(EPISODES, constant['folds']):
    require(fold['heldout_episode'] == episode and fold['training_states'] == 28 and
            fold['training_episodes'] == [x for x in EPISODES if x != episode],
            'B constant train/heldout isolation changed')
    shape(fold['centered_cost_vector'], [7])
  shape(constant['centered_cost_vectors'], [8, 7])
  shape(constant['control_by_episode'], [8], lambda x: type(x) is int and 0 <= x < 7)
  shape(constant['state_mse'], [8, 4])
  shape(constant['state_selected_reward'], [8, 4])
  require(constant['correction'] == 0., 'B constant MC correction changed')
  subsets = value['subsets']
  blocks = dict(first32=[0], first64=[0, 1], pooled128=[0, 1, 2, 3],
                last64=[2, 3], block0=[0], block1=[1], block2=[2], block3=[3])
  require(set(subsets) == set(blocks), 'Partial B MC subsets')
  for name, block_list in blocks.items():
    subset = subsets[name]
    indices = [32 * block + i for block in block_list for i in range(32)]
    require(subset['blocks'] == block_list and subset['paths'] == len(indices) and
            subset['path_indices'] == indices and set(subset['interfaces']) == {'q', 'p'},
            'B fixed path ordering/coverage changed')
    for key in ('cumulative_mean', 'centered_cumulative_mean', 'centered_path_sample_variance',
                'centered_variance_over_n'): shape(subset[key], [8, 4, 4, 2, 7])
    for branch in subset['interfaces'].values():
      shape(branch['choices'], [8, 4, 4], lambda x: type(x) is int and 0 <= x < 7)
      for key in ('bc_mse', 'plugin_centered_mse', 'mc_correction', 'plugin_level_mse',
                  'selected_reward', 'finite_library_regret'):
        metric = branch[key]
        shape(metric['per_initial'], [8, 4, 4]); shape(metric['state_values'], [8, 4])
        shape(metric['episode_values'], [8]); shape(metric['primary'], [])
    for group, keys in (('mse_gaps', {'p_minus_q', 'constant_minus_q'}),
                        ('utility_gaps', {'q_minus_p', 'q_minus_constant'})):
      require(set(subset[group]) == keys, 'Partial B necessary comparisons')
      for metric in subset[group].values():
        shape(metric['per_initial'], [8, 4, 4]); shape(metric['state_values'], [8, 4])
        shape(metric['episode_values'], [8]); shape(metric['primary'], [])
        require(metric['lower95'] <= metric['upper95'], 'Reversed B episode interval')
  for key, group, limit, family in (('mse_reliability', 'mse_gaps', .0004, 'mean'),
                                   ('utility_reliability', 'utility_gaps', .002, 'choice')):
    reliable = value[key]
    require(reliable['epsilon_limit'] == limit and
            set(reliable['metrics']) == set(subsets['pooled128'][group]), 'Changed B reliability object')
    positives, negatives, stable = [], [], []
    for name, metric in reliable['metrics'].items():
      primary = subsets['pooled128'][group][name]
      point, lower, upper = primary['primary'], primary['lower95'], primary['upper95']
      epsilon, components = metric['epsilon'], metric['components']
      require(set(components) == {'nested32_64', 'nested64_128', 'independent_halves', 'block_t_margin'} and
              all(x >= 0 for x in components.values()) and epsilon == max(components.values()),
              'B epsilon components/max changed')
      require(metric['point'] == point and metric['lower95'] == lower and metric['upper95'] == upper and
              metric['lower95_minus_epsilon'] == lower - epsilon and
              metric['upper95_plus_epsilon'] == upper + epsilon and
              metric['relative_positive_margin_limit'] == .2, 'B necessary metric summary changed')
      positive = point > 0 and epsilon <= .2 * point and lower - epsilon > 0
      negative, is_stable = upper + epsilon <= 0, epsilon <= limit
      require(metric['positive_margin_robust'] is positive and metric['negative_margin_robust'] is negative and
              metric['stable'] is is_stable, 'B necessary margin predicate changed')
      positives.append(positive); negatives.append(negative); stable.append(is_stable)
    status = ('MC-inconclusive' if not all(stable) else
              'finite_native_' + family + '_not_supported' if any(negatives) else
              'finite_native_' + family + '_exploratory_supported' if all(positives) else
              'finite_native_' + family + '_inconclusive')
    require(reliable['status'] == status and reliable['mc_stable'] is all(stable) and
            reliable['necessary_directions_robust'] is (all(stable) and all(positives)),
            'B finite status changed')
  require(value['primary_status'] == value['mse_reliability']['status'], 'B primary status changed')


def write_json(path, value):
  with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(value, stream, indent=2, allow_nan=False)
    stream.write('\n')


def _owned_output(command):
  """A bounded metadata helper owns its own session, including descendants."""
  child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           start_new_session=os.name != 'nt')
  _OWNED_HELPER_GROUPS.add(child.pid)
  try:
    stdout, stderr = child.communicate(timeout=30)
    if child.returncode:
      raise subprocess.CalledProcessError(child.returncode, command, stdout, stderr)
    return stdout
  finally:
    if os.name == 'nt':
      if child.poll() is None: child.kill()
    else:
      try: os.killpg(child.pid, signal.SIGKILL)
      except ProcessLookupError: pass
    _OWNED_HELPER_GROUPS.discard(child.pid)
    child.wait(timeout=2)


def git_binding(expected, sources):
  require(re.fullmatch('[0-9a-f]{40}', expected) is not None, 'Require full execution Git SHA')
  require(_owned_output(['git', 'rev-parse', 'HEAD']).decode().strip() == expected,
          'Execution Git SHA mismatch')
  require(not _owned_output(['git', 'status', '--porcelain']),
          'Require clean execution checkout')
  for name, digest in sources.items():
    path = (ROOT / name).resolve(strict=True)
    require(path.is_relative_to(ROOT) and Path(name).as_posix() == name and not Path(name).is_absolute(),
            'Unsafe execution source path')
    blob = _owned_output(['git', 'show', f'{expected}:{name}'])
    require(hashlib.sha256(blob).hexdigest() == digest and path.read_bytes() == blob,
            'Execution source differs from Git: ' + name)


def verify_completion(output, expected_git):
  """B-specific stdlib metadata/hash audit; never imports a numerical worker."""
  output = Path(output)
  budget, plan, result, complete = [read_json(output / name) for name in (
      'budget.json', 'plan.json', 'result.json', 'production_reward_complete.json')]
  for value in (budget, plan, result, complete):
    finite_tree(value)
    require(value['version'] == VERSION and value['actual_git_sha'] == expected_git,
            'Wrong B artifact version/Git')
  require(budget['seconds'] == plan['seconds'] == SECONDS and
          budget['planned_numeric_episode_reads'] == 8 and
          budget['actual_numeric_episode_reads'] is None and budget['planned_physical_gpu'] == 4 and
          budget['planned_gpu_devices'] == 1 and 'real_dataset_reads' not in budget and
          'source_provenance_verified' not in budget, 'Wrong B planning budget')
  require(plan['episodes'] == list(EPISODES) and plan['positions'] == list(POSITIONS) and
          plan['hostname'] == 'lyg0326' and plan['platform'].startswith('linux'),
          'Wrong fixed B execution object/host')
  for key, expected in dict(horizon=20, initial_samples=4, branches=['q', 'p'], controls=7,
      future_paths=128, blocks=4, paths_per_block=32, sample_chunk=8,
      latent_reward_steps=4587520, private_seed=20261024, statistical_seed=20261025,
      oracle_atol=1e-6, oracle_rtol=1e-5, stats_audit_atol=1e-10).items():
    require(type(plan[key]) is type(expected) and plan[key] == expected,
            'B fixed plan changed: ' + key)
  for metadata in (plan['model_metadata'], plan['source_model_provenance']):
    require(all(metadata[k] == v for k, v in MODEL_PINS.items()) and
            metadata['checkpoint_counters'] == dict(updates=74493, batches=74493, actions=18769,
                eval_actions=1001, eval_seed=3404107006), 'B model pins/counters changed')
  for value in (plan, result, complete):
    for key, expected in NO_AUTHORITY.items():
      require(value[key] is expected, 'B scientific authority changed: ' + key)
    require(value['source_provenance_verified'] is True and value['real_dataset_reads'] == 8 and
            value['records_decoded'] is False and value['gpu_devices_used'] == 1 and
            value['exploratory_calibration_only'] is True and value['actor_calls'] == 0 and
            value['initial_latents_resampled'] is False and
            value['physics_steps'] == value['render_calls'] == value['optimizer_updates'] == 0,
            'Incomplete B actual source/GPU scope')
  require(result['source_inputs_unchanged'] is True and result['state_unchanged'] is True and
          result['physics_steps'] == result['render_calls'] == result['optimizer_updates'] == 0,
          'B source/state/no-environment boundary changed')
  require(result['checkpoint_state_unchanged'] is True and result['params_unchanged'] is True and
          result['counters_unchanged'] is True and result['stopped_after_fixed_object'] is True and
          result['checkpoint_state_sha256'] == plan['checkpoint_state_sha256'] and
          re.fullmatch('[0-9a-f]{64}', plan['checkpoint_state_sha256']) is not None and
          plan['params_unchanged'] is True and plan['gate_changed'] is False and
          plan['physics_provenance_reused'] is True,
          'B full checkpoint state changed')
  resource = plan['resource']
  runtime = resource['runtime_compute']
  require(resource['physical_gpu'] == runtime['physical_gpu'] == 4 and
          resource['logical_server'] == 'sv3' and resource['exclusive'] is True and
          resource['gpu_uuid'] == resource['compute_uuid'] == resource['egl_uuid'] ==
              resource['cuda_visible_devices'] and
          runtime['compute_uuid'] == [resource['gpu_uuid']] and runtime['graphics_verified'] is False,
          'B actual exclusive CUDA identity changed')
  require(sha(resource['resource_source_path']) == resource['resource_source_sha256'],
          'B initial resource record changed')
  dataset = Path(plan['source_dataset']).resolve(strict=True)
  require(plan['source_git_sha'] == SOURCE_GIT and plan['source_json_sha256'] == SOURCE_PINS,
          'B source identity changed')
  for name, digest in SOURCE_PINS.items():
    require(sha(dataset / name) == digest, 'B source JSON changed: ' + name)
  inputs = plan['input_artifacts']
  require(len(inputs) == 8 and [item['episode'] for item in inputs] == list(EPISODES),
          'Partial/reordered B source inputs')
  source_shapes = dict(episode=([], 'int32'), positions=([4], 'int32'),
      h=([4, 4096], 'float64'), latent=([4, 4, 2, 32, 32], 'float64'),
      reward_f64=([4, 15, 20], 'float64'), first_commands=([4, 15, 12], 'float32'),
      h_native=([4, 4096], 'bfloat16'), latent_native=([4, 4, 2, 32, 32], 'bfloat16'),
      controls=([4, 7, 12], 'float32'), true_cost=([4, 7], 'float64'))
  for item in inputs:
    require(item['path'] == f"episode_{item['episode']:03}.npz" and
            (dataset / item['path']).resolve(strict=True).parent == dataset and
            sha(dataset / item['path']) == item['sha256'], 'B input archive changed/path unsafe')
    require(set(item['typed_hashes']) == set(source_shapes), 'Partial B source typed receipts')
    for name, (dimensions, dtype) in source_shapes.items():
      proof = item['typed_hashes'][name]
      require(proof['shape'] == dimensions and proof['dtype_name'] == dtype and
              type(proof['sha256']) is str and re.fullmatch('[0-9a-f]{64}', proof['sha256']) is not None,
              'Changed B source axes/dtype/proof: ' + name)
  sources = plan['source_files_sha256']
  require(complete['source_files_sha256'] == result['source_files_sha256'] == sources and
          set(sources) == set(SOURCE_FILES),
          'Incomplete B execution source chain')
  git_binding(expected_git, sources)
  require(budget['supervisor_sha256'] == sources['scripts/dl_production_reward_supervisor.py'],
          'B parent source differs from worker binding')
  budget_hash, plan_hash, result_hash = [sha(output / name) for name in
                                       ('budget.json', 'plan.json', 'result.json')]
  require(plan['budget_sha256'] == result['budget_sha256'] == complete['budget_sha256'] == budget_hash and
          result['plan_sha256'] == complete['plan_sha256'] == plan_hash and
          complete['result_sha256'] == result_hash, 'B budget/plan/result hash mismatch')
  for key, expected in dict(complete_episodes=8, complete_states=32, initial_samples=4,
      branches=2, controls=7, future_paths=128, horizon=20, complete_chunks=512,
      latent_reward_steps=4587520).items():
    require(type(complete[key]) is int and complete[key] == expected, 'Partial B coverage: ' + key)
  artifacts = result['evidence_artifacts']
  require(type(artifacts) is list and len(artifacts) == 8 and
          [x['episode'] for x in artifacts] == list(EPISODES), 'Missing B prediction artifacts')
  names = []
  for item in artifacts:
    path = (output / item['path']).resolve(strict=True)
    require(item['path'] == f"episode_{item['episode']:03}.npz" and
            path.is_relative_to(output.resolve()) and not Path(item['path']).is_absolute() and
            item['path'] not in names and sha(path) == item['sha256'],
            'Unsafe/changed/duplicate B prediction artifact')
    names.append(item['path'])
    for key in ('reward_typed_sha256', 'root_keys_typed_sha256', 'sample_keys_typed_sha256'):
      require(re.fullmatch('[0-9a-f]{64}', item[key]) is not None, 'Missing B prediction typed proof')
  oracle = result['first_chunk_oracle']
  require(oracle['path'] == 'first_chunk_oracle.npz' and
          sha(output / oracle['path']) == oracle['sha256'] and
          oracle['atol'] == 1e-6 and oracle['rtol'] == 1e-5 and
          oracle['finite_point_oracle_passed'] is True and
          oracle['full_model_independently_reproduced'] is False and
          type(oracle['compared_points']) is int and oracle['compared_points'] == 8960 and
          type(oracle['failed_points']) is int and oracle['failed_points'] == 0 and
          read_json(output / 'first_chunk_oracle_check.json') == oracle,
          'B first-chunk oracle changed')
  verify_analysis(result['analysis'])
  require(complete['primary_status'] == result['analysis']['primary_status'], 'B statistics status mismatch')
  require(result['elapsed_parent_entry_seconds'] <= SECONDS, 'B worker exceeded whole-entry deadline')
  return dict(verification_kind='production_reward_artifact_chain_schema_and_hashes',
      worker_boundary=dict(NO_AUTHORITY, source_provenance_verified=True, real_dataset_reads=8,
          actual_numeric_episode_reads=8, actual_gpu_devices_used=1, records_decoded=False,
          gpu_devices_used=1, physics_steps=0, render_calls=0, optimizer_updates=0,
          actor_calls=0, initial_latents_resampled=False, exploratory_calibration_only=True),
      worker_pid=runtime['pid'],
      budget_sha256=budget_hash, plan_sha256=plan_hash, result_sha256=result_hash,
      first_chunk_oracle_check_sha256=sha(output / 'first_chunk_oracle_check.json'),
      complete_sha256=sha(output / 'production_reward_complete.json'))


def real_worker_command(command, output, expected_git):
  require(type(command) in (list, tuple) and len(command) == 15 and
          Path(command[0]).resolve() == Path(sys.executable).resolve() and
          Path(command[1]).resolve() == ROOT / 'scripts/dl_production_reward.py' and
          list(command[2::2]) == ['--dataset', '--checkpoint', '--config', '--resource-record',
                            '--output', '--expected-git', '--worker'] and
          all(Path(command[i]).is_absolute() for i in (3, 5, 7, 9, 11)) and
          Path(command[11]).resolve() == output and command[13] == expected_git,
          'Real B provenance requires fixed production worker command')


def supervise(command, output, expected_git, seconds=SECONDS, verify=None, started=None):
  """Generic engineering calls retain unknown actual reads/GPU provenance."""
  started = time.monotonic() if started is None else started
  require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0 and
          math.isfinite(started) and started <= time.monotonic(), 'Invalid B parent budget')
  output, child = Path(output), None
  posix_alarm = os.name == 'posix'
  timed_out, verified, cleanup_requested = False, False, False
  failure, verification, code, old_handler = None, None, 1, None
  def expire(signum, frame):
    raise TimeoutError('Whole-entry B budget expired')
  def remaining():
    value = started + seconds - time.monotonic()
    if value <= 0: raise TimeoutError('Whole-entry B budget expired')
    return value
  if posix_alarm:
    inherited = (_CLI_ENTRY_ALARM and started == ENTRY_STARTED and seconds == SECONDS and
                 signal.getsignal(signal.SIGALRM) is _entry_expire)
    require(inherited or signal.getitimer(signal.ITIMER_REAL) == (0., 0.),
            'Cannot replace unrelated process alarm')
    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, remaining())
  try:
    output = output.resolve()
    if verify is verify_completion: real_worker_command(command, output, expected_git)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'budget.json', dict(version=VERSION, implementation=IMPLEMENTATION,
        actual_git_sha=expected_git, seconds=seconds, parent_pid=os.getpid(),
        parent_entry_monotonic=started, supervisor_sha256=sha(__file__),
        planned_numeric_episode_reads=8, actual_numeric_episode_reads=None,
        planned_gpu_devices=1, planned_physical_gpu=4))
    with (output / 'production_reward.log').open('x', encoding='utf-8') as log:
      try:
        remaining()
        env = dict(os.environ, DL_PRODUCTION_REWARD_PARENT=str(os.getpid()))
        for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'): env[key] = '1'
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=os.name != 'nt', env=env)
        code = child.wait(timeout=remaining())
        if code == 0 and verify is not None:
          verification = verify(output, expected_git)
          require(verification is not False, 'B completion verification rejected')
          if verify is verify_completion:
            require(verification['worker_pid'] == child.pid, 'B CUDA receipt belongs to another worker')
          remaining()
          verified = True
      except (subprocess.TimeoutExpired, TimeoutError) as error:
        timed_out, verified, code = True, False, 124
        failure = dict(type=type(error).__name__, message=str(error))
      except Exception as error:
        verified, code = False, 1
        failure = dict(type=type(error).__name__, message=str(error))
      finally:
        if posix_alarm: signal.setitimer(signal.ITIMER_REAL, 0)
        if child is not None:
          cleanup_requested = os.name != 'nt'
          if os.name == 'nt':
            if child.poll() is None: child.kill()
          else:
            try: os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError: pass
          try: child.wait(timeout=2)
          except subprocess.TimeoutExpired:
            verified, code = False, 1
            failure = dict(type='CleanupFailure', message='Owned B worker did not terminate')
  finally:
    if posix_alarm:
      signal.setitimer(signal.ITIMER_REAL, 0)
      signal.signal(signal.SIGALRM, old_handler)
  elapsed = time.monotonic() - started
  if elapsed > seconds: timed_out, verified, code = True, False, 124
  if posix_alarm and not timed_out:
    signal.signal(signal.SIGALRM, _entry_expire)
    signal.setitimer(signal.ITIMER_REAL, max(1e-6, started + seconds - time.monotonic()))
  real_chain = verified and verify is verify_completion
  actual = verification['worker_boundary'] if real_chain else dict(NO_AUTHORITY, **UNKNOWN_ACTUAL)
  receipt = dict(version=VERSION, implementation=IMPLEMENTATION, actual_git_sha=expected_git,
      **actual, planned_numeric_episode_reads=8, planned_gpu_devices=1, planned_physical_gpu=4,
      verified_worker_boundary=verification['worker_boundary'] if real_chain else None,
      exit_code=code, timed_out=timed_out, worker_pid=child.pid if child else None,
      worker_terminal=child is not None and child.poll() is not None,
      owned_group_cleanup_requested=cleanup_requested, complete_verified=verified,
      real_artifact_chain_verified=real_chain, verification=verification, failure=failure,
      budget_sha256=sha(output / 'budget.json'), elapsed_seconds=elapsed,
      elapsed_parent_entry_seconds=elapsed, hostname=socket.gethostname(), platform=sys.platform,
      generic_fixture_without_scientific_verification=verify is not verify_completion,
      windows_cleanup_scope='known direct child only' if os.name == 'nt' else None)
  completion = output / 'supervisor_complete.json'
  try:
    if not timed_out:
      if posix_alarm:
        signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, remaining())
      remaining()
    write_json(completion, receipt)
    if not timed_out: remaining()
  except TimeoutError as error:
    if posix_alarm: signal.setitimer(signal.ITIMER_REAL, 0)
    if completion.exists(): completion.rename(output / 'supervisor_complete.incomplete.json')
    receipt.update(**UNKNOWN_ACTUAL, verified_worker_boundary=None, real_artifact_chain_verified=False,
        exit_code=124, timed_out=True, complete_verified=False,
        failure=dict(type=type(error).__name__, message=str(error)),
        elapsed_seconds=time.monotonic() - started,
        elapsed_parent_entry_seconds=time.monotonic() - started)
    write_json(completion, receipt)
  finally:
    if posix_alarm:
      signal.setitimer(signal.ITIMER_REAL, 0)
      signal.signal(signal.SIGALRM, old_handler)
  receipt['terminal_elapsed_seconds'] = time.monotonic() - started
  return receipt


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('dataset', 'checkpoint', 'config', 'resource-record', 'output'):
    parser.add_argument('--' + name, type=Path, required=True)
  parser.add_argument('--expected-git', required=True)
  args = parser.parse_args()
  require(os.name == 'posix' and socket.gethostname() == 'lyg0326' and sys.platform.startswith('linux'),
          'Require fixed sv3 Linux production B runtime')
  # Check execution source before any model process is launched; the worker
  # binds the complete dependency set again before/after predictions.
  git_binding(args.expected_git, {'scripts/dl_production_reward_supervisor.py': sha(__file__)})
  command = [sys.executable, str(ROOT / 'scripts/dl_production_reward.py')]
  for name in ('dataset', 'checkpoint', 'config', 'resource_record', 'output'):
    path = getattr(args, name).resolve(strict=name != 'output')
    command.extend(['--' + name.replace('_', '-'), str(path)])
  command.extend(['--expected-git', args.expected_git, '--worker'])
  return supervise(command, args.output, args.expected_git, verify=verify_completion,
                   started=ENTRY_STARTED)['exit_code']


if __name__ == '__main__':
  try: raise SystemExit(main())
  except TimeoutError: os._exit(124)
