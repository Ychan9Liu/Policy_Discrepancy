"""DL-production-reward-r1: fixed production-model conditional mean assay.

The stdlib supervisor owns the whole-entry 1800s deadline. This worker only
reads authenticated calibration numeric fields and the trusted checkpoint.
No environment, renderer, optimizer, actor or new initial latent is invoked.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VERSION = 'DL-production-reward-r1'
IMPLEMENTATION = 'DL-production-reward-engineering-r1'
SECONDS = 1800
PRED_ATOL, PRED_RTOL = 1e-6, 1e-5
STAT_ATOL = 1e-10
SOURCE_FILES = (
    'scripts/dl_production_reward.py', 'scripts/dl_production_reward_source.py',
    'scripts/dl_production_reward_model.py', 'scripts/dl_production_reward_stats.py',
    'scripts/dl_production_reward_supervisor.py', 'docs/DL_PRODUCTION_REWARD.md',
    'scripts/dl_predictive_mc.py', 'scripts/dl_predictive.py',
    'scripts/dl_diagnostic.py', 'scripts/dl_resources.py',
    'dreamerv3/agent.py', 'dreamerv3/rssm.py', 'embodied/jax/agent.py',
    'embodied/jax/heads.py', 'embodied/jax/outs.py', 'embodied/jax/nets.py',
    'scripts/probe_formal_devices.py',
    'tests/test_dl_production_reward_source.py',
    'tests/test_dl_production_reward_model.py',
    'tests/test_dl_production_reward_stats.py',
    'tests/test_dl_production_reward_supervisor.py',
    'tests/test_dl_production_reward_worker.py',
    'analysis/dl_production_reward_audit.py',
    'tests/test_dl_production_reward_audit.py')
BOUNDARY = dict(exploratory_calibration_only=True, source_provenance_verified=True,
    real_dataset_reads=8, blind_arrays_read=False, physics_steps=0,
    render_calls=0, optimizer_updates=0, gpu_devices_used=1,
    records_decoded=False, method_evidence=False,
    initial_latents_resampled=False, actor_calls=0, training_authorized=False,
    proceed_short_training=False, eligible_method_training=False,
    real_gate_authority=False, method_effectiveness_established=False,
    current_rgb_increment_over_complete_history_established=False,
    continuing_kl_compression_effect_established=False)


def require(condition, message):
  if not condition:
    raise ValueError(message)


def sha(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for block in iter(lambda: stream.read(2 ** 20), b''):
      digest.update(block)
  return digest.hexdigest()


def write_json(path, value):
  with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write('\n')


def typed_sha(value):
  value = np.ascontiguousarray(value)
  return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() +
      value.tobytes()).hexdigest()


def tree_sha(value):
  """Host checkpoint leaves include all optimizer and normalization state."""
  digest = hashlib.sha256()
  def visit(node):
    if isinstance(node, dict):
      digest.update(b'dict')
      for key in sorted(node):
        digest.update(str(key).encode()); visit(node[key])
    elif isinstance(node, (tuple, list)):
      digest.update(type(node).__name__.encode() + str(len(node)).encode())
      for item in node:
        visit(item)
    else:
      array = np.ascontiguousarray(node)
      require(not array.dtype.hasobject and np.isfinite(array).all(),
              'Invalid checkpoint leaf')
      digest.update(str(array.dtype).encode() + str(array.shape).encode())
      digest.update(array.tobytes())
  visit(value)
  return digest.hexdigest()


def git_binding(expected):
  actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                   text=True).strip()
  require(actual == expected, 'Execution Git SHA differs')
  require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT),
          'Require a clean committed checkout')
  pins = {}
  for name in SOURCE_FILES:
    blob = subprocess.check_output(['git', 'show', f'{expected}:{name}'], cwd=ROOT)
    require((ROOT / name).read_bytes() == blob, 'Execution Git bytes differ: ' + name)
    pins[name] = hashlib.sha256(blob).hexdigest()
  return pins


def symmetric_pred_numpy(logits, bins):
  """Independent F64 physical-bin mean, preserving native symmetric pairing."""
  logits, bins = np.asarray(logits), np.asarray(bins)
  require(logits.dtype == np.float32 and bins.dtype == np.float32 and
          bins.shape == (255,) and logits.shape[-1] == 255 and
          np.isfinite(logits).all() and np.isfinite(bins).all(),
          'Invalid raw TwoHot oracle axes/dtypes')
  values = logits.astype(np.float64)
  probs = np.exp(values - values.max(-1, keepdims=True))
  probs /= probs.sum(-1, keepdims=True)
  products = probs * bins.astype(np.float64)
  return products[..., 127] + (
      products[..., :127][..., ::-1] + products[..., 128:]).sum(-1)


def validate_chunk(output, roots, reference_bins=None):
  require(set(output) == {'reward', 'logits', 'bins', 'root_keys', 'sample_keys'},
          'Unexpected native model output schema')
  reward, logits, bins = [output[name] for name in ('reward', 'logits', 'bins')]
  require(reward.dtype == np.float32 and reward.shape == (4, 2, 8, 7, 20) and
          logits.dtype == np.float32 and logits.shape == (*reward.shape, 255) and
          bins.dtype == np.float32 and bins.shape == (255,) and
          np.isfinite(reward).all() and np.isfinite(logits).all() and
          np.isfinite(bins).all() and np.all(np.diff(bins) > 0),
          'Invalid native reward/logit/bin contract')
  for name in ('root_keys', 'sample_keys'):
    require(output[name].dtype == np.uint32 and output[name].shape == (4, 8, 20, 2),
            'Invalid saved private RNG evidence: ' + name)
  require(output['root_keys'].tobytes() == roots.tobytes(), 'Private roots changed')
  if reference_bins is not None:
    require(bins.tobytes() == reference_bins.tobytes(), 'Reward bins changed')


def worker(args):
  from scripts import dl_production_reward_source as source
  from scripts import dl_production_reward_model as model
  from scripts import dl_production_reward_stats as stats
  output, dataset = args.output.resolve(strict=True), args.dataset.resolve(strict=True)
  budget = json.loads((output / 'budget.json').read_text(encoding='utf-8'))
  require(os.name == 'posix' and sys.platform.startswith('linux') and
          socket.gethostname() == 'lyg0326', 'Frozen execution is sv3 Linux only')
  require(budget['version'] == VERSION and budget['seconds'] == SECONDS and
          budget['actual_git_sha'] == args.expected_git and
          budget['parent_pid'] == os.getppid() ==
              int(os.environ.get('DL_PRODUCTION_REWARD_PARENT', '-1')) and
          budget['supervisor_sha256'] == sha(ROOT / SOURCE_FILES[4]),
          'Bounded live supervisor/source binding required')
  def check():
    if time.monotonic() >= budget['parent_entry_monotonic'] + SECONDS:
      raise TimeoutError('Fixed B object exceeded whole-entry deadline')
  check()
  files = git_binding(args.expected_git)
  data = source.load_dataset(dataset, check=check)
  require(np.__version__ == '1.26.4', 'Require pinned NumPy1.26.4')
  packages = {name: importlib.metadata.version(name)
              for name in data['source_plan']['packages']}
  require(packages == data['source_plan']['packages'], 'Source runtime packages differ')
  model_packages = {name: importlib.metadata.version(name)
                    for name in ('jax', 'jaxlib', 'ninjax')}
  require(model_packages == {'jax': '0.4.33', 'jaxlib': '0.4.33', 'ninjax': '3.6.3'},
          'Require pinned native model/private RNG runtime')
  packages.update(model_packages)
  from scripts.dl_resources import require_resource, verify_runtime
  resource = require_resource(args.resource_record)
  require(resource['physical_gpu'] == 4, 'B-r1 uses assigned physical GPU4 only')
  from scripts import dl_predictive_mc as pinned
  require(source.MODEL_PINS['checkpoint_sha256'] == pinned.CHECKPOINT_SHA and
          source.MODEL_PINS['config_source_sha256'] == pinned.CONFIG_SHA and
          source.MODEL_PINS['params_sha256'] == pinned.PARAMS_SHA,
          'No-environment loader model pins differ')
  agent, config, metadata = pinned.load_agent_no_env(args)
  require(metadata['params_sha256'] == pinned.PARAMS_SHA and
          metadata['checkpoint_counters'] == data['source_plan']['model_provenance']['checkpoint_counters'],
          'Loaded model checkpoint provenance differs')
  metadata['loader_code_version'] = metadata['code_version']
  metadata['loader_scope'] = 'Pinned model factory/checkpoint loader only; no legacy assay inputs'
  metadata['code_version'] = VERSION
  resource['runtime_compute'] = verify_runtime(resource, compute=True, graphics=False)
  before = tree_sha(agent.save())
  adapter = model.NativeRewardAdapter(agent)
  plan = dict(version=VERSION, implementation=IMPLEMENTATION, **BOUNDARY,
      actual_git_sha=args.expected_git, source_files_sha256=files,
      source_dataset=str(dataset), source_git_sha=source.SOURCE_GIT,
      source_json_sha256=source.SOURCE_PINS, input_artifacts=data['input_artifacts'],
      source_proof_scope='authenticated existing f1 physics/carry/key proof; no fresh physics',
      source_model_provenance=data['source_plan']['model_provenance'],
      model_metadata=metadata, checkpoint_state_sha256=before,
      budget_sha256=sha(output / 'budget.json'), resource=resource,
      python=sys.version, packages=packages,
      episodes=list(source.EPISODES), positions=list(source.POSITIONS),
      horizon=20, initial_samples=4, branches=['q', 'p'], controls=7,
      future_paths=128, blocks=4, paths_per_block=32, sample_chunk=8,
      latent_reward_steps=4587520, private_seed=20261024, statistical_seed=20261025,
      seconds=SECONDS, hostname=socket.gethostname(), platform=sys.platform,
      oracle_atol=PRED_ATOL, oracle_rtol=PRED_RTOL, stats_audit_atol=STAT_ATOL,
      params_unchanged=True, gate_changed=False, physics_provenance_reused=True)
  write_json(output / 'plan.json', plan)
  rows, artifacts, reference_bins, oracle = [], [], None, None
  for ei, episode in enumerate(source.EPISODES):
    rewards = np.empty((4, 4, 2, 128, 7, 20), np.float32)
    roots_all = np.empty((4, 4, 128, 20, 2), np.uint32)
    samples_all = np.empty_like(roots_all)
    for si, position in enumerate(source.POSITIONS):
      controls = np.zeros((7, 20, 12), np.float32)
      controls[:, 0] = data['controls'][ei, si]
      for block in range(4):
        for start in (0, 8, 16, 24):
          check()
          roots = model.private_keys(episode, position, block, start)
          prediction = adapter.call_chunk(data['h_native'][ei, si],
              data['latent_native'][ei, si], controls, roots)
          validate_chunk(prediction, roots, reference_bins)
          if reference_bins is None:
            reference_bins = prediction['bins'].copy()
          require(prediction['sample_keys'].tobytes() ==
                  model.sample_keys(roots).tobytes(), 'Actual private sample keys differ')
          if ei == si == block == start == 0:
            cpu = symmetric_pred_numpy(prediction['logits'], prediction['bins'])
            error = np.abs(prediction['reward'].astype(np.float64) - cpu)
            first = output / 'first_chunk_oracle.npz'
            np.savez_compressed(first, **prediction)
            passed = bool(np.all(error <= PRED_ATOL + PRED_RTOL * np.abs(cpu)))
            oracle = dict(path=first.name, sha256=sha(first),
                max_abs_error=float(error.max()), atol=PRED_ATOL, rtol=PRED_RTOL,
                finite_point_oracle_passed=passed, full_model_independently_reproduced=False,
                compared_points=int(error.size), failed_points=int(np.count_nonzero(
                    error > PRED_ATOL + PRED_RTOL * np.abs(cpu))))
            write_json(output / 'first_chunk_oracle_check.json', oracle)
            require(passed, 'First-chunk native symmetric point oracle failed')
          index = slice(block * 32 + start, block * 32 + start + 8)
          rewards[si, :, :, index] = prediction['reward']
          roots_all[si, :, index] = prediction['root_keys']
          samples_all[si, :, index] = prediction['sample_keys']
      print(json.dumps(dict(episode=episode, position=position, completed_future_paths=128)), flush=True)
    path = output / f'episode_{episode:03}.npz'
    require(not path.exists(), 'Episode output already exists')
    np.savez_compressed(path, reward=rewards, root_keys=roots_all,
        sample_keys=samples_all, bins=reference_bins,
        episode=np.asarray(episode, np.int32), positions=np.asarray(source.POSITIONS, np.int32),
        true_cost=data['true_cost'][ei], controls=data['controls'][ei])
    artifacts.append(dict(episode=episode, path=path.name, sha256=sha(path),
        reward_typed_sha256=typed_sha(rewards), root_keys_typed_sha256=typed_sha(roots_all),
        sample_keys_typed_sha256=typed_sha(samples_all)))
    rows.append(rewards)
  check()
  require(tree_sha(agent.save()) == before, 'Complete model/optimizer/norm/counters changed')
  require(agent.save()['counters'] == metadata['checkpoint_counters'], 'Counters changed')
  source.authority(dataset, check=check)
  require(git_binding(args.expected_git) == files, 'Execution source files changed')
  analysis = stats.report(np.stack(rows), data['true_cost'], np.asarray(source.EPISODES, np.int32))
  check()
  source.authority(dataset, check=check)
  require(git_binding(args.expected_git) == files, 'Execution source files changed during statistics')
  require(sha(pinned.checkpoint_path(args.checkpoint)) == pinned.CHECKPOINT_SHA and
          sha(args.config) == pinned.CONFIG_SHA, 'Checkpoint/config input bytes changed')
  require(adapter.accessed_keys and all(name.startswith(('dyn/', 'rew/'))
          for name in adapter.accessed_keys), 'Unexpected native model parameter access')
  final_runtime = verify_runtime(resource, compute=True, graphics=False)
  check()
  result = dict(version=VERSION, implementation=IMPLEMENTATION, **BOUNDARY,
      actual_git_sha=args.expected_git, plan_sha256=sha(output / 'plan.json'),
      budget_sha256=sha(output / 'budget.json'), analysis=analysis,
      evidence_artifacts=artifacts, first_chunk_oracle=oracle,
      checkpoint_state_unchanged=True, checkpoint_state_sha256=before,
      state_unchanged=True,
      native_model_accessed_keys=sorted(adapter.accessed_keys),
      checkpoint_and_config_inputs_unchanged=True, final_runtime_compute=final_runtime,
      params_unchanged=True, counters_unchanged=True, source_inputs_unchanged=True,
      source_files_sha256=files, stopped_after_fixed_object=True,
      elapsed_parent_entry_seconds=time.monotonic() - budget['parent_entry_monotonic'])
  write_json(output / 'result.json', result)
  check()
  write_json(output / 'production_reward_complete.json', dict(version=VERSION,
      implementation=IMPLEMENTATION, **BOUNDARY, actual_git_sha=args.expected_git,
      budget_sha256=sha(output / 'budget.json'), plan_sha256=sha(output / 'plan.json'),
      result_sha256=sha(output / 'result.json'), complete_episodes=8, complete_states=32,
      complete_chunks=512, latent_reward_steps=4587520, source_files_sha256=files,
      initial_samples=4, branches=2, controls=7, future_paths=128, horizon=20,
      primary_status=analysis['primary_status']))
  check()
  print(json.dumps(dict(complete=True, primary_status=analysis['primary_status'])), flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('dataset', 'checkpoint', 'config', 'resource-record', 'output'):
    parser.add_argument('--' + name, type=Path, required=True)
  parser.add_argument('--expected-git', required=True)
  parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
  args = parser.parse_args()
  require(args.worker, 'Use the bounded production-reward supervisor CLI')
  args.platform, args.task = 'cuda', 'dmc_quadruped_walk'
  worker(args)


if __name__ == '__main__':
  main()
