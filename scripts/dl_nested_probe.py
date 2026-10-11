"""Fixed, CPU-only exploratory nested probe on authenticated cal8 numeric fields."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

# Must precede NumPy import, including CLI imports of the statistics module.
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
  os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VERSION = 'DL-nested-probe-r1'
IMPLEMENTATION = 'DL-nested-probe-engineering-r1'
SOURCE_GIT = 'f1ae4d6513e81ed680992b64ba8ebade7d256d52'
EPISODES = (9, 14, 15, 18, 20, 24, 30, 31)
POSITIONS = (1, 300, 600, 900)
SOURCE_PINS = {
    'plan.json': '474bad6adf6558b7a47c40711546dc4cf2adc870684b8cff2766a802605f44d6',
    'result.json': '06df6efa5f9ce84b428921b59fa508df8badd1dd54e30335bf2d3a273e585fad',
    'action_information_complete.json': '28481a749476be0b3e8c05f28e719cb22e4cfffb638e56c04d8f86128026d37e',
    'budget.json': 'ddc8f43a3f4b57c11858be61f4c31315484c8c42b1e887f48391e71d077e2a68',
    'supervisor_complete.json': 'a920a0191429c7ffd05af5a897a30839da47d046d9126a1002a4997928dc1ea3',
    'independent-numeric-audit-v1.json': 'cbb99e43ad72a6960d86e5044455a18fecc38f068172defe60deb0d82f8678e5'}
SOURCES = ('scripts/dl_nested_probe.py', 'scripts/dl_nested_probe_stats.py',
           'scripts/dl_nested_probe_supervisor.py', 'docs/DL_NESTED_PROBE.md')
BOUNDARY = dict(synthetic_only=False, exploratory_calibration_only=True,
    source_provenance_verified=True, actor_applicable=False, method_evidence=False,
    proceed_short_training=False, decision='stop_after_fixed_nested_probe',
    real_gate_authority=False, eligible_method_training=False, training_authorized=False,
    method_effectiveness_established=False, real_dataset_reads=8, blind_arrays_read=False,
    model_calls=0, physics_steps=0, optimizer_updates=0, gpu_devices_used=0,
    current_rgb_increment_over_complete_history_established=False,
    continuing_kl_compression_effect_established=False)


def require(condition, message):
  if not condition:
    raise ValueError(message)


def sha(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
  return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_json(path, value):
  with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(value, stream, indent=2, allow_nan=False)
    stream.write('\n')


def git_binding(expected):
  actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
  require(actual == expected, 'Execution Git SHA mismatch')
  require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT),
          'Execution checkout must be clean')
  pins = {}
  for name in SOURCES:
    blob = subprocess.check_output(['git', 'show', f'{expected}:{name}'], cwd=ROOT)
    require((ROOT / name).read_bytes() == blob, 'Execution bytes differ from Git: ' + name)
    pins[name] = hashlib.sha256(blob).hexdigest()
  return pins


def load_numeric_episode(path, episode):
  """Lazy whitelist only: never decode the opaque records envelope."""
  with np.load(path, allow_pickle=False) as archive:
    names = ('episode', 'positions', 'h', 'latent', 'reward_f64')
    require(set(names).issubset(archive.files), 'Missing required numeric fields')
    data = {name: archive[name].copy() for name in names}
  require(data['episode'].dtype == np.int32 and data['episode'].shape == () and
          int(data['episode']) == episode, 'Episode identity/dtype/shape changed')
  require(data['positions'].dtype == np.int32 and data['positions'].shape == (4,) and
          np.array_equal(data['positions'], POSITIONS), 'Positions changed')
  for name, shape in (('h', (4, 4096)), ('latent', (4, 4, 2, 32, 32)),
                      ('reward_f64', (4, 15, 20))):
    value = data[name]
    require(value.dtype == np.float64 and value.shape == shape and np.isfinite(value).all(),
            'Invalid finite F64 numeric field: ' + name)
  z = data['latent']
  require(np.logical_or(z == 0., z == 1.).all() and np.all(z.sum(-1) == 1.),
          'Actual sampled latent must be exactly onehot')
  costs = np.sum(data['reward_f64'][:, :7], axis=-1, dtype=np.float64)
  require(np.isfinite(costs).all(), 'Cost sum overflow')
  return data['h'], z, costs


def authority(dataset):
  documents = {}
  for name, expected in SOURCE_PINS.items():
    require(sha(dataset / name) == expected, 'Published source identity differs: ' + name)
    documents[name] = json.loads((dataset / name).read_text(encoding='utf-8'))
  plan, result, complete, parent, audit = [documents[name] for name in
      ('plan.json', 'result.json', 'action_information_complete.json',
       'supervisor_complete.json', 'independent-numeric-audit-v1.json')]
  for value in (plan, result, complete, parent):
    require(value['actual_git_sha'] == SOURCE_GIT and value['version'] == 'DL-action-information-r1',
            'Wrong source Git/version')
  require(complete['plan_sha256'] == SOURCE_PINS['plan.json'] and
          complete['result_sha256'] == SOURCE_PINS['result.json'] and
          parent['budget_sha256'] == SOURCE_PINS['budget.json'] and
          parent['exit_code'] == 0 and parent['timed_out'] is False and
          parent['worker_terminal'] is True and complete['complete_episodes'] == 8 and
          complete['complete_states'] == 32, 'Source completion chain failed')
  require(plan['episodes'] == list(EPISODES) and plan['positions'] == list(POSITIONS) and
          plan['horizon'] == 20 and plan['delta'] == .2, 'Frozen source object changed')
  require(audit['status'] == 'passed' and audit['expected_git_sha'] == SOURCE_GIT and
          audit['source_h_sampled_z_and_actor_raw_byte_exact'] is True and
          audit['blind_arrays_read'] is False, 'Missing independent source feature proof')
  items = result['evidence_artifacts']
  require([x['episode'] for x in items] == list(EPISODES), 'Source episode order changed')
  for item in items:
    require(item['path'] == f"episode_{item['episode']:03}.npz", 'Unsafe/unexpected source path')
    require(sha(dataset / item['path']) == item['sha256'], 'Numeric input bytes changed')
  return plan, result, items


def worker(args):
  from scripts import dl_nested_probe_stats as stats
  output, dataset = args.output.resolve(strict=True), args.dataset.resolve(strict=True)
  budget = json.loads((output / 'budget.json').read_text(encoding='utf-8'))
  require(os.name == 'posix' and socket.gethostname() == 'lyg0326' and sys.platform.startswith('linux'),
          'Frozen execution requires sv3 Linux CPU')
  require(np.__version__ == '1.26.4', 'Require pinned NumPy1.26.4')
  require(budget['version'] == VERSION and budget['seconds'] == 120 and
          budget['actual_git_sha'] == args.expected_git and
          budget['parent_pid'] == os.getppid() == int(os.environ.get('DL_NESTED_PROBE_PARENT', '-1')) and
          budget['supervisor_sha256'] == sha(ROOT / 'scripts/dl_nested_probe_supervisor.py'),
          'Actual bounded parent/Git/source binding required')
  def check():
    if time.monotonic() >= budget['parent_entry_monotonic'] + 120:
      raise TimeoutError('Frozen nested object incomplete at deadline')
  check()
  source_files = git_binding(args.expected_git)
  source_plan, source_result, items = authority(dataset)
  hs, zs, costs, proofs = [], [], [], []
  for item in items:
    check()
    h, z, cost = load_numeric_episode(dataset / item['path'], item['episode'])
    hs.append(h); zs.append(z); costs.append(cost)
    proofs.append(dict(episode=item['episode'], path=item['path'], sha256=item['sha256'],
        h_sha256=array_sha(h), latent_sha256=array_sha(z), cost_sha256=array_sha(cost)))
  h, z, cost = np.stack(hs), np.stack(zs), np.stack(costs)
  old_cost = np.asarray(source_result['external_control']['rewards'], dtype=np.float64)
  require(old_cost.shape == cost.shape and old_cost.tobytes() == cost.tobytes(),
          'True cost bytes differ from original published result')
  require(tuple(stats.LAMBDAS) == (1., .1, .01, .001), 'Frozen lambda grid changed')
  plan = dict(version=VERSION, implementation=IMPLEMENTATION, **BOUNDARY,
      actual_git_sha=args.expected_git, source_files_sha256=source_files,
      budget_sha256=sha(output / 'budget.json'),
      source_dataset=str(dataset), source_git_sha=SOURCE_GIT, source_json_sha256=SOURCE_PINS,
      source_proof_scope='authenticated reuse; no fresh model/physics revalidation',
      input_artifacts=proofs, episodes=list(EPISODES), positions=list(POSITIONS),
      lambdas=list(stats.LAMBDAS), hostname=socket.gethostname(), platform=sys.platform,
      numpy_version=np.__version__, python=sys.version, threads=1, seconds=120,
      fixed_inner_lambda_selection_permitted=True,
      additional_parameter_search_permitted=False, outer_selection_permitted=False,
      current_rgb_independent_cue_claim_permitted=False,
      source_external_control_sufficient=source_result['external_control']['sufficient'],
      source_external_delta=.2, source_model_provenance=source_plan['model_provenance'])
  write_json(output / 'plan.json', plan)
  nested = stats.nested_readout(cost, h, z, np.asarray(EPISODES, np.int32))
  check()
  git_binding(args.expected_git)
  authority(dataset)
  require('jax' not in sys.modules and 'torch' not in sys.modules, 'GPU framework imported')
  result = dict(version=VERSION, implementation=IMPLEMENTATION, **BOUNDARY,
      actual_git_sha=args.expected_git, plan_sha256=sha(output / 'plan.json'), nested=nested,
      budget_sha256=sha(output / 'budget.json'),
      source_inputs_unchanged=True, cost_bytes_match_original=True,
      source_external_control_sufficient=source_result['external_control']['sufficient'],
      source_external_delta=.2, no_prior_real_result_rejudged=True,
      no_parameters_or_checkpoints_loaded=True,
      elapsed_parent_entry_seconds=time.monotonic() - budget['parent_entry_monotonic'])
  write_json(output / 'result.json', result)
  check()
  write_json(output / 'nested_probe_complete.json', dict(version=VERSION,
      implementation=IMPLEMENTATION, **BOUNDARY, actual_git_sha=args.expected_git,
      plan_sha256=sha(output / 'plan.json'), result_sha256=sha(output / 'result.json'),
      budget_sha256=sha(output / 'budget.json'), complete_outer_folds=8,
      source_files_sha256=source_files,
      complete_inner_folds=56, eigendecompositions=128, lambda_evaluations=464))
  check()
  print(json.dumps(dict(complete=True, decision=BOUNDARY['decision'])), flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--dataset', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--expected-git', required=True)
  args = parser.parse_args()
  worker(args)


if __name__ == '__main__':
  main()
