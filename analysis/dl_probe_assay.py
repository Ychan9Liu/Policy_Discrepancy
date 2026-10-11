"""DL-probe-assay-r1: finite synthetic detection-range evidence, CPU only.

No real artifact/dataset input is accepted. The unchanged, byte-pinned r1
readout is evaluated on four predetermined artificial objects. No actor,
simulator, Dreamer, gate, training, or real scientific decision is evaluated.
"""
import time
ENTRY_STARTED = time.monotonic()
import argparse
import copy
import functools
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
  os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MUJOCO_GL'] = 'disable'
os.environ['JAX_PLATFORMS'] = 'cpu'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_action_information_stats as stats

VERSION = 'DL-probe-assay-r1'
IMPLEMENTATION = 'DL-probe-assay-engineering-r1'
SEED = 20261022
SECONDS = 120
CASES = ('sparse', 'replicated', 'history', 'identity')
STATS_SHA256 = 'c1d4cfa85707616ffaf0ca35581cf2d90b1ce6b9d48dc1f56bded2a3e0613b35'
SOURCE_FILES = ('analysis/dl_probe_assay.py',
                'scripts/dl_action_information_stats.py',
                'scripts/dl_probe_assay_supervisor.py',
                'tests/test_dl_probe_assay.py',
                'tests/test_dl_probe_assay_supervisor.py',
                'docs/DL_PROBE_ASSAY.md')
BOUNDARY = dict(
    synthetic_only=True, source_provenance_verified=False,
    actor_applicable=False, source_provenance_applicable=False, method_evidence=False,
    proceed_short_training=False, decision='stop_after_fixed_synthetic_assay',
    real_scientific_decision='not_evaluated', real_gate_authority=False,
    eligible_method_training=False, training_authorized=False,
    method_effectiveness_established=False, real_dataset_reads=0,
    blind_arrays_read=False, model_calls=0, physics_steps=0,
    optimizer_updates=0, gpu_devices_used=0,
    current_rgb_increment_over_complete_history_established=False,
    continuing_kl_compression_effect_established=False)


def require(condition, message):
  if not condition:
    raise ValueError(message)


def sha(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
  a = np.ascontiguousarray(value)
  require(not a.dtype.hasobject, 'Object data is forbidden')
  return dict(dtype=a.dtype.str, shape=list(a.shape),
              sha256=hashlib.sha256(a.tobytes()).hexdigest())


def write_json(path, value):
  with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
    stream.write('\n')


@functools.lru_cache(maxsize=1)
def _canonical_inputs():
  # All cases share these base draws; case iteration never consumes RNG.
  children = np.random.SeedSequence(SEED).spawn(2)
  h_rng, z_rng = [np.random.Generator(np.random.PCG64(x)) for x in children]
  initial_states = [copy.deepcopy(x.bit_generator.state) for x in (h_rng, z_rng)]
  base_h = h_rng.standard_normal((8, 4, 4096)).astype(np.float64)
  codes = z_rng.integers(0, 32, (8, 4, 4, 32), dtype=np.int32)
  base_z = np.eye(32, dtype=np.float64)[codes]
  classes = (np.arange(4)[None, :] == (np.arange(8) % 4)[:, None]).astype(np.uint8)
  rewards = np.zeros((8, 4, 7), np.float64)
  rewards[..., 2] = 1 - classes
  rewards[..., 3] = classes
  h_signal = np.broadcast_to((2 * classes.astype(np.float64) - 1)[..., None],
                            (8, 4, 4096)).copy()
  signal_z = np.eye(32, dtype=np.float64)[classes]
  result = {}
  for case in CASES:
    h = h_signal.copy() if case == 'history' else base_h.copy()
    latent = np.stack((base_z, base_z), axis=3)
    if case == 'sparse':
      latent[:, :, :, 0, 0, :] = signal_z[:, :, None, :]
    elif case == 'replicated':
      latent[:, :, :, 0, :, :] = signal_z[:, :, None, None, :]
    result[case] = dict(classes=classes.copy(), rewards=rewards.copy(),
        h=h, latent=latent, episodes=np.arange(8, dtype=np.int32))
  rng_receipt = dict(root_entropy=SEED, children=[dict(
      stream=name, entropy=child.entropy, spawn_key=list(child.spawn_key),
      bit_generator='PCG64', initial_state=initial_states[i],
      final_state=copy.deepcopy(rng.bit_generator.state))
      for i, (name, child, rng) in enumerate(zip(('h', 'base_z'), children, (h_rng, z_rng)))],
      base_h=digest(base_h), base_codes=digest(codes), base_onehot=digest(base_z))
  for arrays in result.values():
    for value in arrays.values():
      value.setflags(write=False)
  return result, rng_receipt


def generate_inputs():
  # Copy a single cached fixed realization; validation never advances a stream.
  canonical, _ = _canonical_inputs()
  return {case: {name: value.copy() for name, value in arrays.items()}
          for case, arrays in canonical.items()}


def feature_decoder(case, arrays, branch):
  require(case in CASES and branch in (0, 1), 'Unknown case/interface')
  if case in ('sparse', 'replicated') and branch == 0:
    # Feature-only teacher: no classes, episode IDs, or reward arguments.
    q = arrays['latent'][:, :, :, 0, 0, :]
    label = q[..., 1]
    require(np.all((label == 0) | (label == 1)), 'Cue teacher has nonbinary input')
    return (2 + label).astype(np.int32)
  if case == 'history':
    label = (arrays['h'][..., 0] > 0).astype(np.int32)
    return np.repeat((2 + label)[..., None], 4, axis=2)
  return None


def validate_input(case, arrays):
  require(case in CASES and set(arrays) == {'classes', 'rewards', 'h', 'latent', 'episodes'},
          'Unknown synthetic schema')
  specs = dict(classes=(np.dtype('uint8'), (8, 4)),
               rewards=(np.dtype('float64'), (8, 4, 7)),
               h=(np.dtype('float64'), (8, 4, 4096)),
               latent=(np.dtype('float64'), (8, 4, 4, 2, 32, 32)),
               episodes=(np.dtype('int32'), (8,)))
  for name, (dtype, shape) in specs.items():
    a = arrays[name]
    require(isinstance(a, np.ndarray) and a.dtype == dtype and a.shape == shape,
            f'{name}: wrong synthetic dtype/shape')
    require(np.isfinite(a).all(), f'{name}: nonfinite synthetic input')
  z = arrays['latent']
  require(np.all((z == 0) | (z == 1)) and np.all(z.sum(-1) == 1),
          'Each synthetic factor must be exactly onehot')
  expected = _canonical_inputs()[0][case]
  for name in specs:
    require(np.array_equal(arrays[name], expected[name]),
            f'{case}/{name}: changed fixed construction')
  teacher = {}
  for branch, label in enumerate(('q', 'p')):
    choice = feature_decoder(case, arrays, branch)
    if choice is None:
      teacher[label] = dict(available=False,
          unavailable_is_not_proof_of_no_information=True)
      continue
    utilities = np.broadcast_to(arrays['rewards'][:, :, None, :], (8, 4, 4, 7))
    selected = np.take_along_axis(utilities, choice[..., None], -1)[..., 0]
    require(np.all(selected == 1), 'Feature-only teacher does not attain known oracle')
    teacher[label] = dict(available=True, choices=choice.tolist(),
        selected_reward=selected.tolist(), mean_selected_reward=1.,
        input='q factor0 only' if case != 'history' else 'h coordinate0 sign only',
        episode_ids_or_classes_or_rewards_given_to_decoder=False)
  return dict(input_arrays={k: digest(v) for k, v in arrays.items()},
      feature_only_teacher=teacher, construction_verified=True,
      privileged_oracle_reward=1., constant_mean_reward=.75,
      oracle_minus_constant=.25, classes=arrays['classes'].tolist(),
      classes_are_balanced_constructed_labels_not_random_events=True)


def ensure_synthetic_boundary(result):
  for key, expected in BOUNDARY.items():
    require(key in result and type(result[key]) is type(expected) and result[key] == expected,
            f'Synthetic authority boundary changed: {key}')


def analyze_case(case, arrays):
  proof = validate_input(case, arrays)
  before = {k: digest(v) for k, v in arrays.items()}
  external = stats._external(arrays['rewards'], arrays['episodes'])
  # Original POSITIONS are only algorithm display labels; expose actual toy indices.
  position_index = {p: i for i, p in enumerate(stats.POSITIONS)}
  for pair in external['cross_episode_disjoint_pairs']:
    pair['state_index_a'] = position_index[pair.pop('position_a')]
    pair['state_index_b'] = position_index[pair.pop('position_b')]
  external['interpretation'] = 'Known synthetic utility library; no real control validity'
  readout, qp, qc = stats._readouts(arrays['rewards'], arrays['h'],
                                  arrays['latent'], arrays['episodes'])
  readout['features'] = 'synthetic F64 h and exact onehot z; not actual Dreamer samples'
  readout['prediction_units'] = 'synthetic unit reward; no physical rollout or horizon'
  readout.pop('independent_episode_units')
  readout['constructed_group_units'] = 8
  readout['episode_units_are_constructed_groups_not_natural_samples'] = True
  draws = np.random.default_rng(stats.STAT_SEED).integers(
      0, 8, (stats.BOOTSTRAP_DRAWS, 8))
  comparisons = {name: stats._interval(values, draws)
                 for name, values in (('q_minus_p', qp), ('q_minus_constant', qc))}
  both_positive = all(x['lower95'] > 0 for x in comparisons.values())
  any_nonpositive = any(x['upper95'] <= 0 for x in comparisons.values())
  old_status = ('interface_increment_exploratory_supported'
      if both_positive and external['sufficient'] else
      'interface_increment_exploratory_not_supported' if any_nonpositive else
      'interface_increment_exploratory_inconclusive')
  require(before == {k: digest(v) for k, v in arrays.items()}, 'Readout modified inputs')
  known_q = case in ('sparse', 'replicated') and proof['feature_only_teacher']['q']['available']
  missed = known_q and external['sufficient'] and comparisons['q_minus_constant']['primary'] <= 0
  q_reward = readout['interfaces']['q']['primary_selected_reward']
  p_reward = readout['interfaces']['p']['primary_selected_reward']
  identity_exact = (readout['interfaces']['q'] == readout['interfaces']['p'])
  if case in ('history', 'identity'):
    require(identity_exact and np.all(qp == 0), 'Exact shared interface control failed')
  return dict(case=case, **BOUNDARY, proof=proof, external_synthetic_control=external,
      frozen_readout=readout, comparisons=comparisons,
      original_readout_status_on_synthetic_only=old_status,
      synthetic_selection_status={
          'interface_increment_exploratory_supported': 'synthetic_selection_detected',
          'interface_increment_exploratory_not_supported': 'synthetic_selection_not_detected',
          'interface_increment_exploratory_inconclusive': 'synthetic_selection_inconclusive'}[old_status],
      interval_scope='Descriptive constructed-group bootstrap; not inference about real data',
      feature_oracle_available_but_probe_did_not_beat_constant=bool(missed),
      known_q_specific_decoder=bool(known_q),
      shared_interface_outputs_exact=bool(identity_exact),
      expected_control_behavior_observed=(
          q_reward == 1 and comparisons['q_minus_constant']['lower95'] > 0
          if case == 'replicated' else
          q_reward == p_reward == 1 if case == 'history' else
          identity_exact if case == 'identity' else None),
      control_expectation_failure_is_not_permission_to_change_parameters=True,
      unchanged_input_arrays=before)


def git_sha():
  return subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
      check=True, capture_output=True, text=True).stdout.strip()


def source_proof(expected_git):
  require(git_sha() == expected_git, 'Unexpected execution Git SHA')
  dirty = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain'],
      check=True, capture_output=True, text=True).stdout
  require(not dirty, 'Require clean committed source')
  require(sha(ROOT / 'scripts/dl_action_information_stats.py') == STATS_SHA256,
          'Original frozen readout bytes changed')
  result = {}
  for name in SOURCE_FILES:
    actual = (ROOT / name).read_bytes()
    committed = subprocess.run(['git', '-C', str(ROOT), 'show', f'{expected_git}:{name}'],
        check=True, capture_output=True).stdout
    require(actual == committed, f'Checkout differs from committed source: {name}')
    result[name] = hashlib.sha256(actual).hexdigest()
  require(stats.RIDGE == 1 and stats.DELTA == .2 and stats.TIE == 1e-10 and
          stats.BOOTSTRAP_DRAWS == 2000 and stats.STAT_SEED == 20261021,
          'Original numerical contract changed')
  return result


def plan(expected_git, sources):
  return dict(version=VERSION, implementation=IMPLEMENTATION, **BOUNDARY,
      actual_git_sha=expected_git, source_sha256=sources, seed=SEED,
      actual_random_stream_receipt=copy.deepcopy(_canonical_inputs()[1]),
      numpy_version=np.__version__, python_version=sys.version,
      platform=sys.platform, hostname=socket.gethostname(),
      rng='NumPy1.26.4 SeedSequence(seed).spawn(2); child0 PCG64 h, child1 PCG64 z',
      h_shape=[8, 4, 4096], latent_shape=[8, 4, 4, 2, 32, 32],
      reward_shape=[8, 4, 7], cases=list(CASES), state_indices=list(range(4)),
      episode_group_ids=list(range(8)), classes='state_index == episode_group_id % 4',
      utility='maximize: a2=1-C, a3=C, other five=0',
      interfaces='q/p share base nuisance; only declared cue factors differ',
      teacher='q factor0 or shared h coordinate0 sign; no labels given to decoder',
      ridge_lambda=1., group_normalization='train-only total energy RMS for h and z',
      bootstrap_draws=2000, bootstrap_seed=20261021,
      seconds=SECONDS, threads=1, parameter_search_permitted=False,
      no_real_dataset_argument=True)


def run_worker(output, expected_git):
  output = Path(output)
  budget = json.loads((output / 'budget.json').read_text(encoding='utf-8'))
  require(os.name != 'nt' and socket.gethostname() == 'lyg0326',
          'Frozen complete execution requires sv3 Linux CPU')
  require(str(os.getppid()) == os.environ.get('DL_PROBE_ASSAY_PARENT') ==
          str(budget['parent_pid']), 'Worker requires its own bounded parent')
  ensure_synthetic_boundary(budget)
  require(budget['version'] == VERSION and budget['seconds'] == SECONDS and
          budget['actual_git_sha'] == expected_git and
          budget['supervisor_sha256'] == sha(ROOT / 'scripts/dl_probe_assay_supervisor.py'),
          'Changed parent budget/source binding')
  deadline = float(budget['parent_entry_monotonic']) + SECONDS
  def check_deadline():
    require(np.isfinite(deadline) and time.monotonic() <= deadline,
            'Complete assay exceeds fixed parent budget')
  check_deadline()
  sources = source_proof(expected_git)
  require(np.__version__ == '1.26.4', 'Require pinned NumPy1.26.4')
  write_json(output / 'plan.json', plan(expected_git, sources))
  inputs = generate_inputs()
  results = []
  for case in CASES:
    check_deadline()
    validate_input(case, inputs[case])
    artifact = output / f'{case}_inputs.npz'
    with artifact.open('xb') as stream:
      np.savez_compressed(stream, **inputs[case])
    result = analyze_case(case, inputs[case])
    check_deadline()
    ensure_synthetic_boundary(result)
    result['input_artifact'] = dict(path=artifact.name, sha256=sha(artifact))
    write_json(output / f'{case}_result.json', result)
    results.append(result)
    print(json.dumps(dict(stage='synthetic_case_complete', case=case,
        q_reward=result['frozen_readout']['interfaces']['q']['primary_selected_reward'],
        p_reward=result['frozen_readout']['interfaces']['p']['primary_selected_reward'],
        measurement_counterexample=result['feature_oracle_available_but_probe_did_not_beat_constant'])),
        flush=True)
  source_proof(expected_git)
  check_deadline()
  require(not any(x in sys.modules for x in ('jax', 'mujoco', 'dm_control', 'dreamerv3')),
          'Forbidden real model/simulator module imported')
  summary = dict(version=VERSION, implementation=IMPLEMENTATION, **BOUNDARY,
      actual_git_sha=expected_git, plan_sha256=sha(output / 'plan.json'),
      completed_cases=list(CASES), synthetic_construction_and_identity_valid=True,
      finite_detection_counterexamples=[r['case'] for r in results
          if r['feature_oracle_available_but_probe_did_not_beat_constant']],
      expected_control_behavior={r['case']: r['expected_control_behavior_observed'] for r in results},
      case_result_sha256={case: sha(output / f'{case}_result.json') for case in CASES},
      no_parameter_search_performed=True, no_prior_real_result_rejudged=True,
      elapsed_worker_entry_seconds=time.monotonic() - ENTRY_STARTED)
  ensure_synthetic_boundary(summary)
  write_json(output / 'summary.json', summary)
  check_deadline()
  write_json(output / 'assay_complete.json', dict(version=VERSION, **BOUNDARY,
      actual_git_sha=expected_git, plan_sha256=sha(output / 'plan.json'),
      summary_sha256=sha(output / 'summary.json'), complete_cases=4))
  check_deadline()
  print(json.dumps(dict(complete=True, **BOUNDARY)), flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--expected-git', required=True)
  args = parser.parse_args()
  run_worker(args.output, args.expected_git)
  return 0


if __name__ == '__main__':
  sys.exit(main())
