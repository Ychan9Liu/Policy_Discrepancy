"""DL-production-reward-r1 authenticated numeric inputs, without runtime effects.

No environment variables, CUDA/JAX runtime, model, simulator, renderer, pickle
or existing CPU worker is touched. Source state/physics/key truth is certified
reuse of the pinned f1 source and its audit, not fresh verification by B.
"""
import hashlib
import json
from pathlib import Path
import re

import ml_dtypes
import numpy as np

VERSION = 'DL-production-reward-r1'
IMPLEMENTATION = 'DL-production-reward-source-engineering-r1'
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
MODEL_PINS = dict(
    checkpoint_sha256='2ab693c8ded4d8f2186bd489aa8b77498448ba2f2c331090159da4e5f08056b9',
    params_sha256='14859c4e129b51aa488c10eab09f16d56fd110c76239b2c9072f415829d111c2',
    config_source_sha256='11428512fdc9b9e04a5384e419c5abcef93e5429db0a3327d05ccb0d4bd20445')
COUNTERS = dict(updates=74493, batches=74493, actions=18769,
                eval_actions=1001, eval_seed=3404107006)
NUMERIC_FIELDS = ('episode', 'positions', 'h', 'latent', 'reward_f64', 'first_commands')
NO_AUTHORITY = dict(training_authorized=False, method_evidence=False,
    real_gate_authority=False, eligible_method_training=False,
    proceed_short_training=False, method_effectiveness_established=False,
    current_rgb_increment_over_complete_history_established=False,
    continuing_kl_compression_effect_established=False)


def require(condition, message):
  if not condition:
    raise ValueError(message)


def sha(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def read_json(path):
  def reject(value):
    raise ValueError('Nonfinite JSON constant: ' + value)
  return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)


def typed_hash(value):
  value = np.asarray(value)
  require(not value.dtype.hasobject and np.isfinite(value).all(), 'Invalid typed numeric proof')
  return dict(dtype=value.dtype.str, dtype_name=str(value.dtype), shape=list(value.shape),
      sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())


def command_library():
  values = np.zeros((7, 12), np.float32)
  values[1] = .25
  values[2] = -.25
  values[3, [1, 10]] = .25
  values[3, [4, 7]] = -.25
  values[4] = -values[3]
  values[5, [2, 8]] = .25
  values[5, [5, 11]] = -.25
  values[6] = -values[5]
  return values


def native_bf16(value, name):
  require(value.dtype == np.float64 and np.isfinite(value).all(),
          'Require finite source F64: ' + name)
  native = value.astype(ml_dtypes.bfloat16)
  restored = native.astype(np.float64)
  require(np.isfinite(native).all() and restored.tobytes() == value.tobytes(),
          'Source BF16 promotion is not byte-exact: ' + name)
  return native


def load_numeric_episode(path, episode):
  """Load exactly six lazy numeric fields; never request an opaque envelope."""
  require(type(episode) is int and episode in EPISODES, 'Unapproved calibration episode')
  with np.load(path, allow_pickle=False) as archive:
    require(set(NUMERIC_FIELDS).issubset(archive.files), 'Missing required numeric fields')
    data = {name: archive[name].copy() for name in NUMERIC_FIELDS}
  require(data['episode'].dtype == np.int32 and data['episode'].shape == () and
          int(data['episode']) == episode, 'Episode identity/dtype/shape changed')
  require(data['positions'].dtype == np.int32 and data['positions'].shape == (4,) and
          np.array_equal(data['positions'], POSITIONS), 'Positions changed')
  for name, dtype, dimensions in (
      ('h', np.float64, (4, 4096)), ('latent', np.float64, (4, 4, 2, 32, 32)),
      ('reward_f64', np.float64, (4, 15, 20)), ('first_commands', np.float32, (4, 15, 12))):
    value = data[name]
    require(value.dtype == dtype and value.shape == dimensions and np.isfinite(value).all(),
            'Invalid finite numeric field: ' + name)
  z = data['latent']
  require(np.logical_or(z == 0., z == 1.).all() and np.all(z.sum(-1) == 1.),
          'Actual sampled latent must be exactly onehot')
  controls = np.ascontiguousarray(data['first_commands'][:, :7])
  require(controls.tobytes() == np.broadcast_to(command_library(), (4, 7, 12)).tobytes(),
          'First raw normalized control bytes changed')
  cost = np.sum(data['reward_f64'][:, :7], axis=-1, dtype=np.float64)
  require(np.isfinite(cost).all(), 'Cost sum overflow')
  h_native, z_native = native_bf16(data['h'], 'h'), native_bf16(z, 'latent')
  proofs = {name: typed_hash(data[name]) for name in NUMERIC_FIELDS}
  proofs.update(h_native=typed_hash(h_native), latent_native=typed_hash(z_native),
                controls=typed_hash(controls), true_cost=typed_hash(cost))
  return dict(data, h_native=h_native, latent_native=z_native,
              controls=controls, true_cost=cost, typed_hashes=proofs)


def authority(dataset, check=lambda: None):
  """Authenticate fixed source JSON and all eight archive bytes before decoding."""
  dataset = Path(dataset).resolve(strict=True)
  require(dataset.is_dir(), 'Require fixed source directory')
  documents = {}
  for name, digest in SOURCE_PINS.items():
    check()
    path = dataset / name
    require(path.resolve(strict=True).parent == dataset and sha(path) == digest,
            'Published source identity differs: ' + name)
    documents[name] = read_json(path)
  plan, result, complete, budget, parent, audit = [documents[name] for name in (
      'plan.json', 'result.json', 'action_information_complete.json', 'budget.json',
      'supervisor_complete.json', 'independent-numeric-audit-v1.json')]
  for value in (plan, result, complete, parent):
    require(value['actual_git_sha'] == SOURCE_GIT and value['version'] == 'DL-action-information-r1',
            'Wrong source Git/version')
  require(budget['version'] == 'DL-action-information-r1' and budget['git_sha'] == SOURCE_GIT and
          budget['seconds'] == 180 and complete['plan_sha256'] == SOURCE_PINS['plan.json'] and
          complete['result_sha256'] == SOURCE_PINS['result.json'] and
          parent['budget_sha256'] == SOURCE_PINS['budget.json'] and
          parent['exit_code'] == 0 and parent['timed_out'] is False and
          parent['worker_terminal'] is True and complete['complete_episodes'] == 8 and
          complete['complete_states'] == 32 and complete['complete_records'] == 480,
          'Source completion chain failed')
  require(plan['episodes'] == list(EPISODES) and plan['positions'] == list(POSITIONS) and
          plan['horizon'] == 20 and plan['delta'] == .2 and
          plan['actor_command_order'] == [f'pair{i}{b}' for i in range(4) for b in ('q', 'p')],
          'Frozen source object changed')
  metadata = plan['model_provenance']
  require(all(metadata[k] == v for k, v in MODEL_PINS.items()) and
          metadata['checkpoint_counters'] == COUNTERS, 'Source model pins/counters changed')
  config = metadata['config']
  expected_config = {'task': 'dmc_quadruped_walk', 'env.dmc.proprio': False,
      'env.dmc.size': [64, 64], 'env.dmc.repeat': 1, 'jax.compute_dtype': 'bfloat16',
      'agent.dyn.typ': 'rssm', 'agent.dyn.rssm.deter': 4096, 'agent.dyn.rssm.stoch': 32,
      'agent.dyn.rssm.classes': 32, 'agent.dyn.rssm.unimix': .01,
      'agent.rewhead.output': 'symexp_twohot', 'agent.dt_latch.mode': 'off',
      'agent.rep_probe.mode': 'off'}
  require(all(type(config[k]) is type(v) and config[k] == v for k, v in expected_config.items()) and
          config['env.dmc.camera'] in (-1, 2), 'Source size50m/observation contract changed')
  require(audit['status'] == 'passed' and audit['expected_git_sha'] == SOURCE_GIT and
          audit['source_h_sampled_z_and_actor_raw_byte_exact'] is True and
          audit['source_feedback_bytes_verified'] is True and audit['blind_arrays_read'] is False,
          'Missing independent source feature proof')
  require(result['main_physics_steps'] == result['replay_physics_steps'] == 9600 and
          result['fresh_model_calls'] == result['render_calls'] == result['fresh_actor_rng_calls'] == 0 and
          result['source_artifacts_unchanged'] is True, 'Source physical/readonly coverage changed')
  items = result['evidence_artifacts']
  require(len(items) == 8 and [x['episode'] for x in items] == list(EPISODES),
          'Source episode order changed')
  for item in items:
    check()
    require(item['path'] == f"episode_{item['episode']:03}.npz" and
            type(item['sha256']) is str and re.fullmatch('[0-9a-f]{64}', item['sha256']) is not None,
            'Unsafe/unexpected source path or hash')
    path = dataset / item['path']
    require(path.resolve(strict=True).parent == dataset and sha(path) == item['sha256'],
            'Numeric input bytes changed')
    require(item['records'] == 60 and item['main_steps'] == item['replay_steps'] == 1200 and
            len(item['verifications']) == 60, 'Source per-episode physical coverage changed')
    for position in POSITIONS:
      proofs = [x for x in item['verifications'] if x['position'] == position]
      require(len(proofs) == 15 and [x['control'] for x in proofs] == list(range(15)) and
              all(x['executed_steps'] == x['physics_steps'] == 20 and x['padding_steps'] == 0 and
                  x['end_type'] == 'window' and x['float64_reward_byte_exact'] is True and
                  x['float32_reward_encoding_byte_exact'] is True and
                  x['discount_byte_exact'] is True for x in proofs),
              'Source early-terminal/full-H20 semantics changed')
  return plan, result, items


def load_dataset(dataset, check=lambda: None):
  """Return fixed identity axes and native BF16, retaining typed input receipts."""
  dataset = Path(dataset).resolve(strict=True)
  plan, result, items = authority(dataset, check)
  episodes, proofs = [], []
  for item in items:
    check()
    path = dataset / item['path']
    require(sha(path) == item['sha256'], 'Numeric input bytes changed before numeric read')
    data = load_numeric_episode(path, item['episode'])
    require(sha(path) == item['sha256'], 'Numeric input bytes changed during numeric read')
    episodes.append(data)
    proofs.append(dict(episode=item['episode'], path=item['path'], sha256=item['sha256'],
                       typed_hashes=data['typed_hashes']))
  cost = np.stack([x['true_cost'] for x in episodes])
  old_cost = np.asarray(result['external_control']['rewards'], np.float64)
  require(old_cost.shape == (8, 4, 7) and np.isfinite(old_cost).all() and
          old_cost.tobytes() == cost.tobytes(), 'True cost bytes differ from original published result')
  authority(dataset, check)
  return dict(episodes=np.asarray(EPISODES, np.int32), positions=np.asarray(POSITIONS, np.int32),
      h_native=np.stack([x['h_native'] for x in episodes]),
      latent_native=np.stack([x['latent_native'] for x in episodes]),
      controls=np.stack([x['controls'] for x in episodes]), true_cost=cost,
      source_plan=plan, source_result=result, input_artifacts=proofs,
      source_git_sha=SOURCE_GIT, source_json_sha256=dict(SOURCE_PINS),
      source_proof_scope='authenticated reuse; no fresh physics/carry/key or model revalidation',
      records_decoded=False, blind_arrays_read=False, **NO_AUTHORITY)
