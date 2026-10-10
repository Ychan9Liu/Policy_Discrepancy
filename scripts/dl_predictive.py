"""DL-predictive-r1: calibration-only finite-control prediction diagnostics.

No optimizer updates, gate fitting, blind episode arrays, or future observations
in predictions. Trusted project snapshots are read only after byte verification.
"""
import argparse
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_diagnostic as d

VERSION = 'DL-predictive-r1'
HORIZON = 10
PREFIXES = (1, 3, 10)


def control_sequences(recorded):
  recorded = np.asarray(recorded)
  if recorded.dtype != np.float32 or recorded.ndim != 2 or len(recorded) != HORIZON:
    raise ValueError('Require exactly ten float32 vector actions')
  if not np.isfinite(recorded).all():
    raise ValueError('Nonfinite recorded action')
  # The verified source stores raw sampled Gaussian actions, as returned by
  # Agent.policy and passed to dm_control. Preserve their actual encoding;
  # neither silently clip them nor mistake bounded means for bounded samples.
  fixed = d.fixed_actions(recorded.shape[1:])
  return np.concatenate([recorded[None], np.repeat(fixed[:, None], HORIZON, 1)])


def model_rollout(model, deter, qmode, pmode, sequences):
  """Pure deterministic mode paths; no reward/observation/seed input."""
  import jax.numpy as jnp
  from embodied.jax import nets as nn
  n, a, horizon = deter.shape[0], sequences.shape[0], sequences.shape[1]
  if horizon < 1 or qmode.shape != pmode.shape or qmode.shape[0] != n:
    raise ValueError('Invalid rollout axes')
  expand = lambda x: jnp.repeat(x, a, axis=0)
  h = nn.cast(jnp.concatenate([expand(deter), expand(deter)], 0))
  z = nn.cast(jnp.concatenate([expand(qmode), expand(pmode)], 0))
  logits, means = [], []
  for step in range(horizon):
    action = jnp.tile(sequences[:, step], (n, 1))
    action = nn.cast(jnp.concatenate([action, action], 0))
    h = model.dyn._core(h, z, action)
    z = nn.cast(model.dyn._dist(model.dyn._prior(h)).pred())
    reward = model.rew(model.feat2tensor(dict(deter=h, stoch=z)), 1)
    logits.append(reward.logits)
    means.append(reward.pred())
  logits = jnp.stack(logits, 1).reshape((2, n, a, horizon, -1)).swapaxes(0, 1)
  means = jnp.stack(means, 1).reshape((2, n, a, horizon)).swapaxes(0, 1)
  return dict(logits=logits, reward_mean=means, bins=reward.bins)


def readonly_jit(function):
  import jax
  import ninjax as nj
  def invoke(*args, **kwargs):
    state, result, _, modified, created = nj.pure(function)(
        *args, create=False, modify=True, track=True, **kwargs)
    if modified or created:
      raise AssertionError('Predictive diagnostic attempted checkpoint state mutation')
    return state, result
  return jax.jit(invoke)


class Predictor:
  def __init__(self, agent):
    import jax
    import jax.numpy as jnp
    self.agent = agent
    model = agent.model
    def observe(carry, images, previous, first):
      n = len(images)
      enc, dyn = jax.tree.map(lambda x: jnp.repeat(x, n, 0), carry)
      reset = jnp.full((n,), first)
      _, _, tokens = model.enc(enc, {'image': images}, reset, False, single=True)
      dyn, _, feat = model.dyn.observe(dyn, tokens,
          {'action': jnp.broadcast_to(previous, (n, *previous.shape))},
          reset, False, single=True)
      return (carry[0], jax.tree.map(lambda x: x[:1], dyn)), feat
    def advance(carry, image, previous, first):
      return observe(carry, image[None], previous, first)[0]
    def predict(carry, images, previous, first, sequences):
      _, feat = observe(carry, images, previous, first)
      h, q = feat['deter'], feat['logit']
      from dreamerv3.dt_latch import require_valid
      require_valid(jnp.all(h == h[:1]))
      p = model.dyn._prior(h)
      return model_rollout(model, h, model.dyn._dist(q).pred(),
          model.dyn._dist(p).pred(), sequences)
    self.advance = readonly_jit(advance)
    self.predict = readonly_jit(predict)

  def call(self, function, carry, *args, seed):
    import embodied.jax.internal as internal
    inputs = internal.device_put(args, self.agent.train_mirrored)
    key = internal.device_put(np.asarray([seed, 0], np.uint32), self.agent.train_mirrored)
    _, output = function(self.agent.params, carry, *inputs, seed=key)
    return output


def report(rows, artifacts):
  data = {k: np.concatenate([r[k] for r in rows]) for k in rows[0] if k != 'bins'}
  logits, truth = data['logits'], data['true_rewards']
  labels = np.broadcast_to(truth[:, None], logits[:, :, 0].shape[:-1]).astype(np.float32)
  score = d.compatibility_numpy(logits[:, :, 0], logits[:, :, 1], labels, rows[0]['bins'])
  episodes = data['episode']
  by_prefix = {}
  for h in PREFIXES:
    gap = score['loss_gap'][..., :h].mean(-1)
    ceq, cep = score['ell_q'][..., :h].mean(-1), score['ell_p'][..., :h].mean(-1)
    groups = {}
    for name, controls in (('recorded', [0]), ('fixed', [1, 2, 3, 4, 5])):
      interval = lambda x: d.cluster_interval(x, episodes, draws=2000, seed=20261016)
      groups[name] = dict(clean_minus_prior_log_score=interval(gap[:, 0, controls].mean(-1)),
          clean_over_deprived_log_score=interval((ceq[:, 3, controls] - ceq[:, 0, controls]).mean(-1)),
          mean_CE_q=float(ceq[:, 0, controls].mean()), mean_CE_p=float(cep[:, 0, controls].mean()),
          background_a_minus_clean_CE=interval((ceq[:, 1, controls] - ceq[:, 0, controls]).mean(-1)),
          background_b_minus_clean_CE=interval((ceq[:, 2, controls] - ceq[:, 0, controls]).mean(-1)))
    by_prefix[str(h)] = groups
  primary = by_prefix['10']['fixed']
  score_supported = all(primary[k]['lower95'] is not None and primary[k]['lower95'] > 0
      for k in ('clean_minus_prior_log_score', 'clean_over_deprived_log_score'))
  span = np.ptp(truth[:, 1:].sum(-1), axis=-1)
  sensitive = span >= .1
  sensitivity_supported = sensitive.sum() >= 16 and len(np.unique(episodes[sensitive])) >= 4
  supported = score_supported and sensitivity_supported
  return dict(version=VERSION, analysis_git_sha=d.git_sha(), calibration_episodes=len(np.unique(episodes)),
      positions=len(episodes), primary_horizon=10, prefixes=by_prefix,
      fixed_control_true_return_span=d.cluster_interval(span, episodes, draws=2000, seed=20261016),
      control_sensitivity=dict(threshold_cumulative=.1, positions=int(sensitive.sum()),
          episodes=len(np.unique(episodes[sensitive])), sufficient=sensitivity_supported),
      prediction_skill_direction_supported=score_supported,
      maximum_physics_repeat_error=float(data['physics_repeat_error'].max()),
      evidence_artifacts=artifacts, necessary_skill_supported_on_calibration=supported,
      next_step='Design independent revision only' if supported else 'Do not adopt multi-step reward proxy from this evidence',
      interpretation='Exploratory calibration skill, not causal correctness, gate discrimination or performance',
      training_authorized=False, proceed_short_training=False)


def run(args):
  from scripts.dl_resources import require_resource, verify_runtime
  resource = require_resource(args.resource_record)
  source, output = Path(args.dataset), Path(args.output)
  if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
    raise RuntimeError('Actual diagnostic requires clean committed checkout')
  if output.exists():
    raise FileExistsError(output)
  plan = json.loads((source / 'collection_plan.json').read_text())
  complete = json.loads((source / 'collection_complete.json').read_text())
  if (complete['plan_sha256'] != d.sha256(source / 'collection_plan.json') or
      not complete['params_unchanged'] or plan['seed'] != 20261012 or
      plan['episodes'] != d.episode_plan(32, 8, 20261012) or len(complete['artifacts']) != 32 or
      [{k: x[k] for k in ('episode', 'split', 'seed', 'policy_seed')}
       for x in complete['artifacts']] != plan['episodes'] or
      plan['git_sha'] != '5765c893a419a59db46e5a3e3197af20b9059757'):
    raise ValueError('Require complete unchanged H100 collection')
  selected = [x for x in complete['artifacts'] if x['split'] == 'calibration']
  if len(selected) != 8:
    raise ValueError('Require exactly the eight declared calibration episodes')
  if d.sha256(args.checkpoint) != plan['checkpoint_sha256'] or d.sha256(args.config) != plan['config_source_sha256']:
    raise ValueError('Checkpoint/config differ from collection')
  if (plan['checkpoint_sha256'] not in (
      '2ab693c8ded4d8f2186bd489aa8b77498448ba2f2c331090159da4e5f08056b9',
      'c9c893081485c8ad22468611d046bc5bf9fbe463d55996fda15cf85b2a0323fe') or
      plan['config_source_sha256'] != '11428512fdc9b9e04a5384e419c5abcef93e5429db0a3327d05ccb0d4bd20445'):
    raise ValueError('Source is not a frozen checkpoint/config pair')
  if args.platform != 'cuda':
    raise ValueError('Actual predictive diagnostic requires assigned CUDA resource')
  import jax
  from embodied.run.protocol_v1 import parameter_digest
  agent, config, metadata = d.load_agent(args)
  resource['runtime_cuda'] = verify_runtime(resource, compute=True, graphics=False)
  if metadata['params_sha256'] != plan['params_sha256']:
    raise ValueError('Loaded checkpoint parameters differ')
  output.mkdir()
  d.write_json(output / 'plan.json', dict(version=VERSION, source=plan,
      source_complete_sha256=d.sha256(source / 'collection_complete.json'), metadata=metadata,
      resource=resource, selected_episodes=[x['episode'] for x in selected],
      selection='calibration only; same 16 H100 positions', controls=6,
      recorded_action_semantics='Verified raw sampled float32 actions; retained without clipping',
      horizon=HORIZON, prefixes=PREFIXES, diagnostic_seed=20261012,
      statistical_seed=20261016, optimizer_updates=0, gate_changed=False))
  predictor, rows, artifacts = Predictor(agent), [], []
  started = time.perf_counter()
  for entry in selected:
    path = source / entry['path']
    if not path.resolve().is_relative_to(source.resolve()) or d.sha256(path) != entry['sha256']:
      raise ValueError('Source path/hash changed')
    with np.load(path, allow_pickle=False) as a:
      data = {k: a[k] for k in a.files}
    states = pickle.loads(data.pop('snapshots').tobytes())
    positions = d.sample_positions(len(data['reward']), 16, 100)
    if len(positions) != 16:
      raise ValueError('No complete predetermined window')
    env = d.make_dmc(args.task, entry['seed'])
    env.reset()
    carry, records = agent.init_train(1)[:2], []
    try:
      for t in range(int(positions[-1]) + 1):
        previous = data['action'][t - 1] if t else np.zeros_like(data['action'][0])
        seed = int(np.random.SeedSequence([20261012, entry['episode'], t]).generate_state(1)[0])
        before = carry
        carry = predictor.call(predictor.advance, before, data['image'][t], previous, data['is_first'][t], seed=seed)
        if t not in positions:
          continue
        sequences = control_sequences(data['action'][t:t + HORIZON])
        images = d.intervention_images(data['image'][t], data['preserve'][t], data['body'][t], [20261012, entry['episode']])
        # Construct predictions before obtaining any counterfactual reward.
        prediction = predictor.call(predictor.predict, before, images, previous, data['is_first'][t], sequences, seed=seed)
        with jax._src.config.explicit_device_get_scope():
          prediction = jax.tree.map(lambda x: np.asarray(x), prediction)
        if not np.array_equal(prediction['logits'][:, 1], np.repeat(prediction['logits'][:1, 1], 4, 0)):
          raise AssertionError('Observational variants changed prior rollout')
        rewards, errors, hashes = [], [], []
        for sequence in sequences:
          rew, error, digest = d.verified_consequence(env, states[t], sequence[0], sequence[1:], HORIZON)
          rewards.append(rew)
          errors.append(error)
          hashes.append(digest)
        if not np.array_equal(np.asarray(rewards[0], np.float32), data['reward'][t + 1:t + 1 + HORIZON]):
          raise AssertionError('Recorded full-window reward encoding changed')
        records.append(dict(**prediction, position=t, true_rewards=np.asarray(rewards),
            controls=sequences, physics_repeat_error=max(errors), final_state_hashes=hashes))
      row = {k: np.stack([r[k] for r in records]) for k in records[0] if k != 'bins'}
      row['bins'] = records[0]['bins']
      row['episode'] = np.full(len(records), entry['episode'], np.int32)
      target = output / f'episode_{entry["episode"]:03}.npz'
      np.savez_compressed(target, **row)
      rows.append(row)
      artifacts.append(dict(episode=entry['episode'], source_sha256=entry['sha256'],
          path=target.name, sha256=d.sha256(target), positions=len(records)))
      print(json.dumps(artifacts[-1]), flush=True)
    finally:
      env.close()
  if parameter_digest(agent) != metadata['params_sha256']:
    raise AssertionError('Predictive diagnostic changed checkpoint state')
  result = report(rows, artifacts)
  result.update(params_unchanged=True, elapsed_seconds=time.perf_counter() - started,
      plan_sha256=d.sha256(output / 'plan.json'))
  d.write_json(output / 'result.json', result)
  print(json.dumps(dict(complete=True, necessary_skill_supported=result['necessary_skill_supported_on_calibration'],
      training_authorized=False)), flush=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  for field in ('checkpoint', 'config', 'dataset', 'output', 'resource_record'):
    parser.add_argument('--' + field.replace('_', '-'), required=True)
  parser.add_argument('--task', default='dmc_quadruped_walk')
  parser.add_argument('--platform', default='cuda')
  run(parser.parse_args())
