"""DL-predictive-MC-r1 CPU engineering, never actual 50m skill acceptance."""
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_predictive_mc as mc


def write(path, obj):
  path.write_text(json.dumps(obj, sort_keys=True, indent=2) + '\n', encoding='utf-8')


def synthetic_sources(root):
  """Only 8 calibration NPZ exist: attempted blind-array access would fail."""
  source, scores = root / 'collection', root / 'modal'
  source.mkdir(); scores.mkdir()
  expected = mc.d.episode_plan(32, 8, mc.HISTORY_SEED)
  selected = [x for x in expected if x['split'] == 'calibration']
  plan = dict(git_sha=mc.COLLECTION_SHA, seed=mc.HISTORY_SEED, episodes=expected,
      task='dmc_quadruped_walk', camera=2, shape=[64, 64],
      checkpoint_sha256=mc.CHECKPOINT_SHA, params_sha256=mc.PARAMS_SHA,
      config_source_sha256=mc.CONFIG_SHA)
  write(source / 'collection_plan.json', plan)
  artifacts, modal_artifacts = [], []
  for definition in expected:
    episode = definition['episode']
    path = source / f'episode_{episode:03}.npz'
    entry = dict(**definition, path=path.name, frames=202, sha256='a'*64)
    if definition['split'] == 'calibration':
      frames = 202
      first = np.zeros(frames, bool); first[0] = True
      last = np.zeros(frames, bool); last[-1] = True
      raw = dict(image=np.zeros((frames, 64, 64, 3), np.uint8),
          preserve=np.ones((frames, 64, 64), bool), body=np.zeros((frames, 64, 64), bool),
          reward=np.zeros(frames, np.float32), is_first=first, is_last=last,
          is_terminal=np.zeros(frames, bool), action=np.zeros((frames, 12), np.float32))
      # Above-one raw samples deliberately retained and tested against source.
      raw['action'][:, 0] = 2.5
      np.savez_compressed(path, **raw)
      entry['sha256'] = mc.d.sha256(path)
      positions = mc.d.sample_positions(frames, 16, 100)
      truth = np.zeros((16, 6, 10), np.float64)
      count = 9 if len(modal_artifacts) < 4 else 8
      truth[:count, 1] = .02
      row = dict(logits=np.zeros((16, 4, 2, 6, 10, 255), np.float32),
          bins=np.linspace(-1, 1, 255, dtype=np.float32),
          position=positions, episode=np.full(16, episode, np.int32),
          true_rewards=truth, physics_repeat_error=np.zeros(16),
          controls=np.stack([mc.modal.control_sequences(raw['action'][t:t+10]) for t in positions]))
      target = scores / path.name
      np.savez_compressed(target, **row)
      modal_artifacts.append(dict(episode=episode, positions=16, path=target.name,
          sha256=mc.d.sha256(target), source_sha256=entry['sha256']))
    artifacts.append(entry)
  write(source / 'collection_complete.json', dict(plan_sha256=mc.d.sha256(source / 'collection_plan.json'),
      params_unchanged=True, artifacts=artifacts))
  modal_plan = dict(version=mc.modal.VERSION, source=plan,
      source_complete_sha256=mc.d.sha256(source / 'collection_complete.json'),
      metadata=dict(checkpoint_sha256=mc.CHECKPOINT_SHA, params_sha256=mc.PARAMS_SHA,
          config_source_sha256=mc.CONFIG_SHA, checkpoint_counters=dict(updates=74493)),
      horizon=10, controls=6, prefixes=[1, 3, 10], diagnostic_seed=mc.HISTORY_SEED,
      statistical_seed=mc.STAT_SEED, selected_episodes=[x['episode'] for x in selected],
      gate_changed=False, optimizer_updates=0)
  write(scores / 'plan.json', modal_plan)
  result = dict(version=mc.modal.VERSION, primary_horizon=10, positions=128,
      calibration_episodes=8, analysis_git_sha=mc.MODAL_SHA, params_unchanged=True,
      maximum_physics_repeat_error=0, plan_sha256=mc.d.sha256(scores / 'plan.json'),
      evidence_artifacts=modal_artifacts, control_sensitivity=dict(positions=68, episodes=8))
  write(scores / 'result.json', result)
  audit_path = root / 'audit.json'
  write(audit_path, dict(source_result_sha256=mc.d.sha256(scores / 'result.json'),
      all_eight_calibration_artifact_hashes_verified=True, blind_arrays_read=False,
      prior_rollouts_equal=True, fixed_controls_exact=True, maximum_physics_repeat_error=0))
  return source, scores, audit_path


@contextlib.contextmanager
def synthetic_pins(scores, audit):
  # Fixtures cannot bypass production CLI: only test-time in-memory constants.
  with mock.patch.multiple(mc, MODAL_RESULT_SHA=mc.d.sha256(scores / 'result.json'),
      MODAL_PLAN_SHA=mc.d.sha256(scores / 'plan.json'), MODAL_AUDIT_SHA=mc.d.sha256(audit)):
    yield


class MCTest(unittest.TestCase):
  def test_real_tiny_model_private_sampler_paths_and_readonly_contract(self):
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--fixture'],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
        env=dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT)))
    self.assertEqual(child.returncode, 0, child.stdout[-4000:] + child.stderr[-5000:])
    facts = json.loads(child.stdout.strip().splitlines()[-1])
    for key in ('q_equals_p_exact', 'private_seed_repeat_exact', 'chunk_partition_exact',
        'future_key_prefix_exact', 'future_key_changes_prediction', 'future_action_prefix_exact',
        'all_latent_steps_sampled', 'no_ninjax_seed', 'state_unchanged', 'writes_rejected',
        'actual_unimix_frequency', 'strict_transfer_guard_keys'):
      self.assertTrue(facts[key], key)

  def test_probability_before_CE_is_not_path_CE_average(self):
    # Asymmetric paths also distinguish mean-logits from probability mixing.
    logits = np.log(np.array([[[[[.9, .1]]], [[[.3, .7]]]]]))
    logits = np.broadcast_to(logits, (5, 2, 6, 10, 2)).copy()
    logits[4] = np.log([.6, .4])
    sums = mc.log_probability_sum(logits)[None]
    score = mc.scores_from_sums(sums, 2, np.zeros((1, 6, 10)), np.array([0, 1], np.float32))
    np.testing.assert_allclose(score['ce_q'], -np.log(.6), atol=1e-14)
    np.testing.assert_allclose(score['ce_p'], -np.log(.6), atol=1e-14)
    path_ce = float(-mc.log_probabilities(logits)[0, :, 0, 0, 0].mean())
    mean_logit_ce = float(-mc.log_probabilities(logits.mean(1))[0, 0, 0, 0])
    self.assertGreater(path_ce, -np.log(.6)+.1)
    self.assertLess(mean_logit_ce, -np.log(.6)-.05)

  def test_stopping_is_fixed_to_two_primary_gaps_and_finite_MC_error(self):
    positive = {key: dict(point=.01, lower95=.004) for key in mc.GAPS}
    values = {key: .01 for key in mc.GAPS}
    good = mc.reliability(values, values, values, values, [values]*4, positive)
    self.assertEqual(good['decision'], 'design_independent_revision_only')
    large = {key: .012 for key in mc.GAPS}
    bad = mc.reliability(large, values, values, values, [values]*4, positive)
    self.assertEqual(bad['decision'], 'MC-inconclusive')
    weak = {key: dict(point=.001, lower95=.0001) for key in mc.GAPS}
    shifted = {key: .0105 for key in mc.GAPS}
    ambiguous = mc.reliability(shifted, values, values, values, [values]*4, weak)
    self.assertTrue(ambiguous['mc_stable'])
    self.assertEqual(ambiguous['decision'], 'stop_reward_support_candidate')
    negative = dict(positive)
    negative[mc.GAPS[1]] = dict(point=-.01, lower95=-.02)
    self.assertEqual(mc.reliability(values, values, values, values, [values]*4,
        negative)['decision'], 'stop_reward_support_candidate')

  def test_artifact_path_reads_only_calibration_and_rejects_rehashed_position(self):
    with tempfile.TemporaryDirectory() as temp:
      source, scores, audit = synthetic_sources(Path(temp))
      with synthetic_pins(scores, audit):
        entries, proof = mc.load_sources(source, scores, audit)
      self.assertEqual(len(entries), 8)
      self.assertFalse(proof['blind_arrays_read'])
      self.assertEqual(proof['control_sensitivity_positions'], 68)
      self.assertEqual(len(list(source.glob('*.npz'))), 8)
      result = json.loads((scores / 'result.json').read_text())
      artifact = result['evidence_artifacts'][0]
      path = scores / artifact['path']
      with np.load(path, allow_pickle=False) as data:
        row = {k: data[k].copy() for k in data.files}
      row['position'][1] += 1  # Still ordered and legal, but not frozen selection.
      np.savez_compressed(path, **row)
      artifact['sha256'] = mc.d.sha256(path)
      write(scores / 'result.json', result)
      metadata = json.loads(audit.read_text())
      metadata['source_result_sha256'] = mc.d.sha256(scores / 'result.json')
      write(audit, metadata)
      with synthetic_pins(scores, audit), self.assertRaisesRegex(ValueError, 'outcome-independent'):
        mc.load_sources(source, scores, audit)

  def test_summary_requires_control_sensitivity_and_never_authorizes_training(self):
    truth = np.zeros((128, 6, 10))
    episodes = np.repeat(np.arange(8), 16)
    probabilities = np.tile([.05, .9, .05], (128, 4, 5, 6, 10, 1))
    probabilities[:, :, 3] = [.4, .2, .4]
    probabilities[:, :, 4] = [.3, .4, .3]
    result = mc.report(np.log(probabilities*32), truth, episodes, np.array([-1,0,1], np.float32))
    self.assertTrue(result['numerical_reliability']['mc_stable'])
    self.assertFalse(result['necessary_skill_supported_on_calibration'])
    self.assertEqual(result['numerical_reliability']['decision'], 'stop_reward_support_candidate')
    for key in ('training_authorized', 'proceed_short_training', 'eligible_method_training'):
      self.assertFalse(result[key])


def fixture():
  dll = (os.add_dll_directory(os.environ['DL_DLL_DIRECTORY'])
      if os.name == 'nt' and os.environ.get('DL_DLL_DIRECTORY') else None)
  import elements
  import jax
  import jax.numpy as jnp
  import ninjax as nj
  from ruamel.yaml import YAML
  from dreamerv3.rssm import RSSM
  from types import SimpleNamespace
  presets = YAML(typ='safe').load((ROOT / 'dreamerv3/configs.yaml').read_text(encoding='utf-8'))
  config = elements.Config(presets['defaults']).update(presets['debug'])
  config = elements.Config({**config.flat, **{'jax.platform': 'cpu', 'jax.compute_dtype': 'float32',
      'jax.precompile': False, 'jax.profiler': False, 'jax.enable_policy': False,
      'batch_size': 2, 'batch_length': 3, 'replay_context': 0,
      'agent.imag_length': 2, 'agent.imag_last': 2,
      'agent.rep_probe.mode': 'off', 'agent.dt_latch.mode': 'off'}})
  config = config.update(logdir='/tmp/dl-mc-fixture', seed=7)
  before_env_modules = {name for name in sys.modules if name.startswith('dm_control')}
  agent = mc.agent_from_spaces(config)
  assert {name for name in sys.modules if name.startswith('dm_control')} == before_env_modules
  original_counters = agent.save()['counters']
  # Agent initialization enables strict transfer guards. Constants are JIT-built.
  def nonuniform(params):
    result = dict(params)
    name = 'rew/head/logits/kernel'
    result[name] = jnp.sin(jnp.arange(result[name].size, dtype=jnp.float32)).reshape(result[name].shape)*.2
    return result
  agent.params = jax.jit(nonuniform)(agent.params)
  carry = agent.init_train(1)
  h = jax.jit(lambda x: x[0])(carry[1]['deter'])
  factors, classes = config.agent.dyn.rssm.stoch, config.agent.dyn.rssm.classes
  q, p, controls = jax.jit(lambda: (
      jnp.zeros((4, factors, classes), jnp.float32), jnp.zeros((factors, classes), jnp.float32),
      jnp.asarray(mc.modal.control_sequences(np.zeros((10, 12), np.float32)))))()
  predictor = mc.Predictor(agent)
  images = np.random.default_rng(31).integers(0, 256, (4,64,64,3), dtype=np.uint8)
  before = carry[:2]
  prepared = predictor.history_call(predictor.prepare, before, images,
      np.zeros(12,np.float32), np.bool_(True), seed=31)
  same_mode = predictor.history_call(predictor.mode_predict, before, images,
      np.zeros(12,np.float32), np.bool_(True), np.asarray(mc.modal.control_sequences(
          np.zeros((10,12),np.float32))), seed=31)
  rerun_mode = predictor.history_call(predictor.mode_predict, before, images,
      np.zeros(12,np.float32), np.bool_(True), np.asarray(mc.modal.control_sequences(
          np.zeros((10,12),np.float32))), seed=31)
  keys = mc.private_keys(0, 10, 0, 0, 8)
  repeated = mc.private_keys(0, 10, 0, 0, 8)
  split = [mc.private_keys(0, 10, 0, x, 4) for x in (0, 4)]
  run = mc.modal.readonly_jit(lambda h, q, p, a, k: mc.model_chunk(agent.model, h, q, p, a, k))
  sampled = []
  original = mc.coupled_sample
  def counted(*args):
    sampled.append(True)
    return original(*args)
  with mock.patch.object(mc, 'coupled_sample', counted), mock.patch.object(nj, 'seed',
      side_effect=AssertionError('Private MC must not request NJ seed')):
    state, result = run(agent.params, h, q, p, controls, keys)
  _, again = run(agent.params, h, q, p, controls, repeated)
  split_results = [run(agent.params, h, q, p, controls, k)[1] for k in split]
  changed_keys = jax.jit(lambda a,b: a.at[:, :, 5:].set(b[:, :, 5:]))(
      keys, mc.private_keys(0, 10, 1, 0, 8))
  _, changed = run(agent.params, h, q, p, controls, changed_keys)
  changed_controls = jax.jit(lambda a: a.at[:, 9].set(a[:, 9]+.1))(controls)
  _, action_changed = run(agent.params, h, q, p, changed_controls, keys)
  first_param = next(iter(agent.params))
  def write_state():
    nj.context()[first_param] = nj.context()[first_param] + 1
    return 0
  writes_rejected = False
  try:
    mc.modal.readonly_jit(write_state)(agent.params)
  except AssertionError:
    writes_rejected = True
  # Actual production _dist implementation, large unimix makes double mixing detectable.
  dyn = RSSM(dict(action=elements.Space(np.float32, (2,))), unimix=.4, name='frequency_dyn')
  unmixed = RSSM(dict(action=elements.Space(np.float32, (2,))), unimix=0., name='frequency_unmixed')
  surrogate = SimpleNamespace(dyn=dyn)
  def frequencies():
    logs = jnp.log(jnp.array([.8,.1,.1]))
    logs = jnp.broadcast_to(logs, (2, 50000, 1, 1, 3))
    seeds = jax.random.split(jax.random.PRNGKey(71), 50000)[:, None]
    draws = mc.coupled_sample(surrogate, logs, seeds)
    reference = jax.vmap(jax.vmap(lambda value, key: dyn._dist(value).sample(seed=key)))(logs[0], seeds)
    other = mc.coupled_sample(SimpleNamespace(dyn=unmixed), logs, seeds)
    return draws.mean((1,2,3)), other.mean((1,2,3)), jnp.all(draws[0] == draws[1]) & jnp.all(draws[0] == reference)
  _, (frequencies, no_mix_frequencies, identical) = mc.modal.readonly_jit(frequencies)(agent.params)
  with jax._src.config.explicit_device_get_scope():
    np.testing.assert_array_equal(np.asarray(same_mode['logits']), np.asarray(rerun_mode['logits']))
    assert np.asarray(prepared[1]).shape == (4, factors, classes)
    result, again, changed, action_changed = [jax.tree.map(np.asarray, v)
        for v in (result, again, changed, action_changed)]
    split_values = [np.asarray(v['logits']) for v in split_results]
    np.testing.assert_array_equal(result['logits'][0], result['logits'][4])
    for branch in range(4):
      np.testing.assert_array_equal(result['logits'][branch], result['logits'][4])
    np.testing.assert_array_equal(result['logits'], again['logits'])
    np.testing.assert_array_equal(result['logits'], np.concatenate(split_values, 1))
    np.testing.assert_array_equal(result['logits'][..., :4, :], changed['logits'][..., :4, :])
    assert np.any(result['logits'][..., 4:, :] != changed['logits'][..., 4:, :])
    np.testing.assert_array_equal(result['logits'][..., :9, :], action_changed['logits'][..., :9, :])
    for key in state:
      np.testing.assert_array_equal(np.asarray(state[key]), np.asarray(agent.params[key]))
    np.testing.assert_allclose(np.asarray(frequencies), np.tile([.613333333,.193333333,.193333333], (2,1)), atol=.006)
    np.testing.assert_allclose(np.asarray(no_mix_frequencies), np.tile([.8,.1,.1], (2,1)), atol=.006)
    assert bool(np.asarray(identical))
    assert len(sampled) == 11, len(sampled)
  assert agent.save()['counters'] == original_counters
  print(json.dumps(dict(q_equals_p_exact=True, private_seed_repeat_exact=True,
      chunk_partition_exact=True, future_key_prefix_exact=True, future_key_changes_prediction=True,
      future_action_prefix_exact=True, all_latent_steps_sampled=True, no_ninjax_seed=True,
      state_unchanged=True, writes_rejected=writes_rejected, actual_unimix_frequency=True,
      strict_transfer_guard_keys=True, scope='Tiny float32 CPU engineering only')))


if __name__ == '__main__':
  fixture() if sys.argv[1:] == ['--fixture'] else unittest.main()
