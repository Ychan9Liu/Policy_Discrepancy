"""CPU fixtures only; no production source, checkpoint, blind, physics or CUDA."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_reset_features as f


def fake_episode():
  frames = f.SOURCE_FRAMES
  first, last = np.zeros(frames, bool), np.zeros(frames, bool)
  first[0], last[-1] = True, True
  raw = np.linspace(-.3, .3, frames*12).astype(np.float32).reshape((frames, 12))
  raw[-1] = 0
  executed = np.ones(frames, bool); executed[-1] = False
  return dict(image=np.zeros((frames, 64, 64, 3), np.uint8),
      reward=np.zeros(frames, np.float32), is_first=first, is_last=last,
      is_terminal=np.zeros(frames, bool), action=raw, action_executed=executed)


class ResetFeaturesTest(unittest.TestCase):
  def test_import_is_loader_and_device_free(self):
    code = "import os,sys; before=dict(os.environ); import scripts.dl_reset_features; assert dict(os.environ)==before; assert 'jax' not in sys.modules; assert 'dreamerv3.agent' not in sys.modules; assert not any(x.startswith('dm_control') for x in sys.modules)"
    child = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=str(ROOT)), capture_output=True, text=True, timeout=30)
    self.assertEqual(child.returncode, 0, child.stdout+child.stderr)

  def test_common_roots_only_have_four_sample_labels(self):
    roots = f.private_roots()
    self.assertEqual(roots.shape, (4, 2)); self.assertEqual(roots.dtype, np.uint32)
    self.assertEqual(len({key.tobytes() for key in roots}), 4)
    for index in reversed(range(4)):
      expected = np.random.default_rng([20261026, 0x52535231, index]).integers(
          0, np.iinfo(np.uint32).max, (2,), np.uint32)
      np.testing.assert_array_equal(f.private_root_key(index), expected)
      np.testing.assert_array_equal(roots[index], expected)
    for invalid in (-1, 4, True, np.int64(0), 0.):
      with self.assertRaises(ValueError): f.private_root_key(invalid)

  def test_typed_numeric_digest_distinguishes_shapes_dtypes_and_bytes(self):
    digest = f.tree_digest({'x': np.array(1., np.float32)})
    for value in (np.array([1.], np.float32), np.array(1., np.float64), np.array(2., np.float32)):
      self.assertNotEqual(digest, f.tree_digest({'x': value}))
    self.assertNotEqual(f.tree_digest([np.float32(1)]), f.tree_digest((np.float32(1),)))
    self.assertEqual(f.tree_digest({'b': 2, 'a': 1}), f.tree_digest({'a': 1, 'b': 2}))
    for invalid in (np.array(np.nan), np.array('bad'), np.array(None, object)):
      with self.assertRaises(ValueError): f.tree_digest({'x': invalid})

  def test_annotation_shifts_actual_raw999_without_terminal_action(self):
    episodes = [fake_episode(), fake_episode()]
    originals = f.tree_digest(episodes)
    obs, previous = f.annotation_inputs(episodes)
    self.assertEqual(obs['image'].shape, (2, 1001, 64, 64, 3))
    self.assertEqual(previous['action'].shape, (2, 1001, 12))
    self.assertTrue(np.all(previous['action'][:, 0] == 0))
    for index, ep in enumerate(episodes):
      np.testing.assert_array_equal(previous['action'][index, 1:], ep['action'][:-1])
      self.assertTrue(previous['action'][index, -1].tobytes() == ep['action'][999].tobytes())
      self.assertFalse(np.all(previous['action'][index, -1] == 0))
    self.assertEqual(originals, f.tree_digest(episodes))

  def test_annotation_rejects_incomplete_or_fabricated_terminal(self):
    for change in ('image_dtype', 'short', 'first', 'early_last', 'terminal', 'executed', 'placeholder'):
      ep = fake_episode()
      if change == 'image_dtype': ep['image'] = ep['image'].astype(np.float32)
      if change == 'short': ep['image'] = ep['image'][:-1]
      if change == 'first': ep['is_first'][5] = True
      if change == 'early_last': ep['is_last'][4] = True
      if change == 'terminal': ep['is_terminal'][-1] = True
      if change == 'executed': ep['action_executed'][-1] = True
      if change == 'placeholder': ep['action'][-1] = .25
      with self.subTest(change=change), self.assertRaises(ValueError):
        f.annotation_inputs([ep])
    for invalid in ([], {}, [dict(image=0)]):
      with self.assertRaises(ValueError): f.annotation_inputs(invalid)

  def test_actual_tiny_current_model_reset_and_full_episode_annotation(self):
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--fixture'],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
        env=dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT)))
    self.assertEqual(child.returncode, 0, child.stdout[-6500:]+child.stderr[-10000:])
    facts = json.loads(child.stdout.strip().splitlines()[-1])
    for key in ('official_post_sample_exact', 'official_prior_sample_exact',
        'original_mode_D_exact', 'original_raw_K_exact', 'all32_common_bytes',
        'four_h_logits_mode_signal_exact', 'new_private_q_not_source_claim',
        'current_params_not_stale_policy', 'strict_transfer_guard',
        'full_state_and_counters_unchanged', 'attempted_writes_rejected',
        'attempted_creation_rejected', 'extra_seed_rejected', 'counter_change_rejected',
        'invalid_common_anchor_rejected', 'invalid_reset_rejected',
        'official_full1001_batch_annotation_exact', 'terminal_raw999_exact',
        'all_actual_entries_returned', 'annotation_actor_never_called',
        'annotation_attempted_writes_rejected', 'annotation_attempted_creation_rejected'):
      self.assertTrue(facts[key], key)


def fixture():
  dll = (os.add_dll_directory(os.environ['DL_DLL_DIRECTORY'])
      if os.name == 'nt' and os.environ.get('DL_DLL_DIRECTORY') else None)
  import elements
  import jax
  import jax.numpy as jnp
  import ninjax as nj
  from embodied.jax import internal
  from embodied.jax import nets as nn
  from dreamerv3 import dt_latch
  from ruamel.yaml import YAML
  from scripts.dl_predictive_mc import agent_from_spaces
  from unittest import mock
  presets = YAML(typ='safe').load((ROOT/'dreamerv3/configs.yaml').read_text(encoding='utf-8'))
  config = elements.Config(presets['defaults']).update(presets['debug'])
  config = elements.Config({**config.flat, **{'jax.platform': 'cpu',
      'jax.compute_dtype': 'bfloat16', 'jax.precompile': False, 'jax.profiler': False,
      'jax.enable_policy': True, 'jax.transfer_guard': True, 'batch_size': 2,
      'batch_length': 3, 'replay_context': 1, 'agent.imag_length': 2,
      'agent.imag_last': 2, 'agent.rep_probe.mode': 'off', 'agent.dt_latch.mode': 'off'}}).update(
          logdir='/tmp/dl-reset-feature-fixture', seed=7)
  agent = agent_from_spaces(config)
  rng = np.random.default_rng(31)
  def observation():
    return dict(image=rng.integers(0, 256, (64, 64, 3), np.uint8),
        reward=np.float32(0), is_first=np.bool_(True),
        is_last=np.bool_(False), is_terminal=np.bool_(False))
  observations = [observation() for _ in range(32)]
  input_digest = f.tree_digest(observations)
  adapter = f.ResetFeatureAdapter(agent)
  full_before = f.tree_digest(dict(params=f.host_copy(agent.params), counters=f.counters(agent)))
  no_public = mock.patch.multiple(agent, _seeds=mock.Mock(side_effect=AssertionError('public seeds')),
      _eval_seeds=mock.Mock(side_effect=AssertionError('public eval seeds')),
      policy=mock.Mock(side_effect=AssertionError('public policy')))
  with no_public:
    cal = adapter.extract(observations[:8])
    blind_adapter = f.ResetFeatureAdapter(agent, cal['aux']['common_reference'])
    blind = blind_adapter.extract(observations[8:])
  assert blind['aux']['common_reference']['verified_observations'] == 32
  for key in ('h', 'prior_logit', 'zp', 'roots'):
    assert f._same(cal['aux']['common_reference'][key], blind['aux']['common_reference'][key])
  assert cal['h'].shape == (8, config.agent.dyn.rssm.deter)
  assert cal['zq'].shape == (8, 4, config.agent.dyn.rssm.stoch, config.agent.dyn.rssm.classes)
  assert np.any(cal['aux']['post_logit'][0, 0] != cal['aux']['post_logit'][1, 0])
  assert f.tree_digest(observations) == input_digest
  roots = f.private_roots()
  def official(obs):
    enc, dyn, _, previous = agent.model.init_policy(1)
    enc, _, tokens = agent.model.enc(enc, obs, obs['is_first'], training=False, single=True)
    _, _, feat = agent.model.dyn.observe(dyn, tokens, previous, obs['is_first'],
        training=False, single=True)
    prior = agent.model.dyn._prior(feat['deter'])
    btfeat = dict(deter=feat['deter'][:, None], logit=feat['logit'][:, None])
    signal = dt_latch.representative_signal(agent.model, btfeat, prior[:, None])
    return feat, prior, signal, agent.model.dyn._dist(feat['logit']).kl(agent.model.dyn._dist(prior))
  reference = jax.jit(nj.pure(official))
  placed_obs = internal.device_put(adapter._observation(observations[0]), agent.train_mirrored)
  for k in range(4):
    key = internal.device_put(roots[k], agent.train_mirrored)
    _, result = reference(agent.params, placed_obs, seed=key, create=False, modify=False)
    feat, prior, signal, rawkl = f.host_copy(result)
    assert f._same(cal['zq'][0, k], feat['stoch'][0])
    assert f._same(cal['aux']['prior_logit'][0, k], prior[0])
    assert f._same(cal['aux']['D'][0, k], signal['D'][0, 0])
    assert f._same(cal['aux']['K'][0, k], rawkl[0])
    samplekey = f.host_copy(jax.jit(lambda root: jax.random.split(root, 16)[1])(key))
    np.testing.assert_array_equal(samplekey, cal['aux']['sample_key'][0, k])
    def sample(logit, samplekey):
      return nn.cast(agent.model.dyn._dist(logit).sample(seed=samplekey))
    expected_p = f.host_copy(jax.jit(sample)(internal.device_put(prior, agent.train_mirrored),
        internal.device_put(samplekey, agent.train_mirrored)))
    assert f._same(expected_p[0], cal['zp'][0, k])
  for key in ('h', 'post_logit', 'prior_logit', 'D', 'K', 'qmode', 'pmode'):
    assert all(f._same(cal['aux'][key][0, 0], cal['aux'][key][0, k]) for k in range(4))
  # Host F64 oracle for the original mode Normal discrepancy; numerical only.
  aux = cal['aux']
  qm, pm, qs, ps = [aux[key][0, 0].astype(np.float64) for key in ('qmean', 'pmean', 'qstd', 'pstd')]
  qv, pv = qs**2, ps**2
  expected_d = np.mean(.25*((qv-pv)**2/(qv*pv)+(qm-pm)**2*(1/qv+1/pv)))
  np.testing.assert_allclose(aux['D'][0, 0], expected_d, atol=1e-6, rtol=1e-5)
  assert not aux['source_q_sample_exact_claimed'] and aux['learned_only']
  assert f.tree_digest(dict(params=f.host_copy(agent.params), counters=f.counters(agent))) == full_before
  for bad in ({**observations[0], 'is_first': np.bool_(False)},
      {**observations[0], 'reward': np.float32(1)},
      {**observations[0], 'image': observations[0]['image'].astype(np.float32)}):
    try: adapter.extract([bad])
    except ValueError: pass
    else: raise AssertionError('Invalid reset accepted')
  bad_anchor = {**cal['aux']['common_reference'], 'h': cal['h'][0].copy()}
  bad_anchor['h'][0] += 1
  try: f.ResetFeatureAdapter(agent, bad_anchor).extract([observations[0]])
  except ValueError as error: assert 'common' in str(error)
  else: raise AssertionError('Wrong common anchor accepted')
  with agent.n_eval_actions.lock: agent.n_eval_actions.value += 1
  try:
    try: adapter.extract([observations[0]])
    except AssertionError as error: assert 'unchanged' in str(error)
    else: raise AssertionError('Counter change accepted')
  finally:
    with agent.n_eval_actions.lock: agent.n_eval_actions.value -= 1
  original_observe = agent.model.dyn.observe
  def writes(*args, **kwargs):
    name = next(iter(nj.context())); nj.context()[name] = nj.context()[name]
    return original_observe(*args, **kwargs)
  def creates(*args, **kwargs):
    nj.context()['forbidden_reset_scratch'] = jnp.int32(0)
    return original_observe(*args, **kwargs)
  def extra_seed(*args, **kwargs):
    result = original_observe(*args, **kwargs); nj.seed(); return result
  for replacement, message in ((writes, 'state writes'), (creates, 'create=True'),
      (extra_seed, 'exactly one')):
    agent.model.dyn.observe = replacement
    try:
      rejecting = f.ResetFeatureAdapter(agent)
      try: rejecting.extract([observations[0]])
      except (AssertionError, RuntimeError) as error: assert message in str(error), str(error)
      else: raise AssertionError('Forbidden model trace accepted')
    finally: agent.model.dyn.observe = original_observe
  # Full synthetic1001 histories, two independent episodes, one batched scan.
  episodes = [fake_episode(), fake_episode()]
  for index, ep in enumerate(episodes): ep['image'][:] = observations[index]['image']
  annotation_key = f.private_root_key(0)
  with mock.patch.object(agent.model, 'pol', side_effect=AssertionError('actor annotation prohibited')):
    annotated = adapter.annotate_episode_terminal(episodes, annotation_key)
  obs, previous = f.annotation_inputs(episodes)
  def official_annotation(obs, previous):
    enc, dyn, dec, _ = agent.model.init_policy(obs['is_first'].shape[0])
    enc, enc_entry, tokens = agent.model.enc(enc, obs, obs['is_first'], training=False, single=False)
    dyn, dyn_entry, _ = agent.model.dyn.observe(dyn, tokens, previous,
        obs['is_first'], training=False, single=False)
    return enc, dyn, dec, enc_entry, dyn_entry
  official_pure = nj.pure(official_annotation)
  def oracle_annotation_readonly(params, obs, previous, key):
    _, result, _, modified, created = official_pure(params, obs, previous,
        seed=key, create=False, modify=True, track=True)
    assert not modified and not created
    return result
  call = jax.jit(oracle_annotation_readonly)
  inputs = internal.device_put((obs, previous, annotation_key), agent.train_mirrored)
  expected = call(agent.params, *inputs)
  expected = f.host_copy(expected)
  for name in ('deter', 'stoch'):
    assert f._same(annotated['entries']['dyn/'+name], expected[4][name])
    assert f._same(annotated['terminal_entry']['dyn/'+name], expected[1][name])
    promoted = annotated['terminal_entry']['dyn/'+name].astype(np.float32)
    assert f._same(promoted.astype(annotated['terminal_entry']['dyn/'+name].dtype),
        annotated['terminal_entry']['dyn/'+name])
  # The generic official sequence helper also rejects attempts hidden in the
  # sequence entry point. Short synthetic sequences avoid redundant full1001
  # work; the public complete-episode wrapper remains fixed to1001 frames.
  short_obs = {name: value[:, :3] for name, value in obs.items()}
  short_previous = {'action': previous['action'][:, :3]}
  short_inputs = internal.device_put((short_obs, short_previous, annotation_key), agent.train_mirrored)
  for replacement, message in ((writes, 'state writes'), (creates, 'create=True')):
    agent.model.dyn.observe = replacement
    try:
      rejecting_annotation = jax.jit(f.readonly_annotations(agent.model))
      try: rejecting_annotation(agent.params, *short_inputs)
      except (AssertionError, RuntimeError) as error: assert message in str(error), str(error)
      else: raise AssertionError('Annotation attempted write/create accepted')
    finally: agent.model.dyn.observe = original_observe
  assert f._same(annotated['carry'][3]['action'], np.stack([ep['action'][999] for ep in episodes]))
  assert annotated['entries']['dyn/deter'].shape[:2] == (2, 1001)
  assert annotated['aux']['actor_calls'] == 0 and annotated['aux']['environment_actions'] == 0
  assert f.tree_digest(dict(params=f.host_copy(agent.params), counters=f.counters(agent))) == full_before
  # Current model changes while the public policy copy intentionally stays old.
  stale_policy = f.tree_digest(f.host_copy(agent.policy_params))
  def equal_heads(params):
    return {name: jnp.zeros_like(value) if name.startswith(('dyn/obslogit/', 'dyn/priorlogit/'))
        else value for name, value in params.items()}
  agent.params = jax.jit(equal_heads)(agent.params)
  equal = f.ResetFeatureAdapter(agent).extract(observations[:2])
  assert np.all(equal['aux']['post_logit'] == 0) and np.all(equal['aux']['prior_logit'] == 0)
  assert equal['zq'].tobytes() == equal['zp'].tobytes()
  assert np.all(equal['D'] == 0) and np.all(equal['K'] == 0)
  assert f.tree_digest(f.host_copy(agent.policy_params)) == stale_policy
  try: adapter.extract(observations[:1])
  except AssertionError: pass
  else: raise AssertionError('Old adapter accepted replaced params')
  print(json.dumps(dict(official_post_sample_exact=True, official_prior_sample_exact=True,
      original_mode_D_exact=True, original_raw_K_exact=True, all32_common_bytes=True,
      four_h_logits_mode_signal_exact=True, new_private_q_not_source_claim=True,
      current_params_not_stale_policy=True, strict_transfer_guard=jax.config.jax_transfer_guard == 'disallow',
      full_state_and_counters_unchanged=True, attempted_writes_rejected=True,
      attempted_creation_rejected=True, extra_seed_rejected=True, counter_change_rejected=True,
      invalid_common_anchor_rejected=True, invalid_reset_rejected=True,
      official_full1001_batch_annotation_exact=True, terminal_raw999_exact=True,
      all_actual_entries_returned=True, annotation_actor_never_called=True,
      annotation_attempted_writes_rejected=True, annotation_attempted_creation_rejected=True,
      scope='Actual tiny BF16 CPU fixture only; no real source/checkpoint/physics/CUDA')))


if __name__ == '__main__':
  fixture() if sys.argv[1:] == ['--fixture'] else unittest.main()
