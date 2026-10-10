"""DL-code-r1: independent controls, episode statistics, masks and alignment.

These are engineering fixtures. They do not validate any real checkpoint,
simulator restoration, task cue, or scientific hypothesis.
"""

import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import os
import tempfile
import json

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/dl_diagnostic.py'
SPEC = importlib.util.spec_from_file_location('dl_diagnostic', SOURCE)
dl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dl)


class DLDiagnosticTest(unittest.TestCase):

  def test_all_geometry_and_antialias_edges_preserved(self):
    seg = np.full((9, 9, 2), -1, np.int32)
    seg[5:, :, :] = [0, 5]  # Ground includes contact shadows.
    seg[2:5, 3:6, :] = [1, 5]
    seg[0, 8, :] = [99, 8]  # Unknown non-geometry object is kept.
    preserve, body = dl.segmentation_masks(seg, np.array([0, 1]))
    self.assertTrue(preserve[5:, :].all())
    self.assertTrue(preserve[1:6, 2:7].all())
    self.assertTrue(preserve[0, 8])
    self.assertFalse(body[7:].any())
    image = np.arange(9 * 9 * 3, dtype=np.uint8).reshape((9, 9, 3))
    transformed = dl.intervention_images(image, preserve, body, 5)
    for variant in (1, 2):
      np.testing.assert_array_equal(transformed[variant][preserve], image[preserve])
      self.assertTrue(np.any(transformed[variant][~preserve] != image[~preserve]))
    np.testing.assert_array_equal(transformed[3][~body], image[~body])
    self.assertTrue(np.all(transformed[3][body] == 127))

  def test_episode_split_is_frozen_and_independent(self):
    plan = dl.episode_plan()
    self.assertEqual(len(plan), 32)
    self.assertEqual(sum(x['split'] == 'calibration' for x in plan), 8)
    self.assertEqual(len({x['seed'] for x in plan}), 32)
    self.assertEqual(plan, dl.episode_plan())
    self.assertNotEqual(plan, dl.episode_plan(seed=3))
    with self.assertRaises(ValueError):
      dl.episode_plan(count=8, calibration=8)

  def test_sampling_and_reward_alignment_have_full_future_window(self):
    positions = dl.sample_positions(1001, maximum=32, horizon=10)
    self.assertEqual(len(positions), 32)
    self.assertTrue(np.all((positions >= 1) & (positions + 10 < 1001)))
    self.assertFalse(len(dl.sample_positions(10, horizon=10)))
    expected = dl.fixed_actions((12,))
    self.assertEqual(expected.shape, (5, 12))
    np.testing.assert_array_equal(expected[0], 0)
    np.testing.assert_array_equal(expected[3], -expected[4])

  def test_true_consequence_uses_candidate_then_common_recorded_actions(self):
    class Env:
      def __init__(self):
        self.value = 0
        self.actions = []
      def step(self, action):
        self.actions.append(action.copy())
        self.value += float(action[0])
        return types.SimpleNamespace(reward=self.value, last=lambda: False)
    env = Env()
    def restore(environment, snapshot):
      environment.value = snapshot
      environment.actions = []
    with patch.object(dl, 'restore_physics', restore):
      reward = dl.true_consequence(env, 10, np.array([2.]),
          np.array([[3.], [4.], [99.]]), horizon=3)
      np.testing.assert_array_equal(reward, [12, 15, 19])
      np.testing.assert_array_equal(np.asarray(env.actions).reshape(-1), [2, 3, 4])
      np.testing.assert_array_equal(reward, dl.true_consequence(env, 10,
          np.array([2.]), np.array([[3.], [4.]]), horizon=3))
      with self.assertRaises(ValueError):
        dl.true_consequence(env, 0, np.array([1.]), [], horizon=2)

  def test_twohot_exact_bins_endpoints_and_positive_part(self):
    bins = np.array([-3, 0, 2], np.float32)
    reward = np.array([-10, -1.5, 0, 1, 20], np.float32)
    logits = np.zeros((5, 3))
    result = dl.compatibility_numpy(logits, logits, reward, bins)
    expected = [[1, 0, 0], [.5, .5, 0], [0, 1, 0], [0, .5, .5], [0, 0, 1]]
    np.testing.assert_allclose(result['target'], expected)
    np.testing.assert_allclose(result['ell_q'], np.log(3))
    np.testing.assert_array_equal(result['e'], 0)
    changed = logits.copy()
    changed[:, 1] = 2
    result = dl.compatibility_numpy(changed, logits, reward, bins)
    self.assertGreater(result['signed_support'][2], 0)
    self.assertLess(result['signed_support'][0], 0)
    self.assertEqual(result['e'][0], 0)
    with self.assertRaises(ValueError):
      dl.compatibility_numpy(logits, logits, reward, bins[::-1])

  def test_episode_mean_weights_not_frame_mean(self):
    values, episodes = np.array([0, 1, 1, 1, 1]), np.array([0, 1, 1, 1, 1])
    interval = dl.cluster_interval(values, episodes, draws=200)
    self.assertEqual(interval['point'], .5)
    self.assertEqual(interval['episodes'], 2)
    self.assertLessEqual(interval['lower95'], interval['point'])
    self.assertGreaterEqual(interval['upper95'], interval['point'])

  def test_auroc_ties_and_clustered_missing_class(self):
    self.assertEqual(dl.auc([True, False], [1, 0]), 1)
    self.assertEqual(dl.auc([True, False], [0, 1]), 0)
    self.assertEqual(dl.auc([True, False], [1, 1]), .5)
    self.assertIsNone(dl.auc([True, True], [1, 2]))
    labels = np.array([True, False])
    values = np.array([1, 0])
    interval = dl.cluster_interval(values, np.array([0, 1]),
        statistic=lambda idx: dl.auc(labels[idx], values[idx]), draws=200)
    # Many episode draws contain only one class; do not pretend a valid CI.
    self.assertEqual(interval['point'], 1)
    self.assertIsNone(interval['lower95'])

  def test_label_shuffle_never_same_episode_or_cross_split(self):
    episodes = np.array([0, 0, 1, 1, 1, 2])
    rewards = episodes * 10 + np.arange(len(episodes))
    shuffled, donors = dl.different_episode_labels(rewards, episodes)
    self.assertTrue(np.all(donors != episodes))
    self.assertEqual(set(donors), set(episodes))
    for destination, donor, reward in zip(episodes, donors, shuffled):
      self.assertIn(reward, rewards[episodes == donor])
    with self.assertRaises(ValueError):
      dl.different_episode_labels([1, 2], [0, 0])

  def test_numpy_target_matches_repository_twohot(self):
    import jax.numpy as jnp
    from embodied.jax.outs import TwoHot
    bins = np.array([-3, -.2, 0, .5, 2], np.float32)
    reward = np.array([-10, -.3, -.2, -.1, 0, .3, .5, 20], np.float32)
    rng = np.random.default_rng(7)
    q = rng.normal(size=(len(reward), len(bins))).astype(np.float32)
    p = rng.normal(size=q.shape).astype(np.float32)
    expected = TwoHot(jnp.asarray(q), jnp.asarray(bins))
    result = dl.compatibility_numpy(q, p, reward, bins)
    np.testing.assert_allclose(result['target'], np.asarray(expected.target(
        jnp.asarray(reward))), atol=1e-7)
    np.testing.assert_allclose(result['ell_q'], np.asarray(expected.loss(
        jnp.asarray(reward))), atol=1e-6)

  def test_analysis_reports_insufficient_coverage_without_training_authorization(self):
    with tempfile.TemporaryDirectory() as temporary:
      source = Path(temporary)
      plan = dict(free_nats=1, rho=.5)
      dl.write_json(source / 'score_plan.json', plan)
      artifacts = []
      for entry in dl.episode_plan():
        episode = entry['episode']
        values = {key: np.zeros((1, 4), np.float32) for key in (
            'e', 'signed_support', 'loss_gap', 'mode_perturb_e', 'mode_perturb_C',
            'mode_perturb_q_K', 'mode_perturb_p_K', 'D_std_component')}
        values.update(D=np.ones((1, 4)), K=np.ones((1, 4)) * 2,
            m=np.ones((1, 4)), v=np.ones((1, 4)), f_D=np.ones((1, 4)) * .5,
            D_mean_component=np.ones((1, 4)), q_margin_min=np.ones((1, 4)),
            mode_perturb_applied=np.zeros((1, 4)),
            fixed_e=np.zeros((1, 4, 5)), fixed_signed_support=np.zeros((1, 4, 5)),
            true_q_horizon=np.asarray([[.02 if episode % 2 else -.02, 0, 0, 0]]),
            true_p_horizon=np.zeros(1), next_reward=np.zeros(1),
            physics_repeat_error=np.zeros(1), background_frac=np.ones(1) * .5,
            body_frac=np.ones(1) * .1, raw_logits_q=np.zeros((1, 4, 5)),
            raw_logits_p=np.zeros((1, 4, 5)), bins=np.linspace(-2, 2, 5, dtype=np.float32))
        path = source / f'episode_{episode:03}.npz'
        np.savez_compressed(path, **values)
        artifacts.append(dict(episode=episode, split=entry['split'], positions=1,
            path=path.name, sha256=dl.sha256(path)))
      dl.write_json(source / 'score_complete.json', dict(artifacts=artifacts,
          plan_sha256=dl.sha256(source / 'score_plan.json')))
      output = source / 'analysis.json'
      dl.analyze(types.SimpleNamespace(scores=source, output=output, bootstrap=50, seed=7))
      result = json.loads(output.read_text())
      self.assertEqual(result['screening']['signal'], 'inconclusive')
      self.assertEqual(result['screening']['cue_deprivation'], 'inconclusive')
      self.assertFalse(result['screening']['proceed_short_training'])
      self.assertFalse(result['screening']['eligible_for_short_training_review'])
      self.assertIn('C', result['signal_auc'])
      self.assertEqual(result['calibration_only']['kappa'], 0)
      self.assertIsNone(result['calibration_only']['reward_only_lambda'])
      self.assertEqual(result['offline_strength_controls']['W']['status'], 'uncalibrated')


@unittest.skipUnless(os.environ.get('DL_FULL_AGENT_FIXTURE') == '1',
    'Enable separate tiny full-agent engineering fixture explicitly')
class DLFrozenScorerFixture(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    import elements
    from ruamel.yaml import YAML
    from dreamerv3.agent import Agent
    source = SOURCE.parents[1] / 'dreamerv3/configs.yaml'
    presets = YAML(typ='safe').load(source.read_text())
    config = elements.Config(presets['defaults']).update(presets['debug'])
    config = elements.Config({**config.flat, 'batch_size': 1,
        'batch_length': 3, 'replay_context': 0, 'jax.compute_dtype': 'float32',
        'jax.precompile': False, 'jax.profiler': False,
        'agent.imag_length': 2, 'agent.imag_last': 2})
    spaces = dict(image=elements.Space(np.uint8, (64, 64, 3)),
        reward=elements.Space(np.float32), is_first=elements.Space(bool),
        is_last=elements.Space(bool), is_terminal=elements.Space(bool))
    acts = {'action': elements.Space(np.float32, (2,))}
    cls.agent = Agent(spaces, acts, elements.Config(**config.agent,
        logdir='analysis/outputs/dl-prevalidation/tiny-scorer', seed=7,
        jax=config.jax, batch_size=1, batch_length=3, replay_context=0,
        report_length=2, replica=0, replicas=1))
    cls.scorer = dl.FrozenScorer(cls.agent, alpha=20, rho=.5)

  def test_shared_h_uniform_head_and_frozen_params(self):
    import jax
    from embodied.run.protocol_v1 import parameter_digest
    before = parameter_digest(self.agent)
    image = np.arange(64 * 64 * 3, dtype=np.uint8).reshape((64, 64, 3))
    preserve = np.zeros((64, 64), bool)
    preserve[32:] = True
    body = np.zeros((64, 64), bool)
    body[20:30, 20:30] = True
    images = dl.intervention_images(image, preserve, body, 7)
    carry = self.agent.init_train(1)[:2]
    _, result = self.scorer.call(self.scorer.diagnose, carry, images,
        np.zeros(2, np.float32), np.bool_(True), np.ones(2, np.float32) * .1,
        np.float32(.2), dl.fixed_actions((2,)), np.zeros(5, np.float32), seed=11)
    with jax._src.config.explicit_device_get_scope():
      arrays = jax.tree.map(np.asarray, result)
    np.testing.assert_array_equal(arrays['e'], 0)
    np.testing.assert_array_equal(arrays['v'], 1)
    np.testing.assert_array_equal(arrays['p_action'], np.repeat(
        arrays['p_action'][:1], len(dl.VARIANTS), 0))
    self.assertEqual(before, parameter_digest(self.agent))

  def test_diagnostic_selection_does_not_change_clean_history(self):
    import jax
    carry_a = self.agent.init_train(1)[:2]
    carry_b = self.agent.init_train(1)[:2]
    for t in range(4):
      image = np.full((64, 64, 3), t * 40, np.uint8)
      previous = np.ones(2, np.float32) * .1
      before_b = carry_b
      carry_a = self.scorer.call(self.scorer.advance, carry_a, image,
          previous, np.bool_(t == 0), seed=100 + t)
      carry_b = self.scorer.call(self.scorer.advance, carry_b, image,
          previous, np.bool_(t == 0), seed=100 + t)
      if t % 2:
        self.scorer.call(self.scorer.diagnose, before_b, np.repeat(image[None], 4, 0),
            previous, np.bool_(False), previous, np.float32(.2),
            dl.fixed_actions((2,)), np.zeros(5, np.float32), seed=100 + t)
    with jax._src.config.explicit_device_get_scope():
      for a, b in zip(jax.tree.leaves(carry_a), jax.tree.leaves(carry_b)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


if __name__ == '__main__':
  unittest.main()
