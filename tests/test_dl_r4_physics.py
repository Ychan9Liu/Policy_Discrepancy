"""Fresh CPU quadruped/wrappers; policy and RGB are explicit engineering fakes."""
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MUJOCO_GL', 'disable')

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_diagnostic as d
from scripts import dl_r4_physics as p


def dummy_render(self, height=64, width=64, camera_id=2, **kwargs):
  # CPU-only stub changes with actual physics time. No renderer is initialized.
  value = int(round(self.data.time * 1000)) % 256
  return np.full((height, width, 3), value, np.uint8)


def initial_carry():
  return ({}, dict(deter=[np.zeros(4, np.float32)],
      stoch=[np.eye(4, dtype=np.float32)[:3].copy()]), {},
      dict(action=[np.zeros(12, np.float32)]))


class FakePolicy:
  """No real model, production NJ key derivation or policy oracle is tested."""
  def __init__(self):
    self.normal_observations = []
    self.current_calls = 0
    self.first_calls = []

  def current(self, carry, obs, root_key):
    self.current_calls += 1
    q, prior = copy.deepcopy(carry), copy.deepcopy(carry)
    h = carry[1]['deter'][0] + np.float32(1)
    qz = np.eye(4, dtype=np.float32)[:3].copy()
    pz = np.roll(qz, 1, axis=1).copy()
    signal = float(obs['image'][0, 0, 0]) / 10000
    qraw = np.linspace(-1.5, 1.5, 12, dtype=np.float32) + np.float32(signal)
    praw = -qraw.copy()
    for after, z, raw in ((q, qz, qraw), (prior, pz, praw)):
      after[1]['deter'][0] = h.copy()
      after[1]['stoch'][0] = z.copy()
      after[3]['action'][0] = raw.copy()
    return dict(q_after=q, p_after=prior, q_raw=qraw, p_raw=praw,
        h=h, qlogits=qz.copy(), plogits=pz.copy(), qsample=qz, psample=pz,
        root_key=root_key.copy(), latent_key=root_key[::-1].copy(),
        actor_key=(root_key ^ np.uint32(79)).copy(),
        original_nj_seed_calls=np.array(2, np.int32),
        before_carry_sha256=p.policy_tree_digest(carry),
        observation_sha256=p.policy_tree_digest(obs))

  def first(self, bundle, *, arm):
    self.first_calls.append(arm)
    action, memory = arm
    after = copy.deepcopy(bundle[memory + '_after'])
    raw = bundle[action + '_raw'].copy()
    after[3]['action'][0] = raw.copy()
    proof = dict(root_key=bundle['root_key'].copy(),
        latent_key=bundle['latent_key'].copy(), actor_key=bundle['actor_key'].copy(),
        h=bundle['h'].copy(), selected_logit=bundle[memory + 'logits'].copy(),
        latent=bundle[memory + 'sample'].copy(), action_logit=bundle[action + 'logits'].copy(),
        action_latent=bundle[action + 'sample'].copy(), raw=raw.copy(),
        rng_consumption=np.array(2, np.int32),
        memory_source=np.array(memory == 'p', np.int8),
        action_source=np.array(action == 'p', np.int8))
    return after, raw, proof

  def normal_step(self, carry, obs, root_key):
    self.normal_observations.append(p.source.tree_digest(obs))
    bundle = self.current(carry, obs, root_key)
    return self.first(bundle, arm='qq')


class R4PhysicsTest(unittest.TestCase):
  def setUp(self):
    from dm_control.mujoco.engine import Physics
    patch = mock.patch.object(Physics, 'render', dummy_render)
    patch.start()
    self.addCleanup(patch.stop)
    self.adapter = p.feedback.FeedbackEnvironment(d.make_dmc('dmc_quadruped_walk', 77))
    self.addCleanup(self.adapter.close)
    self.obs = self.adapter.reset()
    for _ in range(64):
      self.obs, _ = self.adapter.step(np.zeros(12, np.float32))
    self.snapshot = self.adapter.snapshot()
    self.visual = p.visual_model_identity(self.adapter)
    self.keys = np.arange(200, dtype=np.uint32).reshape(100, 2)
    self.policy = FakePolicy()
    self.carry = initial_carry()

  def branch(self, arm='qq', policy=None):
    policy = policy or self.policy
    bundle = policy.current(self.carry, self.obs, self.keys[0])
    return p.run_branch(policy, self.adapter, self.snapshot, self.carry,
        self.keys, arm=arm, bundle=bundle, initial_obs=self.obs,
        visual_identity=self.visual)

  def test_three_real_cpu_branches_null_and_both_separate_replays(self):
    result = p.run_state_pair(self.policy, self.adapter, self.snapshot,
        self.carry, self.keys, initial_obs=self.obs, visual_identity=self.visual, pair=0)
    self.assertEqual(result['order'], list(p.ARM_ORDERS[0]))
    self.assertTrue(result['null_performed'])
    self.assertTrue(np.all(result['null_reward_difference'] == 0))
    self.assertTrue(result['pairing']['qq_pq_current_memory_byte_exact'])
    self.assertTrue(result['pairing']['pq_pp_first_raw_applied_reward_snapshot_byte_exact'])
    self.assertFalse(result['pairing']['future_rendered_pixels_paired'])
    self.assertFalse(result['pairing']['interaction_estimated'])
    self.assertEqual(set(result['records']), set(p.ARMS))
    for arm, record in result['records'].items():
      self.assertEqual(p.validate_record(record), 100)
      self.assertEqual(record['arm'], arm)
      self.assertEqual(record['future_normal_policy_calls'], 99)
      self.assertEqual(record['first_assembly_model_calls'], 0)
      self.assertEqual(len(record['observations']), 101)
      self.assertEqual(len(record['snapshots']), 101)
      self.assertEqual(len(record['after_carries']), 100)
      self.assertGreater(len({p.source.tree_digest(o) for o in record['observations']}), 1)
      policy = p.replay_cached_policy(self.policy, record)
      self.assertTrue(policy['cached_own_rgb'])
      self.assertFalse(policy['renderer_replayed'])
      self.assertFalse(policy['physics_replayed'])
      with mock.patch.object(self.adapter.suite.physics, 'render',
          side_effect=AssertionError('stored physics replay must not render')):
        physical = p.verify_physics_replay(self.adapter, record)
      self.assertEqual(physical['physics_steps'], 100)
      self.assertTrue(physical['full_snapshot_byte_exact'])
      self.assertFalse(physical['renderer_replayed'])
      self.assertFalse(physical['policy_replayed'])
    qq, pq, pp = (result['records'][a] for a in p.ARMS)
    self.assertNotEqual(p.source.tree_digest(qq['after_carries'][0][3]),
        p.source.tree_digest(pq['after_carries'][0][3]))
    self.assertEqual(p.source.tree_digest(pq['after_carries'][0][3]),
        p.source.tree_digest(pp['after_carries'][0][3]))
    self.assertTrue(np.any(np.abs(qq['raw']) > 1))
    self.assertFalse(np.array_equal(qq['raw'], qq['applied']))

  def test_orders_all_six_permutations_and_null_only_pair_zero(self):
    orders = [p.branch_order(i) for i in range(16)]
    self.assertEqual(set(orders), set(p.ARM_ORDERS))
    self.assertEqual(orders[:6], list(p.ARM_ORDERS))
    counts = [orders.count(order) for order in p.ARM_ORDERS]
    self.assertEqual(max(counts) - min(counts), 1)
    for invalid in (-1, 16, True, 1.5):
      with self.assertRaises(ValueError):
        p.branch_order(invalid)
    result = p.run_state_pair(self.policy, self.adapter, self.snapshot,
        self.carry, self.keys, initial_obs=self.obs, visual_identity=self.visual, pair=1)
    self.assertEqual(result['order'], list(p.ARM_ORDERS[1]))
    self.assertIsNone(result['null_record'])
    self.assertIsNone(result['null_reward_difference'])

  def test_current_rgb_shared_without_new_render_and_visual_guard(self):
    original = copy.deepcopy(self.obs)
    original['image'][:] = 123
    digest = p.source.tree_digest(original)
    with mock.patch.object(self.adapter.suite.physics, 'render',
        side_effect=AssertionError('current shared RGB must not be rerendered')):
      observed, receipt = p.prepare_initial_observation(self.adapter,
          self.snapshot, original, visual_identity=self.visual)
    self.assertEqual(p.source.tree_digest(observed), digest)
    self.assertEqual(receipt['fresh_initial_render_count'], 0)
    self.assertTrue(receipt['shared_actual_source_rgb'])
    self.assertFalse(receipt['renderer_byte_exact_established'])
    observed['image'][0, 0, 0] ^= np.uint8(1)
    self.assertEqual(p.source.tree_digest(original), digest)
    for change in ('hash', 'adapter', 'falseclaim'):
      broken = copy.deepcopy(self.visual)
      if change == 'hash': broken['mjb_sha256'] = '0' * 64
      elif change == 'adapter': broken['adapter_identity'] += 1
      else: broken['renderer_byte_exact_established'] = True
      with self.assertRaises(ValueError):
        p.prepare_initial_observation(self.adapter, self.snapshot, original,
            visual_identity=broken)
    for change in ('reset', 'position', 'inactive'):
      bad = copy.deepcopy(self.snapshot)
      obs = copy.deepcopy(original)
      if change == 'reset': obs['is_first'][...] = True
      elif change == 'position': bad['suite']['step_count'] = 0
      else: bad['from_dm_done'] = True
      with self.assertRaises(ValueError):
        p.prepare_initial_observation(self.adapter, bad, obs, visual_identity=self.visual)
    wrong_obs = copy.deepcopy(self.obs)
    wrong_obs['image'][0, 0, 0] ^= np.uint8(1)
    bundle = self.policy.current(self.carry, wrong_obs, self.keys[0])
    with self.assertRaisesRegex(ValueError, 'different history or observation'):
      p.run_branch(self.policy, self.adapter, self.snapshot, self.carry, self.keys,
          arm='qq', bundle=bundle, initial_obs=self.obs, visual_identity=self.visual)

  def test_absorbing_true_terminal_has_no_fabricated_future(self):
    cls = type(self.adapter.suite._task)
    with mock.patch.object(cls, 'get_termination', return_value=0.):
      record = self.branch('pp')
      self.assertEqual(record['end_type'], 'true_terminal')
      self.assertEqual(record['executed_steps'], 1)
      self.assertEqual(record['future_normal_policy_calls'], 0)
      self.assertEqual(len(record['observations']), 2)
      self.assertTrue(np.all(record['reward_f64'][1:] == 0))
      self.assertTrue(np.all(~record['action_executed'][1:]))
      self.assertEqual(p.verify_physics_replay(self.adapter, record)['absorbing_padding_steps'], 99)
      self.assertEqual(p.replay_cached_policy(self.policy, record)['policy_steps'], 1)
      damaged = copy.deepcopy(record)
      damaged['observations'].append(copy.deepcopy(record['observations'][-1]))
      with self.assertRaises(ValueError): p.validate_record(damaged)
      damaged = copy.deepcopy(record)
      damaged['reward_f64'][-1] = .1
      with self.assertRaises(ValueError): p.validate_record(damaged)
    with mock.patch.object(cls, 'get_termination', return_value=1.):
      with self.assertRaisesRegex(ValueError, 'Early or unexpected time limit'):
        self.branch()

  def test_independent_r4_proof_schema_and_crossed_first_carry(self):
    record = self.branch('pq')
    self.assertEqual(int(record['proofs'][0]['memory_source']), 0)
    self.assertEqual(int(record['proofs'][0]['action_source']), 1)
    self.assertTrue(np.array_equal(record['proofs'][0]['latent'], record['current_bundle']['qsample']))
    self.assertTrue(np.array_equal(record['proofs'][0]['action_latent'], record['current_bundle']['psample']))
    for damage in ('old_schema', 'source', 'action_latent', 'prevact', 'first_model_call'):
      broken = copy.deepcopy(record)
      if damage == 'old_schema': broken['proofs'][0].pop('action_latent')
      elif damage == 'source': broken['proofs'][0]['memory_source'][...] = 1
      elif damage == 'action_latent': broken['proofs'][0]['action_latent'][0, 0] += 1
      elif damage == 'prevact':
        broken['after_carries'][0][3]['action'][0] = broken['applied'][0].astype(np.float32)
      else: broken['first_assembly_model_calls'] = 1
      with self.assertRaises((ValueError, AssertionError)):
        p.validate_record(broken)

  def test_one_ulp_f64_and_sub_tolerance_fullstate_tamper_rejected(self):
    record = self.branch()
    broken = copy.deepcopy(record)
    value = broken['reward_f64'][0]
    broken['reward_f64'][0] = np.nextafter(value, np.inf)
    self.assertEqual(np.float32(value).tobytes(), np.float32(broken['reward_f64'][0]).tobytes())
    with self.assertRaisesRegex(ValueError, 'F64/F32 reward'):
      p.verify_physics_replay(self.adapter, broken)
    broken = copy.deepcopy(record)
    value = broken['snapshots'][1]['suite']['integration'][0]
    broken['snapshots'][1]['suite']['integration'][0] = np.nextafter(value, np.inf)
    with self.assertRaisesRegex(ValueError, 'integration snapshot differs byte-exactly'):
      p.verify_physics_replay(self.adapter, broken)
    broken = copy.deepcopy(record)
    broken['applied'][0, 0] += .001
    with self.assertRaisesRegex(ValueError, 'applied action'):
      p.verify_physics_replay(self.adapter, broken)

  def test_nonfinite_mutation_wrong_keys_and_deadline_are_fail_closed(self):
    class MutatingPolicy(FakePolicy):
      def normal_step(self, carry, obs, key):
        obs['image'][0, 0, 0] ^= np.uint8(1)
        return super().normal_step(carry, obs, key)
    with self.assertRaisesRegex(ValueError, 'mutated'):
      self.branch(policy=MutatingPolicy())
    class WrongKeyPolicy(FakePolicy):
      def first(self, bundle, *, arm):
        result = super().first(bundle, arm=arm)
        result[2]['root_key'][0] ^= np.uint32(1)
        return result
    with self.assertRaisesRegex(ValueError, 'different private root'):
      self.branch(policy=WrongKeyPolicy())
    record = self.branch()
    for damage in ('reward', 'state', 'proof', 'rgb'):
      broken = copy.deepcopy(record)
      if damage == 'reward': broken['reward_f64'][0] = np.nan
      elif damage == 'state': broken['snapshots'][0]['suite']['integration'][0] = np.inf
      elif damage == 'proof': broken['proofs'][0]['h'][0] = np.nan
      else: broken['observations'][0]['image'] = broken['observations'][0]['image'].astype(np.float32)
      with self.assertRaises((ValueError, AssertionError)):
        p.validate_record(broken)
    called = []
    def expired():
      called.append(True)
      raise TimeoutError('test enclosing owner deadline')
    with self.assertRaises(TimeoutError):
      p.run_state_pair(self.policy, self.adapter, self.snapshot,
          self.carry, self.keys, initial_obs=self.obs,
          visual_identity=self.visual, pair=0, check=expired)
    self.assertEqual(called, [True])

  def test_cached_policy_proof_or_rgb_and_pair_action_tamper_rejected(self):
    result = p.run_state_pair(self.policy, self.adapter, self.snapshot,
        self.carry, self.keys, initial_obs=self.obs, visual_identity=self.visual, pair=1)
    record = result['records']['qq']
    broken = copy.deepcopy(record)
    # Current RGB is bound separately; later RGB must be consumed by the real
    # cached replay rather than substituted with the source image.
    broken['observations'][2]['image'][...] = 77
    with self.assertRaisesRegex(ValueError, 'Cached policy raw'):
      p.replay_cached_policy(self.policy, broken)
    class WrongProofPolicy(FakePolicy):
      def normal_step(self, *args):
        after, raw, proof = super().normal_step(*args)
        proof['actor_key'][0] ^= np.uint32(1)
        return after, raw, proof
    with self.assertRaisesRegex(ValueError, 'numeric policy receipt'):
      p.replay_cached_policy(WrongProofPolicy(), record)
    wrong_pair = dict(result['records'])
    wrong_pair['pp'] = result['records']['qq']
    with self.assertRaisesRegex(ValueError, 'Arm label mismatch'):
      p.validate_pair(wrong_pair)


if __name__ == '__main__':
  unittest.main()
