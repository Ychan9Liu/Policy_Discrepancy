"""CPU engineering fixtures; no real source/checkpoint/GPU/scientific result."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest

import numpy as np

from scripts import dl_rep_observation_guard as guard
from scripts import dl_reset_learning as original
from scripts import dl_reset_all_rep as full
from tests import test_dl_reset_learning as fixtures


WHOLE = {'dl_allrep/active_count': np.int32(64),
    'dl_allrep/excess_sum': np.float32(10.5),
    'dl_allrep/position_count': np.int32(1024),
    'dl_allrep/nonreset_active_count': np.int32(56),
    'dl_allrep/nonreset_excess_sum': np.float32(7.)}


class HistoricalA(fixtures.FixtureAgent):
  def take_train_result(self):
    outs, metrics = super().take_train_result()
    return outs, {**metrics, **WHOLE}


class GuardAgent(fixtures.FixtureAgent):
  def __init__(self, mode, change=None):
    super().__init__()
    self.mode, self.change = mode, change

  def train(self, carry, prepared):
    super().train(carry, prepared)
    if self.change == 'state_bit' and self.train_calls == 3:
      value = self.params['enc/kernel']
      value[0] = np.nextafter(value[0], np.float32(np.inf))
    if callable(self.change): self.change(self)
    return ({'carry': np.array([self.train_calls], np.float32)}, {}, {})

  def take_train_result(self):
    _, metrics = super().take_train_result()
    if self.mode == 'Bwhole': metrics.update(WHOLE)
    if self.drain_calls == 3:
      if self.change == 'metric_bit':
        metrics['opt/grad_norm'] = np.nextafter(metrics['opt/grad_norm'], np.float32(np.inf))
      elif self.change == 'metric_dtype': metrics['opt/grad_norm'] = np.float64(metrics['opt/grad_norm'])
      elif self.change == 'metric_shape': metrics['opt/grad_norm'] = np.array([metrics['opt/grad_norm']])
      elif self.change == 'nan': metrics['opt/grad_norm'] = np.float32(np.nan)
      elif self.change == 'bad_counter_dtype': metrics['opt/updates'] = np.float32(metrics['opt/updates'])
    return {'replay': {'fixture': np.array(self.train_calls, np.int32)}}, metrics


class ObservationGuardContracts(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.cal, cls.pins, cls.contexts = fixtures.inputs()
    cls.checkpoint = fixtures.checkpoint()
    with original.rssm_branch('B', fixtures.FixtureDyn):
      b = original.run_branch(fixtures.FixtureAgent(), cls.checkpoint,
          cls.cal, cls.pins, cls.contexts, 'B')
    with full.allrep_branch(fixtures.FixtureDyn):
      a = full.run_a(HistoricalA(), cls.checkpoint, cls.cal, cls.pins,
          cls.contexts, reference_b=b)
    # Simulate actual historical JSON, not an invented typed reference.
    cls.b, cls.a = guard.json_projection(b), guard.json_projection(a)

  def run_guard(self, mode, change=None, *, b=None, a=None, cal=None, check=lambda: None):
    with guard.observation_branch(mode, fixtures.FixtureDyn):
      agent = GuardAgent(mode, change)
      result = guard.run_guard(agent, self.checkpoint, self.cal if cal is None else cal,
          self.pins, self.contexts, mode, reference_b=self.b if b is None else b,
          reference_a=self.a if a is None else a, check=check)
    return result, agent

  def test_both_exact128_historical_endpoints_and_authenticated_metric_anchors(self):
    initial_pin = guard.tree_digest(self.checkpoint)
    for mode in guard.MODES:
      with self.subTest(mode=mode):
        result, agent = self.run_guard(mode)
        self.assertEqual((agent.init_calls, agent.train_calls, agent.drain_calls), (128, 128, 128))
        self.assertEqual(result['updates'], 128)
        self.assertTrue(result['equivalent'])
        self.assertTrue(result['common_metrics_comparison']['passed'])
        self.assertTrue(result['common_metrics_comparison']['typed_reference_available'])
        self.assertTrue(result['endpoint_comparison']['full_state_typed_exact'])
        self.assertEqual(result['final_state_digest'], self.b['final_state_digest'])
        self.assertEqual(len(result['receipts']), 128)
        self.assertTrue(all(row['common_metrics_typed_exact'] for row in result['receipts']))
        self.assertTrue(all(len(row['full_state_digest']) == 64 and len(row['carry_digest']) == 64 and
            len(row['outs_digest']) == 64 for row in result['receipts']))
        self.assertFalse(result['historical_per_update_fullstate_carry_outs_equivalence_claimed'])
        self.assertFalse(result['source_authentication_performed'])
        self.assertFalse(result['real_size50m_cuda_equivalence_established'])
    self.assertEqual(guard.tree_digest(self.checkpoint), initial_pin)

  def test_metric_bit_dtype_shape_and_endpoint_bit_keep_all128_difference_evidence(self):
    for change in ('metric_bit', 'metric_dtype', 'metric_shape', 'state_bit'):
      with self.subTest(change=change):
        result, agent = self.run_guard('Bwhole', change)
        self.assertEqual(agent.train_calls, 128)
        self.assertFalse(result['equivalent'])
        self.assertIsNotNone(result['first_difference'])
        self.assertEqual(result['equivalence_verdict'], 'not_equivalent')
        self.assertEqual(result['common_metrics_comparison']['passed'], change == 'state_bit')
        self.assertEqual(result['endpoint_comparison']['full_state_typed_exact'], change != 'state_bit')

  def test_a_json_exact_can_be_weaker_without_invented_typed_history(self):
    result, agent = self.run_guard('Amin', 'metric_dtype')
    self.assertEqual(agent.train_calls, 128)
    self.assertTrue(result['endpoint_comparison']['full_state_typed_exact'])
    self.assertTrue(result['common_metrics_comparison']['passed'])
    self.assertFalse(result['common_metrics_comparison']['typed_reference_available'])
    row = result['receipts'][2]
    self.assertTrue(row['common_metrics_json_exact'])
    self.assertIsNone(row['common_metrics_typed_exact'])
    self.assertFalse(row['reference_common_typed_anchor']['available'])
    self.assertIn('hash_differs', row['reference_common_typed_anchor']['reason'])
    changed, _ = self.run_guard('Amin', 'metric_bit')
    self.assertFalse(changed['equivalent'])
    self.assertFalse(changed['common_metrics_comparison']['passed'])

  def test_reference_key_or_source_mismatch_is_invalid_input_not_graph_difference(self):
    changed = copy.deepcopy(self.b)
    changed['receipts'][3]['train_seed'][0] ^= 1
    with self.assertRaisesRegex(ValueError, 'normal key contract differs'):
      self.run_guard('Bwhole', b=changed)
    changed = copy.deepcopy(self.a)
    changed['initial_state_digest'] = 'a'*64
    with self.assertRaisesRegex(ValueError, 'original input contract differs'):
      self.run_guard('Amin', a=changed)

  def test_nonfinite_badcounter_mutation_and_deadline_raise_contract_failures(self):
    for issue in ('nan', 'bad_counter_dtype'):
      with self.subTest(issue=issue):
        with self.assertRaises(ValueError): self.run_guard('Bwhole', issue)
    cal = copy.deepcopy(self.cal)
    def mutate(agent):
      if agent.train_calls == 1: cal[9]['reward'][2] += np.float32(1)
    with self.assertRaisesRegex(ValueError, 'calibration arrays mutated'):
      self.run_guard('Bwhole', mutate, cal=cal)
    count = [0]
    def check():
      count[0] += 1
      if count[0] == 4: raise TimeoutError('fixture owned deadline')
    with self.assertRaises(TimeoutError): self.run_guard('Amin', check=check)

  def test_typed_shape_dtype_signedzero_and_json_exact_difference_oracles(self):
    self.assertEqual(guard.typed_difference(np.zeros(1, np.float32), np.zeros(1, np.float64))['kind'], 'dtype')
    self.assertEqual(guard.typed_difference(np.zeros(1, np.float32), np.zeros((1, 1), np.float32))['kind'], 'shape')
    self.assertEqual(guard.typed_difference(np.array(-0., np.float32), np.array(0., np.float32))['kind'], 'array_bytes')
    self.assertEqual(guard.json_difference(-0., 0.)['kind'], 'json_float_bytes')
    self.assertIsNone(guard.json_difference(np.array([.25], np.float32), [.25]))
    with self.assertRaises(ValueError): guard.typed_difference(np.array(np.inf), np.array(1.))
    common = {'metric': np.array(.25, np.float32)}
    full_metrics = {**common, **{k: np.asarray(v) for k, v in WHOLE.items()}}
    old = guard.json_projection(full_metrics)
    accepted = guard._restore_a_anchor(common, old, guard.tree_digest(full_metrics))
    self.assertTrue(accepted['available'])
    invalid = copy.deepcopy(old); invalid['dl_allrep/active_count'] = 1e40
    self.assertFalse(guard._restore_a_anchor(common, invalid, guard.tree_digest(full_metrics))['available'])
    invalid = copy.deepcopy(old); invalid['dl_allrep/excess_sum'] = [.25]
    self.assertFalse(guard._restore_a_anchor(common, invalid, guard.tree_digest(full_metrics))['available'])
    # Synthetic official-output schema, without constructing any real Agent.
    import ml_dtypes
    proxy = type('OfficialOutputSchemaProxy', (), {'__module__': 'embodied.jax.agent'})()
    raw = {'action': np.zeros((16, 65, 12), np.float32),
        'stepid': np.arange(16*65*20, dtype=np.uint8).reshape(16, 65, 20),
        'dyn/deter': np.zeros((16, 65, 4), np.float32),
        'dyn/stoch': np.zeros((16, 65, 2, 2), np.float32)}
    carry = ({}, {'deter': np.zeros((16, 4), ml_dtypes.bfloat16),
        'stoch': np.broadcast_to(np.eye(2, dtype=ml_dtypes.bfloat16), (16, 2, 2)).copy()},
        {}, {'action': raw['action'][:, -1].copy()})
    outs = {'replay': {k: v[:, 1:].copy() for k, v in raw.items() if k != 'action'}}
    guard._returned_contract(proxy, carry, outs, raw)
    broken = copy.deepcopy(outs); broken['replay']['stepid'][0, 0, 0] ^= 1
    with self.assertRaisesRegex(ValueError, 'step identities'):
      guard._returned_contract(proxy, carry, broken, raw)
    broken = copy.deepcopy(carry); broken[3]['action'][0, 0] = 1
    with self.assertRaisesRegex(ValueError, 'previous raw'):
      guard._returned_contract(proxy, broken, outs, raw)

  def test_constructor_late_stale_nested_and_original_patch_ownership(self):
    before_loss, before_init = fixtures.FixtureDyn.loss, fixtures.FixtureDyn.__init__
    early = GuardAgent('Bwhole')
    with guard.observation_branch('Bwhole', fixtures.FixtureDyn):
      with self.assertRaisesRegex(ValueError, 'precede'): guard._assert_agent(early, 'Bwhole')
      stale = GuardAgent('Bwhole')
      with self.assertRaisesRegex(ValueError, 'Nested'):
        with guard.observation_branch('Amin', fixtures.FixtureDyn): pass
    with guard.observation_branch('Bwhole', fixtures.FixtureDyn):
      with self.assertRaisesRegex(ValueError, 'exact construction-time'): guard._assert_agent(stale, 'Bwhole')
      with original.rssm_branch('B', fixtures.FixtureDyn):
        with self.assertRaises(ValueError): guard._assert_agent(GuardAgent('Bwhole'), 'Bwhole')
    for owner in ('B', 'A'):
      context = original.rssm_branch('B', fixtures.FixtureDyn) if owner == 'B' else full.allrep_branch(fixtures.FixtureDyn)
      with context:
        with self.assertRaisesRegex(ValueError, 'Nested'):
          with guard.observation_branch('Bwhole', fixtures.FixtureDyn): pass
    self.assertIs(fixtures.FixtureDyn.loss, before_loss)
    self.assertIs(fixtures.FixtureDyn.__init__, before_init)

  def test_module_import_is_lazy_without_jax_devices_loaders_or_simulator(self):
    root = Path(__file__).resolve().parents[1]
    code = ('import sys; from scripts import dl_rep_observation_guard; '
        'assert not any(k in sys.modules for k in ("jax", "dreamerv3", "dm_control", "mujoco")); '
        'print("lazy CPU helper import")')
    result = subprocess.run([sys.executable, '-c', code], cwd=root, text=True,
        capture_output=True, timeout=30, env={**os.environ, 'PYTHONPATH': str(root)})
    self.assertEqual(result.returncode, 0, result.stderr)

  def test_cpu_wrapper_one_original_call_forward_gradients_and_pressure(self):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--jax-fixture'],
        cwd=root, text=True, capture_output=True, timeout=60,
        env={**os.environ, 'PYTHONPATH': str(root), 'CUDA_VISIBLE_DEVICES': '', 'JAX_PLATFORMS': 'cpu'})
    self.assertEqual(result.returncode, 0, result.stdout[-1000:]+result.stderr[-3000:])
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    self.assertEqual(receipt['devices'], ['cpu'])
    self.assertTrue(receipt['all_contracts_passed'])
    self.assertFalse(receipt['real_size50m_cuda_equivalence'])


def jax_fixture():
  dll = (os.add_dll_directory(os.environ['DL_DLL_DIRECTORY'])
      if os.name == 'nt' and os.environ.get('DL_DLL_DIRECTORY') else None)
  import jax
  import jax.numpy as jnp
  calls = [0]
  class Tiny:
    free_nats = 1
    def __init__(self): pass
    def loss(self, carry, tokens, acts, reset, training, rep_probe=False):
      calls[0] += 1
      return carry, {'entry': tokens}, {'rep': jnp.maximum(tokens**2, 1.),
          'dyn': acts**2}, tokens, {'base': jnp.sum(acts)}, {'key_marker': acts}
  x = jnp.array([[2., .5, 3.], [4., 2., .25]], jnp.float32)
  y = jnp.full_like(x, .7)
  reset = jnp.array([[True, False, False], [True, False, False]])
  for mode in guard.MODES:
    instance = Tiny()
    baseline = instance.loss((), x, y, reset, True, True)
    with guard.observation_branch(mode, Tiny):
      instance = Tiny(); calls[0] = 0
      protected = instance.loss((), x, y, reset, True, True)
      assert calls[0] == 1
      for index in (0, 1, 2, 3, 5):
        assert guard.typed_difference(guard._host(protected[index]), guard._host(baseline[index])) is None
      rep_grad = jax.grad(lambda z: instance.loss((), z, y, reset, True)[2]['rep'].sum())(x)
      expected = np.zeros((2, 3), np.float32) if mode == 'Amin' else np.where(np.asarray(x)**2 > 1, 2*np.asarray(x), 0)
      assert guard.typed_difference(np.asarray(rep_grad), expected) is None
      dyn_grad = jax.grad(lambda z: instance.loss((), x, z, reset, True)[2]['dyn'].sum())(y)
      assert guard.typed_difference(np.asarray(dyn_grad), 2*np.asarray(y)) is None
      inference = instance.loss((), x, y, reset, False)
      assert set(inference[4]) == {'base'}
      infer_grad = jax.grad(lambda z: instance.loss((), z, y, reset, False)[2]['rep'].sum())(x)
      assert guard.typed_difference(np.asarray(infer_grad), np.where(np.asarray(x)**2 > 1, 2*np.asarray(x), 0)) is None
      assert int(protected[4]['dl_reset/active_count']) == 2
      assert float(protected[4]['dl_reset/excess_sum']) == 18.
      if mode == 'Bwhole':
        assert int(protected[4]['dl_allrep/active_count']) == 4
        assert float(protected[4]['dl_allrep/excess_sum']) == 29.
      else: assert not set(protected[4]).intersection(guard.WHOLE_FIELDS)
  assert all(d.platform == 'cpu' for d in jax.devices())
  print(json.dumps(dict(all_contracts_passed=True, devices=['cpu'], real_size50m_cuda_equivalence=False)))
  if dll is not None: dll.close()


if __name__ == '__main__':
  if '--jax-fixture' in sys.argv: jax_fixture()
  else: unittest.main()
