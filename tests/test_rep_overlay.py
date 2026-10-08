"""Engineering fixtures for Dt, fixed constant, and global shuffle."""

import unittest
from types import SimpleNamespace

import elements
import jax
import jax.numpy as jnp
import numpy as np

from dreamerv3 import rep_probe


class OverlayTest(unittest.TestCase):

  def test_configuration(self):
    act = {'a': elements.Space(np.float32, (2,))}
    def config(mode, alpha=0.7, c=0.6):
      return SimpleNamespace(
          rep_probe=SimpleNamespace(mode=mode, alpha=alpha, c=c),
          dyn=SimpleNamespace(typ='rssm'),
          policy_dist_cont='bounded_normal')
    for mode in ('dt', 'constant', 'shuffle'):
      rep_probe.validate(config(mode), act)
    for c in (-1, 0, 1.01, float('nan'), float('inf')):
      with self.subTest(c=c), self.assertRaises(ValueError):
        rep_probe.validate(config('constant', c=c), act)
    with self.assertRaises(ValueError):
      rep_probe.validate(config('shuffle', alpha=-1), act)
    # Identity settings bypass the probe even if alpha was not calibrated.
    rep_probe.validate(config('dt', alpha=0), act)
    rep_probe.validate(config('constant', alpha=-1, c=1), act)

  def test_free_nats_order_and_local_gradients(self):
    raw = jnp.array([[0.2, 0.8, 2.0, 3.0]])
    weight = jnp.array([[0.25, 0.5, 0.75, 1.0]])
    def loss(x, tau):
      rep = jnp.maximum(x, tau) if tau else x
      return (weight * rep).mean()
    np.testing.assert_allclose(loss(raw, 1.0),
        np.mean(np.asarray(weight) * np.maximum(np.asarray(raw), 1.0)), rtol=1e-6)
    self.assertNotEqual(float(loss(raw, 1.0)),
        float(jnp.maximum(weight * raw, 1.0).mean()))
    np.testing.assert_allclose(jax.grad(loss)(raw, 1.0),
        [[0, 0, 0.75 / 4, 1 / 4]], rtol=1e-6, atol=0)
    np.testing.assert_allclose(jax.grad(loss)(raw, 0.0),
        np.asarray(weight) / 4, rtol=1e-6, atol=0)
    boundary = jnp.array([[1.0]])
    scale = 0.4
    base_grad = jax.grad(lambda x: jnp.maximum(x, 1.0).mean())(boundary)
    gated_grad = jax.grad(lambda x: (scale * jnp.maximum(x, 1.0)).mean())(boundary)
    np.testing.assert_array_equal(gated_grad, scale * base_grad)

  def test_global_shuffle_replay_and_multiset(self):
    source = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 20 + 0.2
    first, moved, perm = rep_probe.global_shuffle(source, 7, 3)
    repeat, moved2, perm2 = rep_probe.global_shuffle(source, 7, 3)
    np.testing.assert_array_equal(first, repeat)
    np.testing.assert_array_equal(perm, perm2)
    np.testing.assert_array_equal(moved, moved2)
    np.testing.assert_array_equal(np.sort(first.reshape(-1)),
        np.sort(source.reshape(-1)))
    np.testing.assert_array_equal(first.reshape(-1), source.reshape(-1)[perm])
    self.assertAlmostEqual(float(moved),
        float(np.mean(np.asarray(perm) != np.arange(12))), places=6)
    other, _, other_perm = rep_probe.global_shuffle(source, 7, 4)
    self.assertFalse(np.array_equal(first, other))
    self.assertFalse(np.array_equal(perm, other_perm))

  def test_global_shuffle_cross_shard(self):
    if len(jax.devices()) < 2:
      self.skipTest('requires at least two devices')
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:2]), ('d',))
    sharded = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('d'))
    mirrored = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    source = jax.device_put(np.arange(12, dtype=np.float32).reshape(4, 3), sharded)
    fn = jax.jit(lambda w: rep_probe.global_shuffle(w, 7, 3),
        in_shardings=sharded, out_shardings=(sharded, mirrored, sharded))
    with mesh:
      result, moved, perm = fn(source)
    values = np.asarray(jax.device_get(result))
    indices = np.asarray(jax.device_get(perm))
    np.testing.assert_array_equal(values.reshape(-1),
        np.asarray(jax.device_get(source)).reshape(-1)[indices])
    np.testing.assert_array_equal(np.sort(values.reshape(-1)), np.arange(12))
    self.assertTrue(np.any((indices // 6) != (np.arange(12) // 6)))
    self.assertAlmostEqual(float(moved),
        float(np.mean(indices != np.arange(12))), places=6)

  def test_matching_stats_strict_threshold_and_global_sum(self):
    raw = jnp.array([[0.5, 1.0, 1.2], [2.0, 0.9, 1.5]])
    weight = jnp.array([[0.1, 0.2, 0.5], [0.25, 0.4, 0.8]])
    total, count = rep_probe.matching_stats(raw, weight, 1.0)
    self.assertEqual(int(count), 3)
    self.assertAlmostEqual(float(total), 0.5 + 0.25 + 0.8, places=6)
    if len(jax.devices()) >= 2:
      mesh = jax.sharding.Mesh(np.array(jax.devices()[:2]), ('d',))
      sharded = jax.sharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec('d'))
      mirrored = jax.sharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec())
      fn = jax.jit(lambda r, w: rep_probe.matching_stats(r, w, 1.0),
          in_shardings=(sharded, sharded),
          out_shardings=(mirrored, mirrored))
      with mesh:
        result = fn(jax.device_put(raw, sharded),
                    jax.device_put(weight, sharded))
      self.assertAlmostEqual(float(result[0]), 1.55, places=6)
      self.assertEqual(int(result[1]), 3)


if __name__ == '__main__':
  unittest.main()
