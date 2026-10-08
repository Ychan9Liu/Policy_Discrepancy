"""Numerical fixtures for the logging-only policy discrepancy probe."""

import unittest
from types import SimpleNamespace

import embodied.jax.outs as outs
import elements
import jax
import jax.numpy as jnp
import numpy as np

from dreamerv3 import rep_probe


def policy(mean, std):
  return outs.Normal(jnp.asarray(mean, jnp.float32), jnp.asarray(std, jnp.float32))


class ProbeTest(unittest.TestCase):

  def test_gaussian_kl_fixtures(self):
    spaces = {'a': elements.Space(np.float32, (2,)),
              'b': elements.Space(np.float32, ())}
    q = {'a': outs.Agg(policy([[[1., 0.]]], [[[1., 1.]]]), 1),
         'b': policy([[0.]], [[1.]])}
    p = {'a': outs.Agg(policy([[[0., 0.]]], [[[1., 1.]]]), 1),
         'b': policy([[0.]], [[2.]])}
    d, md, sd = rep_probe.discrepancy(q, p, spaces)
    expected = (0.5 + 0.25 * (4 - 1) ** 2 / 4) / 3
    np.testing.assert_allclose(d, [[expected]], rtol=1e-6)
    np.testing.assert_allclose(md, [[1 / 3]], rtol=1e-6)
    np.testing.assert_allclose(sd, [[1 / 3]], rtol=1e-6)
    reversed_d = rep_probe.discrepancy(p, q, spaces)[0]
    np.testing.assert_allclose(d, reversed_d, rtol=1e-6)
    np.testing.assert_array_equal(rep_probe.discrepancy(q, q, spaces)[0], [[0.]])

  def test_invalid_parameters_are_visible(self):
    space = {'a': elements.Space(np.float32, ())}
    d = rep_probe.discrepancy(
        {'a': policy([[0.]], [[0.]])},
        {'a': policy([[0.]], [[1.]])}, space)[0]
    self.assertTrue(np.isnan(np.asarray(d)).all())

  def test_configuration_validation(self):
    act = {'a': elements.Space(np.float32, (2,))}
    config = SimpleNamespace(
        rep_probe=SimpleNamespace(mode='logging', alpha=0.5),
        dyn=SimpleNamespace(typ='rssm'),
        policy_dist_cont='bounded_normal')
    rep_probe.validate(config, act)
    for alpha in (None, -1, float('inf'), float('nan')):
      with self.subTest(alpha=alpha), self.assertRaises(ValueError):
        rep_probe.validate(SimpleNamespace(**{
            **vars(config), 'rep_probe': SimpleNamespace(mode='logging', alpha=alpha)}), act)
    with self.assertRaises(ValueError):
      rep_probe.validate(config, {'a': elements.Space(np.int32, (), 0, 3)})
    rep_probe.validate(SimpleNamespace(rep_probe=SimpleNamespace(mode='off')),
        {'a': elements.Space(np.int32, (), 0, 3)})

  def test_probe_mode_and_gradient_isolation(self):
    space = {'action': elements.Space(np.float32, (2,))}
    class Dyn:
      free_nats = 0.1
      def _dist(self, logits):
        return outs.Agg(outs.OneHot(logits, 0.01), 1)
    class FakeAgent:
      act_space = space
      dyn = Dyn()
      def __init__(self, scale):
        self.scale = scale
      def feat2tensor(self, feat):
        z = feat['stoch'].reshape((*feat['stoch'].shape[:2], -1))
        return jnp.concatenate([feat['deter'], z], -1)
      def pol(self, x, bdims):
        assert bdims == 2
        mean = self.scale * x[..., 1:3]
        std = 0.5 + 0.1 * jax.nn.sigmoid(x[..., 3:5])
        return {'action': outs.Agg(outs.Normal(mean, std), 1)}
    q = jnp.array([[[[3., 0., 0., 0.], [0., 2., 0., 0.]]]])
    p = jnp.array([[[[0., 3., 0., 0.], [0., 2., 0., 0.]]]])
    h = jnp.ones((1, 1, 1))
    def probe(scale, deter, qlogit, plogit):
      feat = {'deter': deter, 'logit': qlogit}
      return rep_probe.metrics(FakeAgent(scale), feat, plogit,
          jnp.array([[1.5]]), jnp.array([[1.5]]), 0.7)
    values = probe(1., h, q, p)
    self.assertGreater(float(values['dt/D_mean']), 0)
    self.assertLess(float(values['dt/w_mean']), 1)
    for key in ('dt/D_mean', 'dt/w_mean'):
      grads = jax.grad(lambda *xs: probe(*xs)[key], argnums=(0, 1, 2, 3))(
          1., h, q, p)
      for grad in grads:
        np.testing.assert_array_equal(grad, np.zeros_like(grad))
    # Different logits with the same mode give the same representative input.
    same_mode = q.at[..., 0, 0].set(4.)
    same = probe(1., h, q, same_mode)
    self.assertEqual(float(same['dt/D_mean']), 0.)
    self.assertEqual(float(same['dt/w_mean']), 1.)

  def test_rep_gradient_remains_active(self):
    prior = jnp.zeros((1, 1, 2, 4))
    post = prior.at[..., 0].set(2.)
    def rep_loss(logits):
      q = outs.Agg(outs.OneHot(logits, 0.01), 1)
      p = outs.Agg(outs.OneHot(prior, 0.01), 1)
      raw = q.kl(p)
      return jnp.maximum(raw, 0.1).mean()
    grad = jax.grad(rep_loss)(post)
    self.assertGreater(float(jnp.linalg.norm(grad)), 0)


if __name__ == '__main__':
  unittest.main()
