"""Numerical fixtures for the logging-only policy discrepancy probe."""

import unittest
from types import SimpleNamespace

import embodied.jax.outs as outs
import elements
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


if __name__ == '__main__':
  unittest.main()
