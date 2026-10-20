"""Frozen-input CPU probe checks; no environment or optimizer update.

Compare the trusted pre-fix Git implementation, including all unchanged metrics,
loss/gradients and surrounding Ninjax keys. Parameters here are test fixtures.
"""

import subprocess
import unittest

import elements
import embodied.jax.outs as outs
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from dreamerv3 import rep_probe


class Dyn:
    free_nats = 1.

    def _dist(self, logit):
        return outs.Agg(outs.OneHot(logit, .01), 1)


class FrozenAgent:
    dyn = Dyn()
    act_space = {'action': elements.Space(np.float32, (2,))}

    def __init__(self, scale):
        self.scale = scale

    def feat2tensor(self, feat):
        z = feat['stoch'].reshape((*feat['stoch'].shape[:2], -1))
        return jnp.concatenate([feat['deter'], z], -1)

    def pol(self, x, bdims):
        assert bdims == 2
        # Every class/factor can influence the actor; probe itself never samples.
        coeff = jnp.arange(x.shape[-1], dtype=jnp.float32) / 10
        mean = self.scale * (x * coeff).sum(-1, keepdims=True)
        mean = jnp.concatenate([mean, -.5 * mean], -1)
        std = .5 + .1 * jax.nn.sigmoid(mean)
        return {'action': outs.Agg(outs.Normal(mean, std), 1)}


class ModeProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = subprocess.check_output([
            'git', 'show', 'd8124f9157dbddf653354ba98bc0b90ca4debe58:dreamerv3/rep_probe.py'],
            text=True)
        namespace = {}
        exec(compile(source, '<trusted pre-fix rep_probe>', 'exec'), namespace)
        cls.legacy = staticmethod(namespace['overlay'])

    def rates(self, q_indices, p_indices, classes=32):
        q = jax.nn.one_hot(jnp.array(q_indices), classes) * 6
        p = jax.nn.one_hot(jnp.array(p_indices), classes) * 6
        shape = q.shape[:2]
        _, metrics = rep_probe.overlay(FrozenAgent(1.),
            {'deter': jnp.ones((*shape, 1)), 'logit': q}, p,
            jnp.full(shape, 2.), jnp.full(shape, 2.), 'logging', 20)
        return metrics

    def test_coordinate_factor_vector_counterexample_and_extremes(self):
        # [B=1,T=2,S=2,C=32], one differing factor at only one time.
        rates = self.rates([[[0, 0], [0, 0]]], [[[1, 0], [0, 0]]])
        self.assertEqual(float(rates['dt/mode_coordinate_diff_frac']), .015625)
        self.assertEqual(float(rates['dt/mode_factor_diff_frac']), .25)
        self.assertEqual(float(rates['dt/mode_vector_diff_frac']), .5)
        self.assertEqual(int(rates['dt/mode_diagnostic_version']), 2)
        for prior, expected in (([[[0, 0], [0, 0]]], (0., 0., 0.)),
                                ([[[1, 1], [1, 1]]], (2/32, 1., 1.))):
            r = self.rates([[[0, 0], [0, 0]]], prior)
            self.assertEqual(tuple(float(r['dt/mode_' + k + '_diff_frac'])
                                   for k in ('coordinate', 'factor', 'vector')), expected)

    def test_equal_factor_rates_do_not_determine_vector_rate(self):
        q = [[[0, 0], [0, 0]]]
        together = self.rates(q, [[[1, 1], [0, 0]]])
        separated = self.rates(q, [[[1, 0], [1, 0]]])
        self.assertEqual(float(together['dt/mode_factor_diff_frac']), .5)
        self.assertEqual(float(separated['dt/mode_factor_diff_frac']), .5)
        self.assertEqual(float(together['dt/mode_vector_diff_frac']), .5)
        self.assertEqual(float(separated['dt/mode_vector_diff_frac']), 1.)

    def test_batch_time_factor_axes_and_tied_modes(self):
        q = [[[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]]]
        p = [[[1, 0], [0, 0], [0, 0]], [[1, 1], [0, 0], [1, 0]]]
        r = self.rates(q, p, classes=4)
        np.testing.assert_allclose(
            [r['dt/mode_coordinate_diff_frac'], r['dt/mode_factor_diff_frac'],
             r['dt/mode_vector_diff_frac']], [8/48, 4/12, 3/6], rtol=1e-6)
        logits = jnp.zeros((2, 3, 2, 4))  # deterministic shared argmax on ties
        _, metrics = rep_probe.overlay(FrozenAgent(1.),
            {'deter': jnp.ones((2, 3, 1)), 'logit': logits}, logits,
            jnp.ones((2, 3)), jnp.ones((2, 3)), 'logging', 20)
        self.assertEqual(float(metrics['dt/mode_vector_diff_frac']), 0.)

    def test_old_new_loss_gradients_metrics_and_rng_exact(self):
        qidx = jnp.array([[[0, 0], [0, 0], [0, 0]],
                         [[0, 0], [0, 0], [0, 0]]])
        pidx = jnp.array([[[1, 0], [0, 0], [1, 1]],
                         [[0, 1], [1, 0], [1, 1]]])
        q, p = jax.nn.one_hot(qidx, 32) * 6, jax.nn.one_hot(pidx, 32) * 6
        h = jnp.ones((2, 3, 1))
        control = nj.pure(lambda: (nj.seed(), nj.seed()))({}, seed=19)[1]
        changed = {'dt/mode_factor_diff_frac', 'dt/mode_vector_diff_frac',
                   'dt/mode_coordinate_diff_frac', 'dt/mode_diagnostic_version'}
        for mode in ('logging', 'dt', 'constant', 'shuffle'):
            with self.subTest(mode=mode):
                def make(overlay):
                    def objective(scale, q, p, h):
                        before = nj.seed()
                        raw = Dyn()._dist(q).kl(Dyn()._dist(jax.lax.stop_gradient(p)))
                        rep = jnp.maximum(raw, 1.)
                        weight, metrics = overlay(FrozenAgent(scale),
                            {'deter': h, 'logit': q}, p, raw, rep, mode, 20., .6, 7, 3)
                        loss = .1 * (weight * rep).mean() + .03 * (h ** 2).mean() + scale ** 2
                        return loss, (weight, metrics, before, nj.seed(), raw)
                    pure = nj.pure(objective)
                    return jax.jit(jax.value_and_grad(
                        lambda *xs: pure({}, *xs, seed=19)[1],
                        has_aux=True, argnums=(0, 1, 2, 3)))
                old = make(self.legacy)(1., q, p, h)
                new = make(rep_probe.overlay)(1., q, p, h)
                (loss, (weight, metrics, first, last, raw)), grads = new
                self.assertTrue(bool(jnp.any(raw > 1)))
                self.assertGreater(float(jnp.linalg.norm(grads[1])), 0)
                if mode != 'logging':
                    self.assertTrue(bool(jnp.any(weight < .999)))
                np.testing.assert_array_equal(old[0][0], loss)
                for a, b in zip(jax.tree.leaves(old[1]), jax.tree.leaves(grads)):
                    np.testing.assert_array_equal(a, b)
                np.testing.assert_array_equal(old[0][1][0], weight)
                for key in metrics.keys() - changed:
                    np.testing.assert_array_equal(old[0][1][1][key], metrics[key])
                np.testing.assert_array_equal(metrics['dt/mode_factor_diff_frac'],
                                              old[0][1][1]['dt/mode_vector_diff_frac'])
                np.testing.assert_array_equal(metrics['dt/mode_coordinate_diff_frac'],
                                              old[0][1][1]['dt/mode_factor_diff_frac'])
                for a, b in zip((first, last), control):
                    np.testing.assert_array_equal(a, b)
                for a, b in zip(old[0][1][2:4], (first, last)):
                    np.testing.assert_array_equal(a, b)
                # Adding the boolean diagnostics to a scalar cannot create
                # gradients through actor, logits or recurrent features.
                for name in ('coordinate', 'factor', 'vector'):
                    def diagnostic(scale, q, p, h):
                        return rep_probe.overlay(FrozenAgent(scale),
                            {'deter': h, 'logit': q}, p, raw, jnp.maximum(raw, 1.),
                            mode, 20., .6, 7, 3)[1]['dt/mode_' + name + '_diff_frac']
                    for grad in jax.grad(diagnostic, argnums=(0, 1, 2, 3))(1., q, p, h):
                        np.testing.assert_array_equal(grad, np.zeros_like(grad))


if __name__ == '__main__':
    unittest.main()
