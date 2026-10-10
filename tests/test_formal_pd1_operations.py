"""Boundaries for exact summary transport and declared config differences."""
import math
from types import SimpleNamespace
import unittest
from scripts.audit_formal_pd1 import exact_payload, verify_transport

class ExactTransport(unittest.TestCase):
  def summary(self, values):
    encoded, digest = exact_payload(values)
    return dict(values, **{'formal/exact_values_json': encoded,
                          'formal/exact_values_sha256': digest}), encoded, digest

  def test_rounding_retained_with_exact_source(self):
    values = {'formal/c': 0.9287514026396865, 'formal/count': 3}
    remote, encoded, digest = self.summary(values)
    remote['formal/c'] = math.nextafter(values['formal/c'], 0)
    result = verify_transport(remote, values, encoded, digest)
    self.assertEqual(result['formal/c']['authoritative'], values['formal/c'])
    self.assertNotEqual(result['formal/c']['numeric_display'], values['formal/c'])

  def test_exact_payload_corruption_refused(self):
    values = {'formal/c': 0.9}
    remote, encoded, digest = self.summary(values)
    remote['formal/exact_values_json'] = encoded.replace('0.9', '0.8')
    with self.assertRaises(RuntimeError):
      verify_transport(remote, values, encoded, digest)

  def test_missing_float_and_changed_integer_refused(self):
    for key, replacement in [('formal/c', None), ('formal/count', 4)]:
      values = {'formal/c': 0.9, 'formal/count': 3}
      remote, encoded, digest = self.summary(values)
      remote[key] = replacement
      with self.assertRaises(RuntimeError):
        verify_transport(remote, values, encoded, digest)

class DeclaredConfig(unittest.TestCase):
  def test_only_constant_fields_allowed(self):
    from scripts.run_formal_baselines import config_differences
    baseline = {'agent.rep_probe.mode': 'logging', 'agent.rep_probe.c': 1.,
                'run.frozen_c_file': '', 'logdir': 'baseline',
                'agent.reward_grad': True}
    current = dict(baseline, logdir='constant')
    current.update({'agent.rep_probe.mode': 'constant', 'agent.rep_probe.c': .9,
                    'run.frozen_c_file': 'formal-source.json'})
    self.assertEqual(len(config_differences(SimpleNamespace(flat=current),
                                          baseline, 'constant')), 4)
    current['agent.reward_grad'] = False
    with self.assertRaises(ValueError):
      config_differences(SimpleNamespace(flat=current), baseline, 'constant')

  def test_shuffle_c_change_refused(self):
    from scripts.run_formal_baselines import config_differences
    baseline = {'agent.rep_probe.c': 1.}
    with self.assertRaises(ValueError):
      config_differences(SimpleNamespace(flat={'agent.rep_probe.c': .9}),
                         baseline, 'shuffle')

if __name__ == '__main__':
  unittest.main()
