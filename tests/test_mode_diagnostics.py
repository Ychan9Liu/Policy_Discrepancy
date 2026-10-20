"""Compatibility checks independent of JAX or training dependencies."""

import unittest

from analysis.mode_diagnostics import normalize


class ModeCompatibilityTest(unittest.TestCase):
    def test_legacy_is_explicit_and_vector_stays_unknown(self):
        row = {'dt/mode_factor_diff_frac': .015625, 'dt/mode_vector_diff_frac': .25}
        with self.assertRaises(ValueError):
            normalize(row, prefix='dt/')
        result = normalize(row, legacy=True, prefix='dt/')
        self.assertEqual(result['mode_coordinate_diff_frac'], .015625)
        self.assertEqual(result['mode_factor_diff_frac'], .25)
        self.assertIsNone(result['mode_vector_diff_frac'])
        self.assertFalse(result['vector_available'])
        self.assertEqual(row['dt/mode_vector_diff_frac'], .25)

    def test_v2_uses_correct_fields_even_in_legacy_reader(self):
        row = dict(mode_diagnostic_version=2, mode_coordinate_diff_frac=.015625,
                   mode_factor_diff_frac=.25, mode_vector_diff_frac=.5)
        result = normalize(row, legacy=True, prefix='')
        self.assertEqual(result['mode_vector_diff_frac'], .5)
        self.assertTrue(result['vector_available'])

    def test_unknown_mixed_missing_and_invalid_refused(self):
        valid = dict(mode_diagnostic_version=2, mode_coordinate_diff_frac=.015625,
                     mode_factor_diff_frac=.25, mode_vector_diff_frac=.5)
        for version in (1, 1.5, 2.5, 3):
            with self.subTest(version=version), self.assertRaises(ValueError):
                normalize(dict(valid, mode_diagnostic_version=version), prefix='')
        with self.assertRaises(KeyError):
            normalize({'mode_diagnostic_version': 2}, prefix='')
        for value in (float('nan'), float('inf'), -1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize(dict(valid, mode_factor_diff_frac=value), prefix='')


if __name__ == '__main__':
    unittest.main()
