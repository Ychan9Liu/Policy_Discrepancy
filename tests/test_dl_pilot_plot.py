"""Small CPU fixtures for authenticated plot inputs; no model or renderer runs."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import dl_pilot_plot as plot


def fixture():
  # Ten returns 0..9 plus a line rising 0..100; fixed curve integrals are known.
  offsets = {'B': 0, 'D': 10, 'N': 5}
  points = {group: [dict(action_step=step, mean=4.5+step/10000+offset,
      returns_raw=[float(i+step/10000+offset) for i in range(10)])
      for step in range(0, 1000001, 50000)] for group, offset in offsets.items()}
  summaries = {group: dict(auc_per_1m_actions=54.5+offset,
      tail_mean_return=94.5+offset, endpoint_return=104.5+offset)
      for group, offset in offsets.items()}
  bins = []
  for i in range(20):
    n = 11993 if i == 0 else 12500
    bins.append(dict(action_start=i*50000, action_end=(i+1)*50000,
        actual_v_mean=dict(mean=.9996, n=n), valid_active_frac=dict(mean=.95, n=n),
        active_protected_frac=dict(mean=.3, n=n)))
  review = dict(status='independent_evidence_review_passed', train_sha=plot.TRAIN_SHA,
      source_run='/data/Policy_Discrepancy/runs/fixture/dmc_quadruped_walk/N-seed0-attempt01',
      counts=dict(train_action_steps=1000000, evaluations=21, eval_action_steps=210000,
          updates=249493), summaries=summaries, gate=dict(updates=249493, bins=bins),
      comparisons={group: {key: dict(absolute=difference) for key in (
          'auc_per_1m_actions', 'tail_mean_return', 'endpoint_return')}
          for group, difference in (('B', 5), ('D', -5))})
  return review, points


class PlotInputs(unittest.TestCase):
  def write_inputs(self, root, review, points):
    paths = root/'review.json', root/'points.json'
    for path, payload in zip(paths, (review, points)):
      path.write_bytes(payload if isinstance(payload, bytes) else json.dumps(payload).encode())
    return paths, [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]

  def reject(self, review, points, message=None, wrong_hash=False):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      paths, hashes = self.write_inputs(root, review, points)
      if wrong_hash:
        hashes[1] = '0'*64
      output = root/'must_not_be_created'/'figures'
      with self.assertRaisesRegex(ValueError, message or '.'):
        plot.render(*paths, output, expected_review_sha=hashes[0], expected_points_sha=hashes[1])
      self.assertFalse(output.parent.exists(), 'Validation must precede every output write')

  def test_normal_hand_calculated_fixture_and_single_payload_reads(self):
    review, points = fixture()
    with tempfile.TemporaryDirectory() as temporary:
      paths, hashes = self.write_inputs(Path(temporary), review, points)
      original, calls = Path.read_bytes, []
      def tracked(path):
        calls.append(path)
        return original(path)
      with patch.object(Path, 'read_bytes', tracked):
        actual_review, actual_points, binding, contributions = plot.validate_inputs(*paths, *hashes)
      self.assertEqual(calls, list(paths))
      self.assertEqual(actual_review, review)
      self.assertEqual(actual_points, points)
      self.assertEqual(binding, dict(zip(('review.json', 'points.json'), hashes)))
      self.assertEqual(contributions['B']['total'], 5)
      self.assertEqual(contributions['D']['total'], -5)
      self.assertEqual(contributions['B']['points'][0]['auc_weight'], .025)
      self.assertEqual(contributions['B']['points'][10]['auc_weight'], .05)

  def test_common_point_shift_cannot_hide_behind_unchanged_relative_auc(self):
    review, points = fixture()
    before = [points['N'][5]['mean']-points[g][5]['mean'] for g in ('B', 'D')]
    for group in points:
      points[group][5]['mean'] += 100
      points[group][5]['returns_raw'] = [x+100 for x in points[group][5]['returns_raw']]
    self.assertEqual(before, [points['N'][5]['mean']-points[g][5]['mean'] for g in ('B', 'D')])
    self.reject(review, points, 'B auc_per_1m_actions')

  def test_missing_misordered_grid_raw_count_and_wrong_mean_are_rejected_before_writes(self):
    review, complete = fixture()
    for kind in ('missing', 'misgrid', 'raw_count', 'raw_mean', 'extra_group'):
      points = copy.deepcopy(complete)
      if kind == 'missing': points['N'].pop(5)
      if kind == 'misgrid': points['D'][5]['action_step'] += 1
      if kind == 'raw_count': points['B'][5]['returns_raw'].pop()
      if kind == 'raw_mean': points['N'][5]['mean'] += 1
      if kind == 'extra_group': points['other'] = []
      with self.subTest(kind=kind):
        self.reject(review, points)

  def test_wrong_hash_is_rejected_before_decode_or_output(self):
    review, _ = fixture()
    self.reject(review, b'not even JSON', 'Input SHA256 differs', wrong_hash=True)

  def test_duplicate_json_keys_and_nonfinite_literals_are_rejected(self):
    review, points = fixture()
    self.reject(b'{"status":"x","status":"y"}', points, 'Duplicate JSON key')
    for value in (math.nan, math.inf, -math.inf):
      altered = copy.deepcopy(points)
      altered['N'][1]['returns_raw'][0] = value
      with self.subTest(value=value):
        self.reject(review, altered, 'Nonfinite JSON number')
    self.reject(b'{"unused":1e999}', points, 'must be finite')

  def test_bools_cannot_impersonate_numeric_values(self):
    review, points = fixture()
    for kind in ('step', 'raw_return', 'mean', 'count', 'bin_count'):
      altered_review, altered_points = copy.deepcopy(review), copy.deepcopy(points)
      if kind == 'step': altered_points['N'][0]['action_step'] = False
      if kind == 'raw_return': altered_points['B'][0]['returns_raw'][0] = False
      if kind == 'mean': altered_points['N'][1]['mean'] = True
      if kind == 'count': altered_review['counts']['evaluations'] = True
      if kind == 'bin_count': altered_review['gate']['bins'][0]['actual_v_mean']['n'] = True
      with self.subTest(kind=kind):
        self.reject(altered_review, altered_points, 'not bool')

  def test_review_identity_absolute_summaries_comparisons_and_gate_binding(self):
    complete, points = fixture()
    for kind in ('status', 'sha', 'task', 'budget', 'tail', 'endpoint', 'comparison', 'gate_grid', 'gate_denominator'):
      review = copy.deepcopy(complete)
      if kind == 'status': review['status'] = 'pending'
      if kind == 'sha': review['train_sha'] = '0'*40
      if kind == 'task': review['source_run'] = review['source_run'].replace('quadruped_walk', 'other')
      if kind == 'budget': review['counts']['train_action_steps'] = 100000
      if kind == 'tail': review['summaries']['N']['tail_mean_return'] += 1
      if kind == 'endpoint': review['summaries']['B']['endpoint_return'] += 1
      if kind == 'comparison': review['comparisons']['D']['endpoint_return']['absolute'] += 1
      if kind == 'gate_grid': review['gate']['bins'][0]['action_end'] += 1
      if kind == 'gate_denominator': review['gate']['bins'][0]['active_protected_frac']['n'] += 1
      with self.subTest(kind=kind):
        self.reject(review, points)

  def test_existing_output_is_preserved(self):
    review, points = fixture()
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      paths, hashes = self.write_inputs(root, review, points)
      output = root/'existing'
      output.mkdir()
      marker = output/'keep.txt'
      marker.write_text('old plot stays')
      with self.assertRaisesRegex(ValueError, 'new plot directory'):
        plot.render(*paths, output, expected_review_sha=hashes[0], expected_points_sha=hashes[1])
      self.assertEqual(marker.read_text(), 'old plot stays')
      self.assertEqual(list(output.iterdir()), [marker])


if __name__ == '__main__':
  unittest.main()
