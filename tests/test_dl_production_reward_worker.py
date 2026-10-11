"""Worker boundary/oracle counterexamples, constructed values only."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_production_reward as worker


class WorkerGuards(unittest.TestCase):
  def test_physical_bin_oracle_uses_probabilities_and_symmetric_pairing(self):
    bins = np.arange(-127, 128, dtype=np.float32)
    logits = np.full((2, 255), -100., np.float32)
    logits[0, 128] = np.log(np.float32(.25))
    logits[0, 129] = np.log(np.float32(.75))
    logits[1, 125] = 0.
    pred = worker.symmetric_pred_numpy(logits, bins)
    np.testing.assert_allclose(pred, [1.75, -2.], atol=1e-7, rtol=0)
    np.testing.assert_array_equal(worker.symmetric_pred_numpy(
        np.zeros((3, 255), np.float32), bins), np.zeros(3))
    # A point oracle is distinct from choosing a reward-bin mode.
    self.assertNotEqual(pred[0], bins[np.argmax(logits[0])])

  def test_checkpoint_tree_identity_includes_typed_optimizer_and_norm_leaves(self):
    value = dict(params={'dyn/a': np.array([1.], np.float32),
                         'opt/m': np.array([0.], np.float32),
                         'retnorm/mean': np.array(.3, np.float32)},
                 counters={'updates': 74493, 'eval_seed': 3404107006})
    before = worker.tree_sha(value)
    for path in ('opt/m', 'retnorm/mean'):
      changed = copy.deepcopy(value)
      changed['params'][path] += np.float32(.1)
      self.assertNotEqual(worker.tree_sha(changed), before)
    changed = copy.deepcopy(value)
    changed['params']['dyn/a'] = changed['params']['dyn/a'].astype(np.float64)
    self.assertNotEqual(worker.tree_sha(changed), before)
    changed = copy.deepcopy(value)
    changed['counters']['eval_seed'] += 1
    self.assertNotEqual(worker.tree_sha(changed), before)
    with self.assertRaises(ValueError):
      worker.tree_sha({'params': np.array([np.nan])})

  def test_chunk_rejects_axis_dtype_seed_and_bin_changes(self):
    roots = np.arange(4 * 8 * 20 * 2, dtype=np.uint32).reshape(4, 8, 20, 2)
    value = dict(reward=np.zeros((4, 2, 8, 7, 20), np.float32),
        logits=np.zeros((4, 2, 8, 7, 20, 255), np.float32),
        bins=np.arange(-127, 128, dtype=np.float32), root_keys=roots.copy(),
        sample_keys=roots.copy())
    worker.validate_chunk(value, roots, value['bins'])
    changes = (
        lambda x: x.update(reward=x['reward'].swapaxes(1, 2)),
        lambda x: x.update(reward=x['reward'].astype(np.float64)),
        lambda x: x['root_keys'].__setitem__((0, 0, 0, 0), np.uint32(900)),
        lambda x: x['bins'].__setitem__(0, np.float32(-130)),
        lambda x: x['reward'].__setitem__((0, 0, 0, 0, 0), np.nan),
        lambda x: x.update(extra='unreviewed'))
    for change in changes:
      with self.subTest(change=change):
        bad = copy.deepcopy(value); change(bad)
        with self.assertRaises(ValueError):
          worker.validate_chunk(bad, roots, value['bins'])


if __name__ == '__main__':
  unittest.main()
