"""Resource guard: actual competing jobs refuse even with zero memory."""
import unittest
from unittest.mock import patch

class GPUResourceGuard(unittest.TestCase):
  def check(self, memory, compute=(), graphics=(), pmon='3 - - -\n'):
    from scripts.run_formal_baselines import require_free, UUIDS
    entries = ''.join(f'<process_info><pid>{p}</pid><type>{typ}</type></process_info>'
        for typ, values in [('C', compute), ('G', graphics)] for p in values)
    xml = f'<nvidia_smi_log><gpu><uuid>{UUIDS[3]}</uuid><processes>{entries}</processes></gpu></nvidia_smi_log>'
    with patch('scripts.run_formal_baselines.gpu_state',
               return_value={3: (UUIDS[3], memory)}), \
         patch('scripts.run_formal_baselines.subprocess.check_output', side_effect=[xml, pmon]):
      return require_free(3)

  def test_no_job_memory_is_recorded(self):
    self.assertEqual(self.check(1)['memory_used_mib'], 1)

  def test_compute_job_blocks_zero_memory(self):
    with self.assertRaises(RuntimeError):
      self.check(0, compute=[123])

  def test_graphics_job_blocks_zero_memory(self):
    with self.assertRaises(RuntimeError):
      self.check(0, graphics=[123])

  def test_pmon_job_blocks_empty_inventory(self):
    with self.assertRaises(RuntimeError):
      self.check(0, pmon='3 123 G 0 0\n')

if __name__ == '__main__':
  unittest.main()
