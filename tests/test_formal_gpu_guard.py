"""Resource guard: actual competing jobs refuse even with zero memory."""
import sys
import types
import unittest
from unittest.mock import patch

class GPUResourceGuard(unittest.TestCase):
  def check(self, memory, compute=(), graphics=(), pmon='3 - - -\n'):
    from scripts.run_formal_baselines import require_free, UUIDS
    nvml = types.SimpleNamespace(nvmlInit=lambda: None,
        nvmlShutdown=lambda: None, nvmlDeviceGetHandleByUUID=lambda uuid: uuid,
        nvmlDeviceGetComputeRunningProcesses=lambda h:
            [types.SimpleNamespace(pid=p) for p in compute],
        nvmlDeviceGetGraphicsRunningProcesses=lambda h:
            [types.SimpleNamespace(pid=p) for p in graphics])
    with patch.dict(sys.modules, pynvml=nvml), \
         patch('scripts.run_formal_baselines.gpu_state',
               return_value={3: (UUIDS[3], memory)}), \
         patch('scripts.run_formal_baselines.subprocess.check_output', return_value=pmon):
      return require_free(3)

  def test_no_job_memory_is_recorded(self):
    self.assertEqual(self.check(1)['memory_used_mib'], 1)

  def test_compute_job_blocks_zero_memory(self):
    with self.assertRaises(RuntimeError):
      self.check(0, compute=[123])

  def test_graphics_job_blocks_zero_memory(self):
    with self.assertRaises(RuntimeError):
      self.check(0, graphics=[123])

  def test_pmon_job_blocks_empty_nvml(self):
    with self.assertRaises(RuntimeError):
      self.check(0, pmon='3 123 G 0 0\n')

if __name__ == '__main__':
  unittest.main()
