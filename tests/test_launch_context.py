"""Regression for the observed interactive CUDA12.4 launch contamination.

The actual GPU compile comparison is preserved in the engineering report. These
CPU checks verify the existing explicit inherited-context API used to restore the
approved runtime, without changing global shell state or training configuration.
"""
import os
import unittest
from unittest.mock import patch

from scripts.run_hopper_seeds123 import child_environment


class LaunchContextTest(unittest.TestCase):
  def test_verified_context_overrides_interactive_CUDA_pollution(self):
    interactive=dict(PATH='/opt/miniforge3/bin:/usr/local/cuda-12.4/bin:/usr/bin',
        LD_LIBRARY_PATH=':/usr/local/cuda-12.4/lib64',CUDA_HOME='/usr/local/cuda-12.4',
        JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='wrong-mask')
    approved=dict(PATH='/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
        JAX_PLATFORMS='cuda',CUDA_VISIBLE_DEVICES='source-GPU7',
        MUJOCO_EGL_DEVICE_ID='7',WANDB_RUN_ID='source-GPU7-run')
    target=dict(CUDA_VISIBLE_DEVICES='assigned-GPU2',MUJOCO_EGL_DEVICE_ID='2',
        JAX_PLATFORMS='cuda',WANDB_RUN_ID='target-GPU2-retry',WANDB_MODE='online')
    before=dict(approved)
    with patch.dict(os.environ,interactive,clear=True):
      actual=child_environment(target,inherited=approved)
      self.assertEqual(os.environ['CUDA_HOME'],'/usr/local/cuda-12.4')
    for key in ('LD_LIBRARY_PATH','LD_PRELOAD','CUDA_HOME','CUDA_PATH','PYTHONPATH'):
      self.assertNotIn(key,actual)
    self.assertEqual(actual['PATH'],approved['PATH'])
    self.assertEqual(actual['CUDA_VISIBLE_DEVICES'],'assigned-GPU2')
    self.assertEqual(actual['MUJOCO_EGL_DEVICE_ID'],'2')
    self.assertEqual(actual['WANDB_RUN_ID'],'target-GPU2-retry')
    self.assertEqual(approved,before)

  def test_supervisor_CPU_flags_do_not_leak_into_restored_CUDA_context(self):
    approved=dict(PATH='/usr/local/bin:/usr/bin:/bin',CUDA_VISIBLE_DEVICES='',
        JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=2',
        CUDA_MPS_PIPE_DIRECTORY='/unused',PYTHONFAULTHANDLER='1')
    target=dict(CUDA_VISIBLE_DEVICES='assigned-UUID',MUJOCO_EGL_DEVICE_ID='2',
        JAX_PLATFORMS='cuda',WANDB_RUN_ID='unique-target')
    actual=child_environment(target,inherited=approved)
    for key in ('XLA_FLAGS','JAX_PLATFORM_NAME','CUDA_MPS_PIPE_DIRECTORY'):
      self.assertNotIn(key,actual)
    self.assertEqual(actual['JAX_PLATFORMS'],'cuda')
    self.assertEqual(actual['PYTHONFAULTHANDLER'],'1')
    self.assertEqual(actual['WANDB_RUN_ID'],'unique-target')
    self.assertEqual(approved['JAX_PLATFORMS'],'cpu')


if __name__=='__main__':unittest.main()
