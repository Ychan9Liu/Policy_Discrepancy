"""Supervisor boundary checks; no subprocess launches or network/GPU access."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from dreamerv3.protocol_scope import HOPPER_123
from scripts.run_hopper_seeds123 import MATRIX, child_environment, require_free, run, validate_matrix


class HopperSupervisorTest(unittest.TestCase):
  def plan(self):
    return dict(protocol=HOPPER_123,authorized_to_train=False,runs=[dict(
        seed=seed,mode=mode,physical_gpu=gpu,server='sv3',host='lyg0326',
        task='dmc_hopper_hop') for (seed,mode),gpu in MATRIX.items()])

  def test_exact_matrix_no_extra_or_substituted_resources(self):
    validate_matrix(self.plan())
    for field,value in (('physical_gpu',0),('server','sv2'),('mode','constant'),('seed',0)):
      plan=self.plan();plan['runs'][0][field]=value
      with self.subTest(field=field),self.assertRaises(ValueError):
        validate_matrix(plan)
    plan=self.plan();plan['runs'].append(plan['runs'][0])
    with self.assertRaises(ValueError): validate_matrix(plan)

  def test_C_and_G_processes_and_pmon_refuse_but_idle_memory_is_not_a_job(self):
    xml='<nvidia_smi_log><gpu><uuid>assigned</uuid><fb_memory_usage><used>1 MiB</used></fb_memory_usage><processes>{}</processes></gpu></nvidia_smi_log>'
    with patch('scripts.run_hopper_seeds123.subprocess.check_output',side_effect=[xml.format(''),'# header\n1 - - -\n']):
      self.assertTrue(require_free(1,'assigned')['no_jobs'])
    for kind in ('C','G','C+G'):
      process=f'<process_info><pid>3</pid><type>{kind}</type></process_info>'
      with patch('scripts.run_hopper_seeds123.subprocess.check_output',return_value=xml.format(process)):
        with self.assertRaisesRegex(RuntimeError,'has jobs'):require_free(1,'assigned')
    with patch('scripts.run_hopper_seeds123.subprocess.check_output',side_effect=[xml.format(''),'1 99 G 0\n']):
      with self.assertRaisesRegex(RuntimeError,'C/G jobs'):require_free(1,'assigned')
    with patch('scripts.run_hopper_seeds123.subprocess.check_output',return_value=xml.format('')):
      with self.assertRaisesRegex(RuntimeError,'UUID differs'):require_free(1,'wrong')

  def test_training_child_isolated_from_supervisor_CPU_and_MPS_environment(self):
    inherited=dict(JAX_PLATFORM_NAME='cpu',JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=2',
                   CUDA_MPS_PIPE_DIRECTORY='/not-used',CUDA_VISIBLE_DEVICES='',WANDB_RUN_ID='old')
    template=dict(CUDA_VISIBLE_DEVICES='assigned-UUID',JAX_PLATFORMS='cuda',MUJOCO_EGL_DEVICE_ID='5',
                  WANDB_RUN_ID='new-specific-id',WANDB_MODE='online')
    actual=child_environment(template,inherited)
    for key in ('XLA_FLAGS','JAX_PLATFORM_NAME','CUDA_MPS_PIPE_DIRECTORY'):
      self.assertNotIn(key,actual)
    self.assertEqual(actual['JAX_PLATFORMS'],'cuda')
    self.assertEqual(actual['WANDB_RUN_ID'],'new-specific-id')
    self.assertEqual(inherited['WANDB_RUN_ID'],'old')
    with self.assertRaises(ValueError):child_environment(dict(template,MUJOCO_EGL_DEVICE_ID=None),inherited)

  def test_unauthorized_plan_cannot_start_process_or_create_campaign(self):
    with TemporaryDirectory() as directory:
      path=Path(directory)/'plan.json';path.write_text(json.dumps(self.plan()))
      with patch('scripts.run_hopper_seeds123.subprocess.Popen') as start:
        with self.assertRaises(RuntimeError):run(path)
        start.assert_not_called()


if __name__=='__main__':unittest.main()
