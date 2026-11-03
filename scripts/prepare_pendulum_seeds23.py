"""Prepare the four explicitly approved pendulum repeats, without launching."""
import argparse
import json
from pathlib import Path

from dreamerv3.protocol_scope import PENDULUM_23, PENDULUM_SCRIPT, validate_scope
from scripts.prepare_dmc16 import source_state, PYTHON
from scripts.prepare_m2_v1 import resolve, check
from scripts.run_hopper_seeds123 import dump, sha, now, ENTITY

HOST = 'lyg0326'
TASK = 'dmc_pendulum_swingup'
CARDS = {2: 1, 3: 4}
PREFIX = 'formal-m2-pendulum-seeds23-'


def check_config(config):
  if (config.script != PENDULUM_SCRIPT or
      validate_scope(config.script, config.task, config.seed, config.agent.rep_probe.mode) != PENDULUM_23 or
      tuple(config.logger.outputs) != ('jsonl', 'wandb')):
    raise ValueError('Wrong pendulum repeat entry/logger')
  # Only the checker copy normalizes explicitly approved scope/task/seed/logger.
  check(config.update({'script': 'protocol_v1', 'task': 'dmc_hopper_hop',
                       'seed': 0, 'logger.outputs': ['jsonl']}))


def prepare(output, resources, references, campaign):
  output = Path(output)
  if output.exists():
    raise FileExistsError(output)
  if not campaign.startswith(PREFIX) or any(s in campaign for s in ('/', '\\', '..')):
    raise ValueError('Fresh pendulum-seeds23 campaign required')
  commit, drafts = source_state()
  resource = json.loads(Path(resources).read_text(encoding='utf-8'))
  reference = json.loads(Path(references).read_text(encoding='utf-8'))
  if resource['host'] != HOST:
    raise ValueError('Assigned host differs')
  devices = {r['physical_gpu']: r for r in resource['cards']}
  if not set(CARDS.values()) <= devices.keys() or len({devices[g]['uuid'] for g in CARDS.values()}) != 2:
    raise ValueError('Missing/duplicate assigned physical devices')
  root = '/data/Policy_Discrepancy/runs/' + campaign
  rows = []
  for seed, gpu in CARDS.items():
    for position, mode in enumerate(('logging', 'dt'), 1):
      group = 'baseline' if mode == 'logging' else 'dt'
      directory = f'{root}/seed{seed}/{mode}-attempt01'
      argv = [PYTHON, '-u', '-m', 'dreamerv3.main', '--configs', 'm2_pendulum_seeds23',
          '--task', TASK, '--seed', str(seed), '--logdir', directory,
          '--agent.rep_probe.mode', mode, '--jax.prealloc', 'False',
          '--logger.outputs', 'jsonl', 'wandb']
      c = resolve(argv[4:]); check_config(c)
      config = json.loads(json.dumps(dict(c.flat)))
      original = reference[group]['config']
      differences = {k: [original.get(k), config.get(k)] for k in original.keys() | config.keys()
                     if original.get(k) != config.get(k)}
      if set(differences) != {'script', 'seed', 'logdir'}:
        raise ValueError('Scientific config differs from the original pendulum run')
      run_id = f'{campaign}-pendulum-seed{seed}-{mode}-01'
      env = dict(CUDA_VISIBLE_DEVICES=devices[gpu]['uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
          MUJOCO_EGL_DEVICE_ID=None, MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
          JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false',
          OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
          PYTHONUNBUFFERED='1', TMPDIR=directory+'/tmp', WANDB_DIR=directory,
          WANDB_ENTITY=ENTITY, WANDB_PROJECT='PD_1', WANDB_MODE='online',
          WANDB_RESUME='never', WANDB_RUN_GROUP=campaign, WANDB_RUN_ID=run_id)
      rows.append(dict(task=TASK, seed=seed, mode=mode, group=group, protocol=PENDULUM_23,
          code_sha=commit, server='sv3', host=HOST, physical_gpu=gpu,
          gpu_uuid=devices[gpu]['uuid'], directory=directory, queue_position=position,
          run_id='sv3:'+run_id, state='prepared_not_started', authorized_to_train=False,
          launch_argv=None, review_argv=argv, config=config, egl_index=None,
          verified_compute_and_graphics_same_gpu=False, environment_template=env,
          reference_config_differences=differences,
          baseline_c_source=f'{root}/seed{seed}/logging-attempt01',
          serial_predecessor='own fully audited baseline' if mode=='dt' else None))
  plan = dict(protocol=PENDULUM_23, code_sha=commit, campaign=campaign, root=root,
      server='sv3', host=HOST, runs=rows, authorized_to_train=False,
      auto_start_when_idle=False, created_utc=now(), local_drafts=drafts,
      resource_snapshot=resource, references=reference, reference_file_sha256=sha(references),
      resource_file_sha256=sha(resources), required_storage_bytes=240_000_000_000,
      authorization='User explicitly approved original-config four repeats and launch, 2026-11-03',
      pending=['same_SHA_CPU_tests', 'fresh_CUDA_EGL_mapping', 'four_real_CUDA_fixtures',
               'online_logger', 'fresh_exclusive_resources_and_capacity'])
  output.mkdir(parents=True)
  dump(output/'planned-run-index.json', plan)
  for r in rows:
    dump(output/f'seed{r["seed"]}-{r["mode"]}-config.json', r['config'])
  print(json.dumps(dict(code_sha=commit, runs=4, training_started=False)))


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  for name in ('output', 'resources', 'references', 'campaign'):
    p.add_argument('--'+name, required=True)
  a = p.parse_args()
  prepare(a.output, a.resources, a.references, a.campaign)
