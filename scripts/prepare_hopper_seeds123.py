"""CPU-only preparation of the six user-specified Hopper repeat runs.

No launch action exists. GPU/EGL binding and separate training authorization are
required later by 04; resource snapshots are never reservations.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

import numpy as np

from dreamerv3.protocol_scope import HOPPER_123, HOPPER_SCRIPT, validate_scope
from scripts.prepare_m2_v1 import check, resolve

REPO = Path(__file__).resolve().parents[1]
PYTHON = '/data/Policy_Discrepancy/envs/dreamer/bin/python'
ALLOCATION = ((1, 'logging', 1), (1, 'dt', 3), (2, 'logging', 4),
              (2, 'dt', 5), (3, 'logging', 6), (3, 'dt', 7))


def dump(path, value):
  Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def sha(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed32(root, namespace, *indices):
  parts = [str(root), 'dmc_hopper_hop', namespace, *(str(x) for x in indices)]
  return int.from_bytes(hashlib.sha256('\0'.join(parts).encode()).digest()[:4], 'big')


def check_repeat(config):
  validate_scope(config.script, config.task, config.seed, config.agent.rep_probe.mode)
  if config.script != HOPPER_SCRIPT or tuple(config.logger.outputs) != ('jsonl', 'wandb'):
    raise ValueError('Repeat script/logger mismatch')
  # The existing strict scientific checker stays seed0-only. Normalize only
  # explicitly approved extension/operational fields in this checking copy.
  check(config.update({'script': 'protocol_v1', 'seed': 0, 'logger.outputs': ['jsonl']}))


def prepare(output, campaign, resources):
  if not campaign.startswith('formal-m2-hopper-seeds123-') or '/' in campaign or '\\' in campaign:
    raise ValueError('Use a new formal-m2-hopper-seeds123-* campaign')
  commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
  if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO):
    raise ValueError('Preparation requires a clean committed checkout')
  output = Path(output)
  if output.exists():
    raise FileExistsError(output)
  mapping = json.loads(Path(resources).read_text(encoding='utf-8'))
  if mapping['server'] != 'sv3' or mapping['host'] != 'lyg0326' or set(
      mapping['gpus']) != {str(row[2]) for row in ALLOCATION}:
    raise ValueError('Resources differ from user-specified sv3 GPUs1,3,4,5,6,7')
  if len({r['uuid'] for r in mapping['gpus'].values()}) != 6:
    raise ValueError('Duplicate GPU UUID')
  root = '/data/Policy_Discrepancy/runs/' + campaign
  if Path(root).exists():
    raise FileExistsError('Future campaign root must not exist')
  output.mkdir(parents=True)
  rows, checks, references = [], [], {}
  for seed in (1, 2, 3):
    rng = dict(root_seed=seed, train_env_seeds=[seed32(seed, 'train_env', i) for i in range(16)],
               replay_seed=seed32(seed, 'train_replay'),
               training_policy_and_update='separate PCG64([root_seed, own_counter])',
               report='disabled; existing SHA256(root_seed,task,report,report_id) mapping unchanged',
               evaluations=[],
               shuffle='not run in this campaign')
    for point, step in enumerate(range(0, 1000001, 50000)):
      episodes = []
      for episode in range(10):
        env_seed = seed32(seed, 'eval_env', point, episode)
        first_key = np.random.default_rng([seed, 0x4556414C, env_seed, 0]).integers(
            0, np.iinfo(np.uint32).max, (2,), np.uint32).tolist()
        episodes.append(dict(episode_id=episode, environment_seed=env_seed,
                             first_eval_policy_key=first_key))
      rng['evaluations'].append(dict(grid_index=point, action_step=step, episodes=episodes))
    dump(output / f'seed{seed}-random-streams.json', rng)
  for seed, mode, gpu in ALLOCATION:
    card = mapping['gpus'][str(gpu)]
    directory = f'{root}/seed{seed}/{mode}-attempt01'
    argv = ['--configs', 'm2_hopper_seeds123', '--task', 'dmc_hopper_hop',
            '--seed', str(seed), '--logdir', directory, '--agent.rep_probe.mode', mode,
            '--jax.prealloc', 'False', '--logger.outputs', 'jsonl', 'wandb']
    config = resolve(argv)
    check_repeat(config)
    full = dict(config.flat)
    references.setdefault(seed, full)
    differences = [key for key in full if full[key] != references[seed][key]]
    if set(differences) - {'agent.rep_probe.mode', 'logdir'}:
      raise ValueError('Undeclared within-seed group difference')
    name = f'seed{seed}-{mode}-config'
    dump(output / (name + '.json'), full)
    config.save(str(output / (name + '.yaml')))
    command = [PYTHON, '-u', '-m', 'dreamerv3.main', *argv]
    environment = dict(CUDA_VISIBLE_DEVICES=card['uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
        MUJOCO_EGL_DEVICE_ID=None, MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
        JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        PYTHONUNBUFFERED='1', TMPDIR=directory+'/tmp', WANDB_PROJECT='PD_1',
        WANDB_MODE='online', WANDB_RESUME='never', WANDB_DIR=directory,
        WANDB_RUN_GROUP=campaign, WANDB_JOB_TYPE='formal-'+('baseline' if mode == 'logging' else mode),
        WANDB_RUN_ID=f'{campaign}-hopper-seed{seed}-{mode}-01')
    rows.append(dict(run_id=f'sv3:{campaign}:seed{seed}:{mode}:attempt01',
        task='dmc_hopper_hop', group='baseline' if mode == 'logging' else mode,
        seed=seed, mode=mode, protocol=HOPPER_123, state='not_started',
        authorized_to_train=False, code_sha=commit, directory=directory,
        server='sv3', host=mapping['host'], physical_gpu=gpu, gpu_uuid=card['uuid'],
        egl_index=None, verified_compute_and_graphics_same_gpu=False,
        config=full, config_file=name+'.json', config_sha256=sha(output/(name+'.json')),
        review_argv=command, command_for_review=shlex.join(command), launch_argv=None,
        environment_template=environment,
        blockers=['explicit_training_authorization', 'fresh_assigned_gpu_all_process_check',
                  'verified_actual_cuda_egl_binding', 'clean_exact_execution_SHA',
                  'new_directory_and_dependency_disk_RAM_checks'],
        baseline_c_source=(dict(root_seed=seed, baseline_directory=directory,
            window=[100000, 300000], c=None,
            freeze_argv=[PYTHON, '-m', 'dreamerv3.freeze_c', '--baseline_dir', directory,
                         '--output', directory+'/c_frozen.json']) if mode == 'logging' else None),
        depends_on_baseline_c=False, auc_return_per_budget=None, tail_mean_return=None))
    checks.append(dict(seed=seed, mode=mode, physical_gpu=gpu,
                       configuration_check=True, allowed_group_differences=differences))
  reference = rows[0]['config']
  for row in rows:
    if {k for k in reference if reference[k] != row['config'][k]} - {
        'seed', 'logdir', 'agent.rep_probe.mode'}:
      raise ValueError('Undeclared cross-seed differences')
  dump(output/'planned-run-index.json', dict(kind='planned_not_results', protocol=HOPPER_123,
      created_utc=datetime.now(timezone.utc).isoformat(), code_sha=commit, campaign=campaign,
      root=root, authorized_to_train=False, task='dmc_hopper_hop', seeds=[1, 2, 3],
      groups=['baseline', 'dt'], run_count=6, runs=rows,
      resources_sha256=sha(resources), total_train_actions=6000000,
      evaluation_points_per_run=21, complete_episodes_per_point=10,
      total_evaluation_episodes=1260, expected_evaluation_actions=1260000,
      c_policy='own-seed full logging baseline then original successful-update active S/N; '
               'c retained for audit, no constant run in this campaign',
      original_seed0_results='preserved separately; never pooled into primary seeds1/2/3 table',
      result_audit_argv=[PYTHON, '-m', 'analysis.return_metrics', '--profile', 'hopper-seeds123',
          '--index', root+'/actual-RUN_INDEX.json', '--output', root+'/NEW_RETURN_AUDIT',
          '--require-complete']))
  dump(output/'configuration-audit.json', dict(code_sha=commit, checks=checks,
      imports_agent=False, creates_environments=False, uses_GPU=False,
      starts_training=False, source_config_sha256=sha(REPO/'dreamerv3/configs.yaml'),
      binding_pending=True, recovery='No seamless continuation; incomplete run preserved and '
      'fresh attempt after engineering repair; no return-dependent retry/selection'))
  dump(output/'artifact-hashes.json', {p.name:sha(p) for p in output.iterdir() if p.is_file()})
  print(json.dumps(dict(output=str(output), count=6, code_sha=commit, training_started=False)))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', required=True)
  parser.add_argument('--campaign', required=True)
  parser.add_argument('--resources', required=True)
  args = parser.parse_args()
  prepare(args.output, args.campaign, args.resources)
