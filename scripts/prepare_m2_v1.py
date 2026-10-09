"""Resolve the frozen formal plan without importing an agent or starting runs.

Requires the existing dreamer environment's elements, numpy and ruamel.yaml.
Only writes a new analysis directory. Constant c remains explicitly unbound.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from datetime import datetime, timezone

import elements
import numpy as np
from ruamel.yaml import YAML


REPO = Path(__file__).resolve().parents[1]
TASKS = ('dmc_hopper_hop', 'dmc_quadruped_run',
         'dmc_quadruped_walk', 'dmc_reacher_hard')
MODES = ('logging', 'dt', 'constant', 'shuffle')
PYTHON = '/data/Policy_Discrepancy/envs/dreamer/bin/python'
COMMON = {'jax.prealloc': False, 'logger.outputs': ['jsonl']}
GRID = list(range(0, 1000001, 50000))


def dump(path, data):
  Path(path).write_text(json.dumps(data, sort_keys=True, indent=2,
                                  allow_nan=True) + '\n', encoding='utf-8')


def sha(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed32(task, namespace, *indices):
  parts = ['0', task, namespace, *(str(x) for x in indices)]
  return int.from_bytes(hashlib.sha256('\0'.join(parts).encode()).digest()[:4],
                        'big')


def resolve(argv):
  presets = YAML(typ='safe').load((REPO / 'dreamerv3/configs.yaml').read_text())
  parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
  config = elements.Config(presets['defaults'])
  for name in parsed.configs:
    config = config.update(presets[name])
  return elements.Flags(config).parse(other)


def check(config):
  expected = {
      'script': 'protocol_v1', 'seed': 0,
      'run.action_budget': 1000000, 'run.eval_every_actions': 50000,
      'run.match_start': 100000, 'run.match_end': 300000, 'run.eval_eps': 10,
      'run.eval_envs': 1, 'run.envs': 16, 'run.train_ratio': 256,
      'run.report_every_actions': 0, 'run.engineering_fixture': False,
      'run.engineering_stop_after_actions': 0,
      'run.engineering_trace_actions': 0, 'run.engineering_trace_updates': 0,
      'run.from_checkpoint': '', 'run.remote_replay': False,
      'env.dmc.engineering_trace_physics': False, 'env.dmc.repeat': 1,
      'env.dmc.image': True, 'env.dmc.proprio': False,
      'env.dmc.use_seed': True, 'env.dmc.size': (64, 64), 'env.dmc.camera': -1,
      'agent.rep_probe.alpha': 20, 'agent.dyn.rssm.free_nats': 1,
      'agent.loss_scales.dyn': 1, 'agent.loss_scales.rep': .1,
      'agent.ac_grads': False, 'agent.reward_grad': True,
      'agent.repval_grad': True, 'agent.repval_loss': True,
      'agent.dyn.rssm.deter': 4096, 'agent.dyn.rssm.hidden': 512,
      'agent.dyn.rssm.classes': 32, 'agent.enc.simple.depth': 32,
      'agent.dec.simple.depth': 32, 'agent.policy.units': 512,
      'agent.value.units': 512, 'batch_size': 16, 'batch_length': 64,
      'replay_context': 1, 'consec_train': 1, 'replay.size': 5000000,
      'replay.chunksize': 1024, 'replay.online': True,
      'replay.fracs.uniform': 1, 'replay.fracs.priority': 0,
      'replay.fracs.recency': 0, 'jax.platform': 'cuda',
      'jax.compute_dtype': 'bfloat16', 'jax.train_devices': (0,),
      'jax.policy_devices': (0,), 'jax.jit': True, 'jax.debug': False,
      'run.debug': False, 'jax.prealloc': False,
      'logger.outputs': ('jsonl',), 'random_agent': False,
  }
  flat = dict(config.flat)
  for key, value in expected.items():
    if flat[key] != value:
      raise ValueError(f'Frozen config mismatch: {key}: {flat[key]} != {value}')
  if flat['task'] not in TASKS or flat['agent.rep_probe.mode'] not in MODES:
    raise ValueError('Unknown frozen task/mode')


def prepare(output, campaign, resources):
  if not campaign.startswith('formal-m2-v1-') or '/' in campaign:
    raise ValueError('Use an explicit formal-m2-v1-* campaign ID')
  source_sha = subprocess.check_output(
      ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
  dirty = subprocess.check_output(
      ['git', 'status', '--porcelain'], cwd=REPO, text=True).strip()
  if dirty:
    raise ValueError('Preparation requires a clean pinned Git checkout')
  output = Path(output)
  if output.exists():
    raise FileExistsError(output)
  output.mkdir(parents=True)
  assignments = json.loads(Path(resources).read_text(encoding='utf-8'))
  if set(assignments['tasks']) != set(TASKS):
    raise ValueError('Need exactly the four task resource mappings')
  cards = [(x['server'], x['gpu_uuid']) for x in assignments['tasks'].values()]
  if len(set(cards)) != 4:
    raise ValueError('The four concurrent task slots must use distinct GPUs')
  for mapping in assignments['tasks'].values():
    if not mapping['verified_compute_and_graphics_same_gpu']:
      raise ValueError('Unverified CUDA/EGL assignment')
  root = '/data/Policy_Discrepancy/runs/' + campaign
  runs, checks = [], []
  for task in TASKS:
    mapping = assignments['tasks'][task]
    seeds = {
        'root_seed': 0,
        'derivation': 'first 4 big-endian bytes of SHA256(NUL-joined strings)',
        'train_env_seeds': [seed32(task, 'train_env', i) for i in range(16)],
        'replay_seed': seed32(task, 'train_replay'),
        'parameter_initialization_key': [0, 0],
        'train_policy_and_update': 'PCG64([0, own_call_or_batch_counter])',
        'first_train_counter_key': np.random.default_rng([0, 0]).integers(
            0, np.iinfo(np.uint32).max, (2,), np.uint32).tolist(),
        'report': 'disabled; seed32(task, report, report_id) if separately audited',
        'shuffle': 'PRNGKey(0); fold_in(0x4454); fold_in(optimizer_step)',
        'evaluations': [],
    }
    for grid_id, step in enumerate(GRID):
      episodes = []
      for episode in range(10):
        seed = seed32(task, 'eval_env', grid_id, episode)
        key = np.random.default_rng([0, 0x4556414C, seed, 0]).integers(
            0, np.iinfo(np.uint32).max, (2,), np.uint32).tolist()
        episodes.append(dict(episode_id=episode, environment_seed=seed,
                             first_eval_policy_key=key))
      seeds['evaluations'].append(dict(grid_index=grid_id, action_step=step,
                                       episodes=episodes))
    dump(output / (task + '-seeds.json'), seeds)
    reference = None
    for mode in MODES:
      directory = root + '/' + task + '/' + mode
      baseline = root + '/' + task + '/logging'
      argv = ['--configs', 'm2_v1', '--task', task, '--logdir', directory,
              '--agent.rep_probe.mode', mode,
              '--jax.prealloc', 'False', '--logger.outputs', 'jsonl']
      config = resolve(argv)
      check(config)
      full = dict(config.flat)
      if reference is None:
        reference = full.copy()
      differences = sorted(k for k in full if full[k] != reference[k])
      allowed = {'logdir', 'agent.rep_probe.mode'}
      if not set(differences).issubset(allowed):
        raise ValueError(f'Undeclared group differences: {differences}')
      complete_config = mode != 'constant'
      if not complete_config:
        full['agent.rep_probe.c'] = None
        full['run.frozen_c_file'] = baseline + '/c_frozen.json'
      config_file = task + '-' + mode + (
          '-config.json' if complete_config else '-config.pending.json')
      dump(output / config_file, full)
      if complete_config:
        config.save(str(output / (task + '-' + mode + '-config.yaml')))
      env = dict(CUDA_VISIBLE_DEVICES=mapping['gpu_uuid'],
          CUDA_DEVICE_ORDER='PCI_BUS_ID',
          MUJOCO_EGL_DEVICE_ID=str(mapping['egl_index']), MUJOCO_GL='egl',
          JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false',
          PYOPENGL_PLATFORM='egl', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
          OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', TMPDIR=directory+'/tmp')
      command = [PYTHON, '-u', '-m', 'dreamerv3.main', *argv]
      template = command + ['--agent.rep_probe.c', '${VERIFIED_FORMAL_C}',
                           '--run.frozen_c_file', baseline+'/c_frozen.json']
      run = dict(run_id=mapping['server']+':'+campaign+':'+task+':'+mode,
          task=task, group='baseline' if mode == 'logging' else mode,
          mode=mode, seed=0, server=mapping['server'], host=mapping['host'],
          gpu_index=mapping['gpu_index'], gpu_uuid=mapping['gpu_uuid'],
          egl_index=mapping['egl_index'], binding_evidence=mapping['evidence'],
          binding_evidence_sha256=mapping['evidence_sha256'],
          directory=directory, source_directory=directory,
          status='not_started', authorized=False, execution_sha=source_sha,
          configuration_complete=complete_config,
          config_file=config_file, config_sha256=sha(output/config_file),
          environment=env, cwd='/data/Policy_Discrepancy/repo',
          launch_argv=command if complete_config else None,
          command_for_review=shlex.join(command) if complete_config else None,
          launch_argv_template=template if not complete_config else None,
          prerequisites=['explicit_user_formal_authorization',
              'clean_exact_SHA', 'fresh_compute_and_graphics_job_check',
              'binding_recheck', 'disk_and_RAM_margin', 'new_logdir'],
          completion=dict(train_action_steps=1000000, evaluation_points=GRID,
              episodes_per_point=10, snapshots=21, episode_count=210),
          formal_auc=None, formal_tail=None,
          c_dependency=dict(baseline_dir=baseline, file=baseline+'/c_frozen.json',
              value=None, sha256=None, window=[100000, 300000],
              freeze_argv=[PYTHON, '-m', 'dreamerv3.freeze_c',
                           '--baseline_dir', baseline, '--output',
                           baseline+'/c_frozen.json']))
      if mode != 'logging':
        run['prerequisites'] += ['this_task_full_baseline_and_audited_formal_c']
      runs.append(run)
      checks.append(dict(task=task, mode=mode,
          differing_keys=differences + (['agent.rep_probe.c',
              'run.frozen_c_file'] if mode == 'constant' else []),
          scientific_config_check=True, unresolved=('formal_c' if mode ==
                                                    'constant' else None)))
  dump(output/'planned-run-index.json', dict(kind='planned_not_results',
      created_utc=datetime.now(timezone.utc).isoformat(), campaign=campaign,
      code_sha=source_sha, authorized=False, run_count=16, runs=runs,
      resource_input_sha256=sha(resources), config_source_sha256=sha(
          REPO/'dreamerv3/configs.yaml'), shared_operational_overrides=COMMON,
      schedule='four task baselines; audit/freeze each; dt, constant, shuffle waves',
      recovery='no incomplete checkpoint continuation; preserve failure, new run ID',
      analysis='mirror actual small files and make a new result index after runs'))
  dump(output/'configuration-audit.json', dict(code_sha=source_sha,
      training_launched=False, runs=checks, resolved=12, pending_formal_c=4,
      imports_training_code=False,
      note='JSON retains preset Infinity for inactive priority.initial; not NaN loss'))
  dump(output/'artifact-hashes.json', {p.name: sha(p) for p in output.iterdir()
                                      if p.is_file()})
  print(json.dumps(dict(output=str(output), code_sha=source_sha,
                       count=16, training_launched=False)))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', required=True)
  parser.add_argument('--campaign', required=True)
  parser.add_argument('--resources', required=True)
  args = parser.parse_args()
  prepare(args.output, args.campaign, args.resources)
