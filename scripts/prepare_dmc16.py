"""CPU-only preparation of the remaining DMC16 baseline/Dt matrix.

There is deliberately no launch action or idle-GPU watcher. The user paused real
tests and training after the resource conflict; this plan cannot authorize runs.
Unrelated dirty documents are retained, while runtime source must match HEAD.
"""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

import numpy as np

from dreamerv3.protocol_scope import (
    DMC16, DMC16_SCRIPT, DMC16_TASKS, V1_TASKS, validate_scope)
from scripts.prepare_m2_v1 import check, resolve, seed32

REPO = Path(__file__).resolve().parents[1]
PYTHON = '/data/Policy_Discrepancy/envs/dreamer/bin/python'
MODES = ('logging', 'dt')
ENTITY = 'ychan9liu-tsinghua-university'
SCORE_ALIASES = {'dmc_cup_catch': 'dmc_ball_in_cup_catch'}
SOURCE_PATHS = ('dreamerv3', 'embodied', 'scripts', 'analysis/return_metrics.py')


def dump(path, value):
  Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def sha(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_task_inventory():
  source = REPO/'scores/dmc_vision-dreamerv3.json.gz'
  tasks = {r['task'] for r in json.loads(gzip.decompress(source.read_bytes()))}
  expected = {SCORE_ALIASES.get(t, t) for t in DMC16_TASKS} | set(V1_TASKS)
  if len(tasks) != 20 or tasks != expected or set(DMC16_TASKS) & set(V1_TASKS):
    raise ValueError('DMC16 must be exactly the original visual20 minus the four tested tasks')
  return dict(source=str(source), sha256=sha(source), visual_tasks=sorted(tasks),
              excluded_tasks=list(V1_TASKS), runtime_to_score_aliases=SCORE_ALIASES)


def check_dmc16(config):
  validate_scope(config.script, config.task, config.seed, config.agent.rep_probe.mode)
  if config.script != DMC16_SCRIPT or tuple(config.logger.outputs) != ('jsonl', 'wandb'):
    raise ValueError('DMC16 script/logger mismatch')
  # Validate all original numerical settings with the unchanged strict checker.
  # This checking copy alone normalizes explicitly different scope/task/logger.
  check(config.update({'script': 'protocol_v1', 'task': V1_TASKS[0],
                       'logger.outputs': ['jsonl']}))


def allocation():
  return [(task, mode, (i % 4)*2+j, i//4+1)
          for i, task in enumerate(DMC16_TASKS) for j, mode in enumerate(MODES)]


def validate_plan(plan):
  """Check a review plan; validity never means training authorization."""
  expected = set(allocation())
  rows = plan['runs']
  actual = {(r['task'], r['mode'], r['physical_gpu'], r['queue_position']) for r in rows}
  if (plan['protocol'] != DMC16 or len(rows) != 32 or actual != expected or
      plan['authorized_to_train'] or plan['auto_start_when_idle']):
    raise ValueError('Expected a paused, exact 16x2 preparation matrix')
  for row in rows:
    if (row['server'] != 'sv1' or type(row['seed']) is not int or row['seed'] != 0 or
        row['authorized_to_train'] or row['launch_argv'] is not None or
        row['state'] != 'not_started_user_paused'):
      raise ValueError('Run is not a paused sv1 seed0 preparation')
    validate_scope(DMC16_SCRIPT, row['task'], row['seed'], row['mode'])
    check_dmc16(resolve(row['review_argv'][4:]))
    if json.loads(json.dumps(dict(resolve(row['review_argv'][4:]).flat))) != row['config']:
      raise ValueError('Command and resolved configuration differ')
  if len({r['directory'] for r in rows}) != 32 or len({r['run_id'] for r in rows}) != 32:
    raise ValueError('Run paths and IDs must be unique')


def source_state():
  commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
  if subprocess.check_output(['git', 'diff', 'HEAD', '--', *SOURCE_PATHS], cwd=REPO):
    raise ValueError('Commit runtime/preparation code before generating a pinned plan')
  if subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard',
                              '--', *SOURCE_PATHS], cwd=REPO):
    raise ValueError('Uncommitted runtime source is not a pinned preparation')
  dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)
  return commit, dirty


def prepare(output, campaign, resources):
  if (not campaign.startswith('formal-m2-dmc16-') or
      any(c in campaign for c in '/\\') or '..' in campaign):
    raise ValueError('Use a new formal-m2-dmc16-* campaign ID')
  output = Path(output)
  if output.exists():
    raise FileExistsError(output)
  inventory = verify_task_inventory()
  mapping = json.loads(Path(resources).read_text(encoding='utf-8'))
  if (mapping['server'] != 'sv1' or not mapping['host'] or
      set(mapping['gpus']) != {str(i) for i in range(8)} or
      len({r['uuid'] for r in mapping['gpus'].values()}) != 8):
    raise ValueError('Need exactly sv1 physical GPUs0-7, with eight distinct UUIDs')
  commit, dirty = source_state()
  root = '/data/Policy_Discrepancy/runs/' + campaign
  output.mkdir(parents=True)
  dump(output/'task-inventory.json', inventory)
  for task in DMC16_TASKS:
    rng = dict(root_seed=0, task=task,
        derivation='first four big-endian bytes of SHA256(NUL-joined strings)',
        train_env_seeds=[seed32(task, 'train_env', i) for i in range(16)],
        replay_seed=seed32(task, 'train_replay'), initialization_key=[0, 0],
        training_policy_and_update='independent PCG64([root_seed, own_counter])',
        report='disabled; existing seed32(root_seed,task,report,id) mapping',
        shuffle='not used in this matrix', evaluations=[])
    for point, step in enumerate(range(0, 1000001, 50000)):
      episodes = []
      for episode in range(10):
        env_seed = seed32(task, 'eval_env', point, episode)
        key = np.random.default_rng([0, 0x4556414C, env_seed, 0]).integers(
            0, np.iinfo(np.uint32).max, (2,), np.uint32).tolist()
        episodes.append(dict(episode_id=episode, environment_seed=env_seed,
                             first_eval_policy_key=key))
      rng['evaluations'].append(dict(grid_index=point, action_step=step, episodes=episodes))
    dump(output/(task+'-random-streams.json'), rng)
  rows = []
  for task, mode, gpu, position in allocation():
    directory = f'{root}/{task}/{mode}-attempt01'
    argv = ['--configs', 'm2_dmc16', '--task', task, '--seed', '0',
        '--logdir', directory, '--agent.rep_probe.mode', mode,
        '--jax.prealloc', 'False', '--logger.outputs', 'jsonl', 'wandb']
    config = resolve(argv)
    check_dmc16(config)
    name = task+'-'+mode+'-config'
    dump(output/(name+'.json'), dict(config.flat))
    config.save(str(output/(name+'.yaml')))
    environment = dict(CUDA_VISIBLE_DEVICES=mapping['gpus'][str(gpu)]['uuid'],
        CUDA_DEVICE_ORDER='PCI_BUS_ID', MUJOCO_EGL_DEVICE_ID=None,
        MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', JAX_PLATFORMS='cuda',
        XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1',
        TMPDIR=directory+'/tmp', WANDB_ENTITY=ENTITY, WANDB_PROJECT='PD_1',
        WANDB_MODE='online', WANDB_RESUME='never', WANDB_DIR=directory,
        WANDB_RUN_GROUP=campaign, WANDB_JOB_TYPE='formal-'+('baseline' if mode=='logging' else mode),
        WANDB_RUN_ID=f'{campaign}-{task[4:]}-seed0-{mode}-01')
    review = [PYTHON, '-u', '-m', 'dreamerv3.main', *argv]
    rows.append(dict(run_id=f'sv1:{campaign}:{task}:seed0:{mode}:attempt01',
        task=task, score_task=SCORE_ALIASES.get(task,task), mode=mode,
        group='baseline' if mode=='logging' else mode, seed=0, protocol=DMC16,
        server='sv1', host=mapping['host'], physical_gpu=gpu,
        gpu_uuid=mapping['gpus'][str(gpu)]['uuid'], queue_position=position,
        directory=directory, code_sha=commit, state='not_started_user_paused',
        authorized_to_train=False, launch_argv=None, review_argv=review,
        command_for_review=shlex.join(review), config=dict(config.flat),
        config_file=name+'.json', config_sha256=sha(output/(name+'.json')),
        environment_template=environment, egl_index=None,
        verified_compute_and_graphics_same_gpu=False, depends_on_baseline_c=False,
        baseline_c_source=(dict(root_seed=0, baseline_directory=directory,
            window=[100000,300000], c=None, freeze_argv=[PYTHON,'-m','dreamerv3.freeze_c',
                '--baseline_dir',directory,'--output',directory+'/c_frozen.json'])
                if mode=='logging' else None),
        auc_return_per_budget=None, tail_mean_return=None,
        blockers=['user_resume_required_after_pause','fresh_gpu_no_C_or_G_jobs',
            'fresh_host_UUID_and_actual_CUDA_EGL_binding','clean_pinned_server_source',
            'real_task_interfaces_and_size50m_CUDA_acceptance',
            'online_wandb_and_dependency_validation','disk_RAM_and_new_run_root_check']))
  # JSON normalization makes elements' tuple/list representation portable.
  rows = json.loads(json.dumps(rows))
  for task in DMC16_TASKS:
    a,b = [r['config'] for r in rows if r['task']==task]
    if {k for k in a if a[k]!=b[k]} != {'agent.rep_probe.mode','logdir'}:
      raise ValueError('Undeclared paired-group configuration difference')
  plan = dict(protocol=DMC16, campaign=campaign, root=root, code_sha=commit,
      kind='planned_not_results', created_utc=datetime.now(timezone.utc).isoformat(),
      authorized_to_train=False, auto_start_when_idle=False,
      authorization_status='user_paused_real_tests_and_training; preparation only',
      run_count=32, tasks=list(DMC16_TASKS), seeds=[0], groups=['baseline','dt'],
      runs=rows, resource_snapshot_sha256=sha(resources),
      local_unrelated_worktree_dirty=bool(dirty), local_worktree_status=dirty,
      server_checkout_requirement='clean exact SHA; local unrelated drafts are not deployed',
      schedule='8 fixed physical-GPU queues, 4 runs each; at most one run per GPU; no idle watcher',
      total_train_actions=32000000, evaluation_points_per_run=21,
      complete_episodes_per_point=10, total_evaluation_episodes=6720,
      c_policy='own-task seed0 completed logging baseline active S/N, audit only; no constant or shuffle',
      analysis_scope='separate 16-task single-seed descriptive cohort; no automatic 20-task pooling or strong M2 attribution',
      result_audit_argv=[PYTHON,'-m','analysis.return_metrics','--profile','dmc16',
          '--index',root+'/actual-RUN_INDEX.json','--output',root+'/NEW_RETURN_AUDIT','--require-complete'])
  validate_plan(plan)
  dump(output/'planned-run-index.json', plan)
  dump(output/'gpu-queues.json', {str(gpu):[dict(task=r['task'],mode=r['mode'],
      queue_position=r['queue_position'],run_id=r['run_id']) for r in rows
      if r['physical_gpu']==gpu] for gpu in range(8)})
  dump(output/'configuration-audit.json', dict(code_sha=commit, resolved_configs=32,
      paired_differences=['agent.rep_probe.mode','logdir'], training_actions=0,
      agent_updates=0, creates_environments=False, uses_GPU=False,
      training_authorized=False, task_inventory_verified=True,
      source_config_sha256=sha(REPO/'dreamerv3/configs.yaml')))
  dump(output/'artifact-hashes.json', {p.name:sha(p) for p in output.iterdir() if p.is_file()})
  print(json.dumps(dict(output=str(output),code_sha=commit,count=32,training_started=False)))


if __name__ == '__main__':
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',required=True)
  parser.add_argument('--campaign',required=True)
  parser.add_argument('--resources',required=True)
  args=parser.parse_args()
  prepare(args.output,args.campaign,args.resources)
