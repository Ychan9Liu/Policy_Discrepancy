"""DL-code-r1: prepare independent short-run plans, then execute one gated run.

Default command is prepare; preparing never creates an agent or starts training.
run requires real Stage-C evidence, a separate engineering/visual acceptance,
and a fresh external sv3 GPU4--7 resource record. This is a DL runner; it never
invokes or weakens the M2-v1 configuration checker. Incomplete attempts cannot
resume because environment/replay RNG state is not completely checkpointed.
"""

import argparse
import datetime as dt
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VERSION = 'DL-code-r1'
GROUPS = ('B', 'D', 'N', 'S', 'P', 'W')
SEEDS = (0, 1, 2)
BUDGET, INTERVAL, EPISODES, ENVS = 100000, 25000, 5, 16


def digest(path):
  value = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for block in iter(lambda: stream.read(2 ** 20), b''):
      value.update(block)
  return value.hexdigest()


def git_state():
  sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
      text=True).strip()
  dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'],
      cwd=ROOT, text=True).strip())
  return sha, dirty


def safe_json(value):
  if isinstance(value, dict):
    return {str(k): safe_json(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [safe_json(x) for x in value]
  if isinstance(value, float) and not math.isfinite(value):
    return str(value)
  return value


def write_json(path, value):
  path = Path(path)
  if path.exists():
    raise FileExistsError(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(safe_json(value), indent=2, sort_keys=True,
      allow_nan=False) + '\n', encoding='utf-8')


def read_report(path):
  if not path:
    return None, None
  path = Path(path).resolve()
  report = json.loads(path.read_text(encoding='utf-8'))
  return report, dict(path=str(path), sha256=digest(path))


def calibration(report):
  if report is None:
    return None
  if report.get('calibration_episodes') != 8 or report.get('blind_episodes') != 24:
    raise ValueError('DL calibration requires the frozen 8/24 episode split')
  source = report['calibration_only']
  result = dict(kappa=source['kappa'], reward_only_lambda=source['reward_only_lambda'],
      d_bins=source['d_bins'], k_bins=source['k_bins'])
  for key in ('kappa', 'reward_only_lambda'):
    value = result[key]
    if value is None or not math.isfinite(value) or not 0 <= value <= .5:
      raise ValueError(f'Uncalibrated or illegal frozen {key}')
  for key in ('d_bins', 'k_bins'):
    edges = result[key]
    if not edges or any(not math.isfinite(x) or x < 0 for x in edges) or any(
        a >= b for a, b in zip(edges, edges[1:])):
      raise ValueError(f'Illegal frozen {key}; no default-bin fallback')
  return result


def group_overrides(group, seed, logdir, frozen=None, platform_name='cuda'):
  if group not in GROUPS or seed not in SEEDS:
    raise ValueError('DL plan requires B/D/N/S/P/W and seeds 0/1/2')
  changes = {
      'script': 'dl_short_train', 'task': 'dmc_quadruped_walk',
      'seed': seed, 'logdir': str(logdir), 'replica': 0, 'replicas': 1,
      'random_agent': False, 'batch_size': 16, 'batch_length': 64,
      'consec_train': 1, 'replay_context': 1,
      'jax.platform': platform_name, 'jax.policy_devices': [0],
      'jax.train_devices': [0], 'jax.prealloc': False, 'jax.profiler': False,
      'jax.enable_policy': True, 'logger.outputs': ['jsonl'],
      'logger.filter': '.*', 'run.action_budget': BUDGET,
      'run.eval_every_actions': INTERVAL, 'run.eval_eps': EPISODES,
      'run.envs': ENVS, 'run.eval_envs': 1, 'run.train_ratio': 256,
      'run.debug': False, 'run.engineering_fixture': False,
      'run.from_checkpoint': '', 'run.report_every_actions': 0,
      'agent.rep_probe.mode': 'dt' if group == 'D' else 'off',
      'agent.rep_probe.alpha': 20.0, 'agent.dt_latch.alpha': 20.0,
      'agent.dt_latch.rho': .5, 'agent.dt_latch.kl_tolerance': 1e-5,
      'agent.dt_latch.mode': dict(B='logging', D='logging', N='dtlatch',
          S='strength', P='conditional_shuffle', W='reward_only')[group]}
  if frozen is not None:
    if group == 'S':
      changes['agent.dt_latch.kappa'] = frozen['kappa']
    if group == 'P':
      changes.update({'agent.dt_latch.d_bins': frozen['d_bins'],
          'agent.dt_latch.k_bins': frozen['k_bins']})
    if group == 'W':
      changes['agent.dt_latch.rho'] = frozen['reward_only_lambda']
  return changes


def validate_config(config, group, frozen):
  flat = dict(config.flat) if hasattr(config, 'flat') else config
  expected = group_overrides(group, flat['seed'], flat['logdir'], frozen,
      platform_name='cuda')
  for key, value in expected.items():
    if safe_json(flat[key]) != safe_json(value):
      raise ValueError(f'DL critical configuration changed: {key}')
  required = {
      'agent.dyn.typ': 'rssm', 'agent.dyn.rssm.deter': 4096,
      'agent.dyn.rssm.hidden': 512, 'agent.dyn.rssm.stoch': 32,
      'agent.dyn.rssm.classes': 32, 'agent.dyn.rssm.unimix': .01,
      'agent.dyn.rssm.free_nats': 1.0, 'agent.loss_scales.dyn': 1.0,
      'agent.loss_scales.rep': .1, 'agent.enc.simple.depth': 32,
      'agent.dec.simple.depth': 32, 'agent.policy.units': 512,
      'agent.value.units': 512, 'agent.reward_grad': True,
      'agent.repval_grad': True, 'agent.repval_loss': True,
      'agent.ac_grads': False, 'agent.rewhead.output': 'symexp_twohot',
      'agent.policy_dist_cont': 'bounded_normal', 'env.dmc.repeat': 1,
      'env.dmc.proprio': False, 'env.dmc.image': True, 'env.dmc.use_seed': True,
      'env.dmc.camera': -1, 'env.dmc.engineering_trace_physics': False}
  for key, value in required.items():
    if flat[key] != value:
      raise ValueError(f'DL model or scientific configuration changed: {key}')
  if tuple(flat['env.dmc.size']) != (64, 64):
    raise ValueError('DL requires vision64')


def validate_stage_c(report):
  if not report or not report.get('screening', {}).get('eligible_for_short_training_review'):
    raise ValueError('Stage C is absent or did not pass the actual frozen screening')
  plan = report['score_plan']
  if (plan['task'] != 'dmc_quadruped_walk' or plan['sample_count'] != 32 or
      plan['horizon'] != 10 or plan['alpha'] != 20 or plan['rho'] != .5 or
      plan['free_nats'] != 1 or not plan.get('checkpoint_sha256') or
      int(plan['checkpoint_counters']['updates']) < 1):
    raise ValueError('Stage C source differs from the real frozen-model protocol')
  if not report.get('score_complete_sha256'):
    raise ValueError('Stage C lacks a complete scored-artifact evidence hash')
  return calibration(report)


def validate_acceptance(record, report_sha, code_sha):
  if record.get('protocol') != 'DL-protocol-r1' or record.get('git_sha') != code_sha:
    raise ValueError('DL acceptance does not cover this actual execution SHA')
  if record.get('stage_c_report_sha256') != report_sha:
    raise ValueError('DL acceptance is not bound to the selected actual report')
  for key in ('real_size50m_engineering_accepted', 'segmentation_visual_accepted',
      'stage_c_joint_accepted', 'training_control_design_accepted'):
    if record.get(key) is not True:
      raise ValueError(f'DL prerequisite remains unaccepted: {key}')
  if not record.get('evidence_paths'):
    raise ValueError('Acceptance requires auditable evidence locations')


def verify_stage_c_artifacts(report, directory=None):
  """Read actual scored evidence; report booleans alone cannot start training."""
  directory = Path(directory or report['score_plan']['config']['logdir']).resolve()
  complete_path = directory / 'score_complete.json'
  if digest(complete_path) != report['score_complete_sha256']:
    raise ValueError('Actual Stage C complete artifact differs from report')
  complete = json.loads(complete_path.read_text(encoding='utf-8'))
  plan_path = directory / 'score_plan.json'
  if digest(plan_path) != complete['plan_sha256']:
    raise ValueError('Stage C scored plan hash differs')
  if json.loads(plan_path.read_text(encoding='utf-8')) != report['score_plan']:
    raise ValueError('Report describes a different actual scored plan')
  entries = complete['artifacts']
  if len(entries) != 32 or len({x['episode'] for x in entries}) != 32 or sum(
      x['split'] == 'calibration' for x in entries) != 8:
    raise ValueError('Actual independent scored episode evidence differs from 8/24 plan')
  for entry in entries:
    artifact = (directory / entry['path']).resolve()
    if not artifact.is_relative_to(directory) or digest(artifact) != entry['sha256']:
      raise ValueError('Actual scored episode changed or escaped the evidence directory')
  return str(directory)


def validate_resource(record, environ=None, now=None):
  environ = os.environ if environ is None else environ
  now = dt.datetime.now(dt.timezone.utc) if now is None else now
  physical = record.get('physical_gpu')
  if record.get('logical_server') != 'sv3' or physical not in (4, 5, 6, 7):
    raise ValueError('Only user-assigned sv3 physical GPU4--7 is authorized')
  if record.get('occupancy') != 'idle' or record.get('exclusive') is not True:
    raise ValueError('Assigned GPU has not been verified idle and exclusive')
  checked = dt.datetime.fromisoformat(record['checked_at_utc'].replace('Z', '+00:00'))
  if checked.tzinfo is None or not -5 <= (now - checked).total_seconds() <= 120:
    raise ValueError('Resource occupancy check must be current (within 120 seconds)')
  uuid = record.get('gpu_uuid')
  if not uuid or record.get('compute_uuid') != uuid or record.get('egl_uuid') != uuid:
    raise ValueError('Actual compute/EGL physical mapping is unverified or differs')
  for variable, key in (('CUDA_VISIBLE_DEVICES', 'cuda_visible_devices'),
      ('MUJOCO_EGL_DEVICE_ID', 'mujoco_egl_device_id')):
    if not record.get(key) or environ.get(variable) != str(record[key]):
      raise ValueError(f'Execution environment differs from resource record: {variable}')


def prepare(args):
  import elements
  from ruamel.yaml import YAML
  output = Path(args.output).resolve()
  output.mkdir(parents=True, exist_ok=False)
  report, source = read_report(args.stage_c)
  frozen = calibration(report) if report is not None else None
  eligible = bool(report and report.get('screening', {}).get('eligible_for_short_training_review'))
  yaml = YAML(typ='safe')
  presets = yaml.load((ROOT / 'dreamerv3/configs.yaml').read_text(encoding='utf-8'))
  base = elements.Config(presets['defaults']).update(presets['m2_v1'])
  sha, dirty = git_state()
  runs = []
  for group in GROUPS:
    for seed in SEEDS:
      run_id = f'{group}-seed{seed}'
      run_root = (PurePosixPath(args.run_root) if str(args.run_root).startswith('/')
          else Path(args.run_root))
      logdir = run_root / run_id
      config = elements.Config({**base.flat, **group_overrides(group, seed,
          logdir, frozen, args.platform)})
      path = output / f'{run_id}.yaml'
      with path.open('w', encoding='utf-8') as stream:
        yaml.dump(dict(config.flat), stream)
      runs.append(dict(run_id=run_id, group=group, seed=seed, logdir=str(logdir),
          config_file=path.name, config_sha256=digest(path),
          status='planned_awaiting_engineering' if eligible else 'planned_waiting_stage_C'))
  index = dict(protocol='DL-protocol-r1', code_version=VERSION,
      planning_git_sha=sha, planning_checkout_dirty=dirty,
      stage_c=source, frozen_calibration=frozen, train_actions_per_run=BUDGET,
      evaluation_grid=list(range(0, BUDGET + 1, INTERVAL)), episodes_per_point=EPISODES,
      train_envs=ENVS, seeds=list(SEEDS), groups=list(GROUPS), runs=runs,
      resource_scope='sv3 physical GPU4--7, one exclusive GPU per independent run',
      limitation='Plan only. No agent constructed, GPU work, or performance evidence.')
  write_json(output / 'RUN_INDEX.json', index)
  print(str(output / 'RUN_INDEX.json'), flush=True)


def summary(evaluations, budget=BUDGET):
  points = sorted(int(x) for x in evaluations)
  if points != list(range(0, budget + 1, INTERVAL)):
    raise ValueError('Short-run evaluation grid is incomplete')
  means = [float(evaluations[x]['mean']) for x in points]
  if not all(math.isfinite(x) for x in means):
    raise ValueError('Nonfinite raw returns')
  area = sum(.5 * (a + b) * (y - x) for x, y, a, b in zip(
      points[:-1], points[1:], means[:-1], means[1:])) / budget
  return dict(return_auc_per_action_budget=area, endpoint_return_raw=means[-1],
      grid=points, mean_returns_raw=means, budget=budget,
      limitation='DL short-budget exploratory metrics, not M2-v1 formal AUC/tail.')


def run(args):
  index_path = Path(args.index).resolve()
  index = json.loads(index_path.read_text(encoding='utf-8'))
  selected = [x for x in index['runs'] if x['run_id'] == args.run_id]
  if len(selected) != 1 or index.get('protocol') != 'DL-protocol-r1':
    raise ValueError('Unknown or duplicate DL run identity')
  entry = selected[0]
  sha, dirty = git_state()
  if dirty or index['planning_checkout_dirty'] or sha != index['planning_git_sha']:
    raise RuntimeError('DL execution requires the clean, exact prepared Git SHA')
  report, report_source = read_report(args.stage_c)
  frozen = validate_stage_c(report)
  score_artifacts = verify_stage_c_artifacts(report, args.score_artifacts)
  if (not index['stage_c'] or index['stage_c']['sha256'] != report_source['sha256']
      or index['frozen_calibration'] != frozen):
    raise ValueError('Prepared calibration/report changed; make a new plan')
  acceptance = json.loads(Path(args.acceptance).read_text(encoding='utf-8'))
  validate_acceptance(acceptance, report_source['sha256'], sha)
  resource = json.loads(Path(args.resource_record).read_text(encoding='utf-8'))
  validate_resource(resource)
  from scripts.dl_resources import require_resource
  resource = require_resource(args.resource_record)
  config_path = index_path.parent / entry['config_file']
  if digest(config_path) != entry['config_sha256']:
    raise ValueError('Frozen run configuration changed')
  import multiprocessing as mp
  if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn')
  if mp.get_start_method() != 'spawn':
    raise RuntimeError('DL environment workers require spawn; refusing to fork JAX runtime')
  import elements
  import embodied
  import jax
  from ruamel.yaml import YAML
  from dreamerv3 import main
  from embodied.run.protocol_v1 import (ProtocolState, EpisodeReturn,
      parameter_digest, package_versions, seed32, training_fingerprint)
  config = elements.Config(YAML(typ='safe').load(config_path.read_text(encoding='utf-8')))
  validate_config(config, entry['group'], frozen)
  logdir = Path(config.logdir)
  if not logdir.as_posix().startswith('/data/Policy_Discrepancy/runs/'):
    raise ValueError('Actual DL training artifacts must be under the assigned server runs root')
  logdir.mkdir(parents=True, exist_ok=False)  # Never recover/splice an old attempt.
  (logdir / 'config.yaml').write_bytes(config_path.read_bytes())
  agent = replay = logger = driver = None
  started = time.perf_counter()
  state = ProtocolState(BUDGET, INTERVAL, EPISODES, 0, BUDGET)
  try:
    agent = main.make_agent(config)
    if int(agent.n_updates) or int(agent.n_actions):
      raise RuntimeError('DL short runs must start from newly initialized parameters')
    replay_seed = seed32(config.seed, config.task, 'DL_train_replay')
    replay = main.make_replay(config, 'replay', seed=replay_seed)
    logger = main.make_logger(config)
    step = logger.step
    cp = elements.Checkpoint(elements.Path(str(logdir / 'ckpt')))
    cp.step, cp.agent, cp.replay, cp.protocol = step, agent, replay, state
    cp.save()
    manifest = dict(protocol='DL-protocol-r1', code_version=VERSION, group=entry['group'],
        git_sha=sha, config=dict(config.flat), config_sha256=entry['config_sha256'],
        index_sha256=digest(index_path), root_seed=config.seed,
        stage_c=report_source, frozen_calibration=frozen,
        verified_score_artifacts=score_artifacts,
        acceptance_sha256=digest(args.acceptance), resource=resource,
        resource_record_sha256=digest(args.resource_record), actual_hostname=platform.node(),
        python=sys.version, jax=jax.__version__, devices=[str(x) for x in jax.devices()],
        packages=package_versions('dm-control', 'mujoco', 'numpy', 'jaxlib'),
        dependency_lock_sha256=digest(ROOT / 'env/requirements-dreamer.lock'),
        replay_seed=replay_seed, evaluation_policy='sampled independent eval RNG',
        train_env_seeds=[seed32(config.seed, config.task, 'DL_train_env', i)
            for i in range(ENVS)],
        restoration='No resumption: environment/replay RNG not fully checkpointed',
        matched_forward_warning='Zero kappa/lambda static bypass may have less overhead; report actual cost.')
    write_json(logdir / 'DL_MANIFEST.json', manifest)
    batch_steps = config.batch_size * config.batch_length
    should_train = elements.when.Ratio(config.run.train_ratio / batch_steps)
    stream = iter(main.make_stream(config, replay, 'train'))
    carry = [agent.init_train(config.batch_size)]
    agg = elements.Agg()
    def on_transition(tran, worker):
      state.on_train_transition(tran)
      if not tran['is_first']:
        step.increment()
      replay.add(tran, worker)
      if len(replay) < batch_steps:
        return
      for _ in range(should_train(step)):
        update_id, action_step = state.updates, state.actions
        batch = agent.prepare_batch(next(stream))
        carry[0], delayed_outs, delayed_metrics = agent.train(carry[0], dict(batch))
        if delayed_outs or delayed_metrics:
          raise RuntimeError('Unconsumed asynchronous training result')
        outs, metrics = agent.take_train_result()
        finite = all(math.isfinite(float(metrics[k])) for k in ('opt/grad_norm', 'opt/loss'))
        if int(metrics['opt/updates']) != update_id + 1:
          raise RuntimeError('Optimizer did not commit the expected update')
        state.record_update(update_id, action_step, 0, 0, finite=finite)
        if int(agent.n_updates) != state.updates:
          raise RuntimeError('Agent update counter and persisted receipt disagree')
        receipt = dict(update_id=update_id, start_action=action_step,
            match_sum=0.0, match_count=0, invalid_count=0,
            metrics={k: float(v) for k, v in metrics.items()
                if k.startswith(('dl/', 'dt/')) and getattr(v, 'ndim', 0) == 0})
        with (logdir / 'update_receipts.jsonl').open('a', encoding='utf-8') as file:
          file.write(json.dumps(safe_json(receipt), sort_keys=True, allow_nan=False) + '\n')
        if 'replay' in outs:
          replay.update(outs['replay'])
        agg.add(metrics, prefix='train')
    fns = [partial(main.make_env, config, i, seed=seed32(
        config.seed, config.task, 'DL_train_env', i)) for i in range(ENVS)]
    driver = embodied.Driver(fns, parallel=True)
    driver.on_step(on_transition)
    driver.reset(agent.init_policy)
    train_policy = lambda *xs: agent.policy(*xs, mode='train')
    def evaluate(point):
      before = training_fingerprint(agent, replay)
      agent.sync_policy()
      directory = logdir / 'eval_snapshots' / f'{point:07d}'
      snapshot = elements.Checkpoint(elements.Path(str(directory)))
      snapshot.agent = agent
      snapshot.save()
      snapshot_id = str(directory.relative_to(logdir)) + '/' + (directory / 'latest').read_text().strip()
      write_json(directory / 'evaluation_snapshot.json', dict(snapshot_id=snapshot_id,
          action_step=state.actions, update_id=state.updates, git_sha=sha))
      details, scores, lengths, seeds = [], [], [], []
      for episode_id in range(EPISODES):
        seed = seed32(config.seed, config.task, 'DL_eval_env', point // INTERVAL, episode_id)
        evaluator = embodied.Driver([partial(main.make_env, config, 0, seed=seed)], parallel=False)
        result = EpisodeReturn()
        def on_eval(tran, _):
          if not tran['is_first']:
            state.eval_actions += 1
          result.add(tran)
        evaluator.on_step(on_eval)
        try:
          agent.set_eval_seed(seed)
          evaluator.reset(lambda count: agent.init_policy(count, mode='eval'))
          evaluator(lambda *xs: agent.policy(*xs, mode='eval'), episodes=1)
        finally:
          evaluator.close()
        item = dict(result.result(), episode_id=episode_id, environment_seed=seed)
        details.append(item)
        scores.append(item['return_raw'])
        lengths.append(item['length'])
        seeds.append(seed)
      after = training_fingerprint(agent, replay)
      if before != after:
        raise RuntimeError('DL evaluation changed training parameters/counters/replay state')
      state.record_evaluation(point, scores, lengths, seeds,
          episode_details=details, snapshot_id=snapshot_id)
      with (logdir / 'evaluations.jsonl').open('a', encoding='utf-8') as file:
        file.write(json.dumps(dict(action_step=point, **state.evaluations[point]),
            sort_keys=True, allow_nan=False) + '\n')
      with (logdir / 'evaluation_state_audit.jsonl').open('a', encoding='utf-8') as file:
        file.write(json.dumps(dict(action_step=point, before=before, after=after),
            sort_keys=True) + '\n')
      logger.add(dict(return_raw=state.evaluations[point]['mean'], action_steps=state.actions), prefix='DL_eval')
      logger.add(agg.result())
      logger.write()
      cp.save()
    evaluate(0)
    while state.actions < BUDGET:
      boundary = state.next_boundary()
      driver.step_selected(train_policy, range(min(ENVS, boundary - state.actions)))
      if state.actions == boundary:
        evaluate(boundary)
    if state.actions != BUDGET or int(step) != BUDGET or len(state.evaluations) != 5:
      raise RuntimeError('DL exact budget or complete evaluation grid not satisfied')
    result = dict(protocol='DL-protocol-r1', git_sha=sha, group=entry['group'], seed=config.seed,
        status='training_ended_pending_audit', train_action_steps=state.actions,
        train_resets=state.resets, updates=state.updates, eval_action_steps=state.eval_actions,
        evaluations=len(state.evaluations), params_sha256=parameter_digest(agent),
        elapsed_seconds=time.perf_counter() - started, **summary(state.evaluations))
    write_json(logdir / 'final_state.json', result)
    cp.save()
    print(json.dumps(result, sort_keys=True), flush=True)
  except BaseException as error:
    write_json(logdir / 'FAILED.json', dict(status='failed', type=type(error).__name__,
        message=str(error), train_action_steps=state.actions, updates=state.updates,
        elapsed_seconds=time.perf_counter() - started, git_sha=sha,
        continuation='Forbidden. Keep evidence and use a new attempt directory/index.'))
    raise
  finally:
    if driver is not None:
      driver.close()
    if logger is not None:
      logger.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest='stage')
  plan = sub.add_parser('prepare')
  plan.add_argument('--output', required=True)
  plan.add_argument('--run-root', required=True)
  plan.add_argument('--stage-c')
  plan.add_argument('--platform', choices=('cpu', 'cuda'), default='cuda')
  execute = sub.add_parser('run')
  for argument in ('index', 'run-id', 'stage-c', 'acceptance', 'resource-record'):
    execute.add_argument('--' + argument, required=True)
  execute.add_argument('--score-artifacts', help='Actual scored files, if relocated from source logdir')
  argv = sys.argv[1:]
  if not argv or argv[0] not in ('prepare', 'run', '-h', '--help'):
    argv = ['prepare', *argv]
  args = parser.parse_args(argv)
  (run if args.stage == 'run' else prepare)(args)


if __name__ == '__main__':
  main()
