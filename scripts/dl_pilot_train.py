"""DL-pilot-r3: user-corrected 1M online DtLatch exploration.

The original DL screening gates and M2-v1 runner remain unchanged. Only N,
quadruped_walk, seed 0 is trained; pinned PD_1 B/D complete runs are historical
controls. Preparing imports no agent. The supervisor owns one Linux process
group and bounds the entire attempt, including initialization and evaluation.
"""
import time
ENTRY_STARTED = time.monotonic()
import os
import signal
OWNED = set()
SECONDS = 57600
DEADLINE = ENTRY_STARTED + SECONDS


def expire(signum, frame):
  for pid in tuple(OWNED):
    try:
      os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
      pass
  raise TimeoutError('Whole DL pilot entry expired or was terminated')


if __name__ == '__main__' and os.name == 'posix':
  signal.signal(signal.SIGALRM, expire)
  signal.signal(signal.SIGTERM, expire)
  signal.setitimer(signal.ITIMER_REAL, max(.001, DEADLINE - time.monotonic()))

import argparse
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.dl_train import digest, git_state, safe_json, write_json

VERSION = 'DL-pilot-r3'
TASK, SEED, BUDGET, INTERVAL, EPISODES, ENVS = (
    'dmc_quadruped_walk', 0, 1000000, 50000, 10, 16)
HOST = 'lyg0360'
UUIDS = {
    4: 'GPU-7bf636ec-fa03-aa3f-7314-dd803c1658be',
    5: 'GPU-ac733835-08ee-7a17-a2fa-1e49d62d385b',
    6: 'GPU-548be1c8-86f0-b7a2-de92-5c455b12221b',
    7: 'GPU-4cbf5b69-adb3-5210-2396-ca5d4e400f26'}
RUNS_ROOT = Path('/data/Policy_Discrepancy/runs')
SOURCE_GIT = '48d71f1541721208322442f5a1d5977124f3e888'
LOCK_SHA = 'cd50c7936690f9adbc5c35f99b3df2f7a977cef3222582f988979b11f65b6f22'
INITIAL_PARAMS = '18199ec5fb45b13f18ec29ae6dcc438d3018807888ce9f2c5e6ee64b621a9267'
INITIAL_CHECKPOINT = '6ad6dfae1a8b6a160b877eac9e663f12dbe232a0d0e4daae01911bf1d9661f94'
CONTROLS = {
    'B': ('formal-m2-v1-pd1-seed0-20261009/dmc_quadruped_walk/logging-attempt01', {
        'protocol_manifest.json': 'c304d77a03515b34aafc8bd4e9d613ed005dbe4be127ee789e0b49bf17d43af4',
        'evaluations.jsonl': 'e7cbd36ccf3a27537b12c8f52731c99df475799cb4a7fb2513b5491eb252db4c',
        'final_state.json': '04c37ec98512a5cda54d16fb85c9837a1d52ce8b90294c67b8d5a02ab3038322'}),
    'D': ('formal-m2-v1-pd1-dt-seed0-20261009/dmc_quadruped_walk/dt-attempt01', {
        'protocol_manifest.json': 'e402828a8c5d645206ed86b9557475fd0d7a3171b49c16e2cf25b978a58d78f1',
        'evaluations.jsonl': 'acdd16d9bf5da3abd68f1d10d9c9dee149cc8f9b2b65792705e9c15c8ca44de8',
        'final_state.json': '2de19b669434b23740dff566aeb39331a65da680cd5a44045b4cecd0fa533ab6'})}


def require(condition, message):
  if not condition:
    raise ValueError(message)


def read_json(path):
  return json.loads(Path(path).read_text(encoding='utf-8'))


def append_json(path, value):
  with Path(path).open('a', encoding='utf-8') as stream:
    stream.write(json.dumps(safe_json(value), sort_keys=True, allow_nan=False) + '\n')


def now():
  from datetime import datetime, timezone
  return datetime.now(timezone.utc).isoformat()


def command(argv):
  return subprocess.check_output(argv, text=True, timeout=15).strip()


def binding(expected):
  sha, dirty = git_state()
  require(sha == expected and not dirty, 'Clean exact execution SHA required')
  require(digest(ROOT / 'env/requirements-dreamer.lock') == LOCK_SHA,
      'Dependency lock differs from PD_1')
  return sha


def gpu_state():
  document = ET.fromstring(command(['nvidia-smi', '-q', '-x']))
  rows = {}
  for gpu in document.findall('gpu'):
    number = int(gpu.findtext('minor_number'))
    section = gpu.find('processes')
    require(section is not None and (section.find('process_info') is not None or
        (section.text or '').strip() in ('', 'None')), 'GPU inventory unavailable')
    rows[number] = dict(physical_gpu=number, uuid=gpu.findtext('uuid'),
        memory_used=gpu.findtext('fb_memory_usage/used'),
        processes=[dict(pid=int(p.findtext('pid')), type=p.findtext('type'))
            for p in section.findall('process_info')])
  require(all(i in rows and rows[i]['uuid'] == u for i, u in UUIDS.items()),
      'Assigned sv2 physical GPU UUIDs changed')
  return dict(checked_utc=now(), actual_hostname=platform.node(), gpus=rows)


def require_server():
  require(platform.system() == 'Linux' and platform.node().split('.')[0] == HOST,
      'Only user-assigned sv2 is allowed')


def require_idle(gpu):
  require_server()
  require(gpu in UUIDS, 'Outside user physical GPU4--7 allocation')
  snapshot = gpu_state()
  require(not snapshot['gpus'][gpu]['processes'],
      f'Assigned GPU{gpu} is occupied; no sharing or substitution')
  return snapshot


def overrides(logdir):
  return {
      'script': 'dl_pilot_r3', 'task': TASK, 'seed': SEED, 'logdir': str(logdir),
      'run.action_budget': BUDGET, 'run.eval_every_actions': INTERVAL,
      'run.eval_eps': EPISODES, 'run.envs': ENVS,
      'logger.outputs': ['jsonl'], 'logger.filter': '.*',
      'jax.prealloc': False, 'agent.rep_probe.mode': 'off',
      'agent.dt_latch.mode': 'dtlatch', 'agent.dt_latch.alpha': 20.0,
      'agent.dt_latch.rho': .5, 'agent.dt_latch.kl_tolerance': 1e-5}


def validate_config(config, baseline):
  flat = safe_json(dict(config.flat))
  expected = safe_json(overrides(flat['logdir']))
  for key, value in expected.items():
    require(flat[key] == value, 'DL pilot configuration changed: ' + key)
  allowed = {'script', 'logdir', 'run.action_budget', 'agent.rep_probe.mode',
      'logger.outputs', 'logger.filter'}
  differences = {k: dict(historical=baseline.get(k), actual=flat.get(k))
      for k in set(flat) | set(baseline) if flat.get(k) != baseline.get(k)}
  require(all(k in allowed or k.startswith('agent.dt_latch.') for k in differences),
      'Undeclared configuration differences: ' + str(set(differences) - allowed))
  require(flat['agent.dt_latch.kappa'] == -1 and
      flat['agent.dt_latch.d_bins'] == [-1.] and flat['agent.dt_latch.k_bins'] == [-1.],
      'Unused strength/shuffle calibration must remain unbound')
  return differences


def validate_evaluations(rows):
  from embodied.run.protocol_v1 import seed32, REWARD_SOURCE, RETURN_DEFINITION
  require([r['action_step'] for r in rows] == list(range(0, BUDGET + 1, INTERVAL)),
      'Evaluation grid incomplete or unordered')
  for row in rows:
    scores, lengths, seeds = row['scores'], row['lengths'], row['seeds']
    require(len(scores) == len(lengths) == len(seeds) == len(row['episodes']) == EPISODES,
        'Wrong episode count')
    require(all(math.isfinite(float(x)) for x in scores), 'Nonfinite return')
    require(lengths == [1000] * EPISODES, 'Incomplete quadruped episode')
    require(seeds == [seed32(SEED, TASK, 'eval_env', row['action_step'] // INTERVAL, i)
        for i in range(EPISODES)], 'PD_1 evaluation RNG mapping changed')
    require(row['snapshot_action_step'] == row['target_action_step'] == row['action_step']
        and row['snapshot_update_id'] == row['update_id'], 'Wrong evaluation snapshot')
    require(row['reward_source'] == REWARD_SOURCE and row['return_definition'] == RETURN_DEFINITION,
        'Return measurement differs from PD_1')
    for i, item in enumerate(row['episodes']):
      require(item['complete'] and item['episode_id'] == i and item['length'] == lengths[i]
          and item['environment_seed'] == seeds[i] and item['return_raw'] == scores[i],
          'Episode detail differs from raw return/seed')
    require(row['returns_raw'] == scores and row['mean'] == row['mean_return_raw']
        and abs(row['mean'] - math.fsum(scores) / EPISODES) < 1e-10,
        'Stored return reduction differs')


def summarize(rows):
  validate_evaluations(rows)
  values = [r['mean_return_raw'] for r in rows]
  area = math.fsum(.5 * (a + b) * INTERVAL for a, b in zip(values[:-1], values[1:])) / BUDGET
  tail = [r for r in rows if r['action_step'] >= 4 * BUDGET // 5]
  return dict(grid=[r['action_step'] for r in rows], mean_returns_raw=values,
      action_budget=BUDGET, return_auc_per_action_budget=area, endpoint_return_raw=values[-1],
      tail_grid=[r['action_step'] for r in tail],
      tail_mean_return=math.fsum(r['mean_return_raw'] for r in tail) / len(tail))


def control_evidence():
  result = {}
  for group, (relative, pins) in CONTROLS.items():
    directory = RUNS_ROOT / relative
    for name, expected in pins.items():
      require(digest(directory / name) == expected, 'Historical PD_1 artifact changed: ' + name)
    manifest = read_json(directory / 'protocol_manifest.json')
    require(manifest['git_commit'] == SOURCE_GIT and manifest['root_seed'] == SEED and
        manifest['task'] == TASK and manifest['dependency_lock_sha256'] == LOCK_SHA,
        'Historical control provenance differs')
    zero = directory / 'eval_snapshots/0000000'
    checkpoint = zero / (zero / 'latest').read_text().strip() / 'agent.pkl'
    require(digest(checkpoint) == INITIAL_CHECKPOINT, 'PD_1 initial checkpoint changed')
    rows = [json.loads(line) for line in (directory / 'evaluations.jsonl').read_text().splitlines()
        if json.loads(line)['action_step'] <= BUDGET]
    result[group] = dict(directory=str(directory), source_sha256=pins,
        manifest=manifest, evaluations=rows, summary=summarize(rows),
        initial_checkpoint=dict(path=str(checkpoint), sha256=INITIAL_CHECKPOINT,
            params_sha256=INITIAL_PARAMS))
  return result


def prepare(args):
  binding(args.git_sha)
  require_server()
  from scripts.prepare_m2_v1 import resolve
  output = Path(args.output)
  run_root = Path(args.run_root)
  require(run_root.parent == RUNS_ROOT and run_root.name.startswith('dl-pilot-r3-'),
      'A distinct DL-pilot-r3 run campaign is required')
  require(not run_root.exists() and not output.exists(), 'Use new plan and attempt directories')
  controls = control_evidence()
  logdir = run_root / TASK / 'N-seed0-attempt01'
  config = resolve(['--configs', 'm2_v1', '--task', TASK]).update(overrides(logdir))
  differences = validate_config(config, controls['B']['manifest']['config'])
  output.mkdir(parents=True)
  config.save(str(output / 'config.yaml'))
  write_json(output / 'historical_controls.json', controls)
  index = dict(protocol=VERSION, git_sha=args.git_sha, group='N', task=TASK, seed=SEED,
      logical_server='sv2', physical_gpu=args.gpu, uuid=UUIDS[args.gpu],
      logdir=str(logdir), config_sha256=digest(output / 'config.yaml'),
      controls_sha256=digest(output / 'historical_controls.json'),
      config_differences_from_PD1_B=differences, hard_wall_seconds=SECONDS,
      authorization='User corrected budget to 1m; sv2 GPU4--7; one DL quadruped_walk seed0.',
      hypothesis='Selective rep protection may improve early learning; mechanism unestablished.',
      limitation='Single training seed, historical controls, distinct code SHA; exploratory only.')
  write_json(output / 'RUN_INDEX.json', index)
  print(json.dumps(dict(prepared=str(output / 'RUN_INDEX.json'), **{
      k: index[k] for k in ('task', 'group', 'seed', 'physical_gpu', 'git_sha')})), flush=True)


def load_plan(path, expected):
  path = Path(path).resolve()
  index = read_json(path)
  binding(expected)
  require(index['protocol'] == VERSION and index['git_sha'] == expected and
      index['task'] == TASK and index['group'] == 'N' and index['seed'] == SEED and
      index['logical_server'] == 'sv2' and index['uuid'] == UUIDS[index['physical_gpu']]
      and index['hard_wall_seconds'] == SECONDS, 'Prepared identity differs')
  require(digest(path.parent / 'config.yaml') == index['config_sha256'] and
      digest(path.parent / 'historical_controls.json') == index['controls_sha256'],
      'Prepared config/control evidence changed')
  return index


def spawn_owned(argv, **kwargs):
  # This stdlib parent has no JAX threads. Mask the registration race only.
  previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM, signal.SIGTERM})
  try:
    child = subprocess.Popen(argv, start_new_session=True,
        preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, previous), **kwargs)
    OWNED.add(child.pid)
    return child
  finally:
    signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def cleanup(child):
  try:
    os.killpg(child.pid, signal.SIGKILL)
  except ProcessLookupError:
    pass
  child.wait(timeout=5)
  OWNED.discard(child.pid)


def runtime_mapping(child, gpu):
  import psutil
  try:
    root = psutil.Process(child.pid)
    processes = [root, *root.children(recursive=True)]
  except psutil.NoSuchProcess:
    require(child.poll() is not None, 'Live worker process inventory unavailable')
    processes = []
  pids = set()
  for proc in processes:
    try:
      if not proc.is_running():
        continue
      require(os.getpgid(proc.pid) == child.pid,
          'Owned environment worker escaped the supervised process group')
      pids.add(proc.pid)
    except (ProcessLookupError, psutil.NoSuchProcess):
      pass
  snapshot = gpu_state()
  own = []
  for number, row in snapshot['gpus'].items():
    for proc in row['processes']:
      if proc['pid'] in pids:
        require(number == gpu and row['uuid'] == UUIDS[gpu], 'Actual CUDA/EGL mapping differs')
        own.append(dict(**proc, physical_gpu=number, uuid=row['uuid']))
      elif number == gpu:
        raise RuntimeError('Unknown job occupied allocated GPU; stop only this owned attempt')
  return dict(checked_utc=snapshot['checked_utc'], parent_pid=child.pid,
      owned_pids=sorted(pids), actual_devices=own, physical_gpu=gpu,
      memory_used=snapshot['gpus'][gpu]['memory_used'])


def supervise_child(argv, log, *, deadline, environment, gpu=None, evidence=None):
  with Path(log).open('xb') as stream:
    child = spawn_owned(argv, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    try:
      while True:
        remaining = deadline - time.monotonic()
        require(remaining > 0, 'Whole pilot deadline exceeded')
        try:
          code = child.wait(timeout=min(30, remaining))
          require(time.monotonic() < deadline, 'Late child completion')
          return code, child.pid
        except subprocess.TimeoutExpired:
          if gpu is not None:
            append_json(evidence, runtime_mapping(child, gpu))
    finally:
      cleanup(child)


def launch(args):
  index = load_plan(args.index, args.git_sha)
  gpu = index['physical_gpu']
  output = Path(args.output)
  require(output.parent == Path(args.index).resolve().parent, 'Supervisor must be beside its plan')
  output.mkdir(exist_ok=False)
  initial = require_idle(gpu)
  write_json(output / 'allocation_check.json', initial)
  env = dict(os.environ, CUDA_VISIBLE_DEVICES=UUIDS[gpu], CUDA_DEVICE_ORDER='PCI_BUS_ID',
      MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID=str(gpu),
      OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
      XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONUNBUFFERED='1')
  try:
    probe = output / 'device_probe.json'
    code, probe_pid = supervise_child([sys.executable, str(ROOT / 'scripts/probe_formal_devices.py'),
        '--gpu', str(gpu), '--uuid', UUIDS[gpu], '--output', str(probe)],
        output / 'device_probe.log', deadline=min(DEADLINE, time.monotonic() + 180), environment=env)
    require(code == 0 and read_json(probe)['verified_compute_and_graphics_same_gpu'],
        'Actual zero-action CUDA/EGL device probe failed')
    fresh = require_idle(gpu)
    write_json(output / 'startup.json', dict(protocol=VERSION, git_sha=args.git_sha,
        started_utc=now(), supervisor_pid=os.getpid(), physical_gpu=gpu, uuid=UUIDS[gpu],
        probe_sha256=digest(probe), probe_pid=probe_pid, fresh_allocation=fresh,
        command=[sys.executable, str(ROOT / 'scripts/dl_pilot_train.py'), 'worker',
            '--index', str(Path(args.index).resolve()), '--git-sha', args.git_sha,
            '--supervisor', str(output.resolve())],
        environment={k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
            'MUJOCO_GL', 'PYOPENGL_PLATFORM', 'MUJOCO_EGL_DEVICE_ID')},
        whole_entry_limit_seconds=SECONDS, restoration='No resume or artifact splicing'))
    code, worker_pid = supervise_child(read_json(output / 'startup.json')['command'],
        output / 'worker.log', deadline=DEADLINE, environment=env, gpu=gpu,
        evidence=output / 'resources.jsonl')
    require(code == 0, f'Training worker failed with exit {code}')
    result = audit(args.index, output)
    write_json(output / 'supervisor_complete.json', dict(protocol=VERSION, git_sha=args.git_sha,
        ended_utc=now(), worker_pid=worker_pid, worker_exit=code, status='audited_complete',
        elapsed_seconds=time.monotonic() - ENTRY_STARTED, audit=result,
        resource_release=gpu_state()))
    print(json.dumps(result, sort_keys=True), flush=True)
  except BaseException as error:
    write_json(output / 'supervisor_failed.json', dict(protocol=VERSION, git_sha=args.git_sha,
        failed_utc=now(), type=type(error).__name__, message=str(error),
        elapsed_seconds=time.monotonic() - ENTRY_STARTED,
        continuation='Keep failed attempt; fix locally, commit, and prepare a new attempt.'))
    raise


def worker_binding(args, index):
  supervisor = Path(args.supervisor)
  startup = read_json(supervisor / 'startup.json')
  gpu = index['physical_gpu']
  require_server()
  require(startup['supervisor_pid'] == os.getppid() and startup['git_sha'] == args.git_sha,
      'Worker must be the direct child of its bound supervisor')
  require(os.getpgrp() == os.getpid() and os.environ['CUDA_VISIBLE_DEVICES'] == UUIDS[gpu]
      and os.environ['MUJOCO_EGL_DEVICE_ID'] == str(gpu), 'Worker CUDA/EGL/session binding differs')
  require_idle(gpu)
  return supervisor, gpu


def worker_environment(packages, controls, jax_version):
  require(packages == controls['B']['manifest']['packages'] and jax_version == '0.4.33'
      and platform.python_version() == '3.11.16', 'Actual environment differs from PD_1')


def worker(args):
  index = load_plan(args.index, args.git_sha)
  supervisor, gpu = worker_binding(args, index)
  import multiprocessing as mp
  if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn')
  require(mp.get_start_method() == 'spawn', 'Refusing fork after JAX initialization')
  import elements
  import embodied
  import jax
  from ruamel.yaml import YAML
  from dreamerv3 import main
  from embodied.run.protocol_v1 import (ProtocolState, EpisodeReturn,
      package_versions, seed32, training_fingerprint, parameter_digest)
  plan = Path(args.index).resolve().parent
  config = elements.Config(YAML(typ='safe').load((plan / 'config.yaml').read_text()))
  controls = read_json(plan / 'historical_controls.json')
  validate_config(config, controls['B']['manifest']['config'])
  packages = package_versions('dm-control', 'mujoco', 'numpy', 'jaxlib')
  worker_environment(packages, controls, jax.__version__)
  logdir = Path(config.logdir)
  require(logdir == Path(index['logdir']) and RUNS_ROOT in logdir.parents,
      'Actual artifact path differs')
  logdir.mkdir(parents=True, exist_ok=False)
  (logdir / 'config.yaml').write_bytes((plan / 'config.yaml').read_bytes())
  state = ProtocolState(BUDGET, INTERVAL, EPISODES, 0, BUDGET)
  agent = replay = logger = driver = None
  started = time.perf_counter()
  try:
    agent = main.make_agent(config)
    initial = parameter_digest(agent)
    require(initial == INITIAL_PARAMS and not int(agent.n_updates) and not int(agent.n_actions),
        'New N initialization differs from the common PD_1 untrained state')
    replay_seed = seed32(SEED, TASK, 'train_replay')
    replay = main.make_replay(config, 'replay', seed=replay_seed)
    logger = main.make_logger(config)
    step = logger.step
    cp = elements.Checkpoint(elements.Path(str(logdir / 'ckpt')))
    cp.step, cp.agent, cp.replay, cp.protocol = step, agent, replay, state
    cp.save()
    write_json(logdir / 'protocol_manifest.json', dict(protocol=VERSION, git_commit=args.git_sha,
        task=TASK, group='N', root_seed=SEED, config=dict(config.flat),
        config_sha256=index['config_sha256'], index_sha256=digest(args.index),
        controls_sha256=index['controls_sha256'], historical_git_commit=SOURCE_GIT,
        initial_params_sha256=initial, actual_hostname=platform.node(), python=sys.version,
        jax=jax.__version__, devices=[str(x) for x in jax.devices()], packages=packages,
        dependency_lock_sha256=LOCK_SHA, replay_seed=replay_seed,
        train_env_seeds=[seed32(SEED, TASK, 'train_env', i) for i in range(ENVS)],
        train_policy='JAX _seeds(root_seed, train_policy_call_index)',
        train_update='JAX _seeds(root_seed, train_batch_index)',
        eval_policy='PCG64(root_seed, EVAL namespace, environment_seed, call)',
        evaluation='SHA256(root_seed, task, eval_env, grid_index, episode_id)',
        resource=dict(logical_server='sv2', physical_gpu=gpu, uuid=UUIDS[gpu], logical_cuda=0,
            mujoco_egl_device_id=os.environ['MUJOCO_EGL_DEVICE_ID'], supervisor=str(supervisor),
            startup_sha256=digest(supervisor / 'startup.json')),
        matching='No constant or shuffle arm; no c/kappa matching in this N run',
        restoration='From scratch only; environment/replay RNG cannot seamlessly resume',
        limitation=index['limitation']))
    batch_steps = config.batch_size * config.batch_length
    should_train = elements.when.Ratio(config.run.train_ratio / batch_steps)
    stream = iter(main.make_stream(config, replay, 'train'))
    carry = [agent.init_train(config.batch_size)]
    agg = elements.Agg()
    last_progress = [time.monotonic()]

    def on_transition(tran, worker_id):
      state.on_train_transition(tran)
      if not tran['is_first']:
        step.increment()
      replay.add(tran, worker_id)
      if len(replay) < batch_steps:
        return
      for _ in range(should_train(step)):
        update_id, action_step = state.updates, state.actions
        batch = agent.prepare_batch(next(stream))
        carry[0], delayed_outs, delayed_metrics = agent.train(carry[0], dict(batch))
        require(not delayed_outs and not delayed_metrics, 'Unconsumed asynchronous training result')
        outs, metrics = agent.take_train_result()
        finite = all(math.isfinite(float(metrics[k])) for k in ('opt/grad_norm', 'opt/loss'))
        require(int(metrics['opt/updates']) == update_id + 1, 'Optimizer did not commit expected update')
        state.record_update(update_id, action_step, 0., 0, finite=finite)
        require(int(agent.n_updates) == state.updates, 'Agent and receipt update counters disagree')
        append_json(logdir / 'update_receipts.jsonl', dict(update_id=update_id,
            start_action=action_step, match_sum=0., match_count=0, invalid_count=0,
            optimizer_loss=float(metrics['opt/loss']), optimizer_grad_norm=float(metrics['opt/grad_norm']),
            metrics={k: float(v) for k, v in metrics.items()
                if k.startswith(('dl/', 'dt/')) and getattr(v, 'ndim', 0) == 0}))
        if 'replay' in outs:
          replay.update(outs['replay'])
        agg.add(metrics, prefix='train')
      if time.monotonic() - last_progress[0] >= 30:
        append_json(logdir / 'progress.jsonl', dict(checked_utc=now(),
            actions=state.actions, resets=state.resets, updates=state.updates,
            elapsed_seconds=time.perf_counter() - started))
        last_progress[0] = time.monotonic()

    fns = [partial(main.make_env, config, i, seed=seed32(SEED, TASK, 'train_env', i))
        for i in range(ENVS)]
    driver = embodied.Driver(fns, parallel=True)
    driver.on_step(on_transition)
    driver.reset(agent.init_policy)
    policy = lambda *xs: agent.policy(*xs, mode='train')

    def evaluate(point):
      before = training_fingerprint(agent, replay)
      agent.sync_policy()
      directory = logdir / 'eval_snapshots' / f'{point:07d}'
      snapshot = elements.Checkpoint(elements.Path(str(directory)))
      snapshot.agent = agent
      snapshot.save()
      snapshot_id = str(directory.relative_to(logdir)) + '/' + (directory / 'latest').read_text().strip()
      write_json(directory / 'evaluation_snapshot.json', dict(snapshot_id=snapshot_id,
          action_step=state.actions, update_id=state.updates, git_commit=args.git_sha))
      details, scores, lengths, seeds = [], [], [], []
      for episode_id in range(EPISODES):
        seed = seed32(SEED, TASK, 'eval_env', point // INTERVAL, episode_id)
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
        item = dict(result.result(), episode_id=episode_id, environment_seed=seed,
            action_rng='PCG64(root_seed, EVAL namespace, environment_seed, call)')
        details.append(item)
        scores.append(item['return_raw'])
        lengths.append(item['length'])
        seeds.append(seed)
        append_json(logdir / 'evaluation_episode_receipts.jsonl', dict(action_step=point, **item))
      after = training_fingerprint(agent, replay)
      require(before == after, 'DL evaluation changed training parameters/counters/replay state')
      state.record_evaluation(point, scores, lengths, seeds,
          episode_details=details, snapshot_id=snapshot_id)
      append_json(logdir / 'evaluations.jsonl', dict(action_step=point, **state.evaluations[point]))
      append_json(logdir / 'evaluation_state_audit.jsonl', dict(action_step=point, before=before, after=after))
      logger.add(dict(return_raw=state.evaluations[point]['mean'], action_steps=state.actions), prefix='DL_eval')
      logger.add(agg.result())
      logger.write()
      cp.save()
      print(json.dumps(dict(event='evaluation_complete', action_step=point,
          updates=state.updates, mean_return_raw=state.evaluations[point]['mean'])), flush=True)

    evaluate(0)
    while state.actions < BUDGET:
      boundary = state.next_boundary()
      driver.step_selected(policy, range(min(ENVS, boundary - state.actions)))
      if state.actions == boundary:
        evaluate(boundary)
    require(state.actions == int(step) == BUDGET and len(state.evaluations) == len(state.grid),
        'Exact 1M budget and complete evaluation grid required')
    result = dict(protocol=VERSION, git_commit=args.git_sha, group='N', task=TASK, seed=SEED,
        status='training_ended_pending_audit', train_action_steps=state.actions,
        train_resets=state.resets, updates=state.updates, eval_action_steps=state.eval_actions,
        evaluations=len(state.evaluations), params_sha256=parameter_digest(agent),
        train_policy_calls=int(agent.n_actions), train_batch_keys=int(agent.n_batches),
        elapsed_seconds=time.perf_counter() - started,
        **summarize([dict(action_step=p, **r) for p, r in sorted(state.evaluations.items())]))
    write_json(logdir / 'final_state.json', result)
    cp.save()
    print(json.dumps(result, sort_keys=True), flush=True)
  except BaseException as error:
    write_json(logdir / 'FAILED.json', dict(status='failed', type=type(error).__name__,
        message=str(error), train_action_steps=state.actions, updates=state.updates,
        elapsed_seconds=time.perf_counter() - started, git_commit=args.git_sha,
        continuation='No resume. Keep evidence and start a new attempt after local fix.'))
    raise
  finally:
    if driver is not None:
      driver.close()
    if logger is not None:
      logger.close()


def audit(index_path, supervisor):
  index = read_json(index_path)
  directory = Path(index['logdir'])
  require(not (directory / 'FAILED.json').exists(), 'Failed artifact cannot pass')
  manifest, final = read_json(directory / 'protocol_manifest.json'), read_json(directory / 'final_state.json')
  require(final['git_commit'] == manifest['git_commit'] == index['git_sha'] and
      final['train_action_steps'] == BUDGET and final['evaluations'] == BUDGET // INTERVAL + 1 and
      final['eval_action_steps'] == (BUDGET // INTERVAL + 1) * EPISODES * 1000 and
      final['updates'] == BUDGET // 4 - 507 and
      final['train_batch_keys'] == final['updates'], 'Final exact counts differ from PD_1 full budget')
  rows = [json.loads(x) for x in (directory / 'evaluations.jsonl').read_text().splitlines()]
  calculated = summarize(rows)
  require(all(final[k] == v for k, v in calculated.items()), 'Final summary disagrees with raw evaluations')
  receipts = [json.loads(x) for x in (directory / 'update_receipts.jsonl').read_text().splitlines()]
  require([r['update_id'] for r in receipts] == list(range(final['updates'])), 'Update ledger incomplete')
  require(all(0 <= r['start_action'] <= BUDGET and not r['invalid_count'] and
      math.isfinite(r['optimizer_loss']) and math.isfinite(r['optimizer_grad_norm']) for r in receipts),
      'Invalid optimizer receipt')
  for r in rows:
    require((directory / r['snapshot_id'] / 'agent.pkl').is_file(), 'Missing actual evaluation snapshot')
    require(r['update_id'] == (r['action_step'] // 4 - 507 if r['action_step'] else 0),
        'Update count at evaluation differs from PD_1')
  fingerprints = [json.loads(x) for x in (directory / 'evaluation_state_audit.jsonl').read_text().splitlines()]
  require([r['action_step'] for r in fingerprints] == list(range(0, BUDGET + 1, INTERVAL)) and
      all(r['before'] == r['after'] for r in fingerprints), 'Evaluation mutated training state')
  resources = [json.loads(x) for x in (Path(supervisor) / 'resources.jsonl').read_text().splitlines()]
  devices = [p for r in resources for p in r['actual_devices']]
  require(any('C' in p['type'] for p in devices) and any('G' in p['type'] for p in devices) and
      all(p['uuid'] == index['uuid'] for p in devices), 'Actual training compute/EGL mapping not evidenced')
  controls = read_json(Path(index_path).parent / 'historical_controls.json')
  result = dict(protocol=VERSION, git_commit=index['git_sha'], status='audited_complete',
      actual_run=str(directory), counts={k: final[k] for k in
          ('train_action_steps', 'updates', 'evaluations', 'eval_action_steps')},
      N=calculated, historical={g: controls[g]['summary'] for g in ('B', 'D')},
      differences={g: {k: calculated[k] - controls[g]['summary'][k] for k in
          ('return_auc_per_action_budget', 'endpoint_return_raw', 'tail_mean_return')}
          for g in ('B', 'D')},
      artifact_sha256={name: digest(directory / name) for name in
          ('protocol_manifest.json', 'final_state.json', 'evaluations.jsonl', 'update_receipts.jsonl',
           'evaluation_state_audit.jsonl')},
      limitation='Exploratory one-seed historical comparison; no mechanism, paired-pixel, or cross-seed claim.')
  write_json(directory / 'audit.json', result)
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest='stage', required=True)
  plan = sub.add_parser('prepare')
  plan.add_argument('--output', required=True)
  plan.add_argument('--run-root', required=True)
  plan.add_argument('--gpu', type=int, choices=tuple(UUIDS), required=True)
  plan.add_argument('--git-sha', required=True)
  start = sub.add_parser('launch')
  start.add_argument('--index', required=True)
  start.add_argument('--output', required=True)
  start.add_argument('--git-sha', required=True)
  run = sub.add_parser('worker')
  run.add_argument('--index', required=True)
  run.add_argument('--git-sha', required=True)
  run.add_argument('--supervisor', required=True)
  args = parser.parse_args()
  {'prepare': prepare, 'launch': launch, 'worker': worker}[args.stage](args)


if __name__ == '__main__':
  main()
