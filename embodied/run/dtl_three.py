"""Dedicated DtL runner; original M2 runtime remains byte-identical."""
import hashlib
import json
import math
import platform
import subprocess
import sys
from functools import partial as bind
from pathlib import Path
import elements
import embodied
import jax
from dreamerv3.protocol_scope import (
    validate_scope, tasks_for_protocol, validate_dtl_scope, DTL_THREE)
from dreamerv3.return_stats import EpisodeReturn
from embodied.run.protocol_v1 import (
    ProtocolState, seed32, package_versions, array_hashes, tree_hash,
    parameter_hashes, report_fingerprint, training_fingerprint, report_on_batch,
    persisted_rows, audit_persisted_history, audit_complete_outputs, parameter_digest)


def dtl_receipt(update_id, start_action, metrics):
  keys = ('actual_v_mean', 'actual_v_min', 'actual_v_max', 'active_frac',
          'valid_frac', 'valid_active_frac', 'active_protected_frac',
          'excess_release_mean', 'weighted_excess_mean',
          'rep_before_mean', 'rep_after_mean')
  values = {key: float(metrics['dl/' + key]) for key in keys}
  values.update(loss=float(metrics['opt/loss']),
                grad_norm=float(metrics['opt/grad_norm']))
  if not all(math.isfinite(x) for x in values.values()):
    raise RuntimeError('Nonfinite DtL update diagnostics')
  if not 0.5 <= values['actual_v_min'] <= values['actual_v_max'] <= 1:
    raise RuntimeError('DtL weight outside the frozen rho=0.5 bounds')
  return dict(update_id=update_id, start_action=start_action, **values)


def audit_dtl_receipts(logdir, state):
  rows = persisted_rows(Path(logdir) / 'dtl_updates.jsonl')
  seen = {}
  for row in rows:
    uid = int(row['update_id'])
    if uid in seen:
      raise RuntimeError(f'Duplicate DtL update {uid}')
    seen[uid] = int(row['start_action'])
  expected = {uid: record[0] for uid, record in state.records.items()}
  if seen != expected:
    raise RuntimeError('Checkpoint/DtL receipts disagree')


def run(make_agent, make_replay, make_env, make_stream, make_logger,
        args, config):
  protocol = validate_scope(config.script, config.task, config.seed,
                            config.agent.rep_probe.mode)
  validate_dtl_scope(config)
  is_dtl = protocol == DTL_THREE
  if args.eval_envs != 1:
    raise ValueError('M2 v1 uses one eval environment for exact 10 episodes')
  if not is_dtl and config.agent.rep_probe.mode not in (
      'logging', 'dt', 'constant', 'shuffle'):
    raise ValueError('M2 v1 requires one of the four declared modes')
  if config.agent.rep_probe.mode != 'constant' and (
      config.agent.rep_probe.alpha != 20):
    raise ValueError('M2 v1 requires alpha=20')
  if config.agent.rep_probe.mode == 'constant' and (
      config.agent.rep_probe.alpha != 20 or
      not 0 < config.agent.rep_probe.c <= 1):
    raise ValueError('M2 v1 constant requires frozen c and alpha=20')
  if config.agent.rep_probe.mode == 'constant' and not args.engineering_fixture:
    if not args.frozen_c_file:
      raise ValueError('M2 v1 constant requires a frozen c artifact')
    frozen_path = Path(str(args.frozen_c_file)).expanduser()
    frozen_bytes = frozen_path.read_bytes()
    checksum = frozen_path.with_suffix(frozen_path.suffix + '.sha256')
    if checksum.read_text(encoding='utf-8').strip() != hashlib.sha256(
        frozen_bytes).hexdigest():
      raise ValueError('Frozen c artifact checksum mismatch')
    frozen = json.loads(frozen_bytes)
    if (frozen['protocol'] != 'M2-v1' or frozen['task'] != config.task or
        frozen['root_seed'] != 0 or frozen['alpha'] != 20 or
        frozen['window'] != [100000, 300000] or
        frozen['raw_rep_threshold'] != 1 or
        frozen['engineering_fixture'] or
        abs(frozen['c'] - config.agent.rep_probe.c) > 1e-12):
      raise ValueError('Frozen c artifact does not match this run')
  if not args.engineering_fixture:
    if (args.engineering_stop_after_actions or
        args.engineering_trace_actions or args.engineering_trace_updates or
        config.env.dmc.engineering_trace_physics):
      raise ValueError('Engineering stop/trace is forbidden in formal runs')
    if (int(args.action_budget), int(args.eval_every_actions),
        int(args.eval_eps), int(args.match_start), int(args.match_end)) != (
            1000000, 50000, 10, 100000, 300000):
      raise ValueError('M2 v1 budget, grid, episodes, or window differs')
    if config.task not in tasks_for_protocol(protocol):
      raise ValueError('Task is outside the declared protocol task set')
    rssm = config.agent.dyn.rssm
    if (rssm.deter, rssm.hidden, rssm.classes) != (4096, 512, 32):
      raise ValueError('M2 v1 requires size50m RSSM')
    if (config.agent.enc.simple.depth, config.agent.dec.simple.depth,
        config.agent.policy.units, config.agent.value.units) != (
            32, 32, 512, 512):
      raise ValueError('M2 v1 requires size50m head/encoder settings')
    if (config.agent.ac_grads or not config.agent.reward_grad or
        not config.agent.repval_grad or not config.agent.repval_loss):
      raise ValueError('M2 v1 gradient switches differ from protocol')
  if (not args.engineering_fixture and (
      config.agent.dyn.rssm.free_nats != 1 or
      config.agent.loss_scales.dyn != 1 or
      config.agent.loss_scales.rep != 0.1)):
    raise ValueError('M2 v1 KL settings differ from frozen protocol')
  if (not args.engineering_fixture and (
      config.env.dmc.repeat != 1 or config.env.dmc.proprio or
      not config.env.dmc.image or not config.env.dmc.use_seed)):
    raise ValueError('M2 v1 requires clean vision-only DMC repeat=1')

  logdir = Path(str(args.logdir)).expanduser()
  logdir.mkdir(parents=True, exist_ok=True)
  repo = Path(__file__).resolve().parents[2]
  git_commit = subprocess.check_output(
      ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
  if subprocess.check_output(
      ['git', 'status', '--porcelain'], cwd=repo, text=True).strip():
    raise RuntimeError('Protocol runner requires a clean Git checkout')
  manifest_path = logdir / 'protocol_manifest.json'
  if manifest_path.exists():
    previous = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected_config = json.loads(json.dumps(dict(config.flat), default=str))
    if (previous.get('config') != expected_config or
        previous.get('git_commit') != git_commit):
      raise ValueError('Existing protocol config or Git commit differs')
  state = ProtocolState(
      args.action_budget, args.eval_every_actions, args.eval_eps,
      args.match_start, args.match_end)
  agent = make_agent()
  replay = make_replay()
  logger = make_logger()
  step = logger.step
  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  last_batch = [None]
  agg = elements.Agg()

  cp = elements.Checkpoint(elements.Path(str(logdir / 'ckpt')))
  cp.step = step
  cp.agent = agent
  cp.replay = replay
  cp.protocol = state
  had_checkpoint = (logdir / 'ckpt' / 'latest').exists()
  try:
    cp.load_or_save()
    if int(step) != state.actions or int(agent.n_updates) != state.updates:
      raise ValueError('Checkpoint counters disagree')
    audit_persisted_history(logdir, state)
    if is_dtl:
      audit_dtl_receipts(logdir, state)
    if had_checkpoint:
      if state.actions < state.budget:
        raise RuntimeError(
            'Training continuation refused: environment and replay RNG '
            'state are not completely checkpointed; start a fresh run')
      if len(state.evaluations) != len(state.grid):
        raise RuntimeError(
            'Terminal evaluation recovery needs the fixed endpoint snapshot')
      audit_complete_outputs(
          logdir, state, config.agent.rep_probe.mode, git_commit)
      print('Completed protocol run already checkpointed')
      logger.close()
      return
  except Exception:
    logger.close()
    raise

  stream_train = iter(make_stream(replay, 'train'))
  carry_train = [agent.init_train(args.batch_size)]

  manifest = dict(
      protocol=protocol, task=config.task, root_seed=config.seed,
      git_commit=git_commit, python=sys.version,
      jax=jax.__version__, host=platform.node(),
      devices=[str(x) for x in jax.devices()],
      device_kinds=[x.device_kind for x in jax.devices()],
      packages=package_versions('dm-control', 'mujoco', 'numpy', 'jaxlib'),
      dependency_lock_sha256=hashlib.sha256(
          (repo / 'env' / 'requirements-dreamer.lock').read_bytes()
      ).hexdigest(),
      train_env_seeds=[seed32(config.seed, config.task, 'train_env', i)
                       for i in range(args.envs)],
      replay_seed=seed32(config.seed, config.task, 'train_replay'),
      train_policy='JAX _seeds(root_seed, train_policy_call_index)',
      train_update='JAX _seeds(root_seed, train_batch_index)',
      report='SHA256(root_seed, task, report, report_id) then PCG64',
      evaluation='SHA256(root_seed, task, eval_env, grid_index, episode_id)',
      eval_policy='JAX PCG64(root_seed, EVAL namespace, episode_seed, call)',
      shuffle='JAX PRNGKey(root_seed), fold_in(0x4454), fold_in(opt_step)',
      config=dict(config.flat))
  manifest_payload = json.dumps(
      manifest, sort_keys=True, indent=2, default=str)
  if manifest_path.exists():
    if manifest_path.read_text(encoding='utf-8') != manifest_payload:
      raise ValueError('Existing protocol manifest differs from this run')
  else:
    manifest_path.write_text(
        manifest_payload, encoding='utf-8')

  def on_transition(tran, worker):
    state.on_train_transition(tran)
    if (args.engineering_fixture and args.engineering_trace_actions and
        state.actions <= args.engineering_trace_actions):
      with (logdir / 'engineering_action_trace.jsonl').open(
          'a', encoding='utf-8') as file:
        file.write(json.dumps(dict(action_step=state.actions, worker=worker,
            hashes=array_hashes(tran, include_logs=True)),
            sort_keys=True) + '\n')
    if not tran['is_first']:
      step.increment()
    replay.add(tran, worker)
    if len(replay) < batch_steps:
      return
    for _ in range(should_train(step)):
      start_action = state.actions
      update_id = state.updates
      raw_batch = next(stream_train)
      batch = agent.prepare_batch(raw_batch)
      last_batch[0] = raw_batch
      if (args.engineering_fixture and args.engineering_trace_updates and
          update_id < args.engineering_trace_updates):
        with (logdir / 'engineering_update_trace.jsonl').open(
            'a', encoding='utf-8') as file:
          file.write(json.dumps(dict(
              update_id=update_id, start_action=start_action,
              batch_hashes=array_hashes(raw_batch, skip=('stepid',)),
              seed_hash=tree_hash(batch['seed']),
              carry_hash=tree_hash(carry_train[0]),
              parameter_hashes=parameter_hashes(agent)),
              sort_keys=True) + '\n')
      carry_train[0], delayed_outs, delayed_mets = agent.train(
          carry_train[0], dict(batch))
      if delayed_outs or delayed_mets:
        raise RuntimeError('Unconsumed delayed train result')
      outs, metrics = agent.take_train_result()
      grad_norm = float(metrics['opt/grad_norm'])
      finite = math.isfinite(grad_norm) and math.isfinite(
          float(metrics['opt/loss']))
      if int(metrics['opt/updates']) != update_id + 1:
        raise RuntimeError(f'Optimizer did not commit update {update_id}')
      if config.agent.rep_probe.mode == 'logging':
        match_sum = float(metrics['dt/match_sum'])
        match_count = int(metrics['dt/match_count'])
        invalid = int(metrics['dt/match_invalid_count'])
      else:
        match_sum, match_count, invalid = 0.0, 0, 0
      state.record_update(
          update_id, start_action, match_sum, match_count, invalid, finite)
      if int(agent.n_updates) != state.updates:
        raise ValueError('Agent update counter disagrees with receipt')
      if is_dtl:
        receipt = dtl_receipt(update_id, start_action, metrics)
        with (logdir / 'dtl_updates.jsonl').open('a', encoding='utf-8') as file:
          file.write(json.dumps(receipt, sort_keys=True, allow_nan=False) + '\n')
      with (logdir / 'update_receipts.jsonl').open(
          'a', encoding='utf-8') as file:
        file.write(json.dumps(dict(
            update_id=update_id, start_action=start_action,
            match_sum=match_sum, match_count=match_count,
            invalid_count=invalid), sort_keys=True) + '\n')
      if 'replay' in outs:
        replay.update(outs['replay'])
      agg.add(metrics, prefix='train')

  fns = [bind(make_env, i, seed=seed32(
      config.seed, config.task, 'train_env', i)) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(on_transition)
  driver.reset(agent.init_policy)
  train_policy = lambda *xs: agent.policy(*xs, mode='train')

  def evaluate(point):
    before = (training_fingerprint(agent, replay)
              if args.engineering_fixture or is_dtl else None)
    grid_index = point // state.interval
    agent.sync_policy()
    snapshot = elements.Checkpoint(elements.Path(
        str(logdir / 'eval_snapshots' / f'{point:07d}')))
    snapshot.agent = agent
    snapshot.save()
    snapshot_id = f'eval_snapshots/{point:07d}/' + (
        logdir / 'eval_snapshots' / f'{point:07d}' / 'latest').read_text().strip()
    (logdir / 'eval_snapshots' / f'{point:07d}' / 'evaluation_snapshot.json').write_text(
        json.dumps(dict(snapshot_id=snapshot_id, actual_action_step=state.actions,
            update_id=state.updates, git_commit=git_commit), sort_keys=True) + '\n',
        encoding='utf-8')
    scores, lengths, seeds, episode_details = [], [], [], []
    for episode_id in range(state.episodes):
      seed = seed32(config.seed, config.task, 'eval_env',
                    grid_index, episode_id)
      seeds.append(seed)
      env = make_env(0, seed=seed)
      evaluator = embodied.Driver([lambda env=env: env], parallel=False)
      episode_return = EpisodeReturn()
      def on_eval(tran, _):
        if not tran['is_first']:
          state.eval_actions += 1
        episode_return.add(tran)
      evaluator.on_step(on_eval)
      agent.set_eval_seed(seed)
      evaluator.reset(lambda count: agent.init_policy(count, mode='eval'))
      evaluator(lambda *xs: agent.policy(*xs, mode='eval'), episodes=1)
      evaluator.close()
      details = dict(episode_return.result(), episode_id=episode_id,
          environment_seed=seed,
          action_rng='PCG64(root_seed, EVAL namespace, environment_seed, call)')
      episode_details.append(details)
      scores.append(details['return_raw'])
      lengths.append(details['length'])
    state.record_evaluation(point, scores, lengths, seeds,
        episode_details=episode_details, snapshot_id=snapshot_id)
    (logdir / 'evaluations.jsonl').open('a', encoding='utf-8').write(
        json.dumps(dict(action_step=point, **state.evaluations[point]),
                   sort_keys=True) + '\n')
    logger.add(dict(
        eval_return=state.evaluations[point]['mean'],
        eval_episodes=state.episodes, train_action_steps=state.actions,
        train_resets=state.resets, eval_action_steps=state.eval_actions,
        updates=state.updates), prefix='protocol')
    logger.write()
    cp.save()
    if before is not None:
      after = training_fingerprint(agent, replay)
      if before != after:
        raise RuntimeError('Evaluation changed training state')
      with (logdir / 'engineering_evaluation_audit.jsonl').open(
          'a', encoding='utf-8') as file:
        file.write(json.dumps(dict(action_step=point, fingerprint=before),
            sort_keys=True) + '\n')

  try:
    if 0 not in state.evaluations:
      evaluate(0)
    next_report = int(args.report_every_actions)
    while state.actions < state.budget:
      boundary = state.next_boundary()
      remaining = boundary - state.actions
      driver.step_selected(train_policy, range(min(args.envs, remaining)))
      if (args.report_every_actions and last_batch[0] is not None and
          state.actions >= next_report):
        before = (report_fingerprint(agent, replay, last_batch[0])
                  if args.engineering_fixture else None)
        metrics = report_on_batch(
            agent, last_batch[0], state.reports, config.seed, config.task,
            config.report_length + config.replay_context)
        if before is not None:
          after = report_fingerprint(agent, replay, last_batch[0])
          if before != after:
            raise RuntimeError('Read-only report changed training state')
          with (logdir / 'engineering_report_audit.jsonl').open(
              'a', encoding='utf-8') as file:
            file.write(json.dumps(dict(action_step=state.actions,
                report_id=state.reports, fingerprint=before),
                sort_keys=True) + '\n')
        state.reports += 1
        logger.add(metrics, prefix='report')
        next_report += int(args.report_every_actions)
      if state.actions == boundary:
        evaluate(boundary)
        logger.add(agg.result())
        logger.write()
        if (args.engineering_fixture and
            args.engineering_stop_after_actions and
            state.actions >= args.engineering_stop_after_actions):
          raise RuntimeError('Intentional fixture stop after checkpoint')
    if config.agent.rep_probe.mode == 'logging':
      c = state.frozen_c()
      (logdir / 'matching_result.json').write_text(
          json.dumps(dict(match_sum=state.match_sum,
                          match_count=state.match_count, c=c,
                          start=state.match_start, end=state.match_end),
                     sort_keys=True, indent=2), encoding='utf-8')
    (logdir / 'final_state.json').write_text(json.dumps(dict(
        git_commit=git_commit, train_action_steps=state.actions,
        train_resets=state.resets, eval_action_steps=state.eval_actions,
        train_policy_calls=int(agent.n_actions),
        train_batch_keys=int(agent.n_batches),
        eval_policy_calls=int(agent.n_eval_actions),
        updates=state.updates, evaluations=len(state.evaluations),
        match_sum=state.match_sum, match_count=state.match_count,
        params_sha256=parameter_digest(agent)),
        sort_keys=True, indent=2), encoding='utf-8')
    cp.save()
  finally:
    driver.close()
    logger.close()
