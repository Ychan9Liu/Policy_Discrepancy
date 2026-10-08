"""Exact-action M2 v1 runner and auditable active-weight matching."""

import hashlib
import json
import math
from pathlib import Path
from functools import partial as bind

import elements
import embodied
import numpy as np

from embodied.jax import internal


def seed32(root, task, namespace, *indices):
  """Stable, named seed derivation; independent of Python's process hash."""
  parts = [str(root), task, namespace, *(str(x) for x in indices)]
  digest = hashlib.sha256('\0'.join(parts).encode()).digest()
  return int.from_bytes(digest[:4], 'big')


class ProtocolState:

  def __init__(self, budget, interval, episodes, match_start, match_end):
    budget, interval, episodes = map(int, (budget, interval, episodes))
    match_start, match_end = int(match_start), int(match_end)
    if budget < 1 or interval < 1 or budget % interval or episodes < 1:
      raise ValueError('Budget must be a positive multiple of eval interval')
    if not 0 <= match_start < match_end <= budget:
      raise ValueError('Invalid half-open matching window')
    self.budget = budget
    self.interval = interval
    self.episodes = episodes
    self.match_start = match_start
    self.match_end = match_end
    self.actions = 0
    self.resets = 0
    self.eval_actions = 0
    self.updates = 0
    self.reports = 0
    self.match_sum = 0.0
    self.match_count = 0
    self.records = {}
    self.evaluations = {}

  @property
  def grid(self):
    return tuple(range(0, self.budget + 1, self.interval))

  def next_boundary(self):
    return next((point for point in self.grid if point > self.actions),
                self.budget)

  def on_train_transition(self, transition):
    if bool(transition['is_first']):
      self.resets += 1
    else:
      if self.actions >= self.budget:
        raise RuntimeError('Training action budget exceeded')
      self.actions += 1

  def record_update(self, update_id, start_action, match_sum, match_count,
                    invalid_count=0, finite=True):
    update_id, start_action = int(update_id), int(start_action)
    match_sum = float(match_sum)
    match_count, invalid_count = int(match_count), int(invalid_count)
    receipt = (start_action, match_sum, match_count, invalid_count)
    if update_id in self.records:
      if self.records[update_id] != receipt:
        raise ValueError(f'Conflicting receipt for update {update_id}')
      return False
    if update_id != self.updates:
      raise ValueError(f'Expected update ID {self.updates}, got {update_id}')
    if not 0 <= start_action <= self.budget:
      raise ValueError('Update action step outside budget')
    if (not finite or invalid_count or not math.isfinite(match_sum) or
        match_sum < 0 or match_count < 0 or match_sum > match_count + 1e-4):
      raise ValueError(f'Invalid matching/gradient receipt: {receipt}')
    self.records[update_id] = receipt
    self.updates += 1
    if self.match_start <= start_action < self.match_end:
      self.match_sum += match_sum
      self.match_count += match_count
    return True

  def frozen_c(self):
    if self.match_count == 0:
      raise ValueError('No active rep loss positions in matching window')
    c = self.match_sum / self.match_count
    if not math.isfinite(c) or not 0 < c <= 1:
      raise ValueError(f'Invalid frozen constant: {c}')
    return c

  def record_evaluation(self, action_step, scores, lengths, seeds):
    action_step = int(action_step)
    if action_step not in self.grid or action_step != self.actions:
      raise ValueError('Evaluation is not on the current action grid point')
    if action_step in self.evaluations:
      raise ValueError(f'Evaluation point {action_step} already recorded')
    if len(scores) != self.episodes or len(lengths) != self.episodes:
      raise ValueError('Evaluation requires exactly the planned episodes')
    if len(seeds) != self.episodes or not all(
        math.isfinite(float(x)) for x in scores):
      raise ValueError('Invalid episode result or seed mapping')
    self.evaluations[action_step] = dict(
        scores=[float(x) for x in scores],
        lengths=[int(x) for x in lengths],
        seeds=[int(x) for x in seeds],
        mean=float(np.mean(scores)),
        update_id=self.updates)

  def save(self):
    return dict(
        budget=self.budget, interval=self.interval, episodes=self.episodes,
        match_start=self.match_start, match_end=self.match_end,
        actions=self.actions, resets=self.resets,
        eval_actions=self.eval_actions, updates=self.updates,
        reports=self.reports, match_sum=self.match_sum,
        match_count=self.match_count, records=self.records,
        evaluations=self.evaluations)

  def load(self, data):
    expected = (self.budget, self.interval, self.episodes,
                self.match_start, self.match_end)
    actual = tuple(int(data[k]) for k in (
        'budget', 'interval', 'episodes', 'match_start', 'match_end'))
    if actual != expected:
      raise ValueError(f'Checkpoint protocol mismatch: {actual} != {expected}')
    for key in ('actions', 'resets', 'eval_actions', 'updates', 'reports',
                'match_sum', 'match_count'):
      setattr(self, key, data[key])
    self.records = {int(k): tuple(v) for k, v in data['records'].items()}
    self.evaluations = {int(k): v for k, v in data['evaluations'].items()}
    if len(self.records) != self.updates:
      raise ValueError('Checkpoint update ledger is incomplete')


def report_on_batch(agent, batch, report_id, root, task, length):
  """Report from an already sampled batch without touching train replay/RNG."""
  seed = seed32(root, task, 'report', report_id)
  rng = np.random.default_rng(seed=[seed, 0])
  key = rng.integers(0, np.iinfo(np.uint32).max, (2,), np.uint32)
  key = internal.device_put(key, agent.train_mirrored)
  if batch['is_first'].shape[1] < length:
    raise ValueError('Training batch is shorter than report length')
  batch = {k: v[:, :length] for k, v in batch.items() if k != 'seed'}
  carry = agent.init_report(len(batch['is_first']))
  _, metrics = agent.report(carry, dict(batch, seed=key))
  return metrics


def run(make_agent, make_replay, make_env, make_stream, make_logger,
        args, config):
  if config.seed != 0:
    raise ValueError('M2 v1 requires training seed 0')
  if args.eval_envs != 1:
    raise ValueError('M2 v1 uses one eval environment for exact 10 episodes')
  if config.agent.rep_probe.mode not in (
      'logging', 'dt', 'constant', 'shuffle'):
    raise ValueError('M2 v1 requires one of the four declared modes')
  if config.agent.rep_probe.mode != 'constant' and (
      config.agent.rep_probe.alpha != 20):
    raise ValueError('M2 v1 requires alpha=20')
  if config.agent.rep_probe.mode == 'constant' and (
      config.agent.rep_probe.alpha != 20 or
      not 0 < config.agent.rep_probe.c <= 1):
    raise ValueError('M2 v1 constant requires frozen c and alpha=20')
  if not args.engineering_fixture:
    if (int(args.action_budget), int(args.eval_every_actions),
        int(args.eval_eps), int(args.match_start), int(args.match_end)) != (
            1000000, 50000, 10, 100000, 300000):
      raise ValueError('M2 v1 budget, grid, episodes, or window differs')
    if config.task not in (
        'dmc_hopper_hop', 'dmc_quadruped_run',
        'dmc_quadruped_walk', 'dmc_reacher_hard'):
      raise ValueError('M2 v1 task is outside the frozen four-task set')
    rssm = config.agent.dyn.rssm
    if (rssm.deter, rssm.hidden, rssm.classes) != (4096, 512, 32):
      raise ValueError('M2 v1 requires size50m RSSM')
  if (config.agent.dyn.rssm.free_nats != 1 or
      config.agent.loss_scales.dyn != 1 or
      config.agent.loss_scales.rep != 0.1):
    raise ValueError('M2 v1 KL settings differ from frozen protocol')
  if (not args.engineering_fixture and (
      config.env.dmc.repeat != 1 or config.env.dmc.proprio or
      not config.env.dmc.image or not config.env.dmc.use_seed)):
    raise ValueError('M2 v1 requires clean vision-only DMC repeat=1')

  logdir = Path(str(args.logdir)).expanduser()
  logdir.mkdir(parents=True, exist_ok=True)
  state = ProtocolState(
      args.action_budget, args.eval_every_actions, args.eval_eps,
      args.match_start, args.match_end)
  agent = make_agent()
  replay = make_replay()
  logger = make_logger()
  step = logger.step
  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  stream_train = iter(agent.stream(make_stream(replay, 'train')))
  carry_train = [agent.init_train(args.batch_size)]
  last_batch = [None]
  agg = elements.Agg()

  cp = elements.Checkpoint(elements.Path(str(logdir / 'ckpt')))
  cp.step = step
  cp.agent = agent
  cp.replay = replay
  cp.protocol = state
  cp.load_or_save()
  if int(step) != state.actions or int(agent.n_updates) != state.updates:
    raise ValueError('Checkpoint counters disagree')
  if state.actions:
    if state.actions < state.budget:
      raise RuntimeError(
          'Training continuation refused: environment and replay RNG state '
          'are not completely checkpointed; start a fresh run')
    if len(state.evaluations) == len(state.grid):
      print('Completed protocol run already checkpointed')
      return
    raise RuntimeError(
        'Terminal evaluation recovery needs the fixed endpoint snapshot')

  manifest = dict(
      protocol='M2-v1', task=config.task, root_seed=config.seed,
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
  (logdir / 'protocol_manifest.json').write_text(
      json.dumps(manifest, sort_keys=True, indent=2, default=str),
      encoding='utf-8')

  def on_transition(tran, worker):
    state.on_train_transition(tran)
    if not tran['is_first']:
      step.increment()
    replay.add(tran, worker)
    if len(replay) < batch_steps:
      return
    for _ in range(should_train(step)):
      start_action = state.actions
      update_id = state.updates
      batch = next(stream_train)
      last_batch[0] = dict(batch)
      carry_train[0], delayed_outs, delayed_mets = agent.train(
          carry_train[0], dict(batch))
      if delayed_outs or delayed_mets:
        raise RuntimeError('Unconsumed delayed train result')
      outs, metrics = agent.take_train_result()
      grad_norm = float(metrics['opt/grad_norm'])
      finite = math.isfinite(grad_norm) and math.isfinite(
          float(metrics['opt/loss']))
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
    grid_index = point // state.interval
    agent.sync_policy()
    snapshot = elements.Checkpoint(elements.Path(
        str(logdir / 'eval_snapshots' / f'{point:07d}')))
    snapshot.agent = agent
    snapshot.save()
    scores, lengths, seeds = [], [], []
    for episode_id in range(state.episodes):
      seed = seed32(config.seed, config.task, 'eval_env',
                    grid_index, episode_id)
      seeds.append(seed)
      env = make_env(0, seed=seed)
      evaluator = embodied.Driver([lambda env=env: env], parallel=False)
      score, length = [0.0], [0]
      def on_eval(tran, _):
        if not tran['is_first']:
          state.eval_actions += 1
          length[0] += 1
        score[0] += float(tran['reward'])
      evaluator.on_step(on_eval)
      agent.set_eval_seed(seed)
      evaluator.reset(agent.init_policy)
      evaluator(lambda *xs: agent.policy(*xs, mode='eval'), episodes=1)
      evaluator.close()
      scores.append(score[0])
      lengths.append(length[0])
    state.record_evaluation(point, scores, lengths, seeds)
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
        metrics = report_on_batch(
            agent, last_batch[0], state.reports, config.seed, config.task,
            config.report_length + config.replay_context)
        state.reports += 1
        logger.add(metrics, prefix='report')
        next_report += int(args.report_every_actions)
      if state.actions == boundary:
        evaluate(boundary)
        logger.add(agg.result())
        logger.write()
    if config.agent.rep_probe.mode == 'logging':
      c = state.frozen_c()
      (logdir / 'matching_result.json').write_text(
          json.dumps(dict(match_sum=state.match_sum,
                          match_count=state.match_count, c=c,
                          start=state.match_start, end=state.match_end),
                     sort_keys=True, indent=2), encoding='utf-8')
    cp.save()
  finally:
    driver.close()
    logger.close()
