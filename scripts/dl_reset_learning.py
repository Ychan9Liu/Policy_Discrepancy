"""DL reset learning: bounded offline full-Agent.train B/R helper.

This module has no CLI, filesystem/source loader, simulator or authorization
logic. Its caller must authenticate the checkpoint, eight complete calibration
episodes, and frozen baseline terminal entries before calling it. B and R run
in separate owned workers. Install rssm_branch BEFORE constructing the Agent.
Synthetic fixtures exercise this helper, never establish actual learning.
"""

import contextlib
import functools
import hashlib
import json
import math

VERSION = 'DL-reset-learning-r1'
EPISODES = (9, 14, 15, 18, 20, 24, 30, 31)
UPDATES = 128
BATCH_SIZE, LOSS_LENGTH, CONTEXT = 16, 64, 1
FIELDS = ('image', 'reward', 'is_first', 'is_last', 'is_terminal', 'action')
OPTIMIZER_STEP_KEY = 'opt/step/value'


def require(condition, message):
  if not condition:
    raise ValueError(message)


def tree_digest(value):
  """Include shape, dtype, container identity, and signed-zero bytes."""
  import numpy as np
  digest = hashlib.sha256()

  def visit(item):
    if isinstance(item, dict):
      require(all(isinstance(k, str) for k in item), 'String tree keys required')
      digest.update(b'dict')
      for key in sorted(item):
        digest.update(json.dumps(key).encode())
        visit(item[key])
    elif isinstance(item, (tuple, list)):
      digest.update(type(item).__name__.encode())
      digest.update(str(len(item)).encode())
      for child in item:
        visit(child)
    elif item is None or type(item) in (bool, int, float, str):
      require(type(item) is not float or np.isfinite(item), 'Nonfinite tree scalar')
      digest.update(type(item).__name__.encode())
      digest.update(json.dumps(item).encode())
    else:
      array = np.asarray(item)
      numeric = array.dtype.kind in 'buifc'
      if array.dtype.kind == 'V':
        import ml_dtypes
        numeric = array.dtype == np.dtype(ml_dtypes.bfloat16)
      require(numeric, 'Numeric tree leaf required; arbitrary void/object is forbidden')
      require(np.isfinite(array).all(), 'Nonfinite tree array')
      digest.update(json.dumps([array.dtype.str, str(array.dtype)]).encode())
      digest.update(json.dumps(array.shape).encode())
      digest.update(array.tobytes(order='C'))
  visit(value)
  return digest.hexdigest()


def stop_reset_rep(rep, reset, training=True):
  """Stop only new reset-position rep gradients; retain forward values."""
  import jax
  import jax.numpy as jnp
  require(rep.shape == reset.shape, 'Rep/reset BT shape differs')
  require(reset.dtype == jnp.bool_, 'Boolean reset mask required')
  return jnp.where(reset, jax.lax.stop_gradient(rep), rep) if training else rep


@contextlib.contextmanager
def rssm_branch(group, rssm_class=None):
  """Process-local patch, no source edits or added model/optimizer variables.

  The optional class argument is for small contract fixtures, not a production
  identity receipt. The actual runner uses the default official RSSM class.
  """
  require(group in ('B', 'R'), 'Only fixed B/R groups allowed')
  if rssm_class is None:
    from dreamerv3.rssm import RSSM
    rssm_class = RSSM
  original = rssm_class.loss
  original_init = rssm_class.__init__
  require(not getattr(original, '_dl_reset_patch_owned', False), 'Nested B/R patch forbidden')
  @functools.wraps(original_init)
  def protected_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self._dl_reset_learning_branch_at_construction = group
    self._dl_reset_rep_stop_at_construction = (group == 'R')
  @functools.wraps(original)
  def protected(self, carry, tokens, acts, reset, training, rep_probe=False):
    import jax.numpy as jnp
    result = original(self, carry, tokens, acts, reset, training, rep_probe)
    carry, entries, losses, feat, metrics, probe = result
    if training:
      rep = losses['rep']
      require(self.free_nats == 1, 'Reset pressure observation requires original free-nats1')
      metrics = {**metrics,
          'dl_reset/active_count': jnp.sum(reset & (rep > 1)),
          'dl_reset/excess_sum': jnp.sum(jnp.where(reset, jnp.maximum(rep - 1, 0), 0)),
          'dl_reset/reset_count': jnp.sum(reset)}
    if group == 'R':
      losses = {**losses, 'rep': stop_reset_rep(losses['rep'], reset, training)}
    return carry, entries, losses, feat, metrics, probe
  protected._dl_reset_rep_stop = (group == 'R')
  protected._dl_reset_learning_wrapper = group
  protected._dl_reset_patch_owned = True
  rssm_class.loss = protected
  rssm_class.__init__ = protected_init
  try:
    yield
  finally:
    rssm_class.loss = original
    rssm_class.__init__ = original_init


def _numeric(value, dtype, shape, name):
  import numpy as np
  require(isinstance(value, np.ndarray) and value.dtype == dtype and
      value.shape == shape and np.isfinite(value).all(), f'Invalid {name}')


def validate_calibration(calibration, source_hashes):
  import numpy as np
  require(isinstance(calibration, dict) and set(calibration) == set(EPISODES),
      'Exactly the eight fixed calibration episodes required')
  require(isinstance(source_hashes, dict) and set(source_hashes) == set(EPISODES),
      'Exactly eight authenticated source hashes required')
  for episode in EPISODES:
    pin = source_hashes[episode]
    require(isinstance(pin, str) and len(pin) == 64 and
        all(c in '0123456789abcdef' for c in pin), 'Invalid source hash')
    row = calibration[episode]
    require(isinstance(row, dict) and set(FIELDS + ('action_executed',)) <= set(row),
        'Complete real observations/actions/execution mask required')
    _numeric(row['image'], np.dtype('uint8'), (1001, 64, 64, 3), 'source RGB')
    _numeric(row['reward'], np.dtype('float32'), (1001,), 'source rewards')
    _numeric(row['action'], np.dtype('float32'), (1001, 12), 'source raw actions')
    for name in ('is_first', 'is_last', 'is_terminal', 'action_executed'):
      _numeric(row[name], np.dtype('bool'), (1001,), name)
    require(row['is_first'][0] and not row['is_first'][1:].any(), 'Genuine reset differs')
    require(row['is_last'][-1] and not row['is_last'][:-1].any(), 'Complete episode differs')
    require(not row['is_terminal'][:-1].any(), 'Unexpected early terminal')
    require(row['reward'][0] == 0, 'Reset reward differs')
    require(row['action_executed'][:-1].all() and not row['action_executed'][-1],
        'Source real-action/terminal-placeholder execution differs')
    require(np.array_equal(row['action'][-1], np.zeros(12, np.float32)),
        'Terminal raw action must be the unexecuted zero placeholder')


def offline_batch(agent, calibration, source_hashes, frozen_context_entries):
  """Real terminal from previous cal episode + current reset0..63, twice.

  This is an explicit concatenation of independently collected episodes, not
  an original continuous worker/replay restoration. Terminal cached entries
  come from the caller's authenticated baseline full-history reconstruction.
  Replay's usual first-column annotation is applied only in the copied batch.
  """
  import numpy as np
  validate_calibration(calibration, source_hashes)
  require(isinstance(frozen_context_entries, dict) and
      set(frozen_context_entries) == set(EPISODES), 'Eight frozen terminal entries required')
  expected_extra = set(agent.spaces) - set(FIELDS) - {'consec', 'stepid'}
  require(expected_extra and 'dyn/deter' in expected_extra and 'dyn/stoch' in expected_extra,
      'Production context1 RSSM entries required')
  raw = agent._zeros(agent.spaces, (BATCH_SIZE, LOSS_LENGTH + CONTEXT))
  require(set(raw) == set(agent.spaces) and set(FIELDS) <= set(raw), 'Agent spaces differ')
  layout = []
  for row_index, episode in enumerate(EPISODES * 2):
    context_episode = EPISODES[(EPISODES.index(episode) - 1) % len(EPISODES)]
    previous, current = calibration[context_episode], calibration[episode]
    for field in FIELDS:
      raw[field][row_index, 0] = previous[field][-1]
      raw[field][row_index, 1:] = current[field][:LOSS_LENGTH]
    # core Replay._annotate_batch(data, ..., True) marks sampled prefix first.
    raw['is_first'][row_index, 0] = True
    raw['consec'][row_index] = 0
    identities = [(context_episode, 1000)] + [(episode, t) for t in range(LOSS_LENGTH)]
    for col, (ep, position) in enumerate(identities):
      raw['stepid'][row_index, col] = np.frombuffer(hashlib.sha256(
          f'{VERSION}:{source_hashes[ep]}:{position}'.encode()).digest()[:20], np.uint8)
    entries = frozen_context_entries[context_episode]
    require(isinstance(entries, dict) and set(entries) == expected_extra,
        'Frozen terminal entry keys differ')
    for field in sorted(expected_extra):
      value = entries[field]
      _numeric(value, raw[field].dtype, raw[field].shape[2:], f'context {field}')
      raw[field][row_index, 0] = value
    stoch = entries['dyn/stoch']
    require(stoch.ndim == 2 and np.all((stoch == 0) | (stoch == 1)) and
        np.all(stoch.sum(-1) == 1), 'Context stochastic entry is not a real categorical sample')
    layout.append(dict(row=row_index, episode=episode, context_episode=context_episode,
        context_position=1000, loss_positions=[0, 63],
        terminal_action_executed=False, context_first_is_replay_annotation=True))
  return raw, dict(layout=layout, batch_digest=tree_digest(raw),
      context_entries_digest=tree_digest({str(k): v for k, v in frozen_context_entries.items()}),
      source_hashes={str(k): v for k, v in source_hashes.items()},
      offline_episode_concatenation=True, original_continuous_replay_restored=False,
      cached_entries='caller reconstructed baseline full histories; common frozen stale entries',
      diagnostic_step_ids=True, balanced_duplicates_per_episode=2,
      reset_loss_positions_per_update=16, loss_positions_per_update=1024)


def _counters(agent):
  return dict(updates=int(agent.n_updates), batches=int(agent.n_batches),
      actions=int(agent.n_actions), eval_actions=int(agent.n_eval_actions),
      eval_seed=int(agent.eval_seed))


def optimizer_step(params):
  """Read the official Ninjax Variable leaf without converting its dtype.

  Optimizer.step is a Variable named 'step'; Variable.read creates the leaf
  'value'. A sibling opt/step is not this optimizer's counter.
  """
  import numpy as np
  value = params.get(OPTIMIZER_STEP_KEY)
  require(value is not None, 'Complete original optimizer step required: ' + OPTIMIZER_STEP_KEY)
  array = np.asarray(value)
  require(array.shape == () and array.dtype == np.int32 and
      np.isfinite(array) and int(array) >= 0,
      'Original optimizer step must be a finite nonnegative int32 scalar')
  return int(array)


def _readonly_features(agent, callback, stage):
  before = tree_digest(agent.save())
  result = callback(agent, stage) if callback is not None else None
  require(tree_digest(agent.save()) == before, 'Reset feature callback modified full Agent state')
  return result


def run_branch(agent, checkpoint, calibration, source_hashes,
               frozen_context_entries, group, reset_feature_callback=None):
  """Execute exactly 128 full normal updates on one preconstructed B/R Agent.

  Returns an in-memory complete Agent save plus reset feature bundles; caller
  owns authenticated storage, cal readout freeze and later single blind open.
  No outcome-dependent update count, readout fit, physics or blind access here.
  """
  import numpy as np
  require(group in ('B', 'R'), 'Only B/R groups allowed')
  config = agent.config
  require(config.batch_size == 16 and config.batch_length == 64 and
      config.replay_context == 1, 'Requires B16/T64/context1')
  require(not agent.jaxcfg.enable_policy, 'Offline learning must disable policy synchronization')
  model = agent.model
  require(model.config.rep_probe.mode == 'off' and model.config.dt_latch.mode == 'off',
      'Normal B/R must disable existing weighting/logging overlays')
  require(model.dyn.free_nats == 1 and model.scales['rep'] == .1 and
      model.scales['dyn'] == 1 and not model.config.ac_grads,
      'Original free-nats/scales/gradient configuration differs')
  require(model.config.reward_grad and model.config.repval_grad,
      'Original reward/repval gradient configuration differs')
  require(bool(getattr(type(model.dyn).loss, '_dl_reset_rep_stop', False)) == (group == 'R'),
      'RSSM patch must match branch and be installed before Agent construction')
  require(getattr(type(model.dyn).loss, '_dl_reset_learning_wrapper', None) == group,
      'B/R readonly pressure-observation wrapper must match branch')
  require(bool(getattr(model.dyn, '_dl_reset_rep_stop_at_construction', False)) == (group == 'R'),
      'RSSM patch must precede construction; late patch cannot change compiled train')
  require(getattr(model.dyn, '_dl_reset_learning_branch_at_construction', None) == group,
      'B/R observation wrapper must precede Agent construction')
  require(isinstance(checkpoint, dict) and set(checkpoint) == {'params', 'counters'},
      'Complete Agent checkpoint required; no weights-only or regex restore')
  for name in ('updates', 'batches', 'actions', 'eval_actions', 'eval_seed'):
    require(type(checkpoint['counters'].get(name)) is int and
        checkpoint['counters'][name] >= 0, f'Missing/invalid checkpoint counter {name}')
  checkpoint_digest = tree_digest(checkpoint)
  agent.load(checkpoint)
  initial = agent.save()
  require(tree_digest(initial['params']) == tree_digest(checkpoint['params']),
      'Full loaded params/optimizer/norm/target state differs')
  expected = {**checkpoint['counters'], 'batches': checkpoint['counters']['updates']}
  require(initial['counters'] == expected, 'Standard Agent.load counter semantics differ')
  initial_opt_step = optimizer_step(initial['params'])
  calibration_digest = tree_digest({str(k): v for k, v in calibration.items()})
  raw, binding = offline_batch(agent, calibration, source_hashes, frozen_context_entries)
  initial_features = _readonly_features(agent, reset_feature_callback, 0)
  require(_counters(agent) == expected, 'Feature callback changed counters')
  receipts = []
  for index in range(UPDATES):
    before = _counters(agent)
    require(before == {**expected, 'updates': expected['updates'] + index,
        'batches': expected['batches'] + index}, 'Unexpected pre-update counters')
    carry = agent.init_train(BATCH_SIZE)
    prepared = agent.prepare_batch({key: value.copy() for key, value in raw.items()})
    seed = prepared.get('seed')
    require(seed is not None, 'Actual prepare_batch must supply its normal train seed')
    # Scalar seed placement follows actual Agent transfer guard, when available.
    if type(agent).__module__ == 'embodied.jax.agent':
      import jax
      with jax._src.config.explicit_device_get_scope():
        actual_seed = np.asarray(jax.device_get(seed)).copy()
        canonical_seed = np.asarray(jax.device_get(agent._seeds(
            before['batches'], agent.train_mirrored))).copy()
    else:
      actual_seed = np.asarray(seed).copy()
      canonical_seed = np.asarray(agent._seeds(before['batches'], agent.train_mirrored)).copy()
    require(actual_seed.dtype == np.uint32 and actual_seed.shape == (2,) and
        np.array_equal(actual_seed, canonical_seed), 'Prepared train seed differs from normal counter stream')
    agent.train(carry, prepared)
    _, metrics = agent.take_train_result()
    require(isinstance(metrics, dict) and metrics, 'Missing completed train metrics')
    metrics_digest = tree_digest(metrics)  # Reject all nonfinite numeric metrics.
    pressure = {}
    for name in ('active_count', 'excess_sum', 'reset_count'):
      value = np.asarray(metrics.get('dl_reset/' + name))
      require(value.shape == () and value.dtype.kind in 'iuf' and np.isfinite(value),
          'Missing/nonfinite actual reset pressure metric')
      pressure[name] = float(value) if name == 'excess_sum' else int(value)
      require(float(value) == pressure[name], 'Reset pressure count must be an exact integer')
    require(pressure['reset_count'] == 16 and 0 <= pressure['active_count'] <= 16 and
        pressure['excess_sum'] >= 0 and
        (pressure['excess_sum'] > 0) == (pressure['active_count'] > 0),
        'Actual reset pressure metric bounds/invariants differ')
    after = _counters(agent)
    require(after == {**expected, 'updates': expected['updates'] + index + 1,
        'batches': expected['batches'] + index + 1}, 'Full train counter progression differs')
    receipts.append(dict(update=index + 1, seed_counter=before['batches'],
        train_seed=actual_seed.astype(int).tolist(), counters=after, metrics=metrics,
        metrics_digest=metrics_digest, reset_pressure=pressure))
    require(tree_digest(raw) == binding['batch_digest'], 'Frozen common batch mutated')
  final_features = _readonly_features(agent, reset_feature_callback, UPDATES)
  final = agent.save()
  require(tree_digest({str(k): v for k, v in calibration.items()}) == calibration_digest,
      'Caller calibration arrays changed during learning/features')
  require(tree_digest({str(k): v for k, v in frozen_context_entries.items()}) ==
      binding['context_entries_digest'], 'Frozen baseline terminal entries changed')
  require(tree_digest(checkpoint) == checkpoint_digest, 'Input checkpoint mutated')
  require(final['counters'] == {**expected, 'updates': expected['updates'] + UPDATES,
      'batches': expected['batches'] + UPDATES}, 'Final standard counters differ')
  final_opt_step = optimizer_step(final['params'])
  require(final_opt_step == initial_opt_step + UPDATES,
      'Complete optimizer did not perform exactly 128 updates')
  return dict(version=VERSION, group=group, updates=UPDATES, binding=binding,
      source_checkpoint_digest=checkpoint_digest, initial_state_digest=tree_digest(initial),
      calibration_arrays_digest=calibration_digest,
      initial_params_digest=tree_digest(initial['params']), final_state_digest=tree_digest(final),
      final_params_digest=tree_digest(final['params']), restored_counters=expected,
      optimizer_counter=dict(key=OPTIMIZER_STEP_KEY, dtype='int32', shape=[],
          initial=initial_opt_step, final=final_opt_step, updates=UPDATES),
      saved_source_counters=checkpoint['counters'].copy(), receipts=receipts,
      features={0: initial_features, UPDATES: final_features}, final_checkpoint=final,
      reset_rep_gradient_stopped=(group == 'R'), rep_forward_values_retained=True,
      historical_optimizer_momentum_retained=True, per_update_init_train=True,
      reset_pressure=dict(observation='actual original post-free-nats rep before reset sg',
          active_count=sum(row['reset_pressure']['active_count'] for row in receipts),
          reset_count=sum(row['reset_pressure']['reset_count'] for row in receipts),
          excess_sum=math.fsum(row['reset_pressure']['excess_sum'] for row in receipts),
          entire_128_update_metrics_observed=True),
      total_loss_positions=UPDATES * 1024, total_reset_loss_positions=UPDATES * 16,
      source_authentication_performed=False, blind_arrays_read=False,
      simulator_calls=0, method_effectiveness_established=False,
      scope='helper calls and state/counter checks; caller authenticates real source and runtime')
