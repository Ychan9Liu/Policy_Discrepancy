"""Full-window current-rep gradient-stop helper for an adopted finite assay.

This module has no CLI, source/checkpoint loader, simulator or device setup.
The finite diagnostic is adopted within Agent-authorized prevalidation;
it is not a formal method/configuration or evidence of method effectiveness.
The caller must
authenticate the old B receipts, original checkpoint/calibration/contexts,
own the execution budget, and separately specify native actor evaluation.

All 19 producer files of the completed 0/B/R object remain unchanged. The
original helper generates this assay's batch and step IDs, so its layout
VERSION, normal train counters and random stream are not replaced by A IDs.
CPU construction fixtures establish interfaces, never real learning effects.
"""

import contextlib
import functools
import math

from scripts import dl_reset_learning as original_learning

VERSION = 'DL-reset-allrep-learning-r1'
LAYOUT_VERSION = 'DL-reset-learning-r1'
GROUP = 'A'
UPDATES = original_learning.UPDATES
BATCH_SIZE = original_learning.BATCH_SIZE
LOSS_LENGTH = original_learning.LOSS_LENGTH
CONTEXT = original_learning.CONTEXT

require = original_learning.require
tree_digest = original_learning.tree_digest


@contextlib.contextmanager
def allrep_branch(rssm_class=None):
  """Install before Agent construction; stop only the entire rep loss branch.

  One original loss call supplies every forward value and RNG consumption.
  Python ownership flags introduce no Ninjax variables or model parameters.
  The optional class argument is for construction fixtures, not provenance.
  """
  if rssm_class is None:
    from dreamerv3.rssm import RSSM
    rssm_class = RSSM
  loss_before = rssm_class.loss
  init_before = rssm_class.__init__
  require(not getattr(loss_before, '_dl_reset_patch_owned', False) and
      not getattr(loss_before, '_dl_allrep_patch_owned', False),
      'Nested original B/R or full-window A patch forbidden')
  construction_token = object()

  @functools.wraps(init_before)
  def initialize(self, *args, **kwargs):
    init_before(self, *args, **kwargs)
    self._dl_allrep_branch_at_construction = GROUP
    self._dl_allrep_stop_at_construction = True
    self._dl_allrep_loss_identity_at_construction = id(loss)
    self._dl_allrep_construction_token = construction_token

  @functools.wraps(loss_before)
  def loss(self, carry, tokens, acts, reset, training, rep_probe=False):
    import jax
    import jax.numpy as jnp
    result = loss_before(self, carry, tokens, acts, reset, training, rep_probe)
    carry, entries, losses, features, metrics, probe = result
    if training:
      rep = losses['rep']
      require(rep.shape == reset.shape and reset.dtype == jnp.bool_,
          'Full-window rep/reset BT contract differs')
      require(self.free_nats == 1, 'Original free-nats1 pressure contract differs')
      active = rep > 1
      excess = jnp.maximum(rep - 1, 0)
      metrics = {**metrics,
          'dl_reset/active_count': jnp.sum(reset & active),
          'dl_reset/excess_sum': jnp.sum(jnp.where(reset, excess, 0)),
          'dl_reset/reset_count': jnp.sum(reset),
          'dl_allrep/active_count': jnp.sum(active),
          'dl_allrep/excess_sum': jnp.sum(excess),
          'dl_allrep/position_count': jnp.int32(rep.size),
          'dl_allrep/nonreset_active_count': jnp.sum(~reset & active),
          'dl_allrep/nonreset_excess_sum': jnp.sum(jnp.where(~reset, excess, 0))}
      losses = {**losses, 'rep': jax.lax.stop_gradient(rep)}
    return carry, entries, losses, features, metrics, probe

  loss._dl_allrep_patch_owned = True
  loss._dl_allrep_gradient_scope = 'entire_rep_loss_window'
  loss._dl_allrep_construction_token = construction_token
  rssm_class.__init__ = initialize
  rssm_class.loss = loss
  try:
    yield
  finally:
    rssm_class.loss = loss_before
    rssm_class.__init__ = init_before


def _assert_a_agent(agent):
  config, model = agent.config, agent.model
  require(config.batch_size == BATCH_SIZE and config.batch_length == LOSS_LENGTH and
      config.replay_context == CONTEXT, 'Requires original B16/T64/context1')
  require(not agent.jaxcfg.enable_policy, 'Offline A must disable policy synchronization')
  require(model.config.rep_probe.mode == model.config.dt_latch.mode == 'off',
      'Existing probe/weighting/logging overlays must remain off')
  require(model.dyn.free_nats == 1 and model.scales['rep'] == .1 and
      model.scales['dyn'] == 1 and not model.config.ac_grads and
      model.config.reward_grad and model.config.repval_grad,
      'Original loss/free-nats/gradient configuration differs')
  loss = type(model.dyn).loss
  require(getattr(loss, '_dl_allrep_patch_owned', False) is True and
      getattr(loss, '_dl_allrep_gradient_scope', None) == 'entire_rep_loss_window' and
      not getattr(loss, '_dl_reset_patch_owned', False),
      'Independent A wrapper must own the entire rep loss window')
  require(getattr(model.dyn, '_dl_allrep_branch_at_construction', None) == GROUP and
      getattr(model.dyn, '_dl_allrep_stop_at_construction', False) is True,
      'A wrapper must precede Agent construction and compiled train creation')
  require(getattr(model.dyn, '_dl_allrep_loss_identity_at_construction', None) == id(loss) and
      getattr(model.dyn, '_dl_allrep_construction_token', None) is
      getattr(loss, '_dl_allrep_construction_token', None),
      'A compiled Agent must belong to this exact construction-time wrapper')
  require(original_learning.VERSION == LAYOUT_VERSION and
      (BATCH_SIZE, LOSS_LENGTH, CONTEXT, UPDATES) == (16, 64, 1, 128),
      'Frozen original batch layout/version/update count differs')


def _metric_scalar(metrics, name, *, count=False):
  import numpy as np
  value = np.asarray(metrics.get(name))
  require(value.shape == () and value.dtype.kind in 'iuf' and np.isfinite(value),
      'Missing/nonfinite scalar pressure metric: ' + name)
  number = int(value) if count else float(value)
  require(float(value) == number and number >= 0,
      'Pressure must be nonnegative with exact integral counts: ' + name)
  return number


def _pressure(metrics):
  reset = dict(active_count=_metric_scalar(metrics, 'dl_reset/active_count', count=True),
      excess_sum=_metric_scalar(metrics, 'dl_reset/excess_sum'),
      reset_count=_metric_scalar(metrics, 'dl_reset/reset_count', count=True))
  whole = dict(active_count=_metric_scalar(metrics, 'dl_allrep/active_count', count=True),
      excess_sum=_metric_scalar(metrics, 'dl_allrep/excess_sum'),
      position_count=_metric_scalar(metrics, 'dl_allrep/position_count', count=True),
      nonreset_active_count=_metric_scalar(metrics, 'dl_allrep/nonreset_active_count', count=True),
      nonreset_excess_sum=_metric_scalar(metrics, 'dl_allrep/nonreset_excess_sum'))
  require(reset['reset_count'] == 16 and 0 <= reset['active_count'] <= 16 and
      whole['position_count'] == 1024 and 0 <= whole['active_count'] <= 1024 and
      0 <= whole['nonreset_active_count'] <= 1008 and
      whole['active_count'] == reset['active_count'] + whole['nonreset_active_count'],
      'Original whole-window/reset pressure position/count invariants differ')
  for count, excess in ((reset['active_count'], reset['excess_sum']),
      (whole['active_count'], whole['excess_sum']),
      (whole['nonreset_active_count'], whole['nonreset_excess_sum'])):
    require((count > 0) == (excess > 0), 'Active/excess pressure invariant differs')
  # The three positive F32 reductions can have different association. Do not
  # invent a tolerance or require whole_excess == reset_excess+nonreset_excess.
  return reset, whole


def run_a(agent, checkpoint, calibration, source_hashes, frozen_context_entries,
          reset_feature_callback=None, check=lambda: None, *, reference_b):
  """Adopted finite 128-update diagnostic with original B input/key contracts.

  reference_b is the learning dictionary from authenticated B-complete.json.
  This helper matches its values, without performing filesystem provenance
  authentication. Native evaluation needs a fresh loaded Agent and its own
  frozen evaluation protocol/budget.
  """
  import numpy as np
  check(); _assert_a_agent(agent)
  require(isinstance(checkpoint, dict) and set(checkpoint) == {'params', 'counters'},
      'Complete original Agent checkpoint required; no weights-only/regex restore')
  for name in ('updates', 'batches', 'actions', 'eval_actions', 'eval_seed'):
    require(type(checkpoint['counters'].get(name)) is int and checkpoint['counters'][name] >= 0,
        'Missing/invalid original checkpoint counter ' + name)
  source_checkpoint_digest = tree_digest(checkpoint)
  agent.load(checkpoint)
  initial = agent.save()
  require(tree_digest(initial['params']) == tree_digest(checkpoint['params']),
      'Full loaded params/optimizer/norm/target state differs')
  expected = {**checkpoint['counters'], 'batches': checkpoint['counters']['updates']}
  require(initial['counters'] == expected, 'Standard Agent.load counter semantics differ')
  initial_step = original_learning.optimizer_step(initial['params'])
  cal_digest = tree_digest({str(k): v for k, v in calibration.items()})
  # Do not substitute this module's VERSION into the established step IDs.
  raw, binding = original_learning.offline_batch(
      agent, calibration, source_hashes, frozen_context_entries)
  require(isinstance(reference_b, dict) and reference_b.get('version') == LAYOUT_VERSION and
      reference_b.get('group') == 'B' and reference_b.get('updates') == UPDATES and
      len(reference_b.get('receipts', [])) == UPDATES,
      'Complete original B learning reference required')
  for name, value in dict(source_checkpoint_digest=source_checkpoint_digest,
      initial_state_digest=tree_digest(initial), initial_params_digest=tree_digest(initial['params']),
      calibration_arrays_digest=cal_digest, binding=binding,
      restored_counters=expected, saved_source_counters=checkpoint['counters']).items():
    require(reference_b.get(name) == value, 'Original B matched input/state differs: ' + name)
  require(reference_b.get('optimizer_counter') == dict(key=original_learning.OPTIMIZER_STEP_KEY,
      dtype='int32', shape=[], initial=initial_step, final=initial_step+UPDATES, updates=UPDATES),
      'Original B official optimizer counter reference differs')
  features0 = original_learning._readonly_features(agent, reset_feature_callback, 0)
  require(original_learning._counters(agent) == expected, 'Feature callback changed counters')
  rows = []
  for index in range(UPDATES):
    check(); _assert_a_agent(agent)
    before = original_learning._counters(agent)
    require(before == {**expected, 'updates': expected['updates'] + index,
        'batches': expected['batches'] + index}, 'Unexpected pre-update counters')
    carry = agent.init_train(BATCH_SIZE)
    prepared = agent.prepare_batch({k: v.copy() for k, v in raw.items()})
    seed = prepared.get('seed')
    require(seed is not None, 'Actual normal prepare_batch must supply train seed')
    if type(agent).__module__ == 'embodied.jax.agent':
      import jax
      with jax._src.config.explicit_device_get_scope():
        actual_seed = np.asarray(jax.device_get(seed)).copy()
        normal_seed = np.asarray(jax.device_get(agent._seeds(
            before['batches'], agent.train_mirrored))).copy()
    else:
      actual_seed = np.asarray(seed).copy()
      normal_seed = np.asarray(agent._seeds(before['batches'], agent.train_mirrored)).copy()
    require(actual_seed.dtype == np.uint32 and actual_seed.shape == (2,) and
        np.array_equal(actual_seed, normal_seed), 'Actual normal train counter/key differs')
    reference_row = reference_b['receipts'][index]
    require(reference_row['update'] == index+1 and
        reference_row['seed_counter'] == before['batches'] and
        reference_row['train_seed'] == actual_seed.astype(int).tolist(),
        'A actual normal train root differs from original B receipt')
    agent.train(carry, prepared)
    _, metrics = agent.take_train_result()
    require(isinstance(metrics, dict) and metrics, 'Missing current completed train metrics')
    digest = tree_digest(metrics)
    reset, whole = _pressure(metrics)
    opt_metric = np.asarray(metrics.get('opt/updates'))
    require(opt_metric.shape == () and opt_metric.dtype == np.int32 and
        int(opt_metric) == initial_step + index + 1,
        'Actual official optimizer metric did not advance exactly once')
    after = original_learning._counters(agent)
    require(after == {**expected, 'updates': expected['updates'] + index + 1,
        'batches': expected['batches'] + index + 1}, 'Full train counter progression differs')
    require(reference_row['counters'] == after, 'A counters differ from original B receipt')
    rows.append(dict(update=index + 1, seed_counter=before['batches'],
        train_seed=actual_seed.astype(int).tolist(), counters=after, metrics=metrics,
        metrics_digest=digest, reset_pressure=reset, whole_window_pressure=whole,
        optimizer_step=int(opt_metric)))
    require(tree_digest(raw) == binding['batch_digest'], 'Original frozen common batch mutated')
    check()
  features128 = original_learning._readonly_features(agent, reset_feature_callback, UPDATES)
  final = agent.save()
  require(tree_digest({str(k): v for k, v in calibration.items()}) == cal_digest,
      'Caller calibration arrays changed during learning/features')
  require(tree_digest({str(k): v for k, v in frozen_context_entries.items()}) ==
      binding['context_entries_digest'], 'Original frozen terminal context changed')
  require(tree_digest(checkpoint) == source_checkpoint_digest, 'Input original checkpoint mutated')
  require(final['counters'] == {**expected, 'updates': expected['updates'] + UPDATES,
      'batches': expected['batches'] + UPDATES}, 'Final standard counters differ')
  final_step = original_learning.optimizer_step(final['params'])
  require(final_step == initial_step + UPDATES, 'Complete optimizer did not perform exactly 128 updates')
  check()
  return dict(version=VERSION, protocol_adopted=True,
      working_draft_not_adopted=False, group=GROUP,
      updates=UPDATES, layout_version=LAYOUT_VERSION, binding=binding,
      source_checkpoint_digest=source_checkpoint_digest,
      initial_state_digest=tree_digest(initial), initial_params_digest=tree_digest(initial['params']),
      calibration_arrays_digest=cal_digest,
      final_state_digest=tree_digest(final), final_params_digest=tree_digest(final['params']),
      optimizer_counter=dict(key=original_learning.OPTIMIZER_STEP_KEY, dtype='int32', shape=[],
          initial=initial_step, final=final_step, updates=UPDATES),
      restored_counters=expected, saved_source_counters=checkpoint['counters'].copy(),
      receipts=rows, features={0: features0, UPDATES: features128}, final_checkpoint=final,
      reset_rep_gradient_stopped=True, entire_window_rep_gradient_stopped=True,
      new_rep_gradient_scope='entire_rep_loss_window', rep_forward_values_retained=True,
      other_loss_branches_retained=True, posterior_features_and_recurrent_carry_not_stopped=True,
      historical_optimizer_momentum_retained=True, per_update_init_train=True,
      reset_pressure=dict(observation='actual original post-free-nats rep before full-window sg',
          active_count=sum(r['reset_pressure']['active_count'] for r in rows),
          reset_count=sum(r['reset_pressure']['reset_count'] for r in rows),
          excess_sum=math.fsum(r['reset_pressure']['excess_sum'] for r in rows),
          entire_128_update_metrics_observed=True),
      whole_window_pressure=dict(observation='actual original post-free-nats rep before full-window sg',
          active_count=sum(r['whole_window_pressure']['active_count'] for r in rows),
          position_count=sum(r['whole_window_pressure']['position_count'] for r in rows),
          excess_sum=math.fsum(r['whole_window_pressure']['excess_sum'] for r in rows),
          nonreset_active_count=sum(r['whole_window_pressure']['nonreset_active_count'] for r in rows),
          nonreset_excess_sum=math.fsum(r['whole_window_pressure']['nonreset_excess_sum'] for r in rows),
          entire_128_update_metrics_observed=True),
      total_loss_positions=UPDATES * 1024, total_reset_loss_positions=UPDATES * 16,
      source_authentication_performed=False, original_B_comparison_authenticated=False,
      original_B_receipt_values_matched=True,
      simulator_calls=0, native_actor_evaluation_performed=False,
      method_effectiveness_established=False,
      scope='array-only finite assay; caller owns authenticated inputs, resource/budget and native evaluation')
