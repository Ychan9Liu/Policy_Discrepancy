"""Candidate engineering negative controls for pressure-observation graphs.

No loader, CLI, source authentication, simulator, policy or device setup.
Install observation_branch BEFORE constructing a fresh Agent. The caller
authenticates both historical learning receipts and owns resource/deadline.
Differences are evidence: complete all128 updates and return their verdict.
Bad finite/state/counter/input contracts raise instead of publishing success.
Fixtures do not establish size50m CUDA equivalence or a scientific result.
"""

import contextlib
import functools
import hashlib
import struct

from scripts import dl_reset_learning as original

VERSION = 'DL-rep-observation-guard-candidate-r0'
MODES = ('Bwhole', 'Amin')
UPDATES = original.UPDATES
WHOLE_FIELDS = ('dl_allrep/active_count', 'dl_allrep/excess_sum',
    'dl_allrep/position_count', 'dl_allrep/nonreset_active_count',
    'dl_allrep/nonreset_excess_sum')
RESET_FIELDS = ('dl_reset/active_count', 'dl_reset/excess_sum', 'dl_reset/reset_count')
require = original.require
tree_digest = original.tree_digest


@contextlib.contextmanager
def observation_branch(mode, rssm_class=None):
  """Bwhole has original rep gradients; Amin stops all current rep gradients.

  Both call the original RSSM.loss once and retain its forward outputs.
  This is a NEW compiled graph, not a claim of historical graph identity.
  """
  require(mode in MODES, 'Only Bwhole/Amin engineering paths allowed')
  if rssm_class is None:
    from dreamerv3.rssm import RSSM
    rssm_class = RSSM
  old_loss, old_init = rssm_class.loss, rssm_class.__init__
  require(not any(getattr(old_loss, name, False) for name in
      ('_dl_reset_patch_owned', '_dl_allrep_patch_owned', '_dl_observation_guard_owned')),
      'Nested reset/allrep/observation patch forbidden')
  token = object()

  @functools.wraps(old_init)
  def initialize(self, *args, **kwargs):
    old_init(self, *args, **kwargs)
    self._dl_observation_guard_mode = mode
    self._dl_observation_guard_token = token
    self._dl_observation_guard_loss_identity = id(loss)

  @functools.wraps(old_loss)
  def loss(self, carry, tokens, acts, reset, training, rep_probe=False):
    import jax
    import jax.numpy as jnp
    result = old_loss(self, carry, tokens, acts, reset, training, rep_probe)
    carry, entries, losses, features, metrics, probe = result
    if training:
      rep = losses['rep']
      require(rep.shape == reset.shape and reset.dtype == jnp.bool_,
          'Original rep/reset BT contract differs')
      require(self.free_nats == 1, 'Original free-nats1 contract differs')
      require(not set(metrics).intersection((*RESET_FIELDS, *WHOLE_FIELDS)),
          'Original loss already contains pressure-observation metrics')
      if mode == 'Bwhole':
        # Same expressions/order as the adopted full-window A wrapper.
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
      else:
        # Same three observation expressions as the historical B wrapper.
        metrics = {**metrics,
            'dl_reset/active_count': jnp.sum(reset & (rep > 1)),
            'dl_reset/excess_sum': jnp.sum(jnp.where(reset, jnp.maximum(rep - 1, 0), 0)),
            'dl_reset/reset_count': jnp.sum(reset)}
        losses = {**losses, 'rep': jax.lax.stop_gradient(rep)}
    return carry, entries, losses, features, metrics, probe

  loss._dl_observation_guard_owned = True
  loss._dl_observation_guard_mode = mode
  loss._dl_observation_guard_token = token
  rssm_class.__init__, rssm_class.loss = initialize, loss
  try:
    yield
  finally:
    rssm_class.loss, rssm_class.__init__ = old_loss, old_init


def _assert_agent(agent, mode):
  require(mode in MODES and
      (original.UPDATES, original.BATCH_SIZE, original.LOSS_LENGTH, original.CONTEXT) ==
      (128, 16, 64, 1), 'Frozen original layout/update count differs')
  cfg, model = agent.config, agent.model
  require((cfg.batch_size, cfg.batch_length, cfg.replay_context) == (16, 64, 1),
      'Require original B16/T64/context1')
  require(not agent.jaxcfg.enable_policy and model.config.rep_probe.mode == 'off' and
      model.config.dt_latch.mode == 'off', 'Existing policy/weighting/logging overlays must be off')
  require(model.dyn.free_nats == 1 and model.scales['rep'] == .1 and
      model.scales['dyn'] == 1 and not model.config.ac_grads and
      model.config.reward_grad and model.config.repval_grad,
      'Original loss/free-nats/gradient contract differs')
  loss = type(model.dyn).loss
  require(getattr(loss, '_dl_observation_guard_owned', False) is True and
      getattr(loss, '_dl_observation_guard_mode', None) == mode and
      not getattr(loss, '_dl_reset_patch_owned', False) and
      not getattr(loss, '_dl_allrep_patch_owned', False), 'Exact engineering wrapper required')
  require(getattr(model.dyn, '_dl_observation_guard_mode', None) == mode,
      'Engineering wrapper must precede Agent construction')
  require(getattr(model.dyn, '_dl_observation_guard_token', None) is
      getattr(loss, '_dl_observation_guard_token', None) and
      getattr(model.dyn, '_dl_observation_guard_loss_identity', None) == id(loss),
      'Agent must belong to this exact construction-time wrapper')


def json_projection(value):
  """Historical producer jsonify semantics; this projection loses dtype."""
  if isinstance(value, dict): return {str(k): json_projection(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)): return [json_projection(v) for v in value]
  if hasattr(value, 'tolist'): return json_projection(value.tolist())
  return value


def json_difference(actual, expected, path='$'):
  """First exact JSON difference, including float signed zero; no tolerance."""
  actual, expected = json_projection(actual), json_projection(expected)
  tree_digest(actual); tree_digest(expected)  # Nonfinite references are invalid.
  def visit(a, b, location):
    if type(a) is not type(b):
      return dict(path=location, kind='json_type', actual=type(a).__name__, expected=type(b).__name__)
    if isinstance(a, dict):
      if set(a) != set(b):
        return dict(path=location, kind='keys', actual=sorted(a), expected=sorted(b))
      for key in sorted(a):
        found = visit(a[key], b[key], location+'/'+key)
        if found: return found
    elif isinstance(a, list):
      if len(a) != len(b): return dict(path=location, kind='length', actual=len(a), expected=len(b))
      for index, (x, y) in enumerate(zip(a, b)):
        found = visit(x, y, location+'/'+str(index))
        if found: return found
    elif type(a) is float:
      if struct.pack('>d', a) != struct.pack('>d', b):
        return dict(path=location, kind='json_float_bytes', actual=a.hex(), expected=b.hex())
    elif a != b:
      return dict(path=location, kind='value', actual=a, expected=b)
    return None
  return visit(actual, expected, path)


def typed_difference(actual, expected, path='$'):
  """First difference where BOTH numeric trees actually retain their types."""
  import numpy as np
  tree_digest(actual); tree_digest(expected)
  if isinstance(actual, dict) or isinstance(expected, dict):
    if not isinstance(actual, dict) or not isinstance(expected, dict):
      return dict(path=path, kind='container_type')
    if set(actual) != set(expected): return dict(path=path, kind='keys')
    for key in sorted(actual):
      found = typed_difference(actual[key], expected[key], path+'/'+key)
      if found: return found
    return None
  if isinstance(actual, (list, tuple)) or isinstance(expected, (list, tuple)):
    if type(actual) is not type(expected): return dict(path=path, kind='container_type')
    if len(actual) != len(expected): return dict(path=path, kind='length')
    for i, (a, b) in enumerate(zip(actual, expected)):
      found = typed_difference(a, b, path+'/'+str(i))
      if found: return found
    return None
  if (actual is None or type(actual) in (bool, int, float, str) or
      expected is None or type(expected) in (bool, int, float, str)):
    return json_difference(actual, expected, path)
  a, b = np.asarray(actual), np.asarray(expected)
  if a.dtype != b.dtype: return dict(path=path, kind='dtype', actual=str(a.dtype), expected=str(b.dtype))
  if a.shape != b.shape: return dict(path=path, kind='shape', actual=list(a.shape), expected=list(b.shape))
  if a.tobytes(order='C') != b.tobytes(order='C'):
    return dict(path=path, kind='array_bytes', actual_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
        expected_sha256=hashlib.sha256(b.tobytes()).hexdigest())
  return None


def _pin(value):
  require(type(value) is str and len(value) == 64 and
      all(c in '0123456789abcdef' for c in value), 'Valid historical typed SHA256 required')
  return value


def _host(tree):
  import numpy as np
  if isinstance(tree, dict): return {k: _host(v) for k, v in tree.items()}
  if isinstance(tree, tuple): return tuple(_host(v) for v in tree)
  if isinstance(tree, list): return [_host(v) for v in tree]
  if tree is None or type(tree) in (bool, int, float, str): return tree
  if type(tree).__module__.startswith(('jax', 'jaxlib')):
    import jax
    with jax._src.config.explicit_device_get_scope(): return np.asarray(jax.device_get(tree)).copy()
  return np.asarray(tree).copy()


def leaf_manifest(tree, path='$'):
  """Small typed leaf receipts; never retain another full tensor payload."""
  import numpy as np
  if isinstance(tree, dict):
    return [row for key in sorted(tree) for row in leaf_manifest(tree[key], path+'/'+key)]
  if isinstance(tree, (list, tuple)):
    return [row for index, value in enumerate(tree) for row in leaf_manifest(value, path+'/'+str(index))]
  tree_digest(tree)
  if tree is None or type(tree) in (bool, int, float, str):
    return [dict(path=path, dtype='python:'+type(tree).__name__, shape=[], typed_digest=tree_digest(tree))]
  value = np.asarray(tree)
  return [dict(path=path, dtype=str(value.dtype), dtype_str=value.dtype.str,
      shape=list(value.shape), typed_digest=tree_digest(value))]


def _returned_contract(agent, carry, outs, raw):
  """Validate actual official train outputs outside the compiled graph."""
  import numpy as np
  tree_digest(carry); tree_digest(outs)
  if type(agent).__module__ != 'embodied.jax.agent': return  # Sequencing doubles only.
  require(type(carry) is tuple and len(carry) == 4 and type(carry[3]) is dict and
      set(carry[3]) == {'action'} and carry[0] == {} and carry[2] == {} and
      type(carry[1]) is dict and set(carry[1]) == {'deter', 'stoch'},
      'Original train returned carry structure differs')
  for name in ('deter', 'stoch'):
    value = carry[1][name]
    require(str(value.dtype) == 'bfloat16' and value.shape == (16, *raw['dyn/'+name].shape[2:]),
        'Actual returned BF16 RSSM carry dtype/shape differs: '+name)
  stoch = carry[1]['stoch'].astype(np.float32)
  require(np.isin(stoch, (0, 1)).all() and (stoch.sum(-1) == 1).all(),
      'Actual returned posterior sample must be categorical onehot')
  previous = carry[3]['action']
  expected = raw['action'][:, -1]
  require(previous.dtype == expected.dtype == np.float32 and previous.shape == (16, 12) and
      previous.tobytes() == expected.tobytes(), 'Actual final carry previous raw action differs')
  require(set(outs) == {'replay'} and type(outs['replay']) is dict,
      'Original context1 replay output required')
  fields = (set(raw) - set(original.FIELDS) - {'consec', 'stepid'}) | {'stepid'}
  require(set(outs['replay']) == fields, 'Original current replay output keys differ')
  for name, value in outs['replay'].items():
    require(value.dtype == raw[name].dtype and value.shape == (16, 64, *raw[name].shape[2:]),
        'Actual returned replay output dtype/shape differs: '+name)
  require(outs['replay']['stepid'].tobytes() == raw['stepid'][:, 1:].tobytes(),
      'Actual returned replay step identities differ')


def _restore_a_anchor(common, reference_metrics, reference_digest):
  """Recover types only if the ENTIRE old full metric digest authenticates them.

  Current common schema is a hypothesis, not historical evidence. Unknown
  array shape/dtype, rounding/overflow, missing keys or hash mismatch leave
  common typed authentication unavailable. Never default to a guessed dtype.
  """
  import numpy as np
  old = json_projection(reference_metrics)
  require(type(old) is dict, 'Historical metrics dictionary required')
  required = set(common) | set(WHOLE_FIELDS)
  if set(old) != required: return dict(available=False, reason='full_metric_keys_differ')
  restored = {}
  for name in sorted(required):
    if name in common:
      actual = common[name]
      if not isinstance(actual, np.ndarray):
        return dict(available=False, reason='common_schema_is_not_actual_numpy_array', metric=name)
      dtype, shape = actual.dtype, actual.shape
    else:
      dtype = np.dtype(np.float32 if 'excess_sum' in name else np.int32)
      shape = ()
    try:
      with np.errstate(over='raise', invalid='raise'):
        value = np.asarray(old[name], dtype=dtype)
      if value.shape != shape or json_difference(value, old[name]) is not None:
        return dict(available=False, reason='schema_roundtrip_not_exact', metric=name)
      tree_digest(value)
    except (ValueError, TypeError, OverflowError, FloatingPointError):
      return dict(available=False, reason='schema_roundtrip_failed', metric=name)
    restored[name] = value
  digest = tree_digest(restored)
  if digest != _pin(reference_digest):
    return dict(available=False, reason='reconstructed_full_metric_anchor_hash_differs',
        reconstructed_full_metrics_digest=digest, expected_full_metrics_digest=reference_digest)
  anchored_common = {k: v for k, v in restored.items() if k not in WHOLE_FIELDS}
  return dict(available=True, reason='entire_historical_typed_metric_hash_verified',
      full_metrics_digest=digest, common_metrics_digest=tree_digest(anchored_common),
      difference=typed_difference(common, anchored_common))


def _pressure(metrics, whole):
  import numpy as np
  def scalar(name, dtype):
    value = np.asarray(metrics.get(name))
    require(value.dtype == dtype and value.shape == () and np.isfinite(value) and value >= 0,
        'Invalid exact typed pressure metric: '+name)
    return float(value) if dtype == np.float32 else int(value)
  reset = dict(active_count=scalar(RESET_FIELDS[0], np.int32),
      excess_sum=scalar(RESET_FIELDS[1], np.float32), reset_count=scalar(RESET_FIELDS[2], np.int32))
  require(reset['reset_count'] == 16 and reset['active_count'] <= 16 and
      (reset['active_count'] > 0) == (reset['excess_sum'] > 0), 'Invalid original reset pressure')
  if not whole: return dict(reset=reset, whole=None)
  full = {name.split('/')[-1]: scalar(name, np.float32 if 'excess_sum' in name else np.int32)
      for name in WHOLE_FIELDS}
  require(full['position_count'] == 1024 and full['active_count'] <= 1024 and
      full['nonreset_active_count'] <= 1008 and
      full['active_count'] == reset['active_count'] + full['nonreset_active_count'] and
      (full['active_count'] > 0) == (full['excess_sum'] > 0) and
      (full['nonreset_active_count'] > 0) == (full['nonreset_excess_sum'] > 0),
      'Invalid original whole-window pressure')
  return dict(reset=reset, whole=full)


def run_guard(agent, checkpoint, calibration, source_hashes, frozen_context_entries,
              mode, *, reference_b, reference_a, check=lambda: None):
  """New128-update engineering path; differences never shorten the budget.

  Reference arguments are externally authenticated learning dictionaries.
  Old receipts lack per-step fullstate/carry/outs anchors. New hashes are NEW
  evidence only; endpoint complete-state hashes are the historical anchors.
  """
  import numpy as np
  check(); _assert_agent(agent, mode)
  for reference, group, version in ((reference_b, 'B', original.VERSION),
      (reference_a, 'A', 'DL-reset-allrep-learning-r1')):
    require(type(reference) is dict and reference.get('version') == version and
        reference.get('group') == group and reference.get('updates') == UPDATES and
        type(reference.get('receipts')) is list and len(reference['receipts']) == UPDATES,
        'Complete fixed128 historical B/A learning references required')
    _pin(reference['final_state_digest']); _pin(reference['final_params_digest'])
    for row in reference['receipts']: _pin(row['metrics_digest'])
  reference = reference_b if mode == 'Bwhole' else reference_a
  references_digest = tree_digest(json_projection([reference_b, reference_a]))
  require(type(checkpoint) is dict and set(checkpoint) == {'params', 'counters'},
      'Complete original checkpoint required')
  for name in ('updates', 'batches', 'actions', 'eval_actions', 'eval_seed'):
    require(type(checkpoint['counters'].get(name)) is int and checkpoint['counters'][name] >= 0,
        'Invalid original checkpoint counter: '+name)
  source_digest = tree_digest(checkpoint)
  agent.load(checkpoint)
  require(tree_digest(checkpoint) == source_digest, 'Caller checkpoint mutated by load')
  initial = agent.save()
  require(tree_digest(initial['params']) == tree_digest(checkpoint['params']),
      'Actual standard load changed original complete parameter state')
  expected = {**checkpoint['counters'], 'batches': checkpoint['counters']['updates']}
  require(initial['counters'] == expected, 'Actual standard load counter semantics differ')
  initial_step = original.optimizer_step(initial['params'])
  calibration_digest = tree_digest({str(k): v for k, v in calibration.items()})
  raw, binding = original.offline_batch(agent, calibration, source_hashes, frozen_context_entries)
  input_values = dict(source_checkpoint_digest=source_digest,
      initial_state_digest=tree_digest(initial), initial_params_digest=tree_digest(initial['params']),
      calibration_arrays_digest=calibration_digest, binding=binding, restored_counters=expected,
      saved_source_counters=checkpoint['counters'], optimizer_counter=dict(
          key=original.OPTIMIZER_STEP_KEY, dtype='int32', shape=[], initial=initial_step,
          final=initial_step+UPDATES, updates=UPDATES))
  input_differences = []
  for group, ref in (('B', reference_b), ('A', reference_a)):
    for name, value in input_values.items():
      difference = json_difference(value, ref.get(name), '$/reference_'+group+'/'+name)
      if difference: input_differences.append(difference)
  require(not input_differences, 'Authenticated original input contract differs: '+str(
      input_differences[0] if input_differences else ''))
  require(not any(getattr(agent, name, None) is not None for name in
      ('pending_outs', 'pending_mets', 'pending_sync')), 'Fresh Agent with no delayed/pending outputs required')
  rows, differences, typed_available = [], list(input_differences), []
  final_carry = final_outs = None
  for index in range(UPDATES):
    check(); _assert_agent(agent, mode)
    before = original._counters(agent)
    require(before == {**expected, 'updates': expected['updates']+index,
        'batches': expected['batches']+index}, 'Invalid pre-update normal counters')
    carry = agent.init_train(original.BATCH_SIZE)
    prepared = agent.prepare_batch({k: v.copy() for k, v in raw.items()})
    require('seed' in prepared, 'Actual prepare_batch seed missing')
    seed = _host(prepared['seed'])
    canonical = _host(agent._seeds(before['batches'], agent.train_mirrored))
    require(seed.dtype == np.uint32 and seed.shape == (2,) and
        seed.tobytes() == canonical.tobytes(), 'Actual normal train root differs')
    key_differences = []
    for group, ref in (('B', reference_b), ('A', reference_a)):
      target = ref['receipts'][index]
      actual_key = dict(update=index+1, seed_counter=before['batches'], train_seed=seed.astype(int).tolist())
      expected_key = {name: target.get(name) for name in actual_key}
      found = json_difference(actual_key, expected_key, f'$/update/{index+1}/reference_{group}/key')
      require(found is None, 'Authenticated original normal key contract differs: '+str(found))
    returned = agent.train(carry, prepared)
    require(isinstance(returned, tuple) and len(returned) == 3,
        'Actual Agent.train carry/delayed-outs/delayed-metrics return required')
    outs, metrics = agent.take_train_result()
    require(type(metrics) is dict and metrics and isinstance(outs, dict), 'Missing completed train result')
    metrics = _host(metrics)
    metrics_digest = tree_digest(metrics)
    if mode == 'Bwhole':
      require(set(WHOLE_FIELDS) <= set(metrics), 'Bwhole must contain exactly the five added pressure fields')
      common = {k: v for k, v in metrics.items() if k not in WHOLE_FIELDS}
    else:
      require(not set(metrics).intersection(WHOLE_FIELDS), 'Amin must not contain whole-window fields')
      common = dict(metrics)
    pressure = _pressure(metrics, mode == 'Bwhole')
    step_value = np.asarray(metrics.get('opt/updates'))
    require(step_value.dtype == np.int32 and step_value.shape == () and
        int(step_value) == initial_step+index+1, 'Actual optimizer metric contract differs')
    after = original._counters(agent)
    require(after == {**expected, 'updates': expected['updates']+index+1,
        'batches': expected['batches']+index+1}, 'Invalid post-update normal counters')
    target = reference['receipts'][index]
    counters_difference = json_difference(after, target.get('counters'), f'$/update/{index+1}/counters')
    require(counters_difference is None,
        'Authenticated original post-update counter contract differs: '+str(counters_difference))
    reference_common = {k: v for k, v in target['metrics'].items() if k not in WHOLE_FIELDS}
    json_diff = json_difference(common, reference_common, f'$/update/{index+1}/common_metrics')
    common_digest = tree_digest(common)
    if mode == 'Bwhole':
      typed = dict(available=True, common_metrics_digest=target['metrics_digest'],
          reason='original_B_typed_metrics_digest_is_common_anchor')
    else:
      typed = _restore_a_anchor(common, target['metrics'], target['metrics_digest'])
    typed_available.append(typed['available'])
    typed_equal = common_digest == typed['common_metrics_digest'] if typed['available'] else None
    metric_equal = typed_equal if mode == 'Bwhole' else json_diff is None
    if not metric_equal:
      differences.append(json_diff or dict(path=f'$/update/{index+1}/common_metrics',
          kind='typed_digest', actual=common_digest, expected=typed['common_metrics_digest']))
    if mode == 'Amin' and typed_equal is False:
      differences.append(dict(path=f'$/update/{index+1}/common_metrics', kind='authenticated_typed_difference',
          actual=common_digest, expected=typed['common_metrics_digest']))
    final_carry, final_outs = _host(returned[0]), _host(outs)
    _returned_contract(agent, final_carry, final_outs, raw)
    state = agent.save()
    require(state['counters'] == after and original.optimizer_step(state['params']) == int(step_value),
        'Actual saved complete state/counter contract differs')
    rows.append(dict(update=index+1, seed_counter=before['batches'], train_seed=seed.astype(int).tolist(),
        counters=after, optimizer_step=int(step_value), metrics=metrics, metrics_digest=metrics_digest,
        common_metrics=common, common_metrics_digest=common_digest,
        common_metrics_json_exact=json_diff is None, common_metrics_typed_exact=typed_equal,
        common_metrics_equal=bool(metric_equal), common_metrics_first_difference=json_diff,
        reference_common_typed_anchor=typed, key_differences=key_differences,
        full_state_digest=tree_digest(state), full_params_digest=tree_digest(state['params']),
        carry_digest=tree_digest(final_carry), outs_digest=tree_digest(final_outs), pressure=pressure,
        carry_leaf_manifest=leaf_manifest(final_carry), outs_leaf_manifest=leaf_manifest(final_outs),
        historical_per_update_fullstate_carry_outs_anchor_available=False))
    require(tree_digest(raw) == binding['batch_digest'], 'Original common raw batch mutated')
    check()
  final = agent.save()
  require(tree_digest(checkpoint) == source_digest, 'Caller original checkpoint mutated')
  require(tree_digest({str(k): v for k, v in calibration.items()}) == calibration_digest,
      'Caller calibration arrays mutated')
  require(tree_digest({str(k): v for k, v in frozen_context_entries.items()}) ==
      binding['context_entries_digest'], 'Caller frozen context entries mutated')
  require(tree_digest(json_projection([reference_b, reference_a])) == references_digest,
      'Caller historical learning references mutated')
  final_step = original.optimizer_step(final['params'])
  require(final['counters'] == {**expected, 'updates': expected['updates']+UPDATES,
      'batches': expected['batches']+UPDATES} and final_step == initial_step+UPDATES,
      'Complete fixed128 final optimizer/counter contract differs')
  endpoint = dict(actual_full_state_digest=tree_digest(final),
      expected_full_state_digest=reference['final_state_digest'],
      actual_params_digest=tree_digest(final['params']), expected_params_digest=reference['final_params_digest'])
  endpoint['full_state_typed_exact'] = endpoint['actual_full_state_digest'] == endpoint['expected_full_state_digest']
  endpoint['params_typed_exact'] = endpoint['actual_params_digest'] == endpoint['expected_params_digest']
  if not endpoint['full_state_typed_exact']:
    differences.append(dict(path='$/final_state', kind='historical_typed_endpoint_digest',
        actual=endpoint['actual_full_state_digest'], expected=endpoint['expected_full_state_digest']))
  if not endpoint['params_typed_exact']:
    differences.append(dict(path='$/final_params', kind='historical_typed_endpoint_digest',
        actual=endpoint['actual_params_digest'], expected=endpoint['expected_params_digest']))
  comparison = dict(typed_available=all(typed_available), typed_available_updates=sum(typed_available),
      typed_reference_available=all(typed_available),
      reference_original_common_typed_digest_available=(mode == 'Bwhole'),
      A_typed_schema_reconstruction_only_after_full_historical_hash_verified=(mode == 'Amin'),
      all_updates_equal=all(r['common_metrics_equal'] for r in rows),
      all_json_numeric_exact=all(r['common_metrics_json_exact'] for r in rows),
      all_available_typed_exact=all(r['common_metrics_typed_exact'] is not False for r in rows),
      per_update_equal=[r['common_metrics_equal'] for r in rows],
      first_difference=next((d for d in differences if '/common_metrics' in d['path']), None),
      fallback_when_A_typed_unavailable='exact JSON numeric only; never claimed typed-byte equality')
  comparison['passed'] = comparison['all_updates_equal'] and comparison['all_available_typed_exact']
  comparison['first_diff'] = comparison['first_difference']
  comparison['rows'] = [dict(update=r['update'], equal=r['common_metrics_equal'],
      json_numeric_exact=r['common_metrics_json_exact'], typed_exact=r['common_metrics_typed_exact'],
      typed_reference_available=r['reference_common_typed_anchor']['available'],
      typed_reference_reason=r['reference_common_typed_anchor']['reason'],
      first_difference=r['common_metrics_first_difference']) for r in rows]
  check()
  equivalent = not differences and endpoint['full_state_typed_exact'] and endpoint['params_typed_exact']
  return dict(version=VERSION, candidate_not_adopted=True, mode=mode, updates=UPDATES,
      layout_version=original.VERSION, **input_values,
      final_state_digest=tree_digest(final), final_params_digest=tree_digest(final['params']),
      final_checkpoint=final,
      final_carry_digest=tree_digest(final_carry), final_outs_digest=tree_digest(final_outs),
      final_carry_leaf_manifest=leaf_manifest(final_carry), final_outs_leaf_manifest=leaf_manifest(final_outs),
      receipts=rows, common_metrics_comparison=comparison, endpoint_comparison=endpoint,
      equivalence_verdict='equivalent_within_declared_evidence' if equivalent else 'not_equivalent',
      equivalent=equivalent, first_difference=differences[0] if differences else None,
      differences=differences, input_reference_values_matched=not input_differences,
      all_fixed128_updates_completed=True, rep_gradient_stopped=(mode == 'Amin'),
      new_graph_not_historical_compiled_graph=True,
      historical_per_update_fullstate_carry_outs_equivalence_claimed=False,
      source_authentication_performed=False, native_actor_evaluation_performed=False,
      simulator_calls=0, real_size50m_cuda_equivalence_established=False,
      method_effectiveness_established=False,
      scope='new path endpoint and declared common-metric comparison only; caller authenticates real run')
