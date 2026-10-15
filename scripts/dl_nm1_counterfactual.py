"""DL-NM1 calibration primitives; no source loader, runner or GPU authorization.

Reference gradients have their own complete loss/gradient JIT. They are not
claimed to be byte-identical to the fused production optimizer's intermediates.
Production displacements use the original Agent._train core on disposable copies
of ALL state leaves; original params, Python counters and replay are untouched.
"""
import numpy as np

from scripts.dl_reset_learning import require, tree_digest

VERSION = 'DL-NM1-counterfactual-r1'
NEURAL = ('enc/', 'dyn/', 'dec/', 'rew/', 'con/', 'pol/', 'val/')


def host(tree):
  import jax
  with jax._src.config.explicit_device_get_scope():
    return jax.tree.map(lambda x: np.asarray(x).copy(), jax.device_get(tree))


def prepare_readonly(agent, raw, counter):
  """Same placement/seed as prepare_batch, without incrementing n_batches."""
  from embodied.jax import internal
  require(type(counter) is int and counter >= 0, 'Explicit train counter required')
  require(set(raw) == set(agent.spaces), 'Full raw replay batch schema required')
  for key,value in raw.items():
    require(type(value) is np.ndarray and np.isfinite(value).all(),
        'Finite raw numpy replay field required: '+key)
  placed=internal.device_put(raw,agent.train_sharded)
  return dict(placed,seed=agent._seeds(counter,agent.train_mirrored))


class Counterfactual:
  """Finite one-step probes from one immutable complete loaded Agent state."""
  def __init__(self,agent):
    import jax
    import ninjax as nj
    require(len(agent.train_devices) == 1 and not agent.jaxcfg.use_shardmap,
        'One-device unsharded-map calibration required')
    require(not agent.jaxcfg.enable_policy, 'Offline probe must disable policy synchronization')
    require(not agent.model.opt.scaling, 'Only original F32/BF16 optimizer without FP16 scaling')
    require(all(getattr(agent,k) is None for k in ('pending_outs','pending_mets','pending_sync')),
        'No pending public train outputs/synchronization allowed')
    self.agent=agent
    self._model=agent.model
    self._configuration=tree_digest(dict(agent.model.config.flat))
    self._active=(agent.model.dt_latch_active,agent.model.rep_probe_active)
    self._params=agent.params
    self._identities={k:id(v) for k,v in agent.params.items()}
    self._counters=agent.save()['counters']
    self._digest=tree_digest(agent.save())
    self.reference_accessed=()
    self.reference_modified=()
    self._context=jax.jit(agent.model._apply_replay_context)
    pure=nj.pure(agent.model.log_train_collect)
    def reference(params,carry,obs,prevact,seed):
      scratch,bundle,accessed,modified,created=pure(params,carry,obs,prevact,
          seed=seed,create=False,modify=True,track=True)
      forbidden={k for k in modified if not k.startswith(('retnorm/','valnorm/','advnorm/'))}
      require(not forbidden and not created, 'Reference attempted non-normalizer writes/state creation')
      self.reference_accessed=tuple(sorted(accessed))
      self.reference_modified=tuple(sorted(modified))
      return scratch,bundle
    self._reference=jax.jit(reference)

  def fence(self,full=False):
    agent=self.agent
    require(agent.model is self._model and
        tree_digest(dict(agent.model.config.flat)) == self._configuration and
        (agent.model.dt_latch_active,agent.model.rep_probe_active) == self._active,
        'Model/configuration changed after compilation; use separate configured Agents')
    require(agent.params is self._params and {k:id(v) for k,v in agent.params.items()} == self._identities,
        'Original Agent parameter identity changed')
    require(all(not v.is_deleted() for v in agent.params.values()), 'Original Agent buffer donated/deleted')
    current=dict(updates=int(agent.n_updates),batches=int(agent.n_batches),
        actions=int(agent.n_actions),eval_actions=int(agent.n_eval_actions),eval_seed=int(agent.eval_seed))
    require(current == self._counters and all(getattr(agent,k) is None for k in
        ('pending_outs','pending_mets','pending_sync')), 'Original Python counters/pending state changed')
    if full:
      require(tree_digest(agent.save()) == self._digest, 'Original complete Agent state changed')

  def reference(self,carry,prepared):
    import jax
    self.fence()
    before=tree_digest(host((carry,prepared)))
    data={k:v for k,v in prepared.items() if k != 'seed'}
    ca,obs,prevact,_=self._context(carry,data)
    scratch,bundle=self._reference(self.agent.params,ca,obs,prevact,prepared['seed'])
    jax.block_until_ready((scratch,bundle))
    loss,params,grads,aux=bundle
    result=dict(loss=host(loss),params=host(params),grads=host(grads),aux=aux,
        gradient_scope='separate-complete-reference-JIT-not-production-intermediate')
    tree_digest((result['loss'],result['params'],result['grads']))
    require(tree_digest(host((carry,prepared))) == before, 'Reference changed input/carry')
    self.fence()
    return result

  def production(self,carry,prepared):
    """Unmodified production train JIT, no extra outputs inside its graph."""
    import jax
    self.fence()
    before=tree_digest(host((carry,prepared)))
    # The production core donates its non-policy dictionary. No leaf, including
    # the allocated policy subset, may refer to the original Agent state.
    scratch={k:v.copy() for k,v in self.agent.params.items()}
    allo={k:v for k,v in scratch.items() if k in self.agent.policy_keys}
    dona={k:v for k,v in scratch.items() if k not in self.agent.policy_keys}
    data={k:v for k,v in prepared.items() if k != 'seed'}
    params,next_carry,outs,metrics=self.agent._train(dona,allo,prepared['seed'],carry,data)
    jax.block_until_ready((params,next_carry,outs,metrics))
    result=dict(params=host(params),carry=host(next_carry),outs=host(outs),metrics=host(metrics),
        displacement_scope='original-Agent._train-core-disposable-complete-state')
    # Some predeclared unavailable diagnostic ratios can be NaN; callers must
    # classify those explicitly. Parameters/carry/replay outputs never may.
    tree_digest((result['params'],result['carry'],result['outs']))
    require(tree_digest(host((carry,prepared))) == before, 'Production probe changed input/carry')
    self.fence()
    return result


def neural_difference(reference,alternative):
  """Host F64 summaries of actual F32 neural tensors; signed zero is separate."""
  require(set(reference) == set(alternative), 'Tensor key sets differ')
  keys=sorted(k for k in reference if k.startswith(NEURAL))
  require(keys, 'Neural tensors required')
  result={}
  groups={'all':keys,'encoder':[k for k in keys if k.startswith('enc/')],
      'posterior':[k for k in keys if k.startswith('dyn/obs')]}
  for name,selected in groups.items():
    count=changed=leaves=zeros=0;delta2=left2=right2=dot=maximum=ulp=nonzero_ulp=0.;byte_identical=True
    for key in selected:
      a,b=reference[key],alternative[key]
      require(type(a) is type(b) is np.ndarray and a.dtype == b.dtype == np.float32 and
          a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
          'Matched finite F32 neural tensors required: '+key)
      unequal=(a != b);n=int(unequal.sum());count+=a.size;changed+=n;leaves+=int(n > 0)
      zeros+=int((a == 0).sum())
      byte_identical &= a.tobytes() == b.tobytes()
      x,y=a.astype(np.float64),b.astype(np.float64);d=y-x
      delta2+=float(np.sum(d*d));left2+=float(np.sum(x*x));right2+=float(np.sum(y*y));dot+=float(np.sum(x*y))
      maximum=max(maximum,float(np.abs(d).max(initial=0)))
      # Exact F32 binade spacing, including zero/subnormals and max finite;
      # np.spacing(max_float32) overflows even though the reference is finite.
      exponent=((a.view(np.uint32) >> 23) & 255).astype(np.int32)
      spacing=np.exp2(np.maximum(exponent-150,-149).astype(np.float64))
      ulp=max(ulp,float(np.max(np.abs(d)/spacing,initial=0)))
      nonzero_ulp=max(nonzero_ulp,float(np.max((np.abs(d)/spacing)[a != 0],initial=0)))
    result[name]=dict(tensors=len(selected),elements=count,changed_elements=changed,
        changed_tensors=leaves,byte_identical=bool(byte_identical),difference_l2=float(np.sqrt(delta2)),
        reference_l2=float(np.sqrt(left2)),alternative_l2=float(np.sqrt(right2)),
        tensor_cosine=float(dot/np.sqrt(left2*right2)) if left2 > 0 and right2 > 0 else None,
        max_abs_difference=maximum,max_reference_ulp_difference=ulp,
        reference_zero_elements=zeros,max_nonzero_reference_ulp_difference=nonzero_ulp)
  return result


def neural_displacements(initial,reference,alternative):
  """Actual update-vector direction, using F64 after-minus-common-before.

  This differs from the cosine between the usually much larger endpoint
  parameters. No gradient/Adam decomposition or optimizer state is inferred.
  """
  require(set(initial) == set(reference) == set(alternative), 'Update tensor keys differ')
  keys=sorted(k for k in initial if k.startswith(NEURAL))
  require(keys, 'Neural update tensors required')
  groups={'all':keys,'encoder':[k for k in keys if k.startswith('enc/')],
      'posterior':[k for k in keys if k.startswith('dyn/obs')]}
  result={}
  for name,selected in groups.items():
    left2=right2=dot=delta2=0.;left_changed=right_changed=different=0
    for key in selected:
      values=[tree[key] for tree in (initial,reference,alternative)]
      require(all(type(v) is np.ndarray and v.dtype == np.float32 and
          v.shape == values[0].shape and np.isfinite(v).all() for v in values),
          'Matched finite F32 update tensors required: '+key)
      before,left,right=[v.astype(np.float64) for v in values]
      a,b=left-before,right-before
      left2+=float(np.sum(a*a));right2+=float(np.sum(b*b));dot+=float(np.sum(a*b))
      delta2+=float(np.sum((b-a)**2))
      left_changed+=int(np.count_nonzero(a));right_changed+=int(np.count_nonzero(b))
      different+=int(np.count_nonzero(b-a))
    result[name]=dict(reference_update_l2=float(np.sqrt(left2)),
        alternative_update_l2=float(np.sqrt(right2)),update_difference_l2=float(np.sqrt(delta2)),
        update_cosine=float(dot/np.sqrt(left2*right2)) if left2 > 0 and right2 > 0 else None,
        reference_updated_elements=left_changed,alternative_updated_elements=right_changed,
        different_update_elements=different)
  return result
