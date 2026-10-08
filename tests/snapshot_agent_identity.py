"""One-process full-agent snapshot for baseline and overlay comparisons.

Run in separate Python processes so JAX and Ninjax initialization are identical.
The baseline run uses this script with PYTHONPATH pointing at commit e935ff7.
"""

import argparse
import json
import resource
import time

import elements
import embodied.jax.internal as internal
import jax
import ninjax as nj
import numpy as np
import ruamel.yaml as yaml

from dreamerv3.agent import Agent


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', choices=(
      'baseline', 'off', 'logging', 'dt', 'constant', 'shuffle'), required=True)
  parser.add_argument('--output', required=True)
  parser.add_argument('--replay_context', type=int, default=0)
  parser.add_argument('--free_nats', type=float, default=1.0)
  parser.add_argument('--imag_last', type=int, default=2)
  parser.add_argument('--alpha', type=float, default=0.7)
  parser.add_argument('--c', type=float, default=0.6)
  parser.add_argument('--seed', type=int, default=7)
  parser.add_argument('--bench_updates', type=int, default=0)
  parser.add_argument('--platform', choices=('cpu', 'cuda'), default='cpu')
  parser.add_argument('--dtype', choices=('float32', 'bfloat16'), default='float32')
  args = parser.parse_args()
  with open('dreamerv3/configs.yaml', encoding='utf-8') as file:
    configs = yaml.YAML(typ='safe').load(file)
  config = elements.Config(configs['defaults']).update(configs['debug'])
  overrides = {
      'batch_size': 2, 'batch_length': 3,
      'replay_context': args.replay_context,
      'jax.platform': args.platform, 'jax.compute_dtype': args.dtype,
      'jax.precompile': False, 'jax.profiler': False,
      'jax.enable_policy': False,
      'agent.imag_length': 2, 'agent.imag_last': args.imag_last,
      'agent.dyn.rssm.free_nats': args.free_nats,
  }
  if args.mode != 'baseline':
    overrides.update({
        'agent.rep_probe.mode': args.mode,
        'agent.rep_probe.alpha': (
            args.alpha if args.mode in ('logging', 'dt', 'constant', 'shuffle') else -1.0),
        'agent.rep_probe.c': args.c if args.mode == 'constant' else -1.0,
    })
  config = elements.Config({**config.flat, **overrides})
  obs_space = {
      'vector': elements.Space(np.float32, (5,)),
      'reward': elements.Space(np.float32),
      'is_first': elements.Space(bool),
      'is_last': elements.Space(bool),
      'is_terminal': elements.Space(bool),
  }
  act_space = {'action': elements.Space(np.float32, (2,))}
  agent_config = elements.Config(**config.agent, logdir='/tmp/pd-identity',
      seed=args.seed, jax=config.jax, batch_size=config.batch_size,
      batch_length=config.batch_length, replay_context=config.replay_context)
  start = time.perf_counter()
  agent = Agent(obs_space, act_space, agent_config)
  init_seconds = time.perf_counter() - start
  length = 3 + args.replay_context
  data = agent._zeros(agent.spaces, (2, length))
  data['vector'][:] = np.arange(10 * length, dtype=np.float32).reshape(
      2, length, 5) / (10 * length)
  data['reward'][:] = np.linspace(-0.5, 1.0, 2 * length,
      dtype=np.float32).reshape(2, length)
  data['is_first'][:, 0] = True
  data['is_last'][0, -1] = True
  data['is_terminal'][0, -1] = True
  data['action'][:] = 0.2
  data = internal.device_put(data, agent.train_sharded)
  carry = agent.init_train(2)
  loss_carry, obs, prevact, _ = jax.jit(agent.model._apply_replay_context)(carry, data)
  seed = agent._seeds(0, agent.train_mirrored)

  arrays = {}
  def add(prefix, tree):
    with jax._src.config.explicit_device_get_scope():
      for key, value in tree.items():
        if value.dtype == jax.numpy.bfloat16:
          value = value.astype(jax.numpy.float32)
        arrays[prefix + key] = np.asarray(value)
  add('init/', agent.params)
  start = time.perf_counter()
  grad_fn = jax.jit(nj.pure(
      lambda c, o, p: nj.grad(
          agent.model.loss, agent.model.modules, has_aux=True)(c, o, p, True)))
  grad_state, (loss, _, grads, aux) = grad_fn(
      agent.params, loss_carry, obs, prevact, seed=seed)
  grad_seconds = time.perf_counter() - start
  with jax._src.config.explicit_device_get_scope():
    arrays['loss'] = np.asarray(loss)
  add('grad/', grads)
  rep_grad_fn = jax.jit(nj.pure(
      lambda c, o, p: nj.grad(
          lambda c, o, p: agent.model.loss(c, o, p, True)[1][2][
              'losses']['rep'].mean() * agent.model.scales['rep'],
          agent.model.modules)(c, o, p)))
  _, (_, _, rep_grads) = rep_grad_fn(
      agent.params, loss_carry, obs, prevact, seed=seed)
  add('repgrad/', rep_grads)
  add('gradstate/', grad_state)
  add('loss/', aux[2]['losses'])
  add('repfeat/', aux[2]['repfeat'])
  if args.mode in ('logging', 'dt', 'constant', 'shuffle') and not (
      args.mode == 'dt' and args.alpha == 0) and not (
      args.mode == 'constant' and args.c == 1):
    add('diagnostic/', {k: v for k, v in aux[3].items() if k.startswith('dt/')})

  start = time.perf_counter()
  carry, _, update_metrics = agent.train(carry, dict(data, seed=seed))
  # The first call returns delayed metrics; parameters are already updated.
  del update_metrics
  update_seconds = time.perf_counter() - start
  add('updated/', agent.params)
  np.savez_compressed(args.output, **arrays)
  bench_seconds = None
  if args.bench_updates:
    carry, _, _ = agent.train(carry, dict(data, seed=seed))
    jax.block_until_ready(next(iter(agent.params.values())))
    start = time.perf_counter()
    for _ in range(args.bench_updates):
      carry, _, _ = agent.train(carry, dict(data, seed=seed))
    jax.block_until_ready(next(iter(agent.params.values())))
    bench_seconds = time.perf_counter() - start
  print(json.dumps({
      'mode': args.mode, 'jax': jax.__version__, 'backend': jax.default_backend(),
      'compute_dtype': str(config.jax.compute_dtype), 'seed': args.seed,
      'batch': [2, 3], 'replay_context': args.replay_context,
      'free_nats': args.free_nats, 'imag_last': args.imag_last,
      'alpha': args.alpha if args.mode in (
          'logging', 'dt', 'constant', 'shuffle') else None,
      'c': args.c if args.mode == 'constant' else None,
      'param_count': len(agent.params),
      'init_seconds': init_seconds, 'grad_seconds': grad_seconds,
      'update_seconds': update_seconds, 'loss': float(arrays['loss']),
      'bench_updates': args.bench_updates, 'bench_seconds': bench_seconds,
      'maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'diagnostic_keys': sorted(k for k in arrays if k.startswith('diagnostic/')),
  }, indent=2))


if __name__ == '__main__':
  main()
