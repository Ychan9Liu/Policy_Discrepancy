"""Real tiny CPU learning with private replay feedback; never N1M acceptance.

Only the generated fixture's dimensions/counts are adapted. Original losses,
optimizer, normalization, target updates and consec=0/context=1 are retained.
"""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def learning_fixture(output):
  # Keep the optional Windows runtime directory handle alive through all JAX use.
  dll_directory = (os.add_dll_directory(os.environ['DL_DLL_DIRECTORY'])
      if os.name == 'nt' and os.environ.get('DL_DLL_DIRECTORY') else None)
  import elements
  import jax
  import jax.numpy as jnp
  from ruamel.yaml import YAML
  from dreamerv3.agent import Agent
  from scripts import dl_normal_replay as replay
  from scripts import dl_replay_inventory as inventory
  from tests.test_dl_replay_inventory import uuid

  def host(tree):
    with jax._src.config.explicit_device_get_scope():
      return jax.tree.map(lambda x: np.array(x, copy=True), jax.device_get(tree))

  def exact(left, right):
    lhs, ls = jax.tree.flatten(host(left))
    rhs, rs = jax.tree.flatten(host(right))
    assert ls == rs
    for a, b in zip(lhs, rhs):
      assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())

  def changed(left, right):
    assert set(left) == set(right)
    return any(left[k].tobytes() != right[k].tobytes() for k in left)

  def finite(tree):
    leaves = jax.tree.leaves(host(tree))
    assert leaves and all(np.isfinite(x).all() for x in leaves)

  presets = YAML(typ='safe').load((ROOT / 'dreamerv3/configs.yaml').read_text())
  defaults = elements.Config(presets['defaults'])
  config = defaults.update(presets['debug'])
  config = elements.Config({**config.flat, **{
      'batch_size': 2, 'batch_length': 3, 'replay_context': 1,
      'jax.platform': 'cpu', 'jax.compute_dtype': 'float32',
      'jax.train_devices': [0], 'jax.precompile': False,
      'jax.profiler': False, 'jax.enable_policy': False}})
  for key in ('loss_scales', 'opt', 'retnorm', 'valnorm', 'advnorm',
      'slowvalue', 'imag_length', 'imag_last', 'imag_loss', 'repl_loss',
      'reward_grad', 'repval_grad', 'repval_loss', 'ac_grads'):
    assert config.agent[key] == defaults.agent[key], key
  assert config.agent.dyn.rssm.free_nats == defaults.agent.dyn.rssm.free_nats == 1.
  assert config.agent.dt_latch.mode == config.agent.rep_probe.mode == 'off'
  assert config.consec_train == 1 and config.replay_context == 1
  agent = Agent(dict(image=elements.Space(np.uint8, (64, 64, 3)),
      reward=elements.Space(np.float32), is_first=elements.Space(bool),
      is_last=elements.Space(bool), is_terminal=elements.Space(bool)),
      dict(action=elements.Space(np.float32, (12,))), elements.Config(
          **config.agent.flat, logdir=str(Path(output).parent / 'agent'),
          seed=20261035, jax=config.jax, batch_size=2, batch_length=3,
          replay_context=1))
  assert all(device.platform == 'cpu' for device in jax.devices())
  fields = dict(inventory.FIELDS, **{
      'dyn/deter': ('float32', (8,)), 'dyn/stoch': ('float32', (2, 4))})
  original_chains = inventory.validate_chains

  def fixture_chains(rows):
    chains = original_chains(rows)  # Retain real topology/flags/coverage checks.
    assert len(chains) == 1 and chains[0]['rows'] == 12
    # The production metadata field is named length65_windows and uses 64.
    # This explicit T3/context1 fixture has nine legal four-row windows.
    chains[0]['length65_windows'] = 9
    return chains

  with tempfile.TemporaryDirectory(prefix='dl-normal-learning-') as temporary, \
      contextlib.ExitStack() as patches:
    for module, name, value in (
        (inventory, 'FIELDS', fields), (inventory, 'EXPECTED_FILES', 3),
        (inventory, 'CHAINS', 1), (inventory, 'validate_chains', fixture_chains),
        (replay, 'BATCH', 2), (replay, 'LENGTH', 3),
        (replay, 'CACHE', {'dyn/deter': (8,), 'dyn/stoch': (2, 4)}),
        (replay, 'EXPECTED_COUNTS', (3, 1, 12, 1, 11))):
      patches.enter_context(mock.patch.object(module, name, value))
    directory = Path(temporary) / 'run' / 'replay'
    directory.mkdir(parents=True)
    originals = []
    for c in range(3):
      arrays = {k: np.zeros((4, *tail), dtype) for k, (dtype, tail) in fields.items()}
      for i in range(4):
        at = 4 * c + i
        arrays['image'][i] = (np.arange(64 * 64 * 3).reshape(64, 64, 3) + at) % 256
        arrays['reward'][i] = 0 if at == 0 else np.float32(at / 12)
        arrays['action'][i] = 0 if at == 11 else np.linspace(-.4, .4, 12) + at / 100
        arrays['is_first'][i] = at == 0
        arrays['is_last'][i] = arrays['is_terminal'][i] = at == 11
        arrays['dyn/deter'][i] = np.float32(1 + at / 10)
        arrays['dyn/stoch'][i, :, at % 4] = 1
      arrays['stepid'][:, :16] = np.frombuffer(inventory.uuid_bytes(uuid(c + 1)), np.uint8)
      arrays['stepid'][:, 16:] = np.arange(4, dtype='>u4').view(np.uint8).reshape(4, 4)
      name = f'20261015T005815F{c * 1000:06d}-{uuid(c+1)}-{uuid(c+2)}-4.npz'
      np.savez_compressed(directory / name, **arrays)
      originals.append(arrays)
    source = {k: np.concatenate([a[k] for a in originals]) for k in fields}
    before = inventory.snapshot(directory)
    chunks = [inventory.validate_chunk(directory / entry['path']) for entry in before]
    report = dict(version=inventory.VERSION, schema=1, stable=True,
        source_git=inventory.SOURCE_GIT, replay_directory=str(directory.resolve()),
        run_directory=str(directory.parent.resolve()), before_inventory=before,
        after_inventory=before, rows=12, resets=1, train_actions=11,
        corrupt_chunks_skipped=0, model_loaded=False, gpu_used=False,
        sampler_rng_restored=False, online_queue_restored=False,
        environment_restored=False, seamless_resume_established=False,
        fields={k: [v[0], list(v[1])] for k, v in fields.items()},
        chunks=chunks, chains=inventory.validate_chains(chunks),
        object_scope='Generated tiny fixture: 12 rows/3 chunks; NOT original N1M')
    path = Path(temporary) / 'fixture_inventory.json'
    path.write_text(json.dumps(report), encoding='utf-8')
    pin = hashlib.sha256(path.read_bytes()).hexdigest()
    cache = replay.load_private_cache(directory, path, expected_inventory_sha256=pin)
    receipt = cache.receipt()
    assert receipt['original_inventory_authenticated'] is False
    assert len(cache._replay) == 9 and cache._replay.length == 4
    initial = host(agent.save())
    finite(initial)
    initial_step = int(initial['params']['opt/step/value'])
    assert any(k.startswith('opt/state/') for k in initial['params'])
    assert any(k.startswith('slowval/') for k in initial['params'])
    assert any(k.startswith('retnorm/') for k in initial['params'])
    carry = agent.init_train(2)
    context_fn = jax.jit(agent.model._apply_replay_context)
    previous_writers, previous_carry, seeds, updates = {}, None, [], []
    feedback_checks, override_checks, cross_chunk_checked = 0, 0, False
    for step, start in enumerate((2, 3, 4)):
      if previous_carry is not None:
        exact(carry, previous_carry)  # Returned carry is actually passed onward.
      starts = ((uuid(start // 4 + 1), start % 4),) * 2
      raw = cache.batch(starts)
      assert set(raw) == set(agent.spaces)
      assert raw['image'].shape == (2, 4, 64, 64, 3)
      assert not raw['consec'].any() and raw['is_first'][:, 0].all()
      if step == 0:
        assert len({identifier[:16].tobytes() for identifier in raw['stepid'][0]}) == 2
        cross_chunk_checked = True
      for key in set(fields) - set(replay.CACHE) - {'is_first'}:
        exact(raw[key], np.stack([source[key][start:start + 4]] * 2))
      assert not raw['is_first'][:, 1:].any()
      assert not raw['is_last'].any() and not raw['is_terminal'].any()
      if step:
        for row in range(2):
          for key in replay.CACHE:
            exact(raw[key][row, 0], previous_writers[raw['stepid'][row, 0].tobytes()][key])
        assert raw['dyn/deter'][0, 0].tobytes() != source['dyn/deter'][start].tobytes()
        feedback_checks += 1
      pre = host(agent.save())
      prepared = agent.prepare_batch(raw)
      exact(prepared['seed'], agent._seeds(step, agent.train_mirrored))
      seeds.append(host(prepared['seed']).tobytes().hex())
      assert int(agent.n_batches) == step + 1 and int(agent.n_updates) == step
      exact(pre['params'], agent.params)
      data = {k: v for k, v in prepared.items() if k != 'seed'}
      loss_carry, obs, previous_action, stepid = context_fn(carry, data)
      exact(loss_carry[1], dict(deter=raw['dyn/deter'][:, 0], stoch=raw['dyn/stoch'][:, 0]))
      exact(previous_action, dict(action=raw['action'][:, :-1]))
      exact(stepid, raw['stepid'][:, 1:])
      exact(obs, {k: raw[k][:, 1:] for k in agent.obs_space})
      # A read-only counterfactual carry checks the real method's consec0 path;
      # this altered carry is NEVER used for the actual update below.
      altered = jax.jit(lambda ca: (ca[0],
          dict(ca[1], deter=ca[1]['deter'] + 7), ca[2],
          dict(action=jnp.full_like(ca[3]['action'], -.9))))(carry)
      exact(context_fn(altered, data), (loss_carry, obs, previous_action, stepid))
      exact(pre['params'], agent.params)
      override_checks += 1
      carry, delayed_outs, delayed_metrics = agent.train(carry, dict(prepared))
      assert delayed_outs == delayed_metrics == {}
      assert agent.pending_outs is not None and agent.pending_mets is not None
      outs, metrics = agent.take_train_result()
      assert agent.pending_outs is agent.pending_mets is None
      assert set(outs) == {'replay'}
      finite((outs, metrics, carry))
      assert {k for k in metrics if k.startswith('loss/')} == {
          'loss/image', 'loss/rew', 'loss/con', 'loss/dyn', 'loss/rep',
          'loss/policy', 'loss/value', 'loss/repval'}
      assert float(metrics['opt/grad_norm']) > 0
      assert int(metrics['opt/updates']) == initial_step + step + 1
      saved = host(agent.save())
      assert saved['counters'] == dict(updates=step + 1, batches=step + 1,
          actions=0, eval_actions=0, eval_seed=20261035)
      assert int(saved['params']['opt/step/value']) == initial_step + step + 1
      finite(saved)
      assert changed(pre['params'], saved['params'])
      old_opt = {k: v for k, v in pre['params'].items() if k.startswith('opt/state/')}
      new_opt = {k: v for k, v in saved['params'].items() if k.startswith('opt/state/')}
      assert changed(old_opt, new_opt)
      assert agent.take_train_result() == ({}, {})
      exact(saved, agent.save())
      out = outs['replay']
      exact(out['stepid'], raw['stepid'][:, 1:])
      result = cache.strict_update(out, raw)
      assert cache._pending is None and not cache._poisoned
      assert result['updated_rows'] == 6 and result['unique_written_rows'] == 3
      previous_writers = {}
      for row in range(2):
        for time in range(3):
          identifier = out['stepid'][row, time].tobytes()
          previous_writers[identifier] = {key: out[key][row, time].copy() for key in replay.CACHE}
      # Independent identifier-addressed readback includes overlapping rows:
      # the later batch row's ACTUAL network result must be the final writer.
      for identifier, values in previous_writers.items():
        chunk = cache._replay.chunks[elements.UUID(identifier[:16])]
        index = int.from_bytes(identifier[16:], 'big')
        for key, value in values.items():
          exact(chunk.data[key][index], value)
      previous_carry = host(carry)
      if step == 0:
        assert any(np.any(v != 0) for v in new_opt.values())
        agent.load(saved)  # Full real moments/norm/targets, no reset or regex.
        exact(saved, agent.save())
      updates.append(dict(update=step + 1, global_start=start,
          opt_updates=int(metrics['opt/updates']), cache_rows=3,
          grad_norm=float(metrics['opt/grad_norm'])))
    final = host(agent.save())
    assert len(set(seeds)) == 3
    neural_keys = {k for k in final['params'] if k.startswith(('enc/', 'dyn/', 'dec/', 'rew/', 'con/', 'pol/', 'val/'))}
    assert changed({k: initial['params'][k] for k in neural_keys},
        {k: final['params'][k] for k in neural_keys})
    assert inventory.snapshot(directory) == before
    assert cache._pending is None
    Path(output).write_text(json.dumps(dict(updates=updates, receipt=receipt,
        real_agent_updates=3, input_rows=4, consumed_rows=3, cross_chunk=cross_chunk_checked,
        feedback_checks=feedback_checks, consec0_override_checks=override_checks,
        full_optimizer_checkpoint_preserved=True, original_npz_sha_unchanged=True,
        all_original_loss_families=True, original_optimizer_and_norm_config=True,
        all_saved_state_carry_outputs_metrics_finite=True,
        individual_gradient_leaves_checked=False, gpu_used=False,
        source_git_binding_authenticated=False, fixture_dimensions_counts_patched=True,
        real_n1m_authenticated=False, method_effectiveness_established=False,
        scope='Generated tiny CPU image64/action12; 4-row inputs, not real65/N1M; no environment/model-scale/online-resume acceptance')))
  del dll_directory


class NormalLearningTest(unittest.TestCase):
  def test_three_real_updates_feed_written_context_to_shifted_nonreset_batches(self):
    with tempfile.TemporaryDirectory(prefix='dl-learning-result-') as temporary:
      output = Path(temporary) / 'result.json'
      result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
          '--fixture', str(output)], cwd=ROOT, text=True, capture_output=True,
          timeout=240, env=dict(os.environ, PYTHONPATH=str(ROOT),
              CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', MUJOCO_GL='disable'))
      self.assertEqual(result.returncode, 0, result.stdout[-4000:] + result.stderr[-6000:])
      proof = json.loads(output.read_text())
      self.assertEqual(proof['real_agent_updates'], 3)
      self.assertEqual(proof['feedback_checks'], 2)
      self.assertEqual(proof['consec0_override_checks'], 3)
      self.assertFalse(proof['receipt']['original_inventory_authenticated'])
      self.assertFalse(proof['real_n1m_authenticated'])
      self.assertFalse(proof['method_effectiveness_established'])
      self.assertTrue(proof['original_npz_sha_unchanged'])
      print(json.dumps(proof, sort_keys=True))


if __name__ == '__main__':
  if len(sys.argv) == 3 and sys.argv[1] == '--fixture':
    learning_fixture(sys.argv[2])
  else:
    unittest.main()
