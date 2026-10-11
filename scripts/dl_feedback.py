"""DL-feedback-r1: one current-observation omission, then normal feedback.

This diagnostic is calibration-only and read-only. True finite-window rewards
are not inferred from the learned reward head and never become gate labels.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dl_diagnostic as d
from scripts import dl_predictive_mc as pinned

VERSION = 'DL-feedback-r1'
IMPLEMENTATION = 'DL-feedback-engineering-r2'
PRIVATE_SEED, STAT_SEED = 20261018, 20261019
EPISODES, POSITIONS, PAIRS, HORIZON, BUDGET = 8, 4, 4, 100, 3600
POSITION_INDICES = (0, 5, 10, 15)
SOURCE_SHA = 'f2d24da1831da2ffca22d313e7f534c96fc0d3eb'
SOURCE_PLAN_SHA = '633efe28d551d5793ac8f0ed5f0e36e937e7f1af3c8edefc1d9f5fa278146fcc'
SOURCE_COMPLETE_SHA = 'd155f577d19d2bbc051ec56b95f15e26afc774c130c3c9728a7fe599f295f3cf'
ARTIFACT_AUDIT_SHA = '8f93788ca212118139fab135036fd47ccbad79f04a34139b5dfa7d2dd26f7516'
PHYSICS_AUDIT_SHA = '8ed3528fae18dc7a3043c767da87f873642a40a1a028ce86247f0521c3264722'
CONTRACT = 'production_clip_unify_normalize_v1'


def require(condition, message):
  if not condition:
    raise ValueError(message)


def analyze_utility(rewards, executed, episodes):
  """Fixed signed, episode-clustered analysis; masks do not prove termination."""
  rewards, executed, episodes = [np.asarray(x) for x in (rewards, executed, episodes)]
  shape = (EPISODES, POSITIONS, PAIRS, 2, HORIZON)
  require(rewards.shape == executed.shape == shape and rewards.dtype == np.float64 and
      executed.dtype == np.bool_, 'Frozen utility shape/dtype differs')
  require(episodes.shape == (EPISODES,) and episodes.dtype.kind in 'iu' and
      (episodes >= 0).all() and len(np.unique(episodes)) == EPISODES,
      'Require eight unique integer episode IDs')
  require(np.isfinite(rewards).all(), 'Nonfinite actual utility rewards')
  require(executed[..., 0].all() and not np.any(executed[..., 1:] & ~executed[..., :-1]),
      'Executed masks must be nonempty contiguous prefixes')
  require(np.all(rewards[~executed] == 0), 'Absorbing padding rewards must be zero')
  signed = {}
  for prefix in (1, 10, 50, 100):
    signed[str(prefix)] = (rewards[:, :, :, 0, :prefix].sum(-1, dtype=np.float64) -
        rewards[:, :, :, 1, :prefix].sum(-1, dtype=np.float64))
  u = signed['100']
  u_episode = u.mean((1, 2), dtype=np.float64)
  require(all(np.isfinite(v).all() for v in signed.values()) and
      np.isfinite(u_episode).all(), 'Nonfinite utility reduction')
  primary = float(u_episode.mean(dtype=np.float64))
  draws = np.random.default_rng(STAT_SEED).integers(0, EPISODES, (2000, EPISODES))
  bootstrap = u_episode[draws].mean(1, dtype=np.float64)
  interval = np.quantile(bootstrap, [.025, .975], method='linear')
  require(np.isfinite(bootstrap).all() and np.isfinite(interval).all() and
      np.isfinite(u.std(-1, ddof=1)).all(), 'Nonfinite utility uncertainty')
  if interval[0] > 0:
    status = 'exploratory_average_opportunity_supported'
  elif interval[1] <= 0:
    status = 'exploratory_average_opportunity_not_supported'
  else:
    status = 'exploratory_average_opportunity_inconclusive'
  return dict(version=VERSION, episodes=episodes.astype(int).tolist(),
      primary=primary, interval95=interval.tolist(), u_pair=u.tolist(),
      u_episode=u_episode.tolist(), signed_prefix={k:v.tolist() for k,v in signed.items()},
      within_state=dict(std=u.std(-1, ddof=1).tolist(), range=np.ptp(u, axis=-1).tolist()),
      per_rng=u.mean((0, 1)).tolist(), first_two=float(u[:, :, :2].mean()),
      last_two=float(u[:, :, 2:].mean()),
      leave_one_rng=[float(np.delete(u, r, axis=2).mean()) for r in range(PAIRS)],
      position_mean=u.mean((0, 2)).tolist(), status=status,
      statistical_seed=STAT_SEED, bootstrap_draws=2000,
      bootstrap_unit='whole episode with all fixed positions and RNG pairs',
      interval_scope='Exploratory nominal episode percentile interval for this fixed four-RNG ensemble; '
          'not an exact policy-path integral, rare-state coverage bound or cross-model replication',
      average_opportunity_is_selective_necessity=False, exact_state_labels_established=False,
      training_authorized=False, proceed_short_training=False,
      eligible_method_training=False, method_effectiveness_established=False)


def load_sources(dataset, artifact_audit, physics_audit):
  """Hash all source bytes; decode only the eight fixed calibration files."""
  dataset = Path(dataset).resolve()
  plan_path, complete_path = dataset/'collection_plan.json', dataset/'collection_complete.json'
  require(d.sha256(plan_path) == SOURCE_PLAN_SHA and d.sha256(complete_path) == SOURCE_COMPLETE_SHA and
      d.sha256(artifact_audit) == ARTIFACT_AUDIT_SHA and d.sha256(physics_audit) == PHYSICS_AUDIT_SHA,
      'Pinned completed production source/audit hashes differ')
  read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
  plan, complete, audit, physical = [read(p) for p in
      (plan_path, complete_path, artifact_audit, physics_audit)]
  expected_packages = plan['environment']['packages']
  require({name:importlib.metadata.version(name) for name in expected_packages} == expected_packages,
      'Production source and actual runtime dependency versions differ')
  definitions = d.episode_plan(32, 8, 20261020)
  calibration = [x for x in definitions if x['split'] == 'calibration']
  require(plan['version'] == complete['version'] == 'DL-source-r1' and
      plan['implementation'] == complete['implementation'] == 'DL-source-engineering-r2' and
      plan['action_contract'] == complete['action_contract'] == CONTRACT and
      plan['metadata']['git_sha'] == SOURCE_SHA and plan['episodes'] == definitions and
      complete['plan_sha256'] == SOURCE_PLAN_SHA, 'Incorrect production source contract')
  require(complete['params_unchanged'] is True and complete['training_counters_unchanged'] is True and
      complete['eval_counters_legitimately_advanced'] is True and complete['optimizer_updates'] == 0 and
      plan['metadata']['checkpoint_sha256'] == pinned.CHECKPOINT_SHA and
      plan['metadata']['params_sha256'] == pinned.PARAMS_SHA and
      plan['metadata']['config_source_sha256'] == pinned.CONFIG_SHA,
      'Source checkpoint/state identity differs')
  require(audit['source_contract_artifact_audit_passed'] is True and
      audit['expected_source_sha'] == physical['expected_source_sha'] == SOURCE_SHA and
      audit['blind_arrays_read'] is False and physical['status'] == 'passed' and
      physical['blind_npz_arrays_opened'] is False and physical['physics_steps'] == 12800 and
      physical['source_json_sha256']['complete'] == SOURCE_COMPLETE_SHA and
      audit['metadata_sha256']['collection_complete.json'] == SOURCE_COMPLETE_SHA,
      'Completed independent source audits are missing or mismatched')
  require(len(complete['artifacts']) == 32, 'Incomplete source artifacts')
  selected = []
  expected_positions = d.sample_positions(1001, 16, 100)
  import ml_dtypes  # Register trusted BF16 carry dtype before project pickle decode.
  for entry, definition in zip(complete['artifacts'], definitions):
    require({k:entry[k] for k in definition} == definition and
        entry['path'] == f'episode_{definition["episode"]:03}.npz', 'Source episode identity differs')
    path = (dataset/entry['path']).resolve()
    require(path.is_relative_to(dataset) and path.is_file() and
        d.sha256(path) == entry['sha256'], 'Source artifact unsafe or altered')
    if definition['split'] != 'calibration':
      continue
    with np.load(path, allow_pickle=False) as artifact:
      fields = ('image', 'reward', 'is_first', 'is_last', 'is_terminal', 'action',
          'action_applied', 'action_executed', 'positions', 'before_carries', 'snapshots', 'action_contract')
      data = {k:artifact[k].copy() for k in fields}
    require(data['action_contract'].item() == CONTRACT and
        np.array_equal(data['positions'], expected_positions) and data['image'].shape == (1001,64,64,3) and
        data['image'].dtype == np.uint8, 'Source frames/positions differ')
    # Only pinned project-generated state pickles, after byte verification above.
    carries = pickle.loads(data.pop('before_carries').tobytes())
    snapshots = pickle.loads(data.pop('snapshots').tobytes())
    require(len(carries) == 16 and len(snapshots) == 1001,
        'Actual source carry/snapshot coverage differs')
    selected.append(dict(definition=definition, artifact=entry, data=data,
        carries=carries, snapshots=snapshots))
  require([x['definition'] for x in selected] == calibration, 'Calibration coverage differs')
  return selected, dict(source_plan_sha256=SOURCE_PLAN_SHA, source_complete_sha256=SOURCE_COMPLETE_SHA,
      artifact_audit_sha256=ARTIFACT_AUDIT_SHA, physics_audit_sha256=PHYSICS_AUDIT_SHA,
      source_implementation_sha=SOURCE_SHA, calibration_episodes=[x['episode'] for x in calibration],
      all_32_artifact_hashes_verified=True, blind_arrays_read=False)


def replay_policy(policy, record):
  """Cached observed RGB replay; this does not replay the renderer."""
  import copy
  from scripts.dl_feedback_model import tree_digest
  from scripts.dl_feedback_physics import validate_record
  count = validate_record(record)
  carry = copy.deepcopy(record['initial_carry'])
  for step in range(count):
    before = tree_digest(carry)
    require(before == record['pre_carry_sha256'][step],
        'Cached policy initial/previous carry differs')
    obs, key = copy.deepcopy(record['observations'][step]), record['keys'][step].copy()
    observation_hash, key_bytes = tree_digest(obs), key.tobytes()
    after, raw, proof = policy.step(carry, obs, key, omit=(record['prior'] and step == 0))
    require(tree_digest(carry) == before and tree_digest(obs) == observation_hash and
        key.tobytes() == key_bytes, 'Cached policy callback mutated its input carry/observation/key')
    require(raw.dtype == record['raw'].dtype and raw.shape == record['raw'][step].shape and
        raw.tobytes() == record['raw'][step].tobytes() and
        tree_digest(after) == record['after_carry_sha256'][step],
        'Cached actual policy raw/after carry byte replay differs')
    require(set(proof) == set(record['proofs'][step]), 'Cached policy proof fields differ')
    for name, expected in record['proofs'][step].items():
      value = proof[name]
      require(value.dtype == expected.dtype and value.shape == expected.shape and
          value.tobytes() == expected.tobytes(), 'Cached policy numeric proof differs: ' + name)
    carry = after
  return dict(policy_steps=count, raw_byte_exact=True, after_carry_byte_hash_exact=True,
      all_numeric_proofs_byte_exact=True, cached_own_rgb=True, renderer_replayed=False)


def supervise(command, output, seconds=BUDGET):
  """Own one process group, with a hard wall budget including all child work."""
  require(np.isfinite(seconds) and seconds > 0, 'Require positive finite wall budget')
  output = Path(output).resolve()
  if output.exists():
    raise FileExistsError('New feedback attempt output must not exist')
  started = time.monotonic()
  output.mkdir(parents=True)
  d.write_json(output/'budget.json', dict(version=VERSION, implementation=IMPLEMENTATION,
      parent_pid=os.getpid(), git_sha=d.git_sha(), seconds=seconds,
      monotonic_start=started, script_sha256=d.sha256(Path(__file__))))
  timed_out = False
  with (output/'feedback.log').open('x', encoding='utf-8') as log:
    child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=os.name != 'nt',
        env=dict(os.environ, DL_FEEDBACK_PARENT=str(os.getpid())))
    try:
      code = child.wait(timeout=max(.001, started+seconds-time.monotonic()))
    except subprocess.TimeoutExpired:
      timed_out = True
      if os.name != 'nt':
        try:
          os.killpg(child.pid, signal.SIGKILL)  # Only the freshly owned child group.
        except ProcessLookupError:
          pass
      else:
        child.kill()
      child.wait()
      code = 124
    finally:
      # A failed leader may leave descendants. This is our start_new_session
      # group only; never search/kill other project or system processes.
      if os.name != 'nt':
        try:
          os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
          pass
  (output/'worker_exit_code.txt').write_text(str(code)+'\n', encoding='utf-8')
  receipt = dict(version=VERSION, implementation=IMPLEMENTATION, worker_pid=child.pid,
      exit_code=code, timed_out=timed_out, worker_terminal=child.poll() is not None,
      owned_group_cleanup_requested=os.name != 'nt',
      reason='incomplete_hard_budget' if timed_out else ('worker_completed' if code == 0 else 'worker_failed'),
      elapsed_seconds=time.monotonic()-started, actual_git_sha=d.git_sha(),
      budget_sha256=d.sha256(output/'budget.json'),
      training_authorized=False, proceed_short_training=False,
      eligible_method_training=False, method_effectiveness_established=False)
  d.write_json(output/'supervisor_complete.json', receipt)
  return receipt


def source_observation(data, position):
  # Source indexing returns NumPy scalar flags/reward; actual wrappers return
  # zero-dimensional arrays. Restore the container, preserving dtype/bytes.
  return {key:np.asarray(data[key][position]).copy() for key in
      ('image','reward','is_first','is_last','is_terminal')}


def worker(args):
  """Frozen calibration-only actual model/physics run, never an update."""
  pinned.require_clean()
  output = Path(args.output).resolve()
  budget = json.loads((output/'budget.json').read_text(encoding='utf-8'))
  require(budget['parent_pid'] == os.getppid() == int(os.environ.get('DL_FEEDBACK_PARENT', '-1')) and
      budget['seconds'] == BUDGET and budget['git_sha'] == d.git_sha() and
      budget['script_sha256'] == d.sha256(Path(__file__)),
      'Worker must be direct child of the actual frozen bounded supervisor')
  os.kill(os.getppid(), 0)
  deadline = budget['monotonic_start'] + BUDGET
  def check_deadline():
    if time.monotonic() >= deadline:
      raise TimeoutError('DL-feedback-r1 incomplete: hard budget exhausted')
  from scripts.dl_resources import require_resource, verify_runtime
  resource = require_resource(args.resource_record)
  require(resource['physical_gpu'] == 4, 'DL-feedback-r1 is limited to sv3 physical GPU4')
  sources, binding = load_sources(args.dataset, args.artifact_audit, args.physics_audit)
  check_deadline()
  agent, config, metadata = pinned.load_agent_no_env(args)
  from embodied.run.protocol_v1 import parameter_digest
  from scripts.dl_feedback_model import FeedbackPolicy, named_root_key, counters, tree_digest
  from scripts.dl_feedback_physics import (FeedbackEnvironment, prepare_initial_observation,
      run_branch, verify_physics_replay)
  require(metadata['params_sha256'] == pinned.PARAMS_SHA, 'Loaded baseline parameter identity differs')
  metadata['code_version'] = VERSION
  frozen = counters(agent)
  frozen_saved_counters = agent.save()['counters']
  policy = FeedbackPolicy(agent)
  positions = d.sample_positions(1001, 16, HORIZON)[list(POSITION_INDICES)]
  require(np.array_equal(positions, [1,300,600,900]), 'Frozen eligible positions differ')
  source_oracles, calibration = [], []
  # Finish every actual production/source oracle before computing new returns.
  for item in sources:
    definition, data = item['definition'], item['data']
    chosen = [item['carries'][i] for i in POSITION_INDICES]
    for pos, carry in zip(positions, chosen):
      check_deadline()
      require(carry['position'] == pos and carry['eval_counter'] == pos and
          carry['eval_seed'] == definition['policy_seed'] and
          carry['carry_sha256'] == tree_digest(carry['carry']) and
          data['action_executed'][pos] and not data['is_last'][pos] and
          item['snapshots'][pos]['suite']['step_count'] == pos,
          'Actual source carry/key/clock identity differs')
      proof = policy.oracle_source(carry, source_observation(data, int(pos)), data['action'][pos])
      source_oracles.append(dict(episode=definition['episode'],position=int(pos),**proof))
    calibration.append(dict(definition=definition, source_artifact=item['artifact'],
        carry_positions=positions.tolist(), carry_sha256=[x['carry_sha256'] for x in chosen]))
  require(len(source_oracles) == 32 and counters(agent) == frozen,
      'Actual model oracles incomplete or changed original counters')
  resource['runtime_compute'] = verify_runtime(resource, compute=True)
  plan = dict(version=VERSION, implementation=IMPLEMENTATION, action_contract=CONTRACT,
      metadata=metadata, source_binding=binding, calibration=calibration,
      positions=positions.tolist(), position_indices=list(POSITION_INDICES), pairs=PAIRS,
      horizon=HORIZON, private_seed=PRIVATE_SEED, stat_seed=STAT_SEED,
      key_convention=dict(tag=0x46445231, absolute_step='position+s',
          high_exclusive=int(np.iinfo(np.uint32).max), branch_absent=True),
      source_oracles=source_oracles, resource=resource, budget_seconds=BUDGET,
      branch_order=['posterior','one_step_prior_then_posterior'],
      source_files_sha256={name:d.sha256(ROOT/name) for name in
          ('scripts/dl_feedback.py','scripts/dl_feedback_model.py','scripts/dl_feedback_physics.py',
           'scripts/dl_production_source.py','scripts/dl_action_contract.py','dreamerv3/agent.py',
           'dreamerv3/rssm.py','embodied/jax/agent.py','embodied/core/wrappers.py')},
      environment=dict(python=sys.version, packages={name:importlib.metadata.version(name) for name in
          ('jax','jaxlib','numpy','dm-control','mujoco','elements','ninjax')}),
      optimizer_updates=0, training_authorized=False, gate_changed=False,
      blind_arrays_read=False, exact_state_labels_established=False)
  d.write_json(output/'plan.json', plan)
  print(json.dumps(dict(stage='source_oracles', positions=32, actual_byte_exact=True)),flush=True)
  rewards, masks, artifacts = [], [], []
  for item in sources:
    definition, data = item['definition'], item['data']
    episode = definition['episode']
    adapter = FeedbackEnvironment(d.make_dmc(args.task, definition['seed']))
    records, verifications, fresh_images, render_proofs = [], [], [], []
    try:
      spec = item['artifact']['action_spec']
      require(spec['dtype'] == str(adapter.low.dtype) == str(adapter.high.dtype) == 'float64' and
          spec['raw_dtype'] == 'float32' and
          np.array_equal(np.asarray(spec['low'],np.float64),adapter.low) and
          np.array_equal(np.asarray(spec['high'],np.float64),adapter.high),
          'Actual action specification differs from the completed production source')
      for index, pos in zip(POSITION_INDICES, positions):
        pos = int(pos)
        check_deadline()
        initial, rendered = prepare_initial_observation(adapter,
            item['snapshots'][pos], source_observation(data,pos))
        fresh_images.append(initial['image'].copy())
        render_proofs.append(rendered)
        if not artifacts and len(fresh_images) == 1:
          resource['runtime_cuda_egl'] = verify_runtime(resource, compute=True, graphics=True)
        for pair in range(PAIRS):
          keys = np.stack([named_root_key(episode,pos,pair,pos+s) for s in range(HORIZON)])
          branches = []
          for branch in range(2):
            check_deadline()
            record = run_branch(policy,adapter,item['snapshots'][pos],item['carries'][index]['carry'],
                keys,prior=bool(branch),initial_obs=initial,horizon=HORIZON)
            try:
              physical = verify_physics_replay(adapter, record)
              replay = replay_policy(policy, record)
            except Exception:
              # Retain a failed conditional replay for diagnosis, without any
              # utility classification or complete-window claim.
              failure = output/f'failed-replay-ep{episode:03}-pos{pos:04}-pair{pair}-branch{branch}.npz'
              if not failure.exists():
                np.savez_compressed(failure,episode=np.array(episode,np.int32),
                    position=np.array(pos,np.int32),pair=np.array(pair,np.int32),
                    branch=np.array(branch,np.int32),
                    records=np.frombuffer(pickle.dumps(record,protocol=5),np.uint8),
                    semantic_verification_passed=np.array(False))
              raise
            record.update(episode=episode, position=pos, pair=pair, branch=branch)
            records.append(record)
            branches.append(record)
            verifications.append(dict(position=pos,pair=pair,branch=branch,
                end_type=record['end_type'],executed_steps=record['executed_steps'],
                physics=physical,policy=replay))
          q,p = branches
          require(q['proofs'][0]['h'].dtype == p['proofs'][0]['h'].dtype and
              q['proofs'][0]['h'].tobytes() == p['proofs'][0]['h'].tobytes(),
              'Initial posterior and prior do not use the same actual new h')
          for step in range(min(q['executed_steps'],p['executed_steps'])):
            require(all(q['proofs'][step][name].tobytes() == p['proofs'][step][name].tobytes()
                for name in ('root_key','latent_key','actor_key','rng_consumption')),
                'Branches do not share actual production sampling positions')
          check_deadline()
          print(json.dumps(dict(stage='paired_feedback',episode=episode,position=pos,pair=pair,
              branch_steps=[x['executed_steps'] for x in branches],semantic_replays_passed=True)),flush=True)
      require(len(records) == 32 and counters(agent) == frozen, 'Episode coverage/counter isolation differs')
      shape = (POSITIONS,PAIRS,2)
      fields = ('keys','raw','applied','reward_f64','reward_f32','discount','is_first','is_last',
          'is_terminal','action_executed','observation_executed')
      arrays = {name:np.stack([r[name] for r in records]).reshape(*shape,*records[0][name].shape)
          for name in fields}
      path = output/f'episode_{episode:03}.npz'
      require(not path.exists(), 'Feedback episode output already exists')
      np.savez_compressed(path, **arrays, episode=np.array(episode,np.int32),
          positions=positions.astype(np.int32),source_image=data['image'][positions],
          fresh_image=np.stack(fresh_images),
          records=np.frombuffer(pickle.dumps(records,protocol=5),np.uint8))
      artifacts.append(dict(episode=episode,path=path.name,sha256=d.sha256(path),records=32,
          physics_steps=sum(x['physics']['physics_steps'] for x in verifications),
          policy_replay_steps=sum(x['policy']['policy_steps'] for x in verifications),
          initial_render=render_proofs,verifications=verifications))
      rewards.append(arrays['reward_f64'])
      masks.append(arrays['action_executed'])
    finally:
      adapter.close()
    check_deadline()
  require(parameter_digest(agent) == pinned.PARAMS_SHA and counters(agent) == frozen and
      agent.save()['counters'] == frozen_saved_counters,
      'Read-only feedback changed model/optimizer/state/counters')
  summary = analyze_utility(np.stack(rewards),np.stack(masks),
      np.array([x['episode'] for x in artifacts],np.int32))
  summary.update(implementation=IMPLEMENTATION,actual_git_sha=d.git_sha(),
      plan_sha256=d.sha256(output/'plan.json'),evidence_artifacts=artifacts,
      params_unchanged=True,counters_unchanged=True,optimizer_updates=0,
      counters_before=frozen,counters_after=counters(agent),source_binding=binding,
      resource=resource,elapsed_seconds=time.monotonic()-budget['monotonic_start'],
      action_contract=CONTRACT,blind_arrays_read=False,
      full_closed_loop_bit_exact_established=False,renderer_byte_exact_established=False)
  check_deadline()
  d.write_json(output/'result.json',summary)
  d.write_json(output/'feedback_complete.json',dict(version=VERSION,implementation=IMPLEMENTATION,
      result_sha256=d.sha256(output/'result.json'),plan_sha256=summary['plan_sha256'],
      actual_git_sha=d.git_sha(),complete_episodes=8,complete_records=256,
      params_unchanged=True,counters_unchanged=True,optimizer_updates=0,
      training_authorized=False,proceed_short_training=False,
      eligible_method_training=False,method_effectiveness_established=False))
  print(json.dumps(dict(complete=True,decision=summary['status'],training_authorized=False)),flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for field in ('checkpoint','config','dataset','artifact_audit','physics_audit','output','resource_record'):
    parser.add_argument('--'+field.replace('_','-'), required=True)
  parser.add_argument('--task',default='dmc_quadruped_walk',choices=['dmc_quadruped_walk'])
  parser.add_argument('--platform',default='cuda',choices=['cuda'])
  parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.worker:
    worker(args)
    return 0
  pinned.require_clean()
  from scripts.dl_resources import require_resource
  require(require_resource(args.resource_record)['physical_gpu'] == 4,
      'DL-feedback-r1 is limited to assigned sv3 physical GPU4')
  command = [sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker']
  receipt = supervise(command,args.output,seconds=BUDGET)
  print(json.dumps(receipt),flush=True)
  return receipt['exit_code']


if __name__ == '__main__':
  sys.exit(main())
