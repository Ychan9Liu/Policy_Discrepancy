"""Synthetic CPU fixtures only; no real source/model/render/physics data."""
import copy
import contextlib
import hashlib
import math
import os
from pathlib import Path
import pickle
import tempfile
import subprocess
import sys
import time
import types
import unittest
from unittest import mock

import numpy as np

from analysis import dl_rep_native_audit as audit


class FakePhysics:
  def __init__(self, seed):
    self.seed, self.clock = seed, 0
    self.low = np.full(12, -2., np.float64)
    self.high = np.full(12, 3., np.float64)
    self.closed = False

  def snapshot(self):
    return dict(suite=dict(integration=np.array([self.seed, self.clock], np.float64),
        task={'fixture_seed': self.seed}, step_count=self.clock,
        reset_next_step=self.clock == 1000), from_dm_done=self.clock == 1000)

  def restore(self, state):
    self.clock = state['suite']['step_count']

  def step(self, applied):
    self.clock += 1
    return types.SimpleNamespace(reward=np.float64(1), discount=np.float64(1),
        first=lambda: False, last=lambda: self.clock == 1000)

  def close(self):
    self.closed = True


def fake_record(seed=1024, group='0'):
  physics = FakePhysics(seed)
  snapshots = [physics.snapshot()]
  for _ in range(1000):
    physics.step(np.zeros(12, np.float64))
    snapshots.append(physics.snapshot())
  observations = dict(image=np.broadcast_to(np.zeros((64, 64, 3), np.uint8),
      (1001, 64, 64, 3)), reward=np.r_[np.float32(0), np.ones(1000, np.float32)],
      is_first=np.arange(1001) == 0, is_last=np.arange(1001) == 1000,
      is_terminal=np.zeros(1001, bool))
  raw = np.zeros((1000, 12), np.float32)
  applied = np.full((1000, 12), .5, np.float64)
  keys = audit.named_roots(seed)
  trajectory = dict(version=audit.VERSION, seed=seed, observations=observations,
      raw=raw, applied=applied, reward_f64=np.ones(1000, np.float64),
      discount_f64=np.ones(1000, np.float64), root_keys=keys,
      latent_keys=keys.copy(), actor_keys=keys.copy(), carry_hashes=['a'*64]*1001,
      counts=dict(transitions=1000, observations=1001, policy_calls=1000, resets=1,
          oracle_calls=3), return_f64=1000.,
      readonly_full_state_and_counters_unchanged=True,
      public_policy_copy_used_for_scientific_calls=False, model_state_sha256='b'*64,
      oracle_receipts=[dict(step=step, receipt=dict(public_raw_byte_exact=True,
          public_after_carry_byte_exact=True, currentparams_policycopy_typed_byte_exact=True,
          eval_counters_restored=True, full_state_and_counters_unchanged=True,
          actual_native_sample=True, policy_copy_sha256='c'*64, root_key=keys[step].copy(),
          public_model_calls=1)) for step in (0,500,999)])
  trajectory['arrays_sha256'] = audit.numeric_digest({name: trajectory[name] for name in
      ('observations','raw','applied','reward_f64','discount_f64','root_keys','latent_keys','actor_keys')})
  return dict(version=audit.VERSION, group=group, seed=seed, trajectory=trajectory,
      snapshots=snapshots, stages=[dict(raw=raw[i].copy(), clipped=raw[i].copy(),
          unified=raw[i].copy(), applied=applied[i].copy()) for i in range(1000)],
      reset_receipt=dict(seed=seed, initial_snapshot_sha256=audit.snapshot_hash(snapshots[0]),
          initial_rgb_sha256=hashlib.sha256(observations['image'][0].tobytes()).hexdigest(),
          rendering_full_state_byte_exact=True), camera=2, size=[64,64], proprio=False,
      action_repeat=1, actual_physics_transitions=1000, actual_RGB_frames=1001,
      production_chain=['ClipAction', 'CheckSpaces', 'UnifyDtypes', 'NormalizeAction',
          'DMC', 'ActionRepeat', 'FromDM'])


def fake_envelope(directory):
  """Tiny files and typed receipts only; known pins overridden for this fixture."""
  root = Path(directory)/'primary'; root.mkdir()
  old = Path(directory)/'old-primary'; old.mkdir()
  git = 'a'*40
  def write(path, value):
    path.write_text(__import__('json').dumps(value,allow_nan=False),encoding='utf-8')
    return audit._sha(path)
  inputs, override = {}, {}
  for constant, name, parent in (('INITIAL_CHECKPOINT','initial.pkl',old),
      ('B_CHECKPOINT','B-checkpoint.pkl',old),('CONFIG_SHA','config.yaml',old),
      ('CONTEXT_SHA','contexts.pkl',old),('OLD_AUDIT_SUPERVISOR_SHA','old-supervisor.json',old),
      ('OLD_AUDIT_TERMINAL_SHA','old-external.json',old)):
    path = parent/name; path.write_bytes(('synthetic-only-'+name).encode())
    pin = audit._sha(path); inputs[str(path.resolve())] = pin; override[constant] = pin
  source_rows = []
  for index in range(32):
    path = old/f'synthetic-source-{index}.npz'; path.write_bytes(str(index).encode())
    pin = audit._sha(path); inputs[str(path.resolve())] = pin
    source_rows.append(dict(path=str(path.resolve()),sha256=pin))
  counters = dict(updates=0,batches=0,actions=0,eval_actions=0,eval_seed=0)
  original = dict(binding={'fixture':'original-common'},source_checkpoint_digest='f'*64,
      initial_state_digest=audit.INITIAL_STATE,initial_params_digest='e'*64,
      calibration_arrays_digest='d'*64,restored_counters=counters,saved_source_counters=counters,
      optimizer_counter=dict(key='opt/step/value',dtype='int32',shape=[],initial=74493,final=74621,updates=128),
      receipts=[dict(update=index+1,seed_counter=index,train_seed=[index,index+1],
          counters={**counters,'updates':index+1,'batches':index+1}) for index in range(128)])
  override['OLD_B_COMPLETE_SHA'] = write(old/'B-complete.json',{'learning':original})
  inputs[str((old/'B-complete.json').resolve())] = override['OLD_B_COMPLETE_SHA']
  old_audit = dict(status='passed',primary_git_sha=audit.OLD_PRIMARY_GIT,
      auditor_git_sha=audit.OLD_AUDITOR_GIT,primary=str(old.resolve()),
      complete_primary_exit_and_parent_preblind_seal_verified=True,
      actual_gpu_devices_used=0,actual_independent_physics_steps=3360)
  override['OLD_AUDIT_SHA'] = write(old/'audit.json',old_audit)
  inputs[str((old/'audit.json').resolve())] = override['OLD_AUDIT_SHA']
  authentication = dict(version=audit.VERSION,actual_git_sha=git,primary=str(old.resolve()),
      primary_git_sha=audit.OLD_PRIMARY_GIT,auditor_git_sha=audit.OLD_AUDITOR_GIT,
      old_audit_sha256=override['OLD_AUDIT_SHA'],immutable_inputs_sha256=inputs,
      original_19_producer_files_unchanged=True,old_full_exit_and_independent_audit_authenticated=True,
      old_B_retrained=False,original_blind_arrays_decoded=False,original_B_learning=original,
      old_files_sha256={'B-checkpoint.pkl':override['B_CHECKPOINT'],'contexts.pkl':override['CONTEXT_SHA']},
      source_binding={'all_32_artifact_hashes':source_rows})
  auth_pin = write(root/'old-authenticated.json',authentication)
  learning = copy.deepcopy(original)
  reset = dict(active_count=8,reset_count=16,excess_sum=3.5)
  whole = dict(active_count=64,position_count=1024,nonreset_active_count=56,
      excess_sum=10.5,nonreset_excess_sum=7.)
  for row in learning['receipts']:
    row.update(reset_pressure=reset.copy(),whole_window_pressure=whole.copy())
  learning.update(version='DL-reset-allrep-learning-r1',protocol_adopted=True,
      working_draft_not_adopted=False,group='A',layout_version='DL-reset-learning-r1',updates=128,
      final_state_digest='c'*64,entire_window_rep_gradient_stopped=True,
      rep_forward_values_retained=True,other_loss_branches_retained=True,
      posterior_features_and_recurrent_carry_not_stopped=True,historical_optimizer_momentum_retained=True,
      per_update_init_train=True,original_B_receipt_values_matched=True,total_loss_positions=131072,
      total_reset_loss_positions=2048,reset_pressure=dict(active_count=1024,reset_count=2048,
          excess_sum=448.,entire_128_update_metrics_observed=True),
      whole_window_pressure=dict(active_count=8192,position_count=131072,nonreset_active_count=7168,
          excess_sum=1344.,nonreset_excess_sum=896.,entire_128_update_metrics_observed=True))
  def device(graphics):
    def runtime(enabled):
      return dict(pid=101,physical_gpu=4,compute_uuid=[audit.GPU_UUID],graphics_verified=enabled,
          live_state=dict(gpu_uuid=audit.GPU_UUID,physical_gpu=4,processes=[{'pid':101}],
              pmon_jobs=[dict(pid=101,type='CG' if enabled else 'C')]))
    return dict(physical_gpu=4,gpu_uuid=audit.GPU_UUID,compute_uuid=audit.GPU_UUID,egl_uuid=audit.GPU_UUID,
        logical_server='sv3',actual_hostname='lyg0326',cuda_visible_devices=audit.GPU_UUID,
        mujoco_egl_device_id='4',occupancy='idle',exclusive=True,runtime_compute_start=runtime(False),
        **({'runtime_compute_and_graphics_start':runtime(True),
            'runtime_compute_and_graphics_end':runtime(True)} if graphics else
            {'runtime_compute_end':runtime(False)}))
  (root/'A-checkpoint.pkl').write_bytes(b'synthetic-A-state-not-decoded')
  a_checkpoint_pin = audit._sha(root/'A-checkpoint.pkl')
  a = dict(version=audit.VERSION,actual_git_sha=git,phase='A',learning=learning,
      old_authentication_sha256=auth_pin,original_B_learning_reference_authenticated=True,
      original_blind_arrays_decoded=False,additional_F_measurements_performed=False,
      checkpoint_sha256=a_checkpoint_pin,cal_image_frames_decoded=8008,
      source_binding=authentication['source_binding'],device=device(False))
  a_pin = write(root/'A-complete.json',a)
  budget_pin = write(root/'budget.json',dict(version=audit.VERSION,seconds=7200,actual_git_sha=git,
      parent_pid=100,planned_physical_gpu=4,max_gpu_devices=1,max_gpu_hours=2,monotonic_start=1.,
      source_files_sha256={'fixture':'numeric-source-only'}))
  plan_pin = write(root/'plan.json',dict(version=audit.VERSION,groups=list(audit.GROUPS),
      evaluation_seeds=list(audit.SEEDS),primary='A_minus_B',transitions_per_episode=1000,
      actual_training_updates_per_new_arm=128,old_B_retrained=False,
      additional_F_measurements_performed=False,independent_signal_identified=False,selective_gate_validated=False))
  states = {'0':audit.INITIAL_STATE,'B':audit.B_STATE,'A':'c'*64}
  freeze_pin = write(root/'freeze.json',dict(version=audit.VERSION,actual_git_sha=git,
      groups=list(audit.GROUPS),evaluation_seeds=list(audit.SEEDS),transitions_per_episode=1000,
      expected_physics_transitions=96000,primary='A_minus_B',bootstrap_seed=20261026,bootstrap_draws=2000,
      root_definition='default_rng([0,0x4556414C,environment_seed,step]).uint32_key2',
      new_evaluation_started=False,additional_F_measurements_performed=False,old_B_authenticated_reuse=True,
      old_B_retrained=False,native_actor_action='official_sample',monotonic_frozen=2.,
      old_authentication_sha256=auth_pin,full_state_digests=states,
      complete_checkpoint_sha256={'0':override['INITIAL_CHECKPOINT'],'B':override['B_CHECKPOINT'],'A':a_checkpoint_pin},
      files_sha256={'budget.json':budget_pin,'plan.json':plan_pin,'old-authenticated.json':auth_pin,
          'A-complete.json':a_pin,'A-checkpoint.pkl':a_checkpoint_pin}))
  seal_pin = write(root/'freeze-seal-parent.json',dict(version=audit.VERSION,actual_git_sha=git,
      freeze_sha256=freeze_pin,monotonic_sealed=3.,new_evaluation_started=False))
  manifest, eval_pins = [], {}
  for group in audit.GROUPS:
    rows = []
    for seed in audit.SEEDS:
      filename = f'{group}-episode-{seed}.pkl'
      (root/filename).write_bytes(b'synthetic-record-byte-pin-only')
      rows.append(dict(group=group,seed=seed,file=filename,sha256=audit._sha(root/filename),return_f64=1000.))
    manifest.extend(rows)
    native = dict(version=audit.VERSION,current_full_state_sha256=states[group],
        currentparams_identity_unchanged=True,scientific_policy_reads_currentparams=True,
        readonly_full_state_and_counters_unchanged=True,public_policy_copy_used_for_scientific_calls=False,
        source_physics_renderer_resource_authenticated_by_this_helper=False,private_initial_calls=32,
        private_policy_calls=32000,public_oracle_calls=96,accessed_parameter_keys=['fixture-only'],
        attempted_model_writes_rejected=True,attempted_model_creation_rejected=True,
        synchronized_policy_copy_sha256='d'*64)
    eval_pins[group] = write(root/f'eval{group}-complete.json',dict(version=audit.VERSION,
        actual_git_sha=git,group=group,phase='eval'+group,freeze_sha256=freeze_pin,
        actual_physics_transitions=32000,actual_RGB_frames=32032,native_model_calls=32000,
        production_oracle_calls=96,additional_F_measurements_performed=False,
        complete_normal_posterior_feedback=True,full_state_digest_before_and_after=states[group],
        native_interface_receipt=native,device=device(True),records=rows,
        reset_receipts=[dict(seed=seed,rendering_full_state_byte_exact=True,
            initial_snapshot_sha256='e'*64,initial_rgb_sha256='f'*64) for seed in audit.SEEDS]))
  result = dict(version=audit.VERSION,actual_git_sha=git,actual_physics_transitions=96000,
      actual_RGB_frames=96096,actual_training_updates=128,original_B_retrained=False,
      original_inputs_unchanged=True,independent_signal_identified=False,selective_gate_validated=False,
      ordinary_replay_training_validated=False,method_effectiveness_established=False,
      minimal_practical_effect_declared=False,full_goal_complete=False,freeze_sha256=freeze_pin,
      freeze_seal_parent_sha256=seal_pin,old_authentication_sha256=auth_pin,
      evaluation_complete_sha256=eval_pins,record_manifest=manifest)
  result_pin = write(root/'result.json',result)
  phases = [dict(phase=phase,worker_pid=101,worker_terminal=True,exit_code=0,timed_out=False,
      elapsed_seconds=1.,is_gpu=phase in ('A','eval0','evalB','evalA')) for phase in audit.PHASES]
  for row in phases: write(root/(row['phase']+'-exit.json'),row)
  write(root/'supervisor_complete.json',dict(version=audit.VERSION,actual_git_sha=git,
      parent_pid=100,phases=phases,exit_code=0,timed_out=False,complete_verified=True,
      actual_gpu_devices_used=1,independent_audit_complete=False,method_effectiveness_established=False,
      elapsed_parent_entry_seconds=8.,budget_sha256=budget_pin,result_sha256=result_pin))
  external = Path(directory)/'external.json'
  write(external,dict(exit_code=0,worker_terminal=True,actual_git_sha=git,worker_pid=100,elapsed_cli_seconds=9.))
  return root, external, git, override


class NativeAuditContracts(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.record = fake_record()

  def test_complete_fake_h1000_physics_and_fsum(self):
    checks = []
    result = audit.replay_record(self.record, FakePhysics(1024), lambda: checks.append(1))
    self.assertEqual(result['physics_steps'], 1000)
    self.assertEqual(result['return_fsum'], 1000.)
    self.assertGreaterEqual(len(checks), 1002)
    self.assertTrue(result['all_state_byte_exact'])
    self.assertFalse(result['model_replayed'])

  def test_early_terminal_wrong_reset_and_missing_final_last_rejected(self):
    for issue in ('earlylast', 'first', 'finallast', 'discount'):
      with self.subTest(issue=issue):
        record = copy.deepcopy(self.record)
        obs = record['trajectory']['observations']
        if issue == 'earlylast': obs['is_last'][3] = True
        elif issue == 'first': obs['is_first'][2] = True
        elif issue == 'finallast': obs['is_last'][-1] = False
        else: record['trajectory']['discount_f64'][5] = 0
        with self.assertRaises(ValueError): audit.validate_record(record)

  def test_dtype_shape_return_and_private_roots_rejected(self):
    for issue in ('dtype', 'shape', 'return', 'root', 'count', 'carry'):
      with self.subTest(issue=issue):
        record = copy.deepcopy(self.record)
        trajectory = record['trajectory']
        if issue == 'dtype': trajectory['reward_f64'] = trajectory['reward_f64'].astype(np.float32)
        elif issue == 'shape': trajectory['raw'] = trajectory['raw'][:-1]
        elif issue == 'return': trajectory['return_f64'] += .1
        elif issue == 'root': trajectory['root_keys'][3, 0] ^= 1
        elif issue == 'count': trajectory['counts']['policy_calls'] += 1
        else: trajectory['carry_hashes'].pop()
        with self.assertRaises(ValueError): audit.validate_record(record)

  def test_source_reset_full_state_and_raw_normalization_mismatch_rejected(self):
    for issue in ('freshreset', 'state', 'task', 'clock', 'stage'):
      with self.subTest(issue=issue):
        record = copy.deepcopy(self.record)
        physics = FakePhysics(1024)
        if issue == 'freshreset': physics.seed += 1
        elif issue == 'state': record['snapshots'][2]['suite']['integration'][1] += 1e-12
        elif issue == 'task': record['snapshots'][2]['suite']['task']['fixture_seed'] += 1
        elif issue == 'clock': record['snapshots'][2]['suite']['step_count'] += 1
        else: record['stages'][1]['unified'][0] = .2
        with self.assertRaises(ValueError): audit.replay_record(record, physics)

  def test_nonzero_reset_reward_and_false_render_receipt_rejected(self):
    for issue in ('reward', 'render', 'snapshot', 'rgb'):
      with self.subTest(issue=issue):
        record = copy.deepcopy(self.record)
        if issue == 'reward': record['trajectory']['observations']['reward'][0] = 1
        else: record['reset_receipt'][{
            'render': 'rendering_full_state_byte_exact', 'snapshot': 'initial_snapshot_sha256',
            'rgb': 'initial_rgb_sha256'}[issue]] = False if issue == 'render' else '0'*64
        with self.assertRaises(ValueError): audit.validate_record(record)

  def test_fsum_whole_ep_paired_bootstrap_uses_all_signed_values(self):
    returns = {'0': [10.]*32, 'B': [20.]*32,
        'A': [21.]*31 + [0.]}
    statistics = audit.independent_statistics(returns)
    primary = statistics['comparisons']['A_minus_B']
    self.assertEqual(primary['values'], [1.]*31 + [-20.])
    self.assertEqual(primary['mean'], 11/32)
    draws = np.random.default_rng(20261026).integers(0, 32, (2000, 32))
    direct = [math.fsum(primary['values'][int(i)] for i in row)/32 for row in draws]
    self.assertEqual(primary['ci95'], np.quantile(direct, [.025, .975], method='linear').tolist())
    self.assertEqual(primary['classification'], 'inconclusive')
    self.assertFalse(statistics['method_effectiveness_established'])

  def test_positive_zero_negative_and_cross_zero_classifications(self):
    for gain, expected in ((1., 'positive_finite_net_effect'),
        (0., 'positive_effect_not_supported'), (-1., 'positive_effect_not_supported')):
      statistics = audit.independent_statistics({'0': [0.]*32,
          'B': [0.]*32, 'A': [gain]*32})
      self.assertEqual(statistics['comparisons']['A_minus_B']['classification'], expected)
    statistics = audit.independent_statistics({'0': [0.]*32,
        'B': [0.]*32, 'A': [1., -1.]*16})
    self.assertEqual(statistics['comparisons']['A_minus_B']['classification'], 'inconclusive')

  def test_ci_zero_side_cannot_pass_using_numeric_tolerance(self):
    expected = {'ci95': [1e-12, 2e-12]}
    with self.assertRaisesRegex(ValueError, 'zero-side'):
      audit.compare_statistics(expected, {'ci95': [-1e-12, 2e-12]})
    with self.assertRaises(ValueError):
      audit.compare_statistics({'classification': 'inconclusive'},
          {'classification': 'positive_finite_net_effect'})

  def test_statistic_schema_full_coverage_and_nonfinite_rejected(self):
    for returns in ({'A': [0.]*32, 'B': [0.]*32},
        {'0': [0.]*31, 'B': [0.]*32, 'A': [0.]*32},
        {'0': [0.]*32, 'B': [0.]*32, 'A': [np.nan]*32}):
      with self.assertRaises(ValueError): audit.independent_statistics(returns)

  def test_missing_extra_reordered_and_cross_arm_reset_records_rejected(self):
    def records():
      for group in audit.GROUPS:
        for seed in audit.SEEDS:
          yield dict(group=group, seed=seed, reset_receipt=dict(
              initial_snapshot_sha256=str(seed), initial_rgb_sha256='same'))
    def replay(record, adapter, check):
      return dict(return_fsum=1000., physics_steps=1000)
    stats = audit.independent_statistics({g: [1000.]*32 for g in audit.GROUPS})
    with mock.patch.object(audit, 'replay_record', side_effect=replay):
      result = audit.audit_records(records(), stats, adapter_factory=FakePhysics)
      self.assertEqual(result['physics_steps'], 96000)
      for stream in (list(records())[:-1], list(records()) + [dict(group='A', seed=1055)],
          [dict(group='B', seed=1024)] + list(records())[1:]):
        with self.assertRaises(ValueError):
          audit.audit_records(stream, stats, adapter_factory=FakePhysics)
      rows = list(records())
      rows[33]['reset_receipt']['initial_rgb_sha256'] = 'different'
      with self.assertRaisesRegex(ValueError, 'Cross-arm'):
        audit.audit_records(rows, stats, adapter_factory=FakePhysics)

  def test_same_payload_hash_before_decode_and_safe_new_record_path(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)/'record.pkl'
      payload = pickle.dumps({'fake': 1}, protocol=5)
      path.write_bytes(payload)
      self.assertEqual(audit.load_record(directory, path.name,
          hashlib.sha256(payload).hexdigest()), {'fake': 1})
      with mock.patch.object(audit.pickle, 'load') as decode:
        with self.assertRaises(ValueError): audit.load_record(directory, path.name, '0'*64)
        decode.assert_not_called()
      with self.assertRaises(ValueError): audit.load_record(directory, '../record.pkl', '0'*64)

  def test_full_fake_filesystem_binding_no_model_or_old_source_decode(self):
    with tempfile.TemporaryDirectory() as directory:
      root,external,git,constants = fake_envelope(directory)
      with contextlib.ExitStack() as stack:
        for name,value in constants.items(): stack.enter_context(mock.patch.object(audit,name,value))
        stack.enter_context(mock.patch.object(audit,'_code_binding',return_value={'audit-only':'fixture'}))
        decode = stack.enter_context(mock.patch.object(audit.pickle,'load'))
        authority = audit.authenticate_primary(root,external,git)
        self.assertEqual(authority['primary_git'],git)
        self.assertEqual(len(authority['result']['record_manifest']),96)
        self.assertEqual(authority['pressure']['active_count'],8192)
        self.assertFalse(authority['pressure']['actual_nonzero_rep_gradient_or_information_deletion_independently_proved'])
        decode.assert_not_called()

  def test_full_fake_filesystem_tampered_exit_seal_model_and_counts_rejected(self):
    # Every mutation leaves the original parent/complete pins intact; failures
    # must not be patched by regenerating those scientific endpoints.
    for issue in ('external_exit','late','seal','Acheckpoint','record','pressure','state'):
      with self.subTest(issue=issue),tempfile.TemporaryDirectory() as directory:
        root,external,git,constants = fake_envelope(directory)
        if issue == 'external_exit':
          value = audit._read_json(external);value['exit_code']=1;external.write_text(__import__('json').dumps(value))
        elif issue == 'late': (root/'late-supervisor.json').write_text('{}')
        elif issue == 'seal': (root/'freeze-seal-parent.json').write_text('{}')
        elif issue == 'Acheckpoint': (root/'A-checkpoint.pkl').write_bytes(b'different')
        elif issue == 'record': (root/'0-episode-1024.pkl').write_bytes(b'different')
        elif issue == 'pressure': (root/'A-complete.json').write_text('{}')
        else: (root/'evalB-complete.json').write_text('{}')
        with contextlib.ExitStack() as stack:
          for name,value in constants.items(): stack.enter_context(mock.patch.object(audit,name,value))
          stack.enter_context(mock.patch.object(audit,'_code_binding',return_value={}))
          with self.assertRaises(ValueError): audit.authenticate_primary(root,external,git)

  def test_pressure_independent_partition_fsum_and_zero_source_limit(self):
    rows=[dict(reset_pressure=dict(active_count=0,reset_count=16,excess_sum=0.),
        whole_window_pressure=dict(active_count=0,position_count=1024,nonreset_active_count=0,
            excess_sum=0.,nonreset_excess_sum=0.)) for _ in range(128)]
    learning=dict(receipts=rows,total_loss_positions=131072,total_reset_loss_positions=2048,
        reset_pressure=dict(active_count=0,reset_count=2048,excess_sum=0.,entire_128_update_metrics_observed=True),
        whole_window_pressure=dict(active_count=0,position_count=131072,nonreset_active_count=0,
            excess_sum=0.,nonreset_excess_sum=0.,entire_128_update_metrics_observed=True))
    result=audit._pressure_receipts(learning)
    self.assertEqual(result['active_count'],0)
    self.assertFalse(result['actual_nonzero_rep_gradient_or_information_deletion_independently_proved'])
    changed=copy.deepcopy(learning);changed['whole_window_pressure']['position_count']+=1
    with self.assertRaises(ValueError): audit._pressure_receipts(changed)
    changed=copy.deepcopy(learning);changed['receipts'][3]['whole_window_pressure']['excess_sum']=1.
    with self.assertRaises(ValueError): audit._pressure_receipts(changed)

  def test_real_owned_child_success_failure_timeout_and_late_creation(self):
    with tempfile.TemporaryDirectory() as directory:
      for body,expected in (('print(1)',0),('raise SystemExit(7)',7),
          ('import time;time.sleep(10)',124)):
        with (Path(directory)/('worker'+str(expected)+'.log')).open('w') as log:
          result=audit._run_owned([sys.executable,'-c',body],log,
              time.monotonic()+(.15 if expected==124 else 10),dict(os.environ))
        self.assertEqual(result['exit_code'],expected)
        self.assertTrue(result['worker_terminal'])
      with mock.patch.object(audit.subprocess,'Popen') as create:
        with self.assertRaises(ValueError):audit._run_owned(['not-created'],None,time.monotonic()-1,{})
        create.assert_not_called()
      self.assertFalse(audit.OWNED)

  def test_late_cpu_parent_write_or_rename_never_publishes_complete(self):
    for late_at in ('write','rename'):
      with self.subTest(late_at=late_at),tempfile.TemporaryDirectory() as directory:
        output=Path(directory)/'audit';clock=[0.]
        args=types.SimpleNamespace(output=str(output),expected_git='a'*40,
            primary='synthetic-only',external_cli_terminal='synthetic-only')
        write_original,replace_original=audit._write_json,audit.os.replace
        def run(*unused):
          write_original(output/'audit.json',dict(version=audit.IMPLEMENTATION,status='passed',passed=True,
              auditor_git_sha=args.expected_git,episodes=96,physics_steps=96000,actual_gpu_devices_used=0))
          return dict(worker_pid=101,worker_terminal=True,exit_code=0,timed_out=False,
              elapsed_seconds=.01,owned_group_cleanup_requested=True)
        def write(path,value):
          write_original(path,value)
          if late_at=='write' and Path(path).name=='supervisor-receipt-pending.json':clock[0]=2.
        def replace(source,target):
          replace_original(source,target)
          if late_at=='rename' and Path(target).name=='supervisor_complete.json':clock[0]=2.
        original_require=audit.require
        def fixture_require(condition,message):
          if message != 'Owned whole-entry CPU audit requires POSIX process groups':
            original_require(condition,message)
        with mock.patch.object(audit,'require',side_effect=fixture_require),mock.patch.object(audit,'ENTRY_STARTED',0.),\
            mock.patch.object(audit,'SECONDS',1),mock.patch.object(audit.time,'monotonic',side_effect=lambda:clock[0]),\
            mock.patch.object(audit,'_run_owned',side_effect=run),\
            mock.patch.object(audit,'_write_json',side_effect=write),mock.patch.object(audit.os,'replace',side_effect=replace):
          self.assertEqual(audit._parent(args),124)
        self.assertFalse((output/'supervisor_complete.json').exists())
        self.assertTrue((output/'late-supervisor.json').exists())
        with self.assertRaises(ValueError):audit.verify_completion(output)

  def test_published_cpu_parent_requires_full_external_exit_and_audit_pin(self):
    with tempfile.TemporaryDirectory() as directory:
      output=Path(directory)/'audit';clock=[.1]
      args=types.SimpleNamespace(output=str(output),expected_git='a'*40,
          primary='synthetic-only',external_cli_terminal='synthetic-only')
      original_require=audit.require
      def fixture_require(condition,message):
        if message != 'Owned whole-entry CPU audit requires POSIX process groups':
          original_require(condition,message)
      def run(*unused):
        audit._write_json(output/'audit.json',dict(version=audit.IMPLEMENTATION,status='passed',passed=True,
            auditor_git_sha=args.expected_git,episodes=96,physics_steps=96000,actual_gpu_devices_used=0))
        clock[0]=.2
        return dict(worker_pid=101,worker_terminal=True,exit_code=0,timed_out=False,
            elapsed_seconds=.1,owned_group_cleanup_requested=True)
      with mock.patch.object(audit,'require',side_effect=fixture_require),\
          mock.patch.object(audit,'ENTRY_STARTED',0.),\
          mock.patch.object(audit.time,'monotonic',side_effect=lambda:clock[0]),\
          mock.patch.object(audit,'_run_owned',side_effect=run):
        self.assertEqual(audit._parent(args),0)
      self.assertFalse(audit.verify_completion(output)['complete_external_exit_verified'])
      receipt=audit._read_json(output/'supervisor_complete.json')
      external=Path(directory)/'external.json'
      audit._write_json(external,dict(actual_git_sha=args.expected_git,exit_code=0,worker_terminal=True,
          worker_pid=receipt['parent_pid'],elapsed_cli_seconds=.3))
      self.assertTrue(audit.verify_completion(output,external)['complete_external_exit_verified'])
      value=audit._read_json(external);value['exit_code']=124
      external.write_text(__import__('json').dumps(value))
      with self.assertRaises(ValueError):audit.verify_completion(output,external)
      (output/'audit.json').write_text('{}')
      with self.assertRaises(ValueError):audit.verify_completion(output)


if __name__ == '__main__':
  unittest.main()
