"""DL-code-r1 planning/contracts only; no checkpoint, GPU, or method evidence."""

import datetime as dt
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/dl_train.py'
SPEC = importlib.util.spec_from_file_location('dl_train_fixture', SOURCE)
dl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dl)


def report_fixture():
  # This is a synthetic schema fixture, never execution authorization/evidence.
  return dict(calibration_episodes=8, blind_episodes=24,
      screening={'eligible_for_short_training_review': True},
      score_plan=dict(task='dmc_quadruped_walk', sample_count=32,
          horizon=10, alpha=20, rho=.5, free_nats=1,
          checkpoint_sha256='fixture-not-a-checkpoint', checkpoint_counters={'updates': 1}),
      score_complete_sha256='fixture-not-real-evidence',
      calibration_only=dict(kappa=.12, reward_only_lambda=.09,
          d_bins=[.1, .2], k_bins=[2.]))


class DLTrainContracts(unittest.TestCase):

  def test_actual_screening_and_frozen_calibration_required(self):
    with self.assertRaises(ValueError):
      dl.validate_stage_c(None)
    report = report_fixture()
    self.assertEqual(dl.validate_stage_c(report)['reward_only_lambda'], .09)
    report['screening']['eligible_for_short_training_review'] = False
    with self.assertRaises(ValueError):
      dl.validate_stage_c(report)
    report = report_fixture()
    report['score_plan']['checkpoint_counters']['updates'] = 0
    with self.assertRaises(ValueError):
      dl.validate_stage_c(report)
    report = report_fixture()
    report['calibration_only']['kappa'] = None
    with self.assertRaises(ValueError):
      dl.calibration(report)
    report = report_fixture()
    report['calibration_only']['d_bins'] = [.2, .2]
    with self.assertRaises(ValueError):
      dl.calibration(report)

  def test_group_semantics_and_matched_reward_only_coefficient(self):
    frozen = dl.calibration(report_fixture())
    for group in dl.GROUPS:
      values = dl.group_overrides(group, 1, '/fixture/run', frozen)
      self.assertEqual(values['run.action_budget'], 100000)
      self.assertEqual(values['run.envs'], 16)
      self.assertEqual(values['run.eval_every_actions'], 25000)
      self.assertEqual(values['run.eval_eps'], 5)
      self.assertEqual(values['run.from_checkpoint'], '')
    self.assertEqual(dl.group_overrides('D', 0, '/fixture', frozen)[
        'agent.rep_probe.mode'], 'dt')
    self.assertEqual(dl.group_overrides('S', 0, '/fixture', frozen)[
        'agent.dt_latch.kappa'], .12)
    self.assertEqual(dl.group_overrides('W', 0, '/fixture', frozen)[
        'agent.dt_latch.rho'], .09)
    self.assertEqual(dl.group_overrides('N', 0, '/fixture', frozen)[
        'agent.dt_latch.rho'], .5)
    with self.assertRaises(ValueError):
      dl.group_overrides('N', 3, '/fixture', frozen)

  def test_engineering_acceptance_is_bound_to_report_and_actual_sha(self):
    acceptance = dict(protocol='DL-protocol-r1', git_sha='fixture-SHA',
        stage_c_report_sha256='fixture-report-hash', evidence_paths=['fixture-path'],
        real_size50m_engineering_accepted=True, segmentation_visual_accepted=True,
        stage_c_joint_accepted=True, training_control_design_accepted=True)
    dl.validate_acceptance(acceptance, 'fixture-report-hash', 'fixture-SHA')
    with self.assertRaises(ValueError):
      dl.validate_acceptance(acceptance, 'changed-report', 'fixture-SHA')
    with self.assertRaises(ValueError):
      dl.validate_acceptance(acceptance, 'fixture-report-hash', 'changed-code')
    acceptance['real_size50m_engineering_accepted'] = False
    with self.assertRaises(ValueError):
      dl.validate_acceptance(acceptance, 'fixture-report-hash', 'fixture-SHA')

  def test_actual_scored_artifacts_are_required_and_hash_bound(self):
    # Synthetic hash contract only; never presented as real Stage C acceptance.
    with tempfile.TemporaryDirectory() as directory:
      directory = Path(directory)
      plan = {'config': {'logdir': str(directory)}}
      dl.write_json(directory / 'score_plan.json', plan)
      rows = []
      for episode in range(32):
        path = directory / f'fixture{episode}.txt'
        path.write_text(str(episode))
        rows.append(dict(episode=episode, split='calibration' if episode < 8 else 'blind',
            path=path.name, sha256=dl.digest(path)))
      dl.write_json(directory / 'score_complete.json', dict(artifacts=rows,
          plan_sha256=dl.digest(directory / 'score_plan.json')))
      report = dict(score_plan=plan, score_complete_sha256=dl.digest(directory / 'score_complete.json'))
      self.assertEqual(dl.verify_stage_c_artifacts(report), str(directory.resolve()))
      (directory / 'fixture0.txt').write_text('changed')
      with self.assertRaises(ValueError):
        dl.verify_stage_c_artifacts(report)

  def test_resource_scope_occupancy_age_and_compute_render_mapping(self):
    now = dt.datetime(2026, 10, 10, tzinfo=dt.timezone.utc)
    record = dict(logical_server='sv3', physical_gpu=4, exclusive=True,
        occupancy='idle', checked_at_utc=now.isoformat(), gpu_uuid='GPU-fixture',
        compute_uuid='GPU-fixture', egl_uuid='GPU-fixture',
        cuda_visible_devices='4', mujoco_egl_device_id='4')
    environ = dict(CUDA_VISIBLE_DEVICES='4', MUJOCO_EGL_DEVICE_ID='4')
    dl.validate_resource(record, environ, now)
    for key, value in (('physical_gpu', 3), ('occupancy', 'unknown'),
        ('egl_uuid', 'GPU-other'), ('logical_server', 'sv2')):
      changed = dict(record, **{key: value})
      with self.assertRaises(ValueError):
        dl.validate_resource(changed, environ, now)
    with self.assertRaises(ValueError):
      dl.validate_resource(record, environ, now + dt.timedelta(seconds=121))
    with self.assertRaises(ValueError):
      dl.validate_resource(record, dict(CUDA_VISIBLE_DEVICES='5'), now)

  def test_prepare_only_creates_eighteen_plans_and_refuses_overwrite(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / 'plan'
      args = types.SimpleNamespace(output=output, run_root='/fixture/future-runs',
          stage_c=None, platform='cuda')
      with patch.object(dl, 'git_state', return_value=('fixture-SHA', False)):
        dl.prepare(args)
        index = json.loads((output / 'RUN_INDEX.json').read_text())
        self.assertEqual(len(index['runs']), 18)
        self.assertEqual(len({r['run_id'] for r in index['runs']}), 18)
        self.assertEqual(index['evaluation_grid'], [0, 25000, 50000, 75000, 100000])
        self.assertIsNone(index['frozen_calibration'])
        self.assertTrue(all(r['status'] == 'planned_waiting_stage_C' for r in index['runs']))
        self.assertTrue(all(r['logdir'].startswith('/fixture/future-runs/') for r in index['runs']))
        self.assertFalse(any('\\' in r['logdir'] for r in index['runs']))
        self.assertFalse((output / 'ckpt').exists())
        with self.assertRaises(FileExistsError):
          dl.prepare(args)

  def test_prepared_full_size50m_config_validation_and_semantic_mismatch(self):
    import elements
    from ruamel.yaml import YAML
    yaml = YAML(typ='safe')
    presets = yaml.load((SOURCE.parents[1] / 'dreamerv3/configs.yaml').read_text())
    base = elements.Config(presets['defaults']).update(presets['m2_v1'])
    frozen = dl.calibration(report_fixture())
    for group in dl.GROUPS:
      config = elements.Config({**base.flat, **dl.group_overrides(group, 2,
          '/fixture/run', frozen)})
      dl.validate_config(config, group, frozen)
    changed = dict(config.flat)
    changed['agent.dyn.rssm.stoch'], changed['agent.dyn.rssm.classes'] = 64, 16
    with self.assertRaises(ValueError):
      dl.validate_config(changed, 'W', frozen)

  def test_exact_action_count_with_sixteen_workers_and_reset_boundaries(self):
    from embodied.run.protocol_v1 import ProtocolState
    state = ProtocolState(100000, 25000, 5, 0, 100000)
    counters, seen_grid = [0] * 16, [0]
    while state.actions < state.budget:
      boundary = state.next_boundary()
      for worker in range(min(16, boundary - state.actions)):
        reset = counters[worker] % 997 == 0
        counters[worker] += 1
        state.on_train_transition(dict(is_first=reset))
      if state.actions == boundary:
        seen_grid.append(boundary)
    self.assertEqual(state.actions, 100000)
    self.assertGreater(state.resets, 16)
    self.assertEqual(seen_grid, [0, 25000, 50000, 75000, 100000])
    with self.assertRaises(RuntimeError):
      state.on_train_transition(dict(is_first=False))

  def test_short_auc_is_raw_and_incomplete_grid_is_rejected(self):
    values = {x: {'mean': x / 1000} for x in range(0, 100001, 25000)}
    result = dl.summary(values)
    self.assertEqual(result['return_auc_per_action_budget'], 50)
    self.assertEqual(result['endpoint_return_raw'], 100)
    del values[50000]
    with self.assertRaises(ValueError):
      dl.summary(values)


if __name__ == '__main__':
  unittest.main()
