"""DL-code-r1 runner flow fixture: real Driver/Checkpoint, no model/render/GPU.

Exercise dl_train.run itself, not a duplicate counting loop. The agent/replay
are controlled doubles; evidence covers orchestration and failure guards only.
"""

from contextlib import ExitStack
import datetime as dt
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import elements
import embodied
import numpy as np
from ruamel.yaml import YAML

from embodied.envs.dummy_cont import DummyCont

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/dl_train.py'
SPEC = importlib.util.spec_from_file_location('dl_train_flow_fixture', SOURCE)
dl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dl)


class AgentDouble:

  def __init__(self, corrupt_evaluation=False, nonfinite_train=False):
    self.n_updates = elements.Counter()
    self.n_actions = elements.Counter()
    self.n_batches = elements.Counter()
    self.eval_actions = 0
    self.eval_seed = 0
    self.pending = False
    self.params = {'model/weight': np.array([1.], np.float32)}
    self.corrupt_evaluation = corrupt_evaluation
    self.nonfinite_train = nonfinite_train

  def init_train(self, count):
    return count

  def init_policy(self, count, mode='train'):
    return ([0] * count,)

  def prepare_batch(self, data):
    self.n_batches.increment()
    return dict(data, seed=int(self.n_batches))

  def train(self, carry, data):
    assert 'seed' in data and not self.pending
    self.pending = True
    self.n_updates.increment()
    self.params['model/weight'] += np.float32(.001)
    return carry, {}, {}

  def take_train_result(self):
    assert self.pending
    self.pending = False
    return {}, {'opt/updates': np.float32(int(self.n_updates)),
        'opt/grad_norm': np.float32(np.nan if self.nonfinite_train else 1.),
        'opt/loss': np.float32(2.), 'dl/release': np.float32(.01)}

  def sync_policy(self):
    pass

  def set_eval_seed(self, seed):
    self.eval_seed, self.eval_actions = seed, 0

  def policy(self, carry, obs, mode='train'):
    if mode == 'train' or self.corrupt_evaluation:
      self.n_actions.increment()
    if mode == 'eval':
      self.eval_actions += 1
    return ([x + 1 for x in carry[0]],), {
        'action': np.full((len(obs['is_first']), 2), .2, np.float32)}, {}

  def save(self):
    return {'params': {k: v.copy() for k, v in self.params.items()},
        'counters': {'updates': int(self.n_updates),
            'batches': int(self.n_batches), 'actions': int(self.n_actions),
            'eval_actions': self.eval_actions, 'eval_seed': self.eval_seed}}

  def load(self, data):
    raise AssertionError('Runner must never resume a fixture attempt')


class ReplayDouble:

  def __init__(self):
    self.added = 0
    self.metrics = {}
    self.sampler = types.SimpleNamespace(rng=np.random.default_rng(4))

  def add(self, tran, worker):
    self.added += 1

  def __len__(self):
    # Preserve initial prefill and then make the production 1024 threshold pass.
    return max(0, self.added - 30) * 50

  def save(self):
    return {'added': self.added}

  def load(self, data):
    raise AssertionError('Runner must never resume a fixture attempt')

  def update(self, data):
    raise AssertionError('This double does not return replay updates')


class LoggerDouble:

  def __init__(self):
    self.step = elements.Counter()
    self.closed = False

  def add(self, *args, **kwargs):
    pass

  def write(self):
    pass

  def close(self):
    self.closed = True


def report_fixture():
  return {'calibration_episodes': 8, 'blind_episodes': 24,
      'screening': {'eligible_for_short_training_review': True},
      'score_plan': {'task': 'dmc_quadruped_walk', 'sample_count': 32,
          'horizon': 10, 'alpha': 20, 'rho': .5, 'free_nats': 1,
          'checkpoint_sha256': 'fixture-not-real',
          'checkpoint_counters': {'updates': 1}},
      'score_complete_sha256': 'fixture-not-real',
      'calibration_only': {'kappa': .12, 'reward_only_lambda': .09,
          'd_bins': [.1, .2], 'k_bins': [2.]}}


class DLTrainFlow(unittest.TestCase):

  def exercise(self, directory, *, corrupt_evaluation=False,
      nonfinite_train=False, wrong_resource=False, changed_config=False):
    from dreamerv3 import main
    root = Path(directory)
    logdir = root / 'attempt'
    agent = AgentDouble(corrupt_evaluation, nonfinite_train)
    replay, logger = ReplayDouble(), LoggerDouble()
    created, drivers = [], []
    original_driver = embodied.Driver
    original_summary = dl.summary
    original_checkpoint = elements.Checkpoint

    def driver(fns, parallel=True):
      # Use the actual Driver control flow, serially to avoid worker/render work.
      result = original_driver(fns, parallel=False)
      drivers.append(result)
      return result

    def make_agent(config):
      created.append(config)
      return agent

    def make_env(config, index, seed=0):
      return DummyCont('fixture', length=3 + index % 2, seed=seed)

    class FixturePath(type(Path())):
      def as_posix(self):
        # Only adapt the Linux artifact-root check to this explicit temp fixture.
        if Path(self) == logdir:
          return '/data/Policy_Discrepancy/runs/DL-flow-fixture'
        return super().as_posix()

    with ExitStack() as stack:
      for name, value in [('BUDGET', 80), ('INTERVAL', 20), ('EPISODES', 2), ('ENVS', 3)]:
        stack.enter_context(patch.object(dl, name, value))
      # Python binds the production default budget at function definition time.
      stack.enter_context(patch.object(dl, 'summary',
          side_effect=lambda evaluations: original_summary(evaluations, budget=80)))
      stack.enter_context(patch.object(dl, 'Path', FixturePath))
      stack.enter_context(patch.object(dl, 'git_state', return_value=('fixture-SHA', False)))
      # Explicit test double for live Linux/NVIDIA/EGL evidence. This fixture
      # validates runner orchestration only and does not validate GPU resources.
      stack.enter_context(patch('scripts.dl_resources.require_resource',
          side_effect=lambda path: json.loads(Path(path).read_text())))
      stack.enter_context(patch.object(main, 'make_agent', side_effect=make_agent))
      stack.enter_context(patch.object(main, 'make_replay', return_value=replay))
      stack.enter_context(patch.object(main, 'make_logger', return_value=logger))
      stack.enter_context(patch.object(main, 'make_env', side_effect=make_env))
      stack.enter_context(patch.object(main, 'make_stream',
          side_effect=lambda *args: iter(lambda: {'fixture': np.zeros((1,), np.float32)}, None)))
      stack.enter_context(patch.object(embodied, 'Driver', side_effect=driver))
      # elements LocalPath cleanup assumes POSIX basename separators. Preserve
      # fixture snapshots on Windows; production Linux cleanup is not covered.
      stack.enter_context(patch.object(elements, 'Checkpoint',
          side_effect=lambda directory: original_checkpoint(directory, keep=None)))
      now = dt.datetime.now(dt.timezone.utc)
      resource = dict(logical_server='sv3', physical_gpu=3 if wrong_resource else 4,
          exclusive=True, occupancy='idle', checked_at_utc=now.isoformat(),
          gpu_uuid='GPU-fixture', compute_uuid='GPU-fixture', egl_uuid='GPU-fixture',
          cuda_visible_devices='4', mujoco_egl_device_id='4')
      stack.enter_context(patch.dict('os.environ', {
          'CUDA_VISIBLE_DEVICES': '4', 'MUJOCO_EGL_DEVICE_ID': '4'}))
      report = root / 'stage-c.json'
      scored = root / 'score-artifacts'
      scored.mkdir()
      report_data = report_fixture()
      dl.write_json(scored / 'score_plan.json', report_data['score_plan'])
      scored_entries = []
      for episode in range(32):
        artifact = scored / f'episode_{episode}.npz'
        np.savez(artifact, fixture=np.array([episode]))
        scored_entries.append(dict(episode=episode,
            split='calibration' if episode < 8 else 'blind',
            path=artifact.name, sha256=dl.digest(artifact)))
      dl.write_json(scored / 'score_complete.json', dict(
          plan_sha256=dl.digest(scored / 'score_plan.json'), artifacts=scored_entries))
      report_data['score_complete_sha256'] = dl.digest(scored / 'score_complete.json')
      dl.write_json(report, report_data)
      acceptance = root / 'acceptance.json'
      dl.write_json(acceptance, dict(protocol='DL-protocol-r1', git_sha='fixture-SHA',
          stage_c_report_sha256=dl.digest(report), evidence_paths=['fixture-not-real'],
          real_size50m_engineering_accepted=True, segmentation_visual_accepted=True,
          stage_c_joint_accepted=True, training_control_design_accepted=True))
      record = root / 'resource.json'
      dl.write_json(record, resource)
      plan = root / 'plan'
      dl.prepare(types.SimpleNamespace(output=plan, run_root=root, stage_c=report,
          platform='cuda'))
      index = json.loads((plan / 'RUN_INDEX.json').read_text())
      chosen = index['runs'][0]
      # Bind B-seed0 to the temporary root, retaining all other production fields.
      config_path = plan / chosen['config_file']
      yaml = YAML(typ='safe')
      config = yaml.load(config_path.read_text())
      config['logdir'] = str(logdir)
      if changed_config:
        config['agent.loss_scales.rep'] = .2
      with config_path.open('w') as stream:
        yaml.dump(config, stream)
      chosen['logdir'], chosen['config_sha256'] = str(logdir), dl.digest(config_path)
      (plan / 'RUN_INDEX.json').write_text(json.dumps(index))
      args = types.SimpleNamespace(index=plan / 'RUN_INDEX.json', run_id='B-seed0',
          stage_c=report, acceptance=acceptance, resource_record=record,
          score_artifacts=scored)
      try:
        dl.run(args)
        error = None
      except BaseException as caught:
        error = caught
    return logdir, agent, replay, logger, created, drivers, error

  def test_complete_actual_run_flow_exact_actions_returns_snapshots_and_receipts(self):
    with tempfile.TemporaryDirectory(prefix='DL-flow-') as directory:
      logdir, agent, replay, logger, created, drivers, error = self.exercise(directory)
      if error:
        raise error
      final = json.loads((logdir / 'final_state.json').read_text())
      self.assertEqual(final['train_action_steps'], 80)
      self.assertEqual(int(logger.step), 80)
      self.assertEqual(final['evaluations'], 5)
      self.assertEqual(final['eval_action_steps'], 30)
      self.assertGreater(final['train_resets'], 3)
      self.assertGreater(final['updates'], 0)
      self.assertEqual(final['updates'], int(agent.n_updates))
      evaluations = [json.loads(x) for x in (logdir / 'evaluations.jsonl').read_text().splitlines()]
      self.assertEqual([x['action_step'] for x in evaluations], [0, 20, 40, 60, 80])
      for result in evaluations:
        self.assertEqual(len(result['episodes']), 2)
        self.assertTrue(all(x['complete'] and x['length'] == 3 for x in result['episodes']))
        self.assertTrue(all((logdir / result['snapshot_id'] / 'agent.pkl').is_file()
            for _ in result['episodes']))
      audits = [json.loads(x) for x in (logdir / 'evaluation_state_audit.jsonl').read_text().splitlines()]
      self.assertTrue(all(x['before'] == x['after'] for x in audits))
      receipts = [json.loads(x) for x in (logdir / 'update_receipts.jsonl').read_text().splitlines()]
      self.assertEqual([x['update_id'] for x in receipts], list(range(final['updates'])))
      self.assertTrue(logger.closed)
      self.assertEqual(len(created), 1)
      self.assertEqual(len(drivers), 11)

  def test_evaluation_training_rng_mutation_stops_actual_flow(self):
    with tempfile.TemporaryDirectory(prefix='DL-flow-') as directory:
      logdir, agent, _, logger, _, _, error = self.exercise(directory, corrupt_evaluation=True)
      self.assertIsInstance(error, RuntimeError)
      self.assertIn('evaluation changed training', str(error))
      failure = json.loads((logdir / 'FAILED.json').read_text())
      self.assertEqual(failure['train_action_steps'], 0)
      self.assertFalse((logdir / 'final_state.json').exists())
      self.assertTrue(logger.closed)

  def test_nonfinite_optimizer_result_stops_without_valid_receipt(self):
    with tempfile.TemporaryDirectory(prefix='DL-flow-') as directory:
      logdir, agent, _, logger, _, _, error = self.exercise(directory, nonfinite_train=True)
      self.assertIsInstance(error, ValueError)
      self.assertIn('Invalid matching/gradient receipt', str(error))
      failure = json.loads((logdir / 'FAILED.json').read_text())
      self.assertEqual(failure['updates'], 0)
      self.assertEqual(int(agent.n_updates), 1)
      self.assertFalse((logdir / 'final_state.json').exists())
      self.assertFalse((logdir / 'update_receipts.jsonl').exists())
      self.assertTrue(logger.closed)

  def test_invalid_resource_never_constructs_agent_or_attempt(self):
    with tempfile.TemporaryDirectory(prefix='DL-flow-') as directory:
      logdir, _, _, _, created, drivers, error = self.exercise(directory, wrong_resource=True)
      self.assertIsInstance(error, ValueError)
      self.assertEqual(created, [])
      self.assertEqual(drivers, [])
      self.assertFalse(logdir.exists())

  def test_changed_scientific_config_never_constructs_agent_or_attempt(self):
    with tempfile.TemporaryDirectory(prefix='DL-flow-') as directory:
      logdir, _, _, _, created, drivers, error = self.exercise(directory, changed_config=True)
      self.assertIsInstance(error, ValueError)
      self.assertIn('scientific configuration changed', str(error))
      self.assertEqual(created, [])
      self.assertEqual(drivers, [])
      self.assertFalse(logdir.exists())


if __name__ == '__main__':
  unittest.main()
