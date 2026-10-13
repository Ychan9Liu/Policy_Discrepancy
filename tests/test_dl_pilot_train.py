"""CPU fixtures for the new pilot orchestration, not scientific validation."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
import elements
import embodied
import numpy as np
from embodied.envs.dummy_cont import DummyCont
from scripts import dl_pilot_train as pilot
from scripts.prepare_m2_v1 import resolve
from test_dl_train_flow import AgentDouble, ReplayDouble, LoggerDouble


def baseline_config():
  config = resolve(['--configs', 'm2_v1', '--task', pilot.TASK,
      '--jax.prealloc', 'False']).update({
          'run.eval_every_actions': pilot.INTERVAL, 'run.eval_eps': pilot.EPISODES,
          'run.envs': pilot.ENVS})
  return config, {k: pilot.safe_json(v) for k, v in config.flat.items()
      if not k.startswith('agent.dt_latch.')}


class PilotContract(unittest.TestCase):

  def test_declared_differences_and_unapproved_method_change(self):
    config, baseline = baseline_config()
    config = config.update(pilot.overrides('/data/Policy_Discrepancy/runs/dl-pilot-r2-fixture/N'))
    differences = pilot.validate_config(config, baseline)
    self.assertIn('run.action_budget', differences)
    self.assertIn('agent.rep_probe.mode', differences)
    self.assertNotIn('run.eval_every_actions', differences)
    for change in ({'agent.dt_latch.rho': .2}, {'agent.loss_scales.rep': .2},
        {'run.train_ratio': 128}, {'agent.dt_latch.kappa': 0.}):
      with self.subTest(change=change), self.assertRaises(ValueError):
        pilot.validate_config(config.update(change), baseline)

  def test_historical_rng_namespaces_and_complete_episodes(self):
    from embodied.run.protocol_v1 import seed32, REWARD_SOURCE, RETURN_DEFINITION
    rows = []
    for point in (0, 50000, 100000):
      scores = [float(i + point / 10000) for i in range(10)]
      seeds = [seed32(0, pilot.TASK, 'eval_env', point // 50000, i) for i in range(10)]
      details = [dict(complete=True, episode_id=i, environment_seed=seeds[i],
          length=1000, return_raw=scores[i]) for i in range(10)]
      rows.append(dict(action_step=point, target_action_step=point, snapshot_action_step=point,
          snapshot_update_id=0, update_id=0, scores=scores, returns_raw=scores, seeds=seeds,
          lengths=[1000]*10, episodes=details, mean=float(np.mean(scores)),
          mean_return_raw=float(np.mean(scores)), reward_source=REWARD_SOURCE,
          return_definition=RETURN_DEFINITION))
    result = pilot.summarize(rows)
    self.assertEqual(result['return_auc_per_100k_actions'], 9.5)
    self.assertEqual(result['endpoint_return_raw'], 14.5)
    bad = json.loads(json.dumps(rows))
    bad[1]['seeds'][0] = seed32(0, pilot.TASK, 'DL_eval_env', 1, 0)
    with self.assertRaisesRegex(ValueError, 'RNG mapping'):
      pilot.summarize(bad)
    bad = json.loads(json.dumps(rows))
    bad[2]['episodes'][0]['complete'] = False
    with self.assertRaisesRegex(ValueError, 'Episode detail'):
      pilot.summarize(bad)

  @unittest.skipUnless(os.name == 'posix', 'Actual Linux owned process group required')
  def test_timeout_kills_owned_child_and_nested_process(self):
    with tempfile.TemporaryDirectory() as directory:
      pidfile = Path(directory) / 'pid'
      # A real child and nested helper share the owned session. No GPU work.
      script = ('import subprocess,sys,time,pathlib; '
          "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
          f'pathlib.Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(30)')
      with self.assertRaisesRegex(ValueError, 'deadline'):
        pilot.supervise_child([sys.executable, '-c', script], Path(directory) / 'stdout',
            deadline=time.monotonic() + 1, environment=dict(os.environ))
      self.assertEqual(pilot.OWNED, set())
      import psutil
      pid = int(pidfile.read_text())
      try:
        process = psutil.Process(pid)
        self.assertTrue(not process.is_running() or process.status() == psutil.STATUS_ZOMBIE)
      except psutil.NoSuchProcess:
        pass

  def exercise(self, root, *, corrupt_eval=False, nonfinite=False):
    from dreamerv3 import main
    logdir, plan = root / 'attempt', root / 'plan'
    plan.mkdir()
    agent = AgentDouble(corrupt_eval, nonfinite)
    replay, logger = ReplayDouble(), LoggerDouble()
    real_driver, real_checkpoint = embodied.Driver, elements.Checkpoint
    made, closed_drivers = [], []
    from embodied.run.protocol_v1 import seed32
    train_seeds = {seed32(0, pilot.TASK, 'train_env', i) for i in range(3)}

    def driver(fns, parallel=True):
      result = real_driver(fns, parallel=False)
      closed_drivers.append(result)
      return result

    def environment(config, index, seed=0):
      return DummyCont('fixture', length=(3 + index % 2) if seed in train_seeds else 1000, seed=seed)

    def make_agent(config):
      made.append(config)
      return agent

    with ExitStack() as stack:
      for name, value in [('BUDGET', 80), ('INTERVAL', 40), ('EPISODES', 2), ('ENVS', 3),
          ('RUNS_ROOT', root)]:
        stack.enter_context(patch.object(pilot, name, value))
      config, baseline = baseline_config()
      config = config.update(pilot.overrides(logdir))
      config.save(str(plan / 'config.yaml'))
      pilot.write_json(plan / 'historical_controls.json', {'B': {'manifest': {'config': baseline}}})
      index = dict(logdir=str(logdir), config_sha256=pilot.digest(plan / 'config.yaml'),
          controls_sha256=pilot.digest(plan / 'historical_controls.json'), limitation='fixture only')
      pilot.write_json(plan / 'RUN_INDEX.json', index)
      stack.enter_context(patch.object(pilot, 'load_plan', return_value=index))
      stack.enter_context(patch.object(pilot, 'worker_binding', return_value=(root, 4)))
      stack.enter_context(patch.object(pilot, 'worker_environment'))
      stack.enter_context(patch('embodied.run.protocol_v1.parameter_digest', return_value=pilot.INITIAL_PARAMS))
      stack.enter_context(patch.object(main, 'make_agent', side_effect=make_agent))
      stack.enter_context(patch.object(main, 'make_replay', return_value=replay))
      stack.enter_context(patch.object(main, 'make_logger', return_value=logger))
      stack.enter_context(patch.object(main, 'make_env', side_effect=environment))
      stack.enter_context(patch.object(main, 'make_stream', side_effect=lambda *xs:
          iter(lambda: {'fixture': np.zeros((1,), np.float32)}, None)))
      stack.enter_context(patch.object(embodied, 'Driver', side_effect=driver))
      stack.enter_context(patch.object(elements, 'Checkpoint', side_effect=lambda path:
          real_checkpoint(path, keep=None)))
      # Snapshot binding is a physical-runtime check and explicitly doubled.
      stack.enter_context(patch.dict(os.environ, {'MUJOCO_EGL_DEVICE_ID': '4'}))
      pilot.write_json(root / 'startup.json', {'fixture_only': True})
      try:
        pilot.worker(types.SimpleNamespace(index=plan / 'RUN_INDEX.json', git_sha='fixture', supervisor=root))
        error = None
      except BaseException as caught:
        error = caught
    return logdir, agent, logger, made, closed_drivers, error

  def test_actual_worker_flow_exact_budget_and_evaluation_isolation(self):
    with tempfile.TemporaryDirectory() as directory:
      logdir, agent, logger, made, drivers, error = self.exercise(Path(directory))
      if error:
        raise error
      final = pilot.read_json(logdir / 'final_state.json')
      self.assertEqual(final['train_action_steps'], 80)
      self.assertEqual(final['eval_action_steps'], 6000)
      self.assertEqual(final['updates'], int(agent.n_updates))
      self.assertGreater(final['updates'], 0)
      self.assertGreater(final['train_resets'], 3)
      audits = [json.loads(x) for x in (logdir / 'evaluation_state_audit.jsonl').read_text().splitlines()]
      self.assertEqual([x['action_step'] for x in audits], [0, 40, 80])
      self.assertTrue(all(x['before'] == x['after'] for x in audits))
      self.assertEqual(len(made), 1)
      self.assertEqual(len(drivers), 7)
      self.assertTrue(logger.closed)

  def test_actual_worker_rejects_nonfinite_update_and_eval_rng_mutation(self):
    for options, expected in [({'corrupt_eval': True}, 'evaluation changed training'),
        ({'nonfinite': True}, 'Invalid matching/gradient receipt')]:
      with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
        logdir, _, logger, _, _, error = self.exercise(Path(directory), **options)
        self.assertIsNotNone(error)
        self.assertIn(expected, str(error))
        self.assertFalse((logdir / 'final_state.json').exists())
        self.assertTrue((logdir / 'FAILED.json').exists())
        self.assertTrue(logger.closed)


if __name__ == '__main__':
  unittest.main()
