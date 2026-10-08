"""Run small continuous-agent v1 fixtures in isolated processes."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from dreamerv3.freeze_c import freeze


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_dir', required=True)
  args = parser.parse_args()
  root = Path(args.output_dir).resolve()
  root.mkdir(parents=True, exist_ok=True)
  repo = Path(__file__).resolve().parents[1]
  common = [
      sys.executable, '-m', 'dreamerv3.main', '--configs', 'm2_v1', 'debug',
      '--task', 'dummycont_test',
      '--run.engineering_fixture', 'True',
      '--run.action_budget', '12',
      '--run.eval_every_actions', '3',
      '--run.match_start', '3', '--run.match_end', '9',
      '--run.envs', '2', '--run.eval_envs', '1',
      '--run.train_ratio', '4',
      '--batch_size', '2', '--batch_length', '2', '--report_length', '2',
      '--replay_context', '0',
      '--agent.dyn.rssm.free_nats', '0.1',
      '--jax.platform', 'cpu', '--jax.prealloc', 'False']
  variants = [
      ('no-report', 2, 0, 0, False),
      ('with-report', 2, 3, 0, False),
      ('one-eval-episode', 1, 0, 0, False),
      ('parallel-envs', 2, 0, 0, True),
      ('interrupted', 2, 0, 6, False),
  ]
  for name, episodes, report_every, stop_after, parallel in variants:
    directory = root / name
    command = common + [
        '--logdir', str(directory),
        '--run.eval_eps', str(episodes),
        '--run.report_every_actions', str(report_every),
        '--run.engineering_stop_after_actions', str(stop_after),
        '--run.debug', str(not parallel)]
    with (root / f'{name}.log').open('w', encoding='utf-8') as output:
      result = subprocess.run(command, cwd=repo, stdout=output,
                              stderr=subprocess.STDOUT)
    if name == 'interrupted':
      assert result.returncode != 0
      assert 'Intentional fixture stop after checkpoint' in (
          root / f'{name}.log').read_text(encoding='utf-8')
      with (root / 'interrupted-restore.log').open(
          'w', encoding='utf-8') as output:
        resumed = subprocess.run(command, cwd=repo, stdout=output,
                                 stderr=subprocess.STDOUT)
      assert resumed.returncode != 0
      assert 'Training continuation refused' in (
          root / 'interrupted-restore.log').read_text(encoding='utf-8')
    elif result.returncode:
      raise RuntimeError(f'{name} failed: {result.returncode}')
    print(f'{name}: exit={result.returncode}', flush=True)

  compare = repo / 'tests' / 'compare_protocol_runs.py'
  for other in ('with-report', 'one-eval-episode', 'parallel-envs'):
    subprocess.run([sys.executable, str(compare),
        str(root / 'no-report'), str(root / other)], cwd=repo, check=True)
  for name, episodes in (
      ('no-report', 2), ('with-report', 2), ('one-eval-episode', 1),
      ('parallel-envs', 2)):
    directory = root / name
    evaluations = [json.loads(line) for line in (
        directory / 'evaluations.jsonl').read_text().splitlines()]
    assert [x['action_step'] for x in evaluations] == [0, 3, 6, 9, 12]
    assert all(len(x['scores']) == episodes for x in evaluations)
    final = json.loads((directory / 'final_state.json').read_text())
    assert final['train_action_steps'] == 12
    assert final['evaluations'] == 5
    assert final['train_resets'] >= 2
    assert final['eval_action_steps'] == 5 * episodes * 3
    assert final['match_count'] > 0
    print(name, json.dumps(final, sort_keys=True), flush=True)

  frozen_path = root / 'no-report' / 'c_frozen_fixture.json'
  frozen = freeze(root / 'no-report', frozen_path, engineering_fixture=True)
  for mode in ('dt', 'constant', 'shuffle'):
    directory = root / mode
    command = common + [
        '--logdir', str(directory), '--run.eval_eps', '2',
        '--run.report_every_actions', '0',
        '--agent.rep_probe.mode', mode,
        '--agent.rep_probe.c', str(frozen['c'] if mode == 'constant' else -1)]
    with (root / f'{mode}.log').open('w', encoding='utf-8') as output:
      subprocess.run(command, cwd=repo, stdout=output,
                     stderr=subprocess.STDOUT, check=True)
    evaluations = [json.loads(line) for line in (
        directory / 'evaluations.jsonl').read_text().splitlines()]
    assert [x['action_step'] for x in evaluations] == [0, 3, 6, 9, 12]
    assert all(len(x['scores']) == 2 for x in evaluations)
    final = json.loads((directory / 'final_state.json').read_text())
    assert final['train_action_steps'] == 12
    assert final['match_count'] == 0
    print(mode, json.dumps(final, sort_keys=True), flush=True)


if __name__ == '__main__':
  main()
