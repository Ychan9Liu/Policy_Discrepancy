"""Freeze the M2 v1 constant from one completed logging baseline."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from embodied.run.protocol_v1 import ProtocolState


def freeze(baseline_dir, output, engineering_fixture=False):
  baseline_dir, output = Path(baseline_dir), Path(output)
  manifest = json.loads((baseline_dir / 'protocol_manifest.json').read_text())
  if manifest['protocol'] != 'M2-v1':
    raise ValueError('Not an M2 v1 baseline')
  config = manifest['config']
  if config['agent.rep_probe.mode'] != 'logging' or float(
      config['agent.rep_probe.alpha']) != 20:
    raise ValueError('Matching source must be the alpha=20 logging baseline')
  if engineering_fixture:
    state = ProtocolState(*(
        config[key] for key in ('run.action_budget',
            'run.eval_every_actions', 'run.eval_eps',
            'run.match_start', 'run.match_end')))
  else:
    if config.get('run.engineering_fixture') or manifest['task'] not in (
        'dmc_hopper_hop', 'dmc_quadruped_run',
        'dmc_quadruped_walk', 'dmc_reacher_hard'):
      raise ValueError('Cannot freeze a formal c from an engineering fixture')
    state = ProtocolState(1000000, 50000, 10, 100000, 300000)
  receipt_path = baseline_dir / 'update_receipts.jsonl'
  for line in receipt_path.read_text(encoding='utf-8').splitlines():
    row = json.loads(line)
    state.record_update(row['update_id'], row['start_action'],
        row['match_sum'], row['match_count'], row['invalid_count'])
  evaluations = [
      json.loads(line) for line in
      (baseline_dir / 'evaluations.jsonl').read_text(
          encoding='utf-8').splitlines()]
  if len(evaluations) != len(state.grid) or sorted(
      row['action_step'] for row in evaluations) != list(state.grid):
    raise ValueError('Baseline does not contain the full evaluation grid')
  c = state.frozen_c()
  final = json.loads((baseline_dir / 'final_state.json').read_text(
      encoding='utf-8'))
  matching = json.loads((baseline_dir / 'matching_result.json').read_text(
      encoding='utf-8'))
  if (final['git_commit'] != manifest['git_commit'] or
      final['train_action_steps'] != state.budget or
      final['updates'] != len(state.records) or
      final['evaluations'] != len(state.grid) or
      final['match_sum'] != state.match_sum or
      final['match_count'] != state.match_count or
      matching != dict(match_sum=state.match_sum,
                       match_count=state.match_count, c=c,
                       start=state.match_start, end=state.match_end)):
    raise ValueError('Baseline final state disagrees with matching receipts')
  included = [update_id for update_id, row in state.records.items()
              if state.match_start <= row[0] < state.match_end]
  artifact = dict(
      protocol='M2-v1', task=manifest['task'],
      baseline_git_commit=manifest['git_commit'],
      baseline_dir=str(baseline_dir.resolve()),
      baseline_config_sha256=hashlib.sha256(
          json.dumps(config, sort_keys=True).encode()).hexdigest(),
      receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
      root_seed=0, alpha=20,
      raw_rep_threshold=config['agent.dyn.rssm.free_nats'],
      window=[state.match_start, state.match_end],
      action_step_definition='executed non-reset train environment actions',
      generator_version='dreamerv3.freeze_c M2-v1',
      engineering_fixture=bool(engineering_fixture),
      included_update_ids=included,
      match_sum=state.match_sum, match_count=state.match_count,
      invalid_count=0, c=c)
  if output.exists():
    checksum = output.with_suffix(output.suffix + '.sha256')
    if checksum.read_text(encoding='utf-8').strip() != hashlib.sha256(
        output.read_bytes()).hexdigest():
      raise ValueError('Existing frozen c checksum mismatch')
    existing = json.loads(output.read_text(encoding='utf-8'))
    if {k: v for k, v in existing.items() if k != 'frozen_at_utc'} != artifact:
      raise ValueError('Existing frozen c artifact differs')
    return existing
  artifact['frozen_at_utc'] = datetime.now(timezone.utc).isoformat()
  payload = json.dumps(artifact, sort_keys=True, indent=2) + '\n'
  # DL-engineering-r1: preserve the exact hashed UTF-8 payload on Windows too.
  output.write_bytes(payload.encode('utf-8'))
  output.with_suffix(output.suffix + '.sha256').write_text(
      hashlib.sha256(payload.encode()).hexdigest() + '\n',
      encoding='utf-8')
  return artifact


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--baseline_dir', required=True)
  parser.add_argument('--output', required=True)
  parser.add_argument('--engineering_fixture', action='store_true')
  args = parser.parse_args()
  print(json.dumps(freeze(
      args.baseline_dir, args.output, args.engineering_fixture), indent=2))


if __name__ == '__main__':
  main()
