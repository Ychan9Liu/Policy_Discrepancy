"""Freeze the M2 v1 constant from one completed logging baseline."""

import argparse
import hashlib
import json
from pathlib import Path

from embodied.run.protocol_v1 import ProtocolState


def freeze(baseline_dir, output):
  baseline_dir, output = Path(baseline_dir), Path(output)
  manifest = json.loads((baseline_dir / 'protocol_manifest.json').read_text())
  if manifest['protocol'] != 'M2-v1':
    raise ValueError('Not an M2 v1 baseline')
  config = manifest['config']
  if config['agent.rep_probe.mode'] != 'logging' or float(
      config['agent.rep_probe.alpha']) != 20:
    raise ValueError('Matching source must be the alpha=20 logging baseline')
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
  if sorted({row['action_step'] for row in evaluations}) != list(state.grid):
    raise ValueError('Baseline does not contain the full evaluation grid')
  c = state.frozen_c()
  included = [update_id for update_id, row in state.records.items()
              if 100000 <= row[0] < 300000]
  artifact = dict(
      protocol='M2-v1', task=manifest['task'],
      baseline_dir=str(baseline_dir.resolve()),
      baseline_config_sha256=hashlib.sha256(
          json.dumps(config, sort_keys=True).encode()).hexdigest(),
      receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
      root_seed=0, alpha=20, raw_rep_threshold=1,
      window=[100000, 300000], included_update_ids=included,
      match_sum=state.match_sum, match_count=state.match_count, c=c)
  payload = json.dumps(artifact, sort_keys=True, indent=2) + '\n'
  if output.exists():
    if output.read_text(encoding='utf-8') != payload:
      raise ValueError('Existing frozen c artifact differs')
    return artifact
  output.write_text(payload, encoding='utf-8')
  return artifact


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--baseline_dir', required=True)
  parser.add_argument('--output', required=True)
  args = parser.parse_args()
  print(json.dumps(freeze(args.baseline_dir, args.output), indent=2))


if __name__ == '__main__':
  main()
