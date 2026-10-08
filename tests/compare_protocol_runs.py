"""Compare fixture training after changing only report/evaluation work."""

import argparse
import json
from pathlib import Path


def read_lines(path):
  return [json.loads(line) for line in path.read_text().splitlines()]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('reference')
  parser.add_argument('comparison')
  args = parser.parse_args()
  first, second = Path(args.reference), Path(args.comparison)
  a = json.loads((first / 'final_state.json').read_text())
  b = json.loads((second / 'final_state.json').read_text())
  keys = (
      'params_sha256', 'train_action_steps', 'train_resets',
      'train_policy_calls', 'train_batch_keys', 'updates',
      'match_sum', 'match_count')
  for key in keys:
    assert a[key] == b[key], (key, a[key], b[key])
  receipts_a = read_lines(first / 'update_receipts.jsonl')
  receipts_b = read_lines(second / 'update_receipts.jsonl')
  assert receipts_a == receipts_b
  eval_a = read_lines(first / 'evaluations.jsonl')
  eval_b = read_lines(second / 'evaluations.jsonl')
  assert [x['action_step'] for x in eval_a] == [
      x['action_step'] for x in eval_b]
  for x, y in zip(eval_a, eval_b):
    common = min(len(x['scores']), len(y['scores']))
    assert x['seeds'][:common] == y['seeds'][:common]
    assert x['scores'][:common] == y['scores'][:common]
    assert x['lengths'][:common] == y['lengths'][:common]
  print(json.dumps(dict(
      exact_train_state={key: a[key] for key in keys},
      receipts=len(receipts_a), evaluation_points=len(eval_a),
      episodes_a=len(eval_a[0]['scores']),
      episodes_b=len(eval_b[0]['scores'])), indent=2))


if __name__ == '__main__':
  main()
