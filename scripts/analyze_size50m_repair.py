"""Compare two engineering action/update traces without altering run evidence."""

import argparse
import json
from pathlib import Path


def rows(path):
  return [json.loads(line) for line in path.read_text().splitlines() if line]


def fields(row, label):
  if label == 'action':
    return row['hashes']
  values = {key: value for key, value in row.items()
            if key not in ('batch_hashes', 'parameter_hashes')}
  values.update({f'batch/{key}': value
                 for key, value in row['batch_hashes'].items()})
  values.update({f'param/{key}': value
                 for key, value in row['parameter_hashes'].items()})
  return values


def compare(first, repeat):
  action_a = rows(first / 'engineering_action_trace.jsonl')
  action_b = rows(repeat / 'engineering_action_trace.jsonl')
  update_a = rows(first / 'engineering_update_trace.jsonl')
  update_b = rows(repeat / 'engineering_update_trace.jsonl')
  result = {'action_rows': [len(action_a), len(action_b)],
            'update_rows': [len(update_a), len(update_b)],
            'action_first_difference': None, 'action_key_differences': {},
            'update_first_difference': None, 'update_key_differences': {}}
  for label, left, right in (('action', action_a, action_b),
                             ('update', update_a, update_b)):
    if len(left) != len(right):
      raise ValueError(f'{label} trace lengths differ')
    for index, (a, b) in enumerate(zip(left, right)):
      values_a, values_b = fields(a, label), fields(b, label)
      keys = set(values_a) | set(values_b)
      changed = sorted(key for key in keys if values_a.get(key) != values_b.get(key))
      if changed:
        if result[f'{label}_first_difference'] is None:
          result[f'{label}_first_difference'] = dict(index=index,
              action_step=a.get('action_step', a.get('start_action')),
              worker=a.get('worker'), key_count=len(changed),
              first_keys=changed[:12])
        for key in changed:
          result[f'{label}_key_differences'][key] = (
              result[f'{label}_key_differences'].get(key, 0) + 1)
  return result


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('first', type=Path)
  parser.add_argument('repeat', type=Path)
  parser.add_argument('--output', type=Path)
  args = parser.parse_args()
  result = compare(args.first, args.repeat)
  payload = json.dumps(result, indent=2, sort_keys=True) + '\n'
  if args.output:
    if args.output.exists():
      raise FileExistsError(args.output)
    args.output.write_text(payload)
  print(payload, end='')


if __name__ == '__main__':
  main()
