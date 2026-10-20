"""Versioned mode diagnostics and read-only audit of reviewed v1 log mirrors.

Unversioned logs are accepted only with explicit legacy=True, never by name alone.
This module does not import JAX, create environments, or change original logs.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess


LEGACY_SHA = '48d71f1541721208322442f5a1d5977124f3e888'


def normalize(row, *, legacy=False, prefix='train/dt/'):
    """Return semantic rates; missing whole-vector observations remain None."""
    version = row.get(prefix + 'mode_diagnostic_version')
    if version is None:
        if not legacy:
            raise ValueError('Unversioned mode fields require reviewed legacy source')
        version = 1
        coordinate = row[prefix + 'mode_factor_diff_frac']
        factor = row[prefix + 'mode_vector_diff_frac']
        vector = None
    elif version == 2:
        coordinate = row[prefix + 'mode_coordinate_diff_frac']
        factor = row[prefix + 'mode_factor_diff_frac']
        vector = row[prefix + 'mode_vector_diff_frac']
    else:
        raise ValueError(f'Unsupported mode diagnostic version: {version}')
    for value in (coordinate, factor, vector):
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError('Invalid mode diagnostic fraction')
    return dict(mode_diagnostic_version=version,
                mode_coordinate_diff_frac=coordinate,
                mode_factor_diff_frac=factor, mode_vector_diff_frac=vector,
                vector_available=vector is not None,
                interpretation=('reviewed legacy coordinate/factor mapping; '
                                'whole vector not recorded' if version == 1 else
                                'v2 coordinate / factor / whole vector reductions'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(inputs, output):
    """Audit all 16 original runs, retaining original fields and hashes."""
    from analysis.analyze_m2_v1 import AUDIT_REL
    from analysis.return_metrics import TASKS, GROUPS
    inputs, output = Path(inputs).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    collection = inputs / 'COLLECTION.json'
    proof = json.loads(collection.read_text(encoding='utf-8'))
    index_path = inputs / AUDIT_REL / 'RUN_INDEX.json'
    if sha(index_path) != proof['files'][index_path.relative_to(inputs).as_posix()]['sha256']:
        raise ValueError('Source index hash mismatch')
    index = json.loads(index_path.read_text(encoding='utf-8'))
    expected = {(task, group, 0) for task in TASKS for group in GROUPS}
    actual = [(r['task'], r['group'], r['seed']) for r in index['runs']]
    if len(actual) != 16 or set(actual) != expected:
        raise ValueError('Not the reviewed 16-run matrix')
    rows, files = [], {str(collection): sha(collection), str(index_path): sha(index_path)}
    for run in index['runs']:
        if run['code_sha'] != LEGACY_SHA:
            raise ValueError('Legacy source not reviewed')
        directory = inputs / run['directory'].lstrip('/')
        path = directory / 'metrics.jsonl'
        digest = sha(path)
        if digest != proof['files'][path.relative_to(inputs).as_posix()]['sha256']:
            raise ValueError('Original metrics hash mismatch')
        files[str(path)] = digest
        config = run['config']
        classes = config['agent.dyn.rssm.classes']
        factors = config['agent.dyn.rssm.stoch']
        steps = []
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            log = json.loads(line)
            if 'train/dt/mode_vector_diff_frac' not in log:
                continue
            rates = normalize(log, legacy=True)
            # For valid one-hot vectors exactly two coordinates differ when a
            # factor differs. Float64 logger aggregation check, not a new fit.
            if not math.isclose(rates['mode_factor_diff_frac'],
                                rates['mode_coordinate_diff_frac'] * classes / 2,
                                rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('One-hot coordinate/factor relation inconsistent')
            steps.append(log['step'])
            rows.append(dict(task=run['task'], group=run['group'], seed=run['seed'],
                             run_id=run['run_id'], training_sha=run['code_sha'],
                             action_step=log['step'], source_file=str(path),
                             source_line=number, source_sha256=digest,
                             original_mode_factor_diff_frac=log['train/dt/mode_factor_diff_frac'],
                             original_mode_vector_diff_frac=log['train/dt/mode_vector_diff_frac'],
                             factors=factors, classes=classes, **rates))
        if steps != list(range(50000, 1000001, 50000)):
            raise ValueError('Missing/conflicting original diagnostic intervals')
    if any(sha(path) != digest for path, digest in files.items()):
        raise ValueError('Source changed during read-only audit')
    (output / 'mode_intervals.json').write_text(
        json.dumps(rows, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (output / 'mode_intervals.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(tool_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  source_files_unchanged=True, input_sha256=files,
                  runs=16, diagnostic_intervals=len(rows),
                  vector_unavailable_intervals=sum(not r['vector_available'] for r in rows),
                  onehot_relation_atol=1e-12, onehot_relation_rtol=1e-12,
                  limitation='Historical whole-vector disagreement cannot be recovered '
                  'from aggregate factor rates. No causal attribution or new training.')
    (output / 'audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.inputs, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'input_sha256'}))


if __name__ == '__main__':
    main()
