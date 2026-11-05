"""Audit local DAVIS assets and write environment-only plans; never train."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--dataset', required=True,
                      help='DAVIS/JPEGImages/480p directory')
  parser.add_argument('--output', required=True)
  args = parser.parse_args()
  os.environ['JAX_PLATFORMS'] = 'cpu'
  from PIL import Image
  from embodied.envs.dcs_background import (
      BackgroundDataset, GROUND_ALPHA, IMPLEMENTATION, UPSTREAM_REVISION)
  from dreamerv3.protocol_scope import VISION20_TASKS
  output = Path(args.output)
  output.mkdir(parents=True, exist_ok=False)
  manifest_hashes = {}
  split_names = {}
  for split in ('train', 'val'):
    dataset = BackgroundDataset(args.dataset, split, dynamic=True)
    for paths in dataset.frames.values():
      for path in paths:
        with Image.open(path) as image:
          if image.mode != 'RGB':
            raise ValueError(f'Non-RGB DAVIS frame: {path}')
          image.load()
    split_names[split] = set(dataset.frames)
    value = dataset.manifest(hash_frames=True)
    value['all_images_decoded'] = True
    content = (json.dumps(value, indent=2) + '\n').encode()
    (output / f'{split}-assets.json').write_bytes(content)
    manifest_hashes[split] = hashlib.sha256(content).hexdigest()
  tasks = []
  for task in VISION20_TASKS:
    domain = task.split('_')[1]
    domain = 'ball_in_cup' if domain == 'cup' else domain
    tasks.append(dict(task=task, dcs_ground_alpha=GROUND_ALPHA.get(domain),
        requires_explicit_ground_alpha=domain not in GROUND_ALPHA,
        conditions=['clean', 'static', 'dynamic'],
        task_selected_for_formal_experiment=False))
  repo = Path(__file__).resolve().parents[1]
  overlap = sorted(split_names['train'] & split_names['val'])
  if overlap:
    raise ValueError(f'Train/val videos overlap: {overlap}')
  result = dict(schema_version=1, implementation=IMPLEMENTATION,
      upstream_revision=UPSTREAM_REVISION,
      created_utc=datetime.now(timezone.utc).isoformat(),
      git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
          cwd=repo, text=True).strip(),
      runtime_worktree_diff=subprocess.check_output(['git', 'diff', '--name-only',
          '--', 'embodied', 'dreamerv3', 'scripts', 'tests'], cwd=repo, text=True).splitlines(),
      dataset=str(Path(args.dataset).resolve()), asset_manifests=manifest_hashes,
      train_val_overlap=overlap, tasks=tasks, authorized_to_train=False,
      scientific_protocol=None, model_or_optimizer_constructed=False,
      background_rng='MT19937 seeded by first 4 big-endian bytes of SHA256 '
                     'of NUL-joined [pd-dcs-background-v1, env_seed, background.seed]',
      unresolved=['formal tasks and seeds', 'train/eval condition and video split',
                  'video count and opacity', 'ground opacity for non-DCS domains',
                  'budget and evaluation grid', 'server/GPU allocation'])
  (output / 'environment-plan.json').write_text(
      json.dumps(result, indent=2) + '\n', encoding='utf-8')
  print(json.dumps(dict(output=str(output), assets_verified=True,
                        tasks=len(tasks), authorized_to_train=False)))


if __name__ == '__main__':
  main()
