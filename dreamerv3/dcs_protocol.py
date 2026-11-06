"""Explicit DCS declaration; configuration does not imply launch authorization."""
import hashlib
import json
from pathlib import Path

from dreamerv3.protocol_scope import DCS_DYNAMIC, DCS_SCRIPT, VISION20_TASKS, validate_scope


def load_declaration(config):
  path = config.dcs_protocol.declaration_file
  if not path or not config.dcs_protocol.declaration_sha256:
    raise ValueError('DCS requires an explicit, hashed campaign declaration')
  data = Path(path).read_bytes()
  if hashlib.sha256(data).hexdigest() != config.dcs_protocol.declaration_sha256:
    raise ValueError('DCS declaration checksum differs')
  declaration = json.loads(data)
  if (declaration.get('protocol') != DCS_DYNAMIC or
      declaration.get('tasks') != list(VISION20_TASKS) or
      declaration.get('groups') != ['baseline', 'dt'] or
      type(declaration.get('seed')) is not int or
      not 0 <= declaration['seed'] < 2 ** 32):
    raise ValueError('DCS declaration task/group/seed scope differs')
  return declaration


def backgrounds(config, declaration=None):
  declaration = declaration or load_declaration(config)
  base = declaration['background']
  if set(base) != {'dataset', 'train_videos', 'eval_videos', 'num_videos',
                  'video_alpha', 'ground_by_task', 'seed', 'cache_size'}:
    raise ValueError('DCS declaration background fields differ')
  if set(base['ground_by_task']) != set(VISION20_TASKS):
    raise ValueError('DCS requires explicit ground opacity for all twenty tasks')
  common = dict(mode='dynamic', dataset=base['dataset'], num_videos=base['num_videos'],
      video_alpha=base['video_alpha'], ground_plane_alpha=base['ground_by_task'][config.task],
      seed=base['seed'], cache_size=base['cache_size'])
  return {role: dict(common, videos=base[role + '_videos']) for role in ('train', 'eval')}


def validate_settings(config):
  validate_scope(config.script, config.task, config.seed, config.agent.rep_probe.mode)
  if config.script != DCS_SCRIPT or config.agent.dt_latch.mode != 'off':
    raise ValueError('DCS scope uses the original baseline/Dt agent')
  declaration = load_declaration(config)
  if config.seed != declaration['seed']:
    raise ValueError('DCS seed differs from campaign declaration')
  choices = backgrounds(config, declaration)
  if dict(config.env.dmc.background) != choices['train']:
    raise ValueError('DCS training background differs from campaign declaration')
  from embodied.envs.dcs_background import resolve_options
  domain = config.task.split('_')[1]
  domain = 'ball_in_cup' if domain == 'cup' else domain
  for value in choices.values():
    resolve_options(value, domain, config.seed)
  return declaration


def validate_background(config, actual):
  if dict(actual) not in backgrounds(config).values():
    raise ValueError('DCS environment override differs from declared train/eval background')


def check_config(config):
  """Check inherited production settings without changing the actual config."""
  from scripts.prepare_m2_v1 import check
  validate_settings(config)
  if tuple(config.logger.outputs) not in (('jsonl',), ('jsonl', 'wandb')):
    raise ValueError('DCS requires JSONL, optionally accompanied by online W&B')
  normalized = {'script': 'protocol_v1', 'seed': 0, 'task': 'dmc_hopper_hop',
      'env.dmc.background.mode': 'clean', 'logger.outputs': ['jsonl']}
  if config.run.engineering_fixture:
    fixture = {'run.action_budget': 4098, 'run.eval_every_actions': 2049,
        'run.eval_eps': 1, 'run.match_start': 2200, 'run.match_end': 3600,
        'run.report_every_actions': 2049}
    if any(config.flat[key] != value for key, value in fixture.items()):
      raise ValueError('DCS engineering fixture counters differ')
    normalized.update({'run.engineering_fixture': False,
        'run.action_budget': 1000000, 'run.eval_every_actions': 50000,
        'run.eval_eps': 10, 'run.match_start': 100000, 'run.match_end': 300000,
        'run.report_every_actions': 0})
  check(config.update(normalized))
