"""DCS split routing around the unchanged action-budget/evaluation runner.

The original runner passes named, derived seeds to its environment factory.
Resolve those exact seeds to roles, checking collisions before any agent exists.
No RNG is sampled and no unknown seed or environment index is accepted.
"""
import json
from pathlib import Path

from dreamerv3.dcs_protocol import backgrounds, check_config, load_declaration
from embodied.envs.dcs_background import BackgroundDataset, seed_for_background
from embodied.run import protocol_v1


def seed_routes(config):
  routes = {}
  background_seeds = set()
  def add(seed, value):
    if seed in routes:
      raise ValueError('DCS environment seed collision; routing would be ambiguous')
    background_seed = seed_for_background(seed, config.env.dmc.background.seed)
    if background_seed in background_seeds:
      raise ValueError('DCS background seed collision; independent streams required')
    background_seeds.add(background_seed)
    routes[seed] = dict(value, environment_seed=seed, background_seed=background_seed)
  for index in range(config.run.envs):
    seed = protocol_v1.seed32(config.seed, config.task, 'train_env', index)
    add(seed, dict(role='train', index=index))
  for point in range(0, int(config.run.action_budget) + 1, int(config.run.eval_every_actions)):
    grid = point // int(config.run.eval_every_actions)
    for episode in range(config.run.eval_eps):
      seed = protocol_v1.seed32(config.seed, config.task, 'eval_env', grid, episode)
      add(seed, dict(role='eval', index=0, action_step=point, episode=episode))
  return routes


class SplitFactory:
  """Serializable immutable routing table, also used in parallel train workers."""
  def __init__(self, make_env, config):
    self.make_env = make_env
    self.routes = seed_routes(config)
    self.backgrounds = backgrounds(config)

  def __call__(self, index, *, seed):
    route = self.routes.get(seed)
    if route is None or route['index'] != index:
      raise ValueError('Unknown DCS environment seed or index')
    return self.make_env(index, seed=seed, background=self.backgrounds[route['role']])


def environment_manifest(config):
  choices = backgrounds(config)
  assets = {}
  for role, options in choices.items():
    dataset = BackgroundDataset(options['dataset'], options['videos'],
        options['num_videos'], dynamic=True)
    assets[role] = dataset.manifest(hash_frames=True)
  train = {x['name'] for x in assets['train']['videos']}
  evaluation = {x['name'] for x in assets['eval']['videos']}
  if train & evaluation:
    raise ValueError('DCS training and evaluation video sets overlap')
  return dict(protocol='M2-DCS-dynamic-vision20', task=config.task,
      root_seed=config.seed, declaration=load_declaration(config),
      declaration_sha256=config.dcs_protocol.declaration_sha256,
      backgrounds=choices, assets=assets, routes=list(seed_routes(config).values()),
      routing='Exact original protocol seed32 mapping; reject unknown seeds or collisions',
      background_rng='Independent MT19937; SHA256(pd-dcs-background-v1, env_seed, background.seed)',
      restoration='Environment/replay continuation remains unsupported; fresh attempts only')


def run(make_agent, make_replay, make_env, make_stream, make_logger, args, config):
  check_config(config)
  factory = SplitFactory(make_env, config)
  manifest = environment_manifest(config)
  root = Path(str(args.logdir))
  root.mkdir(parents=True, exist_ok=True)
  path = root / 'dcs_environment.json'
  payload = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
  # Keep the old runner's exact recovery refusal; never overwrite a changed declaration.
  if path.exists():
    if path.read_bytes() != payload:
      raise ValueError('Existing DCS environment/asset manifest differs')
  else:
    with path.open('xb') as stream:
      stream.write(payload)
  protocol_v1.run(make_agent, make_replay, factory, make_stream, make_logger, args, config)
