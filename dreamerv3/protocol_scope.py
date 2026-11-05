"""Explicit experiment scopes; the original M2 v1 seed0 scope stays closed."""

V1 = 'M2-v1'
DTL_THREE = 'DtL-three-seed0-20261105'
DTL_THREE_SCRIPT = 'protocol_dtl_three'
DTL_THREE_TASKS = ('dmc_quadruped_run', 'dmc_quadruped_walk', 'dmc_finger_spin')
HOPPER_123 = 'M2-v1-hopper-seeds123-baseline-dt'
HOPPER_SCRIPT = 'protocol_hopper_seeds123'
DMC16 = 'M2-v1-dmc16-seed0-baseline-dt'
DMC16_SCRIPT = 'protocol_dmc16'
VISION20_SEED1 = 'M2-v1-vision20-seed1-baseline-dt'
VISION20_SCRIPT = 'protocol_vision20_seed1'
PENDULUM_23 = 'M2-v1-pendulum-seeds23-baseline-dt'
PENDULUM_SCRIPT = 'protocol_pendulum_seeds23'
DMC8_FIRST = 'M2-v1-dmc8-first8-seed0-baseline-dt-serial'
DMC8_REMAINING = 'M2-v1-dmc8-remaining-seed0-baseline-dt-serial'
V1_TASKS = ('dmc_hopper_hop', 'dmc_quadruped_run',
            'dmc_quadruped_walk', 'dmc_reacher_hard')
# Scores use ball_in_cup_catch; the current DMC wrapper accepts cup_catch.
DMC16_TASKS = ('dmc_acrobot_swingup', 'dmc_cup_catch',
    'dmc_cartpole_balance', 'dmc_cartpole_balance_sparse',
    'dmc_cartpole_swingup', 'dmc_cartpole_swingup_sparse', 'dmc_cheetah_run',
    'dmc_finger_spin', 'dmc_finger_turn_easy', 'dmc_finger_turn_hard',
    'dmc_hopper_stand', 'dmc_pendulum_swingup', 'dmc_reacher_easy',
    'dmc_walker_run', 'dmc_walker_stand', 'dmc_walker_walk')
VISION20_TASKS = V1_TASKS + DMC16_TASKS


def tasks_for_protocol(protocol):
  if protocol == DTL_THREE:
    return DTL_THREE_TASKS
  if protocol == PENDULUM_23:
    return ('dmc_pendulum_swingup',)
  if protocol == VISION20_SEED1:
    return VISION20_TASKS
  if protocol == DMC16:
    return DMC16_TASKS
  if protocol in (V1, HOPPER_123):
    return V1_TASKS
  raise ValueError(f'Unknown protocol: {protocol}')


def validate_scope(script, task, seed, mode):
  if script == DTL_THREE_SCRIPT:
    if (type(seed) is not int or seed != 0 or task not in DTL_THREE_TASKS or
        mode != 'off'):
      raise ValueError('DtL extension requires the declared three tasks, seed0, rep_probe off')
    return DTL_THREE
  if script == PENDULUM_SCRIPT:
    if (type(seed) is not int or seed not in (2, 3) or
        task != 'dmc_pendulum_swingup' or mode not in ('logging', 'dt')):
      raise ValueError('Pendulum extension requires pendulum_swingup, seeds 2/3, logging or dt')
    return PENDULUM_23
  if script == VISION20_SCRIPT:
    if (type(seed) is not int or seed != 1 or task not in VISION20_TASKS or
        mode not in ('logging', 'dt')):
      raise ValueError('Vision20 extension requires the visual20, seed1, logging or dt')
    return VISION20_SEED1
  if script == 'protocol_v1':
    if seed != 0:
      raise ValueError('M2 v1 requires training seed 0')
    return V1
  if script == HOPPER_SCRIPT:
    if (type(seed) is not int or seed not in (1, 2, 3) or
        task != 'dmc_hopper_hop' or mode not in ('logging', 'dt')):
      raise ValueError('Hopper extension requires hopper_hop, seeds 1/2/3, logging or dt')
    return HOPPER_123
  if script == DMC16_SCRIPT:
    if (type(seed) is not int or seed != 0 or task not in DMC16_TASKS or
        mode not in ('logging', 'dt')):
      raise ValueError('DMC16 requires the remaining 16 tasks, seed0, logging or dt')
    return DMC16
  raise ValueError(f'Unknown protocol script: {script}')


def validate_dtl_scope(config):
  """Keep DtL opt-in and prevent mislabelling it as any frozen M2 arm."""
  dl = config.agent.get('dt_latch', {})
  if config.script != DTL_THREE_SCRIPT:
    if dl.get('mode', 'off') != 'off':
      raise ValueError('Existing M2 protocols do not authorize DtLatch')
    return
  expected = dict(mode='dtlatch', alpha=20.0, rho=0.5, normal_only=False,
                  kappa=-1.0, kl_tolerance=1e-5, d_bins=(-1.0,), k_bins=(-1.0,))
  for key, value in expected.items():
    actual = dl.get(key)
    if key.endswith('_bins') and actual is not None:
      actual = tuple(actual)
    if actual != value:
      raise ValueError(f'DtL three-task setting differs: {key}')
  if config.env.dmc.get('background', {}).get('mode', 'clean') != 'clean':
    raise ValueError('DtL clean preparation does not authorize a DCS training protocol')
