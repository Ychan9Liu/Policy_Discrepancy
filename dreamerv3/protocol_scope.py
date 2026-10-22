"""Explicit experiment scopes; the original M2 v1 seed0 scope stays closed."""

V1 = 'M2-v1'
HOPPER_123 = 'M2-v1-hopper-seeds123-baseline-dt'
HOPPER_SCRIPT = 'protocol_hopper_seeds123'
DMC16 = 'M2-v1-dmc16-seed0-baseline-dt'
DMC16_SCRIPT = 'protocol_dmc16'
DMC8_FIRST = 'M2-v1-dmc8-first8-seed0-baseline-dt-serial'
V1_TASKS = ('dmc_hopper_hop', 'dmc_quadruped_run',
            'dmc_quadruped_walk', 'dmc_reacher_hard')
# Scores use ball_in_cup_catch; the current DMC wrapper accepts cup_catch.
DMC16_TASKS = ('dmc_acrobot_swingup', 'dmc_cup_catch',
    'dmc_cartpole_balance', 'dmc_cartpole_balance_sparse',
    'dmc_cartpole_swingup', 'dmc_cartpole_swingup_sparse', 'dmc_cheetah_run',
    'dmc_finger_spin', 'dmc_finger_turn_easy', 'dmc_finger_turn_hard',
    'dmc_hopper_stand', 'dmc_pendulum_swingup', 'dmc_reacher_easy',
    'dmc_walker_run', 'dmc_walker_stand', 'dmc_walker_walk')


def tasks_for_protocol(protocol):
  if protocol == DMC16:
    return DMC16_TASKS
  if protocol in (V1, HOPPER_123):
    return V1_TASKS
  raise ValueError(f'Unknown protocol: {protocol}')


def validate_scope(script, task, seed, mode):
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
