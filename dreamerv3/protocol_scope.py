"""Explicit experiment scopes; the original M2 v1 seed0 scope stays closed."""

V1 = 'M2-v1'
HOPPER_123 = 'M2-v1-hopper-seeds123-baseline-dt'
HOPPER_SCRIPT = 'protocol_hopper_seeds123'


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
  raise ValueError(f'Unknown protocol script: {script}')
