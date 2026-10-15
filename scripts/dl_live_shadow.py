"""DL-r5 official CPU shadow, never a live/render clone or model worker.

The parent starts --serve by exec before initializing CUDA/EGL. The framed
protocol is trusted-local pickle5, not a network/untrusted pickle endpoint.
No files, live environment, renderer or JAX runtime are accessed here.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VERSION = 'DL-r5-live-shadow-r1'
# Complete MJB identity is checked by SHA/length, without repeating its assets.
MAX_FRAME = 4 * 1024 * 1024


def require(condition, message):
  if not condition:
    raise ValueError(message)


def _sha(value):
  return hashlib.sha256(value).hexdigest()


def _array(value, dtype, shape, name):
  require(isinstance(value, (np.ndarray, np.generic)) and
      value.dtype == np.dtype(dtype) and value.shape == shape and
      np.isfinite(value).all(), 'Invalid ' + name)
  return value


def array_sha256(value):
  value = np.asarray(value)
  return _sha(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes())


def static_model_identity(model):
  binary = model.to_bytes()
  require(type(binary) is bytes and binary and int(model.ncam) > 2,
      'Actual complete official MJB and camera2 required')
  return dict(mjb_sha256=_sha(binary), mjb_bytes=len(binary), camera=2,
      camera_name=model.id2name(2, 'camera'))


def snapshot_sha256(value):
  require(type(value) is dict and set(value) == {'suite', 'from_dm_done'},
      'Require complete wrapper/suite snapshot')
  state = value['suite']
  require(type(state) is dict and set(state) ==
      {'integration', 'step_count', 'reset_next_step', 'task'}, 'Incomplete full state')
  integration = state['integration']
  require(type(integration) is np.ndarray and integration.dtype == np.float64 and
      integration.ndim == 1 and np.isfinite(integration).all(), 'Invalid integration')
  require(type(state['step_count']) is int and state['step_count'] >= 0 and
      isinstance(state['reset_next_step'], (bool, np.bool_)) and
      isinstance(value['from_dm_done'], (bool, np.bool_)) and type(state['task']) is dict,
      'Invalid task/clock/wrapper state')
  meta = [array_sha256(integration), state['step_count'],
      bool(state['reset_next_step']), bool(value['from_dm_done']),
      _sha(pickle.dumps(state['task'], protocol=5))]
  return _sha(json.dumps(meta, separators=(',', ':')).encode())


def request_sha256(request):
  require(type(request) is dict and request.get('version') == VERSION,
      'Unknown shadow request')
  unit = request['unit']
  require(type(unit) is int and 0 <= unit < 512 and type(request['env_seed']) is int and
      request['env_seed'] == 8192 + unit // 16, 'Shadow unit/env seed differs')
  model = request['static_mjb']
  require(type(model) is dict and set(model) ==
      {'mjb_sha256', 'mjb_bytes', 'camera', 'camera_name'} and
      type(model['mjb_sha256']) is str and re.fullmatch('[0-9a-f]{64}', model['mjb_sha256']) and
      type(model['mjb_bytes']) is int and model['mjb_bytes'] > 0 and
      model['camera'] == 2 and isinstance(model['camera_name'], (str, type(None))),
      'Actual complete MJB SHA/length/camera identity required')
  _array(request['raw_q'], np.float32, (12,), 'baseline-q raw')
  source_sha = snapshot_sha256(request['actual_full_snapshot'])
  require(source_sha == request['source_snapshot_sha256'], 'Source snapshot SHA differs')
  t = 64 + 64 * int(np.random.default_rng([20261030, 0x54494D45, unit]).integers(0, 14))
  require(request['actual_full_snapshot']['suite']['step_count'] == t,
      'Shadow source clock differs from the frozen ordinary position')
  require(type(request['request_id']) is str and request['request_id'] and
      type(request['bundle_sha256']) is str and re.fullmatch('[0-9a-f]{64}', request['bundle_sha256']),
      'Explicit request/bundle identity required')
  meta = [VERSION, unit, request['env_seed'], request['request_id'],
      request['bundle_sha256'], source_sha, model,
      array_sha256(request['raw_q'])]
  return _sha(json.dumps(meta, sort_keys=True, separators=(',', ':')).encode())


def make_request(unit, snapshot, mjb, raw_q, *, bundle_sha256, request_id):
  request = dict(version=VERSION, unit=unit, env_seed=8192 + unit // 16,
      request_id=request_id, bundle_sha256=bundle_sha256,
      actual_full_snapshot=copy.deepcopy(snapshot), static_mjb=copy.deepcopy(mjb),
      source_snapshot_sha256=snapshot_sha256(snapshot), raw_q=raw_q.copy())
  request['request_sha256'] = request_sha256(request)
  return request


def response_sha256(receipt):
  fields = ('version', 'unit', 'env_seed', 'request_id', 'request_sha256',
      'bundle_sha256', 'source_snapshot_sha256', 'static_mjb_sha256',
      'official_steps', 'render_calls', 'model_calls', 'live_environment_touched')
  meta = {name: receipt[name] for name in fields}
  for name in ('raw_q', 'applied', 'reward_f64', 'reward_f32', 'discount',
      'is_first', 'is_last', 'is_terminal'):
    meta[name] = array_sha256(receipt[name])
  meta['next_snapshot_sha256'] = snapshot_sha256(receipt['next_snapshot'])
  require(meta['next_snapshot_sha256'] == receipt['next_snapshot_sha256'],
      'Shadow next snapshot SHA differs')
  return _sha(json.dumps(meta, sort_keys=True, separators=(',', ':')).encode())


def _snapshot(env, done):
  import mujoco
  bits = mujoco.mjtState.mjSTATE_INTEGRATION
  integration = np.empty(mujoco.mj_stateSize(env.physics.model.ptr, bits), np.float64)
  mujoco.mj_getState(env.physics.model.ptr, env.physics.data.ptr, integration, bits)
  return dict(suite=dict(integration=integration, step_count=env._step_count,
      reset_next_step=env._reset_next_step, task=copy.deepcopy(env._task.__dict__)),
      from_dm_done=bool(done))


def shadow_step(request):
  """Exactly one official CPU transition on an independently constructed suite."""
  require('jax' not in sys.modules, 'Shadow must not share a JAX/model process')
  digest = request_sha256(request)
  require(request.get('request_sha256') == digest, 'Shadow request SHA differs')
  from dm_control import suite
  import mujoco
  from scripts.dl_action_contract import production_action_stages
  env = suite.load('quadruped', 'walk', task_kwargs={'random': request['env_seed']})
  try:
    env.reset()  # Official reset has no image/render call.
    require(static_model_identity(env.physics.model) == request['static_mjb'],
        'Shadow/live complete MJB identity differs')
    saved = request['actual_full_snapshot']
    state = saved['suite']
    require(not saved['from_dm_done'] and not state['reset_next_step'] and
        state['step_count'] > 0, 'Shadow requires an ordinary live source state')
    env._task.__dict__.clear()
    env._task.__dict__.update(copy.deepcopy(state['task']))
    env._step_count, env._reset_next_step = state['step_count'], state['reset_next_step']
    mujoco.mj_setState(env.physics.model.ptr, env.physics.data.ptr,
        state['integration'], mujoco.mjtState.mjSTATE_INTEGRATION)
    env.physics.forward()
    require(snapshot_sha256(_snapshot(env, False)) == request['source_snapshot_sha256'],
        'Shadow full-state restoration is not byte exact')
    spec = env.action_spec()
    require(spec.shape == (12,) and spec.dtype == np.float64, 'Official action spec differs')
    applied = production_action_stages(request['raw_q'], spec.minimum, spec.maximum)['applied']
    ts = env.step(applied.copy())
    reward, discount = np.asarray(ts.reward), np.asarray(ts.discount, np.float64)
    _array(reward, np.float64, (), 'official next F64 reward')
    _array(discount, np.float64, (), 'official discount')
    require(not ts.first() and discount in (0, 1), 'Unexpected shadow transition')
    after = _snapshot(env, ts.last())
    receipt = dict(version=VERSION, unit=request['unit'], env_seed=request['env_seed'],
        request_id=request['request_id'], request_sha256=digest,
        bundle_sha256=request['bundle_sha256'],
        source_snapshot_sha256=request['source_snapshot_sha256'],
        static_mjb_sha256=request['static_mjb']['mjb_sha256'], raw_q=request['raw_q'].copy(),
        applied=applied.copy(), reward_f64=reward.copy(), reward_f32=np.asarray(reward, np.float32),
        discount=discount.copy(), is_first=np.bool_(ts.first()), is_last=np.bool_(ts.last()),
        is_terminal=np.bool_(ts.last() and ts.discount == 0), next_snapshot=after,
        next_snapshot_sha256=snapshot_sha256(after), official_steps=1,
        render_calls=0, model_calls=0, live_environment_touched=False)
    receipt['response_sha256'] = response_sha256(receipt)
    return receipt
  finally:
    env.close()


def read_frame(stream):
  header = stream.read(8)
  if not header:
    return None
  require(len(header) == 8, 'Truncated shadow frame header')
  size = struct.unpack('>Q', header)[0]
  require(0 < size <= MAX_FRAME, 'Shadow frame size exceeds fixed bound')
  blocks, remaining = [], size
  while remaining:
    block = stream.read(remaining)
    require(block, 'Truncated shadow frame body')
    blocks.append(block)
    remaining -= len(block)
  return pickle.loads(b''.join(blocks))


def write_frame(stream, value):
  payload = pickle.dumps(value, protocol=5)
  require(0 < len(payload) <= MAX_FRAME, 'Shadow response exceeds fixed bound')
  stream.write(struct.pack('>Q', len(payload))+payload)
  stream.flush()


def serve():
  # Configure only this independent exec process, before simulator import.
  os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', MUJOCO_GL='disable')
  incoming, outgoing = sys.stdin.buffer, sys.stdout.buffer
  # Third-party imports/loaders must never print into the binary response pipe.
  import contextlib
  while True:
    request = read_frame(incoming)
    if request is None:
      return
    with contextlib.redirect_stdout(sys.stderr):
      receipt = shadow_step(request)
    write_frame(outgoing, receipt)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--serve', action='store_true', required=True)
  parser.parse_args()
  serve()
