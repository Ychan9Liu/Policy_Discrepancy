"""DL-r5 family worker: fresh live units, one intervention, no live restore.

Imports are standard-library only until an independent CPU shadow is exec'd.
The parent owns this worker's process group, including the shadow child.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import signal
import socket
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VERSION = 'DL-r5-live-trajectory-r1'
FAMILY_VERSION = 'DL-r5-live-family-r1'


def require(condition, message):
  if not condition:
    raise ValueError(message)


def sha_bytes(value):
  return hashlib.sha256(value).hexdigest()


def sha(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for block in iter(lambda: stream.read(1024*1024), b''):
      digest.update(block)
  return digest.hexdigest()


def read_json(path):
  def pairs(items):
    result = {}
    for key, value in items:
      require(key not in result, 'Duplicate JSON key')
      result[key] = value
    return result
  return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs,
      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def write_bytes(path, payload, check):
  check()
  with Path(path).open('xb') as stream:
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())
  if os.name == 'posix':
    directory = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
      os.fsync(directory)
    finally:
      os.close(directory)
  check()
  return dict(path=Path(path).name, bytes=len(payload), sha256=sha_bytes(payload))


def write_json(path, value, check):
  return write_bytes(path, (json.dumps(value, sort_keys=True, indent=2,
      allow_nan=False)+'\n').encode(), check)


def _process_identity(pid):
  result = dict(pid=pid)
  if os.name == 'posix':
    stat = Path(f'/proc/{pid}/stat').read_text()
    result.update(starttime_ticks=int(stat[stat.rfind(')')+2:].split()[19]),
        pgid=os.getpgid(pid))
  return result


class ShadowClient:
  def __init__(self, check, *, stderr=None):
    check()
    self.check = check
    self.child = subprocess.Popen([sys.executable, str(ROOT/'scripts/dl_live_shadow.py'), '--serve'],
        cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=stderr, env=dict(os.environ, CUDA_VISIBLE_DEVICES='',
            JAX_PLATFORMS='cpu', MUJOCO_GL='disable'))
    self.identity = _process_identity(self.child.pid)
    if os.name == 'posix':
      require(os.getpgid(self.child.pid) == os.getpgrp(), 'Shadow escaped family process group')
    self.exchanges = 0

  def exchange(self, request):
    from scripts.dl_live_shadow import MAX_FRAME
    self.check()
    require(self.child.poll() is None, 'CPU shadow exited before request')
    body = pickle.dumps(request, protocol=5)
    require(0 < len(body) <= MAX_FRAME, 'Shadow request exceeds fixed frame bound')
    self.child.stdin.write(struct.pack('>Q', len(body))+body)
    self.child.stdin.flush()
    header = self.child.stdout.read(8)
    require(len(header) == 8, 'CPU shadow returned no complete header')
    count = struct.unpack('>Q', header)[0]
    require(0 < count <= MAX_FRAME, 'CPU shadow response exceeds fixed frame bound')
    response = self.child.stdout.read(count)
    require(len(response) == count, 'CPU shadow returned a truncated body')
    self.check()
    self.exchanges += 1
    return pickle.loads(response), body, response

  def close(self):
    if not self.child.stdin.closed:
      self.child.stdin.close()
    try:
      self.child.wait(timeout=5)
    except subprocess.TimeoutExpired:
      self.child.kill()
      self.child.wait(timeout=5)
    self.child.stdout.close()
    return dict(identity=self.identity, returncode=self.child.returncode,
        exchanges=self.exchanges, eof_sent=True)


def run_unit(policy, adapter, shadow_client, global_u, output, check, runtime_check=None):
  """Record one actual continuous prefix/outcome; caller owns fresh Env lifetime."""
  import numpy as np
  from scripts import dl_live_policy as live
  from scripts import dl_live_shadow as shadow
  from scripts import dl_feedback_physics as feedback
  from scripts import dl_r4_physics as proofcheck
  from scripts.dl_action_contract import production_action_stages
  plan = live.unit_plan(global_u)
  output = Path(output)
  if not output.exists():
    output.mkdir()
  require(output.is_dir() and not any(output.iterdir()), 'Unit output must be new or empty')
  check()
  started = time.monotonic()
  carry = policy.initial(global_u)
  initial_carry = copy.deepcopy(carry)
  obs = adapter.reset()
  feedback._observation(obs)
  ts = adapter.recorder.last_timestep
  require(ts is not None and ts.first() and bool(obs['is_first']) and
      not bool(obs['is_last']) and obs['reward'] == 0, 'Expected one genuine reset observation')
  initial_snapshot = adapter.snapshot()
  require(initial_snapshot['suite']['step_count'] == 0 and
      not initial_snapshot['from_dm_done'], 'Fresh live unit must start at clock0')
  static = shadow.static_model_identity(adapter.suite.physics.model)
  if runtime_check:
    runtime_check()
  record = dict(version=VERSION, unit=global_u, plan=plan, static_model=static,
      initial_carry=initial_carry, observations=[copy.deepcopy(obs)],
      snapshots=[initial_snapshot], outcome_proofs=[], after_carries=[], oracles=[],
      pre_carry_sha256=[], after_carry_sha256=[], keys=[], raw=[], applied=[],
      reward_f64=[], reward_f32=[], discount=[], is_first=[], is_last=[], is_terminal=[])
  artifacts, last_normal = {}, None
  for step in range(plan['t']+100):
    check()
    require(not bool(obs['is_last']) and not bool(obs['is_terminal']), 'Live loop crossed last/reset')
    before = feedback._carry(carry)
    seen = live.kernel.tree_digest(obs)
    key = live.policy_root_key(global_u, step)
    if step == plan['t']:
      require(not bool(obs['is_first']), 'Intervention must be ordinary active source')
      record['intervention_before_carry'] = copy.deepcopy(carry)
      bundle = policy.current(carry, obs, global_u)
      require(bundle['before_carry_sha256'] == live.kernel.tree_digest(carry) and
          bundle['observation_sha256'] == seen, 'Current policy input provenance differs')
      record['bundle'] = copy.deepcopy(bundle)
      record['oracles'].append(policy.public_oracle(carry, obs, global_u, step,
          expected_after=bundle['q_after'], expected_raw=bundle['q_raw']))
      source_snapshot = adapter.snapshot()
      source_sha = shadow.snapshot_sha256(source_snapshot)
      before_model = shadow.static_model_identity(adapter.suite.physics.model)
      require(before_model == static, 'Live static model changed before shadow')
      request = shadow.make_request(global_u, source_snapshot, before_model, bundle['q_raw'],
          bundle_sha256=bundle['bundle_sha256'], request_id=f'unit{global_u:03d}')
      response, request_body, response_body = shadow_client.exchange(request)
      # Bind the exact sent/received bytes before using the shadow target.
      wire_request,wire_response = pickle.loads(request_body),pickle.loads(response_body)
      require(shadow.request_sha256(wire_request) == wire_request['request_sha256'] ==
          request['request_sha256'] == shadow.request_sha256(request) and
          response['request_sha256'] == request['request_sha256'] and
          shadow.response_sha256(wire_response) == wire_response['response_sha256'] ==
          response['response_sha256'] == shadow.response_sha256(response),
          'Shadow wire bodies/request/response identity differ')
      after_sha = shadow.snapshot_sha256(adapter.snapshot())
      require(source_sha == after_sha and before_model ==
          shadow.static_model_identity(adapter.suite.physics.model), 'CPU shadow changed live state/model')
      require(feedback._carry(carry) == before and live.kernel.tree_digest(obs) == seen,
          'CPU shadow changed live policy carry/RGB')
      artifacts['shadow_request.pkl'] = write_bytes(output/'shadow_request.pkl',request_body,check)
      artifacts['shadow_response.pkl'] = write_bytes(output/'shadow_response.pkl',response_body,check)
      seal = policy.seal_pretreatment(bundle,response,live_before_sha256=source_sha,
          live_after_sha256=after_sha,r_t_f64=record['reward_f64'][step-1])
      artifacts['pretreatment.pkl'] = write_bytes(output/'pretreatment.pkl',
          pickle.dumps(seal,protocol=5),check)
      signal_event = 3*global_u + int(global_u >= 128)
      signals = dict(version=FAMILY_VERSION, unit=global_u, event=signal_event,
          monotonic=time.monotonic(), pretreatment_sha256=seal['pretreatment_sha256'],
          bundle_sha256=bundle['bundle_sha256'], source_snapshot_sha256=source_sha,
          pretreatment_file=artifacts['pretreatment.pkl'],
          shadow_request=artifacts['shadow_request.pkl'], shadow_response=artifacts['shadow_response.pkl'],
          D=float(seal['Dmode']), K=float(seal['K']), e=float(seal['e']),
          r_t=float(seal['r_t']), r_t_f32=float(seal['r_t_f32']))
      artifacts['signals_sealed.json'] = write_json(output/'signals_sealed.json',signals,check)
      # The durable write and fsync above precede even drawing/revealing the label.
      label_receipt = live.reveal_label(global_u,seal['pretreatment_sha256'])
      require(all(sha(output/name) == entry['sha256'] and
          (output/name).stat().st_size == entry['bytes'] for name,entry in artifacts.items()),
          'Sealed pretreatment artifacts changed before assignment')
      assignment = dict(version=FAMILY_VERSION, unit=global_u, event=signal_event+1,
          monotonic=time.monotonic(), label_receipt=label_receipt,
          pretreatment_file=artifacts['pretreatment.pkl'],
          signals_file=artifacts['signals_sealed.json'],
          pretreatment_sha256=seal['pretreatment_sha256'])
      artifacts['assignment.json'] = write_json(output/'assignment.json',assignment,check)
      record.update(seal=seal,label_receipt=label_receipt)
      after, raw, proof = policy.first(bundle,label_receipt['label'],label_receipt=label_receipt)
      arm = 'qq' if label_receipt['label'] in ('qqA','qqB') else label_receipt['label']
      proofcheck._first_matches(bundle['kernel_bundle'],arm,after,raw,proof)
    else:
      after, raw, proof = policy.normal_step(carry,obs,global_u,step)
      arm = 'qq'
      last_normal = (step,copy.deepcopy(carry),copy.deepcopy(obs),copy.deepcopy(after),raw.copy())
      if step == 0:
        record['oracles'].append(policy.public_oracle(carry,obs,global_u,step,
            expected_after=after,expected_raw=raw))
    require(feedback._carry(carry) == before and live.kernel.tree_digest(obs) == seen,
        'Policy mutated its actual carry/observation input')
    proofcheck._proof(proof,key,raw,after,arm=arm)
    following, applied = adapter.step(raw.copy())
    feedback._observation(following)
    reward, discount = feedback._timestep(adapter.recorder.last_timestep)
    feedback._flags(following,adapter.recorder.last_timestep)
    expected_applied = production_action_stages(raw,adapter.low,adapter.high)['applied']
    require(applied.dtype == np.float64 and applied.shape == (12,) and
        applied.tobytes() == expected_applied.tobytes(), 'Actual physical action mapping differs')
    require(np.asarray(reward,np.float32).tobytes() == following['reward'].tobytes(),
        'Actual F64/F32 reward encoding differs')
    state = adapter.snapshot()
    shadow.snapshot_sha256(state)
    require(state['suite']['step_count'] == step+1 and
        bool(state['from_dm_done']) == bool(following['is_last']) and
        bool(state['suite']['reset_next_step']) == bool(following['is_last']),
        'Actual live clock/wrapper/reset state differs')
    if step == plan['t'] and arm == 'qq':
      require(reward.tobytes() == response['reward_f64'].tobytes() and
          applied.tobytes() == response['applied'].tobytes() and
          discount.tobytes() == response['discount'].tobytes() and
          shadow.snapshot_sha256(state) == response['next_snapshot_sha256'],
          'Actual qq first transition differs from its baseline-q CPU shadow')
    record['keys'].append(key.copy())
    record['pre_carry_sha256'].append(before)
    record['after_carry_sha256'].append(feedback._carry(after,raw))
    for name,value in (('raw',raw),('applied',applied),('reward_f64',reward),
        ('reward_f32',following['reward']),('discount',discount)):
      record[name].append(value.copy())
    for name in ('is_first','is_last','is_terminal'):
      record[name].append(following[name].copy())
    if step >= plan['t']:
      record['outcome_proofs'].append(copy.deepcopy(proof))
      record['after_carries'].append(copy.deepcopy(after))
    record['observations'].append(copy.deepcopy(following))
    record['snapshots'].append(state)
    carry,obs = copy.deepcopy(after),following
    if bool(following['is_last']):
      require(step >= plan['t'] and discount == 0, 'Premature prefix last or unexpected time limit')
      break
  require(last_normal is not None and 'seal' in record, 'Missing real prefix/intervention')
  step,before,last_obs,last_after,last_raw = last_normal
  record['oracles'].append(policy.public_oracle(before,last_obs,global_u,step,
      expected_after=last_after,expected_raw=last_raw))
  record['final_carry'] = copy.deepcopy(carry)
  for name in ('keys','raw','applied','reward_f64','reward_f32','discount',
      'is_first','is_last','is_terminal'):
    record[name] = np.stack(record[name])
  count = len(record['raw'])
  record.update(executed_steps=count,outcome_steps=count-plan['t'],
      end_type='true_terminal' if bool(obs['is_terminal']) else 'window',
      live_restore_calls=0,extra_live_render_calls=0,
      last_normal_oracle_step=step)
  require(1 <= record['outcome_steps'] <= 100 and
      shadow.static_model_identity(adapter.suite.physics.model) == static,
      'Incomplete outcome or changed live static model')
  policy.assert_frozen(full=True)
  artifacts['trajectory.pkl'] = write_bytes(output/'trajectory.pkl',pickle.dumps(record,protocol=5),check)
  row = dict(family_id=global_u//16,unit_id=global_u%16,env_seed=plan['env_seed'],
      policy_seed=plan['eval_seed'],position=plan['t'],label=label_receipt['label'],
      pretreatment={k:signals[k] for k in ('D','K','e','r_t','r_t_f32')},
      outcome=dict(reward_sum=math.fsum(float(x) for x in record['reward_f64'][plan['t']: ])),
      chronology=dict(signals_sealed=signal_event,assignment=signal_event+1,outcome_complete=signal_event+2))
  complete = dict(version=FAMILY_VERSION,unit=global_u,plan=plan,unit_row=row,
      event=signal_event+2,monotonic=time.monotonic(),started_monotonic=started,
      artifacts=artifacts,executed_steps=count,outcome_steps=record['outcome_steps'],
      policy_calls=count,original_nj_seed_calls=2*count,derived_nj_seed_calls=0,
      first_assembly_model_calls=0,public_oracle_calls=len(record['oracles']),
      oracle_steps=[x['step'] for x in record['oracles']],live_restore_calls=0,
      extra_live_render_calls=0,renderer_byte_exact_claimed=False)
  write_json(output/'complete.json',complete,check)
  return complete


def worker(args):
  require(os.name == 'posix' and socket.gethostname().split('.')[0] == 'lyg0326',
      'DL-r5 GPU worker requires the assigned sv3 Linux host')
  require(type(args.family) is int and 0 <= args.family < 32 and
      os.getpgrp() == os.getpid(), 'Require one fresh owned family process group')
  from scripts.dl_live_trial import verify_budget
  budget,check = verify_budget(args.budget,args.expected_git,args.checkpoint,
      args.config,args.resource_record)
  deadline = budget['monotonic_deadline']
  signal.signal(signal.SIGALRM, lambda signum,frame: os._exit(124))
  signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
  check()
  output = Path(args.output).resolve(strict=True)
  require(output.is_dir() and not (output/'family_complete.json').exists(),
      'Fresh pre-created family output directory required')
  require('jax' not in sys.modules and not any(name.startswith('dm_control') for name in sys.modules),
      'Family must spawn CPU shadow before model/GL imports')
  units = []
  with (output/'shadow.stderr.log').open('xb') as shadow_log:
    client = ShadowClient(check,stderr=shadow_log)
    write_json(output/'shadow_process.json',dict(version=FAMILY_VERSION,
        worker_identity=_process_identity(os.getpid()),parent_pid=os.getppid(),
        shadow_identity=client.identity,spawned_monotonic=time.monotonic(),
        shadow_before_gpu_import=True),check)
    try:
      from scripts import dl_r4_utility as r4
      r4.DEADLINE = deadline
      from scripts.dl_resources import refresh,verify_runtime
      resource = refresh(read_json(args.resource_record))
      require(resource['physical_gpu'] == 4 and resource['gpu_uuid'] == budget['gpu_uuid'],
          'Family resource changed the assigned physical GPU4')
      agent,metadata = r4.load_agent(args.checkpoint,args.config,output)
      from scripts.dl_live_policy import LivePolicy
      from scripts.dl_feedback_physics import FeedbackEnvironment
      from scripts import dl_diagnostic as diagnostic
      policy = LivePolicy(agent)
      write_json(output/'model_authenticated.json',dict(version=FAMILY_VERSION,
          metadata=metadata,resource=resource,authenticated_monotonic=time.monotonic()),check)
      for local_unit in range(16):
        check()
        global_u = 16*args.family+local_unit
        adapter = FeedbackEnvironment(diagnostic.make_dmc('dmc_quadruped_walk',8192+args.family))
        try:
          runtime = (lambda: write_json(output/'runtime_cuda_egl.json',
              verify_runtime(resource,compute=True,graphics=True),check)) if local_unit == 0 else None
          complete = run_unit(policy,adapter,client,global_u,
              output/f'unit_{local_unit:02d}',check,runtime_check=runtime)
          units.append(dict(unit=global_u,path=f'unit_{local_unit:02d}/complete.json',
              sha256=sha(output/f'unit_{local_unit:02d}'/'complete.json'),unit_row=complete['unit_row']))
          if local_unit == 15:
            write_json(output/'runtime_after.json',verify_runtime(resource,
                compute=True,graphics=True),check)
        finally:
          adapter.close()
      policy.assert_frozen(full=True)
      final_policy_receipt = policy.receipt()
      check()
    finally:
      closed = client.close()
      write_json(output/'shadow_closed.json',closed,check)
    require(closed['returncode'] == 0 and closed['exchanges'] == 16,
        'CPU shadow did not finish all16 requests and EOF cleanly')
  check()
  # Re-authenticate complete source/input provenance after all work, before success.
  verify_budget(args.budget,args.expected_git,args.checkpoint,args.config,args.resource_record)
  family_complete = dict(version=FAMILY_VERSION,family=args.family,units=units,
      completed_monotonic=time.monotonic(),policy_receipt=final_policy_receipt,
      shadow=closed,artifacts={name:dict(path=name,sha256=sha(output/name),bytes=(output/name).stat().st_size)
          for name in ('shadow_process.json','shadow_closed.json','model_authenticated.json',
              'runtime_cuda_egl.json','runtime_after.json','shadow.stderr.log')})
  write_json(output/'family_complete.json',family_complete,check)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--family',type=int,required=True)
  for name in ('output','budget','expected-git','checkpoint','config','resource-record'):
    parser.add_argument('--'+name,required=True)
  worker(parser.parse_args())


if __name__ == '__main__':
  main()
