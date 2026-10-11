"""Bounded CPU DL-action-information-r1, from immutable calibration records.

No Dreamer/JAX/renderer is imported. Existing actor-input samples are read,
never regenerated. Fitted readouts are an analysis, not Dreamer updates.
"""
import time
ENTRY_STARTED = time.monotonic()
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import socket
import subprocess
import sys

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
  os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MUJOCO_GL'] = 'disable'
os.environ['JAX_PLATFORMS'] = 'cpu'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import dl_task_semantics as semantics
from scripts import dl_action_information_physics as physical
from scripts import dl_action_information_stats as stats

VERSION = 'DL-action-information-r1'
IMPLEMENTATION = 'DL-action-information-runner-r1'
SECONDS = 180
FEEDBACK_SHA = semantics.FEEDBACK_SHA
EPISODES = semantics.EPISODES
POSITIONS = semantics.POSITIONS
JSON_PINS = dict(
    plan='f046120aa9a54fe52abdd6a9c86adaa679fc976239aff27f9304f90ca74e74f0',
    result='4592e363a54385a36e849f937fae259aba06890d0f5e11a86e44393135915309',
    complete='a4bfda1872553fb82706f90e79870d69cf69202ab1bd6885c4a3745c5917657c',
    budget='91e38a756d131ae279a8b8da0c6372422fafcd1ee49b1c76a88fc677fa81f4af',
    supervisor='69c1cac7e217dc9660779ed9661a54b986b1b71ade3b23063f124e431df0ed41')
JSON_NAMES = dict(plan='plan.json', result='result.json', complete='feedback_complete.json',
                  budget='budget.json', supervisor='supervisor_complete.json')
PROOF_KEYS = {'root_key', 'latent_key', 'actor_key', 'selected_logit',
              'h', 'latent', 'raw', 'rng_consumption'}
SOURCES = ('scripts/dl_action_information.py', 'scripts/dl_action_information_physics.py',
           'scripts/dl_action_information_stats.py', 'analysis/dl_task_semantics.py',
           'analysis/dl_feedback_physics_audit_v1.py', 'docs/DL_ACTION_INFORMATION.md')


def require(condition, message):
  if not condition:
    raise ValueError(message)


def sha(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def read_json(path):
  def reject(value):
    raise ValueError('Nonfinite JSON constant: ' + value)
  return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)


def write_json(path, value):
  # Exclusive creation preserves incomplete/failure attempts as evidence.
  with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def git_sha():
  return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def require_clean():
  require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(),
          'Committed clean worktree required')


def tree_digest(tree):
  """Same typed leaf/container identity used by the source; no JAX import."""
  digest = hashlib.sha256()
  def visit(node):
    if isinstance(node, dict):
      digest.update(b'dict')
      for key in sorted(node):
        digest.update(str(key).encode()); visit(node[key])
    elif isinstance(node, (list, tuple)):
      digest.update(type(node).__name__.encode() + str(len(node)).encode())
      for value in node:
        visit(value)
    else:
      value = np.ascontiguousarray(node)
      require(not value.dtype.hasobject and np.isfinite(value).all(), 'Nonfinite/object carry evidence')
      digest.update(str(value.dtype).encode() + str(value.shape).encode())
      digest.update(value.tobytes())
  visit(tree)
  return digest.hexdigest()


def features(records, calibration, audit, model_size):
  """Validate only existing first proofs and their actual source provenance."""
  hd, factors, classes = model_size
  h = np.empty((4, hd), np.float64)
  latent = np.empty((4, 4, 2, factors, classes), np.float64)
  commands = np.empty((4, 8, 12), np.float32)
  receipts = []
  require(calibration['carry_positions'] == POSITIONS and
      len(calibration['carry_sha256']) == 4, 'Source carry position coverage differs')
  for i, pos in enumerate(POSITIONS):
    reference = records[8*i]
    initial = reference['snapshots'][0]
    expected_carry = calibration['carry_sha256'][i]
    require(tree_digest(reference['initial_carry']) == expected_carry,
            'First carry differs from original source byte identity')
    require(initial['suite']['step_count'] == pos and not initial['from_dm_done'] and
            not initial['suite']['reset_next_step'], 'Inactive source/absolute clock differs')
    reference_h = reference['proofs'][0]['h']
    for j, record in enumerate(records[8*i:8*i+8]):
      pair, branch = divmod(j, 2)
      require((record['episode'],record['position'],record['pair'],record['branch']) ==
              (calibration['definition']['episode'],pos,pair,branch), 'First proof record identity differs')
      require(pickle.dumps(record['snapshots'][0], protocol=5) == pickle.dumps(initial, protocol=5),
              'Eight first commands do not share complete initial state/task/RNG/clock bytes')
      require(tree_digest(record['initial_carry']) == record['pre_carry_sha256'][0] == expected_carry,
              'First pre-policy carry/source binding differs')
      for key in audit.OBS_KEYS:
        require(audit.byte_equal(record['observations'][0][key],reference['observations'][0][key]),
                'Eight first proofs do not share actual initial RGB/flags/reward bytes')
      proof = record['proofs'][0]
      require(isinstance(proof, dict) and set(proof) == PROOF_KEYS, 'Unreviewed first proof schema')
      for name in ('root_key', 'latent_key', 'actor_key'):
        audit.array(proof[name], np.uint32, (2,), name)
      require(audit.byte_equal(record['keys'][0], audit.named_keys(record['episode'],pos,pair)[0]) and
              audit.byte_equal(proof['root_key'],record['keys'][0]), 'Frozen actual first root key differs')
      audit.array(proof['rng_consumption'], np.int32, (), 'Actual NJ RNG consumption')
      require(int(proof['rng_consumption']) == 2, 'Actual first sampling count differs')
      for name, shape in (('h',(hd,)), ('latent',(factors,classes)),
                          ('selected_logit',(factors,classes))):
        require(isinstance(proof[name],np.ndarray) and str(proof[name].dtype)=='bfloat16' and
                proof[name].shape==shape and np.isfinite(proof[name]).all(),
                'Actual BF16 first sample/config differs: '+name)
      require(audit.byte_equal(proof['h'],reference_h), 'Actual new h differs across first paired samples')
      z = proof['latent'].astype(np.float64)
      require(np.all((z==0)|(z==1)) and np.all(z.sum(-1)==1),
              'Readout latent must be the actual sampled onehot, not logits/probabilities')
      audit.array(proof['raw'],np.float32,(12,),'actual first raw')
      require(record['action_executed'][0] and audit.byte_equal(proof['raw'],record['raw'][0]),
              'Actor first raw command/proof differs')
      h[i] = proof['h'].astype(np.float64); latent[i,pair,branch] = z
      commands[i,j] = proof['raw']
      if branch:
        paired = records[8*i+j-1]['proofs'][0]
        require(all(audit.byte_equal(proof[k],paired[k]) for k in
            ('root_key','latent_key','actor_key','rng_consumption')), 'Paired actual sampling positions differ')
      receipts.append(dict(position=pos,pair=pair,branch=branch,
          first_proof_sha256=tree_digest(proof),source_carry_sha256=expected_carry,
          initial_snapshot_sha256=hashlib.sha256(pickle.dumps(initial,protocol=5)).hexdigest(),
          root_key=proof['root_key'].astype(int).tolist(),latent_key=proof['latent_key'].astype(int).tolist(),
          actor_key=proof['actor_key'].astype(int).tolist(),raw_sha256=hashlib.sha256(proof['raw'].tobytes()).hexdigest()))
  require('jax' not in sys.modules, 'Feature receipt imported JAX')
  return h, latent, commands, receipts


def worker(args):
  require_clean()
  require(os.name=='posix' and socket.gethostname()=='lyg0326', 'Frozen runtime is sv3 CPU')
  output=Path(args.output).resolve(strict=True)
  budget=read_json(output/'budget.json')
  require(budget['parent_pid']==os.getppid()==int(os.environ.get('DL_ACTION_INFORMATION_PARENT','-1')) and
          budget['seconds']==SECONDS and budget['git_sha']==git_sha() and
          budget['script_sha256']==sha(__file__), 'Actual bounded parent/Git/source binding required')
  def check():
    if time.monotonic() >= budget['monotonic_start'] + SECONDS:
      raise TimeoutError('Frozen CPU object incomplete at 180s hard deadline')
  check()
  dataset=Path(args.dataset).resolve(strict=True)
  for key, filename in JSON_NAMES.items():
    require(sha(dataset/filename)==JSON_PINS[key], 'Published feedback identity differs: '+key)
  audit, dependency=semantics.load_auditor(ROOT/'analysis/dl_feedback_physics_audit_v1.py')
  official=semantics.official_modules()
  suite, _, _, modules, packages=official
  old_plan, old_result, old_hashes=audit.authority(dataset,FEEDBACK_SHA)
  require(old_hashes==JSON_PINS, 'Completed feedback authority changed')
  config=old_plan['metadata']['config']
  require(config['task']=='dmc_quadruped_walk' and config['env.dmc.proprio'] is False and
          config['jax.compute_dtype']=='bfloat16', 'Source task/feature contract differs')
  model_size=tuple(config['agent.dyn.rssm.'+k] for k in ('deter','stoch','classes'))
  require(model_size==(4096,32,32), 'Source size50m feature dimensions differ')
  run_sha=git_sha()
  plan=dict(version=VERSION,implementation=IMPLEMENTATION,actual_git_sha=run_sha,
      source_files_sha256={name:sha(ROOT/name) for name in SOURCES},
      feedback_dataset=str(dataset),feedback_git_sha=FEEDBACK_SHA,feedback_json_sha256=old_hashes,
      source_binding=old_plan['source_binding'],model_provenance=old_plan['metadata'],
      input_artifacts=[dict(episode=x['episode'],path=x['path'],sha256=x['sha256'])
                       for x in old_result['evidence_artifacts']],
      derived_auditor_dependency=dependency,official_modules=modules,packages=packages,
      python=sys.version,server=socket.gethostname(),gpu_devices_used=0,thread_budget=1,
      episodes=EPISODES,positions=POSITIONS,control_names=list(stats.CONTROL_NAMES),
      actor_command_order=['pair0q','pair0p','pair1q','pair1p','pair2q','pair2p','pair3q','pair3p'],
      horizon=20,common_tail='19 raw float32 normalized-zero commands',
      maximum_main_steps=9600,maximum_replay_steps=9600,budget_seconds=SECONDS,
      delta=.2,lambda_mean_objective=1.,tie=1e-10,statistical_seed=20261021,bootstrap_draws=2000,
      sampled_h_shape=[4096],sampled_z_shape=[32,32],feature_dtype='source BF16 exactly promoted to F64',
      fresh_model_or_renderer_or_actor_rng_calls=0,blind_arrays_read=False,gate_changed=False,
      optimizer_updates=0,training_authorized=False)
  write_json(output/'plan.json',plan)
  returns=[];hs=[];zs=[];artifacts=[]; main_steps=0;replay_steps=0
  for item, cal in zip(old_result['evidence_artifacts'],old_plan['calibration']):
    check(); ep=item['episode']
    require(cal['definition']['episode']==ep and cal['definition']['split']=='calibration' and
            cal['definition']['seed']==audit.environment_seed(ep), 'Frozen calibration seed/split differs')
    source=semantics.records_for_episode(dataset,item,ep,audit)
    h,z,actor_commands,first_receipts=features(source,cal,audit,model_size)
    env=suite.load('quadruped','walk',task_kwargs={'random':audit.environment_seed(ep)})
    records=[];verification=[]
    try:
      fixed=physical.controls(env)
      for i,pos in enumerate(POSITIONS):
        first_commands=np.concatenate([fixed,actor_commands[i]],axis=0)
        initial=source[8*i]['snapshots'][0]
        for control,command in enumerate(first_commands):
          check()
          record=physical.run_sequence(env,initial,command,audit,check)
          try:
            replay=physical.replay_sequence(env,record,audit,check)
          except Exception:
            failure=output/f'failed-replay-ep{ep:03}-pos{pos:04}-control{control:02}.npz'
            with failure.open('xb') as stream:
              np.savez_compressed(stream,records=np.frombuffer(pickle.dumps(record,protocol=5),np.uint8),
                                  semantic_verification_passed=np.array(False))
            raise
          record.update(episode=ep,position=pos,control=control)
          records.append(record);verification.append(dict(position=pos,control=control,
              end_type=record['end_type'],executed_steps=record['executed_steps'],**replay))
          main_steps+=record['executed_steps'];replay_steps+=replay['physics_steps']
          require(main_steps<=9600 and replay_steps<=9600, 'Frozen physical step budget exceeded')
        print(json.dumps(dict(stage='first_action',episode=ep,position=pos,commands=15,
                              main_steps=main_steps,replay_steps=replay_steps)),flush=True)
      arrays={name:np.stack([x[name] for x in records]).reshape(4,15,*shape)
              for name,(_,shape) in physical.ARRAYS.items()}
      path=output/f'episode_{ep:03}.npz'
      with path.open('xb') as stream:
        np.savez_compressed(stream,**arrays,episode=np.array(ep,np.int32),positions=np.array(POSITIONS,np.int32),
            h=h,latent=z,first_commands=np.concatenate([np.broadcast_to(fixed,(4,7,12)),actor_commands],axis=1),
            records=np.frombuffer(pickle.dumps(records,protocol=5),np.uint8))
      artifacts.append(dict(episode=ep,path=path.name,sha256=sha(path),records=60,
          main_steps=sum(r['executed_steps'] for r in records),
          replay_steps=sum(v['physics_steps'] for v in verification),
          first_proof_receipts=first_receipts,verifications=verification))
      returns.append(arrays['reward_f64'].sum(-1,dtype=np.float64));hs.append(h);zs.append(z)
    finally:
      env.close()
    require(sha(dataset/item['path'])==item['sha256'], 'Read-only input artifact changed')
    check()
  require(main_steps==replay_steps and len(artifacts)==8 and 'jax' not in sys.modules,
          'Complete CPU object/readonly model boundary failed')
  rewards=np.stack(returns)
  result=stats.analyze(rewards[:,:,:7],rewards[:,:,7:].reshape(8,4,4,2),
                       np.stack(hs),np.stack(zs),np.array(EPISODES,np.int32))
  for key,name in JSON_NAMES.items():
    require(sha(dataset/name)==JSON_PINS[key], 'Read-only source authority changed')
  require_clean();check()
  result.update(runner_implementation=IMPLEMENTATION,actual_git_sha=run_sha,plan_sha256=sha(output/'plan.json'),
      evidence_artifacts=artifacts,feedback_json_sha256=old_hashes,feedback_git_sha=FEEDBACK_SHA,
      main_physics_steps=main_steps,replay_physics_steps=replay_steps,gpu_devices_used=0,
      fresh_model_calls=0,render_calls=0,fresh_actor_rng_calls=0,source_artifacts_unchanged=True,
      no_dreamer_parameters_loaded=True,jax_imported=False,
      actual_from_dm_wrapper_executed=False,new_independent_physics_audit=False,
      elapsed_seconds=time.monotonic()-budget['monotonic_start'])
  write_json(output/'result.json',result);check()
  write_json(output/'action_information_complete.json',dict(version=VERSION,implementation=IMPLEMENTATION,
      actual_git_sha=run_sha,plan_sha256=sha(output/'plan.json'),result_sha256=sha(output/'result.json'),
      complete_episodes=8,complete_states=32,complete_records=480,
      main_physics_steps=main_steps,replay_physics_steps=replay_steps,
      optimizer_updates=0,training_authorized=False,eligible_method_training=False,
      method_effectiveness_established=False,blind_arrays_read=False,gpu_devices_used=0))
  print(json.dumps(dict(complete=True,decision=result['decision'],status=result['status'])),flush=True)


def supervise(command, output, seconds=SECONDS, started=None):
  """Own a new process group; total parent budget includes load/JIT/write."""
  require(np.isfinite(seconds) and seconds>0, 'Positive finite deadline required')
  output=Path(output).resolve()
  require(not output.exists(), 'New attempt output must not exist')
  started=time.monotonic() if started is None else started
  output.mkdir(parents=True)
  write_json(output/'budget.json',dict(version=VERSION,implementation=IMPLEMENTATION,
      parent_pid=os.getpid(),git_sha=git_sha(),seconds=seconds,monotonic_start=started,
      script_sha256=sha(__file__),gpu_devices_used=0))
  timed_out=False
  with (output/'action_information.log').open('x',encoding='utf-8') as log:
    child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
        start_new_session=os.name!='nt',env=dict(os.environ,DL_ACTION_INFORMATION_PARENT=str(os.getpid())))
    try:
      code=child.wait(timeout=max(.001,started+seconds-time.monotonic()))
    except subprocess.TimeoutExpired:
      timed_out=True
      if os.name!='nt':
        try:
          os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:
          pass
      else:
        child.kill()
      child.wait();code=124
    finally:
      if os.name!='nt':
        try:
          os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:
          pass
  (output/'worker_exit_code.txt').write_text(str(code)+'\n',encoding='utf-8')
  receipt=dict(version=VERSION,implementation=IMPLEMENTATION,actual_git_sha=git_sha(),worker_pid=child.pid,
      exit_code=code,timed_out=timed_out,worker_terminal=child.poll() is not None,
      owned_group_cleanup_requested=os.name!='nt',elapsed_seconds=time.monotonic()-started,
      reason='incomplete_hard_budget' if timed_out else 'worker_completed' if code==0 else 'worker_failed',
      budget_sha256=sha(output/'budget.json'),gpu_devices_used=0,training_authorized=False)
  write_json(output/'supervisor_complete.json',receipt)
  return receipt


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--dataset',required=True)
  parser.add_argument('--output',required=True)
  parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
  args=parser.parse_args()
  if args.worker:
    worker(args);return 0
  require_clean()
  require(os.name=='posix' and socket.gethostname()=='lyg0326','Actual frozen execution is sv3 CPU')
  command=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker']
  receipt=supervise(command,args.output,started=ENTRY_STARTED)
  print(json.dumps(receipt),flush=True)
  return receipt['exit_code']


if __name__=='__main__':
  sys.exit(main())
