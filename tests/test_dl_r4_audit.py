"""Independent-audit fixtures: real CPU physics, explicit fake policy/RGB."""
import copy
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock

import numpy as np

from analysis import dl_r4_audit as audit
from tests import test_dl_r4_physics as fixture
from tests.test_dl_r4_statistics import fixture as statistical_fixture


class IndependentR4Tests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    # Reuse fixture construction only; audit never imports the producer helper.
    cls.case = fixture.R4PhysicsTest(methodName='test_three_real_cpu_branches_null_and_both_separate_replays')
    cls.case.setUp()
    cls.case.keys = audit.branch_keys(0,64,0)
    bundle = cls.case.policy.current(cls.case.carry,cls.case.obs,cls.case.keys[0])
    cls.records = {arm:fixture.p.run_branch(cls.case.policy,cls.case.adapter,
        cls.case.snapshot,cls.case.carry,cls.case.keys,arm=arm,bundle=bundle,
        initial_obs=cls.case.obs,visual_identity=cls.case.visual) for arm in audit.ARMS}

  @classmethod
  def tearDownClass(cls):
    cls.case.doCleanups()

  def test_official_three_arm_300_steps_independent_no_render(self):
    audit.pair_equal(self.records)
    with mock.patch.object(self.case.adapter.suite.physics,'render',side_effect=AssertionError('No render')):
      total = sum(audit.replay_record(self.case.adapter.suite,record,
          episode=0,position=64,pair=0,arm=arm) for arm,record in self.records.items())
    self.assertEqual(total,300)

  def test_real_physics_f64_single_ulp_same_f32_and_mapping_tamper(self):
    for mutation in ('f64','applied','integration'):
      record = copy.deepcopy(self.records['qq'])
      if mutation == 'f64':
        value = record['reward_f64'][9]
        record['reward_f64'][9] = np.nextafter(value,np.inf)
        self.assertEqual(np.float32(value).tobytes(),np.float32(record['reward_f64'][9]).tobytes())
      elif mutation == 'applied':
        record['applied'][0,0] = np.nextafter(record['applied'][0,0],np.inf)
      else:
        record['snapshots'][3]['suite']['integration'][0] = np.nextafter(
            record['snapshots'][3]['suite']['integration'][0],np.inf)
      with self.subTest(mutation=mutation), self.assertRaises(ValueError):
        audit.replay_record(self.case.adapter.suite,record,episode=0,position=64,pair=0,arm='qq')

  def test_wrong_action_memory_prevraw_namespace_and_future_prior_rejected(self):
    mutations = (
        lambda x:x['after_carries'][0][3]['action'][0].__setitem__(0,np.float32(9)),
        lambda x:x['proofs'][0].update(memory_source=np.array(1,np.int8)),
        lambda x:x['proofs'][2].update(action_source=np.array(1,np.int8)),
        lambda x:x['keys'].__setitem__((0,0),np.uint32(7)),
        lambda x:x['reward_f64'].__setitem__(4,np.nan),
        lambda x:x['action_executed'].__setitem__(4,False),
        lambda x:x['proofs'][0]['action_logit'].__setitem__((0,0),np.nan),
    )
    for mutation in mutations:
      record = copy.deepcopy(self.records['qq'])
      mutation(record)
      with self.subTest(mutation=mutation), self.assertRaises(ValueError):
        audit.validate_record(record,episode=0,position=64,pair=0,arm='qq')

  def test_pair_wrong_initial_physics_and_shared_memory_rejected(self):
    changed = copy.deepcopy(self.records)
    changed['pq']['after_carries'][0][1]['stoch'][0][0,0] += 1
    with self.assertRaises(ValueError): audit.pair_equal(changed)

  def test_static_model_identity_is_checked_without_python_ids_or_gl_claim(self):
    expected=copy.deepcopy(self.records['qq']['visual_identity'])
    expected['adapter_identity']=-1; expected['model_identity']=-2
    actual=audit.static_visual(self.case.adapter.suite,expected)
    self.assertEqual(actual['camera'],2)
    expected['mjb_sha256']='0'*64
    with self.assertRaisesRegex(ValueError,'static MJB/camera'):
      audit.static_visual(self.case.adapter.suite,expected)
    changed = copy.deepcopy(self.records)
    changed['pp']['snapshots'][1]['suite']['step_count'] += 1
    with self.assertRaises(ValueError): audit.pair_equal(changed)

  def test_hash_must_precede_pickle_decode_and_path_stays_in_run(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory).resolve()
      path = root/'source.pkl'; path.write_bytes(pickle.dumps({'fixture':1},protocol=5))
      digest = hashlib.sha256(path.read_bytes()).hexdigest()
      artifacts = {'source.pkl':(path,digest)}
      self.assertEqual(audit.trusted_pickle(artifacts,'source.pkl'),{'fixture':1})
      path.write_bytes(b'changed, untrusted bytes')
      with mock.patch.object(audit.pickle,'loads',side_effect=AssertionError('Unsafe decode')) as decode:
        with self.assertRaisesRegex(ValueError,'changed before trusted decode'):
          audit.trusted_pickle(artifacts,'source.pkl')
        decode.assert_not_called()
      with self.assertRaises(ValueError): audit.contained(root,str(path))
      with self.assertRaises((ValueError,FileNotFoundError)):
        audit.contained(root,'../outside.pkl')

  def test_failure_terminal_never_decodes_pickle(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      expected = 'a'*40
      metadata = dict(version=audit.SCIENCE,git_sha=expected)
      for name in ('budget.json','supervisor_terminal.json','worker_complete.json'):
        (root/name).write_text(json.dumps(metadata),encoding='utf-8')
      (root/'budget.json').write_text(json.dumps(dict(metadata,seconds=21000)),encoding='utf-8')
      terminal = dict(metadata,worker_returncode=1,worker_terminal=True,
          main_generation_complete=False,owned_worker_group_cleaned=True,
          elapsed_seconds=1,scientific_support_claimed=False)
      (root/'supervisor_terminal.json').write_text(json.dumps(terminal),encoding='utf-8')
      def git(command):
        return (expected+'\n').encode() if command[1] == 'rev-parse' else b''
      with mock.patch.object(audit,'owned_output',side_effect=git), \
          mock.patch.object(audit.pickle,'loads',side_effect=AssertionError('Unsafe decode')) as decode:
        with self.assertRaisesRegex(ValueError,'successful bounded supervisor'):
          audit.authenticate(root,expected)
        decode.assert_not_called()

  def test_complete_synthetic_authority_hash_chain_before_decode(self):
    """No physics claim: mock Git, tiny non-pickle artifacts, full hash graph."""
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory).resolve(); expected = 'a'*40
      def write(name,value):
        path=root/name; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,allow_nan=False),encoding='utf-8')
        return hashlib.sha256(path.read_bytes()).hexdigest()
      common=root/'nonpickle.bin'; common.write_bytes(b'not a valid or trusted decoded object')
      common_sha=hashlib.sha256(common.read_bytes()).hexdigest()
      def artifact(relative):
        path=root/relative
        path.write_bytes(common.read_bytes())
        return dict(path=path.name,sha256=common_sha)
      payload=statistical_fixture(); evidence=[]
      names=('analysis/dl_r4_statistics.py','scripts/dl_r4_utility.py',
          'scripts/dl_r4_policy.py','scripts/dl_r4_physics.py','docs/DL_R4_UTILITY.md')
      pins={name:hashlib.sha256((audit.ROOT/name).read_bytes()).hexdigest() for name in names}
      budget=dict(version=audit.SCIENCE,git_sha=expected,seconds=21000,parent_pid=100,
          monotonic_deadline=21000,source_files=pins,
          input_pins=dict(checkpoint_sha256=audit.CHECKPOINT,config_sha256=audit.CONFIG))
      write('budget.json',budget)
      process=dict(parent_pid=100,worker_pid=101,worker_pgid=101)
      write('process.json',process)
      runtime=dict(physical_gpu=4,pid=101,compute_uuid=['fake-gpu'],graphics_verified=True,
          live_state=dict(gpu_uuid='fake-gpu'))
      write('runtime_cuda_egl.json',runtime)
      write('model_authenticated.json',dict(metadata=dict(checkpoint_sha256=audit.CHECKPOINT,
          config_sha256=audit.CONFIG,params_sha256=audit.PARAMS,full_state_sha256='9'*64),
          resource=dict(physical_gpu=4,logical_server='sv3',compute_uuid='fake-gpu',
              egl_uuid='fake-gpu',gpu_uuid='fake-gpu')))
      for ep, episode in enumerate(payload['episodes']):
        epdir=f'episode_{ep:03}';(root/epdir).mkdir()
        source=artifact(epdir+'/source.pkl'); signals=artifact(epdir+'/signals.pkl')
        top,selected=audit.statistics.selected_positions(episode['candidates'])
        oracles=[artifact(epdir+f'/t{t:03}_branch_public_oracle.pkl') for t in selected]
        branches=[]
        for t in selected:
          for pair in range(16):
            for arm in audit.ARMS:
              branches.append(dict(position=t,pair=pair,arm=arm,
                  **artifact(epdir+f'/t{t:03}_pair{pair:02}_{arm}.pkl')))
          branches.append(dict(position=t,pair=0,arm='qq_null',
              **artifact(epdir+f'/t{t:03}_pair00_qq_null.pkl')))
        receipt=dict(version=audit.SCIENCE,episode_id=ep,source=source,signals=signals,
            branches=branches,branch_public_oracles=oracles,
            episode_started_monotonic=2*ep,episode_completed_monotonic=2*ep+1,
            model_full_state_unchanged=True)
        relative=epdir+'/complete.json'
        evidence.append(dict(path=relative,sha256=write(relative,receipt)))
      write('calibration_frozen.json',dict(version=audit.SCIENCE,held_episodes_generated=0,
          calibration_episodes=payload['episodes'][:8],frozen_models=payload['frozen_models'],
          calibration_evidence=evidence[:8],sealed_monotonic=15.5))
      input_sha=write('statistics_input.json',payload)
      provisional_sha=write('statistics_provisional.json',{'fixture':True})
      policy=dict(version=audit.SCIENCE,key_namespace='DL_R4_U1',scientific_reads_current_params=True,
          full_state_and_counters_unchanged=True,public_rng_not_consumed_by_private_calls=True,
          current_original_nj_seed_calls=2,derived_nj_seed_calls=0,h1_nj_seed_calls=0,
          first_is_host_assembly_no_policy_or_random_call=True,current_full_state_sha256='9'*64)
      fixed={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
          ('calibration_frozen.json','model_authenticated.json','runtime_cuda_egl.json','budget.json','process.json')}
      worker=dict(version=audit.SCIENCE,git_sha=expected,complete_episodes=32,optimizer_updates=0,
          training_actions=0,full_model_state_unchanged=True,source_seconds=100,labels_seconds=200,
          qualification_seconds=10,statistics_input_sha256=input_sha,
          provisional_statistics_sha256=provisional_sha,fixed_artifact_hashes=fixed,
          runtime_final=runtime,policy_receipt=policy,evidence=evidence)
      worker_sha=write('worker_complete.json',worker)
      (root/'worker.log').write_text('\n'.join(json.dumps(dict(event='episode_complete',episode_id=ep))
          for ep in range(32)),encoding='utf-8')
      (root/'resources.jsonl').write_text(json.dumps(dict(physical_gpu=4,gpu_uuid='fake-gpu',
          processes=[dict(pid=101)]))+'\n',encoding='utf-8')
      terminal=dict(version=audit.SCIENCE,git_sha=expected,worker_returncode=0,worker_terminal=True,
          main_generation_complete=True,owned_worker_group_cleaned=True,elapsed_seconds=100,
          scientific_support_claimed=False,worker_complete_sha256=worker_sha,
          worker_log_sha256=hashlib.sha256((root/'worker.log').read_bytes()).hexdigest(),
          resources_sha256=hashlib.sha256((root/'resources.jsonl').read_bytes()).hexdigest(),parent_failure=None)
      write('supervisor_terminal.json',terminal)
      def git(command):
        if command[1]=='rev-parse':return (expected+'\n').encode()
        if command[1]=='status':return b''
        if command[1]=='ls-files':return ('\n'.join(names)+'\n').encode()
        return (audit.ROOT/command[2].split(':',1)[1]).read_bytes()
      with mock.patch.object(audit,'owned_output',side_effect=git), \
          mock.patch.object(audit.pickle,'loads',side_effect=AssertionError('No decode before all hashes')) as decode:
        _,_,manifests,artifacts=audit.authenticate(root,expected)
        self.assertEqual(len(manifests),32)
        self.assertGreater(len(artifacts),3000)
        decode.assert_not_called()
        # A branch byte corruption invalidates the entire graph before decode.
        (root/'episode_000/t512_pair00_qq.pkl').write_bytes(b'changed artifact bytes')
        with self.assertRaisesRegex(ValueError,'Artifact hash differs'):
          audit.authenticate(root,expected)
        decode.assert_not_called()

  def test_compact_fsum_arithmetic_bound_and_exact_structure(self):
    audit.compare_compact({'x':[1.,2.,3]}, {'x':[1.+1e-14,2.,3]})
    with self.assertRaises(ValueError):
      audit.compare_compact({'x':[1.]},{'x':[1.+1e-8]})
    with self.assertRaises(ValueError):
      audit.compare_compact({'x':[1.]},{'x':[1.],'extra':False})
    with self.assertRaises(ValueError):
      audit.compare_compact({'x':[True]},{'x':[1.]})

  def test_owned_metadata_timeout_cleans_only_own_child(self):
    old_deadline = audit.DEADLINE
    try:
      audit.DEADLINE = audit.time.monotonic()+.1
      with self.assertRaises((ValueError,audit.subprocess.TimeoutExpired)):
        audit.owned_output([audit.sys.executable,'-c','import time; time.sleep(10)'])
      self.assertFalse(audit.OWNED)
    finally: audit.DEADLINE = old_deadline


if __name__ == '__main__': unittest.main()
