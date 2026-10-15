"""Calibration-stop audit fixtures: no real R4 scientific results are read."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from analysis import dl_r4_calibration_audit as c
from tests import test_dl_r4_audit as physics_fixture
from tests import test_dl_r4_failure_capture as capture_fixture
from tests.test_dl_r4_statistics import fixture as statistics_fixture


class CalibrationFailureTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    physics_fixture.IndependentR4Tests.setUpClass()
    cls.record=physics_fixture.IndependentR4Tests.records['qq']
    cls.environment=physics_fixture.IndependentR4Tests.case.adapter.suite

  @classmethod
  def tearDownClass(cls):
    physics_fixture.IndependentR4Tests.tearDownClass()

  def test_real_cpu_qq_null_200_steps_no_render_and_no_model(self):
    q,n=copy.deepcopy(self.record),copy.deepcopy(self.record)
    with mock.patch.object(self.environment.physics,'render',side_effect=AssertionError('No render')):
      total=sum(c.a.replay_record(self.environment,record,episode=0,position=64,pair=0,arm=arm)
          for record,arm in ((q,'qq'),(n,'qq_null')))
    self.assertEqual(total,200)
    c.snapshot_equal(q['snapshots'][0],n['snapshots'][0])
    stored=dict(returns=dict(qq=[float(q['reward_f64'].sum())]*16),
        qq_null_return=float(n['reward_f64'].sum()),null_max_abs_step_reward_diff=0.)
    self.assertFalse(c.null_difference(q,n,stored)['failed'])
    self.assertTrue(all(value is None for value in
        c.stored_difference_order(q,n)['first_byte_difference'].values()))

  def test_null_boundary_equality_and_one_ulp_failure_are_not_relaxed(self):
    q=dict(reward_f64=np.zeros(100,np.float64))
    n=dict(reward_f64=np.zeros(100,np.float64));n['reward_f64'][0]=.25
    stored=dict(returns=dict(qq=[0.]*16),qq_null_return=.25,null_max_abs_step_reward_diff=.25)
    self.assertEqual(c.null_difference(q,n,stored)['absolute_average_difference'],.0025)
    self.assertFalse(c.null_difference(q,n,stored)['failed'])
    n['reward_f64'][0]=np.nextafter(.25,np.inf)
    stored.update(qq_null_return=float(n['reward_f64'][0]),
        null_max_abs_step_reward_diff=float(n['reward_f64'][0]))
    self.assertTrue(c.null_difference(q,n,stored)['failed'])
    # Numerically close is insufficient if arithmetic changes the frozen side.
    stored['qq_null_return']=.25
    with self.assertRaisesRegex(ValueError,'threshold side'):
      c.null_difference(q,n,stored)

  def test_initial_snapshot_entire_tree_not_only_integration(self):
    initial=self.record['snapshots'][0]
    changed=copy.deepcopy(initial);changed['additional_unreviewed_field']=False
    with self.assertRaisesRegex(ValueError,'serialized tree'):
      c.snapshot_equal(changed,initial)
    changed=copy.deepcopy(initial);changed['suite']['step_count']+=1
    with self.assertRaises(ValueError):c.snapshot_equal(changed,initial)

  def test_stored_first_difference_indices_and_integer_pixel_counts(self):
    q,n=copy.deepcopy(self.record),copy.deepcopy(self.record)
    original=int(n['observations'][1]['image'][0,0,0])
    n['observations'][1]['image'][0,0,0]=original+1 if original < 255 else original-1
    n['raw'][2,0]+=np.float32(.1)
    n['snapshots'][3]['suite']['integration'][0]=np.nextafter(
        n['snapshots'][3]['suite']['integration'][0],np.inf)
    n['proofs'][4]['latent'][0,0]+=np.float32(.1)
    result=c.stored_difference_order(q,n)
    self.assertEqual(result['first_byte_difference']['image_frame'],1)
    self.assertEqual(result['first_byte_difference']['observation_frame'],1)
    self.assertEqual(result['first_byte_difference']['raw'],2)
    self.assertEqual(result['first_byte_difference']['physics_integration_frame'],3)
    self.assertEqual(result['first_byte_difference']['latent_action'],4)
    self.assertEqual(result['first_image_difference'],dict(frame=1,unequal_pixels=1,
        unequal_channel_elements=1,max_abs_integer_difference=1))

  def test_external_capture_sha_failure_precedes_pickle_decode(self):
    with tempfile.TemporaryDirectory() as directory:
      root=Path(directory);capture=root/'capture.json';capture.write_text('{}')
      with mock.patch.object(c.a.pickle,'loads',side_effect=AssertionError('Unsafe decode')) as decode:
        with self.assertRaisesRegex(ValueError,'capture SHA'):
          c.authenticate(root,capture,'0'*64,c.RUN_GIT,'a'*40)
        decode.assert_not_called()

  def test_no_held_even_empty_directory_and_no_success_output(self):
    with tempfile.TemporaryDirectory() as directory:
      root=Path(directory)
      for i in range(8):(root/f'episode_{i:03}').mkdir()
      c.no_held(root)
      (root/'episode_008').mkdir()
      with self.assertRaisesRegex(ValueError,'Held directory'):c.no_held(root)
      (root/'episode_008').rmdir();(root/'worker_complete.json').write_text('{}')
      with self.assertRaisesRegex(ValueError,'successful full-assay'):c.no_held(root)

  def test_full_auditor_still_rejects_cal_stop_without_worker_complete(self):
    with tempfile.TemporaryDirectory() as directory:
      root=Path(directory)
      for name in ('budget.json','supervisor_terminal.json'):(root/name).write_text('{}')
      def git(command):return (c.RUN_GIT+'\n').encode() if command[1] == 'rev-parse' else b''
      with mock.patch.object(c.a,'owned_output',side_effect=git), \
          mock.patch.object(c.a.pickle,'loads',side_effect=AssertionError('Unsafe decode')) as decode:
        with self.assertRaises(FileNotFoundError):c.a.authenticate(root,c.RUN_GIT)
        decode.assert_not_called()

  def test_external_capture_full_cal_metadata_chain_and_corruption_before_decode(self):
    """Mock Git/proc; all branch files are deliberately opaque non-pickle bytes."""
    with tempfile.TemporaryDirectory() as directory:
      base=Path(directory);root=capture_fixture.make_failure_run(base/'run')
      writer=capture_fixture.write_json
      source_name='scripts/dl_r4_utility.py'
      source_bytes=(c.ROOT/source_name).read_bytes()
      pins={source_name:hashlib.sha256(source_bytes).hexdigest()}
      budget=c.a.read_json(root/'budget.json')
      budget.update(seconds=21000,monotonic_deadline=21000,source_files=pins,
          input_pins=dict(checkpoint_sha256=c.a.CHECKPOINT,config_sha256=c.a.CONFIG))
      writer(root/'budget.json',budget)
      resource=dict(physical_gpu=4,logical_server='sv3',gpu_uuid='fixture-gpu',
          compute_uuid='fixture-gpu',egl_uuid='fixture-gpu')
      writer(root/'model_authenticated.json',dict(metadata=dict(checkpoint_sha256=c.a.CHECKPOINT,
          config_sha256=c.a.CONFIG,params_sha256=c.a.PARAMS),resource=resource))
      runtime=dict(pid=247226,physical_gpu=4,compute_uuid=['fixture-gpu'],graphics_verified=True,
          live_state=dict(gpu_uuid='fixture-gpu',processes=[dict(pid=247226)]))
      writer(root/'runtime_cuda_egl.json',runtime)
      (root/'resources.jsonl').write_text(json.dumps(dict(physical_gpu=4,gpu_uuid='fixture-gpu',
          processes=[dict(pid=247226)]))+'\n',encoding='utf-8')
      episodes=statistics_fixture()['episodes'][:8]
      episodes[0]['selected'][0]['qq_null_return']+=.3
      evidence=[]
      def artifact(epdir,name):
        path=epdir/name;path.write_bytes(b'intentional opaque fixture, never decode')
        return dict(path=name,sha256=capture_fixture.file_sha(path))
      for ep,episode in enumerate(episodes):
        epdir=root/f'episode_{ep:03}'
        source=artifact(epdir,'source.pkl');signals=artifact(epdir,'signals.pkl')
        _,selected=c.s.selected_positions(episode['candidates'])
        branches=[]
        for t in selected:
          for pair in range(16):
            for arm in c.a.ARMS:
              branches.append(dict(position=t,pair=pair,arm=arm,
                  **artifact(epdir,f't{t:03}_pair{pair:02}_{arm}.pkl')))
          branches.append(dict(position=t,pair=0,arm='qq_null',
              **artifact(epdir,f't{t:03}_pair00_qq_null.pkl')))
        writer(epdir/'complete.json',dict(version=c.a.SCIENCE,episode_id=ep,
            model_full_state_unchanged=True,source=source,signals=signals,branches=branches,
            episode_started_monotonic=ep*2,episode_completed_monotonic=ep*2+1,
            branch_public_oracles=[artifact(epdir,f't{t:03}_branch_public_oracle.pkl') for t in selected]))
        evidence.append(dict(path=f'episode_{ep:03}/complete.json',sha256=capture_fixture.file_sha(epdir/'complete.json')))
      writer(root/'calibration_frozen.json',dict(version=c.a.SCIENCE,held_episodes_generated=0,
          calibration_evidence=evidence,calibration_episodes=episodes,sealed_monotonic=15.5,
          frozen_models=c.s.freeze_calibration(episodes)))
      terminal=c.a.read_json(root/'supervisor_terminal.json')
      terminal['resources_sha256']=capture_fixture.file_sha(root/'resources.jsonl')
      writer(root/'supervisor_terminal.json',terminal)
      external=base/'capture.json'
      with mock.patch.object(capture_fixture.c,'git_binding',return_value=dict(sourcepins=pins,audit_sourcepins={})), \
          mock.patch.object(capture_fixture.c,'process_set',side_effect=capture_fixture.absent_processes):
        capture_fixture.c.capture(root,external,c.RUN_GIT,'a'*40)
      capture_sha=capture_fixture.file_sha(external)
      def git(command):
        if command[1]=='rev-parse':return ('a'*40+'\n').encode()
        if command[1]=='status':return b''
        if command[1]=='ls-tree':return (source_name+'\n').encode()
        return source_bytes
      with mock.patch.object(c.a,'owned_output',side_effect=git), \
          mock.patch.object(c.a.pickle,'loads',side_effect=AssertionError('No decode before all hashes')) as decode:
        _,capture,data,artifacts=c.authenticate(root,external,capture_sha,c.RUN_GIT,'a'*40)
        self.assertEqual(len(data['episodes']),8)
        self.assertEqual(capture['held_episode_directories'],[])
        self.assertEqual(len(artifacts),8*(2+3+147))
        decode.assert_not_called()
        (root/'episode_003/t512_pair00_qq_null.pkl').write_bytes(b'changed opaque branch')
        with self.assertRaisesRegex(ValueError,'hash differs'):
          c.authenticate(root,external,capture_sha,c.RUN_GIT,'a'*40)
        decode.assert_not_called()


if __name__ == '__main__':unittest.main()
