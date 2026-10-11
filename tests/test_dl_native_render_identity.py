"""Pure CPU engineering fixtures; no official env/render/GPU construction."""
import copy
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import dl_native_render_identity as d


class NumericStruct(SimpleNamespace):
  __module__='mujoco.fixture'


class FakeModel:
  def __init__(self):
    self.ptr=NumericStruct(tex_data=np.arange(8,dtype=np.uint8),mat_rgba=np.zeros((1,4),np.float32),
        cam_pos=np.zeros((3,3),np.float64),light_pos=np.zeros((1,3),np.float64),
        vis=NumericStruct(quality=NumericStruct(shadowsize=2048)),ncam=3)
  def id2name(self,index,kind): return 'fixture-camera'
  def to_bytes(self): return b'CPUfixtureonly'+self.ptr.tex_data.tobytes()


class FakePhysics:
  def __init__(self,captures,vary=False,mutate=False):
    self.model=FakeModel(); self.data=NumericStruct(**{name:np.zeros((3,3),np.float64) for name in d.DERIVED})
    self.captures=captures; self.vary=vary; self.mutate=mutate; self.buffer=np.zeros((64,64,3),np.uint8); self.calls=0
  def render(self,*args,**kwargs):
    before=hashlib.sha256(self.buffer.tobytes()).hexdigest(); self.calls+=1
    self.buffer[:]=self.calls if self.vary else 4
    if self.mutate: self.data.cam_xpos[0,0]+=1
    scene={'fixture/actual_scene':np.array(2,np.int32)}
    self.captures.append(dict(scene=dict(leaves=scene,manifest=d.leaf_manifest(scene),
        unknown=[dict(field='fixture',reason='not real renderer')]),old_buffer_sha256=before,
        post_buffer_sha256=hashlib.sha256(self.buffer.tobytes()).hexdigest(),
        returned_sha256=hashlib.sha256(self.buffer.tobytes()).hexdigest(),returned_shares_buffer=True))
    return self.buffer


class FakeAdapter:
  def __init__(self,captures,vary=False,mutate=False):
    self.suite=SimpleNamespace(physics=FakePhysics(captures,vary,mutate))
    self.recorder=SimpleNamespace(steps=0,last_action=None); self.resets=0
  def snapshot(self):
    return dict(suite=dict(integration=np.zeros(12,np.float64),task={'fixture_only':np.array(1,np.int32)},
        step_count=0,reset_next_step=False),from_dm_done=False)
  def reset(self):
    self.resets+=1
    return dict(image=self.suite.physics.render(64,64,camera_id=2),reward=np.float32(0),
        is_first=np.bool_(True),is_last=np.bool_(False),is_terminal=np.bool_(False))


class IdentityFixtures(unittest.TestCase):
  def test_parent_import_is_stdlib_only_and_has_no_execution(self):
    code="import os,sys; before=dict(os.environ); import scripts.dl_native_render_identity; assert before==dict(os.environ); assert 'numpy' not in sys.modules; assert 'jax' not in sys.modules; assert not any(x.startswith('dm_control') for x in sys.modules)"
    result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=30)
    self.assertEqual(result.returncode,0,result.stdout+result.stderr)

  def test_typed_numeric_dtype_shape_signedzero_and_nonfinite(self):
    rows=[np.array(0,np.float32),np.array(-0.,np.float32),np.array(0,np.float64),np.array([0],np.float32)]
    self.assertEqual(len({d.manifest_digest(d.leaf_manifest({'x':x})) for x in rows}),4)
    for x in (np.array(np.nan),np.array(np.inf),np.array('x'),np.array([object()],object)):
      with self.subTest(value=str(x)),self.assertRaises(ValueError): d.array_record(x)

  def test_snapshot_schema_task_and_state_bytes(self):
    value=FakeAdapter([]).snapshot(); original=d.snapshot_bytes(value)
    for field in ('integration','task','clock','flag'):
      changed=copy.deepcopy(value)
      if field=='integration': changed['suite']['integration'][0]=-0.
      elif field=='task': changed['suite']['task']['fixture_only'][...]=2
      elif field=='clock': changed['suite']['step_count']=1
      else: changed['from_dm_done']=True
      self.assertNotEqual(d.snapshot_bytes(changed),original)
    for change in ('dtype','shape','nan','extra'):
      bad=copy.deepcopy(value)
      if change=='dtype': bad['suite']['integration']=bad['suite']['integration'].astype(np.float32)
      elif change=='shape': bad['suite']['integration']=bad['suite']['integration'].reshape(3,4)
      elif change=='nan': bad['suite']['integration'][0]=np.nan
      else: bad['extra']=1
      with self.subTest(change=change),self.assertRaises(ValueError): d.snapshot_bytes(bad)

  def test_rgb_shape_dtype_copy_difference_and_png(self):
    image=np.zeros((64,64,3),np.uint8); saved=d.rgb_copy(image); image[1,2,0]=9
    difference=d.rgb_difference(image,saved)
    self.assertEqual(difference,dict(byte_equal=False,changed_channels=1,changed_pixels=1,maximum_absolute_difference=9))
    self.assertFalse(np.shares_memory(saved,image)); self.assertTrue(d.png_bytes(saved).startswith(b'\x89PNG\r\n\x1a\n'))
    for bad in (image.astype(np.float32),image[:-1],image[:,:,0]):
      with self.assertRaises(ValueError): d.rgb_copy(bad)

  def test_reset_observation_strict_scalars_flags_and_reward(self):
    row=FakeAdapter([]).reset(); d.reset_observation(row)
    for name,value in (('reward',np.float64(0)),('reward',np.array([0],np.float32)),
        ('reward',np.float32(np.nan)),('is_first',np.int32(1)),('is_last',np.bool_(True))):
      bad=copy.deepcopy(row); bad[name]=value
      with self.subTest(name=name,value=str(value)),self.assertRaises(ValueError): d.reset_observation(bad)

  def test_complete_three_renders_and_alias_difference_are_valid_results(self):
    for vary in (False,True):
      with tempfile.TemporaryDirectory() as directory:
        captures=[]; adapter=FakeAdapter(captures,vary)
        row=d.collect_model(adapter,Path(directory),0,0,captures,check_callback=lambda:None)
        self.assertEqual((adapter.resets,adapter.suite.physics.calls,adapter.recorder.steps),(1,3,0))
        self.assertEqual(len(row['renders']),3); self.assertEqual(row['actor_calls'],0)
        self.assertIs(row['within_model'][0]['rgb_byte_equal'],not vary)
        self.assertFalse(row['complete_render_input_equivalence_proven'])
        if vary:
          self.assertTrue(row['renders'][1]['old_live_views_changed'][0])
          with (Path(directory)/row['renders'][0]['file']).open('rb') as stream: rgb=np.load(stream,allow_pickle=False)
          self.assertTrue(np.all(rgb==1))
        self.assertEqual(len(list(Path(directory).glob('*-snapshot.pkl'))),1)
        self.assertEqual(len(list(Path(directory).glob('*.npy'))),3)

  def test_visual_derived_mutation_is_not_a_valid_render(self):
    with tempfile.TemporaryDirectory() as directory:
      captures=[]; adapter=FakeAdapter(captures,mutate=True)
      with self.assertRaisesRegex(ValueError,'CPU visual/derived'):
        d.collect_model(adapter,Path(directory),0,0,captures,check_callback=lambda:None)

  def test_cross_model_classification_keeps_unknown_and_input_difference(self):
    row=dict(mjb_sha256='a',model_manifest={},model_unknown=[],derived_manifest={},derived_unknown=[],scene_manifest={},
        scene_unknown=[],rgb_sha256='b')
    changed=copy.deepcopy(row); changed['rgb_sha256']='c'
    result=d.compare_recorded(row,changed)
    self.assertEqual(result['classification'],'same_measured_CPU_inputs_RGB_difference_unresolved')
    self.assertFalse(result['graphical_noise_proven']); self.assertFalse(result['full_render_input_equivalence_proven'])
    changed['derived_manifest']={'camera':'different'}
    self.assertEqual(d.compare_recorded(row,changed)['classification'],'measured_CPU_numeric_input_difference')
    changed=copy.deepcopy(row); changed['mjb_sha256']='padding-different'
    result=d.compare_recorded(row,changed)
    self.assertEqual(result['classification'],'MJB_byte_difference_without_measured_numeric_difference_unresolved')
    self.assertFalse(result['mjb_equal']); self.assertTrue(result['typed_model_equal'])

  def test_opaque_or_nonfinite_optional_field_is_explicitly_unknown(self):
    obj=NumericStruct(valid=np.array(1,np.int32),invalid=np.array(np.nan),opaque=object())
    leaves,unknown={},[]; d.capture_object(obj,'fixture',leaves,unknown)
    self.assertEqual(set(leaves),{'fixture/valid'})
    self.assertEqual({row['field'] for row in unknown},{'fixture/invalid','fixture/opaque'})

  @unittest.skipUnless(os.name=='posix','POSIX signal mask ownership fixture')
  def test_spawn_registers_before_unblocking_and_child_restores_mask(self):
    previous=set(); events=[]; callbacks=[]; child=SimpleNamespace(pid=987654)
    def mask(how,values):
      events.append(('mask',how,set(values)))
      if how==d.signal.SIG_SETMASK and values==previous: self.assertIn(child.pid,d.OWNED)
      return previous
    def popen(command,**kwargs):
      self.assertTrue(kwargs['start_new_session']); self.assertIn('preexec_fn',kwargs)
      callbacks.append(kwargs['preexec_fn'])
      events.append(('spawn',command)); return child
    with mock.patch.object(d.signal,'pthread_sigmask',side_effect=mask),mock.patch.object(d.subprocess,'Popen',side_effect=popen):
      result=d.spawn_owned(['fixture'],time.monotonic()+1)
      callbacks[0]()
    self.assertIs(result,child); self.assertEqual([row[0] for row in events],['mask','spawn','mask','mask'])
    d.OWNED.discard(child.pid)

  def test_late_publication_cannot_leave_success_receipt(self):
    for clocks in ([2.],[.5,2.]):
      with tempfile.TemporaryDirectory() as directory,mock.patch.object(d.time,'monotonic',side_effect=clocks):
        output=Path(directory)
        self.assertEqual(d.publish_final(output,{'exit_code':0,'complete_verified':True},1.),124)
        self.assertFalse((output/'supervisor_complete.json').exists())
        self.assertTrue((output/'late-supervisor.json').exists())

  def test_owned_timeout_reaps_only_its_child(self):
    with tempfile.TemporaryDirectory() as directory:
      with (Path(directory)/'child.log').open('w') as log:
        terminal=d.run_owned([sys.executable,'-c','import time; time.sleep(60)'],log,
            time.monotonic()+.1,dict(os.environ))
    self.assertEqual(terminal['exit_code'],124); self.assertTrue(terminal['timed_out'])
    self.assertTrue(terminal['worker_terminal']); self.assertFalse(d.OWNED)

  def test_synthetic_completion_hash_chain_and_missing_timeout_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      output=Path(directory); git='a'*40; phases=[]; models=[]; files={}
      for worker in range(2):
        group=[]
        for index in range(2):
          captures=[]; adapter=FakeAdapter(captures,vary=True)
          row=d.collect_model(adapter,output,worker,index,captures,check_callback=lambda:None)
          group.append(row); models.append(row)
          for key,pinkey in (('snapshot_file','snapshot_sha256'),('mjb_file','mjb_file_sha256'),
              ('CPU_inputs_file','CPU_inputs_file_sha256')): files[row[key]]=row[pinkey]
          for frame in row['renders']:
            files[frame['file']]=frame['file_sha256']; files[frame['debug_png']]=frame['debug_png_sha256']
        phase=dict(worker=worker,worker_pid=9870+worker,worker_terminal=True,exit_code=0,timed_out=False,elapsed_seconds=.1)
        phases.append(phase); d.write_json(output/f'worker{worker}-exit.json',phase)
        complete=dict(version=d.VERSION,actual_git_sha=git,worker=worker,worker_pid=9870+worker,parent_pid=1234,
            complete_verified=True,actual_resets=2,actual_RGB=6,this_worker_cuda_compute_context_created=False,models=group)
        d.write_json(output/f'worker{worker}-complete.json',complete)
        files[f'worker{worker}-complete.json']=d.sha(output/f'worker{worker}-complete.json')
      budget=dict(version=d.VERSION,actual_git_sha=git,parent_pid=1234,seconds=180,max_gpu_devices=1,
          planned_physical_gpu=4,max_gpu_hours=.05)
      d.write_json(output/'budget.json',budget)
      result=dict(version=d.VERSION,actual_git_sha=git,actual_resets=4,actual_RGB=12,actual_suite_transitions=0,
          actual_actor_calls=0,agent_instances=0,checkpoint_reads=0,old_episode_reads=0,models=models,pairs=[{}]*6,
          root_cause_fully_identified=False,graphical_noise_proven=False,r1_reinterpreted=False,files_sha256=files,
          synthetic_fixture_only=True)
      d.write_json(output/'result.json',result)
      supervisor=dict(version=d.VERSION,actual_git_sha=git,parent_pid=1234,complete_verified=True,exit_code=0,
          timed_out=False,actual_gpu_devices_used=1,elapsed_parent_entry_seconds=.5,phases=phases,
          budget_sha256=d.sha(output/'budget.json'),result_sha256=d.sha(output/'result.json'))
      d.write_json(output/'supervisor_complete.json',supervisor)
      external=dict(actual_git_sha=git,worker_pid=1234,worker_terminal=True,exit_code=0,timed_out=False,elapsed_cli_seconds=.6)
      terminal=output/'external.json'; d.write_json(terminal,external)
      verified=d.verify_completion(output,terminal)
      self.assertTrue(verified['diagnostic_complete_external_exit_verified'])
      self.assertFalse(verified['model_or_renderer_independently_replayed'])
      terminal.unlink(); del external['timed_out']; d.write_json(terminal,external)
      with self.assertRaises((KeyError,ValueError)): d.verify_completion(output,terminal)
      terminal.unlink(); external['timed_out']=False; d.write_json(terminal,external)
      first=output/models[0]['renders'][0]['file']
      with first.open('ab') as stream: stream.write(b'tamper')
      with self.assertRaisesRegex(ValueError,'artifact bytes'): d.verify_completion(output,terminal)


if __name__=='__main__': unittest.main()
