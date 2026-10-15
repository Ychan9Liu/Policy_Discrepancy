"""Generated/mocked artifact interfaces and NumPy orchestration, never real data."""
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from analysis import dl_normal_readout_run as r
from scripts import dl_normal_feature_run as f
from tests.test_dl_normal_feature_run import generated_worker_graph

CPU_ENV = dict(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',MUJOCO_GL='disable')


def write(path,value): path.write_bytes(r.canonical(value))


def graph(root):
  """The old compact512 fabricated metadata fixture, not source certification."""
  index,labels = generated_worker_graph(root)
  for row in index['rows']:
    row.update(family=row['global_unit']//16,replicate=row['global_unit']%16)
  generation = dict(version=f.VERSION,git_sha=r.PRODUCER_GIT,role='reference0',complete_units=512,complete_families=32,
      fit_families=list(range(8)),held_families=list(range(8,32)),source_complete_and_stable=True,
      labels_complete_and_stable=True,real_N1M_inputs_authenticated=True,whole_seconds=12600,elapsed_seconds=2.,
      independent_NN_reexecution=False,physics_reexecuted=False,readout_fitted=False,scientific_verdict=False,
      subsequent_learning_authorized=False,render_calls=0,training_updates=0,output_inventory=f.output_inventory(root),
      budget=r.pin(root/'budget.json'),source_index=r.pin(root/'source_index.json'),worker_complete=r.pin(root/'worker_complete.json'))
  write(root/'generation_complete.json',generation)
  parent=dict(pid=123,pgid=123,sid=123,start_ticks=1)
  supervisor=dict(version=f.VERSION,git_sha=r.PRODUCER_GIT,worker_returncode=0,failure=None,owned_groups_cleaned=True,
      generation_complete=r.pin(root/'generation_complete.json'),elapsed_seconds=2.1,parent_identity=parent)
  write(root/'supervisor_terminal.json',supervisor)
  allfiles=r.inventory(root)
  proof=dict(generation=r.pin(root/'generation_complete.json'),supervisor=r.pin(root/'supervisor_terminal.json'),
      inventory_sha256=r.digest(r.canonical(allfiles)),files=len(allfiles['files']),directories=512,
      **{name:generation[name] for name in ('budget','source_index','worker_complete')})
  terminal=dict(output_proof=proof,parent_identity=parent,elapsed_seconds=2.2)
  return terminal,index,labels


class ReadoutRunTests(unittest.TestCase):
  def setUp(self):
    self.environment=mock.patch.dict(os.environ,CPU_ENV);self.environment.start()
    self.removed={name:os.environ.pop(name,None) for name in ('MUJOCO_EGL_DEVICE_ID','PYOPENGL_PLATFORM')}
  def tearDown(self):
    self.environment.stop()
    for name,value in self.removed.items():
      if value is not None:os.environ[name]=value

  def test_import_is_stdlib_and_explicit_budget_and_outer_pins_are_required(self):
    result=subprocess.run([sys.executable,'-c',"import sys;import analysis.dl_normal_readout_run;assert not any(n in sys.modules for n in ('jax','numpy','mujoco','dm_control'))"],
        cwd=r.ROOT,capture_output=True,text=True,timeout=30)
    self.assertEqual(result.returncode,0,result.stderr)
    result=subprocess.run([sys.executable,str(Path(r.__file__))],cwd=r.ROOT,capture_output=True,text=True,timeout=30)
    self.assertEqual(result.returncode,2)
    for argument in ('--seconds','--feature-controller-sha','--feature-outer-sha'):
      self.assertIn(argument,result.stderr)
    with mock.patch.object(r,'DEADLINE',0):
      with self.assertRaisesRegex(ValueError,'deadline'):r.pin(__file__)

  def test_actual_outer_authority_is_distinct_from_successful_inner_terminal(self):
    with tempfile.TemporaryDirectory() as temp:
      root=Path(temp);run=root/'features';run.mkdir();inputs=root/'inputs';inputs.mkdir()
      parent=dict(pid=123,parent_pid=100,pgid=123,sid=123,start_ticks=5)
      child=dict(pid=124,parent_pid=123,pgid=124,sid=124,start_ticks=6)
      terminal=dict(version='DL-normal-reference0-launcher-r1',git_sha=r.PRODUCER_GIT,run_directory=str(run),
          role='reference0',whole_seconds=12600,parent_returncode=0,parent_terminal=True,success=True,
          parent_identity_gone=True,owned_groups_cleaned=True,external_failure=None,cleanup_errors=[],remaining_helper_registry=[],
          elapsed_seconds=10.,physical_gpu=r.GPU,gpu_uuid=r.GPU_UUID,parent_identity=parent,owned_children=[child],
          scope=dict(complete_units=512,fit_families=8,held_families=24,model_role='reference0',
              independent_NN_reexecution=False,training_updates=0,readout_fitted=False,scientific_verdict=False,subsequent_learning_authorized=False))
      path=inputs/'terminal.json';write(path,terminal)
      outer=dict(git_sha=r.PRODUCER_GIT,command=r.OUTER_COMMAND,returncode=0,physical_gpu=r.GPU,external_failure=None,
          actual_outer_complete_within12600=True,elapsed_seconds=11.,log_sha256='a'*64,controller_terminal=r.pin(path))
      external=root/'outer.json';write(external,outer)
      args=SimpleNamespace(features_run=str(run),feature_controller_terminal=str(path),feature_controller_sha=r.pin(path)['sha256'],
          feature_outer_execution=str(external),feature_outer_sha=r.pin(external)['sha256'])
      with mock.patch.object(r,'FEATURE_RUN',run),mock.patch.object(r,'FEATURE_INPUTS',inputs),mock.patch.object(r,'inactive') as retired:
        r.authority(args);self.assertEqual(retired.call_count,2)
        for key,value in (('returncode',1),('returncode',False),('physical_gpu',4),('physical_gpu',5),('elapsed_seconds',12600.01),
            ('actual_outer_complete_within12600',False),('controller_terminal',{}),('command',['ssh','wrong'])):
          write(external,dict(outer,**{key:value}));args.feature_outer_sha=r.pin(external)['sha256']
          with self.subTest(key=key,value=value),self.assertRaisesRegex(ValueError,'SSH exit0'):
            r.authority(args)
        write(external,outer);args.feature_outer_sha=r.pin(external)['sha256']
        # A successful-looking old GPU4/attempt01 terminal cannot stand in for
        # this fresh GPU6 attempt. Re-signing generated metadata does not make
        # its fixed execution identity valid.
        old_run=r.BASE/'runs/dl-normal-reference0-features-20261015-2a15c91/attempt01'
        for change in (dict(run_directory=str(old_run)),dict(physical_gpu=4),
            dict(gpu_uuid='GPU-9c070d5a-ac68-4c26-9522-a790ad874b23'),
            dict(success=False),dict(parent_returncode=1)):
          write(path,dict(terminal,**change));args.feature_controller_sha=r.pin(path)['sha256']
          write(external,dict(outer,controller_terminal=r.pin(path)));args.feature_outer_sha=r.pin(external)['sha256']
          with self.subTest(old_or_failed=change),self.assertRaisesRegex(ValueError,'controller terminal'):
            r.authority(args)
        original_run=args.features_run;args.features_run=str(old_run)
        with self.assertRaisesRegex(ValueError,'fixed reference0 run'):r.authority(args)
        args.features_run=original_run
        write(path,terminal);args.feature_controller_sha=r.pin(path)['sha256']
        write(external,outer);args.feature_outer_sha=r.pin(external)['sha256']
        path.write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError,'raw SHA'):r.authority(args)

  def test_whole_graph_before_stage_view_and_no_hidden_same_basename(self):
    with tempfile.TemporaryDirectory() as temp:
      root=Path(temp).resolve();terminal,index,labels=graph(root)
      first,_,_=r.validate_graph(root,terminal)
      _,complete=r.verify_worker_stage(root,first,index,labels)
      self.assertEqual(complete['complete_units'],512)
      self.assertFalse(any(p.relative_to(root).as_posix() in r.STAGE_ONLY for p in r.WorkerView(root).rglob('*')))
      nested=root/'unit_000/generation_complete.json';nested.write_bytes(b'{}')
      self.assertIn(nested,list(r.WorkerView(root).rglob('*')))
      with mock.patch.object(np,'load') as load,self.assertRaisesRegex(ValueError,'inventory'):
        r.validate_graph(root,terminal)
      load.assert_not_called();nested.unlink()
      unknown=root/'unknown.json';unknown.write_bytes(b'{}')
      with self.assertRaisesRegex(ValueError,'inventory'):r.validate_graph(root,terminal)
      unknown.unlink()
      missing=root/'unit_511/history.json';body=missing.read_bytes();missing.unlink()
      with self.assertRaisesRegex(ValueError,'inventory'):r.validate_graph(root,terminal)
      missing.write_bytes(body)
      excluded=root/'supervisor_terminal.json';body=excluded.read_bytes();excluded.write_bytes(b'{}')
      with self.assertRaisesRegex(ValueError,'top-level terminal'):r.validate_graph(root,terminal)
      excluded.write_bytes(body)
      with mock.patch.object(f,'verify_worker',side_effect=lambda *_: (unknown.write_bytes(b'{}'),{})):
        with self.assertRaisesRegex(ValueError,'changed across'):r.verify_worker_stage(root,first,index,labels)

  def test_partial_generation_and_foreign_worker_keys_cannot_decode_as_complete(self):
    with tempfile.TemporaryDirectory() as temp:
      root=Path(temp).resolve();terminal,index,labels=graph(root)
      path=root/'generation_complete.json';value=r.decode(path.read_bytes());value['complete_units']=511;write(path,value)
      terminal['output_proof']['generation']=r.pin(path)
      with mock.patch.object(np,'load') as load,self.assertRaisesRegex(ValueError,'generation'):
        r.validate_graph(root,terminal)
      load.assert_not_called()
      # Existing verifier must retain its actual key and conditional carry checks.
      (root/'generation_complete.json').unlink();(root/'supervisor_terminal.json').unlink()
      receipt=root/'unit_000/complete.json';value=r.decode(receipt.read_bytes())
      archive=root/'unit_000/current.npz'
      with np.load(archive,allow_pickle=False) as saved:arrays={name:saved[name] for name in saved.files}
      arrays['latent_key']=np.array([999,45],np.uint32)
      with archive.open('wb') as stream:np.savez(stream,**arrays)
      value['feature_schema']['latent_key']=f.typed_spec(arrays['latent_key']);value['artifacts']['current.npz']=r.pin(archive);write(receipt,value)
      top=root/'worker_complete.json';value=r.decode(top.read_bytes());value['units'][0]=r.pin(receipt);write(top,value)
      with self.assertRaisesRegex(ValueError,'feature/oracle key'):f.verify_worker(r.WorkerView(root),index,labels)

  def test_bf16_bits_targets_and_original_probability_arrays_are_used(self):
    with tempfile.TemporaryDirectory() as temp:
      root=Path(temp).resolve();terminal,index,labels=graph(root)
      for row,label in zip(index['rows'],labels['rows']):
        label['current_values']=dict(velocity_x=float(row['global_unit']),upright=.75)
        path=root/f"unit_{row['global_unit']:03d}/complete.json";value=r.decode(path.read_bytes())
        value['current_label_sha256']=r.digest(r.canonical(label));write(path,value)
      archive=root/'unit_000/current.npz'
      with np.load(archive,allow_pickle=False) as saved:arrays={name:saved[name] for name in saved.files}
      arrays['h'][:3]=np.array([0x8000,0x3f80,0xc020],np.uint16)
      with archive.open('wb') as stream:np.savez(stream,**arrays)
      receipt=root/'unit_000/complete.json';value=r.decode(receipt.read_bytes())
      import ml_dtypes
      value['feature_schema']['h']=f.typed_spec(arrays['h'].view(ml_dtypes.bfloat16));write(receipt,value)
      auth=dict(index=index,labels=labels,inventory=r.inventory(root))
      features=r.feature_arrays(auth)
      self.assertTrue(np.signbit(features['h'][0,0]));np.testing.assert_array_equal(features['h'][0,1:3],[1.,-2.5])
      self.assertEqual(features['h'].dtype,np.float64);self.assertEqual(features['q_prob'].dtype,np.float64)
      np.testing.assert_array_equal(features['q_prob'][0],arrays['q_prob'].astype(np.float64))
      self.assertEqual(features['targets'][511,0],511.)
      labels['rows'][0]['source_id']='wrong'
      with self.assertRaisesRegex(ValueError,'source/order'):r.feature_arrays(auth)
      labels['rows'][0]['source_id']=index['rows'][0]['source_id']
      archive.write_bytes(b'changed')
      with self.assertRaisesRegex(ValueError,'raw SHA'):r.feature_arrays(auth)

  def test_mocked_provenance_full_graph_wires_original_cpu_stage_without_model_import(self):
    # Source/Git/physical-label authorities are explicitly mocked; actual graph,
    # worker argv, byte pins,512 receipts/tensors and stage verifier remain real.
    with tempfile.TemporaryDirectory() as temp:
      root=Path(temp)/'features';root.mkdir();external=Path(temp)/'inputs';external.mkdir()
      terminal,index,labels=graph(root)
      for name in r.STAGE_ONLY:(root/name).unlink()
      inputs={}
      for name in ('checkpoint','config','launcher_terminal','launcher_launch','audit_report','audit_terminal','audit_launch',
          'labels','labels_terminal','labels_outer_execution'):
        path=external/(name+'.json');write(path,dict(generated=name));inputs[name]=r.pin(path)
      index.update(run_directory=str(Path(temp)/'original'),external_pins={name:inputs[name] for name in
          ('launcher_terminal','launcher_launch','audit_report','audit_terminal','audit_launch')})
      write(root/'source_index.json',index)
      plan=dict(version=f.PLAN_VERSION,git_sha=r.PRODUCER_GIT,role='reference0',logical_server='sv3',
          physical_gpu=r.GPU,gpu_uuid=r.GPU_UUID,whole_seconds=12600,complete_units=512,
          labels_outer_execution=inputs['labels_outer_execution'])
      path=external/'execution-plan.json';write(path,plan);inputs['execution_plan']=r.pin(path)
      labelpins=dict(output=inputs['labels'],terminal=inputs['labels_terminal'],outer_execution=inputs['labels_outer_execution'])
      parent=dict(pid=123,pgid=123,sid=123,start_ticks=1)
      worker=dict(pid=124,pgid=124,sid=124,start_ticks=2)
      budget=dict(version=f.VERSION,git_sha=r.PRODUCER_GIT,role='reference0',complete_units=512,plan=plan,
          source_files={'explicit_mock_source':{}},parent_identity=parent,source_index=r.pin(root/'source_index.json'),
          entry_started=1.,monotonic_deadline=12601.,execution_plan=inputs['execution_plan'],labels=labelpins,
          inputs={n:inputs[n] for n in ('checkpoint','config')})
      write(root/'budget.json',budget)
      values=dict(run=index['run_directory'],expected_git=r.PRODUCER_GIT,prior_physics_audit_git=r.PRIOR_GIT,
          launcher_terminal=inputs['launcher_terminal']['path'],launcher_sha=inputs['launcher_terminal']['sha256'],
          launcher_launch=inputs['launcher_launch']['path'],launcher_launch_sha=inputs['launcher_launch']['sha256'],
          audit_report=inputs['audit_report']['path'],audit_report_sha=inputs['audit_report']['sha256'],
          audit_terminal=inputs['audit_terminal']['path'],audit_terminal_sha=inputs['audit_terminal']['sha256'],
          audit_launch=inputs['audit_launch']['path'],audit_launch_sha=inputs['audit_launch']['sha256'],
          labels=inputs['labels']['path'],labels_sha=inputs['labels']['sha256'],labels_terminal=inputs['labels_terminal']['path'],
          labels_terminal_sha=inputs['labels_terminal']['sha256'],checkpoint=inputs['checkpoint']['path'],config=inputs['config']['path'],
          execution_plan=inputs['execution_plan']['path'],execution_plan_sha=inputs['execution_plan']['sha256'],output=str(root))
      command=[str(r.BASE/'envs/dreamer/bin/python'),str(r.PRODUCER_CODE/'scripts/dl_normal_feature_run.py')]
      for name in f.COMMON:command+=['--'+name.replace('_','-'),values[name]]
      write(root/'process.json',dict(parent_identity=parent,worker_identity=worker,budget=r.pin(root/'budget.json'),
          command=command+['--worker','--budget-sha',r.pin(root/'budget.json')['sha256']],
          environment=dict(CUDA_VISIBLE_DEVICES=r.GPU_UUID,CUDA_DEVICE_ORDER='PCI_BUS_ID',JAX_PLATFORMS='cuda',
              MUJOCO_GL='disable',MUJOCO_EGL_DEVICE_ID=None,PYOPENGL_PLATFORM=None)))
      for name in ('runtime_before.json','runtime_after.json'):
        write(root/name,dict(pid=124,compute_uuid=[r.GPU_UUID],physical_gpu=r.GPU,egl_uuid=None,graphics_verified=False))
      initial=r.decode((root/'initial.json').read_bytes());initial['model'].update(checkpoint=inputs['checkpoint'],config_pin=inputs['config'])
      write(root/'initial.json',initial)
      complete=r.decode((root/'worker_complete.json').read_bytes())
      complete.update(git_sha=r.PRODUCER_GIT,labels=labelpins,full_state_sha256=f.FULL_STATE,params_sha256=f.PARAMS,
          source_index_sha256=r.pin(root/'source_index.json')['sha256'],model_calls_readonly=True,readout_fitted=False,
          scientific_verdict=False,training_updates=0,physics_reexecuted=False,render_calls=0,
          artifacts={name:r.pin(root/name) for name in complete['artifacts']})
      write(root/'worker_complete.json',complete)
      generation=dict(version=f.VERSION,git_sha=r.PRODUCER_GIT,role='reference0',complete_units=512,complete_families=32,
          fit_families=list(range(8)),held_families=list(range(8,32)),source_complete_and_stable=True,
          labels_complete_and_stable=True,real_N1M_inputs_authenticated=True,whole_seconds=12600,elapsed_seconds=2.,
          independent_NN_reexecution=False,physics_reexecuted=False,readout_fitted=False,scientific_verdict=False,
          subsequent_learning_authorized=False,render_calls=0,training_updates=0,output_inventory=f.output_inventory(root),
          budget=r.pin(root/'budget.json'),source_index=r.pin(root/'source_index.json'),worker_complete=r.pin(root/'worker_complete.json'),labels=labelpins)
      write(root/'generation_complete.json',generation)
      write(root/'supervisor_terminal.json',dict(version=f.VERSION,git_sha=r.PRODUCER_GIT,worker_returncode=0,
          failure=None,owned_groups_cleaned=True,generation_complete=r.pin(root/'generation_complete.json'),elapsed_seconds=2.1,parent_identity=parent,plan=plan))
      actual=r.inventory(root)
      terminal.update(inputs=inputs,source_files=budget['source_files'],
          owned_children=[dict(worker,parent_pid=123)],output_proof=dict(
              generation=r.pin(root/'generation_complete.json'),supervisor=r.pin(root/'supervisor_terminal.json'),
              inventory_sha256=r.digest(r.canonical(actual)),files=len(actual['files']),directories=512,worker_identity=worker,
              public_policy_calls=513,**{n:generation[n] for n in ('budget','source_index','worker_complete')}))
      args=SimpleNamespace(features_run=str(root))
      with mock.patch.object(r,'authority',return_value=(terminal,{})),mock.patch.object(r,'producer_binding',return_value={}), \
          mock.patch.object(r,'validate_index'),mock.patch.object(r,'labels_for',return_value=labels), \
          mock.patch.object(r,'CHECKPOINT',inputs['checkpoint']['sha256']),mock.patch.object(r,'CONFIG',inputs['config']['sha256']):
        result=r.authenticate(args)
      self.assertEqual(len(result['index']['rows']),512)
      self.assertIn('official unimix',result['probability_source'])
      r.cpu_only()

  def test_physical_target_hex_and_torso_frame_rejections_after_mocked_label_authority(self):
    row=dict(plan=dict(t=64));values=dict(velocity_x=-.0,upright=.75)
    label=dict(velocity_frame='torso-local-velocimeter-x',current_source_raw_used=False,replayed_steps=64,official_resets=1,
        render_calls=0,query_readonly_snapshot_and_model_byte_exact=True,current_values=values,
        current_f64_hex={name:np.float64(value).tobytes().hex() for name,value in values.items()})
    report=dict(git_sha=r.LABELS_GIT,GPU_used=False,D_e_Y_used_for_labels_or_selection=False,scientific_verdict=False,
        subsequent_learning_authorized=False,model_calls=0,render_calls=0,rows=[label])
    pins={name:dict(path=name,sha256='a'*64) for name in ('output','terminal','outer_execution')}
    generation=dict(labels=pins);budget=dict(labels=pins,plan={})
    with mock.patch.object(f,'labels_authority',return_value=(pins,report)):
      r.labels_for(dict(rows=[row]),generation,budget)
      label['current_values']['velocity_x']=0.
      with self.assertRaisesRegex(ValueError,'F64 hex'):r.labels_for(dict(rows=[row]),generation,budget)
      label['current_values']['velocity_x']=-0.;label['velocity_frame']='world-x'
      with self.assertRaisesRegex(ValueError,'Physical target'):r.labels_for(dict(rows=[row]),generation,budget)

  def test_fixed_numpy_orchestration_keeps_heads_all_rows_risks_and_numeric_closure(self):
    from analysis import dl_normal_readout as math
    h=np.zeros((512,4096),np.float64);h[:,0]=np.arange(512)/512;h[:,1]=np.sin(np.arange(512))
    q=np.full((512,32,32),1/32,np.float64);q[:,0,0]+=h[:,0]/100;q[:,0,1]-=h[:,0]/100
    p=q.copy();p[:,0,0]-=h[:,0]/200;p[:,0,1]+=h[:,0]/200
    y=np.c_[h[:,0]**2,.5+.1*h[:,1]]
    features=dict(h=h,q_prob=q,p_prob=p,targets=y,families=np.repeat(np.arange(32),16))
    with mock.patch.object(math,'fit',wraps=math.fit) as fit:
      result=r.calculate(features)
    self.assertEqual(fit.call_count,3)
    self.assertTrue(all(call.args[0].shape==(128,4096) and call.args[2].shape==(128,2) for call in fit.call_args_list))
    for name in ('h','hq','hp'):
      self.assertEqual(result['heads'][name]['predictions'].shape,(512,2))
      self.assertEqual(result['heads'][name]['fitted']['weights'].shape[1],2)
      self.assertEqual(len(result['absolute_risks'][name]['fit']['families']),8)
      self.assertEqual(len(result['absolute_risks'][name]['held']['families']),24)
    comparison=result['comparisons']['hq_over_hp']['held']
    np.testing.assert_allclose(comparison['mean'],result['absolute_risks']['hp']['held']['mean']-result['absolute_risks']['hq']['held']['mean'],atol=1e-14)
    self.assertEqual((comparison['draws'],comparison['seed']),(10000,20261040))
    with tempfile.TemporaryDirectory() as temp:
      root=Path(temp);item=r.write_result(root,result);self.assertEqual(item,r.pin(root/'readout.json'))
      meta=r.decode(r.bound(item))
      with np.load(io.BytesIO(r.bound(meta['numeric_artifact'])),allow_pickle=False) as stored:
        def verify(value):
          if type(value) is dict and 'array' in value:
            data=stored[value['array']];self.assertEqual(r.digest(data.tobytes()),value['raw_sha256'])
          elif type(value) is dict:
            for child in value.values():verify(child)
          elif type(value) is list:
            for child in value:verify(child)
        verify(meta)
    self.assertFalse(result['scientific_verdict'])

  def test_identity_source_order_probability_inputs_and_output_escape_rejections(self):
    from scripts import dl_normal_source_index as source
    with self.assertRaisesRegex(ValueError,'identity'):r.identity(dict(pid=1,pgid=1,sid=2,start_ticks=3))
    # Production source validator authenticates identity/order, never just counts.
    with mock.patch.object(source,'validate_order',side_effect=ValueError('Source identity/order')) as validate:
      index=dict(version=source.VERSION,metadata_authentication_git=r.PRODUCER_GIT,run_git=r.RUN_GIT,
          prior_physics_audit_git=r.PRIOR_GIT,complete_units=512,complete_families=32,fit_rows=128,held_rows=384,
          fit_families=list(range(8)),held_families=list(range(8,32)),rows=[{}]*512,run_inventory_sha256='a'*64,
          run_directory=str(r.BASE/'runs/dl-r5-live-r2-20261015-1584ef3/attempt01'),source_kind='complete-audited-R5-pretreatment',
          source_graph_and_prior_physics_audit_authenticated=True,source_recomputed_latents=False,physics_reexecuted=False,
          model_executed=False,scientific_verdict_recomputed=False,source_selected_by_scientific_sign=False,
          GPU_used=False,subsequent_learning_authorized=False)
      with self.assertRaisesRegex(ValueError,'identity/order'):r.validate_index(index)
      validate.assert_called_once()
    with tempfile.TemporaryDirectory() as temp:
      root=Path(temp);immutable=root/'input';immutable.mkdir()
      with self.assertRaisesRegex(ValueError,'outside'):r.fresh_output(root/'x/../input/new',[immutable])
      self.assertFalse((immutable/'new').exists())
    with mock.patch.dict(sys.modules,{'jax':object()}):
      with self.assertRaisesRegex(ValueError,'runtimes'):r.cpu_only()


if __name__ == '__main__': unittest.main()
