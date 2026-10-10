"""DL-code-r1 pure mocked resource contracts; no real hardware is accessed."""

from contextlib import ExitStack
import datetime as dt
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import dl_resources as resource

UUID = 'GPU-fixture-assigned-4'
ENV = dict(CUDA_VISIBLE_DEVICES=UUID, MUJOCO_EGL_DEVICE_ID='4',
    CUDA_DEVICE_ORDER='PCI_BUS_ID', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl')


def record_fixture(probe=None):
  value = dict(logical_server='sv3', actual_hostname=resource.HOST,
      physical_gpu=4, gpu_uuid=UUID, compute_uuid=UUID, egl_uuid=UUID,
      cuda_visible_devices=UUID, mujoco_egl_device_id='4',
      checked_at_utc=resource.now().isoformat(), occupancy='idle', exclusive=True)
  if probe:
    value.update(device_probe_path=str(probe), device_probe_sha256=resource.digest(probe))
  return value


def probe_fixture():
  return dict(physical_gpu=4, uuid=UUID,
      verified_compute_and_graphics_same_gpu=True, training_actions=0,
      agent_updates=0, logical_cuda_device=0, pid=22,
      start_utc=resource.now().isoformat(), end_utc=resource.now().isoformat(),
      environment=ENV,
      source_sha256=resource.digest(resource.ROOT / 'scripts/probe_formal_devices.py'))


def fake_commands(command):
  if '--query-gpu=uuid,memory.used' in command:
    return UUID + ', 0\n'
  if '-x' in command:
    return '<nvidia_smi_log><gpu><uuid>' + UUID + '</uuid><processes></processes></gpu></nvidia_smi_log>'
  if 'pmon' in command:
    return '# gpu pid type\n0 900 C - - - -\n4 - - - - - -\n'
  if '--query-compute-apps=pid,gpu_uuid' in command:
    return ''
  raise AssertionError(command)


class DLResourceContracts(unittest.TestCase):

  def runtime_mocks(self):
    stack = ExitStack()
    stack.enter_context(patch.object(resource.platform, 'node', return_value=resource.HOST))
    stack.enter_context(patch.object(resource.platform, 'system', return_value='Linux'))
    stack.enter_context(patch.dict(resource.os.environ, ENV))
    stack.enter_context(patch.object(resource, 'command_output', side_effect=fake_commands))
    return stack

  def files(self, root):
    probe, record = Path(root) / 'probe.json', Path(root) / 'resource.json'
    resource.write_json(probe, probe_fixture())
    resource.write_json(record, record_fixture(probe))
    return probe, record

  def test_host_physical_uuid_and_all_environment_bindings(self):
    value = record_fixture()
    resource.validate_binding(value, ENV, resource.HOST, 'Linux')
    for field, changed in [('actual_hostname', 'other'), ('logical_server', 'sv2'),
        ('physical_gpu', 3), ('physical_gpu', True), ('compute_uuid', 'other'),
        ('egl_uuid', 'other'), ('cuda_visible_devices', '4')]:
      with self.assertRaises(ValueError):
        resource.validate_binding(dict(value, **{field: changed}), ENV, resource.HOST, 'Linux')
    for field in ENV:
      with self.assertRaises(ValueError):
        resource.validate_binding(value, dict(ENV, **{field: 'changed'}), resource.HOST, 'Linux')
    for hostname, system in [('other', 'Linux'), (resource.HOST, 'Windows')]:
      with self.assertRaises(RuntimeError):
        resource.validate_binding(value, ENV, hostname, system)

  def test_record_age_idle_and_unknown_are_fail_closed(self):
    instant = dt.datetime(2026, 10, 10, tzinfo=dt.timezone.utc)
    value = dict(record_fixture(), checked_at_utc=instant.isoformat())
    resource.validate_age(value, instant + dt.timedelta(seconds=120))
    for delta in (-6, 121):
      with self.assertRaises(ValueError):
        resource.validate_age(value, instant + dt.timedelta(seconds=delta))
    with self.assertRaises(ValueError):
      resource.validate_age(dict(value, occupancy='unknown'), instant)
    with self.assertRaises(ValueError):
      resource.validate_age(dict(value, exclusive=False), instant)

  def test_live_selected_physical_only_unknown_other_card_is_not_interfered_with(self):
    with self.runtime_mocks():
      state = resource.live_state(4)
      resource.require_idle(state, UUID)
      self.assertEqual(state['processes'], [])
      self.assertEqual(state['pmon_jobs'], [])
      self.assertIn('900 C', state['raw_pmon'])

  def test_occupied_memory_xml_or_pmon_and_uuid_mismatch_refuse(self):
    with self.runtime_mocks():
      state = resource.live_state(4)
      for changed in [dict(memory_used_mib=1), dict(processes=[{'pid': 9, 'type': 'G'}]),
          dict(pmon_jobs=[{'pid': 8, 'type': 'C'}]), dict(gpu_uuid='other')]:
        with self.assertRaises(RuntimeError):
          resource.require_idle(dict(state, **changed), UUID)

  def test_absent_xml_inventory_missing_pmon_row_and_non_numeric_memory_refuse(self):
    for replaced, output in [('-x', '<nvidia_smi_log><gpu><uuid>'+UUID+'</uuid></gpu></nvidia_smi_log>'),
        ('pmon', '# unavailable\n'), ('--query-gpu=uuid,memory.used', UUID+', N/A')]:
      def commands(command):
        return output if replaced in command else fake_commands(command)
      with patch.object(resource, 'command_output', side_effect=commands):
        with self.assertRaises(RuntimeError):
          resource.live_state(4)

  def test_initial_entry_reads_actual_probe_and_fresh_live_state(self):
    with tempfile.TemporaryDirectory() as root, self.runtime_mocks():
      probe, record = self.files(root)
      observed = resource.require_resource(record)
      self.assertEqual(observed['gpu_uuid'], UUID)
      self.assertEqual(observed['device_mapping_evidence']['sha256'], resource.digest(probe))
      self.assertEqual(observed['live_check']['memory_used_mib'], 0)
      self.assertEqual(observed['resource_source_sha256'], resource.digest(record))

  def test_changed_probe_bytes_or_wrong_probe_binding_refuse(self):
    with tempfile.TemporaryDirectory() as root, self.runtime_mocks():
      probe, record = self.files(root)
      data = probe_fixture()
      data['environment'] = dict(ENV, MUJOCO_EGL_DEVICE_ID='5')
      probe.write_text(json.dumps(data))
      with self.assertRaises(ValueError):
        resource.require_resource(record)
      value = record_fixture(probe)
      record.write_text(json.dumps(value))
      with self.assertRaises(ValueError):
        resource.require_resource(record)

  def test_paired_session_children_refresh_actual_occupancy_without_stale_age_fallback(self):
    with tempfile.TemporaryDirectory() as root, self.runtime_mocks():
      probe, record = self.files(root)
      session = Path(root) / 'session.json'
      with patch.object(resource.os, 'getpid', return_value=111):
        resource.start_session(record, session)
      value = json.loads(session.read_text())
      value['resource']['checked_at_utc'] = (resource.now()-dt.timedelta(hours=2)).isoformat()
      session.write_text(json.dumps(value))
      with patch.object(resource.os, 'getppid', return_value=111), patch.object(resource.os, 'kill') as kill:
        observed = resource.require_session(session)
        kill.assert_called_once_with(111, 0)
      self.assertNotEqual(observed['checked_at_utc'], value['resource']['checked_at_utc'])
      self.assertEqual(observed['live_check']['memory_used_mib'], 0)
      with patch.object(resource.os, 'getppid', return_value=222):
        with self.assertRaises(ValueError):
          resource.require_session(session)

  def test_runtime_actual_compute_context_must_match_physical_uuid(self):
    def commands(command):
      if '--query-compute-apps=pid,gpu_uuid' in command:
        return '111, '+UUID+'\n'
      if 'pmon' in command:
        return '# gpu pid type\n4 111 C+G 0 0 0 0\n'
      if '-x' in command:
        return ('<nvidia_smi_log><gpu><uuid>'+UUID+'</uuid><processes>'
            '<process_info><pid>111</pid><type>C+G</type></process_info>'
            '</processes></gpu></nvidia_smi_log>')
      return fake_commands(command)
    with self.runtime_mocks(), patch.object(resource.os, 'getpid', return_value=111), patch.object(
        resource, 'command_output', side_effect=commands):
      observed = resource.verify_runtime(record_fixture(), graphics=True)
      self.assertTrue(observed['graphics_verified'])
      with self.assertRaises(RuntimeError):
        resource.verify_runtime(record_fixture(), compute=False)
    with self.runtime_mocks(), patch.object(resource.os, 'getpid', return_value=111):
      with self.assertRaises(RuntimeError):
        resource.verify_runtime(record_fixture())

  def test_record_cli_reuses_real_probe_but_gets_current_idle_and_refuses_overwrite(self):
    with tempfile.TemporaryDirectory() as root, self.runtime_mocks():
      probe, _ = self.files(root)
      output = Path(root) / 'new-record.json'
      args = types.SimpleNamespace(physical_gpu=4, device_probe=probe, output=output)
      resource.record(args)
      result = json.loads(output.read_text())
      self.assertEqual(result['device_probe_sha256'], resource.digest(probe))
      self.assertIn('live_check', result)
      with self.assertRaises(FileExistsError):
        resource.record(args)

  def test_cpu_collect_and_score_are_guarded_before_any_model_or_render(self):
    spec = importlib.util.spec_from_file_location('dl_diag_guard_fixture',
        resource.ROOT / 'scripts/dl_diagnostic.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = types.SimpleNamespace(resource_record='fixture', platform='cpu')
    for function in (module.collect, module.score):
      with patch.object(resource, 'require_resource', side_effect=RuntimeError('fixture-denied')), patch.object(
          module, 'load_agent') as load:
        with self.assertRaisesRegex(RuntimeError, 'fixture-denied'):
          function(args)
        load.assert_not_called()


if __name__ == '__main__':
  unittest.main()
