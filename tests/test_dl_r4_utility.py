"""DL-r4 runner boundaries; no checkpoint, renderer or GPU is loaded."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from scripts import dl_r4_utility as r


class RunnerTests(unittest.TestCase):
  def test_duplicate_and_nonfinite_json_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)/'input.json'
      for bad in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
        path.write_text(bad, encoding='utf-8')
        with self.assertRaises(ValueError): r.read_json(path)

  def test_bad_input_rejected_before_pickle_or_model(self):
    with tempfile.TemporaryDirectory() as directory:
      cp, cfg = Path(directory)/'agent.pkl', Path(directory)/'config.yaml'
      cp.write_bytes(b'not the pinned project checkpoint')
      cfg.write_text('not: the pinned config', encoding='utf-8')
      with mock.patch.object(r.pickle, 'loads', side_effect=AssertionError('must not decode')) as decode:
        with self.assertRaisesRegex(ValueError, 'checkpoint bytes'):
          r.load_agent(cp, cfg, directory)
        decode.assert_not_called()

  def test_selection_does_not_read_utility_or_e_and_ties_are_fixed(self):
    candidates = [dict(position=t, D=1., e=999-t, utility=-t) for t in r.POSITIONS]
    selected, top = r.select_positions(candidates)
    self.assertEqual(top, [64,128])
    self.assertEqual(selected, [64,128,512])
    for x in candidates: x['e'] *= -100; x['utility'] *= -100
    self.assertEqual(r.select_positions(candidates), (selected,top))
    candidates[7]['D'] = 3.
    self.assertEqual(r.select_positions(candidates), ([64,512],[512,64]))
    for bad in (float('nan'),float('inf'),-1., True):
      candidates[0]['D'] = bad
      with self.assertRaises(ValueError): r.select_positions(candidates)

  def test_selection_refuses_reset_short_window_missing_or_reordered_candidates(self):
    rows = [dict(position=t,D=float(t)) for t in r.POSITIONS]
    for bad in (rows[:-1], rows[::-1], [dict(position=0,D=0.)]+rows[1:],
        rows[:-1]+[dict(position=960,D=0.)]):
      with self.assertRaises(ValueError): r.select_positions(bad)

  def test_late_process_never_spawned(self):
    with mock.patch.object(r.subprocess, 'Popen') as popen:
      with self.assertRaisesRegex(ValueError,'Late process'):
        r.spawn_owned(['unused'], time.monotonic()-1)
      popen.assert_not_called()

  def test_late_result_not_written_and_no_overwrite(self):
    with tempfile.TemporaryDirectory() as directory:
      path=Path(directory)/'result.json'
      with mock.patch.object(r,'DEADLINE',time.monotonic()-1):
        with self.assertRaisesRegex(ValueError,'deadline'): r.write_json(path, {'complete':True})
      self.assertFalse(path.exists())
      r.write_json(path, {'complete':False})
      with self.assertRaises(FileExistsError): r.write_json(path, {'complete':True})
      self.assertFalse(json.loads(path.read_text())['complete'])

  def test_phase_cannot_extend_whole_deadline(self):
    # Patch alarms: unit fixture never changes the real test process timer.
    before = time.monotonic()+100
    with mock.patch.object(r,'DEADLINE',before), mock.patch.object(r.signal,'setitimer',create=True):
      saved = r.arm_phase(10)
      self.assertEqual(saved,before)
      self.assertLess(r.DEADLINE,time.monotonic()+11)
      r.restore_phase(saved)
      self.assertEqual(r.DEADLINE,before)
      r.arm_phase(1000)
      self.assertEqual(r.DEADLINE,before)
      with self.assertRaises(ValueError): r.arm_phase(0)

  def test_telemetry_unknown_job_is_not_accepted_as_owned(self):
    xml=b'<nvidia_smi_log><gpu><uuid>GPU-test</uuid><processes><process_info><pid>22</pid><type>C</type></process_info></processes><fb_memory_usage><used>8 MiB</used></fb_memory_usage></gpu></nvidia_smi_log>'
    with mock.patch.object(r,'owned_output',return_value=xml):
      self.assertEqual(r.sample_resource(22)['processes'],[{'pid':22,'type':'C'}])
      with self.assertRaisesRegex(ValueError,'Unknown work'):r.sample_resource(33)
    with mock.patch.object(r,'owned_output',return_value=b'<nvidia_smi_log><gpu/></nvidia_smi_log>'):
      with self.assertRaisesRegex(ValueError,'inventory missing'):r.sample_resource(22)

  @unittest.skipUnless(os.name=='posix','Linux process group contract')
  def test_owned_group_grandchild_is_cleaned(self):
    child = r.spawn_owned([sys.executable,'-c',
        'import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]); '
        'print(p.pid,flush=True);time.sleep(60)'], time.monotonic()+10,
        stdout=subprocess.PIPE,text=True)
    grandchild = int(child.stdout.readline())
    try:
      self.assertIn(child.pid,r.OWNED)
      self.assertEqual(os.getpgid(grandchild),child.pid)
    finally:
      r.cleanup(child)
      child.stdout.close()
    self.assertIsNotNone(child.poll())
    self.assertNotIn(child.pid,r.OWNED)
    # Zombie grandchildren are terminal and cannot allocate or run code.
    status=Path(f'/proc/{grandchild}/status')
    if status.exists(): self.assertIn('State:\tZ',status.read_text())


if __name__=='__main__': unittest.main()
