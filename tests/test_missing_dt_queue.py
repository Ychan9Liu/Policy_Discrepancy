import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from scripts import queue_missing_dt as queue


class QueueTest(unittest.TestCase):
  def setUp(self):
    self.temp=TemporaryDirectory();self.root=Path(self.temp.name)
    self.directory=self.root/'cup'/'dt-attempt02';self.directory.mkdir(parents=True)
    self.wait=dict(directory=str(self.directory),task='dmc_cup_catch',code_sha='fixture-sha')
    self.plan=dict(wait_for=self.wait)
    self.prior=dict(self.wait,state='running',pid=123)
    (self.root/'actual-RUN_INDEX.json').write_text(json.dumps(dict(runs=[self.prior])))
    (self.root/'prelaunch.json').write_text(json.dumps(dict(supervisor_pid=999)))

  def tearDown(self):self.temp.cleanup()

  def test_exited_agent_still_waits_for_live_project_supervisor_audit(self):
    with patch.object(queue.os,'kill') as alive:
      self.assertEqual(queue.predecessor_stage(self.plan),'waiting_for_own_predecessor_or_its_audit')
    alive.assert_called_once_with(999,0)

  def test_ready_requires_original_task_sha_and_full_audit(self):
    row=dict(self.prior,state='run_audited_pending_group_and_WB',run_audit=dict(updates=249493,evaluations=21))
    path=self.directory/'execution-final.json';path.write_text(json.dumps(row))
    self.assertEqual(queue.predecessor_stage(self.plan),'ready')
    for key,value in [('state','failed'),('code_sha','other'),('directory','other'),('task','other'),('run_audit',dict(updates=1,evaluations=21))]:
      path.write_text(json.dumps(dict(row,**{key:value})))
      with self.assertRaises(RuntimeError):queue.predecessor_stage(self.plan)

  def test_failed_or_dead_supervisor_does_not_wait_for_arbitrary_GPU_jobs(self):
    with patch.object(queue.os,'kill',side_effect=ProcessLookupError):
      with self.assertRaises(ProcessLookupError):queue.predecessor_stage(self.plan)
    (self.root/'actual-RUN_INDEX.json').write_text(json.dumps(dict(runs=[dict(self.prior,state='failed')])))
    with self.assertRaises(RuntimeError):queue.predecessor_stage(self.plan)


if __name__=='__main__':unittest.main()
