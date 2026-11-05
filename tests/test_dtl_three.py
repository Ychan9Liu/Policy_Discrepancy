"""Closed run contract and durable DtL diagnostics, not performance evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dreamerv3.protocol_scope import validate_dtl_scope, DTL_THREE_TASKS
from embodied.run.protocol_v1 import ProtocolState
from embodied.run.dtl_three import dtl_receipt, audit_dtl_receipts
from scripts.prepare_m2_v1 import resolve
from scripts import run_dtl_three as runner


class DtlThreeTests(unittest.TestCase):
  def config(self, task='dmc_quadruped_run'):
    return resolve(['--configs', 'dtl_three', '--task', task,
        '--jax.prealloc', 'False', '--logger.outputs', 'jsonl', 'wandb'])

  def test_all_three_inherit_production_settings_and_gate(self):
    for task in DTL_THREE_TASKS:
      c = self.config(task)
      runner.check_config(c)
      self.assertEqual((c.agent.dt_latch.alpha, c.agent.dt_latch.rho), (20., .5))
      self.assertFalse(c.agent.dt_latch.normal_only)
      self.assertEqual(c.run.action_budget, 1000000)
      self.assertEqual((c.run.eval_every_actions, c.run.eval_eps), (50000, 10))
      self.assertEqual((c.agent.dyn.rssm.deter, c.batch_size, c.batch_length), (4096,16,64))
    for delta in ({'seed': 1}, {'task': 'dmc_hopper_hop'},
        {'agent.dt_latch.rho': .6}, {'agent.dt_latch.normal_only': True},
        {'agent.rep_probe.mode': 'dt'}, {'env.dmc.background.mode': 'static'},
        {'run.action_budget': 100000}, {'agent.ac_grads': True}):
      with self.assertRaises(ValueError):
        runner.check_config(self.config().update(delta))

  def test_m2_scope_does_not_silently_allow_dtl(self):
    c = resolve(['--configs', 'm2_v1'])
    validate_dtl_scope(c)
    with self.assertRaisesRegex(ValueError, 'do not authorize'):
      validate_dtl_scope(c.update({'agent.dt_latch.mode': 'dtlatch'}))

  def test_capacity_exception_is_explicit_and_bounded(self):
    with self.assertRaises(RuntimeError):
      runner.check_capacity(175_000_000_000)
    self.assertEqual(runner.check_capacity(175_000_000_000,
        accept_initial_capacity_deficit=True), 180_000_000_000)
    with self.assertRaises(RuntimeError):
      runner.check_capacity(4_000_000_000, accept_initial_capacity_deficit=True)
    with self.assertRaises(RuntimeError):
      runner.check_capacity(8_000_000_000, engineering=True,
          accept_initial_capacity_deficit=True)

  def test_prepared_plan_exact_cards_destinations_and_no_overwrite(self):
    with tempfile.TemporaryDirectory() as tmp:
      out = Path(tmp)/'plan'
      with patch.object(runner, 'clean_sha', return_value='fixture-sha'):
        runner.prepare(out, 'dtl-three-clean-seed0-fixture')
        with self.assertRaises(FileExistsError):
          runner.prepare(out, 'dtl-three-clean-seed0-fixture')
      plan = json.loads((out/'plan.json').read_text())
      runner.validate(plan)
      self.assertEqual([r['physical_gpu'] for r in plan['runs']], [5,6,7])
      self.assertEqual([r['state'] for r in plan['runs']], ['prepared_not_started']*3)
      for key, value in (('physical_gpu',4),('seed',1),('directory','/existing'),
                         ('task','dmc_hopper_hop'),('code_sha','other')):
        bad = copy.deepcopy(plan)
        bad['runs'][0][key] = value
        with self.assertRaises(ValueError):
          runner.validate(bad)
      bad = copy.deepcopy(plan)
      bad['runs'][0]['environment']['CUDA_VISIBLE_DEVICES'] = runner.UUIDS[1]
      with self.assertRaises(ValueError):
        runner.validate(bad)

  def test_nontrivial_diagnostics_and_invalid_values(self):
    metrics = {'dl/'+k: .25 for k in ('active_frac', 'valid_frac',
        'valid_active_frac', 'active_protected_frac', 'excess_release_mean',
        'weighted_excess_mean', 'rep_before_mean', 'rep_after_mean')}
    metrics.update({'dl/actual_v_mean': .8, 'dl/actual_v_min': .6,
                    'dl/actual_v_max': 1., 'opt/loss': 2., 'opt/grad_norm': 3.})
    row = dtl_receipt(3, 2044, metrics)
    self.assertEqual(row['update_id'], 3)
    self.assertLess(row['actual_v_mean'], 1)
    self.assertGreater(row['excess_release_mean'], 0)
    for key, value in (('dl/actual_v_min', .49),('opt/loss', float('nan'))):
      with self.assertRaises(RuntimeError):
        dtl_receipt(3,2044,dict(metrics, **{key:value}))

  def test_resume_receipts_no_duplicates_missing_or_future_updates(self):
    state = ProtocolState(2048,2048,1,2032,2048)
    state.record_update(0,2032,0.,0,0)
    state.record_update(1,2036,0.,0,0)
    rows = [dict(update_id=0,start_action=2032),dict(update_id=1,start_action=2036)]
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp)/'dtl_updates.jsonl'
      def write(values):
        path.write_text(''.join(json.dumps(v)+'\n' for v in values))
      write(rows)
      audit_dtl_receipts(tmp,state)
      for values in (rows[:1],rows+[rows[0]],
                     rows+[dict(update_id=2,start_action=2040)]):
        write(values)
        with self.assertRaises(RuntimeError):
          audit_dtl_receipts(tmp,state)


if __name__ == '__main__':
  unittest.main()
