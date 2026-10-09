"""Hand-calculated raw-return statistics; multi-seed data is fixture only."""

import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.return_metrics import (
    MetricSpec, TASKS, aggregate, analyze_run, normalize_points, seed_metrics,
    snapshot_evidence, trapezoid_auc, write_tables)
from dreamerv3.return_stats import EpisodeReturn, REWARD_SOURCE, RETURN_DEFINITION


FIXTURE = MetricSpec(budget=10, grid=(0, 2, 5, 10), tail=(5, 10),
                     episodes=2, seeds=(0, 1), scope='hand-calculated-fixture')


def metadata(task=TASKS[0], group='baseline', seed=0):
  return dict(task=task, group=group, seed=seed, run_id=f'{task}/{group}/{seed}',
      source_eval_episodes=2, engineering_fixture=False,
      legacy_raw_return_verified=False, eval_policy='fixture-eval-rng')


def point(step, values=(2.0, 4.0)):
  return dict(schema_version=2, action_step=step, target_action_step=step,
      snapshot_action_step=step, update_id=step, snapshot_update_id=step,
      snapshot_id=f'eval_snapshots/{step:07d}/fixture',
      reward_source=REWARD_SOURCE, return_definition=RETURN_DEFINITION,
      returns_raw=list(values), scores=list(values), mean=sum(values)/len(values),
      mean_return_raw=sum(values)/len(values), lengths=[1, 3], seeds=[7, 8],
      episodes=[dict(episode_id=i, return_raw=value, length=length,
          complete=True, environment_seed=seed, action_rng='fixture-eval-rng',
          reward_source=REWARD_SOURCE, return_definition=RETURN_DEFINITION)
          for i, (value, length, seed) in enumerate(zip(values, (1, 3), (7, 8)))])


def snapshots(records):
  return {row['target_action_step']: dict(actual_step=row['snapshot_action_step'],
      update_id=row['update_id'], snapshot_id=row['snapshot_id']) for row in records}


def calculate(records, meta=None, spec=FIXTURE, evidence=None):
  meta = meta or metadata()
  episodes, points, issues = normalize_points(meta, records,
      snapshots(records) if evidence is None else evidence, spec)
  return episodes, points, seed_metrics(meta, points, issues, spec)


class ReturnMetricsTest(unittest.TestCase):

  def test_raw_reward_terminal_and_repeat_already_summed(self):
    returns = []
    for rewards in ((3, 7), (-2,)):
      acc = EpisodeReturn()
      acc.add(dict(reward=0, is_first=True, is_last=False))
      for index, reward in enumerate(rewards):
        acc.add(dict(reward=reward, is_first=False,
                     is_last=index == len(rewards)-1))
      returns.append(acc.result())
    self.assertEqual([item['return_raw'] for item in returns], [10, -2])
    self.assertEqual([item['length'] for item in returns], [2, 1])
    # Rewards 3 and 7 may each already be a repeat-sum: count each once.
    self.assertEqual(returns[0]['return_definition'], RETURN_DEFINITION)
    acc = EpisodeReturn()
    with self.assertRaisesRegex(ValueError, 'incomplete'):
      acc.result()
    acc.add(dict(reward=0, is_first=True, is_last=False))
    acc.add(dict(reward=3, is_first=False, is_last=False))
    with self.assertRaisesRegex(ValueError, 'splice'):
      acc.add(dict(reward=0, is_first=True, is_last=False))
    with self.assertRaisesRegex(ValueError, 'incomplete'):
      acc.result()
    with self.assertRaisesRegex(ValueError, 'zero reward'):
      EpisodeReturn().add(dict(reward=1, is_first=True, is_last=False))
    with self.assertRaisesRegex(ValueError, 'Non-finite'):
      acc.add(dict(reward=float('nan'), is_first=False, is_last=True))

  def test_episode_arithmetic_mean_not_length_weighted(self):
    _, points, _ = calculate([point(0, (3, 9))])
    self.assertEqual(points[0]['mean_return_raw'], 6)
    self.assertTrue(points[0]['source_complete'])

  def test_actual_x_trapezoid_and_fixed_budget(self):
    # Areas: 2*(1+3)/2=4, 3*(3+1)/2=6, 5*(1+5)/2=15.
    values = [(10, 5), (0, 1), (5, 1), (2, 3)]
    self.assertEqual(trapezoid_auc(values, 1000000), 25/1000000)
    self.assertEqual(trapezoid_auc([(0,1),(2,3),(5,1)], 1000000), 10/1000000)
    with self.assertRaises(ValueError):
      trapezoid_auc([(0, 1), (0, 2)], 1000000)

  def test_seed_then_task_then_equal_task_aggregation_and_deltas(self):
    seed_rows = []
    # Task means (2,4,20,30), overall 14; pooled episodes are not repetitions.
    for task, pair, delta in zip(TASKS, ((1, 3), (2, 6), (10, 30), (20, 40)),
                                  (2, -1, 4, -2)):
      for group in ('baseline', 'dt', 'constant', 'shuffle'):
        for seed, value in enumerate(pair):
          value += delta if group == 'dt' else 0
          records = [point(step, (value, value)) for step in FIXTURE.grid]
          seed_rows.append(calculate(records, metadata(task, group, seed))[2])
    tasks, overall, differences = aggregate(seed_rows, FIXTURE)
    self.assertEqual(next(row for row in overall if row['group']=='baseline')[
        'auc_return_per_budget'], 14)
    self.assertEqual(next(row for row in overall if row['group']=='dt')[
        'tail_mean_return'], 14.75)
    self.assertEqual([row['auc_return_per_budget'] for row in differences
        if row['level']=='task' and row['comparison']=='dt-baseline'], [2, -1, 4, -2])
    self.assertEqual(next(row for row in differences if row['level']=='overall'
        and row['comparison']=='dt-baseline')['auc_return_per_budget'], .75)
    self.assertEqual(len(tasks), 16)
    # Each seed is integrated separately before averaging.
    a = calculate([point(x, (y,y)) for x,y in zip(FIXTURE.grid,(0,0,0,10))],
                  metadata(seed=0))[2]
    b = calculate([point(x, (4,4)) for x in FIXTURE.grid], metadata(seed=1))[2]
    task = aggregate([a,b], FIXTURE)[0][0]
    self.assertEqual(a['auc_return_per_budget'], 2.5)
    self.assertEqual(task['auc_return_per_budget'], 3.25)

  def test_unordered_records_and_extra_point_do_not_replace_tail(self):
    records = [point(10,(10,10)),point(2),point(11,(999,999)),
               point(0),point(5,(6,6))]
    _, _, result = calculate(records)
    self.assertEqual(result['tail_mean_return'], 8)
    self.assertTrue(result['tail_complete'])
    self.assertEqual(result['observed_action_steps'], [0,2,5,10,11])

  def test_missing_grid_and_boundary_preserve_independent_tail_completeness(self):
    for omitted in (0,2):
      result=calculate([point(x) for x in FIXTURE.grid if x != omitted])[2]
      self.assertFalse(result['auc_complete'])
      self.assertIsNone(result['auc_return_per_budget'])
      self.assertTrue(result['tail_complete'])
      self.assertIn(f'missing_point:{omitted}',result['auc_issues'])
      self.assertEqual(result['fixed_budget'],10)
    result=calculate([point(x) for x in FIXTURE.grid if x != 10])[2]
    self.assertFalse(result['tail_complete'])
    self.assertIsNone(result['tail_mean_return'])

  def test_partial_episode_count_snapshot_and_mean_conflicts(self):
    for change, reason in (
        (lambda row: row['episodes'][0].update(complete=False),'incomplete_episode'),
        (lambda row: row.update(mean=123),'declared_mean_conflict'),
        (lambda row: row.update(snapshot_action_step=6),'snapshot_action_step_mismatch'),
        (lambda row: row.update(snapshot_id='other'),'snapshot_id_conflict'),
        (lambda row: row.update(scores=[0,0]),'scores_returns_alias_conflict')):
      records=[point(x) for x in FIXTURE.grid]
      evidence=snapshots(records)
      change(records[2])
      _, points, result=calculate(records,evidence=evidence)
      self.assertFalse(result['tail_complete'],reason)
      self.assertIn(reason, points[2]['issues'])
    records=[point(x) for x in FIXTURE.grid]
    records[2]['episodes'].pop()
    self.assertFalse(calculate(records)[2]['tail_complete'])
    evidence=snapshots([point(x) for x in FIXTURE.grid])
    evidence[5]['update_id']=99
    self.assertFalse(calculate([point(x) for x in FIXTURE.grid],evidence=evidence)[2]['tail_complete'])

  def test_identical_rewrite_dedup_and_conflicting_snapshot_refusal(self):
    records=[point(x) for x in FIXTURE.grid]
    records.append(copy.deepcopy(records[2]))
    episodes, points, result=calculate(records)
    self.assertEqual(len(episodes),8)
    self.assertTrue(result['auc_complete'])
    self.assertEqual(points[2]['identical_rewrites_ignored'],1)
    records[-1]['snapshot_id']='another-fixed-snapshot'
    self.assertFalse(calculate(records)[2]['tail_complete'])

  def test_legacy_scores_need_verified_raw_reward_source(self):
    legacy=dict(action_step=0,update_id=0,scores=[-2,8],lengths=[1,7],
                seeds=[7,8],mean=3)
    evidence={0:dict(actual_step=0,update_id=0,snapshot_id='old-snapshot')}
    _, points, _=calculate([legacy],evidence=evidence)
    self.assertFalse(points[0]['source_complete'])
    meta=dict(metadata(),legacy_raw_return_verified=True)
    episodes,points,_=calculate([legacy],meta,evidence=evidence)
    self.assertTrue(points[0]['source_complete'])
    self.assertEqual(points[0]['mean_return_raw'],3)
    self.assertEqual([x['return_raw'] for x in episodes],[-2,8])
    self.assertTrue(all(x['evidence_basis']=='reviewed_legacy_source' for x in episodes))

  def test_engineering_rows_are_not_formal_metrics(self):
    meta=dict(metadata(),engineering_fixture=True)
    _,points,result=calculate([point(x) for x in FIXTURE.grid],meta)
    self.assertTrue(all(x['source_complete'] for x in points))
    self.assertIsNone(result['auc_return_per_budget'])
    self.assertIsNone(result['tail_mean_return'])

  def test_missing_seed_task_and_multiple_runs_are_not_reweighted(self):
    rows=[calculate([point(x) for x in FIXTURE.grid],metadata(seed=s))[2] for s in (0,1)]
    task,overall,_=aggregate(rows,FIXTURE)
    self.assertTrue(task[0]['auc_complete'])
    self.assertFalse(overall[0]['auc_complete'])
    self.assertIsNone(overall[0]['auc_return_per_budget'])
    self.assertFalse(aggregate(rows[:1],FIXTURE)[0][0]['auc_complete'])
    self.assertFalse(aggregate(rows+[copy.deepcopy(rows[0])],FIXTURE)[0][0]['auc_complete'])

  def test_formal_grid_has_exact_21_points_10_episodes_and_five_tail_points(self):
    formal=MetricSpec()
    self.assertEqual(len(formal.grid),21)
    self.assertEqual(formal.tail,(800000,850000,900000,950000,1000000))
    self.assertEqual(formal.seeds,(0,))
    with self.assertRaisesRegex(ValueError,'Fixture'):
      MetricSpec(seeds=(0,1))

  def test_complete_formal_hand_curve_and_exact_five_tail_points(self):
    spec=MetricSpec()
    records=[]
    for step in spec.grid:
      row=point(step)
      value=1+step/100000
      row.update(returns_raw=[value]*10,scores=[value]*10,
                 mean=value,mean_return_raw=value,lengths=[1]*10,seeds=list(range(10)))
      row['episodes']=[dict(row['episodes'][0],episode_id=i,return_raw=value,
                            length=1,environment_seed=i) for i in range(10)]
      records.append(row)
    records.reverse()
    # Additional record does not change the designated tail set.
    records.insert(0,point(1050000,(999,999)))
    meta=dict(metadata(),source_eval_episodes=10,source_budget=1000000)
    _,_,result=calculate(records,meta,spec)
    self.assertEqual(result['auc_return_per_budget'],6)
    self.assertEqual(result['tail_mean_return'],10)
    missing=[row for row in records if row['action_step']!=250000]
    result=calculate(missing,meta,spec)[2]
    self.assertFalse(result['auc_complete'])
    self.assertTrue(result['tail_complete'])
    self.assertEqual(result['tail_mean_return'],10)

  def test_disk_schema_and_read_only_snapshot_audit(self):
    with TemporaryDirectory() as temp:
      root=Path(temp)/'run'
      root.mkdir()
      manifest=dict(task=TASKS[0],root_seed=0,git_commit='fixture-commit',
          eval_policy='fixture-eval-rng',config={'agent.rep_probe.mode':'logging',
          'run.engineering_fixture':False,'run.action_budget':10,'run.eval_eps':2})
      (root/'protocol_manifest.json').write_text(json.dumps(manifest))
      records=[point(x) for x in FIXTURE.grid]
      (root/'evaluations.jsonl').write_text('\n'.join(json.dumps(row) for row in records))
      for row in records:
        folder=root/'eval_snapshots'/f"{row['action_step']:07d}"
        (folder/'fixture').mkdir(parents=True)
        (folder/'fixture'/'done').touch()
        (folder/'fixture'/'agent.pkl').write_bytes(b'fixture-checkpoint-placeholder')
        (folder/'latest').write_text('fixture')
        (folder/'evaluation_snapshot.json').write_text(json.dumps(dict(
            snapshot_id=row['snapshot_id'],actual_action_step=row['action_step'],
            update_id=row['update_id'],git_commit='fixture-commit')))
      before={p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
      episodes,points,result,audit=analyze_run(dict(directory=str(root)),FIXTURE)
      self.assertTrue(result['auc_complete'])
      self.assertTrue(result['tail_complete'])
      output=Path(temp)/'output'
      write_tables(output,dict(episodes=episodes,evaluation_points=points),audit)
      self.assertEqual(before,{p:p.read_bytes() for p in root.rglob('*') if p.is_file()})
      self.assertEqual(audit['field_mapping']['scores'],'episode_return_raw')
      self.assertIn('input_sha256',audit)
      with self.assertRaises(FileExistsError):
        write_tables(output,{},audit)
      with (root/'evaluations.jsonl').open('a') as file:
        file.write('\n\ninvalid-json\n'+json.dumps(point(11)))
      episodes,_,result,audit=analyze_run(dict(directory=str(root)),FIXTURE)
      self.assertEqual([row['source_line'] for row in episodes
                        if row['target_action_step']==11],[7,7])
      self.assertFalse(result['auc_complete'])
      self.assertTrue(any('malformed_json_at_line:6' in issue
                          for issue in audit['parse_issues']))

  def test_external_snapshot_audit_conflicts_are_not_selected(self):
    with TemporaryDirectory() as temp:
      root=Path(temp)
      evaluation=root/'evaluations.jsonl'
      evaluation.write_text(json.dumps(point(0)))
      item=dict(action_step=0,actual_action_step=0,
          snapshot_id='eval_snapshots/0000000/fixture',
          counters={'updates':0},params_sha256='first')
      audit=root/'snapshot-audit.json'
      payload=dict(runs=[dict(directory='source-run',
          git_commit='fixture-commit',artifact_sha256={'evaluations.jsonl':
          hashlib.sha256(evaluation.read_bytes()).hexdigest()},
          snapshots=[item,item])])
      audit.write_text(json.dumps(payload))
      evidence,_=snapshot_evidence(root,{0},dict(snapshot_audit_file=str(audit),
          source_directory='source-run'),dict(git_commit='fixture-commit'))
      self.assertEqual(evidence[0]['snapshot_id'],item['snapshot_id'])
      payload['runs'][0]['snapshots'][1]=dict(item,params_sha256='different')
      audit.write_text(json.dumps(payload))
      with self.assertRaisesRegex(ValueError,'Conflicting snapshot'):
        snapshot_evidence(root,{0},dict(snapshot_audit_file=str(audit),
            source_directory='source-run'),dict(git_commit='fixture-commit'))


if __name__ == '__main__':
  unittest.main()
