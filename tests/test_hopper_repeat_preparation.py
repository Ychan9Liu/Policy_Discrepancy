"""CPU-only configuration, immutable per-seed c and seed-mapping fixtures."""

import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

from dreamerv3.freeze_c import freeze
from dreamerv3.protocol_scope import HOPPER_123, HOPPER_SCRIPT
from embodied.run.protocol_v1 import seed32 as actual_seed32
from scripts.prepare_m2_v1 import resolve
from scripts.prepare_hopper_seeds123 import ALLOCATION, check_repeat, seed32


class HopperPreparationTest(unittest.TestCase):
  def test_six_production_configs_only_differ_in_declared_fields(self):
    configs = []
    for seed, mode, gpu in ALLOCATION:
      config = resolve(['--configs','m2_hopper_seeds123','--seed',str(seed),
          '--agent.rep_probe.mode',mode,'--jax.prealloc','False',
          '--logger.outputs','jsonl','wandb'])
      check_repeat(config)
      configs.append(dict(config.flat))
    for config in configs:
      self.assertLessEqual({key for key in config if config[key] != configs[0][key]},
                           {'seed','agent.rep_probe.mode'})
    self.assertEqual([r[2] for r in ALLOCATION], [1,3,4,5,6,7])
    self.assertEqual(resolve(['--configs','m2_v1']).seed, 0)
    self.assertEqual(resolve(['--configs','m2_v1']).script, 'protocol_v1')

  def test_exact_existing_random_mapping_with_new_roots(self):
    for seed in (1,2,3):
      for i in range(16):
        self.assertEqual(seed32(seed,'train_env',i), actual_seed32(seed,'dmc_hopper_hop','train_env',i))
      for point in range(21):
        for episode in range(10):
          self.assertEqual(seed32(seed,'eval_env',point,episode),
                           actual_seed32(seed,'dmc_hopper_hop','eval_env',point,episode))
    self.assertEqual(len({seed32(seed,'train_replay') for seed in (0,1,2,3)}), 4)

  def source(self, root, seed=1, protocol=HOPPER_123):
    # Artifact-only hand fixture: never creates an environment or simulates a million steps.
    config = {'script':HOPPER_SCRIPT if protocol==HOPPER_123 else 'protocol_v1', 'seed':seed,
        'agent.rep_probe.mode':'logging','agent.rep_probe.alpha':20.,
        'agent.dyn.rssm.free_nats':1., 'run.action_budget':1000000,
        'run.eval_every_actions':50000,'run.eval_eps':10,'run.match_start':100000,
        'run.match_end':300000,'run.engineering_fixture':False}
    def write(name, value):
      (root/name).write_text(json.dumps(value),encoding='utf-8')
    write('protocol_manifest.json',dict(protocol=protocol,task='dmc_hopper_hop',root_seed=seed,
        git_commit='artifact-fixture-not-a-training',config=config))
    receipts=[dict(update_id=i,start_action=step,match_sum=s,match_count=n,invalid_count=0)
        for i,(step,s,n) in enumerate(((99999,.4,1),(100000,.5,1),(299999,1.5,3),(300000,.2,1)))]
    (root/'update_receipts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in receipts))
    (root/'evaluations.jsonl').write_text(''.join(json.dumps(dict(action_step=s))+'\n'
        for s in range(0,1000001,50000)))
    write('final_state.json',dict(git_commit='artifact-fixture-not-a-training',train_action_steps=1000000,
        updates=4,evaluations=21,match_sum=2.,match_count=4))
    write('matching_result.json',dict(match_sum=2.,match_count=4,c=.5,start=100000,end=300000))

  def test_per_seed_c_metadata_and_wrong_identity_refused(self):
    for seed in (1,2,3):
      with TemporaryDirectory() as directory:
        root=Path(directory)
        self.source(root,seed)
        result=freeze(root,root/'c.json')
        self.assertEqual((result['protocol'],result['root_seed'],result['c']), (HOPPER_123,seed,.5))
        self.assertEqual(result['included_update_ids'],[1,2])
        self.assertEqual(freeze(root,root/'c.json'), result)
        p=root/'protocol_manifest.json'
        manifest=json.loads(p.read_text()); manifest['root_seed']=0
        p.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
          freeze(root,root/'wrong.json')

  def test_original_seed0_artifact_matches_trusted_pre_extension(self):
    source=subprocess.check_output(['git','show',
        'b3a4dd72ecd5c852be941de92ad66de59a6029ec:dreamerv3/freeze_c.py'],text=True)
    namespace={}
    exec(compile(source,'<trusted pre-extension freeze_c>','exec'),namespace)
    with TemporaryDirectory() as directory:
      root=Path(directory); self.source(root,0,'M2-v1')
      old=namespace['freeze'](root,root/'old.json')
      new=freeze(root,root/'new.json')
      for value in (old,new):
        value.pop('frozen_at_utc')
      self.assertEqual(old,new)


if __name__=='__main__':
  unittest.main()
