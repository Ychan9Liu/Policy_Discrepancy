"""Synthetic numeric cached-policy replay; no model, physics or GPU proof."""
import copy
from pathlib import Path
import pickle
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.dl_feedback import replay_policy
from scripts.dl_feedback_model import named_root_key, tree_digest


def observation(value, terminal=False):
  return dict(image=np.full((64,64,3),value,np.uint8),reward=np.array(.2,np.float32),
      is_first=np.array(False),is_last=np.array(terminal),is_terminal=np.array(terminal))


class NumericPolicy:
  def __init__(self):
    self.calls = []

  def step(self, carry, obs, key, omit=False):
    self.calls.append((obs['image'].copy(),key.copy(),omit))
    after = copy.deepcopy(carry)
    signal = 0 if omit else int(obs['image'][0,0,0])
    raw = np.full(12,signal/100 + int(key[0]%17)/100,np.float32)
    after[1]['deter'][0] += np.float32(1)
    after[3]['action'][0] = raw.copy()
    proof = dict(root_key=key.copy(),latent_key=key[::-1].copy(),actor_key=key.copy(),
        rng_consumption=np.array(2,np.int32),raw=raw.copy(),
        h=after[1]['deter'][0].copy(),latent=after[1]['stoch'][0].copy(),
        selected_logit=np.full((3,4),signal,np.float32))
    return after,raw,proof


def numeric_record(prior=False, terminal=False):
  """Valid numeric receipt fixture; snapshots are deliberately synthetic."""
  h, count = 3, (1 if terminal else 3)
  carry = ({},dict(deter=[np.zeros(4,np.float32)],stoch=[np.zeros((3,4),np.float32)]),
      {},dict(action=[np.zeros(12,np.float32)]))
  record = dict(version='DL-feedback-r1',implementation='DL-feedback-physics-r1',
      action_contract='production_clip_unify_normalize_v1',horizon=h,prior=prior,
      initial_carry=copy.deepcopy(carry),keys=np.stack([named_root_key(9,1,0,1+s) for s in range(h)]),
      raw=np.zeros((h,12),np.float32),applied=np.zeros((h,12),np.float64),
      reward_f64=np.zeros(h,np.float64),reward_f32=np.zeros(h,np.float32),
      discount=np.zeros(h,np.float64),is_first=np.zeros(h,bool),is_last=np.zeros(h,bool),
      is_terminal=np.zeros(h,bool),action_executed=np.arange(h)<count,
      observation_executed=np.arange(h+1)<=count,observations=[observation(1)],
      snapshots=[],proofs=[],pre_carry_sha256=['']*h,after_carry_sha256=['']*h,
      executed_steps=count,end_type='true_terminal' if terminal else 'window')
  def state(clock,done):
    return dict(from_dm_done=bool(done),suite=dict(integration=np.array([clock],np.float64),
        task={'synthetic_only':True},step_count=clock,reset_next_step=bool(done)))
  record['snapshots'].append(state(1,False))
  policy = NumericPolicy()
  for s in range(count):
    record['pre_carry_sha256'][s] = tree_digest(carry)
    carry,raw,proof = policy.step(carry,record['observations'][s],record['keys'][s],prior and s==0)
    record['after_carry_sha256'][s] = tree_digest(carry)
    record['raw'][s],record['applied'][s] = raw,raw.astype(np.float64)
    record['reward_f64'][s],record['reward_f32'][s] = .2,np.float32(.2)
    done = terminal and s==count-1
    record['discount'][s] = 0 if done else 1
    record['is_last'][s] = record['is_terminal'][s] = done
    record['proofs'].append(proof)
    record['observations'].append(observation(10+s,done))
    record['snapshots'].append(state(2+s,done))
  record['final_carry'] = carry
  return record


class FeedbackReplayTest(unittest.TestCase):
  def test_exact_cached_own_observations_keys_prior_only_first_and_no_mutation(self):
    for prior in (False,True):
      record,policy = numeric_record(prior),NumericPolicy()
      original = pickle.dumps(record,protocol=5)
      proof = replay_policy(policy,record)
      self.assertEqual(proof['policy_steps'],3)
      self.assertTrue(proof['all_numeric_proofs_byte_exact'])
      self.assertFalse(proof['renderer_replayed'])
      self.assertEqual([x[2] for x in policy.calls],[prior,False,False])
      for s,(image,key,_) in enumerate(policy.calls):
        self.assertEqual(image.tobytes(),record['observations'][s]['image'].tobytes())
        self.assertEqual(key.tobytes(),named_root_key(9,1,0,1+s).tobytes())
      self.assertEqual(pickle.dumps(record,protocol=5),original)

  def test_numeric_proof_and_own_rgb_corruption_rejected(self):
    for field in ('latent_key','actor_key','selected_logit','h','latent'):
      record = numeric_record()
      record['proofs'][1][field].flat[0] += 1
      with self.subTest(field=field),self.assertRaises(ValueError):
        replay_policy(NumericPolicy(),record)
    record = numeric_record()
    record['observations'][1]['image'][0,0,0] += np.uint8(1)
    with self.assertRaises(ValueError):
      replay_policy(NumericPolicy(),record)
    record = numeric_record()
    record['after_carry_sha256'][-1] = '0'*64
    with self.assertRaises(AssertionError):
      replay_policy(NumericPolicy(),record)

  def test_true_terminal_replays_actual_prefix_and_rejects_fake_padding(self):
    record,policy = numeric_record(True,True),NumericPolicy()
    self.assertEqual(replay_policy(policy,record)['policy_steps'],1)
    self.assertEqual(len(policy.calls),1)
    invalid = copy.deepcopy(record)
    invalid['observations'].append(observation(99,True))
    with self.assertRaises(ValueError):
      replay_policy(NumericPolicy(),invalid)
    invalid = copy.deepcopy(record)
    invalid['end_type'] = 'window'
    with self.assertRaises(ValueError):
      replay_policy(NumericPolicy(),invalid)

  def test_callback_mutation_and_reshaped_raw_cannot_claim_exact_replay(self):
    for field in ('carry','observation','key','raw_shape'):
      class DamagedPolicy(NumericPolicy):
        def step(self,carry,obs,key,omit=False):
          after,raw,proof = super().step(carry,obs,key,omit)
          if field == 'carry':
            carry[1]['deter'][0][0] += np.float32(1)
          elif field == 'observation':
            obs['image'][-1,-1,-1] ^= np.uint8(1)
          elif field == 'key':
            key[0] ^= np.uint32(1)
          else:
            raw = raw.reshape(3,4)  # Same dtype/bytes, different action semantics.
          return after,raw,proof
      record = numeric_record()
      original = pickle.dumps(record,protocol=5)
      with self.subTest(field=field),self.assertRaises(ValueError):
        replay_policy(DamagedPolicy(),record)
      self.assertEqual(pickle.dumps(record,protocol=5),original)


if __name__ == '__main__':
  unittest.main()
