"""DL-identifiability-r1: frozen-batch algebra and a finite H1 counterexample.

Standard-library CPU analysis only. No agent, training, environment, replay or
GPU is loaded. The finite categorical example is not a Dreamer experiment.
"""
import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import subprocess

VERSION = 'DL-identifiability-r1'
TRAIN_SHA = '13ec72a214b0c03b886b2c7fac52c8f089966dec'
ROOT = Path(__file__).resolve().parents[1]


def require(value, message):
  if not value:
    raise ValueError(message)


def number(value):
  require(type(value) in (int, float) and math.isfinite(value), 'Finite numeric scalar required')
  return float(value)


def decode(payload):
  def pairs(items):
    result = {}
    for key, value in items:
      require(key not in result, 'Duplicate JSON key')
      result[key] = value
    return result
  return json.loads(payload, object_pairs_hook=pairs,
      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)),
      parse_float=lambda x: number(float(x)))


def same_batch_bounds(gate):
  """Real-arithmetic projection using recorded float32 aggregate statistics."""
  n = gate['updates']
  require(type(n) is int and n > 0, 'Actual update count required')
  stats = gate['statistics']
  num, den = (stats[key] for key in ('calibration_num', 'calibration_den'))
  require(type(num['n']) is type(den['n']) is int and num['n'] == den['n'] == n,
      'Common update denominator required')
  numerator, denominator = number(num['mean']), number(den['mean'])
  require(0 <= numerator <= denominator and denominator > 0, 'Positive valid calibration denominator required')
  coefficient = numerator / denominator
  rho_half_ratio = .5 * coefficient
  reported = number(gate['relative_to_D_only_release_on_valid_N_positions'])
  require(math.isclose(rho_half_ratio, reported, rel_tol=1e-12, abs_tol=1e-15),
      'Existing same-mask release ratio disagrees')
  means = {key: number(stats[key]['valid_position_weighted_mean']) for key in
      ('e_valid_mean', 'signed_support_valid_mean', 'u_q_valid_mean', 'u_p_valid_mean')}
  e, signed, uq, up = (means[key] for key in
      ('e_valid_mean', 'signed_support_valid_mean', 'u_q_valid_mean', 'u_p_valid_mean'))
  require(0 <= e <= 1 and -1 <= signed <= 1 and 0 <= uq <= 1 and 0 <= up <= 1,
      'Support summary outside its range')
  require(e >= signed and e <= 1-up, 'Positive-part/headroom inequalities disagree')
  return dict(
      scope='Fixed observed N batches, alpha20, same H1-valid mask, same D/e/K; no new trajectory or gradient prediction',
      numerical_scope='Real-arithmetic identities applied to recorded float32 reductions; not a bit-exact gate re-execution',
      empirical_current_rho=.5, allowed_rho=[0, 1],
      same_mask_D_potential_release_coefficient=coefficient,
      current_release_ratio_to_same_mask_D=rho_half_ratio,
      maximum_ratio_with_rho_at_most_one=coefficient,
      rho_needed_to_equal_same_mask_D=(1/coefficient if coefficient else None),
      finite_rho_can_match=coefficient > 0,
      legal_rho_can_match=coefficient >= 1,
      formal_calibration_kappa_established=False,
      current_actor_or_optimizer_effect_established=False,
      mean_positive_support=e, mean_negative_support=e-signed,
      mean_absolute_support=2*e-signed, mean_one_sided_headroom=1-up,
      headroom_fraction_used=(e/(1-up) if up < 1 else None),
      exponential_saturation_cause_established=False)


def validate_latents(q, p):
  require(len(q) == len(p) == 4, 'Exactly four cue distributions required')
  for row in (*q, *p):
    require(len(row) == 4 and all(isinstance(x,Fraction) and x > 0 for x in row) and
        sum(row) == 1, 'Finite exact positive normalized probabilities required')
  require(all(row == (Fraction(1,4),)*4 for row in p),
      'Prior must be uniform and must not receive the cue')


def validate_readouts(readouts):
  require(len(readouts) == 256 and all(type(a) is tuple and len(a) == 4 and
      all(type(x) is int and x in range(4) for x in a) for a in readouts) and
      set(readouts) == set(itertools.product(range(4),repeat=4)),
      'All 256 distinct cue-independent latent readouts required')


def one_hot_support(probabilities, label):
  require(type(label) is int and label in (0,1), 'One-hot H1 label must be bin0 or bin1')
  require(len(probabilities) == 2 and all(isinstance(x,Fraction) and x > 0 for x in probabilities)
      and sum(probabilities) == 1, 'Exact normalized two-bin probabilities required')
  loss = -math.log(float(probabilities[label]))
  return dict(label=label,target_entropy=0.,loss=loss,KL=loss,support=math.exp(-loss))


def calibrated_H1(q, p, law):
  require(tuple(q) == tuple(p) == tuple(law), 'Both H1 heads must match the independent true law')
  rows = []
  for label in (0,1):
    qr,pr = one_hot_support(q,label),one_hot_support(p,label)
    signed = qr['support']-pr['support']
    require(signed == 0., 'Counterexample requires identical H1 support')
    rows.append(dict(label=label,probability=str(law[label]),q=qr,p=pr,e=max(signed,0.)))
  return rows


def categorical_witness(confidence=Fraction(97, 100)):
  """Enumerate true cue/latent outcomes and all 4**4 possible readouts."""
  require(isinstance(confidence, Fraction) and Fraction(1, 4) <= confidence < 1,
      'Finite symmetric four-cue confidence in [1/4,1) required')
  cues = tuple(range(4))
  q_matrix = tuple(tuple(confidence if z == cue else (1-confidence)/3 for z in cues) for cue in cues)
  p_matrix = ((Fraction(1,4),)*4,)*4
  validate_latents(q_matrix,p_matrix)
  q = lambda cue, z: q_matrix[cue][z]
  p = lambda cue, z: p_matrix[cue][z]
  law = (Fraction(99,100),Fraction(1,100))
  rewards = {
      'useful_current_readout': lambda cue, a: int(a == cue),
      'neutral': lambda cue, a: 1,
      'harmful_current_readout': lambda cue, a: int(a != cue)}
  def value(distribution, readout, reward):
    return sum((Fraction(1,4)*distribution(cue,z)*law[r1]*reward(cue,readout[z])
        for cue in cues for z in cues for r1 in (0,1)), Fraction(0))
  worlds = {}
  identity = cues
  all_readouts = tuple(itertools.product(cues, repeat=4))
  validate_readouts(all_readouts)
  for name, reward in rewards.items():
    actor_q, actor_p = value(q,identity,reward), value(p,identity,reward)
    scores = [(a,value(q,a,reward),value(p,a,reward)) for a in all_readouts]
    best_q = max(vq for a,vq,vp in scores)
    best_p = max(vp for a,vq,vp in scores)
    witness_q = next(a for a,vq,vp in scores if vq == best_q)
    witness_p = next(a for a,vq,vp in scores if vp == best_p)
    worlds[name] = dict(delayed_return_q=float(actor_q), delayed_return_p=float(actor_p),
        full_H2_return_q=float(actor_q+Fraction(1,100)),
        full_H2_return_p=float(actor_p+Fraction(1,100)),
        current_actor_return_difference=float(actor_q-actor_p),
        best_latent_readout_return_q=float(best_q), best_latent_readout_return_p=float(best_p),
        best_readout_information_advantage=float(best_q-best_p),
        current_actor_difference_exact=str(actor_q-actor_p),
        best_readout_advantage_exact=str(best_q-best_p),
        best_q_readout_witness=list(witness_q),best_p_readout_witness=list(witness_p),
        all_readouts=[dict(actions=list(a),q_return_exact=str(vq),p_return_exact=str(vp))
            for a,vq,vp in scores],
        information_should_be_discarded_established=False)
  qr = [float(q(0,z)) for z in cues]
  kl = math.fsum(x*math.log(x/.25) for x in qr)
  r = float(confidence)
  derivative = math.log(3*r/(1-r))
  H1 = calibrated_H1(law,law,law)
  def K_at(x):
    values=[x]+[(1-x)/3]*3
    return math.fsum(v*math.log(4*v) for v in values)
  delta = 1e-6
  finite_difference = (K_at(r+delta)-K_at(r-delta))/(2*delta)
  return dict(
      scope='Finite analytical counterexample; not Dreamer, continuous actor, production Dt or quadruped evidence',
      timing='t0: observe cue and choose action; t1: shared independent Bernoulli reward; t2: terminal reward of initial cue/action, no further decision',
      cue_count=4, readouts_exhaustively_checked=len(all_readouts), posterior_confidence=str(confidence),
      exact_posterior_probabilities=[[str(x) for x in row] for row in q_matrix],
      exact_prior_probabilities=[[str(x) for x in row] for row in p_matrix],
      K=kl, D_toy_action_distribution_KL=kl,
      D_toy_equals_project_Dt_claimed=False, free_nats=1., rep_active=kl > 1,
      rep_dK_dconfidence_with_prior_fixed=derivative,
      rep_finite_difference_check=finite_difference,
      rep_gradient_scope='Only the isolated active rep term and direct confidence parameter; not the full optimizer or information-loss evidence',
      H1_bins=[0, 1], H1_true_and_both_predicted_probabilities=[.99, .01],
      H1_label_receipts=H1,common_H1_expected_reward=.01,
      H1_independent_of_cue_action_and_world=True,
      H1_support_on_zero_reward=.99, H1_support_on_one_reward=.01,
      H1_e_identically_zero=True,
      N_weight_for_any_finite_legal_alpha_rho=1.,
      worlds=worlds, protection_benefit_measured=False,
      current_quadruped_failure_cause_established=False,
      conclusion='Identical D_toy/K/H1 summaries can coexist with positive, neutral or negative current actor utility; negative actor utility is not information uselessness')


def code_binding(expected):
  require(re.fullmatch('[0-9a-f]{40}', expected), 'Full committed SHA required')
  def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)
  require(git('rev-parse','HEAD').decode().strip() == expected, 'Actual HEAD differs')
  require(not git('status','--porcelain','--untracked-files=all'), 'Clean checkout required')
  names = ('analysis/dl_signal_identifiability.py','analysis/dl_pilot_review.py',
      'dreamerv3/dt_latch.py','docs/DL_R4_IDENTIFIABILITY.md')
  pins = {}
  for name in names:
    raw = (ROOT/name).read_bytes()
    require(raw == git('show',expected+':'+name), 'Checkout/Git bytes differ: '+name)
    pins[name] = hashlib.sha256(raw).hexdigest()
  return pins


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  for name in ('review','expected-review-sha','expected-git','output'):
    parser.add_argument('--'+name,required=True)
  args=parser.parse_args()
  source_pins = code_binding(args.expected_git)
  payload = Path(args.review).read_bytes()
  pin = hashlib.sha256(payload).hexdigest()
  require(re.fullmatch('[0-9a-f]{64}',args.expected_review_sha) and pin == args.expected_review_sha,
      'Pinned review bytes differ before decode')
  review = decode(payload)
  require(review['status']=='independent_evidence_review_passed' and
      review['train_sha']==TRAIN_SHA and review['gate']['updates']==249493,
      'Completed fixed pilot review required')
  changes=review['config_differences']
  require(changes['agent.dt_latch.mode']['actual']=='dtlatch' and
      number(changes['agent.dt_latch.alpha']['actual'])==20 and
      number(changes['agent.dt_latch.rho']['actual'])==.5,
      'Fixed alpha20/rho.5 N configuration required')
  result=dict(version=VERSION,analysis_git_sha=args.expected_git,train_sha=TRAIN_SHA,
      review_sha256=pin,source_files_sha256=source_pins,
      same_batch=same_batch_bounds(review['gate']),finite_counterexample=categorical_witness(),
      new_training=False,new_model_environment_or_replay_loads=0,GPU_calls=0,
      independent_signal_or_method_validated=False)
  require(code_binding(args.expected_git)==source_pins,'Analysis sources changed')
  output=Path(args.output)
  output.parent.mkdir(parents=True,exist_ok=True)
  with output.open('x',encoding='utf-8',newline='\n') as stream:
    json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
  print(json.dumps(dict(output=str(output),maximum_legal_rho_ratio=result['same_batch'][
      'maximum_ratio_with_rho_at_most_one'],rho_needed=result['same_batch']['rho_needed_to_equal_same_mask_D'],
      finite_worlds={name:{k:v for k,v in row.items() if k != 'all_readouts'}
          for name,row in result['finite_counterexample']['worlds'].items()}),sort_keys=True))


if __name__=='__main__': main()
