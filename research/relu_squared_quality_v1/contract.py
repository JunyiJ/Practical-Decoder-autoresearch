"""Prospective exact own FFN panel source; no seal, roles or launch exists yet."""
import json
import os
from pathlib import Path
from research.body_allocation_model_v1.contract import ROOT,require,plain,fingerprint
from research.body_allocation_model_v1.evidence import reference
from research.relu_squared_model_v1.contract import ungated_seed
from research.relu_squared_resume_v1.data import config
from research.block_norm_v1.mps_guard import StrictMpsGuard
from .statistics import ARMS,LRS
from research.relu_squared_resume_v1.contract import runtime as base_environment
import numpy as np

B=ROOT/'research/tokenizer-evidence/relu-squared-quality-preparation-001'
RUN=ROOT/'research_runs/tokenizer-relu-squared-quality-v1-001'
PLAN=B/'relu-squared-quality001-plan.json';FREEZE=B/'relu-squared-quality001-freeze.json'
SELECTION=B/'relu-squared-quality001-selection.json';CONFIRM=B/'relu-squared-quality001-confirmation-plan.json'
CAPS={a:640*1024**2 for a in ARMS};RESERVE=8*1024**3;ENTRY=2*1024**2
PREREQUISITE_WEIGHTS=3*640*1024**2;CONSUMER_SCRATCH=128*1024**2
FLOOR=18*640*1024**2+512*1024**2+128*1024**2+PREREQUISITE_WEIGHTS+CONSUMER_SCRATCH+RESERVE+ENTRY
TARGET=FLOOR+2*1024**3
SCREEN_KIND='matched_matrix_ffn_finite_nine_screen_v1'
FRESH_KIND='matched_matrix_ffn_original_fresh_triples_v1'
DESIGN=dict(arms=list(ARMS),candidate='ungated_relu_squared',controls=['native_swiglu','ungated_silu'],steps=1200,targets=1228800,
 ema_snapshots=41,learning_rates=list(LRS),screen_trajectories=9,fresh_triples=3,fresh_trajectories=9,
 selection='Minimum complete H200 MPS native CE independently per arm;lower LR exact tie;one screen cohort;not LR optima/no grid extension.',
 fresh_order=[list(ARMS[p:]+ARMS[:p]) for p in range(3)],role_layout='screen plus three fresh cohorts,each initializer/private_ungated/train/eval/global_cpu/global_mps;one bootstrap role;25distinctroles',
 paired='Common native nonFFN values/order and train-eval sampling withincohort;ungated SiLU/ReLU-squared same private up/down initialization. New explicit global CPU/MPS roles.',
 mean_ce_delta_threshold=-.02,every_seed_negative=True,native_token_blocks=[2048,4096,8192],draws=2000,generator='PCG64',pcg64_reinitialized_each_width=True,
 same_indices_across_seeds_and_contrasts=True,quantile='linear',upper_quantile=1-.05/6,
 target_tokens=34278,target_bytes=111536,target_contract_sha256='6dcd972d01877136396b4a7c1bd11be02097623b7becbd77c58c469ec2c060d1',
 target_ids_sha256='dad00eaab7c2076736432b83eb9f1cc404c87aa5e4820b5f7abebf384c7fcdaf',
 category='LOCAL FFN architecture followup in incumbent-improvement allocation;broad/interaction allocation carried forward',
 resource_axis='Equal targets/updates,full28492416elements,FFNmatrixbudget516096/layer and finiteLRallowance;not equal FLOPs/time/tensorfactorization or isolated gating causation.',
 confidence_scope='Conditional on trained seeds and reused validation text;not holdout or population-seed confidence.',
 keep_all_screens_and_fresh=True,full_weight_high_water=18*640*1024**2,old_payload_or_future_retirement_or_hardlink_credit=False,
 rescue=False,seed_replacement=False,extra_tuning=False,holdout_scored=False,cost_only_adoption=False,control_fallback=False,separate_adoption_consumer_review=True)


def environment():
 return dict(**base_environment(),numpy_version=str(np.__version__))


def checked(ref):
 require(type(ref) is dict and set(ref) in ({'path','sha256'},{'path','sha256','bytes'})
     and reference(ref['path'])=={k:ref[k] for k in ('path','sha256')},'Immutable evidence reference changed')
 require('bytes' not in ref or type(ref['bytes']) is int and Path(ref['path']).stat().st_size==ref['bytes'],'Evidence size differs')
 return json.loads(Path(ref['path']).read_text())


def sources():
 e=ROOT/'research/tokenizer-evidence/post-projection-boundary-001'
 closure=e/'relu-squared-prerequisite-chain001-closure001'
 require(reference(closure/'review.json')['sha256']=='fdfeb06a5101a640228a208bb50a7b63a4632d99e92761d160219e23f5e798c6','Original accepted FFN predecessor witness changed')
 review=checked(reference(closure/'review.json'))
 require(review['status']=='closed_accepted_original_real_ffn_four_twenty_chain' and len(review['sources'])==1201
     and review['protected_weight_bytes']==3355182476 and review['optimizer_updates']==72,'Accepted original own real4/twenty predecessor required')
 require(checked(reference(closure/'guard.json'))['status']=='finished'
     and checked(reference(closure/'STOP'))['status']=='complete_one_time_original_ffn_prerequisite_chain_closure','Original closure guard/STOP required')
 for phase in ('worker','pilot'):
  stop=ROOT/f'research_runs/tokenizer-relu-squared-{phase}-v1-001/STOP'
  prior=[v for v in review['original_metadata'] if v['original']==str(stop)]
  require(len(prior)==1 and reference(stop)['sha256']==prior[0]['sha256']
      and checked(reference(stop))['status']=='three_arm_ffn_'+phase+'_complete','Original phase closure changed')
 pins=dict(review['sources'])
 for name,h in pins.items():require(reference(ROOT/name)['sha256']==h,'Immutable predecessor changed: '+name)
 expected={'__init__.py','contract.py','training.py','validation.py','statistics.py','worker.py','dispatch.py','execution.py','benchmark.py','final_review.py'}
 present={p.name for p in Path(__file__).parent.glob('*.py')}
 require(present==expected,'All ten own scientific source files required;partial SOURCE preparation cannot run')
 for name in sorted(expected):
  p=Path(__file__).parent/name;pins[str(p.relative_to(ROOT))]=reference(p)['sha256']
 require(len(pins)==1211,'Exact1201predecessors+tenfile full namespace required');return pins


class Guard(StrictMpsGuard):
 def __init__(self,path,seconds=7200):
  require(seconds in (600,7200),'Fixed fullworker/freshinference guard horizons required');super().__init__(path,ROOT);self.limits['timeout_seconds']=seconds


def cohort_valid(cohort):
 names={'initializer','private_ungated','train','eval','global_cpu','global_mps'}
 require(type(cohort) is dict and set(cohort)==names and all(type(v) is int and 0<=v<2**63 for v in cohort.values())
     and len(set(cohort.values()))==6,'Six distinct explicit audited cohort roles required')
 require(cohort['private_ungated']==ungated_seed(cohort['initializer']),'Exact audited private FFN domain role required');return cohort


def roles_valid(roles):
 require(type(roles) is dict and set(roles)=={'screen','fresh','bootstrap'} and type(roles['fresh']) is list and len(roles['fresh'])==3,'Screen/three fresh/bootstrap role layout required')
 cohorts=[cohort_valid(roles['screen']),*[cohort_valid(v) for v in roles['fresh']]]
 values=[roles['bootstrap'],*[v for c in cohorts for v in c.values()]]
 require(all(type(v) is int and 0<=v<2**63 for v in values) and len(set(values))==25,'All25 scientific roles must be distinct and separately audited');return roles


def spec(slot,arm,cohort,lr):
 cohort_valid(cohort);require(type(slot) is int and 0<=slot<18 and arm in ARMS and type(lr) is float and lr in LRS,'Exact finite scientific trajectory required')
 cfg=config(arm,1200,cohort['initializer'],cohort['train'],cohort['eval'],lr,global_cpu_seed=cohort['global_cpu'],global_mps_seed=cohort['global_mps'])
 base=RUN/f'evidence/quality/{slot:02d}';cfg.training.eval_every=300;cfg.training.eval_iters=1
 cfg.relu_squared_quality_execution=dict(kind='ffn_full_quality_execution_v1',sources_sha256=fingerprint(sources()),target_contract_sha256=DESIGN['target_contract_sha256'])
 cfg.training.checkpoint_dir=str(base/'checkpoints');cfg.training.metrics_file=str(base/'metrics.jsonl');cfg.training.summary_file=str(base/'worker.json')
 return dict(stage='quality',slot=slot,arm=arm,cohort=plain(cohort),config=plain(cfg),weight_cap=CAPS[arm])


def screen_specs(roles):
 roles_valid(roles);return [spec(3*i+j,a,roles['screen'],lr) for i,lr in enumerate(LRS) for j,a in enumerate(ARMS)]


def fresh_specs(roles,choices):
 roles_valid(roles);require(type(choices) is dict and set(choices)==set(ARMS) and all(type(v) is float and v in LRS for v in choices.values()),'Only independent finite LR choices required')
 return [spec(9+3*p+i,a,cohort,choices[a]) for p,cohort in enumerate(roles['fresh']) for i,a in enumerate(ARMS[p:]+ARMS[:p])]


def prefix_order(accepted,attempted):
 require(type(accepted) is list and type(attempted) is list and len(accepted)==len(attempted)==18 and all(type(v) is bool for v in accepted+attempted),'Exact eighteen-slot state inventory required')
 seen=[i for i,v in enumerate(accepted) if v];require(seen==list(range(len(seen))),'Noncontiguous accepted prefix forbidden')
 require(not any(a and not b for a,b in zip(attempted,accepted)),'Attempted incomplete slot cannot rerun');return len(seen)


def weight_credit(actual,expected):
 require(type(actual) is dict and actual==expected and all(type(n) is str and type(v) is int and v>0 for n,v in expected.items()),'Only accepted current-chain weights may be credited once');return sum(expected.values())


def publish(path,value):
 raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();require(len(raw)<=1024**2,'Bounded immutable JSON required')
 path=Path(path)
 if path.resolve().is_relative_to(B.resolve()):
  require(sum(p.stat().st_size for p in B.rglob('*') if p.is_file())+len(raw)<=128*1024**2,'Complete own sidecar cap exceeded')
 with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
