"""Exact full-panel design,configs,source bridge and retained-output budget."""
import json,math
from pathlib import Path
from research.body_allocation_model_v1.contract import ROOT,require,plain
from research.body_allocation_model_v1.evidence import reference
from research.tokenizer_mla_gate_resume_v1.data import config
from research.block_norm_v1.mps_guard import StrictMpsGuard
from .statistics import ARMS,LRS,select
B=ROOT/'research/tokenizer-evidence/post-diagnostic-boundary-001';RUN=ROOT/'research_runs/tokenizer-mla-gate-quality-v1-001'
PLAN=B/'gate-quality001-plan.json';FREEZE=B/'gate-quality001-freeze.json';SELECTION=B/'gate-quality001-selection.json';CONFIRM=B/'gate-quality001-confirmation-plan.json'
CAPS=dict(none=560*1024**2,static_half=560*1024**2,sigmoid_elementwise_zero=640*1024**2);RESERVE=8*1024**3;ENTRY=1458176
FLOOR=6*sum(CAPS.values())+512*1024**2+64*1024**2+RESERVE+1267904512+ENTRY;TARGET=FLOOR+2*1024**3
SCREEN_KIND='tokenizer_mla_gate_finite_nine_screen_v1';FRESH_KIND='tokenizer_mla_gate_resolved_original_fresh_panel_v1'
DESIGN=dict(arms=list(ARMS),candidate='sigmoid_elementwise_zero',controls=['none','static_half'],steps=1200,targets=1228800,ema_snapshots=41,learning_rates=list(LRS),screen_trajectories=9,fresh_triples=3,fresh_trajectories=9,
 selection='Minimum complete H200 MPS native CE independently per arm;lower LR exact tie.',fresh_order=['none,static_half,sigmoid_elementwise_zero','static_half,sigmoid_elementwise_zero,none','sigmoid_elementwise_zero,none,static_half'],
 mean_ce_delta_threshold=-.02,every_seed_negative=True,native_token_blocks=[2048,4096,8192],draws=2000,generator='PCG64',pcg64_reinitialized_each_width=True,same_indices_across_seeds_and_contrasts=True,quantile='linear',upper_quantile=1-.05/6,
 target_tokens=34278,target_bytes=111536,target_contract_sha256='de78a14fa439a1952c6d0af7372901f8751f77f3dd12b385ff02ec510aa769ad',target_ids_sha256='dad00eaab7c2076736432b83eb9f1cc404c87aa5e4820b5f7abebf384c7fcdaf',
 category='Evidence-grounded structural exploration',resource_axis='Equal native token targets/updates,sampling seeds and finite LR allowance;not equal FLOPs,time or total active/stored parameters.',
 confidence_scope='Conditional on trained seeds and reused validation text;not holdout or population-seed confidence.',
 keep_all_screens_and_fresh=True,full_weight_high_water=6*sum(CAPS.values()),old_payload_or_future_retirement_or_hardlink_credit=False,rescue=False,seed_replacement=False,extra_tuning=False,holdout_scored=False,cost_only_adoption=False,control_fallback=False,separate_adoption_consumer_review=True)

def checked(ref):
 require(type(ref) is dict and reference(ref['path'])==ref,'Immutable evidence reference changed');return json.loads(Path(ref['path']).read_text())
def sources():
 from research.tokenizer_mla_gate_worker_v1.execution import sources as previous
 pins=previous();require(len(pins)==1074,'Actual1074 immutable predecessor bindings required')
 # This witness is constant even in isolated generated path-locator tests.
 source_e=ROOT/'research/tokenizer-evidence/post-diagnostic-boundary-001'
 close=checked(reference(source_e/'continuation-completed003.json'))
 require(close['status']=='all_native_gate_worker_and_resource_prerequisites_closed_independently_reviewed' and close['source_bindings']==1074,'Actual gate worker/pilot closure required')
 for key in ('worker_source_gate','worker_source_guard','worker_builder','pilot_builder','worker_plan','worker_original_review','worker_independent_review','worker_independent_guard','worker_terminal','pilot_plan','pilot_original_review','pilot_independent_review','pilot_independent_guard','pilot_terminal'):
  checked(close[key]) if key not in ('worker_builder','pilot_builder') else require(reference(close[key]['path'])==close[key],'Immutable prepared builder changed')
 for p in sorted(Path(__file__).parent.glob('*.py')):pins[str(p.relative_to(ROOT))]=reference(p)['sha256']
 require(len(pins)==1084,'Exact new ten-file quality runtime required');return pins
class Guard(StrictMpsGuard):
 def __init__(self,path,seconds=7200):
  require(seconds in (600,7200),'Original full/final MPS guard horizons required');super().__init__(path,ROOT);self.limits['timeout_seconds']=seconds

def seeds_valid(seeds):
 require(type(seeds) is dict and set(seeds)=={'screen','evaluation','fresh','bootstrap'} and type(seeds['fresh']) is list and len(seeds['fresh'])==3,'Complete prospective seed panel required')
 values=[seeds['screen'],seeds['evaluation'],*seeds['fresh'],seeds['bootstrap']];require(all(type(v) is int and v>=0 for v in values) and len(set(values))==6,'Distinct declared scientific seed roles required');return seeds

def spec(slot,arm,seed,lr,evaluation):
 require(type(slot) is int and 0<=slot<18 and arm in ARMS and type(lr) is float and lr in LRS,'Exact finite trajectory required')
 cfg=config(arm,steps=1200,seed=seed,eval_seed=evaluation,lr=lr);base=RUN/f'evidence/quality/{slot:02d}';cfg.training.eval_seed=evaluation
 cfg.training.checkpoint_dir=str(base/'checkpoints');cfg.training.metrics_file=str(base/'metrics.jsonl');cfg.training.summary_file=str(base/'worker.json')
 return dict(stage='quality',slot=slot,arm=arm,config=plain(cfg),weight_cap=CAPS[arm])
def screen_specs(seeds):
 seeds_valid(seeds);return [spec(i*3+j,a,seeds['screen'],lr,seeds['evaluation']) for i,lr in enumerate(LRS) for j,a in enumerate(ARMS)]
def fresh_specs(seeds,choices):
 seeds_valid(seeds);require(set(choices)==set(ARMS) and all(type(v) is float and v in LRS for v in choices.values()),'Finite equally bounded LR choices required')
 return [spec(9+3*p+i,a,seed,choices[a],seeds['evaluation']) for p,seed in enumerate(seeds['fresh']) for i,a in enumerate(ARMS[p:]+ARMS[:p])]
def prefix_order(accepted,attempted):
 require(type(accepted) is list and type(attempted) is list and len(accepted)==len(attempted)==18 and all(type(v) is bool for v in accepted+attempted),'Exact eighteen-slot state inventory required')
 seen=[i for i,v in enumerate(accepted) if v];require(seen==list(range(len(seen))),'Noncontiguous accepted prefix forbidden')
 require(not any(a and not b for a,b in zip(attempted,accepted)),'Attempted incomplete slot cannot rerun');return len(seen)
def weight_credit(actual,expected):
 require(type(actual) is dict and actual==expected and all(type(n) is str and type(v) is int and v>0 for n,v in expected.items()),'Extra/unaccepted/foreign weight credit forbidden');return sum(expected.values())
def publish(path,value):
 raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();require(len(raw)<=1024**2,'Bounded immutable record required')
 with Path(path).open('xb') as f:f.write(raw)
