"""Exact full-panel design,configs,source bridge and retained-output budget."""
import json,math
from pathlib import Path
from research.body_allocation_model_v1.contract import ROOT,require,plain
from research.body_allocation_model_v1.evidence import reference
from research.gpt2_prefix_data_v1.data import config
from research.block_norm_v1.mps_guard import StrictMpsGuard
from research.gpt2_prefix_statistics_v1.statistics import ARMS,LRS,select
B=ROOT/'research/tokenizer-evidence/tokenizer-capacity-boundary-001';RUN=ROOT/'research_runs/tokenizer-gpt2-prefix-quality-v1-001'
PLAN=B/'quality001-plan.json';FREEZE=B/'quality001-freeze.json';SELECTION=B/'quality001-selection.json';CONFIRM=B/'quality001-confirmation-plan.json'
CAPS=dict(native=560*1024**2,prefix8192=160*1024**2);RESERVE=8*1024**3;ENTRY=1458176
FLOOR=6*sum(CAPS.values())+256*1024**2+64*1024**2+RESERVE+1267904512+ENTRY;TARGET=FLOOR+2*1024**3
SCREEN_KIND='gpt2_prefix_finite_six_screen_v1';FRESH_KIND='gpt2_prefix_resolved_original_fresh_panel_v1'
DESIGN=dict(arms=list(ARMS),candidate='prefix8192',control='native',steps=1200,targets=1228800,ema_snapshots=41,learning_rates=list(LRS),screen_trajectories=6,fresh_pairs=3,fresh_trajectories=6,
 selection='Minimum complete H200 common-byte BPB independently per arm;lower LR exact tie.',fresh_order=['native,prefix8192','prefix8192,native','native,prefix8192'],
 mean_bpb_delta_threshold=-.01,every_seed_negative=True,raw_byte_blocks=[4096,8192,16384],draws=2000,generator='PCG64',pcg64_reinitialized_each_width=True,quantile='linear',upper_quantile=1-.05/3,
 common_target_bytes=111475,common_target_sha256='ad7da8cdcacc24b5312461c23f2d00aead59d544fcc2fad4fb2de40da4964fe8',prefix_bytes=64,token_targets={'native':34257,'prefix8192':39963},
 category='Evidence-grounded structural exploration',resource_axis='Equal token targets/updates and finite LR allowance;not equal raw context,raw examples/bytes,FLOPs or time.',
 confidence_scope='Conditional on trained seeds and reused validation text;not holdout or population-seed confidence.',coding_scope='Deterministic token coding,not marginalization over all tokenizations;uniform-within-token byte loss allocation is bookkeeping.',
 keep_all_screens_and_fresh=True,full_weight_high_water=6*sum(CAPS.values()),old_payload_or_future_retirement_or_hardlink_credit=False,rescue=False,seed_replacement=False,extra_tuning=False,holdout_scored=False,cost_only_adoption=False,separate_adoption_consumer_review=True)

def checked(ref):
 require(type(ref) is dict and reference(ref['path'])==ref,'Immutable evidence reference changed');return json.loads(Path(ref['path']).read_text())
def sources():
 old=checked(reference(B/'statistics-contracts001.json'));require(old['status']=='passed_independent_common_byte_grouping_selection_and_bootstrap_contracts' and len(old['sources'])==995,'Actual995-source predecessor required');pins=dict(old['sources'])
 for n,h in pins.items():require(reference(ROOT/n)['sha256']==h,'Proven predecessor source changed '+n)
 for p in sorted(Path(__file__).parent.glob('*.py')):pins[str(p.relative_to(ROOT))]=reference(p)['sha256']
 require(len(pins)==1004,'Exact nine-file full-quality source inventory required');return pins
class Guard(StrictMpsGuard):
 def __init__(self,path,seconds=7200):
  require(seconds in (600,7200),'Original full/final MPS guard horizons required');super().__init__(path,ROOT);self.limits['timeout_seconds']=seconds

def seeds_valid(seeds):
 require(type(seeds) is dict and set(seeds)=={'screen','evaluation','fresh','bootstrap'} and type(seeds['fresh']) is list and len(seeds['fresh'])==3,'Complete prospective seed panel required')
 values=[seeds['screen'],seeds['evaluation'],*seeds['fresh'],seeds['bootstrap']];require(all(type(v) is int and v>=0 for v in values) and len(set(values))==6,'Distinct declared scientific seed roles required');return seeds

def spec(slot,arm,seed,lr,evaluation):
 require(type(slot) is int and 0<=slot<12 and arm in ARMS and type(lr) is float and lr in LRS,'Exact finite trajectory required')
 cfg=config(arm,steps=1200,seed=seed,lr=lr);base=RUN/f'evidence/quality/{slot:02d}';cfg.training.eval_seed=evaluation
 cfg.training.checkpoint_dir=str(base/'checkpoints');cfg.training.metrics_file=str(base/'metrics.jsonl');cfg.training.summary_file=str(base/'worker.json')
 return dict(stage='quality',slot=slot,arm=arm,config=plain(cfg),weight_cap=CAPS[arm])
def screen_specs(seeds):
 seeds_valid(seeds);return [spec(i*2+j,a,seeds['screen'],lr,seeds['evaluation']) for i,lr in enumerate(LRS) for j,a in enumerate(ARMS)]
def fresh_specs(seeds,choices):
 seeds_valid(seeds);require(set(choices)==set(ARMS) and all(type(v) is float and v in LRS for v in choices.values()),'Finite equally bounded LR choices required')
 return [spec(6+2*p+i,a,seed,choices[a],seeds['evaluation']) for p,seed in enumerate(seeds['fresh']) for i,a in enumerate(ARMS[p%2:]+ARMS[:p%2])]
def prefix_order(accepted,attempted):
 require(type(accepted) is list and type(attempted) is list and len(accepted)==len(attempted)==12 and all(type(v) is bool for v in accepted+attempted),'Exact twelve-slot state inventory required')
 seen=[i for i,v in enumerate(accepted) if v];require(seen==list(range(len(seen))),'Noncontiguous accepted prefix forbidden')
 require(not any(a and not b for a,b in zip(attempted,accepted)),'Attempted incomplete slot cannot rerun');return len(seen)
def weight_credit(actual,expected):
 require(type(actual) is dict and actual==expected and all(type(n) is str and type(v) is int and v>0 for n,v in expected.items()),'Extra/unaccepted/foreign weight credit forbidden');return sum(expected.values())
def publish(path,value):
 raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();require(len(raw)<=1024**2,'Bounded immutable record required')
 with Path(path).open('xb') as f:f.write(raw)
