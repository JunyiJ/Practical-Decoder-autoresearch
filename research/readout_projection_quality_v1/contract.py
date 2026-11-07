"""Exact full-panel design,configs,source bridge and retained-output budget."""
import json,math,os
from pathlib import Path
from research.body_allocation_model_v1.contract import ROOT,require,plain,fingerprint
from research.body_allocation_model_v1.evidence import reference
from research.readout_projection_resume_v1.data import config
from research.block_norm_v1.mps_guard import StrictMpsGuard
from .statistics import ARMS,LRS,select
B=ROOT/'research/tokenizer-evidence/projection-quality-preparation-001';RUN=ROOT/'research_runs/tokenizer-readout-projection-quality-v1-001'
PLAN=B/'projection-quality001-plan.json';FREEZE=B/'projection-quality001-freeze.json';SELECTION=B/'projection-quality001-selection.json';CONFIRM=B/'projection-quality001-confirmation-plan.json'
CAPS=dict(native=640*1024**2,linear_projection1=640*1024**2,tanh_projection1=640*1024**2);RESERVE=8*1024**3;ENTRY=2*1024**2
QUALITY_FLOOR=6*sum(CAPS.values())+512*1024**2+128*1024**2+RESERVE+2*1024**3+ENTRY
FLOOR=QUALITY_FLOOR;TARGET=FLOOR+2*1024**3
SCREEN_KIND='readout_projection_finite_nine_screen_v1';FRESH_KIND='readout_projection_resolved_original_fresh_panel_v1'
DESIGN=dict(arms=list(ARMS),candidate='tanh_projection1',controls=['native','linear_projection1'],steps=1200,targets=1228800,ema_snapshots=41,learning_rates=list(LRS),screen_trajectories=9,fresh_triples=3,fresh_trajectories=9,
 selection='Minimum complete H200 MPS native CE independently per arm;lower LR exact tie.',fresh_order=['native,linear_projection1,tanh_projection1','linear_projection1,tanh_projection1,native','tanh_projection1,native,linear_projection1'],
 mean_ce_delta_threshold=-.02,every_seed_negative=True,native_token_blocks=[2048,4096,8192],draws=2000,generator='PCG64',pcg64_reinitialized_each_width=True,same_indices_across_seeds_and_contrasts=True,quantile='linear',upper_quantile=1-.05/6,
 target_tokens=34278,target_bytes=111536,target_contract_sha256='61945b178404c035a322de7b7513e6ec4399b06a2275a036931f785018dccd2a',target_ids_sha256='dad00eaab7c2076736432b83eb9f1cc404c87aa5e4820b5f7abebf384c7fcdaf',
 category='LOCAL readout architecture followup in incumbent-improvement allocation; broad/interaction allocation carried forward',resource_axis='Equal native token targets/updates,sampling seeds and finite LR allowance;not equal FLOPs,time or total active/stored parameters.',
 confidence_scope='Conditional on trained seeds and reused validation text;not holdout or population-seed confidence.',
 keep_all_screens_and_fresh=True,full_weight_high_water=6*sum(CAPS.values()),old_payload_or_future_retirement_or_hardlink_credit=False,rescue=False,seed_replacement=False,extra_tuning=False,holdout_scored=False,cost_only_adoption=False,control_fallback=False,separate_adoption_consumer_review=True)

def checked(ref):
 require(type(ref) is dict and set(ref) in ({'path','sha256'},{'path','sha256','bytes'}) and reference(ref['path'])=={k:ref[k] for k in ['path','sha256']},'Immutable evidence reference changed')
 require('bytes' not in ref or type(ref['bytes']) is int and Path(ref['path']).stat().st_size==ref['bytes'],'Evidence size differs')
 return json.loads(Path(ref['path']).read_text())
def sources():
 # Fixed actual predecessor witness, also in generated path-locator fixtures.
 source_e=ROOT/'research/tokenizer-evidence/post-readout-boundary-001'
 close_path=source_e/'projection-prerequisite-chain001-closure001/review.json'
 require(reference(close_path)['sha256']=='5c5d210584500fbf0b02fa205c1c989f5816d7ff3f0719a663f68754c13a5c53','Original projection prerequisite closure changed')
 close=checked(reference(close_path))
 require(close['status']=='accepted_closed_original_projection_real_four_twenty_prerequisite_chain' and close['sources_preserved']==1151 and close['retained_archive_bytes']==3359496280,'All actual projection prerequisites required')
 require(checked(reference(source_e/'projection-prerequisite-chain001-closure001/guard.json'))['status']=='finished' and checked(reference(source_e/'projection-prerequisite-chain001-closure001/STOP'))['status']=='complete_read_only_projection_prerequisite_closure','Original closed prerequisite guard and STOP required')
 for phase in ('worker','pilot'):
  refs=close['original_phase_receipts'][phase];checked(refs['review']);checked(refs['STOP'])
 pilot=checked(close['original_phase_receipts']['pilot']['plan']);pins=dict(pilot['sources']);require(len(pins)==1151,'All1151 immutable projection predecessors required')
 for name,h in pins.items():require(reference(ROOT/name)['sha256']==h,'Immutable predecessor changed: '+name)
 for p in sorted(Path(__file__).parent.glob('*.py')):pins[str(p.relative_to(ROOT))]=reference(p)['sha256']
 require(len(pins)==1161,'Exact additive ten-file full-quality namespace required');return pins
class Guard(StrictMpsGuard):
 def __init__(self,path,seconds=7200):
  require(seconds in (600,7200),'Original full/final MPS guard horizons required');super().__init__(path,ROOT);self.limits['timeout_seconds']=seconds

def seeds_valid(seeds):
 require(type(seeds) is dict and set(seeds)=={'screen','evaluation','fresh','bootstrap'} and type(seeds['fresh']) is list and len(seeds['fresh'])==3,'Complete prospective seed panel required')
 values=[seeds['screen'],seeds['evaluation'],*seeds['fresh'],seeds['bootstrap']];require(all(type(v) is int and 0<=v<2**63 for v in values) and len(set(values))==6,'Distinct declared scientific seed roles required');return seeds

def spec(slot,arm,seed,lr,evaluation):
 require(type(slot) is int and 0<=slot<18 and arm in ARMS and type(lr) is float and lr in LRS,'Exact finite trajectory required')
 cfg=config(arm,1200,seed,seed,evaluation,lr);base=RUN/f'evidence/quality/{slot:02d}';cfg.training.eval_seed=evaluation
 cfg.training.eval_every=300;cfg.training.eval_iters=1
 cfg.projection_quality_execution=dict(kind='projection_full_quality_execution_v1',sources_sha256=fingerprint(sources()),target_contract_sha256=DESIGN['target_contract_sha256'])
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
 path=Path(path)
 if path.resolve().is_relative_to(B.resolve()):
  require(sum(p.stat().st_size for p in B.rglob('*') if p.is_file())+len(raw)<=128*1024**2,'Complete new quality sidecar cap exceeded')
 with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
