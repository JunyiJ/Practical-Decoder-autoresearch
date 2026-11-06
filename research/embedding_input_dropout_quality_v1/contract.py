"""Six equal-LR screens plus three fresh pairs,with tested stage high-water accounting."""
import math
from pathlib import Path
from research.body_allocation_model_v1.contract import ROOT,plain,require
from research.body_allocation_model_v1.evidence import reference
from research.embedding_input_dropout_resume_v1.contract import config
from research.embedding_input_dropout_campaign_v1.rules import ARMS,PROBABILITIES,PAIR_CAP,RESERVE,EXTERNAL,ENTRY
B=ROOT/'research/tokenizer-evidence/embedding-input-dropout-preparation-001';RUN=ROOT/'research_runs/tokenizer-embedding-input-dropout-quality-v1-001'
PLAN=B/'quality001-plan.json';FREEZE=B/'quality001-freeze.json';LRS=(.000625,.00125,.0025);WEIGHT_CAPS={a:PAIR_CAP for a in ARMS}
# Before stage retirement:2pilots+6screens=8pairs. After exact pilot/loser
# retirement:2selected+6fresh=8pairs. No future deletion is subtracted from used.
PEAK_WEIGHTS=8*PAIR_CAP;METADATA=256*1024**2;SIDECARS=64*1024**2
FLOOR=RESERVE+EXTERNAL+ENTRY+PEAK_WEIGHTS+METADATA+SIDECARS;TARGET=FLOOR+2*1024**3
DESIGN=dict(arms=list(ARMS),probabilities=list(PROBABILITIES),candidate='input_feature_p10',control='input_zero',steps=1200,targets=1228800,
    learning_rates=list(LRS),screens_per_arm=3,fresh_seeds=3,fresh_trajectories=6,
    selection='Minimum complete H200 validation CE independently per arm;lower LR breaks exact ties.',comparisons=['input_feature_p10-minus-input_zero'],
    mean_delta_threshold=-.02,every_seed_negative=True,blocks=[2048,4096,8192],draws=2000,bootstrap_generator='PCG64',quantile='linear',upper_quantile=1-.05/3,
    simultaneous_scope='One contrast times three block lengths,Bonferroni,conditional on three trained seeds and adaptively reused validation text.',
    rescue=False,replacement_seeds=False,retry_failed_attempt=False,holdout_scored=False,cost_only_adoption=False,
    retained_high_water='2pilots+6screens=8pairs;after exact retirement2selected+6fresh=8pairs.',no_retirement_credit_until_actual=True,selected_and_fresh_protected=True,
    equality='Same28492416parameters/88tensors/sharedinit/Muon44/AdamW44/data/targets/schedule/EMA41;only independent inverted input feature mask. Eval no mask;all attention/residual/FFN dropout stays0.',
    category='Evidence-grounded local regularization/exploitation,not quota-driven structural exploration.',compute_axis='Equal sampled targets and equally bounded LR tuning,not equal training time/FLOPs.')
def quality_sources():return {str(p.relative_to(ROOT)):reference(p)['sha256'] for p in sorted(Path(__file__).parent.glob('*.py'))}
def spec(slot,arm,seed,lr,evaluation):
    require(arm in ARMS,'Exact two-arm input policy required');cfg=config(PROBABILITIES[ARMS.index(arm)],steps=1200,seed=seed,lr=lr);t=cfg.training;base=RUN/f'evidence/quality/{slot:02d}'
    t.eval_seed=evaluation;t.checkpoint_dir=str(base/'checkpoints');t.metrics_file=str(base/'metrics.jsonl');t.summary_file=str(base/'worker.json')
    return dict(slot=slot,stage='quality',arm=arm,config=plain(cfg),weight_cap=PAIR_CAP)
def screen_specs(seeds):return [spec(i*2+j,a,seeds['screen'],lr,seeds['evaluation']) for i,lr in enumerate(LRS) for j,a in enumerate(ARMS)]
def fresh_specs(seeds,selection):
    require(set(selection)==set(ARMS) and all(type(v) is float and v in LRS for v in selection.values()),'Finite equally bounded per-arm selection required')
    return [spec(6+p*2+i,a,seed,selection[a],seeds['evaluation']) for p,seed in enumerate(seeds['fresh']) for i,a in enumerate(ARMS[p%2:]+ARMS[:p%2])]
def select(rows):
    require(type(rows) is list and len(rows)==6,'Complete six-trajectory screen required')
    require(sorted(r['slot'] for r in rows)==list(range(6)) and all(r['arm']==ARMS[r['slot']%2] and r['lr']==LRS[r['slot']//2] and type(r['ce']) is float and math.isfinite(r['ce']) and r['ce']>=0 for r in rows),'Screen layout/finite loss differs')
    choices={};audits={}
    for arm in ARMS:
        values=[r for r in rows if r['arm']==arm];require(sorted(r['lr'] for r in values)==list(LRS),'Exact equal three-rate allowance required');best=min(values,key=lambda r:(r['ce'],r['lr']))
        choices[arm]=best['lr'];audits[arm]=dict(selected_slot=best['slot'],learning_rate=best['lr'],ce=best['ce'])
    return choices,audits
def weight_credit(actual,expected):
    require(type(actual) is dict and actual==expected and all(type(v) is int and v>0 for v in expected.values()),'Extra/unaccepted/duplicated weight credit forbidden')
    return sum(expected.values())
