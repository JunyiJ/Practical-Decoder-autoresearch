"""Exact prospective screen/fresh slots and complete retained high-water."""
from pathlib import Path
import math
from research.body_allocation_model_v1.contract import ROOT,config,plain,require
from research.body_allocation_model_v1.evidence import reference
from research.body_allocation_campaign_v1.rules import ARMS,LRS,WEIGHT_CAPS,DESIGN,FLOOR as PILOT_FLOOR,RESERVE

B=ROOT/'research/tokenizer-evidence/body-allocation-preparation-001'
RUN=ROOT/'research_runs/tokenizer-body-allocation-quality-v1-001'
PLAN=B/'quality001-plan.json';FREEZE=B/'quality001-freeze.json'
# Additional allowance for prospective exact-retirement shards and portable
# acceptance/statistics sidecars outside the bounded numerical run directories.
SIDECARS=64*1024**2
FLOOR=PILOT_FLOOR+SIDECARS
TARGET=FLOOR+2*1024**3

def quality_sources():
    return {str(p.relative_to(ROOT)):reference(p)['sha256'] for p in sorted(Path(__file__).parent.glob('*.py'))}

def spec(slot,arm,seed,lr,evaluation):
    cfg=config(arm,1200,seed,lr);t=cfg.training;base=RUN/f'evidence/quality/{slot:02d}'
    t.eval_seed=evaluation;t.checkpoint_dir=str(base/'checkpoints');t.metrics_file=str(base/'metrics.jsonl');t.summary_file=str(base/'worker.json')
    return dict(slot=slot,stage='quality',arm=arm,config=plain(cfg),weight_cap=WEIGHT_CAPS[arm])

def screen_specs(seeds):
    return [spec(i*3+j,arm,seeds['screen'],lr,seeds['evaluation']) for i,lr in enumerate(LRS) for j,arm in enumerate(ARMS)]

def fresh_specs(seeds,selection):
    require(set(selection)==set(ARMS) and all(type(v) is float and v in LRS for v in selection.values()),'Finite frozen per-arm selection required')
    rows=[]
    for pair,seed in enumerate(seeds['fresh']):
        order=ARMS[pair:]+ARMS[:pair]
        for index,arm in enumerate(order):rows.append(spec(9+pair*3+index,arm,seed,selection[arm],seeds['evaluation']))
    return rows

def select(rows):
    require(type(rows) is list and len(rows)==9,'Complete nine-trajectory LR screen required')
    require(sorted(r['slot'] for r in rows)==list(range(9)) and all(r['arm']==ARMS[r['slot']%3] and r['lr']==LRS[r['slot']//3]
        and type(r['ce']) is float and math.isfinite(r['ce']) and r['ce']>=0 for r in rows),'Screen layout/finite CE differs')
    choices={};audits={}
    for arm in ARMS:
        values=[r for r in rows if r['arm']==arm];require(len(values)==3,'Three rates per arm required')
        require(sorted(r['lr'] for r in values)==list(LRS),'Exact equally bounded LR allowance required')
        best=min(values,key=lambda r:(r['ce'],r['lr']));choices[arm]=best['lr'];audits[arm]=dict(selected_slot=best['slot'],learning_rate=best['lr'],ce=best['ce'])
    return choices,audits

def weight_credit(actual,expected):
    require(type(actual) is dict and type(expected) is dict and actual==expected
        and all(type(v) is int and v>0 for v in expected.values()),'Extra/unaccepted/duplicated weight credit forbidden')
    return sum(expected.values())
