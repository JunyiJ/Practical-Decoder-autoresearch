import json
from pathlib import Path
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference
from research.tokenizer_moe_resume_v1.contract import runtime

E=ROOT/'research/tokenizer-evidence/post-diagnostic-boundary-001'

def sources():
    review=json.loads((E/'review001.json').read_text())
    require(review['status']=='gate_direction_component_and_generated_model_independently_reviewed','Closed generated gate review required')
    for field in ('component','component_plan','component_terminal','model','model_plan','model_terminal'):
        r=review[field];require(reference(r['path'])==r,'Gate prerequisite evidence changed')
    for r in review['guards'].values():
        require(reference(r['path'])==r,'Gate prerequisite guard changed')
        g=json.loads(Path(r['path']).read_text());require(g['status']=='finished' and g['reason'] is None,'Complete gate prerequisite guard required')
    pins=dict(json.loads(Path(review['model_plan']['path']).read_text())['sources']);require(len(pins)==1062,'All immutable gate model predecessors required')
    for p,h in pins.items():require(reference(ROOT/p)['sha256']==h,'Immutable predecessor changed: '+p)
    for p in sorted(Path(__file__).parent.glob('*.py')):pins[str(p.relative_to(ROOT))]=reference(p)['sha256']
    require(len(pins)==1066,'Own four-file transport namespace required')
    return pins
