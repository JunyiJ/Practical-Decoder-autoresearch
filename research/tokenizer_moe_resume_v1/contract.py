import hashlib
import json
from pathlib import Path
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import environment

E=ROOT/'research/tokenizer-evidence/post-prefix-boundary-001'


def sources():
    plan=json.loads((E/'model001-plan.json').read_text())
    proof=json.loads((E/'model001-result.json').read_text());guard=json.loads((E/'model001-guard.json').read_text())
    require(proof['status']=='three_arm_model_optimizer_and_cache_integration_passed'
        and proof['plan_sha256']==hashlib.sha256((E/'model001-plan.json').read_bytes()).hexdigest()
        and guard['status']=='finished' and guard['reason'] is None,'Actual complete model prerequisite required')
    pins=dict(plan['sources']);require(len(pins)==1014,'Exact model predecessor inventory required')
    for p,h in pins.items():require(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,'Immutable predecessor changed: '+p)
    for p in sorted(Path(__file__).parent.glob('*.py')):pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    require(len(pins)==1018,'Exact four-file additive data/transport namespace required')
    return pins


def runtime():
    actual=environment()
    expected=json.loads((ROOT/'research/tokenizer-evidence/tokenizer-capacity-boundary-001/quality001-plan.json').read_text())['environment']
    require(actual==expected,'Strict proved runtime/thread/library identity changed')
    return actual
