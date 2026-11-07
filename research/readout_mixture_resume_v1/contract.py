import hashlib
import json
from pathlib import Path
from research.body_allocation_model_v1.contract import ROOT, require
from research.tokenizer_moe_resume_v1.contract import runtime

E = ROOT/'research/tokenizer-evidence/post-gate-boundary-001'


def sources():
    model = E/'readout-model001'
    require(json.loads((model/'STOP').read_text())['status'] == 'complete_generated_full_readout_model_proof', 'Closed full-model prerequisite required')
    pins = dict(json.loads((E/'readout-resume-seed-census001/plan.json').read_text())['sources'])
    require(len(pins) == 1095, 'Exact 1094 predecessors plus new census')
    for p, h in pins.items():
        require(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h, 'Immutable source differs: '+p)
    for p in sorted(Path(__file__).parent.glob('*.py')):
        pins[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    require(len(pins) == 1100, 'Exact five-file own state/scorer namespace')
    return pins
