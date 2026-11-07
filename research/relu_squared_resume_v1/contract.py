"""Fixed accepted predecessor and five-file own state/scorer source identity."""
import hashlib
import json
from pathlib import Path
from research.body_allocation_model_v1.contract import ROOT, require
from research.tokenizer_moe_resume_v1.contract import runtime

E = ROOT/'research/tokenizer-evidence/post-projection-boundary-001'


def sources():
    proof = E/'relu-squared-model001'
    require(json.loads((proof/'STOP').read_text())['status'] == 'complete_generated_matched_matrix_ffn_actual_model_proof'
            and json.loads((proof/'result.json').read_text())['source_global_cpu_mps_rng_unchanged'] is True
            and json.loads((proof/'mps-guard.json').read_text())['status'] == 'finished',
            'Accepted own actual-model predecessor required')
    pins = dict(json.loads((proof/'plan.json').read_text())['sources'])
    require(len(pins) == 1184, 'Exact accepted actual-model predecessor source inventory required')
    for p, h in pins.items():
        require(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h, 'Immutable source differs: '+p)
    for name in ('__init__.py', 'contract.py', 'data.py', 'checkpoint.py', 'scoring.py'):
        p = Path(__file__).parent/name
        pins[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    require(len(pins) == 1189, 'Exact five-file own state/scorer namespace required')
    return pins
