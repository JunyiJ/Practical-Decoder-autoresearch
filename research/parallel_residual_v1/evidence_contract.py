"""Static predecessor identities plus additive decay worker adapters."""
import json
from research.local_mix_v1.evidence_contract import ROOT,require,canonical,fingerprint,file_sha
B=ROOT/'research/tokenizer-evidence/parallel-residual-preparation-001'
def source_inventory():
    plan=json.loads((B/'model-resume001-plan.json').read_text())
    result=dict(plan['frozen_sources']);result.update(plan['sources'])
    for n,h in result.items():require(file_sha(ROOT/n,root=ROOT)==h,'Tested predecessor source changed')
    for p in sorted((ROOT/'research/parallel_residual_v1').rglob('*.py')):result[str(p.relative_to(ROOT))]=file_sha(p,root=ROOT)
    return result
def verify_sources(expected):require(expected==source_inventory(),'Parallel residual runtime/acceptance sources changed')
