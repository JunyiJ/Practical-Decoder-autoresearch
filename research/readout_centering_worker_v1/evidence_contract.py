"""Static predecessor identities plus additive readout-centering worker adapters."""
import json
from research.local_mix_v1.evidence_contract import ROOT,require,canonical,fingerprint,file_sha
B=ROOT/'research/tokenizer-evidence/readout-centering-preparation-001'
def source_inventory():
    plan=json.loads((B/'model-resume001-plan.json').read_text())
    bound=dict(plan['frozen_sources']);bound.update(plan['sources'])
    result={str((ROOT/n).relative_to(ROOT)):h for n,h in bound.items()}
    for n,h in result.items():require(file_sha(ROOT/n,root=ROOT)==h,'Tested predecessor source changed')
    for p in sorted((ROOT/'research/readout_centering_worker_v1').rglob('*.py')):result[str(p.relative_to(ROOT))]=file_sha(p,root=ROOT)
    return result
def verify_sources(expected):require(expected==source_inventory(),'Readout-centering runtime/acceptance sources changed')
