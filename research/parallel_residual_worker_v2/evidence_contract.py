"""Preserve failed first adapter source, route new validation explicitly."""
import json
from research.local_mix_v1.evidence_contract import ROOT,require,canonical,fingerprint,file_sha
B=ROOT/'research/tokenizer-evidence/parallel-residual-preparation-001'
def source_inventory():
    plan=json.loads((B/'worker-integration001-plan.json').read_text())
    result=dict(plan['sources'])
    for ref in [plan['runner'],*plan['bound_scripts']]:
        from pathlib import Path
        result[str(Path(ref['path']).relative_to(ROOT))]=ref['sha256']
    for n,h in result.items():require(file_sha(ROOT/n,root=ROOT)==h,'Frozen predecessor changed')
    for p in sorted((ROOT/'research/parallel_residual_worker_v2').rglob('*.py')):result[str(p.relative_to(ROOT))]=file_sha(p,root=ROOT)
    return result
def verify_sources(expected):require(expected==source_inventory(),'Parallel-residual worker sources changed')
