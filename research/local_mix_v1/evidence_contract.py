"""Bind the tested local-mix runtime and immutable predecessors for real workers."""
import json
from research.warmup_quality_v1 import campaign_contract as previous
ROOT=previous.ROOT
require=previous.require
canonical=previous.canonical
fingerprint=previous.fingerprint
file_sha=previous.file_sha

def source_inventory():
    plan=json.loads((ROOT/'research/tokenizer-evidence/warmup-preparation-001/quality001-plan.json').read_text())
    result=dict(plan['sources'])
    for n,h in result.items():require(file_sha(ROOT/n,root=ROOT)==h,'Frozen predecessor source changed')
    model=json.loads((ROOT/'research/tokenizer-evidence/local-mix-preparation-001/model001.json').read_text())
    require(model['status']=='full_model_local_mix_cpu_mps_prerequisite_passed','Passed model prerequisite required')
    for n,h in model['new_sources'].items():require(file_sha(ROOT/n,root=ROOT)==h,'Tested full-model runtime changed')
    for p in sorted((ROOT/'research/local_mix_v1').rglob('*.py')):result[str(p.relative_to(ROOT))]=file_sha(p,root=ROOT)
    return result

def verify_sources(expected):require(expected==source_inventory(),'Local mixing source changed')
