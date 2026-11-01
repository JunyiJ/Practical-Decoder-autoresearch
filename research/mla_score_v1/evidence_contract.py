"""Source bindings for actual score prerequisites; no plan, ledger or dispatcher."""
import json
from research.ffn_quality_v1 import campaign_contract as previous
ROOT=previous.ROOT
require=previous.require
canonical=previous.canonical
fingerprint=previous.fingerprint
file_sha=previous.file_sha

def source_inventory():
    result=previous.source_inventory()
    frozen=json.loads((ROOT/'research/tokenizer-evidence/ffn-structure-preparation-001/quality001-plan.json').read_text())['sources']
    require(result==frozen,'Frozen predecessor source changed')
    for p in sorted((ROOT/'research/mla_score_v1').rglob('*.py')):result[str(p.relative_to(ROOT))]=file_sha(p,root=ROOT)
    return result

def verify_sources(expected):require(expected==source_inventory(),'Score runtime source changed')
