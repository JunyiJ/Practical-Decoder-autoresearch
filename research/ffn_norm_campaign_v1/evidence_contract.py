"""Pin the accepted worker source; add only finite campaign orchestration."""
import json
from research.ffn_norm_v1 import evidence_contract as previous
ROOT=previous.ROOT
require=previous.require
canonical=previous.canonical
fingerprint=previous.fingerprint
file_sha=previous.file_sha

def source_inventory():
    result=previous.source_inventory()
    accepted=json.loads((ROOT/'research/tokenizer-evidence/ffn-norm-preparation-001/trainer-worker001.json').read_text())
    require(result==accepted['sources'],'Accepted worker/runtime source changed')
    for p in sorted((ROOT/'research/ffn_norm_campaign_v1').rglob('*.py')):
        result[str(p.relative_to(ROOT))]=file_sha(p,root=ROOT)
    return result

def verify_sources(expected):require(expected==source_inventory(),'Finite pilot source changed')
