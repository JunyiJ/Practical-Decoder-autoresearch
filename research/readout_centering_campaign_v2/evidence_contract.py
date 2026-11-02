"""Pin the accepted worker source; add only finite campaign orchestration."""
import json
from research.readout_centering_campaign_v1 import evidence_contract as previous
ROOT=previous.ROOT
require=previous.require
canonical=previous.canonical
fingerprint=previous.fingerprint
file_sha=previous.file_sha

def source_inventory():
    result=previous.source_inventory()
    accepted=json.loads((ROOT/'research/tokenizer-evidence/readout-centering-preparation-001/trainer-worker001.json').read_text())
    require(all(result.get(n)==h for n,h in accepted['sources'].items()),'Accepted worker/runtime source changed')
    frozen=json.loads((ROOT/'research/tokenizer-evidence/readout-centering-preparation-001/pilot-integration001-plan.json').read_text())['sources']
    require(result==frozen,'Original closed pilot integration sources changed')
    for p in sorted((ROOT/'research/readout_centering_campaign_v2').rglob('*.py')):
        result[str(p.relative_to(ROOT))]=file_sha(p,root=ROOT)
    return result

def verify_sources(expected):require(expected==source_inventory(),'Finite pilot source changed')
