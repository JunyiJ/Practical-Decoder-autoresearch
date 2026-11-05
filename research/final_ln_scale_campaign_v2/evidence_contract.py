"""Preserve all894 tested sources; add only the metadata-reference continuation."""
from research.final_ln_scale_campaign_v1 import evidence_contract as previous
ROOT=previous.ROOT
require=previous.require
canonical=previous.canonical
fingerprint=previous.fingerprint
file_sha=previous.file_sha
def source_inventory():
    result=previous.source_inventory()
    for p in sorted((ROOT/'research/final_ln_scale_campaign_v2').rglob('*.py')):
        result[str(p.relative_to(ROOT))]=file_sha(p,root=ROOT)
    return result
def verify_sources(expected):require(expected==source_inventory(),'Bound metadata-reference continuation sources changed')
