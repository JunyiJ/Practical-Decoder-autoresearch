"""Bind the unchanged predecessor and only the new warmup integration sources."""
import json
from research.ffn_norm_quality_v1 import campaign_contract as previous
ROOT=previous.ROOT
require=previous.require
canonical=previous.canonical
fingerprint=previous.fingerprint
file_sha=previous.file_sha

def source_inventory():
    frozen=json.loads((ROOT/'research/tokenizer-evidence/warmup-preparation-001/model-resume001-plan.json').read_text())['frozen_sources']
    result=previous.source_inventory()
    require(result==frozen,'Frozen predecessor source changed')
    for p in sorted((ROOT/'research/warmup_v1').rglob('*.py')):
        result[str(p.relative_to(ROOT))]=file_sha(p,root=ROOT)
    return result

def verify_sources(expected):require(expected==source_inventory(),'Warmup source changed')
