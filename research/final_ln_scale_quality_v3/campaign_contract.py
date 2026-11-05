"""Preserve912 sources; add only explicit same-panel continuation contracts."""
import json
from research.final_ln_scale_quality_v2 import campaign_contract as predecessor
ROOT=predecessor.ROOT
LIMITS=predecessor.LIMITS
require=predecessor.require
canonical=predecessor.canonical
fingerprint=predecessor.fingerprint
file_sha=predecessor.file_sha
environment=predecessor.environment
bind_data=predecessor.bind_data
frozen_predecessors=predecessor.frozen_predecessors
trial_config=predecessor.trial_config
validate_trial_config=predecessor.validate_trial_config
ENTRY_HEADROOM=1458176

def source_inventory():
    result=predecessor.source_inventory()
    original=json.loads((ROOT/'research/tokenizer-evidence/final-ln-scale-preparation-001/quality001-plan.json').read_text())
    require(result==original['sources'],'Original912 source contract changed')
    for p in sorted((ROOT/'research/final_ln_scale_quality_v3').rglob('*.py')):result[str(p.relative_to(ROOT))]=file_sha(p,root=ROOT)
    return result

def verify_sources(expected):require(expected==source_inventory(),'Additive continuation sources changed')
