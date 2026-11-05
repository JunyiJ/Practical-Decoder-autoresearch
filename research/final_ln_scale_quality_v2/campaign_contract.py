"""Preserve906 tested sources; add only terminal-evidence verification routing."""
import json
from research.final_ln_scale_quality_v1 import campaign_contract as predecessor
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

def source_inventory():
    result=predecessor.source_inventory()
    review=json.loads((ROOT/'research/tokenizer-evidence/final-ln-scale-preparation-001/quality-integration001-plan.json').read_text())
    require(result==review['sources'],'Tested906 sources changed')
    for p in sorted((ROOT/'research/final_ln_scale_quality_v2').rglob('*.py')):result[str(p.relative_to(ROOT))]=file_sha(p,root=ROOT)
    return result

def verify_sources(expected):require(expected==source_inventory(),'Additive quality verifier sources changed')
