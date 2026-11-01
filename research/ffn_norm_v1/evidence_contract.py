"""Source binding for actual FFN product normalization prerequisites only."""
import json
from research.mla_score_quality_v1 import campaign_contract as previous
ROOT=previous.ROOT
require=previous.require
canonical=previous.canonical
fingerprint=previous.fingerprint
file_sha=previous.file_sha

def source_inventory():
    result=previous.source_inventory()
    freeze=json.loads((ROOT/'research/tokenizer-evidence/mla-score-preparation-001/quality001-freeze.json').read_text())
    frozen=json.loads(open(freeze['plan']['path']).read())['sources']
    require(result==frozen,'Frozen score predecessor source changed')
    prior=json.loads((ROOT/'research/tokenizer-evidence/ffn-norm-preparation-001/model001.json').read_text())
    for n,h in prior['new_sources'].items():
        require(file_sha(ROOT/n,root=ROOT)==h,'Tested full-model runtime changed')
    for p in sorted((ROOT/'research/ffn_norm_v1').rglob('*.py')):result[str(p.relative_to(ROOT))]=file_sha(p,root=ROOT)
    return result

def verify_sources(expected):require(expected==source_inventory(),'FFN product normalization source changed')
