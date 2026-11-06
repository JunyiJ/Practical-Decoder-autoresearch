"""Bind genuine prefix config/data identity and all proved predecessor sources."""
import json
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference
from research.gpt2_prefix_data_v1.data import config,architecture,load

def sources():
    p=ROOT/'research/tokenizer-evidence/tokenizer-capacity-boundary-001/data001.json'
    result=json.loads(p.read_text());require(result['status']=='passed_real_data_and_private_sampler_contracts','Actual data proof required')
    pins=dict(result['source_bindings']);require(len(pins)==984,'Exact predecessor source count required')
    for n,h in pins.items():require(reference(ROOT/n)['sha256']==h,'Proven source changed '+n)
    for p in sorted((ROOT/'research/gpt2_prefix_resume_v1').glob('*.py')):pins[str(p.relative_to(ROOT))]=reference(p)['sha256']
    require(len(pins)==987,'Exact additive transport source count required');return pins
