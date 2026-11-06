"""Real base data/training contract plus separately bound input-regularization policy."""
import json
from pathlib import Path
from research.body_allocation_model_v1.contract import ROOT,require,plain,canonical,config as original_config
from research.body_allocation_model_v1.evidence import reference
from research.embedding_input_dropout_v1.model import config as model_config,signature

def config(probability,steps=4,seed=600404,lr=.00125):
    cfg=original_config('incumbent',steps=steps,seed=seed,lr=lr)
    cfg.input_embedding_regularization=model_config(probability)
    return cfg

def architecture(cfg):
    reg=plain(cfg.input_embedding_regularization)
    require(canonical(cfg.model)==canonical(reg['base_model']),'Genuine original backbone config required')
    return signature(reg)

def sources():
    b=ROOT/'research/tokenizer-evidence/body-allocation-preparation-001/quality001-closure001.json';d=json.load(b.open())
    pins={**d['sources'],**d['pilot_sources'],**d['quality_sources']}
    pins.update(json.load((ROOT/'research/tokenizer-evidence/exploration-boundary-002/continuation-completed001.json').open())['new_inference_diagnostic_sources'])
    for n,h in pins.items():require(reference(ROOT/n)['sha256']==h,'Frozen predecessor changed '+n)
    for ns in ('embedding_input_dropout_v1','embedding_input_dropout_resume_v1'):
        for p in sorted((ROOT/'research'/ns).glob('*.py')):pins[str(p.relative_to(ROOT))]=reference(p)['sha256']
    return pins
