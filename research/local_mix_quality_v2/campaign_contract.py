"""Exact1200-step local-mixing configurations, reusing accepted immutable pilot contracts."""
import copy,re
from pathlib import Path
from research.local_mix_campaign_v1 import pilot002 as pilot, evidence_contract as predecessor
from research.block_norm_v1 import campaign_contract as inherited
from . import campaign_rules as rules
ROOT=inherited.ROOT
LIMITS=inherited.LIMITS
require=inherited.require
canonical=inherited.canonical
fingerprint=inherited.fingerprint
file_sha=inherited.file_sha
environment=inherited.environment
bind_data=inherited.bind_data
frozen_predecessors=inherited.frozen_predecessors

def source_inventory():
    result=predecessor.source_inventory()
    # Bind the accepted pilot source exactly, then add only this new namespace.
    import json
    prior=json.loads((pilot.B/'pilot001-plan.json').read_text())
    require(result==prior['sources'],'Frozen pilot source changed')
    for p in sorted((ROOT/'research/local_mix_quality_v2').rglob('*.py')):result[str(p.relative_to(ROOT))]=file_sha(p,root=ROOT)
    return result

def verify_sources(expected):require(expected==source_inventory(),'Source inventory changed')

def trial_config(output,spec,seeds):
    rules.validate_seeds(seeds);path=Path(output)
    require(path.is_absolute() and '..' not in path.parts and path.is_relative_to(ROOT),'Absolute repository output required')
    parts=path.relative_to(ROOT).parts
    require(len(parts)==5 and parts[0]=='research_runs' and re.fullmatch(r'tokenizer-local-mix-quality-v2-[a-z0-9][a-z0-9-]*',parts[1]) is not None
        and parts[2:4]==('evidence','quality') and re.fullmatch(r'[0-9]{2}',parts[4]) is not None,'Exact local-mixing quality slot path required')
    for p in [path,*path.parents]:
        require(not p.is_symlink(),'Symlink output forbidden')
        if p==ROOT:break
    slot=int(parts[4]);require(slot<12,'Finite12trial panel required')
    expected=[rules.screen_specs(seeds)[slot]] if slot<6 else [rules.confirmation_specs(seeds,dict(zip(rules.ARMS,(a,b))))[slot-6] for a in rules.RATES for b in rules.RATES]
    require(any(canonical(spec)==canonical(s) for s in expected),'Finite trial spec differs')
    cfg=copy.deepcopy(pilot.config(rules.ARMS.index(spec['arm']),seeds));t=cfg['training']
    t.update(seed=spec['seed'],learning_rate=spec['lr'],max_iters=1200,checkpoint_dir=str(path/'checkpoints'),
        metrics_file=str(path/'metrics.jsonl'),summary_file=str(path/'summary.json'),eval_every=300)
    t['lr_schedule'].update(total_steps=1200,start_step=800)
    t['weight_averaging'].update(total_steps=1200,start_step=800,interval=10)
    return cfg

def validate_trial_config(cfg,output,spec,seeds):require(canonical(cfg)==canonical(trial_config(output,spec,seeds)),'Exact finite local-mixing quality config required')
