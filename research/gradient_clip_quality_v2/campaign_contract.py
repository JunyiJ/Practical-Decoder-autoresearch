"""Exact1200-step fixed-LR gradient-clip configurations, reusing accepted immutable pilot contracts."""
import copy,re
from pathlib import Path
from research.gradient_clip_campaign_v1 import pilot as pilot, evidence_contract as predecessor
from research.block_norm_v1 import campaign_contract as inherited
from research.gradient_clip_quality_v1 import campaign_rules as rules
from research.gradient_clip_quality_v1 import campaign_contract as original
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
    result=original.source_inventory()
    import json
    prior=json.loads((pilot.B/'quality001-plan.json').read_text())
    require(result==prior['sources'],'Original687 quality sources changed')
    for p in sorted((ROOT/'research/gradient_clip_quality_v2').rglob('*.py')):result[str(p.relative_to(ROOT))]=file_sha(p,root=ROOT)
    return result

def verify_sources(expected):require(expected==source_inventory(),'Source inventory changed')

def trial_config(output,spec,seeds):
    rules.validate_seeds(seeds);path=Path(output)
    require(path.is_absolute() and '..' not in path.parts and path.is_relative_to(ROOT),'Absolute repository output required')
    parts=path.relative_to(ROOT).parts
    require(len(parts)==5 and parts[0]=='research_runs' and re.fullmatch(r'tokenizer-gradient-clip-quality-v[12]-[a-z0-9][a-z0-9-]*',parts[1]) is not None
        and parts[2:4]==('evidence','quality') and re.fullmatch(r'[0-9]{2}',parts[4]) is not None,'Exact fixed-LR gradient-clip quality slot path required')
    for p in [path,*path.parents]:
        require(not p.is_symlink(),'Symlink output forbidden')
        if p==ROOT:break
    slot=int(parts[4]);require(slot<6,'Finite six fresh trajectories required')
    require(canonical(spec)==canonical(rules.fresh_specs(seeds)[slot]),'Exact fixed-LR paired spec required')
    if slot==0:
        import json
        plan=json.loads((pilot.B/'quality001-plan.json').read_text())
        require(seeds==plan['seeds'] and spec==plan['fresh'][0],'Original slot00 only; no replacement seed')
        return original.trial_config(ROOT/'research_runs/tokenizer-gradient-clip-quality-v1-001/evidence/quality/00',spec,seeds)
    require(parts[1]=='tokenizer-gradient-clip-quality-v2-001','Remaining slots require new continuation root')
    cfg=copy.deepcopy(pilot.config(rules.ARMS.index(spec['arm']),seeds));t=cfg['training']
    t.update(seed=spec['seed'],learning_rate=spec['learning_rate'],max_iters=1200,checkpoint_dir=str(path/'checkpoints'),
        metrics_file=str(path/'metrics.jsonl'),summary_file=str(path/'summary.json'),eval_every=300)
    t['lr_schedule'].update(total_steps=1200,start_step=spec['cooldown_start'])
    t['weight_averaging'].update(total_steps=1200,start_step=800,interval=10)
    return cfg

def validate_trial_config(cfg,output,spec,seeds):require(canonical(cfg)==canonical(trial_config(output,spec,seeds)),'Exact finite fixed-LR gradient-clip quality config required')
