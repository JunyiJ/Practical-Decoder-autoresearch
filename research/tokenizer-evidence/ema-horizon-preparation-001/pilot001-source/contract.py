"""Exact dual-EMA configuration and bounded artifact contract; no dispatch."""
import copy
import json
import re
from pathlib import Path
from research.block_norm_v1 import campaign_contract as inherited, campaign_rules as rules
from research.block_norm_v1.spec import Spec, encode
from research.block_norm_v1.runtime.models.block_norm import GROUPED_POLICY, contract as model_contract
from research.block_norm_v1.runtime.models.grouped_embedding import EMBEDDING_SPEC

ROOT=inherited.ROOT
KIND='tokenizer_mla_dual_ema_horizon_v1'
HALF_LIVES=(100,200)
RAW_LIMIT=640*1024**2
EXPORT_LIMIT=128*1024**2
TRAJECTORY_ALLOWANCE=1051158733
REQUIRED_FREE=13063004980
require=inherited.require

def source_inventory():
    base=inherited.source_inventory()
    # Current numerical code must equal the closed quality campaign exactly.
    freeze=json.loads((ROOT/'research/tokenizer-evidence/block-norm-preparation-001/quality001-freeze.json').read_text())
    ref=freeze['plan'];path=Path(ref['path'])
    require(inherited.file_sha(path)==ref['sha256'],'Prior frozen plan changed')
    require(base==json.loads(path.read_text())['sources'],'Inherited numerical source changed')
    for path in sorted((ROOT/'research/ema_horizon_v1').rglob('*.py')):
        base[str(path.relative_to(ROOT))]=inherited.file_sha(path,root=ROOT)
    return base

def spec(stage,slot,seeds):
    rules.validate_seeds(seeds) # screen is reserved but unused; no screen runs.
    require(type(slot) is int,'Native slot required')
    require(stage in ('pilot','quality'),'Pilot or quality stage required')
    require(slot==0 if stage=='pilot' else 0<=slot<3,'Finite trajectory slot required')
    return dict(stage=stage,slot=slot,seed=seeds['pilot'] if stage=='pilot' else seeds['fresh'][slot],
        learning_rate=.00125,steps=20 if stage=='pilot' else 1200,half_lives=list(HALF_LIVES))

def output_path(output,stage,slot):
    path=Path(output)
    require(path.is_absolute() and '..' not in path.parts and path.is_relative_to(ROOT),'Absolute repository output required')
    parts=path.relative_to(ROOT).parts
    require(len(parts)==5 and parts[0]=='research_runs'
        and re.fullmatch(r'tokenizer-ema-horizon-v1-[a-z0-9][a-z0-9-]*',parts[1]) is not None
        and parts[2]=='evidence' and parts[3]==('pilots' if stage=='pilot' else 'quality')
        and parts[4]==f'{slot:02d}','Dedicated exact EMA slot required')
    for p in [path,*path.parents]:
        require(not p.is_symlink(),'Symlink output forbidden')
        if p==ROOT:break
    return path

def config(output,stage,slot,seeds):
    trial=spec(stage,slot,seeds);path=output_path(output,stage,slot)
    cfg=copy.deepcopy(inherited.baseline())
    cfg['model'].update(tokenizer_architecture_policy=GROUPED_POLICY,
        block_normalization=encode(Spec('rms_incumbent')),embedding_backward=dict(EMBEDDING_SPEC),norm_type='rmsnorm')
    model_contract(cfg['model'])
    pilot=stage=='pilot';start=13 if pilot else 800;t=cfg['training']
    t.update(seed=trial['seed'],eval_seed=seeds['evaluation'],learning_rate=.00125,max_iters=trial['steps'],
        device='mps',strict_device=True,resume_from=None,max_train_seconds=None,
        checkpoint_dir=str(path/'checkpoints'),metrics_file=str(path/'metrics.jsonl'),summary_file=str(path/'training-summary.json'),
        checkpoint_every=0,keep_best_checkpoint=False,eval_every=20 if pilot else 300)
    t['lr_schedule'].update(total_steps=trial['steps'],start_step=start)
    t['weight_averaging'].update(total_steps=trial['steps'],start_step=start,interval=1 if pilot else 10,half_lives=list(HALF_LIVES))
    return cfg

def validate_config(cfg,output,stage,slot,seeds):
    require(inherited.canonical(cfg)==inherited.canonical(config(output,stage,slot,seeds)),
        'Exact averaging-only configuration required')

def artifact_limit(role):
    require(role in ('config','raw','h100','h200','summary','metrics','export100','export200'),'Exact dual-EMA artifact role required')
    return RAW_LIMIT if role=='raw' else EXPORT_LIMIT if role in ('h100','h200') else 2*1024**2

def check_artifact(role,binding):
    require(type(binding) is dict and set(binding)=={'path','sha256'},'Exact artifact binding required')
    require(inherited.file_sha(Path(binding['path']),root=ROOT,max_bytes=artifact_limit(role))==binding['sha256'],
        'Artifact size or hash differs')
