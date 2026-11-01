"""Two finite FFN resource pilots only; quality needs its own reviewed freeze."""
import copy
from pathlib import Path
from research.ema_horizon_v1 import contract as inherited
from research.block_norm_v1 import campaign_contract as base, campaign_rules as rules
from research.block_norm_v1.spec import Spec, encode
from .runtime.models.block_norm import FFN_POLICY, contract as model_contract
from .runtime.models.grouped_embedding import EMBEDDING_SPEC
ROOT=base.ROOT
RUN=ROOT/'research_runs/tokenizer-ffn-structure-pilot-001'
EVIDENCE=ROOT/'research/tokenizer-evidence/ffn-structure-preparation-001'
ARMS=('swiglu','ungated_silu')
RAW_LIMIT=600*1024**2
EXPORT_LIMIT=128*1024**2
TRAJECTORY_ALLOWANCE=788953140
# Keep both pilots, two selected screens and all six fresh trajectories.
REQUIRED_FREE=10*TRAJECTORY_ALLOWANCE+256*1024**2+8*1024**3
require=base.require
canonical=base.canonical
fingerprint=base.fingerprint
file_sha=base.file_sha

def source_inventory():
    result=inherited.source_inventory()
    for p in sorted((ROOT/'research/ffn_structure_v1').rglob('*.py')):
        result[str(p.relative_to(ROOT))]=file_sha(p,root=ROOT)
    return result

def verify_sources(expected):require(expected==source_inventory(),'Frozen FFN or inherited source changed')
def output(slot):
    require(type(slot) is int and 0<=slot<2,'Exactly two pilot slots required')
    return RUN/f'{slot:02d}'

def config(slot,seeds):
    path=output(slot);rules.validate_seeds(seeds);cfg=copy.deepcopy(base.baseline())
    cfg['model'].update(tokenizer_architecture_policy=FFN_POLICY,block_normalization=encode(Spec('rms_incumbent')),
        embedding_backward=dict(EMBEDDING_SPEC),norm_type='rmsnorm',
        ffn='swiglu' if slot==0 else 'silu',ffn_width_policy='swiglu_21_8' if slot==0 else 'ungated_silu_63_16')
    model_contract(cfg['model']);t=cfg['training']
    t.update(seed=seeds['pilot'],eval_seed=seeds['evaluation'],learning_rate=.0025,max_iters=20,
        device='mps',strict_device=True,resume_from=None,max_train_seconds=None,
        checkpoint_dir=str(path/'checkpoints'),metrics_file=str(path/'metrics.jsonl'),summary_file=str(path/'summary.json'),
        checkpoint_every=0,keep_best_checkpoint=False,eval_every=20)
    t['lr_schedule'].update(total_steps=20,start_step=13)
    t['weight_averaging'].update(total_steps=20,start_step=13,interval=1,half_lives=[200])
    return cfg

def validate_config(cfg,slot,seeds):require(canonical(cfg)==canonical(config(slot,seeds)),'Exact finite pilot configuration required')
