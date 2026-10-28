"""Explicit wider attention configs; no training dispatcher or selection logic."""
import copy
import json
from pathlib import Path
from ..models.tokenizer_architecture import WIDE_GQA_POLICY, WIDE_MLA_POLICY, architecture_contract

RECIPE = Path(__file__).resolve().parents[2]/'research/tokenizer-evidence/horizon-001/adopted-width256-gqa-recipe.json'


def config(output, family, width=512, seed=3391, lr=.00125, steps=1200, device='mps'):
    if family not in ('gqa','mla') or type(width) is not int or width not in (512,1024):
        raise ValueError('Wider comparison requires explicit GQA/MLA at512/1024')
    if type(steps) is not int or steps < 4:
        raise ValueError('At least four updates required')
    cfg=copy.deepcopy(json.loads(RECIPE.read_text())['config'])
    cfg['model'].update(dim=width,n_heads=width//32,attn_type=family,
        tokenizer_architecture_policy=WIDE_GQA_POLICY if family=='gqa' else WIDE_MLA_POLICY,
        attention_width_policy='gqa_half_head32' if family=='gqa' else 'mla_quarter_head32',
        depth_width_policy='depth4_gqa_unscaled' if family=='gqa' else 'depth4_unscaled')
    t=cfg['training'];t.update(seed=seed,learning_rate=lr,device=device,max_iters=steps,
        checkpoint_dir=str(output/'checkpoints'),metrics_file=str(output/'metrics.jsonl'),
        summary_file=str(output/'training-summary.json'))
    t['lr_schedule'].update(total_steps=steps,start_step=steps*2//3)
    t['weight_averaging'].update(total_steps=steps,start_step=steps*2//3,
        interval=10 if (steps-steps*2//3)%10==0 else 1)
    architecture_contract(cfg['model'])
    return cfg
