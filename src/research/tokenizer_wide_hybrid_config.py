"""Prospective wider NoPE geometry; no dispatcher, tuning grid or launch plan."""
from ..models.tokenizer_architecture import architecture_contract
from ..models.tokenizer_hybrid import LAYOUTS, WIDE_GEOMETRY, WIDE_POLICY
from .tokenizer_wide_config import config as backbone_config

FAMILIES = ('hybrid', 'gated_mla', 'kda')
LAYOUT = dict(zip(FAMILIES, LAYOUTS))


def config(output, family, width=512, seed=3491, lr=.00125, steps=1200, device='cpu'):
    if family not in FAMILIES:
        raise ValueError('Wider hybrid requires hybrid/gated_mla/kda family')
    cfg = backbone_config(output, 'gqa', width, seed, lr, steps, device)
    cfg['model'].update(attn_type='hybrid', rope=0,
        tokenizer_architecture_policy=WIDE_POLICY,
        attention_width_policy=WIDE_GEOMETRY,
        depth_width_policy='depth4_hybrid_unscaled', attention_layout=list(LAYOUT[family]))
    architecture_contract(cfg['model'])
    return cfg
