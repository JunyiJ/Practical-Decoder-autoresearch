"""Configuration-only tests: no model construction or training."""
import pytest
from omegaconf import OmegaConf
from src.models.depth_scaling import validate_depth6_dense_contract


def cfg(width=64):
    return OmegaConf.create(dict(n_layers=6,dim=width,n_heads=width//32,mlp_type='dense',
        ffn='swiglu',ffn_bias=False,ffn_width_policy='swiglu_21_8',
        depth_width_policy='depth6_dense_unscaled',attn_type='mha',
        attention_width_policy='mha_head32',parameterization='mup',base_width=64,
        norm_type='rmsnorm',rope=1,dropout=0.,attn_dropout=0.,block_size=128))


@pytest.mark.parametrize('width,hidden',[(64,168),(128,336),(256,672)])
def test_explicit_fixed_depth_shape_contract(width,hidden):
    assert validate_depth6_dense_contract(cfg(width))==hidden


@pytest.mark.parametrize('key,value',[
    ('n_layers',4),('n_layers',8),('n_layers',6.0),('dim',96),('dim',True),
    ('n_heads',4),('n_heads',2.0),('mlp_type','moe'),('ffn','gelu'),
    ('ffn_bias',True),('ffn_bias',0),('ffn_width_policy','swiglu_21_4'),
    ('depth_width_policy','depth4_unscaled'),('attn_type','gqa'),
    ('attention_width_policy','mha_head16'),('parameterization','sp'),
    ('base_width',128),('base_width',64.0),('norm_type','layernorm'),('rope',0),
    ('dropout',.1),('attn_dropout',.1),('block_size',512)])
def test_reject_unreviewed_shape_and_scaling(key,value):
    c=cfg();c[key]=value
    with pytest.raises(ValueError):validate_depth6_dense_contract(c)


@pytest.mark.parametrize('key',['hidden_dim','n_kv_heads','n_groups','moe','dimckv',
    'dimcq','head_dim_r','attn_bias','causal','residual_scale','residual_scaling','position_encoding'])
def test_reject_even_null_ignored_override(key):
    c=cfg();c[key]=None
    with pytest.raises(ValueError):validate_depth6_dense_contract(c)
