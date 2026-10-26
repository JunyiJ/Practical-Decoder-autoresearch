"""Pure configuration guards: no torch import, model execution or quality claim."""
import copy
from types import SimpleNamespace

import pytest
from src.models.embedding_scale import (POLICY, input_scale_contract,
    resolve_input_scale, validate_input_scale_load)


def cfg(width=64, scale=.5):
    return dict(n_layers=4, dim=width, n_heads=width//32, mlp_type='dense',
        ffn='swiglu', ffn_bias=False, ffn_width_policy='swiglu_21_8',
        depth_width_policy='depth4_unscaled', attn_type='mha',
        attention_width_policy='mha_head32', parameterization='mup', base_width=64,
        norm_type='rmsnorm', rope=1, dropout=0., attn_dropout=0., block_size=128,
        input_width_policy=POLICY, input_embedding_scale=scale)


@pytest.mark.parametrize('width',[64,128,256])
@pytest.mark.parametrize('scale',[1.,.5,.25])
def test_named_contract_and_saved_signature(width,scale):
    d=cfg(width,scale)
    assert input_scale_contract(d)==input_scale_contract(SimpleNamespace(**d))
    assert resolve_input_scale(d)==scale
    m=SimpleNamespace(cfg=d,input_embedding_scale=scale)
    validate_input_scale_load(m,dict(cfg=dict(model=copy.deepcopy(d))))


def test_legacy_metadata_is_untouched():
    assert input_scale_contract({}) is None and resolve_input_scale({})==1.
    validate_input_scale_load(SimpleNamespace(cfg={}),{})


@pytest.mark.parametrize('field,value',[
    ('input_embedding_scale',True),('input_embedding_scale',float('nan')),
    ('input_embedding_scale',float('inf')),('input_embedding_scale',0.),
    ('input_embedding_scale',-.5),('input_embedding_scale',2.),
    ('input_embedding_scale','0.5'),('input_width_policy','other'),
    ('n_layers',2),('n_layers',4.),('dim',96),('n_heads',4),('n_heads',True),
    ('base_width',128),('ffn_bias',0),('attn_type','kda'),('rope',0),
    ('block_size',512),('dropout',.1),('norm_type','layernorm'),
    ('residual_scale',1.),('tie_embeddings',False),('qk_norm','none'),
    ('attention_output_gate','none'),('ffn_hidden_norm','none'),
    ('hidden_dim',168),('position_encoding','rope')])
def test_reject_unreviewed_combination(field,value):
    d=cfg();d[field]=value
    with pytest.raises(ValueError):input_scale_contract(d)


@pytest.mark.parametrize('field',['input_embedding_scale','input_width_policy'])
def test_incomplete_opt_in_rejected(field):
    d=cfg();del d[field]
    with pytest.raises(ValueError):input_scale_contract(d)


@pytest.mark.parametrize('saved,current',[(.5,.25),(.25,.5),(1.,.5),(.5,1.)])
def test_same_shape_different_semantics_rejected(saved,current):
    with pytest.raises(ValueError,match='contract differs'):
        validate_input_scale_load(SimpleNamespace(cfg=cfg(scale=current),input_embedding_scale=current),
            dict(cfg=dict(model=cfg(scale=saved))))


@pytest.mark.parametrize('direction',['new_into_legacy','legacy_into_new'])
def test_explicit_policy_not_silently_dropped(direction):
    old=cfg(scale=1.);del old['input_width_policy'];del old['input_embedding_scale']
    new=cfg(scale=1.);saved,current=(new,old) if direction=='new_into_legacy' else (old,new)
    with pytest.raises(ValueError,match='contract differs'):
        validate_input_scale_load(SimpleNamespace(cfg=current,input_embedding_scale=1.),dict(cfg=dict(model=saved)))


def test_model_config_mutation_cannot_reinterpret_constructed_scalar():
    d=cfg(scale=.5)
    with pytest.raises(ValueError,match='Constructed'):
        validate_input_scale_load(SimpleNamespace(cfg=d,input_embedding_scale=.25),dict(cfg=dict(model=d)))


def test_deleting_policy_cannot_disguise_constructed_scaled_model_as_legacy():
    d=cfg();del d['input_width_policy'];del d['input_embedding_scale']
    with pytest.raises(ValueError,match='Constructed'):
        validate_input_scale_load(SimpleNamespace(cfg=d,input_embedding_scale=.5),dict(cfg=dict(model=d)))
