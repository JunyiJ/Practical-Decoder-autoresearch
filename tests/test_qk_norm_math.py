"""Small CPU tensor checks only; no GPT training, corpus evaluation or optimizer steps."""
from types import SimpleNamespace
import pytest
import torch
from src.attention.qk_norm import QK_RMS_EPS,make_qk_norms,resolve_qk_norm,validate_qkrms_contract
from src.models.rope import apply_rotary_emb,precompute_freqs_cis


def cfg(**overrides):
    d=dict(dim=64,n_heads=2,n_layers=4,mlp_type='dense',ffn='swiglu',ffn_bias=False,ffn_width_policy='swiglu_21_8',
        depth_width_policy='depth4_qkrms_unscaled',attn_type='mha',attention_width_policy='mha_head32_qkrms',qk_norm='rms_head32',
        parameterization='mup',base_width=64,norm_type='rmsnorm',rope=1,dropout=0.,attn_dropout=0.,block_size=128)
    return SimpleNamespace(**(d|overrides))


@pytest.mark.parametrize('heads',[2,4,8])
def test_exact_formula_gradients_and_shared_gain_shape(heads):
    torch.set_num_threads(1);torch.manual_seed(10)
    q,k=make_qk_norms(cfg(dim=32*heads,n_heads=heads));q=q.double();k=k.double()
    assert q.weight.shape==k.weight.shape==(32,) and q.weight is not k.weight
    x=torch.randn(2,heads,3,32,dtype=torch.float64,requires_grad=True)
    y=q(x);expected=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+QK_RMS_EPS)*q.weight
    torch.testing.assert_close(y,expected,rtol=1e-12,atol=1e-12)
    weight=torch.linspace(.1,1.,y.numel(),dtype=torch.float64).reshape(y.shape)
    grads=torch.autograd.grad((y*weight).sum(),(x,q.weight),retain_graph=True)
    ref=torch.autograd.grad((expected*weight).sum(),(x,q.weight))
    for a,b in zip(grads,ref):torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-11)
    torch.testing.assert_close(q.weight,torch.ones_like(q.weight));torch.testing.assert_close(k.weight,torch.ones_like(k.weight))


def test_ones_initialization_and_disabled_path_consume_no_rng():
    torch.manual_seed(11);state=torch.get_rng_state().clone()
    q,k=make_qk_norms(cfg());assert torch.equal(state,torch.get_rng_state())
    assert make_qk_norms(cfg(qk_norm='none'))==(None,None)
    assert torch.equal(state,torch.get_rng_state())
    assert resolve_qk_norm(SimpleNamespace())=='none'


@pytest.mark.parametrize('scale',[0.,1e-8,1.,1e4])
def test_finite_output_and_gradient_at_zero_and_bounded_scales(scale):
    q,_=make_qk_norms(cfg());x=(torch.arange(32,dtype=torch.float32)*scale).requires_grad_()
    y=q(x);y.square().sum().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all() and torch.isfinite(q.weight.grad).all()
    if scale==0:assert torch.equal(y,torch.zeros_like(y))


def test_normalization_before_rope_preserves_normalized_vector_norm():
    q,_=make_qk_norms(cfg());x=torch.randn(1,2,5,32)
    with torch.no_grad():q.weight.copy_(torch.linspace(.5,1.5,32))
    cos,sin=precompute_freqs_cis(32,5);before=q(x);rotated=apply_rotary_emb(before,cos,sin)
    torch.testing.assert_close(rotated.square().sum(-1),before.square().sum(-1),rtol=2e-6,atol=2e-6)
    # Nonuniform learned gains generally do not commute with rotation; placement is a contract.
    assert not torch.allclose(rotated,q(apply_rotary_emb(x,cos,sin)),atol=1e-5,rtol=1e-5)


@pytest.mark.parametrize('kind',[None,False,0,'rmsnorm','layernorm','l2'])
def test_unknown_normalization_rejected(kind):
    with pytest.raises(ValueError,match='qk_norm'):resolve_qk_norm(cfg(qk_norm=kind))


@pytest.mark.parametrize('change',[dict(n_heads=4),dict(attn_type='gqa'),dict(attn_type='mla'),dict(dim=65),dict(n_heads=True)])
def test_wrong_head_or_attention_rejected(change):
    with pytest.raises(ValueError,match='head dimension32'):resolve_qk_norm(cfg(**change))


@pytest.mark.parametrize('width',[64,128,256])
def test_explicit_width_contract(width):
    assert validate_qkrms_contract(cfg(dim=width,n_heads=width//32))==256


@pytest.mark.parametrize('change',[
    dict(depth_width_policy='depth4_unscaled'),dict(attention_width_policy='mha_head32'),dict(qk_norm='none'),
    dict(n_layers=6),dict(dim=512,n_heads=16),dict(base_width=128),dict(ffn_bias=True),dict(ffn_width_policy='swiglu_21_4'),
    dict(dropout=.1),dict(attn_dropout=.1),dict(block_size=512),dict(norm_type='layernorm'),dict(rope=0),
    dict(hidden_dim=168),dict(qk_norm_eps=1e-5),dict(qk_norm_placement='after_rope'),dict(qk_scale=1),dict(residual_scale=.5),dict(n_kv_heads=1)])
def test_unreviewed_combined_mutations_rejected(change):
    with pytest.raises(ValueError):validate_qkrms_contract(cfg(**change))
