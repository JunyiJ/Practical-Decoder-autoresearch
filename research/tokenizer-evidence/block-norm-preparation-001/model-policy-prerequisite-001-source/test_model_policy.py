"""CPU construction and rejection checks; no training/checkpoint IO."""
import contextlib
import json
from pathlib import Path
import pytest
import torch
from omegaconf import OmegaConf
from research.block_norm_v1.spec import Spec, encode
from research.block_norm_v1.runtime.models.block_norm import POLICY, contract, PLACEMENTS
from research.block_norm_v1.runtime.models.gpt import GPT
from research.block_norm_v1.runtime.models.tokenizer_architecture import validate_architecture_model, validate_architecture_load
from research.block_norm_v1.runtime.train.optimizers import build_optimizer

ROOT = Path(__file__).resolve().parents[2]


def config(kind):
    value = json.loads((ROOT/'research/tokenizer-evidence/mla-001/adopted-width256-mla-recipe.json').read_text())['config']
    cfg = OmegaConf.create(value)
    cfg.model.tokenizer_architecture_policy = POLICY
    cfg.model.block_normalization = encode(Spec(kind))
    cfg.model.norm_type = 'layernorm' if kind == 'layer_matched_epsilon' else 'rmsnorm'
    return cfg


@pytest.fixture(scope='module')
def pair():
    before = torch.get_rng_state().clone()
    try:
        torch.manual_seed(372101); rms = GPT(config('rms_incumbent').model); a = torch.get_rng_state().clone()
        torch.manual_seed(372101); layer = GPT(config('layer_matched_epsilon').model); b = torch.get_rng_state().clone()
        yield rms, layer, a, b
    finally:
        torch.set_rng_state(before)


@contextlib.contextmanager
def replace(obj, name, value):
    old = getattr(obj, name)
    try:
        setattr(obj, name, value)
        yield
    finally:
        setattr(obj, name, old)


def test_pair_shared_initialization_rng_parameters(pair):
    rms, layer, a, b = pair
    assert torch.equal(a, b)
    a, b = rms.state_dict(), layer.state_dict()
    added = {name+'.bias' for name in PLACEMENTS}
    assert set(b)-set(a) == added and not set(a)-set(b)
    assert all(torch.equal(value, b[name]) for name, value in a.items())
    assert all(torch.count_nonzero(b[name]) == 0 for name in added)
    assert [sum(p.numel() for p in m.parameters()) for m in (rms, layer)] == [28492416, 28494464]
    for model in (rms, layer):
        validate_architecture_model(model)


def test_optimizer_parameter_identity_names_and_added_bias_groups(pair):
    rms, layer, _, _ = pair
    opts = [build_optimizer(m, config('rms_incumbent').training) for m in (rms, layer)]
    def groups(opt):
        return {g['parameter_names'][0]:(type(child).__name__,g) for child in opt.optimizers for g in child.param_groups}
    a,b = map(groups, opts)
    assert set(b)-set(a) == {n+'.bias' for n in PLACEMENTS}
    for model, opt in zip((rms,layer),opts):
        named = dict(model.named_parameters()); seen=[]
        for group in opt.param_groups:
            assert len(group['params']) == len(group['parameter_names']) == 1
            name = group['parameter_names'][0]; assert group['params'][0] is named[name]
            seen.append(id(group['params'][0]))
        assert len(seen) == len(set(seen)) == len(named)
        assert opt.decay == .00006
    for name,(kind,g) in b.items():
        if name in a:
            old_kind,old = a[name]
            assert kind == old_kind and {k:v for k,v in g.items() if k!='params'} == {k:v for k,v in old.items() if k!='params'}
        else:
            assert kind=='AdamW' and g['lr']==.00125 and g['width_lr_scale']==1. and g['weight_decay']==0.


def test_both_policies_execute_finite_cpu_fp32(pair):
    tokens=torch.tensor([[0,1,2,3,4]])
    with torch.no_grad():
        for model in pair[:2]:
            logits,loss=model(tokens,targets=tokens)
            assert logits.dtype==torch.float32 and logits.shape==(1,5,50257)
            assert torch.isfinite(logits).all() and torch.isfinite(loss)


@pytest.mark.parametrize('key,value',[
    ('dim',64),('n_layers',6),('n_heads',4),('norm_type','rmsnorm'),
    ('attn_type','mha'),('dropout',.1),('rope',0),('ffn_bias',True),
    ('attention_logit_scale',1.),('hidden_dim',672),('gated_activation_kind','gelu_exact'),
    ('residual_topology_kind','parallel_separate_norms'),('tokenizer_architecture_policy','unknown')])
def test_changed_or_extra_config_rejected(key,value):
    cfg=config('layer_matched_epsilon').model;cfg[key]=value
    from research.block_norm_v1.runtime.models.tokenizer_architecture import architecture_contract
    with pytest.raises(ValueError):architecture_contract(cfg)


@pytest.mark.parametrize('field,value',[
    ('epsilon',1e-5),('dtype','bfloat16'),('width',128),('bias',False),
    ('placement','post_attention'),('affine',False),('kind','unknown')])
def test_same_shape_normalizer_signature_rejected(field,value):
    cfg=config('layer_matched_epsilon').model;cfg.block_normalization[field]=value
    with pytest.raises(ValueError):contract(cfg)


def test_missing_field_rejected():
    cfg=config('layer_matched_epsilon').model;del cfg.block_normalization
    with pytest.raises(ValueError):contract(cfg)


@pytest.mark.parametrize('which', ['epsilon','module','shared','final_epsilon','final_affine','residual','gate','signature','spec'])
def test_actual_module_mutation_rejected(pair,which):
    model=pair[1];block=model.transformer.h[0]
    choices={
        'epsilon':(block.ln1,'eps',1e-5),
        'module':(block,'ln1',torch.nn.RMSNorm(256)),
        'shared':(block,'ln2',block.ln1),
        'final_epsilon':(model.transformer.ln_f,'eps',2.**-23),
        'final_affine':(model.transformer.ln_f,'elementwise_affine',False),
        'residual':(block,'residual_gain_kind','static_tenth'),
        'gate':(block.mlp,'gate_activation','gelu_exact'),
        'signature':(model,'tokenizer_architecture_signature',{}),
        'spec':(model,'block_norm_spec',Spec('rms_incumbent')),
    }
    with replace(*choices[which]):
        with pytest.raises(ValueError):validate_architecture_model(model)
        with pytest.raises(ValueError):build_optimizer(model,config('layer_matched_epsilon').training)
        with pytest.raises(ValueError):model(torch.tensor([[0,1]]))


def test_raw_architecture_metadata_rejects_cross_arm_and_bad_epsilon(pair):
    model=pair[1]
    with pytest.raises(ValueError):validate_architecture_load(model,{'cfg':OmegaConf.to_container(config('rms_incumbent'))})
    cfg=config('layer_matched_epsilon');cfg.model.block_normalization.epsilon=1e-5
    with pytest.raises(ValueError):validate_architecture_load(model,{'cfg':OmegaConf.to_container(cfg)})
    validate_architecture_load(model,{'cfg':OmegaConf.to_container(config('layer_matched_epsilon'))})


def test_autocast_and_non_fp32_rejected(pair):
    model=pair[1]
    with torch.autocast('cpu',dtype=torch.bfloat16):
        with pytest.raises(ValueError,match='Autocast'):model(torch.tensor([[0,1]]))
    with replace(model.transformer.h[0].ln1,'weight',torch.nn.Parameter(torch.ones(256,dtype=torch.float64))):
        with pytest.raises(ValueError):validate_architecture_model(model)
    before=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(ValueError,match='FP32 construction'):GPT(config('layer_matched_epsilon').model)
    finally:torch.set_default_dtype(before)
