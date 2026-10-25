"""Model checks prepared for execution after the active research round exits."""
import copy
import math
from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F

from src.models.gpt import GPT
from src.models.mlp import SwiGLU
from src.attention.cache import KVCache
from src.train.optimizers import build_optimizer
from tests.test_output_gate_integration import config as gate_config


def config(width=64, kind='rms', gate='none'):
    cfg=gate_config(width,False,gate)
    if kind!='none':
        cfg.ffn_hidden_norm=kind
        cfg.ffn_hidden_norm_policy='swiglu_product_norm_21_8'
    return cfg


@pytest.mark.parametrize('width',[64,128,256])
@pytest.mark.parametrize('gate',['none','static_half','sigmoid_headwise_zero'])
def test_common_initialization_counts_and_all_optimizer_roles(width,gate):
    torch.set_num_threads(1);original=None;hidden=21*width//8
    for kind in ('none','rms','layer'):
        torch.manual_seed(9331);model=GPT(config(width,kind,gate));named=dict(model.named_parameters())
        if kind=='none':
            original={n:p.detach().clone() for n,p in named.items()};rng=torch.get_rng_state().clone()
        else:
            assert torch.equal(rng,torch.get_rng_state())
        for name,value in original.items():assert torch.equal(value,named[name])
        norms={n for n in named if '.mlp.hidden_norm.' in n}
        assert len(norms)==(0 if kind=='none' else 4 if kind=='rms' else 8)
        assert sum(p.numel() for n,p in named.items() if n in norms)==(0 if kind=='none' else 4*hidden*(2 if kind=='layer' else 1))
        assert set(named)-set(original)==norms
        for n in norms:
            assert named[n].shape==(hidden,)
            assert torch.equal(named[n],torch.zeros(hidden) if n.endswith('.bias') else torch.ones(hidden))
        opt=build_optimizer(model,SimpleNamespace(optimizer='muon',decay_mode='independent',learning_rate=.005,weight_decay=.00006))
        seen=set()
        for child in opt.optimizers:
            assert not child.state
            for group in child.param_groups:
                name,=group['parameter_names'];assert name not in seen;seen.add(name)
                matrix=name.startswith('transformer.h.') and named[name].ndim==2
                assert isinstance(child,torch.optim.Muon if matrix else torch.optim.AdamW)
                assert group['width_lr_scale']==(1/math.sqrt(width/64) if matrix else 1.)
                assert group['lr']==.005*group['width_lr_scale'] and group['weight_decay']==0
                if name in norms:assert not matrix and group['width_lr_scale']==1.
        assert seen==set(named) and opt.decay==.00006


@pytest.mark.parametrize('width',[64,256])
@pytest.mark.parametrize('kind',['rms','layer'])
def test_independent_complete_swiglu_output_and_all_gradients(width,kind):
    torch.set_num_threads(1);torch.manual_seed(9332);hidden=21*width//8
    model=SwiGLU(width,hidden,0.,hidden_norm=kind).double()
    with torch.no_grad():
        model.hidden_norm.weight.copy_(torch.linspace(.5,1.5,hidden))
        if kind=='layer':model.hidden_norm.bias.copy_(torch.linspace(-.1,.1,hidden))
    x=torch.randn(2,3,width,dtype=torch.float64,requires_grad=True)
    actual=model(x)
    pre=x@model.linear_gate.weight.T
    product=(pre/(1+torch.exp(-pre)))*(x@model.linear_up.weight.T)
    centered=product-product.mean(-1,keepdim=True) if kind=='layer' else product
    normalized=centered/(centered.square().mean(-1,keepdim=True)+1e-5).sqrt()
    normalized=normalized*model.hidden_norm.weight
    if kind=='layer':normalized=normalized+model.hidden_norm.bias
    expected=normalized@model.linear_down.weight.T
    torch.testing.assert_close(actual,expected,atol=1e-11,rtol=1e-11)
    coefficient=torch.linspace(-.2,.3,actual.numel(),dtype=torch.float64).reshape_as(actual)
    params=(x,*model.parameters())
    ga=torch.autograd.grad((actual*coefficient).sum(),params,retain_graph=True)
    gb=torch.autograd.grad((expected*coefficient).sum(),params)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=2e-10,rtol=2e-10)


@pytest.mark.parametrize('width',[64,256])
def test_default_preserves_legacy_keys_rng_and_arithmetic(width):
    torch.set_num_threads(1);hidden=21*width//8;torch.manual_seed(9333)
    model=SwiGLU(width,hidden,0.);rng=torch.get_rng_state().clone()
    torch.manual_seed(9333)
    gate=torch.nn.Linear(width,hidden,bias=False);up=torch.nn.Linear(width,hidden,bias=False);down=torch.nn.Linear(hidden,width,bias=False)
    assert torch.equal(rng,torch.get_rng_state())
    assert set(model.state_dict())=={'linear_gate.weight','linear_up.weight','linear_down.weight'}
    for a,b in [(model.linear_gate,gate),(model.linear_up,up),(model.linear_down,down)]:assert torch.equal(a.weight,b.weight)
    x=torch.randn(2,5,width)
    assert torch.equal(model(x),down(F.silu(gate(x))*up(x)))


@pytest.mark.parametrize('width',[64,256])
@pytest.mark.parametrize('kind',['rms','layer'])
@pytest.mark.parametrize('gate',['none','static_half','sigmoid_headwise_zero'])
def test_causality_cached_decoding_and_strict_reload(tmp_path,width,kind,gate):
    torch.set_num_threads(1);torch.manual_seed(9334);cfg=config(width,kind,gate);model=GPT(cfg).eval()
    with torch.no_grad():
        for block in model.transformer.h:
            block.mlp.hidden_norm.weight.copy_(torch.linspace(.8,1.2,21*width//8))
            if kind=='layer':block.mlp.hidden_norm.bias.copy_(torch.linspace(-.1,.1,21*width//8))
            if gate=='sigmoid_headwise_zero':
                w=block.attn.output_gate.weight;w.copy_(torch.linspace(-.1,.1,w.numel()).reshape_as(w))
        idx=torch.arange(9).reshape(1,9);full=model(idx)[0]
        cache=[KVCache(SimpleNamespace(model=cfg),'cpu',batch_size=1) for _ in range(4)]
        cached=torch.cat([model(idx[:,:3],cache=cache,start_pos=0)[0],model(idx[:,3:7],cache=cache,start_pos=3)[0],model(idx[:,7:],cache=cache,start_pos=7)[0]],1)
        torch.testing.assert_close(cached,full,atol=1e-6,rtol=1e-5)
        changed=idx.clone();changed[:,5:]+=11
        assert torch.equal(model(changed)[0][:,:5],full[:,:5])
        path=tmp_path/'state.pt';torch.save(model.state_dict(),path)
        restored=GPT(copy.deepcopy(cfg)).eval();restored.load_state_dict(torch.load(path,weights_only=True),strict=True)
        assert torch.equal(restored(idx)[0],full)
        for other in ('none','layer' if kind=='rms' else 'rms'):
            with pytest.raises(RuntimeError):GPT(config(width,other,gate)).load_state_dict(model.state_dict(),strict=True)
