"""Actual model formula/gradient/init/cache gates; defer while research is live."""
import copy
from types import SimpleNamespace
import pytest
import torch
from omegaconf import OmegaConf
from src.models.gpt import GPT
from src.attention.cache import KVCache
from tests.test_readout_norm_contract import cfg

class ManualNorm(torch.nn.Module):
    def __init__(self, original, kind):
        super().__init__()
        self.weight=torch.nn.Parameter(original.weight.detach().clone())
        self.bias=torch.nn.Parameter(original.bias.detach().clone()) if kind=='layer_affine' else None
        self.kind=kind
    def forward(self, x):
        centered=x if self.kind=='rms' else x-x.mean(dim=-1,keepdim=True)
        normalized=centered*torch.rsqrt(centered.square().mean(dim=-1,keepdim=True)+1e-5)
        out=normalized*self.weight
        return out if self.bias is None else out+self.bias

@pytest.mark.parametrize('width',[64,256])
@pytest.mark.parametrize('kind',['layer_affine','layer_no_bias','rms'])
def test_independent_formula_gradients_init_rng_cache_and_causality(width,kind):
    torch.set_num_threads(1)
    config=cfg(kind,width,.5);config['vocab_size']=17
    legacy=copy.deepcopy(config)
    for key in ('readout_norm_kind','readout_norm_width_policy','readout_norm_epsilon'):del legacy[key]
    torch.manual_seed(9387);base=GPT(OmegaConf.create(legacy));base_rng=torch.get_rng_state().clone()
    torch.manual_seed(9387);model=GPT(OmegaConf.create(config));rng=torch.get_rng_state().clone()
    assert torch.equal(base_rng,rng)
    missing=set(base.state_dict())-set(model.state_dict())
    assert missing==({'transformer.ln_f.bias'} if kind!='layer_affine' else set())
    assert set(model.state_dict())<=set(base.state_dict())
    for name,tensor in model.state_dict().items():assert torch.equal(tensor,base.state_dict()[name])
    assert sum(p.numel() for p in base.parameters())-sum(p.numel() for p in model.parameters())==(width if kind!='layer_affine' else 0)
    # Independent double-precision formula, including nontrivial gain/bias,
    # checks all gradients rather than merely comparing layer output shapes.
    base=base.double();model=model.double()
    with torch.no_grad():
        model.transformer.ln_f.weight.copy_(torch.linspace(.7,1.3,width,dtype=torch.float64))
        if kind=='layer_affine':model.transformer.ln_f.bias.copy_(torch.linspace(-.1,.1,width,dtype=torch.float64))
    base.transformer.ln_f=ManualNorm(model.transformer.ln_f,kind)
    x=torch.arange(16).reshape(2,8)%17;y=(x+1)%17
    expected,loss=base(x,y);loss.backward()
    observed,actual=model(x,y);actual.backward()
    assert torch.allclose(expected,observed,atol=1e-10,rtol=1e-9)
    assert torch.allclose(loss,actual,atol=1e-10,rtol=1e-9)
    for (name,a),(other,b) in zip(base.named_parameters(),model.named_parameters()):
        assert name==other and a.grad is not None and b.grad is not None
        assert torch.isfinite(b.grad).all() and torch.allclose(a.grad,b.grad,atol=1e-10,rtol=1e-8)
    assert torch.count_nonzero(model.transformer.wte.weight.grad)>0
    assert torch.count_nonzero(model.lm_head.weight.grad)>0
    model=model.float()
    with torch.no_grad():
        changed=x.clone();changed[:,4:]=(changed[:,4:]+1)%17
        assert torch.allclose(model(x)[0][:,:4],model(changed)[0][:,:4],atol=1e-6,rtol=1e-5)
        cache=[KVCache(SimpleNamespace(model=model.cfg),'cpu',batch_size=2) for _ in range(4)]
        parts=[model(x[:,:4],cache=cache)[0]]
        for pos in range(4,8):parts.append(model(x[:,pos:pos+1],cache=cache,start_pos=pos)[0])
        assert torch.allclose(torch.cat(parts,dim=1),model(x)[0],atol=1e-4,rtol=2e-5)
