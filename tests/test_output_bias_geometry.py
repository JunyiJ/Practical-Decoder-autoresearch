"""Deferred actual model tests; no invocation while current research is live."""
import copy
from types import SimpleNamespace
import pytest
import torch
from omegaconf import OmegaConf
from src.models.gpt import GPT
from src.attention.cache import KVCache
from src.models.output_bias import FIELDS
from tests.test_output_bias_contract import config

@pytest.mark.parametrize('width',[64,256])
@pytest.mark.parametrize('kind',['none','learned_zero','learned_train_unigram'])
@pytest.mark.parametrize('norm',['layer_affine','layer_no_bias','rms'])
def test_output_bias_independent_forward_all_gradients_init_rng_cache(width,kind,norm):
    torch.set_num_threads(1)
    cfg=config(kind,width,norm,scale=.5)
    legacy=copy.deepcopy(cfg)
    for k in FIELDS:del legacy[k]
    torch.manual_seed(9391);base=GPT(OmegaConf.create(legacy));rng=torch.get_rng_state().clone()
    torch.manual_seed(9391);model=GPT(OmegaConf.create(cfg))
    assert torch.equal(torch.get_rng_state(),rng)
    old=base.state_dict();new=model.state_dict()
    assert set(new)-set(old)==({'output_bias'} if kind!='none' else set())
    for k,v in old.items():assert torch.equal(v,new[k])
    assert sum(x.numel() for x in model.parameters())-sum(x.numel() for x in base.parameters())==(4 if kind!='none' else 0)
    expected_init=torch.tensor(cfg['output_bias_prior']['initial_values'] if kind=='learned_train_unigram' else [0.]*4,dtype=torch.float32)
    if kind!='none':assert torch.equal(model.output_bias,expected_init)
    base=base.double();model=model.double()
    manual_bias=expected_init.double().requires_grad_()
    x=torch.arange(16).reshape(2,8)%4;y=(x+1)%4
    raw=base(x)[0];want=raw if kind=='none' else raw+manual_bias
    loss=torch.nn.functional.cross_entropy(want.reshape(-1,4),y.reshape(-1));loss.backward()
    got,actual=model(x,y);actual.backward()
    assert torch.allclose(got,want,atol=1e-11,rtol=1e-10)
    assert torch.allclose(actual,loss,atol=1e-11,rtol=1e-10)
    named=dict(model.named_parameters())
    for name,p in base.named_parameters():
        assert p.grad is not None and named[name].grad is not None
        assert torch.allclose(p.grad,named[name].grad,atol=1e-10,rtol=1e-8)
    if kind!='none':
        assert torch.allclose(model.output_bias.grad,manual_bias.grad,atol=1e-12,rtol=1e-10)
        independent=(torch.softmax(got.detach(),-1)-torch.nn.functional.one_hot(y,4)).mean((0,1))
        assert torch.allclose(model.output_bias.grad,independent,atol=1e-12,rtol=1e-10)
    model=model.float()
    with torch.no_grad():
        changed=x.clone();changed[:,4:]=(changed[:,4:]+1)%4
        assert torch.allclose(model(x)[0][:,:4],model(changed)[0][:,:4],atol=1e-6,rtol=1e-5)
        cache=[KVCache(SimpleNamespace(model=model.cfg),'cpu',batch_size=2) for _ in range(4)]
        pieces=[model(x[:,:4],cache=cache)[0]]
        for pos in range(4,8):pieces.append(model(x[:,pos:pos+1],cache=cache,start_pos=pos)[0])
        assert torch.allclose(torch.cat(pieces,1),model(x)[0],atol=1e-4,rtol=2e-5)
