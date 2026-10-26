"""Actual model gates; execute only after the active serial research round exits."""
import copy
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from src.models.gpt import GPT
from src.attention.cache import KVCache
from src.models.embedding_scale import POLICY
from tests.test_embedding_scale_contract import cfg


@pytest.mark.parametrize('width',[64,256])
@pytest.mark.parametrize('scale',[1.,.5,.25])
def test_exact_locus_common_init_rng_gradients_cache_and_causality(width,scale):
    torch.set_num_threads(1)
    config=cfg(width,scale);config['vocab_size']=17
    legacy=copy.deepcopy(config);del legacy['input_width_policy'];del legacy['input_embedding_scale']
    torch.manual_seed(9381);base=GPT(OmegaConf.create(legacy));base_rng=torch.get_rng_state().clone()
    torch.manual_seed(9381);model=GPT(OmegaConf.create(config));rng=torch.get_rng_state().clone()
    assert torch.equal(base_rng,rng)
    assert base.state_dict().keys()==model.state_dict().keys()
    for name,tensor in base.state_dict().items():assert torch.equal(tensor,model.state_dict()[name])
    assert sum(x.numel() for x in model.parameters())==sum(x.numel() for x in base.parameters())
    x=torch.arange(16).reshape(2,8)%17;y=(x+1)%17
    # Independent implementation at the embedding output proves exact placement.
    handle=base.transformer.wte.register_forward_hook(lambda module,args,out:out*scale)
    expected,loss=base(x,y);loss.backward();handle.remove()
    observed,actual=model(x,y);actual.backward()
    assert torch.equal(expected,observed) and torch.equal(loss,actual)
    for (name,a),(other,b) in zip(base.named_parameters(),model.named_parameters()):
        assert name==other and a.grad is not None and b.grad is not None
        assert torch.isfinite(b.grad).all() and torch.equal(a.grad,b.grad)
    assert torch.count_nonzero(model.transformer.wte.weight.grad)>0
    assert torch.count_nonzero(model.lm_head.weight.grad)>0
    if scale==1.:
        with torch.no_grad():assert torch.equal(base(x)[0],model(x)[0])
    with torch.no_grad():
        changed=x.clone();changed[:,4:]=(changed[:,4:]+1)%17
        assert torch.allclose(model(x)[0][:,:4],model(changed)[0][:,:4],atol=1e-6,rtol=1e-5)
        cache=[KVCache(SimpleNamespace(model=model.cfg),'cpu',batch_size=2) for _ in range(4)]
        parts=[model(x[:,:4],cache=cache)[0]]
        for pos in range(4,8):parts.append(model(x[:,pos:pos+1],cache=cache,start_pos=pos)[0])
        assert torch.allclose(torch.cat(parts,dim=1),model(x)[0],atol=1e-4,rtol=2e-5)
