"""Deferred actual model checks; do not execute beside research workers."""
import copy
import math
from types import SimpleNamespace as NS
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from src.models.gpt import GPT
from src.models.attention_scale import FIELDS,validate_attention_scale_model
from src.train.optimizers import build_optimizer
from src.attention.cache import KVCache
from tests.test_attention_scale_contract import config

def rotate(x,cos,sin):
    # Independent real-valued RoPE, without production rotary helper.
    c=cos[:x.shape[2]][None,None];s=sin[:x.shape[2]][None,None]
    even,odd=x[...,0::2],x[...,1::2]
    return torch.stack((even*c-odd*s,odd*c+even*s),dim=-1).flatten(-2)

def attention(block,x,scale):
    a=block.attn;b,t,d=x.shape
    packed=F.linear(x,a.qkv.weight,a.qkv.bias)
    q,k,v=[z.reshape(b,t,d//32,32).transpose(1,2) for z in packed.chunk(3,-1)]
    q=rotate(q,block.freq_cos,block.freq_sin);k=rotate(k,block.freq_cos,block.freq_sin)
    scores=torch.einsum('bhid,bhjd->bhij',q,k)*(scale/math.sqrt(32))
    mask=torch.arange(t)[:,None]>=torch.arange(t)[None,:]
    scores=scores.masked_fill(~mask,float('-inf'))
    # Independent stable exp/sum instead of production softmax.
    e=torch.exp(scores-scores.max(-1,keepdim=True).values);p=e/e.sum(-1,keepdim=True)
    value=torch.einsum('bhij,bhjd->bhid',p,v).transpose(1,2).reshape(b,t,d)
    return F.linear(value,a.out.weight,a.out.bias)

def reference(m,tokens,scale,topology):
    x=m.transformer.wte(tokens)*m.input_embedding_scale
    for b in m.transformer.h:
        original=x;u=original+attention(b,b.ln1(original),scale)
        x=u+b.mlp(b.ln2(original if topology=='parallel_separate_norms' else u))
    h=m.transformer.ln_f(x)
    w=m.transformer.wte.weight if m.lm_head is None else m.lm_head.weight
    out=F.linear(h,w)*m.table_readout_scale/m.width_mult
    return out if m.output_bias is None else out+m.output_bias

@pytest.mark.parametrize('width',[64,256])
@pytest.mark.parametrize('topology',['serial','parallel_separate_norms'])
@pytest.mark.parametrize('table',['legacy_untied','matched_untied','scaled_tied'])
def test_actual_attention_scale_init_all_gradients_cache_optimizer(width,topology,table):
    torch.set_num_threads(1);models=[];rng=[];groups=[]
    for scale in (1.,.5,2.):
        torch.manual_seed(9421);m=GPT(OmegaConf.create(config(scale,width,topology,table)))
        models.append(m);rng.append(torch.get_rng_state().clone());validate_attention_scale_model(m)
        opt=build_optimizer(m,NS(optimizer='muon',decay_mode='independent',learning_rate=.005,weight_decay=.00006))
        groups.append([(g['parameter_names'],g['lr'],g['width_lr_scale']) for g in opt.param_groups])
        ids=[id(p) for g in opt.param_groups for p in g['params']];assert len(ids)==len(set(ids))==len(list(m.parameters()))
    assert all(torch.equal(rng[0],r) for r in rng)
    for m in models[1:]:
        assert m.state_dict().keys()==models[0].state_dict().keys()
        assert all(torch.equal(v,m.state_dict()[k]) for k,v in models[0].state_dict().items())
    assert groups[0]==groups[1]==groups[2]
    x=torch.arange(16).reshape(2,8)%4;y=(x+1)%4
    old=copy.deepcopy(models[0].cfg)
    for f in FIELDS:del old[f]
    torch.manual_seed(9421);historical=GPT(old)
    assert torch.equal(torch.get_rng_state(),rng[0])
    with torch.no_grad():assert torch.equal(historical(x)[0],models[0](x)[0])
    d=width;h=21*d//8;count=8*d+4*(4*d*d+3*d*h+6*d)+2*d-(4*d if table=='scaled_tied' else 0)
    assert all(sum(p.numel() for p in m.parameters())==count for m in models)
    caches=[]
    for scale,m in zip((1.,.5,2.),models):
        m.double();oracle=copy.deepcopy(m);logits,loss=m(x,y);expected=reference(oracle,x,scale,topology)
        assert torch.allclose(logits,expected,atol=1e-11,rtol=1e-9)
        loss.backward();F.cross_entropy(expected.reshape(-1,4),y.reshape(-1)).backward()
        expected_params=dict(oracle.named_parameters())
        for name,p in m.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()
            assert torch.allclose(p.grad,expected_params[name].grad,atol=1e-9,rtol=1e-7),name
        m.float()
        with torch.no_grad():
            changed=x.clone();changed[:,4:]=(changed[:,4:]+1)%4
            assert torch.allclose(m(x)[0][:,:4],m(changed)[0][:,:4],atol=1e-6,rtol=1e-5)
            cache=[KVCache(NS(model=m.cfg),'cpu',batch_size=2) for _ in range(4)]
            parts=[m(x[:,:3],cache=cache)[0]]
            for pos in range(3,8):parts.append(m(x[:,pos:pos+1],cache=cache,start_pos=pos)[0])
            assert torch.allclose(torch.cat(parts,1),m(x)[0],atol=1e-4,rtol=2e-5)
            # First block sees identical inputs across scalars. Cache stores raw
            # rotated keys and values, never a scale-dependent transformed key.
            caches.append((cache[0].k_cache.clone(),cache[0].v_cache.clone()))
    for kc,vc in caches[1:]:
        assert torch.equal(kc,caches[0][0]) and torch.equal(vc,caches[0][1])
