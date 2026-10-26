"""Deferred actual model tests: AST review only until the predecessor exits."""
import copy
import math
from types import SimpleNamespace as NS
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from src.models.gpt import GPT
from src.models.value_residual import FIELDS, validate_value_residual_model, mix_values
from src.train.optimizers import build_optimizer
from src.attention.cache import KVCache
from tests.test_value_residual_contract import config
from tests.test_attention_scale_geometry import rotate
from tests.test_gated_activation_geometry import mlp_reference
from tests.test_residual_gain_geometry import gain


def reference(m,tokens,kind,gain_kind,activation,scale,topology,*,capture=None,split_first=False):
    """Independent masked exp/sum attention and attached first-value graph.

    split_first replaces the four roles of V1 with independent leaves solely
    to audit the chain-rule sum. No production mixing/attention helper is used.
    """
    x=m.transformer.wte(tokens)*m.input_embedding_scale
    first=None;roles=[]
    for index,block in enumerate(m.transformer.h):
        original=x;normalized=block.ln1(original);b,t,d=normalized.shape;a=block.attn
        packed=F.linear(normalized,a.qkv.weight,a.qkv.bias)
        q,k,raw=[z.reshape(b,t,d//32,32).transpose(1,2) for z in packed.chunk(3,-1)]
        if index==0:
            first=raw
            if capture is not None:capture['first_raw']=raw
            if split_first:
                first=raw.detach().clone().requires_grad_(True);roles.append(first)
            value=first
        elif kind=='local':value=raw
        elif kind=='local_half':value=.5*raw
        else:
            reuse=first
            if split_first:
                reuse=first.detach().clone().requires_grad_(True);roles.append(reuse)
            value=.5*raw+.5*reuse
        if capture is not None:capture.setdefault('mixed_values',[]).append(value)
        q=rotate(q,block.freq_cos,block.freq_sin);k=rotate(k,block.freq_cos,block.freq_sin)
        scores=torch.einsum('bhid,bhjd->bhij',q,k)*(scale/math.sqrt(32))
        mask=torch.arange(t,device=x.device)[:,None]>=torch.arange(t,device=x.device)[None,:]
        scores=scores.masked_fill(~mask,float('-inf'))
        exp=torch.exp(scores-scores.max(-1,keepdim=True).values)
        probability=exp/exp.sum(-1,keepdim=True)
        attended=torch.einsum('bhij,bhjd->bhid',probability,value).transpose(1,2).reshape(b,t,d)
        attention=F.linear(attended,a.out.weight,a.out.bias)
        u=original+gain(block,attention,'attention',gain_kind)
        f=mlp_reference(block.mlp,block.ln2(original if topology=='parallel_separate_norms' else u),activation)
        x=u+gain(block,f,'ffn',gain_kind)
    if capture is not None:capture['first_roles']=roles
    h=m.transformer.ln_f(x)
    w=m.transformer.wte.weight if m.lm_head is None else m.lm_head.weight
    logits=F.linear(h,w)*m.table_readout_scale/m.width_mult
    return logits if m.output_bias is None else logits+m.output_bias


def caches(m):
    return [KVCache(NS(model=m.cfg),'cpu',batch_size=2) for _ in range(4)]


def chunked(m,tokens,cache,lengths):
    parts=[];pos=0
    for length in lengths:
        parts.append(m(tokens[:,pos:pos+length],cache=cache,start_pos=pos)[0]);pos+=length
    assert pos==tokens.shape[1]
    return torch.cat(parts,1)


@pytest.mark.parametrize('width',[64,256])
@pytest.mark.parametrize('topology',['serial','parallel_separate_norms'])
@pytest.mark.parametrize('table',['legacy_untied','matched_untied','scaled_tied'])
def test_actual_value_residual_graph_rng_groups_and_cache(width,topology,table):
    torch.set_num_threads(1)
    x=torch.arange(16).reshape(2,8)%4;y=(x+1)%4
    for gain_kind,activation,scale in [('unit','silu',1.),('static_tenth','gelu_exact',.5),('learned_tenth','relu',2.)]:
        models=[];rngs=[];groups=[]
        for kind in ('local','local_half','first_blend'):
            torch.manual_seed(9461)
            m=GPT(OmegaConf.create(config(kind,gain_kind,activation,width,scale,topology,table)))
            validate_value_residual_model(m);models.append(m);rngs.append(torch.get_rng_state().clone())
            opt=build_optimizer(m,NS(optimizer='muon',decay_mode='independent',learning_rate=.005,weight_decay=.00006))
            groups.append([(g['parameter_names'],g['lr'],g['width_lr_scale']) for g in opt.param_groups])
            ids=[id(p) for g in opt.param_groups for p in g['params']]
            assert len(ids)==len(set(ids))==len(list(m.parameters()))
            assert not any('value_residual' in n for n,_ in m.named_parameters())
            assert not any('value_residual' in n for n,_ in m.named_buffers())
        assert all(torch.equal(rngs[0],r) for r in rngs)
        assert groups[0]==groups[1]==groups[2]
        for m in models[1:]:
            assert m.state_dict().keys()==models[0].state_dict().keys()
            assert all(torch.equal(v,m.state_dict()[n]) for n,v in models[0].state_dict().items())
        old=copy.deepcopy(models[0].cfg)
        for f in FIELDS:del old[f]
        torch.manual_seed(9461);historical=GPT(old)
        assert torch.equal(torch.get_rng_state(),rngs[0])
        with torch.no_grad():assert torch.equal(historical(x)[0],models[0](x)[0])
        d=width;h=21*d//8
        count=8*d+4*(4*d*d+3*d*h+6*d)+2*d-(4*d if table=='scaled_tied' else 0)+(8*d if gain_kind=='learned_tenth' else 0)
        assert all(sum(p.numel() for p in m.parameters())==count for m in models)
        first_caches=[]
        for kind,m in zip(('local','local_half','first_blend'),models):
            m.double()
            # Nonzero projection biases make the raw-V and complete-output
            # bias paths observable, rather than relying on zero initialization.
            with torch.no_grad():
                for i,block in enumerate(m.transformer.h):
                    block.attn.qkv.bias.copy_(torch.linspace(-.04,.05,3*d,dtype=torch.float64)+i*.01)
                    block.attn.out.bias.copy_(torch.linspace(-.02,.03,d,dtype=torch.float64))
            if gain_kind=='learned_tenth':
                with torch.no_grad():
                    for i,block in enumerate(m.transformer.h):
                        block.attention_residual_gain.copy_(torch.linspace(-.2,.4,d,dtype=torch.float64)+i*.01)
                        block.ffn_residual_gain.copy_(torch.linspace(.3,-.1,d,dtype=torch.float64))
            oracle=copy.deepcopy(m);actual,loss=m(x,y)
            expected=reference(oracle,x,kind,gain_kind,activation,scale,topology)
            assert torch.allclose(actual,expected,atol=1e-11,rtol=1e-9)
            loss.backward();F.cross_entropy(expected.reshape(-1,4),y.reshape(-1)).backward()
            oracle_params=dict(oracle.named_parameters())
            for name,p in m.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(),name
                assert torch.allclose(p.grad,oracle_params[name].grad,atol=1e-9,rtol=1e-7),name
            m.float()
            with torch.no_grad():
                full=m(x)[0];changed=x.clone();changed[:,4:]=(changed[:,4:]+1)%4
                assert torch.allclose(full[:,:4],m(changed)[0][:,:4],atol=1e-6,rtol=1e-5)
                cache=caches(m)
                for lengths in ([1]*8,[3,1,4],[2,3,1,2]):
                    for c in cache:c.reset()
                    assert torch.allclose(chunked(m,x,cache,lengths),full,atol=1e-4,rtol=2e-5)
                    captured={};reference(m,x,kind,gain_kind,activation,scale,topology,capture=captured)
                    for c,v in zip(cache,captured['mixed_values']):
                        assert c.v_cache.shape==(2,d//32,m.cfg.block_size,32)
                        assert torch.allclose(c.v_cache[:,:,:8],v,atol=1e-5,rtol=2e-5)
                first_caches.append((cache[0].k_cache.clone(),cache[0].v_cache.clone()))
                # Interleave unrelated sequences and reset/reuse the first cache.
                other=(x+2)%4;ca,cb=caches(m),caches(m);aa=[];bb=[];pos=0
                for n in (2,1,5):
                    aa.append(m(x[:,pos:pos+n],cache=ca,start_pos=pos)[0])
                    bb.append(m(other[:,pos:pos+n],cache=cb,start_pos=pos)[0]);pos+=n
                assert torch.allclose(torch.cat(aa,1),full,atol=1e-4,rtol=2e-5)
                assert torch.allclose(torch.cat(bb,1),m(other)[0],atol=1e-4,rtol=2e-5)
                for c in ca:c.reset()
                assert torch.allclose(chunked(m,other,ca,[5,3]),m(other)[0],atol=1e-4,rtol=2e-5)
                validate_value_residual_model(m)
        # First-layer inputs/parameters are common, and its cache is never mixed.
        for kc,vc in first_caches[1:]:
            assert torch.equal(kc,first_caches[0][0]) and torch.equal(vc,first_caches[0][1])


def test_actual_first_value_attached_gradient_is_sum_of_four_roles():
    torch.set_num_threads(1);torch.manual_seed(9463)
    m=GPT(OmegaConf.create(config('first_blend','unit','silu',64,.5))).double()
    x=torch.arange(16).reshape(2,8)%4;y=(x+1)%4;packed=[]
    hook=m.transformer.h[0].attn.qkv.register_forward_hook(lambda mod,args,out:packed.append(out))
    _,loss=m(x,y);hook.remove()
    actual=torch.autograd.grad(loss,packed[0])[0].chunk(3,-1)[2].reshape(2,8,2,32).transpose(1,2)
    attached={};expected=reference(copy.deepcopy(m),x,'first_blend','unit','silu',.5,'serial',capture=attached)
    expected_loss=F.cross_entropy(expected.reshape(-1,4),y.reshape(-1))
    attached_grad,=torch.autograd.grad(expected_loss,attached['first_raw'])
    split={};split_logits=reference(copy.deepcopy(m),x,'first_blend','unit','silu',.5,'serial',capture=split,split_first=True)
    assert torch.allclose(expected,split_logits,atol=1e-12,rtol=1e-10)
    grads=torch.autograd.grad(F.cross_entropy(split_logits.reshape(-1,4),y.reshape(-1)),split['first_roles'])
    assert len(grads)==4 and all(torch.count_nonzero(g)>0 for g in grads)
    assert torch.allclose(actual,attached_grad,atol=1e-10,rtol=1e-8)
    assert torch.allclose(attached_grad,sum(grads),atol=1e-10,rtol=1e-8)
    assert not torch.allclose(attached_grad,grads[0],atol=1e-12,rtol=1e-8)
    local=torch.randn(2,2,3,32,dtype=torch.float64,requires_grad=True)
    first=torch.randn_like(local,requires_grad=True);weight=torch.arange(local.numel(),dtype=local.dtype).reshape_as(local)
    mixed=mix_values(local,first,'first_blend',2)
    dl,df=torch.autograd.grad((mixed*weight).sum(),(local,first))
    assert torch.equal(dl,.5*weight) and torch.equal(df,.5*weight)
    assert mix_values(local,None,'local',3) is local
    assert mix_values(local,None,'first_blend',0) is local
