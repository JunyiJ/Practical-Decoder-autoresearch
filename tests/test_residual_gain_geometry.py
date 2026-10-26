"""DEFERRED actual CPU graphs/gradients/optimizer/cache; never beside live research."""
import copy, math
from types import SimpleNamespace as NS
import pytest, torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from src.models.gpt import GPT
from src.models.residual_gain import FIELDS,PARAMETERS,validate_residual_gain_model,apply_residual_gain
from src.train.optimizers import build_optimizer
from src.attention.cache import KVCache
from tests.test_residual_gain_contract import config
from tests.test_gated_activation_geometry import mlp_reference
from tests.test_attention_scale_geometry import attention

def gain(b,value,branch,kind):
    # Independent test graph, never call production gain dispatcher.
    if kind=='unit':return value
    if kind=='static_tenth':return .1*value
    return torch.einsum('btd,d->btd',value,getattr(b,branch+'_residual_gain'))
def reference(m,tokens,kind,activation,scale,topology):
    x=m.transformer.wte(tokens)*m.input_embedding_scale
    for b in m.transformer.h:
        original=x
        complete_attention=attention(b,b.ln1(x),scale)
        u=x+gain(b,complete_attention,'attention',kind)
        complete_ffn=mlp_reference(b.mlp,b.ln2(original if topology=='parallel_separate_norms' else u),activation)
        x=u+gain(b,complete_ffn,'ffn',kind)
    h=m.transformer.ln_f(x)
    w=m.transformer.wte.weight if m.lm_head is None else m.lm_head.weight
    out=(h@w.T)*m.table_readout_scale/m.width_mult
    return out if m.output_bias is None else out+m.output_bias

def gain_name(name):return name.endswith(tuple('.'+p for p in PARAMETERS))

@pytest.mark.parametrize('width',[64,256])
@pytest.mark.parametrize('topology',['serial','parallel_separate_norms'])
@pytest.mark.parametrize('table',['legacy_untied','matched_untied','scaled_tied'])
def test_actual_residual_gain_full_graph_gradients_init_rng_groups_cache(width,topology,table):
    torch.set_num_threads(1);tokens=torch.arange(16).reshape(2,8)%4;targets=(tokens+1)%4
    for activation,scale in [('silu',1.),('gelu_exact',.5),('relu',2.)]:
        models=[];rngs=[];groups=[]
        for kind in ('unit','static_tenth','learned_tenth'):
            torch.manual_seed(9451);m=GPT(OmegaConf.create(config(kind,activation,width,scale,topology,table)))
            models.append(m);rngs.append(torch.get_rng_state().clone());validate_residual_gain_model(m)
            opt=build_optimizer(m,NS(optimizer='muon',decay_mode='independent',learning_rate=.005,weight_decay=.00006))
            groups.append({g['parameter_names'][0]:(g['lr'],g['width_lr_scale']) for g in opt.param_groups})
            ids=[id(p) for g in opt.param_groups for p in g['params']]
            assert len(ids)==len(set(ids))==len(list(m.parameters()))
            if kind=='learned_tenth':
                gains={n:p for n,p in m.named_parameters() if gain_name(n)}
                assert len(gains)==8 and sum(p.numel() for p in gains.values())==8*width
                for name,p in gains.items():
                    assert p.shape==(width,) and torch.equal(p,torch.full_like(p,.1))
                    assert groups[-1][name]==(.005,1.)
                    assert sum(any(q is p for g in child.param_groups for q in g['params']) for child in opt.optimizers)==1
                    child=next(c for c in opt.optimizers if any(q is p for g in c.param_groups for q in g['params']))
                    assert isinstance(child,torch.optim.AdamW)
                # With zero realized gain gradients and zero initial moments,
                # only the single independent contraction remains.
                before={n:p.detach().clone() for n,p in gains.items()}
                for p in gains.values():p.grad=torch.zeros_like(p)
                opt.step()
                for name,p in gains.items():
                    assert torch.equal(p,before[name]*(1-.00006))
                    with torch.no_grad():p.copy_(before[name])
                m.zero_grad(set_to_none=True)
        assert all(torch.equal(rngs[0],r) for r in rngs)
        common=models[0].state_dict()
        for m in models[1:]:
            assert set(m.state_dict())-set(common)==({f'transformer.h.{i}.{p}' for i in range(4) for p in PARAMETERS} if m is models[2] else set())
            assert all(torch.equal(v,m.state_dict()[k]) for k,v in common.items())
        assert groups[0]==groups[1]=={n:v for n,v in groups[2].items() if not gain_name(n)}
        old=copy.deepcopy(models[0].cfg)
        for f in FIELDS:del old[f]
        torch.manual_seed(9451);historical=GPT(old)
        assert torch.equal(torch.get_rng_state(),rngs[0])
        assert historical.state_dict().keys()==models[0].state_dict().keys()
        with torch.no_grad():
            assert torch.equal(historical(tokens)[0],models[0](tokens)[0])
            assert torch.equal(models[1](tokens)[0],models[2](tokens)[0])
        d=width;h=21*d//8;baseline=8*d+4*(4*d*d+3*d*h+6*d)+2*d-(4*d if table=='scaled_tied' else 0)
        assert [sum(p.numel() for p in m.parameters()) for m in models]==[baseline,baseline,baseline+8*d]
        for kind,m in zip(('unit','static_tenth','learned_tenth'),models):
            m.double()
            # Signed nonuniform values ensure semantics are genuinely per channel
            # and unconstrained after initialization, not a fixed scalar path.
            if kind=='learned_tenth':
                with torch.no_grad():
                    for i,b in enumerate(m.transformer.h):
                        b.attention_residual_gain.copy_(torch.linspace(-.2,.3,width)+i*.01)
                        b.ffn_residual_gain.copy_(torch.linspace(.25,-.15,width)-i*.01)
            oracle=copy.deepcopy(m);logits,loss=m(tokens,targets)
            expected=reference(oracle,tokens,kind,activation,scale,topology)
            assert torch.allclose(logits,expected,atol=1e-11,rtol=1e-9)
            loss.backward();F.cross_entropy(expected.reshape(-1,4),targets.reshape(-1)).backward()
            for name,p in m.named_parameters():
                q=dict(oracle.named_parameters())[name]
                assert p.grad is not None and torch.isfinite(p.grad).all()
                assert q.grad is not None and torch.allclose(p.grad,q.grad,atol=1e-9,rtol=1e-7),name
            m.float()
            with torch.no_grad():
                changed=tokens.clone();changed[:,4:]=(changed[:,4:]+1)%4
                assert torch.allclose(m(tokens)[0][:,:4],m(changed)[0][:,:4],atol=1e-6,rtol=1e-5)
                cache=[KVCache(NS(model=m.cfg),'cpu',batch_size=2) for _ in range(4)]
                parts=[m(tokens[:,:3],cache=cache)[0]]
                for pos in range(3,8):parts.append(m(tokens[:,pos:pos+1],cache=cache,start_pos=pos)[0])
                assert torch.allclose(torch.cat(parts,1),m(tokens)[0],atol=1e-4,rtol=2e-5)

def test_actual_channel_gain_derivatives_include_bias_and_zero_gain():
    torch.set_num_threads(1)
    x=torch.tensor([[[.3,-.7],[.2,.5]]],dtype=torch.float64,requires_grad=True)
    w=torch.tensor([[.2,.8],[-.1,.4]],dtype=torch.float64,requires_grad=True)
    bias=torch.tensor([.9,-.3],dtype=torch.float64,requires_grad=True)
    g=torch.tensor([0.,-.2],dtype=torch.float64,requires_grad=True)
    cot=torch.tensor([[[.4,-.6],[.7,.2]]],dtype=torch.float64)
    complete=x@w.T+bias;b=NS(residual_gain_kind='learned_tenth',attention_residual_gain=g)
    out=apply_residual_gain(b,complete,'attention');(out*cot).sum().backward()
    assert torch.equal(out,complete.detach()*g.detach())
    assert torch.allclose(g.grad,(complete.detach()*cot).sum((0,1)),atol=1e-14,rtol=1e-13)
    inner=cot*g.detach()
    assert torch.allclose(bias.grad,inner.sum((0,1)),atol=1e-14,rtol=1e-13)
    assert torch.allclose(x.grad,inner@w.detach(),atol=1e-14,rtol=1e-13)
    assert torch.allclose(w.grad,torch.einsum('bto,bti->oi',inner,x.detach()),atol=1e-14,rtol=1e-13)
    assert g.grad[0]!=0 and bias.grad[0]==0
