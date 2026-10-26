"""DEFERRED: actual models/gradients/cache. Never execute beside research worker."""
import copy, math
from types import SimpleNamespace as NS
import pytest, torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from src.models.gpt import GPT
from src.models.gated_activation import FIELDS,validate_gated_activation_model
from src.models.mlp import SwiGLU
from src.train.optimizers import build_optimizer
from src.attention.cache import KVCache
from tests.test_gated_activation_contract import config
from tests.test_attention_scale_geometry import attention

def phi(z,kind):
    if kind=='silu':return z/(1+torch.exp(-z))
    if kind=='gelu_exact':return .5*z*(1+torch.erf(z/math.sqrt(2)))
    return torch.where(z>0,z,torch.zeros_like(z))
def mlp_reference(m,x,kind):
    z=x@m.linear_gate.weight.T;u=x@m.linear_up.weight.T
    return (phi(z,kind)*u)@m.linear_down.weight.T
def reference(m,tokens,kind,scale,topology):
    x=m.transformer.wte(tokens)*m.input_embedding_scale
    for b in m.transformer.h:
        original=x;u=x+attention(b,b.ln1(x),scale)
        x=u+mlp_reference(b.mlp,b.ln2(original if topology=='parallel_separate_norms' else u),kind)
    h=m.transformer.ln_f(x)
    w=m.transformer.wte.weight if m.lm_head is None else m.lm_head.weight
    out=(h@w.T)*m.table_readout_scale/m.width_mult
    return out if m.output_bias is None else out+m.output_bias

@pytest.mark.parametrize('width',[64,256])
@pytest.mark.parametrize('topology',['serial','parallel_separate_norms'])
@pytest.mark.parametrize('table',['legacy_untied','matched_untied','scaled_tied'])
def test_actual_gated_activation_reference_all_gradients_init_rng_groups_cache(width,topology,table):
    torch.set_num_threads(1);tokens=torch.arange(16).reshape(2,8)%4;targets=(tokens+1)%4
    for scale in (1.,.5,2.):
        models=[];rngs=[];groups=[]
        for kind in ('silu','gelu_exact','relu'):
            torch.manual_seed(9441);m=GPT(OmegaConf.create(config(kind,width,scale,topology,table)))
            models.append(m);rngs.append(torch.get_rng_state().clone());validate_gated_activation_model(m)
            opt=build_optimizer(m,NS(optimizer='muon',decay_mode='independent',learning_rate=.005,weight_decay=.00006))
            groups.append([(g['parameter_names'],g['lr'],g['width_lr_scale']) for g in opt.param_groups])
            ids=[id(p) for g in opt.param_groups for p in g['params']]
            assert len(ids)==len(set(ids))==len(list(m.parameters()))
        assert all(torch.equal(rngs[0],r) for r in rngs)
        assert groups[0]==groups[1]==groups[2]
        for m in models[1:]:
            assert m.state_dict().keys()==models[0].state_dict().keys()
            assert all(torch.equal(v,m.state_dict()[k]) for k,v in models[0].state_dict().items())
        old=copy.deepcopy(models[0].cfg)
        for f in FIELDS:del old[f]
        torch.manual_seed(9441);historical=GPT(old)
        assert torch.equal(torch.get_rng_state(),rngs[0])
        with torch.no_grad():assert torch.equal(historical(tokens)[0],models[0](tokens)[0])
        d=width;h=21*d//8;count=8*d+4*(4*d*d+3*d*h+6*d)+2*d-(4*d if table=='scaled_tied' else 0)
        assert all(sum(p.numel() for p in m.parameters())==count for m in models)
        for kind,m in zip(('silu','gelu_exact','relu'),models):
            m.double();oracle=copy.deepcopy(m);logits,loss=m(tokens,targets)
            expected=reference(oracle,tokens,kind,scale,topology)
            assert torch.allclose(logits,expected,atol=1e-11,rtol=1e-9)
            loss.backward();F.cross_entropy(expected.reshape(-1,4),targets.reshape(-1)).backward()
            oracle_params=dict(oracle.named_parameters())
            for name,p in m.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all()
                assert torch.allclose(p.grad,oracle_params[name].grad,atol=1e-9,rtol=1e-7),name
            m.float()
            with torch.no_grad():
                changed=tokens.clone();changed[:,4:]=(changed[:,4:]+1)%4
                assert torch.allclose(m(tokens)[0][:,:4],m(changed)[0][:,:4],atol=1e-6,rtol=1e-5)
                cache=[KVCache(NS(model=m.cfg),'cpu',batch_size=2) for _ in range(4)]
                parts=[m(tokens[:,:3],cache=cache)[0]]
                for pos in range(3,8):parts.append(m(tokens[:,pos:pos+1],cache=cache,start_pos=pos)[0])
                assert torch.allclose(torch.cat(parts,1),m(tokens)[0],atol=1e-4,rtol=2e-5)

def test_actual_relu_zero_derivative_and_gate_formula_gradcheck():
    torch.set_num_threads(1)
    zero=torch.zeros(3,dtype=torch.float64,requires_grad=True)
    F.relu(zero).sum().backward();assert torch.equal(zero.grad,torch.zeros_like(zero))
    for kind in ('silu','gelu_exact','relu'):
        # Away from the ReLU kink; verify complete production module VJP.
        torch.manual_seed(9441);m=SwiGLU(2,3,0.,gate_activation=kind).double()
        x=torch.tensor([[.7,-1.1],[1.3,.2]],dtype=torch.float64,requires_grad=True)
        assert torch.min(torch.abs(m.linear_gate(x)))>.001
        oracle=copy.deepcopy(m);xo=x.detach().clone().requires_grad_()
        out=m(x);ref=mlp_reference(oracle,xo,kind)
        assert torch.allclose(out,ref,atol=1e-12,rtol=1e-10)
        cot=torch.tensor([[.3,-.4],[.7,.8]],dtype=torch.float64)
        (out*cot).sum().backward();(ref*cot).sum().backward()
        assert torch.allclose(x.grad,xo.grad,atol=1e-12,rtol=1e-10)
        for p,q in zip(m.parameters(),oracle.parameters()):assert torch.allclose(p.grad,q.grad,atol=1e-12,rtol=1e-10)
        assert torch.autograd.gradcheck(m,(x.detach().clone().requires_grad_(),),eps=1e-6,atol=1e-5,rtol=1e-3)
