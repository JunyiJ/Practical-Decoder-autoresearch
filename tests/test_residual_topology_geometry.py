"""Deferred actual CPU gates. Never execute beside an active research worker."""
import copy
from types import SimpleNamespace as NS
import pytest
import torch
from omegaconf import OmegaConf
from src.models.gpt import GPT
from src.models.residual_topology import FIELDS, validate_residual_topology_model
from src.train.optimizers import build_optimizer
from src.attention.cache import KVCache
from tests.test_residual_topology_contract import config


def reference(model, tokens, parallel):
    # Independent graph construction; deliberately do not invoke block.forward.
    x = model.transformer.wte(tokens) * model.input_embedding_scale
    for block in model.transformer.h:
        original = x
        attn, _ = block.attn(block.ln1(original), freq_cos=block.freq_cos, freq_sin=block.freq_sin)
        after_attention = original + attn
        branch_input = original if parallel else after_attention
        x = after_attention + block.mlp(block.ln2(branch_input))
    x = model.transformer.ln_f(x)
    head = model.transformer.wte.weight if model.lm_head is None else model.lm_head.weight
    logits = torch.nn.functional.linear(x, head) * model.table_readout_scale / model.width_mult
    if model.output_bias is not None: logits = logits + model.output_bias
    return logits


@pytest.mark.parametrize('width', [64,256])
@pytest.mark.parametrize('table', ['legacy_untied','matched_untied','scaled_tied'])
def test_actual_residual_topology_init_formula_gradients_cache_groups(width,table):
    torch.set_num_threads(1)
    models=[];rng=[]
    for kind in ('serial','parallel_separate_norms'):
        torch.manual_seed(9411)
        models.append(GPT(OmegaConf.create(config(kind,width,table))))
        rng.append(torch.get_rng_state().clone())
    serial,parallel=models
    assert torch.equal(rng[0],rng[1])
    assert serial.state_dict().keys()==parallel.state_dict().keys()
    for name,value in serial.state_dict().items():assert torch.equal(value,parallel.state_dict()[name])
    historical_cfg=copy.deepcopy(serial.cfg)
    for key in FIELDS:del historical_cfg[key]
    torch.manual_seed(9411);historical=GPT(historical_cfg)
    assert torch.equal(torch.get_rng_state(),rng[0])
    x=torch.arange(16).reshape(2,8)%4;y=(x+1)%4
    with torch.no_grad():assert torch.equal(serial(x)[0],historical(x)[0])
    d=width;h=21*d//8;V=4
    count=2*V*d+4*(4*d*d+3*d*h+6*d)+2*d-(V*d if table=='scaled_tied' else 0)
    assert all(sum(p.numel() for p in m.parameters())==count for m in models)
    groups=[]
    for m in models:
        validate_residual_topology_model(m)
        opt=build_optimizer(m,NS(optimizer='muon',decay_mode='independent',learning_rate=.005,weight_decay=.00006))
        groups.append([(g['parameter_names'],g['width_lr_scale'],g['lr']) for g in opt.param_groups])
        ids=[id(p) for g in opt.param_groups for p in g['params']]
        assert len(ids)==len(set(ids))==len(list(m.parameters()))
        assert all(b.ln1 is not b.ln2 for b in m.transformer.h)
    assert groups[0]==groups[1]
    for index,m in enumerate(models):
        m.double();oracle=copy.deepcopy(m)
        logits,loss=m(x,y);expected=reference(oracle,x,index==1)
        assert torch.allclose(logits,expected,atol=1e-12,rtol=1e-10)
        expected_loss=torch.nn.functional.cross_entropy(expected.reshape(-1,4),y.reshape(-1))
        loss.backward();expected_loss.backward()
        grads=dict(oracle.named_parameters())
        for name,p in m.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()
            assert torch.allclose(p.grad,grads[name].grad,atol=1e-10,rtol=1e-8)
        m.float()
        with torch.no_grad():
            changed=x.clone();changed[:,4:]=(changed[:,4:]+1)%4
            assert torch.allclose(m(x)[0][:,:4],m(changed)[0][:,:4],atol=1e-6,rtol=1e-5)
            cache=[KVCache(NS(model=m.cfg),'cpu',batch_size=2) for _ in range(4)]
            parts=[m(x[:,:3],cache=cache)[0]]
            for pos in range(3,8):parts.append(m(x[:,pos:pos+1],cache=cache,start_pos=pos)[0])
            assert torch.allclose(torch.cat(parts,1),m(x)[0],atol=1e-4,rtol=2e-5)
    # This graph intervention intentionally changes initial functions.
    with torch.no_grad():assert not torch.equal(serial(x)[0],parallel(x)[0])
