"""Deferred actual GPT/gradient/optimizer/cache gates; do not run beside research."""
import copy
import math
from types import SimpleNamespace
import pytest
import torch
from omegaconf import OmegaConf
from src.models.gpt import GPT
from src.models.weight_tying import KINDS,FIELDS,validate_weight_tying_model
from src.train.optimizers import build_optimizer
from src.attention.cache import KVCache
from tests.test_weight_tying_contract import config

@pytest.mark.parametrize('width',[64,256])
@pytest.mark.parametrize('norm',['layer_affine','layer_no_bias','rms'])
@pytest.mark.parametrize('bias',['none','learned_zero','learned_train_unigram'])
def test_actual_weight_tying_geometry_roles_optimizer_cache(width,norm,bias):
    torch.set_num_threads(1)
    models=[];rngs=[]
    for kind in KINDS:
        torch.manual_seed(9401)
        models.append(GPT(OmegaConf.create(config(kind,width,norm,bias))))
        rngs.append(torch.get_rng_state().clone())
    legacy,matched,tied=models
    assert all(torch.equal(rngs[0],r) for r in rngs)
    default=copy.deepcopy(legacy.cfg)
    for key in FIELDS:del default[key]
    torch.manual_seed(9401);historical=GPT(default)
    assert torch.equal(torch.get_rng_state(),rngs[0])
    for name,value in historical.state_dict().items():assert torch.equal(value,legacy.state_dict()[name])
    assert tied.lm_head is None and 'lm_head.weight' not in tied.state_dict()
    assert matched.lm_head.weight is not matched.transformer.wte.weight
    assert matched.lm_head.weight.data_ptr()!=matched.transformer.wte.weight.data_ptr()
    assert torch.equal(matched.lm_head.weight,matched.transformer.wte.weight)
    for name,value in tied.state_dict().items():
        assert torch.equal(value,matched.state_dict()[name])
        assert torch.equal(value,legacy.state_dict()[name])
    V=4;d=width;h=21*d//8
    base_count=2*V*d+4*(4*d*d+3*d*h+6*d)+(2*d if norm=='layer_affine' else d)+(V if bias!='none' else 0)
    assert sum(p.numel() for p in legacy.parameters())==base_count
    assert sum(p.numel() for p in matched.parameters())==base_count
    assert sum(p.numel() for p in tied.parameters())==base_count-V*d
    x=torch.arange(16).reshape(2,8)%V;y=(x+1)%V
    with torch.no_grad():assert torch.equal(legacy(x)[0],historical(x)[0])
    matched.double();tied.double()
    a,la=matched(x,y);b,lb=tied(x,y)
    assert torch.equal(a,b) and torch.equal(la,lb)
    # Independent final-head formula from the actual normalized hidden values.
    captured=[]
    hook=matched.transformer.ln_f.register_forward_hook(lambda m,i,o:captured.append(o))
    output=matched(x)[0];hook.remove()
    independent=torch.einsum('btd,vd->btv',captured[0],matched.lm_head.weight)/math.sqrt(192)/(width/64)
    if matched.output_bias is not None:independent=independent+matched.output_bias
    assert torch.allclose(output,independent,atol=1e-12,rtol=1e-10)
    la.backward();lb.backward()
    expected=matched.transformer.wte.weight.grad+matched.lm_head.weight.grad
    assert torch.allclose(tied.transformer.wte.weight.grad,expected,atol=1e-10,rtol=1e-8)
    untied=dict(matched.named_parameters())
    for name,p in tied.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
        if name!='transformer.wte.weight':assert torch.allclose(p.grad,untied[name].grad,atol=1e-10,rtol=1e-8)
    for model in models:
        model.float();validate_weight_tying_model(model)
        with torch.no_grad():
            changed=x.clone();changed[:,4:]=(changed[:,4:]+1)%V
            assert torch.allclose(model(x)[0][:,:4],model(changed)[0][:,:4],atol=1e-6,rtol=1e-5)
            cache=[KVCache(SimpleNamespace(model=model.cfg),'cpu',batch_size=2) for _ in range(4)]
            pieces=[model(x[:,:4],cache=cache)[0]]
            for pos in range(4,8):pieces.append(model(x[:,pos:pos+1],cache=cache,start_pos=pos)[0])
            assert torch.allclose(torch.cat(pieces,1),model(x)[0],atol=1e-4,rtol=2e-5)
    tied.zero_grad();tied(x,y)[1].backward()
    training=SimpleNamespace(optimizer='muon',decay_mode='independent',learning_rate=.005,weight_decay=.00006)
    optimizer=build_optimizer(tied,training)
    groups=optimizer.param_groups
    ids=[id(p) for g in groups for p in g['params']]
    assert len(ids)==len(set(ids))==len(list(tied.parameters()))
    table_groups=[g for g in groups if g['parameter_names']==['transformer.wte.weight']]
    assert len(table_groups)==1 and table_groups[0]['width_lr_scale']==1.
    assert not any('lm_head.weight' in g['parameter_names'] for g in groups)
    table=tied.transformer.wte.weight;before=table.detach().double().clone();g=table.grad.detach().double().clone()
    oracle=before*(1-.00006)-.005*g/(g.abs()+1e-8)
    optimizer.step()
    assert torch.allclose(table.double(),oracle,atol=4e-7,rtol=2e-6)
    adam=optimizer.optimizers[-1]
    assert set(adam.state[table])=={'step','exp_avg','exp_avg_sq'}
    assert adam.state[table]['step'].item()==1
    assert torch.allclose(adam.state[table]['exp_avg'].double(),g*.1,atol=1e-8,rtol=2e-6)
