"""Actual model/optimizer tests: defer while any research training is active."""
import copy
import math
import pytest
import torch
from omegaconf import OmegaConf
from src.models.gpt import GPT
from src.train.optimizers import build_optimizer
from src.train.lr_schedule import LearningRateSchedule
from src.train.train import run_training
from src.utils.checkpoint import read_checkpoint
from src.research.capacity_study import suite_spec
from tests.test_capacity_study import plan
from tests.test_schedule_training import equal_nested


def cfg_for(family,width,kind='constant'):
    c=OmegaConf.create(suite_spec(plan(kind),family,.0025)['candidate']['config'])
    c.model.update(dim=width,n_heads=width//32,vocab_size=65)
    return c


@pytest.mark.parametrize('width',[64,128,256])
@pytest.mark.parametrize('family',['dense4','wide4','dense6'])
def test_counts_initialization_groups_and_finite_updates(width,family):
    torch.set_num_threads(1);torch.manual_seed(9302)
    c=cfg_for(family,width);model=GPT(c.model);opt=build_optimizer(model,c.training)
    depth=6 if family=='dense6' else 4;hidden=21*width//(4 if family=='wide4' else 8)
    expected=(2*65+2)*width+depth*(4*width**2+6*width+3*width*hidden)
    assert sum(p.numel() for p in model.parameters())==expected
    assert len(model.transformer.h)==depth and model.width_mult==width/64
    names=[]
    for group in opt.param_groups:
        assert len(group['parameter_names'])==1
        name=group['parameter_names'][0];names.append(name)
        param=group['params'][0]
        hidden_matrix=name.startswith('transformer.h.') and param.ndim==2
        scale=1/math.sqrt(width/64) if hidden_matrix else 1
        assert group['width_lr_scale']==scale and group['lr']==.0025*scale
        assert group['weight_decay']==0
        param.grad=torch.full_like(param,.001)
    assert set(names)==set(dict(model.named_parameters())) and len(names)==len(set(names))
    opt.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    for b in model.transformer.h:
        assert b.hidden_dim==hidden and tuple(b.attn.qkv.weight.shape)==(3*width,width)
        assert tuple(b.mlp.linear_gate.weight.shape)==(hidden,width)
        assert tuple(b.mlp.linear_down.weight.shape)==(width,hidden)
    model.eval();x=torch.arange(16).reshape(2,8)%65
    logits,_=model(x);y=x.clone();y[:,4:]=(y[:,4:]+1)%65
    altered,_=model(y)
    torch.testing.assert_close(logits[:,:4],altered[:,:4],rtol=0,atol=0)
    assert torch.isfinite(logits).all()


def test_depth6_requires_new_contract_and_preserves_prefix_initialization():
    torch.set_num_threads(1)
    c=cfg_for('dense6',64)
    for old in ['depth4_unscaled','depth4_wide_unscaled','depth4_gqa_unscaled']:
        bad=copy.deepcopy(c);bad.model.depth_width_policy=old
        with pytest.raises(ValueError):build_optimizer(GPT(bad.model),bad.training)
    torch.manual_seed(9302);small=GPT(cfg_for('dense4',64).model)
    torch.manual_seed(9302);deep=GPT(c.model)
    for name,value in small.named_parameters():
        if name.startswith('transformer.h.') or name=='transformer.wte.weight':
            torch.testing.assert_close(value,dict(deep.named_parameters())[name],rtol=0,atol=0)
    # Additional blocks consume RNG before readout creation: no claim that full
    # parameter initialization is paired across different architectures.
    assert not torch.equal(small.lm_head.weight,deep.lm_head.weight)


@pytest.mark.parametrize('family',['wide4','dense6'])
@pytest.mark.parametrize('width',[64,256])
@pytest.mark.parametrize('kind',['constant','cosine','linear_cooldown'])
def test_capacity_full_trainer_resume_exact(tmp_path,family,width,kind):
    torch.set_num_threads(1)
    cfg=cfg_for(family,width,kind);corpus=tmp_path/'synthetic.txt';corpus.write_text('abcdefghijklmno\n'*1024)
    cfg.data.path=str(corpus);cfg.training.update(max_iters=8,device='cpu',strict_device=True,seed=42,
        eval_seed=2026,eval_every=2,eval_iters=2,checkpoint_every=4,checkpoint_dir=str(tmp_path/'full'),keep_best_checkpoint=False)
    if kind!='constant':
        cfg.training.lr_schedule.total_steps=8
        if kind=='linear_cooldown':cfg.training.lr_schedule.start_step=5
    full=run_training(copy.deepcopy(cfg));middle=list((tmp_path/'full').glob('step_000004_*.pt'));assert len(middle)==1
    cfg.training.checkpoint_dir=str(tmp_path/'resumed');cfg.training.resume_from=str(middle[0])
    resumed=run_training(copy.deepcopy(cfg));a,b=[read_checkpoint(r['final_checkpoint']) for r in [full,resumed]]
    for key in ['model_state','optimizer_state']:equal_nested(a[key],b[key])
    for key in ['loader_rng','torch_rng','contract']:equal_nested(a['training_state'][key],b['training_state'][key])
    for key in ['val_ce_loss','train_ce_loss','lr_multiplier']:assert full['final_metric'][key]==resumed['final_metric'][key]
    bad=copy.deepcopy(cfg);bad.model.n_layers+=1
    with pytest.raises(ValueError):run_training(bad)
