"""BPE table sharing arithmetic and lineage; no quality selection."""
import copy
import gc
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from src.models.gpt import GPT
from src.models.tokenizer_architecture import architecture_contract
from src.models.tokenizer_tables import POLICY, KINDS
from src.attention.cache import build_caches
from src.train.optimizers import build_optimizer
from src.train.train import run_training
from src.train.weight_averaging import export_averaged_checkpoint
from src.utils.checkpoint import load_model_weights, read_checkpoint, checkpoint_loader

RECIPE = Path('research/tokenizer-evidence/horizon-001/adopted-width256-gqa-recipe.json')


def config(root, kind='scaled_tied', width=64):
    cfg = json.loads(RECIPE.read_text())['config']
    cfg['model'].update(dim=width, n_heads=width//32, tokenizer_architecture_policy=POLICY,
                        weight_tying_kind=kind, weight_tying_width_policy=POLICY)
    cfg['training'].update(device='cpu', seed=3201, max_iters=3, eval_every=3, eval_iters=1,
                           batch_size=8, checkpoint_dir=str(root), metrics_file=None, summary_file=None)
    cfg['training']['weight_averaging'].update(total_steps=3, start_step=1, interval=1)
    cfg['training']['lr_schedule'].update(total_steps=3, start_step=1)
    return OmegaConf.create(cfg)


@pytest.mark.parametrize('width', [64, 256])
def test_initialization_gradient_roles_cache_and_optimizer(tmp_path, width):
    torch.set_num_threads(1)
    models = []; rngs = []
    for kind in KINDS:
        torch.manual_seed(3201)
        models.append(GPT(config(tmp_path, kind, width).model))
        rngs.append(torch.get_rng_state())
    legacy, matched, tied = models
    assert all(torch.equal(rngs[0], r) for r in rngs)
    base = json.loads(RECIPE.read_text())['config']['model'];base.update(dim=width,n_heads=width//32)
    torch.manual_seed(3201);original=GPT(OmegaConf.create(base))
    assert torch.equal(torch.get_rng_state(),rngs[0])
    assert all(torch.equal(v, original.state_dict()[k]) for k,v in legacy.state_dict().items())
    del original
    assert torch.equal(matched.lm_head.weight, matched.transformer.wte.weight)
    assert matched.lm_head.weight.data_ptr() != matched.transformer.wte.weight.data_ptr()
    assert tied.lm_head is None and 'lm_head.weight' not in tied.state_dict()
    assert sum(p.numel() for p in matched.parameters())-sum(p.numel() for p in tied.parameters())==50257*width
    for k,v in tied.state_dict().items():assert torch.equal(v, matched.state_dict()[k])
    x=torch.tensor([[100,200,300,400]]);y=x+1
    with torch.no_grad():
        for model in models:
            model.eval();cache=build_caches(OmegaConf.create({'model': OmegaConf.to_container(model.cfg)}),'cpu',batch_size=1)
            full=model(x)[0]
            actual=torch.cat([model(x[:,:2],cache=cache)[0],model(x[:,2:],cache=cache,start_pos=2)[0]],1)
            torch.testing.assert_close(actual,full,rtol=2e-5,atol=1e-5)
            changed=x.clone();changed[:,2:]+=10
            torch.testing.assert_close(full[:,:2],model(changed)[0][:,:2],rtol=0,atol=0)
    matched.double();tied.double()
    a,la=matched(x,y);b,lb=tied(x,y)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    captured=[];hook=matched.transformer.ln_f.register_forward_hook(lambda m,i,o:captured.append(o))
    logits=matched(x)[0];hook.remove()
    oracle=torch.einsum('btd,vd->btv',captured[0],matched.lm_head.weight)/math.sqrt(192)/(width/64)
    torch.testing.assert_close(logits,oracle,rtol=1e-10,atol=1e-11)
    la.backward();lb.backward()
    torch.testing.assert_close(tied.transformer.wte.weight.grad,matched.transformer.wte.weight.grad+matched.lm_head.weight.grad,rtol=1e-9,atol=1e-10)
    for name,p in tied.named_parameters():
        if name!='transformer.wte.weight':torch.testing.assert_close(p.grad,dict(matched.named_parameters())[name].grad,rtol=1e-9,atol=1e-10)
    # One shared AdamW state and one independent decay, including both gradient roles.
    optimizer=build_optimizer(tied,config(tmp_path,width=width).training)
    groups=[g for g in optimizer.param_groups if g['parameter_names']==['transformer.wte.weight']]
    assert len(groups)==1 and groups[0]['width_lr_scale']==1.
    assert all('lm_head.weight' not in g['parameter_names'] for g in optimizer.param_groups)
    table=tied.transformer.wte.weight;before=table.detach().clone();gradient=table.grad.clone()
    lr=groups[0]['lr'];expected=before*(1-.00006)-lr*gradient/(gradient.abs()+1e-8)
    optimizer.step();torch.testing.assert_close(table,expected,rtol=1e-9,atol=1e-10)
    assert optimizer.optimizers[-1].state[table]['step']==1


@pytest.mark.parametrize('key,value', [('weight_tying_kind','raw_tied'),('weight_tying_width_policy','mha4_scaled_table_sharing'),('readout_scale',.5),('dim',128),('rope',0),('tie_embeddings',True),('attn_type','mha')])
def test_reject_undeclared_contract(tmp_path,key,value):
    cfg=config(tmp_path);cfg.model[key]=value
    with pytest.raises((ValueError,TypeError)):architecture_contract(cfg.model)


def test_checkpoint_mutation_and_aliases(tmp_path):
    torch.set_num_threads(1);cfg=config(tmp_path);model=GPT(cfg.model)
    assert architecture_contract(SimpleNamespace(**OmegaConf.to_container(cfg.model))) == architecture_contract(cfg.model)
    payload=dict(cfg=OmegaConf.to_container(cfg),model_state=model.state_dict())
    payload['cfg']['model']['weight_tying_kind']='matched_untied'
    with pytest.raises(ValueError):load_model_weights(model,payload)
    model.cfg.weight_tying_kind='legacy_untied'
    with pytest.raises(ValueError):build_optimizer(model,cfg.training)


def test_real_resume_and_ema_with_bounded_temporary_storage(tmp_path):
    # Three temporary tied-width64 raw checkpoints plus one EMA fit in190MiB.
    # These disposable test fixtures are removed on success; research checkpoints are untouched.
    if shutil.disk_usage(tmp_path).free < 8*1024**3+190*1024**2:
        pytest.skip('Insufficient headroom above research reserve for CPU checkpoint prerequisite')
    torch.set_num_threads(1);root=tmp_path/'table-resume';root.mkdir()
    cfg=config(root/'full');full=run_training(cfg);expected=read_checkpoint(full['final_checkpoint'])
    cfg.training.checkpoint_dir=str(root/'partial');cfg.training.max_train_seconds=1e-9
    partial=run_training(cfg);assert partial['completed_steps']==1
    cfg.training.max_train_seconds=None;cfg.training.resume_from=partial['final_checkpoint']
    resumed=run_training(cfg);actual=read_checkpoint(resumed['final_checkpoint'])
    for k,v in expected['model_state'].items():torch.testing.assert_close(v,actual['model_state'][k],rtol=0,atol=0)
    for split in ('train','val','test'):assert torch.equal(expected['training_state']['loader_rng'][split],actual['training_state']['loader_rng'][split])
    avg=export_averaged_checkpoint(resumed['final_checkpoint'],root/'h200.pt',200)
    payload=read_checkpoint(avg['checkpoint']);clone=GPT(OmegaConf.create(payload['cfg']['model']));load_model_weights(clone,payload)
    checkpoint_loader(cfg,payload,cfg.data.path,'cpu')
    assert 'lm_head.weight' not in payload['model_state'] and payload['tokenizer_contract']['name']=='gpt2'
    changed=copy.deepcopy(cfg);changed.model.weight_tying_kind='matched_untied'
    with pytest.raises(ValueError):run_training(changed)
    assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) < 190*1024**2
    del expected,actual,payload,clone;gc.collect()
    shutil.rmtree(root)
