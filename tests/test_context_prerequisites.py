import pytest
import torch
from omegaconf import OmegaConf

from src.data.loaders import TinyDataLoader
from src.models.gpt import GPT
from src.train.optimizers import build_optimizer
from src.train.train import _training_contract, _contracts_match
from tests.test_gqa_study import gqa_config
from tests.test_gated_study import training


def loaders(tmp_path):
    p=tmp_path/'text.txt';p.write_text('abcdefghijklmnopqrstuvwxyz\n'*300)
    return [TinyDataLoader(p,1024//c,c,'cpu',seed=2021,train_fraction=.8,val_fraction=.1,
                           eval_seed=2026,batch_layout='paired_blocks512') for c in [128,256,512]]


def test_contexts_receive_exact_same_targets_and_rng(tmp_path):
    ls=loaders(tmp_path)
    for _ in range(5):
        batches=[l.get_batch() for l in ls]
        for x,y in batches[1:]:
            assert torch.equal(x.flatten(),batches[0][0].flatten())
            assert torch.equal(y.flatten(),batches[0][1].flatten())
        assert all(torch.equal(ls[0].generators['train'].get_state(),l.generators['train'].get_state()) for l in ls)
    before=ls[0].generators['train'].get_state().clone();ls[0].get_batch('val')
    assert torch.equal(before,ls[0].generators['train'].get_state())


def test_default_sampling_unchanged_and_layout_is_resume_contract(tmp_path):
    ls=loaders(tmp_path);p=tmp_path/'text.txt'
    a=TinyDataLoader(p,8,128,'cpu',seed=9)
    b=TinyDataLoader(p,8,128,'cpu',seed=9,batch_layout='independent')
    for x,y in zip(a.get_batch(),b.get_batch()):assert torch.equal(x,y)
    cfg=OmegaConf.create(dict(model=dict(dim=64),data=dict(path=str(p)),training=dict(seed=9)))
    old=_training_contract(cfg,'cpu',a);new=_training_contract(cfg,'cpu',ls[0])
    assert 'batch_layout' not in old and new['batch_layout']=='paired_blocks512'
    assert not _contracts_match(old,{**old,'batch_layout':'paired_blocks512'})


@pytest.mark.parametrize('batch,context,layout',[(8,256,'paired_blocks512'),(16,64,'paired_blocks512'),(8,128,'bogus')])
def test_bad_sampling_contract_rejected(tmp_path,batch,context,layout):
    p=tmp_path/'text';p.write_text('abc'*2000)
    with pytest.raises(ValueError):TinyDataLoader(p,batch,context,'cpu',batch_layout=layout)


@pytest.mark.parametrize('width',[64,128,256])
def test_explicit_four_layer_gqa_contract_and_context_independent_initialization(width):
    torch.set_num_threads(1)
    c=gqa_config(width);c.n_layers=4;c.depth_width_policy='depth4_gqa_unscaled'
    saved=None;rng=None
    for context in [128,256,512]:
        c.block_size=context;torch.manual_seed(2021);model=GPT(c)
        weights={n:p.detach().clone() for n,p in model.named_parameters()}
        if saved is not None:
            assert all(torch.equal(p,saved[n]) for n,p in weights.items())
            assert torch.equal(rng,torch.get_rng_state())
        saved=weights;rng=torch.get_rng_state().clone()
    opt=build_optimizer(model,training());x=torch.arange(8).reshape(1,8)
    loss=model(x,x)[1];loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    opt.step()
    assert all(b.attn.n_kv_heads==width//64 for b in model.transformer.h)
    c.depth_width_policy='depth4_unscaled'
    with pytest.raises(ValueError):build_optimizer(GPT(c),training())


@pytest.mark.parametrize('key,value',[('n_layers',3),('attn_type','mha'),('attention_width_policy','mha_head16'),('base_width',128)])
def test_new_depth_policy_does_not_enable_unregistered_shapes(key,value):
    c=gqa_config(64);c.n_layers=4;c.depth_width_policy='depth4_gqa_unscaled';c[key]=value
    with pytest.raises(ValueError):build_optimizer(GPT(c),training())
