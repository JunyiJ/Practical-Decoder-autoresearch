"""Unmodified namespace baseline parity; no candidate-policy validity claimed."""
import importlib,json
from pathlib import Path
import pytest,torch
from omegaconf import OmegaConf
from src.models.gpt import GPT as OriginalGPT
from src.models.tokenizer_architecture import validate_architecture_model as original_validate
from src.train.optimizers import build_optimizer as original_optimizer
from src.attention.cache import build_caches as original_caches
from research.block_norm_v1.runtime.models.gpt import GPT
from research.block_norm_v1.runtime.models.tokenizer_architecture import validate_architecture_model
from research.block_norm_v1.runtime.train.optimizers import build_optimizer
from research.block_norm_v1.runtime.attention.cache import build_caches
ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def cfg():
    return OmegaConf.create(json.loads((ROOT/'research/tokenizer-evidence/mla-001/adopted-width256-mla-recipe.json').read_text())['config'])


@pytest.fixture(scope='module')
def models(cfg):
    before=torch.get_rng_state().clone()
    try:
        torch.manual_seed(372001);old=OriginalGPT(cfg.model);old_rng=torch.get_rng_state().clone()
        torch.manual_seed(372001);new=GPT(cfg.model);new_rng=torch.get_rng_state().clone()
        yield old.eval(),new.eval(),old_rng,new_rng
    finally:torch.set_rng_state(before)


def test_every_manifest_module_resolves_inside_new_namespace():
    m=json.loads((ROOT/'research/tokenizer-evidence/block-norm-preparation-001/runtime-dependency-map.json').read_text())
    for module,ref in m['modules'].items():
        name='research.block_norm_v1.runtime'+module.removeprefix('src')
        loaded=importlib.import_module(name)
        expected=ROOT/'research/block_norm_v1/runtime'/Path(ref['path']).relative_to('src')
        assert Path(loaded.__file__).resolve()==expected.resolve()
    assert GPT is not OriginalGPT and build_optimizer is not original_optimizer


def test_native_control_initialization_state_rng_and_validation(models):
    old,new,a,b=models;assert torch.equal(a,b)
    original_validate(old);validate_architecture_model(new)
    assert old.tokenizer_architecture_signature==new.tokenizer_architecture_signature
    old_state,new_state=old.state_dict(),new.state_dict();assert old_state.keys()==new_state.keys()
    assert all(torch.equal(old_state[k],new_state[k]) for k in old_state)
    assert sum(p.numel() for p in new.parameters())==28492416


def test_control_forward_ce_and_cache_are_bit_exact(models,cfg):
    old,new,_,_=models;tokens=torch.randint(0,50257,(1,7),generator=torch.Generator().manual_seed(372002))
    with torch.no_grad():
        a,loss_a=old(tokens,targets=tokens);b,loss_b=new(tokens,targets=tokens)
        assert torch.equal(a,b) and torch.equal(loss_a,loss_b)
        old_cache=original_caches(cfg,'cpu',batch_size=1,dtype=torch.float32)
        new_cache=build_caches(cfg,'cpu',batch_size=1,dtype=torch.float32)
        for start,end in [(0,3),(3,4),(4,7)]:
            a,_=old(tokens[:,start:end],cache=old_cache,start_pos=start)
            b,_=new(tokens[:,start:end],cache=new_cache,start_pos=start)
            assert torch.equal(a,b)
        for a,b in zip(old_cache,new_cache):
            assert all(torch.equal(a.state_dict()[k],v) for k,v in b.state_dict().items())


def test_optimizer_names_counts_order_and_width_decay_match(models,cfg):
    old,new,_,_=models;a=original_optimizer(old,cfg.training);b=build_optimizer(new,cfg.training)
    def summary(opt):
        return [(type(child).__name__,[{k:v for k,v in g.items() if k!='params'}|{'shapes':[tuple(p.shape) for p in g['params']]} for g in child.param_groups]) for child in opt.optimizers]
    assert a.decay==b.decay and summary(a)==summary(b)
    assert len({id(p) for g in b.param_groups for p in g['params']})==len(list(new.parameters()))


def test_unmodified_copy_rejects_layer_config_and_silent_module_replacement(cfg,models):
    changed=OmegaConf.create(OmegaConf.to_container(cfg.model));changed.norm_type='layernorm'
    with pytest.raises(ValueError,match='norm_type'):GPT(changed)
    _,new,_,_=models;old=new.transformer.h[0].ln1
    try:
        new.transformer.h[0].ln1=torch.nn.LayerNorm(256,eps=2.**-23)
        with pytest.raises(ValueError,match='block semantics'):validate_architecture_model(new)
    finally:new.transformer.h[0].ln1=old
