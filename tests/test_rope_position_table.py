import copy
from pathlib import Path
import pytest
import torch
from omegaconf import OmegaConf
from src.models.gpt import GPT
from src.train.optimizers import build_optimizer
from src.utils.checkpoint import load_model_weights, save_checkpoint, read_checkpoint
from tests.test_gated_study import config, training


def model_config(attn,context):
    c=config(64,'swiglu');c.block_size=context;c.attn_type=attn
    if attn=='mla':c.update(n_layers=4,depth_width_policy='depth4_unscaled',attention_width_policy='mla_quarter_head32')
    if attn=='gqa':c.attention_width_policy='gqa_half_head32'
    return c


def historical_class():
    scope={'__package__':'src.models'}
    exec((Path(__file__).parent/'fixtures/gpt_before_rope_table_fix.txt').read_text(),scope)
    return scope['GPT']

@pytest.mark.parametrize('attn',['mha','gqa','mla'])
def test_rope_context_changes_no_parameters_or_rng_draws(attn):
    torch.set_num_threads(1)
    models=[];states=[]
    for context in [128,256,512]:
        torch.manual_seed(2021);m=GPT(model_config(attn,context));models.append(m);states.append(torch.get_rng_state())
        assert 'wpe' not in m.transformer
        assert all('wpe' not in n for n,_ in m.named_parameters())
        opt=build_optimizer(m,training())
        assert all('wpe' not in n for group in opt.param_groups for n in group['parameter_names'])
    for other,state in zip(models[1:],states[1:]):
        assert dict(models[0].named_parameters()).keys()==dict(other.named_parameters()).keys()
        for a,b in zip(models[0].parameters(),other.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(states[0],state,rtol=0,atol=0)
        x=torch.arange(16).reshape(2,8)%models[0].cfg.vocab_size
        torch.testing.assert_close(models[0](x)[0],other(x)[0],rtol=0,atol=0)

@pytest.mark.parametrize('attn',['mha','gqa','mla'])
def test_historical_rope_forward_exact_and_checkpoint_unchanged(attn,tmp_path):
    torch.set_num_threads(1);cfg=model_config(attn,128);torch.manual_seed(44)
    old=historical_class()(cfg);path=tmp_path/'old.pt';save_checkpoint(path,old,cfg=OmegaConf.create({'model':cfg}))
    before=path.read_bytes();ck=read_checkpoint(path);state=copy.deepcopy(ck['model_state'])
    new=GPT(cfg)
    with pytest.warns(UserWarning,match='historical RoPE'):load_model_weights(new,ck)
    x=torch.arange(32).reshape(4,8)%cfg.vocab_size
    torch.testing.assert_close(old(x)[0],new(x)[0],rtol=0,atol=0)
    assert path.read_bytes()==before
    assert ck['model_state'].keys()==state.keys()
    for name in state:torch.testing.assert_close(state[name],ck['model_state'][name],rtol=0,atol=0)
    with pytest.raises(ValueError,match='original source'):load_model_weights(new,ck,for_resume=True)
    bad=copy.deepcopy(ck);bad['model_state']['transformer.wpe.weight']=torch.zeros(1,1)
    with pytest.raises(ValueError,match='contract'):load_model_weights(new,bad)
    bad=copy.deepcopy(ck);del bad['model_state']['lm_head.weight']
    with pytest.warns(UserWarning),pytest.raises(RuntimeError):load_model_weights(new,bad)


def test_learned_positions_preserve_old_initialization_and_gradients():
    torch.set_num_threads(1);cfg=model_config('mha',128);cfg.rope=0
    torch.manual_seed(7);old=historical_class()(cfg)
    torch.manual_seed(7);new=GPT(cfg)
    assert 'wpe' in new.transformer
    for a,b in zip(old.parameters(),new.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    x=torch.arange(8).reshape(1,8);torch.testing.assert_close(old(x)[0],new(x)[0],rtol=0,atol=0)
    new(x,x)[1].backward();assert new.transformer.wpe.weight.grad.abs().sum()>0


def test_new_checkpoint_optimizer_continuation(tmp_path):
    torch.set_num_threads(1);cfg=model_config('mha',128);m=GPT(cfg);opt=build_optimizer(m,training());x=torch.arange(8).reshape(1,8)
    m(x,x)[1].backward();opt.step();opt.zero_grad()
    path=tmp_path/'new.pt';save_checkpoint(path,m,opt,1,OmegaConf.create({'model':cfg}));ck=read_checkpoint(path)
    assert ck['parameter_layout_version']==2 and 'transformer.wpe.weight' not in ck['model_state']
    other=GPT(cfg);load_model_weights(other,ck,for_resume=True);other_opt=build_optimizer(other,training());other_opt.load_state_dict(ck['optimizer_state'])
    for model,optimizer in [(m,opt),(other,other_opt)]:model(x,x)[1].backward();optimizer.step()
    for a,b in zip(m.parameters(),other.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
