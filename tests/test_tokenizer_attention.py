import copy

import pytest
import torch
from omegaconf import OmegaConf

from src.attention.cache import KVCache
from src.models.gpt import GPT
from src.models.rope import apply_rotary_emb
from src.models.tokenizer_architecture import architecture_contract, validate_architecture_model
from src.research.tokenizer_attention_study import config, choose, specs, quality_conclusion
from src.research.tokenizer_study import make_config
from src.train.optimizers import build_optimizer
from src.train.train import run_training
from src.utils.checkpoint import read_checkpoint, load_model_weights
from src.train.weight_averaging import export_averaged_checkpoint


@pytest.mark.parametrize('width',[64,256])
def test_new_mha_matches_adopted_math_and_optimizer(tmp_path,width):
    old=OmegaConf.create(make_config(tmp_path/'old',width=width,lr=.0025,device='cpu'))
    new=OmegaConf.create(config(tmp_path/'new',width=width,device='cpu'))
    torch.manual_seed(71);a=GPT(old.model)
    torch.manual_seed(71);b=GPT(new.model)
    assert a.state_dict().keys()==b.state_dict().keys()
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    ids=torch.tensor([[100,200,300,400,500]])
    x=a(ids)[0];y=b(ids)[0];torch.testing.assert_close(x,y,rtol=0,atol=0)
    x.square().mean().backward();y.square().mean().backward()
    for (name,p),(other,q) in zip(a.named_parameters(),b.named_parameters()):
        assert name==other;torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    ao=build_optimizer(a,old.training);bo=build_optimizer(b,new.training)
    assert [{k:v for k,v in g.items() if k!='params'} for g in ao.param_groups]==[{k:v for k,v in g.items() if k!='params'} for g in bo.param_groups]
    ao.step();bo.step()
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)


@pytest.mark.parametrize('width',[64,256])
def test_gqa_half_scale_matches_expanded_kv_reference_and_gradient(tmp_path,width):
    cfg=OmegaConf.create(config(tmp_path,family='gqa',width=width,device='cpu'))
    model=GPT(cfg.model).double();block=model.transformer.h[0];att=block.attn
    x=torch.randn(2,7,width,dtype=torch.float64,requires_grad=True)
    actual=att(x,freq_cos=block.freq_cos,freq_sin=block.freq_sin)[0]
    q,k,v=att.qkv(x).split([width,width//2,width//2],dim=-1)
    q=q.view(2,7,width//32,32).transpose(1,2)
    k=k.view(2,7,width//64,32).transpose(1,2)
    v=v.view(2,7,width//64,32).transpose(1,2)
    q=apply_rotary_emb(q,block.freq_cos[:7],block.freq_sin[:7])
    k=apply_rotary_emb(k,block.freq_cos[:7],block.freq_sin[:7]).repeat_interleave(2,dim=1)
    v=v.repeat_interleave(2,dim=1)
    score=(q@k.transpose(-1,-2))*(.5/(32**.5))
    score=score.masked_fill(~torch.ones(7,7,dtype=torch.bool).tril(),float('-inf'))
    expected=att.out((score.softmax(-1)@v).transpose(1,2).reshape(2,7,width))
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    ga=torch.autograd.grad(actual.square().sum(),x,retain_graph=True)[0]
    gb=torch.autograd.grad(expected.square().sum(),x)[0]
    torch.testing.assert_close(ga,gb,rtol=1e-11,atol=1e-11)


@pytest.mark.parametrize('width',[64,256])
@pytest.mark.parametrize('family',['mha','gqa'])
def test_full_cached_causal_reset_and_memory(tmp_path,width,family):
    cfg=OmegaConf.create(config(tmp_path,family=family,width=width,device='cpu'))
    model=GPT(cfg.model).eval();validate_architecture_model(model)
    ids=torch.arange(17).view(1,-1)+100
    cache=[KVCache(cfg,'cpu',batch_size=1) for _ in range(4)]
    with torch.no_grad():
        full=model(ids)[0];parts=[]
        for lo,hi in ((0,3),(3,8),(8,17)):parts.append(model(ids[:,lo:hi],cache=cache,start_pos=lo)[0])
        torch.testing.assert_close(torch.cat(parts,1),full,rtol=1e-5,atol=1e-6)
        changed=ids.clone();changed[:,8:]+=40
        torch.testing.assert_close(model(changed)[0][:,:8],full[:,:8],rtol=0,atol=0)
        # Separate request uses separate state; resetting the original restores it.
        other=[KVCache(cfg,'cpu',batch_size=1) for _ in range(4)]
        model(changed[:,:5],cache=other)
        for c in cache:c.reset()
        torch.testing.assert_close(model(ids,cache=cache)[0],full,rtol=1e-5,atol=1e-6)
    size=sum(b.numel()*b.element_size() for c in cache for b in c.buffers())
    assert size==4*2*128*width*4*(.5 if family=='gqa' else 1)


@pytest.mark.parametrize('key,value',[('dropout',.1),('rope',0),('n_layers',6),('vocab_size',65),('attention_logit_scale',1.),('ffn_bias',True),('n_kv_heads',1),('readout_norm_kind','rms'),('tokenizer_architecture_policy','typo')])
def test_contract_rejects_unreviewed_changes(tmp_path,key,value):
    cfg=config(tmp_path);cfg['model'][key]=value
    with pytest.raises(ValueError):architecture_contract(cfg['model'])


def test_mutation_and_checkpoint_policy_mismatch(tmp_path):
    cfg=OmegaConf.create(config(tmp_path,family='gqa',device='cpu'));model=GPT(cfg.model)
    checkpoint={'cfg':OmegaConf.to_container(cfg),'model_state':model.state_dict()}
    wrong=copy.deepcopy(checkpoint);wrong['cfg']['model']['attention_logit_scale']=1.
    with pytest.raises(ValueError):load_model_weights(model,wrong)
    model.cfg.attn_type='mha'
    with pytest.raises(ValueError):build_optimizer(model,cfg.training)


def test_real_gqa_training_resume_and_ema(tmp_path):
    cfg=OmegaConf.create(config(tmp_path/'full',family='gqa',steps=4,device='cpu'))
    full=run_training(cfg);a=read_checkpoint(full['final_checkpoint'])
    cfg.training.checkpoint_dir=str(tmp_path/'partial');cfg.training.max_train_seconds=1e-9
    partial=run_training(cfg);assert partial['completed_steps']==1
    cfg.training.max_train_seconds=None;cfg.training.resume_from=partial['final_checkpoint']
    resumed=run_training(cfg);b=read_checkpoint(resumed['final_checkpoint'])
    for k,v in a['model_state'].items():torch.testing.assert_close(v,b['model_state'][k],rtol=0,atol=0)
    avg=export_averaged_checkpoint(resumed['final_checkpoint'],tmp_path/'ema.pt',200)
    payload=read_checkpoint(avg['checkpoint']);model=GPT(OmegaConf.create(payload['cfg']['model']));load_model_weights(model,payload)
    assert payload['tokenizer_contract']['name']=='gpt2'


def test_finite_selection_and_neutrality_rules():
    plan=dict(pilot=False,widths=[64,256],screen_seeds=[2701,2702],confirm_seeds=[2711,2712,2713],learning_rates=[.0025,.00125],neutrality_margin=.01)
    rows=[dict(s,h200={'mean_ce':5+s['lr']}) for s in specs(plan)]
    choice=choose(rows,plan);assert choice=={'mha':.00125,'gqa':.00125}
    assert len(specs(plan,choice))==20
    with pytest.raises(ValueError):choose(rows[:-1],plan)
    confirms=[]
    for s in specs(plan,choice):
        if s['stage']!='confirm':continue
        loss=5+(.005 if s['family']=='gqa' else 0.)
        confirms.append(dict(s,h200=dict(mean_ce=loss,context=128,block_tokens=2048,block_target_counts=[2048,2048],block_loss_sums=[loss*2048]*2)))
    result=quality_conclusion(confirms,plan)
    assert all(c['quality_eligible'] for c in result['comparisons']) and result['adopted'] is False
    for r in confirms:
        if r['family']=='gqa':r['h200'].update(mean_ce=5.02,block_loss_sums=[5.02*2048]*2)
    assert not any(c['quality_eligible'] for c in quality_conclusion(confirms,plan)['comparisons'])
