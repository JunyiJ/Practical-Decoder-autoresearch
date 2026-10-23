import copy
import pytest
import torch
from omegaconf import OmegaConf
from src.models.gpt import GPT
from src.train.optimizers import build_optimizer, HybridOptimizer
from src.train.train import run_training, evaluate_checkpoint
from src.utils.checkpoint import read_checkpoint


def config(width=64, parameterization='sp', optimizer='adamw', decay_mode='conventional'):
    return OmegaConf.create({'model': {'dim': width, 'n_heads': max(1,width//32), 'n_layers': 1,
        'vocab_size': 10, 'block_size': 8, 'dropout': 0., 'rope': 1,
        'norm_type': 'rmsnorm', 'attn_type': 'mha', 'mlp_type': 'dense',
        'parameterization': parameterization, 'base_width': 64},
        'training': {'learning_rate': .002, 'weight_decay': .1, 'optimizer': optimizer,
                     'decay_mode': decay_mode}})


def test_base_width_mup_is_identical_to_sp_including_one_update():
    x=torch.tensor([[0,1,2,3]])
    models=[]
    for mode in ['sp','mup']:
        cfg=config(parameterization=mode); torch.manual_seed(9); model=GPT(cfg.model)
        opt=build_optimizer(model,cfg.training); model(x,x)[1].backward(); opt.step(); models.append(model)
    for a,b in zip(models[0].parameters(),models[1].parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('optimizer', ['adamw','muon'])
def test_independent_decay_exact_at_zero_lr_and_excludes_unused_parameters(optimizer):
    cfg=config(optimizer=optimizer, decay_mode='independent'); cfg.training.learning_rate=0
    model=GPT(cfg.model); before={n:p.clone() for n,p in model.named_parameters()}
    opt=build_optimizer(model,cfg.training)
    for n,p in model.named_parameters():
        if 'wpe' not in n: p.grad=torch.zeros_like(p)
    opt.step()
    for n,p in model.named_parameters():
        torch.testing.assert_close(p,before[n]*(1 if 'wpe' in n else .9),rtol=0,atol=0)


@pytest.mark.parametrize('kind,scales',[('adamw',[1.,.5,.25]),('muon',[1.,2**-.5,.5])])
def test_width_scaling_and_disjoint_parameter_routing(kind,scales):
    for w,scale in zip([64,128,256],scales):
        cfg=config(w,'mup',kind); model=GPT(cfg.model); opt=build_optimizer(model,cfg.training)
        found=[]
        for underlying in opt.optimizers:
            for group in underlying.param_groups:
                name=group['parameter_names'][0]; p=group['params'][0]; found.append(id(p))
                hidden=name.startswith('transformer.h.') and p.ndim==2
                assert group['lr']==pytest.approx(.002*(scale if hidden else 1))
                if isinstance(underlying,torch.optim.Muon): assert hidden
        assert len(found)==len(set(found))==len(list(model.parameters()))


def test_mup_initialization_readout_and_bias_scales():
    for w in [64,128,256]:
        cfg=config(w); torch.manual_seed(2); sp=GPT(cfg.model)
        cfg.model.parameterization='mup'; torch.manual_seed(2); mup=GPT(cfg.model)
        factor=(w/64)**.5
        for name,p in mup.named_parameters():
            q=dict(sp.named_parameters())[name]
            bias=name.startswith('transformer.h.') and name.endswith('bias')
            torch.testing.assert_close(p,q*factor if name=='lm_head.weight' or bias else q)


@pytest.mark.parametrize('optimizer,mode,param', [('adamw','independent','sp'),
    ('adamw','independent','mup'),('muon','independent','mup'),('muon','conventional','sp')])
def test_checkpoint_resume_and_contract(tmp_path,optimizer,mode,param):
    torch.set_num_threads(1)
    cfg=config(32,param,optimizer,mode); cfg.model.base_width=16
    corpus=tmp_path/'corpus.txt';corpus.write_text('abcdefghij'*100)
    cfg.data={'path':str(corpus),'train_fraction':.8,'val_fraction':.1}
    cfg.training.update({'device':'cpu','strict_device':True,'seed':42,'eval_seed':123,
        'batch_size':2,'grad_clip':1.,'max_iters':2,'eval_every':1,'eval_iters':2,
        'weight_decay':.00006,'checkpoint_dir':str(tmp_path/'split'),'checkpoint_every':0,
        'keep_best_checkpoint':False})
    first=run_training(cfg)
    cfg.training.max_iters=4;cfg.training.resume_from=first['final_checkpoint']
    second=run_training(cfg)
    cfg.training.resume_from=None;cfg.training.checkpoint_dir=str(tmp_path/'full')
    full=run_training(cfg)
    a,b=[read_checkpoint(r['final_checkpoint']) for r in [second,full]]
    for name,p in a['model_state'].items():
        torch.testing.assert_close(p,b['model_state'][name],rtol=0,atol=0)
    cfg.training.decay_mode='conventional' if mode=='independent' else 'independent'
    with pytest.raises(ValueError,match='matching'):
        evaluate_checkpoint(cfg,full['final_checkpoint'],split='val')
