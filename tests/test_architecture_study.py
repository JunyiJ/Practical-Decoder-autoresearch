import copy
import pytest
import torch
from omegaconf import OmegaConf
from src.models.gpt import GPT
from src.train.optimizers import build_optimizer
from src.research.architecture_study import FAMILIES,LRS,SEEDS,local_jobs,freeze_selection,summarize,validate_plan


def plan():
    return {'families':FAMILIES,'learning_rates':LRS,'seeds':SEEDS,'widths':[64,128,256],
            'min_improvement':.005,'scope':'development'}


def records():
    return [dict(family=f,lr=lr,width=w,seed=s,status='completed',
                 metric=2+abs(lr-.002)*100-(.02 if f=='silu' else 0),
                 training_time_sec=3,rss_mb=100) for f,lr,w,s in local_jobs(plan())]


def test_equal_tuning_budgets_and_fixed_control():
    jobs=local_jobs(plan()); assert len(jobs)==21==len(set(jobs))
    assert {lr for f,lr,w,s in jobs if f=='control'}=={.002}
    for f in ['gelu_tuned','silu']:
        assert len([x for x in jobs if x[0]==f])==9
    assert {s for f,lr,w,s in jobs}=={44,55,66}


def test_selection_requires_both_controls_and_all_seeds():
    p=plan();rs=records();assert freeze_selection(p,rs)['winner']=='silu'
    # A better tuned GELU prevents attributing an LR-screen advantage to activation.
    for r in rs:
        if r['family']=='gelu_tuned' and r['lr']==.002:r['metric']=1.97
    assert freeze_selection(p,rs)['winner']=='control'
    rs=records()
    for r in rs:
        if r['family']=='silu' and r['seed']==44:r['metric']+=.1
    assert freeze_selection(p,rs)['winner']=='control'


@pytest.mark.parametrize('corrupt',['missing','duplicate','transfer','failed'])
def test_selection_rejects_invalid_evidence(corrupt):
    rs=records()
    if corrupt=='missing':rs.pop()
    if corrupt=='duplicate':rs.append(rs[0])
    if corrupt=='transfer':rs[0]['width']=128
    if corrupt=='failed':rs[0]['status']='failed'
    with pytest.raises(ValueError):freeze_selection(plan(),rs)


def test_transfer_cannot_change_frozen_winner():
    p=plan();rs=records()
    for r in rs:
        if r['family']=='silu':r['metric']+=.1
    selected=freeze_selection(p,rs);assert selected['winner']=='control'
    for w in [128,256]:
        for f in FAMILIES:
            for seed in SEEDS:
                rs.append(dict(family=f,lr=selected['choices'][f]['lr'],width=w,seed=seed,
                               status='completed',metric=1 if f=='silu' else 2,
                               training_time_sec=3,rss_mb=100))
    report=summarize(p,dict(records=rs,selection=selected))
    assert report['selection']==selected
    assert report['families']['silu']['widths']['256']['vs_control']['supported']
    assert not report['holdout_consumed']


def model_cfg(width,ffn):
    return OmegaConf.create(dict(dim=width,n_heads=width//32,n_layers=2,vocab_size=65,
        block_size=128,dropout=0.,attn_dropout=0.,rope=1,norm_type='rmsnorm',
        attn_type='mha',mlp_type='dense',ffn=ffn,parameterization='mup',base_width=64))


@pytest.mark.parametrize('width',[64,128,256])
def test_silu_same_initial_parameters_and_optimizer_contract(width):
    torch.set_num_threads(1)
    models=[];opts=[]
    train=OmegaConf.create(dict(optimizer='muon',learning_rate=.002,
                               weight_decay=.00006,decay_mode='independent'))
    for activation in ['gelu','silu']:
        torch.manual_seed(44);model=GPT(model_cfg(width,activation));models.append(model)
        opts.append(build_optimizer(model,train))
    assert dict(models[0].named_parameters()).keys()==dict(models[1].named_parameters()).keys()
    for a,b in zip(models[0].parameters(),models[1].parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    for a,b in zip(opts[0].param_groups,opts[1].param_groups):
        assert a['parameter_names']==b['parameter_names'] and a['lr']==b['lr']
    x=torch.tensor([[0,1,2,3,4,5,6,7]])
    out,loss=models[1](x,x);assert torch.isfinite(out).all()
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in models[1].parameters() if p.grad is not None)
    opts[1].step()
    # Causal logits must not depend on future tokens.
    y=x.clone();y[:,4:]=9
    torch.testing.assert_close(models[1](x)[0][:,:4],models[1](y)[0][:,:4])


def test_unsupported_gated_ffn_still_rejected():
    cfg=model_cfg(64,'swiglu');model=GPT(cfg)
    with pytest.raises(ValueError,match='GELU/SiLU'):
        build_optimizer(model,OmegaConf.create(dict(optimizer='muon',learning_rate=.002,
                        weight_decay=.00006,decay_mode='independent')))


def full_plan():
    p=plan();p.update(study='stage2a-activation-v1',eval_seed=2026,max_trials=39,
        trial_seconds=600,session_seconds=14400,max_disk_mb=4096)
    p['config']={'model':OmegaConf.to_container(model_cfg(64,'gelu')),
        'training':dict(optimizer='muon',decay_mode='independent',learning_rate=.002,
            weight_decay=.00006,grad_clip=1.,batch_size=8,max_iters=3000,eval_every=300,
            eval_iters=30,cpu_threads=4,device='mps')}
    return p


@pytest.mark.parametrize('field,value',[('max_trials',40),('seeds',[11,22,33]),
    ('learning_rates',[.001,.002,.003]),('widths',[64,256,512])])
def test_frozen_protocol_rejects_expansion(field,value):
    p=full_plan();assert validate_plan(p)==p;p[field]=value
    with pytest.raises(ValueError):validate_plan(p)


@pytest.mark.parametrize('field,value',[('ffn','swiglu'),('n_layers',4),('hidden_dim',128),
    ('attn_type','gqa'),('dropout',.01)])
def test_control_cannot_silently_change(field,value):
    p=full_plan();p['config']['model'][field]=value
    with pytest.raises(ValueError):validate_plan(p)


def test_serial_engine_freezes_selection_before_transfer_and_never_retrains(tmp_path,monkeypatch):
    import json
    from src.research import optimizer_study as engine
    from src.research.runner import atomic_json
    p=plan();p.update(source_root=str(tmp_path),max_trials=39,session_seconds=14400,
                     trial_seconds=600,max_disk_mb=4096)
    runtime={'packages':{'torch':'test'}}
    monkeypatch.setattr(engine,'_preflight',lambda spec:None)
    monkeypatch.setattr(engine,'_fingerprint',lambda spec:{'runtime':runtime})
    monkeypatch.setattr(engine,'_verify_checkpoints',lambda *args:None)
    def request(spec,row,d):
        return {'job':row.copy(),'config':{'model':{'block_size':128},
                'training':{'device':'cpu'}},'cpu_threads':1}
    monkeypatch.setattr(engine,'_request',request)
    seen=[]
    def launch(path,output,*args,**kwargs):
        req=json.loads(path.read_text());job=req['job'];seen.append(job)
        if job['width']>64:
            frozen=json.loads((output/'selection.json').read_text())
            assert frozen['winner']=='silu' and len([r for r in seen if r['width']==64])==21
            assert job['lr']==frozen['choices'][job['family']]['lr']
        metric=2+abs(job['lr']-.002)*100-(.02 if job['family']=='silu' else 0)
        atomic_json(path.parent/'result.json',{'metric':metric,'training_time_sec':3,'rss_mb':100,
             'runtime':{'torch':'test','cpu_threads':1,'requested_device':'cpu'}})
        return 'completed',None
    monkeypatch.setattr(engine,'_launch',launch)
    monkeypatch.setattr(engine,'validate_result',lambda raw,*args:{k:raw[k] for k in ['metric','training_time_sec','rss_mb']})
    args=dict(suite_builder=lambda *a:{},job_builder=local_jobs,
              selection_builder=freeze_selection,report_builder=summarize)
    state=engine.run(p,tmp_path/'run',**args)
    assert state['status']=='completed' and len(seen)==39
    engine.run(p,tmp_path/'run',**args)
    assert len(seen)==39
