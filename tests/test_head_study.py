import copy
import pytest
import torch
from src.models.gpt import GPT
from src.train.optimizers import build_optimizer
from src.research import head_study as study
from tests.test_gated_study import config,training


def plan():
    return dict(families=study.FAMILIES,learning_rates=study.LRS,seeds=study.SEEDS,widths=[64,128,256],head_dimensions=study.HEAD_DIMS,min_improvement=.005,scope='development')


@pytest.mark.parametrize('w',[64,128,256])
def test_head16_identical_parameters_groups_and_causal_finite_training(w):
    torch.set_num_threads(1);models=[];opts=[]
    for hd in [32,16]:
        c=config(w,'swiglu');c.n_heads=w//hd
        if hd==16:c.attention_width_policy='mha_head16'
        torch.manual_seed(111);m=GPT(c);models.append(m);opts.append(build_optimizer(m,training()))
    assert sum(p.numel() for p in models[0].parameters())==sum(p.numel() for p in models[1].parameters())
    for (n,p),(m,q) in zip(models[0].named_parameters(),models[1].named_parameters()):
        assert n==m;torch.testing.assert_close(p,q,rtol=0,atol=0)
    for a,b in zip(opts[0].param_groups,opts[1].param_groups):assert a['parameter_names']==b['parameter_names'] and a['lr']==b['lr']
    x=torch.arange(8).reshape(1,8);out,loss=models[1](x,x);loss.backward();assert torch.isfinite(out).all()
    assert all(torch.isfinite(p.grad).all() for p in models[1].parameters() if p.grad is not None);opts[1].step()
    y=x.clone();y[:,4:]=9;torch.testing.assert_close(models[1](x)[0][:,:4],models[1](y)[0][:,:4])


@pytest.mark.parametrize('heads,policy',[(8,'mha_head16'),(4,'mha_head32'),(4,'unknown')])
def test_head_contract_rejects_unregistered_shape(heads,policy):
    c=config(64,'swiglu');c.n_heads=heads;c.attention_width_policy=policy
    with pytest.raises(ValueError):build_optimizer(GPT(c),training())


def test_selection_requires_all_three_comparators():
    p=plan();rs=[dict(family=f,lr=lr,width=w,seed=s,status='completed',metric=2+abs(lr-.0025)*100-(.02 if f=='head16' else 0)) for f,lr,w,s in study.local_jobs(p)]
    assert len(rs)==24;assert study.freeze_selection(p,rs)['winner']=='head16'
    for r in rs:
        if r['family']=='head32_tuned':r['metric']-=.05
    assert study.freeze_selection(p,rs)['winner']=='incumbent'




@pytest.mark.parametrize('field,value',[('max_trials',49),('head_dimensions',{'head16':32}),('seeds',[77,88,99])])
def test_plan_freezes_head_contract(field,value):
    from tests.test_architecture_study import full_plan
    p=full_plan();p.update(study='stage2c-heads-v1',families=study.FAMILIES,seeds=study.SEEDS,head_dimensions=study.HEAD_DIMS,max_trials=48)
    study.validate_plan(p);p[field]=value
    with pytest.raises(ValueError):study.validate_plan(p)


def test_serial_engine_freezes_selection_before_transfer_and_never_retrains(tmp_path,monkeypatch):
    import json
    from src.research import optimizer_study as engine
    from src.research.runner import atomic_json
    p=plan();p.update(source_root=str(tmp_path),max_trials=48,session_seconds=14400,
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
        assert job['n_heads']==job['width']//study.HEAD_DIMS[job['family']]
        if job['width']>64:
            frozen=json.loads((output/'selection.json').read_text())
            assert frozen['winner']=='head16' and len([r for r in seen if r['width']==64])==24
            assert job['lr']==frozen['choices'][job['family']]['lr']
        metric=2+abs(job['lr']-.0025)*100-(.02 if job['family']=='head16' else 0)
        atomic_json(path.parent/'result.json',{'metric':metric,'training_time_sec':3,'rss_mb':100,
             'runtime':{'torch':'test','cpu_threads':1,'requested_device':'cpu'}})
        return 'completed',None
    monkeypatch.setattr(engine,'_launch',launch)
    monkeypatch.setattr(engine,'validate_result',lambda raw,*args:{k:raw[k] for k in ['metric','training_time_sec','rss_mb']})
    args=dict(suite_builder=lambda *a:{},job_builder=study.local_jobs,
              selection_builder=study.freeze_selection,report_builder=study.summarize)
    state=engine.run(p,tmp_path/'run',**args)
    assert state['status']=='completed' and len(seen)==48
    engine.run(p,tmp_path/'run',**args)
    assert len(seen)==48
