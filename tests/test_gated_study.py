import copy
import math
import pytest
import torch
from omegaconf import OmegaConf
from src.models.gpt import GPT
from src.train.optimizers import build_optimizer
from src.research import gated_study as study
from tests.test_architecture_study import model_cfg


def config(w,ffn):
    c=model_cfg(w,ffn)
    c.ffn_bias=False
    if ffn=='swiglu':c.ffn_width_policy='swiglu_21_8'
    return c


def training():
    return OmegaConf.create(dict(optimizer='muon',learning_rate=.002,weight_decay=.00006,decay_mode='independent'))


@pytest.mark.parametrize('w',[64,128,256])
def test_gated_counts_gradients_scaling_and_checkpoint(w):
    torch.set_num_threads(1);torch.manual_seed(77)
    m=GPT(config(w,'swiglu'));dense=GPT(config(w,'silu'));opt=build_optimizer(m,training())
    h=21*w//8
    gate=m.transformer.h[0].mlp
    assert gate.hidden_dim==h and gate.linear_gate.bias is None
    assert sum(p.numel() for p in gate.parameters())==3*w*h
    assert sum(p.numel() for p in dense.transformer.h[0].mlp.parameters())==8*w*w
    assert (3*w*h)/(8*w*w)==63/64
    for group in opt.param_groups:
        name=group['parameter_names'][0]
        if '.mlp.' in name:
            assert group['lr']==pytest.approx(.002/math.sqrt(w/64))
            # RMS-matched Muon shape factor cancels width scaling within this family.
            assert group['lr']*.2*math.sqrt(max(group['params'][0].shape))==pytest.approx(.002*.2*math.sqrt(168))
    x=torch.arange(8).reshape(1,8);out,loss=m(x,x);loss.backward()
    assert torch.isfinite(out).all()
    for name,p in gate.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    opt.step();opt.zero_grad()
    clone=GPT(config(w,'swiglu'));clone.load_state_dict(copy.deepcopy(m.state_dict()));other=build_optimizer(clone,training());other.load_state_dict(copy.deepcopy(opt.state_dict()))
    for model,optimizer in [(m,opt),(clone,other)]:
        model(x,x)[1].backward();optimizer.step()
    for p,q in zip(m.parameters(),clone.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    y=x.clone();y[:,4:]=9
    torch.testing.assert_close(m(x)[0][:,:4],m(y)[0][:,:4])


@pytest.mark.parametrize('key,value',[('hidden_dim',168),('ffn_bias',True),('ffn_width_policy','anything'),('attn_type','gqa'),('n_layers',3),('n_heads',4)])
def test_gated_contract_rejects_unsupported_changes(key,value):
    c=config(64,'swiglu');c[key]=value
    with pytest.raises((ValueError,AttributeError)):build_optimizer(GPT(c),training())


def test_round_selection_and_transfer_are_frozen():
    p=dict(families=study.FAMILIES,learning_rates=study.LRS,seeds=study.SEEDS,widths=[64,128,256],min_improvement=.005,scope='development')
    rows=[dict(family=f,lr=l,width=w,seed=s,status='completed',metric=2+abs(l-.002)*100-(.02 if f=='swiglu' else 0),training_time_sec=1,rss_mb=1) for f,l,w,s in study.local_jobs(p)]
    assert len(rows)==21 and {r['seed'] for r in rows}=={77,88,99}
    sel=study.freeze_selection(p,rows);assert sel['winner']=='swiglu'
    # Beating fixed control is insufficient if the structural comparator is better.
    for r in rows:
        if r['family']=='dense_silu':r['metric']-=.04
    sel=study.freeze_selection(p,rows);assert sel['winner']=='control'
    for w in [128,256]:
        for f in p['families']:
            for s in p['seeds']:rows.append(dict(family=f,lr=sel['choices'][f]['lr'],width=w,seed=s,status='completed',metric=1 if f=='swiglu' else 3,training_time_sec=1,rss_mb=1))
    assert study.summarize(p,dict(records=rows,selection=sel))['selection']==sel


def test_serial_engine_freezes_selection_before_transfer_and_never_retrains(tmp_path,monkeypatch):
    import json
    from src.research import optimizer_study as engine
    from src.research.runner import atomic_json
    p=dict(families=study.FAMILIES,learning_rates=study.LRS,seeds=study.SEEDS,widths=[64,128,256],min_improvement=.005,scope="development");p.update(source_root=str(tmp_path),max_trials=39,session_seconds=14400,
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
            assert frozen['winner']=='swiglu' and len([r for r in seen if r['width']==64])==21
            assert job['lr']==frozen['choices'][job['family']]['lr']
        metric=2+abs(job['lr']-.002)*100-(.02 if job['family']=='swiglu' else 0)
        atomic_json(path.parent/'result.json',{'metric':metric,'training_time_sec':3,'rss_mb':100,
             'runtime':{'torch':'test','cpu_threads':1,'requested_device':'cpu'}})
        return 'completed',None
    monkeypatch.setattr(engine,'_launch',launch)
    monkeypatch.setattr(engine,'validate_result',lambda raw,*args:{k:raw[k] for k in ['metric','training_time_sec','rss_mb']})
    args=dict(suite_builder=lambda *a:{},job_builder=study.local_jobs,
              selection_builder=study.freeze_selection,report_builder=study.summarize)
    state=engine.run(p,tmp_path/'run',**args)
    assert state['status']=='completed' and len(seen)==39
    engine.run(p,tmp_path/'run',**args)
    assert len(seen)==39


@pytest.mark.parametrize('key,value',[('max_trials',40),('seeds',[44,55,66]),('learning_rates',[.001,.002,.003]),('widths',[64,128,512])])
def test_gated_plan_rejects_protocol_drift(key,value):
    from tests.test_architecture_study import full_plan
    p=full_plan();p.update(study='stage2b-gated-v1',families=study.FAMILIES,seeds=study.SEEDS)
    study.validate_plan(p);p[key]=value
    with pytest.raises(ValueError):study.validate_plan(p)
