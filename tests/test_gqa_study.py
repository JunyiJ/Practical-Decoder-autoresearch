import copy
import math
import pytest
import torch
from src.models.gpt import GPT
from src.train.optimizers import build_optimizer
from src.research import gqa_study as study
from tests.test_gated_study import config, training


def gqa_config(w):
    c=config(w,'swiglu');c.attn_type='gqa';c.attention_width_policy='gqa_half_head32'
    return c


@pytest.mark.parametrize('w',[64,128,256])
def test_gqa_shapes_scaling_causality_and_checkpoint(w):
    torch.set_num_threads(1);torch.manual_seed(444)
    model=GPT(gqa_config(w));mha=GPT(config(w,'swiglu'));opt=build_optimizer(model,training())
    att=model.transformer.h[0].attn
    assert att.n_heads==w//32 and att.n_kv_heads==w//64 and att.group_factor==2
    assert att.qkv.weight.shape==(2*w,w)
    assert sum(p.numel() for p in mha.parameters())-sum(p.numel() for p in model.parameters())==2*(w*w+w)
    for group in opt.param_groups:
        if '.attn.qkv.weight' in group['parameter_names'][0]:
            assert group['lr']==pytest.approx(.002/math.sqrt(w/64))
            assert group['lr']*.2*math.sqrt(max(group['params'][0].shape))==pytest.approx(.002*.2*math.sqrt(128))
    x=torch.arange(8).reshape(1,8);out,loss=model(x,x);loss.backward()
    assert torch.isfinite(out).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert att.qkv.weight.grad.abs().sum()>0
    opt.step();opt.zero_grad()
    clone=GPT(gqa_config(w));clone.load_state_dict(copy.deepcopy(model.state_dict()))
    other=build_optimizer(clone,training());other.load_state_dict(copy.deepcopy(opt.state_dict()))
    for m,o in [(model,opt),(clone,other)]:m(x,x)[1].backward();o.step()
    for p,q in zip(model.parameters(),clone.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    y=x.clone();y[:,4:]=9;torch.testing.assert_close(model(x)[0][:,:4],model(y)[0][:,:4])


@pytest.mark.parametrize('field,value',[('n_kv_heads',1),('n_groups',1),('n_heads',4),('n_layers',3),('attention_width_policy','mha_head32'),('attn_type','mha'),('ffn','gelu')])
def test_gqa_contract_rejects_unregistered_changes(field,value):
    c=gqa_config(64);c[field]=value
    with pytest.raises(ValueError):build_optimizer(GPT(c),training())


def plan():
    return dict(families=study.FAMILIES,learning_rates=study.LRS,seeds=study.SEEDS,widths=[64,128,256],head_dimensions=study.HEAD_DIMS,min_improvement=.005,scope='development')


def test_selection_requires_all_comparators_and_every_seed():
    p=plan();rs=[dict(family=f,lr=lr,width=w,seed=s,status='completed',metric=2+abs(lr-.0025)*100-(.02 if f=='gqa_half' else 0)) for f,lr,w,s in study.local_jobs(p)]
    assert len(rs)==24 and study.freeze_selection(p,rs)['winner']=='gqa_half'
    for r in rs:
        if r['family']=='gqa_half' and r['seed']==666:r['metric']+=.021
    assert study.freeze_selection(p,rs)['winner']=='incumbent'


@pytest.mark.parametrize('field,value',[('max_trials',49),('head_dimensions',{'gqa_half':16}),('seeds',[111,222,333])])
def test_frozen_protocol(field,value):
    from tests.test_architecture_study import full_plan
    p=full_plan();p.update(study='stage2d-gqa-v1',families=study.FAMILIES,seeds=study.SEEDS,head_dimensions=study.HEAD_DIMS,max_trials=48)
    study.validate_plan(p);p[field]=value
    with pytest.raises(ValueError):study.validate_plan(p)


def test_serial_engine_propagates_gqa_to_all_widths_before_frozen_transfer(tmp_path,monkeypatch):
    import json
    from omegaconf import OmegaConf
    from tests.test_architecture_study import full_plan
    from src.research import optimizer_study as engine
    from src.research.runner import atomic_json
    p=full_plan();p.update(study='stage2d-gqa-v1',families=study.FAMILIES,seeds=study.SEEDS,
                          head_dimensions=study.HEAD_DIMS,max_trials=48,source_root=str(tmp_path),source_sha256='a'*64,data_sha256=None)
    p['config']['data']=dict(path=str(tmp_path/'corpus.txt'),train_fraction=.8,val_fraction=.1)
    runtime={'packages':{'torch':'test'}}
    monkeypatch.setattr(engine,'_preflight',lambda spec:None)
    monkeypatch.setattr(engine,'_fingerprint',lambda spec:{'runtime':runtime})
    monkeypatch.setattr(engine,'_verify_checkpoints',lambda *args:None)
    original_request=engine._request;seen=[]
    def request(spec,row,d):
        req=original_request(spec,row,d);req['job']=row.copy();return req
    monkeypatch.setattr(engine,'_request',request)
    def launch(path,output,*args,**kwargs):
        req=json.loads(path.read_text());job=req['job'];seen.append(job)
        cfg=OmegaConf.create(req['config']['model']);cfg.vocab_size=65
        model=GPT(cfg);att=model.transformer.h[0].attn
        assert att.n_heads==job['width']//32
        if job['family']=='gqa_half':assert att.n_kv_heads==job['width']//64
        else:assert cfg.attn_type=='mha'
        if job['width']>64:
            frozen=json.loads((output/'selection.json').read_text())
            assert frozen['winner']=='gqa_half' and len([r for r in seen if r['width']==64])==24
            assert job['lr']==frozen['choices'][job['family']]['lr']
        metric=2+abs(job['lr']-.0025)*100-(.02 if job['family']=='gqa_half' else 0)
        atomic_json(path.parent/'result.json',{'metric':metric,'training_time_sec':3,'rss_mb':100,
             'runtime':{'torch':'test','cpu_threads':req['cpu_threads'],'requested_device':'mps'}})
        return 'completed',None
    monkeypatch.setattr(engine,'_launch',launch)
    monkeypatch.setattr(engine,'validate_result',lambda raw,*args:{k:raw[k] for k in ['metric','training_time_sec','rss_mb']})
    args=dict(suite_builder=study.suite_spec,job_builder=study.local_jobs,
              selection_builder=study.freeze_selection,report_builder=study.summarize)
    state=engine.run(p,tmp_path/'run',**args)
    assert state['status']=='completed' and len(seen)==48
    engine.run(p,tmp_path/'run',**args);assert len(seen)==48
