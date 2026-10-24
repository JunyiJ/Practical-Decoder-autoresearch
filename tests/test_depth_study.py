import copy
import math
import pytest
import torch
from src.models.gpt import GPT
from src.train.optimizers import build_optimizer
from src.research import depth_study as study
from tests.test_gated_study import config,training


def deep_config(w):
    c=config(w,'swiglu');c.n_layers=4;c.depth_width_policy='depth4_unscaled';return c


@pytest.mark.parametrize('w',[64,128,256])
def test_depth_initialization_counts_groups_gradients_and_checkpoint(w):
    torch.set_num_threads(1);torch.manual_seed(777);shallow=GPT(config(w,'swiglu'))
    torch.manual_seed(777);model=GPT(deep_config(w));opt=build_optimizer(model,training())
    assert len(model.transformer.h)==4
    assert sum(p.numel() for p in model.parameters())-sum(p.numel() for p in shallow.parameters())==2*(95*w*w//8+6*w)
    # Extra layers are fresh, and the first two retain the existing initialization.
    for p,q in zip(model.transformer.h[:2].parameters(),shallow.transformer.h.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert not torch.equal(model.lm_head.weight,shallow.lm_head.weight)
    names=[]
    for group in opt.param_groups:
        n=group['parameter_names'][0];names.append(n)
        expected=.002/math.sqrt(w/64) if group['params'][0].ndim==2 and n.startswith('transformer.h.') else .002
        assert group['lr']==pytest.approx(expected)
    assert any('h.3.mlp.linear_gate.weight' in n for n in names)
    x=torch.arange(8).reshape(1,8);out,loss=model(x,x);loss.backward()
    assert torch.isfinite(out).all()
    for block in model.transformer.h:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
        assert block.attn.qkv.weight.grad.abs().sum()>0
    opt.step();opt.zero_grad()
    clone=GPT(deep_config(w));clone.load_state_dict(copy.deepcopy(model.state_dict()));other=build_optimizer(clone,training());other.load_state_dict(copy.deepcopy(opt.state_dict()))
    for m,o in [(model,opt),(clone,other)]:m(x,x)[1].backward();o.step()
    for p,q in zip(model.parameters(),clone.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    y=x.clone();y[:,4:]=9;torch.testing.assert_close(model(x)[0][:,:4],model(y)[0][:,:4])


@pytest.mark.parametrize('field,value',[('depth_width_policy','unknown'),('n_layers',3),('n_layers',2),('n_heads',4),('attn_type','gqa'),('attention_width_policy','mha_head16'),('n_kv_heads',1)])
def test_depth_contract_rejects_unsupported_combination(field,value):
    c=deep_config(64);c[field]=value
    with pytest.raises(ValueError):build_optimizer(GPT(c),training())


def plan():
    from tests.test_architecture_study import full_plan
    p=full_plan();p.update(study='stage2e-depth-v1',families=study.FAMILIES,seeds=study.SEEDS,head_dimensions=study.HEAD_DIMS,depths=study.DEPTHS,max_trials=48);return p


@pytest.mark.parametrize('field,value',[('max_trials',49),('depths',{'depth4':3}),('seeds',[444,555,666])])
def test_frozen_protocol(field,value):
    p=plan();study.validate_plan(p);p[field]=value
    with pytest.raises(ValueError):study.validate_plan(p)


def test_selection_rejects_single_seed_regression():
    p=plan();rs=[dict(family=f,lr=lr,width=w,seed=s,status='completed',metric=2+abs(lr-.0025)*100-(.02 if f=='depth4' else 0)) for f,lr,w,s in study.local_jobs(p)]
    assert len(rs)==24 and study.freeze_selection(p,rs)['winner']=='depth4'
    for r in rs:
        if r['family']=='depth4' and r['seed']==999:r['metric']+=.021
    assert study.freeze_selection(p,rs)['winner']=='incumbent'


def test_serial_engine_propagates_depth_to_all_widths_before_frozen_transfer(tmp_path,monkeypatch):
    import json
    from omegaconf import OmegaConf
    from tests.test_architecture_study import full_plan
    from src.research import optimizer_study as engine
    from src.research.runner import atomic_json
    p=plan();p.update(source_root=str(tmp_path),source_sha256='a'*64,data_sha256=None)
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
        assert len(model.transformer.h)==study.DEPTHS[job['family']]
        assert cfg.attn_type=='mha'
        if job['width']>64:
            frozen=json.loads((output/'selection.json').read_text())
            assert frozen['winner']=='depth4' and len([r for r in seen if r['width']==64])==24
            assert job['lr']==frozen['choices'][job['family']]['lr']
        metric=2+abs(job['lr']-.0025)*100-(.02 if job['family']=='depth4' else 0)
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
