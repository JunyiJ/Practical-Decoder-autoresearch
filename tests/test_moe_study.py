import copy
import json
import math
from pathlib import Path
import pytest
from src.research import moe_study as study
from src.research import optimizer_study as engine
from src.research.runner import atomic_json
from src.research.team.suite import RUNTIME_TRAINING


def plan():
    from tests.test_architecture_study import full_plan
    p=full_plan()
    p.update(source_root='/tmp/context-test-root',source_sha256='a'*64,data_sha256=None)
    p['config']['data']=dict(path='/tmp/context-test-corpus',train_fraction=.8,val_fraction=.1)
    for key in RUNTIME_TRAINING:p['config']['training'].pop(key,None)
    p.update(study='moe-quality-v1',families=list(study.CELLS),learning_rates=study.LRS,
        seeds=study.SCREEN_SEEDS,confirmation_seeds=study.CONFIRM_SEEDS,widths=study.WIDTHS,
        max_trials=54,trial_seconds=1800,session_seconds=28800,max_disk_mb=4096,
        primary_metric='full_validation_context128',quality_margin=.01,
        bootstrap_blocks=[2048,4096,8192],bootstrap_draws=2000,bootstrap_seed=20261024,
        evaluation_targets=111538,total_training_targets=3072000,
        batch_layout='paired_blocks512',moe_policy='moe4_top2_half_readout',max_process_rss_mb=4096,max_mps_driver_mb=6144)
    p['screen_jobs']=[list(j) for j in study.local_jobs(p)]
    return study.validate_plan(p)


def screen_rows(p, lr=.0025):
    return [dict(family=f,lr=l,width=w,seed=s,status='completed',metric=2+abs(l-lr)*100)
            for f,l,w,s in study.local_jobs(p)]


def coverage(metric,context=128):
    n=111538;counts=[min(2048,n-i) for i in range(0,n,2048)]
    return dict(context=context,eval_batch_size=1024//context,block_tokens=2048,target_count=n,
        window_count=math.ceil(n/context),mean_ce=metric,block_target_counts=counts,
        block_loss_sums=[metric*x for x in counts])


def test_manifest_fresh_seeds_and_alias_bounds():
    p=plan();assert len(study.local_jobs(p))==18
    for lr,total in [(.00125,54),(.0025,48),(.005,54)]:
        s=study.freeze_selection(p,screen_rows(p,lr));m=s['confirmation']
        assert m['total_distinct_trials']==total
        assert not set(study.SCREEN_SEEDS)&set(j[3] for j in m['jobs'])
        assert {j[2] for j in m['jobs']}=={64,256}
        assert m['reference_aliases']==(study.REFERENCE_CELLS if lr==.0025 else {})
    s['choices']['dense']['lr']=.0025
    assert study.confirmation_manifest(s['choices'])['total_distinct_trials']==48


def test_selection_rejects_confirmation_and_incomplete_or_duplicate_rows():
    p=plan();rs=screen_rows(p)
    for bad in [rs[:-1],rs+[rs[0]],rs+[{**rs[0],'seed':2091}],rs+[{**rs[0],'width':256}]]:
        with pytest.raises(ValueError):study.freeze_selection(p,bad)
    rs[0]['metric']=float('nan')
    with pytest.raises(ValueError):study.freeze_selection(p,rs)


@pytest.mark.parametrize('key,value',[('confirmation_seeds',[2081,2092,2093]),('quality_margin',.02),
    ('widths',[64,128,256]),('max_trials',55),('trial_seconds',1801),('evaluation_targets',111537)])
def test_protocol_guard(key,value):
    p=plan();p[key]=value
    with pytest.raises(ValueError):study.validate_plan(p)


def test_full_and_native_coverage_contracts():
    p=plan();req={'config':{'model':{'block_size':512}}}
    raw=dict(runtime={'torch':'x'},coverage=dict(device='mps',torch='x',holdout_consumed=False,
        evaluations={str(c):coverage(2,c) for c in [128,512]}))
    study.validate_coverage(raw,req,p)
    for field,value in [('window_count',1),('eval_batch_size',8),('target_count',111537),('mean_ce',2.1)]:
        bad=copy.deepcopy(raw);bad['coverage']['evaluations']['512'][field]=value
        with pytest.raises(ValueError):study.validate_coverage(bad,req,p)


def test_serial_engine_freezes_before_fresh_confirmation_and_no_retraining(tmp_path,monkeypatch):
    p=plan();p.update(source_root=str(tmp_path),source_sha256='a'*64,data_sha256=None)
    p['config']['data']['path']=str(tmp_path/'text')
    monkeypatch.setattr(engine,'_preflight',lambda spec:None)
    monkeypatch.setattr(engine,'_fingerprint',lambda spec:{'runtime':{'packages':{'torch':'test'}}})
    monkeypatch.setattr(engine,'_verify_checkpoints',lambda *args:None)
    monkeypatch.setattr(engine,'validate_result',lambda raw,*args:dict(metric=raw['metric'],training_time_sec=3,rss_mb=100))
    seen=[]
    def launch(path,output,*args,**kwargs):
        req=json.loads(path.read_text());cfg=req['config'];t=cfg['training'];m=cfg['model'];seen.append(req)
        assert t['batch_size']*m['block_size']*t['max_iters']==3072000 and t['batch_layout']=='paired_blocks512'
        if t['seed'] in study.CONFIRM_SEEDS:
            selection=json.loads((output/'selection.json').read_text())
            assert selection['confirmation']['total_distinct_trials']==48
            assert len([r for r in seen if r['config']['training']['seed'] in study.SCREEN_SEEDS])==18
            assert req['stage']=='confirmation'
        else:assert req['stage']=='local'
        score=2+abs(t['learning_rate']-.0025)*100
        # One confirmation seed fails the human ceiling even though the mean passes.
        if m['mlp_type']=='moe' and t['seed']==2093:score+=.02
        atomic_json(path.parent/'result.json',dict(metric=score,
            coverage={str(c):coverage(score,c) for c in set([128,m['block_size']])},
            runtime=dict(torch='test',cpu_threads=4,requested_device='mps')))
        return 'completed',None
    kwargs=dict(suite_builder=study.suite_spec,job_builder=study.local_jobs,
        selection_builder=study.freeze_selection,report_builder=study.summarize,launcher=launch,
        result_adapter=lambda raw,*args:dict(coverage=raw['coverage']),
        completion_jobs=lambda p,s:s['confirmation']['jobs'])
    state=engine.run(p,tmp_path/'run',**kwargs)
    assert state['status']=='completed' and len(seen)==48
    report=json.loads((tmp_path/'run/report.json').read_text())
    assert all(not x['near_neutral_common128'] for x in report['comparisons'] if x['contrast']=='tuned_architecture_equal_tokens' and x['candidate']=='moe4_top2' and x['reference']=='dense')
    engine.run(p,tmp_path/'run',**kwargs);assert len(seen)==48



def config(f,width=64):
    from omegaconf import OmegaConf
    c=OmegaConf.create(study.suite_spec(plan(),f,study.FIXED.get(f,.0025))['candidate']['config'])
    c.model.update(dim=width,n_heads=width//32,vocab_size=65)
    return c


def test_moe_sparse_dispatch_matches_dense_oracle_values_and_gradients():
    import torch
    from src.models.gpt import GPT
    torch.set_num_threads(1);torch.manual_seed(42)
    m=GPT(config('moe4_top2').model).transformer.h[0].mlp.double()
    oracle=copy.deepcopy(m)
    x=torch.randn(2,8,64,dtype=torch.double,requires_grad=True);xx=x.detach().clone().requires_grad_()
    y,aux=m(x)
    flat=xx.reshape(-1,64);probs=(oracle.router(flat)/oracle.router_width_mult).softmax(-1)
    values,indices=probs.topk(2,dim=-1);weights=values/values.sum(-1,keepdim=True)
    dense=torch.stack([e(flat) for e in oracle.experts],dim=1)
    expected=(dense.gather(1,indices[:,:,None].expand(-1,-1,64))*weights[:,:,None]).sum(1).view_as(y)
    f=torch.nn.functional.one_hot(indices,4).double().mean((0,1))
    expected_aux=4*(f*probs.mean(0)).sum()
    torch.testing.assert_close(y,expected);torch.testing.assert_close(aux,expected_aux)
    (y.square().mean()+.01*aux).backward();(expected.square().mean()+.01*expected_aux).backward()
    torch.testing.assert_close(x.grad,xx.grad)
    for (n,a),(nn,b) in zip(m.named_parameters(),oracle.named_parameters()):
        assert n==nn;torch.testing.assert_close(a.grad,b.grad,rtol=1e-6,atol=1e-10)
    assert m.router.weight.grad.abs().sum()>0
    assert int(m.last_expert_hist.sum())==32


@pytest.mark.parametrize('width',[64,128,256])
def test_moe_shapes_active_total_counts_router_scaling_and_optimizer(width):
    import torch
    from src.models.gpt import GPT
    from src.train.optimizers import build_optimizer
    torch.set_num_threads(1)
    dense=GPT(config('dense',width).model);wide=GPT(config('dense_wide',width).model)
    cfg=config('moe4_top2',width);moe=GPT(cfg.model);opt=build_optimizer(moe,cfg.training)
    dc=sum(p.numel() for p in dense.transformer.h[0].mlp.parameters())
    wc=sum(p.numel() for p in wide.transformer.h[0].mlp.parameters())
    for b in moe.transformer.h:
        e=b.mlp;ec=sum(p.numel() for p in e.experts[0].parameters())
        assert e.hidden_dim==21*width//16 and ec*2==dc and ec*4==wc
        assert tuple(e.router.weight.shape)==(4,width) and e.router_width_mult==width/64
    grouped=[n for o in opt.optimizers for g in o.param_groups for n in g['parameter_names']]
    assert len(grouped)==len(set(grouped))==len(list(moe.parameters()))
    for o in opt.optimizers:
        for g in o.param_groups:
            n=g['parameter_names'][0]
            if n.endswith('.router.weight'):
                assert isinstance(o,torch.optim.AdamW) and g['width_lr_scale']==1
            elif isinstance(o,torch.optim.Muon):
                assert math.isclose(g['width_lr_scale'],1/math.sqrt(width/64))
    x=torch.arange(32).reshape(2,16)%65
    logits,loss,parts=moe(x,x,return_loss_breakdown=True)
    torch.testing.assert_close(loss,parts['ce_loss']+.01*parts['aux_loss'])
    ce=moe(x,x,include_aux_loss=False)[1]
    torch.testing.assert_close(ce,torch.nn.functional.cross_entropy(logits.reshape(-1,65),x.flatten()))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in moe.parameters())
    opt.step(); assert all(torch.isfinite(p).all() for p in moe.parameters())


@pytest.mark.parametrize('key,value',[('top_k',1),('num_experts',8),('shared_num_experts',1),('aux_loss_weight',float('nan')),('hidden_dim',84)])
def test_moe_rejects_unreviewed_contract(key,value):
    from src.models.gpt import GPT
    c=config('moe4_top2');c.model.moe[key]=value
    with pytest.raises(ValueError):GPT(c.model)


def test_moe_resume_matches_uninterrupted_cpu_and_causality():
    import torch
    from src.models.gpt import GPT
    from src.train.optimizers import build_optimizer
    torch.set_num_threads(1);torch.manual_seed(8);c=config('moe4_top2')
    m=GPT(c.model);o=build_optimizer(m,c.training);x=torch.randint(65,(2,16));y=torch.roll(x,-1,1)
    def step(m,o):
        o.zero_grad();m(x,y)[1].backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);o.step()
    step(m,o);state=copy.deepcopy(m.state_dict());os=copy.deepcopy(o.state_dict())
    resumed=GPT(c.model);resumed.load_state_dict(state,strict=True);oo=build_optimizer(resumed,c.training);oo.load_state_dict(os)
    step(m,o);step(resumed,oo)
    for a,b in zip(m.parameters(),resumed.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    changed=x.clone();changed[:,8:]=(changed[:,8:]+1)%65
    torch.testing.assert_close(m(x)[0][:,:8],m(changed)[0][:,:8],rtol=0,atol=0)


def test_architecture_choice_frozen_from_screens():
    p=plan();rs=screen_rows(p)
    for r in rs:
        if r['family']=='moe4_top2':r['metric']-=.02
    s=study.freeze_selection(p,rs);assert s['winner']=='moe4_top2'
    bad=copy.deepcopy(rs)
    for r in bad:
        if r['family']=='moe4_top2' and r['seed']==2082:r['metric']+=.04
    assert study.freeze_selection(p,bad)['winner']=='historical_incumbent_unchanged'


def test_memory_guard_thresholds_and_invalid_measurement():
    from src.research.memory_guard import exceeded
    assert not exceeded(4096*1024**2-1,6144*1024**2-1)
    assert exceeded(4096*1024**2,0)
    assert exceeded(0,6144*1024**2)
    with pytest.raises(ValueError):exceeded(float('nan'),0)
