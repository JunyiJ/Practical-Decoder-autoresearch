import copy
import json
import math
from pathlib import Path
import pytest
from src.research import context_interaction as study
from src.research import optimizer_study as engine
from src.research.runner import atomic_json
from src.research.team.suite import RUNTIME_TRAINING


def plan():
    from tests.test_architecture_study import full_plan
    p=full_plan()
    p.update(source_root='/tmp/context-test-root',source_sha256='a'*64,data_sha256=None)
    p['config']['data']=dict(path='/tmp/context-test-corpus',train_fraction=.8,val_fraction=.1)
    for key in RUNTIME_TRAINING:p['config']['training'].pop(key,None)
    p.update(study='context-interaction-v1',families=list(study.CELLS),learning_rates=study.LRS,
        seeds=study.SCREEN_SEEDS,confirmation_seeds=study.CONFIRM_SEEDS,widths=study.WIDTHS,
        max_trials=114,trial_seconds=1200,session_seconds=28800,max_disk_mb=4096,
        primary_metric='full_validation_context128',quality_margin=.01,
        bootstrap_blocks=[2048,4096,8192],bootstrap_draws=2000,bootstrap_seed=20261024,
        evaluation_targets=111538)
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
    p=plan();assert len(study.local_jobs(p))==48
    for lr,total in [(.00125,114),(.0025,102),(.005,114)]:
        s=study.freeze_selection(p,screen_rows(p,lr));m=s['confirmation']
        assert m['total_distinct_trials']==total
        assert not set(study.SCREEN_SEEDS)&set(j[3] for j in m['jobs'])
        assert {j[2] for j in m['jobs']}=={64,256}
        assert m['reference_aliases']==(study.REFERENCE_CELLS if lr==.0025 else {})
    s['choices']['d2_mha_c128']['lr']=.0025
    assert study.confirmation_manifest(s['choices'])['total_distinct_trials']==108


def test_selection_rejects_confirmation_and_incomplete_or_duplicate_rows():
    p=plan();rs=screen_rows(p)
    for bad in [rs[:-1],rs+[rs[0]],rs+[{**rs[0],'seed':2031}],rs+[{**rs[0],'width':256}]]:
        with pytest.raises(ValueError):study.freeze_selection(p,bad)
    rs[0]['metric']=float('nan')
    with pytest.raises(ValueError):study.freeze_selection(p,rs)


@pytest.mark.parametrize('key,value',[('confirmation_seeds',[2021,2032,2033]),('quality_margin',.02),
    ('widths',[64,128,256]),('max_trials',115),('trial_seconds',1800),('evaluation_targets',111537)])
def test_protocol_guard(key,value):
    p=plan();p[key]=value
    with pytest.raises(ValueError):study.validate_plan(p)


def test_factorial_contrasts_recover_known_three_way_effect():
    rows=[]
    for f,c in study.CELLS.items():
        d=int(c['depth']==4);a=int(c['attention']=='gqa');t=int(c['context']==512)
        metric=2+.1*d+.2*a+.3*t+.04*d*a+.05*a*t+.06*d*t+.07*d*a*t
        rows.extend(dict(family=f,width=64,seed=s,metric=metric) for s in study.CONFIRM_SEEDS)
    r=study.interaction_values(rows,64,study.CONFIRM_SEEDS)['mean_contrasts']
    assert r['depth_x_attention_c128']==pytest.approx(.04)
    assert r['depth_x_attention_c512']==pytest.approx(.11)
    assert r['depth_x_attention_x_context']==pytest.approx(.07)


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
        assert t['batch_size']*m['block_size']==1024 and t['batch_layout']=='paired_blocks512'
        if t['seed'] in study.CONFIRM_SEEDS:
            selection=json.loads((output/'selection.json').read_text())
            assert selection['confirmation']['total_distinct_trials']==102
            assert len([r for r in seen if r['config']['training']['seed'] in study.SCREEN_SEEDS])==48
            assert req['stage']=='confirmation'
        else:assert req['stage']=='local'
        score=2+abs(t['learning_rate']-.0025)*100
        # One confirmation seed fails the human ceiling even though the mean passes.
        if m['attn_type']=='gqa' and t['seed']==2033:score+=.02
        atomic_json(path.parent/'result.json',dict(metric=score,
            coverage={str(c):coverage(score,c) for c in set([128,m['block_size']])},
            runtime=dict(torch='test',cpu_threads=4,requested_device='mps')))
        return 'completed',None
    kwargs=dict(suite_builder=study.suite_spec,job_builder=study.local_jobs,
        selection_builder=study.freeze_selection,report_builder=study.summarize,launcher=launch,
        result_adapter=lambda raw,*args:dict(coverage=raw['coverage']),
        completion_jobs=lambda p,s:s['confirmation']['jobs'])
    state=engine.run(p,tmp_path/'run',**kwargs)
    assert state['status']=='completed' and len(seen)==102
    report=json.loads((tmp_path/'run/report.json').read_text())
    assert all(not x['quality_eligible'] for x in report['comparisons'] if x['contrast']=='gqa_minus_mha')
    engine.run(p,tmp_path/'run',**kwargs);assert len(seen)==102


def test_every_cell_and_reference_builds_both_widths_with_exact_shapes():
    import torch
    from omegaconf import OmegaConf
    from src.models.gpt import GPT
    from src.train.optimizers import build_optimizer
    torch.set_num_threads(1);p=plan()
    for f in list(study.CELLS)+list(study.FIXED):
        spec=study.suite_spec(p,f,study.FIXED.get(f,.0025))
        for w in study.WIDTHS:
            cfg=OmegaConf.create(spec['candidate']['config']);cfg.model.update(dim=w,n_heads=w//32,vocab_size=65)
            model=GPT(cfg.model);opt=build_optimizer(model,cfg.training);s=study.settings(f)
            assert len(model.transformer.h)==s['depth']
            assert cfg.training.batch_size*cfg.model.block_size==1024 and opt.decay==.00006
            for b in model.transformer.h:
                assert b.attn.qkv.weight.shape==((2 if s['attention']=='gqa' else 3)*w,w)
                if s['attention']=='gqa':assert b.attn.n_kv_heads==w//64


def test_first_failed_worker_stops_without_automatic_retry(tmp_path,monkeypatch):
    p=plan();p.update(source_root=str(tmp_path),source_sha256='a'*64,data_sha256=None)
    p['config']['data']['path']=str(tmp_path/'text')
    monkeypatch.setattr(engine,'_preflight',lambda spec:None)
    monkeypatch.setattr(engine,'_fingerprint',lambda spec:{'runtime':{'packages':{'torch':'test'}}})
    monkeypatch.setattr(engine,'_verify_checkpoints',lambda *args:None)
    calls=[]
    def launch(*args,**kwargs):calls.append(1);return 'failed','synthetic failure'
    kwargs=dict(suite_builder=study.suite_spec,job_builder=study.local_jobs,launcher=launch,
        completion_jobs=lambda p,s:s['confirmation']['jobs'])
    with pytest.raises(RuntimeError):engine.run(p,tmp_path/'run',**kwargs)
    with pytest.raises(ValueError,match='no retry'):engine.run(p,tmp_path/'run',**kwargs)
    assert len(calls)==1
