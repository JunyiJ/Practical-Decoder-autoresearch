import copy
import json
import math
from pathlib import Path
import pytest
from src.research import batch_study as study
from src.research import optimizer_study as engine
from src.research.runner import atomic_json
from src.research.team.suite import RUNTIME_TRAINING


def plan():
    from tests.test_architecture_study import full_plan
    p=full_plan()
    p.update(source_root='/tmp/context-test-root',source_sha256='a'*64,data_sha256=None)
    p['config']['data']=dict(path='/tmp/context-test-corpus',train_fraction=.8,val_fraction=.1)
    for key in RUNTIME_TRAINING:p['config']['training'].pop(key,None)
    p.update(study='batch-sensitivity-v1',families=list(study.CELLS),learning_rates=study.LRS,
        seeds=study.SCREEN_SEEDS,confirmation_seeds=study.CONFIRM_SEEDS,widths=study.WIDTHS,
        max_trials=54,trial_seconds=1200,session_seconds=21600,max_disk_mb=2048,
        primary_metric='full_validation_context128',quality_margin=.01,
        bootstrap_blocks=[2048,4096,8192],bootstrap_draws=2000,bootstrap_seed=20261024,
        evaluation_targets=111538,total_training_targets=3072000,
        batch_layout='paired_batch512',decay_policy='matched_nominal_cumulative')
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
    s['choices']['d4_mha_b8']['lr']=.0025
    assert study.confirmation_manifest(s['choices'])['total_distinct_trials']==48


def test_selection_rejects_confirmation_and_incomplete_or_duplicate_rows():
    p=plan();rs=screen_rows(p)
    for bad in [rs[:-1],rs+[rs[0]],rs+[{**rs[0],'seed':2071}],rs+[{**rs[0],'width':256}]]:
        with pytest.raises(ValueError):study.freeze_selection(p,bad)
    rs[0]['metric']=float('nan')
    with pytest.raises(ValueError):study.freeze_selection(p,rs)


@pytest.mark.parametrize('key,value',[('confirmation_seeds',[2061,2072,2073]),('quality_margin',.02),
    ('widths',[64,128,256]),('max_trials',55),('trial_seconds',1800),('evaluation_targets',111537)])
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
        assert t['batch_size']*m['block_size']*t['max_iters']==3072000 and t['batch_layout']=='paired_batch512'
        if t['seed'] in study.CONFIRM_SEEDS:
            selection=json.loads((output/'selection.json').read_text())
            assert selection['confirmation']['total_distinct_trials']==48
            assert len([r for r in seen if r['config']['training']['seed'] in study.SCREEN_SEEDS])==18
            assert req['stage']=='confirmation'
        else:assert req['stage']=='local'
        score=2+abs(t['learning_rate']-.0025)*100
        # One confirmation seed fails the human ceiling even though the mean passes.
        if t['batch_size']==16 and t['seed']==2073:score+=.02
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
    assert all(not x['near_neutral_common128'] for x in report['comparisons'] if x['contrast']=='tuned_batch_equal_tokens' and x['candidate']=='d4_mha_b16' and x['reference']=='d4_mha_b8')
    engine.run(p,tmp_path/'run',**kwargs);assert len(seen)==48


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
            assert cfg.training.batch_size*cfg.model.block_size*cfg.training.max_iters==3072000
            assert math.isclose((1-opt.decay)**cfg.training.max_iters,(1-.00006)**3000,rel_tol=1e-12)
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


def test_four_layer_mha_active_init_and_optimizer_groups_identical_across_batches():
    import torch
    from omegaconf import OmegaConf
    from src.models.gpt import GPT
    from src.train.optimizers import build_optimizer
    torch.set_num_threads(1);p=plan()
    for width in study.WIDTHS:
        previous=None;previous_rng=None
        for batch in [8,16,32]:
            cfg=OmegaConf.create(study.suite_spec(p,f'd4_mha_b{batch}',.0025)['candidate']['config'])
            cfg.model.update(dim=width,n_heads=width//32,vocab_size=65)
            torch.manual_seed(2061);model=GPT(cfg.model);opt=build_optimizer(model,cfg.training)
            weights={n:t.detach().clone() for n,t in model.named_parameters()}
            if previous is not None:
                assert set(weights)==set(previous)
                assert all(torch.equal(weights[n],previous[n]) for n in weights)
                assert torch.equal(torch.get_rng_state(),previous_rng)
            previous=weights;previous_rng=torch.get_rng_state().clone()
            assert math.isclose((1-opt.decay)**cfg.training.max_iters,(1-.00006)**3000,rel_tol=1e-12)
            assert not any('wpe' in n for n in weights)


def test_grouped_batch_sampler_exact_stream_rng_and_legacy_equality(tmp_path):
    import torch
    from src.data.loaders import TinyDataLoader
    path=tmp_path/'text';path.write_text(''.join(chr(65+i%26) for i in range(30000)))
    streams=[];states=[]
    for b in [8,16,32]:
        loader=TinyDataLoader(path,b,128,'cpu',seed=17,train_fraction=.8,val_fraction=.1,eval_seed=2026,batch_layout='paired_batch512')
        xs=[];ys=[]
        for _ in range(32//b):
            x,y=loader.get_batch();xs.append(x.flatten());ys.append(y.flatten())
            loader.get_batch('val')
        streams.append((torch.cat(xs),torch.cat(ys)));states.append(loader.generators['train'].get_state())
    assert all(torch.equal(a,streams[0][0]) and torch.equal(b,streams[0][1]) for a,b in streams)
    assert all(torch.equal(states[0],s) for s in states)
    old=TinyDataLoader(path,8,128,'cpu',seed=17,train_fraction=.8,val_fraction=.1,eval_seed=2026,batch_layout='paired_blocks512')
    x,y=old.get_batch();assert torch.equal(x.flatten(),streams[0][0][:1024]);assert torch.equal(y.flatten(),streams[0][1][:1024])
    for b,c in [(4,128),(64,128),(8,256),(16,512)]:
        with pytest.raises(ValueError):TinyDataLoader(path,b,c,'cpu',batch_layout='paired_batch512')


def test_batch_budget_decay_and_eval_schedule():
    p=plan()
    for b in [8,16,32]:
        cfg=study.suite_spec(p,f'd4_mha_b{b}',.0025)['candidate']['config'];t=cfg['training']
        assert t['batch_size']==b and t['max_iters']==24000//b
        assert t['eval_every']==2400//b and t['eval_iters']==30
        assert math.isclose(t['max_iters']*math.log1p(-t['weight_decay']),3000*math.log1p(-.00006),abs_tol=1e-14)
