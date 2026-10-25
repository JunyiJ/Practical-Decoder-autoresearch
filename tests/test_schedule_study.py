import copy
import json
import math
from pathlib import Path
import pytest
from src.research import schedule_study as study
from src.research import optimizer_study as engine
from src.research.runner import atomic_json
from src.research.team.suite import RUNTIME_TRAINING


def plan(backbone='dense'):
    from tests.test_architecture_study import full_plan
    p=full_plan()
    p.update(source_root='/tmp/context-test-root',source_sha256='a'*64,data_sha256=None)
    p['config']['data']=dict(path='/tmp/context-test-corpus',train_fraction=.8,val_fraction=.1)
    for key in RUNTIME_TRAINING:p['config']['training'].pop(key,None)
    p.update(study='schedule-quality-v1',schedule_backbone=backbone,families=list(study.CELLS),learning_rates=study.LRS,
        seeds=study.SCREEN_SEEDS,confirmation_seeds=study.CONFIRM_SEEDS,widths=study.WIDTHS,
        max_trials=60,trial_seconds=1800,session_seconds=28800,max_disk_mb=4096,
        primary_metric='full_validation_context128',quality_margin=.01,
        bootstrap_blocks=[2048,4096,8192],bootstrap_draws=2000,bootstrap_seed=20261024,
        evaluation_targets=111538,total_training_targets=3072000,
        batch_layout='paired_blocks512',schedule_policy='incumbent_constant_cosine_cooldown_3000_v1',max_process_rss_mb=4096,max_mps_driver_mb=6144)
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
        assert m['reference_aliases']==(study.reference_cells('dense') if lr==.0025 else {})
    s['choices']['constant']['lr']=.0025
    assert study.confirmation_manifest(s['choices'],'dense')['total_distinct_trials']==48


def test_selection_rejects_confirmation_and_incomplete_or_duplicate_rows():
    p=plan();rs=screen_rows(p)
    for bad in [rs[:-1],rs+[rs[0]],rs+[{**rs[0],'seed':2111}],rs+[{**rs[0],'width':256}]]:
        with pytest.raises(ValueError):study.freeze_selection(p,bad)
    rs[0]['metric']=float('nan')
    with pytest.raises(ValueError):study.freeze_selection(p,rs)


@pytest.mark.parametrize('key,value',[('confirmation_seeds',[2101,2112,2113]),('quality_margin',.02),
    ('widths',[64,128,256]),('max_trials',61),('trial_seconds',1801),('evaluation_targets',111537)])
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


@pytest.mark.parametrize('backbone',['dense','moe4_top2'])
def test_serial_engine_freezes_before_fresh_confirmation_and_no_retraining(tmp_path,monkeypatch,backbone):
    total=54 if backbone=='moe4_top2' else 48
    p=plan(backbone);p.update(source_root=str(tmp_path),source_sha256='a'*64,data_sha256=None)
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
            assert selection['confirmation']['total_distinct_trials']==total
            assert len([r for r in seen if r['config']['training']['seed'] in study.SCREEN_SEEDS])==18
            assert req['stage']=='confirmation'
        else:assert req['stage']=='local'
        score=2+abs(t['learning_rate']-.0025)*100
        # One confirmation seed fails the human ceiling even though the mean passes.
        if t['lr_schedule']['kind']=='linear_cooldown' and t['seed']==2113:score+=.02
        atomic_json(path.parent/'result.json',dict(metric=score,
            coverage={str(c):coverage(score,c) for c in set([128,m['block_size']])},
            runtime=dict(torch='test',cpu_threads=4,requested_device='mps')))
        return 'completed',None
    kwargs=dict(suite_builder=study.suite_spec,job_builder=study.local_jobs,
        selection_builder=study.freeze_selection,report_builder=study.summarize,launcher=launch,
        result_adapter=lambda raw,*args:dict(coverage=raw['coverage']),
        completion_jobs=lambda p,s:s['confirmation']['jobs'])
    state=engine.run(p,tmp_path/'run',**kwargs)
    assert state['status']=='completed' and len(seen)==total
    report=json.loads((tmp_path/'run/report.json').read_text())
    assert all(not x['near_neutral_common128'] for x in report['comparisons'] if x['contrast']=='tuned_schedule_equal_tokens' and x['candidate']=='linear_cooldown' and x['reference']=='constant')
    engine.run(p,tmp_path/'run',**kwargs);assert len(seen)==total


@pytest.mark.parametrize('backbone',['dense','moe4_top2'])
def test_model_and_token_budget_identical_across_schedule_families(backbone):
    p=plan(backbone)
    configs={f:study.suite_spec(p,f,.0025)['candidate']['config'] for f in study.CELLS}
    assert all(c['model']==configs['constant']['model'] for c in configs.values())
    assert configs['constant']['model']['mlp_type']==('moe' if backbone=='moe4_top2' else 'dense')
    assert configs['constant']['model']['n_layers']==4
    assert configs['constant']['model']['ffn_width_policy']==('moe_swiglu_21_16' if backbone=='moe4_top2' else 'swiglu_21_8')
    for family,c in configs.items():
        t=copy.deepcopy(c['training']); schedule=t.pop('lr_schedule')
        base=copy.deepcopy(configs['constant']['training']);base.pop('lr_schedule')
        assert t==base
        assert schedule['kind']==family
    ref=study.suite_spec(p,'incumbent_reference' if backbone=='moe4_top2' else 'deep_reference',.0025)['candidate']['config']
    assert ref==configs['constant']
    assert configs['cosine']['training']['lr_schedule']==dict(kind='cosine',total_steps=3000,min_ratio=.1)
    assert configs['linear_cooldown']['training']['lr_schedule']==dict(kind='linear_cooldown',total_steps=3000,min_ratio=.1,start_step=2000)


def test_schedule_choice_frozen_before_confirmation():
    p=plan();rs=screen_rows(p)
    for r in rs:
        if r['family']=='cosine':r['metric']-=.02
    s=study.freeze_selection(p,rs)
    assert s['winner']=='cosine'
    bad=copy.deepcopy(rs)
    for r in bad:
        if r['family']=='cosine' and r['seed']==2102:r['metric']+=.04
    assert study.freeze_selection(p,bad)['winner']=='historical_incumbent_unchanged'


def test_moe_backbone_has_separate_dense_reference_and_bounded_aliases():
    p=plan('moe4_top2')
    for lr,total in [(.00125,60),(.0025,54),(.005,60)]:
        manifest=study.freeze_selection(p,screen_rows(p,lr))['confirmation']
        assert manifest['total_distinct_trials']==total
        assert manifest['reference_aliases']==({'incumbent_reference':'constant'} if lr==.0025 else {})
        assert sum(j[0]=='deep_reference' for j in manifest['jobs'])==6
    old=study.suite_spec(p,'deep_reference',.0025)['candidate']['config']
    incumbent=study.suite_spec(p,'incumbent_reference',.0025)['candidate']['config']
    assert old['model']['mlp_type']=='dense' and incumbent['model']['mlp_type']=='moe'
    with pytest.raises(ValueError):study.suite_spec(plan(),'incumbent_reference',.0025)
    bad=plan();bad['schedule_backbone']='unchecked'
    with pytest.raises(ValueError):study.validate_plan(bad)


@pytest.mark.parametrize('qualified',[False,True])
def test_backbone_resolved_from_final_review_not_screen_only(tmp_path,qualified):
    from src.research.protocol import file_digest
    selection={'winner':'moe4_top2'}
    comparisons=[dict(candidate='moe4_top2',reference=ref,width=w,sensitivity=[dict(
        mean_delta=-.02 if qualified or w==64 else .01,
        paired_seed_deltas={str(seed):(-.02 if qualified or w==64 else .01) for seed in [2091,2092,2093]})])
        for ref in ['dense','deep_reference'] for w in [64,256]]
    report=dict(plan_sha256='a'*64,selection=selection,comparisons=comparisons)
    for name,obj in [('report',report),('selection',selection)]:
        (tmp_path/f'{name}.json').write_text(json.dumps(obj))
    review=dict(status='verified',distinct_checkpoints=48,report_sha256=file_digest(tmp_path/'report.json'),
        selection_sha256=file_digest(tmp_path/'selection.json'),plan_sha256='a'*64,
        quality_decision=dict(frozen_screen_winner='moe4_top2',qualifies_for_incumbent_replacement=qualified))
    path=tmp_path/'review.json';path.write_text(json.dumps(review))
    assert study.reviewed_backbone(path)==('moe4_top2' if qualified else 'dense')
    review['status']='partial_verified';path.write_text(json.dumps(review))
    with pytest.raises(ValueError,match='completed verified'):study.reviewed_backbone(path)
    review['status']='verified';review['quality_decision']['qualifies_for_incumbent_replacement']=not qualified
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError,match='adoption rule'):study.reviewed_backbone(path)
