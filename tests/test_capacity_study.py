"""Synthetic controller/configuration checks; no model training or MPS."""
import copy
import json
from pathlib import Path
import pytest
from src.research import capacity_study as study
from src.research import optimizer_study as engine
from src.research.protocol import digest,file_digest
from src.research.runner import atomic_json
from tests.test_schedule_study import plan as schedule_plan,coverage


def plan(kind='constant',lr=.0025):
    p=schedule_plan()
    del p['schedule_backbone'];del p['schedule_policy']
    p.update(study='capacity-quality-v1',incumbent_schedule=dict(kind=kind,learning_rate=lr),
        families=list(study.CELLS),seeds=study.SCREEN_SEEDS,confirmation_seeds=study.CONFIRM_SEEDS,
        capacity_policy='dense4_wide4_dense6_reviewed_schedule_v1')
    p['screen_jobs']=[list(x) for x in study.local_jobs(p)]
    return study.validate_plan(p)


def screen(p,lr=.0025):
    return [dict(family=f,lr=l,width=w,seed=s,status='completed',metric=2+abs(l-lr)*100)
        for f,l,w,s in study.local_jobs(p)]


@pytest.mark.parametrize('kind',['constant','cosine','linear_cooldown'])
@pytest.mark.parametrize('incumbent_lr',[.00125,.0025,.005])
def test_manifest_and_exact_reference_aliases(kind,incumbent_lr):
    if kind=='constant' and incumbent_lr!=.0025:
        with pytest.raises(ValueError):plan(kind,incumbent_lr)
        return
    p=plan(kind,incumbent_lr)
    assert len(study.local_jobs(p))==18
    for lr in study.LRS:
        sel=study.freeze_selection(p,screen(p,lr));m=sel['confirmation']
        expect=(54 if kind=='constant' else 60)-(6 if lr==incumbent_lr else 0)
        assert m['total_distinct_trials']==expect
        assert len({tuple(x) for x in m['jobs']})==len(m['jobs'])
        assert set(x[3] for x in m['jobs'])==set(study.CONFIRM_SEEDS)
        assert not set(study.SCREEN_SEEDS)&set(study.CONFIRM_SEEDS)
        for alias,original in m['reference_aliases'].items():
            a=study.suite_spec(p,alias,incumbent_lr)['candidate']['config']
            b=study.suite_spec(p,original,lr)['candidate']['config']
            assert a==b
        if kind!='constant':assert 'deep_reference' not in m['reference_aliases']


@pytest.mark.parametrize('kind',['constant','cosine','linear_cooldown'])
def test_capacity_changes_are_explicit_and_schedule_shared(kind):
    p=plan(kind)
    cfgs={f:study.suite_spec(p,f,.0025)['candidate']['config'] for f in study.CELLS}
    assert all(c['training']==cfgs['dense4']['training'] for c in cfgs.values())
    assert cfgs['dense4']['training']['lr_schedule']==study.schedule_spec(kind)
    for f,depth,ffn,policy in [('dense4',4,'swiglu_21_8','depth4_unscaled'),
        ('wide4',4,'swiglu_21_4','depth4_wide_unscaled'),('dense6',6,'swiglu_21_8','depth6_dense_unscaled')]:
        m=cfgs[f]['model'];assert m['n_layers']==depth and m['ffn_width_policy']==ffn and m['depth_width_policy']==policy
        common=copy.deepcopy(m)
        for k in ['n_layers','ffn_width_policy','depth_width_policy']:common.pop(k)
        baseline=copy.deepcopy(cfgs['dense4']['model'])
        for k in ['n_layers','ffn_width_policy','depth_width_policy']:baseline.pop(k)
        assert common==baseline
    for ref in ['gelu_reference','shallow_reference','deep_reference']:
        assert study.suite_spec(p,ref,study.FIXED[ref])['candidate']['config']['training']['lr_schedule']=={'kind':'constant'}


def test_local_choice_requires_both_references_and_all_seeds():
    p=plan('cosine',.005);rows=screen(p)
    for r in rows:
        if r['family']=='wide4':r['metric']-=.02
        if r['family']=='dense6':r['metric']-=.03
    sel=study.freeze_selection(p,rows);assert sel['winner']=='dense6'
    bad=copy.deepcopy(rows)
    # Keep better mean but a worse individual seed disqualifies this candidate.
    for r in bad:
        if r['family']=='dense6' and r['seed']==2122:r['metric']+=.04
    assert study.freeze_selection(p,bad)['winner']=='wide4'
    for invalid in [rows[:-1],rows+[rows[0]],rows+[dict(rows[0],width=256)],rows+[dict(rows[0],seed=2131)]]:
        with pytest.raises(ValueError):study.freeze_selection(p,invalid)


@pytest.mark.parametrize('kind',['constant','cosine','linear_cooldown'])
def test_mock_serial_controller_selection_before_confirmation(tmp_path,monkeypatch,kind):
    p=plan(kind);p.update(source_root=str(tmp_path),source_sha256='a'*64,data_sha256=None)
    p['config']['data']['path']=str(tmp_path/'text')
    monkeypatch.setattr(engine,'_preflight',lambda spec:None)
    monkeypatch.setattr(engine,'_fingerprint',lambda spec:{'runtime':{'packages':{'torch':'test'}}})
    monkeypatch.setattr(engine,'_verify_checkpoints',lambda *args:None)
    monkeypatch.setattr(engine,'validate_result',lambda raw,*args:dict(metric=raw['metric'],training_time_sec=3,rss_mb=100))
    seen=[];total=48 if kind=='constant' else 54
    def launch(path,output,*args,**kwargs):
        req=json.loads(path.read_text());t=req['config']['training'];m=req['config']['model'];seen.append(req)
        assert t['batch_size']*m['block_size']*t['max_iters']==3072000
        if t['seed'] in study.CONFIRM_SEEDS:
            sel=json.loads((output/'selection.json').read_text())
            assert sel['confirmation']['total_distinct_trials']==total
            assert len([r for r in seen if r['config']['training']['seed'] in study.SCREEN_SEEDS])==18
        score=2+abs(t['learning_rate']-.0025)*100
        if m['n_layers']==6:score-=.03
        elif m.get('ffn_width_policy')=='swiglu_21_4':score-=.02
        atomic_json(path.parent/'result.json',dict(metric=score,coverage={'128':coverage(score)},runtime=dict(torch='test',cpu_threads=4,requested_device='mps')))
        return 'completed',None
    kwargs=dict(suite_builder=study.suite_spec,job_builder=study.local_jobs,selection_builder=study.freeze_selection,
        report_builder=study.summarize,launcher=launch,result_adapter=lambda raw,*args:dict(coverage=raw['coverage']),
        completion_jobs=lambda p,s:s['confirmation']['jobs'])
    state=engine.run(p,tmp_path/'run',**kwargs)
    assert state['status']=='completed' and len(seen)==total and state['selection']['winner']=='dense6'
    engine.run(p,tmp_path/'run',**kwargs);assert len(seen)==total


def predecessor(tmp_path,qualified=True,kind='cosine',bad_seed=False):
    p=schedule_plan();p['git_commit']='a'*40
    sel=dict(winner=kind,choices={kind:{'lr':.005}})
    comparisons=[]
    for w in [64,256]:
        for ref in ['constant','deep_reference']:
            ds={str(s):-.02 for s in [2111,2112,2113]}
            if not qualified and w==256:ds['2113']=.001
            if bad_seed:ds['2113']=float('nan')
            comparisons.append(dict(candidate=kind,reference=ref,width=w,sensitivity=[dict(paired_seed_deltas=ds,mean_delta=sum(ds.values())/3)]))
    report=dict(plan_sha256=digest(p),selection=sel,comparisons=comparisons)
    for name,obj in [('plan',p),('selection',sel),('report',report)]:
        # Deliberately malformed external fixture bypasses the strict writer.
        if bad_seed and name=='report':(tmp_path/f'{name}.json').write_text(json.dumps(obj))
        else:atomic_json(tmp_path/f'{name}.json',obj)
    review=dict(status='verified',distinct_checkpoints=48,plan_sha256=digest(p),source_commit=p['git_commit'],source_sha256=p['source_sha256'],
        report_sha256=file_digest(tmp_path/'report.json'),selection_sha256=file_digest(tmp_path/'selection.json'),
        quality_decision=dict(frozen_screen_winner=kind,qualifies_for_incumbent_replacement=qualified))
    atomic_json(tmp_path/'review.json',review);return tmp_path/'review.json'


@pytest.mark.parametrize('kind',['cosine','linear_cooldown'])
@pytest.mark.parametrize('qualified',[True,False])
def test_only_final_audited_schedule_can_set_common_recipe(tmp_path,kind,qualified):
    path=predecessor(tmp_path,qualified,kind)
    assert study.reviewed_schedule(path)==dict(kind=kind if qualified else 'constant',learning_rate=.005 if qualified else .0025)
    review=json.loads(path.read_text());review['quality_decision']['qualifies_for_incumbent_replacement']=not qualified
    atomic_json(path,review)
    with pytest.raises(ValueError,match='Adoption'):study.reviewed_schedule(path)


def test_reject_partial_nonfinite_or_changed_predecessor(tmp_path):
    path=predecessor(tmp_path,bad_seed=True)
    with pytest.raises(ValueError):study.reviewed_schedule(path)
    path=predecessor(tmp_path);r=json.loads(path.read_text());r['status']='partial_verified';atomic_json(path,r)
    with pytest.raises(ValueError):study.reviewed_schedule(path)
    path=predecessor(tmp_path);(tmp_path/'report.json').write_text('{}')
    with pytest.raises(ValueError):study.reviewed_schedule(path)
