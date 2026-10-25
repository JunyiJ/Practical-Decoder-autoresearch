"""Synthetic bounded-controller evidence; no model training or MPS work."""
import copy
import json
import math

import pytest

from src.research import qknorm_study as study, optimizer_study as engine
from src.research.protocol import digest
from src.research.runner import atomic_json
from tests.test_horizon_study import plan as horizon_plan
from tests.test_schedule_study import coverage


def plan(family='control3000'):
    p = horizon_plan()
    total,start = study.horizon.CELLS[family]
    recipe = dict(family=family,total_steps=total,cooldown_start=start)
    p.update(study='qknorm-quality-v1',families=study.FAMILIES,learning_rates=study.LRS,
        seeds=study.SCREEN_SEEDS,confirmation_seeds=study.CONFIRM_SEEDS,widths=study.WIDTHS,
        max_trials=30,max_model_evaluations=60,recipe=recipe,
        primary_metric='H200_full_validation_context128',normalization_policy='rms_head32_pre_rope_v1')
    p['config']['model'],p['config']['training'] = study.backbone(recipe)
    p['screen_jobs'] = [list(j) for j in study.local_jobs(p)]
    return study.validate_plan(p)


def row(job,stage='local',ce=2.,raw_ce=3.):
    f,lr,w,s = job
    return dict(family=f,lr=lr,width=w,seed=s,trial_id=f'{f}_{lr}_{w}_{s}',stage=stage,
        status='completed',coverage={'128':coverage(raw_ce)},
        averaged_outputs={'200':dict(coverage={'evaluations':{'128':coverage(ce)}})})


def screening(p,standard_lr=.005,eligible=True):
    result=[]
    for j in study.local_jobs(p):
        f,lr,_,s=j
        ce = (2. if lr==standard_lr else 2.02) if f=='standard' else (1.98 if lr==.0025 else 2.04)
        if not eligible and f=='qkrms' and lr==.0025 and s==2182:
            ce=2.001
        result.append(row(j,ce=ce))
    return result


def complete(p,standard_lr=.005,eligible=True):
    local=screening(p,standard_lr,eligible)
    selection=study.freeze_selection(p,local)
    conf=[row(j,'confirmation',1.97 if j[0]=='qkrms' else 2.,raw_ce=4. if j[0]=='qkrms' else 3.)
          for j in selection['confirmation']['jobs']]
    return dict(records=local+conf,selection=selection)


@pytest.mark.parametrize('family',list(study.horizon.CELLS))
@pytest.mark.parametrize('arm',['standard','qkrms',study.REFERENCE])
def test_explicit_shared_horizon_and_only_qk_intervention(family,arm):
    p=plan(family)
    c=study.suite_spec(p,arm,.005)['candidate']['config']
    assert c['training']==p['config']['training']
    m=copy.deepcopy(c['model'])
    if arm=='qkrms':
        assert m.pop('qk_norm')=='rms_head32'
        assert m['depth_width_policy']=='depth4_qkrms_unscaled'
        assert m['attention_width_policy']=='mha_head32_qkrms'
        m.update(depth_width_policy='depth4_unscaled',attention_width_policy='mha_head32')
    else:
        assert 'qk_norm' not in m
    assert m==p['config']['model']
    assert len(study.local_jobs(p))==12
    assert c['training']['weight_averaging']['half_lives']==[200]
    assert c['training']['max_iters']*8*128==p['recipe']['total_steps']*1024


@pytest.mark.parametrize('lr,count,aliases',[(.005,24,{study.REFERENCE:'standard'}),(.0025,30,{})])
def test_frozen_choices_exact_alias_counts_and_primary_only_adoption(lr,count,aliases):
    p=plan();s=complete(p,lr);selection=s['selection'];report=study.summarize(p,s)
    assert selection['choices']['standard']['lr']==lr
    assert selection['choices']['qkrms']['lr']==.0025
    assert selection['candidate_eligible']
    assert selection['confirmation']['reference_aliases']==aliases
    assert len(s['records'])==count==report['physical_training_trajectories']
    assert report['model_evaluations']==2*count
    assert report['quality_decision']['qualifies_for_incumbent_replacement']
    assert len(report['comparisons'])==6 and len(report['fixed_lr_screen'])==3
    for c in report['comparisons']:
        if c['candidate']=='qkrms':
            assert c['quality_improvement'] and c['raw_sensitivity'][0]['mean_delta']>0
        elif aliases:
            assert c['sensitivity'][0]['mean_delta']==0
            assert all(x['conditional_text_bootstrap_95pct']==[0.,0.] for x in c['sensitivity'])


def test_tie_chooses_lower_lr_and_screen_bad_seed_cannot_be_overridden():
    p=plan();rs=[row(j,ce=2. if j[0]=='standard' else 1.98) for j in study.local_jobs(p)]
    sel=study.freeze_selection(p,rs)
    assert all(c['lr']==.00125 for c in sel['choices'].values())
    s=complete(p,eligible=False)
    assert not s['selection']['candidate_eligible']
    assert not study.summarize(p,s)['quality_decision']['qualifies_for_incumbent_replacement']


def test_standard_lr_can_win_prospectively_without_posthoc_qk_promotion():
    p=plan();s=complete(p,.0025,eligible=False)
    assert s['selection']['winner']=='standard'
    assert s['selection']['standard_vs_incumbent']['supported']
    for r in s['records']:
        if r['stage']=='confirmation':
            ce={'standard':1.97,'qkrms':1.95,study.REFERENCE:2.}[r['family']]
            r['averaged_outputs']['200']['coverage']['evaluations']['128']=coverage(ce)
    decision=study.summarize(p,s)['quality_decision']
    assert decision==dict(frozen_screen_winner='standard',qualifies_for_incumbent_replacement=True)
    # A wider bad seed vetoes the frozen standard winner despite excellent QK RMS.
    for r in s['records']:
        if r['stage']=='confirmation' and r['family']=='standard' and r['width']==256 and r['seed']==2193:
            r['averaged_outputs']['200']['coverage']['evaluations']['128']=coverage(2.001)
    decision=study.summarize(p,s)['quality_decision']
    assert decision==dict(frozen_screen_winner='standard',qualifies_for_incumbent_replacement=False)


def test_failed_frozen_qk_cannot_fall_back_to_successful_standard_confirmation():
    p=plan();s=complete(p,.0025)
    assert s['selection']['winner']=='qkrms'
    for r in s['records']:
        if r['stage']=='confirmation':
            ce={'standard':1.95,'qkrms':1.94,study.REFERENCE:2.}[r['family']]
            if r['family']=='qkrms' and r['width']==256 and r['seed']==2193:ce=1.96
            r['averaged_outputs']['200']['coverage']['evaluations']['128']=coverage(ce)
    report=study.summarize(p,s)
    assert not report['quality_decision']['qualifies_for_incumbent_replacement']
    assert all(c['quality_improvement'] for c in report['comparisons'] if c['candidate']=='standard')


@pytest.mark.parametrize('reference',[study.REFERENCE,'standard'])
def test_must_beat_both_confirmation_references_at_both_widths(reference):
    p=plan();s=complete(p,.0025)
    # The other reference and the entire small-width panel still pass.
    for r in s['records']:
        if r['stage']=='confirmation' and r['family']==reference and r['width']==256 and r['seed']==2193:
            r['averaged_outputs']['200']['coverage']['evaluations']['128']=coverage(1.96)
    report=study.summarize(p,s)
    assert not report['quality_decision']['qualifies_for_incumbent_replacement']
    assert any(c['quality_improvement'] for c in report['comparisons'] if c['candidate']=='qkrms')


def test_text_upper_bound_vetoes_all_negative_seed_means():
    p=plan();s=complete(p)
    for r in s['records']:
        if r['stage']!='confirmation' or r['family']!='qkrms':continue
        v=coverage(1.994);counts=v['block_target_counts']
        offsets=[.3 if i<len(counts)//2 else -.3 for i in range(len(counts))]
        center=sum(x*c for x,c in zip(offsets,counts))/sum(counts)
        v['block_loss_sums']=[(1.994+x-center)*c for x,c in zip(offsets,counts)]
        v['mean_ce']=sum(v['block_loss_sums'])/sum(counts)
        r['averaged_outputs']['200']['coverage']['evaluations']['128']=v
    report=study.summarize(p,s)
    assert not report['quality_decision']['qualifies_for_incumbent_replacement']
    for c in report['comparisons']:
        if c['candidate']=='qkrms':
            assert c['sensitivity'][0]['mean_delta']<-.005
            assert max(c['sensitivity'][0]['paired_seed_deltas'].values())<0
            assert any(v['conditional_text_bootstrap_95pct'][1]>0 for v in c['sensitivity'])


@pytest.mark.parametrize('change',['missing','duplicate','wrong_seed','wrong_stage','failed','extra_average','bad_coverage'])
def test_invalid_screen_evidence_rejected(change):
    p=plan();rs=screening(p)
    if change=='missing':rs.pop()
    elif change=='duplicate':rs.append(rs[0])
    elif change=='wrong_seed':rs[0]['seed']=2191
    elif change=='wrong_stage':rs[0]['stage']='confirmation'
    elif change=='failed':rs[0]['status']='failed'
    elif change=='extra_average':rs[0]['averaged_outputs']['50']=rs[0]['averaged_outputs']['200']
    else:rs[0]['coverage']['128']['target_count']=1
    with pytest.raises(ValueError):study.freeze_selection(p,rs)


@pytest.mark.parametrize('change',['manifest','choices','eligibility','alias','extra_record','missing_record','wrong_stage'])
def test_confirmation_cannot_reselect_or_change_aliases(change):
    p=plan();s=complete(p)
    if change=='manifest':s['selection']['confirmation']['jobs'].pop()
    elif change=='choices':s['selection']['choices']['qkrms']['lr']=.005
    elif change=='eligibility':s['selection']['candidate_eligible']=False
    elif change=='alias':s['selection']['confirmation']['reference_aliases']={}
    elif change=='extra_record':s['records'].append(s['records'][-1])
    elif change=='missing_record':s['records'].pop()
    else:s['records'][-1]['stage']='transfer'
    with pytest.raises(ValueError):study.summarize(p,s)


@pytest.mark.parametrize('field,value',[('max_trials',31),('seeds',[2181,2191]),('returned_half_life',50),
    ('evaluation_targets',1),('bootstrap_seed',1),('max_model_evaluations',61),('widths',[64,128,256]),
    ('min_improvement',0.),('normalization_policy','rms_after_rope'),('trial_seconds',9999)])
def test_protocol_extensions_rejected(field,value):
    p=plan();p[field]=value
    with pytest.raises(ValueError):study.validate_plan(p)


@pytest.mark.parametrize('change',['schedule','averaging','model','horizon','steps'])
def test_unreviewed_backbone_change_rejected(change):
    p=plan()
    if change=='schedule':p['config']['training']['lr_schedule']['start_step']=1990
    elif change=='averaging':p['config']['training']['weight_averaging']['half_lives']=[50]
    elif change=='model':p['config']['model']['qk_norm']='rms_head32'
    elif change=='horizon':p['recipe']['family']='long6000_fixedtail'
    else:p['recipe']['total_steps']=3000.0
    with pytest.raises(ValueError):study.validate_plan(p)


@pytest.mark.parametrize('standard_lr',[.005,.0025])
def test_real_engine_freezes_before_confirmation_and_first_failure_stops(tmp_path,monkeypatch,standard_lr):
    p=plan();s=complete(p,standard_lr);fixtures=s['records'];calls=[]
    monkeypatch.setattr(engine,'_preflight',lambda s:None)
    monkeypatch.setattr(engine,'_verify_checkpoints',lambda *a:None)
    monkeypatch.setattr(engine,'_fingerprint',lambda s:dict(runtime=dict(packages=dict(torch='x'))))
    monkeypatch.setattr(engine,'_request',lambda spec,r,d:dict(cpu_threads=4,config=dict(training=dict(device='mps'),model=dict(block_size=128))))
    monkeypatch.setattr(engine,'validate_result',lambda raw,*a:raw['row'])
    def launch(request,output,*a,**kw):
        i=len(calls)
        if i>=12:
            saved=json.loads((output/'selection.json').read_text())
            assert saved['choices']==s['selection']['choices']
            assert saved['confirmation']==s['selection']['confirmation']
        calls.append(i)
        atomic_json(request.parent/'result.json',dict(row=fixtures[i],runtime=dict(torch='x',cpu_threads=4,requested_device='mps')))
        return 'completed',None
    kw=dict(suite_builder=lambda *a:{},job_builder=study.local_jobs,selection_builder=study.freeze_selection,
            report_builder=study.summarize,completion_jobs=lambda p,s:s['confirmation']['jobs'])
    result=engine.run(p,tmp_path/'pass',launcher=launch,**kw)
    assert result['status']=='completed' and len(calls)==len(fixtures)
    calls.clear()
    def fail(*a,**kw):calls.append(1);return 'failed','synthetic failure'
    with pytest.raises(RuntimeError):engine.run(p,tmp_path/'failed',launcher=fail,**kw)
    assert len(calls)==1
    with pytest.raises(ValueError):engine.run(p,tmp_path/'failed',launcher=fail,**kw)
    assert len(calls)==1
