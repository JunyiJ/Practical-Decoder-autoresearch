"""Synthetic horizon controller and schedule algebra; no model training or MPS."""
import copy,json,math
from pathlib import Path
import pytest
from omegaconf import OmegaConf
from src.research import horizon_study as study,averaging_study as averaging,optimizer_study as engine
from src.research.protocol import digest,file_digest
from src.research.runner import atomic_json
from src.train.lr_schedule import schedule_contract,multiplier
from src.train.weight_averaging import averaging_contract
from tests.test_averaging_study import plan as averaging_plan,row as averaging_row
from tests.test_schedule_study import coverage


def plan(h=200):
    p=averaging_plan();p['config']['training'].pop('weight_averaging')
    p.update(study='horizon-quality-v1',families=list(study.CELLS),learning_rates=[.005],seeds=study.SCREEN_SEEDS,
        confirmation_seeds=study.CONFIRM_SEEDS,widths=study.WIDTHS,min_improvement=.005,max_trials=24,
        trial_seconds=2400,session_seconds=21600,max_disk_mb=4096,returned_half_life=h,total_model_evaluations=48 if h else 24,
        horizon_cells={k:dict(total_steps=v[0],cooldown_start=v[1]) for k,v in study.CELLS.items()})
    p['screen_jobs']=[list(x) for x in study.local_jobs(p)];return study.validate_plan(p)


def row(p,job,stage='local',ce=None):
    f,lr,w,s=job;ce={'control3000':2.,'long6000_stretched':1.98,'long6000_fixedtail':1.97}[f] if ce is None else ce
    r=dict(family=f,lr=lr,width=w,seed=s,status='completed',stage=stage,coverage={'128':coverage(ce+.001 if p['returned_half_life'] else ce)})
    if p['returned_half_life']:r['averaged_outputs']={str(p['returned_half_life']):dict(coverage={'evaluations':{'128':coverage(ce)}})}
    return r


@pytest.mark.parametrize('h',[None,10,50,200])
@pytest.mark.parametrize('family',list(study.CELLS))
def test_fixed_horizons_exact_schedules_output_counts_and_decay(h,family):
    p=plan(h);cfg=study.suite_spec(p,family,.005)['candidate']['config'];t=cfg['training'];total,start=study.CELLS[family]
    assert t['max_iters']==total and t['learning_rate']==.005 and t['weight_decay']==.00006 and t['eval_every']==300
    spec=schedule_contract(OmegaConf.create(t));assert spec==study.schedule(family)
    assert multiplier(spec,start)==1 and multiplier(spec,total-1)==.1
    assert math.isclose(sum(multiplier(spec,i)*.005 for i in range(total)),dict(control3000=12.75,long6000_stretched=25.5,long6000_fixedtail=27.75)[family],abs_tol=1e-10)
    a=averaging_contract(OmegaConf.create(t))
    if h:
        assert a['half_lives']==[h] and a['start_step']==start and a['total_steps']==total
        assert 1+(total-start)//a['interval']==(201 if family=='long6000_stretched' else 101)
    else:assert a is None
    assert len(study.local_jobs(p))==6 and len(study.confirmation_jobs(p))==18
    assert len(set(map(tuple,p['screen_jobs']+study.confirmation_jobs(p))))==24


@pytest.mark.parametrize('field,value',[('max_trials',25),('returned_half_life',201),('returned_half_life',True),('total_model_evaluations',49),('min_improvement',.001),('seeds',[2161,2171]),('widths',[64,128,256]),('trial_seconds',9999),('evaluation_targets',1)])
def test_changed_protocol_rejected(field,value):
    p=plan();p[field]=value
    with pytest.raises(ValueError):study.validate_plan(p)


@pytest.mark.parametrize('h',[None,200])
def test_local_frozen_choice_and_all_width_rule(h):
    p=plan(h);rs=[row(p,j) for j in study.local_jobs(p)];sel=study.freeze_selection(p,rs);assert sel['winner']=='long6000_fixedtail'
    conf=[row(p,j,'confirmation') for j in study.confirmation_jobs(p)];report=study.summarize(p,dict(records=rs+conf,selection=sel))
    assert report['quality_decision']['qualifies_for_incumbent_replacement'] and len(report['comparisons'])==6
    for i,r in enumerate(conf):
        if r['family']=='long6000_fixedtail' and r['width']==256 and r['seed']==2173:conf[i]=row(p,(r['family'],.005,256,2173),'confirmation',2.001)
    report=study.summarize(p,dict(records=rs+conf,selection=sel))
    assert report['quality_decision']==dict(frozen_screen_winner='long6000_fixedtail',qualifies_for_incumbent_replacement=False)
    # Better transfer from an unselected alternative cannot promote it.
    assert any(c['candidate']=='long6000_stretched' and c['quality_improvement'] for c in report['comparisons'])


def test_bad_screen_seed_retains_control_even_if_confirmation_is_excellent():
    p=plan();rs=[row(p,j,ce=2.001 if j[3]==2162 and j[0]!='control3000' else None) for j in study.local_jobs(p)]
    sel=study.freeze_selection(p,rs);assert sel['winner']=='control3000'
    conf=[row(p,j,'confirmation',1. if j[0]!='control3000' else 2.) for j in study.confirmation_jobs(p)]
    assert not study.summarize(p,dict(records=rs+conf,selection=sel))['quality_decision']['qualifies_for_incumbent_replacement']


@pytest.mark.parametrize('change',['missing','duplicate','wrong_seed','extra_average','missing_average','bad_coverage','failed','wrong_stage'])
def test_incomplete_or_extra_screen_rejected(change):
    p=plan();rs=[row(p,j) for j in study.local_jobs(p)]
    if change=='missing':rs.pop()
    elif change=='duplicate':rs.append(rs[0])
    elif change=='wrong_seed':rs[0]['seed']=2171
    elif change=='extra_average':rs[0]['averaged_outputs']['50']=rs[0]['averaged_outputs']['200']
    elif change=='missing_average':rs[0].pop('averaged_outputs')
    elif change=='bad_coverage':rs[0]['coverage']['128']['target_count']=1
    elif change=='failed':rs[0]['status']='failed'
    elif change=='wrong_stage':rs[0]['stage']='confirmation'
    with pytest.raises(ValueError):study.freeze_selection(p,rs)


def predecessor(tmp_path,adopt=True):
    p=averaging_plan();p['git_commit']='a'*40
    local=[averaging_row(j,deltas=(-.003,-.004,-.01)) for j in averaging.local_jobs(p)];sel=averaging.freeze_selection(p,local)
    conf=[averaging_row(j,'confirmation',(-.003,-.004,-.01)) for j in averaging.confirmation_jobs(p)]
    if not adopt:conf[-1]['averaged_outputs']['200']['coverage']['evaluations']['128']=coverage(2.01)
    report=averaging.summarize(p,dict(records=local+conf,selection=sel))
    for name,value in [('plan',p),('selection',sel),('report',report)]:atomic_json(tmp_path/f'{name}.json',value)
    review=dict(status='verified',distinct_trajectories=8,distinct_checkpoints=32,plan_sha256=digest(p),source_commit=p['git_commit'],source_sha256=p['source_sha256'],report_sha256=file_digest(tmp_path/'report.json'),selection_sha256=file_digest(tmp_path/'selection.json'),quality_decision=report['quality_decision'])
    atomic_json(tmp_path/'review.json',review);return tmp_path/'review.json'


@pytest.mark.parametrize('adopt',[True,False])
def test_only_final_review_resolves_returned_weight_policy(tmp_path,adopt):
    path=predecessor(tmp_path,adopt);h,cfg=study.reviewed_averaging(path);assert h==(200 if adopt else None) and 'weight_averaging' not in cfg['training']
    review=json.loads(path.read_text());review['quality_decision']['qualifies_for_incumbent_replacement']=not adopt;atomic_json(path,review)
    with pytest.raises(ValueError):study.reviewed_averaging(path)


def test_partial_predecessor_rejected(tmp_path):
    path=predecessor(tmp_path);v=json.loads(path.read_text());v['status']='partial_verified';atomic_json(path,v)
    with pytest.raises(ValueError):study.reviewed_averaging(path)


def test_real_engine_freezes_after_six_and_first_failure_stops(tmp_path,monkeypatch):
    p=plan();calls=[];jobs=study.local_jobs(p)+list(map(tuple,study.confirmation_jobs(p)))
    monkeypatch.setattr(engine,'_preflight',lambda s:None);monkeypatch.setattr(engine,'_verify_checkpoints',lambda *a:None)
    monkeypatch.setattr(engine,'_fingerprint',lambda s:dict(runtime=dict(packages=dict(torch='x'))))
    monkeypatch.setattr(engine,'_request',lambda spec,r,d:dict(cpu_threads=4,config=dict(training=dict(device='mps'),model=dict(block_size=128))))
    monkeypatch.setattr(engine,'validate_result',lambda raw,*a:raw['row'])
    def launch(request,output,*a,**kw):
        i=len(calls)
        if i>=6:assert (output/'selection.json').exists()
        calls.append(jobs[i]);atomic_json(request.parent/'result.json',dict(row=row(p,jobs[i],'local' if i<6 else 'confirmation'),runtime=dict(torch='x',cpu_threads=4,requested_device='mps')));return 'completed',None
    args=dict(suite_builder=lambda *a:{},job_builder=study.local_jobs,selection_builder=study.freeze_selection,report_builder=study.summarize,completion_jobs=lambda p,s:s['confirmation']['jobs'])
    s=engine.run(p,tmp_path/'pass',launcher=launch,**args);assert s['status']=='completed' and len(calls)==24
    calls.clear()
    def fail(*a,**kw):calls.append(1);return 'failed','synthetic failure'
    with pytest.raises(RuntimeError):engine.run(p,tmp_path/'fail',launcher=fail,**args)
    assert len(calls)==1
    with pytest.raises(ValueError):engine.run(p,tmp_path/'fail',launcher=fail,**args)
    assert len(calls)==1


def test_positive_text_upper_bound_vetoes_all_negative_seed_means():
    p=plan();rs=[row(p,j) for j in study.local_jobs(p)];sel=study.freeze_selection(p,rs)
    conf=[row(p,j,'confirmation') for j in study.confirmation_jobs(p)]
    for r in conf:
        if r['family']!='long6000_fixedtail':continue
        v=coverage(1.994);counts=v['block_target_counts'];offsets=[.3 if i<len(counts)//2 else -.3 for i in range(len(counts))]
        center=sum(x*c for x,c in zip(offsets,counts))/sum(counts)
        v['block_loss_sums']=[(1.994+x-center)*c for x,c in zip(offsets,counts)];v['mean_ce']=sum(v['block_loss_sums'])/sum(counts)
        r['averaged_outputs']['200']['coverage']['evaluations']['128']=v
    report=study.summarize(p,dict(records=rs+conf,selection=sel))
    assert not report['quality_decision']['qualifies_for_incumbent_replacement']
    for c in report['comparisons']:
        if c['candidate']=='long6000_fixedtail' and c['reference']=='control3000':
            assert c['sensitivity'][0]['mean_delta']<-.005 and all(x<0 for x in c['sensitivity'][0]['paired_seed_deltas'].values())
            assert any(x['conditional_text_bootstrap_95pct'][1]>0 for x in c['sensitivity'])
