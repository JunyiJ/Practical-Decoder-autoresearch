"""Synthetic controller gates only: no model training, checkpoint scoring or MPS."""
import copy
import json
import pytest
from pathlib import Path
from src.research import averaging_study as study
from src.research import capacity_study as capacity
from src.research import optimizer_study as engine
from src.research.protocol import digest,file_digest
from src.research.runner import atomic_json
from tests.test_capacity_study import plan as capacity_plan
from tests.test_schedule_study import coverage


def plan(family='dense4',lr=.005):
    p=capacity_plan('linear_cooldown',.005)
    p['config']=capacity.suite_spec(p,family,lr)['candidate']['config']
    p['config']['training']['weight_averaging']=copy.deepcopy(study.AVERAGING)
    p.update(study='averaging-quality-v1',recipe=dict(family=family,learning_rate=lr,schedule=capacity.schedule_spec('linear_cooldown')),
        averaging=copy.deepcopy(study.AVERAGING),families=['trajectory'],learning_rates=[lr],seeds=study.SCREEN_SEEDS,
        confirmation_seeds=study.CONFIRM_SEEDS,widths=study.WIDTHS,min_improvement=.002,max_trials=8,
        trial_seconds=1800,session_seconds=14400,max_disk_mb=2048,model_outputs_per_trajectory=4,total_model_evaluations=32)
    p['screen_jobs']=[list(x) for x in study.local_jobs(p)]
    return study.validate_plan(p)


def row(job,stage='local',deltas=(-.01,-.02,-.005)):
    f,lr,w,s=job
    return dict(family=f,lr=lr,width=w,seed=s,status='completed',stage=stage,coverage={'128':coverage(2)},
        averaged_outputs={str(h):dict(coverage={'evaluations':{'128':coverage(2+d)}}) for h,d in zip(study.HALF_LIVES,deltas)})


def screens(p,deltas=(-.01,-.02,-.005)):
    return [row(j,deltas=deltas) for j in study.local_jobs(p)]


@pytest.mark.parametrize('family',['dense4','wide4','dense6'])
@pytest.mark.parametrize('lr',[.00125,.0025,.005])
def test_fixed_backbone_single_lr_and_eight_physical_trajectories(family,lr):
    p=plan(family,lr);spec=study.suite_spec(p,'trajectory',lr)
    assert spec['candidate']['config']==p['config']==spec['baseline']['config']
    assert len(study.local_jobs(p))==2 and len(study.confirmation_jobs(p))==6
    assert len(set(map(tuple,study.confirmation_jobs(p)+p['screen_jobs'])))==8
    assert not set(study.SCREEN_SEEDS)&set(study.CONFIRM_SEEDS)
    with pytest.raises(ValueError):study.suite_spec(p,'trajectory',.01)
    with pytest.raises(ValueError):study.suite_spec(p,'extra',lr)


@pytest.mark.parametrize('field,value',[('max_trials',9),('total_model_evaluations',33),('min_improvement',.001),
    ('widths',[64,128,256]),('confirmation_seeds',[2141,2152,2153]),('evaluation_targets',111537),
    ('max_disk_mb',4096),('averaging',dict(kind='spaced_tail_ema',total_steps=3000,start_step=2000,interval=10,half_lives=[10,50,201]))])
def test_protocol_changes_rejected(field,value):
    p=plan();p[field]=value
    with pytest.raises(ValueError):study.validate_plan(p)


def test_selection_tie_shorter_and_bad_seed_veto_without_pruning():
    p=plan();rs=screens(p,(-.01,-.01,-.005));sel=study.freeze_selection(p,rs)
    assert sel['half_life']==10 and sel['screen_eligible']
    rs[1]['averaged_outputs']['10']['coverage']['evaluations']['128']=coverage(2.001)
    rs[1]['averaged_outputs']['50']['coverage']['evaluations']['128']=coverage(2.002)
    rs[0]['averaged_outputs']['10']['coverage']['evaluations']['128']=coverage(1.95)
    sel=study.freeze_selection(p,rs)
    assert sel['half_life']==10 and not sel['screen_eligible']
    assert len(sel['confirmation']['jobs'])==6
    for bad in [rs[:-1],rs+[rs[0]],[rs[0],{**rs[1],'seed':2151}]]:
        with pytest.raises(ValueError):study.freeze_selection(p,bad)


def test_frozen_half_life_and_local_eligibility_survive_larger_width_results():
    p=plan();rs=screens(p);sel=study.freeze_selection(p,rs)
    conf=[row(j,'confirmation') for j in study.confirmation_jobs(p)]
    report=study.summarize(p,dict(records=rs+conf,selection=sel))
    assert report['quality_decision']==dict(frozen_half_life=50,qualifies_for_incumbent_replacement=True)
    # A different half-life wins by more later; that cannot replace the frozen choice.
    for r in conf:
        r['averaged_outputs']['10']['coverage']['evaluations']['128']=coverage(1.9)
        if r['width']==256 and r['seed']==2153:
            r['averaged_outputs']['50']['coverage']['evaluations']['128']=coverage(2.001)
    report=study.summarize(p,dict(records=rs+conf,selection=sel))
    assert report['quality_decision']==dict(frozen_half_life=50,qualifies_for_incumbent_replacement=False)
    rs=screens(p,(-.001,-.001,-.001));sel=study.freeze_selection(p,rs)
    report=study.summarize(p,dict(records=rs+conf,selection=sel))
    assert not report['quality_decision']['qualifies_for_incumbent_replacement']
    with pytest.raises(ValueError):study.summarize(p,dict(records=rs+conf[:-1],selection=sel))


def predecessor(tmp_path,qualified=True,winner='dense6'):
    p=capacity_plan('linear_cooldown',.005);p['git_commit']='b'*40
    sel=dict(winner=winner,choices={f:dict(lr=.005) for f in capacity.CELLS})
    comps=[]
    for w in study.WIDTHS:
        for ref in ('dense4','incumbent_reference','deep_reference'):
            ds={str(s):-.02 for s in [2131,2132,2133]}
            if not qualified and w==256:ds['2133']=.001
            comps.append(dict(candidate=winner,reference=ref,width=w,sensitivity=[dict(mean_delta=sum(ds.values())/3,paired_seed_deltas=ds)]))
    report=dict(plan_sha256=digest(p),selection=sel,comparisons=comps)
    for name,value in [('plan',p),('selection',sel),('report',report)]:atomic_json(tmp_path/f'{name}.json',value)
    review=dict(status='verified',distinct_checkpoints=54,plan_sha256=digest(p),source_commit=p['git_commit'],source_sha256=p['source_sha256'],
        report_sha256=file_digest(tmp_path/'report.json'),selection_sha256=file_digest(tmp_path/'selection.json'),
        quality_decision=dict(frozen_screen_winner=winner,qualifies_for_incumbent_replacement=qualified))
    atomic_json(tmp_path/'review.json',review);return tmp_path/'review.json'


@pytest.mark.parametrize('winner',['wide4','dense6'])
@pytest.mark.parametrize('qualified',[True,False])
def test_only_final_capacity_adoption_resolves_backbone(tmp_path,winner,qualified):
    path=predecessor(tmp_path,qualified,winner);recipe,cfg=study.reviewed_capacity(path)
    assert recipe['family']==(winner if qualified else 'dense4')
    assert cfg['model']['n_layers']==(6 if qualified and winner=='dense6' else 4)
    review=json.loads(path.read_text());review['quality_decision']['qualifies_for_incumbent_replacement']=not qualified
    atomic_json(path,review)
    with pytest.raises(ValueError):study.reviewed_capacity(path)


def test_partial_or_changed_predecessor_cannot_launch(tmp_path):
    path=predecessor(tmp_path);review=json.loads(path.read_text());review['status']='partial_verified';atomic_json(path,review)
    with pytest.raises(ValueError):study.reviewed_capacity(path)
    path=predecessor(tmp_path);(tmp_path/'report.json').write_text('{}')
    with pytest.raises(ValueError):study.reviewed_capacity(path)


def test_real_serial_engine_freezes_after_two_trajectories_and_stops_first_failure(tmp_path,monkeypatch):
    p=plan();calls=[]
    monkeypatch.setattr(engine,'_preflight',lambda spec:None)
    monkeypatch.setattr(engine,'_fingerprint',lambda spec:dict(runtime={}))
    monkeypatch.setattr(engine,'_verify_checkpoints',lambda rows,output:None)
    monkeypatch.setattr(engine,'_request',lambda spec,r,d:dict(config=dict(training=dict(device='mps'),model=dict(block_size=128))))
    monkeypatch.setattr(engine,'validate_result',lambda raw,*a:raw['row'])
    def launch(request,output,*args,**kwargs):
        i=len(calls);job=(study.local_jobs(p)+list(map(tuple,study.confirmation_jobs(p))))[i]
        if i>=2:assert (output/'selection.json').exists()
        calls.append(job);d=request.parent
        atomic_json(d/'result.json',dict(row=row(job,'local' if i<2 else 'confirmation'),runtime=dict(torch='x',cpu_threads=4,requested_device='mps')))
        return 'completed',None
    monkeypatch.setattr(engine,'_fingerprint',lambda spec:dict(runtime=dict(packages=dict(torch='x'))))
    monkeypatch.setattr(engine,'_request',lambda spec,r,d:dict(cpu_threads=4,config=dict(training=dict(device='mps'),model=dict(block_size=128))))
    result=engine.run(p,tmp_path/'pass',suite_builder=lambda *a:{},job_builder=study.local_jobs,
        selection_builder=study.freeze_selection,report_builder=study.summarize,launcher=launch,
        completion_jobs=lambda p,s:s['confirmation']['jobs'])
    assert result['status']=='completed' and len(calls)==8
    calls.clear()
    def fail(*a,**k):calls.append(1);return 'failed','synthetic failure'
    with pytest.raises(RuntimeError):engine.run(p,tmp_path/'fail',suite_builder=lambda *a:{},job_builder=study.local_jobs,
        selection_builder=study.freeze_selection,report_builder=study.summarize,launcher=fail,
        completion_jobs=lambda p,s:s['confirmation']['jobs'])
    assert len(calls)==1
    with pytest.raises(ValueError):engine.run(p,tmp_path/'fail',suite_builder=lambda *a:{},job_builder=study.local_jobs,
        selection_builder=study.freeze_selection,report_builder=study.summarize,launcher=fail,
        completion_jobs=lambda p,s:s['confirmation']['jobs'])
    assert len(calls)==1


def test_positive_text_upper_bound_vetoes_mean_and_every_seed_gain():
    p=plan();rs=screens(p);sel=study.freeze_selection(p,rs)
    conf=[row(j,'confirmation') for j in study.confirmation_jobs(p)]
    for r in conf:
        val=coverage(1.995);counts=val['block_target_counts'];n=len(counts)
        # Preserve a negative mean but deliberately heterogeneous text differences.
        offsets=[.2 if i<n//2 else -.2 for i in range(n)]
        center=sum(x*c for x,c in zip(offsets,counts))/sum(counts)
        val['block_loss_sums']=[(1.995+x-center)*c for x,c in zip(offsets,counts)]
        val['mean_ce']=sum(val['block_loss_sums'])/sum(counts)
        r['averaged_outputs']['50']['coverage']['evaluations']['128']=val
    report=study.summarize(p,dict(records=rs+conf,selection=sel))
    for c in report['comparisons']:
        if c['half_life']==50:
            assert c['sensitivity'][0]['mean_delta']<-.002
            assert all(x<0 for x in c['sensitivity'][0]['paired_seed_deltas'].values())
            assert any(x['conditional_text_bootstrap_95pct'][1]>0 for x in c['sensitivity'])
            assert not c['quality_improvement']
    assert not report['quality_decision']['qualifies_for_incumbent_replacement']


@pytest.mark.parametrize('change',['extra_average','bad_coverage','wrong_stage'])
def test_report_rejects_extra_or_malformed_confirmation_outputs(change):
    p=plan();rs=screens(p);sel=study.freeze_selection(p,rs)
    conf=[row(j,'confirmation') for j in study.confirmation_jobs(p)]
    if change=='extra_average':conf[0]['averaged_outputs']['999']=conf[0]['averaged_outputs']['10']
    elif change=='bad_coverage':conf[0]['averaged_outputs']['10']['coverage']['evaluations']['128']['target_count']=1
    else:conf[0]['stage']='local'
    with pytest.raises(ValueError):study.summarize(p,dict(records=rs+conf,selection=sel))


@pytest.mark.parametrize('change',[None,'parent','half_life','targets','runtime','snapshot','names'])
def test_result_adapter_checks_all_derived_lineage_and_coverage(tmp_path,change):
    p=plan();request=dict(config=copy.deepcopy(p['config']))
    (tmp_path/'request.json').write_text('{}');(tmp_path/'result.json').write_text('{}')
    parent=tmp_path/'raw.pt';parent.write_bytes(b'synthetic raw metadata test; not a model')
    checked=dict(checkpoint=str(parent),checkpoint_sha256=file_digest(parent),metric=2.)
    raw=dict(runtime=dict(torch='synthetic'),coverage=dict(checkpoint_sha256=file_digest(parent),device='mps',
        torch='synthetic',holdout_consumed=False,evaluations={'128':coverage(2.)}),averaged_outputs={})
    for h in study.HALF_LIVES:
        cp=tmp_path/f'average_half_life_{h}.pt';cp.write_bytes(b'synthetic derived metadata test; not a model')
        raw['averaged_outputs'][str(h)]=dict(checkpoint=str(cp),checkpoint_sha256=file_digest(cp),
            parent_checkpoint=dict(path=str(parent),sha256=file_digest(parent)),
            averaged_weights=dict(spec=copy.deepcopy(study.AVERAGING),half_life=h,snapshots=101,completed_steps=3000,
                parameter_names=['synthetic.weight'],purpose='inference_only_not_optimizer_resume'),
            coverage=dict(checkpoint_sha256=file_digest(cp),parent_checkpoint_sha256=file_digest(parent),half_life=h,
                holdout_consumed=False,device='mps',torch='synthetic',evaluations={'128':coverage(1.99)}))
    bad=raw['averaged_outputs']['10']
    if change=='parent':bad['parent_checkpoint']['sha256']='0'*64
    elif change=='half_life':bad['coverage']['half_life']=50
    elif change=='targets':bad['coverage']['evaluations']['128']['target_count']=1
    elif change=='runtime':bad['coverage']['device']='cpu'
    elif change=='snapshot':bad['averaged_weights']['snapshots']=100
    elif change=='names':bad['averaged_weights']['parameter_names']=['x','x']
    if change:
        with pytest.raises(ValueError):study.adapter(p,raw,checked,request,tmp_path)
    else:
        result=study.adapter(p,raw,checked,request,tmp_path)
        assert result['averaged_outputs']==raw['averaged_outputs'] and result['metric']==2.


@pytest.mark.parametrize('fail_half_life',[None,50])
def test_actual_bootstrap_exports_and_scores_in_order_preserving_first_failure(tmp_path,monkeypatch,fail_half_life):
    # Execute the actual orchestration string with only training/scoring/guard
    # endpoints replaced. No model is instantiated and no accelerator is used.
    import sys
    import src.research.worker as worker
    import src.research.memory_guard as guard
    import src.research.coverage_eval as evaluator
    import src.research.averaging_eval as average_evaluator
    import src.train.weight_averaging as averaging
    from src.train import train as trainer
    monkeypatch.setattr(trainer,'GPT',trainer.GPT)  # Restore bootstrap class assignment after test.
    monkeypatch.setattr(guard,'start_memory_guard',lambda:None)
    p=plan();cfg=copy.deepcopy(p['config']);corpus=tmp_path/'corpus.txt';corpus.write_text('synthetic data')
    cfg['data']['path']=str(corpus);request=tmp_path/'request.json';result=tmp_path/'result.json';parent=tmp_path/'raw.pt'
    atomic_json(request,dict(config=cfg,result_path=str(result)))
    monkeypatch.setattr(sys,'argv',['synthetic-worker','--request',str(request)])
    events=[]
    def train():
        events.append('train');parent.write_bytes(b'synthetic raw checkpoint')
        atomic_json(result,dict(final_checkpoint=str(parent),completed_steps=3000))
    def raw_score(job):
        assert job['data_sha256']==file_digest(corpus) and job['checkpoint_sha256']==file_digest(parent)
        events.append('raw_score');return {'synthetic':'raw coverage'}
    def export(raw,path,h):
        assert raw==parent and not path.exists();events.append(('export',h));path.write_bytes(str(h).encode())
        return dict(checkpoint=str(path),checkpoint_sha256=file_digest(path),parent_checkpoint=dict(path=str(parent),sha256=file_digest(parent)))
    def score(job):
        assert job['contexts']==[128] and job['completed_steps']==3000 and job['data_sha256']==file_digest(corpus)
        assert job['parent_sha256']==file_digest(parent) and job['checkpoint_sha256']==file_digest(Path(job['checkpoint']))
        events.append(('score',job['half_life']))
        if job['half_life']==fail_half_life:raise RuntimeError('synthetic coverage failure')
        return {'synthetic':job['half_life']}
    monkeypatch.setattr(worker,'main',train);monkeypatch.setattr(evaluator,'evaluate_job',raw_score)
    monkeypatch.setattr(averaging,'export_averaged_checkpoint',export);monkeypatch.setattr(average_evaluator,'evaluate_averaged_job',score)
    if fail_half_life:
        with pytest.raises(RuntimeError,match='synthetic coverage failure'):exec(study.BOOTSTRAP,{})
        assert events==['train','raw_score',('export',10),('score',10),('export',50),('score',50)]
        assert set(json.loads(result.read_text())['averaged_outputs'])=={'10'}
        assert (tmp_path/'average_half_life_50.pt').exists() and not (tmp_path/'average_half_life_200.pt').exists()
    else:
        exec(study.BOOTSTRAP,{})
        assert events==['train','raw_score']+[event for h in study.HALF_LIVES for event in [('export',h),('score',h)]]
        assert set(json.loads(result.read_text())['averaged_outputs'])=={'10','50','200'}
