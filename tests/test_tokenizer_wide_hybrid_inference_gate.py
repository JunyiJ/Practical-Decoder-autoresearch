"""Synthetic frozen-panel gate fixtures; no model, real review or corpus load."""
import json
from pathlib import Path
import pytest
from src.research import tokenizer_wide_hybrid_inference_gate as g
from src.research import tokenizer_wide_hybrid_rules as r
from src.research.protocol import digest,file_digest
from src.research.tokenizer_wide_hybrid_driver import _trial_id


def write(path,obj):path.write_text(json.dumps(obj))


@pytest.fixture
def setup(tmp_path,monkeypatch):
    root=tmp_path;tokenizer={'synthetic':True};token_sha='1'*64;calls=[]
    plan=dict(status='frozen',output=str(root),source={'fixture':'source'},runtime={'fixture':'runtime'},
        inference_contract=dict(g.CONTRACT,tokenizer_sha256=digest(tokenizer),token_sha256=token_sha))
    write(root/'plan.json',plan);plan_sha=file_digest(root/'plan.json')
    def verify(plan):
        calls.append('verify_plan')
        if plan['status']!='frozen':raise ValueError('synthetic plan not frozen')
    monkeypatch.setattr(g,'verify_plan',verify)
    def contract(cfg):calls.append('contract');return {'tokenizer':tokenizer}
    monkeypatch.setattr(g,'_contract',contract)
    rows=[dict(s,status='completed',ce=5.) for s in r.lr_specs()]
    rates=r.choose_rates(rows)
    rows += [dict(s,status='completed',ce=5.) for s in r.rank_specs(rates)]
    selection=r.choose_ranks(rows[:9],rows[9:])
    rows += [dict(s,status='completed',ce=5.) for s in r.confirmation_specs(selection)]
    reviews=[];audits=[]
    for index,row in enumerate(rows):
        spec={k:v for k,v in row.items() if k not in ('status','ce')}
        trial=root/_trial_id(index,spec);trial.mkdir();(trial/'checkpoints').mkdir()
        request=dict(spec=spec,config={'synthetic_family':spec['family']},result_path=str(trial/'result.json'))
        write(trial/'request.json',request);files={str(trial/'request.json'):file_digest(trial/'request.json')}
        result=dict(status='completed',holdout_consumed=False,request_sha256=files[str(trial/'request.json')])
        for kind in ('raw','h200'):
            p=trial/'checkpoints'/(kind+'.synthetic');p.write_text(str(index)+kind)
            result[kind]=dict(checkpoint=str(p),checkpoint_sha256=file_digest(p));files[str(p)]=file_digest(p)
        write(trial/'result.json',result);files[str(trial/'result.json')]=file_digest(trial/'result.json')
        reviews.append(dict(files=files,review=dict(selection_row=row,metadata_reviewed=True,
            checkpoint_hashes={k:result[k]['checkpoint_sha256'] for k in ('raw','h200')})))
        audits.append(dict(trial=str(trial),raw_sha256=result['raw']['checkpoint_sha256'],
            h200_sha256=result['h200']['checkpoint_sha256'],checks=dict.fromkeys(sorted(g.TENSOR_CHECKS),True)))
    state=dict(status='quality_completed_pending_tensor_and_serving_audits',active=None,adopted=False,
        plan_sha256=plan_sha,records=rows,rates=rates,selection=selection,reviews=reviews)
    write(root/'state.json',state);write(root/'lr-selection.json',rates);write(root/'rank-selection.json',selection)
    trace=root/'synthetic-trace';trace.write_text('Not a real tensor audit')
    audit=dict(kind='wide_hybrid_tensor_audit_v1',status='passed',plan_sha256=plan_sha,
        state_sha256=file_digest(root/'state.json'),source_sha256=digest(plan['source']),
        runtime_sha256=digest(plan['runtime']),records=audits,artifacts={str(trace):file_digest(trace)})
    write(root/'audit.json',audit)
    request=dict(quality_run=str(root),plan_sha256=plan_sha,state_sha256=file_digest(root/'state.json'),
        tensor_review=str(root/'audit.json'),tensor_review_sha256=file_digest(root/'audit.json'),
        tokenizer=tokenizer,token_sha256=token_sha)
    return root,request,plan,state,audit,calls


@pytest.mark.parametrize('seed',r.SEEDS)
def test_complete_gate_binds_requested_fresh_triple(setup,seed):
    root,request,plan,state,audit,calls=setup
    out=g.bind_inference(request,seed)
    assert set(out['bindings'])==set(r.FAMILIES) and calls==['verify_plan']+['contract']*3+['verify_plan']
    assert all(b['request']['spec']['seed']==seed for b in out['bindings'].values())
    assert len(out['files'])==102 # plan/state +2 choices +96 trial files +review/trace
    assert out['selection']==state['selection'] and out['contract']['job_seconds']==600
    g.recheck(out['files'])


@pytest.mark.parametrize('bad',['draft','missing_contract','changed_time','changed_tokens','source','runtime'])
def test_plan_binding_rejection(setup,bad):
    root,request,plan,state,audit,calls=setup
    if bad=='draft':plan['status']='draft'
    elif bad=='missing_contract':plan.pop('inference_contract')
    elif bad=='changed_time':plan['inference_contract']['job_seconds']=601
    elif bad=='changed_tokens':plan['inference_contract']['token_sha256']='2'*64
    else:plan[bad]={'changed':True}
    write(root/'plan.json',plan);request['plan_sha256']=file_digest(root/'plan.json')
    state['plan_sha256']=request['plan_sha256'];write(root/'state.json',state)
    request['state_sha256']=file_digest(root/'state.json')
    audit.update(plan_sha256=request['plan_sha256'],state_sha256=request['state_sha256'])
    write(root/'audit.json',audit);request['tensor_review_sha256']=file_digest(root/'audit.json')
    with pytest.raises(ValueError):g.bind_inference(request,3511)
    assert 'contract' not in calls


@pytest.mark.parametrize('bad',['active','status','missing_record','missing_review','changed_row','receipt','selection'])
def test_incomplete_quality_rejected(setup,bad):
    root,request,plan,state,audit,calls=setup
    if bad=='active':state['active']=state['records'][0]
    elif bad=='status':state['status']='failed'
    elif bad=='missing_record':state['records'].pop()
    elif bad=='missing_review':state['reviews'].pop()
    elif bad=='changed_row':state['records'][0]['seed']=3502
    elif bad=='receipt':state['reviews'][0]['review']['metadata_reviewed']=False
    else:write(root/'rank-selection.json',{})
    write(root/'state.json',state);request['state_sha256']=file_digest(root/'state.json')
    with pytest.raises(ValueError):g.bind_inference(request,3511)
    assert 'contract' not in calls


@pytest.mark.parametrize('bad',['missing','false_ema','wrong_hash','source','runtime','state','no_trace','trace_changed'])
def test_complete_bound_tensor_audit_required(setup,bad):
    root,request,plan,state,audit,calls=setup
    if bad=='missing':audit['records'].pop()
    elif bad=='false_ema':audit['records'][0]['checks']['exact_ema']=False
    elif bad=='wrong_hash':audit['records'][0]['h200_sha256']='0'*64
    elif bad in ('source','runtime','state'):audit[bad+'_sha256']='0'*64
    elif bad=='no_trace':audit['artifacts']={}
    else:(root/'synthetic-trace').write_text('changed')
    write(root/'audit.json',audit);request['tensor_review_sha256']=file_digest(root/'audit.json')
    with pytest.raises(ValueError):g.bind_inference(request,3511)
    assert 'contract' not in calls


@pytest.mark.parametrize('kind',['raw','request','result'])
def test_retained_file_mutation_blocks_gate(setup,kind):
    root,request,plan,state,audit,calls=setup
    files=state['reviews'][0]['files']
    suffix={'raw':'raw.synthetic','request':'request.json','result':'result.json'}[kind]
    path=next(Path(p) for p in files if p.endswith(suffix));path.write_text('changed')
    with pytest.raises(ValueError):g.bind_inference(request,3511)
    assert 'contract' not in calls


def test_stop_prevents_any_provenance_or_loader_work(setup):
    root,request,*_=setup;(root/'STOP').touch()
    with pytest.raises(ValueError,match='STOP'):g.bind_inference(request,3511)
    assert not setup[-1]


def test_rebuilt_tokenizer_mismatch_and_final_recheck(setup,monkeypatch):
    root,request,*_=setup
    monkeypatch.setattr(g,'_contract',lambda cfg:{'tokenizer':{'wrong':True}})
    with pytest.raises(ValueError,match='tokenizer'):g.bind_inference(request,3511)


def test_mutation_during_contract_rebuild_rejected(setup,monkeypatch):
    root,request,*_=setup
    def changed(cfg):
        (root/'synthetic-trace').write_text('changed during reconstruction')
        return {'tokenizer':request['tokenizer']}
    monkeypatch.setattr(g,'_contract',changed)
    with pytest.raises(ValueError,match='evidence changed'):g.bind_inference(request,3511)


def test_stop_during_contract_rebuild_rejected(setup,monkeypatch):
    root,request,*_=setup
    def changed(cfg):
        (root/'STOP').touch()
        return {'tokenizer':request['tokenizer']}
    monkeypatch.setattr(g,'_contract',changed)
    with pytest.raises(ValueError,match='STOP'):g.bind_inference(request,3511)


def test_source_reverified_after_contract_rebuild(setup,monkeypatch):
    root,request,*_=setup
    calls=[]
    def verify(plan):
        calls.append(True)
        if len(calls)==2:raise ValueError('synthetic changed source')
    monkeypatch.setattr(g,'verify_plan',verify)
    with pytest.raises(ValueError,match='changed source'):g.bind_inference(request,3511)
