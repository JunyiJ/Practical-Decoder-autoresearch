"""Synthetic frozen documents only; no real numerical review or MPS dispatch."""
import copy
import json
from pathlib import Path

import pytest

from src.research import tokenizer_wide_hybrid_pilot_gate as gate
from src.research.protocol import digest, file_digest


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root=tmp_path/'run';root.mkdir()
    monkeypatch.delenv('PYTORCH_ENABLE_MPS_FALLBACK', raising=False)
    monkeypatch.setattr(gate, 'source_fingerprint', lambda: {'synthetic': 'source'})
    monkeypatch.setattr(gate, 'runtime_identity', lambda: {'synthetic': 'runtime'})
    for name in ('PROTOCOL', 'PROJECTION', 'RECIPE'):
        path=tmp_path/name;path.write_text('synthetic '+name);monkeypatch.setattr(gate,name,path)
    corpus=tmp_path/'corpus';corpus.write_text('Synthetic identity; not research corpus')
    monkeypatch.setattr(gate,'CORPUS_SHA256',file_digest(corpus))
    original=gate.config
    def config(out,family):
        cfg=original(out,family);cfg['data']['path']=str(corpus);return cfg
    monkeypatch.setattr(gate,'config',config)
    artifact=tmp_path/'numerical.log';artifact.write_text('Synthetic fixture, not numerical results')
    review_path=tmp_path/'numerical.json'
    source=gate.source_fingerprint();runtime=gate.runtime_identity()
    review=dict(kind='numerical',status='passed',source_sha256=digest(source),
        runtime_sha256=digest(runtime),corpus_sha256=file_digest(corpus),
        checks=dict.fromkeys(('all_declared_ranks',)+gate.CHECKS,True),
        cases=gate.numerical_cases(),artifacts={str(artifact):file_digest(artifact)})
    write(review_path,review)
    requests=[]
    for i,spec in enumerate(gate.specs()):
        trial=root/f"{i:02d}-pilot-{spec['family']}-s3491-d2"
        requests.append(dict(spec=spec,config=config(trial,spec['family']),result_path=str(trial/'result.json')))
    files=[corpus,gate.PROTOCOL,gate.PROJECTION,gate.RECIPE,review_path,artifact]
    plan=dict(kind='wide_hybrid_pilot_v1',status='frozen',output=str(root),source=source,runtime=runtime,
        protocol_sha256=file_digest(gate.PROTOCOL),projection_sha256=file_digest(gate.PROJECTION),
        pilot_contract=copy.deepcopy(gate.CONTRACT),panel_design=gate.panel_design(),
        cache_contract=copy.deepcopy(gate.CACHE_CONTRACT),tokenizer={'synthetic':True},
        training_token_sha256='1'*64,
        storage_contract=gate.storage_contract(),requests=requests,numerical_review=str(review_path),
        files=[dict(path=str(p),sha256=file_digest(p)) for p in files])
    return root,plan,review_path,review


def bind(setup, family='hybrid'):
    root,plan,_,_=setup;path=root/'plan.json';write(path,plan)
    return gate.bind_pilot(path,file_digest(path),family)


def update_review(setup):
    _,plan,path,review=setup;write(path,review)
    for item in plan['files']:
        if item['path']==str(path): item['sha256']=file_digest(path)


@pytest.mark.parametrize('family', gate.rules.FAMILIES)
def test_bind_complete_synthetic_plan(setup, family):
    root,plan,_,_=setup;report=bind(setup,family)
    assert report['request']==next(r for r in plan['requests'] if r['spec']['family']==family)
    assert report['storage_contract']['full_panel_required_free_bytes']==57_224_879_357
    assert not report['adopted']
    assert list(root.iterdir())==[root/'plan.json']
    gate.recheck(report['files'])


@pytest.mark.parametrize('bad', ['draft','kind','source','runtime','protocol','projection','budget',
    'reserve','time','requests','request_config','seed','rate','threshold','quality_order',
    'quality_count','quality_holdout','corpus','recipe','review_binding','duplicate_file'])
def test_changed_plan_rejected(setup,bad):
    _,plan,path,_=setup
    if bad=='draft': plan['status']='draft'
    elif bad=='kind': plan['kind']='wide_hybrid_quality_v1'
    elif bad in ('source','runtime'): plan[bad]={}
    elif bad in ('protocol','projection'): plan[bad+'_sha256']='0'*64
    elif bad=='budget': plan['storage_contract']['required_free_bytes']-=1
    elif bad=='reserve': plan['pilot_contract']['reserve_bytes']-=1
    elif bad=='time': plan['pilot_contract']['job_seconds']=601
    elif bad=='requests': plan['requests'].reverse()
    elif bad=='request_config': plan['requests'][0]['config']['model']['rope']=1
    elif bad=='seed': plan['requests'][0]['spec']['seed']=3501
    elif bad=='rate': plan['requests'][0]['spec']['lr']=.00125
    elif bad=='threshold': plan['panel_design']['neutral_delta']=.02
    elif bad=='quality_order': plan['panel_design']['confirmations'].reverse()
    elif bad=='quality_count': plan['panel_design']['trajectories']=26
    elif bad=='quality_holdout': plan['panel_design']['test_holdout']='open'
    elif bad in ('corpus','recipe','review_binding'):
        target=plan['requests'][0]['config']['data']['path'] if bad=='corpus' else str(gate.RECIPE if bad=='recipe' else path)
        plan['files']=[r for r in plan['files'] if r['path']!=target]
    elif bad=='duplicate_file': plan['files'].append(copy.deepcopy(plan['files'][0]))
    with pytest.raises(ValueError): bind(setup)


@pytest.mark.parametrize('bad',['failed','source','runtime','corpus','check','case_missing','case_duplicate',
    'case_width','case_resume','case_float_divisor','no_trace','unbound_trace','self_trace'])
def test_incomplete_numerical_cannot_be_fixed_by_rehash(setup,bad):
    _,plan,path,review=setup
    if bad=='failed': review['status']='failed'
    elif bad in ('source','runtime','corpus'): review[bad+'_sha256']='0'*64
    elif bad=='check': review['checks']['resume']=False
    elif bad=='case_missing': review['cases'].pop()
    elif bad=='case_duplicate': review['cases'][-1]=copy.deepcopy(review['cases'][0])
    elif bad=='case_width': review['cases'][0]['width']=64
    elif bad=='case_resume': review['cases'][0]['checks']['resume']=False
    elif bad=='case_float_divisor': review['cases'][0]['divisor']=8.0
    elif bad=='no_trace': review['artifacts']={}
    elif bad=='unbound_trace': plan['files']=[r for r in plan['files'] if r['path'] not in review['artifacts']]
    elif bad=='self_trace': review['artifacts']={str(path):'0'*64}
    update_review(setup)
    with pytest.raises(ValueError): bind(setup)


@pytest.mark.parametrize('when',['before','during'])
def test_stop(setup,monkeypatch,when):
    root,_,_,_=setup
    if when=='before': (root/'STOP').touch()
    else:
        original=gate.runtime_identity
        def stop(): (root/'STOP').touch();return original()
        monkeypatch.setattr(gate,'runtime_identity',stop)
    with pytest.raises(ValueError,match='STOP'): bind(setup)


def test_source_changed_during_review(setup,monkeypatch):
    calls=[]
    def source(): calls.append(1);return {'synthetic':'source' if len(calls)==1 else 'changed'}
    monkeypatch.setattr(gate,'source_fingerprint',source)
    with pytest.raises(ValueError,match='changed during'): bind(setup)


def test_fallback(setup,monkeypatch):
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK','1')
    with pytest.raises(ValueError,match='Strict MPS'): bind(setup)


def test_bound_evidence_changed(setup):
    _,_,_,review=setup
    Path(next(iter(review['artifacts']))).write_text('changed')
    with pytest.raises(ValueError,match='evidence changed'): bind(setup)


def test_hash_and_family(setup):
    root,plan,_,_=setup;write(root/'plan.json',plan)
    with pytest.raises(ValueError,match='hash mismatch'):
        gate.bind_pilot(root/'plan.json','0'*64,'hybrid')
    with pytest.raises(ValueError,match='family'):
        gate.bind_pilot(root/'plan.json',file_digest(root/'plan.json'),'mla')


@pytest.mark.parametrize('field', ['cache_contract','tokenizer','training_token_sha256'])
def test_cache_binding_required(setup, field):
    setup[1][field]=None
    with pytest.raises(ValueError, match='cache/tokenizer/input'): bind(setup)
