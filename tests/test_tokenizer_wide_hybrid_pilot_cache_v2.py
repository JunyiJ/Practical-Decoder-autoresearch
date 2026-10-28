"""Tiny checkpoints and mocked MPS/model/cache APIs; no real research model."""
import json
from pathlib import Path

import pytest
import torch

from src.research import tokenizer_wide_hybrid_pilot_cache_v2 as producer
from src.research import tokenizer_wide_hybrid_pilot_cache as cache
from src.research.protocol import digest,file_digest
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_pilot_cache import setup


@pytest.fixture
def prepared(writer,setup,monkeypatch):
    binding,tokenizer,token_sha,events,models,files,corpus=setup
    trial=writer.root/'evidence/pilots/00';trial.parent.mkdir(parents=True)
    Path(binding['trial']).rename(trial)
    cfg=cache.config(trial,'hybrid')
    request=dict(binding['request'],config=cfg,result_path=str(trial/'result.json'))
    (trial/'request.json').write_text(json.dumps(request))
    result={}
    for role in ('raw','h200'):
        path=trial/'checkpoints'/(role+'.pt')
        payload=torch.load(path,weights_only=True);payload['cfg']=cfg;torch.save(payload,path)
        result[role]={'checkpoint':str(path)}
    (trial/'result.json').write_text(json.dumps(result))
    files.clear();files.update({str(p):file_digest(p) for p in trial.rglob('*') if p.is_file()})
    binding.update(trial=str(trial),request=request,request_sha256=files[str(trial/'request.json')],
                   result_sha256=files[str(trial/'result.json')])
    projection=json.loads(cache.PROJECTION.read_text())
    path=writer.publish_json('evidence/meta/projection.json',projection)
    pref=dict(path='meta/projection.json',sha256=file_digest(path))
    rows=[]
    def check(model,cfg,family,ids,sha,row):
        rows.append(row)
        return cache.check_model(model,cfg,family,ids,sha)
    monkeypatch.setattr(cache,'_check_model',check)
    # Neither the copied-projection adapter nor shared reviewer may fall back
    # to reading the source-tree projection during this audit.
    monkeypatch.setattr(cache,'PROJECTION',writer.root/'missing-source-projection.json')
    monkeypatch.setattr(cache,'_projection',lambda family:pytest.fail('source projection accessed'))
    return dict(writer=writer,slot=0,binding=binding,expected_tokenizer=tokenizer,
                expected_token_sha256=token_sha,projection_ref=pref),events,models,rows


def test_serial_tiny_endpoints_bind_campaign_copy_and_bounded_publication(prepared):
    args,events,models,rows=prepared
    ref=producer.review_and_publish(**args)
    evidence=args['writer'].root/'evidence'
    report=json.loads((evidence/ref['path']).read_text())
    assert report['kind']==producer.KIND
    assert report['contract']==cache.CACHE_CONTRACT and report['adopted'] is False
    assert ref['sha256']==file_digest(evidence/ref['path'])
    assert report['files'][str(evidence/'meta/projection.json')]==args['projection_ref']['sha256']
    assert str(cache.PROJECTION) not in report['files']
    assert len(rows)==2 and all(r['family']=='hybrid' and r['bottleneck_divisor']==2 for r in rows)
    assert events==['values','loader','new','load','check','empty','loader','new','load','check','empty']
    assert all(m() is None for m in models)
    marker=report['attempt'];attempt=json.loads((evidence/marker['path']).read_text())
    assert marker['sha256']==file_digest(evidence/marker['path'])
    assert attempt==producer.attempt_contract(0,args['binding']['request_sha256'],args['binding']['result_sha256'],
        args['projection_ref'],args['expected_tokenizer'],args['expected_token_sha256'])
    with pytest.raises(ValueError,match='cannot be retried'):producer.review_and_publish(**args)


@pytest.mark.parametrize('endpoint',[1,2])
def test_cache_failure_is_durable_and_blocks_retry(prepared,monkeypatch,endpoint):
    args,events,_,rows=prepared;original=cache._check_model;calls=[]
    def fail(*a):
        calls.append(1)
        if len(calls)==endpoint:raise RuntimeError('synthetic kernel failure')
        return original(*a)
    monkeypatch.setattr(cache,'_check_model',fail)
    with pytest.raises(RuntimeError,match='synthetic'):producer.review_and_publish(**args)
    trial=Path(args['binding']['trial'])
    failure=json.loads((trial/'audit-failure.json').read_text())
    assert failure['status']=='failed' and failure['retry_allowed'] is False
    assert not (trial/'audit.json').exists()
    assert events.count('empty')==endpoint
    with pytest.raises(ValueError,match='cannot be retried'):producer.review_and_publish(**args)
    assert len(calls)==endpoint


@pytest.mark.parametrize('bad',['slot','trial','spec','projection_hash','projection_path','projection_shape','STOP'])
def test_invalid_binding_prevents_model_work(prepared,bad):
    args,events,_,_=prepared;writer=args['writer']
    if bad=='slot':args['slot']=True
    elif bad=='trial':args['binding']['trial']+='/../00'
    elif bad=='spec':args['binding']['request']['spec']['family']='kda'
    elif bad=='projection_hash':args['projection_ref']['sha256']='0'*64
    elif bad=='projection_path':args['projection_ref']['path']='../outside.json'
    elif bad=='projection_shape':
        path=writer.root/'evidence/meta/projection.json';data=json.loads(path.read_text())
        data['records'].pop(0);path.write_text(json.dumps(data));args['projection_ref']['sha256']=file_digest(path)
    else:writer.publish_json('STOP',{'stop':True})
    with pytest.raises(ValueError):producer.review_and_publish(**args)
    assert events==[]
    assert not (Path(args['binding']['trial'])/'audit-attempt.json').exists()


def test_changed_projection_during_cache_work_stops_before_second_endpoint(prepared,monkeypatch):
    args,events,_,_=prepared;original=cache._check_model
    def mutate(*a):
        result=original(*a)
        (args['writer'].root/'evidence/meta/projection.json').write_text('{}')
        return result
    monkeypatch.setattr(cache,'_check_model',mutate)
    with pytest.raises(ValueError):producer.review_and_publish(**args)
    assert events.count('new')==1 and events.count('empty')==1
    assert (Path(args['binding']['trial'])/'audit-failure.json').exists()


def test_stop_during_audit_is_preserved_and_no_second_endpoint(prepared,monkeypatch):
    args,events,_,_=prepared;original=cache._check_model
    def stop(*a):
        result=original(*a);args['writer'].publish_json('STOP',{'stop':True});return result
    monkeypatch.setattr(cache,'_check_model',stop)
    with pytest.raises(ValueError,match='STOP'):producer.review_and_publish(**args)
    assert events.count('new')==1


def test_audit_attempt_tampering_is_not_published_as_success(prepared,monkeypatch):
    args,_,_,_=prepared;original=cache._check_model
    def mutate(*a):
        result=original(*a)
        (Path(args['binding']['trial'])/'audit-attempt.json').write_text('{}')
        return result
    monkeypatch.setattr(cache,'_check_model',mutate)
    with pytest.raises(ValueError):producer.review_and_publish(**args)
    assert not (Path(args['binding']['trial'])/'audit.json').exists()


def test_bounded_publication_failure_preserves_attempt_and_original_error(prepared,monkeypatch):
    args,_,_,_=prepared;original=args['writer'].publish_json
    def publish(name,value,**kw):
        if name.endswith('/audit.json'):raise OSError('synthetic publication failure')
        return original(name,value,**kw)
    monkeypatch.setattr(args['writer'],'publish_json',publish)
    with pytest.raises(OSError,match='publication failure'):producer.review_and_publish(**args)
    trial=Path(args['binding']['trial'])
    assert (trial/'audit-attempt.json').exists() and (trial/'audit-failure.json').exists()


@pytest.mark.parametrize('name',['audit-attempt.json','audit.json.tmp','audit-failure.json'])
def test_any_existing_attempt_blocks_expensive_work(prepared,name):
    args,events,_,_=prepared
    (Path(args['binding']['trial'])/name).write_text('{}')
    with pytest.raises(ValueError,match='cannot be retried'):producer.review_and_publish(**args)
    assert events==[]


@pytest.mark.parametrize('bad',['fallback','unavailable','token_hash'])
def test_actual_runtime_or_input_failure_is_not_a_retryable_attempt(prepared,monkeypatch,bad):
    args,events,_,_=prepared
    if bad=='fallback':monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK','1')
    elif bad=='unavailable':monkeypatch.setattr(torch.backends.mps,'is_available',lambda:False)
    else:args['expected_token_sha256']='0'*64
    with pytest.raises((ValueError,RuntimeError)):producer.review_and_publish(**args)
    assert 'new' not in events
    assert (Path(args['binding']['trial'])/'audit-attempt.json').exists()
    assert (Path(args['binding']['trial'])/'audit-failure.json').exists()
    with pytest.raises(ValueError,match='cannot be retried'):producer.review_and_publish(**args)
