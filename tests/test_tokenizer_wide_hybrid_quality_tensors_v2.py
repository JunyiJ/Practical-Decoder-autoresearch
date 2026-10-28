"""Tiny real mapped tensor checks, mocked metadata/model and MPS observations."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.research import tokenizer_wide_hybrid_quality_tensors_v2 as producer
from src.research import tokenizer_wide_hybrid_tensors as tensors
from src.research import tokenizer_wide_hybrid_resources as resource_module
from src.research.protocol import digest,file_digest
from src.research.tokenizer_wide_hybrid_campaign_resources import CampaignQualityResourceRecorder
from src.research.tokenizer_wide_hybrid_plan_v2 import PROJECTION
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_tensors import pair


@pytest.fixture
def prepared(writer,monkeypatch):
    root=writer.root/'evidence';stem='quality/00';trial=root/stem
    request={'spec':{'synthetic':True},'config':{'synthetic':True}}
    contract={'tokenizer':{'synthetic':True}}
    rq=writer.publish_json('evidence/'+stem+'/request.json',request);rqsha=file_digest(rq)
    projection=writer.publish_json('evidence/meta/projection.json',json.loads(PROJECTION.read_text()))
    pref=dict(path='meta/projection.json',sha256=file_digest(projection))
    (trial/'checkpoints').mkdir()
    raw,export=pair();torch.save(raw,trial/'checkpoints/raw.pt');torch.save(export,trial/'checkpoints/h200.pt')
    cp={k:dict(path=str(trial/f'checkpoints/{k}.pt'),size=(trial/f'checkpoints/{k}.pt').stat().st_size,
               sha256=file_digest(trial/f'checkpoints/{k}.pt')) for k in ('raw','h200')}
    training_record=dict(cp,kind='wide_hybrid_training_outputs_v2',status='completed',
                         config_sha256=digest(request['config']),numerical_audits_passed=False,dispatch_authorized=False)
    training=writer.publish_json('evidence/'+stem+'/training-outputs.json',training_record)
    result=dict(status='completed',holdout_consumed=False,request_sha256=rqsha,
                **{k:dict(checkpoint=v['path'],checkpoint_sha256=v['sha256']) for k,v in cp.items()})
    rs=writer.publish_json('evidence/'+stem+'/result.json',result);rssha=file_digest(rs)
    writer.publish_json('evidence/'+stem+'/evaluation-attempt.json',dict(kind='wide_hybrid_evaluation_v2',status='started',
        request_sha256=rqsha,training_outputs_sha256=file_digest(training),training_contract_sha256=digest(contract),
        training_token_sha256=digest('synthetic96')))
    monkeypatch.setattr(producer,'request_options',lambda root:{'quality':{'0':[request]}})
    monkeypatch.setattr(resource_module,'observation',lambda root:dict(peak_rss_bytes=1000,mps_driver_bytes=2000,free_bytes=64*1024**3))
    calls=[]
    def metadata(*args):
        calls.append(args)
        return dict(checkpoint_hashes={k:v['sha256'] for k,v in cp.items()})
    monkeypatch.setattr(tensors,'review_trial',metadata)
    args=dict(writer=writer,slot=0,binding=dict(request=request,request_sha256=rqsha,result_sha256=rssha,
        training_contract=contract),projection_ref=pref,training_token_sha256=digest('synthetic96'))
    return SimpleNamespace(writer=writer,root=root,trial=trial,args=args,calls=calls,plan=digest('synthetic plan'),
                           raw=raw,export=export,cp=cp)


def run(case):
    with CampaignQualityResourceRecorder(case.writer,0,case.plan,case.args['binding']['request_sha256']) as guard:
        ref=producer.review_and_publish(**case.args,resources=guard)
        guard.mark_success(case.args['binding']['result_sha256'])
    return ref


def test_real_mapping_values_bounded_publication_and_readonly_provenance(prepared):
    before={str(p):file_digest(p) for p in prepared.trial.rglob('*.pt')}
    ref=run(prepared);report=json.loads((prepared.root/ref['path']).read_text())
    assert report['values']['exact_export_matches_stored_ema']
    assert not report['values']['ema_recurrence_audited'] and not report['adopted']
    assert len(prepared.calls)==1 and report['kind']=='quality'
    evidence={str(p.relative_to(prepared.root)):file_digest(p) for p in prepared.trial.rglob('*') if p.is_file()}
    producer.verify_producer(prepared.root,'quality/00',0,report,evidence=evidence,
        context={'plan_sha256':prepared.plan},projection_ref=prepared.args['projection_ref'],
        training_token_sha256=digest('synthetic96'))
    assert before=={str(p):file_digest(p) for p in prepared.trial.rglob('*.pt')}
    assert json.loads((prepared.trial/'resource-audit.json').read_text())['status']=='completed'


@pytest.mark.parametrize('bad',['slot','options','guard','contract','token','receipt','STOP','projection'])
def test_invalid_binding_never_reaches_mapping(prepared,monkeypatch,bad):
    if bad=='slot':prepared.args['slot']=True
    elif bad=='options':monkeypatch.setattr(producer,'request_options',lambda root:{'quality':{'0':[]}})
    elif bad=='contract':prepared.args['binding']['training_contract']={}
    elif bad=='token':prepared.args['training_token_sha256']='0'*64
    elif bad=='receipt':(prepared.trial/'training-outputs.json').write_text('{}')
    elif bad=='STOP':(prepared.writer.root/'STOP').touch()
    elif bad=='projection':prepared.args['projection_ref']['sha256']='0'*64
    if bad=='guard':
        with pytest.raises(ValueError,match='Active matching'):producer.review_and_publish(**prepared.args,resources=None)
    else:
        with pytest.raises(ValueError):run(prepared)
    assert not prepared.calls and not (prepared.trial/'tensor-audit.json').exists()


@pytest.mark.parametrize('bad',['nan','wrong_ema','wrong_buffer'])
def test_actual_tensor_failure_preserved_without_success(prepared,bad):
    exported=copy.deepcopy(prepared.export)
    if bad=='nan':exported['model_state']['weight'][0]=float('nan')
    elif bad=='wrong_ema':exported['model_state']['weight'][0]+=1
    else:exported['model_state']['buffer'][0]+=1
    # Publish this synthetic invalid pair as an input, then bind its real hash.
    path=prepared.trial/'checkpoints/h200.pt';torch.save(exported,path)
    prepared.cp['h200'].update(size=path.stat().st_size,sha256=file_digest(path))
    training=json.loads((prepared.trial/'training-outputs.json').read_text());training.update(prepared.cp)
    (prepared.trial/'training-outputs.json').write_text(json.dumps(training))
    result=json.loads((prepared.trial/'result.json').read_text());result['h200']['checkpoint_sha256']=file_digest(path)
    (prepared.trial/'result.json').write_text(json.dumps(result))
    prepared.args['binding']['result_sha256']=file_digest(prepared.trial/'result.json')
    ep=prepared.trial/'evaluation-attempt.json';ev=json.loads(ep.read_text())
    ev['training_outputs_sha256']=file_digest(prepared.trial/'training-outputs.json');ep.write_text(json.dumps(ev))
    with pytest.raises(ValueError):run(prepared)
    assert len(prepared.calls)==1 and not (prepared.trial/'tensor-audit.json').exists()
    failure=json.loads((prepared.trial/'tensor-failure.json').read_text())
    assert not failure['retry_allowed']
    assert json.loads((prepared.trial/'resource-audit.json').read_text())['status']=='failed'
    assert path.exists() and (prepared.trial/'tensor-attempt.json').exists()


@pytest.mark.parametrize('target',['request.json','result.json','checkpoints/raw.pt','tensor-attempt.json','STOP'])
def test_changed_input_or_stop_during_mapping_is_terminal(prepared,monkeypatch,target):
    original=producer.review_values
    def changed(*a,**kw):
        report=original(*a,**kw)
        if target=='STOP':(prepared.writer.root/'STOP').touch()
        else:
            with (prepared.trial/target).open('ab') as f:f.write(b'changed')
        return report
    monkeypatch.setattr(producer,'review_values',changed)
    with pytest.raises((ValueError,json.JSONDecodeError)):run(prepared)
    assert not (prepared.trial/'tensor-audit.json').exists()
    assert (prepared.trial/'tensor-failure.json').exists()


def test_successful_attempt_cannot_be_repeated_inside_same_guard(prepared):
    with CampaignQualityResourceRecorder(prepared.writer,0,prepared.plan,prepared.args['binding']['request_sha256']) as guard:
        producer.review_and_publish(**prepared.args,resources=guard)
        with pytest.raises(ValueError,match='retried'):
            producer.review_and_publish(**prepared.args,resources=guard)
        guard.mark_success(prepared.args['binding']['result_sha256'])
    assert len(prepared.calls)==1


def test_closed_guard_cannot_audit(prepared):
    run(prepared)
    with pytest.raises(ValueError):producer.review_and_publish(**prepared.args,resources=None)
    assert len(prepared.calls)==1


def test_oversized_report_cannot_escape_metadata_cap(prepared,monkeypatch):
    original=producer.review_values
    def oversized(*a,**kw):
        result=original(*a,**kw);result['synthetic_oversized']='x'*(256*1024);return result
    monkeypatch.setattr(producer,'review_values',oversized)
    with pytest.raises(ValueError,match='cap'):run(prepared)
    assert not (prepared.trial/'tensor-audit.json').exists()
    assert (prepared.trial/'tensor-attempt.json').exists()
    assert (prepared.trial/'tensor-audit.json.tmp').exists()
    assert json.loads((prepared.trial/'resource-audit.json').read_text())['status']=='failed'


@pytest.mark.parametrize('bad',['plan','request','slot','stopped','completed','limits'])
def test_mismatched_active_guard_cannot_audit(prepared,bad):
    with CampaignQualityResourceRecorder(prepared.writer,0,prepared.plan,prepared.args['binding']['request_sha256']) as guard:
        attribute={'plan':'plan_sha','request':'request_sha','slot':'trial','limits':'LIMITS'}.get(bad)
        # Plan identity is established by the parent. The producer records it;
        # invalid digests are rejected here and any different valid plan is
        # independently rejected by verify_producer against frozen context.
        previous=getattr(guard,attribute) if attribute else None
        if bad=='plan':guard.plan_sha='invalid'
        elif bad=='request':guard.request_sha='0'*64
        elif bad=='slot':guard.trial=prepared.root/'quality/01'
        elif bad=='stopped':guard.stop.set()
        elif bad=='completed':guard.result_sha=prepared.args['binding']['result_sha256']
        else:guard.LIMITS=dict(guard.LIMITS,timeout_seconds=8000)
        try:
            with pytest.raises(ValueError):producer.review_and_publish(**prepared.args,resources=guard)
        finally:
            if attribute:setattr(guard,attribute,previous)
            guard.stop.clear();guard.result_sha=None
        assert not prepared.calls
        guard.mark_success(prepared.args['binding']['result_sha256'])
