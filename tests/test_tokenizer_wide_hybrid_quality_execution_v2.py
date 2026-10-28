"""Mocked scientific phases with real bounded writers/recorder; no MPS model."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from src.research import tokenizer_wide_hybrid_quality_execution_v2 as execution
from src.research import tokenizer_wide_hybrid_resources as resources
from src.research.protocol import digest,file_digest
from src.research.tokenizer_wide_hybrid_plan_v2 import request_options
from test_tokenizer_wide_hybrid_metadata_writer import writer


@pytest.fixture(params=[0,9,15,23])
def case(writer,monkeypatch,request):
    slot=request.param;stem=f'quality/{slot:02d}'
    evidence=writer.root/'evidence';trial=evidence/stem
    spec=request_options(writer.root)['quality'][str(slot)][-1]['spec']
    corpus=writer.publish_json('evidence/meta/corpus.json',{'synthetic':True})
    cfg=dict(model={'synthetic':True},data={'synthetic':True,'path':str(corpus)},training=dict(max_iters=1200,
        checkpoint_every=0,keep_best_checkpoint=False,resume_from=None,max_train_seconds=None,
        checkpoint_dir=str(trial/'checkpoints'),metrics_file=str(trial/'metrics.jsonl'),
        summary_file=str(trial/'training-summary.json'),weight_averaging={'synthetic':True}))
    request=dict(spec=spec,config=cfg,result_path=str(trial/'result.json'))
    tokenizer={'synthetic':True};token_sha=digest('synthetic96');contract=dict(tokenizer=tokenizer,
        model=cfg['model'],weight_averaging=cfg['training']['weight_averaging'])
    plan={'tokenizer':tokenizer,'training_token_sha256':token_sha}
    planpath=writer.publish_json('evidence/meta/plan.json',plan);plansha=file_digest(planpath)
    projection=json.loads(execution.PROJECTION.read_text())
    # Preserve the exact source bytes so the frozen-copy equality is exercised.
    projpath=evidence/'meta/projection.json';projpath.write_bytes(execution.PROJECTION.read_bytes())
    pref=dict(path='meta/projection.json',sha256=file_digest(projpath))
    rq=writer.publish_json('evidence/'+stem+'/request.json',request);rqsha=file_digest(rq)
    bound=dict(requests={'quality':{str(slot):[request]}},files={str(planpath):plansha,str(projpath):pref['sha256'],
        str(execution.PROJECTION.resolve()):pref['sha256']})
    order=[]
    def bind(p,sha):
        assert p==planpath and sha==plansha
        order.append('bind');return copy.deepcopy(bound)
    monkeypatch.setattr(execution,'bind_campaign',bind)
    monkeypatch.setattr(resources,'observation',lambda root:dict(peak_rss_bytes=1000,mps_driver_bytes=2000,
                                                               free_bytes=64*1024**3))
    monkeypatch.setattr(torch.mps,'empty_cache',lambda:None)
    def inputs(r,t,s):
        assert r==request and t==tokenizer and s==token_sha
        order.append('inputs');return contract
    monkeypatch.setattr(execution,'_inputs',inputs)
    def train(c,*,output_sink):
        order.append('train');output_sink.validate_config(c)
        metric=dict(step=0,loss=1.);output_sink.metric(trial/'metrics.jsonl',metric)
        path=trial/'checkpoints/final_001200_1.pt'
        payload=dict(cfg=OmegaConf.to_container(c,resolve=True),step=1200,model_state={'w':torch.ones(1)},
                     optimizer_state={},training_state={'contract':contract})
        output_sink.checkpoint(path,payload)
        result=dict(completed_steps=1200,final_checkpoint=str(path),metrics=[metric],weight_averaging=dict(snapshots=41))
        output_sink.summary(trial/'training-summary.json',result)
        return result
    monkeypatch.setattr(execution,'run_training',train)
    def export(raw,path,h,*,payload_writer):
        order.append('export');assert h==200
        payload_writer(path,dict(cfg=cfg,model_state={'w':torch.ones(1)},averaged_weights={'half_life':200},
            parent_checkpoint=dict(path=str(raw),sha256=file_digest(raw))))
    monkeypatch.setattr(execution,'export_averaged_checkpoint',export)
    def evaluate(w,stage,index,r,rsha,osha,c,ts,*,started):
        order.append('evaluate');assert w is writer and stage=='quality' and index==slot and c==contract
        assert (trial/'training-outputs.json').exists() and started>0
        from src.research.tokenizer_wide_hybrid_pilot_audit_v2 import SAMPLE_SETTINGS
        from test_tokenizer_wide_hybrid_retention_audit import score
        raw=trial/'checkpoints/final_001200_1.pt';h200=trial/'checkpoints/h200.pt'
        files={str(p):file_digest(p) for p in (trial/'request.json',trial/'training-outputs.json',
            trial/'training-summary.json',corpus,raw,h200)}
        writer.publish_json('evidence/'+stem+'/evaluation-attempt.json',dict(kind='wide_hybrid_evaluation_v2',
            status='started',request_sha256=rsha,training_outputs_sha256=osha,
            training_contract_sha256=digest(c),training_token_sha256=ts,files=files,sample_settings=SAMPLE_SETTINGS))
        result=dict(r['spec'],status='completed',holdout_consumed=False,request_sha256=rsha,
            training=json.loads((trial/'training-summary.json').read_text()),
            raw=score(1,raw,file_digest(raw)),h200=score(1,h200,file_digest(h200)))
        result['h200'].update(sample='synthetic',sample_settings=SAMPLE_SETTINGS)
        path=writer.publish_json('evidence/'+stem+'/result.json',result)
        return dict(path=stem+'/result.json',sha256=file_digest(path))
    monkeypatch.setattr(execution,'evaluate_and_publish',evaluate)
    def tensor(w,index,binding,*,projection_ref,resources,training_token_sha256):
        order.append('tensor');assert resources.thread.is_alive() and not resources.stop.is_set()
        assert resources.result_sha is None and index==slot and projection_ref==pref
        path=writer.publish_json('evidence/'+stem+'/tensor-audit.json',{'synthetic':True})
        return dict(path=stem+'/tensor-audit.json',sha256=file_digest(path))
    monkeypatch.setattr(execution,'review_tensors',tensor)
    return SimpleNamespace(writer=writer,trial=trial,args=(writer,slot,plansha,rqsha),pref=pref,
                           order=order,bound=bound,contract=contract)


def run(case):return execution.execute_quality_outputs(*case.args,projection_ref=case.pref)


def test_composition_orders_all_phases_with_bounded_outputs_and_resource_binding(case):
    ref=run(case)
    assert case.order==['bind','inputs','train','export','evaluate','tensor','bind']
    report=json.loads((case.trial/'worker-outputs.json').read_text())
    assert ref['sha256']==file_digest(case.trial/'worker-outputs.json')
    assert report['dispatch_authorized'] is False and report['stage_audits_bound'] is False
    assert set(report['artifacts'])=={'training_outputs','evaluation_attempt','result','resources','tensor'}
    resource=json.loads((case.trial/'resource-audit.json').read_text())
    assert resource['status']=='completed' and resource['limits']['timeout_seconds']==7200
    assert resource['result_sha256']==report['artifacts']['result']['sha256']
    assert len(list((case.trial/'checkpoints').glob('*.pt')))==2
    with pytest.raises(ValueError,match='retried'):run(case)


@pytest.mark.parametrize('phase',['inputs','train','export','evaluate','tensor','end_bind'])
def test_phase_failure_is_terminal_and_preserves_preceding_outputs(case,monkeypatch,phase):
    names=dict(inputs='_inputs',train='run_training',export='export_averaged_checkpoint',
               evaluate='evaluate_and_publish',tensor='review_tensors',end_bind='bind_campaign')
    previous=getattr(execution,names[phase]);calls=0
    def fail(*a,**kw):
        nonlocal calls
        calls+=1
        if phase=='end_bind' and calls==1:return previous(*a,**kw)
        raise RuntimeError('synthetic phase failure')
    monkeypatch.setattr(execution,names[phase],fail)
    with pytest.raises(RuntimeError,match='phase failure'):run(case)
    failure=json.loads((case.trial/'worker-failure.json').read_text())
    assert not failure['retry_allowed'] and not (case.trial/'worker-outputs.json').exists()
    resource=json.loads((case.trial/'resource-audit.json').read_text())
    assert resource['status']==('completed' if phase=='end_bind' else 'failed')
    assert (case.trial/'worker-attempt.json').exists()
    if phase in ('evaluate','tensor','end_bind'):
        assert len(list((case.trial/'checkpoints').glob('*.pt')))==2
    monkeypatch.setattr(execution,names[phase],previous)
    with pytest.raises(ValueError):run(case)


@pytest.mark.parametrize('bad',['plan','request','projection','slot','prior','checkpoint'])
def test_preflight_rejects_before_any_scientific_phase(case,monkeypatch,bad):
    if bad=='plan':
        def fail(*a):raise ValueError('numerical evidence unavailable')
        monkeypatch.setattr(execution,'bind_campaign',fail)
    elif bad=='request':(case.trial/'request.json').write_text('{}')
    elif bad=='projection':case.pref['sha256']='0'*64
    elif bad=='slot':case.args=(case.writer,True,*case.args[2:])
    elif bad=='prior':(case.trial/'worker-failure.json').write_text('{}')
    else:
        (case.trial/'checkpoints').mkdir();(case.trial/'checkpoints/old.pt').write_bytes(b'old')
    with pytest.raises(ValueError):run(case)
    assert not any(p in case.order for p in ('inputs','train','export','evaluate'))


@pytest.mark.parametrize('bad',['request','attempt','projection','stop','evaluation_identity'])
def test_changed_evidence_between_phases_cannot_complete(case,monkeypatch,bad):
    previous=execution.evaluate_and_publish
    def evaluate(*a,**kw):
        ref=previous(*a,**kw)
        if bad=='request':(case.trial/'request.json').write_text('{}')
        elif bad=='attempt':(case.trial/'worker-attempt.json').write_text('{}')
        elif bad=='projection':(case.writer.root/'evidence/meta/projection.json').write_text('{}')
        elif bad=='stop':(case.writer.root/'STOP').touch()
        else:
            path=case.trial/'evaluation-attempt.json';v=json.loads(path.read_text())
            v['training_contract_sha256']='0'*64;path.write_text(json.dumps(v))
        return ref
    monkeypatch.setattr(execution,'evaluate_and_publish',evaluate)
    with pytest.raises(ValueError):run(case)
    assert not (case.trial/'worker-outputs.json').exists()


def test_frozen_options_do_not_accept_arbitrary_request_or_ambiguous_binding(case):
    slot=str(case.args[1]);case.bound['requests']['quality'][slot]=[]
    with pytest.raises(ValueError,match='exact frozen slot option'):run(case)
    assert 'train' not in case.order


def test_body_does_not_perform_selection_or_stage_acceptance(case):
    run(case)
    report=json.loads((case.trial/'worker-outputs.json').read_text())
    assert not report['stage_audits_bound'] and not report['adopted']
    assert not (case.writer.root/'evidence/meta/lr-selection.json').exists()


def test_body_outputs_match_independent_producer_binding(case):
    from src.research.tokenizer_wide_hybrid_pilot_audit_v2 import _worker_review
    from src.research.tokenizer_wide_hybrid_retention import allocations
    run(case)
    root=case.writer.root/'evidence';stem=str(case.trial.relative_to(root))
    evidence={str(p.relative_to(root)):file_digest(p) for p in case.trial.rglob('*')
              if p.is_file() and p.suffix!='.pt'}
    request=json.loads((case.trial/'request.json').read_text())
    result=json.loads((case.trial/'result.json').read_text())
    training=json.loads((case.trial/'training-outputs.json').read_text())
    pair={k:dict(training[k],path=str(Path(training[k]['path']).relative_to(root))) for k in ('raw','h200')}
    context=dict(plan_sha256=case.args[2],corpus_sha256=file_digest(Path(request['config']['data']['path'])))
    allowance=allocations(json.loads(execution.PROJECTION.read_text()))[request['spec']['family'],request['spec']['divisor']]
    _worker_review(root,stem,case.args[1],context,request,result,pair,evidence,case.pref,
                   digest('synthetic96'),allowance,stage='quality')


def test_frozen_binding_does_not_hold_parent_metadata_namespace(case,monkeypatch):
    original=execution.bind_campaign;calls=[]
    def bind(*a,**kw):
        # Parent control updates must remain possible while source hashes run.
        path=case.writer.root/'control/binding-observation.json'
        case.writer.publish_json('control/binding-observation.json',{'call':len(calls)},replace=path.exists())
        calls.append(True);return original(*a,**kw)
    monkeypatch.setattr(execution,'bind_campaign',bind)
    run(case)
    assert len(calls)==2
