"""Independent parent read-back with synthetic endpoints/resource observations.

Metadata, coverage arithmetic, trace provenance and ledger acceptance are real.
Tensor/model/input checks use explicit doubles here; separate audit tests cover
real activation tensors. These fixtures cannot certify a research pilot.
"""
import copy
import json

import pytest

from src.research import tokenizer_activation_worker as w
from src.research import tokenizer_activation_audit as a
from src.research.protocol import digest
from src.research.tokenizer_activation_session import reference, KIND
from src.research.tokenizer_wide_hybrid_checkpoint_writer import CheckpointWriter
from src.research.tokenizer_wide_hybrid_resources import _header
from tests.test_tokenizer_activation_session import ledger,publish
from tests.test_tokenizer_activation_audit import measured_fixture,trace


def completed(ledger,monkeypatch):
    rq=ledger.reserve();request=ledger.document(rq);stem='evidence/pilots/00';trial=ledger.root/stem
    _,measured,outputs,summary,metrics=measured_fixture()
    cfg=request['config'];outputs['config_sha256']=digest(cfg)
    writer=CheckpointWriter(ledger.writer,'pilots',0,1024**2)
    for role,name in [('raw','final_000020_1.pt'),('h200','h200.pt')]:
        stream=writer.open(name);stream.write(b'explicit synthetic test bytes');row=stream.publish()
        outputs[role]=row;measured[role]['checkpoint']=row['path'];measured[role]['checkpoint_sha256']=row['sha256']
    weights={role:reference(ledger.root,str((trial/'checkpoints'/name).relative_to(ledger.root)))
             for role,name in [('raw','final_000020_1.pt'),('h200','h200.pt')]}
    summary['final_checkpoint']=outputs['raw']['path'];summary['weight_averaging']['spec']=cfg['training']['weight_averaging']
    measured['training_contract']={'synthetic':True}
    measured['cache']={role:[dict(batch=b,state_bytes=b*w.CACHE_BYTES,passed=True,max_absolute_error=0.) for b in (1,8)]
                       for role in ('raw','h200')}
    for name,value in [('training-outputs.json',outputs),('training-summary.json',summary),('measurements.json',measured),
                       ('producer-tensors.json',{'synthetic':True}),('worker-attempt.json',{'request':rq})]:
        publish(ledger,stem+'/'+name,value)
    with ledger.writer.open_append(stem+'/metrics.jsonl') as out:
        for row in metrics:out.write((json.dumps(row)+'\n').encode())
    result=publish(ledger,stem+'/result.json',dict(kind=KIND,request=rq,record=dict(request['spec'],status='completed'),
        holdout_consumed=False,status='completed',request_sha256=rq['sha256'],measurements=reference(ledger.root,stem+'/measurements.json')))
    report,events=trace(True)
    report.update(plan_sha256=ledger.plan['sha256'],request_sha256=rq['sha256'],result_sha256=result['sha256'])
    events[0]=_header(ledger.plan['sha256'],rq['sha256'],a.RESOURCE_KIND,report['limits'])
    events[-1]['result_sha256']=result['sha256']
    with ledger.writer.open_append(stem+'/resource-events.jsonl') as out:
        for row in events:out.write((json.dumps(row)+'\n').encode())
    report['events']['sha256']=w.file_digest(trial/'resource-events.jsonl')
    publish(ledger,stem+'/resource-audit.json',report)
    publish(ledger,stem+'/worker-outputs.json',dict(kind='activation_worker_v1',status='complete_pending_parent_review',
        request=rq,result=result,dispatch_authorized=False))
    monkeypatch.setattr(w,'input_contract',lambda *args:{'synthetic':True})
    monkeypatch.setattr(w,'review_pair',lambda *args:({'synthetic':True},weights))
    return rq,trial


def test_independent_parent_binds_complete_producer_and_advances_once(ledger,monkeypatch):
    rq,trial=completed(ledger,monkeypatch)
    ref=w.accept_outputs(ledger,rq,{},lambda:None)
    assert ref==reference(ledger.root,'evidence/pilots/00/acceptance.json')
    state=ledger.inspect()
    assert len(state['pilots'])==1 and state['next']['spec']['family']=='gelu_exact'
    assert state['dispatch_authorized'] is False
    with pytest.raises(ValueError):w.accept_outputs(ledger,rq,{},lambda:None)


@pytest.mark.parametrize('case',['coverage','cache','trace','worker_status','changed_during_review'])
def test_parent_rejects_incomplete_or_altered_evidence_before_acceptance(ledger,monkeypatch,case):
    rq,trial=completed(ledger,monkeypatch)
    if case in ('coverage','cache'):
        path=trial/'measurements.json';doc=json.loads(path.read_text())
        if case=='coverage':doc['h200']['block_target_counts'][-1]-=1
        else:doc['cache']['raw'][1]['passed']=False
        path.write_text(json.dumps(doc))
        # Update the child result references too so the independent semantic
        # checks, not just an old outer hash, must reject this corruption.
        result_path=trial/'result.json';result=json.loads(result_path.read_text())
        result['measurements']=reference(ledger.root,'evidence/pilots/00/measurements.json')
        result_path.write_text(json.dumps(result))
        child_path=trial/'worker-outputs.json';child=json.loads(child_path.read_text())
        child['result']=reference(ledger.root,'evidence/pilots/00/result.json');child_path.write_text(json.dumps(child))
    elif case=='trace':
        path=trial/'resource-events.jsonl';path.write_text(path.read_text().splitlines()[0]+'\n')
    elif case=='worker_status':
        path=trial/'worker-outputs.json';doc=json.loads(path.read_text());doc['status']='failed';path.write_text(json.dumps(doc))
    calls=[]
    def recheck():
        calls.append(1)
        if case=='changed_during_review' and len(calls)==2:
            path=trial/'metrics.jsonl';path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError):w.accept_outputs(ledger,rq,{},recheck)
    assert not (trial/'acceptance.json').exists()
    assert ledger.inspect()['next']['kind']=='pending'
