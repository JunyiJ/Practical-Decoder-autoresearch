"""Import contract tests with explicit provenance doubles/disposable JSON only."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.activation_inference_recovery_v1 import source_binding as binding
from src.research.protocol import digest
from src.research.tokenizer_activation_session import reference


def ref(name):
    return dict(path=name,size=1,sha256='a'*64)


def complete_state():
    return dict(next={'kind':'review_final','adopted':False},pending=None,failed=False,
        pilots=[{}]*2,records=[{}]*12,reviews=[ref(f'review/{i}') for i in range(14)],pairs=[[] for _ in range(14)],
        pilot_boundary=True,screen_boundary=True,selection=dict.fromkeys(binding.FAMILIES,.00125),
        selection_ref=dict(path='evidence/meta/activation-selection.json',size=1,sha256=binding.SELECTION_SHA256))


def test_exact_complete_state():
    binding._complete_state(complete_state())


@pytest.mark.parametrize('key,value',[
    ('next',{'kind':'quality'}),('pending',{}),('failed',True),('pilots',[{}]),('records',[{}]*11),
    ('reviews',[{}]*13),('pairs',[[]]*13),('pilot_boundary',False),('screen_boundary',False),
    ('selection',dict(silu=.00125,gelu_exact=.0025)),
    ('selection_ref',dict(path='evidence/meta/activation-selection.json',size=1,sha256='b'*64)),
    ('selection_ref',dict(path='evidence/meta/other.json',size=1,sha256=binding.SELECTION_SHA256))])
def test_incomplete_or_changed_state_rejected(key,value):
    state=complete_state();state[key]=value
    with pytest.raises(ValueError):binding._complete_state(state)


@pytest.fixture
def pair_fixture():
    state=complete_state();docs={};verified=[]
    for i,spec in enumerate(binding.confirmation_specs(state['selection']),6):
        request=ref(f'evidence/quality/{i:02d}/request.json')
        audits={role:ref(f'audits/{i}/{role}') for role in ('training','evaluation','tensors','resources')}
        for audit in audits.values():docs[audit['path']]={'reviewer':'activation_independent_v1'}
        weights={role:ref(f'evidence/quality/{i:02d}/checkpoints/{role}.pt') for role in ('raw','h200')}
        docs[state['reviews'][i+2]['path']]=dict(request=request,audits=audits,weights=weights)
        docs[request['path']]={'spec':spec}
    ledger=SimpleNamespace(plan=ref('evidence/meta/activation-plan.json'),
        document=lambda r:copy.deepcopy(docs[r['path']]),verify=lambda r:verified.append(r))
    plan={'inputs':{'frozen':'input-identity'}}
    dispatches=[ref(f'dispatch/{i}') for i in range(14)]
    return ledger,state,plan,dispatches,docs,verified


def test_loader_pair_shape_and_order_preserved(pair_fixture):
    ledger,state,plan,dispatches,_,verified=pair_fixture
    result=binding._pairs(ledger,state,plan,dispatches)
    assert list(result)==['3611','3612','3613']
    assert len(verified)==12
    for seed,pair in result.items():
        assert pair['seed']==int(seed) and pair['plan']==ledger.plan
        assert pair['inputs']==plan['inputs'] and pair['dispatches']==dispatches
        assert pair['contract']==binding.TIMING
        assert set(pair['bindings'])=={'silu','gelu_exact'}
        assert all(set(item['weights'])=={'raw','h200'} for item in pair['bindings'].values())


@pytest.mark.parametrize('bad',['request_spec','request_path','missing_audit','wrong_reviewer','missing_raw'])
def test_pair_import_rejects_altered_provenance(pair_fixture,bad):
    ledger,state,plan,dispatches,docs,_=pair_fixture
    review=docs[state['reviews'][8]['path']]
    if bad=='request_spec':docs[review['request']['path']]['spec']['seed']=9999
    elif bad=='request_path':
        original=review['request']['path'];review['request']['path']='wrong/request.json'
        docs['wrong/request.json']=docs[original]
    elif bad=='missing_audit':del review['audits']['resources']
    elif bad=='wrong_reviewer':docs[review['audits']['training']['path']]['reviewer']='self-declared'
    else:del review['weights']['raw']
    with pytest.raises(ValueError):binding._pairs(ledger,state,plan,dispatches)


@pytest.fixture
def failure_fixture(tmp_path,monkeypatch):
    root=tmp_path/'original';root.mkdir()
    def put(name,value):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
        return reference(root,name)
    process=dict(status='failed',returncode=1,pid=12345678)
    process_ref=put('evidence/inference/3511/process/process.json',process)
    fail_ref=put('evidence/inference/3511/dispatch-failure.json',{'retry_allowed':False})
    copied={r['path']:r for r in (process_ref,fail_ref)}
    plan=ref('evidence/meta/activation-plan.json')
    audit=dict(status='training_complete_inference_failed_no_retry',plan=plan,source_sha256=binding.SOURCE_SHA256,
        holdout_scored=False,retry_allowed=False,formal_final_completed=False,copied=copied,process=process)
    anchor=tmp_path/'failure-review.json';anchor.write_text(json.dumps(audit))
    monkeypatch.setattr(binding,'ROOT',tmp_path);monkeypatch.setattr(binding,'FAILURE_REVIEW',anchor)
    monkeypatch.setattr(binding,'FAILURE_REVIEW_SHA256',reference(tmp_path,anchor.name)['sha256'])
    monkeypatch.setattr(binding,'_group_exists',lambda pid:False)
    def verify(r):
        if reference(root,r['path'])!=r:raise ValueError('Original artifact changed')
        return root/r['path']
    ledger=SimpleNamespace(root=root,plan=plan,verify=verify,document=lambda r:json.loads(verify(r).read_text()))
    return ledger,{'files':dict.fromkeys(copied,{})},anchor


def test_exact_failure_remains_failure(failure_fixture):
    ledger,seen,_=failure_fixture
    _,audit=binding._failed_inference(ledger,seen)
    assert audit['retry_allowed'] is False and audit['formal_final_completed'] is False


@pytest.mark.parametrize('name',['evidence/inference/3511/accepted.json','evidence/inference/3511/result.json',
    'evidence/inference/3512/request.json','control/process-inference-3512.json'])
def test_failed_campaign_cannot_gain_results_or_later_attempts(failure_fixture,name):
    ledger,seen,_=failure_fixture;seen['files'][name]={}
    with pytest.raises(ValueError,match='inventory changed'):binding._failed_inference(ledger,seen)


def test_original_failed_artifact_tampering_rejected(failure_fixture):
    ledger,seen,_=failure_fixture
    (ledger.root/'evidence/inference/3511/dispatch-failure.json').write_text('{"retry_allowed":true}')
    with pytest.raises(ValueError,match='artifact changed'):binding._failed_inference(ledger,seen)


def test_failure_anchor_tampering_rejected(failure_fixture):
    ledger,seen,anchor=failure_fixture;anchor.write_text('{}')
    with pytest.raises(ValueError,match='audit changed'):binding._failed_inference(ledger,seen)


def test_live_original_process_group_blocks_import(failure_fixture,monkeypatch):
    ledger,seen,_=failure_fixture
    monkeypatch.setattr(binding,'_group_exists',lambda pid:True)
    with pytest.raises(ValueError,match='remains live'):binding._failed_inference(ledger,seen)


def test_frozen_import_comparison_replays_and_rejects_changes(monkeypatch):
    actual={'files':{'control/a':ref('control/a')},'dispatch_authorized':False};calls=[]
    def replay(ledger):calls.append(ledger);return copy.deepcopy(actual)
    monkeypatch.setattr(binding,'collect_source_import',replay)
    assert binding.verify_source_import('ledger',copy.deepcopy(actual))==actual
    changed=copy.deepcopy(actual);changed['files']['control/a']['sha256']='b'*64
    with pytest.raises(ValueError,match='import differs'):binding.verify_source_import('ledger',changed)
    with pytest.raises(ValueError,match='import differs'):binding.verify_source_import('ledger',None)
    assert calls==['ledger']*3
