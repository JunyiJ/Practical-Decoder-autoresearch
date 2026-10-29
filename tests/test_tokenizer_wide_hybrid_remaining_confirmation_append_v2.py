"""Explicit fourth-fresh preparation with disposable bytes; no model or child."""
import json
import os
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_remaining_confirmation_append_v2 as append
from src.research.protocol import file_digest
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_later_confirmation_history_v2 import setup as completed


PATHS=append.paths(18)


def setup(writer,monkeypatch):
    args=completed(writer,monkeypatch);args['slot']=18;root=writer.root/'evidence'
    # The next controller may differ from the original completed child's parent.
    pid=os.getpid()+500000;monkeypatch.setattr(append,'os',SimpleNamespace(getpid=lambda:pid))
    attempt=dict(kind=append.KIND,status='started',slot=18,parent_pid=pid,
        plan_sha256=args['plan_sha256'],projection_ref=args['projection_ref'],
        rank_completion_ref=args['rank_completion_ref'],prior_dispatch_ref=args['completion_ref'],
        dispatch_authorized=False,deletion_authorized=False)
    guard=dict(kind=append.KIND,status='starting',reason=None,pid=pid,
        limits=dict(timeout_seconds=1800,rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3),
        elapsed_seconds=.1,observation=dict(peak_rss_bytes=1000,free_bytes=64*1024**3))
    path=writer.publish_json('evidence/'+PATHS['attempt'],attempt)
    args['attempt_ref']=dict(path=PATHS['attempt'],sha256=file_digest(path))
    path=writer.publish_json(PATHS['guard'],guard)
    args['guard_ref']=dict(path=PATHS['guard'],sha256=file_digest(path))
    return args


def pending(writer,args):
    root=writer.root/'evidence';saved=append._document(root,args['completion_ref'])
    _,requests=append._frozen(writer,18,args['plan_sha256'],args['projection_ref'],
                                    saved['review']['rates'],saved['review']['selection'])
    path=writer.publish_json('evidence/'+PATHS['request'],requests[-1])
    return dict(args,pending_request_ref=dict(path=PATHS['request'],sha256=file_digest(path)))


def test_exact_attempt_and_pending_request_preserve_historical_audits_and_capacity(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence'
    before=append.history.acceptance._inventory(root)
    initial=append.verify_confirmation_preparation(**args)
    assert initial['preparation_append_verified'] and not initial['pending_request_verified']
    assert len(initial['records'])==18 and len(initial['checkpoint_pairs'])==6
    assert initial['audited_indices']==[9,10,11,15,16,17] and len(initial['stages'])==3
    assert append.history.acceptance._inventory(root)==before
    args=pending(writer,args);before=append.history.acceptance._inventory(root)
    original=append.history.acceptance.bind_storage;calls=[]
    def capacity(*a,**kw):
        assert len(kw['records'])==18 and len(kw['checkpoint_pairs'])==21
        assert PATHS['request'] in kw['current_inventory'];calls.append(1)
        without=dict(kw);without.pop('pending_request_ref')
        with pytest.raises(ValueError,match='Unreviewed trajectory'):original(*a,**without)
        return original(*a,**kw)
    monkeypatch.setattr(append.history.acceptance,'bind_storage',capacity)
    checked=append.verify_confirmation_preparation(**args)
    assert calls==[1] and checked['pending_request_verified']
    for key in ('records','checkpoint_pairs','stages','rates','selection','audited_indices'):
        assert checked[key]==initial[key]
    for key in ('required_free_bytes','projection','occupied_checkpoint_capacity_bytes'):
        assert checked['storage'][key]==initial['storage'][key]
    assert all(checked[k] is False for k in ('adopted','dispatch_authorized','next_stage_authorized'))
    assert len(list(root.rglob('*.pt')))==12
    assert {p.name for p in (root/'quality/18').iterdir()}=={'request.json'}
    assert append.history.acceptance._inventory(root)==before
    # The old public path does not acquire this append capability.
    strict={k:v for k,v in args.items() if k not in ('attempt_ref','guard_ref','pending_request_ref')}
    strict['slot']=17
    with pytest.raises(ValueError):append.history.verify_confirmation_dispatch(**strict)
    guard=append._document(writer.root,args['guard_ref']);guard['status']='finished';guard['elapsed_seconds']=.2
    path=writer.publish_json(PATHS['guard'],guard,replace=True)
    args['guard_ref']=dict(path=PATHS['guard'],sha256=file_digest(path))
    assert append.verify_confirmation_preparation(**args)['pending_request_verified']
    census=append.history.snapshot(writer.root)
    (writer.root.parent/'slot18-append-census.json').write_text(json.dumps(dict(
        metadata_bytes=census['metadata_bytes'],meta_group_bytes=census['groups']['evidence/meta'],
        completed_quality_records=18,pending_slot=18,live_checkpoint_files=12,
        required_free_bytes=checked['storage']['required_free_bytes'],real_panel_fit_proven=False)))


def test_rehashed_claims_and_undeclared_appends_never_get_historical_credit(writer,monkeypatch):
    args=pending(writer,setup(writer,monkeypatch));root=writer.root/'evidence'
    cases={
        'attempt_ref':[('slot',19),('slot',True),('parent_pid',1),('prior_dispatch_ref',{}),
            ('projection_ref',{}),('rank_completion_ref',{}),('dispatch_authorized',True),('deletion_authorized',True)],
        'guard_ref':[('status','failed'),('status','running'),('pid',1),('elapsed_seconds',1800.),
            ('elapsed_seconds',True),('observation',dict(peak_rss_bytes=5*1024**3,free_bytes=64*1024**3)),
            ('observation',dict(peak_rss_bytes=1000,free_bytes=1)),('limits',{})],
        'pending_request_ref':[('spec',{}),('config',{}),('result_path',str(root/'quality/19/result.json'))]}
    for name,mutations in cases.items():
        path=(writer.root if name=='guard_ref' else root)/args[name]['path'];raw=path.read_bytes()
        for key,value in mutations:
            changed=json.loads(raw);changed[key]=value;path.write_text(json.dumps(changed))
            bad=dict(args,**{name:dict(path=args[name]['path'],sha256=file_digest(path))})
            try:
                with pytest.raises((ValueError,KeyError)):append.verify_confirmation_preparation(**bad)
            finally:path.write_bytes(raw)
    for name in ('quality/18/result.json','quality/18/process/launch.json','quality/18/checkpoints/h200.pt',
                 'quality/19/request.json','meta/quality-confirm-startup-18.json','meta/quality-readiness-18.json'):
        path=root/name;parents=[];parent=path.parent
        while not parent.exists():parents.append(parent);parent=parent.parent
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'{}')
        try:
            with pytest.raises(ValueError):append.verify_confirmation_preparation(**args)
        finally:
            path.unlink()
            for parent in parents:parent.rmdir()
    for name in ('control/process-quality-18.json','control/quality-controller-prepare-18.json',
                 'control/quality-confirm-startup-19.json'):
        path=writer.root/name;path.write_bytes(b'{}')
        try:
            with pytest.raises(ValueError):append.verify_confirmation_preparation(**args)
        finally:path.unlink()
    # No unbound pending request or changed original evidence is excused.
    with pytest.raises(ValueError):append.verify_confirmation_preparation(**{**args,'pending_request_ref':None})
    for path in (root/'quality/17/checkpoints/h200.pt',root/'meta/numerical.json',
                 writer.root/'control/process-quality-17.json'):
        raw=path.read_bytes();path.write_bytes(b'{}')
        try:
            with pytest.raises((ValueError,KeyError)):append.verify_confirmation_preparation(**args)
        finally:path.write_bytes(raw)


def test_late_append_guard_and_space_changes_are_rejected_readonly(writer,monkeypatch):
    args=pending(writer,setup(writer,monkeypatch));root=writer.root/'evidence'
    original=append.history.acceptance.bind_storage
    for bad in ('attempt','request','guard','space'):
        reached=[]
        path={'attempt':root/PATHS['attempt'],'request':root/PATHS['request'],'guard':writer.root/PATHS['guard']}.get(bad)
        raw=path.read_bytes() if path else None
        with monkeypatch.context() as patch:
            def late(*a,**kw):
                result=original(*a,**kw)
                reached.append(True)
                if path:path.write_bytes(b'changed during replay')
                else:patch.setattr(append.history,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=8*1024**3)))
                return result
            patch.setattr(append.history.acceptance,'bind_storage',late)
            try:
                with pytest.raises((ValueError,KeyError)):append.verify_confirmation_preparation(**args)
            finally:
                if path:path.write_bytes(raw)
        assert reached==[True]
    (writer.root/'STOP').touch()
    with pytest.raises(ValueError):append.verify_confirmation_preparation(**args)


@pytest.mark.parametrize('slot',[15,16,17])
def test_pending_capability_is_not_exposed_by_immediate_parent_acceptance(writer,slot):
    with pytest.raises(ValueError,match='historical'):
        append.history.acceptance._review_completed(writer,slot,{},{},{},pending_confirmation_request={})
    assert not (writer.root/'evidence').exists()


@pytest.mark.parametrize('slot',[0,15,16,17,19,20,21,22,23,24,-1,True,18.,'18'])
def test_only_exact_slot18_preparation_is_enabled(writer,slot):
    with pytest.raises(ValueError,match='slot18'):
        append.verify_confirmation_preparation(writer,slot,'a'*64,{},projection_ref={},rank_completion_ref={},attempt_ref={},guard_ref={})
    assert not (writer.root/'evidence').exists()


@pytest.mark.parametrize('slot',range(18,24))
def test_shared_names_are_finite_and_cannot_skip_completed_parent(writer,slot):
    paths=append.paths(slot)
    assert paths==dict(attempt=f'meta/quality-confirm-startup-{slot:02d}-attempt.json',
        guard=f'control/quality-confirm-startup-{slot:02d}.json',request=f'quality/{slot:02d}/request.json',
        prior=f'meta/quality-dispatch-{slot-1:02d}.json')
    for saved in ({'slot':slot},{'slot':True},{'slot':slot-2}):
        with pytest.raises(ValueError,match='preceding'):
            append._append(writer,slot,'a'*64,dict(path=paths['prior']),{},{},saved,{})
    assert not (writer.root/'evidence').exists()


@pytest.mark.parametrize('slot',[15,16,17,24,-1,True,18.,'18'])
def test_shared_contract_rejects_other_ranges_and_types(slot):
    with pytest.raises(ValueError):append.paths(slot)
