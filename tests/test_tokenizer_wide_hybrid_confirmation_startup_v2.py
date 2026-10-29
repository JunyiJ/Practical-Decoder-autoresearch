"""Slot16 receipt publication/replay on disposable history; no real training."""
import copy
import json
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_confirmation_startup_v2 as start
from src.research.protocol import file_digest
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_first_confirmation_history_v2 import setup as completed


def setup(writer,monkeypatch):
    args=completed(writer,monkeypatch)
    args['prior_dispatch_ref']=args.pop('completion_ref');args['slot']=16
    monkeypatch.setattr(start,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=64*1024**3)))
    return args


def test_compact_receipt_fully_replays_history_and_capacity_without_child(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence'
    before=start.append.history.acceptance._inventory(root)
    original=start.append.verify_first_confirmation_preparation;calls=[]
    def gate(**kw):
        value=original(**kw)
        calls.append(kw.get('pending_request_ref') is not None)
        assert len(value['records'])==16 and len(value['checkpoint_pairs'])==4 and len(value['stages'])==3
        return value
    monkeypatch.setattr(start.append,'verify_first_confirmation_preparation',gate)
    ref=start.prepare_confirmation_request(**args);saved=start._document(root,ref)
    assert calls==[False,True] and ref['path']==start.RECEIPT
    assert saved['status']=='confirmation_pending_predispatch_review' and saved['slot']==16
    assert (root/ref['path']).stat().st_size<8000
    assert not set(saved)&{'records','checkpoint_pairs','stages','inventory','frozen_files'}
    assert all(saved[k] is False for k in start.NONCLAIMS)
    assert saved['audited_indices']==[9,10,11,15] and len(list(root.rglob('*.pt')))==8
    assert {p.name for p in (root/'quality/16').iterdir()}=={'request.json'}
    assert not (root/'quality/17').exists()
    after=start.append.history.acceptance._inventory(root)
    assert all(after[n]==v for n,v in before.items())
    checked=start.verify_confirmation_request(writer,16,ref)
    assert checked['confirmation_request_verified'] and checked['prior_transport_verified']
    assert checked['selected_and_fresh_prefix_verified'] and not checked['dispatch_authorized']
    assert checked['required_free_bytes']==saved['storage']['required_free_bytes']
    assert start.append.history.acceptance._inventory(root)==after
    with pytest.raises(ValueError,match='cannot be retried'):start.prepare_confirmation_request(**args)
    assert start.append.history.acceptance._inventory(root)==after
    # Neither older public history API silently acquires receipt append support.
    old=dict(writer=writer,plan_sha256=args['plan_sha256'],completion_ref=args['prior_dispatch_ref'],
             projection_ref=args['projection_ref'],rank_completion_ref=args['rank_completion_ref'])
    with pytest.raises(ValueError):start.append.history.verify_first_confirmation_dispatch(**old)
    with pytest.raises(ValueError):start.append.verify_first_confirmation_preparation(**old,
        attempt_ref=saved['attempt'],guard_ref=saved['guard'],pending_request_ref=saved['request'])
    census=start.snapshot(writer.root)
    (writer.root.parent/'slot16-startup-census.json').write_text(json.dumps(dict(receipt_bytes=(root/ref['path']).stat().st_size,
        metadata_bytes=census['metadata_bytes'],meta_group_bytes=census['groups']['evidence/meta'],
        live_checkpoint_files=8,completed_quality_records=16,pending_slot=16,real_panel_fit_proven=False)))


def test_rehashed_false_receipts_and_unreviewed_outputs_are_rejected(writer,monkeypatch):
    args=setup(writer,monkeypatch);ref=start.prepare_confirmation_request(**args);root=writer.root/'evidence'
    path=root/ref['path'];raw=path.read_bytes();saved=start._document(root,ref)
    mutations=[('parent_pid',True),('slot',17),('status','ready'),('dispatch_authorized',True),
        ('deletion_authorized',True),('prior_dispatch_ref',{}),('guard',{}),('attempt',{}),('request',{}),
        ('context',{}),('completed_records_sha256','a'*64),('checkpoint_pairs_sha256','a'*64),
        ('stages_sha256','b'*64),('frozen_files_sha256','c'*64),('inventory_sha256','d'*64),
        ('controls_sha256','e'*64),('audited_indices',[]),('selection',{}),('extra',False),
        ('storage',dict(saved['storage'],required_free_bytes=1))]
    for key,value in mutations:
        changed=copy.deepcopy(saved);changed[key]=value;path.write_text(json.dumps(changed))
        bad=dict(path=ref['path'],sha256=file_digest(path))
        try:
            with pytest.raises((ValueError,KeyError)):start.verify_confirmation_request(writer,16,bad)
        finally:path.write_bytes(raw)
    for name in ('quality/16/result.json','quality/17/request.json','meta/quality-readiness-16.json'):
        extra=root/name;made=not extra.parent.exists();extra.parent.mkdir(parents=True,exist_ok=True);extra.write_bytes(b'{}')
        try:
            with pytest.raises(ValueError):start.verify_confirmation_request(writer,16,ref)
        finally:
            extra.unlink()
            if made:extra.parent.rmdir()
    for name in (start.append.GUARD,'control/process-quality-15.json','control/quality-confirm-startup-17.json'):
        target=writer.root/name;old=target.read_bytes() if target.exists() else None;target.write_bytes(b'{}')
        try:
            with pytest.raises((ValueError,KeyError)):start.verify_confirmation_request(writer,16,ref)
        finally:
            if old is None:target.unlink()
            else:target.write_bytes(old)
    # Preparation is bound to the current parent; broader historical handoff is separate.
    with monkeypatch.context() as patch:
        patch.setattr(start,'os',SimpleNamespace(getpid=lambda:saved['parent_pid']+1))
        with pytest.raises(ValueError):start.verify_confirmation_request(writer,16,ref)


@pytest.mark.parametrize('when',['before_request','after_request','receipt'])
def test_preparation_failure_is_preserved_and_never_retried(writer,monkeypatch,when):
    args=setup(writer,monkeypatch);root=writer.root/'evidence';original=writer.publish_json
    def fail(name,*a,**kw):
        if when=='before_request' and name=='evidence/'+start.append.REQUEST:raise RuntimeError('synthetic request refusal')
        if when=='receipt' and name=='evidence/'+start.RECEIPT:raise RuntimeError('synthetic receipt refusal')
        result=original(name,*a,**kw)
        if when=='after_request' and name=='evidence/'+start.append.REQUEST:(root/'meta/source.txt').write_bytes(b'changed')
        return result
    monkeypatch.setattr(writer,'publish_json',fail)
    with pytest.raises((ValueError,RuntimeError)):start.prepare_confirmation_request(**args)
    failure=json.loads((root/(start.BASE+'-failure.json')).read_text())
    assert failure['retry_allowed'] is False and not (root/start.RECEIPT).exists()
    assert (root/start.append.REQUEST).exists()==(when!='before_request')
    assert len(list(root.rglob('*.pt')))==8
    before=start.append.history.acceptance._inventory(root)
    with pytest.raises(ValueError,match='cannot be retried'):start.prepare_confirmation_request(**args)
    assert start.append.history.acceptance._inventory(root)==before


def test_late_receipt_control_source_space_and_stop_changes_fail_closed(writer,monkeypatch):
    args=setup(writer,monkeypatch);ref=start.prepare_confirmation_request(**args);root=writer.root/'evidence'
    original=start.append.history.acceptance.bind_storage
    for bad in ('receipt','control','source','space','stop'):
        reached=[];path={'receipt':root/ref['path'],'control':writer.root/start.append.GUARD,
                        'source':root/'meta/source.txt'}.get(bad)
        raw=path.read_bytes() if path else None
        with monkeypatch.context() as patch:
            def late(*a,**kw):
                result=original(*a,**kw);reached.append(True)
                if path:path.write_bytes(b'changed after storage audit')
                elif bad=='stop':(writer.root/'STOP').touch()
                else:patch.setattr(start.append.history,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=8*1024**3)))
                return result
            patch.setattr(start.append.history.acceptance,'bind_storage',late)
            try:
                with pytest.raises((ValueError,KeyError)):start.verify_confirmation_request(writer,16,ref)
            finally:
                if path:path.write_bytes(raw)
                if bad=='stop':(writer.root/'STOP').unlink()
        assert reached==[True]


@pytest.mark.parametrize('slot',[15,17,23,24,True,16.,'16'])
def test_other_slots_rejected_before_mutation(writer,slot):
    with pytest.raises(ValueError,match='slot16'):
        start.prepare_confirmation_request(writer,slot,'a'*64,projection_ref={},rank_completion_ref={},prior_dispatch_ref={})
    with pytest.raises(ValueError,match='slot16'):start.verify_confirmation_request(writer,slot,{})
    assert not (writer.root/'evidence').exists()


def test_post_replay_control_append_blocks_receipt_publication(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence'
    original=start.append.verify_first_confirmation_preparation;reached=[]
    def late(**kw):
        checked=original(**kw)
        if kw.get('pending_request_ref') is not None:
            reached.append(True)
            writer.publish_json('control/process-quality-17.json',dict(synthetic_late_append=True))
        return checked
    monkeypatch.setattr(start.append,'verify_first_confirmation_preparation',late)
    with pytest.raises(ValueError,match='controls changed'):start.prepare_confirmation_request(**args)
    assert reached==[True] and not (root/start.RECEIPT).exists()
    assert (root/(start.BASE+'-failure.json')).exists()
    assert len(list(root.rglob('*.pt')))==8 and not (root/'quality/17').exists()
    with pytest.raises(ValueError,match='cannot be retried'):start.prepare_confirmation_request(**args)
