"""Idle compact receipt replay with disposable bytes; no model or real child."""
import copy
import os
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_first_confirmation_history_v2 as history
from src.research.protocol import file_digest,digest
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_first_confirmation_dispatch_v2 import setup as dispatch_setup,dispatch


def setup(writer,monkeypatch):
    args,calls=dispatch_setup(writer,monkeypatch)
    ref=dispatch.run_first_confirmation_once(**args);assert calls==[15]
    monkeypatch.setattr(history,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=64*1024**3)))
    args.pop('slot');args['completion_ref']=ref
    return args


def test_full_replay_under_new_parent_and_descriptors_preserves_all_bytes(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence';before=history.acceptance._inventory(root)
    saved=history._document(root,args['completion_ref']);old_fds=writer.lock_fds
    fresh_fds=tuple(os.dup(fd) for fd in old_fds)
    monkeypatch.setattr(writer,'lock_fds',fresh_fds)
    monkeypatch.setattr(history.acceptance,'os',SimpleNamespace(getpid=lambda:saved['parent_pid']+500000))
    process=history._document(root,saved['review']['child']['process'])
    try:
        # Public immediate acceptance must still require its exact live parent.
        with pytest.raises(ValueError,match='same-parent'):
            history.acceptance.review_completed_first_confirmation(writer,15,saved['review']['startup'],saved['review']['readiness'],process)
        calls=[];original=history.acceptance.bind_storage
        def gate(*a,**kw):
            assert len(kw['records'])==16 and len(kw['stages'])==3 and len(kw['checkpoint_pairs'])==19
            calls.append(1);return original(*a,**kw)
        monkeypatch.setattr(history.acceptance,'bind_storage',gate)
        checked=history.verify_first_confirmation_dispatch(**args)
        assert calls==[1] and checked['historical_transport_verified'] and checked['first_confirmation_audits_bound']
        assert len(checked['records'])==16 and checked['audited_indices']==[9,10,11,15]
        assert len(checked['checkpoint_pairs'])==4 and len(checked['stages'])==3
        for key in ('completed_records','checkpoint_pairs','stages'):
            value=checked['records' if key=='completed_records' else key]
            assert digest(value)==saved['review'][key+'_sha256']
        assert checked['storage']['cleanup_boundaries_verified']==3 and not checked['dispatch_authorized']
        assert history.acceptance._inventory(root)==before and len(list(root.rglob('*.pt')))==8
    finally:
        monkeypatch.setattr(writer,'lock_fds',old_fds)
        for fd in fresh_fds:os.close(fd)


def test_rehashed_receipt_requires_exact_claims_and_reconstructed_audit(writer,monkeypatch):
    import json
    args=setup(writer,monkeypatch);root=writer.root/'evidence';path=root/args['completion_ref']['path']
    raw=path.read_bytes();saved=history._document(root,args['completion_ref'])
    mutations=[('status','lr_reviewed'),('slot',True),('parent_pid',1),('prior_dispatches',[{}]),
        ('rank_completion_ref',{}),('projection_ref',{}),('guards',{})]
    review_mutations=[('representation','unknown'),('adopted',True),('next_stage_authorized',True),
        ('completed_records_sha256','a'*64),('checkpoint_pairs_sha256','b'*64),('stages_sha256','c'*64),
        ('audited_indices',[0,1,2,15]),('evidence_inventory_sha256','d'*64),
        ('selection',{}),('entry_guard',{}),('completed_records',[]),
        ('storage',dict(saved['review']['storage'],required_free_bytes=8*1024**3))]
    for nested,cases in [(False,mutations),(True,review_mutations)]:
        for key,value in cases:
            changed=copy.deepcopy(saved);target=changed['review'] if nested else changed;target[key]=value
            path.write_text(json.dumps(changed));bad=dict(args,completion_ref=dict(path=args['completion_ref']['path'],sha256=file_digest(path)))
            before=history.acceptance._inventory(root)
            try:
                with pytest.raises((ValueError,KeyError)):history.verify_first_confirmation_dispatch(**bad)
                assert history.acceptance._inventory(root)==before,key
            finally:path.write_bytes(raw)


def test_protected_science_transport_and_later_appends_rejected(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence';saved=history._document(root,args['completion_ref'])
    paths=[writer.root/v['path'] for v in saved['guards'].values()]+[
        writer.root/'control/quality-entry-15.json',writer.root/'control/quality-readiness-15.json',
        writer.root/'control/quality-confirm-startup-15.json',writer.root/'control/process-quality-15.json',
        root/'quality/09/checkpoints/final_001200_123456.pt',root/'quality/15/checkpoints/h200.pt',
        root/'meta/numerical.json',root/'meta/source.txt',root/'.retention/rank/deletions.jsonl',
        root/'quality/15/process/worker.log',root/'quality/15/process/process.json',
        root/'quality/15/tensor-audit.json',root/'quality/15/result.json']
    for path in paths:
        raw=path.read_bytes();path.write_bytes(b'{}');before=history.acceptance._inventory(root)
        try:
            with pytest.raises((ValueError,KeyError)):history.verify_first_confirmation_dispatch(**args)
            assert history.acceptance._inventory(root)==before
        finally:path.write_bytes(raw)
    for name in ('quality/16/request.json','meta/quality-confirm-startup-16-attempt.json',
                 'quality/15/worker-failure.json','meta/quality-dispatch-15.json.tmp'):
        path=root/name;created=not path.parent.exists();path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'{}')
        try:
            with pytest.raises(ValueError):history.verify_first_confirmation_dispatch(**args)
        finally:
            path.unlink()
            if created:path.parent.rmdir()
    (writer.root/'STOP').touch()
    with pytest.raises(ValueError):history.verify_first_confirmation_dispatch(**args)


def test_late_disk_source_control_and_receipt_changes_fail_closed(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence';original=history.acceptance.bind_storage
    for bad in ('space','source','control','receipt'):
        path={'source':root/'meta/source.txt','control':writer.root/'control/quality-controller-child-15.json',
              'receipt':root/args['completion_ref']['path']}.get(bad)
        old=path.read_bytes() if path else None
        with monkeypatch.context() as patch:
            def late(*a,**kw):
                result=original(*a,**kw)
                if path:path.write_bytes(b'changed during audit')
                else:patch.setattr(history,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=8*1024**3)))
                return result
            patch.setattr(history.acceptance,'bind_storage',late)
            try:
                with pytest.raises((ValueError,KeyError)):history.verify_first_confirmation_dispatch(**args)
            finally:
                if path:path.write_bytes(old)


@pytest.mark.parametrize('identity',[{},dict(parent_pid=True,lock_fds=(3,4)),
    dict(parent_pid=5,lock_fds=(3,3)),dict(parent_pid=5,lock_fds=(True,4))])
def test_invalid_historical_identity_cannot_enter_audit(writer,identity):
    start=dict(path='meta/quality-confirm-startup-15.json',sha256='a'*64)
    ready=dict(path='meta/quality-readiness-15.json',sha256='b'*64)
    with pytest.raises(ValueError):
        history.acceptance._review_completed(writer,15,start,ready,{},historical_identity=identity)
    assert not (writer.root/'evidence').exists()
