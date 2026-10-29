"""Independent compact slot17 receipt replay; disposable bytes, no model work."""
import copy
import json
import os
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_later_confirmation_history_v2 as history
from src.research.protocol import digest,file_digest
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_slot17_dispatch_v2 import setup as dispatch_setup,controller


def setup(writer,monkeypatch):
    args,calls=dispatch_setup(writer,monkeypatch)
    ref=controller.run_sealed_confirmation_once(**args);assert calls==[17]
    saved=history._document(writer.root/'evidence',ref)
    monkeypatch.setattr(history,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=64*1024**3)))
    return dict(writer=writer,slot=17,plan_sha256=saved['plan_sha256'],completion_ref=ref,
        projection_ref=saved['projection_ref'],rank_completion_ref=saved['rank_completion_ref'])


def test_full_replay_under_new_parent_and_descriptors_preserves_all_bytes(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence';before=history.acceptance._inventory(root)
    controls=history.acceptance.readiness._controls(writer,'not-a-control')
    saved=history._document(root,args['completion_ref']);old_fds=writer.lock_fds
    new_fds=tuple(os.dup(fd) for fd in old_fds);monkeypatch.setattr(writer,'lock_fds',new_fds)
    monkeypatch.setattr(history.acceptance,'os',SimpleNamespace(getpid=lambda:saved['parent_pid']+500000))
    original=history.acceptance.bind_storage;calls=[]
    def capacity(*a,**kw):
        assert len(kw['records'])==18 and len(kw['stages'])==3 and len(kw['checkpoint_pairs'])==21
        calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(history.acceptance,'bind_storage',capacity)
    try:
        with pytest.raises(ValueError,match='same-parent'):
            history.acceptance.review_completed_later_confirmation(writer,17,saved['review']['startup'],saved['review']['readiness'],
                history._document(root,saved['review']['child']['process']))
        report=history.verify_confirmation_dispatch(**args)
        assert calls==[1] and report['historical_transport_verified'] and report['confirmation_audits_bound']
        assert len(report['records'])==18 and len(report['checkpoint_pairs'])==6 and len(report['stages'])==3
        assert report['audited_indices']==[9,10,11,15,16,17]
        for key,report_key in [('completed_records','records'),('checkpoint_pairs','checkpoint_pairs'),('stages','stages')]:
            assert digest(report[report_key])==saved['review'][key+'_sha256']
        assert all(report[k] is False for k in ('adopted','dispatch_authorized','deletion_authorized','next_stage_authorized'))
        assert history.acceptance._inventory(root)==before
        assert history.acceptance.readiness._controls(writer,'not-a-control')==controls
        assert len(list(root.rglob('*.pt')))==12 and not (root/'quality/18').exists()
        census=history.snapshot(writer.root)
        (writer.root.parent/'slot17-history-census.json').write_text(json.dumps(dict(
            metadata_bytes=census['metadata_bytes'],meta_group_bytes=census['groups']['evidence/meta'],
            completed_quality_records=18,live_checkpoint_files=12,cleanup_boundaries=3,
            required_free_bytes=report['storage']['required_free_bytes'],real_panel_fit_proven=False)))
    finally:
        monkeypatch.setattr(writer,'lock_fds',old_fds)
        for fd in new_fds:os.close(fd)


def test_rehashed_false_receipt_claims_cannot_replace_full_reconstruction(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence';path=root/args['completion_ref']['path']
    raw=path.read_bytes();saved=history._document(root,args['completion_ref'])
    mutations=[('status','first_confirmation_reviewed'),('slot',True),('parent_pid',1),('prior_dispatches',[]),
        ('pilot_completion_ref',{}),('rank_completion_ref',{}),('projection_ref',{}),('guards',{}),('extra',False)]
    review_mutations=[('representation','unknown'),('prior_transport_verified',False),('adopted',True),
        ('next_stage_authorized',True),('deletion_authorized',True),('completed_records_sha256','a'*64),
        ('checkpoint_pairs_sha256','b'*64),('stages_sha256','c'*64),('audited_indices',[9,10,11,16]),
        ('evidence_inventory_sha256','d'*64),('selection',{}),('rates',{}),('entry_guard',{}),
        ('supervisor_ref',{}),('prior_dispatch_ref',{}),('completed_records',[]),
        ('storage',dict(saved['review']['storage'],required_free_bytes=8*1024**3))]
    for nested,cases in [(False,mutations),(True,review_mutations)]:
        for key,value in cases:
            changed=copy.deepcopy(saved);target=changed['review'] if nested else changed;target[key]=value
            path.write_text(json.dumps(changed));bad=dict(args,completion_ref=dict(path=args['completion_ref']['path'],sha256=file_digest(path)))
            before=history.acceptance._inventory(root)
            try:
                with pytest.raises((ValueError,KeyError)):history.verify_confirmation_dispatch(**bad)
                assert history.acceptance._inventory(root)==before
            finally:path.write_bytes(raw)
    # Prechild capacity must be reconstructed, not merely copied from two receipts.
    _,_,context=history._transport(writer,args['plan_sha256'],args['completion_ref'],args['projection_ref'],args['rank_completion_ref'])
    current=history.verify_confirmation_dispatch(**args)['storage']
    for key,value in [('required_free_bytes',8*1024**3),('occupied_checkpoint_capacity_bytes',0),
        ('checkpoint_logical_bytes',0),('cleanup_boundaries_verified',2)]:
        for nested in (False,True):
            changed=copy.deepcopy(context);target=changed['prior_context'] if nested else changed
            target['prepared']['storage'][key]=value;target['prior']['review']['storage'][key]=value
            with pytest.raises(ValueError):history._prechild_storage(writer,changed,current)


def test_protected_history_and_later_evidence_or_controls_are_rejected(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence'
    paths=[root/n for n in ('meta/source.txt','meta/numerical.json','meta/quality-dispatch-15.json','meta/quality-dispatch-16.json',
        'meta/quality-confirm-startup-17-attempt.json','meta/quality-confirm-startup-17.json','meta/quality-readiness-17.json',
        '.retention/pilot/receipt.json','.retention/lr/deletions.jsonl','.retention/rank/execution-review.json',
        'quality/09/checkpoints/final_001200_123456.pt','quality/15/checkpoints/h200.pt','quality/16/checkpoints/h200.pt','quality/17/checkpoints/h200.pt',
        'quality/15/result.json','quality/16/result.json','quality/17/result.json','quality/17/worker-outputs.json','quality/17/tensor-audit.json',
        'quality/17/process/launch.json','quality/17/process/process.json','quality/17/process/worker.log')]
    paths += [writer.root/n for n in ('control/quality-confirm-startup-17.json','control/quality-readiness-17.json',
        'control/quality-entry-15.json','control/quality-entry-16.json','control/quality-entry-17.json','control/process-quality-15.json','control/process-quality-16.json','control/process-quality-17.json',
        'control/quality-controller-prepare-17.json','control/quality-controller-child-17.json','control/quality-controller-review-17.json')]
    for path in paths:
        raw=path.read_bytes();path.write_bytes(b'{}');before=history.acceptance._inventory(root)
        try:
            with pytest.raises((ValueError,KeyError)):history.verify_confirmation_dispatch(**args)
            assert history.acceptance._inventory(root)==before
        finally:path.write_bytes(raw)
    for name in ('evidence/quality/18/request.json','evidence/meta/quality-confirm-startup-18-attempt.json',
        'control/quality-confirm-startup-18.json','control/process-quality-18.json',
        'evidence/quality/17/worker-failure.json','evidence/meta/quality-dispatch-17.json.tmp'):
        path=writer.root/name;made=not path.parent.exists();path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'{}')
        try:
            with pytest.raises(ValueError):history.verify_confirmation_dispatch(**args)
        finally:
            path.unlink()
            if made:path.parent.rmdir()
    (writer.root/'STOP').touch()
    with pytest.raises(ValueError):history.verify_confirmation_dispatch(**args)


def test_late_science_transport_capacity_and_stop_changes_fail_closed(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence';original=history.acceptance.bind_storage
    for bad in ('space','source','control','receipt','stop'):
        path={'source':root/'meta/source.txt','control':writer.root/'control/quality-controller-child-17.json',
            'receipt':root/args['completion_ref']['path']}.get(bad);raw=path.read_bytes() if path else None;seen=[]
        with monkeypatch.context() as patch:
            def late(*a,**kw):
                value=original(*a,**kw);seen.append(1)
                if path:path.write_bytes(b'late mutation')
                elif bad=='stop':(writer.root/'STOP').touch()
                else:patch.setattr(history,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=8*1024**3)))
                return value
            patch.setattr(history.acceptance,'bind_storage',late)
            try:
                with pytest.raises((ValueError,KeyError)):history.verify_confirmation_dispatch(**args)
            finally:
                if path:path.write_bytes(raw)
                if bad=='stop':(writer.root/'STOP').unlink()
        assert seen==[1]
    assert len(list(root.rglob('*.pt')))==12 and not (root/'quality/18').exists()


@pytest.mark.parametrize('slot',[0,15,16,18,23,24,-1,True,17.,'17'])
def test_only_exact_slot17_replay_is_enabled(writer,slot):
    with pytest.raises(ValueError,match='slot17'):
        history.verify_confirmation_dispatch(writer,slot,'a'*64,{},projection_ref={},rank_completion_ref={})
    assert not (writer.root/'evidence').exists()


@pytest.mark.parametrize('identity',[{},dict(parent_pid=True,lock_fds=(3,4)),dict(parent_pid=5,lock_fds=(3,3)),dict(parent_pid=5,lock_fds=(True,4))])
def test_invalid_historical_identity_cannot_enter_slot17_audit(writer,identity):
    start=dict(path='meta/quality-confirm-startup-17.json',sha256='a'*64)
    seal=dict(path='meta/quality-readiness-17.json',sha256='b'*64)
    with pytest.raises(ValueError):history.acceptance._review_completed(writer,17,start,seal,{},historical_identity=identity)
    assert not (writer.root/'evidence').exists()
