"""Read-only rank handoff against disposable full-history byte fixtures."""
import copy
import json
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_quality_rank_startup_v2 as start
from src.research.protocol import file_digest
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_quality_rank_startup_v2 import setup


def prepared(writer,monkeypatch,slot=9):
    args=setup(writer,monkeypatch,slot)
    return start.prepare_rank_request(**args)


@pytest.mark.parametrize('slot',[9,14])
def test_complete_history_live_prefix_and_future_space_without_mutation(writer,monkeypatch,slot):
    ref=prepared(writer,monkeypatch,slot);root=writer.root/'evidence';before=start._inventory(root)
    report=start._document(root,ref);original=start.bind_rank_prefix;seen=[]
    def prefix(*a,**kw):
        assert len(kw['lr_requests'])==9 and len(kw['rank_requests'])==slot-9
        assert kw['pending_request']['spec']==start.rank_specs(kw['rates'])[slot-9]
        value=original(*a,**kw)
        assert len(value['records'])==slot and len(value['checkpoint_pairs'])==slot-6
        assert f'quality/{slot:02d}' not in value['checkpoint_pairs']
        seen.append(True);return value
    monkeypatch.setattr(start,'bind_rank_prefix',prefix)
    checked=start.verify_rank_request(writer,slot,ref)
    assert seen==[True]
    assert checked['required_free_bytes']==report['storage']['required_free_bytes']>30_000_000_000
    assert all(checked[k] is True for k in ('pilot_history_verified','lr_history_verified',
                                          'rank_prefix_verified','rank_request_verified'))
    assert checked['dispatch_authorized'] is False and start._inventory(root)==before
    assert len(list(root.rglob('*.pt')))==(slot-6)*2
    assert {p.name for p in (root/f'quality/{slot:02d}').iterdir()}=={'request.json'}


def test_saved_claims_cannot_replace_independent_reconstruction(writer,monkeypatch):
    ref=prepared(writer,monkeypatch);root=writer.root/'evidence';path=root/ref['path']
    original=path.read_bytes();report=json.loads(original)
    storage_cases=[('required_free_bytes',8*1024**3),('occupied_checkpoint_capacity_bytes',0),
        ('checkpoint_logical_bytes',1),('cleanup_boundaries_verified',2.0),
        ('dispatch_authorized',True),('frozen_plan_verified',True),('audit_semantics_verified',True),
        ('storage_projection_fits',False),('current_phase_space_verified',False),
        ('current_inventory_sha256','0'*64),('projection',{}),('free_bytes',True),
        ('free_bytes',8*1024**3),('metadata_bytes',True),('metadata_bytes',256*1024**2+1)]
    cases=[('storage.'+key,value) for key,value in storage_cases]+[
        ('checkpoint_pairs',{}),('completed_records',[]),('audited_indices',[False,1,2]),
        ('context',{}),('frozen_files',{}),('rates',{}),('stages',[]),('selection_ref',{}),
        ('selection',{}),('slot',True),('parent_pid',True),('dispatch_authorized',True)]
    for field,value in cases:
        altered=copy.deepcopy(report)
        if field.startswith('storage.'):altered['storage'][field.split('.')[1]]=value
        else:altered[field]=value
        path.write_text(json.dumps(altered));changed_ref=dict(path=ref['path'],sha256=file_digest(path))
        before=start._inventory(root)
        try:
            with pytest.raises(ValueError):start.verify_rank_request(writer,9,changed_ref)
            assert start._inventory(root)==before,field
        finally:path.write_bytes(original)
    assert file_digest(path)==ref['sha256']


def test_changed_active_failed_or_unknown_evidence_prevents_handoff(writer,monkeypatch):
    ref=prepared(writer,monkeypatch,12);root=writer.root/'evidence'
    paths=['evidence/quality/12/request.json','evidence/quality/00/checkpoints/raw.pt',
        'evidence/quality/09/checkpoints/final_001200_123456.pt','evidence/quality/09/tensor-audit.json',
        'evidence/meta/source.txt','evidence/meta/numerical.json',
        'evidence/.retention/pilot/receipt.json','evidence/.retention/lr/deletions.jsonl',
        'evidence/meta/quality-rank-startup-12-attempt.json','control/quality-rank-startup-12.json',
        'evidence/quality/12/process/launch.json','evidence/quality/12/checkpoints/raw.pt',
        'evidence/quality/13/request.json','evidence/meta/quality-rank-startup-12-failure.json',
        'evidence/meta/quality-rank-startup-12.json.tmp','STOP']
    for name in paths:
        path=writer.root/name;old=path.read_bytes() if path.exists() else None;created=[];parent=path.parent
        while not parent.exists():created.append(parent);parent=parent.parent
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'{}')
        before=start._inventory(root)
        try:
            with pytest.raises(ValueError):start.verify_rank_request(writer,12,ref)
            assert start._inventory(root)==before,name
        finally:
            if old is not None:path.write_bytes(old)
            else:path.unlink()
            for directory in created:directory.rmdir()


def test_exact_actual_capacity_and_late_space_loss(writer,monkeypatch):
    ref=prepared(writer,monkeypatch);root=writer.root/'evidence';before=start._inventory(root)
    needed=start._document(root,ref)['storage']['required_free_bytes']
    for values,match in [([needed,needed],None),([needed-1],'remaining panel no longer fits'),
                         ([needed,needed-1],'Remaining storage changed')]:
        readings=iter(values)
        with monkeypatch.context() as patch:
            patch.setattr(start,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=next(readings))))
            if match:
                with pytest.raises(ValueError,match=match):start.verify_rank_request(writer,9,ref)
            else:assert start.verify_rank_request(writer,9,ref)['required_free_bytes']==needed
        assert start._inventory(root)==before


def test_late_history_checkpoint_and_lock_changes_are_rejected(writer,monkeypatch):
    ref=prepared(writer,monkeypatch);root=writer.root/'evidence';old_frozen=start._frozen
    for change in ('history','weight','locks'):
        calls=[];path=root/('.retention/lr/deletions.jsonl' if change=='history' else 'quality/00/checkpoints/raw.pt')
        old=path.read_bytes()
        with monkeypatch.context() as patch:
            def frozen(*a,**kw):
                value=old_frozen(*a,**kw);calls.append(True)
                if len(calls)==2:
                    if change=='locks':patch.setattr(writer,'lock_fds',(writer.lock_fds[0],writer.lock_fds[0]))
                    else:path.write_bytes(b'changed after initial full audit')
                return value
            patch.setattr(start,'_frozen',frozen)
            try:
                with pytest.raises(ValueError):start.verify_rank_request(writer,9,ref)
                assert len(calls)==2
            finally:path.write_bytes(old)


@pytest.mark.parametrize('slot',[8,15,23,-1,True,'9'])
def test_only_exact_rank_slots_before_mutation(writer,slot):
    with pytest.raises(ValueError):start.verify_rank_request(writer,slot,{})
    assert not (writer.root/'evidence/quality').exists()
