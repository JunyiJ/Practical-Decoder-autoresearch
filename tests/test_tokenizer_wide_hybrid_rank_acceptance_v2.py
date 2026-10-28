"""Completed rank child review with disposable byte fixtures; no actual child."""
import copy
import json
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_quality_acceptance_v2 as accept
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_rank_readiness_v2 import prepared
from test_tokenizer_wide_hybrid_quality_acceptance_v2 import complete


def setup(writer,monkeypatch,slot=9):
    startup=prepared(writer,monkeypatch,slot)
    ready=accept.readiness.seal_quality_readiness(writer,slot,startup)
    return complete(writer,monkeypatch,slot,startup,ready)


@pytest.mark.parametrize('slot',[9,12,14])
def test_rank_parent_reaudits_every_live_member_history_and_full_storage(writer,monkeypatch,slot):
    args=setup(writer,monkeypatch,slot);root=writer.root/'evidence';before=accept._inventory(root)
    audit=accept.bind_rank_prefix;storage=accept.bind_storage;calls=[]
    def prefix(*a,**kw):
        writer.publish_json('control/synthetic-rank-parent-observation.json',{'idle_audit':True})
        assert len(kw['lr_records'])==9 and len(kw['lr_requests'])==9 and len(kw['rank_requests'])==slot-8
        assert 'pending_request' not in kw;calls.append('prefix');return audit(*a,**kw)
    def capacity(*a,**kw):
        assert len(kw['pilots'])==3 and len(kw['records'])==slot+1 and len(kw['stages'])==2
        assert len(kw['checkpoint_pairs'])==slot+4 and kw['rates'] is not None and kw['selection'] is None
        calls.append('storage');return storage(*a,**kw)
    monkeypatch.setattr(accept,'bind_rank_prefix',prefix);monkeypatch.setattr(accept,'bind_storage',capacity)
    checked=accept.review_completed_rank(**args)
    assert calls==['prefix','storage'] and checked['kind']==accept.RANK_KIND
    assert checked['status']=='rank_child_reviewed_pending_parent_receipt'
    assert len(checked['completed_records'])==slot+1 and len(checked['checkpoint_pairs'])==slot-5
    assert checked['audited_indices']==[0,1,2]+list(range(9,slot+1))
    assert checked['storage']['cleanup_boundaries_verified']==2 and checked['storage']['storage_projection_fits']
    assert checked['selection'] is None and len(checked['stages'])==2
    assert all(checked[k] is False for k in ('adopted','dispatch_authorized','deletion_authorized',
        'stage_cleanup_complete','next_stage_authorized'))
    assert accept._inventory(root)==before
    assert not (root/'.retention/rank').exists() and not (root/'meta/rank-selection.json').exists()


def test_rank_success_exit_never_excuses_changed_evidence(writer,monkeypatch):
    args=setup(writer,monkeypatch,12);root=writer.root/'evidence'
    names=[f'evidence/{args[k]["path"]}' for k in ('startup_ref','readiness_ref')]+[
        'evidence/meta/numerical.json','evidence/meta/source.txt',
        'evidence/quality/00/checkpoints/raw.pt','evidence/quality/09/checkpoints/final_001200_123456.pt',
        'evidence/quality/12/checkpoints/h200.pt','evidence/quality/12/process/launch.json',
        'control/quality-entry-12.json','evidence/quality/12/result.json',
        'evidence/quality/12/worker-outputs.json','evidence/quality/12/tensor-audit.json',
        'evidence/quality/12/resource-audit.json','evidence/.retention/lr/deletions.jsonl',
        'evidence/.retention/pilot/deletions.jsonl','evidence/quality/12/unreviewed.json',
        'evidence/quality/13/request.json','STOP']
    for name in names:
        path=writer.root/name;old=path.read_bytes() if path.exists() else None;created=[];parent=path.parent
        while not parent.exists():created.append(parent);parent=parent.parent
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'{}');before=accept._inventory(root)
        try:
            with pytest.raises(ValueError):accept.review_completed_rank(**args)
            assert accept._inventory(root)==before,name
        finally:
            if old is None:path.unlink()
            else:path.write_bytes(old)
            for directory in created:directory.rmdir()


def test_rehashed_rank_terminal_still_requires_exact_process_contract(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence';path=root/'quality/09/process/process.json'
    saved=path.read_bytes()
    for key,value in [('pid',1),('parent_pid',1),('returncode',True),('status','running'),
        ('timeout_seconds',600),('required_free_bytes',8*1024**3),('capture_kind','file'),
        ('log_truncated',True),('elapsed_seconds',7200.),
        ('observation',dict(free_bytes=64*1024**3,output_bytes=0,log_bytes=0))]:
        terminal=copy.deepcopy(args['process']);terminal[key]=value;path.write_text(json.dumps(terminal))
        before=accept._inventory(root)
        try:
            with pytest.raises(ValueError):accept.review_completed_rank(**{**args,'process':terminal})
            assert accept._inventory(root)==before,key
        finally:path.write_bytes(saved)


def test_final_rank_parent_storage_source_and_control_rechecks(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence';original=accept.bind_storage
    for bad in ('space','source','control','receipt'):
        name={'source':'evidence/meta/source.txt','control':'control/quality-readiness-09.json',
              'receipt':'evidence/.retention/lr/receipt.json'}.get(bad)
        path=writer.root/name if name else None;old=path.read_bytes() if path else None
        with monkeypatch.context() as patch:
            def late(*a,**kw):
                value=original(*a,**kw)
                if bad=='space':patch.setattr(accept,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=8*1024**3)))
                else:path.write_bytes(b'changed after complete storage audit')
                return value
            patch.setattr(accept,'bind_storage',late)
            try:
                with pytest.raises(ValueError):accept.review_completed_rank(**args)
            finally:
                if path:path.write_bytes(old)


@pytest.mark.parametrize('slot',[-1,0,8,15,23,True,'9'])
def test_rank_acceptance_only_declared_rank_slots(writer,slot):
    with pytest.raises(ValueError,match='Exact rank acceptance'):
        accept.review_completed_rank(writer,slot,{},{},{})
    assert not (writer.root/'evidence').exists()


@pytest.mark.parametrize('slot',[9,14,15,True])
def test_lr_acceptance_cannot_be_used_as_rank_shortcut(writer,slot):
    with pytest.raises(ValueError,match='Exact LR acceptance'):
        accept.review_completed_lr(writer,slot,{},{},{})
    assert not (writer.root/'evidence').exists()
