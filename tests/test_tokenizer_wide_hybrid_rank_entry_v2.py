"""Synthetic rank parent seal -> child handoff/storage; never real training."""
import copy
import json
import os
import sys
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_quality_entry_v2 as entry
from src.research.protocol import file_digest
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_rank_readiness_v2 import prepared


def setup(writer,monkeypatch,slot=9):
    startup=prepared(writer,monkeypatch,slot)
    ref=entry.readiness.seal_quality_readiness(writer,slot,startup);root=writer.root/'evidence'
    seal=entry._document(root,ref)
    monkeypatch.setattr(entry.os,'getppid',os.getpid)
    monkeypatch.setattr(entry,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=64*1024**3)))
    argv=entry.worker_argv(writer,slot,startup,ref)
    launch=dict(status='preflight',argv=argv,parent_pid=os.getpid(),pid=None,returncode=None,error=None,
        required_free_bytes=seal['required_free_bytes'],reserve_bytes=8*1024**3,
        output_limit_bytes=seal['output_limit_bytes'],log_limit_bytes=4*1024**2,timeout_seconds=7200,
        observation=None,capture_kind='bounded_pipe_v2',log_truncated=False)
    writer.publish_json(f'evidence/quality/{slot:02d}/process/launch.json',launch)
    with writer.open_append(f'evidence/quality/{slot:02d}/process/worker.log') as out:out.write(b'synthetic child imports\n')
    return dict(writer=writer,slot=slot,startup_ref=startup,readiness_ref=ref,actual_argv=argv)


@pytest.mark.parametrize('slot',[9,14])
def test_rank_handoff_counts_live_capacity_and_preserves_deleted_history(writer,monkeypatch,slot):
    args=setup(writer,monkeypatch,slot);root=writer.root/'evidence';before=entry.readiness._inventory(root)
    original=entry.required_free;seen=[]
    def capacity(state,occupied):
        assert state['cleanup_prefix_length']==2 and state['phase']=='rank'
        actual=sum(min(p.stat().st_size,p.stat().st_blocks*512) for p in root.rglob('*.pt'))
        assert occupied==actual>0;seen.append(True);return original(state,occupied)
    monkeypatch.setattr(entry,'required_free',capacity)
    checked=entry.verify_ready_worker(**args)
    assert seen==[True] and all(checked[k] is True for k in (
        'parent_idle_audit_bound','frozen_plan_verified','live_transport_verified','storage_verified'))
    assert checked['dispatch_authorized'] is False and entry.readiness._inventory(root)==before
    assert len(list(root.rglob('*.pt')))==(slot-6)*2
    assert not (root/'quality/03/checkpoints/raw.pt').exists()
    assert args['actual_argv'][args['actual_argv'].index('--slot')+1]==str(slot)


def test_rank_changed_evidence_and_live_transport_reject_without_writes(writer,monkeypatch):
    args=setup(writer,monkeypatch,12);root=writer.root/'evidence'
    names=[args['readiness_ref']['path'],'meta/source.txt','meta/numerical.json',
        '.retention/lr/deletions.jsonl','.retention/pilot/receipt.json',
        'quality/00/checkpoints/raw.pt','quality/09/checkpoints/final_001200_123456.pt',
        'quality/12/process/launch.json','quality/12/process/process.json',
        'quality/12/checkpoints/raw.pt','meta/quality-unreviewed.json']
    for name in names:
        path=root/name;old=path.read_bytes() if path.exists() else None;created=[];parent=path.parent
        while not parent.exists():created.append(parent);parent=parent.parent
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'{}');before=entry.readiness._inventory(root)
        try:
            with pytest.raises(ValueError):entry.verify_ready_worker(**args)
            assert entry.readiness._inventory(root)==before,name
        finally:
            if old is None:path.unlink()
            else:path.write_bytes(old)
            for directory in created:directory.rmdir()
    for bad in ('parent','locks','argv'):
        with monkeypatch.context() as patch:
            changed=dict(args)
            if bad=='parent':patch.setattr(entry.os,'getppid',lambda:1)
            elif bad=='locks':patch.setattr(writer,'lock_fds',(writer.lock_fds[0],writer.lock_fds[0]))
            else:changed['actual_argv']=args['actual_argv']+['extra']
            with pytest.raises(ValueError):entry.verify_ready_worker(**changed)


def test_rank_storage_reconstructs_original_pairs_and_selection(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence'
    startup=entry._document(root,args['startup_ref']);seal=entry._document(root,args['readiness_ref'])
    cases=[('stages',[]),('rates',{}),('selection_ref',{}),('completed_records',[]),
        ('audited_indices',[False,1,2]),('checkpoint_pairs',{}),('projection_ref',{}),('pilot_completion_ref',{})]
    for field,value in cases:
        altered=copy.deepcopy(startup);altered[field]=value
        with pytest.raises((ValueError,KeyError)):entry._storage(writer,9,seal,altered,seal['inventory'])
    altered=copy.deepcopy(startup)
    altered['checkpoint_pairs']['quality/03']['raw']['sha256']='0'*64
    with pytest.raises(ValueError,match='Original deleted'):entry._storage(writer,9,seal,altered,seal['inventory'])
    altered=copy.deepcopy(startup);altered['storage']['cleanup_boundaries_verified']=1
    with pytest.raises(ValueError,match='actual allocation'):entry._storage(writer,9,seal,altered,seal['inventory'])


def test_rank_capacity_exact_threshold_late_loss_and_log_append_rules(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence';seal=entry._document(root,args['readiness_ref'])
    needed=seal['required_free_bytes']
    for values,valid in [([needed,needed],True),([needed-1],False),([needed,needed-1],False)]:
        readings=iter(values)
        with monkeypatch.context() as patch:
            patch.setattr(entry,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=next(readings))))
            if valid:assert entry.verify_ready_worker(**args)['required_free_bytes']==needed
            else:
                with pytest.raises(ValueError):entry.verify_ready_worker(**args)
    original=entry._storage;log=root/'quality/09/process/worker.log';saved=log.read_bytes()
    for change in ('append','rewrite','history'):
        journal=root/'.retention/lr/deletions.jsonl';old=journal.read_bytes()
        with monkeypatch.context() as patch:
            def late(*a,**kw):
                result=original(*a,**kw)
                if change=='append':
                    with log.open('ab') as out:out.write(b'bounded append\n')
                elif change=='rewrite':log.write_bytes(b'x'*len(saved))
                else:journal.write_bytes(b'late mutation')
                return result
            patch.setattr(entry,'_storage',late)
            try:
                if change=='append':assert entry.verify_ready_worker(**args)['storage_verified']
                else:
                    with pytest.raises(ValueError):entry.verify_ready_worker(**args)
            finally:log.write_bytes(saved);journal.write_bytes(old)


@pytest.mark.parametrize('valid',[True,False])
def test_rank_child_gates_synthetic_producer_and_never_retries(writer,monkeypatch,valid):
    args=setup(writer,monkeypatch);calls=[]
    def produce(w,slot,plan_sha,request_sha,*,projection_ref):
        assert w is writer and slot==9
        guard=json.loads((writer.root/'control/quality-entry-09.json').read_text())
        assert guard['limits']['timeout_seconds']==7200 and guard['status']=='starting'
        calls.append(True);return dict(synthetic_only=True)
    monkeypatch.setitem(sys.modules,'src.research.tokenizer_wide_hybrid_quality_execution_v2',
                        SimpleNamespace(execute_quality_outputs=produce))
    if not valid:(writer.root/'STOP').touch()
    kwargs={**args,'argv':args['actual_argv']};del kwargs['actual_argv']
    if valid:assert entry.run_worker(**kwargs)==dict(synthetic_only=True)
    else:
        with pytest.raises(ValueError):entry.run_worker(**kwargs)
    assert len(calls)==int(valid)
    with pytest.raises(ValueError,match='cannot be retried'):entry.run_worker(**kwargs)


@pytest.mark.parametrize('slot',[9,14])
def test_rank_cli_builds_exact_sealed_request_without_launch(writer,monkeypatch,slot):
    startup=dict(path=f'meta/quality-rank-startup-{slot:02d}.json',sha256='a'*64)
    readiness=dict(path=f'meta/quality-readiness-{slot:02d}.json',sha256='b'*64)
    argv=entry.worker_argv(writer,slot,startup,readiness);seen=[]
    def capture(w,s,start,ready,args):
        assert w.root==writer.root and w.lock_fds==writer.lock_fds and s==slot
        assert start==startup and ready==readiness and args==argv
        seen.append(True);return 'no producer'
    monkeypatch.setattr(entry,'run_worker',capture)
    assert entry.main(argv[3:])=='no producer' and seen==[True]
    with pytest.raises(ValueError,match='requires an immutable readiness'):
        entry.transport.worker_argv(writer,slot,startup)


def test_later_confirmation_cli_stays_closed(writer):
    with pytest.raises(SystemExit) as exc:entry.main(['worker','--slot','16'])
    assert exc.value.code==2
