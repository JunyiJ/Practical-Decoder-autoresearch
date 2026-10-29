"""First-fresh sealed handoff with disposable complete history, never training."""
import copy
import json
import os
import sys
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_quality_entry_v2 as entry
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_first_confirmation_readiness_v2 import prepared


def setup(writer,monkeypatch):
    startup=prepared(writer,monkeypatch)
    ref=entry.readiness.seal_quality_readiness(writer,15,startup)
    seal=entry._document(writer.root/'evidence',ref)
    monkeypatch.setattr(entry.os,'getppid',os.getpid)
    monkeypatch.setattr(entry,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=64*1024**3)))
    argv=entry.worker_argv(writer,15,startup,ref)
    launch=dict(status='preflight',argv=argv,parent_pid=os.getpid(),pid=None,returncode=None,error=None,
        required_free_bytes=seal['required_free_bytes'],reserve_bytes=8*1024**3,
        output_limit_bytes=seal['output_limit_bytes'],log_limit_bytes=4*1024**2,timeout_seconds=7200,
        observation=None,capture_kind='bounded_pipe_v2',log_truncated=False)
    writer.publish_json('evidence/quality/15/process/launch.json',launch)
    with writer.open_append('evidence/quality/15/process/worker.log') as out:out.write(b'synthetic imports\n')
    return dict(writer=writer,slot=15,startup_ref=startup,readiness_ref=ref,actual_argv=argv)


def test_fresh_handoff_reconstructs_three_boundaries_and_only_selected_capacity(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence';before=entry.readiness._inventory(root)
    original=entry.remaining_projection;seen=[]
    def projection(p,pilots,records,rates,selection,boundaries):
        assert len(pilots)==3 and len(records)==15 and boundaries==3
        assert all(v['divisor']==8 for v in selection.values())
        seen.append(1);return original(p,pilots,records,rates,selection,boundaries)
    monkeypatch.setattr(entry,'remaining_projection',projection)
    checked=entry.verify_ready_worker(**args)
    assert seen==[1] and all(checked[k] for k in
        ('parent_idle_audit_bound','frozen_plan_verified','live_transport_verified','storage_verified'))
    assert not checked['dispatch_authorized'] and entry.readiness._inventory(root)==before
    assert len(list(root.rglob('*.pt')))==6 and not (root/'quality/15/checkpoints').exists()
    assert not (root/'quality/00/checkpoints/raw.pt').exists()
    census=entry.snapshot(writer.root)
    (writer.root.parent/'first-confirmation-entry-census.json').write_text(json.dumps(dict(
        groups=census['groups'],metadata_bytes=census['metadata_bytes'],file_count=len(census['files']),
        live_checkpoint_files=6,completed_fresh=0,real_panel_fit_proven=False),sort_keys=True))


def test_fresh_changed_history_transport_and_selected_weights_reject(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence'
    for name in (args['readiness_ref']['path'],'meta/source.txt','meta/numerical.json',
        '.retention/pilot/receipt.json','.retention/lr/deletions.jsonl','.retention/rank/execution-review.json',
        'meta/rank-selection.json','quality/09/checkpoints/final_001200_123456.pt',
        'quality/15/process/launch.json','quality/15/result.json'):
        path=root/name;old=path.read_bytes() if path.exists() else None;path.write_bytes(b'{}')
        before=entry.readiness._inventory(root)
        try:
            with pytest.raises((ValueError,KeyError)):entry.verify_ready_worker(**args)
            assert entry.readiness._inventory(root)==before
        finally:
            if old is None:path.unlink()
            else:path.write_bytes(old)
    for bad in ('parent','locks','argv','stop'):
        with monkeypatch.context() as patch:
            changed=dict(args)
            if bad=='parent':patch.setattr(entry.os,'getppid',lambda:1)
            elif bad=='locks':patch.setattr(writer,'lock_fds',(writer.lock_fds[0],writer.lock_fds[0]))
            elif bad=='argv':changed['actual_argv']=args['actual_argv']+['extra']
            else:(writer.root/'STOP').touch()
            try:
                with pytest.raises(ValueError):entry.verify_ready_worker(**changed)
            finally:
                if bad=='stop':(writer.root/'STOP').unlink()


def test_compact_fresh_hashes_and_capacity_cannot_be_forged(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence'
    startup=entry._document(root,args['startup_ref']);seal=entry._document(root,args['readiness_ref'])
    for key,value in [('completed_records_sha256','a'*64),('checkpoint_pairs_sha256','b'*64),
        ('stages_sha256','c'*64),('frozen_files_sha256','d'*64),('rates',{}),('selection',{}),
        ('selection_ref',{}),('rates_ref',{}),('pilot_completion_ref',{}),('audited_indices',[0,1,2])]:
        altered=copy.deepcopy(startup);altered[key]=value
        with pytest.raises((ValueError,KeyError)):entry._storage(writer,15,seal,altered,seal['inventory'])
    altered=copy.deepcopy(startup);altered['storage']['cleanup_boundaries_verified']=2
    with pytest.raises(ValueError):entry._storage(writer,15,seal,altered,seal['inventory'])
    needed=seal['required_free_bytes']
    for values,valid in [([needed,needed],True),([needed-1],False),([needed,needed-1],False)]:
        readings=iter(values)
        with monkeypatch.context() as patch:
            patch.setattr(entry,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=next(readings))))
            if valid:assert entry.verify_ready_worker(**args)['required_free_bytes']==needed
            else:
                with pytest.raises(ValueError):entry.verify_ready_worker(**args)


def test_fresh_late_mutation_and_log_append_rules(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence';original=entry._storage
    log=root/'quality/15/process/worker.log';journal=root/'.retention/rank/deletions.jsonl'
    for change in ('append','rewrite','history'):
        old_log=log.read_bytes();old_journal=journal.read_bytes()
        def late(*a,**kw):
            result=original(*a,**kw)
            if change=='append':
                with log.open('ab') as out:out.write(b'bounded append\n')
            elif change=='rewrite':log.write_bytes(b'x'*len(old_log))
            else:journal.write_bytes(b'late mutation')
            return result
        with monkeypatch.context() as patch:
            patch.setattr(entry,'_storage',late)
            try:
                if change=='append':assert entry.verify_ready_worker(**args)['storage_verified']
                else:
                    with pytest.raises(ValueError):entry.verify_ready_worker(**args)
            finally:log.write_bytes(old_log);journal.write_bytes(old_journal)


@pytest.mark.parametrize('valid',[True,False])
def test_fresh_guard_precedes_synthetic_producer_and_blocks_retry(writer,monkeypatch,valid):
    args=setup(writer,monkeypatch);calls=[]
    def produce(w,slot,plan_sha,request_sha,*,projection_ref):
        assert w is writer and slot==15
        guard=json.loads((writer.root/'control/quality-entry-15.json').read_text())
        assert guard['limits']['timeout_seconds']==7200 and guard['status']=='starting'
        calls.append(1);return 'synthetic only'
    monkeypatch.setitem(sys.modules,'src.research.tokenizer_wide_hybrid_quality_execution_v2',
                        SimpleNamespace(execute_quality_outputs=produce))
    if not valid:(writer.root/'STOP').touch()
    args['argv']=args.pop('actual_argv')
    if valid:assert entry.run_worker(**args)=='synthetic only'
    else:
        with pytest.raises(ValueError):entry.run_worker(**args)
    assert len(calls)==int(valid)
    with pytest.raises(ValueError,match='cannot be retried'):entry.run_worker(**args)


def test_first_fresh_cli_exact_seal_and_later_slots_stay_closed(writer,monkeypatch):
    start=dict(path='meta/quality-confirm-startup-15.json',sha256='a'*64)
    seal=dict(path='meta/quality-readiness-15.json',sha256='b'*64)
    argv=entry.worker_argv(writer,15,start,seal);seen=[]
    def capture(w,slot,startup,ready,args):
        assert w.root==writer.root and slot==15 and startup==start and ready==seal and args==argv
        seen.append(1);return 'no producer'
    monkeypatch.setattr(entry,'run_worker',capture)
    assert entry.main(argv[3:])=='no producer' and seen==[1]
    with pytest.raises(ValueError):entry.transport.worker_argv(writer,15,start)
    for slot in (17,23,24,-1,True,15.0):
        with pytest.raises(ValueError):entry.worker_argv(writer,slot,start,seal)
