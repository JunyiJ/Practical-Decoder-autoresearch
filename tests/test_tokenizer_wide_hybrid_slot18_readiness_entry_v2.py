"""Slot18 sealed handoff on disposable history, with no real model execution."""
import copy
import json
import os
import sys
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_quality_entry_v2 as entry
from src.research import tokenizer_wide_hybrid_remaining_confirmation_startup_v2 as start
from src.research.protocol import file_digest
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_remaining_confirmation_startup_v2 import setup as history


def prepared(writer,monkeypatch):
    args=history(writer,monkeypatch)
    ref=start.prepare_confirmation_request(**args)
    monkeypatch.setattr(entry.readiness,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=64*1024**3)))
    return ref


def setup(writer,monkeypatch):
    startup=prepared(writer,monkeypatch)
    ref=entry.readiness.seal_quality_readiness(writer,18,startup)
    seal=entry._document(writer.root/'evidence',ref)
    monkeypatch.setattr(entry.os,'getppid',os.getpid)
    monkeypatch.setattr(entry,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=64*1024**3)))
    argv=entry.worker_argv(writer,18,startup,ref)
    launch=dict(status='preflight',argv=argv,parent_pid=os.getpid(),pid=None,returncode=None,error=None,
        required_free_bytes=seal['required_free_bytes'],reserve_bytes=8*1024**3,
        output_limit_bytes=seal['output_limit_bytes'],log_limit_bytes=4*1024**2,timeout_seconds=7200,
        observation=None,capture_kind='bounded_pipe_v2',log_truncated=False)
    writer.publish_json('evidence/quality/18/process/launch.json',launch)
    with writer.open_append('evidence/quality/18/process/worker.log') as out:out.write(b'synthetic imports\n')
    return dict(writer=writer,slot=18,startup_ref=startup,readiness_ref=ref,actual_argv=argv)


def test_seal_binds_explicit_guard_and_full_history_without_weakening_idle_reader(writer,monkeypatch):
    startup=prepared(writer,monkeypatch);root=writer.root/'evidence';before=entry.readiness._inventory(root)
    original=start.verify_confirmation_for_readiness;seen=[]
    def checked(w,slot,ref,guard):
        # Only the explicitly declared, bounded current-parent guard is allowed.
        with pytest.raises(ValueError):start.verify_confirmation_request(w,slot,ref)
        path=w.root/guard['path'];raw=path.read_bytes();saved=json.loads(raw)
        for key,value in [('kind','wrong'),('pid',True),('status','failed'),('limits',{}),('elapsed_seconds',1800)]:
            changed=copy.deepcopy(saved);changed[key]=value;path.write_text(json.dumps(changed))
            try:
                with pytest.raises(ValueError):start._readiness_control(w,dict(path=guard['path'],sha256=file_digest(path)))
            finally:path.write_bytes(raw)
        value=original(w,slot,ref,guard);seen.append(1);return value
    monkeypatch.setattr(start,'verify_confirmation_for_readiness',checked)
    ref=entry.readiness.seal_quality_readiness(writer,18,startup);seal=entry._document(root,ref)
    assert seen==[1] and seal['inventory']==before and seal['slot']==18
    wire=json.loads((root/ref['path']).read_text())
    assert wire['inventory']['base']['path']=='meta/quality-readiness-16.json'
    chain=[];cursor=ref;chain_bytes=0
    while cursor is not None:
        assert cursor['path'] not in chain
        path=root/cursor['path'];assert file_digest(path)==cursor['sha256']
        chain.append(cursor['path']);chain_bytes+=path.stat().st_size
        item=json.loads(path.read_text())
        cursor=item['inventory']['base'] if '_inventory_encoding' in item else None
    assert len(chain)==11 and 'meta/quality-readiness-17.json' not in chain and chain_bytes<8*1024**2
    assert seal['control_inventory']==entry.readiness._controls(writer,'not-a-control')
    assert entry._document(writer.root,seal['guard'])['status']=='finished'
    assert seal['predispatch_review']==entry._expected_review(18,startup,seal)
    assert all(entry.readiness._inventory(root)[n]==v for n,v in before.items())
    assert len(list(root.rglob('*.pt')))==12 and not (root/'quality/18/process').exists()
    with pytest.raises(ValueError):start.verify_confirmation_request(writer,18,startup)
    with pytest.raises(ValueError,match='cannot be retried'):entry.readiness.seal_quality_readiness(writer,18,startup)
    census=entry.snapshot(writer.root)
    (writer.root.parent/'slot18-readiness-census.json').write_text(json.dumps(dict(
        readiness_bytes=(root/ref['path']).stat().st_size,metadata_bytes=census['metadata_bytes'],
        meta_group_bytes=census['groups']['evidence/meta'],live_checkpoint_files=12,
        completed_quality_records=18,inventory_chain_documents=len(chain),inventory_chain_bytes=chain_bytes,real_panel_fit_proven=False)))


def test_entry_reconstructs_fresh_pair_and_guard_precedes_synthetic_producer(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence';before=entry.readiness._inventory(root)
    original=entry.remaining_projection;seen=[]
    def projection(p,pilots,records,rates,selection,boundaries):
        assert len(pilots)==3 and len(records)==18 and boundaries==3
        assert records[-1]['stage']=='confirm' and records[-1]['seed']==3511
        seen.append(1);return original(p,pilots,records,rates,selection,boundaries)
    monkeypatch.setattr(entry,'remaining_projection',projection)
    value=entry.verify_ready_worker(**args)
    assert seen==[1] and value['storage_verified'] and not value['dispatch_authorized']
    assert before==entry.readiness._inventory(root) and len(list(root.rglob('*.pt')))==12
    assert not (root/'quality/18/checkpoints').exists()
    calls=[]
    def produce(w,slot,plan_sha,request_sha,*,projection_ref):
        guard=json.loads((w.root/'control/quality-entry-18.json').read_text())
        assert slot==18 and guard['status']=='starting' and guard['limits']['timeout_seconds']==7200
        calls.append(1);return 'synthetic only'
    monkeypatch.setitem(sys.modules,'src.research.tokenizer_wide_hybrid_quality_execution_v2',SimpleNamespace(execute_quality_outputs=produce))
    args['argv']=args.pop('actual_argv')
    assert entry.run_worker(**args)=='synthetic only' and calls==[1]
    with pytest.raises(ValueError,match='cannot be retried'):entry.run_worker(**args)


def test_entry_rejects_history_transport_capacity_and_late_mutations(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence'
    for name in ('meta/source.txt','meta/numerical.json','meta/quality-dispatch-16.json','meta/quality-dispatch-17.json','quality/15/result.json','quality/16/result.json','quality/17/result.json',
        '.retention/pilot/receipt.json','.retention/rank/deletions.jsonl',
        'quality/09/checkpoints/final_001200_123456.pt','quality/18/process/launch.json','quality/18/result.json'):
        path=root/name;raw=path.read_bytes() if path.exists() else None;path.write_bytes(b'{}')
        try:
            with pytest.raises((ValueError,KeyError)):entry.verify_ready_worker(**args)
        finally:
            if raw is None:path.unlink()
            else:path.write_bytes(raw)
    # Both files of the completed fresh pair remain protected and hash bound.
    for path in [* (root/'quality/15/checkpoints').iterdir(),* (root/'quality/16/checkpoints').iterdir(),* (root/'quality/17/checkpoints').iterdir()]:
        raw=path.read_bytes();path.write_bytes(b'bad')
        try:
            with pytest.raises(ValueError):entry.verify_ready_worker(**args)
        finally:path.write_bytes(raw)
    for bad in ('parent','locks','argv','stop','guard'):
        with monkeypatch.context() as patch:
            changed=dict(args);path=writer.root/'control/quality-readiness-18.json';raw=path.read_bytes()
            if bad=='parent':patch.setattr(entry.os,'getppid',lambda:1)
            elif bad=='locks':patch.setattr(writer,'lock_fds',(writer.lock_fds[0],writer.lock_fds[0]))
            elif bad=='argv':changed['actual_argv']=args['actual_argv']+['extra']
            elif bad=='guard':path.write_bytes(b'{}')
            else:(writer.root/'STOP').touch()
            try:
                with pytest.raises(ValueError):entry.verify_ready_worker(**changed)
            finally:
                if bad=='stop':(writer.root/'STOP').unlink()
                if bad=='guard':path.write_bytes(raw)
    startup=entry._document(root,args['startup_ref']);seal=entry._document(root,args['readiness_ref'])
    for key,value in [('completed_records_sha256','a'*64),('checkpoint_pairs_sha256','b'*64),
        ('stages_sha256','c'*64),('frozen_files_sha256','d'*64),('audited_indices',[]),('prior_dispatch_ref',{})]:
        altered=copy.deepcopy(startup);altered[key]=value
        with pytest.raises((ValueError,KeyError)):entry._storage(writer,18,seal,altered,seal['inventory'])
    needed=seal['required_free_bytes']
    for values,valid in [([needed,needed],True),([needed-1],False),([needed,needed-1],False)]:
        readings=iter(values)
        with monkeypatch.context() as patch:
            patch.setattr(entry,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=next(readings))))
            if valid:assert entry.verify_ready_worker(**args)['required_free_bytes']==needed
            else:
                with pytest.raises(ValueError):entry.verify_ready_worker(**args)
    original=entry._storage;log=root/'quality/18/process/worker.log';journal=root/'.retention/rank/deletions.jsonl'
    for change in ('append','rewrite','history'):
        raw_log=log.read_bytes();raw_journal=journal.read_bytes()
        def late(*a,**kw):
            value=original(*a,**kw)
            if change=='append':
                with log.open('ab') as out:out.write(b'bounded append\n')
            elif change=='rewrite':log.write_bytes(b'x'*len(raw_log))
            else:journal.write_bytes(b'late mutation')
            return value
        with monkeypatch.context() as patch:
            patch.setattr(entry,'_storage',late)
            try:
                if change=='append':assert entry.verify_ready_worker(**args)['storage_verified']
                else:
                    with pytest.raises(ValueError):entry.verify_ready_worker(**args)
            finally:log.write_bytes(raw_log);journal.write_bytes(raw_journal)


@pytest.mark.parametrize('bad',['claims','late_control','publication'])
def test_failed_seal_preserves_attempt_and_never_retries(writer,monkeypatch,bad):
    startup=prepared(writer,monkeypatch);root=writer.root/'evidence'
    if bad=='claims':
        original=start.verify_confirmation_for_readiness
        def changed(*a,**kw):
            value=original(*a,**kw);value['prior_transport_verified']=False;return value
        monkeypatch.setattr(start,'verify_confirmation_for_readiness',changed)
    elif bad=='late_control':
        original=entry.readiness._ReadinessGuard.__exit__
        def late(*a,**kw):
            value=original(*a,**kw);writer.publish_json('control/process-quality-19.json',{});return value
        monkeypatch.setattr(entry.readiness._ReadinessGuard,'__exit__',late)
    else:
        original=writer.publish_json
        def fail(name,*a,**kw):
            if name=='evidence/meta/quality-readiness-18.json':
                (writer.root/(name+'.tmp')).write_bytes(b'partial');raise OSError('synthetic publication failure')
            return original(name,*a,**kw)
        monkeypatch.setattr(writer,'publish_json',fail)
    with pytest.raises((ValueError,OSError)):entry.readiness.seal_quality_readiness(writer,18,startup)
    assert not (root/'meta/quality-readiness-18.json').exists()
    assert (root/'meta/quality-readiness-18-failure.json').exists()
    assert len(list(root.rglob('*.pt')))==12 and not (root/'quality/18/process').exists()
    before=entry.readiness._inventory(root) if bad!='publication' else None
    with pytest.raises(ValueError,match='cannot be retried'):entry.readiness.seal_quality_readiness(writer,18,startup)
    if before is not None:assert entry.readiness._inventory(root)==before


@pytest.mark.parametrize('slot',[19,20,21,22,23,24,True,18.,'18',-1])
def test_unsupported_handoff_and_readiness_slots_fail_before_mutation(writer,slot):
    with pytest.raises(ValueError):entry.worker_argv(writer,slot,{}, {})
    with pytest.raises(ValueError):entry.readiness.seal_quality_readiness(writer,slot,{})
    assert not (writer.root/'evidence').exists()


def test_slot18_cli_requires_canonical_sealed_arguments_before_worker(writer,monkeypatch):
    startup=dict(path='meta/quality-confirm-startup-18.json',sha256='a'*64)
    ready=dict(path='meta/quality-readiness-18.json',sha256='b'*64)
    argv=entry.worker_argv(writer,18,startup,ready);seen=[]
    def run(w,slot,s,r,actual):
        assert w.root==writer.root and w.lock_fds==writer.lock_fds
        assert slot==18 and s==startup and r==ready and actual==argv
        seen.append(True);return 'canonical CLI only'
    monkeypatch.setattr(entry,'run_worker',run)
    assert entry.main(argv[3:])=='canonical CLI only' and seen==[True]
    with pytest.raises(ValueError,match='Noncanonical'):
        entry.main([*argv[3:8],*argv[10:],*argv[8:10]])
    unsupported=list(argv[3:]);unsupported[4]='19'
    with pytest.raises(SystemExit):entry.main(unsupported)
    assert seen==[True] and not (writer.root/'evidence').exists()
