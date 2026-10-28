"""Synthetic full idle seal -> live child checks; no real child/model launched."""
import json
import os
import sys
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_quality_entry_v2 as entry
from src.research.protocol import file_digest
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_quality_readiness_v2 import prepared


def setup(writer,monkeypatch,slot=1):
    startup=prepared(writer,monkeypatch,slot)
    ref=entry.readiness.seal_quality_readiness(writer,slot,startup);root=writer.root/'evidence'
    seal=entry._document(root,ref)
    # The real shared-lock fixture stays in one process; simulate its child PID
    # relation only. Real process capture/parent identity has separate tests.
    monkeypatch.setattr(entry.os,'getppid',os.getpid)
    monkeypatch.setattr(entry,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=64*1024**3)))
    argv=entry.worker_argv(writer,slot,startup,ref)
    launch=dict(status='preflight',argv=argv,parent_pid=os.getpid(),pid=None,returncode=None,error=None,
        required_free_bytes=seal['required_free_bytes'],reserve_bytes=8*1024**3,
        output_limit_bytes=seal['output_limit_bytes'],log_limit_bytes=4*1024**2,timeout_seconds=7200,
        observation=None,capture_kind='bounded_pipe_v2',log_truncated=False)
    writer.publish_json(f'evidence/quality/{slot:02d}/process/launch.json',launch)
    with writer.open_append(f'evidence/quality/{slot:02d}/process/worker.log') as stream:stream.write(b'child imports\n')
    return dict(writer=writer,slot=slot,startup_ref=startup,readiness_ref=ref,actual_argv=argv)


@pytest.mark.parametrize('slot',[0,1,4,8])
def test_same_parent_seal_rebinds_full_inputs_and_actual_storage(writer,monkeypatch,slot):
    args=setup(writer,monkeypatch,slot);before=entry.readiness._inventory(writer.root/'evidence')
    checked=entry.verify_ready_worker(**args)
    assert all(checked[k] for k in ('parent_idle_audit_bound','frozen_plan_verified','live_transport_verified','storage_verified'))
    assert not checked['dispatch_authorized'] and entry.readiness._inventory(writer.root/'evidence')==before
    assert args['actual_argv'][-2:]==['--readiness-sha256',args['readiness_ref']['sha256']]


@pytest.mark.parametrize('bad',['seal','argv','source','numeric','weight','control','startup_guard','pilot_receipt',
    'extra','process_terminal','new_weight','stop','locks','parent'])
def test_changed_inputs_never_reach_quality_entry(writer,monkeypatch,bad):
    args=setup(writer,monkeypatch);root=writer.root/'evidence'
    paths={'seal':args['readiness_ref']['path'],'source':'meta/source.txt','numeric':'meta/numerical.json',
           'weight':'quality/00/checkpoints/raw.pt','pilot_receipt':'.retention/pilot/receipt.json'}
    if bad in paths:(root/paths[bad]).write_text('{}')
    elif bad=='argv':args['actual_argv'][-1]='0'*64
    elif bad=='control':(writer.root/'control/quality-readiness-01.json').write_text('{}')
    elif bad=='startup_guard':(writer.root/'control/quality-lr-startup-01.json').write_text('{}')
    elif bad=='stop':(writer.root/'STOP').touch()
    elif bad=='locks':monkeypatch.setattr(writer,'lock_fds',(writer.lock_fds[0],writer.lock_fds[0]))
    elif bad=='parent':monkeypatch.setattr(entry.os,'getppid',lambda:1)
    else:
        name={'extra':'meta/quality-unreviewed.json','process_terminal':'quality/01/process/process.json',
              'new_weight':'quality/01/checkpoints/raw.pt'}[bad]
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    before=entry.readiness._inventory(root)
    with pytest.raises(ValueError):entry.verify_ready_worker(**args)
    assert entry.readiness._inventory(root)==before


@pytest.mark.parametrize('key,value',[('required_free_bytes',8*1024**3),('output_limit_bytes',1024**3),
    ('frozen_files',{}),('inventory',{}),('predispatch_review',{}),('parent_pid',1),('slot',True),
    ('dispatch_authorized',True)])
def test_rehashed_seal_cannot_change_reconstructed_contract(writer,monkeypatch,key,value):
    args=setup(writer,monkeypatch);root=writer.root/'evidence';path=root/args['readiness_ref']['path']
    seal=json.loads(path.read_text());seal[key]=value;path.write_text(json.dumps(seal))
    args['readiness_ref']['sha256']=file_digest(path)
    args['actual_argv']=entry.worker_argv(writer,1,args['startup_ref'],args['readiness_ref'])
    launch=root/'quality/01/process/launch.json';v=json.loads(launch.read_text());v['argv']=args['actual_argv']
    launch.write_text(json.dumps(v))
    with pytest.raises(ValueError):entry.verify_ready_worker(**args)


@pytest.mark.parametrize('offset',[0,-1])
def test_current_full_remaining_storage_boundary(writer,monkeypatch,offset):
    args=setup(writer,monkeypatch);root=writer.root/'evidence'
    needed=json.loads((root/args['readiness_ref']['path']).read_text())['required_free_bytes']
    monkeypatch.setattr(entry,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=needed+offset)))
    if offset==0:assert entry.verify_ready_worker(**args)['required_free_bytes']==needed
    else:
        with pytest.raises(ValueError,match='remaining panel'):entry.verify_ready_worker(**args)


@pytest.mark.parametrize('bad',['space','evidence','control','stop'])
def test_changes_after_live_transport_are_rejected(writer,monkeypatch,bad):
    args=setup(writer,monkeypatch);root=writer.root/'evidence';original=entry._storage
    needed=json.loads((root/args['readiness_ref']['path']).read_text())['required_free_bytes']
    def late(*a,**kw):
        value=original(*a,**kw)
        if bad=='space':monkeypatch.setattr(entry,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=needed-1)))
        elif bad=='evidence':(root/'quality/00/metrics.jsonl').write_text('changed')
        elif bad=='control':(writer.root/'control/quality-lr-startup-01.json').write_text('{}')
        else:(writer.root/'STOP').touch()
        return value
    monkeypatch.setattr(entry,'_storage',late)
    with pytest.raises(ValueError):entry.verify_ready_worker(**args)


def test_current_allocation_is_reconstructed_not_taken_from_seal(writer,monkeypatch):
    args=setup(writer,monkeypatch);original=entry.required_free;seen=[]
    def changed(state,occupied):
        assert occupied>0;seen.append(occupied);return original(state,occupied)+1
    monkeypatch.setattr(entry,'required_free',changed)
    with pytest.raises(ValueError,match='actual allocation'):entry.verify_ready_worker(**args)
    assert len(seen)==1


@pytest.mark.parametrize('valid',[True,False])
def test_child_body_gates_producer_and_rejects_reentry(writer,monkeypatch,valid):
    args=setup(writer,monkeypatch,0);calls=[]
    def produce(w,slot,plan_sha,request_sha,*,projection_ref):
        assert w is writer and slot==0
        guard=json.loads((writer.root/'control/quality-entry-00.json').read_text())
        assert guard['limits']['timeout_seconds']==7200 and guard['status']=='starting'
        calls.append(True);return {'synthetic_producer':True}
    monkeypatch.setitem(sys.modules,'src.research.tokenizer_wide_hybrid_quality_execution_v2',
                        SimpleNamespace(execute_quality_outputs=produce))
    if not valid:(writer.root/'STOP').touch()
    kwargs={**args,'argv':args['actual_argv']};del kwargs['actual_argv']
    if valid:assert entry.run_worker(**kwargs)=={'synthetic_producer':True}
    else:
        with pytest.raises(ValueError):entry.run_worker(**kwargs)
    assert len(calls)==int(valid)
    with pytest.raises(ValueError,match='cannot be retried'):entry.run_worker(**kwargs)


@pytest.mark.parametrize('change',['append','rewrite','truncate','replace','overflow'])
def test_log_remains_append_only_through_final_storage_and_source_checks(writer,monkeypatch,change):
    args=setup(writer,monkeypatch);original=entry._storage
    def late(*a,**kw):
        value=original(*a,**kw)
        writer.publish_json('control/synthetic-parent-live.json',{'alive':True})
        log=writer.root/'evidence/quality/01/process/worker.log'
        if change=='append':
            with log.open('ab') as stream:stream.write(b'parent output\n')
        elif change=='rewrite':log.write_bytes(b'x'*log.stat().st_size)
        elif change=='truncate':log.write_bytes(b'x')
        elif change=='overflow':
            with log.open('ab') as stream:stream.truncate(entry.transport.LOG_LIMIT+1)
        else:
            replacement=log.parent/'replacement';replacement.write_bytes(log.read_bytes());os.replace(replacement,log)
        return value
    monkeypatch.setattr(entry,'_storage',late)
    if change=='append':assert entry.verify_ready_worker(**args)['storage_verified']
    else:
        with pytest.raises(ValueError,match='log changed'):entry.verify_ready_worker(**args)
