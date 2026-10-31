"""Activation executable gates on disposable plans; never launch research training.

Prerequisite documents and input identities below are deliberately synthetic.
Real inherited-lock subprocess transport is tested with harmless child commands;
actual training dispatch is intercepted explicitly at the process boundary.
"""
import copy
import fcntl
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from src.research import tokenizer_activation_entry as e
from src.research import tokenizer_activation_retention as retention
from src.research import tokenizer_wide_hybrid_cleanup as cleanup
from src.research.tokenizer_activation_session import Session,reference
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.protocol import digest
from tests.test_tokenizer_activation_session import ledger


@pytest.fixture
def frozen(tmp_path,monkeypatch):
    root=tmp_path/'research_runs/tokenizer-activation-901';root.mkdir(parents=True)
    lock=tmp_path/'global.lock';lock.touch();(root/'.lock').touch()
    monkeypatch.setattr(cleanup,'GLOBAL_LOCK',lock);monkeypatch.setattr(e,'GLOBAL_LOCK',lock)
    monkeypatch.setattr(e,'ROOT',tmp_path)
    with lock.open('rb') as g,(root/'.lock').open('rb') as c:
        fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(c,fcntl.LOCK_EX|fcntl.LOCK_NB)
        writer=MetadataWriter(root,(g.fileno(),c.fileno()))
        prerequisites={};base=tmp_path/'research/tokenizer-evidence/activation-preparation-001';base.mkdir(parents=True)
        inputs=dict(tokenizer={'synthetic':True},training96_sha256='a'*64,validation_sha256='b'*64,
                    validation_targets=34278,validation_target_bytes=111536)
        storage=dict(trajectory_allowance_bytes=e.ALLOWANCE,maximum_weight_bytes=e.HIGH_WATER,
                     metadata_bytes=e.METADATA,reserve_bytes=e.RESERVE,full_required_free_bytes=e.FULL_FREE)
        source=e.source_fingerprint();runtime=e.runtime_identity()
        for name,status in e.PREREQUISITES.items():
            if name=='capacity':continue
            path=base/(name+'.json');path.write_text(json.dumps(dict(status=status,synthetic_fixture=True)))
            prerequisites[name]=reference(tmp_path,str(path.relative_to(tmp_path)))
        path=base/'capacity.json';path.write_text(json.dumps(dict(status='passed',source_sha256=digest(source),
            runtime=runtime,prerequisites=copy.deepcopy(prerequisites),storage=storage,inputs=inputs,complete_executable_review=True)))
        prerequisites['capacity']=reference(tmp_path,str(path.relative_to(tmp_path)))
        plan=dict(kind='activation_campaign_v1',status='frozen',root=str(root),source=source,runtime=runtime,
                  design=e.design(),limits=e.LIMITS,inputs=inputs,prerequisites=prerequisites,storage=storage)
        writer.publish_json('evidence/meta/activation-plan.json',plan)
        yield Session(writer,reference(root,'evidence/meta/activation-plan.json'))


def rewrite_plan(ledger,edit):
    path=ledger.root/ledger.plan['path'];plan=json.loads(path.read_text());edit(plan);path.write_text(json.dumps(plan))
    return Session(ledger.writer,reference(ledger.root,ledger.plan['path']))


def test_complete_frozen_binding_and_capacity(frozen):
    assert e.bind(frozen)['status']=='frozen'
    result=e.require_capacity(frozen)
    assert result['required_free_bytes']==e.FULL_FREE-result['metadata_bytes']
    assert e.FULL_FREE==15169995168


@pytest.mark.parametrize('field',['status','root','source','runtime','design','limits','inputs','prerequisites','storage'])
def test_frozen_gate_rejects_missing_changed_or_partial_fields(frozen,field):
    def edit(p):
        if field=='status':p[field]='draft'
        elif field=='root':p[field]+='/other'
        else:p[field]={}
    changed=rewrite_plan(frozen,edit)
    with pytest.raises((ValueError,KeyError)):e.bind(changed)


def test_changed_prerequisite_and_stop_are_not_authorization(frozen):
    plan=e.bind(frozen);ref=plan['prerequisites']['worker']
    path=e.ROOT/ref['path'];path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='Prerequisite'):e.bind(frozen)
    frozen.writer.publish_json('STOP',{'reason':'fixture stop'})
    with pytest.raises(ValueError,match='STOP'):e.bind(frozen)


def test_full_panel_storage_and_unknown_weight_credit(frozen,monkeypatch):
    monkeypatch.setattr(e.shutil,'disk_usage',lambda p:SimpleNamespace(free=e.RESERVE))
    with pytest.raises(ValueError,match='Full remaining'):e.require_capacity(frozen)
    path=frozen.root/'evidence/quality/11/checkpoints/unknown.pt';path.parent.mkdir(parents=True);path.write_bytes(b'fixture')
    with pytest.raises(ValueError,match='Unknown weights'):e.require_capacity(frozen)


def test_idle_process_inspection_rejects_other_workers(monkeypatch):
    own=os.getpid()
    monkeypatch.setattr(e.subprocess,'check_output',lambda *a,**k:f'{own} 1 /usr/bin/python -m src.research.tokenizer_activation_entry step\n999999 1 /usr/bin/python -m src.research.tokenizer_study worker\n888888 1 /Applications/UnrelatedApp\n')
    with pytest.raises(ValueError,match='Other research'):e.idle()
    assert e.idle((999999,))['checked'] is True


def test_real_subprocess_receipt_and_inherited_two_locks(ledger):
    code='import os,fcntl; fds='+repr(ledger.writer.lock_fds)+'; [(os.fstat(fd),fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)) for fd in fds]; print("inherited locks verified")'
    result=e.run_process(ledger.writer,'pilots',0,[sys.executable,'-c',code],Path.cwd(),
        required_free_bytes=e.RESERVE,output_limit_bytes=8*1024**2,timeout_seconds=10)
    assert result['status']=='completed' and result['returncode']==0
    assert 'inherited locks verified' in (ledger.root/'evidence/pilots/00/process/worker.log').read_text()
    with pytest.raises(FileExistsError):e.run_process(ledger.writer,'pilots',0,[sys.executable,'-c','pass'],Path.cwd(),
        required_free_bytes=e.RESERVE,output_limit_bytes=8*1024**2,timeout_seconds=10)


def test_real_failed_subprocess_is_terminal(ledger):
    with pytest.raises(RuntimeError,match='code 3'):
        e.run_process(ledger.writer,'pilots',0,[sys.executable,'-c','raise SystemExit(3)'],Path.cwd(),
            required_free_bytes=e.RESERVE,output_limit_bytes=8*1024**2,timeout_seconds=10)
    result=json.loads((ledger.root/'evidence/pilots/00/process/process.json').read_text())
    assert result['status']=='failed' and result['returncode']==3


def test_step_refuses_unfrozen_before_any_request(ledger):
    with pytest.raises(ValueError):e.run_once(ledger)
    assert not (ledger.root/'evidence/pilots').exists()


def test_dispatch_failure_latches_reserved_attempt(frozen,monkeypatch):
    monkeypatch.setattr(e,'idle',lambda *args:{'checked':True})
    # Freeze validation and preparation guard execute normally. Intercept only
    # the actual process launch; no model or scientific fixture is dispatched.
    calls=[]
    def fail(*args,**kwargs):calls.append(kwargs);raise RuntimeError('injected child launch failure')
    monkeypatch.setattr(e,'run_process',fail)
    with pytest.raises(RuntimeError,match='injected child'):e.run_once(frozen)
    assert calls and calls[0]['timeout_seconds']==600
    state=frozen.inspect();assert state['failed'] and state['pending'] is not None
    with pytest.raises(ValueError):e.run_once(frozen)
    assert len(calls)==1


def test_child_parent_seal_rejection_precedes_producer(frozen,monkeypatch):
    request=frozen.reserve();name='evidence/meta/dispatch-seal-pilots-00.json'
    frozen.writer.publish_json(name,dict(kind=e.KIND,plan=frozen.plan,request=request,parent_pid=os.getppid()+100000,
                                        lock_fds=list(frozen.writer.lock_fds)))
    seal=reference(frozen.root,name)
    monkeypatch.setattr(e,'produce',lambda *a:pytest.fail('producer reached invalid child gate'))
    with pytest.raises(ValueError,match='parent/locks'):e.worker(frozen,request,seal,os.getppid())


def test_sealed_child_and_guarded_parent_success(frozen,monkeypatch):
    """Real entry/acceptance gates with explicitly synthetic process and numerics.

    Separate subprocess tests exercise OS inheritance/exit. This test connects
    actual seal validation, child preflight, guards, parent checks and ledger;
    it does not claim a real MPS child executed.
    """
    from tests.test_tokenizer_activation_acceptance import completed
    monkeypatch.setattr(e,'idle',lambda *a:{'checked':True})
    monkeypatch.setattr(e,'_group_exists',lambda pid:False)
    monkeypatch.setattr(e.os,'getppid',lambda:os.getpid())
    def synthetic_producer(ledger,request,inputs,recheck):
        recheck()
        with monkeypatch.context() as local:
            local.setattr(ledger,'reserve',lambda:request)
            # Input and tensor reviewers stay mocked for the parent stage;
            # actual metadata/trace/coverage and ledger checks remain active.
            completed(ledger,monkeypatch)
    monkeypatch.setattr(e,'produce',synthetic_producer)
    def synthetic_process(writer,stage,slot,argv,cwd,**kwargs):
        seal=reference(frozen.root,f'evidence/meta/dispatch-seal-{stage}-{slot:02d}.json')
        request=frozen.document(seal)['request']
        assert argv==e.argv_for(frozen,request,seal,os.getpid(),writer.lock_fds)
        kwargs['parent_check']()
        e.worker(frozen,request,seal,os.getpid())
        value=dict(status='completed',returncode=0,error=None,parent_pid=os.getpid(),pid=os.getpid(),
            argv=argv,capture_kind='bounded_pipe_v2',log_truncated=False,reserve_bytes=e.RESERVE,
            timeout_seconds=kwargs['timeout_seconds'],required_free_bytes=kwargs['required_free_bytes'],
            output_limit_bytes=kwargs['output_limit_bytes'],log_limit_bytes=4*1024**2,
            observation=dict(free_bytes=e.FULL_FREE,output_bytes=10000,log_bytes=0),elapsed_seconds=.1,
            synthetic_process_fixture=True)
        writer.publish_json(f'evidence/{stage}/{slot:02d}/process/process.json',value)
        return value
    monkeypatch.setattr(e,'run_process',synthetic_process)
    result=e.run_once(frozen)
    state=frozen.inspect()
    assert len(state['pilots'])==1 and state['next']['spec']['family']=='gelu_exact'
    assert e.prefix(frozen,state)==[result]
    entry=frozen.root/'control/activation-entry-pilots-00.json'
    entry.write_bytes(entry.read_bytes()+b' ')
    # A guard hash is part of the next-prefix proof, even when its JSON is unchanged.
    # Entry guard needs explicit pinning in the completion receipt for this to fail.
    with pytest.raises(ValueError):e.prefix(frozen,state)
