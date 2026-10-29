"""Transport and boundary composition; full quality replay is mocked here only."""
import copy
import json
import os

import pytest

from src.research import tokenizer_wide_hybrid_inference_acceptance_v2 as acceptance
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_process_v2 import _encoded_size
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_inference_entry_v2 import setup

entry=acceptance.entry;outputs=acceptance.outputs;prep=acceptance.prep


def terminal(writer,seed):
    root=writer.root/'evidence';stem=f'inference/{seed}'
    launch=json.loads((root/stem/'process/launch.json').read_text())
    worker=json.loads((root/stem/'worker-outputs.json').read_text())
    total=sum(v['size'] for n,v in outputs.snapshot(writer.root)['files'].items() if n.startswith('evidence/'+stem+'/'))
    log=(root/stem/'process/worker.log').stat().st_size
    state=dict(launch,status='completed',pid=worker['pid'],returncode=0,elapsed_seconds=1.,
        observation=dict(free_bytes=64*1024**3,output_bytes=total,log_bytes=log))
    writer.publish_json(f'control/process-inference-{seed}.json',state,replace=True)
    state=copy.deepcopy(state);state['elapsed_seconds']=1.1
    for _ in range(10):
        size=_encoded_size(state)
        if state['observation']['output_bytes']==total+size:break
        state['observation']['output_bytes']=total+size
    else:raise AssertionError('Synthetic terminal size failed to converge')
    writer.publish_json('evidence/'+stem+'/process/process.json',state)
    return state


def completed(writer,monkeypatch):
    ready,argv=setup(writer,monkeypatch)
    ref=entry.run_worker(writer,3511,ready,argv)
    process=terminal(writer,3511);calls=[]
    def full(w,**kw):
        calls.append(kw['inference_request_ref'])
        root=w.root/'evidence';request=json.loads((root/prep.REQUEST).read_text())
        inventory=prep.gate.history.acceptance._inventory(root)
        controls=prep.gate.history.acceptance.readiness._controls(w,'none')
        files={str(root/n):v['sha256'] for n,v in inventory.items()}
        files.update({str(w.root/n):v['sha256'] for n,v in controls.items()})
        return dict(plan_sha256=request['plan_sha256'],completion_ref=request['completion_ref'],contract=request['contract'],
            selection=request['selection'],tokenizer={'synthetic':True},token_sha256=request['token_sha256'],triples=writer._test_triples,
            records_sha256=request['records_sha256'],checkpoint_pairs_sha256=request['checkpoint_pairs_sha256'],
            retention_sha256=request['retention_sha256'],inventory_sha256=request['quality_inventory_sha256'],
            controls_sha256=request['quality_controls_sha256'],files=files)
    monkeypatch.setattr(prep.gate,'_bind',full)
    monkeypatch.setattr(acceptance,'_group_exists',lambda p:False)
    return ref,process,calls


def test_completed_acceptance_replays_and_rebinds_historical_descriptors(writer,monkeypatch):
    ref,process,calls=completed(writer,monkeypatch)
    before={str(p):p.read_bytes() for p in writer.root.rglob('*') if p.is_file()}
    with monkeypatch.context() as patch:
        patch.setattr(acceptance.os,'getpid',lambda:process['parent_pid'])
        accepted=acceptance.accept_completed(writer,ref,process)
        assert calls==[ref]
        with pytest.raises(ValueError):acceptance.accept_completed(writer,ref,process)
    assert all(__import__('pathlib').Path(n).read_bytes()==v for n,v in before.items())
    fds=tuple(os.dup(fd) for fd in writer.lock_fds)
    try:
        reopened=MetadataWriter(writer.root,fds)
        report=acceptance.verify_boundary(reopened,accepted)
        assert report['quality_history_reaudited'] and not report['dispatch_authorized']
        assert report['scope']['transport']['lock_fds']==list(writer.lock_fds)
        assert list(fds)!=list(writer.lock_fds)
    finally:
        for fd in fds:os.close(fd)
    assert calls==[ref,accepted]
    next_ref=acceptance.prepare_next_readiness(writer,accepted)
    assert next_ref['path']=='inference/3512/ready.json'
    assert acceptance._scope(writer,next_ref)['state']=='ready'
    assert not (writer.root/'evidence/inference/3512/worker-attempt.json').exists()
    with pytest.raises(ValueError):acceptance.prepare_next_readiness(writer,accepted)


@pytest.mark.parametrize('bad',['parent','live_group','returned','current_fds','terminal_status','exit','extra',
    'terminal_pid','terminal_time','terminal_bytes','terminal_log','terminal_free','argv','launch','supervisor',
    'supervisor_bytes','supervisor_time','supervisor_free','worker_pid','guard','future','partial','failure',
    'original','model','claim','receipt'])
def test_transport_and_prefix_fail_closed(writer,monkeypatch,bad):
    ref,process,calls=completed(writer,monkeypatch);root=writer.root/'evidence';stem=root/'inference/3511'
    def change(path,fn):
        value=json.loads(path.read_text());fn(value);path.write_text(json.dumps(value))
    terminal_path=stem/'process/process.json';control=writer.root/'control/process-inference-3511.json'
    if bad=='parent':
        with pytest.raises(ValueError,match='Current parent'):acceptance.accept_completed(writer,ref,process)
        return
    if bad in ('live_group','returned','current_fds'):
        monkeypatch.setattr(acceptance.os,'getpid',lambda:process['parent_pid'])
        if bad=='live_group':monkeypatch.setattr(acceptance,'_group_exists',lambda p:True)
        elif bad=='returned':process=copy.deepcopy(process);process['elapsed_seconds']=2.
        else:
            original=entry.worker_argv
            def changed(*args):
                value=original(*args);value[11]='999';value[13]='998';return value
            monkeypatch.setattr(entry,'worker_argv',changed)
        with pytest.raises(ValueError):acceptance.accept_completed(writer,ref,process)
        assert not (stem/'review.json').exists()
        return
    if bad in ('terminal_status','exit','extra','terminal_pid','terminal_time','argv'):
        def alter(v):
            if bad=='terminal_status':v['status']='running'
            elif bad=='exit':v['returncode']=True
            elif bad=='extra':v['unknown']=True
            elif bad=='terminal_pid':v['pid']=v['parent_pid']
            elif bad=='terminal_time':v['elapsed_seconds']=600
            else:v['argv'][11]='03'
        change(terminal_path,alter)
    elif bad in ('terminal_bytes','terminal_log','terminal_free'):
        key={'terminal_bytes':'output_bytes','terminal_log':'log_bytes','terminal_free':'free_bytes'}[bad]
        change(terminal_path,lambda v:v['observation'].update({key:1}))
    elif bad=='launch':change(stem/'process/launch.json',lambda v:v.update(timeout_seconds=601))
    elif bad.startswith('supervisor'):
        def alter(v):
            if bad=='supervisor':v['status']='running'
            elif bad=='supervisor_time':v['elapsed_seconds']=2.
            else:v['observation']['free_bytes' if bad=='supervisor_free' else 'output_bytes']=1
        change(control,alter)
    elif bad=='worker_pid':change(stem/'worker-outputs.json',lambda v:v.update(pid=v['pid']+1));ref=acceptance._ref(root,ref['path'])
    elif bad=='guard':change(writer.root/'control/inference-worker-3511.json',lambda v:v.update(status='failed'))
    elif bad in ('future','partial','failure','receipt'):
        name={'future':'inference/3513/ready.json','partial':'inference/3511/result.json.tmp','failure':'inference/3511/worker-failure.json','receipt':'inference/3511/review.json'}[bad]
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    elif bad=='original':(root/'quality/23/checkpoints/raw.pt').write_bytes(b'changed')
    elif bad=='model':change(stem/'models.json',lambda v:v['hybrid'].update(parameters=1))
    elif bad=='claim':change(stem/'worker-outputs.json',lambda v:v.update(quality_history_reaudited=True));ref=acceptance._ref(root,ref['path'])
    with pytest.raises(ValueError):acceptance.verify_boundary(writer,ref)
    assert not calls


def test_rehashed_accepted_receipt_cannot_claim_different_history(writer,monkeypatch):
    ref,process,calls=completed(writer,monkeypatch)
    monkeypatch.setattr(acceptance.os,'getpid',lambda:process['parent_pid'])
    ref=acceptance.accept_completed(writer,ref,process)
    path=writer.root/'evidence'/ref['path'];value=json.loads(path.read_text());value['quality_binding']['retention_sha256']='0'*64
    path.write_text(json.dumps(value));ref=acceptance._ref(writer.root/'evidence',ref['path'])
    with pytest.raises(ValueError,match='accepted inference receipt'):acceptance.verify_boundary(writer,ref)
    assert len(calls)==1
