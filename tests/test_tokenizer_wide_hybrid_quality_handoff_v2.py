"""Transport-only synthetic snapshots; no numerical, startup or model claims."""
import json
import os

import pytest

from src.research import tokenizer_wide_hybrid_quality_handoff_v2 as handoff
from src.research.protocol import file_digest
from test_tokenizer_wide_hybrid_metadata_writer import writer


def inventory(root):
    return {str(p.relative_to(root)):dict(size=p.stat().st_size,sha256=file_digest(p))
            for p in root.rglob('*') if p.is_file()}


def setup(writer,slot=0):
    root=writer.root/'evidence';stem=f'quality/{slot:02d}'
    writer.publish_json('evidence/meta/source.json',{'synthetic':True})
    for i in range(slot):
        writer.publish_json(f'evidence/quality/{i:02d}/result.json',{'synthetic':True})
        path=root/f'quality/{i:02d}/checkpoints/raw.pt';path.parent.mkdir();path.write_bytes(b'old synthetic weights')
    rq=writer.publish_json('evidence/'+stem+'/request.json',{'synthetic':True})
    name='meta/quality-first-startup.json' if slot==0 else f'meta/quality-lr-startup-{slot:02d}.json'
    path=writer.publish_json('evidence/'+name,{'request':dict(path=stem+'/request.json',sha256=file_digest(rq))})
    ref=dict(path=name,sha256=file_digest(path));before=inventory(root);argv=handoff.worker_argv(writer,slot,ref)
    limits=dict(required_free_bytes=30_631_281_412,output_limit_bytes=2*1024**3)
    launch=dict(status='preflight',argv=argv,parent_pid=os.getppid(),pid=None,returncode=None,error=None,
        **limits,reserve_bytes=8*1024**3,log_limit_bytes=4*1024**2,timeout_seconds=7200,
        observation=None,capture_kind='bounded_pipe_v2',log_truncated=False)
    writer.publish_json('evidence/'+stem+'/process/launch.json',launch)
    with writer.open_append('evidence/'+stem+'/process/worker.log') as stream:stream.write(b'initial log\n')
    return dict(writer=writer,slot=slot,startup_ref=ref,actual_argv=argv,inventory=before,**limits)


@pytest.mark.parametrize('slot',[0,1,4,8])
def test_exact_transport_is_readonly_and_grants_no_other_credit(writer,slot):
    args=setup(writer,slot);before=inventory(writer.root/'evidence')
    checked=handoff.verify_live_handoff(**args)
    assert checked['live_transport_verified']
    for key in ('quality_prerequisites_verified','storage_verified','dispatch_authorized'):assert checked[key] is False
    assert inventory(writer.root/'evidence')==before


@pytest.mark.parametrize('key,value',[
    ('parent_pid',1),('pid',123),('returncode',0),('status','running'),('error','failed'),
    ('required_free_bytes',8*1024**3),('output_limit_bytes',1),('reserve_bytes',1),
    ('timeout_seconds',600),('log_limit_bytes',4096),('capture_kind','file'),
    ('log_truncated',True),('observation',{}),('unexpected',True)])
def test_launch_identity_and_exact_limits_cannot_change(writer,key,value):
    args=setup(writer);p=writer.root/'evidence/quality/00/process/launch.json'
    data=json.loads(p.read_text());data[key]=value;p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='launch identity'):handoff.verify_live_handoff(**args)


@pytest.mark.parametrize('bad',['argv','startup','request','source','weight','extra','later','terminal',
    'partial','missing_log','new_weight','stop','locks'])
def test_changed_or_incomplete_handoff_rejects(writer,bad,monkeypatch):
    args=setup(writer,1);root=writer.root/'evidence'
    if bad=='argv':args['actual_argv']=args['actual_argv']+['extra']
    elif bad=='startup':args['startup_ref']['sha256']='0'*64
    elif bad=='request':(root/'quality/01/request.json').write_text('{}')
    elif bad=='source':(root/'meta/source.json').write_text('{}')
    elif bad=='weight':(root/'quality/00/checkpoints/raw.pt').write_bytes(b'changed')
    elif bad=='stop':(writer.root/'STOP').touch()
    elif bad=='locks':monkeypatch.setattr(writer,'lock_fds',(writer.lock_fds[0],writer.lock_fds[0]))
    elif bad=='missing_log':(root/'quality/01/process/worker.log').unlink()
    else:
        name={'extra':'meta/new.json','later':'quality/02/request.json','terminal':'quality/01/process/process.json',
            'partial':'quality/01/process/process.json.tmp','new_weight':'quality/01/checkpoints/raw.pt'}[bad]
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    with pytest.raises(ValueError):handoff.verify_live_handoff(**args)


@pytest.mark.parametrize('change',['append','rewrite','truncate','replace','oversize','extra','parent','stop','locks'])
def test_parent_can_append_only_bounded_log_while_immutable_files_are_hashed(writer,monkeypatch,change):
    args=setup(writer,1);root=writer.root/'evidence';original=handoff._check;seen=[]
    def check(r,name,binding,deadline):
        if name=='quality/00/checkpoints/raw.pt':
            # This publication would fail if child hashing held the namespace lock.
            writer.publish_json('control/synthetic-parent-observation.json',{'alive':True})
            log=root/'quality/01/process/worker.log'
            if change=='append':
                with log.open('ab') as stream:stream.write(b'parent appended\n')
            elif change=='rewrite':log.write_bytes(b'changed log\n')
            elif change=='truncate':log.write_bytes(b'x')
            elif change=='replace':
                replacement=root/'quality/01/process/replacement';replacement.write_bytes(log.read_bytes())
                os.replace(replacement,log)
            elif change=='oversize':
                with log.open('ab') as stream:stream.truncate(handoff.LOG_LIMIT+1)
            elif change=='extra':writer.publish_json('evidence/meta/new.json',{})
            elif change=='parent':monkeypatch.setattr(handoff.os,'getppid',lambda:1)
            elif change=='stop':(writer.root/'STOP').touch()
            else:monkeypatch.setattr(writer,'lock_fds',(writer.lock_fds[0],writer.lock_fds[0]))
            seen.append(True)
        return original(r,name,binding,deadline)
    monkeypatch.setattr(handoff,'_check',check)
    if change=='append':assert handoff.verify_live_handoff(**args)['live_transport_verified']
    else:
        with pytest.raises(ValueError):handoff.verify_live_handoff(**args)
    assert seen==[True]


@pytest.mark.parametrize('slot',[-1,9,15,True])
def test_undeclared_stage_is_not_a_transport_shortcut(writer,slot):
    with pytest.raises(ValueError):handoff.worker_argv(writer,slot,{'path':'meta/quality-first-startup.json','sha256':'0'*64})
