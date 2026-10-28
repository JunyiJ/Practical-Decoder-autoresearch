"""Synthetic campaign evidence and producer bodies; never real model dispatch."""
import json
import os
import sys
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_pilot_dispatch_v2 as dispatch
from src.research import tokenizer_wide_hybrid_pilot_startup_v2 as startup
from src.research import tokenizer_wide_hybrid_pilot_execution_v2 as execution
from src.research.protocol import file_digest
from src.research.tokenizer_wide_hybrid_retention import allocations
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_pilot_startup_v2 import setup
from test_tokenizer_wide_hybrid_pilot_audit_v2 import fixture,write


def handoff(writer,monkeypatch,slot=0):
    args=setup(writer,monkeypatch,slot)
    ref=startup.prepare_pilot_request(**args)
    checked=startup.verify_before_process(writer,slot,ref)
    argv=dispatch.worker_argv(writer,slot,ref)
    projection=json.loads((writer.root/'evidence/meta/projection.json').read_text())
    family=fixture_spec(slot)['family']
    launch=dict(status='preflight',argv=argv,parent_pid=os.getppid(),pid=None,returncode=None,error=None,
        required_free_bytes=checked['required_free_bytes'],reserve_bytes=8*1024**3,
        output_limit_bytes=allocations(projection)[family,2],log_limit_bytes=4*1024**2,
        timeout_seconds=600,observation=None,capture_kind='bounded_pipe_v2',log_truncated=False)
    writer.publish_json(f'evidence/pilots/{slot:02d}/process/launch.json',launch)
    with writer.open_append(f'evidence/pilots/{slot:02d}/process/worker.log') as stream:stream.write(b'')
    return ref,argv,launch


def fixture_spec(slot):
    from src.research.tokenizer_wide_hybrid_pilot_config import specs
    return specs()[slot]


@pytest.mark.parametrize('slot',[0,1,2])
def test_child_rechecks_full_receipt_before_body(writer,monkeypatch,slot):
    ref,argv,launch=handoff(writer,monkeypatch,slot);seen=[]
    def body(w,i,plan,rq,**kw):
        assert w is writer and i==slot
        assert rq==file_digest(writer.root/f'evidence/pilots/{slot:02d}/request.json')
        seen.append(kw);return {'synthetic':True}
    monkeypatch.setattr(execution,'execute_pilot_outputs',body)
    assert dispatch.run_worker(writer,slot,ref,argv)=={'synthetic':True} and len(seen)==1
    record=json.loads((writer.root/f'control/pilot-entry-{slot:02d}.json').read_text())
    assert record['status']=='finished' and record['limits']['rss_limit_bytes']==4*1024**3
    with pytest.raises(FileExistsError):dispatch.run_worker(writer,slot,ref,argv)
    assert len(seen)==1


@pytest.mark.parametrize('bad',['argv','parent','timeout','reserve','required','output','loglimit','capture',
    'pid','returncode','error','status','observation','logtruncated','terminal','partial','unexpected','missinglog',
    'newweight','changedsource','startuphash'])
def test_child_handoff_rejects_changed_or_incomplete_launch(writer,monkeypatch,bad):
    ref,argv,launch=handoff(writer,monkeypatch);root=writer.root/'evidence';stem='pilots/00/process/'
    changes={'argv':('argv',argv+['--extra']), 'parent':('parent_pid',os.getppid()+1),
        'timeout':('timeout_seconds',599),'reserve':('reserve_bytes',1),'required':('required_free_bytes',8*1024**3),
        'output':('output_limit_bytes',1),'loglimit':('log_limit_bytes',4096),'capture':('capture_kind','old'),
        'pid':('pid',123),'returncode':('returncode',0),'error':('error','failed'),'status':('status','completed'),
        'observation':('observation',{}),'logtruncated':('log_truncated',True)}
    if bad in changes:
        k,v=changes[bad];launch[k]=v;write(root,stem+'launch.json',launch)
    elif bad=='terminal':write(root,stem+'process.json',{})
    elif bad=='partial':write(root,stem+'launch.json.tmp',{})
    elif bad=='unexpected':write(root,stem+'extra.json',{})
    elif bad=='missinglog':(root/(stem+'worker.log')).unlink()
    elif bad=='newweight':write(root,'pilots/00/checkpoints/raw.pt',b'unreviewed')
    elif bad=='changedsource':(root/'meta/source.txt').write_text('changed')
    else:ref['sha256']='0'*64
    monkeypatch.setattr(execution,'execute_pilot_outputs',lambda *a,**k:pytest.fail('Body must not execute'))
    with pytest.raises(ValueError):dispatch.run_worker(writer,0,ref,argv)
    record=json.loads((writer.root/'control/pilot-entry-00.json').read_text())
    assert record['status']=='failed'


def controller(writer,monkeypatch):
    args=setup(writer,monkeypatch,0)
    monkeypatch.setattr(dispatch,'bind_campaign',startup.bind_campaign)
    calls=[]
    def process(w,stage,slot,argv,cwd,**kw):
        assert w is writer and stage=='pilots' and cwd==dispatch.ROOT
        assert kw['timeout_seconds']==600 and kw['log_limit_bytes']==4*1024**2
        assert kw['required_free_bytes']>30_000_000_000
        assert argv[:4]==[sys.executable,'-m',dispatch.MODULE,'worker']
        assert not (writer.root/f'evidence/pilots/{slot:02d}/process').exists()
        calls.append(slot)
        # Tiny complete saved producer records, no training or real measurements.
        existing={p:p.read_bytes() for p in (writer.root/'evidence').rglob('*') if p.is_file()
                  and not p.is_relative_to(writer.root/f'evidence/pilots/{slot:02d}')}
        fixture(writer.root/'evidence',count=slot+1)
        for p,content in existing.items():
            if p.read_bytes()!=content:p.write_bytes(content)
        path=writer.root/f'evidence/pilots/{slot:02d}/process/process.json'
        p=json.loads(path.read_text());p.update(pid=123456,parent_pid=os.getpid())
        path.write_text(json.dumps(p))
        writer.publish_json(f'control/pilot-entry-{slot:02d}.json',dict(kind=dispatch.KIND,status='finished',reason=None,
            pid=123456,elapsed_seconds=2.,observation=dict(peak_rss_bytes=1000,free_bytes=64*1024**3),
            limits=dict(rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3,timeout_seconds=600)))
        return p
    monkeypatch.setattr(dispatch,'run_process',process)
    return args,calls,process


def test_three_serial_controller_calls_reaudit_all_prefixes_without_cleanup(writer,monkeypatch):
    args,calls,_=controller(writer,monkeypatch)
    for slot in range(3):
        args['slot']=slot;ref=dispatch.run_pilot_once(**args)
        result=json.loads((writer.root/'evidence'/ref['path']).read_text())
        assert result['status']=='pilot_reviewed' and len(result['completed_pilots'])==slot+1
        assert not any(result[k] for k in ('adopted','deletion_authorized','stage_cleanup_complete','next_stage_authorized'))
        assert len(list((writer.root/'evidence/pilots').glob('*/checkpoints/*.pt')))==2*(slot+1)
    assert calls==[0,1,2]
    with pytest.raises(ValueError,match='cannot be rerun'):dispatch.run_pilot_once(**args)


@pytest.mark.parametrize('bad',['process','entry','result','worker','space','postbind'])
def test_controller_failure_preserves_evidence_and_blocks_progress(writer,monkeypatch,bad):
    args,calls,process=controller(writer,monkeypatch)
    def corrupt(*a,**kw):
        p=process(*a,**kw)
        root=writer.root/'evidence'
        if bad=='process':raise RuntimeError('synthetic process failure')
        if bad=='entry':(writer.root/'control/pilot-entry-00.json').write_text('{}')
        elif bad=='result':(root/'pilots/00/result.json').write_text('{}')
        elif bad=='worker':(root/'pilots/00/worker-outputs.json').unlink()
        elif bad=='space':monkeypatch.setattr(startup.shutil,'disk_usage',lambda p:SimpleNamespace(free=30_000_000_000))
        elif bad=='postbind':(root/'meta/source.txt').write_text('changed')
        return p
    monkeypatch.setattr(dispatch,'run_process',corrupt)
    with pytest.raises((ValueError,RuntimeError)):dispatch.run_pilot_once(**args)
    assert (writer.root/'evidence/meta/pilot-dispatch-00-failure.json').exists()
    assert len(list((writer.root/'evidence/pilots').glob('*/checkpoints/*.pt')))==2
    args['slot']=1
    with pytest.raises(ValueError,match='Failed dispatch'):dispatch.run_pilot_once(**args)
    assert calls==[0]


def test_missing_frozen_prerequisite_prevents_process(writer,monkeypatch):
    args,calls,_=controller(writer,monkeypatch)
    (writer.root/'evidence/meta/numerical.json').write_text('{}')
    with pytest.raises(ValueError):dispatch.run_pilot_once(**args)
    assert calls==[] and not (writer.root/'evidence/pilots').exists()


def test_worker_exception_preserves_guard_failure(writer,monkeypatch):
    ref,argv,_=handoff(writer,monkeypatch)
    def fail(*a,**kw):raise RuntimeError('synthetic body failure')
    monkeypatch.setattr(execution,'execute_pilot_outputs',fail)
    with pytest.raises(RuntimeError,match='synthetic body'):dispatch.run_worker(writer,0,ref,argv)
    assert json.loads((writer.root/'control/pilot-entry-00.json').read_text())['status']=='failed'


def test_guard_observes_rss_before_body_and_keeps_failure(writer,monkeypatch):
    monkeypatch.setattr(dispatch,'observation',lambda root:dict(peak_rss_bytes=4*1024**3,free_bytes=64*1024**3))
    with pytest.raises(RuntimeError,match='resource limit'):
        with dispatch._Guard(writer,0,'entry',600):pytest.fail('Guard must reject')
    report=json.loads((writer.root/'control/pilot-entry-00.json').read_text())
    assert report['reason']=='peak_rss_limit'


@pytest.mark.parametrize('extra',[False,True])
def test_child_hashing_releases_namespace_for_parent_monitor(writer,monkeypatch,extra):
    ref,argv,_=handoff(writer,monkeypatch,1);original=startup._check;seen=[]
    def check(root,name,binding,deadline):
        if name=='pilots/00/checkpoints/raw.pt':
            # This would time out if preflight retained its namespace lock.
            writer.publish_json('control/synthetic-parent-observation.json',{'alive':True})
            with writer._serialized():
                (root/'pilots/01/process/worker.log').write_bytes(b'allowed preflight warning')
            if extra:writer.publish_json('evidence/meta/unexpected-during-hash.json',{})
            seen.append(True)
        return original(root,name,binding,deadline)
    monkeypatch.setattr(startup,'_check',check)
    monkeypatch.setattr(execution,'execute_pilot_outputs',lambda *a,**kw:dict(synthetic=True))
    if extra:
        with pytest.raises(ValueError,match='inventory changed'):dispatch.run_worker(writer,1,ref,argv)
    else:assert dispatch.run_worker(writer,1,ref,argv)==dict(synthetic=True)
    assert seen==[True]


def test_sampled_entry_guard_hard_exits_child_and_keeps_bounded_record(writer):
    import subprocess
    from src.research import tokenizer_wide_hybrid_cleanup as cleanup
    code=f'''
from pathlib import Path
import time
from src.research import tokenizer_wide_hybrid_cleanup as cleanup
from src.research import tokenizer_wide_hybrid_pilot_dispatch_v2 as d
cleanup.GLOBAL_LOCK=Path({str(cleanup.GLOBAL_LOCK)!r})
w=d.MetadataWriter(Path({str(writer.root)!r}),{writer.lock_fds!r})
calls=0
def observed(root):
    global calls
    calls+=1
    return dict(peak_rss_bytes=1000 if calls==1 else 4*1024**3,free_bytes=64*1024**3)
d.observation=observed
with d._Guard(w,0,'entry',600):time.sleep(5)
'''
    child=subprocess.run([sys.executable,'-c',code],cwd=dispatch.ROOT,pass_fds=writer.lock_fds,
                         capture_output=True,timeout=12)
    assert child.returncode==90,child.stderr.decode()
    path=writer.root/'control/pilot-entry-00.json';record=json.loads(path.read_text())
    assert record['status']=='failed' and record['reason']=='peak_rss_limit' and path.stat().st_size<2048


@pytest.mark.parametrize('bad',['rss','elapsed','missing','free'])
def test_child_final_guard_requires_actual_bounded_observation(writer,monkeypatch,bad):
    args,calls,process=controller(writer,monkeypatch)
    def corrupt(*a,**kw):
        result=process(*a,**kw);path=writer.root/'control/pilot-entry-00.json'
        value=json.loads(path.read_text())
        if bad=='rss':value['observation']['peak_rss_bytes']=4*1024**3
        elif bad=='elapsed':value['elapsed_seconds']=600.
        elif bad=='missing':value['observation']={}
        else:value['observation']['free_bytes']=8*1024**3-1
        path.write_text(json.dumps(value));return result
    monkeypatch.setattr(dispatch,'run_process',corrupt)
    with pytest.raises(ValueError,match='observation invalid'):dispatch.run_pilot_once(**args)
    assert (writer.root/'evidence/meta/pilot-dispatch-00-failure.json').exists()


def test_source_change_during_postreview_cannot_publish_completion(writer,monkeypatch):
    args,calls,_=controller(writer,monkeypatch);original=dispatch.bind_storage
    def changed(*a,**kw):
        result=original(*a,**kw)
        (writer.root/'evidence/meta/source.txt').write_text('changed after storage audit')
        return result
    monkeypatch.setattr(dispatch,'bind_storage',changed)
    with pytest.raises(ValueError,match='Frozen file changed'):dispatch.run_pilot_once(**args)
    assert not (writer.root/'evidence/meta/pilot-dispatch-00.json').exists()
    assert (writer.root/'evidence/meta/pilot-dispatch-00-failure.json').exists()
