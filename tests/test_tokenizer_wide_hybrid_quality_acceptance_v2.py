"""Synthetic completed child evidence; no process, model or campaign launch."""
import json
import os
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_quality_acceptance_v2 as accept
from src.research.protocol import file_digest
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_quality_readiness_v2 import prepared
from test_tokenizer_wide_hybrid_quality_startup_v2 import setup as first_setup
import test_tokenizer_wide_hybrid_retention_audit as fixtures


def setup(writer,monkeypatch,slot=1):
    root=writer.root/'evidence'
    if slot==0:
        options=first_setup(writer,monkeypatch);original_campaign=accept.readiness.first.bind_campaign
        def campaign(*a):
            bound=original_campaign(*a)
            for choices in bound['requests']['quality'].values():
                for request in choices:
                    request['config']=dict(fixture_only=True,data=dict(path=str(root/'meta/corpus.txt')))
            return bound
        monkeypatch.setattr(accept.readiness.first,'bind_campaign',campaign)
        monkeypatch.setattr(accept.readiness,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=64*1024**3)))
        startup=accept.readiness.first.prepare_first_quality_request(**options)
    else:startup=prepared(writer,monkeypatch,slot)
    ready=accept.readiness.seal_quality_readiness(writer,slot,startup)
    return complete(writer,monkeypatch,slot,startup,ready)


def complete(writer,monkeypatch,slot,startup,ready):
    root=writer.root/'evidence'
    start=json.loads((root/startup['path']).read_text());seal=json.loads((root/ready['path']).read_text())
    bound,_,_=accept.readiness._bound(writer,slot,start)
    plan=json.loads((root/'meta/plan.json').read_text())
    existing={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    original=fixtures.write
    def write(root,name,value):
        if name in existing:
            # Prior completed slots already carry final v2 tensor reports; the
            # fixture constructs an intermediate report before adding producer
            # lineage. Preserve prior bytes instead of replaying those writes.
            if name==f'quality/{slot:02d}/request.json':assert value==json.loads(existing[name])
            return dict(path=name,sha256=file_digest(root/name))
        return original(root,name,value)
    with monkeypatch.context() as patch:
        patch.setattr(fixtures,'write',write)
        if slot<9:
            options=dict(prefix_count=slot+1)
            requests=[bound['requests']['quality'][str(i)][0] for i in range(slot+1)]
        else:
            from src.research.tokenizer_wide_hybrid_quality_rank_startup_v2 import _frozen
            _,requests=_frozen(writer,slot,start['plan_sha256'],start['projection_ref'],start['rates'])
            options=dict(boundary='rank',rank_prefix_count=slot-8,
                         raw_name={i:'final_001200_123456.pt' for i in range(9,15)})
        fixtures.fixture(root,producer_v2=True,**options,bindings=dict(context=bound['context'],
            projection_ref=start['projection_ref'],numerical_ref=plan['numerical_ref'],
            training_token_sha256=plan['training_token_sha256'],requests=requests))
    argv=accept.entry.worker_argv(writer,slot,startup,ready)
    launch=dict(status='preflight',argv=argv,parent_pid=os.getpid(),pid=None,returncode=None,error=None,
        required_free_bytes=seal['required_free_bytes'],reserve_bytes=8*1024**3,output_limit_bytes=seal['output_limit_bytes'],
        log_limit_bytes=4*1024**2,timeout_seconds=7200,observation=None,capture_kind='bounded_pipe_v2',log_truncated=False)
    writer.publish_json(f'evidence/quality/{slot:02d}/process/launch.json',launch)
    terminal=dict(launch,status='completed',pid=os.getpid()+100000,returncode=0,elapsed_seconds=2.,
        observation=dict(free_bytes=64*1024**3,output_bytes=0,log_bytes=13))
    path=root/f'quality/{slot:02d}/process/process.json'
    # Synthetic counterpart to the real producer's self-inclusive final census.
    for _ in range(10):
        path.write_text(json.dumps(terminal))
        total=sum(p.stat().st_size for p in (root/f'quality/{slot:02d}').rglob('*') if p.is_file())
        if total==terminal['observation']['output_bytes']:break
        terminal['observation']['output_bytes']=total
    else:pytest.fail('Synthetic terminal size did not converge')
    writer.publish_json(f'control/quality-entry-{slot:02d}.json',dict(kind=accept.entry.KIND,status='finished',reason=None,
        pid=terminal['pid'],elapsed_seconds=2.,observation=dict(peak_rss_bytes=1000,free_bytes=64*1024**3),
        limits=dict(timeout_seconds=7200,rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3)))
    monkeypatch.setattr(accept,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=64*1024**3)))
    return dict(writer=writer,slot=slot,startup_ref=startup,readiness_ref=ready,process=terminal)


@pytest.mark.parametrize('slot',[0,1,4,8])
def test_complete_child_reaudits_prefix_history_and_storage_readonly(writer,monkeypatch,slot):
    args=setup(writer,monkeypatch,slot);before=accept._inventory(writer.root/'evidence')
    checked=accept.review_completed_lr(**args)
    assert checked['status']=='lr_child_reviewed_pending_parent_receipt'
    assert len(checked['completed_records'])==slot+1 and len(checked['checkpoint_pairs'])==slot+1
    assert checked['storage']['storage_projection_fits'] and checked['selection'] is None
    assert not any(checked[k] for k in ('adopted','dispatch_authorized','deletion_authorized','stage_cleanup_complete','next_stage_authorized'))
    assert accept._inventory(writer.root/'evidence')==before


@pytest.mark.parametrize('bad',['return','launch','guard','startup','seal','numeric','source','prior_weight',
    'result','worker','tensor','resource','journal','extra','later','stop','locks','space'])
def test_success_exit_is_not_enough_and_all_inputs_are_preserved(writer,monkeypatch,bad):
    args=setup(writer,monkeypatch);root=writer.root/'evidence'
    paths={'launch':'quality/01/process/launch.json','startup':args['startup_ref']['path'],
        'seal':args['readiness_ref']['path'],'numeric':'meta/numerical.json','source':'meta/source.txt',
        'prior_weight':'quality/00/checkpoints/raw.pt','result':'quality/01/result.json',
        'worker':'quality/01/worker-outputs.json','tensor':'quality/01/tensor-audit.json',
        'resource':'quality/01/resource-audit.json','journal':'.retention/pilot/deletions.jsonl'}
    if bad in paths:(root/paths[bad]).write_text('{}')
    elif bad=='return':args['process']['returncode']=1
    elif bad=='guard':(writer.root/'control/quality-entry-01.json').write_text('{}')
    elif bad=='stop':(writer.root/'STOP').touch()
    elif bad=='locks':monkeypatch.setattr(writer,'lock_fds',(writer.lock_fds[0],writer.lock_fds[0]))
    elif bad=='space':monkeypatch.setattr(accept,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=8*1024**3)))
    else:writer.publish_json('evidence/'+('quality/01/unreviewed.json' if bad=='extra' else 'quality/02/request.json'),{})
    before=accept._inventory(root)
    with pytest.raises(ValueError):accept.review_completed_lr(**args)
    assert accept._inventory(root)==before


@pytest.mark.parametrize('key,value',[('pid',1),('parent_pid',1),('returncode',True),('status','running'),
    ('timeout_seconds',600),('required_free_bytes',8*1024**3),('capture_kind','file'),('log_truncated',True),
    ('elapsed_seconds',7200.),('observation',dict(free_bytes=64*1024**3,output_bytes=0,log_bytes=0))])
def test_rewritten_terminal_and_return_value_still_require_exact_contract(writer,monkeypatch,key,value):
    args=setup(writer,monkeypatch);args['process'][key]=value
    (writer.root/'evidence/quality/01/process/process.json').write_text(json.dumps(args['process']))
    with pytest.raises(ValueError):accept.review_completed_lr(**args)


def test_full_prefix_and_storage_are_independently_replayed_without_namespace_lock(writer,monkeypatch):
    args=setup(writer,monkeypatch);calls=[];prefix=accept.bind_lr_prefix;storage=accept.bind_storage
    def audit(*a,**kw):
        writer.publish_json('control/parent-postprocess-observation.json',{'audit':True})
        assert len(kw['requests'])==2;calls.append('prefix');return prefix(*a,**kw)
    def capacity(*a,**kw):
        assert len(kw['pilots'])==3 and len(kw['records'])==2 and len(kw['stages'])==1
        assert len(kw['checkpoint_pairs'])==5 and kw['rates'] is None and kw['selection'] is None
        calls.append('storage');return storage(*a,**kw)
    monkeypatch.setattr(accept,'bind_lr_prefix',audit);monkeypatch.setattr(accept,'bind_storage',capacity)
    accept.review_completed_lr(**args);assert calls==['prefix','storage']
