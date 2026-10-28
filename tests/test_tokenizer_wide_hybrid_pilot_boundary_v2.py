"""Full boundary wiring on disposable tiny artifacts; no research deletion."""
import json
import os
from pathlib import Path

import pytest

from src.research import tokenizer_wide_hybrid_pilot_boundary_v2 as boundary
from src.research import tokenizer_wide_hybrid_pilot_dispatch_v2 as dispatch
from src.research import tokenizer_wide_hybrid_pilot_startup_v2 as startup
from src.research import tokenizer_wide_hybrid_cleanup as cleanup
from src.research.protocol import file_digest
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_pilot_dispatch_v2 import controller


def prepared(writer,monkeypatch,count=3):
    args,calls,process=controller(writer,monkeypatch)
    monkeypatch.setattr(boundary,'bind_campaign',startup.bind_campaign)
    monkeypatch.setattr(boundary,'PROJECTION',startup.PROJECTION)
    def recorded(w,stage,slot,argv,cwd,**kw):
        p=process(w,stage,slot,argv,cwd,**kw);root=w.root/f'evidence/pilots/{slot:02d}/process'
        p.update(argv=argv,**{k:kw[k] for k in ('required_free_bytes','output_limit_bytes','log_limit_bytes','timeout_seconds')})
        (root/'process.json').write_text(json.dumps(p))
        w.publish_json(f'evidence/pilots/{slot:02d}/process/launch.json',dict(status='preflight',argv=argv,
            parent_pid=os.getpid(),pid=None,returncode=None,error=None,observation=None,
            required_free_bytes=kw['required_free_bytes'],reserve_bytes=8*1024**3,
            output_limit_bytes=kw['output_limit_bytes'],log_limit_bytes=kw['log_limit_bytes'],
            timeout_seconds=kw['timeout_seconds'],capture_kind='bounded_pipe_v2',log_truncated=False))
        return p
    monkeypatch.setattr(dispatch,'run_process',recorded)
    for i in range(count):args['slot']=i;dispatch.run_pilot_once(**args)
    return dict(writer=writer,plan_sha256=args['plan_sha256'],projection_ref=args['projection_ref'])


def failure(writer):return writer.root/'evidence/meta/pilot-boundary-failure.json'


def test_complete_audit_cleanup_and_post_deletion_reverification(writer,monkeypatch):
    args=prepared(writer,monkeypatch);root=writer.root/'evidence'
    before={str(p.relative_to(root)):file_digest(p) for p in root.rglob('*') if p.is_file()}
    ref=boundary.cleanup_pilot_boundary(**args)
    result=json.loads((root/ref['path']).read_text())
    deleted=result['chain']['deleted_files'];assert len(deleted)==6
    assert all(n.endswith('.pt') and n.startswith('pilots/') for n in deleted)
    assert not list(root.rglob('*.pt'))
    assert all(file_digest(root/n)==sha for n,sha in before.items() if n not in deleted)
    assert result['storage']['projection']['phase']=='lr' and not result['next_stage_authorized']
    checked=boundary.verify_pilot_boundary(writer,args['plan_sha256'],ref)
    assert checked['pilot_boundary_verified'] and checked['chain']['boundaries']==1
    assert not checked['dispatch_authorized'] and len(checked['checkpoint_pairs'])==3
    with pytest.raises(ValueError,match='cannot be retried'):boundary.cleanup_pilot_boundary(**args)


@pytest.mark.parametrize('bad',['partial','dispatch','guard','launch','worker','weight','source','projection','unknown_weight','stop'])
def test_incomplete_or_changed_pilots_never_reach_unlink(writer,monkeypatch,bad):
    args=prepared(writer,monkeypatch,count=2 if bad=='partial' else 3);root=writer.root/'evidence'
    if bad=='dispatch':(root/'meta/pilot-dispatch-01.json').unlink()
    elif bad=='guard':(writer.root/'control/pilot-entry-01.json').write_text('{}')
    elif bad=='launch':(root/'pilots/01/process/launch.json').write_text('{}')
    elif bad=='worker':(root/'pilots/01/worker-outputs.json').unlink()
    elif bad=='weight':(root/'pilots/01/checkpoints/raw.pt').write_bytes(b'changed')
    elif bad=='source':(root/'meta/source.txt').write_text('changed')
    elif bad=='projection':args['projection_ref']['sha256']='0'*64
    elif bad=='unknown_weight':(root/'pilots/00/checkpoints/extra.pt').write_bytes(b'protected unreviewed')
    elif bad=='stop':(writer.root/'STOP').write_text('stop')
    weights={p:p.read_bytes() for p in root.rglob('*.pt')}
    with pytest.raises((ValueError,FileNotFoundError,KeyError)):boundary.cleanup_pilot_boundary(**args)
    assert weights=={p:p.read_bytes() for p in root.rglob('*.pt')}
    assert failure(writer).exists()
    with pytest.raises(ValueError,match='cannot be retried'):boundary.cleanup_pilot_boundary(**args)


def test_stop_after_one_deletion_keeps_journal_and_forbids_continuation(writer,monkeypatch):
    args=prepared(writer,monkeypatch);original=Path.unlink;removed=[]
    def stop(path,*a,**kw):
        result=original(path,*a,**kw)
        if path.suffix=='.pt':
            removed.append(str(path));(writer.root/'STOP').write_text('synthetic interruption')
        return result
    monkeypatch.setattr(Path,'unlink',stop)
    with pytest.raises(ValueError):boundary.cleanup_pilot_boundary(**args)
    assert len(removed)==1 and failure(writer).exists()
    owned=writer.root/'evidence/.retention/pilot'
    events=[json.loads(line) for line in (owned/'deletions.jsonl').read_text().splitlines()]
    assert [e['event'] for e in events]==['intent','deleted']
    assert not (owned/'execution-review.json').exists()
    assert len(list((writer.root/'evidence').rglob('*.pt')))==5


@pytest.mark.parametrize('bad',['journal','guard','failure','completion','review','protected','reappeared','context','extra','boundary_guard'])
def test_post_cleanup_acceptance_requires_complete_unchanged_chain(writer,monkeypatch,bad):
    args=prepared(writer,monkeypatch);ref=boundary.cleanup_pilot_boundary(**args);root=writer.root/'evidence'
    owned=root/'.retention/pilot'
    if bad=='journal':(owned/'deletions.jsonl').write_text('')
    elif bad=='guard':(owned/'guard.json').write_text('{}')
    elif bad=='failure':(owned/'failure.json').write_text('{}')
    elif bad=='completion':ref['sha256']='0'*64
    elif bad=='review':(root/'meta/pilot-boundary-review.json').write_text('{}')
    elif bad=='protected':(root/'pilots/00/result.json').write_text('{}')
    elif bad=='reappeared':(root/'pilots/00/checkpoints/raw.pt').write_bytes(b'recreated')
    elif bad=='context':(root/'meta/source.txt').write_text('changed')
    elif bad=='boundary_guard':(writer.root/'control/pilot-boundary-00.json').write_text('{}')
    else:(root/'meta/unreviewed.json').write_text('{}')
    with pytest.raises((ValueError,FileNotFoundError)):
        boundary.verify_pilot_boundary(writer,args['plan_sha256'],ref)


def test_callback_revalidates_noncheckpoint_evidence_before_next_unlink(writer,monkeypatch):
    args=prepared(writer,monkeypatch);original=Path.unlink;removed=[]
    def changed(path,*a,**kw):
        result=original(path,*a,**kw)
        if path.suffix=='.pt':
            removed.append(str(path));(writer.root/'evidence/pilots/00/metrics.jsonl').write_text('changed')
        return result
    monkeypatch.setattr(Path,'unlink',changed)
    with pytest.raises(ValueError):boundary.cleanup_pilot_boundary(**args)
    assert len(removed)==1 and failure(writer).exists()


def test_post_cleanup_storage_failure_retains_receipts_but_no_acceptance(writer,monkeypatch):
    args=prepared(writer,monkeypatch);original=boundary.bind_storage
    def gate(*a,**kw):
        if kw['stages']:raise ValueError('synthetic future storage shortfall')
        return original(*a,**kw)
    monkeypatch.setattr(boundary,'bind_storage',gate)
    with pytest.raises(ValueError,match='storage shortfall'):boundary.cleanup_pilot_boundary(**args)
    root=writer.root/'evidence'
    assert (root/'.retention/pilot/execution-review.json').exists() and failure(writer).exists()
    assert not (root/'meta/pilot-boundary-complete.json').exists()
    assert not list(root.rglob('*.pt'))


@pytest.mark.parametrize('bad',['dispatch_prefix','dispatch_claim','startup_claim','launch_limit','entry_observation'])
def test_dispatch_semantics_cannot_be_replaced_by_saved_success_flag(writer,monkeypatch,bad):
    args=prepared(writer,monkeypatch);root=writer.root/'evidence'
    name='meta/pilot-dispatch-01.json';value=json.loads((root/name).read_text())
    if bad=='dispatch_prefix':value['completed_pilots'].pop()
    elif bad=='dispatch_claim':value['next_stage_authorized']=True
    elif bad=='startup_claim':
        ref=value['startup'];document=json.loads((root/ref['path']).read_text())
        document['deletion_authorized']=True;(root/ref['path']).write_text(json.dumps(document))
        ref['sha256']=file_digest(root/ref['path'])
    elif bad=='launch_limit':
        path=root/'pilots/01/process/launch.json';launch=json.loads(path.read_text())
        launch['timeout_seconds']=7200;path.write_text(json.dumps(launch))
    else:
        path=writer.root/'control/pilot-entry-01.json';g=json.loads(path.read_text())
        g['observation']['peak_rss_bytes']=4*1024**3;path.write_text(json.dumps(g))
        value['entry_guard']=dict(path=str(path),sha256=file_digest(path),record=g)
    (root/name).write_text(json.dumps(value))
    before={p:p.read_bytes() for p in root.rglob('*.pt')}
    with pytest.raises(ValueError):boundary.cleanup_pilot_boundary(**args)
    assert before=={p:p.read_bytes() for p in root.rglob('*.pt')}
