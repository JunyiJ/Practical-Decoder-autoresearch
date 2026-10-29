"""Read-only fourth-fresh acceptance using disposable evidence, never training."""
import copy
import json
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_quality_acceptance_v2 as accept
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_slot18_readiness_entry_v2 import prepared
from test_tokenizer_wide_hybrid_quality_acceptance_v2 import complete


def setup(writer,monkeypatch):
    startup=prepared(writer,monkeypatch)
    ready=accept.readiness.seal_quality_readiness(writer,18,startup)
    return complete(writer,monkeypatch,18,startup,ready)


def test_fourth_fresh_parent_reaudits_transport_all_four_pairs_and_complete_storage(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence';before=accept._inventory(root)
    original=accept.bind_storage;calls=[]
    def storage(*a,**kw):
        assert len(kw['records'])==19 and len(kw['pilots'])==3 and len(kw['stages'])==3
        assert len(kw['checkpoint_pairs'])==22
        assert kw['records'][-2]['stage']==kw['records'][-1]['stage']=='confirm'
        calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(accept,'bind_storage',storage)
    report=accept.review_completed_remaining_confirmation(**args)
    assert calls==[1] and report['kind']==accept.LATER_CONFIRM_KIND and report['prior_transport_verified']
    assert report['status']=='confirmation_child_reviewed_pending_parent_receipt'
    assert report['audited_indices']==[9,10,11,15,16,17,18] and len(report['checkpoint_pairs'])==7
    assert report['storage']['cleanup_boundaries_verified']==3
    assert report['prior_dispatch_ref']['path']=='meta/quality-dispatch-17.json'
    assert all(report[k] is False for k in ('adopted','dispatch_authorized','deletion_authorized','stage_cleanup_complete','next_stage_authorized'))
    assert accept._inventory(root)==before and len(list(root.rglob('*.pt')))==14
    assert not (root/'quality/19').exists() and not (root/'meta/quality-dispatch-18.json').exists()
    census=accept.snapshot(writer.root)
    (writer.root.parent/'slot18-acceptance-census.json').write_text(json.dumps(dict(
        completed_quality_records=19,live_checkpoint_files=14,cleanup_boundaries=3,
        metadata_bytes=census['metadata_bytes'],meta_group_bytes=census['groups']['evidence/meta'],
        required_free_bytes=report['storage']['required_free_bytes'],real_panel_fit_proven=False)))


def test_current_and_prior_science_transport_and_rehashed_terminal_fail_closed(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence'
    names=['meta/source.txt','meta/numerical.json','meta/quality-dispatch-16.json','meta/quality-dispatch-17.json',
        '.retention/pilot/receipt.json','.retention/lr/deletions.jsonl','.retention/rank/execution-review.json',
        'quality/09/checkpoints/final_001200_123456.pt','quality/15/checkpoints/h200.pt','quality/16/checkpoints/h200.pt','quality/17/checkpoints/h200.pt','quality/18/checkpoints/h200.pt']
    names += [f'quality/{i}/{tail}' for i in ('15','16','17','18') for tail in
              ('result.json','tensor-audit.json','worker-outputs.json','resource-audit.json','process/launch.json')]
    paths=[root/n for n in names]+[writer.root/n for n in
        ('control/quality-readiness-18.json','control/quality-entry-15.json','control/quality-entry-18.json',
         'control/process-quality-15.json','control/process-quality-16.json','control/quality-entry-16.json','control/process-quality-17.json','control/quality-entry-17.json','control/process-quality-18.json')]
    for path in paths:
        raw=path.read_bytes();path.write_bytes(b'{}');before=accept._inventory(root)
        try:
            with pytest.raises((ValueError,KeyError)):accept.review_completed_remaining_confirmation(**args)
            assert accept._inventory(root)==before
        finally:path.write_bytes(raw)
    terminal=root/'quality/18/process/process.json';raw=terminal.read_bytes()
    for key,value in [('returncode',True),('parent_pid',1),('pid',1),('log_truncated',True),('timeout_seconds',600),
        ('elapsed_seconds',7200.),('observation',dict(free_bytes=64*1024**3,output_bytes=0,log_bytes=0))]:
        changed=copy.deepcopy(args['process']);changed[key]=value;terminal.write_text(json.dumps(changed))
        try:
            with pytest.raises(ValueError):accept.review_completed_remaining_confirmation(**dict(args,process=changed))
        finally:terminal.write_bytes(raw)
    for name in ('quality/18/unreviewed.json','quality/19/request.json'):
        path=root/name;made=not path.parent.exists();path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'{}')
        try:
            with pytest.raises(ValueError):accept.review_completed_remaining_confirmation(**args)
        finally:
            path.unlink()
            if made:path.parent.rmdir()
    with monkeypatch.context() as patch:
        patch.setattr(accept,'os',SimpleNamespace(getpid=lambda:1))
        with pytest.raises(ValueError):accept.review_completed_remaining_confirmation(**args)
    # The earlier public API never acquires permission to accept slot18.
    with pytest.raises(ValueError):accept.review_completed_first_confirmation(**args)
    with pytest.raises(ValueError):accept.review_completed_confirmation(**args)
    with pytest.raises(ValueError):accept.review_completed_later_confirmation(**args)


def test_late_history_controls_stop_and_capacity_changes_reject_readonly(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence';original=accept.bind_storage
    for bad in ('source','prior_supervisor','current_supervisor','prior_guard','current_guard','prior_receipt','second_supervisor','second_guard','first_receipt','third_supervisor','third_guard','third_receipt','space','stop'):
        paths=dict(source=root/'meta/source.txt',prior_supervisor=writer.root/'control/process-quality-15.json',
            current_supervisor=writer.root/'control/process-quality-18.json',prior_guard=writer.root/'control/quality-entry-15.json',
            current_guard=writer.root/'control/quality-entry-18.json',prior_receipt=root/'meta/quality-dispatch-16.json',second_supervisor=writer.root/'control/process-quality-16.json',
            second_guard=writer.root/'control/quality-entry-16.json',first_receipt=root/'meta/quality-dispatch-15.json',third_supervisor=writer.root/'control/process-quality-17.json',third_guard=writer.root/'control/quality-entry-17.json',third_receipt=root/'meta/quality-dispatch-17.json')
        path=paths.get(bad);raw=path.read_bytes() if path else None;seen=[]
        with monkeypatch.context() as patch:
            def late(*a,**kw):
                value=original(*a,**kw);seen.append(1)
                if path:path.write_bytes(b'late mutation')
                elif bad=='stop':(writer.root/'STOP').touch()
                else:patch.setattr(accept,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=8*1024**3)))
                return value
            patch.setattr(accept,'bind_storage',late)
            try:
                with pytest.raises((ValueError,KeyError)):accept.review_completed_remaining_confirmation(**args)
            finally:
                if path:path.write_bytes(raw)
                if bad=='stop':(writer.root/'STOP').unlink()
        assert seen==[1]
    assert len(list(root.rglob('*.pt')))==14 and not (root/'quality/19').exists()


@pytest.mark.parametrize('slot',[0,9,15,16,17,21,22,23,24,-1,True,18.,'18'])
def test_only_integrated_remaining_parent_slots_are_enabled(writer,slot):
    with pytest.raises(ValueError,match='slot18'):accept.review_completed_remaining_confirmation(writer,slot,{},{},{})
    assert not (writer.root/'evidence').exists()
