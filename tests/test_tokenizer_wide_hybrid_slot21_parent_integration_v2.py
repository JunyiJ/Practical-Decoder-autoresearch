"""Sequential synthetic parent checks share one prefix; no model/training runs.

Production replay remains uncached. Mutations below are confined to disposable
fixtures and always restored before the next independent read-only audit.
"""
import copy
import json
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_remaining_confirmation_dispatch_v2 as controller
from src.research.protocol import file_digest
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_slot21_readiness_entry_v2 import prepared
from test_tokenizer_wide_hybrid_quality_acceptance_v2 import complete


def setup(writer,monkeypatch):
    startup=prepared(writer,monkeypatch)
    ready=controller.readiness.seal_quality_readiness(writer,21,startup);calls=[]
    free=SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=64*1024**3))
    for module in (controller,controller.dispatch,controller.acceptance):monkeypatch.setattr(module,'shutil',free)
    def process(w,stage,slot,argv,cwd,**limits):
        assert w is writer and stage=='quality' and slot==21 and cwd==controller.ROOT
        assert limits['timeout_seconds']==7200 and limits['log_limit_bytes']==4*1024**2
        guard=json.loads((writer.root/'control/quality-controller-prepare-21.json').read_text())
        assert guard['status']=='finished'
        seal=controller._document(writer.root/'evidence',ready)
        assert limits['required_free_bytes']==seal['required_free_bytes']
        assert limits['output_limit_bytes']==seal['output_limit_bytes']
        assert argv==controller.entry.worker_argv(writer,21,startup,ready)
        limits['parent_check']();calls.append(21)
        return complete(writer,monkeypatch,21,startup,ready)['process']
    monkeypatch.setattr(controller,'run_process',process)
    return dict(writer=writer,slot=21,startup_ref=startup,readiness_ref=ready),calls


def test_sequential_full_audit_and_single_dispatch(writer,monkeypatch):
    args,calls=setup(writer,monkeypatch);root=writer.root/'evidence'
    accept=controller.acceptance;before=accept._inventory(root);cases=[]
    original_verify=controller.verify_sealed_confirmation
    def verify(*a,**kw):
        guard_ref=a[-1];path=writer.root/guard_ref['path'];raw=path.read_bytes();guard=json.loads(raw)
        for key,value in [('kind','wrong'),('pid',True),('status','failed'),('limits',{}),('elapsed_seconds',1800)]:
            changed=copy.deepcopy(guard);changed[key]=value;path.write_text(json.dumps(changed))
            try:
                with pytest.raises(ValueError):controller._prepare_control(writer,dict(path=guard_ref['path'],sha256=file_digest(path)),21)
                cases.append('prepare_guard:'+key)
            finally:path.write_bytes(raw)
        with pytest.raises(ValueError):controller.startup.verify_confirmation_request(writer,21,args['startup_ref'])
        from src.research.tokenizer_wide_hybrid_inventory_delta import encode_inventory
        ready=args['readiness_ref'];path=root/ready['path'];raw=path.read_bytes()
        seal=controller._document(root,ready);startup=controller._document(root,args['startup_ref'])
        prior=controller._document(root,startup['prior_dispatch_ref'])
        path.write_text(json.dumps(encode_inventory(root,seal,prior['review']['readiness'])))
        try:
            with pytest.raises(ValueError,match='wire anchor'):
                controller._stable_anchor(writer,dict(path=ready['path'],sha256=file_digest(path)),startup)
            cases.append('stable_original16_wire_anchor')
        finally:path.write_bytes(raw)
        return original_verify(*a,**kw)
    monkeypatch.setattr(controller,'verify_sealed_confirmation',verify)
    original_review=accept.review_completed_remaining_confirmation
    def review(*a,**kw):
        # One completed synthetic child, audited repeatedly without replaying its
        # expensive setup or granting credit for any parent receipt.
        names=['meta/source.txt','meta/numerical.json','.retention/pilot/receipt.json',
               '.retention/lr/deletions.jsonl','.retention/rank/execution-review.json',
               'quality/09/checkpoints/final_001200_123456.pt']
        names += [f'meta/quality-dispatch-{i}.json' for i in range(15,21)]
        names += [f'quality/{i}/{tail}' for i in range(15,22) for tail in
                  ('checkpoints/h200.pt','checkpoints/final_001200_123456.pt','result.json','tensor-audit.json','process/launch.json')]
        paths=[root/n for n in names]+[writer.root/f'control/{stem}-{i}.json'
                  for i in range(15,22) for stem in ('process-quality','quality-entry')]
        for path in paths:
            raw=path.read_bytes();path.write_bytes(b'{}');inventory=accept._inventory(root)
            try:
                with pytest.raises((ValueError,KeyError)):original_review(*a,**kw)
                assert accept._inventory(root)==inventory
                cases.append('tamper:'+str(path.relative_to(writer.root)))
            finally:path.write_bytes(raw)
        terminal=root/'quality/21/process/process.json';raw=terminal.read_bytes()
        for key,value in [('returncode',True),('parent_pid',1),('pid',1),('log_truncated',True),
                          ('timeout_seconds',600),('elapsed_seconds',7200.)]:
            changed=copy.deepcopy(a[-1]);changed[key]=value;terminal.write_text(json.dumps(changed))
            try:
                with pytest.raises(ValueError):original_review(*a[:-1],changed,**kw)
                cases.append('terminal:'+key)
            finally:terminal.write_bytes(raw)
        storage=accept.bind_storage
        for bad in ('source','old_control','current_control','prior_receipt','capacity','stop'):
            path={'source':root/'meta/source.txt','old_control':writer.root/'control/process-quality-15.json',
                  'current_control':writer.root/'control/quality-entry-21.json',
                  'prior_receipt':root/'meta/quality-dispatch-20.json'}.get(bad)
            raw=path.read_bytes() if path else None;seen=[]
            with monkeypatch.context() as patch:
                def late(*sa,**sk):
                    result=storage(*sa,**sk);seen.append(True)
                    if path:path.write_bytes(b'late mutation')
                    elif bad=='stop':(writer.root/'STOP').touch()
                    else:patch.setattr(accept,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=8*1024**3)))
                    return result
                patch.setattr(accept,'bind_storage',late)
                try:
                    with pytest.raises((ValueError,KeyError)):original_review(*a,**kw)
                    assert seen==[True];cases.append('late:'+bad)
                finally:
                    if path:path.write_bytes(raw)
                    if bad=='stop':(writer.root/'STOP').unlink()
        report=original_review(*a,**kw)
        assert len(report['completed_records'])==22 and len(report['checkpoint_pairs'])==10
        assert len(report['stages'])==3 and report['storage']['cleanup_boundaries_verified']==3
        assert report['audited_indices']==[9,10,11,*range(15,22)]
        assert report['prior_transport_verified'] and report['prior_dispatch_ref']['path']=='meta/quality-dispatch-20.json'
        return report
    monkeypatch.setattr(accept,'review_completed_remaining_confirmation',review)
    ref=controller.run_sealed_confirmation_once(**args);saved=controller._document(root,ref)
    assert calls==[21] and saved['status']=='confirmation_reviewed'
    assert [v['path'] for v in saved['prior_dispatches']]==[f'meta/quality-dispatch-{i}.json' for i in range(15,21)]
    report=saved['review']
    assert report['representation']==controller.REPRESENTATION
    assert not set(report)&{'completed_records','checkpoint_pairs','stages'}
    assert all(report[k] is False for k in ('adopted','dispatch_authorized','deletion_authorized','stage_cleanup_complete','next_stage_authorized'))
    assert len(list(root.rglob('*.pt')))==20 and not (root/'quality/22').exists()
    after=accept._inventory(root);assert all(after[n]==v for n,v in before.items())
    with pytest.raises(ValueError,match='cannot be retried'):controller.run_sealed_confirmation_once(**args)
    assert accept._inventory(root)==after and calls==[21]
    census=controller.snapshot(writer.root)
    (writer.root.parent/'slot21-parent-census.json').write_text(json.dumps(dict(
        receipt_bytes=(root/ref['path']).stat().st_size,metadata_bytes=census['metadata_bytes'],
        meta_group_bytes=census['groups']['evidence/meta'],live_checkpoint_files=20,
        completed_quality_records=22,negative_cases=cases,real_panel_fit_proven=False)))


def test_partial_publication_preserves_failure_and_prevents_second_child(writer,monkeypatch):
    args,calls=setup(writer,monkeypatch);root=writer.root/'evidence';publish=writer.publish_json
    def partial(name,*a,**kw):
        if name=='evidence/meta/quality-dispatch-21.json':
            (writer.root/(name+'.tmp')).write_bytes(b'partial');raise OSError('synthetic partial receipt')
        return publish(name,*a,**kw)
    monkeypatch.setattr(writer,'publish_json',partial)
    with pytest.raises(OSError,match='partial receipt'):controller.run_sealed_confirmation_once(**args)
    failure=json.loads((root/'meta/quality-dispatch-21-failure.json').read_text())
    assert failure['retry_allowed'] is False and calls==[21]
    assert not (root/'meta/quality-dispatch-21.json').exists()
    assert (root/'meta/quality-dispatch-21.json.tmp').read_bytes()==b'partial'
    assert len(list(root.rglob('*.pt')))==20 and not (root/'quality/22').exists()
    before=controller.acceptance._inventory(root)
    with pytest.raises(ValueError):controller.run_sealed_confirmation_once(**args)
    assert calls==[21] and controller.acceptance._inventory(root)==before


@pytest.mark.parametrize('slot',[0,9,15,16,17,24,-1,True,21.,'21'])
def test_only_integrated_parent_slots_are_enabled(writer,slot):
    with pytest.raises(ValueError,match='slot21'):controller.run_sealed_confirmation_once(writer,slot,{},{})
    with pytest.raises(ValueError,match='slot21'):controller.acceptance.review_completed_remaining_confirmation(writer,slot,{},{},{})
    assert not (writer.root/'evidence').exists()
