"""One sealed slot17 transition using a synthetic child, no model or training."""
import copy
import json
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_later_confirmation_dispatch_v2 as controller
from src.research.protocol import file_digest
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_slot17_readiness_entry_v2 import prepared
from test_tokenizer_wide_hybrid_quality_acceptance_v2 import complete


def setup(writer,monkeypatch):
    startup=prepared(writer,monkeypatch)
    ready=controller.readiness.seal_quality_readiness(writer,17,startup);calls=[]
    free=SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=64*1024**3))
    for module in (controller,controller.dispatch,controller.acceptance):monkeypatch.setattr(module,'shutil',free)
    def process(w,stage,slot,argv,cwd,**limits):
        assert w is writer and stage=='quality' and slot==17 and cwd==controller.ROOT
        assert limits['timeout_seconds']==7200 and limits['log_limit_bytes']==4*1024**2
        guard=json.loads((writer.root/'control/quality-controller-prepare-17.json').read_text())
        assert guard['status']=='finished'
        seal=controller._document(writer.root/'evidence',ready)
        assert limits['required_free_bytes']==seal['required_free_bytes']
        assert limits['output_limit_bytes']==seal['output_limit_bytes']
        assert argv==controller.entry.worker_argv(writer,17,startup,ready)
        limits['parent_check']();calls.append(17)
        return complete(writer,monkeypatch,17,startup,ready)['process']
    monkeypatch.setattr(controller,'run_process',process)
    return dict(writer=writer,slot=17,startup_ref=startup,readiness_ref=ready),calls


def test_sealed_serial_transition_replays_history_and_stops_before_slot18(writer,monkeypatch):
    args,calls=setup(writer,monkeypatch);root=writer.root/'evidence';before=controller.acceptance._inventory(root)
    original=controller.verify_sealed_confirmation;checked=[]
    def verify(*a,**kw):
        guard_ref=a[-1];path=writer.root/guard_ref['path'];raw=path.read_bytes();guard=json.loads(raw)
        for key,value in [('kind','wrong'),('pid',True),('status','failed'),('limits',{}),('elapsed_seconds',1800)]:
            changed=copy.deepcopy(guard);changed[key]=value;path.write_text(json.dumps(changed))
            try:
                with pytest.raises(ValueError):controller._prepare_control(writer,dict(path=guard_ref['path'],sha256=file_digest(path)))
            finally:path.write_bytes(raw)
        with pytest.raises(ValueError):controller.startup.verify_confirmation_request(writer,17,args['startup_ref'])
        seal=original(*a,**kw);checked.append(1);return seal
    monkeypatch.setattr(controller,'verify_sealed_confirmation',verify)
    ref=controller.run_sealed_confirmation_once(**args);saved=controller._document(root,ref)
    assert calls==[17] and checked==[1] and saved['status']=='confirmation_reviewed'
    assert [v['path'] for v in saved['prior_dispatches']]==['meta/quality-dispatch-15.json','meta/quality-dispatch-16.json']
    report=saved['review']
    assert report['representation']==controller.REPRESENTATION and report['prior_transport_verified']
    assert report['audited_indices']==[9,10,11,15,16,17]
    assert not set(report)&{'completed_records','checkpoint_pairs','stages'}
    assert all(report[k] is False for k in ('adopted','dispatch_authorized','deletion_authorized','stage_cleanup_complete','next_stage_authorized'))
    assert len(list(root.rglob('*.pt')))==12 and not (root/'quality/18').exists()
    after=controller.acceptance._inventory(root);assert all(after[n]==v for n,v in before.items())
    with pytest.raises(ValueError,match='cannot be retried'):controller.run_sealed_confirmation_once(**args)
    assert controller.acceptance._inventory(root)==after
    census=controller.snapshot(writer.root)
    (writer.root.parent/'slot17-dispatch-census.json').write_text(json.dumps(dict(
        receipt_bytes=(root/ref['path']).stat().st_size,metadata_bytes=census['metadata_bytes'],
        meta_group_bytes=census['groups']['evidence/meta'],live_checkpoint_files=12,
        completed_quality_records=18,real_panel_fit_proven=False)))


@pytest.mark.parametrize('phase',['process','review','late_control','late_evidence','publication'])
def test_failure_keeps_evidence_and_blocks_retry(writer,monkeypatch,phase):
    args,calls=setup(writer,monkeypatch);root=writer.root/'evidence'
    def fail(*a,**kw):raise RuntimeError('synthetic '+phase)
    if phase=='process':monkeypatch.setattr(controller,'run_process',fail)
    elif phase=='review':monkeypatch.setattr(controller.acceptance,'review_completed_later_confirmation',fail)
    elif phase=='publication':
        original=writer.publish_json
        def publish(name,*a,**kw):
            if name=='evidence/meta/quality-dispatch-17.json':
                (writer.root/(name+'.tmp')).write_bytes(b'partial');raise OSError('synthetic partial receipt')
            return original(name,*a,**kw)
        monkeypatch.setattr(writer,'publish_json',publish)
    else:
        original=controller.dispatch._PhaseGuard.__exit__
        def late(guard,*a,**kw):
            value=original(guard,*a,**kw)
            if guard.name=='control/quality-controller-review-17.json':
                path=writer.root/'control/process-quality-17.json' if phase=='late_control' else root/'quality/17/result.json'
                path.write_bytes(b'late mutation')
            return value
        monkeypatch.setattr(controller.dispatch._PhaseGuard,'__exit__',late)
    with pytest.raises((ValueError,RuntimeError,OSError)):controller.run_sealed_confirmation_once(**args)
    failure=json.loads((root/'meta/quality-dispatch-17-failure.json').read_text())
    assert not (root/'meta/quality-dispatch-17.json').exists() and failure['retry_allowed'] is False
    assert calls==([] if phase=='process' else [17])
    assert len(list(root.rglob('*.pt')))==(10 if phase=='process' else 12) and not (root/'quality/18').exists()
    with pytest.raises(ValueError):controller.run_sealed_confirmation_once(**args)


def test_changed_seal_blocks_child_and_preserves_attempt(writer,monkeypatch):
    args,calls=setup(writer,monkeypatch);root=writer.root/'evidence'
    original=controller.verify_sealed_confirmation
    def late(*a,**kw):
        value=original(*a,**kw)
        (root/args['readiness_ref']['path']).write_bytes(b'changed seal after replay')
        return value
    monkeypatch.setattr(controller,'verify_sealed_confirmation',late)
    with pytest.raises(ValueError):controller.run_sealed_confirmation_once(**args)
    assert calls==[] and not (root/'quality/17/process').exists()
    assert (root/'meta/quality-dispatch-17-failure.json').exists()
    with pytest.raises(ValueError):controller.run_sealed_confirmation_once(**args)


@pytest.mark.parametrize('slot',[0,9,15,16,18,23,24,-1,True,17.,'17'])
def test_other_slots_stay_closed(writer,slot):
    with pytest.raises(ValueError,match='slot17'):controller.run_sealed_confirmation_once(writer,slot,{},{})
    assert not (writer.root/'evidence').exists()
