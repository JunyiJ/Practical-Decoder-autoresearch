"""Real serial rank audits and synthetic child bytes; no model or MPS launch."""
import copy
import json
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_quality_dispatch_v2 as dispatch
from src.research import tokenizer_wide_hybrid_quality_rank_startup_v2 as rank
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_quality_rank_startup_v2 import setup as rank_setup
from test_tokenizer_wide_hybrid_quality_acceptance_v2 import complete


def setup(writer,monkeypatch):
    args=rank_setup(writer,monkeypatch);args.pop('slot')
    free=SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=64*1024**3))
    for module in (dispatch,dispatch.readiness,dispatch.acceptance):monkeypatch.setattr(module,'shutil',free)
    calls=[]
    def process(w,stage,slot,argv,cwd,**limits):
        assert w is writer and stage=='quality' and cwd==dispatch.ROOT
        assert limits['timeout_seconds']==7200 and limits['log_limit_bytes']==4*1024**2
        assert not (writer.root/f'control/quality-controller-review-{slot:02d}.json').exists()
        prepare=json.loads((writer.root/f'control/quality-controller-prepare-{slot:02d}.json').read_text())
        assert prepare['status']=='finished'
        startup=dict(path=f'meta/quality-rank-startup-{slot:02d}.json',sha256=argv[9])
        ready=dict(path=f'meta/quality-readiness-{slot:02d}.json',sha256=argv[15])
        seal=dispatch._document(writer.root/'evidence',ready)
        assert limits['required_free_bytes']==seal['required_free_bytes'] and limits['output_limit_bytes']==seal['output_limit_bytes']
        limits['parent_check']();calls.append(slot)
        return complete(writer,monkeypatch,slot,startup,ready)['process']
    monkeypatch.setattr(dispatch,'run_process',process)
    return args,calls


def test_all_six_serial_rank_slots_preserve_history_and_stop_before_selection(writer,monkeypatch):
    args,calls=setup(writer,monkeypatch);root=writer.root/'evidence';before=dispatch.acceptance._inventory(root);refs=[]
    for slot in range(9,15):
        ref=dispatch.run_rank_once(slot=slot,**args);saved=dispatch._document(root,ref)
        assert saved['status']=='rank_reviewed' and saved['prior_dispatches']==refs
        assert saved['lr_completion_ref']==args['lr_completion_ref']
        assert len(saved['review']['completed_records'])==slot+1 and saved['review']['selection'] is None
        assert saved['review']['storage']['cleanup_boundaries_verified']==2
        assert all(saved['review'][k] is False for k in ('adopted','dispatch_authorized','deletion_authorized','next_stage_authorized'))
        refs.append(ref)
        with pytest.raises(ValueError,match='cannot be retried'):dispatch.run_rank_once(slot=slot,**args)
    assert calls==list(range(9,15))
    after=dispatch.acceptance._inventory(root);assert all(after[n]==v for n,v in before.items())
    assert not (root/'quality/15').exists() and not (root/'meta/rank-selection.json').exists() and not (root/'.retention/rank').exists()
    monkeypatch.setattr(dispatch,'os',SimpleNamespace(getpid=lambda:1))
    assert dispatch._prior(writer,15,args['plan_sha256'],args['projection_ref'],saved['pilot_completion_ref'],
        lr_completion_ref=args['lr_completion_ref'])==refs


def test_prior_rank_receipts_reject_changed_transport_and_rehashed_claims(writer,monkeypatch):
    args,calls=setup(writer,monkeypatch);ref=dispatch.run_rank_once(slot=9,**args);root=writer.root/'evidence'
    saved=dispatch._document(root,ref)
    prior=lambda:dispatch._prior(writer,10,args['plan_sha256'],args['projection_ref'],saved['pilot_completion_ref'],
        lr_completion_ref=args['lr_completion_ref'])
    paths=[root/ref['path']]+[writer.root/g['path'] for g in saved['guards'].values()]+[
        root/saved['review'][k]['path'] for k in ('startup','readiness')]+[
        root/v['path'] for v in saved['review']['child'].values()]+[writer.root/saved['review']['entry_guard']['path'],
        writer.root/'control/quality-rank-startup-09.json']
    for path in paths:
        original=path.read_bytes();path.write_bytes(b'{}')
        try:
            with pytest.raises((ValueError,KeyError)):prior()
        finally:path.write_bytes(original)
    receipt=root/ref['path'];original=receipt.read_bytes()
    for key,value in [('status','lr_reviewed'),('slot',True),('lr_completion_ref',{}),('prior_dispatches',[ref]),('parent_pid',1)]:
        changed=copy.deepcopy(saved);changed[key]=value;receipt.write_text(json.dumps(changed))
        try:
            with pytest.raises(ValueError):prior()
        finally:receipt.write_bytes(original)
    terminal=root/saved['review']['child']['process']['path'];old=terminal.read_bytes()
    for key,value in [('timeout_seconds',600),('log_truncated',True),('elapsed_seconds',7200.),
        ('observation',dict(free_bytes=64*1024**3,output_bytes=0,log_bytes=0))]:
        record=json.loads(old);record[key]=value;terminal.write_text(json.dumps(record))
        changed=copy.deepcopy(saved);changed['review']['child']['process']['sha256']=dispatch.file_digest(terminal)
        receipt.write_text(json.dumps(changed))
        try:
            with pytest.raises(ValueError,match='Earlier terminal'):prior()
        finally:terminal.write_bytes(old);receipt.write_bytes(original)
    assert prior()==[ref] and calls==[9]
    # The actual transition must check prior transport before creating a request.
    (writer.root/saved['guards']['child']['path']).write_bytes(b'{}')
    with pytest.raises(ValueError):dispatch.run_rank_once(slot=10,**args)
    assert calls==[9] and not (root/'quality/10').exists()
    assert (root/'meta/quality-dispatch-10-failure.json').exists()


@pytest.mark.parametrize('phase',['prepare','seal','process','review','monitor','publish'])
def test_rank_failure_is_preserved_and_permanently_blocks_retry(writer,monkeypatch,phase):
    args,calls=setup(writer,monkeypatch)
    def fail(*a,**kw):raise RuntimeError('synthetic '+phase)
    if phase=='prepare':monkeypatch.setattr(rank,'prepare_rank_request',fail)
    elif phase=='seal':monkeypatch.setattr(dispatch.readiness,'seal_quality_readiness',fail)
    elif phase=='process':monkeypatch.setattr(dispatch,'run_process',fail)
    elif phase=='review':monkeypatch.setattr(dispatch.acceptance,'review_completed_rank',fail)
    elif phase=='monitor':monkeypatch.setattr(dispatch._ChildMonitor,'check',fail)
    else:
        publish=writer.publish_json
        def publication(name,value,**kw):
            if name=='evidence/meta/quality-dispatch-09.json':fail()
            return publish(name,value,**kw)
        monkeypatch.setattr(writer,'publish_json',publication)
    with pytest.raises(RuntimeError,match='synthetic'):dispatch.run_rank_once(slot=9,**args)
    root=writer.root/'evidence';failed=json.loads((root/'meta/quality-dispatch-09-failure.json').read_text())
    assert failed['retry_allowed'] is False and not (root/'meta/quality-dispatch-09.json').exists()
    before=dispatch.acceptance._inventory(root)
    with pytest.raises(ValueError):dispatch.run_rank_once(slot=9,**args)
    assert dispatch.acceptance._inventory(root)==before


@pytest.mark.parametrize('slot',[True,-1,0,8,15,23,'9'])
def test_rank_dispatch_exact_slots_before_mutation(writer,slot):
    with pytest.raises(ValueError,match='Exact rank writer/slot'):
        dispatch.run_rank_once(writer,slot,'a'*64,projection_ref={},lr_completion_ref={})
    assert not (writer.root/'evidence').exists()


def test_cannot_skip_prior_rank_transport_even_with_scientific_prefix(writer,monkeypatch):
    args=rank_setup(writer,monkeypatch,10)
    with pytest.raises(FileNotFoundError):dispatch.run_rank_once(**args)
    assert not (writer.root/'evidence/quality/10').exists()
    assert (writer.root/'evidence/meta/quality-dispatch-10-failure.json').exists()


def test_child_success_does_not_excuse_late_historical_control_change(writer,monkeypatch):
    args,calls=setup(writer,monkeypatch);first=dispatch.run_rank_once(slot=9,**args)
    saved=dispatch._document(writer.root/'evidence',first);original=dispatch.run_process
    def change(*a,**kw):
        value=original(*a,**kw)
        (writer.root/saved['guards']['review']['path']).write_bytes(b'{}')
        return value
    monkeypatch.setattr(dispatch,'run_process',change)
    with pytest.raises(ValueError):dispatch.run_rank_once(slot=10,**args)
    assert calls==[9,10] and not (writer.root/'evidence/meta/quality-dispatch-10.json').exists()
    assert (writer.root/'evidence/meta/quality-dispatch-10-failure.json').exists()
