"""Synthetic whole-chain first-fresh preparation; no model or child process."""
import copy
import json
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_first_confirmation_startup_v2 as start
from src.research import tokenizer_wide_hybrid_storage_v2 as storage
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_rank_boundary_v2 import setup as rank_setup,boundary


def setup(writer,monkeypatch):
    args=rank_setup(writer,monkeypatch);root=writer.root/'evidence'
    completion=boundary.cleanup_rank_boundary(**args)
    checked=boundary.verify_rank_boundary(writer,args['plan_sha256'],completion)
    monkeypatch.setattr(start,'bind_campaign',boundary.selection._frozen.__globals__['bind_campaign'])
    monkeypatch.setattr(start,'PROJECTION',boundary.selection._frozen.__globals__['PROJECTION'])
    monkeypatch.setattr(start,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=64*1024**3)))
    return dict(writer=writer,slot=15,plan_sha256=args['plan_sha256'],
        projection_ref=checked['projection_ref'],rank_completion_ref=completion)


@pytest.mark.parametrize('slot',[False,True,0,9,14,16,23,24,15.0,'15'])
def test_only_first_fresh_slot_is_supported_without_writing(writer,slot):
    before=start.snapshot(writer.root)
    with pytest.raises(ValueError,match='slot15'):
        start.prepare_first_confirmation_request(writer,slot,'a'*64,projection_ref={},rank_completion_ref={})
    with pytest.raises(ValueError,match='slot15'):
        start.verify_first_confirmation_request(writer,slot,{})
    assert start.snapshot(writer.root)==before


def test_full_history_selected_anchors_and_capacity_precede_first_fresh_request(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence';before=start._inventory(root)
    def audit():return start._audit(writer,15,args['plan_sha256'],args['projection_ref'],args['rank_completion_ref'])
    # Exercise each independent gate before any request exists. These calls are
    # read-only; restore only disposable corruption fixtures between cases.
    for path in (root/'.retention/rank/deletions.jsonl',root/'quality/09/checkpoints/final_001200_123456.pt',
        root/'meta/rank-selection.json',root/'meta/numerical.json',root/'meta/source.txt',
        writer.root/'control/quality-controller-child-14.json'):
        old=path.read_bytes();path.write_bytes(b'changed')
        try:
            with pytest.raises((ValueError,KeyError)):audit()
        finally:path.write_bytes(old)
    with monkeypatch.context() as patch:
        original=start.bind_campaign
        def missing(*a):
            value=original(*a);value['requests']['quality']['15']=[];return value
        patch.setattr(start,'bind_campaign',missing)
        with pytest.raises(ValueError,match='conditional'):audit()
    with monkeypatch.context() as patch:
        patch.setattr(storage,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=8*1024**3)))
        with pytest.raises(ValueError,match='does not fit'):audit()
    assert start._inventory(root)==before and not (root/'quality/15').exists()
    original=start.bind_storage;calls=[]
    def gate(*a,**kw):
        assert not (root/'quality/15').exists()
        assert len(kw['records'])==15 and len(kw['checkpoint_pairs'])==18 and len(kw['stages'])==3
        assert all(c['divisor']==8 for c in kw['selection'].values())
        calls.append(1);return original(*a,**kw)
    with monkeypatch.context() as patch:
        patch.setattr(start,'bind_storage',gate)
        ref=start.prepare_first_confirmation_request(**args)
    assert calls==[1]
    report=start._document(root,ref);after=start._inventory(root)
    assert all(after[n]==v for n,v in before.items())
    history=start.verify_rank_history(writer,args['plan_sha256'],args['rank_completion_ref'],current_inventory=after)
    assert report['completed_records_sha256']==start.digest(history['records']) and report['audited_indices']==[9,10,11]
    assert report['checkpoint_pairs_sha256']==start.digest(history['checkpoint_pairs']) and len(list(root.rglob('*.pt')))==6
    assert report['stages_sha256']==start.digest(history['stages'])
    assert not set(report)&{'completed_records','checkpoint_pairs','stages','frozen_files'}
    assert (root/ref['path']).stat().st_size<16000
    assert report['storage']['cleanup_boundaries_verified']==3
    assert report['storage']['projection']['phase']=='confirm'
    assert report['storage']['required_free_bytes']>8*1024**3
    assert {p.name for p in (root/'quality/15').iterdir()}=={'request.json'}
    assert start._document(root,report['request'])['spec']==start.confirmation_specs(report['selection'])[0]
    assert all(report[k] is False for k in ('adopted','dispatch_authorized','deletion_authorized','next_stage_authorized'))
    checked=start.verify_first_confirmation_request(writer,15,ref)
    assert checked['required_free_bytes']==report['storage']['required_free_bytes']
    assert checked['first_confirmation_request_verified'] and checked['rank_history_verified'] and not checked['dispatch_authorized']
    assert start._inventory(root)==after
    # A rehashed false receipt cannot replace independent history/audits/capacity.
    path=root/ref['path'];raw=path.read_bytes()
    for key,value in [('parent_pid',True),('adopted',True),('completed_records_sha256','a'*64),
        ('selection',dict(report['selection'],hybrid=dict(lr=.0025,divisor=2))),
        ('checkpoint_pairs_sha256','b'*64),('stages_sha256','c'*64),('frozen_files_sha256','d'*64),('audited_indices',[]),
        ('storage',dict(report['storage'],required_free_bytes=1))]:
        changed=copy.deepcopy(report);changed[key]=value;path.write_text(json.dumps(changed))
        bad_ref=dict(path=ref['path'],sha256=start.file_digest(path))
        try:
            with pytest.raises((ValueError,KeyError)):start.verify_first_confirmation_request(writer,15,bad_ref)
        finally:path.write_bytes(raw)
    for path in (root/'quality/15/request.json',root/'quality/10/tensor-audit.json',
        root/'.retention/pilot/deletions.jsonl',writer.root/'control/quality-controller-child-14.json'):
        old=path.read_bytes();path.write_bytes(b'changed')
        try:
            with pytest.raises((ValueError,KeyError)):start.verify_first_confirmation_request(writer,15,ref)
        finally:path.write_bytes(old)
    path=root/'quality/15/result.json';path.write_text('{}')
    try:
        with pytest.raises(ValueError):start.verify_first_confirmation_request(writer,15,ref)
    finally:path.unlink()
    with monkeypatch.context() as patch:
        patch.setattr(start,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=8*1024**3)))
        with pytest.raises(ValueError,match='no longer fits'):start.verify_first_confirmation_request(writer,15,ref)
    with pytest.raises(ValueError):start.prepare_first_confirmation_request(**args)
    assert start._inventory(root)==after and not (root/'quality/16').exists()
    census=start.snapshot(writer.root)
    (writer.root.parent/'first-confirmation-startup-census.json').write_text(json.dumps(dict(
        groups=census['groups'],metadata_bytes=census['metadata_bytes'],file_count=len(census['files']),
        startup_bytes=(root/ref['path']).stat().st_size,completed_fresh=0,live_checkpoint_files=6,
        real_panel_fit_proven=False),sort_keys=True))


@pytest.mark.parametrize('when',['storage','before_request','after_request'])
def test_failed_preparation_retains_evidence_and_never_retries(writer,monkeypatch,when):
    args=setup(writer,monkeypatch);root=writer.root/'evidence';before=start._inventory(root)
    if when=='storage':
        monkeypatch.setattr(storage,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=8*1024**3)))
    elif when=='before_request':
        original=start.bind_storage
        def changed(*a,**kw):
            result=original(*a,**kw);(root/'meta/source.txt').write_text('changed');return result
        monkeypatch.setattr(start,'bind_storage',changed)
    else:
        original=writer.publish_json
        def changed(name,*a,**kw):
            result=original(name,*a,**kw)
            if name=='evidence/quality/15/request.json':(root/'meta/source.txt').write_text('changed')
            return result
        monkeypatch.setattr(writer,'publish_json',changed)
    with pytest.raises(ValueError):start.prepare_first_confirmation_request(**args)
    assert (root/'quality/15/request.json').exists()==(when=='after_request')
    assert (root/'meta/quality-confirm-startup-15-failure.json').exists()
    assert not (root/'meta/quality-confirm-startup-15.json').exists()
    after=start._inventory(root)
    assert all(after[n]==v for n,v in before.items() if n!='meta/source.txt')
    assert len(list(root.rglob('*.pt')))==6
    with pytest.raises(ValueError):start.prepare_first_confirmation_request(**args)
    assert start._inventory(root)==after
