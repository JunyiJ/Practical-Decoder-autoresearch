"""Complete synthetic LR cleanup -> rank preparation; no training or model data."""
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_quality_rank_startup_v2 as start
from src.research import tokenizer_wide_hybrid_lr_boundary_v2 as boundary
from src.research import tokenizer_wide_hybrid_storage_v2 as storage
from src.research.protocol import file_digest
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_lr_boundary_v2 import setup as lr_setup
import test_tokenizer_wide_hybrid_retention_audit as fixture_module


def setup(writer,monkeypatch,slot=9):
    args=lr_setup(writer,monkeypatch);root=writer.root/'evidence'
    completion=boundary.cleanup_lr_boundary(**args)
    # The independent campaign loader is replaced only by the existing bound
    # synthetic source/config fixture. History/cleanup/live audits stay real.
    monkeypatch.setattr(start,'bind_campaign',boundary.selection._frozen.__globals__['bind_campaign'])
    monkeypatch.setattr(start,'PROJECTION',boundary.selection._frozen.__globals__['PROJECTION'])
    monkeypatch.setattr(start,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=64*1024**3)))
    history=boundary.verify_lr_boundary(writer,args['plan_sha256'],completion)
    if slot>9:
        bound,requests=start._frozen(writer,14,args['plan_sha256'],history['projection_ref'],history['rates'])
        plan=start._document(root,dict(path='meta/plan.json',sha256=args['plan_sha256']))
        original=fixture_module.write
        existing={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
        def preserve(root,name,value):
            if name in existing:return dict(path=name,sha256=file_digest(root/name))
            return original(root,name,value)
        with monkeypatch.context() as patch:
            patch.setattr(fixture_module,'write',preserve)
            fixture_module.fixture(root,'rank',producer_v2=True,rank_prefix_count=slot-9,
                raw_name={i:'final_001200_123456.pt' for i in range(9,15)},bindings=dict(
                    context=bound['context'],projection_ref=history['projection_ref'],
                    numerical_ref=plan['numerical_ref'],training_token_sha256=plan['training_token_sha256'],requests=requests))
    return dict(writer=writer,slot=slot,plan_sha256=args['plan_sha256'],
        projection_ref=history['projection_ref'],lr_completion_ref=completion)


@pytest.mark.parametrize('slot',[9,12,14])
def test_full_chain_live_anchors_and_storage_precede_rank_request(writer,monkeypatch,slot):
    args=setup(writer,monkeypatch,slot);root=writer.root/'evidence';before=start._inventory(root)
    original=start.bind_storage;calls=[]
    def gate(*a,**kw):
        assert not (root/f'quality/{slot:02d}').exists()
        assert len(kw['records'])==slot and len(kw['checkpoint_pairs'])==slot+3
        assert len(kw['stages'])==2 and kw['selection'] is None and kw['rates'] is not None
        calls.append(slot);return original(*a,**kw)
    monkeypatch.setattr(start,'bind_storage',gate)
    ref=start.prepare_rank_request(**args);report=start._document(root,ref)
    assert calls==[slot] and len(report['completed_records'])==slot
    assert report['audited_indices']==[0,1,2]+list(range(9,slot))
    assert report['storage']['cleanup_boundaries_verified']==2
    assert report['storage']['required_free_bytes']>30_000_000_000
    assert report['storage']['checkpoint_logical_bytes']==sum(p.stat().st_size for p in root.rglob('*.pt'))
    assert all(report[k] is False for k in ('adopted','dispatch_authorized','deletion_authorized','next_stage_authorized'))
    assert report['selection'] is None
    assert {p.name for p in (root/f'quality/{slot:02d}').iterdir()}=={'request.json'}
    assert start._document(root,report['request'])['spec']==start.rank_specs(report['rates'])[slot-9]
    after=start._inventory(root);assert all(after[n]==v for n,v in before.items())
    assert not (root/'.retention/rank').exists()
    with pytest.raises(ValueError):start.prepare_rank_request(**args)
    assert start._inventory(root)==after


@pytest.mark.parametrize('bad',['history','anchor','conditional','source','storage'])
def test_failed_gate_never_materializes_request_and_cannot_retry(writer,monkeypatch,bad):
    args=setup(writer,monkeypatch);root=writer.root/'evidence'
    if bad=='history':(root/'.retention/lr/deletions.jsonl').write_bytes(b'changed')
    elif bad=='anchor':(root/'quality/00/checkpoints/raw.pt').write_bytes(b'changed')
    elif bad=='source':(root/'meta/source.txt').write_bytes(b'changed')
    elif bad=='storage':
        monkeypatch.setattr(storage,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=8*1024**3)))
    else:
        old=start.bind_campaign
        def changed(*a):
            bound=old(*a);bound['requests']['quality']['9']=[];return bound
        monkeypatch.setattr(start,'bind_campaign',changed)
    before=start._inventory(root)
    with pytest.raises(ValueError):start.prepare_rank_request(**args)
    assert not (root/'quality/09').exists()
    assert (root/'meta/quality-rank-startup-09-failure.json').exists()
    after=start._inventory(root);assert all(after[n]==v for n,v in before.items())
    with pytest.raises(ValueError,match='cannot be retried'):start.prepare_rank_request(**args)
    assert start._inventory(root)==after


@pytest.mark.parametrize('when',['before_request','after_request'])
def test_source_change_during_preparation_retains_failure_and_partial_request(writer,monkeypatch,when):
    args=setup(writer,monkeypatch);root=writer.root/'evidence'
    if when=='before_request':
        old=start.bind_storage
        def changed(*a,**kw):
            value=old(*a,**kw);(root/'meta/source.txt').write_text('changed');return value
        monkeypatch.setattr(start,'bind_storage',changed)
    else:
        old=writer.publish_json
        def changed(name,*a,**kw):
            value=old(name,*a,**kw)
            if name=='evidence/quality/09/request.json':(root/'meta/source.txt').write_text('changed')
            return value
        monkeypatch.setattr(writer,'publish_json',changed)
    with pytest.raises(ValueError):start.prepare_rank_request(**args)
    assert (root/'meta/quality-rank-startup-09-failure.json').exists()
    assert not (root/'meta/quality-rank-startup-09.json').exists()
    assert (root/'quality/09/request.json').exists()==(when=='after_request')
    with pytest.raises(ValueError):start.prepare_rank_request(**args)


@pytest.mark.parametrize('slot',[8,15,23,-1,True,'9'])
def test_only_exact_rank_slot_integers_before_mutation(writer,slot):
    with pytest.raises(ValueError):start.prepare_rank_request(writer,slot,'0'*64,projection_ref={},lr_completion_ref={})
    assert not (writer.root/'evidence/quality').exists()
