"""Rank readiness with real temporary history/retention audits, no child launch."""
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_quality_readiness_v2 as ready
from src.research import tokenizer_wide_hybrid_quality_rank_startup_v2 as rank
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_quality_rank_startup_v2 import setup


def prepared(writer,monkeypatch,slot=9):
    args=setup(writer,monkeypatch,slot)
    ref=rank.prepare_rank_request(**args)
    monkeypatch.setattr(ready,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=64*1024**3)))
    return ref


@pytest.mark.parametrize('slot',[9,14])
def test_seal_binds_complete_rank_review_without_process_or_selection(writer,monkeypatch,slot):
    startup=prepared(writer,monkeypatch,slot);root=writer.root/'evidence';before=ready._inventory(root)
    ref=ready.seal_quality_readiness(writer,slot,startup);report=ready._document(root,ref)
    after=ready._inventory(root)
    assert all(after[n]==v for n,v in before.items()) and set(after)-set(before)=={ref['path']}
    assert report['inventory']==before and report['startup_ref']==startup and report['slot']==slot
    assert all(report['predispatch_review'][k] is True for k in (
        'pilot_history_verified','lr_history_verified','rank_prefix_verified','rank_request_verified'))
    original=ready._document(root,startup)
    bound,request,allowance=ready._bound(writer,slot,original)
    assert report['context']==bound['context'] and report['frozen_files']==bound['files']
    assert report['required_free_bytes']==original['storage']['required_free_bytes']>30_000_000_000
    assert report['output_limit_bytes']==allowance
    assert report['control_inventory'][report['guard']['path']]['sha256']==report['guard']['sha256']
    assert report['dispatch_authorized'] is False and report['deletion_authorized'] is False
    assert {p.name for p in (root/f'quality/{slot:02d}').iterdir()}=={'request.json'}
    assert not (root/'meta/rank-selection.json').exists() and not (root/'.retention/rank').exists()
    with pytest.raises(ValueError,match='cannot be retried'):ready.seal_quality_readiness(writer,slot,startup)
    assert ready._inventory(root)==after


@pytest.mark.parametrize('bad',['history','process','claims'])
def test_failed_rank_review_keeps_evidence_and_permanently_blocks_seal(writer,monkeypatch,bad):
    startup=prepared(writer,monkeypatch);root=writer.root/'evidence'
    if bad=='history':(root/'.retention/lr/deletions.jsonl').write_bytes(b'changed')
    elif bad=='process':writer.publish_json('evidence/quality/09/process/launch.json',{})
    else:
        original=rank.verify_rank_request
        def changed(*a,**kw):
            value=original(*a,**kw);value['lr_history_verified']=False;return value
        monkeypatch.setattr(rank,'verify_rank_request',changed)
    before=ready._inventory(root)
    with pytest.raises(ValueError):ready.seal_quality_readiness(writer,9,startup)
    after=ready._inventory(root)
    assert all(after[n]==v for n,v in before.items())
    assert set(after)-set(before)=={'meta/quality-readiness-09-failure.json'}
    assert not (root/'meta/quality-readiness-09.json').exists()
    with pytest.raises(ValueError,match='cannot be retried'):ready.seal_quality_readiness(writer,9,startup)
    assert ready._inventory(root)==after


@pytest.mark.parametrize('bad',['control','source_after_guard','space_after_guard'])
def test_late_change_prevents_rank_seal(writer,monkeypatch,bad):
    startup=prepared(writer,monkeypatch);root=writer.root/'evidence'
    if bad=='control':
        original=rank.verify_rank_request
        def changed(*a,**kw):
            value=original(*a,**kw)
            (writer.root/'control/quality-rank-startup-09.json').write_bytes(b'{}')
            return value
        monkeypatch.setattr(rank,'verify_rank_request',changed)
    elif bad=='source_after_guard':
        original=ready._ReadinessGuard.__exit__
        def finished(*a,**kw):
            value=original(*a,**kw);(root/'meta/source.txt').write_text('changed after guard');return value
        monkeypatch.setattr(ready._ReadinessGuard,'__exit__',finished)
    else:
        needed=ready._document(root,startup)['storage']['required_free_bytes'];values=iter([needed,needed-1])
        monkeypatch.setattr(ready,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=next(values))))
    with pytest.raises(ValueError):ready.seal_quality_readiness(writer,9,startup)
    assert not (root/'meta/quality-readiness-09.json').exists()
    assert (root/'meta/quality-readiness-09-failure.json').exists()
    with pytest.raises(ValueError,match='cannot be retried'):ready.seal_quality_readiness(writer,9,startup)


def test_partial_rank_seal_publication_is_preserved_and_not_retried(writer,monkeypatch):
    startup=prepared(writer,monkeypatch);root=writer.root/'evidence';original=writer.publish_json
    def fail(name,payload,**kw):
        if name=='evidence/meta/quality-readiness-09.json':
            (writer.root/(name+'.tmp')).write_bytes(b'partial publication')
            raise OSError('synthetic publication failure')
        return original(name,payload,**kw)
    monkeypatch.setattr(writer,'publish_json',fail)
    with pytest.raises(OSError):ready.seal_quality_readiness(writer,9,startup)
    assert (root/'meta/quality-readiness-09.json.tmp').read_bytes()==b'partial publication'
    assert (root/'meta/quality-readiness-09-failure.json').exists()
    assert not (root/'meta/quality-readiness-09.json').exists()
    with pytest.raises(ValueError,match='cannot be retried'):ready.seal_quality_readiness(writer,9,startup)


@pytest.mark.parametrize('slot',[16,23,-1,True,'9'])
def test_invalid_or_later_confirmation_slot_never_creates_readiness_guard(writer,slot):
    with pytest.raises(ValueError):ready.seal_quality_readiness(writer,slot,{})
    assert not (writer.root/'evidence/meta').exists()


def test_rank_slot_requires_its_exact_startup_path_before_mutation(writer):
    with pytest.raises(ValueError,match='Exact startup reference'):
        ready.seal_quality_readiness(writer,9,dict(path='meta/quality-lr-startup-09.json'))
    assert not (writer.root/'evidence/meta').exists()
