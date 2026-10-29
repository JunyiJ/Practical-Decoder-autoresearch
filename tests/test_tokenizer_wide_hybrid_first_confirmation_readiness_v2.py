"""First fresh readiness, using real temporary audits and disposable bytes only."""
import json
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_quality_readiness_v2 as ready
from src.research import tokenizer_wide_hybrid_first_confirmation_startup_v2 as fresh
from src.research import tokenizer_wide_hybrid_quality_handoff_v2 as transport
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_first_confirmation_startup_v2 import setup


def prepared(writer,monkeypatch):
    args=setup(writer,monkeypatch)
    ref=fresh.prepare_first_confirmation_request(**args)
    monkeypatch.setattr(ready,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=64*1024**3)))
    return ref


@pytest.mark.parametrize('slot',[16,23,24,-1,True,15.0,'15'])
def test_unsupported_fresh_slot_never_creates_guard(writer,slot):
    before=ready.snapshot(writer.root)
    with pytest.raises(ValueError):ready.seal_quality_readiness(writer,slot,{})
    assert ready.snapshot(writer.root)==before


def test_first_fresh_requires_exact_startup_path(writer):
    before=ready.snapshot(writer.root)
    with pytest.raises(ValueError,match='Exact startup'):
        ready.seal_quality_readiness(writer,15,dict(path='meta/quality-rank-startup-15.json'))
    assert ready.snapshot(writer.root)==before


def test_first_fresh_seal_binds_full_idle_review_and_does_not_launch(writer,monkeypatch):
    startup=prepared(writer,monkeypatch);root=writer.root/'evidence'
    before=ready._inventory(root)
    ref=ready.seal_quality_readiness(writer,15,startup);seal=ready._document(root,ref)
    after=ready._inventory(root);report=ready._document(root,startup)
    assert seal['inventory']==before and seal['startup_ref']==startup and seal['slot']==15
    assert set(after)-set(before)=={ref['path']} and all(after[n]==v for n,v in before.items())
    assert all(seal['predispatch_review'][k] is True for k in ('pilot_history_verified','rank_history_verified',
        'selected_screen_prefix_verified','first_confirmation_request_verified'))
    bound,request,allowance=ready._bound(writer,15,report)
    assert request['spec']==fresh.confirmation_specs(report['selection'])[0]
    assert seal['context']==bound['context'] and seal['frozen_files']==bound['files']
    assert seal['required_free_bytes']==report['storage']['required_free_bytes']>8*1024**3
    assert seal['output_limit_bytes']==allowance
    assert seal['control_inventory'][seal['guard']['path']]['sha256']==seal['guard']['sha256']
    assert seal['control_inventory']==ready._controls(writer,'not-a-control')
    assert not seal['dispatch_authorized'] and not seal['deletion_authorized']
    assert not seal['predispatch_review']['dispatch_authorized']
    assert {p.name for p in (root/'quality/15').iterdir()}=={'request.json'}
    assert len(list(root.rglob('*.pt')))==6 and not (root/'quality/16').exists()
    assert ready._document(root,report['selection_ref'])==report['selection']
    with pytest.raises(ValueError,match='cannot be retried'):ready.seal_quality_readiness(writer,15,startup)
    assert transport.worker_argv(writer,15,startup,readiness_ref=ref)[2].endswith('quality_entry_v2')
    with pytest.raises(ValueError):fresh.verify_first_confirmation_request(writer,15,startup)
    assert ready._inventory(root)==after
    census=ready.snapshot(writer.root)
    (writer.root.parent/'first-confirmation-readiness-census.json').write_text(json.dumps(dict(
        groups=census['groups'],metadata_bytes=census['metadata_bytes'],file_count=len(census['files']),
        readiness_bytes=(root/ref['path']).stat().st_size,child_transport_implemented=True,
        live_checkpoint_files=6,completed_fresh=0,real_panel_fit_proven=False),sort_keys=True))


@pytest.mark.parametrize('bad',['claims','source_after_guard','space_after_guard'])
def test_failed_or_changed_review_preserves_attempt_and_blocks_retry(writer,monkeypatch,bad):
    startup=prepared(writer,monkeypatch);root=writer.root/'evidence';before=ready._inventory(root)
    if bad=='claims':
        original=fresh.verify_first_confirmation_request
        def changed(*a,**kw):
            value=original(*a,**kw);value['rank_history_verified']=False;return value
        monkeypatch.setattr(fresh,'verify_first_confirmation_request',changed)
    elif bad=='source_after_guard':
        original=ready._ReadinessGuard.__exit__
        def finished(*a,**kw):
            value=original(*a,**kw);(root/'meta/source.txt').write_text('changed after guard');return value
        monkeypatch.setattr(ready._ReadinessGuard,'__exit__',finished)
    else:
        needed=ready._document(root,startup)['storage']['required_free_bytes'];values=iter([needed,needed-1])
        monkeypatch.setattr(ready,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=next(values))))
    with pytest.raises(ValueError):ready.seal_quality_readiness(writer,15,startup)
    after=ready._inventory(root)
    assert all(after[n]==v for n,v in before.items() if n!='meta/source.txt')
    assert set(after)-set(before)=={'meta/quality-readiness-15-failure.json'}
    assert not (root/'meta/quality-readiness-15.json').exists()
    assert len(list(root.rglob('*.pt')))==6 and {p.name for p in (root/'quality/15').iterdir()}=={'request.json'}
    with pytest.raises(ValueError,match='cannot be retried'):ready.seal_quality_readiness(writer,15,startup)
    assert ready._inventory(root)==after


def test_partial_readiness_publication_is_preserved_and_never_retried(writer,monkeypatch):
    startup=prepared(writer,monkeypatch);root=writer.root/'evidence';original=writer.publish_json
    def fail(name,payload,**kw):
        if name=='evidence/meta/quality-readiness-15.json':
            (writer.root/(name+'.tmp')).write_bytes(b'partial publication')
            raise OSError('synthetic publication failure')
        return original(name,payload,**kw)
    monkeypatch.setattr(writer,'publish_json',fail)
    with pytest.raises(OSError):ready.seal_quality_readiness(writer,15,startup)
    assert (root/'meta/quality-readiness-15.json.tmp').read_bytes()==b'partial publication'
    assert (root/'meta/quality-readiness-15-failure.json').exists()
    assert not (root/'meta/quality-readiness-15.json').exists()
    assert len(list(root.rglob('*.pt')))==6
    with pytest.raises(ValueError,match='cannot be retried'):ready.seal_quality_readiness(writer,15,startup)
