"""Synthetic full serial rank boundary; no model, training or real checkpoints."""
import copy
import json
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_rank_selection_v2 as selection
from src.research import tokenizer_wide_hybrid_quality_dispatch_v2 as dispatch
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_rank_dispatch_v2 import setup as dispatch_setup


def setup(writer,monkeypatch,count=6):
    args,calls=dispatch_setup(writer,monkeypatch)
    monkeypatch.setattr(selection,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=64*1024**3)))
    for slot in range(9,9+count):dispatch.run_rank_once(slot=slot,**args)
    return args,calls


def test_final_choice_full_replay_and_readonly_tamper_rejection(writer,monkeypatch):
    args,calls=setup(writer,monkeypatch);root=writer.root/'evidence';before=selection._inventory(root)
    ref=selection.save_rank_selection(**args);saved=selection._document(root,ref)
    checked=selection.verify_rank_selection(writer,args['plan_sha256'],ref)
    assert calls==list(range(9,15)) and checked['audit']['stage']['stage_audits_bound']
    assert checked['audit']['stage']['producer_provenance_bound']
    assert len(checked['audit']['stage']['manifest']['delete'])==12
    records=checked['audit']['summary']['records']
    assert checked['selection']==selection.choose_ranks(records[:9],records[9:])
    assert selection._document(root,saved['selection_ref'])==checked['selection']
    assert checked['audit']['storage']['cleanup_boundaries_verified']==2
    assert all(saved[k] is False and checked[k] is False for k in selection.NONCLAIMS)
    after=selection._inventory(root);assert all(after[n]==v for n,v in before.items())
    assert len(list(root.rglob('*.pt')))==18 and not (root/'quality/15').exists() and not (root/'.retention/rank').exists()
    assert selection.snapshot(writer.root)['groups']['evidence/meta']<2*1024**2
    # Historical transport, live weights, numerical evidence and control bytes
    # remain mandatory even though selection receipts summarize full audits.
    for path in (root/selection.CHOICE,writer.root/selection.GUARD,root/'meta/quality-dispatch-14.json',
        root/'quality/00/checkpoints/raw.pt',root/'quality/14/tensor-audit.json',root/'meta/numerical.json',
        root/'.retention/lr/deletions.jsonl'):
        original=path.read_bytes();path.write_bytes(b'{}')
        try:
            with pytest.raises((ValueError,KeyError)):selection.verify_rank_selection(writer,args['plan_sha256'],ref)
        finally:path.write_bytes(original)
    # Rehashing a forged completion does not replace independently replayed facts.
    path=root/ref['path'];original=path.read_bytes()
    for key,value in [('stage_sha256','a'*64),('manifest_sha256','b'*64),('summary_sha256','c'*64),
        ('parent_pid',True),('adopted',True),('storage',dict(saved['storage'],required_free_bytes=1))]:
        changed=copy.deepcopy(saved);changed[key]=value;path.write_text(json.dumps(changed))
        changed_ref=dict(path=ref['path'],sha256=selection.file_digest(path))
        try:
            with pytest.raises(ValueError):selection.verify_rank_selection(writer,args['plan_sha256'],changed_ref)
        finally:path.write_bytes(original)
    with pytest.raises(ValueError,match='cannot be retried'):selection.save_rank_selection(**args)
    assert selection._inventory(root)==after


def test_incomplete_rank_stage_never_persists_selection(writer,monkeypatch):
    args,calls=setup(writer,monkeypatch,count=0);root=writer.root/'evidence';before=selection._inventory(root)
    with pytest.raises((ValueError,FileNotFoundError)):selection.save_rank_selection(**args)
    assert not (root/selection.CHOICE).exists() and (root/(selection.BASE+'-failure.json')).exists()
    after=selection._inventory(root);assert all(after[n]==v for n,v in before.items())
    with pytest.raises(ValueError):selection.save_rank_selection(**args)
    assert selection._inventory(root)==after


@pytest.mark.parametrize('phase',['choice','stage','completion'])
def test_partial_boundary_never_authorizes_cleanup_or_retry(writer,monkeypatch,phase):
    args,calls=setup(writer,monkeypatch);root=writer.root/'evidence';publish=writer.publish_json
    def fail(*a,**kw):raise RuntimeError('synthetic '+phase)
    if phase=='stage':monkeypatch.setattr(selection,'bind_quality_stage',fail)
    else:
        def publication(name,value,**kw):
            target=selection.CHOICE if phase=='choice' else selection.BASE+'-complete.json'
            if name=='evidence/'+target:fail()
            return publish(name,value,**kw)
        monkeypatch.setattr(writer,'publish_json',publication)
    with pytest.raises(RuntimeError,match='synthetic'):selection.save_rank_selection(**args)
    assert (root/(selection.BASE+'-review.json')).exists()
    assert (root/selection.CHOICE).exists()==(phase!='choice')
    assert not (root/(selection.BASE+'-complete.json')).exists() and not (root/'.retention/rank').exists()
    before=selection._inventory(root)
    with pytest.raises(ValueError):selection.save_rank_selection(**args)
    assert selection._inventory(root)==before and len(list(root.rglob('*.pt')))==18


@pytest.mark.parametrize('changed',['source','control'])
def test_late_provenance_change_blocks_completion(writer,monkeypatch,changed):
    args,calls=setup(writer,monkeypatch);root=writer.root/'evidence';original=selection._guard
    def late(*a,**kw):
        value=original(*a,**kw)
        path=root/'meta/source.txt' if changed=='source' else writer.root/'control/quality-controller-child-14.json'
        path.write_bytes(b'changed after selection guard')
        return value
    monkeypatch.setattr(selection,'_guard',late)
    with pytest.raises(ValueError):selection.save_rank_selection(**args)
    assert (root/selection.CHOICE).exists() and (root/(selection.BASE+'-failure.json')).exists()
    assert not (root/(selection.BASE+'-complete.json')).exists()
    assert len(list(root.rglob('*.pt')))==18 and not (root/'.retention/rank').exists()
