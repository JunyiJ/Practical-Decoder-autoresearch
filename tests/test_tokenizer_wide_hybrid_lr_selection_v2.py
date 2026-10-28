"""Synthetic nine-screen selection boundaries; no model training or deletion."""
import json
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_lr_selection_v2 as selection
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_quality_dispatch_v2 import setup as dispatch_setup
from src.research import tokenizer_wide_hybrid_quality_dispatch_v2 as dispatch


def setup(writer,monkeypatch,count=9):
    args,calls=dispatch_setup(writer,monkeypatch)
    monkeypatch.setattr(selection,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=64*1024**3)))
    for slot in range(count):dispatch.run_lr_once(slot=slot,**args)
    return args,calls


def test_full_stage_selection_preserves_weights_and_can_be_verified_readonly(writer,monkeypatch):
    args,calls=setup(writer,monkeypatch);root=writer.root/'evidence';before=selection._inventory(root)
    ref=selection.save_lr_selection(**args);saved=selection._document(root,ref)
    assert calls==list(range(9)) and saved['audit']['stage']['stage_audits_bound']
    assert saved['audit']['stage']['producer_provenance_bound']
    assert len(saved['audit']['stage']['manifest']['delete'])==12
    assert saved['audit']['rates']==selection.choose_rates(saved['audit']['prefix']['records'])
    assert selection._document(root,saved['selection_ref'])==saved['audit']['rates']
    assert not any(saved[k] for k in ('adopted','dispatch_authorized','deletion_authorized','stage_cleanup_complete','next_stage_authorized'))
    after=selection._inventory(root)
    assert all(after[n]==v for n,v in before.items())
    assert len([n for n in after if n.endswith('.pt')])==18
    checked=selection.verify_lr_selection(writer,args['plan_sha256'],ref)
    assert checked['rates']==saved['audit']['rates'] and checked['audit']['storage']['cleanup_boundaries_verified']==1
    assert selection._inventory(root)==after and not (root/'.retention/lr').exists()
    with pytest.raises(ValueError,match='cannot be retried'):selection.save_lr_selection(**args)
    assert selection._inventory(root)==after


@pytest.mark.parametrize('count',[0,8])
def test_incomplete_stage_cannot_save_choice(writer,monkeypatch,count):
    args,calls=setup(writer,monkeypatch,count);root=writer.root/'evidence';before=selection._inventory(root)
    with pytest.raises((ValueError,FileNotFoundError)):selection.save_lr_selection(**args)
    assert not (root/selection.CHOICE).exists()
    after=selection._inventory(root);assert all(after[n]==v for n,v in before.items())
    assert (root/(selection.BASE+'-failure.json')).exists()
    with pytest.raises(ValueError):selection.save_lr_selection(**args)


@pytest.mark.parametrize('bad',['winner_weight','numeric','dispatch','tensor','space'])
def test_all_audits_and_storage_must_pass_before_selection(writer,monkeypatch,bad):
    args,calls=setup(writer,monkeypatch);root=writer.root/'evidence'
    paths={'winner_weight':'quality/00/checkpoints/raw.pt','numeric':'meta/numerical.json',
        'dispatch':'meta/quality-dispatch-08.json','tensor':'quality/08/tensor-audit.json'}
    if bad=='space':monkeypatch.setattr(selection,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=8*1024**3)))
    else:(root/paths[bad]).write_text('{}')
    before=selection._inventory(root)
    with pytest.raises((ValueError,KeyError)):selection.save_lr_selection(**args)
    assert not (root/selection.CHOICE).exists()
    after=selection._inventory(root);assert all(after[n]==v for n,v in before.items())


@pytest.mark.parametrize('phase',['choice','stage','completion'])
def test_partial_selection_publication_is_preserved_and_blocks_retry(writer,monkeypatch,phase):
    args,calls=setup(writer,monkeypatch);root=writer.root/'evidence';publish=writer.publish_json
    def fail(*a,**kw):raise RuntimeError('synthetic '+phase)
    if phase=='stage':monkeypatch.setattr(selection,'bind_quality_stage',fail)
    else:
        def publication(name,value,**kw):
            target=selection.CHOICE if phase=='choice' else selection.BASE+'-complete.json'
            if name=='evidence/'+target:fail()
            return publish(name,value,**kw)
        monkeypatch.setattr(writer,'publish_json',publication)
    with pytest.raises(RuntimeError,match='synthetic'):selection.save_lr_selection(**args)
    assert (root/(selection.BASE+'-review.json')).exists()
    assert (root/selection.CHOICE).exists()==(phase!='choice')
    assert not (root/(selection.BASE+'-complete.json')).exists()
    before=selection._inventory(root)
    with pytest.raises(ValueError):selection.save_lr_selection(**args)
    assert selection._inventory(root)==before and len(list(root.rglob('*.pt')))==18


@pytest.mark.parametrize('bad',['choice','guard','stage','future'])
def test_readonly_verifier_rejects_changed_boundary(writer,monkeypatch,bad):
    args,calls=setup(writer,monkeypatch);root=writer.root/'evidence';ref=selection.save_lr_selection(**args)
    if bad=='choice':(root/selection.CHOICE).write_text('{}')
    elif bad=='guard':(writer.root/selection.GUARD).write_text('{}')
    elif bad=='stage':
        saved=selection._document(root,ref);saved['audit']['stage']['stage_audits_bound']=False
        (root/ref['path']).write_text(json.dumps(saved));ref['sha256']=selection.file_digest(root/ref['path'])
    else:writer.publish_json('evidence/quality/09/request.json',{})
    before=selection._inventory(root)
    with pytest.raises(ValueError):selection.verify_lr_selection(writer,args['plan_sha256'],ref)
    assert selection._inventory(root)==before
