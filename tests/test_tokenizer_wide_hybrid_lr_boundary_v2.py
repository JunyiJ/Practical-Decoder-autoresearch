"""Real cleanup of disposable byte fixtures; never research checkpoints."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_lr_boundary_v2 as boundary
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_lr_selection_v2 import setup as selection_setup


def setup(writer,monkeypatch):
    args,calls=selection_setup(writer,monkeypatch)
    monkeypatch.setattr(boundary,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=64*1024**3)))
    selected=boundary.selection.save_lr_selection(**args)
    return dict(writer=writer,plan_sha256=args['plan_sha256'],selection_completion_ref=selected)


def test_complete_cleanup_keeps_selected_pairs_and_every_original_evidence_file(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence';before=boundary._inventory(root)
    result=boundary.cleanup_lr_boundary(**args);saved=boundary._document(root,result)
    review=boundary._document(root,saved['review_ref']);deleted={x['path'] for x in review['stage']['manifest']['delete']}
    after=boundary._inventory(root)
    assert len(deleted)==12 and set(before)-set(after)==deleted
    assert all(after[n]==v for n,v in before.items() if n not in deleted)
    assert len(list(root.rglob('*.pt')))==6 and saved['storage']['cleanup_boundaries_verified']==2
    checked=boundary.verify_lr_boundary(writer,args['plan_sha256'],result)
    assert checked['chain']['boundaries']==2 and len(checked['chain']['deleted_files'])==18
    assert len(checked['records'])==9 and len(checked['checkpoint_pairs'])==9
    assert not checked['dispatch_authorized'] and not checked['next_stage_authorized']
    assert boundary._inventory(root)==after and not (root/'quality/09').exists()
    with pytest.raises(ValueError,match='cannot be retried'):boundary.cleanup_lr_boundary(**args)
    assert boundary._inventory(root)==after
    # This is a different verifier, not an exception to the live-tensor gate.
    with pytest.raises(ValueError):boundary.selection.verify_lr_selection(writer,args['plan_sha256'],args['selection_completion_ref'])


@pytest.mark.parametrize('bad',['selection','source','winner','guard'])
def test_failed_full_precleanup_gate_removes_nothing(writer,monkeypatch,bad):
    args=setup(writer,monkeypatch);root=writer.root/'evidence'
    names={'selection':root/'meta/lr-selection.json','source':root/'meta/source.txt',
        'winner':root/'quality/00/checkpoints/raw.pt','guard':writer.root/'control/quality-entry-08.json'}
    names[bad].write_text('{}');before=boundary._inventory(root)
    with pytest.raises(ValueError):boundary.cleanup_lr_boundary(**args)
    assert not (root/'.retention/lr').exists() and len(list(root.rglob('*.pt')))==18
    after=boundary._inventory(root);assert all(after[n]==v for n,v in before.items())
    with pytest.raises(ValueError,match='cannot be retried'):boundary.cleanup_lr_boundary(**args)


@pytest.mark.parametrize('bad',['journal','recreated','winner','control'])
def test_postcleanup_verifier_cannot_excuse_missing_or_changed_protected_data(writer,monkeypatch,bad):
    args=setup(writer,monkeypatch);root=writer.root/'evidence';ref=boundary.cleanup_lr_boundary(**args)
    complete=boundary._document(root,ref);review=boundary._document(root,complete['review_ref'])
    if bad=='journal':(root/'.retention/lr/deletions.jsonl').write_text('')
    elif bad=='recreated':(root/review['stage']['manifest']['delete'][0]['path']).write_bytes(b'recreated')
    elif bad=='winner':(root/review['stage']['manifest']['retain'][0]['path']).unlink()
    else:(writer.root/'control/quality-controller-child-08.json').write_text('{}')
    before=boundary._inventory(root)
    with pytest.raises((ValueError,FileNotFoundError)):boundary.verify_lr_boundary(writer,args['plan_sha256'],ref)
    assert boundary._inventory(root)==before


def test_partial_unlink_failure_preserves_journal_and_is_never_retried(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence';before=boundary._inventory(root)
    original=Path.unlink;removed=[]
    def unlink(path,*a,**kw):
        if path.suffix=='.pt' and path.is_relative_to(root):
            if removed:raise OSError('synthetic second unlink failure')
            removed.append(str(path.relative_to(root)))
        return original(path,*a,**kw)
    with monkeypatch.context() as patch:
        patch.setattr(Path,'unlink',unlink)
        with pytest.raises(OSError,match='synthetic'):boundary.cleanup_lr_boundary(**args)
    assert len(removed)==1 and len(list(root.rglob('*.pt')))==17
    after=boundary._inventory(root)
    assert all(after[n]==v for n,v in before.items() if n not in removed)
    journal=[json.loads(line) for line in (root/'.retention/lr/deletions.jsonl').read_text().splitlines()]
    assert [row['event'] for row in journal]==['intent','deleted','intent']
    assert (root/'.retention/lr/failure.json').exists() and (root/(boundary.BASE+'-failure.json')).exists()
    assert not (root/(boundary.BASE+'-complete.json')).exists()
    with pytest.raises(ValueError,match='cannot be retried'):boundary.cleanup_lr_boundary(**args)
    assert boundary._inventory(root)==after
