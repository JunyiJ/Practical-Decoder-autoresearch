"""Full serial synthetic rank selection/cleanup; only disposable byte fixtures."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_rank_boundary_v2 as boundary
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_rank_selection_v2 import setup as selection_setup


def setup(writer,monkeypatch):
    args,calls=selection_setup(writer,monkeypatch)
    monkeypatch.setattr(boundary,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=64*1024**3)))
    ref=boundary.selection.save_rank_selection(**args)
    return dict(writer=writer,plan_sha256=args['plan_sha256'],selection_completion_ref=ref)


def test_complete_rank_cleanup_and_cumulative_history_reject_tampering(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence';before=boundary._inventory(root)
    ref=boundary.cleanup_rank_boundary(**args);saved=boundary._document(root,ref)
    review=boundary._document(root,saved['review_ref']);view=boundary._view(writer,review)
    deleted={e['path'] for e in view['manifest']['delete']};retained={e['path'] for e in view['manifest']['retain']}
    after=boundary._inventory(root)
    assert len(deleted)==12 and set(before)-set(after)==deleted and len(retained)==6
    assert all(after[n]==v for n,v in before.items() if n not in deleted)
    assert len(list(root.rglob('*.pt')))==6 and not (root/'quality/15').exists()
    # Equal synthetic CEs select D/8; every former D/4 LR anchor is retired.
    assert all(v['divisor']==8 for v in view['summary']['selection'].values())
    assert all(not (root/f'quality/{i:02d}/checkpoints/raw.pt').exists() for i in range(3))
    checked=boundary.verify_rank_boundary(writer,args['plan_sha256'],ref)
    assert checked['chain']['boundaries']==3 and len(checked['chain']['deleted_files'])==30
    assert checked['storage']['cleanup_boundaries_verified']==3
    assert len(checked['checkpoint_pairs'])==18 and len(checked['records'])==15
    assert all(checked[k] is False for k in ('adopted','dispatch_authorized','next_stage_authorized'))
    assert boundary._inventory(root)==after
    for path in [root/'.retention/rank/deletions.jsonl',root/'.retention/lr/deletions.jsonl',
        root/'.retention/pilot/execution-review.json',root/next(iter(retained)),
        root/'meta/rank-selection.json',root/'quality/14/tensor-audit.json',
        writer.root/'control/quality-controller-child-14.json',root/'meta/source.txt']:
        original=path.read_bytes();path.write_bytes(b'changed')
        try:
            with pytest.raises((ValueError,KeyError)):boundary.verify_rank_boundary(writer,args['plan_sha256'],ref)
        finally:path.write_bytes(original)
    lost=root/next(iter(deleted));lost.write_bytes(b'recreated')
    try:
        with pytest.raises(ValueError):boundary.verify_rank_boundary(writer,args['plan_sha256'],ref)
    finally:lost.unlink()
    path=root/ref['path'];original=path.read_bytes()
    for key,value in [('chain_sha256','a'*64),('manifest_sha256','b'*64),('adopted',True),
        ('storage',dict(saved['storage'],cleanup_boundaries_verified=2))]:
        changed=copy.deepcopy(saved);changed[key]=value;path.write_text(json.dumps(changed))
        changed_ref=dict(path=ref['path'],sha256=boundary.file_digest(path))
        try:
            with pytest.raises(ValueError):boundary.verify_rank_boundary(writer,args['plan_sha256'],changed_ref)
        finally:path.write_bytes(original)
    assert boundary._inventory(root)==after
    with pytest.raises(ValueError,match='cannot be retried'):boundary.cleanup_rank_boundary(**args)
    with pytest.raises(ValueError):boundary.selection.verify_rank_selection(writer,args['plan_sha256'],args['selection_completion_ref'])
    # Record portable measurement in the disposable fixture's parent, outside
    # its audited campaign. The driver discovers this explicit unique filename.
    census=boundary.snapshot(writer.root)
    (writer.root.parent/'rank-boundary-census.json').write_text(json.dumps(dict(groups=census['groups'],
        metadata_bytes=census['metadata_bytes'],files={n:dict(size=(writer.root/n).stat().st_size,
        sha256=boundary.file_digest(writer.root/n)) for n in census['files']},real_panel_fit_proven=False),sort_keys=True))


@pytest.mark.parametrize('bad',['choice','control'])
def test_failed_precleanup_audit_removes_nothing(writer,monkeypatch,bad):
    args=setup(writer,monkeypatch);root=writer.root/'evidence'
    path=root/'meta/rank-selection.json' if bad=='choice' else writer.root/'control/quality-entry-14.json'
    path.write_bytes(b'changed');before=boundary._inventory(root)
    with pytest.raises(ValueError):boundary.cleanup_rank_boundary(**args)
    after=boundary._inventory(root)
    assert len(list(root.rglob('*.pt')))==18 and not (root/'.retention/rank').exists()
    assert all(after[n]==v for n,v in before.items())
    with pytest.raises(ValueError,match='cannot be retried'):boundary.cleanup_rank_boundary(**args)
    assert boundary._inventory(root)==after


def test_partial_rank_unlink_failure_keeps_journal_and_blocks_retry(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence';before=boundary._inventory(root)
    original=Path.unlink;removed=[]
    def unlink(path,*a,**kw):
        if path.suffix=='.pt' and path.is_relative_to(root):
            if removed:raise OSError('synthetic second rank unlink failure')
            removed.append(str(path.relative_to(root)))
        return original(path,*a,**kw)
    with monkeypatch.context() as patch:
        patch.setattr(Path,'unlink',unlink)
        with pytest.raises(OSError,match='synthetic'):boundary.cleanup_rank_boundary(**args)
    after=boundary._inventory(root)
    assert len(removed)==1 and len(list(root.rglob('*.pt')))==17
    assert all(after[n]==v for n,v in before.items() if n not in removed)
    rows=[json.loads(line) for line in (root/'.retention/rank/deletions.jsonl').read_text().splitlines()]
    assert [r['event'] for r in rows]==['intent','deleted','intent']
    assert (root/'.retention/rank/failure.json').exists() and (root/(boundary.BASE+'-failure.json')).exists()
    assert not (root/(boundary.BASE+'-complete.json')).exists()
    with pytest.raises(ValueError,match='cannot be retried'):boundary.cleanup_rank_boundary(**args)
    assert boundary._inventory(root)==after


def test_completion_publication_failure_preserves_deletion_receipts(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence';publish=writer.publish_json
    def fail(name,value,**kw):
        if name=='evidence/'+boundary.BASE+'-complete.json':raise RuntimeError('synthetic completion failure')
        return publish(name,value,**kw)
    monkeypatch.setattr(writer,'publish_json',fail)
    with pytest.raises(RuntimeError,match='synthetic'):boundary.cleanup_rank_boundary(**args)
    assert (root/'.retention/rank/execution-review.json').exists() and len(list(root.rglob('*.pt')))==6
    assert (root/(boundary.BASE+'-failure.json')).exists()
    before=boundary._inventory(root)
    with pytest.raises(ValueError,match='cannot be retried'):boundary.cleanup_rank_boundary(**args)
    assert boundary._inventory(root)==before
