"""Historical cleanup plus synthetic rank bytes; no training or rank audit credit."""
import pytest

from src.research import tokenizer_wide_hybrid_lr_boundary_v2 as boundary
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_lr_boundary_v2 import setup


def cleaned(writer,monkeypatch):
    args=setup(writer,monkeypatch)
    ref=boundary.cleanup_lr_boundary(**args)
    return dict(writer=writer,plan_sha256=args['plan_sha256'],completion_ref=ref)


def test_declared_rank_bytes_preserve_history_without_rank_or_storage_credit(writer,monkeypatch):
    args=cleaned(writer,monkeypatch);root=writer.root/'evidence'
    before=boundary._inventory(root)
    original=boundary.verify_lr_boundary(**args)
    for slot in range(9,15):
        # Deliberately not valid scientific outputs. Only names and bytes are
        # bound by this history consumer; a future rank audit must reject them.
        for artifact in boundary.QUALITY_ARTIFACTS:
            path=root/f'quality/{slot:02d}'/artifact
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'{}')
        for name in ('h200.pt','final_001200_123456.pt'):
            path=root/f'quality/{slot:02d}/checkpoints'/name
            path.parent.mkdir(exist_ok=True);path.write_bytes(b'synthetic unreviewed rank weights')
        writer.publish_json(f'evidence/meta/quality-rank-startup-{slot:02d}.json',{})
    writer.publish_json('evidence/meta/rank-selection.json',{})
    current=boundary._inventory(root)
    def forbidden(*a,**kw):raise AssertionError('History must not grant a storage gate')
    monkeypatch.setattr(boundary,'bind_storage',forbidden)
    checked=boundary.verify_lr_history(**args,current_inventory=current)
    for key in ('records','rates','checkpoint_pairs','stages','selection_ref'):
        assert checked[key]==original[key]
    assert checked['lr_history_verified'] and len(checked['chain']['deleted_files'])==18
    assert all(checked[key] is False for key in ('rank_audits_bound','storage_verified',
        'adopted','dispatch_authorized','next_stage_authorized'))
    assert all(current[n]==v for n,v in before.items())
    assert boundary._inventory(root)==current
    with pytest.raises(ValueError,match='inventory changed'):boundary.verify_lr_boundary(**args)
    with pytest.raises(ValueError):boundary.verify_lr_history(**args,current_inventory=before)
    assert boundary._inventory(root)==current


@pytest.mark.parametrize('name',[
    'quality/09/request.json','quality/14/result.json','quality/09/process/worker.log',
    'quality/10/checkpoints/h200.pt','quality/14/checkpoints/final_001200_123.pt',
    'meta/quality-rank-startup-09-attempt.json','meta/quality-dispatch-14.json',
    'meta/quality-readiness-12.json','meta/rank-selection.json',
    'meta/rank-boundary-selection-complete.json'])
def test_rank_names_match_declared_producer_outputs(name):
    assert boundary._rank_append(name)


@pytest.mark.parametrize('name',[
    'quality/08/request.json','quality/15/request.json','quality/009/request.json',
    'quality/09/unknown.json','quality/09/request.json.tmp','quality/09/worker-failure.json',
    'quality/09/checkpoints/raw.pt','quality/09/checkpoints/final_000020_123.pt',
    'quality/09/checkpoints/best_001200_123.pt','quality/09/checkpoints/final_001200_x.pt',
    'quality/09/../09/request.json','quality/09//request.json','/quality/09/request.json',
    'meta/quality-dispatch-15.json','meta/quality-dispatch-09-failure.json',
    'meta/quality-arbitrary.json','meta/rank-selection.json.tmp',
    'meta/rank-boundary-cleanup-attempt.json','.retention/rank/manifest.json',
    'inference/3511/result.json','meta/lr-boundary-selection-complete.json'])
def test_only_precleanup_rank_namespace_is_allowed(name):
    assert not boundary._rank_append(name)


def test_corrupt_history_or_undeclared_append_cannot_be_laundered_with_fresh_inventory(writer,monkeypatch):
    args=cleaned(writer,monkeypatch);root=writer.root/'evidence'
    complete=boundary._document(root,args['completion_ref'])
    review=boundary._document(root,complete['review_ref'])
    writer.publish_json('evidence/quality/09/request.json',{})
    cases=[root/'.retention/lr/deletions.jsonl',root/'.retention/pilot/deletions.jsonl',
        root/review['stage']['manifest']['retain'][0]['path'],
        root/review['stage']['manifest']['delete'][0]['path'],
        root/'meta/lr-selection.json',root/'meta/source.txt',
        writer.root/'control/quality-controller-child-08.json',
        root/'quality/15/request.json',root/'quality/09/unknown.json',
        root/'.retention/rank/manifest.json',root/'quality/09/worker-failure.json']
    for path in cases:
        old=path.read_bytes() if path.exists() else None
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'changed or unauthorized')
        current=boundary._inventory(root)
        with pytest.raises((ValueError,FileNotFoundError)):
            boundary.verify_lr_history(**args,current_inventory=current)
        assert boundary._inventory(root)==current
        # Restore only this disposable corruption fixture to isolate next case.
        if old is None:
            path.unlink()
            if path.parent.name=='rank':path.parent.rmdir()
        else:path.write_bytes(old)
    current=boundary._inventory(root)
    assert boundary.verify_lr_history(**args,current_inventory=current)['lr_history_verified']


def test_rank_bytes_changed_during_history_verification_fail_without_writes(writer,monkeypatch):
    args=cleaned(writer,monkeypatch);root=writer.root/'evidence'
    path=writer.publish_json('evidence/quality/09/request.json',{})
    current=boundary._inventory(root);original=boundary._context;calls=[]
    def changed(*a,**kw):
        result=original(*a,**kw);calls.append(1)
        if len(calls)==1:path.write_bytes(b'{"changed":true}')
        return result
    monkeypatch.setattr(boundary,'_context',changed)
    with pytest.raises(ValueError):boundary.verify_lr_history(**args,current_inventory=current)
    after=boundary._inventory(root)
    assert set(after)==set(current)
    assert all(after[n]==v for n,v in current.items() if n!='quality/09/request.json')
