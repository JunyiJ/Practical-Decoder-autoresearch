"""Full disposable deletion chain plus fresh appends, without dispatch credit."""
import json

import pytest

from src.research import tokenizer_wide_hybrid_rank_history_v2 as history
from src.research import tokenizer_wide_hybrid_confirmation_audit_v2 as audit
from src.research.protocol import digest,file_digest
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_rank_boundary_v2 import setup,boundary
import test_tokenizer_wide_hybrid_retention_audit as fixtures


@pytest.mark.parametrize('name',[
    'quality/15/request.json','quality/23/result.json','quality/18/process/worker.log',
    'quality/20/checkpoints/h200.pt','quality/23/checkpoints/final_001200_123.pt',
    'meta/quality-confirm-startup-15-attempt.json','meta/quality-dispatch-23.json',
    'meta/quality-readiness-19.json'])
def test_fresh_artifact_names(name):
    assert history.fresh_artifact(name)


@pytest.mark.parametrize('name',[
    'quality/14/request.json','quality/24/request.json','quality/015/request.json',
    'quality/15/unknown.json','quality/15/request.json.tmp','quality/15/worker-failure.json',
    'quality/15/checkpoints/raw.pt','quality/15/checkpoints/final_000020_123.pt',
    'quality/15/checkpoints/best_001200_123.pt','quality/15/checkpoints/final_001200_x.pt',
    'quality/15/../15/request.json','quality/15//request.json','/quality/15/request.json',
    'meta/quality-dispatch-24.json','meta/quality-dispatch-15-failure.json',
    'meta/quality-arbitrary.json','meta/rank-selection.json','.retention/rank/manifest.json',
    'inference/3511/result.json'])
def test_unknown_names_do_not_extend_history(name):
    assert not history.fresh_artifact(name)


@pytest.mark.parametrize('name',['control/quality-confirm-startup-15.json','control/quality-readiness-23.json',
    'control/quality-entry-18.json','control/quality-controller-child-21.json',
    'control/process-quality-15.json','control/process-quality-23.json'])
def test_fresh_controls(name):
    assert history.fresh_control(name)


@pytest.mark.parametrize('name',['control/quality-controller-child-14.json','control/quality-entry-24.json',
    'control/quality-entry-15.json.tmp','control/quality-entry-015.json','control/quality-failure-15.json',
    'control/quality-rank-startup-15.json','control/quality-controller-arbitrary-15.json',
    'control/process-quality-14.json','control/process-quality-24.json',
    'control/process-quality-015.json','control/process-quality-15.json.tmp',
    'control/process-pilots-15.json'])
def test_other_controls_rejected(name):
    assert not history.fresh_control(name)


def test_full_history_and_all_fresh_prefixes_keep_deleted_weights_gone(writer,monkeypatch):
    options=setup(writer,monkeypatch);root=writer.root/'evidence'
    complete=boundary.cleanup_rank_boundary(**options)
    args=dict(writer=writer,plan_sha256=options['plan_sha256'],completion_ref=complete)
    before=boundary._inventory(root)
    original=boundary.verify_rank_boundary(**args)
    checked=history.verify_rank_history(**args,current_inventory=before)
    for key in ('records','rates','selection','checkpoint_pairs','stages','selection_ref'):
        assert checked[key]==original[key]
    assert checked['rank_history_verified'] and len(checked['chain']['deleted_files'])==30
    for key in ('fresh_audits_bound','storage_verified','adopted','dispatch_authorized','next_stage_authorized'):
        assert checked[key] is False
    bound,requests=boundary.selection._frozen(writer,14,args['plan_sha256'],checked['projection_ref'],checked['rates'])
    plan=boundary._document(root,dict(path='meta/plan.json',sha256=args['plan_sha256']))
    for i,spec in enumerate(audit.rules.confirmation_specs(checked['selection']),15):
        matches=[r for r in bound['requests']['quality'][str(i)] if digest(r['spec'])==digest(spec)]
        assert len(matches)==1;requests.append(matches[0])
    bindings=dict(context=bound['context'],projection_ref=checked['projection_ref'],numerical_ref=plan['numerical_ref'],
        training_token_sha256=plan['training_token_sha256'],requests=requests)
    audit_args=dict(root=root,context=bound['context'],screen_records=checked['records'],rates=checked['rates'],
        selection=checked['selection'],screen_requests=requests[:15],numerical_ref=plan['numerical_ref'],
        rates_ref=checked['rates_ref'],selection_ref=checked['selection_ref'],projection_ref=checked['projection_ref'],
        training_token_sha256=plan['training_token_sha256'])
    original_write=fixtures.write
    for count in range(10):
        existing=set(boundary._inventory(root))
        def preserve(root,name,value):
            if name in existing:return dict(path=name,sha256=file_digest(root/name))
            return original_write(root,name,value)
        with monkeypatch.context() as patch:
            patch.setattr(fixtures,'write',preserve)
            fixtures.fixture(root,'rank',producer_v2=True,confirmation_prefix_count=count,bindings=bindings,
                raw_name={i:'final_001200_123456.pt' for i in range(9,24)})
        current=boundary._inventory(root)
        retained=history.verify_rank_history(**args,current_inventory=current)
        prefix=audit.bind_confirmation_prefix(**audit_args,confirmation_requests=requests[15:15+count],
            evidence={n:v['sha256'] for n,v in current.items() if not n.endswith('.pt')})
        assert prefix['records'][:15]==retained['records'] and len(prefix['records'])==15+count
        assert len(prefix['checkpoint_pairs'])==3+count and not prefix['adopted']
        assert all(pair==retained['checkpoint_pairs'][name] for name,pair in prefix['checkpoint_pairs'].items()
            if int(name.split('/')[1])<15)
        assert all(not (root/name).exists() for name in retained['chain']['deleted_files'])
        assert len(list(root.rglob('*.pt')))==6+2*count
        assert all(current[n]==v for n,v in before.items()) and boundary._inventory(root)==current
    # History admits declared bytes/control names, without interpreting them as
    # fresh scientific/transport completion or as a new storage gate.
    for slot in range(15,24):
        writer.publish_json(f'evidence/meta/quality-confirm-startup-{slot:02d}.json',{})
        writer.publish_json(f'control/quality-controller-child-{slot:02d}.json',{})
    current=boundary._inventory(root)
    assert history.verify_rank_history(**args,current_inventory=current)['rank_history_verified']
    with pytest.raises(ValueError):boundary.verify_rank_boundary(**args)
    with pytest.raises(ValueError):history.verify_rank_history(**args,current_inventory=before)
    # Fresh inventories cannot launder original evidence changes or resurrection.
    saved=boundary._document(root,complete);review=boundary._document(root,saved['review_ref'])
    view=boundary._view(writer,review)
    paths=[root/'.retention/rank/deletions.jsonl',root/'.retention/lr/deletions.jsonl',
        root/'.retention/pilot/execution-review.json',root/view['manifest']['retain'][0]['path'],
        root/view['manifest']['delete'][0]['path'],root/'meta/rank-selection.json',root/'meta/source.txt',
        writer.root/'control/quality-controller-child-14.json',writer.root/'control/quality-entry-24.json',
        root/'quality/24/request.json',root/'quality/15/unknown.json',root/'quality/15/worker-failure.json']
    for path in paths:
        old=path.read_bytes() if path.exists() else None
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'changed or unauthorized')
        changed=boundary._inventory(root)
        try:
            with pytest.raises((ValueError,FileNotFoundError,KeyError)):
                history.verify_rank_history(**args,current_inventory=changed)
            assert boundary._inventory(root)==changed
        finally:
            if old is None:
                path.unlink()
                if path.parent.name=='24':path.parent.rmdir()
            else:path.write_bytes(old)
    # Detect a concurrent append-byte mutation during revalidation.
    path=root/'meta/quality-confirm-startup-15.json';old=path.read_bytes();context=history._context;calls=[]
    def changed(*a,**kw):
        value=context(*a,**kw);calls.append(1)
        if len(calls)==1:path.write_bytes(b'{"changed":true}')
        return value
    with monkeypatch.context() as patch:
        patch.setattr(history,'_context',changed)
        with pytest.raises(ValueError):history.verify_rank_history(**args,current_inventory=current)
    path.write_bytes(old)
    assert boundary._inventory(root)==current
    census=boundary.snapshot(writer.root)
    (writer.root.parent/'confirmation-history-census.json').write_text(json.dumps(dict(groups=census['groups'],
        metadata_bytes=census['metadata_bytes'],file_count=len(census['files']),live_checkpoint_files=24,
        cumulative_deleted_files=30,fresh_parent_transport_present=False,real_panel_fit_proven=False),sort_keys=True))
