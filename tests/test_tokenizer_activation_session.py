"""New disposable filesystem transport fixtures, not model/quality experiments.

Tiny synthetic tensors exercise unchanged bounded writers with real locks and
real free-space checks. Audit verdicts are fixture data: no scientific readiness
or actual activation tensor correctness is inferred from these ledger tests.
"""
import fcntl
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.research import tokenizer_activation_session as session
from src.research import tokenizer_wide_hybrid_cleanup as cleanup
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_training_outputs import TrainingOutputs
from src.research.tokenizer_wide_hybrid_metadata import snapshot
from src.research.protocol import file_digest


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    root = (tmp_path/'activation').resolve(); root.mkdir()
    global_lock = tmp_path/'global.lock'; global_lock.touch(); (root/'.lock').touch()
    monkeypatch.setattr(cleanup, 'GLOBAL_LOCK', global_lock)
    with global_lock.open('rb') as g, (root/'.lock').open('rb') as c:
        fcntl.flock(g, fcntl.LOCK_EX | fcntl.LOCK_NB); fcntl.flock(c, fcntl.LOCK_EX | fcntl.LOCK_NB)
        writer = MetadataWriter(root, (g.fileno(), c.fileno()))
        writer.publish_json('evidence/meta/activation-plan.json', dict(purpose='transport_fixture', frozen=False))
        yield session.Session(writer, session.reference(root, 'evidence/meta/activation-plan.json'))


def publish(ledger, name, value):
    ledger.writer.publish_json(name, value)
    return session.reference(ledger.root, name)


def outputs(ledger, *, ce=5.):
    ref = ledger.reserve(); request = ledger.document(ref)
    stem = str(Path(ref['path']).parent); cfg = request['config']; trial = ledger.root/stem
    # Use real torch.save/read and the production output sink. Payload shape is
    # deliberately tiny; this is a transport test, not a model tensor audit.
    raw = trial/f'checkpoints/final_{request["spec"]["steps"]:06d}_1.pt'
    with TrainingOutputs(ledger.writer, Path(stem).parts[1], int(Path(stem).parts[2]), 1024**2, cfg) as sink:
        sink.validate_config(OmegaConf.create(cfg))
        sink.metric(trial/'metrics.jsonl', dict(step=0, loss=5.))
        sink.checkpoint(raw, dict(cfg=cfg, step=request['spec']['steps'],
            model_state={'synthetic': torch.arange(4.)}, optimizer_state={}, training_state={}))
        sink.summary(trial/'training-summary.json', dict(completed_steps=request['spec']['steps'],
            final_checkpoint=str(raw), metrics=[dict(step=0, loss=5.)]))
        sink.checkpoint(raw.parent/'h200.pt', dict(cfg=cfg, model_state={'synthetic': torch.arange(4.)},
            parent_checkpoint=dict(path=str(raw), sha256=file_digest(raw)), averaged_weights=dict(half_life=200)))
    assert torch.equal(torch.load(raw, weights_only=True)['model_state']['synthetic'], torch.arange(4.))
    weights = {name: session.reference(ledger.root, str(Path(value['path']).relative_to(ledger.root)))
               for name, value in (('raw', sink.raw), ('h200', sink.export))}
    row = dict(request['spec'], status='completed')
    if request['spec']['stage'] != 'pilot': row['h200_ce'] = ce
    measurement = publish(ledger, stem+'/measurements.json', dict(synthetic_fixture=True))
    result = publish(ledger, stem+'/result.json', dict(kind=session.KIND, request=ref, record=row, holdout_consumed=False,
        status='completed', request_sha256=ref['sha256'], measurements=measurement))
    audits = {name: publish(ledger, stem+'/'+name+'-audit.json',
        dict(status='passed', request=ref, result=result, weights=weights, synthetic_fixture=True)) for name in session.AUDITS}
    review = publish(ledger, stem+'/acceptance.json', dict(kind='activation_acceptance_v1', status='accepted',
        request=ref, result=result, audits=audits, weights=weights,
        outputs=[session.reference(ledger.root, str(p.relative_to(ledger.root)))
                 for p in sorted(trial.rglob('*')) if p.is_file()]))
    return review


def accept(ledger, **kw):
    ref = outputs(ledger, **kw); ledger.record('accept', ref); return ref


def boundary(ledger, which):
    state = ledger.inspect(); deleted = ledger._eligible(state, which)
    # Explicit, exact allowlist on disposable tiny fixture weights only.
    before = {name: session.reference(ledger.root, name) for name in snapshot(ledger.root)['files']
              if name.startswith('evidence/')}
    for name, ref in deleted.items():
        assert before[name] == ref; ledger.verify(ref).unlink()
    for name, ref in before.items():
        if name not in deleted: ledger.verify(ref)
    return publish(ledger, f'evidence/.retention/{"pilot" if which == "pilot" else "lr"}/activation-complete.json',
        dict(kind='activation_retention_v1', status='completed', boundary=which, reviews=state['reviews'],
             selection=state['selection_ref'], deleted=list(deleted.values()),
             preserved=[v for k, v in before.items() if k not in deleted]))


def through_pilots(ledger):
    accept(ledger); accept(ledger)
    assert ledger.inspect()['next']['kind'] == 'review_pilot_boundary'
    ledger.record('boundary', boundary(ledger, 'pilot'))


def through_screen(ledger):
    through_pilots(ledger)
    for _ in range(6): accept(ledger)
    ledger.save_selection()


def test_complete_14_trajectory_filesystem_and_retention_chain(ledger):
    through_screen(ledger)
    assert ledger.inspect()['selection'] == dict(silu=.000625, gelu_exact=.000625)
    ledger.record('boundary', boundary(ledger, 'screen'))
    for _ in range(6): accept(ledger)
    state = ledger.inspect()
    assert state['next'] == dict(kind='review_final', adopted=False)
    assert state['dispatch_authorized'] is False
    assert len(state['pilots']) == 2 and len(state['records']) == 12
    assert len(state['deleted']) == 12
    assert len(list(ledger.root.rglob('*.pt'))) == 16  # two selected + all six fresh pairs
    assert len(state['events']) == 31
    again = session.Session(ledger.writer, ledger.plan).inspect()
    assert again == state
    with pytest.raises(ValueError): ledger.reserve()
    with pytest.raises(ValueError): ledger.record('boundary', session.reference(ledger.root,
        'evidence/.retention/lr/activation-complete.json'))


def test_pending_request_is_exclusive_and_stop_blocks_progress(ledger):
    ref = ledger.reserve(); assert ledger.inspect()['next']['kind'] == 'pending'
    with pytest.raises(ValueError): ledger.reserve()
    failure = publish(ledger, 'evidence/pilots/00/session-failure.json', dict(request=ref, retry_allowed=False))
    ledger.writer.publish_json('STOP', dict(reason='fixture stop'))
    with pytest.raises(ValueError, match='STOP'): ledger.record('failed', failure)
    assert ledger.inspect()['pending']['ref'] == ref


def test_failure_latches_and_preserves_partial_attempt(ledger):
    ref = ledger.reserve()
    failure = publish(ledger, 'evidence/pilots/00/session-failure.json', dict(request=ref, retry_allowed=False))
    ledger.record('failed', failure)
    assert ledger.inspect()['next']['kind'] == 'failed'
    with pytest.raises(ValueError): ledger.reserve()
    with pytest.raises(ValueError): ledger.record('failed', failure)
    assert ledger.verify(failure).is_file() and ledger.verify(ref).is_file()


@pytest.mark.parametrize('target', ['request', 'result', 'audit', 'weights', 'plan', 'event'])
def test_mutation_rejected_after_acceptance(ledger, target):
    review_ref = accept(ledger); review = ledger.document(review_ref)
    ref = {'request': review['request'], 'result': review['result'], 'audit': review['audits']['tensors'],
           'weights': review['weights']['raw'], 'plan': ledger.plan,
           'event': ledger.inspect()['events'][0]}[target]
    path = ledger.root/ref['path']; path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError): ledger.inspect()


@pytest.mark.parametrize('case', ['missing_audit', 'wrong_request', 'wrong_pair', 'holdout', 'wrong_score', 'wrong_spec'])
def test_invalid_acceptance_does_not_append(ledger, case):
    ref = outputs(ledger); review = ledger.document(ref)
    if case == 'missing_audit': del review['audits']['tensors']
    if case == 'wrong_request': review['request'] = ledger.plan
    if case == 'wrong_pair': review['weights']['h200'] = review['weights']['raw']
    if case in ('holdout', 'wrong_score', 'wrong_spec'):
        result = ledger.document(review['result'])
        if case == 'holdout': result['holdout_consumed'] = True
        if case == 'wrong_score': result['record']['h200_ce'] = 4.
        if case == 'wrong_spec': result['record']['seed'] = 3611
        path = ledger.root/review['result']['path']; path.write_text(json.dumps(result))
        review['result'] = session.reference(ledger.root, review['result']['path'])
    path = ledger.root/ref['path']; path.write_text(json.dumps(review))
    bad = session.reference(ledger.root, ref['path'])
    with pytest.raises(ValueError): ledger.record('accept', bad)
    assert len(ledger.inspect()['events']) == 1


def test_incomplete_stage_and_selection_tamper_block_fresh(ledger):
    through_pilots(ledger); accept(ledger)
    with pytest.raises(ValueError): ledger.save_selection()
    for _ in range(5): accept(ledger)
    ref = ledger.save_selection(); value = ledger.document(ref)
    value['choices']['silu'] = .0025
    (ledger.root/ref['path']).write_text(json.dumps(value))
    with pytest.raises(ValueError): ledger.reserve()


def test_missing_weight_needs_complete_exact_boundary_receipt(ledger):
    accept(ledger); accept(ledger)
    ref = boundary(ledger, 'pilot')
    with pytest.raises(FileNotFoundError): ledger.inspect()
    value = ledger.document(ref); value['deleted'].pop()
    (ledger.root/ref['path']).write_text(json.dumps(value))
    with pytest.raises(ValueError): ledger.record('boundary', session.reference(ledger.root, ref['path']))
    assert len(ledger._events()) == 4


def test_protected_selected_pair_cannot_be_in_cleanup_allowlist(ledger):
    through_screen(ledger)
    state = ledger.inspect(); ref = boundary(ledger, 'screen'); value = ledger.document(ref)
    value['deleted'].append(state['pairs'][2][0])
    (ledger.root/ref['path']).write_text(json.dumps(value))
    with pytest.raises(ValueError): ledger.record('boundary', session.reference(ledger.root, ref['path']))
    assert (ledger.root/state['pairs'][2][0]['path']).exists()


def test_orphan_request_and_temporary_ledger_are_not_retried(ledger):
    ref = ledger.reserve(); event = ledger.root/'evidence/meta/session/0000.json'
    event.rename(event.with_suffix('.json.tmp'))
    with pytest.raises(ValueError, match='Partial'): ledger.inspect()
    event.with_suffix('.json.tmp').unlink()
    with pytest.raises(ValueError, match='Unjournaled'): ledger.inspect()
    assert ledger.verify(ref).is_file()


def test_real_bounded_writer_capacity_and_exclusive_checkpoint_failure(ledger):
    from src.research.tokenizer_wide_hybrid_checkpoint_writer import CheckpointWriter
    writer = CheckpointWriter(ledger.writer, 'pilots', 0, 64)
    stream = writer.open('fixture.pt')
    stream.write(b'x'*64)
    with pytest.raises(ValueError, match='allowance'): stream.write(b'x')
    assert (ledger.root/'evidence/pilots/00/checkpoints/fixture.pt.tmp').stat().st_size == 64
    with pytest.raises(FileExistsError): writer.open('fixture.pt')
    assert not (ledger.root/'evidence/pilots/00/checkpoints/fixture.pt').exists()


def test_stop_before_first_request_and_closed_locks(ledger):
    ledger.writer.publish_json('STOP', {'reason': 'fixture'})
    with pytest.raises(ValueError, match='STOP'): ledger.reserve()
    assert not (ledger.root/'evidence/pilots/00/request.json').exists()
    with pytest.raises(ValueError): MetadataWriter(ledger.root, (0, 1))


def test_unexpected_worker_failure_prevents_acceptance(ledger):
    review = outputs(ledger)
    publish(ledger, 'evidence/pilots/00/worker-failure.json', {'status': 'failed'})
    with pytest.raises(ValueError, match='Failed or partial'): ledger.record('accept', review)
    assert len(ledger.inspect()['events']) == 1


@pytest.mark.parametrize('name', ['metrics.jsonl', 'training-summary.json', 'training-outputs.json'])
def test_all_producer_evidence_is_pinned_before_boundary(ledger, name):
    accept(ledger)
    path = ledger.root/'evidence/pilots/00'/name
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='hash/size'): ledger.reserve()


def test_added_unreviewed_output_blocks_acceptance(ledger):
    ref = outputs(ledger)
    publish(ledger, 'evidence/pilots/00/unreviewed.json', {'extra': True})
    with pytest.raises(ValueError, match='Unreviewed'): ledger.record('accept', ref)


def test_metadata_symlink_is_rejected(ledger):
    ref = ledger.reserve()
    (ledger.root/'evidence/pilots/00/alias.json').symlink_to(ledger.root/ref['path'])
    with pytest.raises(ValueError, match='symlink'): ledger.inspect()
