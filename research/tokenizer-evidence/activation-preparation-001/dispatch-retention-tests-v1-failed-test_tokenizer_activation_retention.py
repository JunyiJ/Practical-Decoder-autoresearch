"""Real deletion/journal/preservation mechanics on disposable tiny fixture files.

Freeze and process gates are explicit doubles; accepted audit documents contain
synthetic verdicts. No research checkpoints or prior protected files are touched.
"""
import json
from pathlib import Path

import pytest

from src.research import tokenizer_activation_retention as r
from src.research import tokenizer_activation_entry as e
from src.research.tokenizer_activation_session import reference
from tests.test_tokenizer_activation_session import ledger,outputs


def accepted(ledger):
    ref=outputs(ledger);review=ledger.document(ref);stem=str(Path(ref['path']).parent)
    for name,old in list(review['audits'].items()):
        doc=ledger.document(old);doc['reviewer']='activation_independent_v1'
        (ledger.root/old['path']).write_text(json.dumps(doc));review['audits'][name]=reference(ledger.root,old['path'])
    review['outputs']=[reference(ledger.root,str(p.relative_to(ledger.root)))
                       for p in sorted((ledger.root/stem).rglob('*')) if p.is_file() and p.name!='acceptance.json']
    (ledger.root/ref['path']).write_text(json.dumps(review));ref=reference(ledger.root,ref['path'])
    ledger.record('accept',ref)


@pytest.fixture
def gated(ledger,monkeypatch):
    monkeypatch.setattr(e,'bind',lambda *a,**k:{'synthetic':True})
    monkeypatch.setattr(e,'idle',lambda *a:{'checked':True})
    monkeypatch.setattr(e,'prefix',lambda *a:[])
    return ledger


def pilots(ledger):
    accepted(ledger);accepted(ledger)


def test_real_two_boundaries_keep_selected_and_every_fresh_pair(gated):
    pilots(gated);pilot=r.execute_boundary(gated)
    assert r.verify_execution(gated,gated.inspect(),'pilot')==pilot
    for _ in range(6):accepted(gated)
    gated.save_selection();screen=r.execute_boundary(gated)
    assert r.verify_execution(gated,gated.inspect(),'screen')==screen
    for _ in range(6):accepted(gated)
    state=gated.inspect();r.verify_boundaries(gated,state)
    assert state['next']['kind']=='review_final' and len(state['deleted'])==12
    retained={ref['path']:ref for ref in state['pins'].values() if ref['path'].endswith('.pt') and ref['path'] not in state['deleted']}
    assert len(retained)==16
    for ref in retained.values():gated.verify(ref)
    with pytest.raises(ValueError,match='Complete audited stage'):r.execute_boundary(gated)


def test_no_cleanup_before_complete_stage_or_independent_review(gated):
    accepted(gated)
    with pytest.raises(ValueError):r.execute_boundary(gated)
    assert not (gated.root/'evidence/.retention/pilot').exists()
    # Fixture lacking the independent reviewer cannot become deletion authority.
    ref=outputs(gated);gated.record('accept',ref)
    with pytest.raises(ValueError,match='Independent scientific'):r.execute_boundary(gated)
    assert len(list(gated.root.rglob('*.pt')))==4


def test_partial_deletion_failure_is_journaled_and_never_retried(gated,monkeypatch):
    pilots(gated);original=Path.unlink;removed=[]
    def unlink(path,*args,**kwargs):
        if path.parent.name=='checkpoints' and path.suffix=='.pt':
            if len(removed)==1:raise OSError('injected second unlink failure')
            removed.append(str(path))
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',unlink)
    with pytest.raises(OSError,match='injected second'):r.execute_boundary(gated)
    base=gated.root/'evidence/.retention/pilot'
    failure=json.loads((base/'failure.json').read_text())
    assert failure['retry_allowed'] is False and len(failure['confirmed_deleted'])==1
    assert len(list(gated.root.rglob('*.pt')))==3
    assert not (base/'activation-complete.json').exists()
    with pytest.raises((ValueError,FileNotFoundError)):r.execute_boundary(gated)
    assert len(removed)==1


def test_completed_receipt_cannot_hide_journal_mutation(gated):
    pilots(gated);r.execute_boundary(gated);state=gated.inspect()
    path=gated.root/'evidence/.retention/pilot/deletions.jsonl'
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError):r.verify_execution(gated,state,'pilot')
    with pytest.raises(ValueError):gated.reserve()


def test_retention_refuses_symlink_or_changed_weight(gated):
    pilots(gated);state=gated.inspect();ref=state['pairs'][0][0]
    path=gated.root/ref['path'];path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError):r.execute_boundary(gated)
    assert not (gated.root/'evidence/.retention/pilot').exists()
