"""Final recovery gating on the complete doubled inference chain; no research run."""
import copy
import json

import pytest
from research.activation_inference_recovery_v2 import controller as e,final as f
from src.research.protocol import digest
from research.activation_inference_recovery_v2.test_execution import ready


@pytest.fixture
def completed(ready,monkeypatch):
    r=ready
    for _ in range(3):e.run_once(r.ctx)
    def analysis(source,imported):
        assert source is r.source and imported==r.imported
        return dict(source_import_sha256=digest(imported),formal_final_completed=False,
                    analysis=dict(strict_quality=False,adopted=False),fresh_provenance=[{'synthetic':True}]*6)
    monkeypatch.setattr(f,'analyze_imported_quality',analysis)
    return r


def test_closed_recovery_retains_original_failure_and_all_pairs(completed):
    r=completed;ref=f.complete(r.ctx);receipt=r.ctx.document(ref);decision=r.ctx.document(receipt['decision'])
    assert receipt['status']=='closed' and receipt['adopted'] is False
    assert decision['original_campaign_status']=='training_complete_inference_failed_no_retry'
    assert decision['original_campaign_retried'] is False and decision['holdout_scored'] is False
    assert len(decision['inference_acceptances'])==3 and decision['representative_recipe'] is None
    assert decision['retained_fresh_pairs']==6 and decision['retained_selected_screen_pairs']==2
    assert r.launches==[3612,3613]
    with pytest.raises(ValueError,match='cannot retry'):f.complete(r.ctx)


def test_missing_inference_is_permanent_final_failure(ready):
    with pytest.raises(ValueError,match='All three'):f.complete(ready.ctx)
    stem=ready.ctx.root/f.BASE
    assert not (ready.ctx.root/(f.BASE+'-complete.json')).exists()
    assert json.loads((ready.ctx.root/(f.BASE+'-failure.json')).read_text())['retry_allowed'] is False
    with pytest.raises(ValueError,match='cannot retry'):f.complete(ready.ctx)
    assert not ready.launches


@pytest.mark.parametrize('case',['summary','quality_binding','source_replay'])
def test_changed_final_evidence_cannot_close(completed,monkeypatch,case):
    r=completed
    if case=='summary':(r.ctx.root/e.SUMMARY).write_text('{}')
    elif case=='quality_binding':
        old=f.analyze_imported_quality
        def wrong(*args):
            result=old(*args);result['source_import_sha256']='0'*64;return result
        monkeypatch.setattr(f,'analyze_imported_quality',wrong)
    else:
        old=f.accepted_prefix;calls=[]
        def changed(*args):
            calls.append(True)
            if len(calls)>1:raise ValueError('source replay changed')
            return old(*args)
        monkeypatch.setattr(f,'accepted_prefix',changed)
    with pytest.raises(ValueError):f.complete(r.ctx)
    assert not (r.ctx.root/(f.BASE+'-complete.json')).exists()
    assert (r.ctx.root/(f.BASE+'-failure.json')).exists()


def test_fixed_representative_seed_if_frozen_quality_gate_passes(completed,monkeypatch):
    r=completed;old=f.analyze_imported_quality
    def eligible(*args):
        result=old(*args);result['analysis']['strict_quality']=True;return result
    monkeypatch.setattr(f,'analyze_imported_quality',eligible)
    receipt=r.ctx.document(f.complete(r.ctx));decision=r.ctx.document(receipt['decision'])
    assert decision['adopted'] is True and decision['adopted_family']=='gelu_exact'
    assert decision['representative_recipe']['seed']==3611
    assert decision['representative_recipe']['weights']==r.imported['pair_bindings']['3611']['bindings']['gelu_exact']['weights']


def test_failed_final_blocks_later_controller_step(ready):
    with pytest.raises(ValueError,match='All three'):f.complete(ready.ctx)
    with pytest.raises(ValueError,match='cannot retry or advance'):e.run_once(ready.ctx)
    assert not ready.launches
