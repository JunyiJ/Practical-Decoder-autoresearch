"""Opt-in immutable prefix replay under a new process and fresh lock FDs."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_inference_preparation_v2 as prep
from src.research.tokenizer_wide_hybrid_rank_history_v2 import verify_rank_history
from src.research.tokenizer_wide_hybrid_pilot_boundary_v2 import verify_pilot_history
from tokenizer_wide_hybrid_reusable_fixture import reopen_prefix


def binding():
    name=os.environ.get('TOKENIZER_PREPARED_PREFIX_BINDING')
    if not name:pytest.skip('Explicit completed synthetic prefix binding required')
    return json.loads(Path(name).read_text())


def reference(writer):
    return dict(path=prep.REQUEST,sha256=prep.file_digest(writer.root/'evidence'/prep.REQUEST))


def test_finished_preparation_replays_independently_and_restores_environment():
    original=prep.gate.bind_campaign
    with reopen_prefix(binding()) as writer:
        ref=reference(writer);request=prep._document(writer.root/'evidence',ref)
        assert request['parent_pid']!=os.getpid()
        result=prep.verify_prepared_inference(writer,ref)
        assert result['preparation_verified'] and len(result['history']['records'])==24
        assert len(result['history']['checkpoint_pairs'])==12 and set(result['triples'])=={3511,3512,3513}
        assert all(result[k] is False for k in ('adopted','dispatch_authorized','deletion_authorized','next_stage_authorized'))
        inventory=prep.gate.history.acceptance._inventory(writer.root/'evidence')
        with pytest.raises(ValueError,match='Undeclared fresh evidence'):
            verify_rank_history(writer,request['plan_sha256'],request['rank_completion_ref'],current_inventory=inventory)
        saved=prep._document(writer.root/'evidence',request['completion_ref'])
        with pytest.raises(ValueError,match='outside quality'):
            verify_pilot_history(writer,request['plan_sha256'],saved['pilot_completion_ref'],current_inventory=inventory,stages=result['history']['stages'])
    assert prep.gate.bind_campaign is original


@pytest.mark.parametrize('bad',['stop','space'])
def test_late_changes_after_full_history_replay_are_rejected(monkeypatch,bad):
    with reopen_prefix(binding()) as writer,monkeypatch.context() as patch:
        original=prep.gate._contract;called=[]
        def late(config):
            value=original(config)
            if not called:
                called.append(True)
                if bad=='stop':(writer.root/'STOP').touch()
                else:patch.setattr(prep.gate,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=8*1024**3)))
            return value
        patch.setattr(prep.gate,'_contract',late)
        try:
            with pytest.raises(ValueError,match='STOP|census'):prep.verify_prepared_inference(writer,reference(writer))
            assert called==[True]
        finally:
            if bad=='stop' and (writer.root/'STOP').exists():(writer.root/'STOP').unlink()
