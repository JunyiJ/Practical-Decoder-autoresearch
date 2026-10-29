"""Read-only complete historical panel replay under the new controller code."""
import json
import os
from pathlib import Path

import pytest

from src.research import tokenizer_wide_hybrid_inference_acceptance_v2 as acceptance
from tokenizer_wide_hybrid_reusable_fixture import reopen_prefix
from test_tokenizer_wide_hybrid_inference_acceptance_reused_v2 import synthetic_token_adapter


def test_existing_completed_panel_keeps_full_history_and_exact_bytes(monkeypatch):
    name=os.environ.get('TOKENIZER_COMPLETED_PREFIX_BINDING')
    if not name:pytest.skip('Explicit completed synthetic inference prefix binding required')
    binding=json.loads(Path(name).read_text());calls=[];original=acceptance.prep.gate.history._verify
    def history(*a,**kw):
        value=original(*a,**kw);calls.append(kw.get('inference_request_ref'));return value
    with reopen_prefix(binding) as writer,monkeypatch.context() as patch:
        synthetic_token_adapter(patch)
        patch.setattr(acceptance.prep.gate.history,'_verify',history)
        ref=acceptance._ref(writer.root/'evidence','inference/3513/review.json')
        result=acceptance.verify_boundary(writer,ref)
        assert calls==[ref] and result['quality_history_reaudited']
        assert len(result['bound']['history']['records'])==24
        assert len(result['bound']['history']['checkpoint_pairs'])==12
        assert len(result['scope']['previous_refs'])==3
        assert not result['dispatch_authorized'] and not result['adopted']
        assert len(list((writer.root/'evidence').rglob('*.pt')))==24
