"""Append first readiness to one successful hash-pinned prepared synthetic prefix."""
import json
import os
from pathlib import Path

from src.research import tokenizer_wide_hybrid_inference_entry_v2 as entry
from tokenizer_wide_hybrid_reusable_fixture import reopen_prefix


def test_full_idle_replay_then_compact_child_reconstruction():
    binding=json.loads(Path(os.environ['TOKENIZER_PREPARED_PREFIX_BINDING']).read_text())
    with reopen_prefix(binding,allow_first_readiness=True) as writer:
        root=writer.root/'evidence'
        request=dict(path=entry.prep.REQUEST,sha256=entry.file_digest(root/entry.prep.REQUEST))
        ref=entry.prepare_first_readiness(writer,request)
        ready=entry._document(root,ref)
        proof=entry._immutable(writer,ready)
        entry.outputs._binding(writer,3511,proof)
        assert proof['ready_ref']==ref and proof['parent_pid']==os.getpid()
        assert (root/ref['path']).stat().st_size<1024
        assert len(list(root.rglob('*.pt')))==24
        # Public idle API remains strict once readiness has appended.
        import pytest
        with pytest.raises(ValueError,match='Only the shared inference'):
            entry.prep.verify_prepared_inference(writer,request)
