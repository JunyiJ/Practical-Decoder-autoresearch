"""Pure artifact tests; no Codex process, optimizer or training execution."""
import copy

import pytest

from src.research.team.coordinator import reserve_audit
from src.research.team.knowledge import KnowledgeBank


def test_audit_reserved_before_dispatch_and_shared_across_campaigns(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("fixture")
    bank = KnowledgeBank(tmp_path / "knowledge")
    spec = {"audit_test": True, "baseline": {"config": {"data": {"path": str(corpus)}}}}
    first = copy.deepcopy(spec)
    reserve_audit(bank, {"key": "campaign-1"}, {"id": "iteration-1"}, first, tmp_path)
    assert first["audit_test"] is True
    assert bank.search(first["data_sha256"])[0]["scope"]["holdout_consumed"]
    second = copy.deepcopy(spec)
    iteration = {"id": "iteration-1"}
    reserve_audit(bank, {"key": "campaign-2"}, iteration, second, tmp_path)
    assert second["audit_test"] is False
    assert "previously" in iteration["audit_skipped_reason"]


def test_data_change_cannot_replace_reserved_identity(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("fixture")
    bank = KnowledgeBank(tmp_path / "knowledge")
    spec = {"audit_test": False, "data_sha256": "0" * 64,
            "baseline": {"config": {"data": {"path": str(corpus)}}}}
    with pytest.raises(ValueError, match="Dataset changed"):
        reserve_audit(bank, {"key": "fixture"}, {"id": "fixture"}, spec, tmp_path)
    assert bank.search() == []
