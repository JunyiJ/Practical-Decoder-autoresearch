"""No agents or training: entrypoint/evidence import contracts only."""
import json

import pytest

from src.research.team.__main__ import import_report, main
from src.research.team.knowledge import KnowledgeBank


def test_standalone_team_report_preserves_consumed_dataset_hash(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps({"version": 1, "hypothesis": "fixture",
        "transfer_outcome": "supported", "records": [{"stage": "test"}],
        "contract": {"data": "a" * 64}}))
    bank = KnowledgeBank(tmp_path / "knowledge")
    entry = bank.get(import_report(bank, path))
    assert entry["scope"]["dataset_sha256"] == "a" * 64
    assert entry["scope"]["holdout_consumed"] is True
    assert entry["outcome"] == "reported"  # Import does not independently verify evidence.


def test_imported_smoke_is_never_empirical_support(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"name": "synthetic-smoke", "transfer_outcome": "beneficial", "trials": []}))
    bank = KnowledgeBank(tmp_path / "knowledge")
    assert bank.get(import_report(bank, path))["outcome"] == "inconclusive"


def test_cli_requires_explicit_agent_flag(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("src.research.team.__main__.advance", lambda path, **kw: calls.append(kw) or {"status": "prepared"})
    assert main(["advance", "--campaign", str(tmp_path)]) == 0
    assert calls == [{"run_agents": False, "allow_training": False}]


def test_knowledge_add_verifies_artifact_hash(tmp_path):
    artifact = tmp_path / "observation.txt"
    artifact.write_text("real fixture")
    entry = tmp_path / "entry.json"
    entry.write_text(json.dumps({"kind": "observation", "title": "fixture", "summary": "fixture", "outcome": "supported", "created_by": "test",
        "artifacts": [{"path": str(artifact), "sha256": "0" * 64}]}))
    with pytest.raises(SystemExit) as exc:
        main(["knowledge", "add", "--bank", str(tmp_path / "knowledge"), "--entry", str(entry)])
    assert exc.value.code == 2


def test_report_command_exports_history_without_agents_or_training(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("Reporting must not run agents or training")
    monkeypatch.setattr("src.research.team.__main__.advance", forbidden)
    monkeypatch.setattr("src.research.team.__main__.run_campaign", forbidden)
    (tmp_path / "campaign.json").write_text(json.dumps({"key": "fixture", "status": "prepared", "iterations": [], "agent_calls": []}))
    assert main(["report", "--campaign", str(tmp_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["trajectory_md"] == str(tmp_path / "trajectory.md")
    assert "No iterations" in (tmp_path / "trajectory.md").read_text()
