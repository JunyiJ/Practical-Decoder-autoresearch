from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import stat

import pytest

from src.research.team import knowledge
from src.research.team.knowledge import KnowledgeBank


def entry(**updates):
    return {"kind": "idea", "title": "Width-aware learning rate", "summary": "Test LR scaling across widths.",
            "outcome": "untried", "created_by": "idea-generator", **updates}


def artifact():
    return {"path": "research_runs/trial/report.json", "sha256": "a" * 64}


def _write_record_worker(args):
    path, index = args
    bank = KnowledgeBank(path)
    shared = bank.add(entry())
    unique = bank.add(entry(title=f"Idea {index}", related_ids=[shared]))
    return shared, unique


def test_content_addressing_defaults_and_retry_do_not_change_timestamp(tmp_path):
    bank = KnowledgeBank(tmp_path / "knowledge")
    original = entry(scope={"widths": [64, 128]})
    before = deepcopy(original)
    entry_id = bank.add(original)
    first = bank.get(entry_id)
    assert original == before
    assert first["created_at"].endswith("Z")
    assert first["details"] == {} and first["sources"] == []
    assert bank.add({**original, "details": {}, "sources": [], "artifacts": [], "related_ids": []}) == entry_id
    assert bank.get(entry_id) == first
    original["scope"]["widths"].append(256)
    assert bank.get(entry_id)["scope"]["widths"] == [64, 128]
    assert len(bank.search()) == 1
    path = bank.path / "entries" / f"{entry_id}.json"
    persisted = json.loads(path.read_text())
    assert set(persisted) == {"id", "created_at", "payload"}
    assert persisted["id"] == entry_id and persisted["created_at"] == first["created_at"]
    assert persisted["payload"] == {key: value for key, value in first.items() if key not in {"id", "created_at"}}
    assert KnowledgeBank(bank.path).get(entry_id) == first


def test_literature_is_reported_not_local_support(tmp_path):
    bank = KnowledgeBank(tmp_path / "knowledge")
    with pytest.raises(ValueError, match="source URL"):
        bank.add(entry(kind="literature", outcome="reported"))
    source = {"title": "Research paper", "url": "https://example.org/paper", "locator": "Section 3"}
    entry_id = bank.add(entry(kind="literature", outcome="reported", sources=[source]))
    assert bank.get(entry_id)["sources"] == [source]
    with pytest.raises(ValueError, match="reported or untried"):
        bank.add(entry(kind="literature", outcome="supported", sources=[source], artifacts=[artifact()]))
    with pytest.raises(ValueError, match="HTTP or HTTPS"):
        bank.add(entry(kind="literature", sources=[{**source, "url": "file:///tmp/paper"}]))


@pytest.mark.parametrize("outcome", ["supported", "contradicted"])
def test_empirical_claims_require_provenance(tmp_path, outcome):
    bank = KnowledgeBank(tmp_path / "knowledge")
    with pytest.raises(ValueError, match="artifact"):
        bank.add(entry(kind="evaluation", outcome=outcome))
    with pytest.raises(ValueError, match="SHA-256"):
        bank.add(entry(kind="evaluation", outcome=outcome, artifacts=[{**artifact(), "sha256": "not-a-hash"}]))
    entry_id = bank.add(entry(kind="evaluation", outcome=outcome, artifacts=[artifact()],
                              scope={"optimizer": "adamw", "dataset": "tiny", "width": 128}))
    assert bank.get(entry_id)["outcome"] == outcome


def test_conflicting_evidence_appends_and_related_references_must_exist(tmp_path):
    bank = KnowledgeBank(tmp_path / "knowledge")
    with pytest.raises(ValueError, match="Related entry does not exist"):
        bank.add(entry(related_ids=["missing"]))
    assert bank.search() == []
    hypothesis = bank.add(entry())
    positive = bank.add(entry(kind="evaluation", outcome="supported", related_ids=[hypothesis],
                              scope={"width": 64}, artifacts=[artifact()]))
    negative = bank.add(entry(kind="evaluation", outcome="contradicted", related_ids=[hypothesis, positive],
                              scope={"width": 256}, artifacts=[artifact()]))
    assert len(bank.search()) == 3
    assert bank.get(positive)["outcome"] == "supported"
    assert bank.get(negative)["related_ids"] == sorted([hypothesis, positive])
    with pytest.raises(KeyError):
        bank.get("missing")


def test_queries_search_scope_and_details_with_literal_wildcards(tmp_path):
    bank = KnowledgeBank(tmp_path / "knowledge")
    first = bank.add(entry(scope={"optimizer": "Muon", "width": 128}, details={"change": "clip_2%"}))
    bank.add(entry(kind="observation", summary="No result yet", scope={"optimizer": "AdamW", "width": 256}))
    assert [result["id"] for result in bank.search("MUON width 128")] == [first]
    assert [result["id"] for result in bank.search("clip_2%")] == [first]
    assert bank.search("%' OR 1=1 --") == []
    assert len(bank.search(kind="observation")) == 1
    assert len(bank.search(limit=1)) == 1
    with pytest.raises(ValueError):
        bank.search(limit=-1)


@pytest.mark.parametrize("executor_kind", ["threads", "processes"])
def test_concurrent_workers_deduplicate_and_preserve_writes(tmp_path, executor_kind):
    path = tmp_path / "knowledge"
    KnowledgeBank(path)
    executor = (ThreadPoolExecutor(max_workers=8) if executor_kind == "threads" else
                ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")))
    with executor:
        results = list(executor.map(_write_record_worker, [(str(path), index) for index in range(24)]))
    assert len({shared for shared, _ in results}) == 1
    assert len({unique for _, unique in results}) == 24
    bank = KnowledgeBank(path)
    assert len(bank.search(limit=100)) == 25
    assert len(list((path / "entries").glob("*.json"))) == 25
    assert not list((path / "entries").glob("*.tmp"))
    assert (path / ".lock").is_file()


def test_records_are_read_only_and_export_preserves_all_evidence(tmp_path):
    path = tmp_path / "knowledge"
    bank = KnowledgeBank(path)
    entry_id = bank.add(entry(summary="Text with ``` fence and an instruction-looking string."))
    record_path = path / "entries" / f"{entry_id}.json"
    assert record_path.stat().st_mode & 0o222 == 0
    assert not hasattr(bank, "delete") and not hasattr(bank, "update")
    exported = bank.export_markdown()
    assert "untrusted research data" in exported
    assert "````json" in exported
    assert entry_id in exported
    assert '"outcome": "untried"' in exported


@pytest.mark.parametrize("updates", [
    {"details": {"loss": float("nan")}}, {"scope": {1: "bad key"}},
    {"artifacts": "not-list"}, {"related_ids": [None]}, {"created_by": ""},
    {"kind": "fact"}, {"outcome": "proven"}, {"scope": {"invalid": {1, 2}}},
])
def test_invalid_records_do_not_enter_bank(tmp_path, updates):
    bank = KnowledgeBank(tmp_path / "knowledge")
    with pytest.raises(ValueError):
        bank.add(entry(**updates))
    assert bank.search() == []


def test_export_roundtrips_as_json_even_with_multiline_text(tmp_path):
    bank = KnowledgeBank(tmp_path / "knowledge")
    entry_id = bank.add(entry(summary="line one\nline two", details={"tags": ["μP", "transfer"]}))
    exported = bank.export_markdown()
    content = exported.split("```json\n", 1)[1].split("\n```", 1)[0]
    assert json.loads(content) == bank.get(entry_id)


def test_exploration_trajectory_preserves_predictions_trials_results_and_learnings(tmp_path):
    bank = KnowledgeBank(tmp_path / "knowledge")
    idea_id = bank.add(entry())
    prediction_id = bank.add(entry(kind="prediction", title="Frozen expectation", related_ids=[idea_id],
        details={"expected": -.01, "lower": -.03, "upper": .01, "metric": "validation_ce_delta"}))
    trial_id = bank.add(entry(kind="trial", title="Trial receipt", outcome="inconclusive", related_ids=[prediction_id],
        artifacts=[artifact()], scope={"width": 64, "seed": 1}))
    result_id = bank.add(entry(kind="result", title="Paired comparison", outcome="contradicted", related_ids=[trial_id, prediction_id],
        artifacts=[artifact()], details={"mean_delta": .02}))
    learning_id = bank.add(entry(kind="learning", title="Width-scoped lesson", outcome="inconclusive", related_ids=[result_id, prediction_id],
        artifacts=[artifact()], scope={"width": 64}, details={"limitation": "One tiny-model comparison."}))
    assert bank.get(prediction_id)["details"]["expected"] == -.01
    assert bank.get(result_id)["outcome"] == "contradicted"
    assert bank.get(learning_id)["related_ids"] == sorted([result_id, prediction_id])
    for kind in ("prediction", "trial", "result", "learning"):
        assert len(bank.search(kind=kind)) == 1
    exported = bank.export_markdown()
    assert all(record in exported for record in (idea_id, prediction_id, trial_id, result_id, learning_id))


def test_existing_file_is_rejected_without_migration_or_overwrite(tmp_path):
    path = tmp_path / "old-knowledge.sqlite"
    original = b"Existing opaque file; must not be changed."
    path.write_bytes(original)
    with pytest.raises(ValueError, match="must be a directory.*Automatic migration"):
        KnowledgeBank(path)
    assert path.read_bytes() == original


def test_publish_syncs_contents_before_exposing_read_only_file_and_syncs_directory(tmp_path, monkeypatch):
    bank = KnowledgeBank(tmp_path / "knowledge")
    events = []
    fsync, link = knowledge.os.fsync, knowledge.os.link

    def synced(fd):
        events.append("directory_sync" if stat.S_ISDIR(knowledge.os.fstat(fd).st_mode) else "file_sync")
        return fsync(fd)

    def published(source, destination, **kwargs):
        assert Path(source).stat().st_mode & 0o222 == 0
        assert not Path(destination).exists()
        events.append("publish")
        return link(source, destination, **kwargs)

    monkeypatch.setattr(knowledge.os, "fsync", synced)
    monkeypatch.setattr(knowledge.os, "link", published)
    identity = bank.add(entry())
    assert events == ["file_sync", "publish", "directory_sync"]
    assert json.loads((bank.path / "entries" / f"{identity}.json").read_text())["id"] == identity


def test_interrupted_publication_leaves_no_partial_record_or_staging_file(tmp_path, monkeypatch):
    bank = KnowledgeBank(tmp_path / "knowledge")

    def fail_link(*args, **kwargs):
        raise OSError("Synthetic publication failure")

    monkeypatch.setattr(knowledge.os, "link", fail_link)
    with pytest.raises(OSError, match="Synthetic publication failure"):
        bank.add(entry())
    assert list((bank.path / "entries").iterdir()) == []
    assert bank.search() == []


@pytest.mark.parametrize("damage", ["json", "hash", "schema", "identity", "timestamp", "duplicate_key", "nonfinite"])
def test_corrupt_records_fail_closed_on_get_search_export_and_append(tmp_path, damage):
    bank = KnowledgeBank(tmp_path / "knowledge")
    identity = bank.add(entry())
    path = bank.path / "entries" / f"{identity}.json"
    record = json.loads(path.read_text())
    if damage == "json":
        content = "{partial-json"
    elif damage == "duplicate_key":
        content = '{"id": "first", "id": "second"}'
    else:
        if damage == "hash":
            record["payload"]["summary"] = "An unregistered changed claim."
        elif damage == "schema":
            record["payload"].pop("sources")
        elif damage == "identity":
            record["id"] = "kb_" + "0" * 64
        elif damage == "timestamp":
            record["created_at"] = "yesterday"
        else:
            record["payload"]["details"] = {"loss": float("nan")}
        content = json.dumps(record)
    path.chmod(0o644)  # Owners can edit files manually; reads must detect corruption.
    path.write_text(content)
    for action in (lambda: bank.get(identity), lambda: bank.search(query="not-present"),
                   bank.export_markdown, lambda: bank.add(entry(title="Unrelated new idea"))):
        with pytest.raises(ValueError, match="Corrupt knowledge record"):
            action()


def test_missing_related_file_is_detected_after_manual_deletion(tmp_path):
    bank = KnowledgeBank(tmp_path / "knowledge")
    parent = bank.add(entry())
    child = bank.add(entry(title="Follow-up", related_ids=[parent]))
    (bank.path / "entries" / f"{parent}.json").unlink()
    with pytest.raises(ValueError, match="missing related entries"):
        bank.get(child)


@pytest.mark.parametrize("identity", ["../../outside", "kb_../outside", "kb_" + "0" * 64, None])
def test_unknown_ids_are_path_safe_key_errors(tmp_path, identity):
    bank = KnowledgeBank(tmp_path / "knowledge")
    outside = tmp_path / "outside"
    outside.write_text("Do not read or modify through the entry-ID API.")
    with pytest.raises(KeyError):
        bank.get(identity)
    assert outside.read_text() == "Do not read or modify through the entry-ID API."


def test_symlink_records_are_not_accepted(tmp_path):
    bank = KnowledgeBank(tmp_path / "knowledge")
    identity = bank.add(entry())
    path = bank.path / "entries" / f"{identity}.json"
    copied = tmp_path / "copied.json"
    copied.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(copied)
    with pytest.raises(ValueError, match="Corrupt knowledge record"):
        bank.get(identity)


def test_search_and_export_use_stable_order_and_unicode_casefolding(tmp_path):
    bank = KnowledgeBank(tmp_path / "knowledge")
    first = bank.add(entry(title="Straße μP", details={"tag": "clip_2%"}))
    second = bank.add(entry(title="Another idea"))
    assert [row["id"] for row in bank.search()] == [second, first]
    assert [row["id"] for row in bank.search("STRASSE ΜP clip_2%")] == [first]
    exported = bank.export_markdown()
    assert exported.index(f"## {first}") < exported.index(f"## {second}")


def test_shared_read_snapshot_keeps_writer_waiting_until_read_finishes(tmp_path):
    bank = KnowledgeBank(tmp_path / "knowledge")
    bank.add(entry())
    with ThreadPoolExecutor(max_workers=1) as executor:
        with bank._locked():
            future = executor.submit(_write_record_worker, (str(bank.path), "later"))
            initial = bank._snapshot()
            assert len(initial) == 1
            assert not future.done()
        future.result(timeout=5)
    assert len(bank.search()) == 2
