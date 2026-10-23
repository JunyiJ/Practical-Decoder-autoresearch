"""Pure history views: no models, agent calls, training, or subprocesses."""
from copy import deepcopy
import csv
import json
from pathlib import Path

from src.research.team.history import export_history


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def performance(target, width, metric, delta, base, candidate, complete=True):
    return {"id": f"{target}:{metric}", "target": target, "width": width, "metric": metric,
            "unit": "nats/character" if metric == "validation_ce_delta" else "%", "complete": complete,
            "mean_delta": delta, "baseline_mean": base, "candidate_mean": candidate,
            "pairs": [{"seed": 11, "baseline": base, "candidate": candidate, "delta": delta}],
            "missing_seeds": [] if complete else [22, 33]}


def campaign():
    return {"key": "study-one", "status": "awaiting_training_authorization",
            "config": {"objective": "Find improvements that transfer"},
            "agent_seconds": 17, "training_seconds": 8,
            "agent_calls": [
                {"iteration": "iteration-010", "role": "idea", "duration_seconds": 10, "reserved_seconds": 20,
                 "status": "completed", "usage": [{"input_tokens": 50, "output_tokens": 12}]},
                {"iteration": "iteration-010", "role": "evaluator", "reserved_seconds": 7, "status": "failed"},
                {"iteration": "iteration-002", "role": "coder", "duration_seconds": 3, "status": "completed"}],
            "iterations": [
                {"id": "iteration-010", "stage": "completed", "idea_id": "kb_idea", "prediction_id": "kb_prediction",
                 "execution_id": "kb_execution", "evaluation_id": "kb_eval", "learning_ids": ["kb_learning"],
                 "idea": {"title": "Stable normalization", "hypothesis": "Normalization transfers",
                          "based_on": ["kb_prior", "kb_prior"], "prediction": {"mechanism": "stability", "estimates": []}},
                 "workspace": {"branch": "codex/study/iteration-010"}, "snapshot": {"source_digest": "a" * 64},
                 "implementation": {"config_overrides": [{"path": "training.learning_rate", "value_json": "0.001"}]},
                 "suite": {"targets": [{"name": "small", "width": 64}, {"name": "large", "width": 256}]},
                 "prediction_check": {"comparisons": [{"target": "small", "metric": "validation_ce_delta", "outcome": "within_range"}]},
                 "evaluation": {"learnings": [{"finding": "Benefit seen locally", "explanation": "Lower CE",
                    "limitation": "Large model incomplete", "next_action": "Investigate timeout", "evidence_ids": ["small:validation_ce_delta"]}]},
                 "report": {"local_outcome": "supported", "transfer_outcome": "incomplete", "audit_outcome": "not_requested",
                    "active_seconds": 8, "performance": [
                        performance("small", 64, "validation_ce_delta", -.1, 2., 1.9),
                        performance("small", 64, "training_throughput_pct", 20., 100., 120.),
                        performance("small", 64, "peak_rss_pct", -10., 100., 90.),
                        performance("large", 256, "validation_ce_delta", -.7, 1., .3, complete=False)],
                    "records": [
                        {"trial_id": "local_small_s11_baseline", "stage": "local", "target": "small", "width": 64,
                         "seed": 11, "arm": "baseline", "status": "completed", "metric": 2., "tokens_processed": 1000,
                         "training_time_sec": 10., "duration_seconds": 12., "rss_mb": 100., "accel_mem_mb": 80.,
                         "accel_mem_kind": "mps_current_allocated", "checkpoint_sha256": "b" * 64},
                        {"trial_id": "transfer_large_s11_candidate", "stage": "transfer", "target": "large", "width": 256,
                         "seed": 11, "arm": "candidate", "status": "timeout", "metric": None, "error": "deadline reached"},
                        {"trial_id": "test_small_s11_baseline", "parent_trial_id": "local_small_s11_baseline", "stage": "test",
                         "target": "small", "width": 64, "seed": 11, "arm": "baseline", "status": "completed", "metric": 2.1}]}},
                {"id": "iteration-002", "stage": "failed", "idea": {"title": "Broken idea", "based_on": ["kb_learning"]},
                 "error": "local test failed", "failed_phase": "coder"},
                {"id": "iteration-003", "stage": "idea"}]}


def test_history_keeps_chronology_lineage_predictions_learnings_and_state(tmp_path):
    state = campaign()
    original = deepcopy(state)
    paths = export_history(tmp_path, state)
    assert state == original
    assert set(paths) == {"trajectory_json", "trajectory_csv", "trajectory_md", "trials_json", "trials_csv"}
    assert all(Path(path).is_absolute() and Path(path).is_file() for path in paths.values())
    trajectory = read_json(paths["trajectory_json"])
    assert [item["iteration_id"] for item in trajectory["iterations"]] == ["iteration-010", "iteration-002", "iteration-003"]
    first, failed, pending = trajectory["iterations"]
    assert first["iteration_order"] == 1
    assert first["depends_on"] == ["kb_prior"]
    assert first["prediction_id"] == "kb_prediction"
    assert first["learning_ids"] == ["kb_learning"]
    assert first["learnings"][0]["limitation"] == "Large model incomplete"
    assert first["prediction_check"]["comparisons"][0]["outcome"] == "within_range"
    assert first["source_sha256"] == "a" * 64
    assert failed["error"] == "local test failed" and failed["local_outcome"] == "not_run"
    assert pending["performance"] == [] and pending["stage"] == "idea"
    text = Path(paths["trajectory_md"]).read_text()
    assert "Stable normalization" in text and "local test failed" in text
    assert "Benefit seen locally" in text and "kb_learning" in text
    assert "pending" in text and "incomplete" in text
    assert "no pooled best score" in text


def test_csv_is_one_row_per_target_and_does_not_promote_incomplete_metrics(tmp_path):
    paths = export_history(tmp_path, campaign())
    rows = read_csv(paths["trajectory_csv"])
    assert [(row["iteration_id"], row["target"]) for row in rows] == [
        ("iteration-010", "small"), ("iteration-010", "large"), ("iteration-002", ""), ("iteration-003", "")]
    small, large, failed, pending = rows
    assert float(small["validation_ce_delta_mean_delta"]) == -.1
    assert float(small["training_throughput_pct_baseline_mean"]) == 100.
    assert float(small["peak_rss_pct_mean_delta"]) == -10.
    assert small["validation_ce_delta_complete"] == "true"
    assert large["validation_ce_delta_mean_delta"] == ""
    assert large["validation_ce_delta_baseline_mean"] == ""
    assert large["validation_ce_delta_complete"] == "false"
    assert json.loads(large["validation_ce_delta_missing_seeds"]) == [22, 33]
    assert len(json.loads(large["performance"])[0]["pairs"]) == 1
    assert failed["suite_active_seconds"] == pending["validation_ce_delta_mean_delta"] == ""


def test_actual_agent_time_is_not_confused_with_reservations_or_unknown_cost(tmp_path):
    paths = export_history(tmp_path, campaign())
    first, failed, pending = read_json(paths["trajectory_json"])["iterations"]
    assert first["agent_calls_count"] == 2
    assert first["agent_actual_seconds"] == 10
    assert first["agent_reserved_seconds"] == 27
    assert first["agent_unresolved_reserved_seconds"] == 7
    assert first["agent_calls_missing_actual"] == 1
    assert first["agent_actual_seconds_complete"] is False
    assert first["agent_calls"][0]["usage"][0]["input_tokens"] == 50
    assert failed["agent_actual_seconds"] == 3 and failed["agent_reserved_seconds"] is None
    assert pending["agent_actual_seconds"] is None and pending["suite_active_seconds"] is None
    assert "no monetary cost is estimated" in Path(paths["trajectory_md"]).read_text()


def test_every_trial_including_failure_and_audit_is_exported_with_lineage(tmp_path):
    paths = export_history(tmp_path, campaign())
    records = read_json(paths["trials_json"])["records"]
    rows = read_csv(paths["trials_csv"])
    assert len(records) == len(rows) == 3
    assert [row["stage"] for row in rows] == ["local", "transfer", "test"]
    assert all(row["campaign"] == "study-one" and row["iteration_id"] == "iteration-010" for row in rows)
    assert all(row["prediction_id"] == "kb_prediction" for row in rows)
    assert records[0]["training_time_sec"] == 10 and records[0]["rss_mb"] == 100
    assert records[0]["accel_mem_kind"] == "mps_current_allocated"
    assert rows[1]["status"] == "timeout" and rows[1]["error"] == "deadline reached"
    assert rows[1]["metric"] == rows[1]["duration_seconds"] == ""
    assert rows[2]["parent_trial_id"] == rows[0]["trial_id"]
    assert rows[2]["training_time_sec"] == ""


def test_minimal_and_empty_campaigns_rebuild_without_stale_rows_or_temp_files(tmp_path):
    paths = export_history(tmp_path, campaign())
    minimal = {"iterations": [{"id": "untried"}, {"stage": "failed"}]}
    export_history(tmp_path, minimal)
    rows = read_csv(paths["trajectory_csv"])
    assert len(rows) == 2 and rows[0]["stage"] == "pending"
    assert read_json(paths["trials_json"])["records"] == []
    assert read_csv(paths["trials_csv"]) == []
    export_history(tmp_path, {})
    assert read_json(paths["trajectory_json"])["iterations"] == []
    assert "No iterations have been recorded" in Path(paths["trajectory_md"]).read_text()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["trajectory.csv", "trajectory.json", "trajectory.md", "trials.csv", "trials.json"]


def test_legacy_ce_reports_remain_visible_without_inventing_throughput(tmp_path):
    state = {"iterations": [{"id": "old", "report": {"comparisons": [{"target": "w64", "width": 64,
        "complete": True, "mean_delta": -.2, "pairs": [{"seed": 0, "baseline": 2., "candidate": 1.8}]}]}}]}
    paths = export_history(tmp_path, state)
    rows = read_csv(paths["trajectory_csv"])
    assert len(rows) == 1 and rows[0]["target"] == "w64"
    assert float(rows[0]["validation_ce_delta_baseline_mean"]) == 2.
    assert rows[0]["training_throughput_pct_mean_delta"] == ""
    assert rows[0]["peak_rss_pct_mean_delta"] == ""


def test_csv_text_is_not_a_formula_and_markdown_does_not_break_on_idea_text(tmp_path):
    state = {"iterations": [{"id": "one", "idea": {"title": '=HYPERLINK("https://invalid")|<script>\nheading'}}]}
    paths = export_history(tmp_path, state)
    assert read_csv(paths["trajectory_csv"])[0]["title"].startswith("'=HYPERLINK")
    assert read_json(paths["trajectory_json"])["iterations"][0]["title"].startswith("=HYPERLINK")
    markdown = Path(paths["trajectory_md"]).read_text()
    assert "<script>" not in markdown and "&lt;script&gt;" in markdown and "\\|" in markdown


def test_result_and_trial_knowledge_ids_and_timestamps_survive_export(tmp_path):
    state = campaign()
    state["created_at"] = "2026-10-23T01:00:00Z"
    raw = state["iterations"][0]
    raw.update(result_id="kb_result", created_at="2026-10-23T01:01:00Z", completed_at="2026-10-23T01:04:00Z",
               trial_ids=["kb_trial_one", "kb_trial_failed", "kb_trial_test"])
    paths = export_history(tmp_path, state)
    root = read_json(paths["trajectory_json"])
    row = root["iterations"][0]
    assert root["created_at"] == state["created_at"]
    assert row["result_id"] == "kb_result"
    assert row["id"] == row["iteration_id"] and row["ordinal"] == row["iteration_order"] == 1
    assert row["created_at"] == raw["created_at"] and row["completed_at"] == raw["completed_at"]
    assert read_csv(paths["trajectory_csv"])[0]["result_id"] == "kb_result"
    trials = read_json(paths["trials_json"])["records"]
    assert trials[1]["trial_knowledge_id"] == "kb_trial_failed"
    assert trials[0]["result_id"] == "kb_result" and trials[0]["iteration_completed_at"] == raw["completed_at"]


def test_failure_lessons_do_not_take_ids_from_unpersisted_evaluator_lessons(tmp_path):
    lesson = {"finding": "The code change failed a local test", "explanation": "Shape mismatch",
              "limitation": "No quality measurement", "next_action": "Fix the test", "evidence_ids": ["kb_failure"]}
    state = {"iterations": [{"id": "failed", "stage": "failed", "failure_learning": lesson,
            "failure_learning_id": "kb_operational", "learning_ids": ["kb_first_evaluator_lesson"],
            "evaluation": {"learnings": [{"finding": "First interpretation"}, {"finding": "Unpersisted interpretation"}]}}]}
    paths = export_history(tmp_path, state)
    row = read_json(paths["trajectory_json"])["iterations"][0]
    assert [record["id"] for record in row["learning_records"]] == ["kb_first_evaluator_lesson", None, "kb_operational"]
    assert row["learning_records"][-1]["kind"] == "operational_failure"
    assert row["learnings"][-1] == lesson
    assert row["failure_learning_id"] == "kb_operational"
    text = Path(paths["trajectory_md"]).read_text()
    assert "The code change failed a local test" in text and "kb_operational" in text
    assert "Unpersisted interpretation** (evaluation; learning ID: not recorded)" in text


def test_forecasts_are_readable_before_execution_and_compared_without_fabricated_actuals(tmp_path):
    estimates = [
        {"target": "small", "metric": "validation_ce_delta", "expected": -.1, "lower": -.15, "upper": -.05,
         "confidence": "medium", "rationale": "A moderate CE effect"},
        {"target": "large", "metric": "training_throughput_pct", "expected": None, "lower": None, "upper": None,
         "confidence": "low", "rationale": "Timing is uncertain"}]
    raw = {"id": "forecast", "stage": "coder", "prediction_id": "kb_forecast",
           "idea": {"title": "New attention", "prediction": {"mechanism": "Less work per attention step", "estimates": estimates}}}
    state = {"iterations": [raw]}
    paths = export_history(tmp_path, state)
    text = Path(paths["trajectory_md"]).read_text()
    assert "Registered forecasts:" in text and "[-0.15, -0.05]" in text
    assert "unquantified" in text and "Less work per attention step" in text
    assert [row["target"] for row in read_csv(paths["trajectory_csv"])] == ["small", "large"]
    raw["prediction_check"] = {"comparisons": [
        {"id": "small:validation_ce_delta", "target": "small", "metric": "validation_ce_delta", "unit": "nats/character",
         "prediction": estimates[0], "complete": True, "observed_mean_delta": -.08, "verdict": "within_range", "error": .02},
        {"id": "large:training_throughput_pct", "target": "large", "metric": "training_throughput_pct", "unit": "%",
         "prediction": estimates[1], "complete": False, "observed_mean_delta": 9876., "verdict": "unavailable", "missing_seeds": [11]}]}
    export_history(tmp_path, state)
    text = Path(paths["trajectory_md"]).read_text()
    assert "| -0.08 | +0.02 | within_range |" in text
    assert "9876" not in text
    row = read_json(paths["trajectory_json"])["iterations"][0]
    assert row["prediction_check"] == raw["prediction_check"]
