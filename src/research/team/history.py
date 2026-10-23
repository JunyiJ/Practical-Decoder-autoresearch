"""Rebuildable chronological views of campaign evidence, without model execution.

The campaign and separately recorded evidence remain authoritative. These JSON,
CSV, and Markdown files are derived views, atomically replaced after each save.
They neither rank unlike model widths nor infer missing performance or costs.
"""
from __future__ import annotations

import copy
import csv
import html
import io
import json
import math
import os
from pathlib import Path
import tempfile


METRICS = ("validation_ce_delta", "training_throughput_pct", "peak_rss_pct")
LINEAGE = ("campaign", "iteration_order", "iteration_id", "idea_id", "prediction_id",
           "execution_id", "result_id", "evaluation_id", "learning_ids", "failure_learning_id", "depends_on", "branch", "source_sha256")
COST_FIELDS = ("agent_calls_count", "agent_actual_seconds", "agent_actual_seconds_complete",
               "agent_reserved_seconds", "agent_unresolved_reserved_seconds", "agent_calls_missing_actual",
               "suite_active_seconds")
TRAJECTORY_FIELDS = (*LINEAGE, "created_at", "completed_at", "stage", "title", "hypothesis", "target", "width",
                     "local_outcome", "transfer_outcome", "audit_outcome", "error", "failed_phase",
                     "config_overrides", *COST_FIELDS,
                     *(f"{metric}_{field}" for metric in METRICS
                       for field in ("mean_delta", "baseline_mean", "candidate_mean", "complete", "missing_seeds")),
                     "performance", "prediction_check", "learnings", "learning_records")
TRIAL_FIELDS = (*LINEAGE, "record_order", "trial_id", "parent_trial_id", "stage", "target", "width",
                "n_heads", "seed", "arm", "status", "metric", "tokens_processed", "training_time_sec",
                "duration_seconds", "rss_mb", "accel_mem_mb", "accel_mem_kind", "error",
                "checkpoint", "checkpoint_sha256")


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _list(value) -> list:
    return value if isinstance(value, list) else []


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _json(value, *, pretty=False) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      indent=2 if pretty else None)


def _sum_known(values) -> float | None:
    known = [value for value in values if _number(value)]
    return sum(known) if known else None


def _agent_costs(state: dict, iteration_id: str | None) -> dict:
    calls = [copy.deepcopy(call) for call in _list(state.get("agent_calls"))
             if iteration_id is not None and isinstance(call, dict) and call.get("iteration") == iteration_id]
    missing = [call for call in calls if not _number(call.get("duration_seconds"))]
    return {"agent_calls_count": len(calls),
            "agent_actual_seconds": _sum_known(call.get("duration_seconds") for call in calls),
            "agent_actual_seconds_complete": bool(calls) and not missing,
            "agent_reserved_seconds": _sum_known(call.get("reserved_seconds") for call in calls),
            "agent_unresolved_reserved_seconds": _sum_known(call.get("reserved_seconds") for call in missing),
            "agent_calls_missing_actual": len(missing), "agent_calls": calls}


def _performance(report: dict) -> list:
    if isinstance(report.get("performance"), list):
        return copy.deepcopy(report["performance"])
    # Older reports recorded CE comparisons only. Retain their evidence while
    # leaving new throughput/memory aggregates absent rather than fabricating them.
    result = []
    for comparison in _list(report.get("comparisons")):
        row = _dict(comparison)
        pairs = _list(row.get("pairs"))
        complete = row.get("complete") is True
        performance = {"id": f"{row.get('target')}:validation_ce_delta", "target": row.get("target"),
                       "width": row.get("width"), "metric": "validation_ce_delta", "unit": "nats/character",
                       "complete": complete, "mean_delta": row.get("mean_delta") if complete else None,
                       "sample_std_delta": row.get("sample_std_delta") if complete else None,
                       "pairs": copy.deepcopy(pairs), "missing_seeds": copy.deepcopy(row.get("missing_seeds", []))}
        for arm in ("baseline", "candidate"):
            values = [_dict(pair).get(arm) for pair in pairs]
            if complete and values and all(_number(value) for value in values):
                performance[f"{arm}_mean"] = sum(values) / len(values)
        result.append(performance)
    return result


def _iteration(state: dict, raw: dict, order: int) -> dict:
    idea, report = _dict(raw.get("idea")), _dict(raw.get("report"))
    implementation, snapshot = _dict(raw.get("implementation")), _dict(raw.get("snapshot"))
    evaluation = _dict(raw.get("evaluation"))
    learning_ids = copy.deepcopy(_list(raw.get("learning_ids")))
    learning_records = [{"id": learning_ids[index] if index < len(learning_ids) else None,
                         "kind": "evaluation", "learning": copy.deepcopy(lesson)}
                        for index, lesson in enumerate(_list(evaluation.get("learnings")))]
    if isinstance(raw.get("failure_learning"), dict):
        learning_records.append({"id": raw.get("failure_learning_id"), "kind": "operational_failure",
                                 "learning": copy.deepcopy(raw["failure_learning"])})
    depends_on = list(dict.fromkeys(value for value in
        [*_list(idea.get("based_on")), *_list(raw.get("depends_on"))] if isinstance(value, str)))
    row = {"campaign": state.get("key", _dict(state.get("config")).get("name")),
           "iteration_order": order, "iteration_id": raw.get("id"), "stage": raw.get("stage", "pending"),
           "ordinal": order, "id": raw.get("id"), "created_at": raw.get("created_at"), "completed_at": raw.get("completed_at"),
           "title": idea.get("title", raw.get("title")), "hypothesis": idea.get("hypothesis"),
           "idea_id": raw.get("idea_id"), "prediction_id": raw.get("prediction_id"),
           "execution_id": raw.get("execution_id"), "result_id": raw.get("result_id"), "evaluation_id": raw.get("evaluation_id"),
           "learning_ids": learning_ids, "failure_learning_id": raw.get("failure_learning_id"), "depends_on": depends_on,
           "branch": _dict(raw.get("workspace")).get("branch", snapshot.get("branch")),
           "source_sha256": snapshot.get("source_digest", snapshot.get("source_sha256")),
           "config_overrides": copy.deepcopy(implementation.get("config_overrides", raw.get("config_overrides", []))),
           "idea": copy.deepcopy(idea), "prediction": copy.deepcopy(raw.get("prediction", idea.get("prediction"))),
           "prediction_check": copy.deepcopy(raw.get("prediction_check")),
           "learnings": [record["learning"] for record in learning_records], "learning_records": learning_records,
           "failure_learning": copy.deepcopy(raw.get("failure_learning")),
           "local_outcome": report.get("local_outcome", "not_run"),
           "transfer_outcome": report.get("transfer_outcome", "not_run"),
           "audit_outcome": report.get("audit_outcome", "not_run"),
           "evidence_integrity": report.get("evidence_integrity"),
           "stop_reason": report.get("stop_reason"), "error": raw.get("error"),
           "failed_phase": raw.get("failed_phase"), "performance": _performance(report),
           "suite_active_seconds": report.get("active_seconds"),
           **_agent_costs(state, raw.get("id"))}
    # Preserve target order from the declared suite, followed by any legacy report
    # targets. Never pool scores across widths or sort targets by their results.
    targets = {}
    declared = _list(_dict(raw.get("suite")).get("targets")) + _list(_dict(report.get("protocol")).get("targets"))
    for target in declared:
        target = _dict(target)
        name = target.get("name", target.get("target"))
        if name is not None:
            targets.setdefault(name, {"target": name, "width": target.get("width")})
    for performance in row["performance"]:
        performance = _dict(performance)
        name = performance.get("target")
        if name is not None:
            targets.setdefault(name, {"target": name, "width": performance.get("width")})
    for estimate in _list(_dict(row["prediction"]).get("estimates")):
        name = _dict(estimate).get("target")
        if name is not None:
            targets.setdefault(name, {"target": name, "width": None})
    row["targets"] = list(targets.values())
    return row


def _trajectory_rows(iterations: list) -> list:
    rows = []
    for item in iterations:
        for target in item["targets"] or [{"target": None, "width": None}]:
            row = {key: copy.deepcopy(item.get(key)) for key in TRAJECTORY_FIELDS if key in item}
            row.update(target)
            performance = [p for p in item["performance"] if _dict(p).get("target") == target["target"]]
            row["performance"] = performance
            for metric in METRICS:
                matches = [p for p in performance if _dict(p).get("metric") == metric]
                if len(matches) == 1:
                    result = matches[0]
                    for field in ("mean_delta", "baseline_mean", "candidate_mean", "complete", "missing_seeds"):
                        # Incomplete results retain their pairs in performance JSON,
                        # but must not appear as full paired means in flat views.
                        value = result.get(field)
                        if field in ("mean_delta", "baseline_mean", "candidate_mean") and result.get("complete") is not True:
                            value = None
                        row[f"{metric}_{field}"] = value
            rows.append(row)
    return rows


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return _json(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    # CSV readers may otherwise execute an idea title as a spreadsheet formula.
    # JSON remains the exact text record; numeric deltas retain numeric CSV cells.
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value


def _csv(rows: list, fields) -> str:
    extra = sorted({key for row in rows for key in row} - set(fields))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=[*fields, *extra], lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _cell(value) for key, value in row.items()})
    return stream.getvalue()


def _md(value) -> str:
    if value is None or value == "":
        return "—"
    return html.escape(str(value), quote=False).replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


def _formatted(value, *, signed=False) -> str:
    return f"{value:+.5g}" if signed and _number(value) else f"{value:.5g}" if _number(value) else "—"


def _metric_cell(row, metric) -> str:
    complete = row.get(f"{metric}_complete")
    if complete is not True:
        return "incomplete" if complete is False else "pending"
    return _formatted(row.get(f"{metric}_mean_delta"), signed=True)


def _forecast_table(item: dict) -> list[str]:
    checks = [_dict(value) for value in _list(_dict(item["prediction_check"]).get("comparisons"))]
    estimates = [_dict(value) for value in _list(_dict(item["prediction"]).get("estimates"))]
    keys = {(row.get("target"), row.get("metric")) for row in estimates}
    # Old reports may have comparison records but no retained proposal object.
    estimates += [{**_dict(check.get("prediction")), "target": check.get("target"), "metric": check.get("metric")}
                  for check in checks if (check.get("target"), check.get("metric")) not in keys]
    if not estimates:
        return ["Prediction comparison: pending or not recorded.", ""]
    label = "Registered forecasts" if item["prediction_id"] else "Proposed forecasts (registration pending)"
    lines = [label + ":", "",
             "| Target | Metric (unit) | Expected Δ | Plausible range | Observed Δ | Actual − expected | Verdict | Confidence |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for estimate in estimates:
        target, metric = estimate.get("target"), estimate.get("metric")
        matches = [check for check in checks if check.get("target") == target and check.get("metric") == metric]
        check = matches[0] if len(matches) == 1 else {}
        expected = _formatted(estimate.get("expected"), signed=True) if _number(estimate.get("expected")) else "unquantified"
        bounds = ([estimate.get("lower"), estimate.get("upper")])
        interval = "[" + ", ".join(_formatted(value, signed=True) for value in bounds) + "]" if all(_number(value) for value in bounds) else "unquantified"
        observed = _formatted(check.get("observed_mean_delta"), signed=True) if check.get("complete") is True else "unavailable" if check else "pending"
        error = _formatted(check.get("error"), signed=True) if check.get("complete") is True else "—"
        verdict = check.get("verdict", check.get("outcome", "pending"))
        unit = check.get("unit", "nats/character" if metric == "validation_ce_delta" else "%" if metric in METRICS else "unspecified")
        lines.append(f"| {_md(target)} | {_md(metric)} ({_md(unit)}) | {expected} | {interval} | {observed} | {error} | {_md(verdict)} | {_md(estimate.get('confidence'))} |")
    lines.append("")
    return lines


def _markdown(trajectory: dict, rows: list) -> str:
    lines = ["# Research exploration trajectory", "", f"Campaign: **{_md(trajectory['campaign'])}**. Status: **{_md(trajectory['status'])}**.", "",
             "Chronological, rebuildable view of recorded evidence. Each target is compared with its own baseline; "
             "there is no pooled best score or universal winner across widths.", "",
             "CE deltas are nats/character (lower is better). Throughput changes are percent (higher is better). "
             "RSS changes are percent (lower is better); RSS is process high-water memory, not GPU peak memory. "
             "Pending or incomplete results are not measured gains.", "",
             "| # / iteration | Idea | Stage | Target (width) | Mean CE baseline → candidate | CE Δ | Throughput Δ % | RSS Δ % | Local / transfer / audit |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        target = _md(row.get("target"))
        if row.get("width") is not None:
            target += f" ({_md(row['width'])})"
        outcomes = " / ".join(_md(row.get(field)) for field in ("local_outcome", "transfer_outcome", "audit_outcome"))
        means = " → ".join(_formatted(row.get(f"validation_ce_delta_{arm}_mean")) for arm in ("baseline", "candidate")) if row.get("validation_ce_delta_complete") is True else "incomplete" if row.get("validation_ce_delta_complete") is False else "pending"
        lines.append(f"| {row['iteration_order']} / {_md(row['iteration_id'])} | {_md(row.get('title'))} | {_md(row['stage'])} | "
                     f"{target} | {means} | {_metric_cell(row, METRICS[0])} | {_metric_cell(row, METRICS[1])} | {_metric_cell(row, METRICS[2])} | {outcomes} |")
    if not rows:
        lines.extend(["", "No iterations have been recorded."])
    lines.extend(["", "## Recorded execution cost", "",
                  "Actual agent durations include only reported measurements; reservations are shown separately and must not be added to actual time. "
                  "Suite seconds reproduce the suite's accounting. Missing values do not mean zero, and no monetary cost is estimated. "
                  "CSV cost fields repeat per target: aggregate them once per iteration.", "",
                  "| Iteration | Agent calls | Actual agent s | Reserved agent s | Calls missing actual time | Suite accounted s |",
                  "| --- | --- | --- | --- | --- | --- |"])
    for item in trajectory["iterations"]:
        lines.append(f"| {_md(item['iteration_id'])} | {item['agent_calls_count']} | {_formatted(item['agent_actual_seconds'])} | "
                     f"{_formatted(item['agent_reserved_seconds'])} | {item['agent_calls_missing_actual']} | {_formatted(item['suite_active_seconds'])} |")
    lines.extend(["", "## Idea and learning trail", ""])
    for item in trajectory["iterations"]:
        lines.extend([f"### {item['iteration_order']}. {_md(item['title'] or item['iteration_id'])}", "",
                      f"Hypothesis: {_md(item['hypothesis'])}", "",
                      f"Created: {_md(item['created_at'])}. Completed: {_md(item['completed_at'])}.", "",
                      f"Evidence IDs: idea `{_md(item['idea_id'])}`; prediction `{_md(item['prediction_id'])}`; "
                      f"execution `{_md(item['execution_id'])}`; result `{_md(item['result_id'])}`; evaluation `{_md(item['evaluation_id'])}`.", "",
                      "Depends on: " + (", ".join(_md(value) for value in item["depends_on"]) or "none recorded") + ".", ""])
        if item["error"]:
            lines.extend([f"Failure ({_md(item['failed_phase'])}): {_md(item['error'])}", ""])
        prediction = _dict(item["prediction"])
        if prediction.get("mechanism"):
            lines.extend([f"Predicted mechanism: {_md(prediction['mechanism'])}.", ""])
        lines.extend(_forecast_table(item))
        if item["learnings"]:
            for record in item["learning_records"]:
                learning, learning_id = record["learning"], record["id"] or "not recorded"
                if isinstance(learning, dict):
                    lines.extend([f"- **{_md(learning.get('finding'))}** ({_md(record['kind'])}; learning ID: {_md(learning_id)}). "
                                  f"Explanation: {_md(learning.get('explanation'))}. "
                                  f"Limit: {_md(learning.get('limitation'))}. Next: {_md(learning.get('next_action'))}. "
                                  f"Evidence: {_md(', '.join(_list(learning.get('evidence_ids'))))}."])
                else:
                    lines.append(f"- {_md(learning)} (learning ID: {_md(learning_id)}).")
            lines.append("")
        else:
            lines.extend(["Learnings: pending or not recorded.", ""])
    return "\n".join(lines).rstrip() + "\n"


def _atomic_write(path: Path, content: str):
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def export_history(output: Path, state: dict) -> dict:
    """Write derived history and all trial records; never modify campaign state."""
    output = Path(output).expanduser().resolve()
    iterations = [_iteration(state, _dict(item), order)
                  for order, item in enumerate(_list(state.get("iterations")), start=1)]
    campaign = state.get("key", _dict(state.get("config")).get("name"))
    trajectory = {"version": 1, "derived": True, "campaign": campaign, "status": state.get("status"),
                  "created_at": state.get("created_at"), "completed_at": state.get("completed_at"),
                  "objective": _dict(state.get("config")).get("objective"), "iterations": iterations,
                  "campaign_accounted_agent_seconds": state.get("agent_seconds"),
                  "campaign_accounted_training_seconds": state.get("training_seconds"),
                  "cost_note": "Accounted budgets may include reservations. Actual agent durations are separately recorded; no monetary costs are inferred."}
    rows = _trajectory_rows(iterations)
    trials = []
    for raw, iteration in zip(_list(state.get("iterations")), iterations):
        for order, record in enumerate(_list(_dict(_dict(raw).get("report")).get("records")), start=1):
            trial_ids = _list(_dict(raw).get("trial_ids"))
            trials.append({**copy.deepcopy(_dict(record)), **{key: copy.deepcopy(iteration.get(key)) for key in LINEAGE},
                           "record_order": order, "iteration_created_at": iteration["created_at"],
                           "iteration_completed_at": iteration["completed_at"],
                           "trial_knowledge_id": trial_ids[order - 1] if order <= len(trial_ids) else None})
    contents = {"trajectory_json": ("trajectory.json", _json(trajectory, pretty=True) + "\n"),
                "trajectory_csv": ("trajectory.csv", _csv(rows, TRAJECTORY_FIELDS)),
                "trajectory_md": ("trajectory.md", _markdown(trajectory, rows)),
                "trials_json": ("trials.json", _json({"version": 1, "derived": True, "campaign": campaign, "records": trials}, pretty=True) + "\n"),
                "trials_csv": ("trials.csv", _csv(trials, TRIAL_FIELDS))}
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for key, (name, content) in contents.items():
        path = output / name
        _atomic_write(path, content)
        artifacts[key] = str(path)
    return artifacts
