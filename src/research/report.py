"""Descriptive reports for a preregistered, paired width-transfer experiment.

This module never chooses a candidate. In particular, transfer or held-out test
scores cannot change the selection made on small-width validation results.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
from statistics import mean, stdev
from typing import Any


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _usable(record: dict) -> bool:
    return record.get("status") == "completed" and _finite_number(record.get("metric"))


def _clean_record(record: dict) -> dict:
    result = deepcopy(record)
    if not _finite_number(result.get("metric")):
        result["metric"] = None
    return result


def _baseline_id(plan: dict, records: list[dict], selection: dict) -> str | None:
    if selection.get("baseline_id"):
        return selection["baseline_id"]
    baseline_lr = plan.get("baseline_lr")
    if not _finite_number(baseline_lr):
        return None
    candidates = {
        row["candidate"] for row in records
        if row.get("candidate") and _finite_number(row.get("learning_rate"))
        and math.isclose(row["learning_rate"], baseline_lr, rel_tol=1e-12, abs_tol=0)
    }
    return next(iter(candidates)) if len(candidates) == 1 else None


def _comparison(records: list[dict], stage: str, width: int, seeds: list[int],
                candidate: str, baseline: str) -> dict:
    """Pair only declared seeds, excluding ambiguous duplicate successful runs."""
    by_key: dict[tuple[str, int], list[dict]] = {}
    for row in records:
        if (row.get("stage") == stage and row.get("width") == width
                and row.get("candidate") in {candidate, baseline} and _usable(row)):
            by_key.setdefault((row["candidate"], row.get("seed")), []).append(row)

    pairs, missing, duplicate = [], [], []
    for seed in seeds:
        candidate_rows = by_key.get((candidate, seed), [])
        baseline_rows = by_key.get((baseline, seed), [])
        if len(candidate_rows) > 1 or len(baseline_rows) > 1:
            duplicate.append(seed)
            continue
        if not candidate_rows or not baseline_rows:
            missing.append(seed)
            continue
        candidate_row, baseline_row = candidate_rows[0], baseline_rows[0]
        pairs.append({
            "seed": seed,
            "candidate_metric": candidate_row["metric"],
            "baseline_metric": baseline_row["metric"],
            "delta": candidate_row["metric"] - baseline_row["metric"],
        })
    deltas = [pair["delta"] for pair in pairs]
    complete = bool(seeds) and not missing and not duplicate
    return {
        "stage": stage,
        "width": width,
        "candidate_id": candidate,
        "baseline_id": baseline,
        "expected_seeds": list(seeds),
        "pairs": pairs,
        "n_pairs": len(pairs),
        "missing_seeds": missing,
        "duplicate_seeds": duplicate,
        "complete": complete,
        "mean_candidate": mean(pair["candidate_metric"] for pair in pairs) if pairs else None,
        "mean_baseline": mean(pair["baseline_metric"] for pair in pairs) if pairs else None,
        "mean_delta": mean(deltas) if deltas else None,
        "sample_std_delta": stdev(deltas) if len(deltas) >= 2 else None,
        "improved_seeds": sum(delta < 0 for delta in deltas),
    }


def _transfer_outcome(comparisons: list[dict], selection: dict, widths: list[int]) -> tuple[str, str]:
    transfer = [row for row in comparisons if row["stage"] == "transfer"]
    if (not selection.get("winner_id") or not widths[1:]
            or len(transfer) != len(widths[1:]) or not all(row["complete"] for row in transfer)):
        return "incomplete", "The declared challenger and baseline do not yet have all planned larger-width seed pairs."
    if selection["winner_id"] == selection.get("baseline_id"):
        return "not_beneficial", "Small-width confirmation retained the baseline; no improvement from the locked choice is demonstrated. Larger-width challenger comparisons remain diagnostic."
    if all(row["mean_delta"] < 0 and row["improved_seeds"] == row["n_pairs"] for row in transfer):
        return "beneficial", "The locked candidate improved final validation CE in every planned larger-width seed pair."
    if all(row["mean_delta"] >= 0 for row in transfer):
        return "not_beneficial", "The locked candidate did not improve mean final validation CE at any planned larger width."
    return "mixed", "Benefits vary across widths or seeds; the recorded runs do not show consistent improvement."


def build_report(plan: dict, records: list[dict], selection: dict | None = None) -> dict:
    """Return JSON-serializable evidence without making or revising a selection.

    ``metric`` is final-budget validation CE, or held-out test CE for ``test``
    records. All original trial statuses are retained. Failed, nonfinite and
    duplicate successful observations never enter paired statistics. Statistics
    from partial pairs are descriptive and marked incomplete.
    """
    rows = [_clean_record(row) for row in records]
    selected = deepcopy(selection or {})
    baseline = _baseline_id(plan, rows, selected)
    if baseline:
        selected["baseline_id"] = baseline
    widths, seeds = list(plan.get("widths", [])), list(plan.get("seeds", []))
    comparisons = []
    if baseline and widths and selected.get("challenger_id"):
        comparisons.append(_comparison(rows, "confirm", widths[0], seeds,
                                       selected["challenger_id"], baseline))
    if baseline and selected.get("challenger_id"):
        comparisons.extend(_comparison(rows, "transfer", width, seeds,
                                       selected["challenger_id"], baseline) for width in widths[1:])
        confirm = next((item for item in comparisons if item["stage"] == "confirm"), None)
        for item in comparisons:
            item["candidate_is_locked_winner"] = item["candidate_id"] == selected.get("winner_id")
            item["mean_direction_reversed"] = (
                item["mean_delta"] * confirm["mean_delta"] < 0
                if item["stage"] == "transfer" and confirm is not None
                and item["mean_delta"] is not None and confirm["mean_delta"] is not None else None
            )
    if baseline and selected.get("winner_id"):
        # A held-out evaluation is optional until at least one test trial exists.
        # Once begun, every planned width/seed is shown, including missing ones.
        if any(row.get("stage") == "test" for row in rows):
            comparisons.extend(_comparison(rows, "test", width, seeds,
                                           selected["winner_id"], baseline) for width in widths)
    outcome, explanation = _transfer_outcome(comparisons, selected, widths)
    diagnostics = {"transfer_outcome": outcome, "transfer_explanation": explanation}
    purpose = plan.get("purpose", "research")
    if purpose == "smoke":
        outcome = "smoke_only"
        explanation = "These runs exercise the experiment machinery only. Their scores provide no scientific evidence of hyperparameter transfer; check trial statuses for successful execution."
    candidate_rates: dict[str, set[float]] = {}
    for row in rows:
        if row.get("candidate") and _finite_number(row.get("learning_rate")):
            candidate_rates.setdefault(row["candidate"], set()).add(row["learning_rate"])
    if baseline and _finite_number(plan.get("baseline_lr")):
        candidate_rates.setdefault(baseline, set()).add(plan["baseline_lr"])
    candidates = [{"candidate_id": candidate, "learning_rate": next(iter(rates)) if len(rates) == 1 else None}
                  for candidate, rates in candidate_rates.items()]
    candidates.sort(key=lambda row: (row["learning_rate"] if row["learning_rate"] is not None else math.inf,
                                     row["candidate_id"]))

    def token_total(heldout: bool) -> int | float:
        return sum(row["tokens_processed"] for row in rows
                   if (row.get("stage") == "test") == heldout
                   and _finite_number(row.get("tokens_processed")) and row["tokens_processed"] >= 0)

    training = plan.get("training", {})
    budget_fields = [training.get(key) for key in ("max_iters", "batch_size", "block_size")]
    tokens_per_trial = math.prod(budget_fields) if all(_finite_number(x) for x in budget_fields) else None
    return {
        "schema_version": 1,
        "name": plan.get("name", "Width transfer experiment"),
        "protocol": {
            "purpose": purpose,
            "objective": "final-budget cross-entropy in nats/character (lower is better)",
            "widths": widths,
            "seeds": seeds,
            "screen_seed": plan.get("screen_seed", 0),
            "baseline_lr": plan.get("baseline_lr"),
            "learning_rates": list(plan.get("learning_rates", [])),
            "min_improvement": plan.get("min_improvement", .005),
            "selection_rule": "Retain the challenger only if mean paired validation CE delta < -min_improvement and every paired delta < 0; otherwise retain the baseline. This is a predeclared heuristic, not a significance test.",
            "training": deepcopy(training),
            "tokens_per_training_trial": tokens_per_trial,
        },
        "selection": selected,
        "candidates": candidates,
        "status_counts": dict(sorted(Counter(row.get("status", "unknown") for row in rows).items())),
        "budget": {
            "limits": deepcopy(plan.get("limits", {})),
            "recorded_trials": len(rows),
            "recorded_training_tokens": token_total(heldout=False),
            "recorded_heldout_evaluation_tokens": token_total(heldout=True),
            "recorded_duration_seconds": sum(row["duration_seconds"] for row in rows
                                             if _finite_number(row.get("duration_seconds")) and row["duration_seconds"] >= 0),
        },
        "screen": [row for row in rows if row.get("stage") == "screen"],
        "comparisons": comparisons,
        "transfer_outcome": outcome,
        "transfer_explanation": explanation,
        "diagnostics": diagnostics,
        "caveats": [
            "The screen uses one seed and is provisional. Confirmation uses separate paired training seeds.",
            "Paired deltas are candidate CE minus baseline CE. Negative values favor the candidate; sample standard deviation describes seed variation, not a confidence guarantee.",
            "The candidate is locked using small-width validation. Larger-width and test results must not revise that selection.",
            "Held-out test scores are an audit of the locked choice, not another search objective. Reusing them for a new search consumes the holdout.",
            "These are local finite-width results for this corpus, optimizer, architecture, token budget and hardware. They do not establish transfer to production-scale LLMs, MoE or Muon.",
            "This experiment measures empirical hyperparameter transfer. It does not implement or validate the muP parameterization.",
            "Tokens in this character language model are characters. Training and held-out evaluation counts are reported separately; validation evaluation tokens are not included in these counts. Equal character budgets do not imply equal compute or wall time. Recorded duration and memory constraints should be considered separately.",
            "A paired comparison covers only the declared candidates. It does not estimate regret against an independently optimized larger-model optimum.",
        ],
        "trials": rows,
    }


def _fmt(value: Any, digits: int = 6) -> str:
    return f"{value:.{digits}g}" if _finite_number(value) else "—"


def _cell(value: Any) -> str:
    return str(value if value is not None else "—").replace("|", "\\|").replace("\n", " ")


def render_markdown(report: dict) -> str:
    """Render the evidence, including incomplete runs, for human inspection."""
    protocol, selected, budget = report["protocol"], report["selection"], report["budget"]
    learning_rates = {row["candidate_id"]: row["learning_rate"] for row in report["candidates"]}

    def candidate_label(candidate: str | None) -> str:
        lr = learning_rates.get(candidate)
        return f"LR {_fmt(lr)}" if lr is not None else _cell(candidate)

    lines = [
        f"# {_cell(report['name'])}", "",
        f"**Transfer outcome: {report['transfer_outcome']}.** {report['transfer_explanation']}", "",
        f"Objective: {protocol['objective']}. Widths: {_cell(protocol['widths'])}; paired seeds: {_cell(protocol['seeds'])}.", "",
        f"Planned tokens per training trial: {_fmt(protocol['tokens_per_training_trial'], 12)}. "
        f"Recorded trials: {budget['recorded_trials']}; recorded duration: {_fmt(budget['recorded_duration_seconds'])} seconds.", "",
        f"Recorded training characters: {_fmt(budget['recorded_training_tokens'], 12)}. "
        f"Recorded held-out evaluation characters: {_fmt(budget['recorded_heldout_evaluation_tokens'], 12)}. "
        "Validation evaluation characters are excluded from these two counts.", "",
        f"Session limits: `{_cell(budget['limits'])}`.", "",
        f"Statuses: {_cell(report['status_counts'])}.", "",
        f"Baseline: {candidate_label(selected.get('baseline_id'))}. Screen challenger: {candidate_label(selected.get('challenger_id'))}. "
        f"Locked small-width winner: {candidate_label(selected.get('winner_id'))}.", "",
        f"Selection rule: {protocol['selection_rule']} Minimum improvement: {_fmt(protocol['min_improvement'])} nats/character.", "",
        "| Candidate ID | Learning rate | Role |",
        "| --- | ---: | --- |",
    ]
    for row in report["candidates"]:
        roles = [label for key, label in [("baseline_id", "baseline"), ("challenger_id", "screen challenger"),
                                          ("winner_id", "locked winner")] if selected.get(key) == row["candidate_id"]]
        lines.append(f"| `{_cell(row['candidate_id'])}` | {_fmt(row['learning_rate'])} | {', '.join(roles) or 'screen candidate'} |")
    lines.extend([
        "",
        "## Paired evidence", "",
        "Negative deltas favor the candidate. Confirmation and transfer compare the challenger with the baseline; test compares the locked winner with the baseline. Incomplete comparisons cannot support a completed transfer outcome.", "",
        "| Stage | Width | Candidate | Pairs | Mean baseline CE | Mean candidate CE | Mean CE delta | Sample SD | Improved seeds | Missing / duplicate seeds |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in report["comparisons"]:
        lines.append(f"| {row['stage']} | {row['width']} | {candidate_label(row['candidate_id'])} | "
                     f"{row['n_pairs']}/{len(row['expected_seeds'])} | {_fmt(row['mean_baseline'])} | "
                     f"{_fmt(row['mean_candidate'])} | {_fmt(row['mean_delta'])} | "
                     f"{_fmt(row['sample_std_delta'])} | {row['improved_seeds']} | "
                     f"{_cell(row['missing_seeds'])} / {_cell(row['duplicate_seeds'])} |")
    if not report["comparisons"]:
        lines.extend(["", "No paired candidate comparison is available yet."])
    for row in report["comparisons"]:
        deltas = ", ".join(f"seed {pair['seed']}: {_fmt(pair['delta'])}" for pair in row["pairs"])
        reversal = " Mean ranking reversed relative to small-width confirmation." if row.get("mean_direction_reversed") else ""
        lines.extend(["", f"{row['stage']} at width {row['width']}: {deltas or 'no usable seed pairs'}.{reversal}"])
    lines.extend([
        "", "## Trial ledger", "",
        "Screen results are provisional single-seed evidence. Test rows use held-out test CE; all other rows use validation CE.", "",
        "| Trial | Stage | Width | Seed | Learning rate | Status | Final CE | Training chars | Held-out chars | Error |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ])
    for row in report["trials"]:
        lines.append(f"| {_cell(row.get('trial_id'))} | {_cell(row.get('stage'))} | {_cell(row.get('width'))} | "
                     f"{_cell(row.get('seed'))} | {_fmt(row.get('learning_rate'))} | {_cell(row.get('status'))} | "
                     f"{_fmt(row.get('metric'))} | {_fmt(row.get('tokens_processed') if row.get('stage') != 'test' else 0, 12)} | "
                     f"{_fmt(row.get('tokens_processed') if row.get('stage') == 'test' else 0, 12)} | {_cell(row.get('error', ''))} |")
    lines.extend(["", "## Interpretation limits", ""])
    lines.extend(f"- {caveat}" for caveat in report["caveats"])
    return "\n".join(lines) + "\n"
