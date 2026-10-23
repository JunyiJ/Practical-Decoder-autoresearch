from copy import deepcopy
import json
import math

import pytest

from src.research.report import build_report, render_markdown


def plan():
    return {"name": "LR transfer", "widths": [64, 128, 256], "seeds": [1, 2, 3],
            "screen_seed": 0, "baseline_lr": .001, "learning_rates": [.001, .003],
            "training": {"max_iters": 10, "batch_size": 2, "block_size": 8},
            "limits": {"max_trials": 23}}


def selection(winner="challenger"):
    return {"baseline_id": "baseline", "challenger_id": "challenger",
            "winner_id": winner, "selected_by": "confirm_mean_val_loss"}


def row(stage, width, seed, candidate, metric, **kwargs):
    return {"trial_id": f"{stage}_{width}_{seed}_{candidate}", "stage": stage,
            "width": width, "seed": seed, "candidate": candidate,
            "learning_rate": .001 if candidate == "baseline" else .003,
            "status": "completed", "metric": metric,
            "tokens_processed": 160, "duration_seconds": 2, **kwargs}


def complete_records():
    records = []
    for stage, widths in [("confirm", [64]), ("transfer", [128, 256])]:
        for width in widths:
            for seed in [1, 2, 3]:
                records += [row(stage, width, seed, "baseline", 2 + seed / 10),
                            row(stage, width, seed, "challenger", 1.9 + seed / 10)]
    return records


def test_pairing_uses_matching_seeds_and_descriptive_sample_std():
    records = [row("confirm", 64, 1, "baseline", 2),
               row("confirm", 64, 2, "baseline", 3),
               row("confirm", 64, 2, "challenger", 2.7),
               row("confirm", 64, 1, "challenger", 1.9)]
    comparison = build_report(plan(), records, selection())["comparisons"][0]
    assert comparison["mean_delta"] == pytest.approx(-.2)
    assert comparison["sample_std_delta"] == pytest.approx(math.sqrt(.02))
    assert comparison["missing_seeds"] == [3]
    assert not comparison["complete"]
    assert [pair["seed"] for pair in comparison["pairs"]] == [1, 2]


def test_incomplete_and_nonfinite_runs_cannot_establish_transfer():
    records = complete_records()
    records[-1]["metric"] = float("nan")
    records[-3]["status"] = "timeout"
    report = build_report(plan(), records, selection())
    comparison = report["comparisons"][-1]
    assert report["transfer_outcome"] == "incomplete"
    assert comparison["missing_seeds"] == [2, 3]
    assert comparison["sample_std_delta"] is None
    assert report["status_counts"]["timeout"] == 1
    json.dumps(report, allow_nan=False)


def test_duplicate_successful_seed_is_not_counted_twice():
    records = complete_records()
    records.append(dict(records[-1]))
    comparison = build_report(plan(), records, selection())["comparisons"][-1]
    assert comparison["duplicate_seeds"] == [3]
    assert comparison["n_pairs"] == 2
    assert not comparison["complete"]


def test_positive_transfer_requires_all_planned_width_seed_pairs():
    report = build_report(plan(), complete_records(), selection())
    assert report["transfer_outcome"] == "beneficial"
    assert all(row["complete"] for row in report["comparisons"])
    assert report["protocol"]["tokens_per_training_trial"] == 160


def test_sign_disagreement_is_explicitly_mixed():
    records = complete_records()
    records[-1]["metric"] = records[-2]["metric"] + .05
    assert build_report(plan(), records, selection())["transfer_outcome"] == "mixed"
    for record in records:
        if record["stage"] == "transfer" and record["candidate"] == "challenger":
            record["metric"] = 20
    assert build_report(plan(), records, selection())["transfer_outcome"] == "not_beneficial"


def test_report_never_reselects_using_transfer_or_test_results():
    records = complete_records()
    for width in [64, 128, 256]:
        for seed in [1, 2, 3]:
            records += [row("test", width, seed, "baseline", 1),
                        row("test", width, seed, "challenger", 20)]
    selected, original = selection(), deepcopy(selection())
    report = build_report(plan(), records, selected)
    assert report["selection"] == original == selected
    assert report["transfer_outcome"] == "beneficial"
    assert all(item["candidate_id"] == "challenger" for item in report["comparisons"])
    assert all(item["mean_delta"] > 0 for item in report["comparisons"] if item["stage"] == "test")


def test_baseline_retention_is_not_reported_as_success():
    report = build_report(plan(), complete_records(), selection("baseline"))
    assert report["transfer_outcome"] == "not_beneficial"
    assert "retained the baseline" in report["transfer_explanation"]
    assert all(item["mean_delta"] < 0 for item in report["comparisons"] if item["stage"] == "transfer")
    assert all(not item["candidate_is_locked_winner"] for item in report["comparisons"] if item["stage"] == "transfer")


def test_screen_only_report_is_provisional_and_does_not_choose():
    records = [row("screen", 64, 0, "baseline", 2), row("screen", 64, 0, "challenger", 1)]
    report = build_report(plan(), records)
    assert report["selection"] == {"baseline_id": "baseline"}
    assert report["comparisons"] == []
    assert report["transfer_outcome"] == "incomplete"
    markdown = render_markdown(report)
    assert "provisional" in markdown
    assert "does not implement or validate the muP" in markdown
    assert "screen_64_0_baseline" in markdown


def test_seed_not_in_protocol_does_not_change_statistics():
    records = complete_records() + [row("transfer", 256, 999, "baseline", 10),
                                    row("transfer", 256, 999, "challenger", 100)]
    report = build_report(plan(), records, selection())
    assert report["transfer_outcome"] == "beneficial"
    assert report["comparisons"][-1]["n_pairs"] == 3


def test_rank_reversal_is_visible_without_overriding_baseline_selection():
    records = complete_records()
    for record in records:
        if record["stage"] == "confirm" and record["candidate"] == "challenger":
            record["metric"] += .2
    report = build_report(plan(), records, selection("baseline"))
    assert report["selection"]["winner_id"] == "baseline"
    assert all(item["mean_direction_reversed"] for item in report["comparisons"] if item["stage"] == "transfer")
    assert "Mean ranking reversed" in render_markdown(report)
    assert report["protocol"]["min_improvement"] == .005


def test_smoke_scores_never_claim_scientific_transfer():
    smoke_plan = {**plan(), "purpose": "smoke"}
    report = build_report(smoke_plan, complete_records(), selection())
    assert report["transfer_outcome"] == "smoke_only"
    assert report["diagnostics"]["transfer_outcome"] == "beneficial"
    assert "no scientific evidence" in report["transfer_explanation"]
    markdown = render_markdown(report)
    assert "**Transfer outcome: smoke_only." in markdown
    assert "**Transfer outcome: beneficial." not in markdown


def test_training_and_holdout_character_counts_are_separate():
    records = complete_records() + [row("test", 64, 1, "baseline", 2, tokens_processed=80)]
    report = build_report(plan(), records, selection())
    assert report["budget"]["recorded_training_tokens"] == 18 * 160
    assert report["budget"]["recorded_heldout_evaluation_tokens"] == 80
    markdown = render_markdown(report)
    assert "Recorded training characters: 2880." in markdown
    assert "Recorded held-out evaluation characters: 80." in markdown


def test_opaque_candidate_ids_are_mapped_to_learning_rates():
    records = complete_records()
    selected = selection()
    for record in records:
        record["candidate"] = {"baseline": "lr_a1b2c3", "challenger": "lr_d4e5f6"}[record["candidate"]]
    for key in ["baseline_id", "challenger_id", "winner_id"]:
        selected[key] = {"baseline": "lr_a1b2c3", "challenger": "lr_d4e5f6"}[selected[key]]
    report = build_report(plan(), records, selected)
    markdown = render_markdown(report)
    assert "Baseline: LR 0.001." in markdown
    assert "Screen challenger: LR 0.003." in markdown
    assert "| `lr_a1b2c3` | 0.001 | baseline |" in markdown
    assert "| Mean baseline CE | Mean candidate CE |" in markdown
    assert "| confirm | 64 | LR 0.003 | 3/3 | 2.2 | 2.1 |" in markdown
