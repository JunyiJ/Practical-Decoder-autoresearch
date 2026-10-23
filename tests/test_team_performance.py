"""Pure summaries of recorded evidence; no models, workers or training."""
from copy import deepcopy
import math

import pytest

from src.research.team.performance import summarize_performance


@pytest.fixture
def evidence():
    spec = {"targets": [{"name": "small", "width": 64, "n_heads": 2},
                        {"name": "large", "width": 128, "n_heads": 4}], "seeds": [11, 22]}
    records = []
    for i, target in enumerate(spec["targets"]):
        for j, seed in enumerate(spec["seeds"]):
            for arm in ("baseline", "candidate"):
                records.append({"target": target["name"], "width": target["width"],
                                "stage": "local" if i == 0 else "transfer", "seed": seed,
                                "arm": arm, "status": "completed", "metric": (2., 4.)[j] - (.2 if arm == "candidate" else 0),
                                "tokens_processed": 100, "training_time_sec": ((10., 20.) if arm == "baseline" else (5., 40.))[j],
                                "rss_mb": ((100., 400.) if arm == "baseline" else (80., 500.))[j],
                                "accel_mem_mb": 20., "accel_mem_kind": "mps_current_allocated"})
    return spec, records


def _row(rows, metric, target="small"):
    return next(row for row in rows if row["id"] == f"{target}:{metric}")


def _candidate(records, target="small", seed=22):
    return next(row for row in records if row["target"] == target and row["seed"] == seed and row["arm"] == "candidate")


def test_registered_targets_have_exactly_three_comparable_metrics(evidence):
    spec, records = evidence
    original = deepcopy((spec, records))
    rows = summarize_performance(spec, records)
    assert (spec, records) == original
    assert len(rows) == 6
    assert all(row["complete"] and row["missing_seeds"] == [] for row in rows)
    assert {row["metric"] for row in rows} == {"validation_ce_delta", "training_throughput_pct", "peak_rss_pct"}
    assert [row["target"] for row in rows] == ["small"] * 3 + ["large"] * 3
    assert not any("accel" in row["metric"] for row in rows)


def test_quality_is_an_absolute_ce_delta(evidence):
    row = _row(summarize_performance(*evidence), "validation_ce_delta")
    assert row["unit"] == "nats/character"
    assert row["better"] == "lower"
    assert row["mean_delta"] == pytest.approx(-.2)
    assert row["baseline_mean"] == 3
    assert row["candidate_mean"] == pytest.approx(2.8)
    assert row["sample_std_delta"] == pytest.approx(0)


def test_throughput_averages_paired_percentages_not_ratio_of_means(evidence):
    row = _row(summarize_performance(*evidence), "training_throughput_pct")
    assert row["unit"] == "%" and row["better"] == "higher"
    assert row["pairs"] == [{"seed": 11, "baseline": 10., "candidate": 20., "delta": 100.},
                            {"seed": 22, "baseline": 5., "candidate": 2.5, "delta": -50.}]
    assert row["mean_delta"] == 25
    assert row["baseline_mean"] == 7.5
    assert row["candidate_mean"] == 11.25
    assert row["sample_std_delta"] == pytest.approx(math.sqrt(11250))


def test_rss_is_paired_process_lifetime_peak_percentage(evidence):
    row = _row(summarize_performance(*evidence), "peak_rss_pct")
    assert row["unit"] == "%" and row["better"] == "lower"
    assert [pair["delta"] for pair in row["pairs"]] == pytest.approx([-20, 25])
    assert row["mean_delta"] == pytest.approx(2.5)
    assert row["baseline_mean"] == 250
    assert row["candidate_mean"] == 290


@pytest.mark.parametrize("field,metric,invalid", [
    ("metric", "validation_ce_delta", None),
    ("metric", "validation_ce_delta", False),
    ("metric", "validation_ce_delta", float("nan")),
    ("metric", "validation_ce_delta", float("inf")),
    ("metric", "validation_ce_delta", -1),
    ("training_time_sec", "training_throughput_pct", None),
    ("training_time_sec", "training_throughput_pct", False),
    ("training_time_sec", "training_throughput_pct", 0),
    ("training_time_sec", "training_throughput_pct", -1),
    ("training_time_sec", "training_throughput_pct", float("inf")),
    ("training_time_sec", "training_throughput_pct", float("nan")),
    ("tokens_processed", "training_throughput_pct", True),
    ("tokens_processed", "training_throughput_pct", 0),
    ("tokens_processed", "training_throughput_pct", 99),
    ("tokens_processed", "training_throughput_pct", float("nan")),
    ("rss_mb", "peak_rss_pct", None),
    ("rss_mb", "peak_rss_pct", True),
    ("rss_mb", "peak_rss_pct", 0),
    ("rss_mb", "peak_rss_pct", -1),
    ("rss_mb", "peak_rss_pct", float("inf")),
    ("rss_mb", "peak_rss_pct", float("nan")),
])
def test_invalid_resource_or_quality_value_is_missing_not_zero(evidence, field, metric, invalid):
    spec, records = evidence
    _candidate(records)[field] = invalid
    rows = summarize_performance(spec, records)
    row = _row(rows, metric)
    assert row["complete"] is False
    assert row["missing_seeds"] == [22]
    assert len(row["pairs"]) == 1
    assert row["mean_delta"] is None and row["sample_std_delta"] is None
    assert "baseline_mean" not in row and "candidate_mean" not in row
    assert all(row["complete"] for row in rows if row["target"] == "large" or row["metric"] != metric)


@pytest.mark.parametrize("status", ["failed", "running", "timeout", "invalid", "budget_exhausted"])
def test_noncompleted_trials_cannot_enter_any_metric(evidence, status):
    spec, records = evidence
    _candidate(records)["status"] = status
    rows = summarize_performance(spec, records)
    assert all(row["missing_seeds"] == [22] for row in rows if row["target"] == "small")
    assert all(row["complete"] for row in rows if row["target"] == "large")


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_ambiguous_duplicate_is_never_selected_or_averaged(evidence, status):
    spec, records = evidence
    duplicate = deepcopy(_candidate(records))
    duplicate.update(status=status, metric=0, training_time_sec=.1, rss_mb=1)
    records.append(duplicate)
    assert all(row["missing_seeds"] == [22] for row in summarize_performance(spec, records)
               if row["target"] == "small")


@pytest.mark.parametrize("change", [
    {"stage": "test"}, {"stage": "transfer"}, {"width": 128},
    {"target": "large"}, {"seed": 99}, {"seed": 22.0}, {"width": 64.0},
])
def test_other_stage_width_target_or_seed_cannot_fill_a_missing_pair(evidence, change):
    spec, records = evidence
    _candidate(records).update(change)
    assert all(row["missing_seeds"] == [22] for row in summarize_performance(spec, records)
               if row["target"] == "small")


def test_heldout_and_unregistered_rows_do_not_contaminate_complete_pairs(evidence):
    spec, records = evidence
    expected = summarize_performance(spec, records)
    extras = deepcopy(records)
    for row in extras:
        row.update(stage="test", metric=10000., training_time_sec=.001, rss_mb=1)
    assert summarize_performance(spec, records + extras) == expected


def test_failed_integrity_excludes_every_observation(evidence):
    spec, records = evidence
    rows = summarize_performance(spec, records, integrity=False)
    assert all(not row["complete"] and row["missing_seeds"] == [11, 22] and row["pairs"] == [] for row in rows)
    assert all(row["mean_delta"] is None and row["sample_std_delta"] is None for row in rows)
    assert all("baseline_mean" not in row and "candidate_mean" not in row for row in rows)


def test_absent_fields_and_overflowing_derived_metrics_are_missing(evidence):
    spec, records = evidence
    del _candidate(records)["training_time_sec"]
    assert _row(summarize_performance(spec, records), "training_throughput_pct")["missing_seeds"] == [22]
    _candidate(records)["training_time_sec"] = 1e-320
    assert _row(summarize_performance(spec, records), "training_throughput_pct")["missing_seeds"] == [22]
    _candidate(records)["rss_mb"] = 10 ** 1000
    assert _row(summarize_performance(spec, records), "peak_rss_pct")["missing_seeds"] == [22]


def test_unrepresentable_standard_deviation_is_not_infinity(evidence):
    spec, records = evidence
    for record in records:
        if record["target"] == "small":
            record["metric"] = 1.7e308 if (record["arm"] == "candidate") == (record["seed"] == 11) else 0.
    row = _row(summarize_performance(spec, records), "validation_ce_delta")
    assert row["complete"] and row["mean_delta"] == 0
    assert row["sample_std_delta"] is None
