"""Forecast validation and comparison use synthetic numbers only."""
from copy import deepcopy
import json

import pytest

from src.research.team.predictions import METRICS, compare_prediction, validate_prediction


def targets():
    return [{"name": "width-64", "width": 64, "n_heads": 4},
            {"name": "width-128", "width": 128, "n_heads": 8}]


def prediction():
    return {"mechanism": "A plausible intervention may alter optimization and resource use.",
            "assumptions": ["The same dataset, device, and token budget are used."],
            "estimates": [{"target": target["name"], "metric": metric, "expected": 0., "lower": -1., "upper": 1.,
                           "confidence": "low", "rationale": "A synthetic test fixture, not an empirical estimate."}
                          for target in targets() for metric in METRICS]}


def test_full_target_metric_coverage_is_required_without_mutating_input():
    original = prediction()
    checked = validate_prediction(original, targets())
    assert checked == original and checked is not original
    checked["estimates"][0]["expected"] = -.5
    assert original["estimates"][0]["expected"] == 0
    for missing in range(len(original["estimates"])):
        incomplete = deepcopy(original)
        incomplete["estimates"].pop(missing)
        with pytest.raises(ValueError, match="cover every target and metric"):
            validate_prediction(incomplete, targets())


@pytest.mark.parametrize("change", ["duplicate", "unknown_target", "unknown_metric", "unknown_field"])
def test_ambiguous_or_unregistered_estimates_are_rejected(change):
    data = prediction()
    if change == "duplicate":
        data["estimates"].append(deepcopy(data["estimates"][0]))
    elif change == "unknown_target":
        data["estimates"][0]["target"] = "width-999"
    elif change == "unknown_metric":
        data["estimates"][0]["metric"] = "latency_seconds"
    else:
        data["estimates"][0]["posthoc_adjustment"] = 1
    with pytest.raises(ValueError):
        validate_prediction(data, targets())


@pytest.mark.parametrize("metric", ["training_throughput_pct", "peak_rss_pct"])
def test_resource_percent_cannot_fall_below_physical_floor(metric):
    data = prediction()
    estimate = next(row for row in data["estimates"] if row["metric"] == metric)
    estimate.update(expected=-100, lower=-100, upper=-90)
    validate_prediction(data, targets())
    estimate["lower"] = -100.01
    with pytest.raises(ValueError, match="more than 100 percent"):
        validate_prediction(data, targets())


def test_ce_delta_has_no_resource_percent_floor_and_unknown_forecasts_remain_null():
    data = prediction()
    for estimate in data["estimates"]:
        if estimate["metric"] == "validation_ce_delta":
            estimate.update(expected=-101, lower=-102, upper=-100)
        else:
            estimate.update(expected=None, lower=None, upper=None)
    result = validate_prediction(data, targets())
    assert result["estimates"][0]["expected"] == -101
    assert result["estimates"][1]["expected"] is None


@pytest.mark.parametrize("updates", [
    {"expected": None}, {"lower": None}, {"upper": None}, {"lower": .1}, {"upper": -.1},
    {"expected": True}, {"expected": float("nan")}, {"upper": float("inf")},
])
def test_mixed_nulls_bad_ranges_and_nonfinite_values_are_rejected(updates):
    data = prediction()
    data["estimates"][0].update(updates)
    with pytest.raises(ValueError):
        validate_prediction(data, targets())


def test_compare_reports_ranges_unquantified_missing_and_errors_without_changing_inputs():
    data = prediction()
    data["estimates"][3].update(expected=None, lower=None, upper=None)
    actuals = [-2, 2, .5, 8, 0]
    performance = [{"id": f"{row['target']}:{row['metric']}", "complete": True,
                    "mean_delta": actual, "missing_seeds": []}
                   for row, actual in zip(data["estimates"], actuals)]
    original, original_performance = deepcopy(data), deepcopy(performance)
    report = compare_prediction(data, performance)
    assert [row["verdict"] for row in report["comparisons"]] == [
        "below_range", "above_range", "within_range", "unquantified", "within_range", "unavailable"]
    assert [row["error"] for row in report["comparisons"]] == [-2, 2, .5, None, 0, None]
    assert report["comparisons"][-1]["observed_mean_delta"] is None
    assert data == original and performance == original_performance
    assert "subjective forecasts" in report["caveat"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("actual,complete", [(float("nan"), True), (float("inf"), True),
    (True, True), ("0.1", True), (None, True), (.1, False), (.1, 1)])
def test_incomplete_or_non_numeric_results_never_become_observed_zero(actual, complete):
    data = prediction()
    report = compare_prediction(data, [{"id": "width-64:validation_ce_delta", "complete": complete,
        "mean_delta": actual, "missing_seeds": [2]}])
    first = report["comparisons"][0]
    assert first["verdict"] == "unavailable" and first["complete"] is False
    assert first["observed_mean_delta"] is None and first["error"] is None
    assert first["missing_seeds"] == [2]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("actual", [-1, 1])
def test_registered_range_endpoints_are_inclusive(actual):
    report = compare_prediction(prediction(), [{"id": "width-64:validation_ce_delta", "complete": True, "mean_delta": actual}])
    assert report["comparisons"][0]["verdict"] == "within_range"


def test_duplicate_performance_ids_cannot_choose_a_favorable_observation():
    observations = [{"id": "width-64:validation_ce_delta", "complete": True, "mean_delta": -2},
                    {"id": "width-64:validation_ce_delta", "complete": True, "mean_delta": 2}]
    for order in (observations, list(reversed(observations))):
        first = compare_prediction(prediction(), order)["comparisons"][0]
        assert first["verdict"] == "unavailable" and first["complete"] is False
        assert first["observed_mean_delta"] is None
