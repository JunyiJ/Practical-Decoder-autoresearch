"""Pre-execution forecasts and deterministic checks against paired measurements."""
from __future__ import annotations

import copy
import math


METRICS = {
    "validation_ce_delta": {"unit": "nats/character", "better": "lower",
                            "definition": "Mean paired candidate minus baseline final validation CE"},
    "training_throughput_pct": {"unit": "%", "better": "higher",
                                "definition": "Mean paired 100 * (candidate training tokens/second / baseline - 1)"},
    "peak_rss_pct": {"unit": "%", "better": "lower",
                     "definition": "Mean paired 100 * (candidate process-lifetime peak RSS / baseline - 1); not accelerator peak memory"},
}


def validate_prediction(prediction: dict, targets: list[dict]) -> dict:
    """Require one quality and two resource forecasts for every registered target."""
    # Reuse the role's schema and cross-field validation without inferring values.
    from .roles import _validate, schema_for

    _validate(prediction, schema_for("idea")["properties"]["prediction"], "prediction")
    expected = {(target["name"], metric) for target in targets for metric in METRICS}
    seen = set()
    for estimate in prediction["estimates"]:
        key = (estimate["target"], estimate["metric"])
        if key in seen:
            raise ValueError(f"Duplicate prediction: {key}")
        seen.add(key)
        values = [estimate[k] for k in ("lower", "expected", "upper")]
        if any(value is None for value in values):
            if not all(value is None for value in values):
                raise ValueError("Prediction ranges must be all numeric or all null")
        else:
            if not all(not isinstance(v, bool) and isinstance(v, (float, int)) and math.isfinite(v) for v in values):
                raise ValueError("Predictions must be finite numbers")
            if not values[0] <= values[1] <= values[2]:
                raise ValueError("Prediction must satisfy lower <= expected <= upper")
            if estimate["metric"].endswith("_pct") and values[0] < -100:
                raise ValueError("A positive resource measurement cannot decrease by more than 100 percent")
    if seen != expected:
        raise ValueError(f"Predictions must cover every target and metric; missing={sorted(expected - seen)}, unexpected={sorted(seen - expected)}")
    return copy.deepcopy(prediction)


def compare_prediction(prediction: dict, performance: list[dict]) -> dict:
    """Compare only complete paired means, without changing empirical selection."""
    by_id = {row["id"]: row for row in performance}
    duplicates = {row["id"] for row in performance if sum(other["id"] == row["id"] for other in performance) != 1}
    checks = []
    for estimate in prediction["estimates"]:
        identity = f"{estimate['target']}:{estimate['metric']}"
        row = {} if identity in duplicates else by_id.get(identity, {})
        actual = row.get("mean_delta")
        complete = row.get("complete") is True and isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isfinite(actual)
        actual = actual if complete else None
        expected = estimate["expected"]
        if not complete:
            verdict = "unavailable"
        elif expected is None:
            verdict = "unquantified"
        elif actual < estimate["lower"]:
            verdict = "below_range"
        elif actual > estimate["upper"]:
            verdict = "above_range"
        else:
            verdict = "within_range"
        checks.append({"id": identity, "target": estimate["target"], "metric": estimate["metric"],
                       "unit": METRICS[estimate["metric"]]["unit"], "prediction": copy.deepcopy(estimate),
                       "complete": complete, "observed_mean_delta": actual, "verdict": verdict,
                       "error": actual - expected if complete and expected is not None else None,
                       "missing_seeds": row.get("missing_seeds", [])})
    return {"version": 1, "reference": "fixed campaign baseline at each target; paired seeds",
            "metrics": copy.deepcopy(METRICS), "comparisons": checks,
            "caveat": "Ranges are preregistered subjective forecasts, not statistical confidence intervals. Missing measurements are not zeros; resource deltas do not establish a causal mechanism."}
