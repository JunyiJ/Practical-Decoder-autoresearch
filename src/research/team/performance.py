"""Pure paired quality, throughput, and host-memory summaries.

Resource improvements remain separate from quality and never select a winner.
Each target uses only its registered training-stage observations. Held-out test
measurements and measurements from another width cannot fill missing pairs.
"""
from __future__ import annotations

import math
from statistics import mean, stdev


_METRICS = (
    ("validation_ce_delta", "nats/character", "lower"),
    ("training_throughput_pct", "%", "higher"),
    ("peak_rss_pct", "%", "lower"),
)


def _finite(value, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return (math.isfinite(value) and (not positive or value > 0)
                and (not nonnegative or value >= 0))
    except (OverflowError, ValueError):
        return False


def _values(metric, baseline, candidate):
    if metric == "validation_ce_delta":
        values = (baseline.get("metric"), candidate.get("metric"))
        if not all(_finite(v, nonnegative=True) for v in values):
            return None
    elif metric == "training_throughput_pct":
        tokens = (baseline.get("tokens_processed"), candidate.get("tokens_processed"))
        seconds = (baseline.get("training_time_sec"), candidate.get("training_time_sec"))
        if (not all(_finite(v, positive=True) for v in tokens + seconds)
                or tokens[0] != tokens[1]):
            return None
        values = (tokens[0] / seconds[0], tokens[1] / seconds[1])
        if not all(_finite(v, positive=True) for v in values):
            return None
    else:
        values = (baseline.get("rss_mb"), candidate.get("rss_mb"))
        if not all(_finite(v, positive=True) for v in values):
            return None
    a, b = values
    delta = b - a if metric == "validation_ce_delta" else 100 * (b / a - 1)
    if not _finite(delta):
        return None
    return float(a), float(b), float(delta)


def _sample_std(values):
    if len(values) < 2:
        return None
    try:
        value = stdev(values)
    except (OverflowError, ValueError):
        return None
    return value if _finite(value) else None


def summarize_performance(spec: dict, records: list[dict], integrity: bool = True) -> list[dict]:
    """Summarize all registered targets using paired training observations.

    ``spec`` must provide validated ``targets`` and distinct ``seeds``. The first
    target uses stage ``local``; all subsequent targets use stage ``transfer``.
    Exactly one completed record must exist for each arm/seed/target/width; a
    duplicate identity is ambiguous even if one duplicate failed. Each metric
    validates its own fields without turning missing/invalid values into zero.

    Pair and aggregate arm values are CE in nats/character, training characters
    per second, or process-lifetime peak RSS in MiB. ``unit`` describes the delta:
    quality uses candidate minus baseline; resource deltas use the paired
    percentage ``100 * (candidate / baseline - 1)``. The percentage mean is the
    mean of seed percentages, not a ratio of aggregate means. Training throughput
    excludes evaluation/checkpoint time because ``training_time_sec`` does.

    Partial pairs remain visible, but aggregate means/std are withheld until all
    declared pairs are valid. Failed evidence integrity makes all pairs missing.
    Accelerator memory stays in raw records because MPS current allocation and
    CUDA peak allocation are different measurements.
    """
    rows = []
    seeds = spec["seeds"]
    for index, target in enumerate(spec["targets"]):
        stage = "local" if index == 0 else "transfer"
        observations = {}
        if integrity is True:
            for record in records:
                if (record.get("target") != target["name"] or record.get("stage") != stage
                        or type(record.get("width")) is not int or record["width"] != target["width"]
                        or type(record.get("seed")) is not int or record["seed"] not in seeds
                        or record.get("arm") not in {"baseline", "candidate"}):
                    continue
                observations.setdefault((record["arm"], record["seed"]), []).append(record)
        for metric, unit, better in _METRICS:
            pairs, missing = [], []
            for seed in seeds:
                groups = [observations.get((arm, seed), []) for arm in ("baseline", "candidate")]
                if any(len(group) != 1 or group[0].get("status") != "completed" for group in groups):
                    missing.append(seed)
                    continue
                values = _values(metric, groups[0][0], groups[1][0])
                if values is None:
                    missing.append(seed)
                    continue
                a, b, delta = values
                pairs.append({"seed": seed, "baseline": a, "candidate": b, "delta": delta})
            complete = bool(seeds) and not missing
            deltas = [pair["delta"] for pair in pairs]
            row = {"id": f"{target['name']}:{metric}", "target": target["name"],
                   "width": target["width"], "metric": metric, "unit": unit, "better": better,
                   "complete": complete, "pairs": pairs, "missing_seeds": missing,
                   "mean_delta": mean(deltas) if complete else None,
                   "sample_std_delta": _sample_std(deltas) if complete else None}
            if complete:
                row.update(baseline_mean=mean(pair["baseline"] for pair in pairs),
                           candidate_mean=mean(pair["candidate"] for pair in pairs))
            rows.append(row)
    return rows
