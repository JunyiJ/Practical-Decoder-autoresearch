# Fixed-LR decay quality comparison

This additive entry compares independent decay0.00006 with0 at common peak
LR0.00125. It runs exactly six1200-update trajectories in three fresh pairs,
ordered decay/zero, zero/decay, decay/zero. There is no LR screening, selection,
rescue, or quality-checkpoint retirement command. All six raw/H200 pairs remain
protected. The question concerns a standard recipe, not separately tuned optima.

Run only after the external quality freeze, with its exact SHA256:

```
python -m research.decay_quality_v2.quality_execution run --slot 0 --freeze-sha256 SHA
```

Use the strict four-thread MPS environment recorded in the freeze. Inspect
actual processes and accepted anchors before dispatching the next serial slot.
A missing acceptance or STOP prevents retries; never duplicate an active worker.
Each dispatch rechecks source/data bindings and complete remaining panel capacity.

The accepted resource pilots were re-audited through quality-integration002,
including actual optimizer/EMA/RNG/full-validation/cache checks. Integration001
failed before numerical tests because a metadata hash helper rejected large
checkpoint payloads; its source/log and v1 namespace remain closed. The v2
integration uses the already tested weight-aware hash helper. No model or
scientific design changed as a recovery.

Read-only three-pair inference and independent final statistics are prepared in
decay-preparation-001/quality001-fresh-inference002.py and
quality001-final-review002.py. They require all six accepted endpoints and the
exact freeze. Original strict gates, closed holdout and resource guards apply.
