# Native MLA elementwise gate: closed without adoption

**CLOSED_NOT_ADOPTED. Retain the unchanged native dense MLA incumbent.**
All nine LR screens and nine fresh trajectories independently passed complete
RAW/H200, optimizer/EMA/RNG/private sampler, clipping, parameter exposure, full
34,278-target CPU/MPS validation and cache/reset acceptance. Every trajectory
completed 1,200 updates, 1,228,800 training targets and 41 H200 snapshots.
Each arm selected LR0.00125 from the three frozen choices before confirmation.
All nine fresh serving checks and independent final statistics completed.

| Fresh seed | Unchanged H200 CE | Fixed-half H200 CE | Learned gate H200 CE | Gate−unchanged | Gate−half |
|---|---:|---:|---:|---:|---:|
| 200120011 | 4.693034389 | 4.705853195 | 4.698384407 | +0.005350018 | -0.007468787 |
| 200120012 | 4.674625290 | 4.689046664 | 4.678954365 | +0.004329076 | -0.010092299 |
| 200120013 | 4.697226174 | 4.715179192 | 4.705421579 | +0.008195404 | -0.009757613 |

Mean gate-minus-unchanged is **+0.005958166082 nat/native-token**; all three
paired seeds worsen. Mean gate-minus-half is **−0.009106233077**; all three
improve, but this misses the required mean improvement beyond0.02. All three
joint adoption gates are false. No fixed-half fallback, cost-only adoption,
new LR, seed replacement or gate/optimizer/horizon rescue follows.

The six conditional-text upper bounds (2048/4096/8192 native-target blocks) are
+0.008041515/+0.008369735/+0.007117054 versus unchanged and
−0.008235687/−0.008203264/−0.008275884 versus half. Independent arithmetic agreed
within1e-12. These condition on the trained seeds and reused validation text;
they are not holdout or population-seed confidence. The holdout stayed closed.

Saved-metadata analysis finds RAW gate-minus-unchanged differences positive on
all three seeds, mean+0.002988756134; RAW gate-minus-half mean−0.007905583527.
All nine fresh H200 losses exceed their RAW endpoint losses. RAW is diagnostic,
not an alternative selector. Averaging did not create the ranking reversal;
this does not identify an EMA/horizon optimum or justify replaying the study.
Five sparse online samples do not establish overfitting or its cause.
Fresh clipped-update counts are22/29/20 unchanged,3/3/1 half,3/2/1 gate out of
1,200. Lower clipping counts did not establish better quality or prove an
optimization defect. All88/88/92 learned tensors have complete update exposure.

All fresh serving checks passed B1/B8 full/cache/reset, with unchanged cache
196,608bytes/item. Timing is serial local descriptive evidence, not a selection
or cost-only adoption route. Max accepted CPU/MPS CE difference2.79931408134e-5
passes3e-5. All1,084 frozen source hashes remain unchanged.

The original controller15959/session6668 exited0. Root STOP
complete_finite_scientific_panel closes every original training, serving and
review command; never replay them. All36 original RAW/H200 files totaling
10,169,313,948bytes were hash-verified and remain protected, including screens.
173 closed metadata files totaling2,315,265bytes were mirrored with zero weight
copies and zero deletions. All12 prerequisite weights remain protected.
Closure creates no new retirement eligibility. Historical failed harnesses and
STOPs are unchanged. Source snapshots and earlier preparations are historical.

Evidence: [original final review](tokenizer-evidence/post-diagnostic-boundary-001/gate-quality001-final-review.json),
[independent statistics](tokenizer-evidence/post-diagnostic-boundary-001/gate-quality001-independent-final-review.json),
[closure and retention audit](tokenizer-evidence/post-diagnostic-boundary-001/gate-quality001-closure001/completed.json),
[saved-record analysis](tokenizer-evidence/post-diagnostic-boundary-001/gate-quality001-existing-analysis001.json).

Continue through the [new research boundary](tokenizer-evidence/post-gate-boundary-001/evidence-review.md).
No numerical worker is active and no next scientific panel is admitted.
