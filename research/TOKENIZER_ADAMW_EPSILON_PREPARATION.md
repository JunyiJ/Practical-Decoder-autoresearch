# AdamW epsilon preparation

Select epsilon1e-6 versus incumbent1e-8, holding betas(.9,.999), Muon,
independent decay.00006, model, group membership and width learning rates fixed.
Do not infer benefit or typical full-model denominator values from tiny generated
gradients. Full-model identity must reject foreign epsilon before optimizer child
loading; preserve exact shared initialization/logits/RNG and all unrelated groups.

## November 2 — Final-bias quality closed; AdamW epsilon prerequisite passed

All six1200-update endpoints and three paired full/cache/reset inference reviews
passed. Fixed-zero final LayerNorm bias improved all three seeds by0.01054424,
0.01051816 and0.00685128 nats/token; mean delta-0.009304557298255875. All three
conditional text upper bounds are negative, but the frozen mean improvement
threshold0.02 was not reached. CLOSED_NOT_ADOPTED: retain learned final bias;
no threshold change, extra seeds or rescue. These are three trained seeds on
reused validation text, not holdout or population-seed confidence. Terminal STOP
closes every quality command. All six fresh raw/H200 pairs3,355,558,167B remain
protected; all191 historical RAW archives and H100/H200 originals remain intact.

At the boundary, reviewed architecture (final LayerNorm learned scale versus
fixed unit scale, with learned bias retained) and bounded HP. Defer the scale
ablation and select the previously deferred AdamW epsilon1e-6 versus1e-8 with
betas(.9,.999),Muon,independent decay.00006 and model fixed. Four generated CPU
FP64/FP32 real optimizer cases passed independent moments/bias-correction/update
checks including tiny/zero/ordinary gradients (maxerror2.4649105268004234e-7),
exact shrinkage-before-child order, serialized resume, foreign epsilon rejection,
gradient-absent preservation, unchanged RNG and prescribed-gradient moment/Muon
parity. All813 bindings unchanged,812 predecessors preserved. This establishes
only a generated CPU prerequisite; no full-model/MPS/EMA/H200/cache or quality
claim, scientific seed reservation or active worker.

Next explicit additive full-model epsilon policy/group/checkpoint identity,
CPU/MPS repeat/resume/EMA/RNG/clipping-history/cache integration; then actual
bounded workers, full capacity/census, finite pilots and exactly three fresh pairs
at commonLR.00125/equal zeroLRsearch/original gates. Closure free10,054,123,520B
is an observation, not new-panel readiness. Source/small numerical work proceeds.
Evidence: final-ln-bias-preparation-001/quality001-final-review.json,
quality001-closure.json, quality001-fresh-inference-pair0/1/2-review.json;
adamw-epsilon-preparation-001/direction-decision.json and optimizer-component001-*.
