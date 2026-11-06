# Body-allocation structural comparison results

Decision: **CLOSED_NOT_ADOPTED**. Keep the4×256,H672 incumbent with learned final LayerNorm scale/bias and adopted AdamW epsilon1e-6 recipe. The seven-layer192-wide candidate does not pass the prospectively frozen gates against both controls.

All nine LR screens and all nine fresh trajectories completed and independently passed RAW/H200, optimizer/EMA/RNG/sampler, complete clipping, full-validation and cache/reset checks. All three arms selected nominal LR.00125 from the same three-choice allowance before fresh confirmation. Every trajectory used1200updates/1,228,800targets/41H200snapshots;no retry, extra rate or replacement seed.

| Fresh seed | Incumbent CE | Shallow192 CE | Deeper192 CE | Deeper−incumbent | Deeper−shallow |
|---|---:|---:|---:|---:|---:|
| 200080011 | 4.684805582 | 4.710281956 | 4.691415533 | +0.006609951 | -0.018866423 |
| 200080012 | 4.699310882 | 4.707138488 | 4.697767608 | -0.001543274 | -0.009370880 |
| 200080013 | 4.688363823 | 4.724323769 | 4.705794848 | +0.017431024 | -0.018528921 |

Mean deeper-minus-incumbent is **+0.007499234 nat/token**; two fresh seeds are worse. Its simultaneous conditional-text upper bounds are+.011706365/+.010430525/+.009425925 for2048/4096/8192-token blocks. All three original quality gates fail.

Mean deeper-minus-shallow is **−0.015588741 nat/token**, with all three seeds and all upper bounds negative. The gain nevertheless misses the predeclared−.02 practical threshold. No threshold was changed and no shallow fallback or cost-only adoption is allowed.

The independent explicit grouping/bootstrap reviewer agreed within1e-12. Bounds condition on these three trained seeds and adaptively reused validation text, with Bonferroni accounting for two contrasts × three block sizes. They are not holdout or population-seed confidence. The closed test was not scored.

## Cost and interpretation

| Arm | Mean fresh CE | Parameters | Mean training-loop seconds | FP32 cache bytes/item |
|---|---:|---:|---:|---:|
| incumbent | 4.690826763 | 28,492,416 | 274.96 | 196,608 |
| shallow | 4.713914738 | 22,057,472 | 236.71 | 163,840 |
| deeper | 4.698325997 | 22,029,632 | 257.37 | 286,720 |

Deeper full128 inference latency is approximately unchanged, while median paired decode32 latency ratios versus incumbent are1.17295/1.14605 at batches1/8. Shallow ratios are.90230/.95162. Measurements use one resident model, two warmups and five timed repetitions; loading/cache allocation are excluded. Training and serving time are descriptive local measurements, not equal-compute comparisons or universal hardware claims. Fewer parameters do not imply smaller cache or faster decoding.

The192-wide pair nearly matches body/vocabulary capacity while exchanging FFN width for depth; all three were equally tuned within the finite allowance. These contrasts support a scoped allocation finding, not a pure-depth causal claim or a proof that depth/width exploration is exhausted. Shapes, initialization coordinates and optimizer applicability differ from the incumbent, as prospectively disclosed.

## Preservation and continuation

All939 frozen scientific bindings remain unchanged. The terminal STOP closes every original training/inference/final command. Preserve all24 selected/fresh RAW/H200 files totaling5,680,488,904B in their original paths, all historical selected/fresh archives and every original source/log/config/metric/receipt/failure. Retired losing screens resolve through their later exact manifest; never restore or rerun them. No numerical worker remains active at closure.

The authorized improvement loop continues with existing-data and primary-paper review. New fixed-weight conditioning diagnostics are explicitly separate from this closed scientific study; they do not adopt a model or pool old seeds into a fresh confirmation. No automatic beta1 experiment or random novelty is selected.

Evidence: [original final review](tokenizer-evidence/body-allocation-preparation-001/quality001-final-review.json), [read-only closure](tokenizer-evidence/body-allocation-preparation-001/quality001-closure001.json), [all-fresh inference](tokenizer-evidence/body-allocation-preparation-001/quality001-fresh-inference.json), [preparation/design](TOKENIZER_BODY_ALLOCATION_PREPARATION.md).
