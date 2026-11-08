# Byte-context transforms: complete, not adopted

All nine fresh trajectories completed the fixed 1,200 updates and saved RAW/H200
weights. The independent saved-result review passed. Keep the existing byte
incumbent: neither added transform improves its averaged validation loss on any
of the three seeds. No model, learning rate, seed, endpoint or tolerance was
changed to rescue the result; the holdout remains closed.

## Primary result

Lower cross-entropy is better. Every score uses all 34,278 validation targets,
context128 reset windows and the predefined saved/reloaded H200 checkpoint.

| Fresh bundle | Byte baseline | Linear transform | Tanh transform | Tanh minus byte | Tanh minus linear |
|---|---:|---:|---:|---:|---:|
| 62102200000 | 4.550480 | 4.643989 | 4.616356 | +0.065875 | -0.027634 |
| 62102300000 | 4.566994 | 4.654481 | 4.643983 | +0.076989 | -0.010497 |
| 62102400000 | 4.571279 | 4.656345 | 4.623391 | +0.052112 | -0.032954 |
| Mean | 4.562918 | 4.651605 | 4.627910 | +0.064992 | -0.023695 |

The candidate must beat both controls by a mean of more than .02, improve every
seed and have negative conditional upper bounds at all declared block widths.
Tanh fails every gate against byte. Against linear, it passes the mean and
every-seed gates but fails the uncertainty gate. There is no fallback adoption.

| Contrast | Upper bound: 2048 | Upper bound: 4096 | Upper bound: 8192 |
|---|---:|---:|---:|
| Tanh minus byte | +0.099135 | +0.108591 | +0.112091 |
| Tanh minus linear | +0.027287 | +0.038411 | +0.044644 |

These one-sided text-bootstrap bounds condition on three trained bundles and
adaptively reused validation text. They are not population-seed confidence or
untouched-holdout evidence. The reviewer independently reproduced the saved-loss
contrasts and bounds within 1e-12.

## What the learning curves add

The nonlinear model's mean training loss over its last 100 updates exceeds the
paired byte baseline by .723820, .675350 and .671926. The pattern repeats across
all three seeds. It still improves substantially late in training. By contrast,
the linear model fits the sampled training batches slightly better than byte
but generalizes worse. There is no single uniform optimization explanation for
both failed transforms.

All three tanh averaged checkpoints score worse than their final online weights:
the gaps are +.024026, +.039427 and +.019843. Linear averaging improves its score
by .073735 to .090873, although it still loses to byte. In the last two bundles,
byte averaging improves its score by about .0361. The first byte online curve
used the original defective no-grad normalization path, so it is excluded from
the direct online/averaged comparison.

For the two comparable byte/tanh pairs, final-online tanh-minus-byte differences
are only +.001474 and -.003866. Their H200 differences are +.076989 and +.052112;
the respective differences in averaging effects are +.075515 and +.055977.
This is an exact decomposition of the observed endpoint gaps, not proof of an
averaging mechanism. It does not justify choosing final-online weights instead.

Two previously measured tanh windows had mean local activation derivatives near
.15-.16. They are not representative training samples or full-Jacobian measures.
Delayed fitting, amplitude constraints, optimizer interaction and changing
representations remain competing explanations. The primary literature review
and its scope limits are in the [first-bundle analysis](tokenizer-evidence/byte-context-transform-001/first-bundle-analysis001/analysis.md).
No initialization, activation or averaging sweep is admitted by this result.

## Cost and correctness

The panel performed 10,800 scientific training updates and sampled 11,059,200
targets. Actual update execution took 5,103.879 seconds (85.06 minutes), and
scheduled online scoring took 619.421 seconds (10.32 minutes). These are 84.37%
and 10.24% of the nine original guarded jobs. Total charged execution, including
the earlier pilot/diagnostic recoveries and final independent review, was
6,210.163 seconds (103.50 minutes), below the original 36,000-second allowance.
The independent review itself took 2.351 seconds. These measurements exclude
human/agent engineering wall time and are not deployment throughput claims.

Byte has 16,351,859 parameters. Either transform adds 65,536 parameters (0.401%).
The added parameters did not buy improved validation quality in this recipe.

Original failures remain preserved. The MPS default-epsilon normalization defect
was corrected in the scoring path; native CPU cross-entropy was separately
replaced by a verified FP32 reference. Training was not repeated. The first
three primary MPS reports are byte-identical to their prior accepted scores.
All nine corrected CPU/MPS checks pass the original mean/max tolerances.
Resource accounting was separately corrected without changing model execution.

Slots00 and02 lack original published terminal sampler digests; their expected
states are explicitly source/count-derived. Other observed digests are retained.
Final RAW/H200 artifacts contain weights, not resumable optimizer states or an
independent reconstruction of every intermediate averaging snapshot.

The completed scientific panel is closed. The separately frozen inference check
on the new baseline weights is an implementation-correctness task. It does not
reopen historical quality comparisons or change the selected model.

## Bound evidence

- [Frozen protocol](BYTE_CONTEXT_TRANSFORM_PROTOCOL.md)
- [Recovery plan](tokenizer-evidence/byte-context-transform-001/quality-recovery002/quality-plan.json)
- [Final controller result](tokenizer-evidence/byte-context-transform-001/quality-recovery002/final.json)
- [Independent review](tokenizer-evidence/byte-context-transform-001/quality-recovery002/final-review.json)
- [All-seed curves and cost analysis](tokenizer-evidence/byte-context-transform-001/final-analysis001/result.json)
- [CPU reference recovery](tokenizer-evidence/byte-context-transform-001/cpu-reference-recovery-decision.md)
