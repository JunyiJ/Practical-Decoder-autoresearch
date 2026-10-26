# Input embedding scale quality results

Input embedding scale comparison completed 36 physical trajectories and 72 correlated raw/H200 outputs, independently reviewed. Common warmup 0 updates. Frozen eligible winner: incumbent_unchanged. Fresh H200 width64/256 deltas versus tuned unit-scale baseline: half +0.004697/-0.004641; quarter +0.013697/-0.002240. Retain the prior incumbent under the complete frozen rule.

Historical source `1dadecddd57060aecf1910cb3553d11b90ff0c1a`. Result `kb_eafb07ff1bc9a39a32fb5e22d7857305459684b180a325529931a5eb009acb56`. All 72 checkpoint hashes/payloads, requests/results, full coverage, optimizer/RNG/schedule/H200 contracts, frozen selection and complete report/bootstrap were independently reviewed. No training/evaluation rerun or holdout access.

Controller active time 2,808.85 seconds; output before review 530.44MiB. [Frozen protocol](EMBEDDING_SCALE_PROTOCOL.md).

## Fresh confirmation means

Seeds 2311/2312/2313; 111,538 validation targets once each. H200 primary; raw correlated diagnostic.

| Family | Selected peak LR | H200 CE64 | H200 CE256 | Raw CE64 | Raw CE256 |
|---|---:|---:|---:|---:|---:|
| baseline | 0.005 | 1.506894 | 1.433154 | 1.513094 | 1.450571 |
| half | 0.005 | 1.511590 | 1.428513 | 1.518923 | 1.444901 |
| quarter | 0.0025 | 1.520590 | 1.430914 | 1.525394 | 1.447168 |
| incumbent_reference | 0.005 | 1.506894 | 1.433154 | 1.513094 | 1.450571 |

## Paired contrasts

Positive is worse. Intervals use 4096-token blocks; all three block sizes were audited.

| Endpoint | Width | Contrast | Mean delta | Seed deltas | Conditional95% text interval | Primary support |
|---|---:|---|---:|---|---|---|
| H200 | 64 | half minus baseline | +0.004697 | +0.002509 / +0.006606 / +0.004975 | [+0.000745, +0.009318] | False |
| Raw | 64 | half minus baseline | +0.005829 | +0.005407 / +0.007611 / +0.004470 | [+0.001630, +0.010633] | Diagnostic |
| H200 | 64 | quarter minus baseline | +0.013697 | +0.013512 / +0.014224 / +0.013354 | [+0.010413, +0.017528] | False |
| Raw | 64 | quarter minus baseline | +0.012300 | +0.012267 / +0.013805 / +0.010828 | [+0.008995, +0.016238] | Diagnostic |
| H200 | 64 | quarter minus half | +0.009000 | +0.011003 / +0.007618 / +0.008379 | [+0.006668, +0.011227] | False |
| Raw | 64 | quarter minus half | +0.006471 | +0.006860 / +0.006194 / +0.006358 | [+0.003599, +0.008907] | Diagnostic |
| H200 | 64 | baseline minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 64 | baseline minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 64 | half minus incumbent_reference | +0.004697 | +0.002509 / +0.006606 / +0.004975 | [+0.000745, +0.009318] | False |
| Raw | 64 | half minus incumbent_reference | +0.005829 | +0.005407 / +0.007611 / +0.004470 | [+0.001630, +0.010633] | Diagnostic |
| H200 | 64 | quarter minus incumbent_reference | +0.013697 | +0.013512 / +0.014224 / +0.013354 | [+0.010413, +0.017528] | False |
| Raw | 64 | quarter minus incumbent_reference | +0.012300 | +0.012267 / +0.013805 / +0.010828 | [+0.008995, +0.016238] | Diagnostic |
| H200 | 256 | half minus baseline | -0.004641 | +0.006323 / -0.011490 / -0.008756 | [-0.006983, -0.002177] | False |
| Raw | 256 | half minus baseline | -0.005670 | +0.010201 / -0.015723 / -0.011488 | [-0.008366, -0.002941] | Diagnostic |
| H200 | 256 | quarter minus baseline | -0.002240 | +0.003696 / -0.001449 / -0.008967 | [-0.004705, +0.000092] | False |
| Raw | 256 | quarter minus baseline | -0.003403 | +0.006103 / -0.004912 / -0.011401 | [-0.006353, -0.000506] | Diagnostic |
| H200 | 256 | quarter minus half | +0.002401 | -0.002626 / +0.010041 / -0.000211 | [-0.001039, +0.005720] | False |
| Raw | 256 | quarter minus half | +0.002267 | -0.004097 / +0.010811 / +0.000087 | [-0.001814, +0.006019] | Diagnostic |
| H200 | 256 | baseline minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 256 | baseline minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 256 | half minus incumbent_reference | -0.004641 | +0.006323 / -0.011490 / -0.008756 | [-0.006983, -0.002177] | False |
| Raw | 256 | half minus incumbent_reference | -0.005670 | +0.010201 / -0.015723 / -0.011488 | [-0.008366, -0.002941] | Diagnostic |
| H200 | 256 | quarter minus incumbent_reference | -0.002240 | +0.003696 / -0.001449 / -0.008967 | [-0.004705, +0.000092] | False |
| Raw | 256 | quarter minus incumbent_reference | -0.003403 | +0.006103 / -0.004912 / -0.011401 | [-0.006353, -0.000506] | Diagnostic |

## Fixed-LR screens

| Contrast | H200 mean delta | Two seed deltas | Raw mean delta |
|---|---:|---|---:|
| LR0.0025: half minus baseline | -0.007924 | -0.008994 / -0.006855 | -0.007563 |
| LR0.0025: quarter minus baseline | -0.014672 | -0.017700 / -0.011644 | -0.013330 |
| LR0.0025: quarter minus half | -0.006748 | -0.008706 / -0.004790 | -0.005767 |
| LR0.005: half minus baseline | -0.003713 | -0.005381 / -0.002044 | -0.006360 |
| LR0.005: quarter minus baseline | +0.006249 | +0.007685 / +0.004813 | +0.006331 |
| LR0.005: quarter minus half | +0.009962 | +0.013067 / +0.006857 | +0.012691 |
| LR0.01: half minus baseline | +0.007461 | +0.008026 / +0.006895 | +0.007746 |
| LR0.01: quarter minus baseline | +0.002031 | +0.008123 / -0.004062 | +0.004849 |
| LR0.01: quarter minus half | -0.005430 | +0.000097 / -0.010957 | -0.002898 |

## Measured training costs

| Family | Width | Training seconds | Wall seconds | Ratio versus baseline | Sampled RSS MiB |
|---|---:|---:|---:|---:|---:|
| baseline | 64 | 54.80 | 59.12 | 1.000 | 473.61 |
| baseline | 256 | 125.71 | 133.75 | 1.000 | 482.65 |
| half | 64 | 54.94 | 59.19 | 1.003 | 473.06 |
| half | 256 | 125.70 | 134.21 | 1.000 | 482.86 |
| quarter | 64 | 54.89 | 59.20 | 1.002 | 473.35 |
| quarter | 256 | 125.75 | 134.21 | 1.000 | 482.27 |
| incumbent_reference | 64 | 54.80 | 59.12 | 1.000 | 473.61 |
| incumbent_reference | 256 | 125.71 | 133.75 | 1.000 | 482.65 |

## Limits

- Adaptive development on reused validation; conditional text intervals are not seed-population uncertainty, independent holdout or adjusted significance.
- 36 physical trajectories yield 72 correlated raw/H200 endpoints. Reference aliases {'incumbent_reference': 'baseline'} were declared prospectively and are not independent repeated controls; no historical pooling.
- Only the frozen eligible width64 winner can be adopted; no alternate fallback, larger-width reselection, grid extension or completed-trial rerun.
- Equal finite LR tuning; selected-recipe contrasts include LR choice. A tuned baseline-only gain is not attributed to input scaling. Fixed-LR screens are separate.
- Equal3000 updates and3072000 targets, common reviewed warmup/cooldown/H200, independent decay .00006. Scalar changes relative residual contributions and embedding gradients without changing initialization or parameter shapes; no identified mechanism or optimal muP claim.
- Measured training time is separate from quality; no equal-compute, serving-speed or peak-accelerator-memory claim.
- Final H200 tensors equal saved raw-parent shadows; endpoint evidence cannot reconstruct every historical snapshot. Earlier MPS reproducibility failures remain preserved without an identified cause.
- H200 checkpoints are inference-only; resume optimizer training only from their raw parent.

## Interpretation and preserved limits

Half-scale improved both screening seeds at selected LR 0.005, but its mean gain of 0.003713 CE did not exceed the preregistered 0.005 threshold. Quarter-scale selected the lower grid boundary, LR 0.0025. The frozen winner was `incumbent_unchanged`; larger-width results cannot reopen selection or extend that grid.

Fresh width-64 confirmation reverses the half-scale screening direction: all three seeds worsen (+0.002509, +0.006606, +0.004975). Quarter-scale also worsens all three (+0.013512, +0.014224, +0.013354). All conditional text intervals are positive for both comparisons, conditional on these trained seeds.

At width 256, half-scale improves the mean by 0.004641, but seed 2311 worsens by 0.006323; the other seeds improve by 0.011490 and 0.008756. All three conditional text intervals are negative, which does not override the mixed seed evidence or the failed local rule. Quarter-scale improves the mean by 0.002240 with mixed seeds and intervals that include zero. Neither family meets the complete frozen criterion. The historical H200 incumbent remains; fresh control scores are not pooled with prior rounds.

The fixed-LR screening contrasts depend on LR: at 0.0025, half and quarter improve against the unit-scale control by 0.007924 and 0.014672; at 0.005, their deltas are -0.003713 and +0.006249; at 0.01, +0.007461 and +0.002031. These are two-seed adaptive screens, not an independently established mechanism or optimal scaling law. Selected quarter-scale confirmation includes its different chosen LR.

Training times are within 0.26% of the unit-scale control in this run. This is a descriptive measurement, not a speed-equivalence, equal-compute, or inference claim. Six fixed-reference cells alias selected unit-scale trajectories prospectively; they are not repeated independent controls.

The source always inherited a worker memory watchdog: nominal 0.5-second sampling of process peak RSS and current MPS driver allocation, with thresholds 4,096 and 6,144 MiB. Earlier protocol/review prose describing these as pilot-only was incomplete. The clarification is recorded separately; no source, limit, plan, training, or evidence changed. Sampling does not guarantee instantaneous, total-system, or peak-accelerator enforcement.

All completed studies stay closed. Continue with a distinct final readout-normalization protocol on the retained unit-scale, LR 0.005, zero-warmup recipe after its actual CPU/MPS, resource, source-publication and preregistration gates.
