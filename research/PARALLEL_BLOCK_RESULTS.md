# Parallel-block quality results

Serial versus parallel separate-norm residual blocks completed 24 physical trajectories and 48 correlated raw/H200 outputs, independently reviewed. Common inherited table kind legacy_untied; warmup 0 updates. Frozen eligible winner: incumbent_unchanged. Fresh H200 width64/256 deltas for parallel versus tuned serial: +0.026757/+0.013555. Retain the prior incumbent under the complete frozen rule.

Historical source `fe092eb93af7b9281d06089cd63482ff4b55c93e`. Result `kb_a45549683e77c85173f2340bbf8b3b0bd5f6fdbdb995e437f61c5aeabd653031`. All 48 checkpoint hashes/payloads, requests/results, full coverage, optimizer/RNG/schedule/H200 contracts, frozen selection and complete report/bootstrap were independently reviewed. No training/evaluation rerun or holdout access.

Controller active time 1863.72 seconds; output before review 353.92 MiB. [Frozen protocol](PARALLEL_BLOCK_PROTOCOL.md).

## Fresh confirmation means

Seeds 2391/2392/2393; 111,538 validation targets once each. H200 primary; raw correlated diagnostic.

| Family | Selected peak LR | H200 CE64 | H200 CE256 | Raw CE64 | Raw CE256 |
|---|---:|---:|---:|---:|---:|
| serial | 0.005 | 1.515415 | 1.434856 | 1.521052 | 1.447788 |
| parallel_separate_norms | 0.01 | 1.542172 | 1.448411 | 1.549045 | 1.461025 |
| incumbent_reference | 0.005 | 1.515415 | 1.434856 | 1.521052 | 1.447788 |

## Paired contrasts

Positive is worse. Intervals use 4096-token blocks; all three block sizes were audited.

| Endpoint | Width | Contrast | Mean delta | Seed deltas | Conditional 95% text interval | Primary support |
|---|---:|---|---:|---|---|---|
| H200 | 64 | parallel_separate_norms minus serial | +0.026757 | +0.031364 / +0.022656 / +0.026251 | [+0.024375, +0.028933] | False |
| Raw | 64 | parallel_separate_norms minus serial | +0.027992 | +0.034274 / +0.023514 / +0.026189 | [+0.025398, +0.030545] | Diagnostic |
| H200 | 64 | serial minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 64 | serial minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 64 | parallel_separate_norms minus incumbent_reference | +0.026757 | +0.031364 / +0.022656 / +0.026251 | [+0.024375, +0.028933] | False |
| Raw | 64 | parallel_separate_norms minus incumbent_reference | +0.027992 | +0.034274 / +0.023514 / +0.026189 | [+0.025398, +0.030545] | Diagnostic |
| H200 | 256 | parallel_separate_norms minus serial | +0.013555 | +0.013898 / +0.016997 / +0.009768 | [+0.011118, +0.015852] | False |
| Raw | 256 | parallel_separate_norms minus serial | +0.013237 | +0.017125 / +0.017121 / +0.005464 | [+0.010468, +0.016193] | Diagnostic |
| H200 | 256 | serial minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 256 | serial minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 256 | parallel_separate_norms minus incumbent_reference | +0.013555 | +0.013898 / +0.016997 / +0.009768 | [+0.011118, +0.015852] | False |
| Raw | 256 | parallel_separate_norms minus incumbent_reference | +0.013237 | +0.017125 / +0.017121 / +0.005464 | [+0.010468, +0.016193] | Diagnostic |

## Fixed-LR screens

| Contrast | H200 mean delta | Two seed deltas | Raw mean delta |
|---|---:|---|---:|
| LR0.0025: parallel_separate_norms minus serial | +0.031655 | +0.033431 / +0.029878 | +0.031274 |
| LR0.005: parallel_separate_norms minus serial | +0.034663 | +0.035232 / +0.034094 | +0.033690 |
| LR0.01: parallel_separate_norms minus serial | +0.029818 | +0.019566 / +0.040069 | +0.028194 |

## Measured training costs

| Family | Width | Training seconds | Wall seconds | Ratio versus tuned serial | Sampled RSS MiB |
|---|---:|---:|---:|---:|---:|
| serial | 64 | 54.77 | 58.83 | 1.000 | 474.97 |
| serial | 256 | 125.50 | 133.08 | 1.000 | 484.27 |
| parallel_separate_norms | 64 | 54.66 | 58.72 | 0.998 | 475.24 |
| parallel_separate_norms | 256 | 125.61 | 133.68 | 1.001 | 483.47 |
| incumbent_reference | 64 | 54.77 | 58.83 | 1.000 | 474.97 |
| incumbent_reference | 256 | 125.50 | 133.08 | 1.000 | 484.27 |

## Limits

- Adaptive development on reused validation; conditional text intervals are not seed-population uncertainty, independent holdout or adjusted significance.
- 24 physical trajectories yield 48 correlated raw/H200 endpoints. Reference aliases {'incumbent_reference': 'serial'} were declared prospectively and are not independent repeated controls; no historical pooling.
- Only the frozen eligible width64 winner can be adopted; no alternate fallback, larger-width reselection, grid extension or completed-trial rerun.
- Equal finite LR tuning; selected-recipe contrasts include LR choice. A tuned serial-only gain is attributed to LR selection, not residual topology. Fixed-LR screens are separate.
- Equal 3000 updates and 3072000 targets, common reviewed table/bias/prior/norm/input, warmup/cooldown/H200 and independent decay. Serial uses u=x+A(N1(x)), y=u+F(N2(u)); parallel uses y=u+F(N2(x)), retaining two distinct norms. Construction RNG and parameter layout are identical, while initial functions and gradients differ. No isolated mechanism, PaLM replication, fused kernels, concurrency or empirical optimal muP transfer claim.
- Measured training time is separate from quality; no equal-compute, serving-speed or peak-accelerator-memory claim.
- Final H200 tensors equal saved raw-parent shadows; endpoint evidence cannot reconstruct every historical snapshot. Earlier MPS reproducibility failures remain preserved without an identified cause.
- H200 checkpoints are inference-only; resume optimizer training only from their raw parent.

## Interpretation and decision

Serial selected peak LR 0.005 (interior); parallel selected 0.01 (upper boundary). The best screening recipe was the unchanged serial incumbent, so retention was frozen before confirmation. Parallel then worsened H200 and raw loss on every fresh seed at both widths; every conditional text interval was positive. At all three fixed learning rates, parallel also worsened both screening seeds. Selected-recipe confirmation compares different learning rates and does not isolate topology alone. The grid is closed.

Six fixed-reference cells prospectively alias the selected serial cells; their zero contrasts are identities, not independent repeated evidence. Fresh controls are not pooled with historical studies. Training-time differences were −0.21% at width 64 and +0.09% at width 256, descriptive measurements without a speed or equivalence claim. The retained recipe remains serial, untied, no output bias, affine final LayerNorm, unit input scale, peak LR 0.005, independent decay 0.00006, zero warmup, 3000-update cooldown and H200 returned weights.

Before this run, all 3030 CPU cases and four serial MPS pilots passed. Final auditing independently checked all 24 raw and 24 H200 checkpoints, optimizer and RNG contracts, full validation coverage, selection, aliases and every reported bootstrap. The original auditor passed on the frozen source after both controller and worker exited.
