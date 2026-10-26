# Scaled table-sharing quality results

Scaled table-sharing comparison completed 36 physical trajectories and 72 correlated raw/H200 outputs, independently reviewed. Common warmup 0 updates. Frozen decision: retain the incumbent. Fresh H200 width64/256 deltas versus tuned legacy baseline: matched-untied +0.003225/-0.003865; scaled-tied +0.000447/-0.004328. Retain the prior incumbent under the complete frozen rule.

Historical source `8b636636c7c4b1f800821eb290e5ad365326b8d4`. Result `kb_56328c82e5551e2e07b74d28883511b4c2d48c6015a3219faa77f61275e80062`. All 72 checkpoint hashes/payloads, requests/results, full coverage, optimizer/RNG/schedule/H200 contracts, frozen selection and complete report/bootstrap were independently reviewed. No training/evaluation rerun or holdout access.

Controller active time 2,801.36 seconds; output before review 529.17 MiB. [Frozen protocol](WEIGHT_TYING_PROTOCOL.md).

## Fresh confirmation means

Seeds 2371/2372/2373; 111,538 validation targets once each. H200 primary; raw correlated diagnostic.

| Family | Selected peak LR | H200 CE64 | H200 CE256 | Raw CE64 | Raw CE256 |
|---|---:|---:|---:|---:|---:|
| legacy_untied | 0.005 | 1.522497 | 1.433955 | 1.523213 | 1.445806 |
| matched_untied | 0.005 | 1.525722 | 1.430090 | 1.524229 | 1.437961 |
| scaled_tied | 0.005 | 1.522944 | 1.429627 | 1.522474 | 1.437331 |
| incumbent_reference | 0.005 | 1.522497 | 1.433955 | 1.523213 | 1.445806 |

## Paired contrasts

Positive is worse. Intervals use 4,096-token blocks; all three block sizes were audited.

| Endpoint | Width | Contrast | Mean delta | Seed deltas | Conditional 95% text interval | Primary support |
|---|---:|---|---:|---|---|---|
| H200 | 64 | matched_untied minus legacy_untied | +0.003225 | +0.010099 / +0.000410 / -0.000833 | [-0.000525, +0.006767] | False |
| Raw | 64 | matched_untied minus legacy_untied | +0.001016 | +0.006874 / -0.001627 / -0.002200 | [-0.002936, +0.004664] | Diagnostic |
| H200 | 64 | scaled_tied minus legacy_untied | +0.000447 | +0.007199 / -0.010644 / +0.004785 | [-0.003530, +0.004383] | False |
| Raw | 64 | scaled_tied minus legacy_untied | -0.000740 | +0.004530 / -0.010941 / +0.004192 | [-0.004629, +0.003111] | Diagnostic |
| H200 | 64 | scaled_tied minus matched_untied | -0.002778 | -0.002900 / -0.011054 / +0.005618 | [-0.004604, -0.000978] | False |
| Raw | 64 | scaled_tied minus matched_untied | -0.001755 | -0.002345 / -0.009314 / +0.006392 | [-0.003671, +0.000256] | Diagnostic |
| H200 | 64 | legacy_untied minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 64 | legacy_untied minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 64 | matched_untied minus incumbent_reference | +0.003225 | +0.010099 / +0.000410 / -0.000833 | [-0.000525, +0.006767] | False |
| Raw | 64 | matched_untied minus incumbent_reference | +0.001016 | +0.006874 / -0.001627 / -0.002200 | [-0.002936, +0.004664] | Diagnostic |
| H200 | 64 | scaled_tied minus incumbent_reference | +0.000447 | +0.007199 / -0.010644 / +0.004785 | [-0.003530, +0.004383] | False |
| Raw | 64 | scaled_tied minus incumbent_reference | -0.000740 | +0.004530 / -0.010941 / +0.004192 | [-0.004629, +0.003111] | Diagnostic |
| H200 | 256 | matched_untied minus legacy_untied | -0.003865 | -0.007226 / +0.001004 / -0.005374 | [-0.005920, -0.001998] | False |
| Raw | 256 | matched_untied minus legacy_untied | -0.007844 | -0.010183 / -0.005017 / -0.008334 | [-0.010074, -0.005866] | Diagnostic |
| H200 | 256 | scaled_tied minus legacy_untied | -0.004328 | -0.009672 / -0.000885 / -0.002426 | [-0.006616, -0.001828] | False |
| Raw | 256 | scaled_tied minus legacy_untied | -0.008475 | -0.013471 / -0.008107 / -0.003847 | [-0.011179, -0.005571] | Diagnostic |
| H200 | 256 | scaled_tied minus matched_untied | -0.000462 | -0.002446 / -0.001888 / +0.002948 | [-0.002705, +0.001738] | False |
| Raw | 256 | scaled_tied minus matched_untied | -0.000631 | -0.003288 / -0.003090 / +0.004487 | [-0.003072, +0.001762] | Diagnostic |
| H200 | 256 | legacy_untied minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 256 | legacy_untied minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 256 | matched_untied minus incumbent_reference | -0.003865 | -0.007226 / +0.001004 / -0.005374 | [-0.005920, -0.001998] | False |
| Raw | 256 | matched_untied minus incumbent_reference | -0.007844 | -0.010183 / -0.005017 / -0.008334 | [-0.010074, -0.005866] | Diagnostic |
| H200 | 256 | scaled_tied minus incumbent_reference | -0.004328 | -0.009672 / -0.000885 / -0.002426 | [-0.006616, -0.001828] | False |
| Raw | 256 | scaled_tied minus incumbent_reference | -0.008475 | -0.013471 / -0.008107 / -0.003847 | [-0.011179, -0.005571] | Diagnostic |

## Fixed-LR screens

| Contrast | H200 mean delta | Two seed deltas | Raw mean delta |
|---|---:|---|---:|
| LR0.0025: matched_untied minus legacy_untied | +0.008087 | +0.007646 / +0.008528 | +0.006903 |
| LR0.0025: scaled_tied minus legacy_untied | +0.010180 | +0.009853 / +0.010508 | +0.008787 |
| LR0.0025: scaled_tied minus matched_untied | +0.002094 | +0.002207 / +0.001980 | +0.001883 |
| LR0.005: matched_untied minus legacy_untied | +0.009270 | +0.011790 / +0.006750 | +0.006488 |
| LR0.005: scaled_tied minus legacy_untied | +0.012753 | +0.017439 / +0.008068 | +0.010257 |
| LR0.005: scaled_tied minus matched_untied | +0.003484 | +0.005648 / +0.001319 | +0.003769 |
| LR0.01: matched_untied minus legacy_untied | +0.016115 | +0.018060 / +0.014169 | +0.014844 |
| LR0.01: scaled_tied minus legacy_untied | +0.015691 | +0.016880 / +0.014502 | +0.013047 |
| LR0.01: scaled_tied minus matched_untied | -0.000424 | -0.001181 / +0.000333 | -0.001797 |

## Measured training costs

| Family | Width | Training seconds | Wall seconds | Ratio versus tuned legacy | Sampled RSS MiB |
|---|---:|---:|---:|---:|---:|
| legacy_untied | 64 | 54.82 | 58.82 | 1.000 | 474.86 |
| legacy_untied | 256 | 125.59 | 133.90 | 1.000 | 483.93 |
| matched_untied | 64 | 54.84 | 59.01 | 1.000 | 475.09 |
| matched_untied | 256 | 125.68 | 133.71 | 1.001 | 484.60 |
| scaled_tied | 64 | 54.65 | 58.84 | 0.997 | 475.18 |
| scaled_tied | 256 | 125.45 | 133.96 | 0.999 | 484.33 |
| incumbent_reference | 64 | 54.82 | 58.82 | 1.000 | 474.86 |
| incumbent_reference | 256 | 125.59 | 133.90 | 1.000 | 483.93 |

## Limits

- Adaptive development on reused validation; conditional text intervals are not seed-population uncertainty, independent holdout or adjusted significance.
- 36 physical trajectories yield 72 correlated raw/H200 endpoints. Reference aliases {'incumbent_reference': 'legacy_untied'} were declared prospectively and are not independent repeated controls; no historical pooling.
- Only the frozen eligible width64 winner can be adopted; no alternate fallback, larger-width reselection, grid extension or completed-trial rerun.
- Equal finite LR tuning; selected-recipe contrasts include LR choice. A tuned legacy-only gain is attributed to LR selection, not table sharing. Fixed-LR screens are separate.
- Equal 3000 updates and 3072000 targets, common reviewed bias/prior/norm/input, warmup/cooldown/H200 and independent decay. Matched untied clones E into a distinct Q after all legacy initialization draws; scaled tied removes the head and uses canonical E. Both apply 1/sqrt192 before the existing width division. Matched versus legacy changes readout distribution, correlation and optimizer coordinates. Tied versus matched couples lookup/output gradient roles and AdamW histories. One shared table has one optimizer state, decay application and H200 shadow. Shared initial functions do not imply identical later trajectories, isolated capacity effects or optimal muP transfer. Tied-minus-matched is reported separately.
- Measured training time is separate from quality; no equal-compute, serving-speed or peak-accelerator-memory claim.
- Final H200 tensors equal saved raw-parent shadows; endpoint evidence cannot reconstruct every historical snapshot. Earlier MPS reproducibility failures remain preserved without an identified cause.
- H200 checkpoints are inference-only; resume optimizer training only from their raw parent.

## Interpretation

All three LRs were interior selections. Legacy was best in screening and exactly matched the incumbent, freezing retention before confirmation. On fresh seeds 2371/2372/2373, matched-minus-legacy means were +0.003225 / −0.003865 at widths 64 / 256, with mixed seeds at both widths. Tied-minus-legacy was +0.000447 / −0.004328: two width-64 seeds worsened, while all three width-256 seeds improved. The latter gain remains below 0.005 and cannot reopen the frozen selection.

Tied-minus-matched means were −0.002778 / −0.000462, with one worse seed at each width. Its negative width-64 conditional text interval does not establish training-seed consistency. All three recipes use the same selected peak LR here. Matched versus legacy changes the initial readout distribution, correlation and optimizer coordinates; tying then combines lookup/output gradients and AdamW histories. These comparisons do not isolate a single cause or establish optimal muP transfer.

Retain the historical untied, no-bias, affine-final-LayerNorm, unit-input MHA4/SwiGLU21D/8 recipe: LR 0.005, independent decay 0.00006, zero warmup, 3,000-update cooldown and H200 returned weights. Six fixed-reference cells prospectively alias legacy confirmation; they are not independent repeats. Training-time changes were below 0.32%, descriptive only. Active time was 2,801.36 seconds (0.78 hours), with 529.17 MiB before review. Adaptive validation, no holdout, no historical pooling or grid extension.

