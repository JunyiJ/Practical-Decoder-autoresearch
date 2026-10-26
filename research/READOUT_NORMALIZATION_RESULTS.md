# Final readout normalization quality results

Final readout normalization comparison completed 36 physical trajectories and 72 correlated raw/H200 outputs, independently reviewed. Common warmup 0 updates. Frozen eligible winner: incumbent_unchanged. Fresh H200 width64/256 deltas versus tuned affine LayerNorm: bias-free LayerNorm -0.001072/-0.006786; RMSNorm -0.001101/+0.001167. Retain the prior incumbent under the complete frozen rule.

Historical source `43b9a42b820430ca9a242fc2b142e66cd65be93c`. Result `kb_f6907bd9b2544ce7a50ba862b0ca94bbeee2698381b0dbf32692d139bd6205a1`. All 72 checkpoint hashes/payloads, requests/results, full coverage, optimizer/RNG/schedule/H200 contracts, frozen selection and complete report/bootstrap were independently reviewed. No training/evaluation rerun or holdout access.

Controller active time 2814.51 seconds; output before review 530.36 MiB. [Frozen protocol](READOUT_NORMALIZATION_PROTOCOL.md).

## Interpretation and frozen selection

The affine LayerNorm control was best in screening and exactly matched the fixed incumbent at peak LR 0.005. Bias-free LayerNorm selected 0.01 at the upper grid boundary, and RMSNorm selected the interior 0.005. No changed-normalization candidate was locally eligible. The grid is closed: confirmation cannot select a different winner or extend the LR range.

At width 64, both alternatives slightly improved the mean H200 loss, but seed 2331 worsened for each. Gains of 0.001072 and 0.001101 were below the required 0.005; their 4,096-token conditional text intervals include zero. At width 256, bias-free LayerNorm improved mean loss by 0.006786, but seed 2331 worsened by 0.000973. Its negative conditional text interval does not override that seed failure or the frozen local decision. RMSNorm had mixed seeds and a worse mean by 0.001167. RMSNorm was worse than bias-free LayerNorm for all three width-256 seeds, with mean difference +0.007953; these selected recipes also use different peak LRs.

The six fixed-incumbent confirmation cells alias the affine control exactly; they are not independent repeated controls. Fresh means remain specific to this round and are not pooled into the historical incumbent's estimate. Retain the four-layer MHA, affine final LayerNorm, unit input scale, LR 0.005, no warmup, independent decay 0.00006, 3,000-update cooldown and H200 returned weights.

Measured training times differed by less than 1% in these paired recipes. This is descriptive timing, not equivalence, a speedup claim, or serving performance. The next output-bias proposal remains separate and must inherit this completed decision and pass its own correctness, resource and registration gates.

## Fresh confirmation means

Seeds 2331/2332/2333;111,538 validation targets once each. H200 primary; raw correlated diagnostic.

| Family | Selected peak LR | H200 CE64 | H200 CE256 | Raw CE64 | Raw CE256 |
|---|---:|---:|---:|---:|---:|
| layer_affine | 0.005 | 1.520548 | 1.437485 | 1.523177 | 1.450814 |
| layer_no_bias | 0.01 | 1.519476 | 1.430699 | 1.524889 | 1.443826 |
| rms | 0.005 | 1.519448 | 1.438652 | 1.522772 | 1.452195 |
| incumbent_reference | 0.005 | 1.520548 | 1.437485 | 1.523177 | 1.450814 |

## Paired contrasts

Positive is worse. Intervals use 4096-token blocks; all three block sizes were audited.

| Endpoint | Width | Contrast | Mean delta | Seed deltas | Conditional 95% text interval | Primary support |
|---|---:|---|---:|---|---|---|
| H200 | 64 | layer_no_bias minus layer_affine | -0.001072 | +0.006721 / -0.005448 / -0.004489 | [-0.003874, +0.001481] | False |
| Raw | 64 | layer_no_bias minus layer_affine | +0.001713 | +0.009052 / -0.003177 / -0.000737 | [-0.001704, +0.005258] | Diagnostic |
| H200 | 64 | rms minus layer_affine | -0.001101 | +0.002334 / -0.003591 / -0.002045 | [-0.002368, +0.000182] | False |
| Raw | 64 | rms minus layer_affine | -0.000404 | +0.003873 / -0.002584 / -0.002503 | [-0.001663, +0.000900] | Diagnostic |
| H200 | 64 | rms minus layer_no_bias | -0.000029 | -0.004386 / +0.001856 / +0.002444 | [-0.002384, +0.002637] | False |
| Raw | 64 | rms minus layer_no_bias | -0.002117 | -0.005179 / +0.000593 / -0.001766 | [-0.005213, +0.001082] | Diagnostic |
| H200 | 64 | layer_affine minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 64 | layer_affine minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 64 | layer_no_bias minus incumbent_reference | -0.001072 | +0.006721 / -0.005448 / -0.004489 | [-0.003874, +0.001481] | False |
| Raw | 64 | layer_no_bias minus incumbent_reference | +0.001713 | +0.009052 / -0.003177 / -0.000737 | [-0.001704, +0.005258] | Diagnostic |
| H200 | 64 | rms minus incumbent_reference | -0.001101 | +0.002334 / -0.003591 / -0.002045 | [-0.002368, +0.000182] | False |
| Raw | 64 | rms minus incumbent_reference | -0.000404 | +0.003873 / -0.002584 / -0.002503 | [-0.001663, +0.000900] | Diagnostic |
| H200 | 256 | layer_no_bias minus layer_affine | -0.006786 | +0.000973 / -0.011157 / -0.010173 | [-0.010278, -0.003325] | False |
| Raw | 256 | layer_no_bias minus layer_affine | -0.006988 | +0.003696 / -0.010690 / -0.013971 | [-0.010922, -0.003222] | Diagnostic |
| H200 | 256 | rms minus layer_affine | +0.001167 | +0.008835 / +0.003901 / -0.009234 | [-0.001602, +0.003990] | False |
| Raw | 256 | rms minus layer_affine | +0.001382 | +0.011141 / +0.003019 / -0.010015 | [-0.001251, +0.004038] | Diagnostic |
| H200 | 256 | rms minus layer_no_bias | +0.007953 | +0.007862 / +0.015058 / +0.000939 | [+0.004832, +0.011378] | False |
| Raw | 256 | rms minus layer_no_bias | +0.008370 | +0.007444 / +0.013709 / +0.003956 | [+0.004803, +0.012030] | Diagnostic |
| H200 | 256 | layer_affine minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 256 | layer_affine minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 256 | layer_no_bias minus incumbent_reference | -0.006786 | +0.000973 / -0.011157 / -0.010173 | [-0.010278, -0.003325] | False |
| Raw | 256 | layer_no_bias minus incumbent_reference | -0.006988 | +0.003696 / -0.010690 / -0.013971 | [-0.010922, -0.003222] | Diagnostic |
| H200 | 256 | rms minus incumbent_reference | +0.001167 | +0.008835 / +0.003901 / -0.009234 | [-0.001602, +0.003990] | False |
| Raw | 256 | rms minus incumbent_reference | +0.001382 | +0.011141 / +0.003019 / -0.010015 | [-0.001251, +0.004038] | Diagnostic |

## Fixed-LR screens

| Contrast | H200 mean delta | Two seed deltas | Raw mean delta |
|---|---:|---|---:|
| LR0.0025: layer_no_bias minus layer_affine | -0.003038 | +0.000205 / -0.006281 | -0.002339 |
| LR0.0025: rms minus layer_affine | -0.000578 | -0.001274 / +0.000118 | -0.000456 |
| LR0.0025: rms minus layer_no_bias | +0.002460 | -0.001480 / +0.006399 | +0.001883 |
| LR0.005: layer_no_bias minus layer_affine | +0.006643 | +0.006018 / +0.007268 | +0.007610 |
| LR0.005: rms minus layer_affine | +0.003151 | +0.007317 / -0.001014 | +0.003384 |
| LR0.005: rms minus layer_no_bias | -0.003492 | +0.001299 / -0.008282 | -0.004226 |
| LR0.01: layer_no_bias minus layer_affine | -0.001004 | -0.002255 / +0.000248 | +0.001337 |
| LR0.01: rms minus layer_affine | +0.003841 | +0.001766 / +0.005915 | +0.005369 |
| LR0.01: rms minus layer_no_bias | +0.004844 | +0.004021 / +0.005667 | +0.004031 |

## Measured training costs

| Family | Width | Training seconds | Wall seconds | Ratio versus tuned affine | Sampled RSS MiB |
|---|---:|---:|---:|---:|---:|
| layer_affine | 64 | 55.04 | 59.43 | 1.000 | 473.79 |
| layer_affine | 256 | 125.78 | 134.06 | 1.000 | 482.84 |
| layer_no_bias | 64 | 54.66 | 59.03 | 0.993 | 473.53 |
| layer_no_bias | 256 | 125.56 | 134.15 | 0.998 | 482.42 |
| rms | 64 | 55.25 | 59.57 | 1.004 | 469.60 |
| rms | 256 | 126.96 | 135.17 | 1.009 | 479.32 |
| incumbent_reference | 64 | 55.04 | 59.43 | 1.000 | 473.79 |
| incumbent_reference | 256 | 125.78 | 134.06 | 1.000 | 482.84 |

## Limits

- Adaptive development on reused validation; conditional text intervals are not seed-population uncertainty, independent holdout or adjusted significance.
- 36 physical trajectories yield 72 correlated raw/H200 endpoints. Reference aliases {'incumbent_reference': 'layer_affine'} were declared prospectively and are not independent repeated controls; no historical pooling.
- Only the frozen eligible width64 winner can be adopted; no alternate fallback, larger-width reselection, grid extension or completed-trial rerun.
- Equal finite LR tuning; selected-recipe contrasts include LR choice. A tuned affine-only gain is attributed to LR selection, not normalization. Fixed-LR screens are separate.
- Equal 3,000 updates and 3,072,000 targets, common reviewed warmup/cooldown/H200, independent decay .00006. Inherited input scale is fixed. Bias-free/RMS remove D final-bias parameters and vector groups; RMS versus affine changes bias and formula, and RMS versus bias-free LayerNorm changes centering and denominator statistics. Shared active initialization/RNG and existing width factors do not establish an identified mechanism or optimal muP transfer.
- Measured training time is separate from quality; no equal-compute, serving-speed or peak-accelerator-memory claim.
- Final H200 tensors equal saved raw-parent shadows; endpoint evidence cannot reconstruct every historical snapshot. Earlier MPS reproducibility failures remain preserved without an identified cause.
- H200 checkpoints are inference-only; resume optimizer training only from their raw parent.
