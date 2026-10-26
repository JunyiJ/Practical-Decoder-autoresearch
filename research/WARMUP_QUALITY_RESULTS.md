# Warmup quality results

The 100-update warmup passed the frozen width-64 screen but failed fresh confirmation. Retain the no-warmup MHA/cooldown/H200 incumbent at peak LR 0.005 and independent decay 0.00006. All 24 trajectories and 48 correlated raw/H200 outputs were independently reviewed.

Historical source `69272ce1c72f6d19cb583773d31687a31f191dd8`. Result `kb_f682fa7cf84d4b37c5c35a7ae2d69ad7b054652d96e082edbfcf96b9cefb8c90`. All 48 checkpoint hashes/payloads, requests/results, full coverage, optimizer/RNG/schedule/H200 contracts, frozen selection and complete report/bootstrap were independently reviewed. No training/evaluation rerun or holdout access.

Controller active time 2,096.64 seconds (0.58 hours); output before review 488.92 MiB. [Frozen protocol](WARMUP_QUALITY_PROTOCOL.md).

## Fresh confirmation means

Seeds 2291/2292/2293; 111,538 validation targets once each. H200 primary; raw correlated diagnostic.

| Schedule | Fixed peak LR | H200 CE64 | H200 CE256 | Raw CE64 | Raw CE256 |
|---|---:|---:|---:|---:|---:|
| baseline | 0.005 | 1.514762 | 1.428944 | 1.521473 | 1.446722 |
| warmup100 | 0.005 | 1.510299 | 1.428140 | 1.516783 | 1.444640 |
| warmup300 | 0.005 | 1.515234 | 1.429972 | 1.521885 | 1.447663 |

The screening delta for the 100-update warmup was −0.006476 CE, with both seeds improving. Fresh confirmation reduced the mean by 0.004462 at width 64 and 0.000805 at width 256; neither met the required gain greater than 0.005, and each width had one worsening seed. Width-64 seed 2292 worsened by 0.001030; width-256 seed 2293 worsened by 0.003743. All three conditional text intervals were negative at width 64, but those intervals do not measure seed-population uncertainty and do not override the failed every-seed and mean-gain criteria. The width-256 H200 intervals included zero.

The 300-update warmup had slightly worse mean H200 CE at both widths. Every width-256 seed favored the 100-update over the 300-update warmup, but that contrast does not establish improvement over the no-warmup incumbent. Raw checkpoints are correlated diagnostics and cannot replace the preregistered H200 endpoint. Measured training-time ratios were within 0.2% of the baseline here; this is not an equivalence or serving-speed claim.

## Paired contrasts

Positive is worse. Intervals use 4,096-token blocks; all three block sizes were audited.

| Endpoint | Width | Contrast | Mean delta | Seed deltas | Conditional 95% text interval | Primary support |
|---|---:|---|---:|---|---|---|
| H200 | 64 | warmup100 minus baseline | -0.004462 | -0.005604 / +0.001030 / -0.008813 | [-0.006909, -0.002180] | False |
| Raw | 64 | warmup100 minus baseline | -0.004690 | -0.005183 / +0.002686 / -0.011572 | [-0.007493, -0.002088] | Diagnostic |
| H200 | 64 | warmup300 minus baseline | +0.000473 | -0.002601 / -0.002391 / +0.006410 | [-0.001658, +0.002558] | False |
| Raw | 64 | warmup300 minus baseline | +0.000412 | -0.002108 / -0.001351 / +0.004695 | [-0.001687, +0.002614] | Diagnostic |
| H200 | 64 | warmup300 minus warmup100 | +0.004935 | +0.003003 / -0.003421 / +0.015223 | [+0.003286, +0.006609] | False |
| Raw | 64 | warmup300 minus warmup100 | +0.005102 | +0.003076 / -0.004037 / +0.016266 | [+0.003372, +0.006812] | Diagnostic |
| H200 | 256 | warmup100 minus baseline | -0.000805 | -0.002992 / -0.003165 / +0.003743 | [-0.002900, +0.001302] | False |
| Raw | 256 | warmup100 minus baseline | -0.002082 | -0.004244 / -0.003998 / +0.001995 | [-0.004089, -0.000108] | Diagnostic |
| H200 | 256 | warmup300 minus baseline | +0.001027 | -0.000635 / -0.001540 / +0.005258 | [-0.001551, +0.003888] | False |
| Raw | 256 | warmup300 minus baseline | +0.000941 | +0.000383 / -0.001311 / +0.003752 | [-0.002031, +0.004208] | Diagnostic |
| H200 | 256 | warmup300 minus warmup100 | +0.001832 | +0.002357 / +0.001624 / +0.001515 | [-0.000797, +0.005057] | False |
| Raw | 256 | warmup300 minus warmup100 | +0.003024 | +0.004628 / +0.002686 / +0.001757 | [+0.000027, +0.006734] | Diagnostic |

## Fixed-LR screens

| Contrast | H200 mean delta | Two seed deltas | Raw mean delta |
|---|---:|---|---:|
| warmup100 minus baseline | -0.006476 | -0.005175 / -0.007778 | -0.006016 |
| warmup300 minus baseline | -0.000439 | +0.008114 / -0.008993 | -0.000015 |
| warmup300 minus warmup100 | +0.006037 | +0.013289 / -0.001215 | +0.006001 |

## Measured training costs

| Schedule | Width | Training seconds | Wall seconds | Ratio versus baseline | Sampled RSS MiB |
|---|---:|---:|---:|---:|---:|
| baseline | 64 | 54.79 | 59.08 | 1.000 | 473.72 |
| baseline | 256 | 125.63 | 134.08 | 1.000 | 482.71 |
| warmup100 | 64 | 54.73 | 59.04 | 0.999 | 473.29 |
| warmup100 | 256 | 125.75 | 134.39 | 1.001 | 482.94 |
| warmup300 | 64 | 54.89 | 59.10 | 1.002 | 473.80 |
| warmup300 | 256 | 125.44 | 134.08 | 0.999 | 482.81 |

## Limits

- Adaptive development on reused validation; conditional text intervals are not seed-population uncertainty, independent holdout or adjusted significance.
- 24 independent training trajectories yield 48 correlated raw/H200 endpoints. No reference aliases or historical pooling.
- Only the frozen eligible width 64 winner can be adopted; no alternate fallback, larger-width reselection, grid extension or completed-trial rerun.
- Common peak LR is inherited from the final reviewed KDA decision, not independently tuned for each warmup schedule; no optimized-schedule or empirical optimal muP-transfer claim.
- Equal 3,000 updates, 3,072,000 targets and constant independent decay; warmup lowers the LR multiplier integral from 2,550 to 2,505 or 2,415 and changes update/decay balance. No identified stability cause.
- Measured training time is separate from quality; no equal-compute, serving-speed or peak-accelerator-memory claim.
- Final H200 tensors equal saved raw-parent shadows; endpoint evidence cannot reconstruct every historical snapshot. Earlier MPS reproducibility failures remain preserved without an identified cause.
- H200 checkpoints are inference-only; resume optimizer training only from their raw parent.
