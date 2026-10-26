# Independent-decay quality results

Independent-decay comparison completed 36 physical trajectories and 72 correlated raw/H200 outputs, independently reviewed. Frozen choices: anchor LR 0.005, weak LR 0.005, strong LR 0.005; eligible winner: incumbent_unchanged. Fresh H200 deltas at widths 64/256: Weak decay minus anchor +0.004269/+0.003997; Strong decay minus anchor -0.001221/-0.010659; Strong decay minus weak -0.005491/-0.014657. The complete frozen adoption rule is not met; retain the prior incumbent.

Historical source `267284b002decacf102a7dc2c953ff926c9ae5ef`. Result `kb_c1de5a623f18adcc0a078ff33fcc7de3a896e36090d683c19ff744bf8fc0cf2e`. All 72 raw/H200 checkpoint hashes and payload contracts, requests/results, optimizer/decay/RNG/schedule/common-model-shape/full-coverage contracts, frozen choices, aliases and complete report/bootstrap were independently reviewed. No training/evaluation rerun or holdout access.

Controller active time 2813.93 seconds; output before review 530.42 MiB. [Frozen protocol](DECAY_QUALITY_PROTOCOL.md).

## Fresh confirmation means

Seeds 2251/2252/2253; 111,538 validation targets once each. H200 is primary; raw is a correlated diagnostic.

| Recipe | Decay | Selected LR | H200 CE64 | H200 CE256 | Raw CE64 | Raw CE256 |
|---|---:|---:|---:|---:|---:|---:|
| anchor | 6e-05 | 0.005 | 1.518961 | 1.432232 | 1.524831 | 1.448259 |
| weak | 2e-05 | 0.005 | 1.523230 | 1.436229 | 1.529411 | 1.455176 |
| strong | 0.00018 | 0.005 | 1.517739 | 1.421573 | 1.523552 | 1.434108 |
| incumbent_reference | 6e-05 | 0.005 | 1.518961 | 1.432232 | 1.524831 | 1.448259 |

## Paired contrasts

Positive is worse. Intervals use 4,096-token blocks; all three block sizes were audited. Contrast support alone does not determine adoption.

| Endpoint | Width | Contrast | Mean delta | Paired seed deltas | Conditional 95% text interval | Primary support |
|---|---:|---|---:|---|---|---|
| H200 | 64 | weak minus anchor | +0.004269 | +0.003035 / +0.008681 / +0.001092 | [+0.002475, +0.006246] | False |
| Raw | 64 | weak minus anchor | +0.004580 | +0.000441 / +0.010335 / +0.002964 | [+0.002610, +0.006629] | Diagnostic |
| H200 | 64 | strong minus anchor | -0.001221 | -0.002225 / +0.002907 / -0.004346 | [-0.003765, +0.001085] | False |
| Raw | 64 | strong minus anchor | -0.001280 | -0.003296 / +0.004374 / -0.004916 | [-0.005364, +0.002374] | Diagnostic |
| H200 | 64 | strong minus weak | -0.005491 | -0.005259 / -0.005775 / -0.005438 | [-0.008820, -0.002316] | True |
| Raw | 64 | strong minus weak | -0.005860 | -0.003738 / -0.005962 / -0.007881 | [-0.010552, -0.001530] | Diagnostic |
| H200 | 64 | anchor minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 64 | anchor minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 64 | weak minus incumbent_reference | +0.004269 | +0.003035 / +0.008681 / +0.001092 | [+0.002475, +0.006246] | False |
| Raw | 64 | weak minus incumbent_reference | +0.004580 | +0.000441 / +0.010335 / +0.002964 | [+0.002610, +0.006629] | Diagnostic |
| H200 | 64 | strong minus incumbent_reference | -0.001221 | -0.002225 / +0.002907 / -0.004346 | [-0.003765, +0.001085] | False |
| Raw | 64 | strong minus incumbent_reference | -0.001280 | -0.003296 / +0.004374 / -0.004916 | [-0.005364, +0.002374] | Diagnostic |
| H200 | 256 | weak minus anchor | +0.003997 | +0.004895 / +0.004084 / +0.003014 | [+0.001813, +0.005964] | False |
| Raw | 256 | weak minus anchor | +0.006917 | +0.006995 / +0.009135 / +0.004621 | [+0.004628, +0.009055] | Diagnostic |
| H200 | 256 | strong minus anchor | -0.010659 | -0.012610 / -0.012248 / -0.007120 | [-0.014356, -0.007294] | True |
| Raw | 256 | strong minus anchor | -0.014151 | -0.015036 / -0.014245 / -0.013171 | [-0.018806, -0.009992] | Diagnostic |
| H200 | 256 | strong minus weak | -0.014657 | -0.017505 / -0.016331 / -0.010134 | [-0.018216, -0.011316] | True |
| Raw | 256 | strong minus weak | -0.021067 | -0.022031 / -0.023380 / -0.017791 | [-0.026107, -0.016452] | Diagnostic |
| H200 | 256 | anchor minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 256 | anchor minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 256 | weak minus incumbent_reference | +0.003997 | +0.004895 / +0.004084 / +0.003014 | [+0.001813, +0.005964] | False |
| Raw | 256 | weak minus incumbent_reference | +0.006917 | +0.006995 / +0.009135 / +0.004621 | [+0.004628, +0.009055] | Diagnostic |
| H200 | 256 | strong minus incumbent_reference | -0.010659 | -0.012610 / -0.012248 / -0.007120 | [-0.014356, -0.007294] | True |
| Raw | 256 | strong minus incumbent_reference | -0.014151 | -0.015036 / -0.014245 / -0.013171 | [-0.018806, -0.009992] | Diagnostic |

## Fixed-LR screens

| LR | Contrast | H200 mean delta | Two seed deltas | Raw mean delta |
|---|---|---:|---|---:|
| 0.00125 | weak minus anchor | -0.002258 | -0.002045 / -0.002470 | -0.002674 |
| 0.00125 | strong minus anchor | +0.014315 | +0.015251 / +0.013380 | +0.021398 |
| 0.00125 | strong minus weak | +0.016573 | +0.017296 / +0.015850 | +0.024072 |
| 0.0025 | weak minus anchor | +0.001184 | -0.002005 / +0.004373 | +0.002295 |
| 0.0025 | strong minus anchor | +0.004698 | +0.004841 / +0.004556 | +0.006936 |
| 0.0025 | strong minus weak | +0.003514 | +0.006846 / +0.000183 | +0.004641 |
| 0.005 | weak minus anchor | -0.001964 | +0.004399 / -0.008326 | -0.002824 |
| 0.005 | strong minus anchor | -0.002842 | -0.001303 / -0.004382 | -0.002542 |
| 0.005 | strong minus weak | -0.000879 | -0.005702 / +0.003945 | +0.000282 |

## Measured training costs

| Recipe | Width | Training seconds | Trajectory wall seconds | Ratio versus anchor | Sampled RSS MiB |
|---|---:|---:|---:|---:|---:|
| anchor | 64 | 54.75 | 59.05 | 1.000 | 473.78 |
| anchor | 256 | 125.79 | 134.39 | 1.000 | 482.46 |
| weak | 64 | 54.88 | 59.20 | 1.002 | 473.71 |
| weak | 256 | 125.84 | 134.76 | 1.000 | 482.70 |
| strong | 64 | 54.94 | 59.23 | 1.003 | 473.23 |
| strong | 256 | 125.77 | 133.86 | 1.000 | 482.60 |
| incumbent_reference | 64 | 54.75 | 59.05 | 1.000 | 473.78 |
| incumbent_reference | 256 | 125.79 | 134.39 | 1.000 | 482.46 |

## Limits

- Fresh development seeds and reused full validation remain adaptive evidence. Conditional text intervals are not seed-population uncertainty, independent holdout, or adjusted significance.
- 36 physical trajectories produce 72 correlated raw/H200 outputs. Exact prospective reference aliases are not independent repeats.
- Only the frozen width-64 winner is eligible for adoption. No alternative fallback, larger-width reselection, LR-grid extension or completed-trial rerun.
- Independent coefficients weak=.00002, anchor=.00006, strong=.00018 apply once per update before inner optimization to all parameters with gradients; inner optimizer decay stays zero. No width or LR scaling of the coefficient.
- The same final-reviewed model,3000-update cooldown and H200 policy are shared. All three arms selected peak LR0.005, so confirmation compares decay at the same selected LR. Equal bounded tuning allowed different choices, but none occurred in this round. Nominal cumulative shrinkage is algebra, not final norm or quality prediction.
- Earlier two-layer GELU decay probes remain closed and are not pooled. This recipe interaction comparison does not identify overfitting, a regularization mechanism, or optimal LR/decay/muP transfer.
- Equal 3,000 updates and 3,072,000 targets do not imply equal compute. Training time is measured separately; no serving-speed or peak-accelerator-memory claim.
- H200 tensors match final raw-parent shadows, but saved endpoints cannot reconstruct every historical averaging snapshot. Preserve earlier MPS reproducibility failures without an identified cause.
- H200 checkpoints are inference-only; optimizer continuation requires the original raw training parent.

Documentation clarification: all arms selected LR0.005; the generic tuning caveat was corrected without changing any evidence. Append-only clarification `kb_60e01e3d12b08ad3c56f86b2e56faee1b3afa5b99c5f0ce9f0d18ce9604b15e6`.
