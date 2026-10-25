# SwiGLU hidden-normalization quality results

SwiGLU hidden-normalization completed 36 physical trajectories and 72 correlated raw/H200 outputs, independently reviewed. Frozen choices: none LR 0.005, rms LR 0.005, layer LR 0.005; eligible winner: incumbent_unchanged. Fresh H200 deltas at widths 64/256: RMS minus unchanged -0.000775/-0.003438; LayerNorm minus unchanged -0.002675/-0.002503; LayerNorm minus RMS -0.001900/+0.000935. The complete frozen adoption rule is not met; retain the prior incumbent.

Historical source `5bf7cbfd40e87db0b2f0338248a2b3b6af10cc9b`. Result `kb_4b4d9bebe6865cab1d14608767859ac2d7b18ac11ae373aeef62eaebf43ab045`. All 72 raw/H200 checkpoint hashes and payload contracts, requests/results, optimizer/RNG/schedule/normalization/model-shape/full-coverage contracts, frozen choices, aliases and complete report/bootstrap were independently reviewed. No training/evaluation rerun or holdout access.

Controller active time 2942.80 seconds; output before review 531.54 MiB. [Frozen protocol](FFN_HIDDEN_NORM_PROTOCOL.md).

## Fresh confirmation means

Seeds 2231/2232/2233; 111,538 validation targets once each. H200 is primary; raw is a correlated diagnostic.

| Recipe | Selected LR | H200 CE64 | H200 CE256 | Raw CE64 | Raw CE256 |
|---|---:|---:|---:|---:|---:|
| none | 0.005 | 1.521596 | 1.432887 | 1.526843 | 1.446929 |
| rms | 0.005 | 1.520822 | 1.429449 | 1.525505 | 1.443841 |
| layer | 0.005 | 1.518921 | 1.430384 | 1.523461 | 1.445410 |
| incumbent_reference | 0.005 | 1.521596 | 1.432887 | 1.526843 | 1.446929 |

## Paired contrasts

Positive is worse. Intervals use 4,096-token blocks; all three block sizes were audited. Contrast support alone does not determine adoption.

| Endpoint | Width | Contrast | Mean delta | Paired seed deltas | Conditional 95% text interval | Primary support |
|---|---:|---|---:|---|---|---|
| H200 | 64 | rms minus none | -0.000775 | +0.007558 / -0.002923 / -0.006959 | [-0.003688, +0.002089] | False |
| Raw | 64 | rms minus none | -0.001339 | +0.004187 / -0.003208 / -0.004995 | [-0.004762, +0.001934] | Diagnostic |
| H200 | 64 | layer minus none | -0.002675 | +0.001392 / -0.002016 / -0.007400 | [-0.005598, +0.000253] | False |
| Raw | 64 | layer minus none | -0.003382 | -0.001674 / -0.001822 / -0.006651 | [-0.006382, -0.000309] | Diagnostic |
| H200 | 64 | layer minus rms | -0.001900 | -0.006167 / +0.000906 / -0.000441 | [-0.004243, +0.000561] | False |
| Raw | 64 | layer minus rms | -0.002043 | -0.005861 / +0.001386 / -0.001656 | [-0.004455, +0.000382] | Diagnostic |
| H200 | 64 | none minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 64 | none minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 64 | rms minus incumbent_reference | -0.000775 | +0.007558 / -0.002923 / -0.006959 | [-0.003688, +0.002089] | False |
| Raw | 64 | rms minus incumbent_reference | -0.001339 | +0.004187 / -0.003208 / -0.004995 | [-0.004762, +0.001934] | Diagnostic |
| H200 | 64 | layer minus incumbent_reference | -0.002675 | +0.001392 / -0.002016 / -0.007400 | [-0.005598, +0.000253] | False |
| Raw | 64 | layer minus incumbent_reference | -0.003382 | -0.001674 / -0.001822 / -0.006651 | [-0.006382, -0.000309] | Diagnostic |
| H200 | 256 | rms minus none | -0.003438 | -0.005692 / -0.005285 / +0.000662 | [-0.006514, -0.000088] | False |
| Raw | 256 | rms minus none | -0.003088 | -0.004906 / -0.005888 / +0.001530 | [-0.006654, +0.000724] | Diagnostic |
| H200 | 256 | layer minus none | -0.002503 | -0.006113 / -0.007531 / +0.006135 | [-0.005063, +0.000441] | False |
| Raw | 256 | layer minus none | -0.001520 | -0.005739 / -0.007887 / +0.009066 | [-0.004383, +0.001866] | Diagnostic |
| H200 | 256 | layer minus rms | +0.000935 | -0.000421 / -0.002246 / +0.005473 | [-0.001351, +0.003227] | False |
| Raw | 256 | layer minus rms | +0.001568 | -0.000833 / -0.001999 / +0.007536 | [-0.000944, +0.004187] | Diagnostic |
| H200 | 256 | none minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 256 | none minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 256 | rms minus incumbent_reference | -0.003438 | -0.005692 / -0.005285 / +0.000662 | [-0.006514, -0.000088] | False |
| Raw | 256 | rms minus incumbent_reference | -0.003088 | -0.004906 / -0.005888 / +0.001530 | [-0.006654, +0.000724] | Diagnostic |
| H200 | 256 | layer minus incumbent_reference | -0.002503 | -0.006113 / -0.007531 / +0.006135 | [-0.005063, +0.000441] | False |
| Raw | 256 | layer minus incumbent_reference | -0.001520 | -0.005739 / -0.007887 / +0.009066 | [-0.004383, +0.001866] | Diagnostic |

## Fixed-LR screens

| LR | Contrast | H200 mean delta | Two seed deltas | Raw mean delta |
|---|---|---:|---|---:|
| 0.00125 | rms minus none | -0.014795 | -0.008400 / -0.021191 | -0.014429 |
| 0.00125 | layer minus none | -0.011999 | -0.009694 / -0.014304 | -0.011347 |
| 0.00125 | layer minus rms | +0.002797 | -0.001294 / +0.006887 | +0.003082 |
| 0.0025 | rms minus none | -0.000594 | +0.000628 / -0.001817 | -0.000862 |
| 0.0025 | layer minus none | +0.001001 | +0.000429 / +0.001573 | +0.000344 |
| 0.0025 | layer minus rms | +0.001595 | -0.000199 / +0.003390 | +0.001206 |
| 0.005 | rms minus none | -0.002721 | -0.007348 / +0.001905 | -0.004305 |
| 0.005 | layer minus none | -0.001725 | -0.005100 / +0.001651 | -0.001477 |
| 0.005 | layer minus rms | +0.000997 | +0.002248 / -0.000255 | +0.002827 |

## Measured training costs

| Recipe | Width | Training seconds | Trajectory wall seconds | Ratio versus unchanged | Sampled RSS MiB |
|---|---:|---:|---:|---:|---:|
| none | 64 | 54.50 | 58.76 | 1.000 | 474.73 |
| none | 256 | 125.90 | 133.88 | 1.000 | 482.67 |
| rms | 64 | 62.81 | 67.03 | 1.152 | 476.20 |
| rms | 256 | 141.57 | 149.80 | 1.125 | 485.20 |
| layer | 64 | 56.08 | 60.40 | 1.029 | 477.10 |
| layer | 256 | 130.07 | 138.66 | 1.033 | 485.56 |
| incumbent_reference | 64 | 54.50 | 58.76 | 1.000 | 474.73 |
| incumbent_reference | 256 | 125.90 | 133.88 | 1.000 | 482.67 |

## Limits

- Fresh development seeds and reused full validation remain adaptive evidence. Conditional text intervals are not seed-population uncertainty, independent holdout, or adjusted significance.
- 36 physical trajectories produce 72 correlated raw/H200 outputs. Exact prospective reference aliases are not independent repeats.
- Only the frozen width-64 winner is eligible for adoption. No alternative fallback, larger-width reselection, LR-grid extension or completed-trial rerun.
- Normalization acts on each token’s SwiGLU product, before the unchanged down projection. RMS and LayerNorm change activation and gradient magnitudes; LayerNorm also centers and adds bias. No isolated centering or amplitude causal attribution.
- This is an adaptation motivated by NormFormer, not a full replication or evidence of the same gradient-mismatch mechanism.
- New norm gain/bias vectors use AdamW at base LR and independent decay; hidden matrices retain Muon width factors. Shape algebra does not establish optimal LR transfer.
- Equal 3,000 updates and 3,072,000 targets do not imply equal compute. Training time is measured separately; no serving-speed or peak-accelerator-memory claim.
- H200 tensors match final raw-parent shadows, but saved endpoints cannot reconstruct every historical averaging snapshot. Preserve earlier MPS reproducibility failures without an identified cause.
- H200 checkpoints are inference-only; optimizer continuation requires the original raw training parent.
