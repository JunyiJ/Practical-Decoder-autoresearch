# Logit-z objective quality results

Logit-z objective comparison completed 36 physical trajectories and 72 correlated raw/H200 outputs, independently reviewed. Common warmup 0 updates. Frozen eligible winner: incumbent_unchanged. Fresh H200 width64/256 deltas versus tuned zero-penalty control: lambda1e-4 -0.005551/-0.001172; lambda1e-3 -0.002380/-0.004684. Retain the prior incumbent under the complete frozen rule.

Historical source `37c3150d946c1d4884e88d48021b36801d626465`. Result `kb_78ee39c6d3c927c76877e8f4559b47919fe4e794295b07fb200db9f4728de1c6`. All72 checkpoint hashes/payloads, requests/results, full coverage, optimizer/RNG/schedule/H200 contracts, frozen selection and complete report/bootstrap were independently reviewed. No training/evaluation rerun or holdout access.

Controller active time 3005.89seconds; output before review 531.39MiB. [Frozen protocol](LOGIT_Z_PROTOCOL.md).

## Fresh confirmation means

Seeds2511/2512/2513;111,538 validation targets once each. H200 primary; raw correlated diagnostic.

| Family | Selected peak LR | H200 CE64 | H200 CE256 | Raw CE64 | Raw CE256 |
|---|---:|---:|---:|---:|---:|
| none | 0.01 | 1.515571 | 1.428035 | 1.521296 | 1.445966 |
| z1e4 | 0.005 | 1.510020 | 1.426864 | 1.515396 | 1.443341 |
| z1e3 | 0.01 | 1.513191 | 1.423352 | 1.518824 | 1.442161 |
| incumbent_reference | 0.01 | 1.515571 | 1.428035 | 1.521296 | 1.445966 |

## Paired contrasts

Positive is worse. Intervals use4096-token blocks; all three block sizes were audited.

| Endpoint | Width | Contrast | Mean delta | Seed deltas | Conditional95% text interval | Primary support |
|---|---:|---|---:|---|---|---|
| H200 | 64 | z1e4 minus none | -0.005551 | -0.009698 / -0.012644 / +0.005688 | [-0.008139, -0.003084] | False |
| Raw | 64 | z1e4 minus none | -0.005901 | -0.010008 / -0.014762 / +0.007068 | [-0.008412, -0.003506] | Diagnostic |
| H200 | 64 | z1e3 minus none | -0.002380 | -0.004124 / -0.003156 / +0.000140 | [-0.005544, +0.000442] | False |
| Raw | 64 | z1e3 minus none | -0.002473 | -0.003526 / -0.004084 / +0.000191 | [-0.006097, +0.000820] | Diagnostic |
| H200 | 64 | z1e3 minus z1e4 | +0.003171 | +0.005574 / +0.009487 / -0.005548 | [-0.000206, +0.006834] | False |
| Raw | 64 | z1e3 minus z1e4 | +0.003428 | +0.006482 / +0.010679 / -0.006877 | [+0.000306, +0.006659] | Diagnostic |
| H200 | 64 | none minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 64 | none minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 64 | z1e4 minus incumbent_reference | -0.005551 | -0.009698 / -0.012644 / +0.005688 | [-0.008139, -0.003084] | False |
| Raw | 64 | z1e4 minus incumbent_reference | -0.005901 | -0.010008 / -0.014762 / +0.007068 | [-0.008412, -0.003506] | Diagnostic |
| H200 | 64 | z1e3 minus incumbent_reference | -0.002380 | -0.004124 / -0.003156 / +0.000140 | [-0.005544, +0.000442] | False |
| Raw | 64 | z1e3 minus incumbent_reference | -0.002473 | -0.003526 / -0.004084 / +0.000191 | [-0.006097, +0.000820] | Diagnostic |
| H200 | 256 | z1e4 minus none | -0.001172 | -0.002357 / +0.006333 / -0.007491 | [-0.005811, +0.002659] | False |
| Raw | 256 | z1e4 minus none | -0.002625 | -0.002521 / +0.006047 / -0.011401 | [-0.007961, +0.001970] | Diagnostic |
| H200 | 256 | z1e3 minus none | -0.004684 | -0.005402 / +0.003098 / -0.011747 | [-0.008120, -0.001609] | False |
| Raw | 256 | z1e3 minus none | -0.003805 | -0.002865 / +0.003611 / -0.012162 | [-0.007105, -0.000845] | Diagnostic |
| H200 | 256 | z1e3 minus z1e4 | -0.003512 | -0.003045 / -0.003235 / -0.004256 | [-0.006510, -0.000454] | False |
| Raw | 256 | z1e3 minus z1e4 | -0.001180 | -0.000344 / -0.002436 / -0.000761 | [-0.005017, +0.003042] | Diagnostic |
| H200 | 256 | none minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 256 | none minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 256 | z1e4 minus incumbent_reference | -0.001172 | -0.002357 / +0.006333 / -0.007491 | [-0.005811, +0.002659] | False |
| Raw | 256 | z1e4 minus incumbent_reference | -0.002625 | -0.002521 / +0.006047 / -0.011401 | [-0.007961, +0.001970] | Diagnostic |
| H200 | 256 | z1e3 minus incumbent_reference | -0.004684 | -0.005402 / +0.003098 / -0.011747 | [-0.008120, -0.001609] | False |
| Raw | 256 | z1e3 minus incumbent_reference | -0.003805 | -0.002865 / +0.003611 / -0.012162 | [-0.007105, -0.000845] | Diagnostic |

## Fixed-LR screens

| Contrast | H200 mean delta | Two seed deltas | Raw mean delta |
|---|---:|---|---:|
| LR0.0025: z1e4 minus none | -0.000502 | -0.000697 / -0.000308 | -0.000622 |
| LR0.0025: z1e3 minus none | -0.004365 | -0.000092 / -0.008638 | -0.004488 |
| LR0.0025: z1e3 minus z1e4 | -0.003862 | +0.000605 / -0.008330 | -0.003867 |
| LR0.005: z1e4 minus none | +0.000251 | +0.001500 / -0.000997 | +0.000775 |
| LR0.005: z1e3 minus none | -0.002058 | -0.005139 / +0.001024 | -0.001394 |
| LR0.005: z1e3 minus z1e4 | -0.002309 | -0.006639 / +0.002022 | -0.002169 |
| LR0.01: z1e4 minus none | +0.006940 | +0.010035 / +0.003846 | +0.006098 |
| LR0.01: z1e3 minus none | +0.000695 | -0.000789 / +0.002178 | +0.001078 |
| LR0.01: z1e3 minus z1e4 | -0.006246 | -0.010823 / -0.001668 | -0.005020 |

## Measured training costs

| Family | Width | Training seconds | Wall seconds | Ratio versus tuned zero-penalty control | Sampled RSS MiB |
|---|---:|---:|---:|---:|---:|
| none | 64 | 57.00 | 61.93 | 1.000 | 475.06 |
| none | 256 | 129.73 | 138.35 | 1.000 | 483.84 |
| z1e4 | 64 | 60.19 | 65.08 | 1.056 | 475.68 |
| z1e4 | 256 | 132.65 | 141.86 | 1.022 | 484.92 |
| z1e3 | 64 | 60.18 | 65.05 | 1.056 | 475.71 |
| z1e3 | 256 | 132.50 | 141.68 | 1.021 | 485.94 |
| incumbent_reference | 64 | 57.00 | 61.93 | 1.000 | 475.06 |
| incumbent_reference | 256 | 129.73 | 138.35 | 1.000 | 483.84 |

## Limits

- Adaptive development on reused validation; conditional text intervals are not seed-population uncertainty, independent holdout or adjusted significance.
- 36 physical trajectories yield 72 correlated raw/H200 endpoints. Reference aliases {'incumbent_reference': 'none'} were declared prospectively and are not independent repeated controls; no historical pooling.
- Only the frozen eligible width64 winner can be adopted; no alternate fallback, larger-width reselection, grid extension or completed-trial rerun.
- Equal finite LR tuning; selected-recipe contrasts include LR choice. A gain from tuning the zero-penalty control is attributed to LR selection, not the penalty. Fixed-LR screens are separate.
- Equal 3000 updates and3072000 targets at context128. Final-reviewed position, values, gains, gated activation, attention scale, topology, tables, bias/prior, readout normalization, input scale, schedule/H200 and independent decay are inherited unchanged. Training loss is CE plus lambda times the mean squared logsumexp of FINAL FP32 logits on exactly the CE valid targets, with coefficients0/1e-4/1e-3 applied once. No centering, log-vocabulary subtraction, clipping, detach, temperature or width-dependent coefficient. Zero uses the historical exact CE path and leaves logZ diagnostics null. No added learned parameters or initialization draws. Every validation score and ranking is pure CE; training objective metrics are separate. The penalty is sensitive to a common logit shift unlike CE. Nonzero-coefficient contrasts are separate, not an extra adoption gate. No diagnosed local instability, isolated-confidence cause, published-recipe replication or optimal muP claim.
- Measured training time is separate from quality; no equal-compute, serving-speed or peak-accelerator-memory claim.
- Final H200 tensors equal saved raw-parent shadows; endpoint evidence cannot reconstruct every historical snapshot or historical objective calls, pure-CE evaluation exclusion, attached-value gradients or cache operations. Those correctness claims also require frozen actual CPU/MPS prerequisite evidence. Earlier MPS reproducibility failures remain preserved without an identified cause.
- H200 checkpoints are inference-only; resume optimizer training only from their raw parent.

## Interpretation and preserved prerequisite limitations

Logit-z objective comparison completed 36 physical trajectories and 72 correlated raw/H200 outputs, independently reviewed. Common warmup 0 updates. Frozen eligible winner: incumbent_unchanged. Fresh H200 width64/256 deltas versus tuned zero-penalty control: lambda1e-4 -0.005551/-0.001172; lambda1e-3 -0.002380/-0.004684. Retain the prior incumbent under the complete frozen rule.

Retain the zero-penalty incumbent: RoPE/local values/unit gains/SiLU, attention scale0.5 and peak LR0.01. Screening selected zero and lambda0.001 at LR0.01 (upper boundary), and lambda0.0001 at LR0.005 (interior). Zero was the best screen and exactly matched the fixed incumbent, so retention was frozen before fresh confirmation. No larger-width promotion or grid extension follows.

| Objective | Selected peak LR | Fresh H200 CE64 | Fresh H200 CE256 |
|---|---:|---:|---:|
| none | 0.01 | 1.515571303 | 1.428035473 |
| z1e4 | 0.005 | 1.510019993 | 1.426863895 |
| z1e3 | 0.01 | 1.513191110 | 1.423351680 |

Lambda0.0001 minus zero:

- Width64: mean -0.005551310; mixed seeds; seeds2511/2512/2513 -0.009697821/-0.012643634/+0.005687525; 4,096-token conditional interval [-0.008139098, -0.003084299]; all three text intervals negative.
- Width256: mean -0.001171578; mixed seeds; seeds2511/2512/2513 -0.002356551/+0.006333162/-0.007491344; 4,096-token conditional interval [-0.005810937, +0.002659157]; text intervals do not uniformly exclude zero.

Lambda0.001 minus zero:

- Width64: mean -0.002380193; mixed seeds; seeds2511/2512/2513 -0.004123911/-0.003156224/+0.000139554; 4,096-token conditional interval [-0.005543544, +0.000441937]; text intervals do not uniformly exclude zero.
- Width256: mean -0.004683793; mixed seeds; seeds2511/2512/2513 -0.005402021/+0.003097801/-0.011747159; 4,096-token conditional interval [-0.008120289, -0.001608820]; all three text intervals negative.

Lambda0.001 minus lambda0.0001:

- Width64: mean +0.003171117; mixed seeds; seeds2511/2512/2513 +0.005573911/+0.009487411/-0.005547970; 4,096-token conditional interval [-0.000206288, +0.006834221]; text intervals do not uniformly exclude zero.
- Width256: mean -0.003512215; every seed better; seeds2511/2512/2513 -0.003045470/-0.003235360/-0.004255816; 4,096-token conditional interval [-0.006509739, -0.000454377]; all three text intervals negative.

Both penalties improve fresh mean CE at both widths, but each worsens seed2513 at width64 and seed2512 at width256. Lambda0.0001 meets the mean-gain threshold only at width64; lambda0.001 stays below0.005 at both widths. Negative conditional text intervals for lambda0.0001 at width64 and lambda0.001 at width256 do not remove training-seed failures. These intervals condition on the trained seeds and reused validation text, not seed-population uncertainty or a holdout.

The selected lambda0.0001 recipe uses a different LR from zero and lambda0.001; its confirmation contrasts combine objective and LR choices. The nine fixed-LR screen contrasts are reported separately. Lambda0.001 versus lambda0.0001 improves every fresh width256 seed, but is a separate contrast and not an alternative adoption route. No isolated regularization mechanism, diagnosed instability, confidence effect or optimal muP claim follows.

The penalty is applied once to mean squared logsumexp of final FP32 logits on the CE-valid targets. It changes under a common logit shift, while CE does not. All sampled/full validation, selection and adoption use pure CE. Training CE, logZ statistics, weighted penalty and total are recorded separately. Zero preserves exact historical CE arithmetic and null logZ diagnostics. No new learned parameters or initialization draws were introduced.

All72 saved checkpoint/request/result/hash/config/objective/optimizer/RNG/schedule/H200/full111538-target coverage contracts and the entire report, frozen selection, six aliases and all primary/raw bootstrap intervals passed independent final review on the frozen source before documentation. Final endpoints cannot reconstruct every historical objective call, penalty exclusion, cache operation or101 EMA snapshots; the actual CPU/MPS prerequisite evidence remains necessary.

CPU prerequisite failures remain preserved: an atomic-file monitoring race left one completed test worker with unavailable OS exit code and a monitoring gap; a later continuation counted retained failed scratch in its active budget. Reviewed continuations excluded successful cases and stayed within original budgets. All5689 unique assertions passed, but four interrupted-prefix test durations and their warning summary remain unavailable; no monolithic exit0 or whole-suite zero-warning claim is made. The MPS prerequisite reviewer duplicate-width metadata failure and its recorder indentation failure are preserved; the repaired reviewer checked the same six fresh pilots without retraining or rescoring.

z1e4 descriptive training-time changes versus zero at widths64/256: +5.60%/+2.25%. These are training measurements, not inference-speed claims.

z1e3 descriptive training-time changes versus zero at widths64/256: +5.58%/+2.13%. These are training measurements, not inference-speed claims.

The round used 3005.89 active seconds (0.83 hours) and 531.39 MiB before review. All arms use3,000 updates and3,072,000 targets, not equal compute. Six fixed-reference cells alias zero-penalty trajectories prospectively and are not independent repeats. No historical pooling, holdout access, training/scoring rerun or posthoc promotion occurred. [Full results, fixed-LR screens and costs](LOGIT_Z_RESULTS.md).

Next separate preparation compares standard softmax, visible-prefix attenuation and fixed-zero null attention. Its256 metadata/scalar/controller checks passed;28 actual model/trainer cases remain unexecuted. Final logit review now retains the zero objective, but source integration, bounded CPU validation, fresh serial MPS and resource/publication/preregistration gates remain required. All old studies and permanent stop006 stay closed; quality research continues until human interruption.
