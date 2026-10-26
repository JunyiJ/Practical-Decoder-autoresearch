# First-layer value reuse quality results

First-layer value reuse comparison completed 36 physical trajectories and 72 correlated raw/H200 outputs, independently reviewed. Common warmup 0 updates. Frozen eligible winner: incumbent_unchanged. Fresh H200 width64/256 deltas versus tuned Local values: Half local +0.003625/+0.002584; First-value blend +0.008219/+0.013734. Retain the prior incumbent under the complete frozen rule.

Historical source `c2c8498a7f18735c3e0cb68d355a7ee587377aa4`. Result `kb_95cf001351fe762d1eea3f24432963e1ff4c69b1f41d78a6d6da7d6f5120d35e`. All72 checkpoint hashes/payloads, requests/results, full coverage, optimizer/RNG/schedule/H200 contracts, frozen selection and complete report/bootstrap were independently reviewed. No training/evaluation rerun or holdout access.

Controller active time 2842.15seconds; output before review 530.92MiB. [Frozen protocol](VALUE_RESIDUAL_PROTOCOL.md).

## Fresh confirmation means

Seeds2471/2472/2473;111,538 validation targets once each. H200 primary; raw correlated diagnostic.

| Family | Selected peak LR | H200 CE64 | H200 CE256 | Raw CE64 | Raw CE256 |
|---|---:|---:|---:|---:|---:|
| local | 0.01 | 1.513767 | 1.423661 | 1.518823 | 1.437272 |
| local_half | 0.01 | 1.517391 | 1.426245 | 1.521536 | 1.441336 |
| first_blend | 0.005 | 1.521986 | 1.437396 | 1.525475 | 1.451718 |
| incumbent_reference | 0.01 | 1.513767 | 1.423661 | 1.518823 | 1.437272 |

## Paired contrasts

Positive is worse. Intervals use4096-token blocks; all three block sizes were audited.

| Endpoint | Width | Contrast | Mean delta | Seed deltas | Conditional95% text interval | Primary support |
|---|---:|---|---:|---|---|---|
| H200 | 64 | local_half minus local | +0.003625 | +0.006498 / -0.005055 / +0.009432 | [+0.001640, +0.005642] | False |
| Raw | 64 | local_half minus local | +0.002713 | +0.004709 / -0.004412 / +0.007843 | [+0.000670, +0.004819] | Diagnostic |
| H200 | 64 | first_blend minus local | +0.008219 | +0.000106 / +0.006809 / +0.017742 | [+0.005655, +0.010939] | False |
| Raw | 64 | first_blend minus local | +0.006652 | -0.003954 / +0.007040 / +0.016871 | [+0.003972, +0.009531] | Diagnostic |
| H200 | 64 | first_blend minus local_half | +0.004594 | -0.006391 / +0.011864 / +0.008310 | [+0.001226, +0.007852] | False |
| Raw | 64 | first_blend minus local_half | +0.003939 | -0.008662 / +0.011452 / +0.009028 | [+0.000895, +0.007033] | Diagnostic |
| H200 | 64 | local minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 64 | local minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 64 | local_half minus incumbent_reference | +0.003625 | +0.006498 / -0.005055 / +0.009432 | [+0.001640, +0.005642] | False |
| Raw | 64 | local_half minus incumbent_reference | +0.002713 | +0.004709 / -0.004412 / +0.007843 | [+0.000670, +0.004819] | Diagnostic |
| H200 | 64 | first_blend minus incumbent_reference | +0.008219 | +0.000106 / +0.006809 / +0.017742 | [+0.005655, +0.010939] | False |
| Raw | 64 | first_blend minus incumbent_reference | +0.006652 | -0.003954 / +0.007040 / +0.016871 | [+0.003972, +0.009531] | Diagnostic |
| H200 | 256 | local_half minus local | +0.002584 | +0.006136 / -0.001953 / +0.003568 | [+0.000577, +0.004640] | False |
| Raw | 256 | local_half minus local | +0.004064 | +0.009776 / -0.002366 / +0.004782 | [+0.002063, +0.006023] | Diagnostic |
| H200 | 256 | first_blend minus local | +0.013734 | +0.012592 / +0.013745 / +0.014866 | [+0.011894, +0.015718] | False |
| Raw | 256 | first_blend minus local | +0.014446 | +0.014644 / +0.013569 / +0.015124 | [+0.012673, +0.016427] | Diagnostic |
| H200 | 256 | first_blend minus local_half | +0.011150 | +0.006455 / +0.015698 / +0.011298 | [+0.009226, +0.013138] | False |
| Raw | 256 | first_blend minus local_half | +0.010381 | +0.004868 / +0.015935 / +0.010342 | [+0.008385, +0.012385] | Diagnostic |
| H200 | 256 | local minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 256 | local minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 256 | local_half minus incumbent_reference | +0.002584 | +0.006136 / -0.001953 / +0.003568 | [+0.000577, +0.004640] | False |
| Raw | 256 | local_half minus incumbent_reference | +0.004064 | +0.009776 / -0.002366 / +0.004782 | [+0.002063, +0.006023] | Diagnostic |
| H200 | 256 | first_blend minus incumbent_reference | +0.013734 | +0.012592 / +0.013745 / +0.014866 | [+0.011894, +0.015718] | False |
| Raw | 256 | first_blend minus incumbent_reference | +0.014446 | +0.014644 / +0.013569 / +0.015124 | [+0.012673, +0.016427] | Diagnostic |

## Fixed-LR screens

| Contrast | H200 mean delta | Two seed deltas | Raw mean delta |
|---|---:|---|---:|
| LR0.0025: local_half minus local | -0.004270 | -0.006495 / -0.002045 | -0.003145 |
| LR0.0025: first_blend minus local | +0.000386 | -0.000789 / +0.001560 | +0.000427 |
| LR0.0025: first_blend minus local_half | +0.004655 | +0.005706 / +0.003605 | +0.003572 |
| LR0.005: local_half minus local | -0.000316 | +0.005496 / -0.006128 | -0.001216 |
| LR0.005: first_blend minus local | +0.004863 | +0.005620 / +0.004107 | +0.003690 |
| LR0.005: first_blend minus local_half | +0.005179 | +0.000124 / +0.010234 | +0.004907 |
| LR0.01: local_half minus local | +0.000120 | -0.000348 / +0.000587 | -0.001311 |
| LR0.01: first_blend minus local | +0.007293 | +0.002039 / +0.012547 | +0.006064 |
| LR0.01: first_blend minus local_half | +0.007173 | +0.002387 / +0.011960 | +0.007375 |

## Measured training costs

| Family | Width | Training seconds | Wall seconds | Ratio versus tuned Local values | Sampled RSS MiB |
|---|---:|---:|---:|---:|---:|
| local | 64 | 55.12 | 59.32 | 1.000 | 475.04 |
| local | 256 | 126.56 | 134.98 | 1.000 | 484.31 |
| local_half | 64 | 55.46 | 59.84 | 1.006 | 475.33 |
| local_half | 256 | 126.80 | 135.25 | 1.002 | 484.70 |
| first_blend | 64 | 55.80 | 60.34 | 1.012 | 475.05 |
| first_blend | 256 | 127.67 | 136.55 | 1.009 | 484.55 |
| incumbent_reference | 64 | 55.12 | 59.32 | 1.000 | 475.04 |
| incumbent_reference | 256 | 126.56 | 134.98 | 1.000 | 484.31 |

## Limits

- Adaptive development on reused validation; conditional text intervals are not seed-population uncertainty, independent holdout or adjusted significance.
- 36 physical trajectories yield 72 correlated raw/H200 endpoints. Reference aliases {'incumbent_reference': 'local'} were declared prospectively and are not independent repeated controls; no historical pooling.
- Only the frozen eligible width64 winner can be adopted; no alternate fallback, larger-width reselection, grid extension or completed-trial rerun.
- Equal finite LR tuning; selected-recipe contrasts include LR choice. A tuned Local values-only gain is attributed to LR selection, not value reuse. Fixed-LR screens are separate.
- Equal 3000 updates and 3072000 targets. Final-reviewed residual gains, gated activation, attention scale, residual topology, table, bias/prior, readout norm, input scale, warmup/cooldown/H200 and independent decay are common. Layer1 is unchanged. Layers2-4 use local V, half local V, or half local V plus half raw first-layer V. The first raw V includes QKV projection bias and remains attached to autograd; its gradient sums its own and all three upper-layer roles. Each layer caches the newly mixed chunk once; cached prefixes are not remixed. No parameters, buffers or construction RNG draws are added. Inherited gain parameters, if present, remain explicit in H200. No attention-concentration diagnosis, isolated mechanism, paper replication, cache compression or optimal muP transfer is claimed. Blend-minus-half-local is reported separately and is not an additional adoption gate.
- Measured training time is separate from quality; no equal-compute, serving-speed or peak-accelerator-memory claim.
- Final H200 tensors equal saved raw-parent shadows; endpoint evidence cannot reconstruct every historical snapshot or historical attached-value/cache operations. Those correctness claims also require frozen actual CPU/MPS prerequisite evidence. Earlier MPS reproducibility failures remain preserved without an identified cause.
- H200 checkpoints are inference-only; resume optimizer training only from their raw parent.

## Interpretation and retained recipe

Retain local attention values with unit outer gains, SwiGLU, attention-logit scale 0.5 and peak LR 0.01. Local values were the best screen and exactly matched the fixed incumbent, so retention was frozen before confirmation. Local and half-local selected LR 0.01 at the upper boundary; first-value blending selected the interior LR 0.005. No grid extension or larger-width reselection follows.

First-value blending worsens H200 validation CE for every fresh seed at both widths: mean increases are 0.008218977 at width 64 and 0.013734169 at width 256. The 4,096-token conditional intervals are [+0.005655377, +0.010938883] and [+0.011893968, +0.015717928]; all three block-size intervals are positive. Width64 seed2471 worsens by only 0.000106467, which is still in the unfavorable direction.

Half-local values worsen the means by 0.003624579 / 0.002583844, with mixed seeds at both widths: seed2472 improves while2471 and2473 worsen. All three conditional text intervals are positive despite this seed disagreement. These intervals condition on the trained seeds and reused text; they are not training-seed population intervals. Blending minus half-local is +0.004594398 at width64 with mixed seeds, and +0.011150325 at width256 with every seed worse. This separate contrast is not an additional adoption gate.

Blending and the controls selected different peak LRs, so their confirmation contrasts measure selected recipes, not an isolated value-reuse mechanism. At fixed LR0.005 in the separate screening panel, blending minus local is +0.004863212 with both seeds worse. Half-local versus local uses the same selected LR0.01. No attention-concentration diagnosis or paper reproduction follows from these results.

Layer1 is unchanged. Upper layers mix only the newly projected values; the raw first-layer value includes its projection bias and remains attached to all four gradient roles. No trainable parameters, initialization draws or cache-shape savings are added. Actual CPU/MPS prerequisites independently verified coupled gradients and full/token/uneven/reset/interleaved caches. Final checkpoint auditing verifies saved semantics and endpoint tensors; it cannot reconstruct every historical graph operation or all101 online averaging snapshots.

Half-local training time increases by 0.61% / 0.19%, and first-value blending by 1.25% / 0.87%. These are descriptive training measurements, not inference-speed or equivalence claims. All arms use equal3,000 updates and3,072,000 targets. Six fixed-reference cells prospectively alias local-value trajectories; they are not independent repeats. No historical control pooling, holdout access, training/scoring rerun or alternative promotion occurred. The first MPS prerequisite's device-string failure, reuse of its already-completed training/scoring, and the later registration-schema repair remain preserved and bound in the preregistered evidence.
