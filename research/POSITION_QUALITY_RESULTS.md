# Position-family quality results

Position-family comparison completed 36 physical trajectories and 72 correlated raw/H200 outputs, independently reviewed. Common warmup 0 updates. Frozen eligible winner: incumbent_unchanged. Fresh H200 width64/256 deltas versus tuned RoPE: No explicit position +0.124521/+0.063084; ALiBi +0.092128/+0.006974. Retain the prior incumbent under the complete frozen rule.

Historical source `7c2c6d5a82c20009f0b31f9b8227558004dd96c5`. Result `kb_10c48e5cb5febe97dd32c193aa36c62a2830f5223e79695b179a05c3d0b7b9b5`. All72 checkpoint hashes/payloads, requests/results, full coverage, optimizer/RNG/schedule/H200 contracts, frozen selection and complete report/bootstrap were independently reviewed. No training/evaluation rerun or holdout access.

Controller active time 2754.99seconds; output before review 527.82MiB. [Frozen protocol](POSITION_QUALITY_PROTOCOL.md).

## Fresh confirmation means

Seeds2491/2492/2493;111,538 validation targets once each. H200 primary; raw correlated diagnostic.

| Family | Selected peak LR | H200 CE64 | H200 CE256 | Raw CE64 | Raw CE256 |
|---|---:|---:|---:|---:|---:|
| rope | 0.01 | 1.516183 | 1.426079 | 1.522147 | 1.443061 |
| nope | 0.01 | 1.640704 | 1.489163 | 1.648649 | 1.500697 |
| alibi | 0.01 | 1.608311 | 1.433052 | 1.616107 | 1.449954 |
| incumbent_reference | 0.01 | 1.516183 | 1.426079 | 1.522147 | 1.443061 |

## Paired contrasts

Positive is worse. Intervals use4096-token blocks; all three block sizes were audited.

| Endpoint | Width | Contrast | Mean delta | Seed deltas | Conditional95% text interval | Primary support |
|---|---:|---|---:|---|---|---|
| H200 | 64 | nope minus rope | +0.124521 | +0.150288 / +0.110159 / +0.113116 | [+0.117086, +0.131501] | False |
| Raw | 64 | nope minus rope | +0.126502 | +0.153131 / +0.111116 / +0.115258 | [+0.119999, +0.132949] | Diagnostic |
| H200 | 64 | alibi minus rope | +0.092128 | +0.103185 / +0.094596 / +0.078603 | [+0.084313, +0.099474] | False |
| Raw | 64 | alibi minus rope | +0.093959 | +0.106512 / +0.096676 / +0.078690 | [+0.086715, +0.100916] | Diagnostic |
| H200 | 64 | alibi minus nope | -0.032393 | -0.047104 / -0.015563 / -0.034512 | [-0.035115, -0.029708] | True |
| Raw | 64 | alibi minus nope | -0.032542 | -0.046619 / -0.014440 / -0.036568 | [-0.035704, -0.029487] | Diagnostic |
| H200 | 64 | rope minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 64 | rope minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 64 | nope minus incumbent_reference | +0.124521 | +0.150288 / +0.110159 / +0.113116 | [+0.117086, +0.131501] | False |
| Raw | 64 | nope minus incumbent_reference | +0.126502 | +0.153131 / +0.111116 / +0.115258 | [+0.119999, +0.132949] | Diagnostic |
| H200 | 64 | alibi minus incumbent_reference | +0.092128 | +0.103185 / +0.094596 / +0.078603 | [+0.084313, +0.099474] | False |
| Raw | 64 | alibi minus incumbent_reference | +0.093959 | +0.106512 / +0.096676 / +0.078690 | [+0.086715, +0.100916] | Diagnostic |
| H200 | 256 | nope minus rope | +0.063084 | +0.063807 / +0.062449 / +0.062997 | [+0.056522, +0.068893] | False |
| Raw | 256 | nope minus rope | +0.057636 | +0.052854 / +0.060441 / +0.059614 | [+0.051526, +0.063176] | Diagnostic |
| H200 | 256 | alibi minus rope | +0.006974 | +0.007310 / +0.004770 / +0.008842 | [+0.003082, +0.010596] | False |
| Raw | 256 | alibi minus rope | +0.006894 | +0.005292 / +0.003788 / +0.011601 | [+0.002809, +0.010580] | Diagnostic |
| H200 | 256 | alibi minus nope | -0.056111 | -0.056497 / -0.057679 / -0.054155 | [-0.060244, -0.051370] | True |
| Raw | 256 | alibi minus nope | -0.050742 | -0.047562 / -0.056653 / -0.048013 | [-0.054952, -0.046052] | Diagnostic |
| H200 | 256 | rope minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | False |
| Raw | 256 | rope minus incumbent_reference | +0.000000 | +0.000000 / +0.000000 / +0.000000 | [+0.000000, +0.000000] | Diagnostic |
| H200 | 256 | nope minus incumbent_reference | +0.063084 | +0.063807 / +0.062449 / +0.062997 | [+0.056522, +0.068893] | False |
| Raw | 256 | nope minus incumbent_reference | +0.057636 | +0.052854 / +0.060441 / +0.059614 | [+0.051526, +0.063176] | Diagnostic |
| H200 | 256 | alibi minus incumbent_reference | +0.006974 | +0.007310 / +0.004770 / +0.008842 | [+0.003082, +0.010596] | False |
| Raw | 256 | alibi minus incumbent_reference | +0.006894 | +0.005292 / +0.003788 / +0.011601 | [+0.002809, +0.010580] | Diagnostic |

## Fixed-LR screens

| Contrast | H200 mean delta | Two seed deltas | Raw mean delta |
|---|---:|---|---:|
| LR0.0025: nope minus rope | +0.139803 | +0.146766 / +0.132841 | +0.137730 |
| LR0.0025: alibi minus rope | +0.104834 | +0.107490 / +0.102178 | +0.105041 |
| LR0.0025: alibi minus nope | -0.034969 | -0.039276 / -0.030663 | -0.032689 |
| LR0.005: nope minus rope | +0.120352 | +0.122190 / +0.118514 | +0.120019 |
| LR0.005: alibi minus rope | +0.099804 | +0.098892 / +0.100716 | +0.099611 |
| LR0.005: alibi minus nope | -0.020549 | -0.023299 / -0.017798 | -0.020409 |
| LR0.01: nope minus rope | +0.125511 | +0.122281 / +0.128741 | +0.128396 |
| LR0.01: alibi minus rope | +0.098368 | +0.097569 / +0.099166 | +0.100179 |
| LR0.01: alibi minus nope | -0.027144 | -0.024712 / -0.029575 | -0.028216 |

## Measured training costs

| Family | Width | Training seconds | Wall seconds | Ratio versus tuned RoPE | Sampled RSS MiB |
|---|---:|---:|---:|---:|---:|
| rope | 64 | 55.09 | 59.50 | 1.000 | 474.79 |
| rope | 256 | 126.45 | 134.68 | 1.000 | 484.74 |
| nope | 64 | 48.51 | 52.22 | 0.881 | 473.85 |
| nope | 256 | 117.50 | 124.75 | 0.929 | 483.59 |
| alibi | 64 | 57.51 | 62.47 | 1.044 | 474.59 |
| alibi | 256 | 126.38 | 134.48 | 0.999 | 483.59 |
| incumbent_reference | 64 | 55.09 | 59.50 | 1.000 | 474.79 |
| incumbent_reference | 256 | 126.45 | 134.68 | 1.000 | 484.74 |

## Limits

- Adaptive development on reused validation; conditional text intervals are not seed-population uncertainty, independent holdout or adjusted significance.
- 36 physical trajectories yield 72 correlated raw/H200 endpoints. Reference aliases {'incumbent_reference': 'rope'} were declared prospectively and are not independent repeated controls; no historical pooling.
- Only the frozen eligible width64 winner can be adopted; no alternate fallback, larger-width reselection, grid extension or completed-trial rerun.
- Equal finite LR tuning; selected-recipe contrasts include LR choice. A tuned RoPE-only gain is attributed to LR selection, not position family. Fixed-LR screens are separate.
- Equal 3000 updates and3072000 targets at context128. Final-reviewed value reuse, residual gains, gated activation, attention scale, topology, table, bias/prior, readout norm, input scale, schedule/H200 and independent decay are common. RoPE rotates Q/K; nope uses unrotated Q/K without explicit position, while the causal mask remains order dependent. ALiBi adds fixed head-order slopes2**(-8*(h+1)/H) times negative distance AFTER content scaling and before masking. Head count changes the slope repertoire across widths. No new learned parameters or initialization RNG; deterministic frequency buffers exist only for RoPE. Legacy rope0 outside the explicit named policy still means learned positions. Inherited value mixing and explicit gain shadows remain unchanged. ALiBi-minus-nope is reported separately, not an extra adoption gate. No extrapolation, isolated-mechanism, paper-replication or optimal muP claim.
- Measured training time is separate from quality; no equal-compute, serving-speed or peak-accelerator-memory claim.
- Final H200 tensors equal saved raw-parent shadows; endpoint evidence cannot reconstruct every historical snapshot or historical position-bias placement, attached-value gradients or cache operations. Those correctness claims also require frozen actual CPU/MPS prerequisite evidence. Earlier MPS reproducibility failures remain preserved without an identified cause.
- H200 checkpoints are inference-only; resume optimizer training only from their raw parent.

## Interpretation and retained recipe

Retain RoPE with local attention values, unit outer gains, SwiGLU, attention-logit scale0.5 and peak LR0.01. RoPE was the best screen and exactly matched the fixed incumbent, so retention was frozen before confirmation. All three families selected LR0.01 at the upper boundary. No grid extension or larger-width reselection follows.

No explicit position minus RoPE:

- Width64: mean +0.124521092; every seed worse; seeds2491/2492/2493 +0.150288282/+0.110159242/+0.113115752; 4,096-token conditional interval [+0.117085876, +0.131500713]; all three conditional text intervals positive.
- Width256: mean +0.063084358; every seed worse; seeds2491/2492/2493 +0.063806707/+0.062449468/+0.062996900; 4,096-token conditional interval [+0.056522236, +0.068892507]; all three conditional text intervals positive.

ALiBi minus RoPE:

- Width64: mean +0.092128039; every seed worse; seeds2491/2492/2493 +0.103184531/+0.094596176/+0.078603410; 4,096-token conditional interval [+0.084312615, +0.099473631]; all three conditional text intervals positive.
- Width256: mean +0.006973779; every seed worse; seeds2491/2492/2493 +0.007309636/+0.004770165/+0.008841537; 4,096-token conditional interval [+0.003082031, +0.010596149]; all three conditional text intervals positive.

ALiBi minus no explicit position:

- Width64: mean -0.032393053; every seed better; seeds2491/2492/2493 -0.047103752/-0.015563066/-0.034512343; 4,096-token conditional interval [-0.035115298, -0.029708483]; all three conditional text intervals negative.
- Width256: mean -0.056110579; every seed better; seeds2491/2492/2493 -0.056497070/-0.057679303/-0.054155363; 4,096-token conditional interval [-0.060244414, -0.051369589]; all three conditional text intervals negative.

All selected peak LRs are identical, so these particular confirmation contrasts do not include different selected LRs. Position families nevertheless change functions, gradients and optimizer trajectories. ALiBi minus NoPE is a separate contrast, not an additional adoption gate or evidence of superiority to RoPE. These results apply to context128; no context-length extrapolation or isolated positional mechanism is established. Conditional text intervals condition on these trained seeds and reused validation text; they are not seed-population intervals or an independent holdout.

NoPE has no explicit positional input, but its causal mask remains order dependent. ALiBi adds fixed negative-distance slopes AFTER the inherited content-attention scale. The fixed head repertoire changes with width. RoPE alone has deterministic frequency buffers; all families share trainable parameter shapes and initialization draws. Legacy rope0 outside the named policy still means learned positions. Actual CPU/MPS prerequisites verified gradients, full/token/uneven/reset/interleaved caches and nonzero offsets. Final checkpoints cannot independently reconstruct every historical bias-placement/cache operation or all101 online averaging snapshots.

The first MPS prerequisite's ideal-FP64 versus declared-FP32 frequency-oracle mismatch remains preserved. The repaired independent oracle reproduced the declared FP32 reciprocal-power and outer-product arithmetic without loosening numerical tolerances. Already-completed first-cell training/scoring was reused byte-for-byte; only unfinished checks and the other five cells ran. The metadata bank-path correction and original incomplete single-entry snapshot remain preserved; registration used the complete canonical bank.

nope training-time changes versus RoPE at widths64/256: -11.95%/-7.08%. These are descriptive training measurements, not inference-speed claims.

alibi training-time changes versus RoPE at widths64/256: +4.39%/-0.06%. These are descriptive training measurements, not inference-speed claims.

All arms use equal3,000 updates and3,072,000 targets, not equal compute. Six fixed-reference cells prospectively alias RoPE trajectories and are not independent repeats. No historical control pooling, holdout access, training/scoring rerun or alternative promotion occurred.
