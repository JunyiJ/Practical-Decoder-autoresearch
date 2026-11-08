# Weight dynamics: no uniform larger-movement explanation

All nine completed RAW/H200 pairs were analyzed without model construction, forward passes, training or quality rescoring. The original scientific result stays negative and closed. The guarded computation took0.985seconds and peaked at720.13MiB RSS. All original weight/input hashes and CPU RNG remained unchanged.

## Relative separation

Each entry is the mean across three seeds of `||H200−RAW|| / ||RAW||`, reported as a percentage. These are descriptive parameter-space measurements, not confidence bounds or measures of functional importance.

| Parameter group | Byte | Linear | Tanh |
|---|---:|---:|---:|
| output | 4.129% | 4.240% | 3.772% |
| composer | 7.322% | 7.414% | 7.755% |
| transformer | 9.558% | 9.758% | 9.596% |
| context | — | 8.757% | 8.531% |

Tanh has smaller relative output-table movement than both controls on all three seeds. Its added transform also has smaller total relative movement than the linear transform on every seed. Transformer movement is close to byte and below linear; the composer moves somewhat more. This contradicts a uniform increase in normalized parameter displacement as the explanation for tanh’s worse averaging outcome. It does not rule out late nonstationarity or a functionally important direction of movement.

## Scale and direction still differ

| Arm | RAW output-table RMS | Output displacement RMS | H200/RAW output norm | Transform orthogonal relative displacement |
|---|---:|---:|---:|---:|
| byte | 0.141593 | 0.005845 | 0.994328 | — |
| linear | 0.131057 | 0.005555 | 1.000241 | 6.324% |
| tanh | 0.169959 | 0.006411 | 0.988402 | 7.961% |

Tanh’s final output table has about20% larger RMS than byte and30% larger RMS than linear. Its absolute output displacement is consequently larger despite its smaller relative displacement. Averaging reduces its output norm by about1.16%, compared with0.57% for byte and essentially no change for linear. The tanh transform’s orthogonal component is larger than linear’s even though its total relative displacement is smaller; linear has more radial norm change. None of these weight-space observations identifies the corresponding change in predictions or loss.

The output table contributes roughly78–84% of squared displacement across all arms, but also contains most parameters. This is not evidence that it causes that proportion of the quality gap. Parameter norms depend on the representation and cannot measure curvature, gradient effectiveness, or saturation by themselves.

## Clipping strength

Every run clipped all final100 updates. The average pre-clip norms and actual multipliers distinguish the arms:

| Arm | Mean pre-clip norm | Mean clip multiplier |
|---|---:|---:|
| byte | 2.158364 | 0.467621 |
| linear | 2.357197 | 0.427845 |
| tanh | 1.339394 | 0.756506 |

Tanh’s global gradients were smaller and scaled down less strongly. A simple stronger-clipping explanation for its slower fitting is unsupported. Global norms and multipliers do not reveal per-group gradient directions or the effective Muon/AdamW updates, so optimizer interactions remain unresolved.

## What remains worth measuring

The larger output-table scale, bounded tanh features and previously observed local derivative attenuation motivate one new bounded representation diagnostic: compare RAW/H200 feature RMS, saturation and the analytic local transform Jacobian across all three arms/seeds on fixed training-text windows. This would test whether the earlier two-window derivative observation generalizes and whether readout growth accompanies reduced feature amplitude or sensitivity. It would use no quality labels or validation scoring and would not alter endpoints, weights or the closed comparisons.

A uniformly larger-motion explanation is weakened; amplitude, orientation and optimization effects remain distinct alternatives. Do not launch an activation, initialization or averaging sweep from these results. No new training campaign or adoption is admitted here. This is analysis of existing evidence, not a numerical exploration allocation.

[Izmailov et al., arXiv1803.05407v3](https://arxiv.org/html/1803.05407v3), §§2 and3.1–3.2, studies averaging near pretrained high-performing regions under constant/cyclic SGD and distinguishes it from decaying-rate EMA. Its image-network setting supplies no local guarantee for this short Transformer recipe. Two saved endpoints cannot reconstruct a trajectory or demonstrate the paper’s geometric mechanism.

## Evidence and limitations

- [Prospective protocol](../../../BYTE_CONTEXT_WEIGHT_DYNAMICS_PROTOCOL.md)
- [All parameter rows and group results](result.json)
- [Frozen plan](plan.json)
- [Acceptance](acceptance.json)
- [Completed scientific panel](../../../BYTE_CONTEXT_TRANSFORM_RESULTS.md)

The first byte online curve retains its earlier normalization caveat; no corrected RAW quality score was manufactured. Missing original terminal sampler digests for slots00/02 remain explicitly source/count-derived. No seed-population inference, holdout result or causal claim is made.
