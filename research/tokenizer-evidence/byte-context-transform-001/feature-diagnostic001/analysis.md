# Feature diagnostic: bounded amplitude and local attenuation recur

The fixed training-window diagnostic passed for all nine runs, both RAW/H200 endpoints and both windows:36 cases total. It took8.746seconds. No labels, vocabulary logits, losses, validation/holdout scores, backward passes or training were computed. All loaded weights, model semantics, global RNG and private sampler states remain unchanged.

## Measurements

The table averages three seeds and two windows for each arm/endpoint. These are small, fixed-input descriptive measurements; the six observations are not independent data samples.

| Arm | Endpoint | Feature RMS before map | Feature RMS after map | Mean elementwise derivative | Local map RMS gain | Saturated fraction |
|---|---|---:|---:|---:|---:|---:|
| byte | raw | 1.981600 | 1.981600 | 1.000000 | 1.000000 | — |
| byte | h200 | 1.895795 | 1.895795 | 1.000000 | 1.000000 | — |
| linear | raw | 1.702228 | 2.544373 | 1.000000 | 1.136655 | — |
| linear | h200 | 1.630362 | 2.344256 | 1.000000 | 1.070219 | — |
| tanh | raw | 1.736171 | 0.909203 | 0.173294 | 0.312447 | 66.26% |
| tanh | h200 | 1.711272 | 0.914458 | 0.163704 | 0.292198 | 67.88% |

Saturation is the predeclared fraction of tanh features with absolute value at least.95. Local gain is sqrt(mean_tokens ||J||²_F/256), with J=diag(1−tanh(u)²)A; it includes the learned A scale. It measures the map’s local response to isotropic infinitesimal input variation, not the full-model gradient or every singular value.

Tanh feature RMS and local map gain are lower than both controls in all12 matched seed/endpoint/window comparisons. Its mean elementwise derivative ranges.1585–.1875 for RAW and.1484–.1794 for H200; local map gain ranges.2959–.3288 and.2735–.3081 respectively. About64–71% of features exceed the saturation threshold across these cases. Thus the earlier attenuation observation recurs on the two prospectively fixed training windows, across all three seeds and both endpoints.

## Interpretation and contrary evidence

The nonlinear layer replaces byte features whose RMS is about1.90–1.98 with bounded features near.91. Its learned matrix alone has gain about.93–.96, but applying tanh reduces the local map gain to about.29–.31. The previously measured larger output-table norm accompanies smaller feature amplitude and sensitivity. This supports investigating a representation/optimization constraint, rather than uniformly larger weight displacement or stronger clipping. It does not establish that the constraint caused the validation loss.

Averaging does not simply shrink the tanh features: their RMS rises from.9092 to.9145 and their saturation fraction rises from66.26% to67.88%. Local map gain falls by roughly6.5%, while linear-map gain falls by roughly5.8% even though averaging helps linear. Those facts prevent a simple “averaging reduces amplitude, therefore loss rises” account. Orientation, feature distribution and optimizer interactions remain unresolved.

[Glorot and Bengio,AISTATS2010](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf), §§3.2/4.3, motivates examining saturation and evolving sensitivities. Figure8 cautions that smaller propagated gradients do not necessarily imply smaller weight gradients. Their deep image networks and SGD differ from this single added map and Muon/AdamW. These measurements are not evidence that their initialization change would improve our model.

The limited first256 training tokens are not representative of all training contexts or unseen text. Two endpoint snapshots do not reveal when saturation developed. None of the scalar sensitivities determines useful gradient direction or the sign of a quality change. No endpoint was changed, no historical score was replaced, and no model is adopted.

## Next research decision

Further generic diagnostics of this failed recipe now have diminishing value. A concrete candidate worth checking against historical restrictions and primary literature is an added nonlinear residual readout path that preserves the baseline feature and starts with a zero gate. At initialization its feature and input Jacobian would equal the baseline, avoiding the demonstrated replacement bottleneck. This is an inference from local evidence, not an admitted experiment or a claim of improvement. A matching linear residual control, fresh roles, fixed criteria and complete cost would be required for a new study. Do not replay the closed plain-transform panel or launch an arbitrary activation/initialization sweep.

## Execution note

PyTorch emitted a warning when a read-only output-weight RMS was converted to a Python scalar from a requires-grad parameter. The statistic was not used in any objective, no backward was called, and unchanged weights/absent gradients were verified. The run completed successfully; no retry or numerical correction is warranted. Its source remains frozen. Future diagnostic helpers should detach parameter statistics before scalar conversion.

## Evidence

- [Prospective protocol](../../../BYTE_CONTEXT_FEATURE_DIAGNOSTIC_PROTOCOL.md)
- [Frozen plan](plan.json)
- [All36 cases](result.json)
- [Acceptance](acceptance.json)
- [Prior weight dynamics](../weight-dynamics001/analysis.md)
