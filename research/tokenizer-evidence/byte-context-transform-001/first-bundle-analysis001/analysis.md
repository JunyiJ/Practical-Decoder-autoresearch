# First-bundle learning dynamics: descriptive evidence only

This analysis uses the three completed paired trajectories, not the unfinished
second and third bundles. `result.json` binds every local input and the pure
Python analysis source. No model execution, training, rescoring or selection
change occurred. The frozen H200 endpoint and both controls remain required.

## Measured local observations

| Arm | Mean training CE, updates1101–1200 | Final online validation CE | H200 validation CE | H200 minus final online |
|---|---:|---:|---:|---:|
| byte | 3.241068 | 4.592677* | 4.550480 | −0.042197* |
| linear | 3.196981 | 4.728962 | 4.643989 | −0.084973 |
| tanh | 3.964889 | 4.592330 | 4.616356 | +0.024026 |

*The original byte online curve used the earlier no-grad RMSNorm path. It is
diagnostic only and must not be treated as a corrected RAW comparison. All three
H200 primary scores are the unchanged, accepted MPS scores. Linear and tanh
online curves used the same corrected v2 forward/scoring path as their H200.

Tanh's online validation CE fell by0.131031 between updates900 and1200, whereas
linear's fell by0.026549. Tanh's final100-update training loss remains0.723820 above
byte and0.767908 above linear under the paired sampling specification. Slots00
and02 lack published terminal sampler digests; their pairing has the explicit
source/count limitation recorded in the recovery. Linear fits these
training batches slightly better than byte but has worse validation. Thus adding
a transform does not yield a uniform optimization explanation across arms.

The H200 tanh output RMS measured by the existing numerical diagnostic was
0.922757 on the first validation window and0.914968 on the worst numerical-error
window. From `mean(tanh'(u)) = 1 - RMS(tanh(u))^2`, the corresponding mean local
derivatives are approximately0.148520 and0.162834. This summarizes elementwise
attenuation at those saved weights. These two windows were not a representative
training sample. Surrounding matrices can rescale gradients, and this is not a
measurement of the full model Jacobian or the cause of slow fitting.

The three original guarded runs used2,000.078 seconds:1,687.070 seconds (84.35%)
were actual training updates and205.601 seconds (10.28%) were scheduled online
validation. The remainder includes checkpoint work, final numerical scoring and
other worker overhead. Separate diagnostic/recovery work is not included in
those percentages. This is execution cost, not total engineering wall time.

## External evidence and its limits

[Glorot and Bengio, AISTATS2010 proceedings](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf),
§3.2, §4.2 Eqs12–16, §4.3 and Table1, studies tanh saturation and gradient
propagation. Its initial uniform range matches our matrix's `1/sqrt(fan_in)`
rule. Their normalized initialization increases weight variance relative to that
rule; simply shrinking our initial matrix would not implement their proposal.
They also caution that initialization-based independence and linearity
assumptions cease to describe learning, and observe changing gradient patterns
after training begins. Their experiments use deep feedforward image networks,
SGD, separately chosen learning rates and much longer optimization. They supply
a reason to measure activation/gradient dynamics, not evidence that changing our
initialization or activation will improve this Transformer.

[Izmailov et al., arXiv1803.05407v3, 25February2019](https://arxiv.org/html/1803.05407v3),
§§2–3.2 and §4, distinguishes equal-weight SWA with constant/cyclic SGD from EMA
with a decaying rate. SWA begins from a pretrained point and averages proposals
exploring a high-performing region. Its image-network results do not establish
that our short, fixed H200 tail under Muon/AdamW helps every architecture. This
limits a claim about averaging; it does not prove that EMA caused our loss.

## Inference and next decision

The first bundle is consistent with a candidate that is still improving late,
so its fixed tail average includes less-developed weights. Saturation may be
involved, but delayed fitting, bounded output amplitude, optimizer interaction
and representation changes remain competing explanations. Even the final online
tanh score does not establish a baseline win, and RAW is not an allowed fallback.

Do not launch an initialization, activation or EMA sweep from this one bundle.
First complete the frozen panel and compare these same descriptive quantities
across its remaining paired bundles. If the pattern repeats, a subsequent
bounded analysis should distinguish amplitude/derivative and optimization effects
before a genuinely new hypothesis is admitted. No closed study is reopened and
no additional training budget is selected here.
