# Clipping counts do not identify an optimizer improvement

Retain clipping1.0. The completed depth/context logs show a real difference in
clipping exposure, but they do not show that clipping harms depth4 or that another
threshold improves quality. No optimizer diagnostic or training panel is admitted.

## Measured local facts

The review reads all384968 saved updates in the completed12-run panel. It does
not execute a model, optimizer, checkpoint replay or new score. The final-third
window is defined by elapsed training time before each update,9600–14400 seconds,
with final-update overshoot retained. Entries below list bundles1/2/3.

| Depth/context | Clipped fraction | Mean coefficient | Coefficient standard deviation |
| --- | --- | --- | --- |
| 4/128 | .7025/.7401/.7428 | .8980/.8831/.8823 | .1008/.1069/.1067 |
| 4/256 | .7858/.7637/.7920 | .8743/.8849/.8765 | .1061/.1010/.1040 |
| 16/128 | .0160/.0136/.0134 | .9986/.9985/.9987 | .0150/.0174/.0157 |
| 16/256 | .0181/.0136/.0151 | .9983/.9987/.9986 | .0179/.0158/.0156 |

Thus the previously reported100% depth4 clipping in the **final120 seconds**
should not be described as100% throughout the final third. Those final120-second
counts and fractions match the preexisting completion summary exactly for all12
runs. The new summaries retain all three time thirds, the final120 seconds,
quantiles, minima and adjacent coefficient changes. Coefficients vary over time;
a constant-rescaling argument alone is insufficient for these histories.

## Qualified source semantics

These five files match their hashes in the scientific plan:
`general_rsi_equal_time_training_v1.py`, `general_rsi_depth_context_training_v1.py`,
`general_rsi_depth_context_v1/optimizer.py`, `gradient_clip_model_v1/clipping.py`,
and the pinned PyTorch `torch/optim/_muon.py`.

- Global clipping acts after backward and before both optimizer children.
  It uses c_t=min(1,1/(||g_t||_2+1e-6)). Across every retained update, the
  independently recomputed scalar differs by at most7.5883e-8, below1e-6.
- Body matrices use Muon with momentum.95, Nesterov enabled, five Newton–Schulz
  iterations and `match_rms_adamw`. Its base LR is half the shared scheduled LR.
  In the actual implementation, B_t=.95 B_(t-1)+.05 c_t g_t, then the input to
  normalization is U_t=.05 c_t g_t+.95 B_t. Matrix-size LR adjustment follows.
- Muon converts U_t to BF16, divides by its norm clamped at1e-7, then applies
  the fixed finite polynomial. For a fixed gradient sequence with an identical
  constant positive rescaling across its entire history, normalization cancels
  that scale in exact arithmetic above the clamp. BF16 rounding and the clamp
  limit that ideal statement. This is not a proof of identical trained trajectories.
- If c_t changes over time, earlier gradients receive different relative weights
  in B_t. The resulting matrix need not be a scalar multiple of the unclipped
  matrix, so normalization does not generally remove clipping's effect.
- Embeddings/output tables and non-Muon parameters use AdamW with beta1=.9,
  beta2=.999 and epsilon1e-6. A constant history-wide scale cancels in the ideal
  zero-epsilon moment ratio. Time-varying scaling affects first and second moments
  differently; nonzero epsilon also breaks exact cancellation.
- Both children have zero built-in decay. The wrapper separately multiplies all
  parameters by1−.00006 before their updates. This shrinkage is not multiplied by
  the clipping coefficient and does not become a smaller shrinkage when clipping
  occurs. It is fixed per update, so equal-time arms can accumulate different decay.

The scalar logs do not retain full pre-clip gradient directions, per-parameter
moment histories, update displacements or counterfactual unclipped updates. They
cannot determine which parameter family causes the global norm, whether update
magnitude shrinks proportionally, or whether a changed threshold helps development
loss. No missing transient optimizer payload is recreated for this review.

## Primary literature and contrary local evidence

- Kingma and Ba, [Adam1412.6980v9](https://arxiv.org/pdf/1412.6980v9), Algorithm1
  and §2.1, describe moment estimates and the rescaling cancellation under the
  zero-epsilon argument. Applying a different c_t at every step is outside that
  simple cancellation. Our child uses epsilon1e-6 and separate manual shrinkage;
  this is not a new validation of Adam's convergence claims.
- Liu et al., [Muon2502.16982v1](https://arxiv.org/html/2502.16982v1), §2.1
  equations1–2 and §2.2, normalize momentum before finite matrix iterations and
  discuss update scaling. They report no consistent momentum-tuning gain and no
  quality benefit from ten versus five iterations. Their large-model experiments
  do not establish an optimal clipping threshold for our small Mac runs. The
  pinned implementation's Nesterov/EMA convention, not a substituted paper
  equation, defines the actual local update.
- Original `gradient-clip-preparation-001/quality002-final-review.json` and
  `quality002-closure.json` record a valid earlier clip0.5 negative: all three
  deltas are positive, mean+0.004133149592. This older corpus/horizon/LR is not a
  direct general-corpus depth comparison, but it contradicts assuming that
  stronger clipping helps and does not authorize reopening that closed panel.
- Original `general-rsi-update-response-001/analysis.json` and accepted review
  record that all tested smaller first-update displacements still increased mean
  disjoint training-probe loss. This limits a generic smaller-step remedy, without
  proving that every future clipping intervention fails.
- The saved composer momentum precision diagnostic and its original result are
  arithmetic evidence only. The distinct composer FP32 fixture later failed its
  frozen direction gate. Neither provides permission to replay closed precision
  work or replace the qualified native optimizer.

## Boundary decision and complete cost

Source and log review was the selected bounded action. It tests whether the
constant-rescaling simplification applies to recorded c_t and whether clipping
counts identify a quality mechanism. Measured c_t varies; the needed vector/moment
and counterfactual-quality evidence is absent. This is insufficient identification,
not a negative quality result for an unrun threshold.

Rank a new clipping sweep below hypotheses with a measured causal bottleneck.
No threshold, LR, momentum, precision or schedule variant is selected. A future
intervention would need a distinct rationale, bounded discriminating controls,
and a full fresh-confirmation/resource plan before any training. Carry unspent
exploration budget forward; do not manufacture another short pilot or rerun a
closed fixture from the clipping-frequency observation.

The analysis used1.25 CPU seconds and43876352-byte peak RSS. It acquired the shared
lock after process inspection found no training worker. Free space remained above
12GiB reserve plus16MiB output/publication allowance. No MPS allocation, weights,
corpus/test access, training or generation occurred. Original logs, endpoints,
first outputs, failures and source bindings remain unchanged. Evidence is in
`tokenizer-evidence/general-rsi-depth-context-clipping-review-001/`.

The two-depth study and its bounded position/clipping interpretation are complete.
No scientific worker or admitted next campaign exists. Publication awaits the
already requested explicit authorization required by automatic approval review.
Do not repeat unchanged push attempts, capacity scans or completion checks.
Reopen selection on new human direction or a concrete new evidence-backed
hypothesis, using original local results and primary papers. Keep the original
test closed and all selected/fresh endpoints preserved.
