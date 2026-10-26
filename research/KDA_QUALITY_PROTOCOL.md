# Prospective KDA quality comparison

Draft implementation and protocol only. No research run is registered. Execute
only after decay-quality-001 is completed, independently audited and reconciled,
the current human merge is safely integrated in the isolated research checkout,
and all correctness and resource gates below pass. Do not alter the active decay
source or primary human checkout. A failed feasibility gate is evidence to record,
not permission to expand limits or to launch an incomplete quality comparison.

## Question and evidence boundary

Compare four layers of the human's sequential KDA reference, with the explicit
alpha-width adaptation below, against four layers of fresh MHA. The completed
MoE, horizon, QK normalization, attention-output-gate and FFN-normalization rounds
remain closed. Their negative adoption decisions do not establish that all other
algorithms fail. Successful cooldown and H200 choices remain common. Resolve
learning rate and independent decay only from the final decay review, through
`reviewed_decay`; no partial result, training log or larger-width observation
may decide the next common recipe.

The cited report combines KDA and global attention and employs specialized
kernels. This all-KDA, dense-SwiGLU, short-context experiment is a local adaptation,
not a reproduction of that complete system. [Primary report](https://arxiv.org/html/2607.24653v2).

## Explicit model and optimizer contract

Keep four layers, head dimension32, bias-free SwiGLU hidden21D/8, RMSNorm blocks,
final LayerNorm, zero dropout, unscaled residuals, context128 and batch8. MHA uses
RoPE. KDA uses its causal convolution and recurrent state, with no positional
embedding or RoPE; replacing attention therefore changes several mechanisms
jointly. Do not attribute a contrast solely to the recurrence equation.

Use `attention_width_policy=kda_head32_alpha_quarter` and
`depth_width_policy=depth4_kda_unscaled`. Derive the alpha projection bottleneck
as D/4 after each target width is applied:16/32/64 at64/128/256. Reject manual
`dim_alpha` under this policy. The human default, head_dim/4, remains unchanged
outside this named policy. The derived policy avoids carrying a base-width
literal into larger model requests. Saved configs and checkpoints must bind the
policy and verify both projection shapes independently.

Preserve the human reference equations, four-tap depthwise convolutions, default
linear/conv initialization, per-head zero-initialized A, FP32 recurrent accumulation,
L2-normalized Q/K, headwise RMS gains, sigmoid output gate and output projection.
The experiment does not add a hybrid layer pattern, tune gate initialization,
change convolution length or implement per-head Muon. No paper-exact bias-init
or optimized-kernel claim is made.

All two-dimensional block matrices have both axes proportional to D under the
named policy. Use existing Muon RMS matching with LR factor1/sqrt(D/64). Depthwise
Conv1d filters[D,1,4], A[H], shared RMS gains[32], and biases use AdamW at base LR.
Embedding and readout roles remain unchanged; the readout forward factor is1/(D/64).
Independent decay applies once before inner updates to every parameter with a
gradient, with zero inner decay. Its coefficient comes from the final decay study
and remains constant through cooldown. These are prospective scaling conventions,
not empirical optimal-LR transfer guarantees.

MHA and KDA constructors consume different random draws. Paired seeds give matched
training sequences but do not guarantee identical common initial tensors. Both
arms use fresh same-source controls; do not pool historical scores. The alpha
width adaptation is a declared local design choice, not an invisible harness fix.

## Finite selection and confirmation

Proposed unused screening seeds2261/2262, at width64 only. Both families use the
same three peak LRs[0.00125,0.0025,0.005], giving12 full-budget screen trajectories.
Choose each family's LR by mean final H200 full-validation CE; exact ties use the
lower LR. Choose the best selected family, ties MHA before KDA. Require mean gain
strictly greater than0.005 CE and every paired seed better against the fixed
incumbent. KDA must also meet those conditions against equally tuned MHA. If the
best family fails, freeze retention; no alternate fallback.

Freeze both LR choices, the eligible winner and the entire manifest before any
confirmation. Fresh seeds2271/2272/2273 at widths64/256 give12 selected trajectories.
Add six fixed-incumbent MHA references unless the selected MHA exactly matches the
resolved incumbent LR/config, in which case prospectively alias its six runs.
Total24 or30 physical trajectories;48 or60 correlated raw/H200 outputs. No pruning,
grid extension, larger-width reselection or repeat training to recover interpretation.

Only the frozen eligible family may replace the incumbent. Require mean gain>0.005,
every fresh-seed delta negative, and all three conditional text-bootstrap upper
bounds below zero at both widths against the fixed reference; KDA also requires
that support against tuned MHA. Report all KDA-minus-MHA, selected-minus-fixed and
fixed-LR screening contrasts. Raw final outputs are correlated diagnostics and
cannot replace the primary H200 selection metric. These operational criteria do
not imply seed-population significance, independent holdout or a universal optimum.

## Shared training and evaluation

3,000 updates,1,024 targets/update,3,072,000 targets total. Use paired_blocks512,
Muon/AdamW muP base64, clipping1 and the resolved independent decay. Cooldown starts
at2,000 and ends at10% of peak. H200 snapshots every10 completed updates from2,000
through3,000 produce101 snapshots. The online raw checkpoint remains resumable;
H200 is an inference-only derived checkpoint. No shadow feeds back into training.

Evaluate every one of111,538 validation targets once in disjoint context128
windows, batch8. Preserve raw2048-token loss blocks. Conditional paired bootstrap:
block sizes2048/4096/8192,2,000 draws, seed20261024. Keep all training logs, but never
rank using their sampled losses. The held-out test stays closed. Different
parameter counts and operations mean this is equal tokens, not equal compute.
Cache correctness pilots do not establish inference speed or peak-memory savings.

## Hard feasibility and execution gates

Limits:30 physical trials,2,400 seconds per worker,21,600 seconds per round,
4,096 MiB artifacts and sampled process RSS,6,144 MiB current MPS driver allocation.
Current allocation is not peak memory. One serial worker; first failure stops;
no automatic retry, paid compute or cap expansion.

Before registration, require independent double-precision recurrence/causal-conv
forward and input/parameter gradients; exact geometry/parameter counts and every
optimizer role/width factor; CPU actual training, optimizer state, exact resume,
private sampler/test RNG, strict saved weights, H200/raw noninterference and changed
policy rejection; unchanged default-KDA tests and the full project CPU suite.

Then run four serial MPS correctness pilots: MHA/KDA x widths64/256, seed9363,
20 updates, cooldown start10, H200 snapshots10/20. Full raw/H200 validation coverage,
strict CPU finite loads and optimizer states, and full/cached-logit agreement must
pass. Cached logits use the existing atol1e-4/rtol2e-5 tolerance; no after-failure
relaxation. Limit each pilot to240 seconds and the pilot round to1,800 seconds and
1,024 MiB artifacts. Scores/timings are correctness/resource evidence only.

Forecast the maximum30-trajectory schedule from measured pilot training per step,
retain measured two-output evaluation overhead, and add50% time/artifact margin.
Every trajectory forecast must fit2,400 seconds, the maximum schedule six hours,
and artifacts4,096 MiB before launch. The sequential recurrence may fail these
limits. If so, preserve the failure and consider another algorithm or a separately
reviewed implementation improvement; do not silently shorten the quality budget.

Freeze fresh seed verification, complete knowledge snapshot, final predecessor,
source/runtime/corpus hashes, prerequisite evidence and finite plan only after all
gates pass on clean committed/pushed source. The active decay run remains separate.

At completion independently audit every raw/derived checkpoint, request/result,
actual QKV/alpha/conv/gate/head/depth/FFN shapes and counts, optimizer slots/steps/
width factors/decay, private RNG, snapshot/token counters, full evaluation coverage,
frozen LR choices and aliases, every aggregate/bootstrap and final adoption.
Update the knowledge bank, best-config context and readable learning guide, then
continue the human quality objective through a distinct justified next experiment.
Permanent stop006 and all other completed studies remain closed.
