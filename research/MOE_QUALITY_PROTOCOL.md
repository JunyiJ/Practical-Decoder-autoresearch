# MoE quality round 001

The human renewed the continuing model-quality objective on October 25, explicitly
including MoE and other algorithm evolution until interruption. This supersedes
the old MoE deferral and time expiry. This finite round is one research step;
completion triggers review and the next justified protocol, not routine idling.
All previous panels and the human-stopped campaign006 stay closed. Work remains
in the isolated research checkout, leaving the human `kimi` checkout untouched.

## Question and comparators

Can token-selected expert capacity improve final validation CE over the retained
four-layer MHA SwiGLU recipe? Independently tune three families:

| Family | FFN hidden size | Experts | Active FFN weights/token |
|---|---:|---:|---:|
| dense | 21d/8 | 1 | 3d(21d/8) |
| dense_wide | 21d/4 | 1 | 3d(21d/4) |
| moe4_top2 | 21d/16 each | 4, top2 | 2×3d(21d/16) |

MoE active expert weights equal dense FFN weights, while all four experts equal
wide dense FFN weights. MoE additionally has 4d router weights per layer, executed
for every token. These counts do not imply equal FLOPs, wall time, or memory.
The wide dense comparator distinguishes extra stored capacity from sparse use;
these recipe contrasts do not isolate routing causally. Both alternatives can
improve quality and are evaluated without assuming MoE must win.

Keep MHA head32, four layers, unscaled residuals, RMSNorm, final LayerNorm, RoPE,
zero dropout, context128, B8, paired_blocks512, 3000 updates, 3072000 targets,
Muon/AdamW, base width64, independent decay .00006, global clip1. Fresh original
GELU2L LR.002, shallow SwiGLU2L LR.0025 and deep SwiGLU4L LR.0025 references stay
visible. No historical pooling. Data/splits/character vocabulary stay unchanged;
held-out test stays closed.

## Routing, initialization and optimizer contract

Top2 softmax weights are renormalized over the selected pair, weighted expert
outputs are summed, no shared expert, capacity cap, dropped token, routing noise,
extra output multiplier or z-loss. Bias-free SwiGLU experts use ordinary fan-in
initialization. No residual-depth rescaling. Weighted combinations may change
activation variance; shape matching is not activation equivalence.

The router is a fixed-four-output projection. Its initial standard deviation is
the default fan-in value multiplied by sqrt(d/64), forward logits are divided by
d/64, and it uses AdamW at the base LR. Expert and attention matrices retain
Muon RMS matching with LR factor (d/64)^(-1/2). Embeddings/readout/vectors retain
the prior AdamW policy. This explicit output-like router convention is an
engineering hypothesis, not an established optimal MoE muP transfer law. The
legacy/default MoE path is unchanged. Strict policy guards reject unreviewed
expert counts, top-k, manual hidden dimensions, depth or attention overrides.

Training objective is CE + .01 times the mean layer balancing loss. For each
layer the loss is E sum_i f_i p_i, with f_i the fraction of top-k assignments and
p_i mean full-softmax probability. The assignment term is nondifferentiable;
probabilities carry the balancing gradient. The coefficient and routing design
are fixed, not chosen from this round's quality. The frozen top2 choice permits
task gradients through relative router weights, unlike normalized top1.

Independent decay applies once per update to parameters with gradients, matching
the existing optimizer. An entirely unselected expert has no gradient and does
not decay/update that step. This is explicitly not matched expert exposure or
matched per-parameter cumulative shrinkage. Report sampled per-layer assignment
fractions and auxiliary loss; those snapshots are not whole-run load statistics.
Primary scoring recomputes CE from logits without the auxiliary term.

Design context: [Mixtral](https://arxiv.org/abs/2401.04088) uses token-selected
top2 experts; [Switch Transformers](https://www.jmlr.org/papers/v23/21-0998.html)
describes auxiliary load balancing. Their reported results do not establish
quality, speed, or scaling for our four-expert character model on MPS.

## Frozen evidence and selection

Exactly18 width64 full-budget screens: 3 families × LRs .00125/.0025/.005 ×
fresh seeds2081/2082. Independently select each family's LR by mean final full
validation CE, tie to lower LR. Freeze all three choices and the confirmation
manifest before any confirmation or width256 result. No pruning or extension.

The screen architecture winner must improve mean CE by more than .005 and every
screen seed versus both selected dense and fixed dense LR.0025. If multiple
alternatives qualify, choose lowest screen mean, tie by family name. Otherwise
retain the historical incumbent. Larger-width results never revise this choice.

Confirm all three selected recipes at widths64/256 with fresh2091/2092/2093:
18 trials, plus up to18 fresh fixed references. If dense selects .0025, the deep
reference aliases those six exact cells prospectively. Total48 or54 distinct
trials. Confirmation reports all candidate-versus-dense and fixed-reference
contrasts, including MoE versus wide dense. Qualify a new overall quality recipe
only if the screen-selected alternative improves mean by more than .005 and
every seed at both widths against tuned dense and fixed deep reference. Otherwise
retain the historical incumbent and record narrower evidence without relabeling
the frozen winner. All reports preserve the old original GELU reference.

Final primary evaluation is disjoint full validation context128, batch8, all
111538 target positions exactly once. Raw2048-token blocks support paired
conditional text bootstrap at2048/4096/8192,2000 draws,seed20261024. Report
fixed-LR.0025 screen contrasts separately from independently tuned confirmation.
The +.01 mean/every-seed/all-upper-bound descriptive tolerance is supplementary;
quality improvement is the objective. Intervals are conditional on trained seeds
and reused validation, not seed-population uncertainty, holdout or adjusted
significance. Report training time, RSS, current MPS allocation and parameter
counts separately; current MPS allocation is not peak memory or inference cost.

## Gates, resource limits and completion

Before launch: reviewed source, full project tests; sparse/dense-oracle forward
and gradient equivalence, task/router gradients, exact parameter accounting,
optimizer groups/scales, CPU checkpoint continuation and causality; actual serial
MPS train/save/reload/full-validation pilots at both widths for all three families;
finite checkpoint/optimizer states, sampler/resume contracts and seed freshness.
Pilot loss and timing are correctness/resource evidence only, excluded from study
quality. Fresh configs bind all changes; no historical checkpoint is modified.

One worker; at most54 trials,1800seconds/trial,28800seconds active round and4096MiB
run disk. A worker watchdog samples process-lifetime RSS and MPS driver allocated
memory every0.5seconds and exits at4096MiB RSS or6144MiB driver allocation; failure
to read memory also stops the worker. This does not guarantee instantaneous or
total-system peak memory. First failure stops for diagnosis; no automatic retry. Freeze a full
knowledge snapshot, source/corpus/runtime hashes, prerequisite audit and plan in
a unique run after committing/pushing clean source. Verify all raw requests,
results, checkpoints, optimizer/router/expert contracts, full coverage, frozen
choices/aliases, independent aggregates/bootstrap and source lineage at completion.
Append findings, update best configuration and guide/HTML, commit/push findings,
then prepare the next evidence-driven round under the continuing human objective.
