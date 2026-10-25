# Stage 2: architecture changes against the stage-1 control

The human concluded stage 1 on October 24, 2026. Campaign006 is intentionally
paused after its last training suite and must not resume. Its final numeric
results were reviewed separately without clearing the stop marker or rerunning
training. Historical campaigns, budgets, failed markers and evidence remain intact.

## Frozen control and stage-1 conclusion

The strongest observed arm is the **Muon/AdamW hybrid with muP**, base LR .002,
independent per-step decay .00006, global clipping 1, zero dropout, two-layer
RMSNorm dense GELU MHA, RoPE, final LayerNorm, base width64 and head dimension32.
The hidden FFN width is four times model width. The optimizer uses Muon on hidden
matrices and AdamW on embeddings, readout and vectors, retaining the exact
stage-1 grouping, initialization, bias, decay and width rules.

The controlled optimizer study measured mean validation CE
**1.646054 / 1.559346 / 1.523978** at widths64/128/256, all paired seeds better
than its AdamW reference. Incremental muP benefit over SP Muon was mixed by seed.
The bounded Adam follow-up did not reverse the historical Muon ranking.

The two optimizer studies contain 138 completed runs. Subsequent campaigns005
and006 contain 42 complete 18-trial comparisons (756 runs). None displaced the
fixed Muon control under the frozen local rule. These counts cover the optimizer
studies and subsequent Muon exploration, not all earlier project experiments.
The last decay .00005 comparison changed mean CE by +.000185 / +.001021 / +.000098;
it also retained the baseline. All its 18 checkpoints were verified. The strongest
observed arm is an adaptive development choice, not a universal optimum.

Machine-readable closure and exact control: `research_state/stage1-conclusion.json`.
Full shared bank snapshot: `research_state/stage2-knowledge-snapshot.md`.
Original measurements and source contracts remain authoritative; new source does
not rewrite them. Held-out test remains closed.

## Round 2A: isolate the activation before changing FFN structure

Compare GELU and SiLU in the existing two-matrix dense FFN. This changes only the
nonlinearity at a fixed LR: identical parameter names, counts, dimensions, initial
parameter tensors for a paired seed, dropout and optimizer groups. SiLU uses
PyTorch `nn.SiLU`; GELU retains the existing `nn.GELU`. Gated SwiGLU is NOT enabled
under the width contract by this change. Attention, depth and head dimension stay
fixed. Implementation tests establish supported execution, not improved quality
or invariant optimal learning rates.

Three families:

| Family | Activation | Base LR options | Role |
| --- | --- | --- | --- |
| control | GELU | .002 only | Exact stage-1 recipe, trained fresh |
| gelu_tuned | GELU | .0015, .002, .0025 | Equal-budget tuning comparator |
| silu | SiLU | .0015, .002, .0025 | Architecture candidate |

Use fresh **development** seeds44/55/66, fixed validation seed2026, the same
Shakespeare splits and tokenizer, batch8, context128, 3000 updates and 30-batch
validation every300 updates. Each trial sees 3,072,000 training character targets.
These are new training seeds, not an independent dataset or a held-out confirmation.

Run all 21 width64 trials first, interleaving families and reversing order on the
middle seed. Select each tuned family's LR using lowest mean FINAL validation CE;
ties choose lower LR. No intermediate-checkpoint selection or grid extension.
A boundary choice is reported as unresolved, not an optimum. The separate fixed
control is intentionally trained fresh even when tuned GELU selects .002; these
are contemporaneous repeated controls, not exact-reproducibility evidence.

Before any larger-width results, freeze the selected LRs and architecture winner
in read-only `selection.json` with plan and local-evidence hashes. SiLU wins only
if its mean paired CE delta is below -.005 and every seed improves against BOTH
the fixed control and tuned GELU. Otherwise retain the control. This heuristic
is not a significance test. Then run the three families at their frozen LRs at
widths128/256 with the same three seeds (18 trials). All-family transfer results
are diagnostics; they cannot revise local selection. Report every result,
including failures and reversals. Total: **39 trials**, no held-out evaluations.

This round has equal parameters, shapes, tuning allowance for the activation
families, and training-token budgets. It is not equal-time or equal measured
compute: activation kernels can differ. Report CE, training time/throughput and
process RSS; RSS is not peak accelerator or total unified-memory demand.
Quality effects are unquantified in advance. Throughput/RSS are expected to be
close because matrix shapes are unchanged, but no numerical range is asserted.
A possible mechanism is different activation/gradient behavior; this round does
not instrument or causally identify that mechanism.

## Execution and limits

`src.research.architecture_study` reuses the serial optimizer-study worker engine,
trusted CE scorer, source/corpus/runtime checks, checkpoint verification, fixed-token
validation and durable failure markers. The existing optimizer study defaults are
retained. Freeze a plan only from clean committed source using `make_plan`, then:

```sh
unset PYTORCH_ENABLE_MPS_FALLBACK
.venv/bin/python -m src.research.architecture_study \
  --plan research_state/architecture-stage2a-plan.json \
  --output research_runs/architecture-stage2a-001 --allow-training
```

Limits: exactly39 dispatches, 600 seconds per trial, four hours for the round,
4 GiB output, stop on first failed trial; serial training only. `STOP` pauses at
a trial boundary. No automatic retry of failed/running records. Source changes
require a new run with preserved prior evidence. This finite round is within the
ongoing 48-hour-target research effort; it does not add an aggregate exploration
cap or authorize an indefinite LR grid. Autonomous maintenance expires at the
previously authorized time, October25 1:32 PM Pacific; healthy runs are not killed
solely because that window ends.

### Completed stage 2A: activation and learning-rate transfer

All **39 trials** completed and were reviewed against checkpoint hashes, worker results,
frozen selection, source/corpus/runtime contracts and recomputed report aggregates.
Mean final validation CE at 3000 updates (three fresh development seeds44/55/66):

| Arm | Frozen LR | Width64 | Width128 | Width256 |
| --- | --- | --- | --- | --- |
| Fixed GELU control | .002 | 1.659749 | 1.573942 | **1.520457** |
| Equally tuned GELU | .0025 | **1.647783** | **1.571541** | 1.525777 |
| Tuned SiLU | .0025 | 1.659710 | 1.576024 | 1.521624 |

**Keep the exact stage-1 control.** SiLU failed the predeclared width64 architecture
rule: its mean delta was −.000039 against fixed GELU (one seed worse), and +.011928
against tuned GELU (all three worse). At widths128/256 its deltas against fixed
control were +.002082 / +.001168, both mixed by seed. SiLU was better than tuned
GELU on every width256 seed, but the mean −.004153 missed the .005 threshold;
transfer diagnostics cannot change the frozen architecture decision.

The GELU LR change is a useful transfer counterexample: .0025 improved over .002
by .011966 at width64 and .002401 at width128, then worsened by .005320 at width256.
All three paired seeds followed each direction. Width scaling does not guarantee
that a locally preferred LR preserves its advantage at larger widths. This does
not invalidate muP in general or identify the optimal LR at any width.

Mean training seconds for control / tuned GELU / SiLU were 32.23 / 32.48 / 32.35
at width64, 51.26 / 51.00 / 51.84 at width128, and 66.40 / 66.41 / 66.36 at width256.
Process peak RSS means ranged from 472.1 to 475.6 MiB. These small observed time
and RSS differences do not establish a reliable speed advantage; RSS is not peak
accelerator memory. Equal token budgets and shapes are not equal measured compute.

Both tuned families selected the upper grid boundary; no grid extension is planned.
The validation split remains adaptively reused, despite fresh training seeds; test
was never opened. Historical control scores use different seeds and must not be
pooled as independent replication. Correctness tests and short smoke runs remain
separate from these measured quality results.

Evidence: `research_runs/architecture-stage2a-001/{plan,selection,report,review}.json`.
Next is a separately preregistered gated-FFN round, after parameter-budget,
bias/initialization, optimizer-aspect-ratio and execution checks. Stage1 stays stopped.

## Subsequent rounds

After reviewing 2A, test gated FFNs with explicit near-matched parameter budgets
and bias/initialization controls, then attention variants/head dimensions,
depth-width tradeoffs, and MoE. Each needs its own reviewed scaling contract,
resource limits and preregistration. In particular, three-projection SwiGLU needs
a smaller hidden width than a two-projection dense FFN for comparable matrix
parameters; rounding, bias counts and changed matrix aspect ratios must be
reported. MoE needs both total/active parameter counts, routing/load-balance
metrics and measured runtime. Do not enable those families by bypassing the current
guards or claim width-only scaling establishes depth/expert scaling.

## Round 2B preregistration: bias-free dense SiLU versus gated SwiGLU

Round2A is closed. Round2B retains the exact GELU LR.002 control and compares two
new equal-tuning families: bias-free dense SiLU (hidden width4d) and bias-free
SwiGLU (hidden width21d/8). Each receives LR.0015/.002/.0025, fresh development
seeds77/88/99, and the same 3000-update, batch8, context128 validation protocol.
Run21 width64 screens, freeze the lowest-final-mean-CE LR for each family, then
18 diagnostic transfer trials at widths128/256. SwiGLU replaces the control only
if width64 mean delta <−.005 and every pair improves against BOTH fixed GELU and
selected bias-free dense SiLU. Otherwise retain control. No transfer-informed
selection, test access, automatic retry or grid extension. Limits remain39 trials,
600seconds/trial, four hours/round,4GiB outputs, first failure stops; serial MPS.

This comparator controls FFN bias presence and the SiLU nonlinearity, but is a
comparison of two FFN designs, not a pure isolated multiplication experiment.
SwiGLU computes down(SiLU(gate(x))*up(x)); its third matrix and narrower hidden
space are intentional. Both new families omit FFN biases. The original control
retains biases, so a comparison against it combines FFN structure/activation/bias
changes. No new bias removal is applied to attention or normalization.

The dense FFN has8d² matrix weights; SwiGLU has3d(21d/8)=63d²/8, **1.5625% fewer**.
Hidden widths168/336/672 preserve exact aspect ratio21/8 at d64/128/256 without
per-width rounding. Choosing the exact ratio8/3 would require rounding at these
widths; the chosen near match avoids that additional scaling change. Total model
parameter counts (including the unused learned position table) are reported in
`research_state/stage2b-validation/mps-correctness.json`. These are near-matched
parameter budgets and equal tokens, not equal FLOPs or elapsed time.

Initialization remains PyTorch Linear's fan-in initialization for all projections,
with no extra gate multiplier. The existing muP readout and hidden-bias conventions
remain. Added/removed matrices and biases change RNG consumption: paired seeds do
NOT imply identical initial tensors across architectures. No transplant of shared
weights is used. Gate multiplication changes activation statistics; finite-gradient
checks do not prove identical signal propagation or optimal learning-rate transfer.

The explicit `swiglu_21_8` optimizer contract permits only two-layer dense MHA,
base64, widths64/128/256 and head dimension32, without a hidden_dim override.
Every hidden FFN matrix uses Muon; embeddings/readout/vectors retain AdamW.
Within this fixed aspect ratio, hidden LR scales as1/sqrt(d/64) and the Muon
RMS-adjustment factor .2*sqrt(max(rows,cols)) scales as sqrt(d/64), so their product
is width-independent for each matrix type. The base factor differs from dense4x;
that difference is not silently normalized away. Equal bounded LR screens help
compare recipes, but do not prove optimality or architecture-invariant updates.
Independent decay .00006 and clipping1 remain unchanged. No depth/expert rule is
inferred. Unsupported shapes, policies and architectures remain rejected.

Before launch: focused gradient/causality/count/scaling/checkpoint-resume tests,
repository tests, and six short actual MPS train/checkpoint-evaluation checks must
pass. Short checks are correctness evidence only. Freeze a new plan and complete
shared-bank snapshot from clean committed source. Execute through
`python -m src.research.gated_study --plan PLAN --output NEW_RUN --allow-training`.
Expected quality and runtime differences are unquantified; report measured CE,
training time and process RSS, all failed trials and all transfer reversals.

### Completed stage 2B: SwiGLU wins locally, with mixed largest-width transfer

All39 trials completed and were reviewed against final checkpoint hashes, worker
result contracts, frozen local selection, report aggregates, source/corpus/runtime
and the full shared-knowledge snapshot. Mean final validation CE at3000 updates:

| Arm | Frozen LR | Width64 | Width128 | Width256 |
| --- | --- | --- | --- | --- |
| Fixed GELU control | .002 | 1.656103 | 1.575256 | 1.528558 |
| Bias-free dense SiLU | .0025 | 1.658701 | 1.571751 | 1.536284 |
| **Bias-free SwiGLU** | **.0025** | **1.637745** | **1.553588** | **1.523561** |

SwiGLU passed the frozen width64 selection rule against both comparators.
Its mean deltas versus fixed GELU were−.018358 / −.021668 / −.004997245 at
widths64/128/256. Every seed improved at64/128. At256, seed77 worsened by+.001391,
seed88 improved by−.001545, and seed99 improved by−.014838. Thus **width256 does
not pass the support rule**: the unrounded mean also narrowly misses−.005.
Rounding it to−.005 cannot turn this into a success, and the positive seed would
still fail. The local winner stays frozen; transfer does not retroactively alter it.

Against the equally tuned bias-free dense SiLU comparator, SwiGLU improved every
seed at all three widths, with mean deltas−.020955 / −.018163 / −.012722.
That makes SwiGLU the most promising observed architecture in this development
round. It does not establish uniformly supported superiority over the original
GELU control at all widths. The original recipe remains an explicit reference;
its historical best-config record is preserved separately.

SwiGLU mean training time was33.41 / 54.72 / 67.10 seconds, about2.96% / 3.97% /
1.70% longer than the fixed control's32.45 / 52.63 / 65.98 seconds. This is a
fixed-token quality/time tradeoff, not evidence from equal-time training. Peak
process RSS means were471.31 / 473.40 / 475.38 MiB versus471.58 / 472.74 / 474.17.
Process RSS is not peak accelerator allocation. Parameter-count reductions did
not produce a measured training-speed win in this round.

Both tuned arms selected the upper LR grid boundary; no extension of this completed
grid is planned. Fresh seeds77/88/99 remain development seeds on the reused
validation split. No held-out test was consumed. The change combines gating,
hidden ratio and different initialization draws; comparison to fixed GELU also
changes FFN bias and activation. Do not attribute the result to gating alone.
Evidence: `research_runs/architecture-stage2b-001/{plan,selection,report,review}.json`.
The next attention/head round requires its own scaling and resource review and
preregistration. Neither previous stage nor completed grid will be restarted.

## Round 2C preregistration: MHA head dimension on the SwiGLU incumbent

Round2B is closed. Its SwiGLU local winner is the incumbent, with mixed width256
transfer versus the original GELU reference. Round2C changes only the MHA head
partition within the SwiGLU recipe when comparing equal LRs. It does not change
FFN, depth, projection dimensions, parameter count, biases, dropout or optimizer.
Original GELU remains a separate reference, not the new attention comparator.

| Family | FFN | Head dimension | Base LR |
| --- | --- | --- | --- |
| control | Original GELU | 32 | .002 fixed |
| incumbent | Bias-free SwiGLU21/8 | 32 | .0025 fixed |
| head32_tuned | Bias-free SwiGLU21/8 | 32 | .0015/.002/.0025 |
| head16 | Bias-free SwiGLU21/8 | 16 | .0015/.002/.0025 |

Fresh development seeds111/222/333; validation seed2026; same corpus/splits,
3000updates, batch8, context128, eval30 batches/every300; no held-out test.
There are24 width64 trials (two fixed references plus two three-LR screens).
Select each tuned LR by lowest final mean validation CE, ties lower LR, then
freeze before24 transfer trials at widths128/256. Head16 replaces the incumbent
only if mean paired CE delta <−.005 and all seeds improve versus EACH of fixed
GELU, fixed SwiGLU and tuned head32. Otherwise retain the prior incumbent without
claiming it passed a new confirmation test. Report all contrasts and reversals.
The separately repeated incumbent and tuned-head32 runs at.0025 are deliberate;
small numeric differences are not exact-reproducibility evidence.

Exactly48 serial trials,600seconds/trial,four hours/round,4GiB outputs; first failure
stops, STOP at trial boundary, no retry or grid expansion. These are finite rounds
within the ongoing research effort, not an aggregate exploration cap.

At widths64/128/256, head32 gives2/4/8 heads; head16 gives4/8/16. Packed QKV and
output projections have identical shapes and initialized parameter tensors for
paired seeds across SwiGLU arms. Optimizer groups, fan-in initialization, readout
scaling and independent decay remain identical. RoPE buffers differ: head16 has
8 frequency pairs per head, head32 has16. Scores divide by sqrt16=4 versus
sqrt32≈5.657. These are intended consequences of head dimension, not independently
isolated mechanisms. We do not change the attention scale to manufacture equality.

Within each family head dimension stays fixed as width grows. Matrix aspect ratios
and the existing Muon LR/RMS-factor cancellation remain unchanged. This checks
implementation consistency only; it does not prove optimal LR transfer or a
head-number scaling theorem. Head16 is allowed only by the explicit
`attention_width_policy=mha_head16` with the narrow two-layer SwiGLU contract.
Other head dimensions and attention architectures remain unsupported for that
contract. The study worker now takes each family's head dimension from the frozen
plan instead of assuming32, with a regression check at both local/transfer stages.

All SwiGLU arms have114688/423936/1626112 parameters. At fixed batch/context,
head16 doubles attention-score tensor elements relative to head32; this is not
a claim of doubling total memory or runtime. Measure process RSS and training
time; the GEMM shapes and softmax workload differ despite identical parameters.
Quality and runtime effects are unquantified in advance. Fresh seeds still reuse
adaptive validation, not independent held-out confirmation. No GQA, MLA, depth or
MoE comparison is included. Those require separate scaling and budget reviews.

Before launch: parameter/init/group equality, finite gradients, causal behavior,
protocol drift rejection, serial48-trial selection ordering/head propagation and
full repository tests; three short actual MPS train/checkpoint checks at head16.
Correctness checks are not quality evidence. Freeze clean source, plan and full
bank snapshot, then execute `python -m src.research.head_study --plan PLAN --output
NEW_RUN --allow-training`. Historical study evidence and source contracts remain
unchanged, and the human-stopped stage1 campaign remains stopped.

### Completed stage 2C: retain head32; head16 gains are inconsistent and slower

All 48 trials, checkpoint hashes, per-family worker head counts, frozen selection,
report aggregates and source/corpus/runtime/knowledge contracts were verified.
Mean final validation CE at 3000 updates on development seeds 111/222/333:

| Arm | LR | Width 64 | Width 128 | Width 256 |
| --- | --- | --- | --- | --- |
| Original GELU head32 | .002 | 1.649678 | 1.567840 | 1.534493 |
| Fixed SwiGLU head32 incumbent | .0025 | 1.639925 | 1.560148 | 1.533336 |
| Tuned SwiGLU head32 | .0025 | 1.639156 | 1.559536 | 1.529393 |
| Tuned SwiGLU head16 | .0025 | 1.633562 | 1.559925 | 1.527879 |

The frozen local winner remains the head32 incumbent. Head16's width 64 mean
improved by −.006363 versus the incumbent and−.005594 versus tuned head32, but
seed 333 worsened by +.002028 / +.001835. It failed the all-negative-pair condition.
Both tuned families chose the upper LR boundary; no grid extension is planned.

At 128, head16 versus tuned head32 changed mean CE by +.000389 with mixed seeds.
At 256 it improved by −.001514, again with mixed seeds. Neither comparison passes.
Head16 versus the fixed incumbent at 256 did pass (mean−.005456, every seed better),
but **the separately trained tuned head32 arm uses the same .0025 recipe** and
produces a different comparison. The two head32 repeat means differ by .000768 /
.000612 / .003943 at 64/128/256; width 256 seed 111 differs by .007560. Preserve both
runs. The threshold-sensitive result is not evidence of an independently robust
head16 advantage, and larger-width diagnostics cannot revise local selection.
This round does not establish the cause of repeated-run variation.

Head16 took 36.01 / 60.80 / 74.10 mean training seconds versus 33.45 / 54.73 / 67.10
for tuned head32: **7.65% / 11.10% / 10.43% longer**. Parameter counts are identical;
more heads still change intermediate tensors and execution cost. No equal-time
quality experiment was run. Process RSS differences were small and do not measure
peak accelerator demand. The repeated SwiGLU incumbent beat GELU on all 64/128
seeds, but width 256 remained mixed; this is additional adaptive development
evidence, not independent held-out confirmation.

Evidence: `research_runs/architecture-stage2c-001/{plan,selection,report,review}.json`.
The most promising recipe remains bias-free SwiGLU 21/8, MHA head32, Muon/AdamW+muP,
LR .0025 and independent decay .00006, with the original GELU recipe retained as a
reference. Next is a separately reviewed grouped-query attention comparison;
no GQA/MLA, depth or MoE claim follows from this head-partition experiment.

### Stage 2D: GQA with two query heads per KV head

This bounded round retains both the original GELU/MHA reference (LR .002) and
fixed SwiGLU/MHA incumbent (LR .0025). It separately tunes SwiGLU/MHA and
SwiGLU/GQA over LR .0015/.002/.0025, using fresh development seeds 444/555/666.
All arms use two layers, query head dimension 32, RoPE, the same Muon/AdamW hybrid
with muP base width 64, independent decay .00006, clipping 1, zero dropout,
3000 updates, batch 8 and context 128. SwiGLU remains bias-free with hidden width
21d/8. No simultaneous FFN or query-head change is made among the SwiGLU arms.

The explicit `attention_width_policy=gqa_half_head32` derives KV heads from each
trial width; manual `n_kv_heads`/`n_groups` overrides are rejected. At widths
64/128/256, query heads are 2/4/8 and KV heads are 1/2/4. Each KV head serves two
query heads. This preserves fixed projection aspect ratios as width grows;
keeping one KV head at every width would be a different experiment.

The existing GQA implementation uses a fused QKV projection and broadcast grouped
matmuls. Its QKV weight shape is 2d × d versus MHA's 3d × d; output projection is
unchanged. With biases enabled, each of two layers saves d²+d parameters. Total
GQA parameters are 106368/390912/1494528 versus SwiGLU/MHA's
114688/423936/1626112 (including the unused learned position table). Thus this is
a capacity/runtime tradeoff, not a parameter-matched or equal-compute comparison.
Stored K/V head tensors halve; attention scores still have the same number of
query heads. No inference latency or KV-cache benchmark is included, and process
RSS is not peak accelerator memory. Measure actual training time and RSS.

Initialization remains fan-in scaled for hidden matrices; muP scales hidden
biases by sqrt(d/64), readout initialization by sqrt(d/64), and logits by 64/d.
Different QKV shapes consume different random draws, so equal seeds do not imply
identical initializations. All hidden 2D weights, including fused QKV, use Muon;
other parameters use AdamW. For m=d/64, Muon's hidden LR is base LR/sqrt(m).
The RMS-matching multiplier is .2 sqrt(max(rows, cols)); its product with the LR
is width-invariant within each fixed shape family. At QKV it is
base LR × .2 sqrt(128) for GQA versus base LR × .2 sqrt(192) for MHA. The ratio
sqrt(2/3) is retained, not silently compensated. Equal bounded LR tuning gives
both families the same search budget; it cannot establish an optimal transfer
rule, identical activation statistics or a pure causal effect of KV sharing.

There are exactly 48 serial trials: 24 width-64 screening trials followed by 24
transfer trials at widths 128/256. Select each tuned family's LR by lowest mean
final-budget width-64 CE (tie: lower LR). GQA replaces the incumbent only if its
mean paired CE delta is strictly below -.005 and every seed delta is negative
against **all three** comparators: original GELU, fixed SwiGLU, tuned MHA. Freeze
that decision before transfer. Otherwise retain the prior incumbent without
claiming this round newly confirms it. Report both fixed and tuned MHA repeats
separately if they choose the same recipe; do not pool away repeat variation.
Do not expand the grid if a boundary LR wins. Held-out test remains closed.

Limits: 600 seconds per trial, 4 hours per round, 4 GiB output, first failure
stops, no automatic retry. This is a bounded round within the ongoing research
protocol, not an aggregate cap on the effort. Prior stages remain closed.

Before launch, check all-width parameter counts and optimizer groups, finite
causal gradients, checkpoint continuation, unsupported-config rejection, real
request propagation through all 48 jobs and frozen selection before transfer.
Run repository tests and three short actual MPS training/checkpoint checks.
These are implementation checks, not quality evidence. Bind clean committed
source, corpus, and the full shared knowledge snapshot to a new read-only plan;
launch `python -m src.research.gqa_study --plan PLAN --output NEW_RUN --allow-training`.

### Completed stage 2D: GQA is smaller and modestly faster, but does not replace MHA

All 48 checkpoint hashes, saved QKV shapes, worker requests/results, frozen
selection, report aggregates and source/corpus/runtime/knowledge contracts were
verified. Mean final validation CE at 3000 updates on development seeds 444/555/666:

| Arm | Selected LR | Width 64 | Width 128 | Width 256 |
| --- | --- | --- | --- | --- |
| Original GELU/MHA reference | .002 | 1.661810 | 1.569754 | 1.522443 |
| Fixed SwiGLU/MHA incumbent | .0025 | 1.647115 | 1.558650 | 1.521123 |
| Tuned SwiGLU/MHA | .002 | 1.646205 | 1.559586 | 1.521749 |
| Tuned SwiGLU/GQA | .0025 | 1.643881 | 1.560941 | 1.522255 |

GQA's frozen width-64 mean deltas were −.003234 versus the incumbent and
−.002325 versus tuned MHA; seed 666 worsened by +.003501 and +.001727 respectively.
It failed both the mean threshold and all-seed requirement. GQA chose the upper
LR boundary; MHA chose the interior .002. The grid remains closed, and the
prior SwiGLU/MHA .0025 incumbent is retained. This does not declare its LR
universally optimal or revise the historical stage-2B selection.

At widths 128/256, GQA's mean CE was worse than the incumbent by +.002291/+.001133
and worse than tuned MHA by +.001356/+.000506; every comparison had mixed seed
signs. At width 256, GQA versus tuned MHA ranged from −.009231 to +.011184 across
seeds. GQA beat the original GELU reference under the rule at widths 64/128,
but not at 256. The original reference alone would therefore overstate the case
for changing attention: the stronger SwiGLU/MHA comparator matters.

GQA mean training times were 32.24/53.57/63.94 seconds versus tuned MHA's
33.47/54.16/67.07: **3.68%/1.09%/4.65% faster**. GQA had about 7.3–8.1% fewer
total parameters under this contract. RSS did not fall consistently; process
RSS is not accelerator memory. These measurements support a modest training
speed/capacity tradeoff, not a quality win, statistical equivalence, an inference
speed claim or superiority at an equal wall-clock budget. Shape-dependent Muon
updates, initialization draws and parameter counts differ as preregistered.

The repeated fixed SwiGLU incumbent again beat GELU on every seed at 64/128,
with mixed results at 256. This adds adaptive development evidence while leaving
the largest-width uncertainty intact. Tuned MHA selected .002 here, so its
transfer results and the fixed .0025 reference are different recipes; do not
label them identical-recipe repeats or pool their measurements.

Evidence: `research_runs/architecture-stage2d-001/{plan,selection,report,review}.json`.
The most promising incumbent remains bias-free SwiGLU 21/8, MHA head32,
Muon/AdamW+muP, LR .0025 and independent decay .00006. Next is a separately
reviewed depth comparison; width scaling alone does not validate depth scaling.

### Stage 2E: two versus four layers, with unscaled residual branches

This round asks whether adding two transformer blocks improves quality enough to
justify the additional capacity and training time at a fixed token budget. It
keeps the original two-layer GELU/MHA reference at LR .002 and fixed two-layer
SwiGLU/MHA incumbent at LR .0025. Two-layer and four-layer SwiGLU/MHA each get
LR .0015/.002/.0025, fresh development seeds 777/888/999, and 3000 updates.
All SwiGLU blocks keep head dimension 32, MHA, RoPE, RMSNorm, bias-free FFN with
hidden width 21d/8, zero dropout, clipping 1 and independent decay .00006.
Batch size 8, context 128 and validation settings remain fixed. Held-out test
stays closed. Neither attention nor the per-layer FFN shape changes.

The explicit `depth_width_policy=depth4_unscaled` permits exactly four layers,
SwiGLU 21/8, MHA head32, base width 64, and widths 64/128/256. It rejects GQA,
head16, KV overrides, other depths and unnamed depth policies. Existing
historical two-layer SwiGLU configurations keep their original contract.

**Depth and width are separate axes.** At both depths, each branch is still
`x + attention(norm(x))` followed by `x + FFN(norm(x))`. There is no new 1/L or
1/sqrt(L) residual multiplier, no depth-dependent LR factor and no output-weight
rescaling with depth. Hidden matrices retain fan-in initialization; muP's hidden
bias, readout initialization and logits factors depend only on d/64. Each
fixed-depth family uses hidden Muon LR base/sqrt(d/64), with its existing
RMS-matching shape factor; nonmatrix parameters/readout/embeddings use AdamW at
base LR. The matrix aspect ratios are unchanged across depth and width, so the
width-factor cancellation remains algebraically valid within either family.
This does not prove depth-invariant feature updates, an optimal transferred LR,
stability at arbitrary depth or an infinite-depth parameterization.

Adding layers changes residual accumulation and the number of trained matrices.
Under a matched seed, the first two blocks initially match, but the later blocks
are fresh and the readout draws occur later in the RNG stream. Initialization
is therefore not identical across full models. Global norm clipping can also
interact with the larger parameter set. The comparison evaluates the complete
recipe; it does not isolate depth from capacity or guarantee equal optimization
dynamics. No partial weight transplant or warm start is used.

A SwiGLU/MHA block contains 95d²/8 + 6d parameters. Adding two blocks gives total
counts 212736/814592/3185664 versus 114688/423936/1626112 for two layers: about
85.5%/92.2%/95.9% more. Embeddings, readout and the unused position table remain
shared in shape. This is neither matched capacity nor equal elapsed compute.
Training time and process RSS will be reported; process RSS is not peak
accelerator memory. Short finite-gradient/checkpoint checks establish only
execution correctness, not quality or full-horizon stability.

The frozen plan has exactly 48 serial trials: 24 width-64 screens and 24 transfer
trials at widths 128/256. For each tuned family select the lowest mean final CE
(tie: lower LR). Four layers replace the incumbent only if width-64 mean paired
CE delta is strictly below -.005 and every seed improves versus all three
comparators: original GELU, fixed SwiGLU and tuned two-layer SwiGLU. Freeze the
selection before transfer. Otherwise retain the prior incumbent, without claiming
new confirmation. A larger model passing this quality rule still requires an
explicit resource tradeoff; it is not an automatic efficiency winner. Do not
extend the grid or revise selection after seeing larger-width results.

Limits remain 600 seconds per trial, 4 hours per round, 4 GiB output, one failure
stops, no automatic retry. This bounded round is part of the ongoing effort,
not an aggregate research cap. Verify all-width counts, initialization scope,
optimizer groups, gradients through every block, causality, checkpoint
continuation, policy rejection and depth propagation through all 48 requests.
Run repository tests plus three short actual MPS train/checkpoint checks. Bind
clean committed source, corpus, and the full shared bank into the read-only plan.
Launch `python -m src.research.depth_study --plan PLAN --output NEW_RUN --allow-training`.
Prior stages remain closed; no human-stopped campaign resumes.

### Stage 2E measured conclusion: four layers win, with a resource cost

All 48 preregistered trials completed and passed checkpoint-hash, saved layer/shape,
parameter-count, request/result, source/corpus/runtime, full-bank snapshot,
frozen-selection and aggregate verification. No training was rerun for review.
Both tuned depths chose LR .0025 at the grid boundary; the grid is closed.

| Recipe | LR | CE64 | CE128 | CE256 |
|---|---:|---:|---:|---:|
| Original two-layer GELU/MHA | .002 | 1.661662 | 1.580095 | 1.535058 |
| Fixed two-layer SwiGLU/MHA | .0025 | 1.646487 | 1.559472 | 1.534609 |
| Tuned two-layer SwiGLU/MHA | .0025 | 1.645470 | 1.561114 | 1.529471 |
| Four-layer SwiGLU/MHA | .0025 | **1.583606** | **1.516719** | **1.506713** |

Four layers passed the frozen width-64 rule against all three references.
Its mean CE deltas versus the fixed SwiGLU incumbent were
−.062881/−.042753/−.027896 at widths 64/128/256; versus tuned two layers they were
−.061863/−.044395/−.022758. Every paired seed improved against every comparator
at every tested width, and all mean differences passed −.005. Larger-width
results support this frozen selection; they did not revise it.

The gain uses more resources. Four-layer parameter counts are
212736/814592/3185664, about 85.5%/92.2%/95.9% above two layers. Mean training
seconds were 55.040/63.150/125.819 versus 33.421/54.142/67.025 for tuned two layers:
**+64.69%/+16.64%/+87.72%**. These are measured timings, not a smooth scaling law
or equal-compute comparison. Mean lifetime peak process RSS was
472.93/475.14/481.46 MiB versus 471.74/472.38/475.28 MiB; RSS is not peak accelerator
memory. No inference-speed or efficiency-optimum claim follows.

The fixed and tuned two-layer arms selected the same recipe but differ on repeat:
tuned-minus-fixed mean CE was −.001018/+.001642/−.005138. At width 256, seed 777
alone differed by−.013150. Both references remain recorded; no exact numerical
reproducibility or identified cause is claimed. The four-layer gain survives
comparison with both. The fixed two-layer arm still had mixed width 256 support
against original GELU, while its tuned repeat passed the rule there.

Four-layer bias-free SwiGLU 21/8, MHA head32, Muon/AdamW with muP base64,
LR .0025, independent decay .00006, clipping 1 and unscaled residual branches is
now the most promising observed **quality** recipe in this development setting.
The two-layer SwiGLU recipe remains a cheaper reference and original GELU is
preserved. This is adaptive Shakespeare validation with seeds 777/888/999 and
3000 updates, not independent held-out confirmation, a universal optimum,
matched capacity, pure depth causality or validated infinite-depth muP.
Stage 2E is closed. Future MoE/MLA work requires its own implementation,
initialization, optimizer/scaling and finite parameter/runtime budget review.

Evidence: [report](../research_runs/architecture-stage2e-001/report.json),
[selection](../research_runs/architecture-stage2e-001/selection.json),
[review](../research_runs/architecture-stage2e-001/review.json).

### Stage 2F: quarter-width MLA at fixed four-layer depth

Stage 2E is closed. This finite round compares the repository's MLA implementation
with MHA on four-layer SwiGLU, keeping the winning depth and per-layer FFN fixed.
It retains the original two-layer GELU .002 reference, cheaper two-layer SwiGLU
.0025 reference, and four-layer SwiGLU/MHA .0025 incumbent. Four-layer MHA and
four-layer MLA each receive LR .0015/.002/.0025. Fresh development seeds are
1011/1022/1033; widths 64/128/256, 3000 updates, batch 8, context 128 and validation
seed 2026 are unchanged. No test evaluation or LR-grid extension is allowed.

The explicit `attention_width_policy=mla_quarter_head32` sets both query and
joint-KV latent dimensions to d/4 (16/32/64), with d/32 heads, content head width 32,
rotary head width 32 and value head width 32. Manual `dimcq`, `dimckv`, `head_dim_r`,
KV-head and group overrides are rejected. Under the Muon/muP contract it is
allowed only for four-layer bias-free SwiGLU21/8, base64 and
`depth_width_policy=depth4_unscaled`. Attention and cache use the same latent
shape resolver. Legacy MLA without this policy keeps its historical dimensions.

The implementation down-projects Q and joint KV, then up-projects content Q/K/V
and rotary Q. The rotary key is a separate shared 32-row projection from d.
Within each head it concatenates content and rotary features and divides scores
by sqrt(64); MHA uses sqrt(32). It keeps biases in attention, fan-in initialization,
unscaled residual branches, RMSNorm, final LayerNorm, RoPE and zero dropout.
This changes feature decomposition, score scale, projection shape, initialization
draws and parameter count together; it is not a pure compression ablation.
The low-rank KV and decoupled rotary idea is described in
[DeepSeek-V2, section 2.1](https://arxiv.org/html/2405.04434v5#S2.SS1).
Our tiny biased implementation and ratios are not a replication of that model.
Its optimized inference claims do not establish speedups here: this implementation
expands compressed content keys/values on each cache forward, without weight
absorption. Cache equivalence is a correctness test, not an inference benchmark.

Initialization remains PyTorch fan-in for every projection. At the same seed,
embeddings match but downstream blocks/readout need not match after the changed
attention draws. Hidden linear biases retain sqrt(d/64) scaling; readout init and
forward retain their existing sqrt(d/64) and 1/(d/64) factors. There is no new
latent, depth, residual or attention-temperature scaling factor. All hidden 2-D
weights, including down/up/rotary matrices, use Muon with baseLR/sqrt(d/64) and
its RMS-matching .2*sqrt(max(rows,cols)); vectors/embedding/readout use AdamW at
baseLR. Independent decay .00006 and clipping1 remain unchanged.

The square and d-by-d/4 matrices have fixed aspect ratios as width changes.
The rotary key matrix is 32-by-d and does **not** have fixed aspect ratio. Its
max dimension is d, so the explicit RMS factor still cancels the width LR factor
algebraically over these three widths. This says nothing about equal update
statistics for a fixed-row projection or validated infinite-width MLA muP.
Its spectral geometry and the factorized attention paths differ from fused MHA;
we retain this finite optimizer recipe with equal bounded LR tuning, not a
claim of theoretically optimal or architecture-invariant scaling.

MLA attention matrix weights total 2.5d²+32d, versus MHA 4d². Attention biases
are 5.5d+32 versus 4d. With four layers, the model totals are
196864/733568/2826880 versus MHA212736/814592/3185664, reductions of about
7.46%/9.95%/11.26%. All parameters participate in dense training; there is no
expert routing or auxiliary objective. The persistent inference cache stores
d/4+32 elements per token/layer versus 2d for MHA (calculated shape counts only).
Training still expands per-head keys/values and builds attention scores; neither
these counts nor fewer parameters imply lower measured time or peak memory.
Report full-trial training time and RSS, separating them from short checks and
from analytic counts. RSS is not peak accelerator memory.

Exactly 57 serial trials: 27 local width 64 (three fixed families times 3 seeds plus
two tuned families times 3 LRs times 3 seeds), then 30 transfer (five families times
three seeds times two widths). Select each tuned LR by lowest final mean CE,
tie lower LR. MLA replaces the incumbent only if its width 64 mean delta is
strictly below −.005 and every seed is negative against ALL four comparators:
original GELU, cheaper two-layer SwiGLU, fixed four-layer incumbent and equally
tuned four-layer MHA. Otherwise retain the incumbent. Freeze before transfer;
all larger-width comparisons are diagnostic. Preserve both repeated MHA arms
if the same recipe is selected. Adaptive validation remains distinct from holdout.

Limits: 600 seconds/trial, 4 hours/round, 4 GiB output, first failure stops, no
automatic retry, STOP at trial boundaries, one training worker on this Mac.
Short serial actual MPS train/checkpoint checks at all widths must pass before
preregistration; their runtime informs feasibility only, not quality. CPU checks
cover projected-attention algebra, causal prefixes, cache/full equivalence,
all-projection finite gradients, checkpoint continuation, initialization scope,
parameter counts/optimizer groups, rejection guards and all 57 real request
contracts through the serial engine. Run the complete repository test suite.
Freeze clean committed source, corpus and the full append-only bank snapshot.
Use `src.research.mla_study` and a new unique output directory. Stage 1 and 2A–2E
stay closed; MoE needs its own later routing/scaling/runtime protocol.

Stage2F prelaunch validation: all544 repository tests passed, including20 focused
MLA tests; three serial12-update actual MPS training/checkpoint evaluations passed.
The cache allocator was aligned with the explicit latent-width resolver before
these checks. An existing error-message regex failed the first full run; its
expected GELU/SiLU wording was restored, then the complete suite passed. No
historical evidence changed. These are correctness checks, not quality results.
Short training times were1.141/.496/.624 seconds at widths64/128/256. Linear
3000-update extrapolations are285/124/156 seconds, with startup overhead distorting
the smallest-width result. Applying that proxy to all57 trials gives about3.31hours
before evaluation/process overhead; historical MHA times are lower. This is only
a feasibility estimate under the unchanged600-second/4-hour hard caps. Saved
checkpoint-size extrapolation is about482MB,603MB with25% margin, below4GiB
before small logs. Full-run timing and disk enforcement remain authoritative.

### Completed stage 2F: MLA does not replace four-layer MHA

All 57 serial trials completed on historical source
`5dd790d51f7ee30c44f3633c00648d227ab0f1e8`. Checkpoint hashes, saved layer/head/latent
and projection shapes, parameter counts, requests/results, runtime, frozen local
selection, report aggregates, corpus and full-bank snapshot were verified before
updating the checkout. The held-out test remains closed.

Both tuned families selected base LR .0025 at the frozen grid boundary; the grid
is closed. At width64, MLA is worse on every paired seed than both fixed and tuned
four-layer MHA: mean CE differences +.018238 and +.013924. It fails the replacement
rule. Four-layer SwiGLU/MHA remains the quality incumbent; cheaper two-layer
SwiGLU and original GELU remain references.

| Family | Selected LR | CE width64 | CE width128 | CE width256 |
| --- | ---: | ---: | ---: | ---: |
| Original two-layer GELU | .002 | 1.658560 | 1.573045 | 1.538094 |
| Cheaper two-layer SwiGLU | .0025 | 1.641669 | 1.562098 | 1.519259 |
| Fixed four-layer MHA | .0025 | 1.579393 | 1.518302 | 1.506017 |
| Tuned four-layer MHA | .0025 | 1.583707 | 1.517095 | 1.504022 |
| Tuned four-layer MLA | .0025 | 1.597631 | 1.521783 | 1.498427 |

MLA minus tuned MHA is +.013924 / +.004688 / -.005595 at widths64/128/256.
At128, seeds1022/1033 worsen; at256, seed1033 worsens by+.001803 despite the lower
mean. Against fixed MHA, width256 mean is-.007589 with seed1033+.002252.
Neither larger width passes the all-seed rule against either four-layer reference.
The lower width256 mean is a useful diagnostic, not permission to revise local
selection or an established architecture-by-width mechanism.

Fixed and tuned MHA use identical selected recipes but are separate runs.
Tuned-minus-fixed means are+.004314 / -.001207 / -.001994; keep both references.
There is no identified cause for these repeat differences or claim of exact
training reproducibility. Both four-layer MHA arms beat the original GELU and
cheaper two-layer SwiGLU on every paired seed at every tested width.

MLA's mean training times are86.62 / 101.64 / 131.04 seconds, versus tuned MHA
54.52 / 60.81 / 127.77: increases of58.88% / 67.14% / 2.55%. Historical total
parameter counts are196864 / 733568 / 2826880, about7.46% / 9.95% / 11.26% fewer
than four-layer MHA. Fewer parameters did not yield faster measured training.
These timings are observations on this Mac, not inference benchmarks or equal
compute comparisons. Projection shapes, initialization draws and attention score
decomposition change together; this is not a pure latent-compression effect or
DeepSeek replication. Finite width-scaling algebra does not validate MLA muP.

An agent maintenance stop after trial50 allowed the human-requested RoPE table
repair to be tested separately. Only the seven unfinished trials resumed, on the
unchanged original source and plan. No completed trial was rerun. The repaired
source `7577129` was adopted only after this full study audit. Historical counts
include the dormant position table; new runs omit it and use fresh initialization.
The fix and its 552 tests/three short MPS checks establish correctness, not new
quality evidence.

Evidence: [report](../research_runs/architecture-stage2f-001/report.json),
[selection](../research_runs/architecture-stage2f-001/selection.json), and
[review](../research_runs/architecture-stage2f-001/review.json).

Next: the authorized saved-checkpoint full-validation audit, with paired text-block
uncertainty, no retraining and no held-out test. Context/batch robustness is the
next research priority. A compact interaction design was recommended in discussion
but has not yet been frozen or launched; no MoE launch is planned.

## Prospective efficiency selection after human steering

The human now permits adopting GQA or MLA with a small quality loss when measured
inference savings justify it. Future selection therefore includes a quality
non-inferiority objective alongside the existing quality incumbent. Historical
stage 2D/2F decisions remain frozen; neither is retroactively declared a winner.

The human accepted a maximum loss of +0.01 validation CE nats/token (exp(.01)-1,
about 1.005% higher perplexity). This is a decision preference, not an established
universal threshold. The matched comparator, uncertainty rule and minimum useful
inference benefit must still be frozen before new selection.
Report paired seeds separately from conditional text-block intervals; the latter
do not quantify training-seed variation. Assess each intended width/context rather
than averaging away a harmful setting.

Measure prefill and cached decode separately on matching prompts, lengths,
batches, precision and hardware. Record warmed-up synchronized latency/throughput,
actual KV-cache bytes and peak device memory. Training speed, theoretical cache
sizes and parameter counts do not establish deployment savings. A reduction in
memory can be useful even without a latency gain; report the actual tradeoff.

The completed full-validation audit makes width256 MLA an efficiency candidate:
its mean CE is .007100 below tuned four-layer MHA, with all three paired seeds
better. Width64 is .013453 worse and width128 is .006146 worse on average. These
are adaptive diagnostics, not approval to deploy or proof of non-inferiority.
Fresh context controls still require the repaired RoPE initialization. The next
training grid remains unfrozen; this policy does not authorize extending closed
grids or launching MoE.

The inference preparation uncovered and repaired a GQA cache contract mismatch:
the studied policy derives KV heads, while generation previously required an
explicit field and direct cache creation defaulted to query-head count. Attention
and cache now share a resolver. The older benchmark loader also uses the strict
historical RoPE adapter. All570 project tests and nine serial MPS saved-weight
cached/full-logit checks pass (maximum absolute difference8.11e-6). These establish
correctness, not speed or quality. The bounded saved-checkpoint inference protocol
is drafted in `research/INFERENCE_EFFICIENCY_PROTOCOL.md`; no benchmark is launched.

## Completed inference diagnostic

The24-pair saved-checkpoint inference diagnostic is complete and reviewed.
See [measured inference tradeoffs](INFERENCE_EFFICIENCY_RESULTS.md). Cache savings
and timing are separate observations; historical selections remain frozen.
No training or test access occurred.

## Next bounded interaction protocol

See [depth, GQA and context interactions](CONTEXT_INTERACTION_PROTOCOL.md).
Fresh paired-target controls, screening/confirmation seed separation and shared
reference aliases address interactions without reopening historical selections.
The accepted+.01 quality ceiling is an operational eligibility rule; matched new
inference evidence is still required before adoption.

## Completed context interaction round

All114 trials are completed and reviewed; see [context interaction results](CONTEXT_INTERACTION_RESULTS.md). Three width256 GQA cells meet the frozen +.01 operational quality rule, none at64; four-layer GQA128 loses+.019204 CE versus matched MHA. Four-layer depth improves every matched seed; context512 training worsens mean common-context128 scores in all eight comparisons. Native512 is secondary and uses different evaluation boundaries. New matched inference measurements remain required; the historical quality incumbent stays unchanged. Preserve frozen LRs, all evidence and the human kimi branch. No grid extension or completed-trial rerun. Context256 and batch studies remain separately bounded follow-ups; no MoE.


## Completed context inference follow-up

All 36 paired workloads and 18 saved checkpoints are reviewed; see [context inference results](CONTEXT_INFERENCE_RESULTS.md). The three quality-eligible width 256 GQA settings halve actual KV-cache storage but none passes the prospective all-seed/workload latency guard. Two-layer/context128 fails upper-quartile variability; contexts512 also have median decode regressions above 5%. No automatic adoption or historical quality change. Preserve all raw timings and gates; do not rerun completed jobs or revise thresholds after results. Context256 and batch follow-ups remain separate bounded studies.
