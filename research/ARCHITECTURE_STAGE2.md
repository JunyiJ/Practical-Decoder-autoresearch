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
