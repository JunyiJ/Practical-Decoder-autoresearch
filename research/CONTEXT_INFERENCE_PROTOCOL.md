# Inference follow-up for context-interaction finalists

This finite diagnostic measures the three width256 GQA cells that passed the
completed context-interaction-001 common-context128 quality gate: two layers at
trained contexts128 and512, and four layers at512. Their comparators are the
matched-depth/context selected MHA recipes. Use every confirmation seed2031,
2032,2033, the frozen LR choices and saved final3000 checkpoints:18 distinct
models. No training, optimizer resume, quality rescoring, holdout access, grid
extension or historical selection revision. Four-layer GQA128 and all width64
cells failed the quality gate and are excluded prospectively from this follow-up.

The reviewed quality gate required mean, every seed and every conditional text
bootstrap upper bound<=human+.01 CE. It is operational eligibility on the common
context128 validation task, not population non-inferiority or evidence that all
native512 quality contrasts satisfy the same rule. Training LR differs for the
two-layer512 GQA/MHA recipes; this benchmark compares the selected recipes.

## Workloads and measurement

Exactly36 jobs:3 depth/context pairs x3 checkpoint seeds x batches1/8 x2 lengths.
At capacity128 use prompt/decode32/32 and64/64; at capacity512 use128/128 and
256/256. Within each pair use identical disjoint validation-prefix strings and
teacher-forced continuation, FP32 MPS, four CPU threads, no MPS fallback. Keep
capacity equal to trained context. Never pool lengths or compare context128
latency directly to context512 as an architecture speedup.

Three unscored warmups per model; nine alternating paired repetitions. Offset
first role by job index. Synchronize before and after prefill and after the full
sequential decode loop. Include model forward and Python loop; exclude load,
cache allocation/reset, input transfer, correctness checks and output sampling.
Report every repetition, per-seed median paired ratio and IQR, prefill seconds,
decode milliseconds/step and processed batch tokens/second. IQR is descriptive,
not a confidence interval; three trained seeds on one machine do not provide
independent hardware replication. Both models remain resident.

Count actual cache tensor bytes, both allocated and logically used at the matched
length. GQA half-head shape predicts half MHA storage; verify this against actual
buffers, do not substitute parameter counts. Report current MPS allocation with
its actual label; true peak device memory is unavailable. Cache is only part of
model/system memory. Report latency regressions even if memory improves.

## Prospective engineering benefit gates

Freeze these operational thresholds before pilots or timing results. Each cell
must meet them for EVERY seed and all four workloads, with no averaging away a
regression. A speed benefit requires median paired decode ratio<=.90 and its
upper quartile<=1.00. A memory benefit requires actual allocated cache ratio<=.75.
Both routes also require a latency guard: every prefill and decode median ratio
<=1.05 and upper quartile<=1.10. Efficiency eligibility means the guard plus at
least one benefit route. These are conservative engineering thresholds selected
under the authorized near-neutral-quality/meaningful-inference-benefit objective,
not human-specified numerical latency tolerances, confidence tests or guarantees.
Report the independent flags and raw ratios even on failure. No automatic
adoption or change to the historical four-layer MHA quality incumbent follows.

## Source, correctness and limits

Research source is the isolated codex/context-interaction worktree; the human
kimi checkout stays untouched. Current main includes merged GatedMLA/RoPE code.
This diagnostic uses legacy MHA/GQA only, verifies each exact resolved saved
config and training contract, loads weights strictly and compares full/cached
logits before each timing job at atol1e-4,rtol2e-5. Original checkpoints and hashes
stay immutable. No initialization, residual, width/depth or optimizer rule is
changed; there is no new scaling or training claim.

Tests must cover512 cache allocation/used capacity, cached/full equivalence,
all-seed/workload gates, panel selection and controller dispatch. Actual serial
MPS pilots exercise the largest B8 workload in each of the three cells, using
seed2031 only, before clean source commit/push and readonly preregistration.
Pilot measurements are feasibility checks, excluded from research aggregates.
No pilot-driven threshold or panel revision. On correctness failure diagnose
before launch; never automatically retry a failed research job.

Hard bounds:120seconds/job,3600seconds/round,512MiB output,36jobs, one worker,
first failure stops. Check source, corpus, runtime, prerequisite evidence and
complete shared knowledge hashes at launch/every job/completion; verify18 unique
checkpoint hashes and all36 raw requests/results/324 paired repetitions at
completion, rebuild aggregates and gates independently. Frozen source and plan
remain unchanged during execution. Full shared knowledge includes recovered
optimizer findings, all architecture results, coverage and prior inference
caveats, context interactions and permanent campaign006 stop.

Prelaunch validation:653 project tests passed, followed by20 affected tests after
adding the direct historical request-hash guard. Three actual MPS B8 pilots at
maximum length completed in5.70/17.93/21.15seconds for2L128/2L512/4L512.
Maximum absolute cached/full-logit error was1.824e-5, within the combined frozen
absolute/relative tolerance. All18 saved configuration hashes were verified.
Pilot raw outputs are retained under research_state/context-inference-pilots;
timings are excluded from research. No threshold or workload changed after pilots.
