# Four-layer MHA context256 follow-up

A separate finite robustness study completes the authorized context128/256/512
question after the closed depth/attention/context factorial. It focuses on the
four-layer MHA quality incumbent; GQA and MLA are not adopted, and no MoE or
attention/FFN/depth intervention is combined here. The completed factorial and
inference panels remain closed. This study adds the untested middle context256
and uses fresh simultaneous endpoints and references on a new source contract;
it does not rerun old seeds to recover or change interpretation.

## Design and frozen local selection

Three tuned cells: four-layer SwiGLU MHA at trained contexts128/256/512, batches8/4/2.
Widths64 and256, head dimension32, bias-free SwiGLU hidden width21d/8,
Muon/AdamW hybrid with muP base64, depth4_unscaled, unscaled residuals, RoPE,
zero dropout, independent per-update decay.00006, global clipping1. No new width,
depth, optimizer or initialization rule is inferred. Shape/factor correctness is
not proof that optimal LRs transfer. The current source includes merged human
GatedMLA changes; these jobs explicitly use MHA and verify its unchanged active
initialization and optimizer construction across contexts.

All trials use paired_blocks512: two512-token training chunks/update reshaped
without changing flattened inputs or targets. Every trial trains3000 updates,
1024 targets/update,3072000 targets total. Private training RNG and initial active
weights match across contexts at a given width/seed; validation RNG is separate.
Context changes sequence count and correlation/available history, not just a
positional table. Do not pool these fresh controls with older measurements.

Screen exactly18 full-budget width64 trials:3 contexts x LRs.00125/.0025/.005 x
seeds2041/2042. Interleave contexts and reverse on the second seed. Select each
context's LR by lowest mean final full-validation-context128 CE, tie lower LR.
Freeze all3 LR choices and the complete confirmation manifest before any fresh
confirmation or width256 job. Boundary choices stay boundary choices; no grid
extension, pruning or target-width reselection.

Confirm all3 selected contexts at widths64/256 on seeds2051/2052/2053:18 trials.
Add fresh context128 references at both widths on those same seeds: original
2-layer dense GELU MHA LR.002, cheaper2-layer SwiGLU MHA LR.0025 and fixed4-layer
SwiGLU MHA LR.0025, up to18 more trials. If selected4-layer context128 LR is.0025,
its identical fixed reference aliases prospectively for every width/seed;
no duplicate run or cherry-picking. This gives exactly48 or54 distinct trials
including screening. All5 seeds were checked against existing study plan panels
and are fresh. References share the new sampler and source but retain original
architectural and optimizer recipes.

## Evaluation and interpretation

Primary final metric: common context128 full validation,111538 targets once per
model, batch8. Native trained-context score is secondary, using batch1024/context.
Context256 thus adds native256; context512 adds native512. Different native window
boundaries and available history mean native scores cannot establish a same-task
quality improvement across training contexts. Held-out test remains closed.

Store2048-token raw block loss sums/counts. Paired conditional text bootstrap uses
blocks2048/4096/8192,2000 draws, seed20261024. Confirmatory descriptive contrasts:
256-minus128,512-minus128 and512-minus256, separately at each width; each tuned
cell versus all3 fixed references. Include every paired seed, mean and all three
text intervals. Report fixed-LR.0025 screening contrasts separately from tuned
confirmation comparisons, which include LR choice. The operational near-neutral
flag requires mean, every seed and every text interval upper bound<=human+.01 CE.
This flag is descriptive eligibility, not a population non-inferiority test,
multiple-comparison correction or automatic adoption. Keep the historical quality
incumbent unchanged; report resource costs and all adverse results.

No inference gain is assumed or measured by training time. The prior GQA
benchmark found50%KV savings but failed its latency guard in all3quality-eligible
cells; preserve that tradeoff. Context256 and the separate forthcoming batch
study address robustness, not attempts to revise those thresholds or results.

## Gates, budgets and provenance

Before launch: focused/full tests, exact paired sampler/init checks, all reference
and candidate shapes/optimizer groups, synthetic serial freeze/alias/failure
checks, actual serial MPS short training/reload/full-validation at contexts256
and512 and widths64/256. Pilot seeds are outside research panels and measurements
are excluded from findings. Do not launch until these pass, source is reviewed,
clean, committed/pushed and a uniquely named readonly JSON plan with the entire
shared knowledge bank is frozen. Primary human kimi checkout remains untouched.

Hard limits:54 trials,1200seconds/trial,21600seconds/round,2048MiB output, one
worker, first failure stops, no automatic retry. These bounds never authorize
extra grid points. A historical-duration feasibility proxy uses1.5 times the
maximum measured4-layer MHA trial duration at each width, with36width64 and
18width256 jobs:2.38hours. Historical maximum checkpoint sizes with25%margin give
638.55MiB. These are planning proxies, not new measured runtime guarantees; the
new pilots must verify feasibility within the explicit limits.

At completion verify every distinct checkpoint and raw request/result hash,
source/corpus/runtime/fullbank, sampler/step/token/depth/head/QKV/FFN/optimizer
contracts, exact screen/fresh seed panels and frozen LR/alias manifest, common
and native coverage, report aggregates and independent conditional-bootstrap
reconstruction. Never rerun a completed trial for interpretation. Append findings
and update best-context records and guide/HTML; preserve all older evidence,
reservations and permanent campaign006 stop. Batch sensitivity remains a later
separately reviewed finite protocol.

Prelaunch checks passed:666 project tests; four serial20-update MPS trusted-worker
pilots at widths64/256 and contexts256/512, including checkpoint reload and
111538-target common/native full validation. Pilot wall times were3.11/3.12/
5.29/5.60seconds; these are correctness checks, not estimates of full-budget
training quality or runtime. Initial development tests caught two generated test
assertions with an incorrect token constant; corrected to1024 before the passing
full suite. Production sampler/target counts were unchanged. Raw pilot evidence
and the prerequisite review are retained in research_state/context256-validation.
