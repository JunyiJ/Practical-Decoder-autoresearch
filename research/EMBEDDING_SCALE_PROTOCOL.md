# Input embedding scale: prospective quality proposal

Status: implemented and CPU-verified; serial MPS and resource gates remain before
registration or research launch. All 2056 project cases passed, including twelve
actual input-scale geometry/training/resume cases. These are correctness checks,
not model-quality measurements. Warmup-quality-001 is completed and independently
reviewed; it retained the no-warmup incumbent. The resolved common recipe uses
warmup_steps=0, peak LR .005 and independent decay .00006. All closed studies and
permanent stop006 remain closed; the human quality objective continues.

## Why investigate this next

Source inspection shows GPT passes an unscaled nn.Embedding output into four
pre-normalized residual blocks; it has no explicit input multiplier. The readout
is a separate untied matrix with the existing muP forward factor. Testing a fixed
input multiplier changes the relative contribution of the starting token vector
and subsequent residual updates, and changes embedding gradients. A benefit is
unknown; there is no numerical quality forecast or identified local failure.

Nguyen and Salazar (2019), Transformers without Tears, report normalization and
embedding-length interventions in machine translation. This motivates considering
input representation geometry, but does not validate a constant scalar, our
Muon/AdamW recipe, character modeling, or this budget. The proposed multiplier is
not their FIXNORM method: it does not normalize each token vector or tie readout
weights. Primary source: https://aclanthology.org/2019.iwslt-1.17/ (abstract read).

## Conditional common recipe and one changed locus

Resolve the exact incumbent only after all24 warmup trajectories are completed
and independently audited under the frozen rule. If warmup fails, inherit baseline
late cooldown; if it qualifies, inherit its exact100/300 schedule. Do not select
from partial warmup scores. Inherit MHA, four layers, bias-free SwiGLU21D/8, head32,
RMS blocks/final LayerNorm, unscaled residual paths, RoPE, zero dropout, context128,
batch8, paired_blocks512, Muon/AdamW muPbase64, clip1, independent decay.00006,
3000updates/3072000targets and H200 averaging101snapshots at2000..3000interval10.

Proposed arms multiply the looked-up token embedding by1,0.5,0.25 immediately
before the first residual block. Do not modify initialization, token IDs, readout,
per-layer residual coefficients, trainability, optimizer grouping or data. Use an
explicit narrowly validated named input policy; reject unsupported attention/depth,
learned-position, tied-weight, manual residual and arbitrary multiplier combinations.
No-op default must remain bitwise identical and add no RNG draws or parameters.
All active weights and RNG must match at initialization across arms and widths.

A width-independent constant changes no parameter shape or declared width exponent.
That algebra does not validate optimal LR transfer: embedding effective gradients,
relative residual contributions, optimizer trajectories and independent-decay balance
change. Preserve AdamW embedding/readout groups and Muon hidden-matrix factors;
no compensating LR or init factor is silently introduced. Check every stored
optimizer slot/group/step/factor, model config and resume contract. Persist/reject
changed input scale for training resume and inference loading; raw/H200 must use
exact saved semantics. Cached/full and causality checks must include the multiplier.

## Proposed finite prospective comparison

Equal bounded peak-LR tuning is needed because the input-coordinate change can
alter update geometry. Proposed LR set.0025/.005/.01 (three values, one round,
no extension); this is a new intervention comparison, not a reopening of an old
LR grid. It also tests a wider peak for the contemporaneous control; interpret
candidate-versus-tuned-control separately from candidate-versus-fixed-incumbent.
Review actual stability/resources before freezing this set; never adapt it after
research outcomes. Fresh proposed screens2301/2302, width64,3scales x3LR x2seeds=18.
Freeze each family LR by mean final H200, ties lower LR. Rank tuned families by mean,
ties baseline then larger scale. A changed-scale winner eligible only if mean
gain>.005 and both seeds improve versus both tuned baseline and fixed incumbent.
A tuned baseline winner may qualify only by the same gain>.005/everyseed
rule versus the fixed incumbent, followed by both-width fresh confirmation.
This route is prospectively included to retain a control LR improvement; it
does not attribute that improvement to input scaling. No alternative fallback.

The fixed incumbent LR is included in screening only if inherited.005 as expected;
if predecessor resolves otherwise, revise this draft before preregistration, not
silently alias a nonmatching control. No fallback if best changed arm fails.

Fresh proposed confirmation2311/2312/2313, widths64/256, all3 selected families=18.
Add6fresh fixed incumbent controls, prospectively alias to selected baseline iff
entire saved recipe matches. Exactly36or42physical trajectories,72or84correlated
raw/H200 outputs. Freeze all aliases and schedule before confirmation. No duplicate
exact-recipe reference training, pruning, larger-width reselection or extensions.
Only the frozen eligible candidate may adopt with gain>.005, everyseednegative
and all3conditionaltextupper<0 at both widths. Changed-scale arms must beat both
tuned baseline and fixed incumbent; a tuned baseline winner must beat the fixed
incumbent. Retention is valid, while the overall human objective continues.

All111538validationtargets onceeach, ctx128B8, raw2048blocks. Pairedconditional
bootstrap2048/4096/8192,2000drawsseed20261024. Raw endpoints and half-versus-quarter
contrasts are diagnostic, not additional selection opportunities. Adaptive
validation and conditionaltext uncertainty are not seed-population significance;
holdout stays closed. Report training time and actual saved parameter counts.
Sameparametercount/tokens does not establish equal compute or inference speed.

## Implementation and launch gates

Warmup completion was independently audited before source integration. Actual CPU
geometry/training/resume checks and the complete regression suite passed on source
SHA 20e18d863fd3f6ff4ad0bb9167c89c57c0d9df4ead97dbba836443eee6a90d0a.
The original proposal and CPU evidence are preserved in the preparation folder;
this publication update changes documentation only. Before registration/launch,
verify fresh seed uniqueness,
review initialization/residual/width/optimizer/checkpoint semantics and full bank;
run focused actual CPU forward/backward/train/resume/H200/optimizer/RNG/cached
correctness plus full project regression, then six serial MPS pilots at bothwidths
and all3scales with exact saved-config reload and fullvalidation. Existing cache
atol1e-4/rtol2e-5 remains; preserve any failures with no automatic retry. Bound
pilot jobs and keep measurements out of research rankings. CPU-only metadata
checks cannot substitute for these actual model gates. No competing model work.

Provisional execution caps:42trials,2400s/job,21600s/round,4096MiB artifacts.
Memory clarification recorded during the run: the inherited worker bootstrap also
starts a nominal 0.5-second watchdog sampling process peak RSS (4096 MiB) and
current MPS driver allocation (6144 MiB, not peak). It exits on a sampled limit
breach or guard error; the controller stops without automatic retry. The controller
also bounds wall time and samples output disk usage. Earlier pilot-only wording
was incomplete; source, thresholds, frozen plan and training never changed.
Sampling is not instantaneous or total-system memory enforcement. Pilot
extrapolation must fit with 50 percent time/artifact margin;
there is no resource estimate yet. Oneworker, firstfailurestop. Freeze tested,
reviewed, clean committed/pushed source, corpus/runtime, completedpredecessorreview,
exact selectedschedule, all seeds, finite caps and fullknowledge snapshot before
one uniquely named launch. Final independent audit must cover all checkpoints,
raw requests/results/config/sampler/RNG/optimizer/shapes/steps/tokens/H200/coverage,
selection/aliases, complete report and independently reconstructed bootstrap.


## Completed evidence and documentation clarification

`embedding-scale-quality-001` completed 36 physical trajectories and 72 correlated raw/H200 outputs. All checkpoint payloads/hashes and the entire report passed independent review on historical source `1dadecddd57060aecf1910cb3553d11b90ff0c1a` before this documentation update. The incumbent is retained; [paired findings and limitations](EMBEDDING_SCALE_RESULTS.md). The original source and frozen run plan preserve the historical contract. Later documentation does not change evidence or selections.
