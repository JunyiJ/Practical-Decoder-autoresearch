# Warmup quality protocol

This is a separate, unregistered experiment under the continuing quality objective.
KDA quality001 is completed and independently reviewed on historical source
b254d13a234b5d9075c802dbaa1e32f195553267; its frozen rule retains the MHA/H200
incumbent at peak LR 0.005 and independent decay 0.00006. The local KDA results
and all earlier studies remain closed. This protocol requires actual CPU trainer,
full regression, serial MPS and resource gates before registration and launch.
Prototype metadata checks alone do not establish feasibility or quality benefit.

## Motivation and evidence boundary

The completed schedule study compared constant LR, cosine and late cooldown with
no warmup. Cooldown and H200 tail averaging are now part of the retained recipe.
A search of the current append-only bank found literature/proposals mentioning
warmup and inference-timing warmups, but no completed local training warmup round.
This search is not a new empirical result. Recheck the complete bank before launch.

Kalra and Barkeshli, [Why Warmup the Learning Rate?](https://arxiv.org/abs/2406.09405v2),
report SGD/Adam experiments in which warmup helps tolerate larger target LRs.
Ma and Yarats, [On the adequacy of untuned warmup](https://arxiv.org/abs/1910.04209v3),
discuss Adam update magnitudes and warmup. Neither establishes a benefit for this
Muon/AdamW hybrid, this short budget, or KDA. No local instability or mechanism is
identified here. We propose a fixed-peak test, not a larger-LR search or replication
of their duration rules. A quality improvement is unknown, with no numeric forecast.

## Resolve the common recipe only from final reviewed evidence

After KDA finishes and its full independent audit succeeds, reconstruct its frozen
adoption decision. Use the resulting incumbent's exact attention family, named
width/depth policy, peak LR and independent decay in every arm. If KDA does not
qualify, retain the prior four-layer MHA LR .005, decay .00006. If it qualifies,
use its reviewed selected KDA recipe. Never choose this backbone from partial
scores or larger-width results outside the frozen predecessor rule.

Keep four layers, SwiGLU hidden21D/8, head32, RMS blocks/final LayerNorm, unscaled
residuals, zero dropout, context128, batch8, paired_blocks512, Muon/AdamW muP base64,
clip1, 3000 updates and 3072000 targets. Preserve KDA's named alpha policy if that
is the resolved incumbent. Within each seed all arms have identical initialization
and input order because their model and sampler configurations are identical.
Use fresh same-source controls; do not pool historical measurements.

## Three schedules, fixed common peak LR

The common peak LR is inherited prospectively; no arm-specific LR tuning or new
LR grid is included. This isolates the early schedule at that peak and cannot
establish each schedule's optimized performance or optimal width transfer.

Baseline: existing linear_cooldown, multiplier1 through zero-based update2000,
then 1-.9*(i-2000)/999, ending at .1 at i=2999.

Warmup100 or warmup300: exactly W=100 or300 updates, i=0..W-1, with multiplier
.1+.9*i/(W-1); the first is .1 and the final warmup update is exactly1. Then
multiplier1 through2000 and precisely the same cooldown as baseline. No zero-LR
update, restart, extra update or shifted averaging tail. H200 collects101 snapshots
at completed updates2000,2010,...,3000; raw checkpoint resumes, H200 is inference-only.

All optimizer groups multiply their configured width-scaled base rates by the same
absolute-step scalar, captured before optimizer-state loading. It never compounds.
Independent decay still applies once per update at the unchanged coefficient;
inner decays remain zero. Nominal cumulative decay and tokens match, while the LR
integrals differ: baseline multiplier sum2550, warmup1002505, warmup3002415.
This changes gradient-update/decay balance and momentum trajectories. Do not claim
equal LR integrals, equal compute, pure stability causation or empirical muP proof.

The opt-in schedule kind is warmup_linear_cooldown. Persist total_steps, start_step,
min_ratio, warmup_steps and initial_ratio in the resume contract and reject any
change. Legacy absent/constant/cosine/linear_cooldown behavior must remain exact.

## Finite screening, confirmation and decision

Proposed fresh screen seeds2281/2282, width64: three schedules x two seeds =6
physical full-budget runs. Choose the best mean final H200 full-validation score,
ties baseline then shorter warmup. A warmup winner is eligible only if its mean
gain is strictly greater than .005 CE and both paired seeds improve over baseline.
Otherwise freeze retention, with no alternate fallback. Freeze the choice and
entire confirmation manifest before any fresh-seed or width256 run.

Confirm all three schedules at widths64/256 with proposed fresh seeds2291/2292/2293:
18 physical runs, including the six fresh exact fixed-incumbent baseline cells.
Thus exactly24 physical trajectories and48 correlated raw/H200 outputs, no aliases
or duplicate fixed controls. No pruning, LR or duration extension, larger-width
reselection, or training rerun to recover interpretation. Seeds need a new full
historical-plan check at preregistration; this document does not reserve them.

Only the frozen eligible warmup may replace the incumbent. Require mean gain>.005,
every fresh paired seed negative, and all three conditional text-bootstrap upper
bounds below zero at both widths versus the fresh baseline. No post-hoc fallback
to the other duration. Report every warmup-minus-baseline and300-minus100 contrast,
including failures and raw correlated diagnostics. A retained incumbent is a
legitimate round outcome; the overall quality objective continues.

Primary final H200 validation uses all111538 targets once in disjoint context128
windows, evalbatch8. Keep raw2048-token blocks; paired conditional bootstrap uses
2048/4096/8192 blocks,2000 draws, seed20261024. Sampled training logs and raw finals
are not alternate selection metrics. Holdout stays closed. This is adaptive local
validation, not seed-population significance or a universal optimum.

## Gates and caps before any launch

Keep this prototype separate until KDA completes and is reviewed. Then independently
review and integrate it into clean source; verify the entire project suite plus
actual CPU trainer/resume trajectories, optimizer groups/slots/factors/decay, sampler
and test RNG, raw/H200 noninterference, strict saves, changed-contract rejection and
worker schedule histories. Scalar checks alone do not satisfy these gates.

Perform six serial MPS correctness/resource pilots after the live study has ended:
each schedule at widths64/256,40 updates,40960targets, provisional seed9371; map the
100/300 warmups to2/4 pilot updates, cooldown start20, H200 snapshots20/30/40.
Pilot first-update ratio.1, peak endpoint1 and final.1 are explicit. Preserve the
existing full/cached atol1e-4/rtol2e-5, full raw/H200 coverage, CPU finite tensor and
optimizer audits, private RNG and snapshot counters. No independent MPS repeat
equality claim or tolerance relaxation. Pilot data are excluded from quality ranks.

Proposed hard limits:24 research trials,2400seconds/trial,21600seconds/round,
4096MiB artifacts and process RSS,6144MiB sampled current MPS driver allocation.
One worker, first failure stops, no automatic retry. Pilot caps240seconds/job,
1800seconds total,1024MiB artifacts. Current allocation is not peak memory.
Fresh pilots must show a maximum24-job projection with50% time/artifact margin
fits every cap; prior KDA timings are only preliminary context. If feasibility
fails, preserve the evidence and do not launch or silently expand this protocol.

After all gates pass, freeze the completed predecessor review, exact incumbent,
source/runtime/corpus hashes, prerequisite review, full shared bank and a uniquely
named plan on tested clean committed/pushed source. At completion independently
audit every request/result/raw/H200 hash, saved model/optimizer/schedule/sampler/RNG/
step/token/coverage contract, frozen manifest, aggregates/bootstrap and decision.
Append findings and update best-config context and the readable guide before the
next distinct round. All closed studies and permanent stop006 stay closed.
