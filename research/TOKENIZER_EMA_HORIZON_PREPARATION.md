# Tokenizer EMA-horizon preparation

Status: real pilot accepted and three fresh quality trajectories frozen, November 1 UTC / October 31 Los Angeles. The block-normalization campaign is closed and
RMSNorm/SiLU remain adopted. This is an averaging hyperparameter experiment;
it does not change architecture, optimizer, training horizon or LR schedule.

## Why this question

The historical character study selected H200 over H10/H50 using a 1,000-update
averaging tail. The adopted subword MLA recipe uses a 400-update tail. With
initialization at update800 and snapshots every10 updates through1200, H200
retains coefficient 2^(-400/200)=0.25 on its initial snapshot. H100 retains
2^(-400/100)=0.0625. A shorter half-life may reduce averaging lag during cooldown.
Neither this algebra nor sampled loss curves establishes a quality gain.

The three latest accepted RMSNorm trajectories show declining sampled online
validation loss late in training. Those measurements are diagnostics and are
not directly comparable to full-validation H200 CE. Earlier width256 MLA full
validation found raw better than H200 on two of three seeds and worse on one;
the historical wider MLA result instead favored H200. Averaging transfer is
therefore uncertain and width-specific. No old checkpoint is selected, rescored
or promoted using this hypothesis.

Parallel separate-norm residual topology is deferred: it worsened all historical
character pairs at widths64/256, and this review found no specific new MLA
benefit that outweighs that negative prior. A longer horizon is also deferred:
the closed subword GQA duration study strongly favored1200 over3000; latest MLA
sampled curves do not establish whether a new duration would help. MoE still
needs a feasible complete capacity plan. The closed negative BPE tying study
will not be reopened to save storage. Architecture research remains on the agenda
after this bounded HP question.

## Concrete proposed comparison

Use the unchanged width256, four-layer quarter-latent RoPE MLA, RMSNorm, SiLU
SwiGLU672, serial unit residuals, untied GPT2 tables and explicit grouped-embedding
v2 numerical policy. Keep LR0.00125, context128, batch8, paired_blocks512,
1200 updates, cooldown800 to0.1, independent decay and tokenizer/corpus fixed.

Track **both H100 and H200 on each new online trajectory**. For each learned
parameter, initialize A_h = theta_800 and then apply
A_h <- beta_h A_h + (1-beta_h) theta_t at t=810,...,1200, where
beta_h=2^(-10/h). This gives41 snapshots. Nonlearned persistent buffers come
from the final raw checkpoint. Neither average feeds back into training.

Keep one raw checkpoint with both EMA states and two inference-only exports
per trajectory. The outputs are correlated paired measurements, not independent
training replicas. Shared online weights justify fixing the inherited LR for
both returned-weight choices; no separate LR search or screen is proposed.

One20-update resource/transport pilot, followed by exactly three new fresh
1200-update trajectories. The pilot uses the inherited start13/interval1
correctness tail with eight snapshots; it is excluded from quality decisions.
All four raw/H100/H200 triples remain retained. Seed freshness, exact request
contracts and pilot/quality freezes are still required before execution.

H100 is the sole candidate. Require mean paired full-validation H100-minus-H200
CE below -0.02 nat/token, all three seed deltas negative, and all simultaneous
one-sided conditional-text upper bounds negative for blocks2048/4096/8192.
Use2000 common paired PCG64 resamples, frozen bootstrap seed and linear quantile
1-0.05/3, preserving tail counts. No raw-endpoint selection, cost-only route,
posthoc half-life grid, replacement seed or automatic promotion. Test scoring
remains closed. Report inference/cache checks and costs descriptively.

## Capacity and concrete integration work

The reviewed model has28,492,416 parameters,65,536 persistent-buffer bytes and
216,929,456 optimizer-tensor bytes. Dual-EMA raw plus both exports has a tensor
payload lower bound of786,974,384B. The raw payload alone is558,903,984B, exceeding
the old512MiB acceptance limit: a new explicitly bounded acceptance contract is
necessary. Do not loosen or bypass the closed block-normalization validator.

Preserving the previous conservative envelope and adding the extra average/export
gives ceil((32P + 3B + 2MiB)*1.15)=1,051,158,733B per trajectory. Four retained
triples plus256MiB metadata and the unchanged8GiB reserve require13,063,004,980B.
The guarded read-only audit observed13,361,377,280B free:298,372,300B headroom.
No cleanup credit is used. This narrow projected fit is not launch authority;
actual pilot memory/output must pass, and current free space must be rechecked.
Selected models, all old/new fresh outputs and failure evidence stay protected.

The existing immutable EMA implementation already accepts multiple half-lives.
Create an averaging-only contract/acceptance adapter outside all395 frozen source
files. Reuse unchanged training, EMA, full-validation arithmetic, guarded writer
and statistical primitives. Bind raw/H100/H200 identities explicitly; old
H200-only loaders must continue rejecting H100 rather than silently accepting it.

Verify that adding H100 leaves online weights, optimizer, RNG and H200 unchanged;
check the dual recurrence against a CPU reference, resume and both exports,
parent recomputation, and cache/reset behavior. Then census new seeds, freeze the
finite pilot, run it under600s/4GiB RSS/6GiB driver/8GiB reserve, and freeze the
quality panel only after its actual capacity and transport review. One local
worker, four threads, strict MPS, no paid compute or subagents. No new generic
ledger framework is needed. Preserve every failed attempt and original source.

Evidence: [guarded review](tokenizer-evidence/ema-horizon-preparation-001/review.json)
and its executable source/guard. Historical source documents are bound to their
recorded hashes at commit c337739; no historical campaign was rerun.

## Dual-EMA shadow prerequisite passed

The new averaging-only contract now binds one pilot and three fresh trajectories,
exact RMSNorm/SiLU configuration, ordered half-lives100/200, fixed LR0.00125,
final-only outputs and distinct raw/export size limits. The old395 numerical
source files remain byte-identical. A reserved screen seed is unused; this
comparison has no screening trajectory. No scientific seeds are reserved yet.

The guarded new prerequisite passed29 contract/recurrence checks plus1200 CPU
online/RNG invariance checks. The full41-snapshot H100/H200 recurrence agreed
with an independent float64 calculation within6.04463e-9; resume matched exactly.
On the actual28,492,416-parameter model with generated tokens,20 realized
optimizer updates showed that shadow updates preserve online parameters,
optimizer state, gradients and CPU/MPS RNG byte-for-byte. H200 matched the
single-average tracker at all8 snapshots; dual-state resume matched exactly.

The MPS phase took5.278seconds, with observed peak RSS616,644,608B and driver
2,319,253,504B. These are generated2x5-token prerequisites, not independent-run
reproducibility, training-quality results or batch8/context128 capacity pilots.
No corpus/test was read and no checkpoints were written or deleted. Original
source snapshots, manifest, logs, both guards and results are retained in
ema-horizon-preparation-001/shadow-prerequisite-001*.

Next implement and exercise actual raw/H100/H200 export and parent acceptance,
including oversized-raw rejection in old validators, exact new bounded loads,
full-validation identities and cache/reset checks. Do not add generic ledger
machinery. Only after these pass: reserve fresh seeds, freeze the pilot, obtain
real workload/resource evidence and freeze the final three-trajectory quality plan.


## Real pilot accepted; finite quality comparison frozen

The23 synthetic artifact-contract checks passed, including wrong-half-life,
changed tensors/buffers/config/parent, malformed EMA and cross-scope rejection.
A558,903,984-byte sparse zero probe was rejected by the unchanged512MiB loader
and read completely by the new640MiB bound; the exact manifested synthetic probe
was removed. It contained no model tensors and was not a research checkpoint.

The first prospective seed census conservatively rejected six whole-word textual
matches in historical resource logs. That failed census is preserved; these
matches do not prove prior use as training seeds. A new census scanned41,131
text files and reserved pilot187610001, fresh187610011–187610013,
evaluation187610003 and bootstrap187610004, with no matches. Screen187610002
is reserved but unused. No training used the rejected candidate range.

To stay within the complete retained-panel projection, the single predeclared
20-step batch8/context128 pilot also served as the first real checkpoint,
worker and independent-parent integration test. An extra full-size generated
fixture would have required another approximately787MB. This adjustment was
recorded before dispatch; quality stayed blocked until pilot acceptance.

Pilot001 passed. The unchanged trainer completed20 updates and eight EMA
snapshots. Raw resume preflight, exact H100/H200 exports, both independently
recomputed full-validation reports, and raw/H100/H200 full/cache/reset checks
passed. The worker finished in63.68seconds; parent acceptance took15.39seconds.
Observed worker peak RSS was1,769,930,752B and MPS driver2,518,482,944B;
parent peaks were1,513,930,752B and1,148,502,016B. All guards finished cleanly.
Raw559,083,061B plus two114,074,691B exports totals787,232,443B, below the
1,051,158,733B allowance. Every file remains retained.

Descriptive paired inference used two resident models, first96 training IDs,
32-token prefill/64-token teacher-forced decode, two warmups and eight alternating
repeats per batch. All four correctness checks passed. Median prefill/decode(ms):
H200 batch1 8.890/274.807, H100 8.050/276.972; H200 batch8 13.557/283.373,
H100 13.813/291.401. Cache remains196608B/item. These pilot timings and pilot
validation CE(H10010.345618/H20010.349039) are feasibility evidence, excluded
from quality selection. There is no cost-only adoption route.

The final quality freeze binds399 source files, including395 unchanged inherited
files, the accepted pilot and timing source, exact configurations and seeds.
Exactly three1200-update trajectories at LR0.00125 will produce raw/H100/H200
triples. No screen, replacement seeds, retries or expanded grid. For statistics,
arm order is explicitly H200 then H100 using unchanged paired arithmetic; all
previously stated effect-size, per-seed and simultaneous-text gates apply.
Each fresh pair also requires the same descriptive inference protocol.

At freeze12,519,411,712B was free versus12,275,772,537B required for the remaining
panel, including the8GiB reserve. Recheck at every dispatch; no cleanup credit.
The test holdout remains unscored. The inherited training loader materializes
test tokens; independent validation uses a train/validation-only view.

Evidence: pilot001-review.json, pilot001-evidence/, pilot001-source/,
pilot001-inference-*, and quality001-{plan,freeze,preflight,freeze-guard}.json.
Quality freeze SHA256: a58f1c2c7d8b1c10c28acb4d89c1ecc535c6f3e457d46052e75116c433b64bbf.
Next dispatch fresh slot0 once, then accept serially; never rerun an existing slot.


## EMA quality001: first fresh trajectory accepted

Seed187610011 completed1200updates and41dual-EMA snapshots. H100 full-validation
CE is4.715769963489097 versus H2004.726289522424742, delta-0.010519558935645
nat/token. This is one of three fixed fresh trajectories and is insufficient
for adoption; the required final mean improvement exceeds0.02, with everyseed
and all simultaneous conditional-text bounds also negative. Continue the frozen
panel without interim selection, early stopping or replacement seeds.

Independent raw-resume/export audits and both full-validation recomputations
passed. Worker elapsed292seconds, peak RSS1.827GB and MPS driver2.535GB;
parent acceptance15.24seconds. All guards passed. Paired inference full/cache/reset
checks passed at batches1/8. Median prefill/decode milliseconds were H200
8.212/285.844 versus H1008.839/276.767 at batch1; H20013.918/288.110 versus
H10013.781/291.969 at batch8. Timing is descriptive; cache196608B/item remains
identical. All raw/H100/H200 files are retained, no test scoring or deletion.
Evidence: ema-horizon-preparation-001/quality001-slot00-{review,inference-result,
review-guard}.json and portable slot00-evidence/. Next is frozen quality slot1.


## EMA quality001: second fresh trajectory accepted

Seed187610012 completed1200updates and41EMA snapshots. H100 full-validation CE
is4.6866778909108255 versus H2004.691822250918006, delta-0.005144360007180
nat/token. Both completed seeds favor H100; their descriptive mean delta is
-0.007831959471412. This partial panel does not satisfy or replace the final
three-seed adoption decision. Continue fixed seed187610013 without early stopping.

Independent parent validation, raw/export acceptance and fresh inference checks
passed. Worker peak RSS1.806GB, driver2.527GB; all guards finished. Median
prefill/decode(ms) at batch1: H2008.250/280.984, H1008.924/283.620; batch8:
H20013.882/306.252, H10013.992/287.625. Timing remains descriptive and mixed;
cache196608B/item is unchanged. All three new checkpoint files remain protected.
No test scoring, deletion, architecture or hyperparameter changes occurred.
Evidence: ema-horizon-preparation-001/quality001-slot01-review.json,
slot01-inference-result.json, slot01-review-guard.json and slot01-evidence/.
