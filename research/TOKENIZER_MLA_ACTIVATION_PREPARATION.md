# Tokenizer gate-activation preparation

Status, October 31, 2026: implementation prerequisites in progress; no training
launched and no finite launch plan frozen. Campaign005 remains closed, with the
hybrid not adopted. Existing selected models, every fresh pair and failures remain
protected. The earlier MoE draft is retained as a historical proposal. The October 28
GQA activation draft in TOKENIZER_ACTIVATION_PREPARATION.md also remains unchanged;
this separate MLA proposal follows completion of the later attention studies.

## Why this family now

The prospective three-arm MoE study needs **18,782,075,912 free bytes** under the
existing allowance method, including the 8 GiB reserve. Only 15,515,467,776 bytes
were free at the audit: a **3,266,608,136-byte gap**. Even the calculated final
raw/H200 tensor payload plus reserve is 15,562,137,728 bytes before headers,
RNG, logs or extra overlap. This assumes all experts acquire optimizer state;
idle experts must not be used to justify capacity. No pilot was attempted.

Defer MoE without dropping its wider-dense control, reducing fresh confirmation,
or deleting prior protected evidence. Select the already proposed same-shape
activation fallback instead. This explores another requested architecture family
while retaining the adopted width256 RoPE MLA backbone and a complete paired
confirmation panel. MoE remains queued for sufficient safely available capacity.

The two-arm activation projection requires **15,169,995,168 bytes**, leaving
345,472,608 bytes at that observation. This is a narrow, time-dependent margin,
not launch authorization. It includes two selected screening pairs, all six
fresh raw/H200 pairs, conservative optimizer/EMA/export overlap, per-trajectory
allowances, 256 MiB campaign metadata and the unchanged 8 GiB reserve. Pilot and
losing-screen cleanup requires complete audits and durable boundary receipts.
Current free space must be checked again before freeze and every dispatch.

Evidence: [storage projection](tokenizer-evidence/moe-preparation-001/ffn-storage-projection.json)
and [independent meta-optimizer crosscheck](tokenizer-evidence/moe-preparation-001/storage-crosscheck.json).
These are shape calculations, not observed training memory, time or quality.
Existing protected files already consume disk; observed free space takes no
credit for their deletion. Source references in the projection bind the
pre-implementation baseline, not a promise that later source remains identical.

## Hypothesis and proposed finite comparison

Compare ordinary SwiGLU to an exact GELU gate on the adopted width256,
four-layer, quarter-latent RoPE MLA backbone. Both use three bias-free FFN
matrices with hidden width 672, identical initialization draws, untied GPT-2
tables, block RMSNorm, final affine LayerNorm and attention score scale 0.5.
For normalized block input x:

- SwiGLU: W_down [SiLU(W_gate x) * (W_up x)].
- GELU gate: W_down [GELU_exact(W_gate x) * (W_up x)], with
  GELU_exact(z) = z [1 + erf(z / sqrt(2))] / 2.

Parameter counts and shapes match (28,492,416 parameters each), but functions,
gradients and optimization paths differ. No FFN product normalization, extra
gain, bias, approximate GELU or ReLU arm is permitted by the new contract.

The historical character study did not adopt GEGLU: fresh H200 deltas were
-0.000404 at width64 and +0.006786 at width256. ReGLU was worse at both widths.
This motivates a narrow GELU-only comparison under the changed subword/MLA
condition, not a positive forecast or a reopening of that study. Do not pool
character and subword metrics. See [historical results](GATED_ACTIVATION_RESULTS.md).

Proposed finite design: two 20-update resource pilots; three equal peak-LR
choices (0.000625, 0.00125, 0.0025) per arm on one screening seed; select each
arm's lowest H200 CE with lower-LR tie break; then both selected recipes on
three fresh seeds with alternating arm order. Quality trajectories use 1,200
updates, cooldown from 800, H200 start 800, batch8 and context128. Exact seeds,
requests, source/runtime/data identities and limits remain to be frozen.

Proposed adoption: GELU must improve mean H200 CE by more than 0.02 versus the
equally tuned fresh SwiGLU control, improve every fresh seed, and have negative
one-sided conditional-text upper bounds at blocks 2048/4096/8192 using 2,000
common paired resamples. Freeze RNG and quantile conventions and their tests
before results. Report raw endpoints, all seed deltas, actual training and
inference costs. No posthoc grid extension or quality-neutral cost route.
Validation is adaptively reused; the test holdout remains closed.

## Completed implementation prerequisite

`tokenizer_mla_gate_v1` validates exactly the width256 MLA backbone plus explicit
gate kind/policy. The existing architecture and gate validators delegate only
for that named contract; old policies retain their exact field sets and errors.
Checkpoint signatures distinguish gate semantics despite identical shapes.

154 targeted tests passed under a 120-second CPU prerequisite guard, with
558,956,544-byte peak RSS and the 8 GiB reserve intact. Checks cover exact
SwiGLU incumbent weight/logit/optimizer parity; full parameter coverage/counts;
independent exact-GELU value/input/parameter gradients; full versus cached logits;
causality; same-shape checkpoint mismatch; mutation and unsupported-combination
rejections; and historical gated-activation contract regressions. No optimizer
step, training pilot, quality evaluation or checkpoint file was performed by
these new tests. This is not a complete numerical or training-readiness claim.

Evidence: [test review](tokenizer-evidence/activation-preparation-001/contract-tests-v1-review.json),
[log](tokenizer-evidence/activation-preparation-001/contract-tests-v1.log),
[guard](tokenizer-evidence/activation-preparation-001/contract-tests-v1-guard.json).

## Next prerequisites, before launch

1. New-contract optimizer/RNG/EMA resume, H200 export/load, tokenizer binding,
   objective separation and cache isolation passed the RAM-serialization tests
   recorded below. Filesystem-producer integration remains required; preserve
   both harness failures and do not retry the closed hybrid numerical fixture.
2. Finite two-arm request/selection/statistical rules and seed registry passed
   22 tests (details below). Implement/test bounded artifact manifests, actual
   review/cleanup receipts and the retention-aware serial controller next.
   No launch through old hybrid controllers.
3. Reconcile the implemented model/optimizer/serialization resource projection,
   source/runtime/input pins and all prerequisites. Account for any test artifacts
   and metadata growth. Freeze the full plan before guarded MPS pilots; require
   accepted pilot resource evidence before quality runs.
4. Keep one worker, four CPU threads, strict MPS for training, all existing
   memory/time/storage guards, no paid compute, no subagents and closed holdout.
   If current storage fails, record the exact gap and safe options; never start
   a partial confirmation panel or remove protected pairs.


## Serialized state prerequisite completed October 31

Four new bounded integration tests passed for SiLU and exact GELU. Four-step
synthetic CPU updates with random token IDs produced bit-exact resumed online
weights, optimizer states, EMA and global/loader RNG relative to uninterrupted
execution. Production H200 derivation/export/load accepted the new contract,
retained the raw byte hash and RNG, matched EMA parameters, rejected optimizer
resume from averaged weights and rejected duplicate export. Production tokenizer
loading accepted the local corpus/tokenizer binding and rejected a changed
identity. Reset/interleaved caches and CE-only objective checks passed. No corpus
split was sampled or scored.

Real torch checkpoint serialization was held in RAM through an IO adapter; no
checkpoint file was created or deleted. This preserves capacity but does not
test filesystem publication. The successful 180-second/4 GiB/8 GiB-reserve guard
finished in 17.48 seconds with peak RSS 3,352,215,552 bytes.

Two harness failures remain preserved with their exact source, logs and guards:
v1 incorrectly requested logit-z and stopped before optimizer updates; v2 used
an uncanonicalized /tmp key for releasing a RAM buffer, causing two failures
after resume loading while both cache/objective cases passed. Production source
was unchanged across these attempts. The corrected v3 used the CE-only API and
canonical paths. These are new activation unit fixtures, not retries of the
closed failed hybrid numerical fixture or of an experimental trajectory.

The runtime and checkpoint, EMA, optimizer, schedule, loader, MLA/cache and
filesystem-writer source hashes match the completed005 baseline. Only the three
activation-contract files differ. This supports review of unchanged primitives,
not automatic reuse of hybrid numerical cases or authorization to launch an
activation campaign. [State prerequisite review](tokenizer-evidence/activation-preparation-001/state-prerequisite-review.json).

Next: implement/test the finite two-arm request and selection rules, exact fresh
seed registry and conditional-text statistics, then retention-aware serial
controller and filesystem-producer integration. Bind the state tests, review
source applicability and reconcile exact serialization/metadata capacity before
freezing all prerequisites and launching guarded MPS pilots. Retain all earlier
protected pairs, archives and failures. The plan is still unfrozen.


## Finite rules and requests validated October 31

The draft now fixes pilot seed3591, screening seed3601 and fresh seeds
3611/3612/3613. A recursive audit of seed-named fields in3,024 recorded
plan/request files found no reuse; its complete hashed inventory is preserved.
This is local nonreuse evidence, not population independence.

Two20-update pilots use LR0.0025, cooldown/EMA start13 and EMA interval1
(eight H200 snapshots). Six full screens use the three declared rates for both
arms, reversing arm order at the middle rate. Independent minimum H200 CE
selects each arm's LR, with lower-LR ties. Six full fresh runs alternate arm
order across the three seeds. The complete finite design has14 trajectories.
All nine conditional LR-selection branches are explicitly saved.

The single GELU-minus-SiLU contrast uses2,000 PCG64 resamples, RNG seed20261031
reset separately for each block size. Common text draws apply to all paired
models/seeds; paired seed loss sums are averaged before resampling and divided
by sampled target counts, including the tail. Bounds use sorted linear
interpolation and quantile1-0.05/3 for the three declared block sizes. Adoption
requires mean delta<-0.02, every fresh seed<0 and every upper bound<0.
The helper returns arithmetic eligibility only, never adopted=True.

22 tests passed, including independent loop-oracle weighted-tail bootstrap
comparison, strict threshold boundaries, invalid data/panel rejection, exact
request geometry/schedules, full ordered progression and persisted selection/
cleanup barriers. Missing, duplicated, reordered, failed or extra trials and
raw scores masquerading as H200 are rejected. Both selected screen pairs and
every fresh pair remain protected; only four losing screens are candidates
after complete audits. No deletion or training was performed.

[Finite design](tokenizer-evidence/activation-preparation-001/finite-design-v1.json),
[tests](tokenizer-evidence/activation-preparation-001/rules-tests-v1-review.json),
[seed audit](tokenizer-evidence/activation-preparation-001/seed-audit.json),
[source applicability](tokenizer-evidence/activation-preparation-001/rules-applicability-review.json).
Only two pure research modules were added since model/state validation.
These rules are not a production dispatcher or an executable freeze. Next
implement/test the bounded serial controller and filesystem producer, actual
review/retention receipts and complete resource/source/input/runtime binding.
Recheck full capacity before freezing and launching pilots.
