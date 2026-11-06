# Body allocation: correctness and prospective structural comparison

The human authorizes model correctness checks, followed by the structural
comparison when correctness and its complete resource budget pass, and continued
evidence-grounded improvement until a human pause or stop. This does not relax
the frozen scientific, resource, holdout or retention contracts.

The incumbent remains learned final LayerNorm scale/bias and AdamW epsilon1e-6.
The complete final-scale comparison closed without adoption; none of its commands,
seeds, terminal STOPs or protected fresh weights are reopened.

## Evidence and question

The [boundary review](tokenizer-evidence/exploration-boundary-001/evidence-review.md)
and [geometry audit](tokenizer-evidence/exploration-boundary-001/geometry-review.md)
separate measured local facts, published results and inference. Vocabulary tables
are90.31% of current parameters, but this is not a FLOP fraction. Existing curves
still improve through1200 updates; they do not establish overfitting. Historical
character depth/MoE gains weakened at width256 and cost more. Those negatives
remain relevant but do not answer this different subword-MLA allocation question.

Kaplan2001.08361v1 sections3.1/4 motivates non-embedding matching; Hoffmann
2203.15556v1 sections3.1/3.2 motivates an explicit token/compute axis. Bordelon
2309.16620v2 section2/equation1/table1 and Yang2310.02244v1 table1/sections9.1/9.2
warn against assuming depth/optimizer transfer. The exact primary links and local
applicability gaps are in the evidence notes. No large-data law, vision result or
arbitrary residual multiplier is treated as a local quality prediction.

Hypothesis: reallocating almost matched body capacity from wide FFNs into more
serial layers can improve composition on the current text. The deeper candidate
must beat both the current incumbent and a vocabulary-width-matched shallow
control. Failure against either contrast counts against this candidate. A shallow
control win is diagnostic; it does not become a post-hoc fallback adoption.

| Role | Layers × width, FFN | Parameters | Muon/AdamW tensors | Cache B/item |
| --- | --- | ---: | ---: | ---: |
| Incumbent | 4×256,672 | 28,492,416 | 44/44 | 196,608 |
| Shallow control | 4×192,1024 | 22,057,472 | 44/44 | 163,840 |
| Deeper candidate | 7×192,504 | 22,029,632 | 77/74 | 286,720 |

The two192-wide arms differ about1.01% in body parameters and1.49% in forward
matmul proxy. This is an allocation comparison, not a pure-depth intervention
or a demonstration of equal measured training compute. Against the incumbent,
table width, head count and latent width also change. All keep serial unit
branches, learned final norm, MLA score.5, SiLU/SwiGLU, adopted optimizer/decay,
context128/batch8 and the original1200-update cooldown800/H200 recipe.

## Actual prerequisite evidence

The new isolated `research/body_allocation_model_v1` does not alter old guards
or any of918 frozen predecessors. CPU and MPS
[model checks](tokenizer-evidence/body-allocation-preparation-001/modelchecks001-mps.json)
pass all three geometries, independent equations/gradients, causal suffix/gradient
tests, full/cache/reset batches1/8 and atomic foreign cache/checkpoint rejection.
The incumbent also reproduces exact initialization, RNG, logits, CE, gradients,
optimizer groups and two updates. Core source copies are retained with each check.

Generated [resume001](tokenizer-evidence/body-allocation-preparation-001/resume001-plan.json)
passed all three arms: exact fresh repeat and serialized resume of every model,
optimizer/EMA tensor byte, scalar, clipping entry, private sampler and CPU/MPS RNG.
H200 export/cache checks and corrupt group/history/geometry rejection passed.
Compact SHA identities do not establish comparative memory savings. Fixture
seeds91161–91163 are not scientific seeds. Its nine generated payloads remain
retained until downstream dependencies close and a new exact retirement audit.

Real-data worker integration is in the separate
`research/body_allocation_worker_v1` namespace. It tokenizes only the original
train/validation raw slices, verifies corpus/tokenizer/ID hashes and uses the
original paired512-block sampler. Independent acceptance checks complete RAW,
optimizer/EMA/RNG/sampler/clipping/H200 state, all34,278 validation targets,
CPU/MPS CE agreement and full/cache/reset at context128 and batches1/8.
The four-update [worker001 fixture](tokenizer-evidence/body-allocation-preparation-001/worker001-plan.json)
passed all three arms; CPU/MPS full-validation CE differences are below2.8e-7.
Its [integration](tokenizer-evidence/body-allocation-preparation-001/integration001.json)
rechecked actual transport, rejected18 forgeries and validated18 future configs.
No scientific quality result is assigned by these fixtures.

The three [resource pilots](tokenizer-evidence/body-allocation-preparation-001/pilot001-review.json)
passed20updates/20,480targets/8H200snapshots each, with independent full-validation
and cache checks. Conservative1200-update projections are1103.42/681.26/736.10s.
The complete initial high-water budget16,100,421,632B passed. Fifteen obsolete
generated correctness payloads were retired after exact closure/audit, with46,801
survivors unchanged. They must never be restored or rerun. Additional verified
cache recovery001/003 recovered96,083,968 allocatedB. Recovery002 failed a bottle
binary-identity eligibility check before manifest/deletion; that bottle remains
intact. The extra2GiB target remains unmet after eligible recovery, while the
mandatory remaining budget passes. No new scientific quality trajectory yet.

## Prospective design, not yet frozen

Worker contracts and all three measured20-update resource pilots passed. The
[finite scientific contracts](tokenizer-evidence/body-allocation-preparation-001/quality-contracts001.json)
also pass independent nonconstant statistics/selection/resource-credit tests. Compare the
three exact arms using the same finite LR allowance {.000625,.00125,.0025} and
one prospective screening seed. Each arm selects its minimum full-validation
H200 CE; ties choose lower LR. Then freeze those choices for three NEW shared
fresh seeds with rotated arm order. All trajectories have1200updates/1,228,800
targets/41H200 snapshots. No rescue, extra screen, replacement seed or failed
attempt retry. Same scalar seed and initialization scheme do not imply shared
weights across different shapes; deeper construction also shifts the readout RNG
stream. This comparison evaluates equally tuned recipes, not ideal depth scaling.

Only deeper is an adoption candidate. For BOTH deeper-minus-control contrasts,
require mean<−.02 nat/token, every fresh seed negative, and negative conditional
text upper bounds for blocks2048/4096/8192. Freeze simultaneous multiplicity
accounting for two contrasts × three block sizes,2000 common PCG64 draws and
NumPy linear quantiles before screening. These intervals condition on trained
seeds and reused text; no seed-population or holdout confidence claim. Report
training time, clipping, serving latency/cache and weight cost separately; no
cost-only or interim-result adoption. A separate independent review is required.

Full resource accounting must include all three pilots, all nine LR trajectories,
three selected endpoints plus nine fresh endpoints, serialization overlap and
the complete256MiB metadata envelope and8GiB reserve. Phase-boundary retirement
may remove only dependency-closed pilot/losing screen payloads after exact audits;
no credit exists until actual retirement. Selected and all fresh weights remain
protected. Measure the complete live remaining budget before every unstarted
slot, target another2GiB in cleanup, and recover inline on a new blocker.

The strengthened scientific full high-water is16,167,530,496B, including another
64MiB for prospective retirement shards and portable sidecars beyond the
pilot forecast. Cleanup target18,315,014,144B. Live remaining preflight
14,747,676,750B passed against15,090,638,848B free, after crediting only the
three actual accepted pilot pairs1,419,853,746B once. No future retirement credit.

Census001 reviewed55,707 recorded files. Scientific seeds are pilot200080001
(used), screen200080002 (unused), evaluation200080003 (used), bootstrap200080004
(unused), fresh200080011/012/013 (unused). No old final-scale seed reuse.

Next: freeze the complete finite scientific/retention/resource contract,
retire only six dependency-closed pilot payloads under the exact audit, then
execute the authorized nine-trajectory LR screen and nine fresh trajectories
serially with live complete remaining-budget checks. The controller stops on
a scientific/resource failure without retries or replacement seeds; source and
independent useful work continue after diagnosis. No automatic beta1 study follows; choose the next improvement
from results, existing data and primary literature.


## Scientific execution launched

[quality001-freeze.json](tokenizer-evidence/body-allocation-preparation-001/quality001-freeze.json)
SHA256 `de119b0483f59ab43fadce3172b553d48960f77cd4710e5a07a39dfc541cb591`
freezes the complete original design. Run root
`research_runs/tokenizer-body-allocation-quality-v1-001`, controller
`research.body_allocation_quality_v1.execution run-all`. The first screen
incumbent/LR.000625/seed200080002 has actual completed updates. No accepted
scientific endpoint or quality/adoption conclusion yet at this launch record.
The controller continues the finite screen, selection, exact losing-screen
retirement, fresh panel and independent inference/statistics serially.

Six pilot payloads1,419,853,746logical/1,476,706,304allocatedB were permanently
retired after actual acceptance and finite downstream source closure;46,983
survivors rehashed unchanged. See quality001-pilot-retirement001/manifest.json
and completed.json. Original pilot STOP/source/log/config/metric/receipts stay
immutable; never rerun or restore those pilots. Post-retirement free16,545,169,408B
passed the strengthened16,167,530,496B floor. Extra2GiB target remains unmet
after eligible cleanup; the full mandatory budget and original guards remain.
All selected/fresh historical weights stay protected. Screen/evaluation/pilot
seeds are used; fresh200080011/012/013 andbootstrap200080004 remain unused.


## Six screens accepted; capacity recovered before screen seven

Slots00–05 passed independent full acceptance. Original controller exited
cleanly before unattempted06 because actual free13,315,923,968B was below
remaining13,327,286,044B. No STOP, trial failure, repeated update or relaxed gate.
External recovery004/005 plus new lossless storage038 reclaimed63,270,912
allocatedB;237 closed metadata files retain original paths/readable bytes,
24 protected payload hashes unchanged. No weight deleted or app stopped.
Actual free13,386,407,936B passed the unchanged remaining gate, and the same
freeze continued at previously unstarted06(LR.0025 incumbent). Three LR screens
and all nine fresh endpoints remain. No selection/adoption yet. Extra2GiB
headroom remains unmet after currently eligible recovery classes were exhausted;
new eligible evidence/blockers still trigger inline recovery. Root-owned npm
parent directories require administrator access; unused Keynote/admin and all
selected/fresh/pending updater/shared-runtime protections remain. See
quality001-prefix6-wake001.json and quality001-prefix6-metadata-mirror001.json.


## LR screen complete; fresh confirmation active

All9screen endpoints passed full independent acceptance. The frozen minimum
H200CE/lowerLRtie rule selects.00125 for each model, retaining03/04/05.
The original18-trajectory scientific design is unchanged: its fresh stage
is exactly9endpoints,three shared seeds with rotated order, now7/9accepted
and slot16active at this record. No interim adoption or complete-panel result.
All939bindings unchanged;fresh seeds all used,bootstrap200080004unused.
The exact12losing screen payloads2,840,244,452logical/2,954,346,496allocatedB
were permanently retired after immutable selection,47,266survivors unchanged.
Selected recipes,every fresh endpoint and original source/log/config/metric/receipt
remain protected; never restore/rerun retired members or closed cleanup.
See quality001-selection.json,quality001-confirmation-plan.json,
quality001-loser-retirement001/completed.json and quality001-prefix16-metadata-mirror001.json.
