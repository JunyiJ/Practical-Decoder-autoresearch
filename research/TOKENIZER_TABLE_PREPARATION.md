# Tokenizer embedding/readout sharing preparation

October27,2026. Hybrid001 is fully audited and not adopted. The next structural
hypothesis is sharing GPT-2 input/output vocabulary tables on the retained GQA
backbone. This is preparation, not a frozen quality protocol or permission to
launch training without resources. MoE, activations, normalization, residual,
depth and standalone HP directions remain active.

## Motivation and controls

A50,257-token vocabulary makes tables a major part of this small model. Sharing
could reduce parameters, optimizer/EMA storage and inference parameter memory,
but it combines lookup/output gradients and changes expressivity. Historical
character results did not justify adoption; the enlarged BPE vocabulary is a
new context, not evidence of a gain or a reason to reopen the old controller.
Use the retained width256 GQA1200-update recipe as a source of architecture and
schedule settings; actual quality widths and finite panel await storage checks.

Prepare three explicit kinds: legacy untied, matched untied, and scaled tied.
Preserve all shared initialization draws. The matched control copies the input
table into a separately registered output matrix. The tied arm retains one
canonical input table, with no aliased/separately registered lm_head parameter.
Both matched/tied use readout multiplier1/sqrt(3*64), followed by inherited
1/width_mult; legacy uses its original head initialization and multiplier.
This matches the inherited readout variance scale, not the full distribution
or role-dependent optimization. Matched versus legacy changes initialization,
correlation and coordinates; tied versus matched combines the two gradients
and AdamW histories. Do not attribute all effects to sharing alone.

Keep RoPE GQA head32/half KV heads, dense SwiGLU21D/8, four layers, RMSNorm
blocks/final affine LayerNorm, unit input/residual scales, no output bias,
context128 and the exact BPE/raw-split identity. Width64/256 CPU contracts;
no width transfer claim. Add a distinct tokenizer table-sharing policy rather
than loosening old character or other tokenizer model contracts. Reject
independent scale/init overrides and incompatible checkpoint/table layouts.

## Required validation before training

Check exact legacy equivalence and initialization RNG, matched/tied logits,
FP64 gradient-role sum, causal/cache outputs, canonical parameter registration,
one AdamW table group/state/decay, correct hidden Muon width rules, optimizer
state resume and H200 export/load. Reject mutable configs and incompatible
raw/EMA/checkpoint contracts. Run bounded CPU checks first, then source-frozen
serial MPS pilots with measured checkpoint sizes. Preserve failed attempts.
Freeze fair tuning, fresh seeds and quality/cost criteria before quality work.
No new adoption rule has been chosen yet.

## Storage prerequisite

Hybrid001 left about8.26GiB free, only0.26GiB above the mandatory8GiB reserve.
All prior research checkpoints must remain intact. This cannot fund a new
multi-seed quality comparison. The user has been asked to free roughly5GiB or
provide an external checkpoint location. Continue CPU preparation; do not lower
the reserve, delete research evidence, or treat silence as storage authorization.
The final width/panel and remaining-trial projection will be frozen only after
actual available storage and measured pilot costs are known.

## Implementation and CPU review

The distinct `tokenizer_gqa_scaled_table_sharing_v1` contract now supports the
three declared arms at widths 64 and 256. It delegates the unchanged GQA fields
to the existing exact architecture validator and binds table policy/kind into
model and checkpoint signatures. Historical character contracts remain separate.

The corrected targeted suite passed all 11 checks, including exact three-update
CPU resume, sampler RNG and H200 export/load with the GPT-2 binding. Width 64/256
checks cover preserved initialization draws and legacy weights, matched/tied
logits, the FP64 sum of lookup/readout gradients, a single canonical table and
AdamW state, analytical first-update decay, causal/cache outputs and rejection
of incompatible metadata. Sharing removes exactly 50,257 times width parameters;
this is a parameter count, not a measured quality or latency benefit.

The initial suite had three fixture failures: two calls supplied only the model
config to a cache factory requiring the full config, and the resume fixture used
batch size 2 despite the inherited paired-block layout requiring batch size 8.
Both were corrected; the original failure log is retained. Existing attention,
MLA, hybrid and character table geometry regressions passed 76 checks. Three
existing disk-heavy resume tests were deliberately deselected; the new BPE table
resume test ran successfully in the targeted suite. A final pass after adding
namespace-config compatibility passed 10 checks and skipped the same table resume
fixture because available space had dropped below its prospective 190 MiB
allowance above the 8 GiB reserve. The earlier successful resume evidence is
retained; the skip is not counted as another pass. Only successful disposable
CPU test checkpoints were removed, never research checkpoints.

Logs and source hashes are in `tokenizer-evidence/table-preparation-001/`.
No MPS pilot or quality experiment has been launched. Remaining prerequisites
are a storage-feasible frozen pilot, its raw/H200 artifact review and measured
resource costs, followed by a finite equally tuned quality/cost protocol.

## Storage recheck and panel accounting (October27,16:03–16:08UTC)

An observed increase to11,221,049,344 free bytes (10.45GiB) allowed a deliberate
rerun of the previously skipped CPU fixture. The complete suite passed11 checks
in6.71s, including exact resume and H200 load. This is new CPU evidence only;
the earlier failures and skip remain recorded in their original logs. Successful
disposable CPU fixtures were removed by the test as designed. No research
checkpoint was removed and no MPS worker was started.

The free-space increase was transient. During the subsequent prospective
accounting, available space was8,738,013,184 bytes (8.14GiB). Do not dispatch from
an earlier free-space observation. The pending storage request remains open;
no human response or external destination has arrived.

`storage_projection.py` and `storage-projection.json` record an explicit possible
panel: three retained pilots, two LR screening trials per arm and three fresh
confirmation trials per arm (15 quality trials). This is storage accounting,
not a frozen quality plan: rates, seeds, adoption criteria and pilot validation
must still be declared before training. The tensor bound charges five FP32
copies of every parameter for raw plus H200 storage (model, EMA, two optimizer
states and exported model), conservatively charging two states even to Muon
matrices. Add15% plus2MiB per trial and4MiB for round metadata. It assumes one
raw/H200 pair per trial, one EMA half-life and no periodic/best checkpoints.

| Width | Untied parameters | Tied parameters | Pilot + quality output allowance | Required free bytes including8GiB reserve |
|---|---:|---:|---:|---:|
|64|6,612,480|3,396,032|2,335,639,936|10,925,574,528 (10.18GiB)|
|256|28,588,032|15,722,240|10,101,908,992|18,691,843,584 (17.41GiB)|

Neither panel fits the later observation. The earlier request for roughly5GiB
was a general estimate, not a promise that a full width256 panel would fit.
Width64 would be a scoped table-sharing investigation, with no transfer to
width256 and no reversal of the closed width64 GQA-versus-MHA conclusion.
Do not silently narrow the panel, omit controls, delete artifacts or lower the
reserve to make it fit. Once resources permit, choose and freeze the scope;
then validate the estimate against measured pilot files and recheck the entire
remaining allocation before each serial launch. Broad architecture and standalone
HP candidates remain available under the same evidence and resource requirements.

## Prospective comparison design (October27,19:49UTC)

This design resolves statistical and control choices while storage is blocked.
It is not an executable/source-frozen protocol. Choose one width, validate all
three pilots, and freeze the runner, exact requests and remaining allocation
before quality dispatch. Changes to this design must be recorded before observing
new quality results, rather than selected after seeing a favorable comparison.

Table sharing remains the next prepared hypothesis because vocabulary matrices
dominate these models and its correctness checks are complete. The equal LR
screen is a first-class optimization question within this structural comparison.
Standalone schedule/regularization studies remain valid, but would not answer
whether the enlarged vocabulary makes sharing useful. MoE is deferred pending
its routed/total-capacity accounting and tokenizer contract validation; activation,
normalization, residual and depth comparisons remain untested alternatives. This
choice does not establish a rigid queue or reopen a closed attention experiment.

Prefer width256 for direct relevance to the retained GQA serving recipe. Width64
may be selected prospectively if only its complete panel fits, with an explicit
scope decision before any pilot. Never select a width from new pilot CE. The
width64 case uses the same 1200-update schedule as a common experimental setting;
that horizon has not been independently adopted at width64. Do not pool widths.

Use all three arms, independently tuned over peak LR0.00125/0.0025. Screen each
arm/rate once at seed3301 (six trajectories); select the lower full-validation
H200 CE separately for each arm, breaking an exact tie with the lower LR. Freeze
all three choices before nine fresh confirmation trajectories: each arm at its
selected rate and seeds3311/3312/3313. Thus a favorable screen never eliminates
either control, and confirmation cannot reselect LR. Use cyclic arm order across
confirmation seeds; declare the exact six-screen order in the executable plan.
These seed IDs had no matches in saved request/plan/recipe seed fields at this
review; check the campaign inventory again at freeze time.

Keep1200 updates, batch8, context128, paired512 sampling, cooldown800–1200 to0.1,
H200 start800/interval10, and the inherited optimizer/decay configuration. The
comparison matches target-token exposure, not total parameters or elapsed time.
Full-validation H200 CE is the only quality selector; raw CE, BPB, samples,
training time and sampled-validation curves are diagnostics. Keep test loss closed.

For each fresh-seed pair define delta as candidate minus control H200 CE. The
strict quality rule is mean delta below−0.02, every seed below0, and upper95%
conditional-text bootstrap bounds below0 at blocks2048/4096/8192 (2000 draws,
seed20261024, resampling paired text blocks jointly across the three seeds).
The neutrality rule instead requires the mean, every seed, and every corresponding
upper bound to be at most+0.01. This margin is an agent-chosen prospective
engineering convention for this table experiment, not a human numeric tolerance.
These are conditional engineering gates under adaptive validation reuse, not
seed-population confidence or a multiple-comparison-adjusted significance claim.

The legacy arm is the deployment reference; matched untied is a required
mechanistic control. A tied quality-improvement claim requires the strict rule
against both controls. A tied memory/quality tradeoff requires neutrality against
both controls and at least30% smaller measured unique resident FP32 parameter
storage against each, at the selected width. Measure storage from loaded H200
tensors, deduplicating actual storage; theoretical counts alone do not pass the
cost gate. This is a parameter-memory claim, not a reduction in KV cache, whole
process RSS, peak working memory or cloud cost. Report those separately. A matched
untied recipe may qualify only through the strict rule against legacy, since it
has no prospective parameter-memory advantage. If both recipes qualify, retain
their distinct scoped tradeoffs rather than declaring a universal winner.

Before any adoption, measure all three fresh-seed model triples serially on the
same FP32 MPS context128 workload: batch1/8, first96 training tokens,32-token
prefill and64 teacher-forced decode steps. Use two warmups and eight synchronized
rotating-order measurements per seed and batch. Check full/cached equivalence
first. Report per-seed and median prefill/decode, actual parameter/cache bytes,
and working-memory observations; no speed gate is imposed on the memory route.
Do not infer achievable maximum batch size from these two measured batches.

Three separate20-update pilots at seed3291/LR0.00125 will check correctness and
resource costs only, retaining raw/H200 pairs with a short EMA tail declared in
their frozen plan. They cannot rank arms or select width/rate. Their measured
bytes must validate or increase the existing conservative allowances. The pilot
and quality runners still require explicit600s/7200s timeouts, RSS4GiB/MPS-driver
6GiB sampled guards, four CPU threads, no fallback, one worker, immutable source
and data identities, exclusive trial directories, and fail-stop behavior.
Preserve all failed artifacts; no automatic retries or replacement seeds.

Outstanding work is the storage-feasible scope decision, pilot runner and source
freeze, reviewed MPS correctness/resource evidence, and executable quality/timing
plans with tested selection and guard logic. No training was launched by this
design review, and no new model-quality evidence or adoption is claimed.

## Authorized retention change (October27, Los Angeles)

The human authorized deletion of non-chosen checkpoints. The reviewed cleanup
removed 30 files from 15 losing tokenizer screening configurations (3.444 GiB),
while keeping selected screens, all fresh confirmation pairs, pilot/failure
evidence and historical campaigns. The [cleanup manifest and review](tokenizer-evidence/checkpoint-cleanup-001/README.md)
record hashes and the resulting limits on reloading deleted weights. Earlier
blanket-retention statements describe the policy before this authorization.

Free space was 18.36 GiB after cleanup, with additional filesystem changes
beyond the deleted bytes. This observation supports preparing the preferred
width256 panel; it does not freeze or launch it. Recheck actual free space and
all remaining allocations before every launch; retain the 8 GiB reserve.

## Character-checkpoint cleanup (October27, Los Angeles)

The human subsequently requested deleting all character-model checkpoints.
The [inventory and deletion review](tokenizer-evidence/character-checkpoint-cleanup-001/README.md)
records 3,232 removed files (23.346 GiB), with all 126 remaining GPT-2 checkpoints
retained and hash-verified. Numeric historical evidence, logs, configs, failure
records and stop markers remain. Character checkpoint reloading is no longer
available; those prior studies stay closed. Free storage was 41.69 GiB after
cleanup. Proceed with preferred width256 table-pilot and finite-plan preparation,
rechecking actual storage and the unchanged reserve before every launch.

## Width256 MPS pilot completion (October28,02:11UTC)

After the authorized checkpoint cleanups, all three source-frozen20-update
width256 pilots completed under source30a309a and plan9fc40ed. Raw and H200
loads passed tokenizer/geometry checks and full/cached equivalence at batches1/8.
H200 used three pilot snapshots. Each untied checkpoint pair measured560,602,700
bytes; tied measured303,284,813. Both remain below their original conservative
allowances. Actual unique FP32 parameter storage was114,352,128 bytes untied
and62,888,960 tied (45.004% lower). This is parameter storage only; the GQA cache
was unchanged. Pilot process peak RSS was at most1,515,585,536 bytes.

Evidence: `tokenizer-evidence/table-pilot-001/review.json`, frozen plan, requests,
logs and results. No pilot quality ranking or adoption is made. The following
finite quality round retains all three arms, independent two-rate screens and
three fresh confirmation seeds. Its executable protocol is TOKENIZER_TABLE_PROTOCOL.md.
