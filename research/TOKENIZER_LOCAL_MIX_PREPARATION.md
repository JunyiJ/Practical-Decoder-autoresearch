# Causal FFN-input mixing preparation

Current question: does identity-initialized causal three-tap per-channel mixing
of the normalized FFN input improve the adopted tokenizer D256 MLA recipe under
equal bounded LR tuning? This is an untested architecture hypothesis. Existing
short convolutions in KDA do not establish a benefit here; no Kimi reproduction
or long-context claim is intended.

After the warmup campaign closed without adoption, both architecture and bounded
HP were reviewed. Independent decay0.00006 versus0 remains a valid separate
regularization hypothesis, deferred while addressing the previously queued local
mixing family. No closed campaign or failed fixture is reopened.

For each block, after the attention residual, let u be the existing FFN RMSNorm
output. Feed the existing SwiGLU with
v[t,c]=a0[c]u[t,c]+a1[c]u[t-1,c]+a2[c]u[t-2,c], zero-padding the left edge.
There is no bias or additional nonlinearity. Initialize three learned channel
vectors a0=1,a1=a2=0 without RNG draws, so the initial function and shared model
weights can match the incumbent exactly. Four D256 layers add3072 parameters;
report the small capacity difference explicitly rather than claim parameter
matching. The prospective optimizer assignment is inherited AdamW at base LR
with the same independent decay for these 1D vectors; actual group/state policy
still requires full-model verification.

Streaming inference retains exactly two normalized FFN-input tokens per layer.
Preserve history across uneven chunks; reset to zeros for a new sequence. Four
layers add8192 FP32 bytes per batch item to the current196608-byte MLA cache;
full-model buffer enumeration and timing must verify this projection. A standalone
component is implemented in research/local_mix_v1/component.py. It is not yet
accepted by a full-model/checkpoint policy or wired into an existing runtime.

Frozen component001 passed six generated cases: CPU FP64/FP32 and MPS FP32,
batches1/3, width7, length11, chunks4/1/6. An independent grouped Conv1d oracle
checked forward values and input/all-three-vector gradients. Maximum FP64 errors
were1.11e-16 forward/1.67e-16 gradients; maximum FP32 errors2.98e-8/1.79e-7.
Cached output was exact in all cases; identity initialization, future-token
causality, reset and wrong-history rejection passed. All497 predecessor sources
remained unchanged. No corpus, optimizer training, scientific seed, checkpoint
resume or quality evidence was used. Preserve this successful source/plan/result;
do not rerun it merely for progress.

Next add narrow explicit full-model policy, shared-weight identity, vector optimizer
grouping, raw optimizer/RNG/EMA resume, H200 export and full/cache/reset integration
outside frozen sources. Then real bounded worker and independent parent full
validation/inference, fresh seed census and complete measured output/transient
capacity, finite guarded pilots and final quality freeze. Prospective quality
comparison remains six equal-LR screens and three fresh pairs with strict gates;
no training panel or scientific seed is frozen/reserved yet. Keep all existing
selected/fresh/failed weights and immutable evidence. Small source/numerical work
proceeds while future full-panel capacity is audited; do not reuse stale blockers.

Evidence: tokenizer-evidence/local-mix-preparation-001/direction-decision.json,
component001-plan.json, component001-result.json and component001-guard.json.


Full-model model001 passed both arms with identical shared initialization, RNG
state and initial logits. The candidate adds3,072 parameters in12 AdamW vectors
at base LR and independent decay0.00006; filtered existing optimizer groups match.
CPU/MPS maximum logits difference7.16e-7 and gradient difference1.32e-7;
nonzero lags were exercised. Interleaved uneven-chunk/cache/reset/causality tests
passed. Wrong-arm, malformed and aliased caches were rejected before mutation.
Actual cache storage is196,608B/control versus204,800B/candidate per batch item.
Checkpoint payload signatures reject changed arms/configurations and constructed
mixer mutations. All497 predecessors and the standalone component remain intact.

Candidate serialized model-resume001 produced an intermediate step2 checkpoint
and matched both four-update repeat and interrupted resume exactly for model,
Muon/AdamW optimizer, EMA and CPU/MPS/loader RNG (max difference0). It then failed
when the harness attempted H200 export from step2 of a total4-update horizon.
The exporter correctly requires the final horizon. Preserve the original source,
plan, trace, guard and445MB-class checkpoint as failed evidence; its STOP closes
this fixture without retry. This is not a passed final H200 integration test.

A separate fresh final-horizon two-arm worker prerequisite is prepared in
trainer-worker001.py, with actual run_training, bounded final-only raw/H200
publication, independent parent full-validation acceptance and arm-specific cache
measurement adapters. Its imports and arm contracts passed;515 sources are bound.
It does not load failed resume weights. Actual workers, final H200 and paired
inference have NOT run. No scientific seed or training panel is reserved/frozen.

Capacity001 measures9,032,929,280B free versus10,066,329,600B required for the complete
worker pair (deficit1,033,400,320B), including8GiB reserve, both560MiB weight
allowances, both16MiB trajectory metadata and256MiB round metadata. A read-only
weight/archive inventory found no further eligible large numerical archive or
losing-screen recovery; the remaining large weights are selected/fresh/failed.
The remaining unreviewed numerical weights are only3.69MB. No cleanup was done;
unrelated maintenance remains untouched. One storage action is needed: free2GB
on this volume for the next worker pair, or provide another research volume.
The eventual scientific panel requires a separate full-output projection.
Independent worker source work is complete. Wake at free>=10066329600B or new
eligible recovery evidence, checking hourly while waiting; no unchanged commits,
repeated escalation, retry of closed fixtures or additional metadata scaffolding.


## Storage recovery and final-horizon worker acceptance

The latest human explicitly authorized retiring obsolete failed numerical
checkpoint payloads while keeping selected models and fresh confirmations. Exact
cleanup001/003 retired14 failed or incomplete payloads totaling5,892,893,697B.
The two operations verified36,155 and36,234 retained files unchanged, including
selected/fresh weights and every original failure source/log/config/receipt. The
earlier retain-failed language is superseded only by these later exact manifests.
Cleanup002's preliminary eligibility assertion failed before deletion; corrected
003 checks failed terminal slots and accepted prefixes, rather than treating an
old generic protected inventory label as proof of a selected model. Include
.pt.tmp files in future size-based inventories. AGENTS.md now records this rule.

The initial trainer-worker001 stopped before dispatch because generated.txt was
inside the restricted campaign output namespace. Original source/plan/failure and
STOP are immutable. Separately frozen worker002 uses generated input outside
the campaign root with all515 model/acceptance sources unchanged. Both actual
four-update workers completed, with4,096targets and3H200snapshots each. Independent
parent payload/optimizer/RNG/EMA/full-validation acceptance passed; EMA inventories
were88/control and100/candidate. CPU/MPS full-validation CE differences were below
2.70e-7. Final H200 export passed for both arms. No failed fixture was rerun.

Trained paired inference001 passed full/cache/reset tests at batches1/8 and the
predeclared alternating2warmup/8repeat measurements. Actual cache storage is
196,608B/control and204,800B/candidate per item. These short generated-text results
validate integration and measurement only; no quality or deployment gain claimed.
The scientific corpus and closed holdout were not read and no scientific seeds
or training panel are reserved/frozen. All new generated weights remain retained.

Capacity002 observes13,565,247,488B free after both workers/inference. The prior complete
13,555,990,528B panel allowance has only9,256,960B current margin and is
not dispatch permission: actual serialized outputs, metadata/transient overlap,
finite plan and seed census still need review. Next implement minimal finite
pilot orchestration/design with tested primitives, audit real accepted payloads
and close successful generated-worker dependencies. Their1,118,405,952B pair
weights may then be retired only by another exact audit under current permission.
Do not leave continuation in the resolved storage-wait state.
