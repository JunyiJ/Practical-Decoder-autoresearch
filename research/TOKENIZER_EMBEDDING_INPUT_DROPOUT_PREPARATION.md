# Input embedding feature-dropout preparation

Status: model and stochastic checkpoint correctness passed; finite generated
worker pair is prepared and **unexecuted**. No scientific panel, pilot, new
scientific seed, quality result or adoption exists. Keep the incumbent.

The [evidence review](tokenizer-evidence/generalization-boundary-001/evidence-review.md)
separates measured local facts, literature and inference. Frequency controls
show unigram gaps .153–.173 versus the model gap 1.078; they are not a causal
loss decomposition. Earlier attention/residual dropout experiments mostly lost.
The new hypothesis therefore isolates the input lookup feature site, with one
literature-anchored probability .1 versus .0, rather than repeating those sites
or sweeping rates. It is local regularization/exploitation, not quota-driven
structural exploration. The proposed portfolio allocation stays proposed.

## Correctness evidence

- `correctness001-cpu/mps.json`: exact .0 incumbent initialization, RNG, logits,
  gradients and two optimizer steps; direct stochastic mask equations/gradients
  and device-local RNG repeat; own-policy checkpoint/optimizer/cache rejection;
  full/cache/reset/causality at batches 1/8. Parameters remain 28,492,416 across
  88 tensors, 44 Muon and 44 AdamW groups. Cache remains 196,608 bytes/item.
  Only input features are masked during training. Eval equations are unchanged;
  no attention/residual/FFN dropout, vocabulary, norm, scale or bias change.
- `resume001.json`: both arms passed exact four-update real-data fresh repeat and
  serialized 2+2 resume. Dtype/shape/SHA256 tensor-byte and scalar identities cover
  model/optimizer/EMA/clipping/samplers/CPU+MPS RNG/raw/H200/logits. All 88 EMA
  tensors and three snapshots agree. Serialized split payload is 445,080,807 bytes
  per arm; it lived in memory only, with zero weight files. RSS/MPS peaks were
  1,960,198,144/2,510,094,336 bytes. No comparative memory-saving claim.
- `resume-metadata001.json`: a separate zero-update adversarial proof found that
  inherited validation accepted a short MPS RNG tensor (the invalid resume itself
  was never called). New additive preflight rejects ten RNG/layout/sampler/copied
  configuration forgeries before model/optimizer/EMA/live-RNG mutation. Valid
  numerical transport and its original receipts remain unchanged.
- `worker-source-review001.json`: all 18 finite probability/horizon/LR configs,
  module routes and actual B8/T128 model RNG versus parent mask-only replay pass.
  This is not an independent full worker/validation acceptance yet. The new parent
  must account for mask RNG consumption; the old dropout-free RNG assertion is
  not reused. CPU/MPS full-validation CE agreement awaits actual worker artifacts.

Source sets are additive: 947 component-proof bindings, 950 transport, 952 RNG
preflight, 958 worker bindings. Every original 939 scientific and six prior
inference-diagnostic sources is preserved. Do not edit these namespaces or rerun
closed proofs. New work belongs outside them. All time/RSS/MPS/disk/output guards,
closed holdout and single local worker/four-thread constraints remain.

## Next concrete execution and resources

Prepared source: `tokenizer-evidence/embedding-input-dropout-preparation-001/worker001-build-run001.py`.
Future root: `research_runs/tokenizer-embedding-input-dropout-worker-v1-001`.
Both are new; the root/plan/anchors are absent. Do not invoke this builder below
its measured-entry wake condition. It executes only the two four-update real
workers and independent parent RAW/H200/optimizer/EMA/RNG/clipping/full-validation/
cache acceptance, including CPU/MPS full CE. It uses fixture seeds 600406/600407,
not scientific quality seeds. No repeated updates, retries or scientific claims.

The pair floor is **11,050,479,616 bytes**: two inherited incumbent 560 MiB
RAW/H200 caps, 16 MiB metadata, 8 GiB reserve, measured external headroom
1,267,904,512 and entry headroom 1,458,176. The first-launch wake is
**11,051,937,792 bytes**, adding one measured entry margin before staging. After
an actual accepted endpoint, credit only its retained published bytes once.
Extra cleanup target is 13,197,963,264 bytes; this target does not replace the
mandatory gate. No root/freeze/config/request is created below the gate.

The conservative **full all-output planning bound is 19,590,082,560 bytes**;
cleanup target **21,737,566,208 bytes**. It counts all 16 future 560 MiB pairs:
2 generated workers +2 resource pilots +6 LR screens +6 fresh endpoints, plus
256 MiB metadata/64 MiB portable sidecars, existing reserve and measured external/
entry headroom. This uses original tested per-slot caps, not measured future
training output or a claim of panel readiness. Atomic `.tmp` publication counts
within that slot cap; no second disk serialization copy is planned. Actual
writer/serialization changes require a new forecast. The earlier rough estimate
counted prospective losing-screen retirement too early; the final bound takes
**no future deletion credit**. Existing retirement code is tied to old roots and
three-arm counts; never bypass it for the new study. New tested stage-retention
contracts and actual dependency closure may lower a later forecast, not today.

Current recovery: new external001 removed 8 unused regenerated compiler caches;
external002 removed one more then stopped because its consumer check included
that deleted path. Its source/manifest/journal/error are immutable. Separate003
verified and removed only the remaining three, preserving that failure.
Storage039 losslessly compressed 36 newly closed metadata files with all paths/
readable bytes unchanged and 48 protected payload hashes verified. Total exact
allocation reclaimed is 29,896,704 bytes; this is not the observed free-space
delta. All current external/app/research classes were checked: root-owned npm and
unused Keynote remain normally access-blocked; active/pending updaters, live app
caches, required/shared runtimes/containers, unmatched downloads and personal data
are preserved. Size-based research inventory found no large incomplete write;
all selected/fresh archives and original storage004 remain protected. No new
obsolete checkpoint payload was produced by these in-memory fixtures.

After source integration is exhausted, the unchanged blocker needs only one
short hourly free-space check. Wake for the **worker pair** at the smaller
11,051,937,792-byte condition, not the future full-study bound. New eligible proof,
access, dependency closure, changed measured requirement or added storage triggers
inline recovery. Do not repeatedly scan/commit/audit closed cleanup. A supplied
external APFS volume with 16 GiB free can support verified protected-archive
relocation toward the current conservative all-output target; do not delete or
move protected bytes without that verified relocation protocol.

After both actual worker acceptances: audit their full artifacts and close
finite20-update pilot integration, retain generated payloads until a new exact
retirement audit, then measured pilots/full-budget/finite tuning/fresh design.
The proposed .000625/.00125/.0025 equal LR allowance and three fresh pairs remain
unfrozen. No scientific seed is assigned. Only a fully confirmed mean <−.02,
all seeds negative and simultaneous conditional-text upper bounds negative can
permit a separate adoption review. No rescue, extra dropout sites/rates, partial
panel, cost-only adoption or old-score pooling.
