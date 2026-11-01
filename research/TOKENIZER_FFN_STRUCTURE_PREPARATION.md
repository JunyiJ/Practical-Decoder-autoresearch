# Tokenizer gated versus ungated SiLU FFN preparation

Status: architecture hypothesis selected; integration prerequisites remain.
The renewed obsolete-screen cleanup resolves the current capacity projection
(see latest receipt below). No scientific seeds or launch plan are frozen.

## Question and controls

Compare the adopted width256 four-layer RoPE MLA SwiGLU FFN with a bias-free
ungated SiLU FFN at exactly the same matrix-parameter budget. Keep RMSNorm,
serial unit residuals, final affine LayerNorm, untied GPT2 tables, grouped-embedding
v2 numerical policy, context128, batch8,1200updates, cooldown800 and H200 fixed.
The completed EMA study does not justify changing the H200 endpoint.

Control: W_down[SiLU(W_gate x) * W_up x], hidden672, three matrices.
Candidate: W_down[SiLU(W_up x)], hidden1008, two matrices.

Both have516,096 matrix parameters per layer:3*256*672 =2*256*1008.
Four FFNs therefore retain2,064,384 parameters, and total model parameters would
remain28,492,416. The candidate uses a1.5-times wider hidden representation
in exchange for removing the multiplicative branch. This changes factorization,
hidden width, activation statistics and optimizer geometry together. It keeps
SiLU fixed; it is not a repeat of the completed SiLU-versus-GELU gate study or
an isolated test of multiplication. Equal parameter/matrix-multiply counts do
not guarantee equal wall time, activation memory or optimization behavior.

This is a distinct unresolved subword architecture question. Historical
residual attenuation is a weaker next candidate: fixed0.1 and learned0.1 gains
worsened every fresh seed at widths64/256, and no new MLA-specific instability
has been demonstrated. Parallel residual results were also negative. MoE remains
deferred on its18.782GB full-panel requirement, with its wider-dense control
retained. No closed campaign or failed historical numerical fixture is reopened.

## Bounded proposed panel

Two20-update resource pilots; three equal LR choices0.000625/0.00125/0.0025
per arm on a screening seed; select each arm's minimum H200 full-validation CE
with lower-LR tie break; then both selected recipes on three new paired seeds.
Six screens plus six fresh runs, each1200updates. No grid extension or alternative
winner. Keep both selected screen pairs and every fresh raw/H200 pair.

Proposed adoption rule: candidate mean paired H200 CE delta below-0.02, every
fresh seed below0, all simultaneous conditional-text upper bounds below0 for
2048/4096/8192-token blocks with2000 common paired PCG64 draws and linear
quantile1-0.05/3. Freeze exact seeds, ordering and all rules before results.
No cost-only adoption route. Reused validation text is not an independent test
holdout or population-seed confidence. Test scoring stays closed.

## Capacity finding before implementation

CPU construction validated the unchanged baseline's28,492,416 parameters,
65,536 persistent-buffer bytes and optimizer groups:2,752,512 Muon parameters
in44tensors;25,739,904 Adam parameters in44tensors. Candidate FFN components
have the exact proposed shape/count. A full candidate is not yet supported or
validated. Its intended Muon tensor count is40, with unchanged parameter count;
all Adam moments and Muon momentum are charged in the projection.

Raw model + one H200 shadow + H200 export + optimizer payload lower bound is
558,969,520B per trajectory. The conservative allowance is788,953,140B,
including inherited envelope/headroom; final serialization and transient audits
must independently validate it. With prospective audited pilot/losing-screen
cleanup, the high-water stage retains two selected plus six fresh pairs:
6,311,625,120B, plus256MiB metadata and8GiB reserve = **15,169,995,168B**.
Pre-cleanup observed free space was10,181,873,664B, a4,988,121,504B shortfall.
Even six fresh payload pairs alone plus the reserve require11,943,751,712B;
dropping pilots/screens cannot make the full comparison fit in the observed space.
No protected-weight deletion, reduced confirmation or partial-panel start is
credited. The successful LayerNorm pilot pair is being separately audited under
the retention policy; any actual reclaimed space is recorded in its receipt.

## Concrete numerical prerequisite and limitations

Standalone ungated SiLU forward values and input/up/down gradients passed an
independent NumPy analytic-derivative oracle for three cases, including the
actual256-by1008 component. Maximum absolute discrepancy was3.553e-15 in
float64 CPU. No checkpoint, training corpus, test data or scientific trajectory
was used. This is component arithmetic evidence only, not full-model correctness,
MPS behavior, resume equivalence or quality evidence.

The first shape-only attempt failed before the oracle: existing normalization
validation rejects aliased storage pointers, and all meta tensors report pointer0.
Its original source, failed guard and failure decision remain immutable. A distinct
CPU construction attempt used real storage and passed without modifying or bypassing
any validator. The corrected audit took0.502seconds and peaked at447MB RSS.

The current policy correctly rejects the proposed FFN. New integration must live
outside all399 frozen source files and bind the exact module layout, initialization,
shape-dependent Muon scaling, optimizer coverage, checkpoint/resume and H200
identities. Require actual forward/gradient/MPS/cache parity, full-model resource
and serialization audits, then seed census and finite pilot/quality freezes.
Do not build another generic ledger or materialize large fixtures while the full
retained-output projection fails. Safe source inspection and small component
checks can continue while capacity is unresolved.

Evidence: tokenizer-evidence/ffn-structure-preparation-001/review002.json,
review002.py and guard; review.py, review-guard.json and review001-failure.json
preserve the original unsuccessful meta audit. Historical source-document hashes
are bound to the audit's recorded commit, before subsequent documentation updates.


## Authorized cleanup completed; remaining capacity blocker

Exact retention audit removed only the successful LayerNorm pilot002 raw/H200
pair,559,220,996B. The closed normalization campaign and dependent inference
were fully reviewed before deletion. Failed RMS pilot001, all selected/fresh
weights and the explicitly retained EMA pilot remain protected. All234 retained
checkpoint/archive files and25,974 non-checkpoint metadata files were rehashed
unchanged after removal. Original pilot receipts stay immutable; the deleted
weights cannot be reloaded or re-audited and closed controllers must not rerun.

The first audit stopped before mutation because its combined inventory exceeded
the inherited2MiB receipt cap. Original source and failed guard are preserved.
The completed v2 audit used bounded inventory parts while retaining that cap;
its exact two-file allowlist, per-removal intent/journal and completed guard are
under norm-pilot-retention-v2/. No other file was deleted.

Current free space: **10,740,375,552B**. Full proposed panel requires
**15,169,995,168B**, leaving **4,429,619,616B** additional capacity needed.
No next training launch is authorized by this preparation alone. Resolve safe
capacity or an appropriate output volume before full-panel dispatch. Keep all
three fresh confirmation pairs, both tuned controls and the8GiB reserve; do
not change scope solely to fit. Do not repeat unchanged space polling or add
metadata-only implementation work while this prerequisite remains unresolved.


## Renewed old-checkpoint retirement completed

The user explicitly authorized removing old checkpoints. The narrow exception
retired only two obsolete wider-hybrid screening archives (26 completed endpoints),
reclaiming 9,828,356,231 logical bytes. No selected, fresh-confirmation,
failed, or numerical-fixture checkpoint was removed. All 232 surviving checkpoint/
archive files and 25,974 non-weight evidence files were rehashed unchanged.
Original archive/member hashes and results remain; the retired weights can no
longer be restored or re-evaluated.

Free space at completion was 20,517,961,728 bytes, above the proposed
15,169,995,168-byte high-water requirement by 5,347,966,560 bytes.
The previous capacity blocker is resolved against that projection. Revalidate
actual serialization/transient needs before freezing or dispatching a panel.

The earlier restriction on all implementation while capacity was short was too
broad. Capacity constrains disk-consuming dispatch, not bounded source or CPU
prerequisites. Continue narrow full-model FFN integration and numerical/cache/
optimizer/checkpoint prerequisites outside the frozen sources. No new scientific
seeds, plan freeze or training launch has occurred. AGENTS.md now requires active
blocker recovery and distinguishes blocked actions from independent useful work.

Evidence: tokenizer-evidence/obsolete-screen-cleanup-001/manifest.json,
completed.json, journal.jsonl and both resource guards.


## Full-model CPU and strict MPS integration passed

The isolated research/ffn_structure_v1 runtime now accepts exactly the matched
SwiGLU672 and ungated SiLU1008 arms on the grouped-embedding D256 MLA backbone.
It overrides six runtime modules and resolves unchanged dependencies from the
frozen norm runtime under its own namespace. All399 frozen source files remain
byte-identical; the tested new source snapshot is preserved separately.

The control matches legacy initialization and logits bitwise. Candidate total
parameters are28,492,416, with2,064,384 FFN parameters. Complete non-duplicated
optimizer coverage passed: Muon2,752,512 parameters/40tensors and AdamW
25,739,904/44tensors, retaining width-LR scale0.5 and match_rms_adamw. Full-model
synthetic forward/backward and one CPU update passed with finite gradients.
CPU cached/full maximum error was8.345e-7; causality was exact.

Strict MPS tests passed for both arms with repeated token IDs exercising grouped
embedding gradients. Maximum CPU/MPS logits error8.941e-7, gradient error1.500e-7,
and full/interleaved-cache error9.537e-7. Reset and causality were bitwise exact.
Cache remains196608B/item. Existing tolerances and resource guards were retained.
Wrong-arm cache and checkpoint signatures, a changed activation module and use
of the candidate under the old policy were rejected. No corpus/test/checkpoint
file IO or scientific training occurred. Synthetic fixture seeds are not reserved
scientific seeds; the future census must include these sources.

Candidate initialization uses ordinary bias-free Linear draws. Removing a
projection and changing its shape changes RNG consumption, so paired scientific
seeds will not imply identical attention/readout initial values. The comparison
changes factorization, hidden width and optimizer geometry jointly.

Next: real serialized raw/H200 checkpoint and optimizer-resume equivalence,
worker/parent full-validation and inference acceptance; measure actual output
allowance/transients before seed census and finite pilots/quality freezes.
Current capacity remains above the15.170GB projection. No quality claim or
training launch is supported by these numerical prerequisites.

Evidence: model-integration001.py/json/guard/source and
mps-integration001.py/json/guard in tokenizer-evidence/ffn-structure-preparation-001/.
