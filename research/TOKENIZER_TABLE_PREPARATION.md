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
