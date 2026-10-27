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
