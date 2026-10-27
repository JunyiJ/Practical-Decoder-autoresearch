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
