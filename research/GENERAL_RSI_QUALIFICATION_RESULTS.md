# General-corpus restart and scoring qualification

The bounded prerequisite passed in143.639 seconds, peak RSS1,991,081,984 bytes.
For each arm,40 uninterrupted updates exactly matched a checkpoint restored
at35 then continued to40. The comparison covered every parameter, both
optimizer states, scheduled learning rates, the complete training history,
both private sampler RNGs, CPU/MPS RNG, and EMA tensors/counters. The checkpoint
was taken after the first EMA snapshot and resumed through the second.

| Measurement | Native | Byte |
|---|---:|---:|
| Complete checkpoint bytes at35 | 445,009,719 | 247,901,553 |
| Save seconds | .363 | .267 |
| Read/restore seconds | .587 | .421 |
| Full100,416-target validation seconds | 25.935 | 62.354 |
| Largest independent CPU FP64 mean-CE error | 2.59e-7 | 1.57e-7 |

Both scorers covered all108 documents, within-document targets only, terminal
EOT included. Independent CPU/FP64 checks used one document from each source,
including non128-divisible tails. Their fixed mean-CE tolerance was3e-5.
Validation sampler state and model parameters were unchanged by scoring. Test
contents remain unopened. These are numerical/cost results, not quality evidence.

The general worker implements the plan's exact interval10 averaging directly:
H2048 snapshots1370..2040, with the last average carried to the endpoint;
H10240 snapshots6830..10240. The historical averaging helper requires an
endpoint-aligned horizon and therefore cannot express H2048 unchanged. Its
frozen implementation was not edited and no scientific horizon was changed.

Capacity recovery retired four now-obsolete numerical-prerequisite checkpoints
(two20-update inference payloads and two35-update complete restart payloads),
872,347,120 bytes total. Each exact path/size/hash/inode passed no-open-consumer
and closed-dependency checks under the worker lock, with durable before/after
deletion journals. All23 pinned evidence/source/selected-weight files remained
unchanged. The original results and checkpoint hash receipts are preserved;
no scientific screen or fresh confirmation weights were deleted.

The frozen panel now has four LR screens followed by six fresh confirmations,
71,303,168 total training targets. SHA256:
`ab70714782ccc514809a70832c8ca650c884ca03719eb39616df37e1c3d876b2`.
Its full measured-state forecast is2,673,037,156 additional bytes plus10 GiB
reserve; free space at freeze13,589,147,648 bytes leaves178,692,252 bytes above
that requirement. All final RAW/H200 weight pairs are retained; only obsolete
transient restart predecessors are retired through exact journals. Before each
slot the remaining whole-panel forecast is rechecked. Training estimate plus
30% margin and one-hour evaluation allowance is13.67 hours, below the36-hour
tranche ceiling; actual full evaluator timings fit that allowance. The worker
charges prior guard durations against the same tranche ceiling.

All acceptance rules, LR choices, role bundles, fixed prompts and run order are
prospective in `research/tokenizer-evidence/general-rsi-panel-001/plan.json`.
Screening selects LR only; it cannot establish an architecture win. Qualification
losses do not change those choices. The next execution is slot00: native,
LR.00125,2,048 updates. Continue slot order after reviewing completed receipts;
run `python -m research.general_rsi_panel_v1 select` only after all four screens.

Evidence roots: `general-rsi-qualification-001`, `general-rsi-capacity-001`, and
`general-rsi-panel-001` under `research/tokenizer-evidence`. Original failed
Shakespeare work remains deferred and immutable. No additional historical test
suite or scientific fixture was rerun.
