# Byte-input integration proof: accepted original004

Read [the current research handoff](../RESEARCH_RESUMED_2026-11-07.md).
The intended proof covers real D256/V50257, B8T128, native/raw-byte/nonlexical
arms, CPU/MPS, complete coordinate comparisons and 357 invalid transactions.
No real corpus or holdout was used. Independent saved review004 accepted all
5,392 records / 1,412,427,220 coordinates with all six cases complete.
See `../tokenizer-evidence/byte-composition-boundary-001/model-acceptance004/acceptance.json`.

Current sources are `entry004.py`, `driver004.py`, `model_reference002.py`,
`admission004.py`, `saved_review004.py`, and unchanged `cases.py`,
`codebook_reference.py`, `coordinates.py`, `transactions.py` and candidate
`research/byte_input_model_v2`. They executed once under a frozen plan with fresh exclusive roles and complete
admission. Original004 is now closed accepted. Do not call any old numerical
entry, repeat the proof, or reuse its roles.

Original002 failed a native optimizer differential because it supplied unequal
but close gradients to the BF16 Muon transformation. New003 retained independent
gradient comparisons, then supplied and recorded identical optimizer inputs.
It completed the native CPU case, including62 rejection transactions, before
failing one raw-byte highway-bias gradient coordinate. Both failures and all
original sources/plans/arrays remain closed and preserved.

Saved-array diagnostics established a floating-point branch mismatch in the
Conv1d/separate-bias reference: algebraically equivalent operations selected two
different near-zero ReLU branches, including the failed bias unit. Current
reference002 independently assembles explicit windows while using the declared
fused FP32 primitive. It changes no candidate weights, gradients, scientific
cases, tolerances or acceptance criteria. It shares qualified primitives/native
blocks/library optimizer kernels; this is an integration proof, not an independent
re-proof of those kernels. The earlier byte component proof remains separately
scoped evidence. No failed coordinate is excluded or automatically accepted.

`coordinates.py` saves complete FP32 actual/reference arrays and masks before
assertions, with bounded CPU chunks for FP64 reporting. Initial parameters,
codebooks, logits/loss, every gradient, matched optimizer inputs, two updates and
all child state are covered. The second update uses the same fixed gradients to
isolate state evolution; it is not a second training step. Invalid transactions
snapshot full tensor hashes/object bindings, input/config/optimizer state and
CPU/MPS RNG before and after rejection. RNG-copy addresses are excluded.

Large `.coordinates.gz` payloads remain local under exact tracked manifests.
Do not delete or replay them to save space. Source/evidence changes are pushed
without binary model weights. The 16GiB output cap,4GiB RSS,6GiB MPS,8GiB reserve,
2GiB extra headroom, one-hour runtime and shared nonblocking lock remain fixed.
Next: prepare the separate numerical checkpoint proof for
`research/byte_input_state_v1.py`, whose source-only review passed. No numerical
state qualification exists yet. Reconcile live processes, freeze all new
source/runtime/assets, qualify fresh roles, and verify the complete retained
output, serialization-overlap and memory requirements before admission.
