# Active research: byte-input full-model proof recovery

The human explicitly resumed research on November 7, 2026 and authorized
bounded autonomous correction of new setup, harness and implementation defects.
Historical closed experiments and scientific comparisons remain fixed. The
native incumbent is unchanged; byte composition is unadopted and has no measured
real-data quality improvement.

Start with [current machine-readable progress](tokenizer-evidence/byte-composition-boundary-001/model-proof-recovery001.json)
and [the proof source guide](byte_input_proof_v1/README.md).

## What the new numerical evidence establishes

- The complete public-byte codebooks and nonlexical control mappings passed in
  the attempted proofs, including full vocabulary rows and exhaustive small
  Feistel domains. The full-domain claim remains the separate XOR inverse argument.
- Fresh model proof003 completed the genuine native CPU B8T128 case: all forward,
  loss, initialization and gradient comparisons; two updates with identical
  gradient inputs; complete optimizer state; causality; and all62 invalid-call
  preservation checks. This verifies the tested native CPU integration scope,
  including the new optimizer-state rejection checks, not all model variants.
- Proof003 then failed one raw-byte highway-bias gradient coordinate. The full
  three-arm CPU/MPS proof is incomplete and unqualified. No MPS model case ran.

[Independent saved review003](tokenizer-evidence/byte-composition-boundary-001/model-saved-review003/review.json)
reconciled1,004 records/406,261,069 coordinates and confirmed that failure.
The run exited1 after104.154s; peak RSS3,003,629,568B stayed below4GiB.

## Preserved failures and exact recoveries

The original model-math002 update comparison failed18 coordinates in the native
CPU dkv matrix (max7.812e-6). The two gradients differed by only3.347e-10, but13
coordinates crossed BF16 rounding boundaries. Muon casts to BF16 before its
orthogonalization. That comparison confounded update correctness with sensitivity
to different gradient inputs. [Saved-data diagnosis](tokenizer-evidence/byte-composition-boundary-001/model-update-diagnosis001/result.json)
did not rerun the model or optimizer. Fresh003 preserved the independent gradient
gates, then supplied exactly equal gradients for the update/state comparison.
Native CPU passed. Original002 remains closed failed, never reinterpreted as pass.

The next failure was different: raw-byte CPU highway-bias unit382 differed by
1.34085e-5. A new diagnostic on saved initial arrays found two near-zero ReLU
branch differences between the declared fused FP32 arithmetic and the algebraic
Conv1d/separate-bias reference. The failed unit's preactivation was +1.177e-9 in
the declared path and0 in the alternative. Independently assembled explicit
windows with the declared fused linear primitive reproduced every preactivation
exactly. [Primitive diagnostic](tokenizer-evidence/byte-composition-boundary-001/component-arithmetic-diagnosis001/result.json)
preserves all preactivation arrays and scope limits. It called no historical
model/reference function, full body, backward, optimizer or initializer.

The prepared reference002 derives windows by explicit slicing/stacking and uses
the declared fused FP32 linear operations. It does not copy masks, omit failing
coordinates, move weights away from a kink, or relax tolerances. Its independence
is indexing/assembly and explicit CE/grouping logic; it shares qualified numerical
primitives, native blocks and optimizer kernels. The earlier independent byte
component math proof remains applicable only in its established finite scope.

## Next executable action

Prepare and freeze model-math004 for entry004/driver004/model_reference002, with
three new exclusive initializer/code-key roles and admission004. Keep D256,
V50257, B8T128, all three arms, CPU/MPS, 357 rejection transactions and the exact
4e-6 +4e-5*abs(reference) numerical gates. Keep exact initialization, codebook,
causality and matched-gradient input gates. Then execute once and reconcile saved
coordinates using saved_review004. No004 roles are assigned or drawn yet and
004 is not numerically admitted. Never rerun002/003 or reuse their roles.

No worker is active at this checkpoint. Existing paused automations remain
unchanged; the active autonomous goal owns continuation. Observed free space
32,098,246,656B exceeds the prospective proof requirement27,917,287,424B
(16GiB output +8GiB reserve +2GiB headroom). No storage blocker or cleanup was
triggered. Full training/state/cache/scorer and scientific-panel work remains
later, after actual-model qualification.

The1,205 original coordinate payloads remain local and protected, totaling about
2.14GB. Exact gzip hashes and metadata are tracked in the
[002 manifest](tokenizer-evidence/byte-composition-boundary-001/model-update-diagnosis001/local-coordinate-manifest.json)
and [003 manifest](tokenizer-evidence/byte-composition-boundary-001/model-gradient-diagnosis001/local-coordinate-manifest.json).
Git contains sources, plans, guards, complete per-record metadata, rejection
snapshots, diagnoses and reviews; it does not contain those large arrays or
trained weights. A clone alone cannot reproduce their saved-data review.
