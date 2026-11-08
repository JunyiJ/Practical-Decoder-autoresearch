# Active research: full-model proof accepted; checkpoint proof next

The human resumed autonomous research on November 7, 2026. The native incumbent
remains unchanged. Byte composition has no measured real-data quality gain and
has not been adopted.

The [independent saved review004](tokenizer-evidence/byte-composition-boundary-001/model-saved-review004/review.json)
accepted all 5,392 records and 1,412,427,220 coordinates from the complete actual
D256/V50257, B8T128 integration proof. Native, raw-byte and nonlexical-code arms
passed on CPU and MPS, including all 357 invalid-call preservation transactions.
The proof covers initialization, codebooks, forward/loss, every gradient,
causality, two optimizer updates with identical input gradients, and complete
optimizer state. The second update reuses fixed gradients to isolate state
behavior; it is not a second training step. Numerical tolerances remain
4e-6 + 4e-5*abs(reference), with exact gates where specified by the frozen plan.
No corpus, holdout or quality comparison ran.

The numerical worker finished in 559.704 seconds, with peak RSS 2,915,385,344
bytes under the 4 GiB limit. Recorded output was 6,132,301,517 bytes under 16 GiB.
The guard enforced the 6 GiB MPS driver limit; its final 26,951,680-byte MPS
observation is not a peak measurement. Independent streaming review took 29.621
seconds and 65,617,920 bytes peak RSS. All workers terminated successfully.

## Preserved failures and recoveries

Original002 failed 18 native CPU optimizer-update coordinates. The independent
gradients were within tolerance, but 13 coordinates crossed BF16 rounding
boundaries before Muon orthogonalization. The
[saved-array diagnosis](tokenizer-evidence/byte-composition-boundary-001/model-update-diagnosis001/result.json)
identified the confounded comparison. Subsequent attempts retained the original
independent gradient gates, then recorded identical optimizer inputs before
comparing updates and state. Original002 remains closed failed.

Original003 passed the complete native CPU case, then failed one raw-byte
highway-bias gradient coordinate. The
[bounded primitive diagnostic](tokenizer-evidence/byte-composition-boundary-001/component-arithmetic-diagnosis001/result.json)
found two near-zero ReLU branch differences caused by Conv1d/separate-bias
reference arithmetic. Independently assembled explicit windows using the
declared fused FP32 linear primitive reproduced every preactivation exactly.
Reference002 uses that assembly without copied masks, excluded coordinates,
weight shifts or tolerance changes. It shares native blocks, qualified numerical
primitives and library optimizer kernels; acceptance is an integration proof,
not an independent re-proof of those kernels. Original003 remains closed failed.

Fresh004 used separately censused exclusive roles and passed unchanged cases
and gates. All original plans, sources, guards, failures, coordinates and reviews
remain closed. Never rerun an old entry or reuse its roles.

## Next executable action

The new [numerical checkpoint adapter](byte_input_state_v1.py) covers model
parameters, named optimizer state, immutable-buffer identity, source/runtime
binding and CPU/MPS RNG. It validates the complete payload before changing a
fresh destination and provides bounded exclusive serialization with hash-bound
reads. [Source review001](tokenizer-evidence/byte-composition-boundary-001/state-source001/review.json)
passed without importing Torch or the adapter. No adapter numerical call has
been admitted or executed. It is not a complete training checkpoint: sampler,
data identity, schedule, clipping history and H200 work remain separate.

Prepare a bounded all-three-arm CPU/MPS serialization, restore and next-update
proof, including invalid-payload preservation and exact RNG continuation. Freeze
complete retained-output and memory costs (including staging and serialization),
all source/runtime/assets and new exclusive roles, then perform static admission
before numerical import. Do not reduce cases or guards to fit storage. Current
model proof acceptance does not admit a training panel or establish model quality.

No worker is active at this checkpoint; the active autonomous goal owns the next
continuation. Paused automations remain unchanged. After this proof approximately
25.75 GB was free; no current execution is blocked. The next proof's complete
requirement must be measured before freeze. Future scientific-panel capacity is
unqualified and must be remeasured and recovered when its actual requirement is
known.

The [acceptance receipt](tokenizer-evidence/byte-composition-boundary-001/model-acceptance004/acceptance.json)
and [local coordinate manifest](tokenizer-evidence/byte-composition-boundary-001/model-acceptance004/local-coordinate-manifest.json)
bind all 5,392 retained payloads. Earlier manifests preserve the failed
[002](tokenizer-evidence/byte-composition-boundary-001/model-update-diagnosis001/local-coordinate-manifest.json)
and [003](tokenizer-evidence/byte-composition-boundary-001/model-gradient-diagnosis001/local-coordinate-manifest.json)
arrays. Git contains sources and complete evidence metadata, not these large
coordinate arrays or binary model weights. A clone alone cannot repeat the
saved-array review.
