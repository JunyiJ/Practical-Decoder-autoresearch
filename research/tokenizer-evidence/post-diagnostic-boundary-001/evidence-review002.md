# Own native MLA gate real-data resume prerequisite

The unchanged, fixed-half and learned-gate arms now have a complete real-data
four-update repeat/serialized-resume proof and an independent terminal review.
This is checkpoint correctness evidence, not model-quality evidence. Keep the
native incumbent; no scientific seeds, quality panel or adoption exist.

## Original failed harness remains closed

`gate-resume001` stopped after the first arm's four completed updates because
its harness indexed every `named_buffers()` entry in the serialized state.
Native `causal_mask` is registered with `persistent=False`; it is reconstructed
and is absent from `state_dict()`. The exact exception was
`KeyError('transformer.h.0.attn.causal_mask')`. The original source, plan,
3.220 s fixture-exception guard and no-retry STOP remain unchanged. No arm was
accepted. Its 4,096 training and 4,096 sampled-validation target evaluations
and fixture roles 600520/600521/600522 are recorded separately in
`gate-resume-failure001.json`. They are not a quality failure.

The separate additive `gate-resume002.py` changes only this harness assertion:
persisted RoPE buffers must survive export exactly; nonpersistent causal masks
must equal the exact reconstructed 128×128 native lower-triangular Boolean mask.
It preserves the failed proof source and all unchanged model, optimizer and
transport code. New roles 600523/600524/600525 were explicitly audited unused
in `fixture-role-census003.json` before this new proof. No old command was rerun.

## Measured corrected proof

For every arm, two fresh four-update trajectories matched exactly. A separate
two-update RAW state was moved to CPU, all live MPS training objects were
released, and the complete tree was serialized to a bounded in-memory buffer.
The buffer was loaded with `weights_only=True`, released, and restored into a
new destination before the last two updates. The final RAW, H200, generated H200
logits, losses/clipping history, optimizer moments, per-parameter exposure,
private train/evaluation sampler streams and CPU/MPS RNG hashes matched the
fresh result exactly on the same MPS backend.

All 88/88/92 learned tensors received gradients on all four updates. All three
arms had three H200 snapshots, with only learned parameters averaged. Shared
input/target streams were identical across arms. Initializers used the default
CPU generator inside a CPU RNG fork; global CPU and MPS RNG were unchanged.
No cross-device optimizer-equivalence claim or rescue of the closed response
diagnostic is involved.

Each arm rejected 26 malformed RAW/config/source/RNG/sampler/model/optimizer/
EMA/history payloads and four malformed H200 payloads before destination,
optimizer, EMA, sampler or global RNG mutation. H200 remained inference-only
and was rejected as an optimizer-resume input.

| Arm | Serialized two-update RAW bytes | Learned tensors |
| --- | ---: | ---: |
| Unchanged | 445,091,367 | 88 |
| Fixed-half | 445,091,367 | 88 |
| Learned elementwise gate | 448,242,127 | 92 |

`gate-resume002/guard.json` finished in 37.6407 s, peak RSS 2,403,614,720 B and
peak MPS driver allocation 2,569,945,088 B, within the original 600 s/4 GiB/6 GiB
limits and 8 GiB free reserve. The proof used 36,864 training and 36,864 sampled
validation target evaluations. Validation was sampled solely for replay checks;
the full 34,278-target validation and holdout were not scored. The holdout tail
was not encoded. No tensor files or checkpoints were written.

`resume-review001.json` independently verified all closed evidence, accounting,
inventories, original failure preservation and all 1,068 source bindings without
loading a model or replaying a gradient/proof. The new four-file namespace is
`research/tokenizer_mla_gate_resume_v1`; it uses own gate configuration and RAW/
H200 kinds rather than legacy or MoE model validators. All 1,062 predecessor
bindings, including all 1,035 original scientific files, remain unchanged.
Both resume proof scripts, the transport namespace and independent review are
now closed and immutable. New fixes must be additive.

## Next enabled work

Build the smallest own gate four-update worker and independent RAW/H200 parent
acceptance needed for this test. Use the proven own data/model/transport,
streamed bounded checkpoint publication, complete learned exposure/clipping,
and full common-native CPU/MPS validation plus cache/reset at batches 1 and 8.
Then qualify all three 20-update resource pilots and actual full-horizon
serialized layouts before admitting a finite scientific panel. New worker and
pilot roles remain unassigned and require unused-role audits.

The existing full prospective 24-pair retained floor 25,227,227,136 B and target
27,374,710,784 B remain covered by about 80.69 GB free. No weight outputs were
added, requirement changed or storage block arose; no cleanup is needed.
The current result does not qualify the full 1,200-update serialization caps,
runtime or final inference. Those gates remain mandatory before any freeze.
