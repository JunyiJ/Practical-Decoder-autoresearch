# Corrected public inference adopted

The selected byte model now has a qualified public inference handle that applies
the declared FP32 RMSNorm epsilon automatically. Its token-loss method also uses
the verified stable FP32 CPU CE reduction. MPS retains native cross-entropy.
The original selected checkpoint, model parameters, training recipe and export
format are unchanged. This is an inference-correctness improvement, not a newly
measured model-quality improvement.

Use [the new recipe](tokenizer-evidence/public-inference-correction-001/adoption/recipe.json).
The previous bulk-hash recipe and all historical quality evidence remain immutable.
The new source is [byte_input_corrected_inference_v1.py](byte_input_corrected_inference_v1.py).

## Verification on the selected checkpoint

The actual public loader reconstructed the selected slot10 H200 identity on CPU
and MPS and loaded parameter values exactly. Fixed training-text cases covered
batch1 and batch8, each with context128. This is 1,152 token evaluations per
backend, including overlap between the B1 and B8 cases. No validation or holdout
targets were scored, and no aggregate quality score was produced.

| Backend | Batch | Maximum logit difference from training-equivalent forward | Maximum loss error versus FP64 | Largest cache error/tolerance ratio |
|---|---:|---:|---:|---:|
| CPU | 1 | 0 | 7.10e-7 | .172543 |
| CPU | 8 | 0 | 8.08e-7 | .120840 |
| MPS | 1 | 9.06e-6 | 6.30e-7 | .107800 |
| MPS | 8 | 1.30e-5 | 9.09e-7 | .154100 |

All inherited logits/cache atol3e-5/rtol3e-4 checks pass. CPU/MPS maximum
per-target loss difference is7.629395e-6 for both batches, below3e-4. Mean CE
differences are3.147186e-7 and5.454999e-8, below3e-5. The maximum error against
CPU FP64 loss is below the declared3e-5 limit in every case.

Prefill96 plus32 contiguous single-token cache steps, full-context agreement,
reset and exact replay all passed for both backends and batch sizes. Invalid
IDs, target dtype/shape/range, cache position, training mode, autocast and
torch.inference_mode are rejected. An exact-value parameter self-copy correctly
invalidates the handle and cache through version checks. This deliberate final
probe changes no weight value. All weights, semantic inventories and CPU/MPS
random-number states remain unchanged; no gradients or new weights are created.

The guarded integration check took11.462seconds, within600seconds. Peak RSS was
2.43GiB. Recorded evidence occupied913,407bytes before publication, within16MiB.
The final free-space observation was12.21GB. Source/input bindings verified at
entry and exit. Saved-result review separately reconstructed numerical errors,
complete cache coverage, replay hashes and resource checks before adoption.

## Using the public handle

The caller owns the shared research lock and resource guard and verifies the
recipe's `inference_implementation_record`, source receipts and source inventory
before importing. Use the same source-admission discipline as the existing
consumer. In ordinary FP32 execution:

```python
import json
from pathlib import Path
from research.byte_input_corrected_inference_v1 import load

recipe = json.loads(Path(
    "research/tokenizer-evidence/public-inference-correction-001/adoption/recipe.json"
).read_text())
entry = recipe["loader"]
model = load(entry["implementation_record"], entry["plan_record"],
             entry["acceptance_record"], "mps")
logits = model.logits(token_ids)
losses = model.token_losses(token_ids, next_token_ids)  # unreduced FP32 losses
session = model.session(batch=1)
prefix_logits = session.logits(token_ids[:, :96], start_pos=0)
# Append contiguous chunks through position128; reset before a new sequence.
session.reset()
```

The handle owns ordinary no-grad execution. Do not wrap it in
torch.inference_mode or autocast. The returned object is a Consumer, not the old
three-item loader tuple. For future quality experiments aggregate token losses
only over the data and windows declared in their new protocol. The old scorer
and cache APIs remain frozen historical implementations; they do not apply the
correction automatically. This proof establishes correctness, not a latency win.

## Evidence

- [Prospective protocol](PUBLIC_INFERENCE_CORRECTION_PROTOCOL.md)
- [Plan](tokenizer-evidence/public-inference-correction-001/plan.json), SHA256 `6a2a4e276b7b2938c5d552a8eba22231a6c30d67f5eb34181eea5c6d5ca3caac`
- [Acceptance](tokenizer-evidence/public-inference-correction-001/acceptance.json)
- [Result](tokenizer-evidence/public-inference-correction-001/result.json)
- [Implementation manifest](tokenizer-evidence/public-inference-correction-001/adoption/implementation.json)
