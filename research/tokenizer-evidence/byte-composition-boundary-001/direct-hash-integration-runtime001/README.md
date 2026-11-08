# Equivalent hashing implementation adopted

The seven additive sources passed their bounded exact differential comparison
against the adopted model. The only numerical-source expression change is direct
NumPy byte serialization in the immutable-buffer digest; the remaining changes
route imports to new exact classes. Original sources, type guards, serialized
formats, selected weights and the complete quality comparison remain unchanged.

The adopted optimized recipe is
[direct-hash-adoption001/recipe.json](../direct-hash-adoption001/recipe.json).
Its public consumer is `research.byte_input_checkpoint_consumer_v2.load`, taking
the recipe's implementation, original plan and acceptance receipts plus `cpu` or
`mps`. The caller must hold the shared worker lock and apply resource guards.
The manifest admits all additive sources and numerical evidence; v1 preparation
still reconstructs trusted typed identity from the original plan. Do not pass
JSON-erased identity tuples directly into the lower-level inference loader.

## Measured execution and exact comparisons

Each implementation loaded the same accepted first-fresh slot10 H200 checkpoint.
On CPU and MPS separately, complete tensor fingerprints matched for:

- Cached B1 and B8 prefix96 + continuation32 logits, with reset accepted.
- Fixed-training-prefix B8T128 full logits, loss and all105 parameter gradients.
- Initial, step1 and step2 complete captured/validated training contexts:
  parameters, optimizer state/groups, history, clipping, scheduler, data identity,
  private sampler state, global CPU/MPS RNG and averaging state.
- The first H200 snapshot at step2 of the unchanged four-step fixture recipe.

The fixture performed two actual Context.step calls for each of two implementations
on each backend: eight numerical updates total. Fresh sampler roles62100600001/2
were absent in the pre-freeze exact repository search; the same trajectory is
mirrored across implementations/devices as an equivalence comparison. Historical
model initializer roles reconstruct only the accepted checkpoint identity. This
is not a new quality experiment, training initializer search or old fixture replay.
Validation data was not scored; no holdout was encoded or scored. No weight archive
was written. Before/after output equality was exact; no tolerance was introduced.

| Backend | Original step1 / step2 seconds | Optimized step1 / step2 seconds | Total two-step speedup |
|---|---:|---:|---:|
| cpu | 3.8783 / 3.9833 | 2.1826 / 2.2403 | 1.777x |
| mps | 3.1601 / 3.2952 | 1.3449 / 1.1607 | 2.576x |

CPU ran original then optimized; MPS reversed that order. These are only two
training-step timings per arm: step1 includes lazy optimizer allocation, and
step2 includes initial averaging. They demonstrate lower elapsed work in this
fixture, not sustained throughput, a confidence interval or equal-time quality.
The original14.5-hour quality panel was not repeated or pooled into this check.

The integration guard finished in62.3996seconds with
2,888,744,960bytes peak RSS, within180seconds/4GiB RSS,
6GiB sampled MPS and8GiB disk reserve. An independent read of saved case files
rechecked their hashes and exact comparisons before manifest publication.
The public optimized loader then loaded the same representative on MPS in
1.2761seconds, preserving both RNG states and all16,351,859
parameters. Its separate60-second guard finished successfully. That final loader
check performed zero training or quality evaluations.

## Evidence transfer and remaining scope

The source map proves all model math, optimizer, averaging, capture/restore and
cache algorithms unchanged except byte serialization and consistent class imports.
The new check verifies actual dispatch through the new inference, optimizer,
training, capture/validation and cache classes. It reuses the accepted broader
mathematical, checkpoint-resume and cache qualifications through this exact source
mapping; it does not claim to have rerun disk save/resume or a full quality panel.
The prior actual-buffer comparison002 separately checks every real buffer on
CPU/MPS, including mutation rejection and restoration. Its earlier Boolean
mutation harness failure001 remains preserved and is not a model-quality failure.

The optimized implementation is adopted for the raw-byte baseline. The scientific
quality evidence, uncertainty limits and first-fresh checkpoint are unchanged.
Prior full-panel native-versus-byte costs describe the old implementation; do not
present them as newly measured optimized costs or multiply short-fixture ratios
into an asserted full-panel speedup. Parameter count remains16,351,859.

Next examine the existing padding work-count evidence for a bounded composition
optimization. Keep a new candidate outside frozen sources and require declared
numerical equivalence checks before claiming any implementation or quality gain.
