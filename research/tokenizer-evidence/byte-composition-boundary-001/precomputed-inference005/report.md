# Precomputed byte embeddings: verified inference and measured tradeoffs

The v3 snapshot materializes all 50,257 byte-composed inputs and owns independent copies of the existing transformer and readout. It retains neither composer nor codebook. The original selected H200 checkpoint and default consumer are unchanged. This is an inference implementation result, not a new quality experiment.

## Verification

| Backend | Vocabulary rows checked | Validation targets | Largest logit error | Largest CE error | Changed validation argmax |
|---|---:|---:|---:|---:|---:|
| CPU | 50,257 | 34,278 | 0 | 0 | 0 |
| MPS | 50,257 | 34,278 | 2.19345093e-05 | 1.71661377e-05 | 0 |

The CPU-built exported table was reloaded on both CPU and MPS. Verification includes all vocabulary rows against independent 1,024-row composition (construction used 256-row chunks), every original validation target, B1/B8 cached prefill96 plus decode32, exact reset/replay, and 16 greedy autoregressive steps. Invalid IDs, dtype, cache position, context overflow, training, and table mutation are rejected. Source weights, gradients and CPU/MPS RNG are unchanged.

Tolerances were fixed before execution: logits/embeddings atol 3e-5 and rtol 3e-4; maximum per-target CE difference 3e-4 and mean absolute CE difference 3e-5. All comparisons are against the same backend’s current public no-grad inference. The separate RMSNorm default-epsilon repair is not included or validated by this work.

## Latency

Four CPU threads, FP32, two warmups and seven measured repeats; alternating paired order and MPS synchronization. Reset, loading and precomputation are excluded from steady-state latency. Decode uses the same fixed 32 continuation tokens in both arms; greedy generation is verified separately. Both paths produce the full vocabulary logits. Raw samples remain in the result JSON files. CPU latency comes from the completed sandboxed CPU pass; GPU latency and isolated memory use approved host execution because MPS is unavailable in the sandbox. Each speedup compares two arms in the same process/environment.

| Backend | Batch | Workload | Current median ms | Precomputed median ms | Speedup | Precomputed min–max ms |
|---|---:|---|---:|---:|---:|---:|
| CPU | 1 | full128 | 25.692 | 9.730 | 2.640× | 9.597–9.895 |
| CPU | 1 | prefill96 | 20.967 | 8.284 | 2.531× | 7.739–33.225 |
| CPU | 1 | decode32 | 113.932 | 70.377 | 1.619× | 67.855–111.495 |
| CPU | 8 | full128 | 179.670 | 59.822 | 3.003× | 58.531–62.492 |
| CPU | 8 | prefill96 | 137.720 | 47.415 | 2.905× | 45.460–65.257 |
| CPU | 8 | decode32 | 264.541 | 158.279 | 1.671× | 153.748–190.074 |
| MPS | 1 | full128 | 26.259 | 9.294 | 2.825× | 9.127–10.071 |
| MPS | 1 | prefill96 | 22.685 | 8.861 | 2.560× | 8.526–9.346 |
| MPS | 1 | decode32 | 540.136 | 252.419 | 2.140× | 240.664–345.671 |
| MPS | 8 | full128 | 147.406 | 41.134 | 3.584× | 38.962–57.289 |
| MPS | 8 | prefill96 | 108.503 | 30.000 | 3.617× | 28.985–30.963 |
| MPS | 8 | decode32 | 601.756 | 273.962 | 2.196× | 236.612–338.321 |

## Conversion and storage

- CPU conversion: 6.448 seconds; conversion-process peak RSS 685.11 MiB (includes original loading and simultaneous source/snapshot).
- MPS conversion: 5.206 seconds; conversion-process peak RSS 750.14 MiB (includes original loading and simultaneous source/snapshot).
- Saved H200: 65,445,293 bytes (62.41 MiB). Precomputed export: 114,067,075 bytes (108.78 MiB); increase 46.37 MiB (1.743×).

## Isolated memory

Each backend/arm runs in a fresh process. Tensor storage counts unique live parameter and buffer storage; it is the precise architectural comparison. RSS also includes Python, libraries and allocator retention. MPS current allocation and driver allocation are point observations, not peaks. Process peak RSS includes loading and warmups and must not be interpreted as an isolated inference peak. CPU and MPS share physical memory, so these measures must not be summed.

| Backend | Arm | Model tensor MiB | Loaded RSS MiB | Loaded MPS current MiB | Loaded MPS driver MiB | Process peak RSS MiB |
|---|---|---:|---:|---:|---:|---:|
| CPU | current | 112.73 | 555.52 | — | — | 1139.81 |
| CPU | precomputed | 108.81 | 513.27 | — | — | 874.69 |
| MPS | current | 112.73 | 727.80 | 113.83 | 1040.47 | 779.81 |
| MPS | precomputed | 108.81 | 638.78 | 110.66 | 1040.47 | 677.59 |

Prefill/decode observations with live logits and idle KV caches are retained in each memory result. KV storage is unchanged: 196,608 bytes at B1 and 1,572,864 bytes at B8. The largest B8 prefill output itself contains 154,389,504 bytes; embedding precomputation does not remove that output cost.

The dense table is 51,463,168 bytes. It replaces 2,900,940 composer-weight bytes plus 52,669,336 codebook-buffer bytes in the current implementation. This predicts a 4,107,108-byte resident tensor reduction, despite the larger export. A future compact-codebook implementation could change this memory tradeoff.

## Use

Run from the repository root with the project environment. The caller should own the shared research lock and appropriate resource guard, as for the current consumer.

```python
import json
from pathlib import Path
from research.byte_precomputed_inference_v3 import load, Session

evidence = Path("research/tokenizer-evidence/byte-composition-boundary-001/precomputed-inference005")
plan = json.loads((evidence / "plan.json").read_text())
receipt = json.loads((evidence / "artifact.json").read_text())
model = load(receipt, plan["checkpoint"], "mps")
# token_ids: int64 MPS tensor, batch 1 or 8, context at most 128
logits = model(token_ids)
session = Session(model, batch=1)
prefix_logits = session.logits(token_ids[:, :96], start_pos=0)
# Append contiguous chunks; call session.reset() before a new sequence.
```

The artifact is local under `research_runs/byte-precomputed-inference003/precomputed.pt`; Git retains sources, receipts and measurements. Loading verifies the full artifact hash and expected source-checkpoint identity. Default checkpoint consumers and the training model remain unchanged.

## Preserved failure and scope

Attempt001 stopped on the inherited normalization contract because v1 disabled parameter requires_grad flags. Attempt002 retained the original flags and passed CPU numerical/cache/generation checks, then detected a loader RNG side effect: torch.manual_seed also seeded MPS despite the CPU-only RNG scope. Attempt003 uses the CPU default generator directly and retains no-grad execution at public APIs. Both failed sources, plans, guards, artifacts and tracebacks remain preserved. Attempt003 completed all CPU verification and timings, then its MPS monitor crashed in allocator initialization before model execution. Attempt004 exposed the underlying sandbox restriction on the first MPS allocation: is_available=False in sandbox, True with successful synchronized allocation on the host (macOS26.3). Attempt005 imports the completed CPU receipts unchanged, explicitly checks availability, initializes MPS on the main thread, and runs the remaining workers through approved host execution. No model math, tolerance or case was relaxed. These measurements are one local machine/runtime/checkpoint benchmark, not a CUDA, quantized, long-context or production-throughput claim.
