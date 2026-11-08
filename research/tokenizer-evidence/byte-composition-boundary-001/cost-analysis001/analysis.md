# What the byte-input resource pilots do and do not establish

This is an analysis of existing accepted pilot measurements and frozen source,
not another numerical test or scientific campaign. `analyze.py` reads the three
closed20-update pilot records, verifies the relevant call paths with Python AST,
and emits the source/measurement hashes and arithmetic in `result.json`. It does
not import the model, access holdout, read pilot quality scores, or modify the
running quality comparison.

## Observed costs

| Quantity | Native | Raw bytes | Nonlexical codes |
|---|---:|---:|---:|
| Trainable parameters |28,492,416|16,351,859|16,351,859|
| Mean seconds/update |0.7531|3.1246|3.1392|
| Sampled training targets/second |1,359.6|327.7|326.2|
| Maximum observed step-end MPS driver bytes |2,501,689,344|3,623,665,664|3,623,665,664|
| H200 archive bytes |114,002,715|65,445,293|65,445,357|

The composer has42.61% fewer parameters and smaller saved weights, but the
implemented pilot is4.15x slower per raw-byte update and has44.85% greater
maximum observed step-end driver allocation. Driver samples are not allocator
peaks or whole-machine memory. The20-step pilot does not establish1200-step
throughput, final quality or serving latency. Raw and nonlexical training times
are close, which is consistent with their shared implementation; this is not a
causal timing decomposition.

## Concrete implementation overhead in the measured path

Both composed variants retain50257x130 int64 code IDs and50257 int64 lengths:
52,669,336bytes. Every `ByteInputModel.validate()` recomputes both complete
buffer hashes. `_buffer_digest()` transfers chunks to CPU, views them as uint8,
converts each byte to a Python-list element, reconstructs `bytes`, then hashes.
There is no unchanged-buffer shortcut in this source.

One ordinary `Context.step()` reaches model validation at least six times:

1. Entry `Context.check` -> state `live` -> optimizer `validate`.
2. `forward_train` -> model `validate`.
3. Explicit optimizer gradient validation.
4. Optimizer `step` -> optimizer `validate`.
5. Optimizer `zero_grad` -> optimizer `validate`.
6. Exit `Context.check` -> state `live` -> optimizer `validate`.

Each optimizer validation calls model validation. Thus the composed update
logically reads/converts/hashes at least316,016,016codebook bytes, excluding
other buffers and diagnostics. Across the frozen14,400composed updates that is
at least4,550,630,630,400logical bytes. This is a static work count, not measured
hardware-transfer traffic, peak memory, or a claim that hashing explains a
particular fraction of elapsed time. Shared-memory MPS and CPU copies also
prevent interpreting it directly as PCIe traffic.

`Context.check` additionally rehashes the complete CPU train/validation ID lists
and revisits the accumulated clipping history. These shared checks affect both
native and composed timings. The composer also performs real extra arithmetic:
six padded-window filter banks, highway/projection layers and fixed-order byte
gradient reduction. None of these costs has been isolated by a profiler here.

The extra codebook is itself larger than the48,562,228bytes saved in FP32
parameters. Parameters plus this extra buffer alone are4,107,108bytes larger
than native parameters; this partial accounting excludes optimizer state,
activations and common buffers. Smaller parameter count is therefore not a
sufficient inference about live memory or speed.

## Consequence for the research decision

The fixed comparison answers quality after equal sampled training exposure,
with balanced rate tuning and fresh seeds. It does not answer quality at equal
wall time. Report the observed implementation costs alongside any eventual
quality result; do not describe fewer parameters as a demonstrated runtime win.
The present evidence supports neither adoption nor rejection on quality.

The next justified performance investigation, after the frozen panel, is to
measure codebook serialization/hash validation separately from neural compute.
A narrow optimization candidate is direct contiguous byte serialization in
place of Python byte lists while retaining every full-content hash check.
Validate byte-for-byte digest identity and time the same existing payloads;
this would not require another learning-rate sweep or fresh quality panel merely
to establish serialization equivalence. An actual integration still needs its
own changed-source scope and numerical equivalence evidence. Do not silently
substitute version-only checks, omit mutation rejection, or patch pinned sources
under the running controller.

This note makes no external literature performance claim and admits no new
architecture hypothesis. The existing protocol's literature basis remains in
`../quality-evidence001/evidence.md`; published quality gains do not quantify
this local implementation overhead. Scientific gates, roles, data, horizon and
the incumbent remain unchanged.
