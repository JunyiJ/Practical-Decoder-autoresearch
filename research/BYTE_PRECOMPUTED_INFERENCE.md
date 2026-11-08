# Precomputed byte embeddings for inference

Implemented and qualified on the adopted first-fresh H200 checkpoint using
`research.byte_precomputed_inference_v3`. It materializes all 50,257 token
embeddings, copies the unchanged transformer/readout, and discards the composer
and byte codebook from the inference snapshot. The standard training model and
default checkpoint consumer remain unchanged.

See the [full measured report](tokenizer-evidence/byte-composition-boundary-001/precomputed-inference005/report.md)
for raw-sample locations, numerical limits, isolated memory measurements,
conversion cost, failure recovery, and executable loading/cache examples.

| MPS workload | Batch 1 speedup | Batch 8 speedup |
|---|---:|---:|
| Prefill 96 tokens | 2.56x | 3.62x |
| Cached decode 32 tokens | 2.14x | 2.20x |

These are synchronized medians of seven alternating paired measurements, after
two warmups. CPU prefill improves 2.53–2.90x and cached decode 1.62–1.67x.

All vocabulary embeddings, all 34,278 original validation targets, B1/B8 cache
continuations, resets, and 16-step greedy continuations passed. CPU full-validation
logits are bit-identical; the MPS maximum logit difference is 2.19345e-5, with
no changed validation argmax or tested greedy tokens. Source weights and CPU/MPS
RNG are unchanged. The CPU-built export loads independently on either backend.

Resident model tensors decrease from 112.73 to 108.81 MiB because the dense
49.08 MiB embedding table replaces both composer weights and the current padded
int64 codebook. KV storage is unchanged. The export grows from 62.41 to 108.78 MiB.
CPU precomputation costs 6.45 seconds once; the saved export avoids repeating it.
RSS and MPS allocator measurements are reported separately and are not summed.

The local export is
`research_runs/byte-precomputed-inference003/precomputed.pt`; its trusted receipt
is `tokenizer-evidence/byte-composition-boundary-001/precomputed-inference005/artifact.json`.
Use `load(receipt, expected_source_checkpoint, backend)` and `Session` from the
qualified v3 module. `precompute(model, source_receipt)` and `save` support producing
another snapshot from an admitted raw-byte checkpoint; qualification here is
specific to the recorded checkpoint, FP32, CPU/MPS, batches 1/8 and context 128.

Inference preserves the original public no-grad normalization behavior. The
separate RMSNorm epsilon repair is not included. No training, holdout evaluation,
scientific reselection, or default-consumer change was performed.
