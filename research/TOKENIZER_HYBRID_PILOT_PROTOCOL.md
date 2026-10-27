# Frozen hybrid MPS feasibility pilot

October27,2026. One width64 KDA/KDA/KDA/gated-MLA pilot, seed2991,
LR0.00125,20 updates, cooldown/H200 EMA start10, EMA interval5 (3 snapshots),
batch8/context128 and the explicit `tokenizer_k3_hybrid_nope_v1` model contract.
All geometry, gate, positional and optimizer semantics are described in
[preparation](TOKENIZER_HYBRID_PREPARATION.md). No alternative pilot seeds,
retry, learning-rate selection or quality adoption. Retain any failure.

Purpose: verify actual MPS training, finite checkpoint/optimizer and averaged
export/load, plus full/cached logits at batches1/8 using12 fixed training tokens.
Record each layer's allocated cache bytes. KDA layers store FP32 head32 recurrent
matrices plus three width3 convolution histories; global NoPE gated MLA stores
rank16 latent plus a shared unrotated32-dimensional key per position. Expected
bytes per example:10,496 each KDA layer and24,576 global layer, total56,064.
No explicit positions or learned-position table. No timing comparison here.

The same raw80/10/10 Shakespeare split and GPT-2 BPE contract remain fixed;
test loss is never computed. Sampled online validation is diagnostic and this
20-update feasibility run is excluded from all quality rankings. The full
quality controller and its endpoint/confirmation panel are not yet frozen.

CPU prerequisites:163 tests pass, including FP64 equation/gradient and cached
parity, causal/reset/interleaved requests, exact resume/EMA and old-family
regressions. Initial two FP32 comparisons differed by2.3–3.0e-6 beyond a2e-6
absolute tolerance. Failure logs were retained; strict FP64 parity confirmed
equations/state, and FP32 uses2e-5 relative/1e-5 absolute tolerance. MPS uses the
existing3e-4 relative/3e-5 absolute cache gate; no post-result relaxation.

One local MPS worker/four CPU threads, global and run locks, no CPU fallback,
600-second process limit, existing RSS4GiB/MPS driver6GiB guard,256MiB output
cap and8GiB free reserve plus full256MiB launch headroom. Source, runtime,
helper, protocol, inherited recipe and corpus hashes freeze before launch.
Verify again on completion; no run restart after any terminal state. STOP and
signals prevent further work. Preserve raw and averaged checkpoints and logs.

Width256 is excluded from this storage-bounded pilot, with no width-transfer
claim. After pilot audit, measure the full remaining storage projection for a
finite fair-tuning quality comparison. Do not delete research checkpoints or
ignore the reserve to make a planned round fit.
