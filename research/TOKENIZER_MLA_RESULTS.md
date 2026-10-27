# Tokenizer MLA results

No quality trials yet. The completed horizon round adopted the shorter GQA
recipe; this next round evaluates an MLA architecture using a fresh MHA control.
See [prospective protocol](TOKENIZER_MLA_PROTOCOL.md).

MLA support passed63 CPU checks (22.82seconds), then one source-frozen60-step
width256 MPS feasibility trial (11.29training seconds). Saved-artifact review
passed exact config, finite optimizer/weights,60 optimizer steps,3 EMA snapshots,
checkpoint hashes, training sampler replay, untouched test RNG and validation
arithmetic. This pilot's validation values are sanity data, excluded from quality
selection. [Pilot evidence](tokenizer-evidence/mla-pilot-001/result.json).

Full/cached FP32 MPS logits passed at batches1/8 (maximum absolute differences
3.34e-6/1.67e-6). Actual allocated latent-plus-rotary cache was196608/1572864bytes
at context128. These checks establish implementation feasibility, not comparative
quality or latency. [Cache checks](tokenizer-evidence/mla-pilot-001/cache-check.json).

The raw/H200 checkpoint pair uses559170172bytes. A new ten-trial quality storage
projection includes the larger preserved width256 MHA pair,15% overhead and a
separate8GiB reserve. [Resource review](tokenizer-evidence/mla-pilot-001/resource-review.json).
Completed temporary pytest fixtures were cleared only after saving logs and a
manifest; no research checkpoints were removed. All prior failures remain
preserved. Quality controller and MLA-specific benchmark integration are next;
no quality launch until their tests and source freeze pass.
