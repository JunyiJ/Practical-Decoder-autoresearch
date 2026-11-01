# SwiGLU-product normalization runtime

This additive runtime resolves unchanged modules through the frozen MLA score
runtime. Its explicit `tokenizer_mla_swiglu_product_rms_v1` policy compares
`ffn_hidden_norm: none` with `rms` and requires
`ffn_hidden_norm_policy: mla_swiglu_product_rms_672_v1` in both arms.

Both use the adopted D256 MLA / score0.5 / SwiGLU672 / SiLU / block RMSNorm /
final affine LayerNorm / grouped-embedding v2 architecture. RMS is per token,
after the elementwise gate/up product and before down projection, epsilon1e-5,
affine gain initialized to1, no bias. Four672-wide gains add2,688 parameters.
They inherit the existing AdamW vector rule at base LR with independent decay;
shared optimizer groups and random initialization remain identical.

Constructed module validation, checkpoint signatures/config/tensor inventories
and request cache signatures distinguish the arms. Legacy policies continue to
reject this extension. No frozen source is modified and no scientific panel is
selected by this runtime. Prerequisite evidence and next work are recorded in
`../TOKENIZER_FFN_NORM_PREPARATION.md`.

The additive validation, artifact-acceptance, output and inference adapters reuse
the prior bounded primitives under this policy. The output sink retains448MiB raw,
112MiB H200 and16MiB trajectory-metadata caps. Parent acceptance checks actual
optimizer/RNG/EMA state and independently recomputes every validation target.
Inference binds none/RMS model identities and exact28,492,416/28,495,104 parameter
counts; both cache footprints remain196,608B per batch item. No ledger or new
scientific dispatcher is introduced by these prerequisite adapters.
