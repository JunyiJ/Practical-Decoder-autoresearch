# MLA score-transfer runtime

Only the named `tokenizer_mla_score_transfer_v1` policy permits scalar0.5 or1.0
on the exact adopted D256 MLA/SwiGLU672/RMSNorm/grouped-embedding-v2 backbone.
Both terms of the content-plus-RoPE attention score use the scalar. No parameters
or optimizer groups change. Other fields retain their inherited strict values.

The runtime resolves unchanged modules from `ffn_structure_v1` and its immutable
predecessors under the new namespace. Only block_norm and tokenizer_architecture
are copied to extend the policy and validate the actual constructed scalar.
Architecture, checkpoint and per-request cache signatures include score_scale;
identical tensor shapes do not make the two arms interchangeable.

This is prerequisite integration, not a frozen pilot or quality controller.
See `research/TOKENIZER_MLA_SCORE_PREPARATION.md` for the proposed comparison.
