# Training byte-row sparsity

The pinned GPT-2 training split contains267688tokens and uses67 embedding byte rows
including boundary rows. The first1024tokens use54. Across261 complete, disjoint
1024-token windows, counts range50–63 with median58. The final partial window is
reported separately and excluded from that summary. Token hashes, tokenizer assets
and corpus bytes were verified. No validation/holdout was encoded and no model or
random stream ran in this analysis.

These counts motivate an active-row gradient computation, but do not demonstrate
runtime savings or equivalence. The subsequent active-byte-runtime001 comparison
found two numerical mismatches and did not adopt its prototype. This analysis does
not change the tokenizer, token sequence, output vocabulary or frozen quality panel.
