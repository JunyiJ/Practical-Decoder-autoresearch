# RoPE positional-table repair (October 24, 2026)

The human requested immediate removal of the unused learned positional table.
Previously GPT allocated `transformer.wpe` even when RoPE bypassed it. That cost
`block_size * dim` dormant parameters and consumed random initialization draws
before the transformer blocks, so changing context length also changed active
initial weights at the same seed.

New RoPE models allocate only the character embedding. Learned absolute-position
models (`rope != 1`) keep the position table, parameter order, and initialization.
New RoPE active parameters and CPU RNG state must match exactly at contexts
128/256/512 under the same seed. Deterministic frequency buffers and causal masks
still scale with context. Model forward math is unchanged; fresh RoPE training
initialization changes and therefore requires a new source contract and run.
New checkpoint metadata records parameter layout version 2; model math version
remains 2 because the removed table never participated in RoPE forward math.

Historical checkpoints are never rewritten. `load_model_weights` supports their
inference/evaluation by checking saved RoPE mode and the old table's shape,
dropping exactly that key in a detached in-memory state dictionary, warning, and
strictly loading every active tensor. Missing/extra active keys remain errors.
Historical optimizer continuation is explicitly rejected under this source:
old optimizer parameter groups include the removed table. Use the original
frozen source to continue an old run. New checkpoint continuation is tested.

Stage2F remains on source5dd790d51f7ee30c44f3633c00648d227ab0f1e8, including
its remaining trials, historical parameter counts and initialization contract.
A temporary boundary pause permits serial correctness tests without competing
training. The repair is developed and published separately; the primary
checkout is not updated until the frozen run is complete and verified.

The test fixture `tests/fixtures/gpt_before_rope_table_fix.txt` preserves that
historical GPT implementation for old/new forward equivalence tests. It is not
executed by training and must not be used to reopen historical studies.

Validation: all 552 repository tests passed. Three additional serial MPS checks
(MHA, GQA, MLA) each completed three optimizer updates, exact-logit checkpoint
reload, and a further optimizer update with finite parameters. These are
correctness checks, not quality or runtime measurements.
