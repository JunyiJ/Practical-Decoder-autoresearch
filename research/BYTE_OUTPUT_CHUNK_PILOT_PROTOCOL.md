# Exact chunked output-composer VJP: bounded prerequisite

Frozen before execution. The analytical result in
`tokenizer-evidence/byte-output-composition-cost-001/result.json` measures
42.5752698 billion forward MACs with73 length-grouped chunks. Fixed padding
costs250.81509705 billion. Test the feasible grouped approach only, using a
new prototype outside all frozen namespaces. This is a mechanical prerequisite,
not a training run or quality screen; no scientific coefficients or seeds are
selected and no old failed fixture is replayed.

Exact first-order strategy: compose the entire output table without a graph;
the eventual model computes its gradient with respect to that table; recompute
each composer chunk and apply its corresponding upstream gradient immediately.
The chain rule sums every chunk's parameter contribution before an optimizer
step. This avoids retaining all chunk graphs. It adds one composer forward
pass, so nominal dense work is4×forward MACs, excluding other operations.
For this prerequisite supply a fixed analytic upstream, not a model loss.
No stale rows, skipped vocabulary, sampled softmax or parameter updates.

Use slot10 selected RAW and its exact receipt as in
`BYTE_OUTPUT_MOMENT_PROTOCOL.md`; no initializer or RNG draw is required.
Copy its18 named composer parameters. Reconstruct all codebook rows from the
pinned tokenizer, including begin/end/pad/special symbols, and verify both
packed code and length hashes from the original config. Nonlexical control
cost is only structurally estimated here; it is not numerically executed.

Proposed functional math follows the original unfold/linear implementation and
uses its deterministic MPS byte-embedding primitive. Reference math independently
uses CPU float64 convolution kernels with weight layout(channels,15,width).
Take the first ID of each distinct actual code length, plus repeated IDs0,0 and
special50256. This fixed shape/boundary fixture is chosen without output values.
Use upstream ((id*256+channel) mod17−8)/(8*row_count). Compare proposed grouped
CPU FP32 and MPS FP32 outputs and all18 parameter gradients to the padded CPU
float64 reference, retaining inherited component tolerances atol4e-6/rtol4e-5.
On MPS also compare retained-graph versus recomputed-chunk VJPs on the same
fixture. These cases cover boundaries, repeated rows and all observed lengths;
they do not prove every possible input or a complete training integration.

After these checks, execute exactly one full50257-row MPS no-grad table forward
and one full chunk-recomputed VJP with the same upstream formula divided by50257.
Require finite table and all gradients, exact row coverage, unchanged source,
RAW weights and CPU RNG. Record synchronized forward/VJP elapsed time and MPS
allocation observations plus process peak RSS. No full-model timing or quality
claim; synthetic upstream and one execution do not establish sustained training
throughput. No optimizer, loss scoring, training/validation/holdout tokenization,
weight export or fresh random roles. Do not benchmark the infeasible fixed-pad
whole vocabulary just to demonstrate its cost.

One new attempt at `tokenizer-evidence/byte-output-chunk-pilot-001` under the
shared nonblocking research lock:180seconds wall,4GiB RSS,2MiB output, zero new
weights, entry10GiB reserve/headroom plus full output cap. Local host memory
must have room for this bounded process; do not interrupt other workers. Preserve
failure evidence and use a separate correction attempt only for demonstrated
implementation/environment defects, with all cases and tolerances fixed.
If the prototype fails numerical correctness, do not report resource numbers
as qualification. If it passes, a full-model/optimizer/checkpoint integration
and complete cost forecast remain required before a fresh quality panel.
