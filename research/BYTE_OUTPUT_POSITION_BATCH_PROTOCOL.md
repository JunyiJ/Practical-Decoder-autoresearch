# Position-bounded batching: new execution optimization pilot

The completed chunk-pilot001 passed all numerical checks but its full table
forward plus recomputed VJP took9.637466seconds. This is an execution-cost
finding, not a negative quality result. The exact source primitive
`byte_embedding_v1.py` limits total positions to1024*130, not rows to1024.
The prior prototype's row cap was inherited conservatively from the input
composer. Short-token batches can therefore be enlarged without altering
the primitive, parameters, code rows, mask, objective, or gradient formula.

New isolated source `byte_output_chunk_pilot_v2.py` replaces only grouping:
for each effective code length L, maximum rows=floor(133120/L). All token IDs
participate once, in the same within-length order. No initializer, learned
coefficient, code permutation, approximation, sampled output or stale cache.
Different reduction/batch sizes may change rounding; numerical equivalence
must be checked rather than assumed. This is an execution optimization of the
same candidate, not a replacement scientific hypothesis.

Retain every case, tolerance, CPU float64 reference, receipt and invariant from
`BYTE_OUTPUT_CHUNK_PILOT_PROTOCOL.md`. Add a required batch-boundary case: the
first1025 vocabulary IDs with actual code length6, compared on MPS under old
1024-row chunks versus one new1025-row batch. Compare outputs and every parameter
gradient at atol4e-6/rtol4e-5, with the original deterministic upstream normalized
by1025. This specifically exercises the newly permitted batch size.

Then measure exactly one full-vocabulary forward and recomputed VJP with larger
position-bounded batches. Keep180seconds,4GiB RSS,2MiB output,10GiB reserve
plus output allowance, shared nonblocking lock and no weight outputs. Use new
directory `tokenizer-evidence/byte-output-chunk-pilot-002`, pin original successful
result/source plus this source/protocol, and retain the first pilot unchanged.
No previous command is rerun. Compare descriptive single-pass times, with
warming/measurement variability acknowledged; do not claim sustained throughput.
If the cost is still unattractive, reject this implementation for a quality
campaign rather than launch an expensive panel or change scientific controls.
