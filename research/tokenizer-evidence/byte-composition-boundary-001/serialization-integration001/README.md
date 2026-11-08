# Minimal direct-byte-hash model candidate

`research/byte_input_direct_hash_v1.py` provides `DirectByteHashModel`, an
unqualified subclass of the frozen model-v3 class. It overrides only
`_buffer_digest`, replacing Python-list byte serialization with direct NumPy
byte serialization. Construction, model math, parameter ordering, complete
buffer hashing, chunk boundaries, device transfers and validation call frequency
are inherited unchanged. No frozen file or active worker was modified.

This follows the measured bottleneck in `cost-analysis001` and the accepted
CPU-only component measurement in `serialization-analysis001`. That measurement
found identical hashes and approximately 14.69x faster serialization/hashing
for the synthetic code-ID buffer. It did not establish a model or GPU speedup.
The existing measurement is reused; it was not replayed.

A standard-library AST check verified that the candidate defines exactly one
method and that its body is the original body with exactly the previously
measured expression substitution. Syntax compilation passed without importing
Torch, constructing a model, allocating GPU memory, or executing any training.
The inherited model source matches the original benchmark's recorded SHA256:

- Inherited model: `2a4187c9bf101b3886fa3f36894ffc809d47078812ca789257a2eb3238d6b460`
- Candidate: `3d3fbc588bcf3552657f0e4a0bf8ef559f28e64b96519655f1f84bede0cdcc03`

This is source preparation for a focused comparison, not full training-context
integration or runtime acceptance. The existing optimizer and checkpoint
consumers require the exact model-v3 class and will reject this subclass. Do not
weaken those guards or inject the candidate into the frozen quality panel.
Any later execution must bind the candidate and its inherited dependencies.

After the current worker allocation is available, the next useful check is
equality of full buffer digests, preservation of content-mutation rejection and
descriptive timing with the actual CPU/MPS model buffers. An integrated model
comparison must additionally establish unchanged outputs and training semantics
under its own source binding before any future training consumer uses it.
Neither this source change nor byte identity alone warrants a new learning-rate
sweep or a claim of better model quality. No new scientific campaign, numerical
role, execution permission exception or adoption is introduced here.
