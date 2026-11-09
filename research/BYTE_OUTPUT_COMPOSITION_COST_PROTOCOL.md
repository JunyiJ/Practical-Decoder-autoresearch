# Full-vocabulary composition: analytical cost check

Frozen before execution. Existing tokenizer/source analysis only: no models,
weights, gradients, random roles, data losses or quality selection. Question:
can the existing composer be considered for an independent output lexical
component without mistaking its input workload for full-vocabulary cost?
The supporting and contrary local/primary evidence is in
`BYTE_OUTPUT_LEXICAL_REVIEW.md`; this check qualifies feasibility only.

Use all50257 GPT2 IDs from the pinned tokenizer/version/vocabulary. Regular code
length is byte length+2 boundaries; special50256 has length3. Verify the packed
uint16 length hash against the original codebook identity in
`byte-residual-readout-001/quality001/00/config.json`. No corpus tokenization
or holdout access is needed. Read composer WIDTHS/CHANNELS as AST constants,
verify the known architecture, and bind the composer/model/codebook source hashes.

Compare two exact-computation layouts: fixed length130 with chunks<=1024, and
groups by effective length max(6,code_length), each chunk<=1024. Restore original
ID order in any future implementation. Length grouping is an analytical proposal;
its outputs/gradients and backend speed are not established by this cost check.
The existing mask removes windows beyond actual lengths, so trimming excess
padding is algebraically plausible, subject to finite numerical verification.

For each layout count dense multiply-accumulates (MACs), not FLOPs:
sum_chunks n*[sum_w (L-w+1)*15*w*channels_w +2*525^2+525*256].
Forward plus dense input/weight-gradient matmuls has nominal cost3×forward.
Exclude nonlinearities, masks, reductions, embedding-gradient accumulation,
validation, optimizer work and backend overhead; this is not elapsed time.
Compare to the existing1024×256 by256×50257 output matmul. If composition can
be added to W before that matmul, there need be no second full logit matmul;
that algebra does not settle initialization, optimizer or model quality.

Report exact vocabulary length histogram, chunk count, maximum chunk MACs,
summed logical sizes of embedded bytes, unfolded windows, convolution values,
pooled features and composed output table in FP32. These are tensor inventories,
not measured RSS or a guarantee that all are live simultaneously. Report the
largest chunk inventory separately. Chunking alone does not release saved
backward activations when every chunk graph is retained until one final backward.
Recomputation could reduce memory while adding work; stale cached rows change
the gradient and are not an equivalent optimization.

Report additional composer parameters (725235) and algebraic two-weight-file
payload increments for six candidate/control runs in a prospective nine-run
panel. Also report minimum complete float32 raw/H200 tensor payload for three
incumbents plus six augmented models. This excludes any optional gate, metadata,
serialization overlap and resumable optimizer states, and is explicitly not an
admission forecast. A measured full-model pilot and complete high-water forecast
are required before a panel is frozen; no storage blocker is declared here.

Check total row coverage, codebook hash, nonlexical length-match applicability,
parameter arithmetic, and summed histogram versus direct per-row cost. One new
attempt at `tokenizer-evidence/byte-output-composition-cost-001`, under the
nonblocking shared research lock:120seconds,1GiB RSS,1MiB output, zero weights,
entry10GiB reserve/headroom plus full output cap. Preserve any failure; recovery
must keep these requirements fixed. Preparation and arithmetic are not training
or numerical exploration credit.
