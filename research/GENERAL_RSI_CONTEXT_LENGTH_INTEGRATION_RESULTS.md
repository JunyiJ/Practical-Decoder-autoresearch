# Context-length endpoint, evaluation and generation integration

The changed paths passed. Complete context256 state with active time-EMA restored
and replayed exactly, including cooldown LR, optimizer, sampler, CPU/MPS RNG,
clock, history and EMA. The checkpoint followed26 updates with3 EMA snapshots;
the reference/replay ended with4. Total numerical work was28 optimizer updates.
No prior pilot or historical model qualification was rerun.

Coverage checks passed1,665 synthetic targets across eight boundary-length
patterns at both contexts. The128 adapter matched the original chunk iterator.
Both contexts scored the same653 targets of the predeclared first training
document, including EOT. Maximum per-token CE difference against independent
FP64 CPU logsumexp arithmetic was1.498322348e-6, below the frozen1e-5 tolerance.
This engineering arithmetic check does not reuse the composer qualification gate.
Three additional aggregation checks verified source sums and document-ID remapping.

Both RAW and EMA inference exports matched all88 tensors exactly. Wrong-context
inference loading was rejected. All24 first fixed engineering outputs,1,474new
tokens, remain in the original journals. Generation took59.041196667seconds.
Scoring/oracle work and generation preserved complete training state; inference
weights also stayed unchanged. These outputs are engineering evidence and are
excluded from the future144 scientific outputs. No quality claim or rating-based
selection was made. Neither validation nor test was opened in this integration.

The run took79.890286625 guarded seconds. Peak RSS was1,544,421,376bytes and
peak observed MPS driver allocation2,501,689,344bytes. The active-EMA restart
was445,008,823bytes; write/read/hash verification took1.290205291seconds. Four
retained payloads total1,118,015,524bytes. All5,219 frozen source bindings and
all payload receipts passed the completion audit. No failure occurred.

The prospective serial worker reuses the accepted clock, recovery, publication,
exact retirement and cumulative-budget logic. Seven unchanged utility functions
are AST-verified against the native-horizon worker. Its bindings use the new
context, scorer and uncached generator. SIGTERM also reaches the per-token stop
check. The complete new quality contract is in
`GENERAL_RSI_CONTEXT_LENGTH_QUALITY_DESIGN.md`. Admission and fresh roles must
be frozen before scientific execution; none of the engineering scores selects
an arm or a seed.

Evidence: `tokenizer-evidence/general-rsi-context-length-integration-001/`.
Plan SHA256: `02911661652891a3da8972c59a76dd7852d0215ac4b5e871d11fd8c7debf1a68`.
