# Null-attention prototype: human-stopped, not quality-validated

Stopped by explicit human instruction on 2026-10-27 at 02:52:45 UTC. Do not start research or enable an automation merely by checking out this branch.

This branch preserves the 15-file prototype based on `2f69633da1a2c560952b1738db2c47a0927e9133`. Full production source digest: `c788d543c1451b01c6fcd5c22f05a14fd8a03f7239261b0cb71f4d9cdea18dc4`.

- All 28 new actual CPU geometry/evaluator/training/resume cases passed (338.56 test seconds).
- The 5,973-case regression was interrupted by the human: 3,367 passed IDs in 90 completed batches, plus 8 passed IDs in the interrupted batch. 2,598 IDs are interrupted or unexecuted. This is **not a full-suite pass**.
- No null-attention MPS pilot, research registration, training-quality trial, or adoption occurred.
- The existing controller handled SIGINT and terminated/reaped its worker. Its generic `failed / KeyboardInterrupt()` status records the human stop, not an identified model assertion failure.
- Do not run the fresh CPU driver on the existing evidence. A future explicit restart needs a reviewed continuation that excludes the 3,375 already-passed IDs and preserves the interrupted log/scratch.

The proposed comparison is standard softmax, visible-prefix attenuation `n/(n+1)`, and a fixed zero-score/zero-value null option. The null is never cached or learned. This changes real attention probability mass and is not invariant to a common shift of real logits. The retained backbone comes from the fully reviewed logit-z result; it uses zero objective penalty, RoPE, local values, unit residual gain, SwiGLU, attention scale 0.5 and peak LR 0.01.

Full prerequisite validation and resource review are still required. Pilot/math/cache preparation is not actual MPS evidence. See `research/RESEARCH_HANDOFF.md` on the repository's main branch for the stop snapshot, portable metadata archive, exact partial case lists, findings and restart procedure. Historical closed studies must remain closed; new reproductions require distinct run IDs and new evidence contracts.
