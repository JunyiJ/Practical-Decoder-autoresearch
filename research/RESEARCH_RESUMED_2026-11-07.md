# Research resumed by the human, November 7, 2026

The new AUTONOMY AND CONTINUATION instruction explicitly resumes research and
authorizes bounded autonomous correction of new execution, harness and
implementation defects. The earlier stop handoff remains historical evidence.
Closed experiments, used roles, scientific comparisons and acceptance criteria
remain fixed. No quality improvement or adoption is claimed by this continuation.

The current work is the independent full-codebook and actual-model proof
specified by the previous handoff. Read
[the proof source guide](byte_input_proof_v1/README.md) and
[the current progress record](tokenizer-evidence/byte-composition-boundary-001/resume-model-proof001/progress.json).

Source inspection established that `byte_input_model_v1/optimizer.py` never
reads the child optimizer state. Its preflight could therefore accept corrupted
momentum/Adam state before mutating parameters. New `byte_input_model_v2` adds
state/ownership/alias/step validation before decay and either child step.
All five original v1 source hashes were reverified unchanged. This is a source
finding and prepared correction; numerical rejection/update verification is
still pending. It does not reopen any old fixture.

Independent reference sources now implement public byte reconstruction, the
length-specific Feistel mapping, Conv1d-based byte composition, explicit primary
CE, and expected optimizer grouping/decay. Streaming coordinate recording and
saved-evidence review avoid giant Python lists and preserve full comparisons.
Prospective cases cover D256/V50257, B8T128, three arms, CPU/MPS and invalid
transactions. No candidate/reference function has been called.

The original source consistency gate passed in 0.028 seconds with 27,770,880B
peak RSS. It compiled syntax, checked source pins and static ordering, and
calculated a conservative 12,692,784,780B uncompressed coordinate bound. The
prospective 16GiB proof output cap plus 8GiB reserve and 2GiB headroom requires
27,917,287,424B; observed free space was 34,376,445,952B. No storage deficit or
cleanup trigger was found. Runtime graph/allocator/time costs remain unqualified;
this arithmetic does not admit a numerical run or a scientific panel.

Next complete the actual driver, invalid-transaction implementations and guarded
once-only entry. Then qualify new exclusive initializer/code-key roles, complete
source/runtime pins, implemented live memory/time/output forecasts, and independent
static admission before any candidate/reference import. The genuine B8T128
forward/backward remains required. No roles have been assigned or drawn. The
original source-consistency gate is closed and must not be replayed.

No research worker is running at this checkpoint. Existing paused research
automations remain unchanged; the current autonomous goal owns continuation.
Independent maintenance changes and the human's unstaged AGENTS.md edits are
excluded from this work. Native incumbent, holdout, TOP and all historical
failed studies remain unchanged.
