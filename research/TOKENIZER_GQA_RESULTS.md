# Tokenizer GQA comparison

The [finite protocol](TOKENIZER_GQA_PROTOCOL.md) compares MHA and half-KV-head
GQA on the adopted GPT-2 tokenizer backbone. The previous LR round adopted
0.0025; this round gives each family equal tuning at0.0025/0.00125 before fresh
confirmation. No GQA quality result or adoption is established yet.

## Prerequisites completed

Source `dbc4d9f` passed140 targeted CPU tests in287.11seconds. The20 initial new
checks are included in that total. Checks include exact MHA initialization,
forward, gradients and optimizer-step parity with the adopted implementation;
GQA against an expanded-KV reference and its input gradients; full/cached causal
behavior, uneven/nonzero offsets, separate/reset state, parameter/cache sizes,
strict contract rejection, real training/resume and EMA/tokenizer loading.
This is targeted regression, not the entire historical repository suite.

Four serial40-step MPS pilots (MHA/GQA, widths64/256, seed2691) completed and
passed saved-artifact review, including finite tensors, optimizer steps,
21 EMA snapshots, parent hashes, full validation coverage, replayed training
sampler state and untouched test RNG. Their scores are feasibility data only.
The separate new architecture contract fixes the inherited model semantics and
implements the0.5 score multiplier in GQA; historical contracts stay intact.
MLA and hybrid support are still future work.

MPS full/cached checks passed at batches1/8 and widths64/256. Paired pilot timing
completed with the frozen32-token prefill/64-step teacher-forced decode workload,
two warmups and eight repetitions per arm and batch. Actual KV allocation is
50% lower for GQA. Timings use two resident models and are feasibility evidence,
not a trained quality/cost adoption result or a total-memory reduction claim.
Fresh selected checkpoints must pass the final cost review after quality training.

Projected20-trial raw/H200 output is5,223,069,680bytes (about4.86GiB), or7.30GiB
with50% headroom. The measured free space passed that projection plus the8GiB
reserve; the round cap is8GiB. Successful temporary pytest fixtures were cleared
after preserving logs; research checkpoints and failure evidence remain retained.

An initial lint check found an unused benchmark import; it was removed after
the unchanged-source CPU regression, and final lint passed. No training or
numerical prerequisite failure occurred. [Pilot and timing review](tokenizer-evidence/gqa-pilots-001/review.json).

## Interpretation and continuation

The20 quality trajectories use one serial MPS worker, eight screen cells and
twelve fresh confirmation cells. Frozen per-family LR choices precede confirmation.
GQA may qualify independently by width only after the+0.01-nat quality margin,
cached correctness and measured cache-storage route pass. Report latency tradeoffs
separately. The known late-validation degradation remains unresolved: this round
tests architecture under the inherited3000-update budget. A horizon/schedule
comparison remains active in the backlog alongside MLA, the Kimi hybrid and other
architecture families. The test holdout stays closed.

## Reviewed quality trajectories

Only completed, artifact-reviewed trajectories appear here. Quality eligibility
alone does not adopt GQA: cached correctness, measured inference cost and final
review are also required. The test holdout remains closed.

| Stage | Family | Width | Seed | LR | Raw CE | H200 CE | Training seconds |
|---|---|---:|---:|---:|---:|---:|---:|

Paired GQA minus MHA H200 CE (fixed-LR screens; selected-recipe confirmation):


Screening incomplete; no frozen family LR choices yet.
