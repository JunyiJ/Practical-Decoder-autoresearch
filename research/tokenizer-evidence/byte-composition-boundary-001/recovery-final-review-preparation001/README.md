# Final reviewer for the recovered execution layout

`research/byte_input_quality_recovery_final_review_v1.py` is an additive,
read-only final reader. It routes original accepted slots00–13 and separately
accepted recovery entries14–17 without modifying either execution namespace.
It imports the exact already checked `contrast` routine from the original final
reviewer (SHA256`088039e6783a322f9baf7ebaa00f617233199405975e7dcbb18ffb80a628c58a`).
No arithmetic implementation, scientific comparison or acceptance threshold was
changed, and no additional bootstrap or model experiment was needed for preparation.

The new reader verifies both plan hashes, the exact immutable failure manifest,
all1,211 original bound records, the RAW audit and corrected-guard reproduction,
identical copied step/diagnostic records, the byte-identical post-training suffix,
and absence of a training-step call in the saved-state recovery driver. It verifies
slot14's parent identity, zero additional training updates, separate recovery
guard, and combined elapsed time within the original slot budget. It does not
invent the original missing terminal RSS observation or call the failed guard
passed. Its report retains that limitation explicitly.

At full-panel completion it additionally requires the recovery controller's
successful STOP and statistics guard, all18 exact acceptance evidence sets,
original/recovery routing, unchanged original scientific design, complete target
losses, all retained RAW/H200 hashes, paired sampler identities and selected rates.
It rebuilds both contrasts with the existing independent routine and checks
the stored numerical results within the original1e-12 tolerance and all decisions
exactly. Resource receipts and total elapsed campaign time are checked alongside
the recovery provenance. Incumbent changes remain outside this reader.

`check.py` completed in0.610333seconds under a60-second CPU guard with256MiB RSS
and8GiB reserve. It audited the actual immutable recovery metadata and rejected
three new in-memory invalid cases: a claimed extra training update, a falsely
passed original guard, and removal of the original-failure flag. It rechecked
the unmodified metadata afterward. No input files were changed, model/weight
payloads loaded, new bootstrap draws made or closed scientific cases replayed.
`result.json` binds the reader and preparation source hashes. This preparation
accepts no complete panel and makes no quality or adoption decision.

Reader SHA256: `152188b55971b1d68ede3a42aca8b0330b03468ab62c1c9347489e8009b54200`.

Only after the live recovery controller finishes the full original panel
successfully and the shared worker lock is available, create a fresh directory
`research/tokenizer-evidence/byte-composition-boundary-001/recovery-final-review001`
and run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m research.byte_input_quality_recovery_final_review_v1 research/tokenizer-evidence/byte-composition-boundary-001/recovery-final-review001/review.json
```

The CLI takes the existing nonblocking research lock, uses a300-second/1GiB CPU
guard with8GiB disk reserve, and writes the new review only after its guard
finishes. Original/recovery panel paths cannot be used as output destinations.
Inspect the finished guard, review and explicit failure-lineage limitation before
any adoption. The full-panel invocation has not yet run.
