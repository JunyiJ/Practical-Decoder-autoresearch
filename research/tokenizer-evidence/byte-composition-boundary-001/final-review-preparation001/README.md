# Final saved-data review preparation

The frozen quality001 protocol requires independent final acceptance before
adoption. `research/byte_input_quality_final_review_v1.py` implements this
remaining consumer outside frozen sources. It does not train, run inference,
change selection, draw new scientific roles, or modify the incumbent.

It requires the successful full-panel STOP and finished statistics guard, verifies
all 18 exact acceptance evidence sets and retained RAW/H200 hashes, and reuses
those accepted per-run numerical checks. It independently reconstructs minimum-CE
rate selection and fresh paired bootstrap contrasts directly from per-target
losses, including partial final blocks. Stored statistics must agree within the
existing 1e-12 numerical comparison tolerance and every decision must agree
exactly. Scientific thresholds and comparisons are unchanged. An arithmetic
mismatch is a review failure, never permission to change a scientific criterion.

Preparation checks completed on November 8, 2026:
- Syntax parsed without importing the training model.
- Three new constant-loss synthetic cases checked clear improvement, zero change,
  and a negative average with one losing seed. Deltas and every bootstrap bound
  matched their analytic values within 1e-12; all gates matched. 0.142 seconds.
- A new heterogeneous synthetic case used 34,278 deterministic targets, three
  paired rows, both contrasts and all three block widths. The independent path
  agreed with the frozen statistics implementation to 9.541e-17 maximum absolute
  difference, with identical gates. 0.193 seconds.
These are arithmetic checks, not model tests, scientific observations, replayed
closed fixtures, full-panel acceptance, or evidence of a quality gain.

After successful controller termination and availability of the shared worker
lock, create a new final-review001 directory and run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m research.byte_input_quality_final_review_v1 research/tokenizer-evidence/byte-composition-boundary-001/final-review001/review.json
```

The reader takes the shared nonblocking worker lock, uses the existing CPU guard
with a 300-second deadline, 1 GiB RSS limit and 8 GiB disk reserve, and creates a
fresh sibling guard receipt. It only writes the review after guard completion.
Inspect that receipt and the saved evidence before any adoption. The complete
18-run review has not executed. It waits for all original fresh confirmations
and successful controller termination; current progress is in the research handoff.

Reviewer source SHA256: 088039e6783a322f9baf7ebaa00f617233199405975e7dcbb18ffb80a628c58a

## Recovery layout requires an additive reviewer

The original controller later closed failed on slot14's output-inventory race.
`../guard-recovery001/README.md` preserves the failure and separately qualified
recovery. `../quality-recovery001` routes original accepted slots00–13, recovered
postprocessing of the exact slot14 RAW, and previously unused slots15–17. The
entrypoint above requires the original successful STOP, so it must not be run
as though the original controller completed successfully. Reuse its accepted
arithmetic implementation in a separate final reader that explicitly verifies
this routing, immutable failure lineage, unchanged scientific gates and resource
accounting. That adapter remains to be implemented and reviewed before adoption.

## Adoption consumer inspection

The current incumbent is recorded in
`research/tokenizer-evidence/adamw-epsilon-preparation-001/adopted-width256-mla-epsilon-recipe.json`.
Its adoption implementation,
`research/tokenizer-evidence/adamw-epsilon-preparation-001/quality001-adopt001.py`,
provides an existing convention: retain historical evidence and choose the first
fresh candidate in frozen order as a representative, never the lowest-loss seed.
If this panel is eligible after independent review, that convention would choose
raw-byte slot10. This is an operational artifact choice identified during
inspection, not an additional scientific gate or a claim that adoption occurred.

The byte model requires its own qualified model, training and inference sources:
`research/byte_input_model_v3/`, `research/byte_input_training_v2.py`, and
`research/byte_input_inference_v1.py`. Relabeling a native configuration would not
reconstruct it. A future adoption record must bind these admitted sources, the
original recipe and codebook, the accepted parent/export receipts, and the final
panel/review evidence, preserving the width-256 MLA scope. The unqualified
`research/byte_input_direct_hash_v1.py` optimization is not part of this model.

The accepted inference loader requires an independently supplied typed identity
and parent. `training.Context._identity` retains tuple model/codebook signatures;
JSON converts these to lists. A standalone consumer therefore must reconstruct
the typed identity from the admitted plan and codebook contract, not blindly pass
JSON lists or trust the export's self-description. This is an inspected API
requirement, not a demonstrated inference defect or a reason to repeat accepted
model tests. Existing per-run inference/cache checks remain the numerical evidence.

This inspection added no training, inference, seed, quality comparison or test
suite. Read-only inspection mistakes (a Boolean sliced as text and package/file
path assumptions) were corrected by string conversion and exact file discovery;
their original tool failures remain in the conversation. Frozen sources and the
active controller were unchanged.
