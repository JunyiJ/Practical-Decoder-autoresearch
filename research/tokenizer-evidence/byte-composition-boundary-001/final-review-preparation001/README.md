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
18-run review has not executed; the last screen is still training.

Reviewer source SHA256: 088039e6783a322f9baf7ebaa00f617233199405975e7dcbb18ffb80a628c58a
