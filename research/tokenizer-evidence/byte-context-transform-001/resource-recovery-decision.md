# Correct one recovery resource-root binding

Static inspection during slot01 found that byte_context_transform_run_v2.py
assigned both ORIGINAL_WEIGHTS and WEIGHTS to the new recovery folder. A broad
source-generation replacement accidentally changed the intended original path.
This excludes the two retained slot00 weights and double-counts later new weights.
The initial remaining-space check was conservative because it omitted retained
bytes from the deduction; the growing output counter would eventually stop the
panel unnecessarily. This is a demonstrated guard/reporting implementation defect,
not a scientific result or a change in available capacity.

Keep v2 source and its original plan immutable. The live slot01 worker continues
unchanged, with its correct RSS/MPS/free-space guards and corrected scoring.
The dispatcher alone was verified by PID/command and SIGSTOPed before it could
launch another case. A separately source-bound resource-recovery wrapper waits
for this worker to finish, records correct accounting, and terminates only the
obsolete stopped dispatcher after the child is terminal. No scientific worker,
seed, model, score, comparison, criterion or tolerance is changed or retried.

The wrapper imports the frozen executor, verifies the erroneous constants, then
supplies the correct ORIGINAL_WEIGHTS path before future worker calls. Correct
roots are the original evidence tree, original weight tree and new weight tree,
each exactly once. The1.5GiB cap,8GiB reserve,2GiB headroom and original10-hour
budget remain unchanged. Every remaining worker has a source-bound supplementary
receipt. The independent final reviewer checks those receipts and actual totals.

Slot01's original counter is explicitly defective. Its original fixed slot00
weights remain immutable, new RAW/H200 files are published monotonically without
temporary renames/deletion, and metadata is bounded within the planned allowance.
Its verified endpoint total, including all original/recovery files, therefore
bounds actual retained output with ample margin. Preserve the original guard;
do not rewrite its observations to make them appear correct. This narrower
recovery avoids discarding or rerunning valid training.
