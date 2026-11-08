# CPU reference correction after completed tanh training

The resource-corrected dispatcher completed slot02's 1,200 training updates and
both RAW/H200 writes. It stopped on the unchanged maximum per-target CPU/MPS CE
limit: 0.000364232808 exceeded 0.000300; mean error 0.000010802940 passed 0.000030.
`quality-recovery001/slot02-failure-preservation.json` binds the original source
plan, logs, scores, failed guard and weights. Slots03 through08 were not attempted.

The new `numerical-diagnostic002` used only this completed checkpoint, with the
fixed first and worst-error windows (starts0 and13184). Grad-enabled forward
matched the actual scorer. Propagated logits differed by at most1.34e-5 on these
windows; identical-input projection and readout were exact, and tanh differed by
at most1.20e-7. Identical FP32 logits reproduced the loss discrepancy: CPU native
CE differed from CPU64 logsumexp by up to0.000364087, while MPS CE differed by less
than8.9e-7. This is a CPU verification-reference accuracy defect, not evidence of
a defective tanh transform or a negative model-quality criterion.

The smallest correction changes only CPU verification CE to an explicit shifted
FP32 exponential/sum/log reduction. Model forward, FP32 primary MPS CE and FP64
aggregation remain unchanged. Training never calls the new CPU helper.

`cpu-reference-recovery001` prospectively verified all34,278 targets on every
completed checkpoint, not just the failing window. Each original native CPU
report reproduced bit-for-bit. The stable FP32 reference differed from CPU64 by
at most1.94e-6 across the three models, versus original maxima0.000273127,
0.000262877 and0.000364087. CPU/MPS maximum target errors became1.91e-5,3.06e-5
and1.75e-5; mean errors were below5.13e-8. All original limits pass. The original
MPS primary score files are reused byte-for-byte. No training or checkpoint
rewrite occurred. Verification took55.144406 seconds; localization took2.477130.

## Recovery execution

Use a separately source-bound `quality-recovery002` namespace. Import the first
three completed trajectories with corrected CPU reference reports and unchanged
MPS primary reports. Retain every original failure and reference report in place;
copy only small evidence, never weight payloads. The source/count-derived terminal
sampler state remains explicit for slots00 and02, whose failures preceded runtime
digest publication. Slot01's observed terminal digest provides the paired
reference. This does not reconstruct optimizer state or claim an observed digest
where none was saved.

Execute only original unattempted slots03..08. The three arms, original roles,
rotated order, .0025 learning rate, 1,200 updates, H200, controls, tolerances and
statistics stay fixed. The new executor dispatches primary MPS scoring directly
to the previous v2 scorer. The independent reviewer checks every imported primary
file against its preserved predecessor, all corrected CPU verification receipts,
the original failed guards, weights, coverage and fixed statistical arithmetic.

The first bundle's unchanged primary losses are byte4.550480418,
linear4.643989454 and tanh4.616355683. Tanh beats linear but loses to byte in this
bundle, so it cannot satisfy the all-seeds-negative adoption criterion. Complete
the frozen panel to characterize the outcome; do not tune, replace this seed,
switch controls or use a different endpoint to rescue it.

Keep the original1.5GiB total output cap, 8GiB reserve plus2GiB admission headroom,
3,600 seconds per trial and36,000 seconds overall. Count original and recovery
weights exactly once. Carry all completed execution costs, including both
diagnostics and reference verification. No public inference repair is adopted;
its separate pending check remains independent of this scientific panel.
