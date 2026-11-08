# Numerical failure and targeted scoring recovery

Original source, plan, slot00 weights, complete logs, CPU/MPS target losses and
failed numerical predicate remain unchanged. Training completed1,200 updates;
mean error1.839996e-5 passed3e-5, maximum target error.003434181 failed.0003.
The original controller exited1 and slots01..08 were never attempted. This is
not a negative quality result. No quality gate has been evaluated for a complete
panel, and the byte incumbent remains unchanged.

Two read-only diagnostics on the newly trained slot00 H200 weights localized
the problem. Fixed first and worst-error validation blocks showed composer error
below2.7e-7; the first block introduced ~5.2e-4 output error. Later blocks and
final LayerNorm were accurate on identical inputs. Attention and FFN separately
agreed within6e-7. MPS default RMSNorm differed from its explicit2^-23 epsilon
version by1.55e-4; explicit epsilon agreed with CPU within4.77e-7.
The final-LayerNorm centering probe did not explain the failure and is rejected
as a recovery direction. No old closed native projection/anchor fixture was used.

The installed torch2.14.1 reports Git revision
5c4886908584029761b579af026dcfb627c84070. The corresponding primary runtime source
explains the measured discrepancy:
- https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mps/operations/RMSNorm.mm#L25
  defaults absent epsilon to double machine epsilon in the MPS fused path.
- https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/layer_norm.cpp#L326
  selects that path when gradients are disabled; the MPS training/composite
  path uses the FP32 epsilon for FP32 inputs (lines271–274).
The installed Python documentation and local block-normalizer contract both
specify FP32 epsilon. The runtime path changes the semantics of inference with
implicit epsilon; the successful short pilot was not sufficient to expose the
trained-state amplification.

Correction: `research/byte_context_scoring_v2.py` keeps graph construction enabled
for forward scoring and immediately detaches losses. It calls no backward pass
and leaves parameters, gradient slots, samplers, and global CPU/MPS RNG unchanged.
This selects the same RMSNorm calculation as training without editing the frozen
model, its semantic guards, or optimizer. No system library is patched. This is
a scoped execution workaround, not a new model or a general serving adoption.

Full original34278-target verification succeeded using the SAME saved weights.
CPU target losses are bit-for-bit identical to the original CPU report. Corrected
MPS mean CE4.550480417683 versus CPU4.550459151130 gives mean error2.126655e-5 and
maximum target error.0002725124. Both ORIGINAL3e-5/.0003 tolerances pass. Zero
training updates, absent gradients, unchanged weights and unchanged RNG are
verified in scoring-recovery001. Its full reports remain available.

Admit a separately identified quality-recovery001 executor. Import the completed
slot00 training and verified corrected scores, without retraining it. Execute
only the original eight unattempted slots using their original paired roles,
initialization, arms, .0025 LR, update budget, optimizer, cooldown and H200.
All primary scores use the corrected scorer; both strict quality comparisons,
three fresh bundles and bootstrap rules remain fixed. No scores from older
campaigns enter the panel. Original online slot00 curves used the defective
no-grad path and remain labeled diagnostic-only; they are not reconstructed or
used as an alternative endpoint. Remaining online diagnostics use the correction.

The original slot00 terminal sampler digest was not published because the worker
failed before its final result record. Do not fabricate an observed digest.
The recovery records a missing observed digest and a separately derived expected
state from the frozen seed and1,200 two-integer private CPU draws. Source/count
pairing evidence is compared with the observed digests of the remaining arms.
This recovers no optimizer-resume state and is an explicit audit limitation.

The complete18-weight output requirement remains unchanged because slot00 weights
are referenced, not copied. The1.5GiB retained cap includes original failed
records and all recovery evidence/weights; reserve8GiB plus2GiB extra headroom
is checked again. Carry original/pilot/diagnostic elapsed time into the unchanged
10-hour allowance, retain one-hour trial guards, and retain the original1200s
per-trial scoring allowance (the full corrected scoring check fits easily).
The independent v2 reviewer is frozen before the remaining quality outcomes.
A passed panel still requires separate consumer qualification before adoption.
