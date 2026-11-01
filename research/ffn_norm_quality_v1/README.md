# Finite FFN product-normalization quality comparison

This additive namespace reuses the accepted FFN normalization runtime, validation,
output caps and inference, together with the exercised MLA-score finite controller
and retention rules. All predecessor namespaces remain byte-identical.

Compare none versus affine RMSNorm on the SwiGLU product before down projection.
Both use the adopted D256 MLA backbone, score multiplier0.5 and H200. The RMS arm
has four additional gain tensors, so accepted EMA inventories are88/92 tensors.
No other model or optimizer change is allowed.

The panel contains exactly six1,200-step screens (three equal learning rates per
arm), then three reserved fresh pairs at each arm's minimum full-validation H200
CE rate, breaking ties toward lower LR. Paired inference and a separate final
statistical review follow all twelve endpoints. Pilot loss cannot select anything.

The complete retained-output requirement is13,555,990,528B: eight560MiB checkpoint
pair allowances,256MiB metadata/transients and8GiB reserve. Successful pilot
retirement requires the passing real-payload integration review. Four losing
screens retire only after all six are independently audited and selection is
persisted. Selected screens, all fresh pairs and failed artifacts remain protected.

Run `research.ffn_norm_quality_v1.quality_execution` only with the exact final
freeze SHA, strict MPS environment, four CPU threads and the shared worker lock.
The controller refuses existing/unaccepted attempts and noncontiguous dispatch.
Original pilot receipts remain immutable; completed pilot-retirement manifests
explain intentionally absent pilot weights. Never rerun a closed pilot.
