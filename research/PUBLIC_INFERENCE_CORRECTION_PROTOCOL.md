# Public inference correction: selected-checkpoint integration

The byte-context panel is closed negative. The scoped FP32 RMSNorm repair passed
its complete correctness check on new baseline weights, including CPU/MPS full
scoring and B1/B8 cached decoding. Stable FP32 CPU CE was separately verified
against FP64. These are the measured basis for public integration, not a new
scientific hypothesis or permission to rerun a historical quality comparison.

The new `byte_input_corrected_inference_v1` loader delegates exact selected-weight
loading and typed identity reconstruction to the adopted consumer-v4. It returns
a read-only handle with `logits`, `token_losses` and `session`. Prediction and
cached calls automatically apply the qualified scope. Token losses use the
existing verified stable CPU CE helper; MPS CE remains native. No training code,
model attributes, checkpoint bytes, export format or original module is changed.
The old full-validation scorer is a historical API, not the corrected public
entrypoint. Future evaluation may aggregate the new unreduced token losses over
its prospectively specified data and windows.

Use the already selected slot10 H200 from the immutable bulk-hash-adoption001
recipe, never a loss-selected replacement. On CPU and MPS load through the actual
new public loader, verify original artifact identity and exact parameter values,
then use the first B1/B8 blocks of 128 training tokens with their next-token
labels. These 1,152 fixed token positions per backend are implementation cases;
do not compute or publish aggregate quality scores. No validation or holdout
targets are scored and no scientific roles are drawn.

For each batch compare corrected public logits with the original grad-enabled,
no-backward forward (the declared training-equivalent normalization path).
Require inherited logits/cache atol3e-5 and rtol3e-4. Compare public token losses
against CPU FP64 CE on the same logits, maximum error<=3e-5. Compare paired CPU/MPS
token losses with unchanged mean CE error<=3e-5 and maximum target error<=3e-4.
These are correctness limits, not new model-selection gates.

Check prefill96 plus32 contiguous one-token cached calls, reset and exact replay,
same cache sizes/positions and full-context agreement for CPU/MPS B1/B8. Check
invalid IDs, target dtype/shape/range, cache positions, training mode,
torch.inference_mode and autocast rejection. A final exact-value self-copy of one
parameter must invalidate the handle and its cache via version checks; this
intentional mutation probe changes no parameter value and occurs after numerical
work. Preserve exact weights, semantics and global RNG, and create no gradients.

Freeze new entrypoint/check sources, actual imported helper dependencies and all
historically bound model/loading sources. Reuse the selected artifact without
copying it. Do not bind unrelated mutable report generators. Run under the shared
nonblocking lock, 600seconds, 4GiB RSS, 6GiB MPS, 16MiB new evidence, 8GiB reserve
plus2GiB headroom. Expected retained output is under1MiB; the 16MiB bound includes
plan, logs and receipts. There are no new weights or serialization overlap.

On complete acceptance publish a new implementation recipe that points to the
new public loader, existing selected weights, source receipts and this check.
Keep the historical recipe and quality evidence immutable. This changes the
qualified inference implementation, not the model-selection result or training
recipe. On failure preserve it and diagnose before a separately bound recovery.
