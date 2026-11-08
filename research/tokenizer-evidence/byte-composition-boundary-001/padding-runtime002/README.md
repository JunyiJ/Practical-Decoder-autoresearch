# Padding removal: exact-equivalence gate failed

Execution completed all16 fixed CPU/MPS cases. Fourteen cases failed the declared
exact output/gradient equivalence gate; the only exact cases were the maximum
vocabulary-length controls where cropping removes no columns. The prototype
`research/byte_composer_trim_v1.py` is **not adopted** and is not part of the
active recipe. No tolerance was changed. This is a numerical equivalence result,
not evidence of a quality loss or a harness failure.

Maximum observed output difference was4.842877388000488e-8; maximum parameter
gradient difference was9.059906005859375e-6. The two large CPU training-input
cases preserved outputs exactly but changed gradients. Cases use the saved
first-fresh adopted composer weights, deterministic training/vocabulary inputs
and a fixed upstream vector. No optimizer update, sampler draw or quality score
was performed; original weights and global RNG remained unchanged.

| Backend | Original component forward/backward ms | Trimmed ms | Descriptive speedup |
|---|---:|---:|---:|
| cpu | 238.613 | 24.105 | 9.90x |
| mps | 242.355 | 22.080 | 10.98x |

These are three alternating timing repeats after warmup on the first1024 training
IDs, including input/parameter checks but excluding preparation and reporting.
They are component timings, not whole-model or sustained-training speedups.
Cropping preserved rejection of invalid padding in all four original/candidate
CPU/MPS checks. Full original input validation occurs before cropping, and the
maximum-length vocabulary case remains supported. Changed matrix/reduction shapes
are a plausible source of the observed floating-point differences; this test did
not identify a particular GPU kernel as their cause.

## Reporting failure and separate recovery

Attempt001 stopped while fingerprinting a one-element token-ID tensor. Its
nonzero-derived stride was256; PyTorch considered it contiguous despite that
singleton stride, and viewing Long as Byte raised a stride error. The failure
source, plan, traceback, guard and STOP are preserved. The completed two CPU
case records already contained negative equivalence results.

Attempt002 changed only serialization of the fingerprint to NumPy tobytes, which
handles the logical singleton layout. It copied those two completed CPU results
byte-for-byte and executed only unpublished cases; the third case's computation
had no durable record before the first failure and was performed under the new
reporting attempt. The fixed cases, checkpoint, upstream, numerical source,
exactness gate and limits were unchanged. The first3.8546seconds were charged
against the original120-second budget, leaving116.1454seconds; recovery finished
in5.6649seconds. The original numerical failures remain binding.

This closes the transparent-cropping attempt. Do not rerun it, relax its gate,
or label the gradient differences as a harness defect. A materially different
numerical implementation or a separately justified scientific question would
need its own prospective criteria and evidence. The adopted hash-only baseline
remains active.
