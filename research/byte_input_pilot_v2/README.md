# Closed accepted twenty-update resource pilot

Pilot002 completed all three native/raw-byte/nonlexical arms on MPS:60 updates,
complete RAW/H200 export/load, all34278validation targets and cached correctness
at B1/B8. It used fresh five-role assignments and the same20-update/paired-input/
LR.00125 recipe as original001. Do not replay either entry or reuse their roles.

Original001 remains failed after20 native updates: its driver supplied the wrong
writer arguments. This version changes only that call and removes redundant
capture. The original saved reviewer002 then rejected canonical `mps:0` because
it expected shorthand `mps`. Saved-only reviewer `byte_input_pilot_review_v3.py`
corrected that metadata expectation without repeating training. All original
source, logs, failures and roles remain preserved.

Accepted evidence under `research/tokenizer-evidence/byte-composition-boundary-001/`:
`pilot-math002`, `pilot-review003`, `pilot-acceptance002`. Original failures are
bound by `pilot-recovery001` and `pilot-review-recovery001`.

This source implements only20-step resource pilots. No1200-step scientific
worker/panel launch, model selection, quality improvement or adoption follows.
