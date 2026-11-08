# Closed training-context proof v1

Original admission, numerical entry and saved review completed and accepted.
Do not replay these entries or their frozen five-role assignments.

The six cases combine native/raw-byte/nonlexical input models with CPU/MPS.
Each runs four actual updates, saves after two, and resumes two in a fresh
context: 36 updates total. Exact continuation covers sampled data, RNG, schedule,
clipping, optimizer, parameters and EMA. There are 37 invalid-transaction checks
per case. Saved review independently reconstructs final EMA in FP64 using the
retained step-2/4 RAW archives and step-3 parameter arrays; it does not train.

Evidence: `research/tokenizer-evidence/byte-composition-boundary-001/` under
`training-admission001`, `training-math001`, `training-review001` and
`training-acceptance001`. Binary payloads are local-only and hash-bound.

This verifies the finite four-step training-resume requirement, not model
quality or inference-export/scorer/campaign-worker correctness. Reuse this
accepted evidence in later consumer qualification.
