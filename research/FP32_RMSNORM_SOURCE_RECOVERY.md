# Inference qualification source-binding recovery

Attempt002 completed both full numerical passes, all four B1/B8 CPU/MPS cache
cases, exact reset/replay and the weight/semantics/RNG checks, then failed its
final source census. It is not accepted. Its 42.389451 seconds and all artifacts
remain preserved alongside attempt001's 38.960124 seconds.

Exactly one source changed during attempt002:
`research/byte_precomputed_report_v1.py`, from SHA256
`48043e68b1d1cffb54f9d4833f96adde94ac3f1c0ec430e8ffa6bcb024f14114`
to `b735c395466f49421c1c0e5185608996c6e86f1d0d35d728fd453796d7cf137f`.
This is another task's standalone saved-result report generator. Its full source
was inspected: its only imports are hashlib, json, pathlib and statistics; work
executes through main under its own __main__ entry. A reference search of the
inference checks, byte-context helpers, byte-input modules and src found no
imports or references to it. It supplies no model, cache, scoring or runtime code
to this qualification. Do not revert or modify that unrelated work.

The broad all-research-Python census accidentally coupled this test to an
unrelated report edit. Attempt003 excludes exactly that report generator from
its prospective executable-source census; every other existing research/src
Python source and every original data/checkpoint asset remains pinned. The
exception is explicit and narrow. This is an engineering source-admission
correction, not a relaxed numerical or scientific requirement. Attempt002 remains
failed; its partial measurements are not promoted into acceptance.

The entire original corrected check is rerun, with the same weights, cases,
tolerances, repair module and no training. All three attempt folders count
against 32MiB. Prior elapsed execution is charged to the same 600-second total,
leaving a 518-second guard. A complete pass must verify its own frozen inputs at
entry and exit. No new quality claim or incumbent adoption follows from this
implementation qualification alone.
