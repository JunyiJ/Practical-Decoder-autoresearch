# Composer reuse component result

The standalone candidate passed its frozen component gates. No production
runtime or scientific recipe changed. Full-model optimizer/restart qualification
is the next required test.

- All six cases passed output and all18 parameter-gradient comparisons,108
  gradient comparisons total. Maximum absolute output difference was1.304e-8;
  maximum gradient difference was2.421e-8. The largest gradient tolerance ratio
  was0.009652, below the fixed limit1.
- Three repeats of each implementation produced bitwise-identical outputs and
  every gradient. FP64 accumulation-oracle checks passed. Parameter hashes and
  CPU/MPS global RNG states stayed unchanged.
- Historical first-batch reuse/original median composer training-time ratios
  were0.444708,0.497715 and0.379275. The median reduction was55.53%, exceeding
  the required15%. Every historical case improved. The all-unique ratio was
  0.999029, within the1.10 limit. Repeated and alternating controls had ratios
  0.058496 and0.057829.
- Attempt1 completed in37.35s with758,300,672-byte peak RSS. Evidence before
  completion review totaled47,305 bytes. Free space exceeded25.4GB, above the
  complete64MiB allowance plus10GiB reserve. No cleanup was needed.

Evidence: `tokenizer-evidence/general-rsi-composer-reuse-001/plan.json`,
`attempt-01/*-correctness.json`, `attempt-01/*-timing.json`, `decision.json`
inside that attempt, and `completion-review.json`. Sources and protocol were
committed as `eac0a1b04` before execution.

The result concerns one private initialization and three fixed historical
batches, with ten timed measurements per arm per case. It does not establish
performance across the complete corpus or full updates. Transformer, readout,
optimizer and averaging costs remain. Small gradient differences can affect
Muon trajectories, so no quality equivalence or gain follows from this result.
There was no optimizer update, generation, validation scoring or test access.

Updated hypothesis: whole-token reuse can remove a measured composer cost
without losing component repeatability. Next, measure its actual optimizer
behavior and whole-update cost in a separately identified execution path,
including complete-state serialization/restart. Keep scientific acceptance
rules unchanged and require a fresh complete confirmation before adoption.
