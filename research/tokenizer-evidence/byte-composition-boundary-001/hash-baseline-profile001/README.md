# One warmed step of the adopted hash-only baseline

The bounded profile used the adopted first-fresh H200, a fresh optimizer and
fresh private sampler roles62100700001/2. It performed two warmup numerical updates
and profiled the third through the unchanged1200-step recipe at LR.0025 on MPS.
No quality evaluation, weight archive or scientific confirmation was performed.
The roles are reserved/closed and are not available for future work. The active
adopted checkpoint and source files remain unchanged.

Warmups took1.5965 and1.2324seconds.
The instrumented third step took1.1216seconds.
The guard finished in8.0955seconds, peak
RSS724,811,776bytes, within60seconds/4GiB RSS,
6GiB sampled MPS and8GiB disk reserve. CPU/MPS global RNG stayed unchanged.

Useful observations from the complete saved pstats and JSON:

| Call path | Calls | Cumulative seconds | Exclusive seconds |
|---|---:|---:|---:|
| Model.validate |6|0.394317|0.142424|
| Model._buffer_digest, nested within validation |84|0.196636|0.001400|
| Optimizer.validate, including nested model validation |5|0.405562|0.046080|
| Optimizer._validate_state |5|0.147348|0.138641|
| Training Context.check |2|0.249067|0.093510|
| Fixed-order byte embedding backward |1|0.188019|0.187549|
| Composer.forward |1|0.105663|0.097130|
| Data check_view |2|0.015292|0.001721|

Cumulative times overlap and must not be summed. cProfile reports Python-visible
wall time, including device synchronization waits and instrumentation. Exclusive
Python-frame time is not a pure CPU-work or GPU-kernel measurement. This is one
early-training step without an averaging snapshot, not a representative full-run
throughput distribution. The data-view checks were only about15ms here; a new
data-cache mechanism is not justified by this profile.

The next bounded candidate combines all105 per-parameter finite predicates into
one same-device Boolean conjunction in model.validate. This may reduce repeated
host scalar reads while retaining full structural, finite, ownership and content
checks. It leaves the mathematical forward/backward and optimizer unchanged.
No speed gain is established yet. The isolated source and exact method mapping
are in ../finite-batch-source001; existing exact-type consumers still reject the
subclass. Compare valid-state timing and nonfinite rejection/restoration on
CPU/MPS before deciding whether integration is justified.
