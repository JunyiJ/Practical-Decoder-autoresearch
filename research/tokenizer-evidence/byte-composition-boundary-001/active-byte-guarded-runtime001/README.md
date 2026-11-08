# New shape-guarded primitive: exact comparison passed

This is a separately identified implementation, not a repair or reopening of the
closed active-row-v1 experiment. It compacts only complete2048-position chunks
with15–258 active rows. Smaller active sets, full259-row sets and partial chunks
retain the original full-row arithmetic. Chunk boundaries and accumulation order
are unchanged; no input domain or prior failing case was removed.

All281 comparisons were exact: the original seven cases plus four boundary cases
on CPU/MPS, then all259 active-row counts on MPS with2048full+17tail positions and
a distinct stress upstream. Original failures remain unchanged historical evidence.
Representative repeated gradients were exact and repeatable. Primitive median
latency fell14.906ms to6.365ms, ratio0.417041, with7/7wins. Selection median was1.252ms
and is included in candidate timing. No optimizer update or quality score occurred.

The guard finished in5.902seconds, peak RSS1,062,993,920bytes. Passing admitted a
new full-model comparison only. That subsequent comparison preserved numerics
but missed its predeclared training-speed gate, so no implementation was adopted.
