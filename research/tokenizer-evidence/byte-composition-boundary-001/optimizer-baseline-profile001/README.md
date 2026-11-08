# Profile after the two validation optimizations

This closed profile measured model-v5/optimizer-v6/training-v5, before the later
bulk-buffer hashing change. It loaded the accepted H200, used fresh sampler roles
62101200001/2, and performed two warmup updates followed by one profiled update.
The roles are now closed/reserved. No quality evaluation or weight archive occurred.

Warmups took 1.246160 s and 0.984842 s; the third update took 0.861881 s including
profiling and synchronization. Six model validations accounted for 0.300386 s
cumulative, including 84 buffer digests at 0.206869 s. The 378 SHA updates used
0.104651 s exclusive, and 378 NumPy byte copies used 0.010569 s. All tensor `to`
calls together used 0.090068 s; that total also includes non-hash transfers.
The deterministic byte-embedding backward used 0.195348 s cumulative, almost all
exclusive. Optimizer-state validation used 0.061589 s across five calls.

These are Python-visible profile times, including device waits and instrumentation.
Cumulative times overlap and must not be summed. They are not GPU kernel timings,
a sustained throughput measurement, or a paired speedup against earlier profiles.

The profile justified a complete-buffer transfer and buffer-protocol SHA check,
subsequently completed and adopted in bulk-hash-adoption001. The unchanged
byte-embedding backward is a remaining measured computation cost. Keep its
2048-position chunks and ordered accumulation unless a separately justified
numerical experiment explicitly tests a different algorithm. The old padding
cropping experiment remains closed not adopted.

The guard finished in 6.668 s, with 840,450,048 bytes peak RSS. Source receipts,
full profile statistics and profile.pstats are retained here.
