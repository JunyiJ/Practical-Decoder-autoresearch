# Width256 GPT-2 table-sharing feasibility pilots

Freeze three serial 20-update trajectories, in order legacy_untied, matched_untied,
scaled_tied. All use seed3291, LR0.00125, width256/head32/half-KV GQA, four layers,
RoPE, batch8/context128 and inherited Muon/AdamW settings from the adopted horizon
recipe. The table policies and controls are described in TOKENIZER_TABLE_PREPARATION.md.
Width256 is chosen prospectively because the complete panel fits after the explicitly
authorized checkpoint cleanups; no pilot CE selects width, arm or learning rate.

Cooldown starts10 and ends20 at multiplier0.1. H200 starts10, interval5, total20
(three snapshots). Keep only the raw endpoint and its separate H200 export.
Sampled validation is diagnostic. No full-validation ranking or test loss is used.
Load both endpoints, validate tokenizer/config lineage and actual unique FP32
parameter storage, and check full/cached MPS equivalence for first12 training
tokens at batches1/8 using 3/4/5-token cache chunks. Expected GQA allocation is
524288 bytes per batch item, unchanged across the three table arms.

Before dispatch commit clean source; freeze source/runtime/corpus/recipe/protocol
and all three exact requests in a digest-bound plan. Source and requests are checked
before and after each worker. One global worker lock and exclusive pilot lock are
inherited by each subprocess. Only a freshly prepared controller can start.
Failure, interruption, changed provenance or a guard violation halts the panel;
retain logs and checkpoints, with no automatic retry or replacement seed.
STOP or controller SIGINT/SIGTERM stops the active child and prevents later launches.

Each worker has600 seconds including export and cache checks, four CPU threads,
strict MPS without fallback, sampled4GiB RSS/6GiB MPS-driver guards. Controller checks
disk/output every2seconds. Preserve8GiB free. Charge659621888 bytes per untied
trajectory and363708672 per tied trajectory, plus4MiB pilot metadata. Reserve all
15 prospective quality trajectories as well as remaining pilots before each launch:
initial required free bytes18691843584. Bounds include15% tensor allowance and2MiB
per trajectory. An observed overrun fails; never lower reserve or silently increase
a bound during a run. Pilot resource review may increase future quality allowances
before a separate executable quality plan is frozen.

The pilot result is feasibility only. It cannot authorize quality adoption. The
prospective quality design remains six independent LR screens at seed3301 and nine
fresh confirmations at3311/3312/3313, with both controls and declared quality/cost
gates. Its executable controller and serial timing review must be prepared separately.
