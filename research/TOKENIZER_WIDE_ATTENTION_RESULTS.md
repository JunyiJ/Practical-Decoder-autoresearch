# Wider direct MLA/GQA comparison status

October28,2026. Two width512 resource pilots passed and were audited; the finite
12-trial quality comparison is prepared under source607ef9f, plan digest
`12c60cbf4e01550e09b8317ccbbb90d13ed2390ad01a3d3304db205450f37240`.
Consult research_state/tokenizer-quality-progress.json and the actual run state
for dispatch/progress. No width512 comparative quality result or adoption yet.

The [frozen protocol](TOKENIZER_WIDE_ATTENTION_PROTOCOL.md) independently screens
three LRs per family at3401, then compares selected GQA/MLA recipes at fresh
3411/3412/3413. Width512 has62.88M/62.42M parameters. The common1200-update
schedule is a fixed comparison budget, not a demonstrated wider optimum. No
screen/pilot result is a final ranking. Source and selection rules stay frozen.

Both20-update pilots retained raw/H200 pairs and passed tokenizer/checkpoint/EMA,
finite saved tensor and cached/full parity checks at batches1/8. Measured cache
per batch item/context128 is1,048,576 bytes GQA and327,680 MLA (68.75% less).
This is geometry/feasibility evidence only. The>=60% memory benefit must be
measured again on every fresh pair, together with prospective quality neutrality
and serving latency. No speed win is required for the memory adoption route.

Width1024 full-panel accounting currently exceeds available disk, so it is not
in this finite round. Existing8GiB reserve and sampled memory/time guards remain.
All old campaigns stay closed and cleanup records remain authoritative. Test
holdout is closed; adaptive validation and small-data limitations still apply.

Preparation evidence: [CPU and resource review](tokenizer-evidence/wide-attention-preparation-001/review.json),
[pilot review](tokenizer-evidence/wide-pilot-001/review.json), and
[quality/timing controller review](tokenizer-evidence/wide-attention-preparation-001/quality-timing-review.json).
The initial invalid two-update CPU fixture failure is retained and corrected;
no MPS pilot failure or retry occurred. In-memory CPU resume checks do not stand
in for disk export checks; both were completed separately.

## Reviewed screening progress

October28,04:30UTC: all six screening trials completed1200updates. Checkpoint
hashes, exact config/tokenizer/EMA lineage, parameter counts and saved validation
arithmetic passed review. Full tensor-finiteness/EMA equality audit remains due
after all quality trials, before timing. The first fresh GQA confirmation is running serially.

| Family | Seed | LR | Raw CE | H200 CE | Training seconds | Raw/H200 bytes |
|---|---:|---:|---:|---:|---:|---:|
| GQA | 3401 | 0.000625 | 4.825032 | 4.857613 | 427.64 | 1,212,186,956 |
| GQA | 3401 | 0.00125 | 4.704309 | 4.709348 | 423.57 | 1,212,186,892 |
| GQA | 3401 | 0.0025 | 4.996874 | 4.883027 | 421.32 | 1,212,186,892 |
| MLA | 3401 | 0.0025 | 4.959769 | 4.842255 | 415.56 | 1,205,038,972 |
| MLA | 3401 | 0.00125 | 4.651238 | 4.649909 | 415.38 | 1,205,039,036 |
| MLA | 3401 | 0.000625 | 4.799393 | 4.834718 | 417.59 | 1,205,039,036 |

Both families independently selected LR0.00125, the interior point of the frozen
three-rate grid. The independent selection audit matches the controller's saved
choice, written before the first fresh confirmation request. This is a one-seed
tuning outcome, not proof of an optimum or an attention-family quality advantage.
All six fresh confirmations at3411/3412/3413 remain required, followed by final
tensor/EMA audit and the three frozen inference pairs.
[Screening review](tokenizer-evidence/wide-attention-001/screen-review.json) records
the selection checks.
[Completed-record review](tokenizer-evidence/wide-attention-001/completed-review.json)
records the bounded audit scope; source and prospective rules remain frozen.

## Fresh confirmation progress

October28,04:45UTC: two of six fresh trajectories have completed and passed the
same saved-artifact hash/metadata/arithmetic checks. MLA seed3412 is running.
The first pair has MLA-minus-GQA H200 delta-0.047069nat. This single-seed
observation cannot establish improvement or adoption. Final tensor/EMA audit
and frozen serving measurements remain due after all quality trajectories.

| Family | Seed | Selected LR | Raw CE | H200 CE | Training seconds |
|---|---:|---:|---:|---:|---:|
| GQA | 3411 | 0.00125 | 4.708778 | 4.707809 | 420.93 |
| MLA | 3411 | 0.00125 | 4.671780 | 4.660740 | 413.52 |
