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

October28,04:09UTC: all three full1200-update GQA screens completed. Checkpoint
hashes, exact config/tokenizer/EMA lineage, parameter counts and saved validation
arithmetic passed review. Full tensor-finiteness/EMA equality audit remains due
after all quality trials, before timing. The first MLA screen is running serially.

| Family | Seed | LR | Raw CE | H200 CE | Training seconds | Raw/H200 bytes |
|---|---:|---:|---:|---:|---:|---:|
| GQA | 3401 | 0.000625 | 4.825032 | 4.857613 | 427.64 | 1,212,186,956 |
| GQA | 3401 | 0.00125 | 4.704309 | 4.709348 | 423.57 | 1,212,186,892 |
| GQA | 3401 | 0.0025 | 4.996874 | 4.883027 | 421.32 | 1,212,186,892 |

These screens establish completion within the resource allowances for these
configurations only. LR0.00125 has the lowest H200 CE in this GQA screen grid;
this is a one-seed tuning observation, not a quality improvement claim. All six screens must finish
before selection; all six fresh confirmations remain required.
[Completed-record review](tokenizer-evidence/wide-attention-001/completed-review.json)
records the bounded audit scope; source and prospective rules remain frozen.
