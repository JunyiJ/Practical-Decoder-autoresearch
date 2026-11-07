# Public vocabulary byte layout accepted; metadata scope only

The original read-only full-vocabulary inspection and independent saved-report
review passed. Both expected public asset hashes were verified before parsing;
file path/device/inode/size/link/owner/mtime identities remained unchanged. The
parser used its own fixed byte alphabet and stdlib parsing, with no tokenizer
imports, extracted library functions, network, corpus, model, scoring or RNG.

All 50,256 regular encoder IDs agree with independently constructed ordered merge
ranks, including all 256 one-byte units and 50,000 merges. The default-JSON native
vocabulary hash is bdc599f0a294cff99d60cf83aed3870857b30bc0d1d06164cae137831b55257c,
matching accepted physical-data metadata. This is distinct from the prefix
codec's compact-JSON digest convention. Special ID 50,256 is kept separate;
ordinary literal text policy and the existing GPT2 output identity are unchanged.
This metadata inspection does not execute or requalify the old codec/data tests.

Measured PUBLIC vocabulary facts: regular byte lengths span 1–128; the sole
maximum-length ID is 35,496. Total regular bytes are 320,814. There are 344 tokens
that are invalid UTF-8 when decoded alone, 33,135 starting with a space byte, and
one containing NUL. Use raw bytes without Unicode repair or normalization; NUL
cannot double as padding. No corpus-weighted token-length or rarity-loss claim
follows from this vocabulary-wide inventory.

For future cost design, a full padded int64 ID-to-byte table at width 128 costs
51,463,168B, versus 320,814B packed regular bytes plus 402,056B int64 offsets.
Full 50,257 int64 lengths cost another 402,056B. B8T128 entails at most 131,072
byte positions, or 133,120 with two boundaries. A dense FP32 byte-embedding
activation therefore costs 532,480 * e bytes before convolution, backward,
optimizer, body/readout, copies or allocator workspace. These are symbolic upper
bounds, not implemented storage choices or measured runtime. Proposed distinct
pad/boundary/special codes in the report are planning labels only.

Original commands are closed/no replay/no retry. Source gate pid 81073 exited 0,
0.004079208s / 23,330,816B RSS, without candidate import or asset-content reads.
Original metadata pid 81089 exited 0, 0.171229334s / 59,506,688B RSS. Whole original
output is 169,926B including plan/attempt/guard/result/STOP, below 2MiB. Independent
saved reviewer pid 81103 exited 0, 0.007788167s / 22,364,160B RSS. All original
120s CPU / 4GiB RSS / 8GiB reserve guards finished; entry included 2GiB extra
headroom. Complete anticipated input was 1,682,174B, below 8MiB. Forecast was
30s/256MiB, not a measured claim before execution.

The saved reviewer independently checked every ID length, histogram, extremes,
one-byte-ID coverage, special identity, saved asset pins and symbolic arithmetic.
It did not rerun asset parsing or reconstruct raw bytes/UTF-8 predicates from a
digest. Exact original result SHA is
e44a645be7325c7b9b6bdf215e9650d5a426979236589b1a800893ead095670f.
See original plans/attempts/reviews/guards/STOPs in vocabulary-source001/,
vocabulary-bytes001/ and vocabulary-saved-review001/. Preserve all original
sources, receipts and public assets. Zero tensor/weight files, copies, deletions
or retirement; no numerical exploration or scientific roles.

Two NEW source files add 25,507B: future bindings 1,269 => 1,271; old closed model
proof 1,236 and scientific runtime 1,211 remain unchanged. These counts do not
claim an unseen complete source ledger. The consumer pins only its actual
stdlib guard/package/source dependencies and accepted descriptor metadata.

The exact raw-byte mapping is feasible. This result admits the next bounded
SOURCE-COST/CONTROL DESIGN question only; no byte-composer mathematical,
gradient/device/native/state/cache/serving/scorer/quality qualification. Native,
holdout and future_two remain unchanged; TOP and all old failed proofs stay
closed with their original gates. Read next-source-handoff002.md before any
further work. No model/component code or numerical call until that plan is
complete and independently reviewed.
