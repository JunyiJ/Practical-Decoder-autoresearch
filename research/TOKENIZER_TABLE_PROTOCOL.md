# Table001: equally tuned width256 vocabulary-table comparison

This finite round implements the prospective comparison in TOKENIZER_TABLE_PREPARATION.md.
All three width256 MPS pilots passed under source30a309a and plan9fc40ed; their
raw/H200 files, tokenizer lineage, caches and storage were reviewed in
`tokenizer-evidence/table-pilot-001/review.json`. Pilot CE did not select any arm,
rate or width. No model/training source changed after those pilots.

Use GPT-2 BPE with the original raw80/10/10 split, corpus hash and tokenizer
binding. Test loss remains closed. Preserve four-layer width256/head32/half-KV
RoPE GQA, dense SwiGLU21D/8, normalization/scales/optimizer/decay from the adopted
horizon recipe. Compare legacy_untied, matched_untied and scaled_tied under the
explicit tokenizer_gqa_scaled_table_sharing_v1 policy. The matched control copies
input weights into an independent output matrix; tied shares one canonical
matrix. Matched/tied use1/sqrt(192) readout scaling followed by1/width_mult.
These interventions change coordinates, gradients and expressivity as described
in preparation; do not attribute every contrast to sharing alone.

Every quality trajectory:1200 updates, batch8/context128, paired512 sampling,
cooldown800–1200 ending at0.1, H200start800/interval10 (41 snapshots).
Main budget is equal target-token exposure, not equal parameters or wall time.

## Finite selection and confirmation

Six screens at seed3301, exactly this order:
legacy_untied0.00125, legacy_untied0.0025, matched_untied0.0025,
matched_untied0.00125, scaled_tied0.00125, scaled_tied0.0025.
Select lowest full-validation H200 CE independently per arm; exact ties choose
lower LR. Persist all three rates before confirmation. Retain both controls.
Nine fresh confirmation trajectories use those selected rates, ordered:
3311 legacy/matched/tied;3312 matched/tied/legacy;3313 tied/legacy/matched.
No new rate, replacement seed, confirmation reselection or retrospective narrowing.

Full-validation H200 token CE is primary: every validation target except first,
context128 resets, unpadded tail. Raw CE/BPB/perplexity/samples/elapsed time are
diagnostics. The same validation has been adaptively reused; conditional intervals
are engineering gates, not seed-population or multiple-testing-adjusted inference.

For each candidate-minus-control contrast use paired blocks2048/4096/8192,
2000 joint-across-seed text-bootstrap draws with seed20261024. Strict quality
requires mean delta<−0.02, every seed<0, and every95% interval upper bound<0.
Neutrality requires mean, every seed and every upper bound<=+0.01. The neutrality
margin is an agent-chosen prospective engineering convention, not a human tolerance.

Tied strict adoption requires strict quality against BOTH legacy and matched.
Tied memory adoption requires neutrality against BOTH and >=30% smaller measured
unique resident FP32 parameter storage against each. Matched may qualify only by
strict quality versus legacy. Report all three contrasts. No adoption from screens,
incomplete panels or pilot counts. Final artifact and inference review is required.

## Measured cost review

Load each fresh H200 triple; validate same seed/config/tokenizer and the first96
training token IDs. Measure actual parameter storage deduplicated by storage pointer.
For batches1/8, check cached/full equivalence first, two warmups per arm, then eight
synchronized repeats in cyclic arm order (rotate by repeat index). Use32-token
prefill and64 teacher-forced decode steps, FP32 MPS/context128. Report per-seed
and median prefill/decode, actual cache/parameter bytes and working-memory readings.
All three models remain resident, so working memory is not single-model deployment
peak memory. No speed gate for the parameter-memory route. No KV reduction, maximum
batch-capacity, long-context, cloud-price or cross-width claim follows automatically.

## Execution and failure rules

Source/runtime/corpus/recipe/protocol/resource review and plan digest are frozen
before dispatch. Configuration construction and all rate/seed/order choices are
part of frozen source and plan. One local worker under inherited global and study
locks; strict MPS/no fallback/four CPU threads. Exclusive trial directories preserve
attempts. Limit7200seconds per trial including export/evaluation, sampled4GiB RSS
and6GiB MPS-driver guards; keep8GiB disk reserve. Check every remaining trajectory
before each launch and monitor disk/output every2seconds. Allow659621888 bytes per
untied trajectory and363708672 per tied trajectory, plus4MiB round metadata. Pilot
pairs measured560602700/303284813 bytes respectively, below these bounds.

Failure/STOP/timeout/changed source or data stops dispatch and retains evidence.
No automatic retry, no controller restart from a running state. Explicit diagnosis
and a new reviewed contract are required after interrupted or failed work. Final
quality completion does not itself adopt a model: serial inference and saved tensor
review follow under the same source freeze before a final scoped decision.

Commands after freeze: `python -m src.research.tokenizer_table_study prepare|run
--output <new-run>`. After all15 quality trials complete, use
`python -m src.research.tokenizer_table_bench --study-run <run>` for three serial
fresh-seed timing triples. The timing parent verifies frozen provenance and hashes,
uses the same locks,600seconds per triple and4MiB output budget; a prior timing
attempt prevents reentry. Preserve an explicit review before any failed retry.
