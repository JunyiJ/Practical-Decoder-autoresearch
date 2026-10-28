# Width512 direct MLA/GQA quality and inference protocol

Prospective October28,2026. The human requested wider models as pilots for larger
recipes and a fresh MLA/GQA comparison. Two width51220-step MPS pilots passed;
CPU prerequisites passed68 checks. This protocol is not a large-data or scale
transfer guarantee. Freeze tested executable source/plan before quality training.

## Fixed comparison

Fresh width512 four-layer models, GPT-2 BPE50257 with original raw80/10/10 split
and corpus hash86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed.
RoPE, head32, bias-free SwiGLU21D/8, block RMSNorm/final affine LayerNorm,
legacy untied tables, serial unit residuals, attention multiplier0.5, Muon/AdamW
base64 scaling and independent decay0.00006. GQA:16query/8KV heads; MLA:16heads,
query/KV rank128, shared rotary dimension32. GQA score normalization sqrt32;
MLA sqrt64 for concatenated content/rotary, each with the0.5 multiplier.
Explicit wider policies reject unreviewed overrides. Attention changes consume
RNG differently; paired seeds do not imply identical shared initial tensors.

Both arms use1200updates, batch8/context128, paired512 sampling; cooldown starts800
and ends1200 at0.1; H200start800/interval10 gives41snapshots. Equal token/update
budget, not equal FLOPs, total parameters or elapsed time. This transfers the
shorter GQA recipe as a fixed comparison budget, not a proven width512 optimum.
Primary endpoint final full-validation H200 CE:34278targets/111536bytes,
context128 resets and unpadded tail. Raw CE/BPB/samples/online validation/time are
diagnostics. No earlier checkpoint selection or test loss. Reused validation
and small Shakespeare corpus limit transfer/generalization claims.

## Finite tuning and confirmation

Six screens at seed3401: GQA LR0.000625/0.00125/0.0025 ascending, then MLA rates
descending. Independently choose each family's lowest full-validation H200 CE;
exact ties choose lower LR. Persist both choices before fresh confirmation.
No rate-grid extension, even for a boundary winner; report that limitation.

Six fresh trials at seeds3411/3412/3413: GQA/MLA, MLA/GQA, GQA/MLA in that order.
No replacement seeds, reselection or pooling old width256 observations. All12
quality trajectories remain in the finite panel. A failure stops the round and
requires diagnosis; no automatic retry. Pilot CE never selects rates or arms.

Compute MLA-minus-GQA paired mean/every-seed deltas and conditional95% text
bootstrap intervals (blocks2048/4096/8192,2000draws,seed20261024, joint across
trained seeds). MLA quality neutrality requires mean, every seed and every upper
bound<=+0.01nat. This is an agent-chosen prospective engineering margin, not a
human numeric tolerance. Strict MLA quality additionally requires mean<-0.02,
every seed<0 and every interval upper bound<0. Report the symmetric strict GQA
gate (MLA mean>+0.02, every seed>0, every interval lower bound>0). These intervals
are conditional on trained seeds and adaptive validation, not seed-population
or multiple-comparison-adjusted inference.

No adoption from quality alone. A scoped width512 MLA memory option requires
neutrality plus at least60% smaller actual allocated KV storage versus GQA in
every fresh pair/batch, cached correctness and complete artifact review. No speed
win required. Report any strict quality preference and all cost tradeoffs without
claiming a universal winner. If gates fail, do not adopt MLA from another seed,
old MHA comparison, pilot score or a favorable timing. This round cannot revise
width256 decisions or claim transfer to width1024.

## Frozen serial inference review

After quality completion and artifact audit, load each fresh H200 pair serially,
verify exact config/tokenizer/seed/lineage and use first96 training tokens. FP32
MPS/context128, batches1/8,32prefill plus64teacher-forced decode tokens; cached/full
parity before two warmups/arm and eight synchronized repetitions alternating
which arm goes first. Report per-seed medians and median of paired median ratios,
parameter bytes and every latent/shared-key/KV buffer. Both models resident;
allocator readings are not single-model peak or maximum batch capacity. No
long-context/cloud-cost/optimized-kernel claim.600seconds per pair,4MiB total
measurement-output allowance, same global/run locks and8GiB disk reserve.

## Resources, provenance and stopping

Freeze source/runtime/protocol/corpus/prior recipe/pilot hashes, configs, rates,
seeds and order from a clean commit. Source files present during the pilots must
remain unchanged; added controller/benchmark files are separately tested/frozen.
Preserve all checkpoints, requests/results, failures, scores and logs. The raw
weights and H200 remain separate, no post-result checkpoint replacement.

One serial strict MPS worker/four CPU threads, no fallback/paid compute/subagents.
Quality7200seconds each, sampled4GiB RSS/6GiB driver limits,8GiB free reserve.
Per-trial bounds GQA1,737,836,135bytes and MLA1,725,319,424bytes. Each exceeds
measured pilot raw/H200 pairs1,212,185,804/1,205,037,884bytes. Keep six allowances
per family plus4MiB round metadata; before every launch all remaining allocations
must fit above reserve. Monitor disk/output every2seconds. STOP, failure, timeout,
changed frozen evidence or guards halt dispatch; interrupted/running controller
cannot be restarted automatically. Historical closed campaigns remain closed.

Commands after freeze: python -m src.research.tokenizer_wide_study prepare|run
--output <new-run>. After all12 quality records, use python -m
src.research.tokenizer_wide_bench --study-run <run>. Final review precedes adoption.
Then choose the next finite architecture or HP round; broader hybrid tuning,
activation/MoE/norm/residual/depth and standalone schedule/regularization remain
first-class. Width1024 is a future scale target, not silently substituted here.
