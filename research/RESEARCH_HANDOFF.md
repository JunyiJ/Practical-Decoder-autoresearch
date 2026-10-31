# Research handoff — stopped by the human on 2026-10-27

> **Current tokenizer continuation:** start with the [resume guide](TOKENIZER_RESUME.md) and [compact state index](TOKENIZER_RESUME_STATE.json). These distinguish active tokenizer work from historical stopped campaigns. Check live state before dispatch.

**Research is stopped.** At 02:52:45 UTC the human asked to stop iteration and preserve the learnings for a later restart or reproduction. The CPU controller and its worker exited, the research heartbeat is **PAUSED**, and the continuing goal is **paused**. This instruction supersedes older “continue until interruption” prose, preparation banners and automation prompts. Reading this guide, checking out a branch or restoring the archive does not authorize execution. A new explicit human instruction is required.

The last completed quality study is **logit-z-quality-001**. It retained the incumbent. **Null attention is unfinished preparation, with no quality result.** Campaign006 remains permanently stopped even if other research is later restarted.

## Read these first

1. This handoff: decisions, exact stop state and restart paths.
2. [Learning guide](LEARNING_GUIDE.md): explanations, historical progression and detailed measurements.
3. The relevant `*_PROTOCOL.md` and `*_RESULTS.md`: each study's own frozen comparison, seeds, thresholds, limitations and source revision.
4. [Portable snapshot](snapshots/2026-10-27/README.md): machine-readable conclusions, exact incumbent request, full knowledge bank and evidence metadata. The original local checkpoints are a separate archive.

## What worked

Lower validation cross-entropy (CE, nats per character) is better. Each row below is a **within-study paired comparison**. The gains must not be added together: different stages used different seeds, controls and evaluation endpoints.

| Retained change | Measured evidence | Interpretation |
|---|---|---|
| Muon/AdamW hybrid with explicit μP width rules and independent decay | [Optimizer study](OPTIMIZER_TRANSFER.md); 138 controlled optimizer runs, followed by 42 Muon suites / 756 runs in stage1 | Useful baseline in this setting. Hidden matrices use the reviewed Muon rule; embeddings/readout/vectors use AdamW. “μP” is not permission to apply arbitrary depth or architecture scaling. |
| Four-layer dense MHA/SwiGLU, head dimension32 | [Depth results](LEARNING_GUIDE.md#stage-2e-measured-conclusion-four-layers-win-with-a-resource-cost): versus fixed two-layer SwiGLU, CE deltas −0.062881 / −0.042753 / −0.027896 at widths64/128/256, every paired seed better | Robust depth gain, with more parameters and training cost. Original GELU2L and cheaper SwiGLU2L remain references. |
| Final-third linear cooldown | [Schedule results](SCHEDULE_QUALITY_RESULTS.md): −0.071693 / −0.075062 at64/256 versus tuned constant LR | Adopted cooldown peak0.005 versus constant0.0025 at that stage. Different LR integrals and update/decay balance; not a pure schedule-shape effect. |
| Spaced-tail EMA, half-life200 updates (H200) | [Averaging results](AVERAGING_QUALITY_RESULTS.md): −0.005398 / −0.016797 versus the same trajectory's raw endpoint | Adopted inference weights. Raw and averaged outputs are correlated, not independent trainings. Keep the raw parent for optimizer resume. |
| Attention score multiplier0.5 with peak LR0.01 | [Attention-scale results](ATTENTION_SCALE_RESULTS.md): −0.006210 / −0.009851 versus tuned unit-scale LR0.005, every fresh seed better and all required intervals negative | Latest adopted recipe. Width64 seed2411 improves only0.000108; the rule requires mean gain>0.005, not that gain on every seed. Fixed-LR screens also favored half, but fresh confirmation includes an LR difference. |

The current recipe's adoption-round H200 CE was **1.509797 / 1.423060** at64/256. Its fresh control scores in the later logit-z round were **1.515571 / 1.428035**. These are different seed sets; do not pool them, treat one as a guaranteed score or retrospectively choose the better set.

## What did not transfer, and what to retain from it

These are results for the tested recipes and finite budgets, not universal rejections of the ideas. “Not adopted” can mean a frozen screening failure, insufficient mean gain, a worse seed, a width-transfer failure, or a text-interval veto. A promising later result cannot override the earlier frozen choice.

| Direction | Main learning | Evidence |
|---|---|---|
| More training | 6,000-update stretched cooldown improved64 by0.036435 but worsened256 by0.079981; every large-width seed worsened. Fixed-tail6000 also worsened256. More tokens were not automatically better. | [Horizon](HORIZON_QUALITY_RESULTS.md) |
| More capacity / MoE | Six layers improved64 by0.028180, but only0.000674 at256 with a worse seed; wider dense also failed256. Earlier MoE improved64 but worsened256 and cost substantially more. | [Capacity](CAPACITY_QUALITY_RESULTS.md), [MoE](MOE_QUALITY_RESULTS.md) |
| MLA / GQA / context | Some cache-saving configurations were conditionally acceptable, but no overall quality replacement. Context256/512 often worsened common-context128 CE. Half the KV cache did not guarantee a decode-latency win. | [Architecture](ARCHITECTURE_STAGE2.md), [Context](CONTEXT_INTERACTION_RESULTS.md), [Context256](CONTEXT256_RESULTS.md), [Inference](CONTEXT_INFERENCE_RESULTS.md) |
| Larger batch | B16/B32 were faster at equal tokens and passed the width256 +0.01 operational tolerance, but width64 seed failures prevented adoption across widths. Update count and cumulative decay were controlled explicitly. | [Batch](BATCH_SENSITIVITY_RESULTS.md) |
| KDA reference | H200 CE worsened by0.004848 /0.007836; training cost was7.442× /4.449× MHA. This was the sequential reference with a joint attention/position-family change, not an optimized hybrid-paper reproduction. | [KDA](KDA_QUALITY_RESULTS.md) |
| Q/K RMS normalization | Improved64 mean, worsened every256 seed; local gain also missed the threshold. | [QK normalization](QKNORM_QUALITY_RESULTS.md) |
| Output gating | Static half failed screening seed consistency; learned gating had mixed fresh seeds and no qualifying gain. Learned-minus-static at256 is a separate comparison, not adoption against unchanged attention. | [Output gating](OUTPUT_GATE_QUALITY_RESULTS.md) |
| SwiGLU hidden normalization | RMS/LN had small mean gains but worse seeds at both widths; not adopted. | [Hidden normalization](FFN_HIDDEN_NORM_RESULTS.md) |
| Independent decay | Stronger0.00018 improved256 by0.010659 with every seed better, but failed local/64 requirements. Weaker0.00002 worsened every seed at both widths. Keep0.00006; stronger decay is a scoped future hypothesis, not a retroactive promotion. | [Decay](DECAY_QUALITY_RESULTS.md) |
| Warmup | 100 updates passed screening but fresh gains0.004462 /0.000805 were below threshold and included worse seeds. Keep zero warmup. | [Warmup](WARMUP_QUALITY_RESULTS.md) |
| Input scale / final norm | Smaller input scales and bias-free final LN had some256 gains, but seed consistency or64 transfer failed. Keep input scale1 and affine final LayerNorm. | [Input](EMBEDDING_SCALE_RESULTS.md), [Final norm](READOUT_NORMALIZATION_RESULTS.md) |
| Vocabulary bias / tied tables | Unigram bias helped small width but hurt256; tied tables had a modest256 gain below threshold and no64 gain. Keep no output bias and legacy untied tables. | [Bias](OUTPUT_BIAS_RESULTS.md), [Tables](WEIGHT_TYING_RESULTS.md) |
| Parallel blocks | Parallel separate-norm blocks worsened every fresh seed at both widths: +0.026757 /+0.013555. Keep serial residual blocks. | [Topology](PARALLEL_BLOCK_RESULTS.md) |
| Activation / complete residual gain | GEGLU failed256; ReGLU worsened every seed. Both static and learned0.1 complete-branch gains worsened every seed at both widths. Keep SwiGLU/SiLU and unit residual gain. | [Activation](GATED_ACTIVATION_RESULTS.md), [Gain](RESIDUAL_GAIN_RESULTS.md) |
| First-layer values | Half-local and first-value blend did not qualify; blend worsened mean CE by0.008219 /0.013734. Keep local values. | [Values](VALUE_RESIDUAL_RESULTS.md) |
| Position | NoPE worsened0.124521 /0.063084; ALiBi worsened0.092128 /0.006974, every seed worse than RoPE. ALiBi beat NoPE, which is not sufficient to replace RoPE. No extrapolation claim. | [Position](POSITION_QUALITY_RESULTS.md) |
| Output logit-z penalty | Both penalties improved mean CE at both widths, but each worsened seed2513 at64 and2512 at256. No adoption. Some conditional text intervals are negative despite seed failures. | [Logit-z](LOGIT_Z_RESULTS.md) |

The practical pattern is **width transfer and training-seed consistency are harder than obtaining a lower small-model mean**. Negative results should narrow future proposals, not be deleted. Avoid cycling through the same grid without a new, preregistered reason.

## Exact incumbent to start from

The [portable template](snapshots/2026-10-27/incumbent-config.template.yaml) is derived from an actual width64 zero-penalty control request in the final completed logit-z study. It makes later semantic policies explicit. The [original request](snapshots/2026-10-27/incumbent-request.original.json) preserves exact historical paths/seed; **do not execute it into its old output directory**.

| Component | Retained setting |
|---|---|
| Network | Four serial dense blocks; MHA with head dimension32 (2/8 heads at widths64/256); RoPE; local layer values; attention logits `0.5 * QK / sqrt(32)` before causal masking/softmax |
| FFN / residuals | Bias-free SwiGLU hidden width21D/8, SiLU gate; unscaled/unit residual branches; no output gate, hidden-product normalization or QK normalization |
| Normalization / readout | RMSNorm blocks; affine final LayerNorm epsilon1e-5; input embedding scale1; legacy untied embedding/readout; no vocabulary output bias |
| Optimizer | Reviewed Muon/AdamW hybrid, μP base64; independent decay0.00006 once per participating parameter per update, inner optimizer decay0; global clip1; dropout0 |
| Budget / data | 3,000 updates, context128, batch8, paired_blocks512; 3,072,000 training targets; Shakespeare contiguous80/10/10 split; sorted global character vocabulary65 |
| LR | Peak0.01, no warmup; multiplier1 through zero-based update2000, then `1 - .9*(i-2000)/999`, ending0.1 at2999 |
| Returned weights | H200 spaced-tail EMA initialized after completed update2000, sampled every10 through3000:101 snapshots; beta=`2**(-10/200)`; learned FP32 parameters only; raw buffers retained |
| Objective / evaluation | Training pure CE (logit-z coefficient0); full validation111,538 targets once at context128/B8; H200 primary, raw endpoint diagnostic; test split remains closed |

Template seed and output paths are mandatory placeholders. Choose a new unused seed/run ID and preserve resource limits before a new run. Changing width requires matching `dim` and `n_heads=D/32`; do not alter fixed head size or invent new width/depth scaling. A standalone trainer command does not reproduce the study controller's deadlines, complete validation, frozen selection or audit.

Corpus SHA256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`. The original tokenizer uses the global character alphabet, including characters present outside training; test sequence losses were not used for selection. Training-prefix prior counts are not sampler-weighted target counts.

## Evidence rules worth keeping

- Freeze a finite hypothesis, source/corpus/runtime hashes, full prior knowledge, grid, seeds, budget, primary endpoint and adoption rule **before** quality training. New architecture needs independently checked geometry, optimizer roles, gradients, caching, resume and resource feasibility.
- Screen at small width, freeze the selected recipe before fresh confirmation, and keep all planned arms. No larger-width reselection, fallback winner, grid extension, pruning or historical control pooling. Exact reference aliases are identities, not independent runs.
- Recent studies required mean improvement>0.005 and every paired seed negative at both widths against the required fresh controls, plus negative upper bounds for all declared text-bootstrap block sizes. H200's own study used0.002. **Each historical plan governs its own rule**; do not retrofit later criteria.
- Bootstrap text blocks2048/4096/8192 with2,000 draws are conditional on these trained models and reused text. They do not measure training-seed population uncertainty, correct adaptive multiple comparisons or replace an independent holdout.
- Equal tokens/updates are not equal FLOPs, parameters, time or LR integral. Keep selected-LR recipe contrasts separate from fixed-LR contrasts; keep measured costs separate from forecasts and synthetic correctness.
- MPS independent trajectories were not bitwise reproducible in earlier checks. Preserve the failures and the direct coupled arithmetic checks; no kernel cause was identified. Exact CPU resume evidence does not establish independent MPS replay.
- Preserve failures and repair only the identified layer. Examples include an FP64 oracle incorrectly standing in for the declared FP32 RoPE arithmetic, a temporary-checkpoint `lstat` race, retained scratch accidentally counted as new scratch, and a duplicate-key metadata assembly error. Never hide these by retraining or relaxing a failed threshold.
- Final tensors cannot reconstruct every historical EMA snapshot, objective invocation or cache operation. They complement frozen source plus independently tested CPU/MPS execution; they do not replace it.

## Exact stop point: unfinished null attention

[Paused prototype branch](https://github.com/JunyiJ/Practical-Decoder-autoresearch/tree/codex/null-attention-paused-20261027), commit **`007eb93190e98371d83ab2b26e3b3dce89838e34`**, based on completed findings commit `2f69633da1a2c560952b1738db2c47a0927e9133`. Its full production digest is `c788d543c1451b01c6fcd5c22f05a14fd8a03f7239261b0cb71f4d9cdea18dc4`. This code is deliberately separate from main's completed-study source.

The proposal compares standard softmax, `n/(n+1)` visible-prefix attenuation, and an always-visible fixed-zero-score/zero-value null option. The null is not a learned parameter or cache slot. It is added after content scaling/position bias/causal masking; joint normalization includes the null in the stable denominator. It is not invariant to a common shift of the real logits. The prefix control matches all-zero scores, not actual initial attention mass or complete trajectories.

- **28 actual CPU cases passed**: geometry, all-parameter gradients, primitive extremes/masks, attached first-value gradients, cache prefixes/offsets, pure-CE evaluators, optimizer/RNG, H200 and resume.
- **Full5,973-case suite incomplete**:3,367 IDs in90 completed batches plus8 passed IDs in interrupted `test_position_family_training.py`; **3,375 unique passed assertions,2,598 incomplete/unexecuted IDs**. The latter include the interrupted case. Exact lists are in [stop review](snapshots/2026-10-27/null-cpu-stop-review.json).
- No null MPS pilots, measured resource forecast, quality study registration or quality training occurred. Thus no null-attention quality conclusion exists.
- Primitive/attention/cache/linked-graph/resource helper packages were prepared. The newest pilot driver/reviewer had9 pure metadata/AST test methods pass; neither real CLI was executed. Its final integration review and all actual pilot evidence remain pending. Research screening/full-completion/registration/reporting helpers are not yet prepared.
- The controller's generic `failed / KeyboardInterrupt()` is a **human stop**, not a model assertion failure. The partial batch lacks a final pytest summary and observed worker exit code; do not claim complete timing/warnings or a whole-file pass.

The interrupted scratch, all old failed scratch and original logs are retained locally. Successful earlier scratch was removed only after its passed assertions and path/size/hash inventory were preserved. Do not delete retained failed evidence to manufacture a clean resume.

## Resume later on this machine

Only after a new explicit human request:

1. Read this guide, `research_state/HUMAN_STOP.json`, the complete knowledge bank and the relevant final predecessor review. Inspect live processes and logs; no automatic training restart. Preserve006's permanent stop.
2. Choose whether to continue null preparation or propose a different finite experiment. Null was a hypothesis, not a selected quality improvement.
3. Check out the paused prototype in an isolated checkout. Verify the source digest, all recorded test hashes, logs and cleanup receipts. Use the original primary `.venv`, unset MPS fallback and keep one model worker. Do not modify the human's primary checkout.
4. For the same CPU validation, write and review a **new continuation** against the exact2,598 remaining/interrupted IDs. Exclude all3,375 already-passed IDs. Keep the original manifest/log/scratch; use a new output namespace and fresh active scratch. Account for all prior time and retain the existing900s actual /6000s remaining budget. Do not run the fresh driver or final full-pass recorder blindly against the interrupted state.
5. After a complete exact-ID audit, preserve the tested protocol, reconcile status-only prose, publish the clean tested source, then run six **fresh serial**40-update MPS pilots with the reviewed null-aware attention/cache/objective proofs. Prospective pilot seed9517 was absent from92 inspected plan/intent files, but must be rechecked. No pilot was run or research seed reserved by that check.
6. Keep240s/job,1800s/round,1024MiB pilot caps. Independently audit saved checkpoints, optimizer/RNG/H200/pure-CE coverage,45-row null and12-row objective primitive panels, linked gradients,96 real KV prefixes and108 reconstructed attention outputs, plus the50% resource forecast margin. Preserve any failure; no automatic retries.
7. Only then finish/review registration, screening, final-audit and reporting helpers and freeze a **new unique** research contract. Proposed research seeds2521/2522 and2531/2532/2533 are not proof of availability; rescan. Proposed42-job maximum,2400s/job,21600s/round,4096MiB artifacts,4096MiB sampled process RSS and6144MiB current MPS driver limits remain unchanged.
8. Explicitly record the new human authorization before changing the stop flag or re-enabling an automation. Re-enabling the old heartbeat alone is not a safe restart procedure.

## Reproduce or restart on another machine

A reproduction is a **new study**, not a reopening or extension of completed evidence. The human may authorize repeating an old design, but use a new run ID/output tree and label whether seeds are historical replay seeds or fresh replication seeds. Never mix the new measurements into the original report.

1. Clone the repository and verify the portable snapshot (command below). Read the original plan, report, selection and review; use the plan's historical source commit for an exact code reproduction, or declare a new source as a changed-condition experiment.
2. The metadata bundle preserves absolute original paths as evidence. Extract into a separate directory; map paths explicitly to a new checkout/corpus/output root. Do not rewrite frozen files in place or assume the archived launch helpers are portable executables.
3. Verify the corpus hash and recorded runtime. If platform, library versions, device or numeric behavior differ, report that difference and repeat correctness/resource prerequisites before quality training. Do not expect bitwise MPS replay.
4. Either restore the original full local evidence archive for tensor auditing, or train new models under a new finite preregistration. Metadata alone cannot regenerate old optimizer states/checkpoints or certify the old tensor audit independently.
5. To continue a specific model, restore its **raw training parent**, matching optimizer/config/RNG/runtime and total-step contract. H200 derived checkpoints are inference-only and intentionally cannot resume the optimizer. A new budget question needs a new study rather than extending a closed grid.

Read-only verification from repository root:

```bash
python3 research/snapshots/2026-10-27/verify_snapshot.py
```

Optional metadata extraction, which starts no model or controller:

```bash
python3 research/snapshots/2026-10-27/verify_snapshot.py --extract /absolute/path/to/new-empty-research-metadata
```

For an authorized future null-source recovery:

```bash
git fetch origin
git worktree add --detach /absolute/path/to/new-null-checkout 007eb93190e98371d83ab2b26e3b3dce89838e34
```

There is intentionally no one-line “resume everything” command: historical helpers bind absolute evidence paths, the CPU suite is partially complete, and no null research run exists to resume.

## What GitHub contains versus what must be backed up separately

The [snapshot manifest](snapshots/2026-10-27/metadata-manifest.json) hashes every bundled member. It contains canonical knowledge JSON entries, all27 conclusion records, run-root plans/reports/reviews/selections/state metadata, research helper source, current null preparation logs/manifests, stop records and selected historical failure reviews. Original metadata bytes are preserved. The latest prototype source is on its separate branch.

**Not uploaded:** model weights, optimizer checkpoints, most trial-level raw worker directories, tensor capture payloads, corpus bytes and retained scratch checkpoints. Preserve the original local `research_runs/`, `research_state/`, data, and retained failed/interrupted `/private/tmp` directories in a separate backup if complete independent auditing or optimizer continuation is needed. `/private/tmp` is not durable storage; copy it to a durable archive before machine cleanup. The metadata manifest explicitly states omissions.

The original evidence root is `/Users/junyi/Documents/ChatGPT/LLM-pretraining-RSI`; the research checkout is `/Users/junyi/.codex/worktrees/rope-position-fix/LLM-pretraining-RSI`. These are provenance, not portable defaults. No experiment has been restarted as part of writing this handoff.
