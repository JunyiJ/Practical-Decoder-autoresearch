# Proposed native pseudo-repetition quality study

## Authority, evidence and hypothesis

This is a new direction under the general RSI research authorization. The
human-priority native-horizon and composer-FP32 panels are complete. Preserve
all closed comparisons, original failures and selected/fresh models. No paid
compute, subagents, batch reuse or deferred output-lexical work.

Local evidence: `GENERAL_RSI_CONTEXT_ORDER_RESULTS.md` and its hash-bound281
paired cases show that disrupting earlier order reduces repeat probability in
22/23 terminal loops, but also harms natural repeat prediction. The earlier
`GENERAL_RSI_REPETITION_PROBABILITY_RESULTS.md` reproduces all2,304 greedy choices
and finds rising probability in18/23 loops. Lower native validation CE did not
establish stronger semantic generation. These facts justify testing a targeted
objective; they do not establish that it works or that generic repetition is bad.

External evidence, read in the preceding boundary review: Xu et al.,
[Learning to Break the Loop, NeurIPS2022](https://papers.nips.cc/paper/2022/file/148c0aeea1c5da82f4fa86a09d4190da-Paper-Conference.pdf),
§3/Eq1 mixes ordinary MLE updates with penalties on synthetic repeated sentences.
Their λ0.5 and equal mixing are reported settings for open-ended generation,
not local optima. §4.2/Table1 reports prediction/generation gains, while warning
through competing results that lower repetition need not improve relevance.
Their750M word-level/context1536 setup is far larger than ours. Welleck et al.,
[arXiv:1908.04319v2](https://arxiv.org/pdf/1908.04319v2), §5/Table2, supplies the
unlikelihood foundation and a contrary token-UL perplexity result. Li et al.,
[NeurIPS2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/e6c2e85db1f1039177c4495ccd399ac4-Paper-Conference.pdf),
§7, explicitly notes possible perplexity damage from repetition dropout.

Inference: retaining ordinary CE while penalizing artificially redundant full
copies might reduce learned loop support with less damage to useful contextual
prediction than indiscriminate token penalties. Falsifiable prediction: at a
matched incremental Mac budget, the candidate improves final validation CE under
the established superiority rule. Repetition and semantic ratings remain separate
diagnostics. A loss rejects the recipe at this setting. Inconclusive CE, even with
fewer loops, does not justify a validated overall-quality claim or equivalence.

## Proposed comparison, before any new training scores

Two arms: ordinary native CE continuation and native CE plus pseudo-repetition
updates. Both start from each of the same three selected14,400-second native
averaged checkpoints, slots01/03/05. Each parent creates one fresh paired
continuation bundle. This is conditional adaptation of three existing parents,
not fresh pretraining initialization or independent replication of the parents'
selection. Old scores cannot serve as the new controls or enter confirmation.

Preserve D256/four blocks/context128/GPT-2 IDs and optimizer policies. Initialize
new optimizer state identically in both arms. Use fresh declared sampler/RNG
roles, identical initial model weights within each pair and distinct output paths.
No best-parent selection. Freeze roles, order, full implementation and resource
contract before launching the six-run panel. Use all three pairs for confirmation.

Proposed incremental training budget:1,200 synchronized training seconds per run,
7,200s total. This is a modest8.33% of the parent14,400s training budget, a cost
choice rather than an optimum. Include pseudo-data construction, loss computation
and optimizer work in the clock. Use initial LR0.00025, the parent's terminal
LR, for both arms, with the existing final-third cooldown to0.1. Reset optimizer
and averaging state in both arms and disclose that common adaptation policy.
Use the inherited qualified time-EMA rule and RAW secondary. No LR screen or
post-score hyperparameter search is proposed for this first targeted test.

Candidate alternates one ordinary CE update and one pseudo-repetition update,
starting with CE, corresponding to the paper's equal update mixture. Baseline
uses CE every update. Each uses1,024 prediction positions/update. Because the
mixture changes ordinary-token exposure, this tests the complete adaptation
recipe at equal time, not an isolated penalty at equal natural-token exposure.
Report natural and synthetic target counts separately. Do not call synthetic
positions additional natural-corpus exposure. Keep sampler streams separate so
pseudo draws cannot perturb the ordinary-CE stream; pair its natural prefix
stream between arms. Record all timing and any lost work.

For pseudo updates, construct repeated complete training-text spans that fit
context128, with at least three complete copies and a retained preceding context
where available. No validation prompts or model-generated failures enter training.
The next implementation must freeze an explicit deterministic sentence/span
segmentation and length-eligibility inventory from training only, including how
partial trailing copies and padding are masked. Do not silently call arbitrary
fragments complete sentences. If robust sentence extraction is unavailable,
name the method span repetition, disclose the departure from DITTO and justify
its tokenization/length policy before qualification and panel freeze.

Use the paper's loss −log(1−abs(p_current−0.5*stop_gradient(p_previous_copy)))
on aligned targets after the first complete copy. Retain CE on ordinary updates.
Freeze exact reduction, numerical protection and valid-target masks in new source.
Do not choose a penalty coefficient by replaying the already observed prompts.
No full candidate is qualified merely by writing this equation.

## Decision and outputs

Primary: all108 validation documents/100,416 targets, final averaged endpoint.
Use the established symmetric three-pair rule: all paired deltas favor the winner,
mean magnitude>=0.02nats/token, one-sided99% document-bootstrap upper<0 and no
source regression>0.05. Fresh bootstrap seed declared before training. Neither
passes means inconclusive. Do not weaken the gate after observing CE.

Retain all12 final RAW/averaged checkpoints, first144 completions from the six
averaged endpoints using the unchanged12 prompts and two decoders, no rerolls.
Blind all new arm/parent identities for the fixed semantic rubric. Report actual
annotator and its limits. Repetition/length/EOS/relevance/consistency remain
separate from CE. Test stays closed. No new generation quality composite.

## Next executable engineering and cost boundary

Implement only the new training-span sampler, aligned-copy loss and integration
needed for this comparison outside frozen namespaces. First inventory eligibility
on existing training data, then freeze a bounded integration fixture. Check the
new loss against a scalar/gradient oracle, detach/masks and finite state; verify
only the added sampler/phase state in complete restart and timing. Existing
model loading, inference, scoring, generation and clock prerequisites remain
qualified. Use exactly identified attempts for any implementation failure.

Provisional fixture ceiling900worker seconds and2GiB retained outputs. It may
run only after its concrete resource admission. No quality/short-pilot CE gate.
Measure complete update time, native optimizer/restart size, pseudo-target density,
memory and serialization overlap before admitting scientific training. A source
or small scalar-only check does not require allocating the full training panel.

Full-panel planning ceiling4worker hours includes7,200 training seconds, evaluation,
generation, I/O and failures. Replace the estimate with measured complete cost;
never shorten the six-run comparison or reduce retention to fit. Provisional
storage caps: twelve128MiB final weights, two512MiB active restart/write slots,
1GiB retained qualification restarts,256MiB metadata,5GiB failed-write allowance,
plus10GiB reserve =20,132,659,200bytes free. The2GiB fixture ceiling is separate;
if all of it must remain, add another1GiB to the panel forecast. Existing retained
outputs are already charged by actual free space. Check serialization overlap
for new optimizer/sampler state and qualify any transient retirement policy.

This forecast is not admission. At measured admission, execute authorized exact
audited cleanup if required, aiming at full need plus2GiB headroom. Do not open
a training panel on a partial capacity forecast. Preserve all current evidence
and fresh weights. If the fixture or candidate requires changes to this proposed
policy, document engineering reasons before scientific scores and freeze the
final contract once, rather than adding opportunistic arms.
