# Pseudo-repetition mechanism boundary

The completed negative is not explained solely by fewer ordinary CE batches.
The retained trajectories already differ adversely at matched natural-batch
ordinal and identical constant learning rate. This is a post hoc training-loss
diagnostic. It does not replace the frozen validation decision or isolate the
penalty gradient from extra optimizer advances.

## Measured local evidence

`tokenizer-evidence/general-rsi-pseudo-repetition-001/decision.json` records
ordinary-CE superiority, mean candidate regression+0.144614982nats/token.
The original final comparison has unequal natural exposure,49.35–49.58% for the
candidate. Its fixed RAW probe gaps grow across all three pairs, while generation
loop rates improve and descriptive relevance ratings decline. All original
results remain unchanged.

The new `general_rsi_pseudo_matched_prefix_analysis_v1.py` reads those immutable
logs. It aligns each candidate ordinary-CE batch with the same ordinal batch in
its paired control and verifies exact batch hashes and1,024 targets. The primary
diagnostic window includes every aligned row for which both recorded LR ratios
are exactly1, before either cooldown. Losses are measured before the current
update, so each aligned row has seen the same number of prior natural CE batches.
The candidate has additional pseudo updates. No model, corpus contents or original
test is opened; no optimizer step or additional quality scoring is executed.

| Pair | Constant-LR matched batches | Mean candidate loss delta | Positive deltas | First-batch delta |
|---|---:|---:|---:|---:|
|1|1,157|+0.1380888581|1,151|0|
|2|1,158|+0.1450513074|1,156|0|
|3|1,158|+0.1314525190|1,153|0|

`general-rsi-pseudo-mechanism-001/matched-prefix.json` retains all aligned rows,
100-batch summaries and input hashes. These correlated training observations
are not independent confirmation samples. No inferential confidence interval is
claimed. Full-prefix results are also saved but include different cooldown states.
The endpoint exposure tradeoff remains real even though it is not a sufficient
explanation of these matched-prefix gaps.

All candidate CE updates were clipped at norm1. Median pseudo gradient norms were
0.4323,0.4461,0.4404, with only2,3,1 clipped updates. Low pseudo loss or norm is
not evidence of negligible parameter movement. Inspection of the installed
PyTorch2.14.1 Muon source shows momentum mixing, norm normalization before
Newton–Schulz, and weight decay on every update. The same optimizer instance
receives CE and pseudo gradients. Exact source receipts are in
`optimizer-inspection.json`. This is a source fact, not measured update attribution.

## Primary literature and applicability

Xu et al., [DITTO, NeurIPS2022](https://papers.nips.cc/paper_files/paper/2022/file/148c0aeea1c5da82f4fa86a09d4190da-Paper-Conference.pdf),
§3/Eq1 and§4.1 use equal pseudo/MLE mixing and10k fine-tuning steps from a
validation-selected MLE model. §4.4 discusses mixing and penalty-factor sweeps:
gamma0.5 is preferred, while lambda depends on task. §4.2/Table1 reports a
perplexity benefit. The inspected sections do not isolate equal natural exposure
and identical optimizer state. Their750M word-level/context1536 setting and
validation selection differ from this28.5M/context128 final-endpoint experiment.
The supplemental appendices were not in the retrieved14-page paper; absence of
all possible controls is not established. Figure screenshot retrieval failed;
no new curve coordinates are claimed.

Welleck et al., [arXiv1908.04319v2](https://arxiv.org/pdf/1908.04319v2),
§6 Training uses1,500 sequence-level fine-tuning updates with probability0.5 of
unlikelihood updates. §6.2/Table2 reports MLE perplexity25.64, token-UL26.91 and
sequence-UL25.42. Thus the reported prediction tradeoff depends on the objective.
The inspected setup does not supply our missing optimizer-state control. Its
scale, word vocabulary, sequence construction and model selection differ locally.
These reported results cannot override the measured local negative.

## Inference and ranked next actions

1. **Prioritize a bounded common-state update diagnostic.** The adverse matched
   prefixes and shared optimizer motivate separating gradient effects from extra
   momentum/decay steps. Use each of the three retained native parents, with
   identical reset optimizer state followed by one ordinary CE update to seed
   momentum. Clone that complete state for each comparison. Compare identity,
   explicit zero-gradient optimizer step, pseudo-gradient step and ordinary-CE
   step. Use separate fixed training-only evaluation batches and retain every
   case. Explicit zero gradients are essential: missing gradients can skip an
   optimizer step and would not be the intended momentum/decay control.
2. **Defer a full equal-natural-exposure training panel.** It would alter cost
   and schedule and still confound pseudo gradients with extra optimizer advances.
   The existing matched-prefix evidence makes it a less discriminating next use
   of compute. Any future such panel needs fresh roles, controls and full admission.
3. **Do not run a coefficient/mixing sweep now.** No evidence identifies a new
   optimum. Lower pseudo loss does not imply proportional Muon parameter updates.
   Replaying the closed recipe would add cost without isolating the mechanism.

Falsifiable diagnostic predictions: if pseudo versus zero-gradient advances
consistently raises independent natural-training loss, that supports direct
penalty-gradient interference at the sampled states. If zero-gradient versus
identity explains the shift, optimizer advance/decay is the stronger lead.
Mixed or negligible effects reject a simple one-step explanation and favor
examining accumulated state or span distribution. Natural-CE steps provide a
positive learning control. Measure parameter displacement and gradient/update
alignment separately by Muon and AdamW groups. Do not infer a long-run quality
gain or refute the completed quality result from this diagnostic.

This is a new mechanism question, not a retry of any failed qualification.
Reuse the qualified model loader, sampler, optimizer and complete-state capture.
Only qualify new diagnostic branching and accounting if a demonstrated need
appears. No batch-reuse arm, generation reroll or validation/test tuning is added.

## Cost and next executable action

The diagnostic is not yet admitted. A provisional layout is3 parents×8 sampled
cases×4 branches. It retains scalar measurements and exact input/state digests,
not new final models. Budget72 optimizer steps,96 fixed8,192-target evaluation
passes, baseline evaluations, pseudo-forward measurements, state cloning and
three model loads. Existing measured costs suggest about25s for updates and172s
for branch evaluation; those are extrapolations, not a new timing qualification.
Baseline evaluation, loading/cloning, gradient checks and failure overhead must
be measured or conservatively charged before launch under a600s total ceiling.

Provisional disk caps are16MiB retained measurements plus16MiB atomic-write
overlap and the existing10GiB reserve. No checkpoint serialization is planned;
if failure handling needs one, include its full measured payload and overlap
before admission. RSS4GiB/MPS6GiB limits and the shared nonblocking worker lock
still apply. A future quality panel would require its own complete retained-output
forecast; this diagnostic cannot stand in for confirmation.

Next implement the minimal diagnostic outside all frozen namespaces, declare
training-only input selection, fresh roles, exact branch order and complete cost,
then freeze admission before running it. Stop short of interpreting preparation
as numerical exploration. No training worker is currently active.
