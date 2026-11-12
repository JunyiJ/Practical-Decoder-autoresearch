# Repetition research boundary — 2026-11-11 Pacific

## Decision

Prioritize a bounded inference diagnostic of repetition reinforcement on the
three retained native long-horizon models. No new training panel is admitted.
Do not rerun native horizon, equal-time, exposure, transfer or composer FP32
quality. The latter remains inconclusive, and its original numerical failure
remains unchanged. No decoder tuning, new generations, test access or checkpoint
selection is needed for the diagnostic.

## Measured local evidence

`tokenizer-evidence/general-rsi-repetition-boundary-001/analysis.json` binds the
new analysis source, original six native execution reviews and generation
results, generator/scorer source, and training files by SHA256. Command:
`.venv/bin/python -m research.general_rsi_repetition_boundary_v1`.
The nonblocking shared research lock was acquired. Three synthetic onset cases
passed. Existing generation metrics recomputed exactly. Analysis took0.325s,
read no model weights and produced no new completions.

- All12 prompts contain4–31 GPT-2 tokens. All144 retained native continuations
  fit within95 total tokens. The generator asserts prompt+64<=128. Thus,
  context-window eviction cannot explain these particular failures. A larger
  training context could still change learned representations; this analysis
  does not test that separate hypothesis.
- Among36 greedy outputs per horizon, all contain a repeated generated trigram.
  First repeat endpoint median is23 tokens for short and18 for long. In both,
  35/36 repeat by token32. Terminal loops occur in15/36 and23/36 respectively.
  Mean duplicate-trigram fractions are0.553763 and0.603943. These are descriptive
  observations, not a newly defined hypothesis test or a changed old decision.
- Sampled outputs have no terminal loops. Mean duplicate fractions are0.022141
  and0.018817. Their original weak relevance/consistency ratings remain unchanged.
  Removing repetition alone is insufficient evidence of useful prose.
- One centered64-token non-EOS window was taken from each of9,807 training
  documents. FineWeb-Edu windows:7,815, duplicate fraction0.013283. Wikipedia:
  1,992, fraction0.024906. Neither group has a terminal triple-repeat suffix.
  This is an equal-document descriptive inventory, not matched prompts, an
  exhaustive search for all corpus loops, or a random population sample.
  Token trigrams include legitimate repeated phrases, subwords and punctuation.
- Native's frozen CE result remains favorable to the long recipe:
  `general-rsi-native-horizon-001/decision.json`, mean delta−0.141445795,
 99% upper−0.128041946. Its generation completion review retains all144 ratings.
  `general-rsi-composer-fp32-quality-001/decision.json` and `history-analysis.json`
  show mixed RAW and H200 pair orderings. Neither additional FP32 arithmetic
  precision nor lower CE has established stronger continuation quality here.
- Most continuations hit64 tokens. This cap does not establish an EOS calibration
  failure. Centered corpus windows do not supply a termination reference.

## Primary literature read for this boundary

External findings below are distinct from local measurements. Versions and
sections were retrieved on2026-11-12UTC. No local gain follows from them.

- Welleck et al., *Neural Text Degeneration with Unlikelihood Training*,
  [arXiv:1908.04319v2, 26Sep2019](https://arxiv.org/pdf/1908.04319v2),
  §§5.1–5.2 Eqs3–7, §6/Table2. Token UL penalizes prior context tokens excluding
  the true next token. Sequence UL targets repeated generated n-grams. Table2
  also shows token UL worsening perplexity25.64→26.91 while reducing repetition;
  the paper distinguishes sequence-generation and prediction objectives. Its
  16-layer word-level Wikitext-103 model/context1536 and fine-tuning compute differ
  from our four-block GPT-2-token model/context128. Sequence rollout costs must
  be measured locally. OpenReview access challenged and HTMLv4 failed; this
  record relies on the successfully retrieved v2 PDF, not those failed fetches.
- Xu et al., *Learning to Break the Loop*,
  [NeurIPS2022 proceedings PDF](https://papers.nips.cc/paper/2022/file/148c0aeea1c5da82f4fa86a09d4190da-Paper-Conference.pdf),
  §§2.1–2.2, §3 Eq1, §§4.1–4.2/Table1. They measure rising token probability
  under repeated sentence context and use pseudo-repetition penalties. Their
  comparison finds stronger repetition suppression by UL-token+seq but weaker
  MAUVE than DITTO, with relevance drift discussed. Necessary repetitions also
  matter. The750M word-level model/context1536,100M-word corpus and fine-tuning
  differ greatly from ours. Their mechanism motivates measurement, not direct
  adoption of their penalty or mixing coefficient.
- Li et al., *Repetition In Repetition Out*,
  [NeurIPS2023 proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/e6c2e85db1f1039177c4495ccd399ac4-Paper-Conference.pdf),
  §4/Fig2, §5.1/Algorithm1, §7. Repetition in natural training text correlates
  with generated repetition across data shards. Repetition attention dropout
  reduces loops, but their limitations explicitly include possible perplexity
  damage. Natural repetition is not automatically bad data. GPT-2/OPT/Llama2,
  domains and n-gram definitions differ. Our low centered-window loop count
  does not refute their account or justify deleting useful repeated entities.

## Ranked hypotheses and decision value

1. **Repetition reinforcement / objective intervention.** Local greedy loops
   arise early despite retained context. Inference: repeated phrase context may
   raise the probability of another copy. Measure before training. If copy
   probability rises across most affected outputs in each seed, a controlled
   objective intervention becomes better motivated. Flat/falling probabilities
   weaken this mechanism and redirect attention to stable high-ranked outputs
   and poor conditional discrimination. Neither outcome proves a causal account.
   A later comparison must include continued ordinary CE from identical starting
   weights, matched resources, CE/source guards, length and semantic diagnostics.
   No coefficient or adoption gate is selected from the present repetitions.
2. **Data-dependent repetition learning.** Local natural windows contain some
   legitimate repeats, and literature supplies an alternative mechanism. A later
   attention intervention would require a matched random-mask control to separate
   repetition specificity from generic regularization. Corpus deletion would
   confound information loss and domain mix. Defer such training until probability
   evidence distinguishes it from broad loss/decoder effects.
3. **Longer context or larger model.** These may improve prediction and semantics,
   but eviction is absent in the observed examples. No local scaling mechanism
   yet discriminates this expensive change. Defer a full panel; do not infer that
   a larger model solves repetition or replay the closed longer-horizon study.

These rankings are qualitative. None is a validated improvement. No exploration
training budget is charged for this existing-data analysis.

## Next executable diagnostic and complete cost boundary

Implement outside frozen namespaces and freeze inputs before any new model
probabilities are inspected. Score all72 original long-horizon outputs, preserving
all sampled/greedy cases and each model's own output pairing. Load all three
slot01/03/05 time-EMA checkpoints. Use corrected FP32 inference and retain per-token
log probability, argmax identity and rank for original generated tokens. For each
original terminal loop, use the smallest period already reported by the frozen
scorer and compare aligned tokens across its last three complete copies. Record
all exclusions and per-seed results. This is conditional teacher-forced replay,
not regenerated output or fresh confirmation.

As a natural-text reference, select32 training documents per source by ascending
normalized SHA256, without using model scores. Use the centered64-token windows
already defined here. Score the same64 windows under each of the three models.
Thus72+192=264 sequences, at most95 tokens each for completions and64 for natural
windows. Repeat positions have different prefix lengths and positions; natural
windows are not causal matched controls. This first diagnostic can falsify a
simple increasing-probability description but cannot identify the effect of
repetition in isolation. A causal intervention, if justified, needs a subsequent
prospective matched-context design.

Prospective diagnostic ceiling:600 total worker seconds, one serial worker,
RSS4GiB/MPS6GiB,64MiB retained result allowance, no new checkpoint or generation.
Existing weights remain read-only. Verify hashes, source/evaluator integration,
actual free space >=10GiB reserve+64MiB, processes and nonblocking research lock
before launch. Reuse qualified inference; test only new indexing against a small
oracle and exact original greedy argmax reproduction where applicable. Save any
mismatch as an engineering failure without replacing the original outputs.
No training/validation/test data beyond named training windows and retained
outputs is needed. Write outputs incrementally and preserve failed attempts.

A full intervention study is deliberately not admitted: objective overhead,
negative-candidate density, restart payload and suitable generation criteria are
unmeasured. For planning only, a two-arm three-pair native panel at14,400 training
seconds each would cost24training hours before screens, qualification, evaluation,
I/O, failures and objective overhead. Twelve native final weights alone are
1,367,984,196bytes; two445,008,055byte restart payloads add890,016,110bytes before
new objective state. With256MiB metadata,5GiB failure allowance and10GiB reserve,
this lower-bound forecast requires18,632,563,122 free bytes, plus all screens and
new qualification outputs. It is not a current capacity request or admission.
Complete measured costs and fresh roles/controls/gates must precede any panel.
