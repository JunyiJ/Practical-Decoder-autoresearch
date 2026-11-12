# Accumulated-state diagnostic, 2026-11-12

## Evidence and changed question

The closed pseudo-quality panel regressed by0.144615nats/token. Existing matched
natural-batch logs show positive mean gaps within the first100 updates in all
three pairs, increasing in the next100. The subsequent common-state diagnostic
found mixed pseudo-minus-zero effects after only one prior CE update. Its CE
controls also raised immediate disjoint-window loss. Thus the one-step setup
does not establish the longer-run mechanism. Original evidence:

- `tokenizer-evidence/general-rsi-pseudo-repetition-001/decision.json`
- `tokenizer-evidence/general-rsi-pseudo-mechanism-001/matched-prefix.json`
- `tokenizer-evidence/general-rsi-pseudo-update-001/analysis.json`

The primary-literature basis remains the actual sections reviewed in
`GENERAL_RSI_PSEUDO_MECHANISM_BOUNDARY.md` and its `literature-review.json`.
Xu et al., [DITTO, NeurIPS2022](https://papers.nips.cc/paper_files/paper/2022/file/148c0aeea1c5da82f4fa86a09d4190da-Paper-Conference.pdf),
§3 and§4.1–4.4 motivate interleaved natural/pseudo training but do not establish
this local equal-natural-count optimizer control in the inspected sections.
Welleck et al., [arXiv1908.04319v2](https://arxiv.org/pdf/1908.04319v2),
§6 and Table2 report differing prediction effects for token and sequence
unlikelihood. Scale, vocabulary, context, optimizer and model-selection differences
limit transfer. Published gains do not reverse the measured local negative.

Inference: accumulated optimizer advances may explain some trajectory divergence;
pseudo gradients may add an adverse accumulated effect absent in the first-step
comparison. Both remain hypotheses. This diagnostic distinguishes those effects
at matched natural exposure. It does not rerun the original equal-time quality
comparison or select a coefficient, decoder, checkpoint or improved model.

## Frozen controls

Use all three selected native parents, slots01/03/05, with reset optimizer state
as in the closed continuation setup. Declare new paired roles before execution.
Each parent has three serial trajectories:

1. Ordinary CE:100 natural updates.
2. CE plus zero:100 natural updates, with99 explicit zero-gradient optimizer
   advances between them. Every parameter gets a zero tensor, not missing grads.
3. CE plus pseudo:the same100 natural batches, with99 original pseudo updates
   between them. The pseudo sampler and objective remain qualified and unchanged.

All arms begin at exactly the same complete parent state. Preserve the canonical
two512-token-block natural sampler,8×128 layout, LR0.00025, Muon/AdamW grouping,
clipping1.0 and independent decay0.00006. LR is constant throughout this short
mechanism test. There is no cooldown or EMA, and no final model is adopted.
Zero and pseudo each advance the same optimizer199 times; CE advances100 times.
The zero contrast combines momentum advance and decay, not a separation of them.

Evaluate after natural updates1,10,25,50,100, before the next extra update. At
update n, CE has n optimizer steps and each interleaved arm has2n−1. Also score
the unmodified parent once. Check exact equality after the first CE update across
arms. Rotate arm order across parents to avoid a single fixed thermal order.
Every raw update loss, clipping record, input hash and probe is retained.

Use64 fixed training-document-local128-target windows per parent. Select them
uniformly by eligible document then start, excluding all source positions used
by the100 CE batches and99 pseudo batches, and excluding overlap between windows.
These windows were available during parent training; they are not held-out quality
evidence. Freeze their inventory before model execution. Reuse the corrected
inference RMSNorm and qualified scoring arithmetic. Validation is not scored and
test is not opened. No generations or model checkpoints are produced.

## Falsifiable interpretation

Report all paired probe curves, per-parent endpoints and descriptive means for
CE+zero minus CE, CE+pseudo minus CE+zero, and CE+pseudo minus CE. No confidence
interval, superiority gate or early stopping by score applies.

- Consistently adverse zero-minus-CE trajectories support an accumulated
  optimizer-advance/decay contribution under this protocol.
- Consistently adverse pseudo-minus-zero trajectories support an additional
  accumulated pseudo-gradient contribution, including optimizer interactions.
- Mixed or negligible contrasts weaken those simple explanations. If the early
  gap does not appear, report that limit rather than extending the horizon after
  seeing results. If both arms improve prediction, preserve that outcome too.

Three paired parents and shared training windows do not establish general seed
confidence. The zero control is an artificial mechanism intervention. Contrasts
do not separate momentum from decay or assign a universal additive causal share.
Parameter displacement and optimizer-moment norms at probes are descriptive.
Any later model-improvement panel needs a separate hypothesis and fresh full
quality confirmation. No batch-reuse optimization arm is introduced.

## Complete resource admission

The nine trajectories require1,494 optimizer advances:900 natural CE,297 zero
and297 pseudo, plus48 evaluation passes of8,192 targets. Charge all zero advances
at the conservative0.4s/update rate, above the qualified0.370253s pseudo mean.
At the measured22.063626s/100,416-target scorer rate, evaluation forecasts86.398s.
Add30s loading,9s restores,54s state summaries/displacements,15s bindings/input
selection, and29.88s logging/boundary checks. Forecast821.878s; freeze a1,200s
whole-worker ceiling with378.122s failure/variability allowance. This is a new
diagnostic budget, not an extension or retry of the completed600s study.

Retained metadata cap16MiB plus16MiB write overlap and10GiB disk reserve. No new
model payload or failed checkpoint serialization is planned. Existing retained
outputs are already deducted from measured free space. Require an additional2GiB
headroom at admission. Use RSS4GiB and sampled MPS-driver6GiB limits, the shared
nonblocking lock, and stop checks on every update and evaluation window. Previous
diagnostic peaks were2.047GB RSS and2.498GB MPS; this test keeps only a pristine
parent capture rather than its larger populated optimizer capture.

Preserve any failed execution with its original source, plan, logs and guard.
No automatic replay is authorized by this protocol. Diagnose a demonstrated
engineering failure separately before another attempt. Reuse the established
loader, optimizer, sampler and complete-state restoration; no completed numerical
qualification is rerun. The exact plan and measured admission must exist before
the first model update.
