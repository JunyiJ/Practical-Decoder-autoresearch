# Readout generalization and decay: applicability review, no new experiment

This review ran while the frozen residual panel continued. It changes no active
source, comparison, seed, checkpoint, endpoint or acceptance criterion. The first
complete fresh bundle is unfavorable; all nine prescribed trajectories and the
independent panel review still execute. This review is existing-data/literature
analysis, not numerical exploration or admission of a rescue sweep.

## Local observations and contrary evidence

The first residual bundle's bound analysis is
`tokenizer-evidence/byte-residual-readout-001/quality001/analysis-1-bundles001/result.json`.
Both branches fit late paired training batches better than byte while their
online and H200 validation scores are worse. Nonlinear H200 minus byte is
+.057385; minus linear is-.077839. H200 improves all three, so both endpoints
support the same unfavorable ordering. This is one seed and does not identify
whether regularization, representation or optimizer dynamics caused the gap.
The closed plain-transform panel also supplies contrary evidence to adding
output transformations; see `BYTE_CONTEXT_TRANSFORM_RESULTS.md`.

Previous completed studies must not be treated as untested generic remedies:

| Evidence | Actual scope and result | What it does not establish |
|---|---|---|
| `tokenizer-evidence/decay-preparation-001/quality001-closure.json` and independent final review | Native-token D256 MLA, fixed LR.00125: removing independent decay.00006 worsened all three fresh seeds, mean+.009165192885 | A globally optimal decay, head-specific effect, or gain from stronger decay on the byte model |
| `DECAY_QUALITY_RESULTS.md` | Earlier character-model global strong decay.00018 improved width256 mean by.010659; width64 mean gain.001221 had mixed seeds; full adoption rule failed | Transfer of that coefficient or selection to this byte-composed input model |
| `TOKENIZER_EMBEDDING_INPUT_DROPOUT_RESULTS.md` | Native-token input dropout.1 worsened all three fresh seeds, mean+.014701832892 | A general benefit from regularizing input features, or a permissible rate/site rescue |
| `OUTPUT_BIAS_RESULTS.md` | Character-model learned-zero and train-unigram biases worsened width256 means versus tuned no-bias by.003492/.008795; closed not adopted | Benefit from adding output biases to address the current validation gap |

The input-capacity/byte-composition reviews also found no evidence for uniformly
compressing the old input table to rank128. That input-table spectrum is not an
output-table spectrum. No output low-rank quality claim follows either way, and
no arbitrary rank or frequency cutoff is selected here. Output compression needs
its own discriminating local rationale, not a relabeled input experiment.

## Primary literature actually read

[Loshchilov and Hutter, ICLR2019, arXiv1711.05101v3](https://arxiv.org/pdf/1711.05101v3),
§2/Eq1 and Algorithm2 distinguish decay from an adaptive-gradient L2 penalty.
Algorithm2 includes a schedule multiplier on the decay term. Their image
classification evidence is not evidence for an output-only coefficient in this
model. It motivates recording exact update equations and units.

[D'Angelo et al., NeurIPS2024, arXiv2310.04415v2](https://arxiv.org/pdf/2310.04415v2),
§3/Figs6–7 and AppendixD/Figs19–23 study124M/774M GPT-2 models with near-zero
training/validation gaps in a mostly one-pass regime. Effective learning rate
and precision stability explain their reported effects; their LLM L2 ablation
also reproduces the training-loss effect. Fig20 shows sensitivity to including
normalization parameters at stronger decay. This challenges a universal
capacity-control interpretation and simple coefficient transfer. Their setting
is not our repeated sampling, small corpus, FP32 and mixed Muon/AdamW setup.
Neither paper supplies a local head-only versus whole-model comparison.

## Exact local convention

`byte_context_transform_v1.Optimizer.step`, inherited by the residual optimizer,
multiplies every parameter by1-.00006 independently of LR. Child optimizers have
weight_decay0. The adopted byte optimizer has the same declared independent-decay
recipe; this is intentional, and no source correction is proposed.

The head's LR goes from.0025 to.00025. In a convention whose shrink factor is
`1-lr_t*lambda_t`, matching our fixed shrink requires lambda_t=.00006/lr_t,
which ranges from.024 to.24. This is algebraic unit conversion, not an assertion
of equivalent full optimizer dynamics. The decay-only factor across1200 updates
is approximately.93053; actual weights also receive learned updates, so this is
not a measured7% norm reduction. Each run samples1,228,800 targets from267,688
training tokens, approximately4.59 target-budget equivalents with replacement,
not a measurement of unique data coverage. The input-bound arithmetic is in
`tokenizer-evidence/byte-readout-regularization-review-001/decay-convention.json`.

## Research decision

Do not launch a stronger-decay, dropout, bias or rank sweep from the first-seed
training/validation gap. Complete the existing panel and review its saved curves.
A remaining question is whether the byte incumbent's independent output-table
decay and the body/composer decay have different quality effects. Answering it
would require a genuinely new group-specific hypothesis and discriminating
controls, with exact LR/decay conventions, fixed criteria, fresh roles and a
complete screening/confirmation cost. No coefficient, panel or numerical budget
is admitted here. A positive result would support that recipe; a negative result
would reject it without attributing causality from norm or training-loss changes.

More output maps and an averaging-endpoint change currently have weaker local
support than resolving this recipe question. That ranking is provisional until
the complete residual panel closes. All historical negative results, source
snapshots and restrictions remain intact; no old commands were replayed. This
completed review supplies applicability limits, not an additional model gain.
