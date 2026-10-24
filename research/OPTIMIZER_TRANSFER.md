# Controlled optimizer transfer study

This study separates optimizer, parameterization and weight-decay semantics. It
uses fixed-depth dense GELU MHA at widths 64, 128 and 256, with fixed head dimension
32. It does not establish transfer to deeper or production-scale language models.

## Implementations

The default remains ordinary PyTorch AdamW, with its historical initialization,
parameter grouping and checkpoint format. Optional training settings are
`optimizer: adamw|muon` and `decay_mode: conventional|independent`.
`model.parameterization: sp|mup` and `model.base_width: 64` define width scaling.
All parameters with gradients are decayed, including vectors, just as in the
historical reference. Changing which parameters decay is a separate experiment.

For width multiplier m = width / base_width:

| Component | Standard parameterization | muP |
| --- | --- | --- |
| Hidden matrix initialization | PyTorch fan-in initialization | Same fan-in rule |
| Hidden linear bias initialization | PyTorch fan-in bound | Multiply by sqrt(m), preserving base-width bound |
| Embedding initialization | PyTorch default | Same |
| Output weight initialization | PyTorch fan-in initialization | Multiply by sqrt(m) |
| Output logits | Linear output | Linear output divided by m |
| Hidden Adam learning rate | Base LR | Base LR / m |
| Embedding, output and vector Adam LR | Base LR | Base LR (output forward multiplier supplies its effective scaling) |
| Hidden Muon nominal LR | Base LR | Base LR / sqrt(m) |

Muon is `torch.optim.Muon` with momentum .95, Nesterov enabled, five Newton–Schulz
steps and `adjust_lr_fn='match_rms_adamw'`. PyTorch multiplies the nominal update
LR by `0.2 * sqrt(max(rows, columns))`. Dividing the nominal LR by sqrt(m) cancels
that width factor for our fixed-aspect-ratio hidden matrices. This gives a
width-constant spectral update scale, with per-layer constants calibrated to the
RMS-matched base model. It is not Adam's inverse-width rule. The hybrid uses
AdamW (default betas .9/.999, epsilon 1e-8) on embeddings, the output layer and
vectors. It is not all-parameter Muon. No additional packages are required.

Head dimension stays 32 while the head count increases. The attention scale is
therefore a fixed constant across widths; this protocol retains it rather than
introducing an unrelated base-model attention-temperature change. Depth is fixed.
Only dense GELU MHA with MLP hidden width 4*model width is supported for these
optimizer scaling options. Wider architecture support requires separate checks.

Conventional decay multiplies each updated parameter by `1 - group_lr * wd`.
For Muon, PyTorch applies decay with the **nominal** group LR, before its shape
adjustment. Independent decay multiplies by `1 - wd` once before the gradient
update; inner optimizers receive zero decay. Thus `wd` changes units between
modes. The independent coefficient .00006 matches historical baseline
`.0006 * .1`; copying `.1` to independent mode would shrink 10% every step.
This implementation handles zero LR without dividing by it. No scheduler is
used; schedules would need explicit independent-decay policy.

## Frozen round

Seven families: Adam SP/conventional, Adam SP/independent, Adam muP/conventional,
Adam muP/independent, Muon SP/independent, Muon muP/independent, and the prior
Adam SP zero-decay LR .002 reference. The six experimental families screen base
LRs .001, .002 and .003 at width 64 with seeds 11/22/33. Every run uses 3,000
updates, batch 8, context 128, fixed 30-batch validation, and no held-out test.

After all 57 local runs, select the lowest mean final validation CE per family
(ties choose lower LR), write a read-only selection with hashes, then evaluate
only each family's selected recipe at widths 128 and 256. This adds 42 trials,
for 99 planned trials total. Larger results cannot revise the frozen choices.
A boundary winner means the LR grid may be inadequate; do not call it optimal.
The decay coefficient is a controlled anchor, not an optimizer-specific optimum.

Limits: 105 total dispatches, 600 seconds per process, 4 hours for the round,
8 GiB output budget, stop on the first failed trial. Trials are serial. The
study uses the team suite's trusted causal CE scorer, strict MPS checks,
checkpoint hashes, fixed-token result validation and process time/disk limits.
Each trial records its request, checkpoint, curves and process log. Failure and
running markers are preserved; no automatic retry or partial-score ranking.
Touch the output directory's `STOP` file for a pause after the current trial.

Freeze a plan from a clean checkout using `make_plan` in
`src.research.optimizer_study`; record its source and corpus hashes before launch.
Run with the same project environment:

```bash
unset PYTORCH_ENABLE_MPS_FALLBACK
.venv/bin/python -m src.research.optimizer_study \
  --plan research_state/optimizer-study-plan.json \
  --output research_runs/optimizer-transfer-001 --allow-training
```

Do not run this alongside an MLE training campaign. Finish the current MLE
iteration and pause it first. After reviewing results, append scoped findings
to the shared knowledge bank and initialize a new continuous campaign from the
reviewed source revision. Keep old evidence and the human pause intact. The
research loop's continuous target and uncapped aggregate protocol are unchanged.

## Validation and interpretation

CPU tests verify base-width equivalence, parameter partitioning, width scaling,
exact independent shrinkage, checkpoint continuation and contract rejection.
Synthetic MPS checks verify supported device execution and finite early updates
at all three widths. These are correctness diagnostics, not quality evidence
or proof that finite-width dynamics are perfectly invariant.

Report paired validation loss, training time and process RSS. A support label
requires mean delta < -.005 and every seed negative; it is a heuristic, not a
statistical significance test. Compare against the fresh zero-decay reference.
Present the complete matrix even if no family wins. Never turn adaptive reused
validation seeds into an independent replication claim. Fixed tokens do not
mean fixed compute, and 3,000 steps do not establish convergence.

## Primary references

- [Tensor Programs V and reference implementation](https://github.com/microsoft/mup)
- [Weight Decay may matter more than muP for Learning Rate Transfer in Practice](https://arxiv.org/html/2510.19093v2)
- [Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scales](https://arxiv.org/html/2512.05620v1)
- [PyTorch Muon implementation and API](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html)

The matrix-optimizer paper's inverse-width decay result concerns compute-optimal
scaling. This first fixed-token study does not import that rule automatically.

## Completed rounds and continued research

The initial 99-trial round and 39-trial bounded Adam LR follow-up are complete.
The learning guide records their measured results and limits. The strongest
observed transferable recipe remains Muon hybrid + muP, LR .002 and independent
decay .00006. Its incremental muP benefit over SP Muon remains mixed by seed.

`team_mac_muon_continuous.yaml` starts a new continuous evolutionary campaign
using that recipe as its fixed baseline. The optional team `baseline_options`
mapping explicitly declares `optimizer`, `decay_mode` and `parameterization`.
Only supported enum values are accepted; muP base width is fixed to 64. LR and
decay coefficient come from `mac_muon_transfer_long.yaml`. These options are
applied once before freezing the campaign baseline and do not change the narrow
standalone LR-sweep protocol. Existing profiles without this mapping retain
their old baseline behavior. The corpus, token budget, target schedule, seeds,
finite per-call/suite limits, 48-hour informational target and uncapped aggregate
protocol remain controlled. Continuous exploration keeps held-out test closed.

The first candidate is a zero-decay component ablation. Later candidates use
one or two configuration mutations and scheduled crossover, preserving the
supported fixed-depth dense GELU MHA scope. All candidate overrides are relative
to the new Muon baseline. Historical optimizer rankings remain adaptive
development, not independent replication or proof of a universal optimum.
