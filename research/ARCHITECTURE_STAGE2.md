# Stage 2: architecture changes against the stage-1 control

The human concluded stage 1 on October 24, 2026. Campaign006 is intentionally
paused after its last training suite and must not resume. Its final numeric
results were reviewed separately without clearing the stop marker or rerunning
training. Historical campaigns, budgets, failed markers and evidence remain intact.

## Frozen control and stage-1 conclusion

The strongest observed arm is the **Muon/AdamW hybrid with muP**, base LR .002,
independent per-step decay .00006, global clipping 1, zero dropout, two-layer
RMSNorm dense GELU MHA, RoPE, final LayerNorm, base width64 and head dimension32.
The hidden FFN width is four times model width. The optimizer uses Muon on hidden
matrices and AdamW on embeddings, readout and vectors, retaining the exact
stage-1 grouping, initialization, bias, decay and width rules.

The controlled optimizer study measured mean validation CE
**1.646054 / 1.559346 / 1.523978** at widths64/128/256, all paired seeds better
than its AdamW reference. Incremental muP benefit over SP Muon was mixed by seed.
The bounded Adam follow-up did not reverse the historical Muon ranking.

The two optimizer studies contain 138 completed runs. Subsequent campaigns005
and006 contain 42 complete 18-trial comparisons (756 runs). None displaced the
fixed Muon control under the frozen local rule. These counts cover the optimizer
studies and subsequent Muon exploration, not all earlier project experiments.
The last decay .00005 comparison changed mean CE by +.000185 / +.001021 / +.000098;
it also retained the baseline. All its 18 checkpoints were verified. The strongest
observed arm is an adaptive development choice, not a universal optimum.

Machine-readable closure and exact control: `research_state/stage1-conclusion.json`.
Full shared bank snapshot: `research_state/stage2-knowledge-snapshot.md`.
Original measurements and source contracts remain authoritative; new source does
not rewrite them. Held-out test remains closed.

## Round 2A: isolate the activation before changing FFN structure

Compare GELU and SiLU in the existing two-matrix dense FFN. This changes only the
nonlinearity at a fixed LR: identical parameter names, counts, dimensions, initial
parameter tensors for a paired seed, dropout and optimizer groups. SiLU uses
PyTorch `nn.SiLU`; GELU retains the existing `nn.GELU`. Gated SwiGLU is NOT enabled
under the width contract by this change. Attention, depth and head dimension stay
fixed. Implementation tests establish supported execution, not improved quality
or invariant optimal learning rates.

Three families:

| Family | Activation | Base LR options | Role |
| --- | --- | --- | --- |
| control | GELU | .002 only | Exact stage-1 recipe, trained fresh |
| gelu_tuned | GELU | .0015, .002, .0025 | Equal-budget tuning comparator |
| silu | SiLU | .0015, .002, .0025 | Architecture candidate |

Use fresh **development** seeds44/55/66, fixed validation seed2026, the same
Shakespeare splits and tokenizer, batch8, context128, 3000 updates and 30-batch
validation every300 updates. Each trial sees 3,072,000 training character targets.
These are new training seeds, not an independent dataset or a held-out confirmation.

Run all 21 width64 trials first, interleaving families and reversing order on the
middle seed. Select each tuned family's LR using lowest mean FINAL validation CE;
ties choose lower LR. No intermediate-checkpoint selection or grid extension.
A boundary choice is reported as unresolved, not an optimum. The separate fixed
control is intentionally trained fresh even when tuned GELU selects .002; these
are contemporaneous repeated controls, not exact-reproducibility evidence.

Before any larger-width results, freeze the selected LRs and architecture winner
in read-only `selection.json` with plan and local-evidence hashes. SiLU wins only
if its mean paired CE delta is below -.005 and every seed improves against BOTH
the fixed control and tuned GELU. Otherwise retain the control. This heuristic
is not a significance test. Then run the three families at their frozen LRs at
widths128/256 with the same three seeds (18 trials). All-family transfer results
are diagnostics; they cannot revise local selection. Report every result,
including failures and reversals. Total: **39 trials**, no held-out evaluations.

This round has equal parameters, shapes, tuning allowance for the activation
families, and training-token budgets. It is not equal-time or equal measured
compute: activation kernels can differ. Report CE, training time/throughput and
process RSS; RSS is not peak accelerator or total unified-memory demand.
Quality effects are unquantified in advance. Throughput/RSS are expected to be
close because matrix shapes are unchanged, but no numerical range is asserted.
A possible mechanism is different activation/gradient behavior; this round does
not instrument or causally identify that mechanism.

## Execution and limits

`src.research.architecture_study` reuses the serial optimizer-study worker engine,
trusted CE scorer, source/corpus/runtime checks, checkpoint verification, fixed-token
validation and durable failure markers. The existing optimizer study defaults are
retained. Freeze a plan only from clean committed source using `make_plan`, then:

```sh
unset PYTORCH_ENABLE_MPS_FALLBACK
.venv/bin/python -m src.research.architecture_study \
  --plan research_state/architecture-stage2a-plan.json \
  --output research_runs/architecture-stage2a-001 --allow-training
```

Limits: exactly39 dispatches, 600 seconds per trial, four hours for the round,
4 GiB output, stop on first failed trial; serial training only. `STOP` pauses at
a trial boundary. No automatic retry of failed/running records. Source changes
require a new run with preserved prior evidence. This finite round is within the
ongoing 48-hour-target research effort; it does not add an aggregate exploration
cap or authorize an indefinite LR grid. Autonomous maintenance expires at the
previously authorized time, October25 1:32 PM Pacific; healthy runs are not killed
solely because that window ends.

### Completed stage 2A: activation and learning-rate transfer

All **39 trials** completed and were reviewed against checkpoint hashes, worker results,
frozen selection, source/corpus/runtime contracts and recomputed report aggregates.
Mean final validation CE at 3000 updates (three fresh development seeds44/55/66):

| Arm | Frozen LR | Width64 | Width128 | Width256 |
| --- | --- | --- | --- | --- |
| Fixed GELU control | .002 | 1.659749 | 1.573942 | **1.520457** |
| Equally tuned GELU | .0025 | **1.647783** | **1.571541** | 1.525777 |
| Tuned SiLU | .0025 | 1.659710 | 1.576024 | 1.521624 |

**Keep the exact stage-1 control.** SiLU failed the predeclared width64 architecture
rule: its mean delta was −.000039 against fixed GELU (one seed worse), and +.011928
against tuned GELU (all three worse). At widths128/256 its deltas against fixed
control were +.002082 / +.001168, both mixed by seed. SiLU was better than tuned
GELU on every width256 seed, but the mean −.004153 missed the .005 threshold;
transfer diagnostics cannot change the frozen architecture decision.

The GELU LR change is a useful transfer counterexample: .0025 improved over .002
by .011966 at width64 and .002401 at width128, then worsened by .005320 at width256.
All three paired seeds followed each direction. Width scaling does not guarantee
that a locally preferred LR preserves its advantage at larger widths. This does
not invalidate muP in general or identify the optimal LR at any width.

Mean training seconds for control / tuned GELU / SiLU were 32.23 / 32.48 / 32.35
at width64, 51.26 / 51.00 / 51.84 at width128, and 66.40 / 66.41 / 66.36 at width256.
Process peak RSS means ranged from 472.1 to 475.6 MiB. These small observed time
and RSS differences do not establish a reliable speed advantage; RSS is not peak
accelerator memory. Equal token budgets and shapes are not equal measured compute.

Both tuned families selected the upper grid boundary; no grid extension is planned.
The validation split remains adaptively reused, despite fresh training seeds; test
was never opened. Historical control scores use different seeds and must not be
pooled as independent replication. Correctness tests and short smoke runs remain
separate from these measured quality results.

Evidence: `research_runs/architecture-stage2a-001/{plan,selection,report,review}.json`.
Next is a separately preregistered gated-FFN round, after parameter-budget,
bias/initialization, optimizer-aspect-ratio and execution checks. Stage1 stays stopped.

## Subsequent rounds

After reviewing 2A, test gated FFNs with explicit near-matched parameter budgets
and bias/initialization controls, then attention variants/head dimensions,
depth-width tradeoffs, and MoE. Each needs its own reviewed scaling contract,
resource limits and preregistration. In particular, three-projection SwiGLU needs
a smaller hidden width than a two-projection dense FFN for comparable matrix
parameters; rounding, bias counts and changed matrix aspect ratios must be
reported. MoE needs both total/active parameter counts, routing/load-balance
metrics and measured runtime. Do not enable those families by bypassing the current
guards or claim width-only scaling establishes depth/expert scaling.
