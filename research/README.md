# Bounded MLE experiment harness

The coding agent supplies hypotheses and interprets results. This harness
executes one declared experiment, preserves evidence, and prevents incomplete
runs or larger-width feedback from changing the small-width selection.
It runs locally without an LLM API or an experiment-tracking service.

**Prepared, not launched:** the real-corpus Mac study has not been run. The user
requested that execution wait. Commands below are instructions for a later run.

## First study

Question: does an LR advantage discovered at width 64 persist at widths 128 and
256? [mac_lr_transfer.yaml](mac_lr_transfer.yaml) specifies the complete study:

| Component | Predeclared setting |
| --- | --- |
| Model family | Dense MHA, GELU, RoPE, RMSNorm, 2 layers, dropout 0 |
| Widths | 64 → 128 → 256; head dimension stays 32, head counts 2 → 4 → 8 |
| Optimizer | AdamW, constant LR, PyTorch default betas/epsilon, decay 0.1 |
| Parameterization | Existing standard PyTorch initialization; μP is not implemented |
| LR candidates | 0.0001, 0.0003, **0.0006 baseline**, 0.001, 0.003 |
| Training per trial | 300 updates × batch 8 × context 128 = 307,200 character tokens |
| Validation | 30 batches, fixed sampling seed 2026 across all models/seeds |
| Data | Contiguous 80% train / 10% validation / 10% test; windows never cross splits |
| Confirmation seeds | 11, 22, 33; screening uses seed 0 |
| Primary metric | Final-budget cross-entropy, nats per character; lower is better |
| Limits | 41 jobs, 2 active hours total, 10 minutes per job, 2 GiB output, 4 failures / 2 consecutive |

The default limits are starting constraints, not a measured Mac performance
promise. The models are small and jobs are sequential, but MPS capacity and
throughput on the target Mac mini still need measurement. The process deadline
includes import, training, validation and checkpoint I/O. A capped run is
excluded from ranking. Equal character tokens do not imply equal FLOPs or time.

The tokenizer is the repository's character vocabulary, constructed once from
the corpus's set of characters. All arms use the same mapping. The test split's
characters therefore contribute to vocabulary construction, but its sequences
and loss are not used for training or selection. This is a small character-LM
study, not a subword-token LLM benchmark.

## Scheduling and selection

1. **Screen:** train all five learning rates at width 64, seed 0. The first job
   is baseline. Select the best completed nonbaseline LR as challenger.
2. **Confirm:** baseline and challenger each train from scratch on seeds 11,
   22, 33. For each seed compute challenger CE minus baseline CE. Promote only
   if every delta is negative and their mean is below -0.005 nats/character.
   Otherwise lock baseline. All six jobs must finish successfully.
3. **Transfer:** keep both baseline and challenger unchanged; train each at
   widths 128 and 256 on the same three seeds. No LR tuning at larger widths.
   The small-width winner cannot change. Even if baseline was retained, the
   challenger is measured to expose width-dependent reversals.
4. **Audit:** after all transfer trials complete, evaluate the **final**
   checkpoints of the locked winner and baseline on test, across widths and
   seeds. If baseline won, it is evaluated once per width/seed. These results
   do not feed back into selection.

This is 23 training jobs and 9–18 test-only jobs. The report contains paired
seed differences, means, sample standard deviations, and failures. Three seeds
give a descriptive check, not a confidence guarantee. Testing two candidates
at a larger width does not estimate regret versus an independently tuned
larger-model optimum. Success supports only this finite-width family, corpus,
budget and optimizer; it does not establish production-scale transfer.

## Commands for later execution

Run from the repository root with the project's Python environment active.
First put a local UTF-8 corpus at `data/raw/shakespeare.txt`, or copy the plan and
set `data.path` to your corpus. Relative corpus paths resolve from the repository
root, regardless of the plan's location. The harness records the corpus SHA-256.
It does not download data automatically.

Inspect the declared matrix without training:

```bash
python -m src.research.runner plan --config research/mac_lr_transfer.yaml
```

Once execution is requested, a synthetic CPU smoke can check the entire path:

```bash
python -m src.research.runner smoke --output research_runs/smoke-001
```

This uses width 8/16, two training steps per trial and synthetic text. Its report
is labeled `smoke_only`; its scores have no scientific interpretation.

Launch the real study on the target Mac only when ready:

```bash
python -m src.research.runner run \
  --config research/mac_lr_transfer.yaml \
  --output research_runs/mac-lr-001
```

Reissuing exactly that command resumes the same session. Completed jobs are
not repeated. The frozen plan, source files, dataset, runtime versions and host
must match. Completed studies return success; incomplete/budget-stopped studies
return exit code 2. Study failures are evidence, not grounds to silently change
the plan and continue. To change anything, prepare a new plan/output directory.

Regenerate reports without training (rejected while the study lock is held):

```bash
python -m src.research.runner report --output research_runs/mac-lr-001
```

## Artifacts and recovery

Each ignored `research_runs/<study>/` directory contains:

| Artifact | Purpose |
| --- | --- |
| `plan.json`, `manifest.json` | Frozen protocol, Git revision/status, source hashes, data hash, runtime |
| `state.json` | Atomic trial ledger, frozen selection, consumed active time and stop reason |
| `report.json`, `report.md` | Machine-readable evidence and human-readable paired comparisons |
| `trials/<id>/request.json` | Exact resolved training/evaluation configuration |
| `trials/<id>/worker.log` | Worker output and failure traceback |
| `trials/<id>/result.json` | Accepted worker output, separate from the controller's status |
| `trials/<id>/metrics.jsonl` | Validation trajectory for training jobs |
| `trials/<id>/checkpoints/final_*.pt` | One final checkpoint per training job, with optimizer/RNG state |

Each job runs in a separate process. The controller kills its process group on
timeout/cancellation, counts all attempted jobs, and checks the output-directory
disk limit roughly once per second. Disk can overshoot during a write; this is
not an OS quota or a RAM cap. Timeout termination and bookkeeping can also add
a small amount beyond the nominal wall-time deadline. GPU memory metrics retain
their true meaning: CUDA peak allocation, MPS current allocation, not MPS peak.

A filesystem lock prevents two controllers or report writers from racing.
Workers inherit the lock so an orphan still excludes a second controller.
Source/data/runtime fingerprints are checked before and after each job.
These checks detect persistent edits; they are not a security sandbox against
an agent deliberately modifying and restoring files during a trial.

The ledger is saved before launch with a reservation for the full permitted job
time. Normal completion refunds unused time. A controller crash conservatively
charges that reservation, marks the abandoned job failed on resume, and does not
retry it automatically. Interrupted/failed confirmation is inconclusive rather
than a smaller seed sample. Idle time between invocations does not consume the
active session budget. There is no automatic expansion of budgets, retry loop,
checkpoint deletion, Git commit, or remote push.

## Next protocol after this baseline

Implement and check μP in a separate code change before adding it as an arm:
width-aware initialization, readout and optimizer rules belong together. A
coordinate/update-scale check should precede LR response-curve comparisons.
Keep depth, context, data exposure and batch fixed initially. Muon, MoE, depth
scaling and different token budgets need their own controlled studies. Extend
the harness only after its first study and real-device performance are reviewed.
