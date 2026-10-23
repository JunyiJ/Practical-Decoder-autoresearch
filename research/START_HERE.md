# Start and continue auto-research

This guide runs the local Codex CLI research team: propose an idea, predict its
quality and resource effects, implement it, train paired baseline/candidate
models, evaluate transfer across widths, and save the results and learnings.
The knowledge bank is a folder of JSON files. No database installation or
storage service is required.

All commands below run from the repository root. Setup and preparation do not
launch research agents or training. The execution commands are for when you
explicitly choose to start them; writing this guide does not authorize a run.

| What you want | Where to start |
| --- | --- |
| First run on a Mac mini | [Set up the machine](#1-set-up-the-machine) |
| Inspect the proposed workflow without running it | [Prepare a campaign](#3-prepare-a-new-campaign) |
| Start the agents and training | [Execute](#4-execute-when-ready) |
| Continue a campaign between phases | [Resume the same campaign](#6-resume-the-same-campaign) |
| Start another experiment using earlier learnings | [Reuse the knowledge bank](#7-start-a-new-campaign-with-previous-knowledge) |
| Recover from a failure or interruption | [Recovery limits](#recovery-limits) |
| Move to another machine or back up your work | [Preserve local evidence](#8-back-up-or-move-your-research) |

## 1. Set up the machine

The starter configuration targets a **16 GB Apple Silicon Mac** using PyTorch
MPS. Its actual training speed and memory requirements have not yet been
measured on the target Mac mini. Use a Python installation supported by the
PyTorch version you install, plus Git and an authenticated Codex CLI.

For a fresh checkout:

```bash
mkdir -p ~/Documents/PreTraining
cd ~/Documents/PreTraining
git clone https://github.com/JunyiJ/Practical-Decoder-autoresearch.git
cd Practical-Decoder-autoresearch
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If the repository and environment already exist, enter that directory and run
`source .venv/bin/activate` instead. Use the same environment on later visits.
Do not upgrade dependencies or pull new controller code in the middle of a
campaign you intend to continue.

Install Codex CLI and sign in using the
[official CLI setup instructions](https://learn.chatgpt.com/docs/codex/cli).
Launching `codex` offers sign-in on first use; exit after setup without asking
it to run research. Verify that the executable is available in this terminal:

```bash
codex --version
codex exec --help
```

The team uses `codex exec` with structured output. `runtime.executable` in
[team_mac.yaml](team_mac.yaml) defaults to `codex`; set it to an absolute binary
path before initializing if needed. `runtime.model: null` uses your configured
Codex model. Local JSON storage is free; Codex usage follows your account's
access and usage limits.

Check accelerator availability without training:

```bash
unset PYTORCH_ENABLE_MPS_FALLBACK
python -c 'import torch; print("PyTorch:", torch.__version__); print("MPS:", torch.backends.mps.is_available()); assert torch.backends.mps.is_available(), "This starter requires MPS"'
```

The starter rejects unavailable MPS and enabled CPU fallback. This check only
confirms availability, not that every proposed model operation supports MPS.

## 2. Put the corpus in place and choose limits

The default corpus is `data/raw/shakespeare.txt`, which is not included in Git.
For a new study, download the
[Tiny Shakespeare corpus from karpathy/char-rnn](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare):

```bash
mkdir -p data/raw
if [ ! -e data/raw/shakespeare.txt ]; then
  curl --fail --location \
    https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
    --output data/raw/shakespeare.txt
fi
```

Check that the download succeeded and the file is nonempty. For an existing
campaign, retain the exact original corpus; do not replace or redownload it.
The harness records its hash and uses contiguous train/validation/test splits.

Review these two files **before** initialization:

| File | Settings to review |
| --- | --- |
| [team_mac.yaml](team_mac.yaml) | Objective, baseline Git ref, knowledge-bank path, Codex executable/model, iterations and total budgets |
| [mac_lr_transfer.yaml](mac_lr_transfer.yaml) | Corpus/device, model widths, paired seeds, training steps, batch/context sizes, trial time and disk limits |

The defaults use widths **64 / 128 / 256**, three paired seeds, and **300 steps**
per training trial. One complete iteration has **18 training trials**; the final
planned iteration can add up to 18 held-out checkpoint evaluations. The team
does not run the five-LR screen listed in the standalone study configuration.

The campaign permits at most **3 iterations, 12 agent calls, 3,600 seconds of
agent time, and 7,200 seconds of training-suite time in total**. These are caps,
not a promise that all iterations fit. Per-call and per-trial limits also apply.
For a shorter first campaign, set `max_iterations: 1` before initialization;
it still uses the complete paired-seed/width protocol. Time limits are not
provider billing limits.

Configuration paths such as `base_plan` and `knowledge_bank` are resolved from
the repository root, not from the directory containing a custom YAML file.
Commit reviewed source/configuration changes before starting, and check:

```bash
git status --short
```

Initialization requires a **clean, committed source checkout**. Resolve any
listed changes deliberately; the harness does not commit them for you. Ignored
data, virtual environments, knowledge records, and run outputs can remain.

## 3. Prepare a new campaign

Choose a new output directory for every new campaign:

```bash
python -m src.research.team init --config research/team_mac.yaml --output research_runs/team-001
python -m src.research.team status --campaign research_runs/team-001
python -m src.research.team prepare --campaign research_runs/team-001
```

`init` freezes the baseline and configuration and creates local Git worktrees
and campaign state. It creates the knowledge bank if missing. `prepare` exports
the next role's prompt, schema, and command for inspection. None of these
commands calls a research agent or trains a model.

Stop here if you only want preparation. Do not run `init` again on an existing
campaign directory. Later configuration edits do not change its saved settings.

## 4. Execute when ready

To run idea generation, implementation/local correctness tests, and execution
preflight, then stop at the training gate:

```bash
python -m src.research.team run --campaign research_runs/team-001 --run-agents
```

This **does call Codex agents**, but leaves training disabled. The idea and its
quality/throughput/memory forecasts are recorded before implementation. Review
`iteration-001/prediction.json` and the candidate artifacts under the campaign
directory before enabling training.

To allow the full loop, including training, evaluation, recorded learnings, and
later iterations within the saved limits:

```bash
python -m src.research.team run --campaign research_runs/team-001 --run-agents --allow-training
```

For manual control, use this instead of `run` and inspect status after each call:

```bash
python -m src.research.team advance --campaign research_runs/team-001 --run-agents --allow-training
python -m src.research.team status --campaign research_runs/team-001
```

`advance` executes one phase: idea, coder, executor, training, or evaluator.
**One training phase runs the entire paired suite**, not a single model trial.
Omit `--allow-training` to keep that phase gated. Without `--run-agents`, both
`run` and `advance` only prepare the next handoff, even if `--allow-training`
is present. Training runs serially; avoid competing campaigns on one Mac.

## 5. Read the results and learnings

These commands inspect state and regenerate reports without agents or training:

```bash
python -m src.research.team status --campaign research_runs/team-001
python -m src.research.team report --campaign research_runs/team-001
python -m src.research.team knowledge export --bank research_state/knowledge --output research_state/knowledge.md
```

| Location | What to read |
| --- | --- |
| `research_runs/team-001/trajectory.md` | Exploration history: ideas, predictions, outcomes, performance, costs, and learnings |
| `research_runs/team-001/trajectory.json` / `trajectory.csv` | Machine-readable history for comparison/plotting |
| `research_runs/team-001/trials.json` / `trials.csv` | Trial inventory, including failures and held-out evaluations |
| `research_runs/team-001/iteration-001/prediction.json` | Forecast made before implementation |
| `research_runs/team-001/iteration-001/evidence.json` | Computed quality/resource results and transfer outcome |
| `research_runs/team-001/iteration-001/prediction-check.json` | Observed results versus the forecast |
| `research_runs/team-001/iteration-001/learnings.json` | Evaluator lessons, limitations, evidence links, and next actions |
| `research_runs/team-001/iteration-001/suite/` | Trial requests, logs, metrics, checkpoints, and partial results |
| `research_state/knowledge/entries/` | Persistent JSON records used by future campaigns |
| `research_state/knowledge.md` | Human-readable export of the bank; regenerate when needed |

Iteration files appear as their phases finish. For other iterations, substitute
`iteration-002`, etc. Larger-width improvement is evidence within this small
character-level experiment, not proof of transfer to a production LLM. See
[TEAM.md](TEAM.md) for metric definitions and evidence rules.

## 6. Resume the same campaign

There is **no `resume` subcommand or `--resume` flag**. At a clean phase boundary,
reactivate the same environment, inspect status, and reuse the same campaign:

```bash
source .venv/bin/activate
python -m src.research.team status --campaign research_runs/team-001
python -m src.research.team run --campaign research_runs/team-001 --run-agents --allow-training
```

This continues the saved phase and remaining iterations without repeating
completed phases. Use `advance` instead of `run` to execute only the next phase,
or omit `--allow-training` to keep training gated. Budgets remain cumulative.

Retain the original checkout, source revision, corpus, dependencies, machine,
and artifact paths. Controller source changes block continuation, and training
contracts reject changed inputs. A completed campaign stays complete; extending
the study requires a new campaign and output directory.

### Recovery limits

**Stop between successful `advance` calls for a resumable pause. Ctrl-C is not a
supported pause mechanism.** Interrupting an agent or local test can fail the
phase. An interruption caught inside the training suite can yield incomplete
evidence and let the outer loop proceed to evaluation instead of pausing.

There is currently no CLI command to retry a failed phase, resume a partially
trained team trial, clear an interrupted handoff, or extend an existing budget.
An abrupt termination can leave a running marker or a training reservation that
blocks automatic continuation. Reserved time remains charged conservatively.

For `needs_attention`, an exhausted budget, or an interrupted-run error:

1. Inspect `status`, `campaign.json`, and the affected iteration's role logs,
   `*-failure.json`, or `suite/state.json` and `suite/report.json` when present.
2. Keep the failed/incomplete artifacts and bank records. Do not delete failure
   flags, edit counters, or overwrite the campaign to make it look successful.
3. Diagnose the cause and make any reviewed code/configuration fix. Start a
   **new campaign using the same knowledge bank**, as below.

## 7. Start a new campaign with previous knowledge

The default `knowledge_bank: research_state/knowledge` is shared across
campaigns. Leave that setting unchanged to reuse earlier ideas, successes,
failures, predictions, and learnings automatically. Each agent handoff gets
recent records and a searchable snapshot of the full bank.

After reviewing the next objective/budget and committing any source/config edits:

```bash
python -m src.research.team knowledge search --bank research_state/knowledge --query "transfer"
python -m src.research.team init --config research/team_mac.yaml --output research_runs/team-002
python -m src.research.team prepare --campaign research_runs/team-002
```

When ready, use the execution commands above with `team-002`. For a bank stored
elsewhere, set `knowledge_bank` to its directory in your campaign configuration
**before** initialization. There is no need to import reports already recorded
in that bank.

A new campaign starts from its configured `base_ref` (default: `HEAD`) with
fresh model initialization. It does **not** automatically adopt a prior
candidate's code or pretrained weights. To build on a candidate, review and
commit the intended changes and select that revision as the new baseline.

To add evidence from an older standalone LR study or another team's suite:

```bash
python -m src.research.team knowledge import-report --bank research_state/knowledge --report /absolute/path/to/old-study/report.json
```

The importer accepts a standalone study `report.json`, team suite `report.json`,
or team iteration `evidence.json`. It records the reported conclusion and its
artifact reference; it does not rerun/verify the result, restore jobs, or copy
checkpoints. Keep the source report and associated artifacts. To add a paper or
manual lesson, use the [record example and `knowledge add` command](TEAM.md#knowledge-that-survives-iterations).

Retain held-out audit records when reusing or copying a bank. A new campaign
does not make an already consumed test set fresh. Repeated exploration of the
same corpus can continue on validation data, but the bank prevents reopening
its consumed held-out audit under the same dataset identity.

## 8. Back up or move your research

Git does **not** include the ignored corpus, knowledge bank, run outputs, or
candidate worktrees. A clone/pull alone will not bring them to your Mac mini.
Back up while campaigns are stopped between phases, retaining:

- `research_state/knowledge/`: the complete JSON bank, not just its Markdown export.
- `research_runs/`: campaign state, source snapshots, predictions, trials,
  checkpoints, reports, and learnings.
- `data/raw/`: the exact corpus used by those experiments.
- The repository's Git history/refs and any candidate worktrees you need.
  Worktrees live in **`../.research-worktrees/`**, beside the repository.
- The Python/dependency versions and campaign configuration needed to interpret
  or reproduce the work.

Campaigns contain absolute paths and environment-specific contracts. Copying
their directory to a different path/machine is **not a supported live resume**.
For the separate Mac mini, set up a fresh checkout/environment, copy the complete
knowledge-bank directory to the configured location, retain the old artifacts,
and start a new campaign. Existing artifact references may still point to the
old machine; copying the bank alone does not make those files available locally.
Preserve original records rather than rewriting their contents or evidence IDs.

## Other kinds of resume

The standalone LR sweep has its own [guide](README.md). Reissue its original
command and output directory to skip completed jobs and continue eligible ones:

```bash
python -m src.research.runner run --config research/mac_lr_transfer.yaml --output research_runs/mac-lr-001
```

This command **trains models** and does not start Codex agents. Its plan, source,
data, runtime, executable, and host must match. Abandoned jobs become failed,
failed jobs are not retried, and budgets do not reset.

Individual model continuation uses `training.resume_from` and the exact original
training configuration; `training.max_iters` is the new **total** step target.
See [individual training budgets and checkpoint resume](../README.md#individual-training-budgets-and-checkpoint-resume).
This is separate from continuing a research campaign: team comparisons require
fresh initialization and do not use manual checkpoint continuation as a retry.
