# autoresearch

This repository supports an autonomous config-research loop inspired by `karpathy/autoresearch`, but scoped to this codebase and to config-only changes at first.

## Setup

To start a new research session, work with the user to:

1. Agree on a short run tag, usually based on today's date and machine, for example `mar24-macmini`.
2. Create a base research branch `autoresearch/<tag>` from the current branch tip.
3. Read the in-scope files for context:
   - `README.md`
   - `program.md`
   - `src/config/mac_tinyshakespeare.yaml`
   - `src/train/train.py`
   - `src/utils/eval.py`
4. Verify the local environment can train:
   - confirm dependencies are installed.
   - if dependencies are missing, create a new conda environment named `pd` with `conda create -n pd python=3.10 -y`
   - initialize conda in the shell if needed, then run `conda activate pd`
   - install dependencies into that env with `pip install -r requirements.txt`
   - confirm `data/raw/shakespeare.txt` exists
5. Initialize these untracked artifacts if they do not exist yet:
   - `results.tsv`
   - `research_summary_<tag>.md`
   - `research_runs/<tag>/`
6. Confirm setup and begin.

`results.tsv` is tab-separated and must have exactly this header:

```tsv
branch	commit	best_val_loss	final_val_loss	device	peak_rss_mb	peak_accel_mem_mb	peak_cuda_util_pct	status	description
```

Do not commit `results.tsv` or the generated research reports.

## Scope

Initial scope is config-only research.

### You CAN do

- Modify only `src/config/mac_tinyshakespeare.yaml`
- Create new git branches for each experiment
- Push experiment branches to the configured remote if available
- Run training experiments
- Read generated metrics, logs, checkpoints, and reports
- Maintain untracked experiment reports and a summary page

### You CANNOT do

- Modify Python source files for experiments during this phase
- Change evaluation logic in `src/utils/eval.py`
- Add dependencies
- Change dataset or tokenizer pipeline

If you conclude that config-only progress is saturated, summarize why and stop to ask the human before expanding to code changes.

## Objective

The primary metric is validation loss. Lower is better.

Use these criteria:

- Primary: `best_val_loss` from the training summary
- Secondary: memory and runtime efficiency
- Diagnostic only: training loss trajectory
- Additional diagnostic: MoE expert usage snapshots when `mlp_type: moe`

On CUDA systems, include CUDA utilization and memory usage in the evaluation when available. VRAM is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

## Platform strategy

Research should happen in two stages:

1. Mac or non-CUDA stage: short screening runs only. The training script runs for a fixed time budget of 20 minutes (wall clock training time, excluding startup/compilation).
2. CUDA stage: promote promising configs into longer confirmation runs. The training script runs for a fixed time budget of 10 minutes (wall clock training time, excluding startup/compilation). 

If the current machine is a Mac mini or otherwise non-CUDA, keep runs short and treat them as screening only. Do not make strong final claims from these results.

## Baseline

The baseline is the committed `src/config/mac_tinyshakespeare.yaml` as-is.

The first run in every new session must be the untouched baseline config.

## How to run experiments

Each experiment gets its own branch:

- Branch naming: `autoresearch/<tag>-NNN-<slug>`
- Start each branch from the current best kept commit, not necessarily from the original baseline
- Push the branch if remote access is configured

For local screening runs, use short overrides at launch time so the committed config change stays focused on the research idea:

```bash
python -m src.train.train \
  training.max_iters=200 \
  +training.eval_every=50 \
  +training.eval_iters=25 \
  +training.metrics_file=research_runs/<tag>/<run_id>/metrics.jsonl \
  +training.summary_file=research_runs/<tag>/<run_id>/training_summary.json \
  hydra.run.dir=research_runs/<tag>/<run_id>/hydra
```

Override `training.device` if needed:

- `training.device=mps` on Mac screening
- `training.device=cuda` or `training.device=cuda:0` on GPU

For promoted CUDA confirmation runs, increase `training.max_iters` and `training.eval_iters`, but keep the experiment comparable within that stage.

## Required report for every run

For every experiment, create `research_runs/<tag>/<run_id>/report.md` with:

1. Branch, commit, parent branch, device
2. Exact config changes vs the baseline config
3. Exact config changes vs the parent kept config
4. Loss trajectory:
   - train loss snapshots
   - val loss snapshots
5. Resource usage:
   - RSS memory
   - accelerator memory
   - CUDA utilization and memory if on CUDA
6. MoE diagnostics when applicable:
   - aggregate expert usage
   - per-layer expert usage
7. Conclusion:
   - `keep`, `discard`, or `crash`
   - one-paragraph reasoning

Also maintain `research_summary_<tag>.md` as a comparison page across all runs.

The summary page should have one row per run with:

- run id
- branch
- parent
- status
- short description of the config change
- best val loss
- final val loss
- device
- peak memory
- report path

## Experiment loop

After setup, operate autonomously until interrupted.

Loop FOREVER:

1. Inspect current results and identify the best kept configuration so far.
2. Choose the next experiment.
3. Create a fresh experiment branch from the chosen parent.
4. Edit only `src/config/mac_tinyshakespeare.yaml`.
5. Commit the config change.
6. Push the branch if configured.
7. Run training with untracked output paths under `research_runs/<tag>/<run_id>/`.
8. Read `training_summary.json`, `metrics.jsonl`, and the Hydra `train.log`.
9. Write the per-run report.
10. Append the result to `results.tsv`.
11. Update `research_summary_<tag>.md`.
12. Decide whether to keep building from this branch or revert to a stronger parent for the next branch.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

Don't stop and I'll let you know when to stop.

## Search policy

Do not get stuck doing only tiny local tweaks.

Use roughly this mix:

- 70% exploitation: refine around the best recent configs
- 30% exploration: try a broader jump

Exploration ideas within config-only scope:

- attention family: `mha`, `gqa`, `mla`
- `n_kv_heads`
- `mlp_type`: `dense` vs `moe`
- dropout
- learning rate
- weight decay
- MoE knobs:
  - `num_experts`
  - `shared_num_experts`
  - `top_k`
  - `aux_loss_weight`
  - `ffn`

When a direction looks promising, test follow-ups around it. When a direction stalls, jump out and try a qualitatively different combination.

## Keep or discard

Prefer simplicity when results are close.

Keep a change if:

- it clearly improves validation loss
- or it holds validation loss roughly flat while reducing memory or complexity

Discard a change if:

- validation loss is worse without a strong efficiency gain
- it crashes
- it only improves training loss but not validation loss

## Failure handling

If a run crashes:

1. Read the tail of the log
2. Decide whether it is a trivial fix or a bad idea
3. If trivial, fix the config and rerun on a fresh branch
4. If not trivial, log it as `crash` and move on

Do not spend many retries on a single dead end.

## Human interaction

Once the run starts, do not keep asking the human whether to continue after every experiment.

Only stop and ask if:

- dependencies or data are missing
- git remote access is blocked and pushing is required
- config-only research is saturated and code changes are the next step
- the user explicitly interrupts the loop
