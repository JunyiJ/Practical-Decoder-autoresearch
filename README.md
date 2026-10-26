# Practical-Decoder
A repo to implement the decoder-only GPT-style model and play around with different attention mechanism and MoE structures


## To train the model
`python -m src.train.train`

Select sigmoid-gated MLA with `model.attn_type=gated_mla`. Set
`+model.enable_rope=true` to enable decoupled RoPE or
`+model.enable_rope=false` for no explicit positional encoding. This boolean
overrides `model.rope` for gated MLA throughout model construction and attention;
the disabled mode creates neither rotary frequency tables nor learned position
embeddings. Omit `+` if `enable_rope` is already present in the configuration.
Without the override, the legacy behavior is preserved: `model.rope=1` selects
RoPE, while `model.rope=0` (or an absent field) selects learned positions.

The sigmoid gate acts on each attention output channel before the output
projection. Both modes retain the same projections, attention scale, and cache
layout: compressed KV latents plus shared keys. With RoPE disabled, the shared
keys and corresponding query features are unrotated content features. This
option does not introduce Kimi K3's KDA hybrid architecture.

Select `model.attn_type=kda` for a sequential Kimi Delta Attention reference
implementation using the [Kimi K3 recurrence and gates](https://arxiv.org/html/2607.24653v2#S2.SS1.SSS1).
It applies causal depthwise convolutions, SiLU, normalized queries/keys,
sigmoid write strengths, channel-wise bounded decay, and head-wise RMSNorm
before the full-rank sigmoid output gate. KDA supplies positional mixing through
convolution and recurrence, so this model type uses neither RoPE nor learned
position embeddings, regardless of the legacy `rope` setting. Only causal
attention is supported. `dropout` applies after the output projection;
`attn_dropout` does not apply to this recurrent operator.

Cached KDA generation stores a fixed-size recurrent matrix and the previous
three projected inputs for each Q/K/V convolution. Use eval mode with gradients
disabled, append contiguous chunks, and reset caches before a new sequence.
The model still enforces `block_size` as a generation limit. Recurrence and
normalization use FP32 under mixed precision. This implementation retains
PyTorch's default linear/convolution initialization (except the zero-initialized
per-head decay scale); it does not reproduce the paper's full initialization or
training recipe, optimized kernels, or a hybrid KDA/MLA layer schedule.

## To generate with the trained checkpoint

Replace the checkpoint argument with an actual saved `final_<steps>_<timestamp>.pt` file:

`python -m src.utils.generate --checkpoint /absolute/path/to/saved-checkpoint.pt --data-path data/raw/shakespeare.txt --prompt "To be, or not to be" --device mps`

## MLE research team

**[Start auto-research: fresh setup, execution, resume, and reuse of the knowledge bank](research/START_HERE.md).**
The guide includes copyable commands for the 16 GB Mac mini, where results and
learnings are stored, and what can be recovered after an interrupted run.

The [multi-agent protocol](research/TEAM.md) uses separate Codex CLI roles for
idea generation, code changes, execution preflight, and evaluation. A shared
folder of JSON records retains literature, past attempts, failures, and scoped
results and learnings. Each iteration registers quality, throughput, and memory
predictions before implementation. A deterministic controller tests isolated model/config candidates
against a fixed baseline, freezes the small-model decision, and checks whether
the effect survives larger widths. It compares forecasts with observations and
exports chronological JSON/CSV/Markdown histories of ideas, trials, performance,
costs, failures, and learnings for future iterations.

The bank defaults to `research_state/knowledge/` and is created on first use.
It uses ordinary local files, with no database setup, account, server, or paid
storage service. Agent and training compute are separate and remain disabled
during preparation.

The [16 GB Mac campaign](research/team_mac.yaml) targets widths 64/128/256 with
paired seeds and fixed training-token budgets. It supports model-code changes
and LR/weight-decay/clipping overrides. Full μP and Muon optimizer integration
remain future protocol extensions.

**Current mode: preparation only. No Codex research agents or training have been
launched for this team.** Read [program.md](program.md) for agent instructions and
[TEAM.md](research/TEAM.md) for setup, commands, limits, and evidence rules.
Campaign initialization requires the reviewed implementation to be committed in
a clean source repository; initialization calls no model and starts no training.

## Standalone learning-rate studies

The existing [LR protocol and harness](research/README.md) remain available for
bounded LR sweeps, small-width screening, paired-seed confirmation, larger-width
transfer, and held-out audit without running research agents.

Preview the matrix without training:

```bash
python -m src.research.runner plan --config research/mac_lr_transfer.yaml
```

Reports, logs, and checkpoints stay in ignored `research_runs/` directories.
Neither workflow commits or pushes changes automatically.

## Individual training budgets and checkpoint resume

This section continues one model's training. For the multi-agent workflow, use
the [campaign resume instructions](research/START_HERE.md#6-resume-the-same-campaign).

Set a step budget and optional cumulative training-step time cap:

```bash
python -m src.train.train training.max_iters=1000 training.max_train_seconds=1200
```

Evaluation and checkpoint I/O do not count toward the time cap. A step already
started completes before stopping. Set `training.max_train_seconds=null` for
fixed-step/token experiments. With checkpoint saving enabled (the default), each
completed run saves a final checkpoint and a new best checkpoint whenever
validation improves. Set `+training.summary_file=outputs/train-summary.json`
to also write a summary JSON containing both paths.
Use separate checkpoint/output directories per experiment.

Resume with the desired **total** number of steps (not additional steps):

```bash
python -m src.train.train training.resume_from=/absolute/path/to/saved-checkpoint.pt training.max_iters=2000 training.max_train_seconds=2400
```

Use the same configuration and overrides as the original run. This example
raises the cumulative time cap to 2,400 seconds; set an explicit cap or `null`
on resume. The prior cap itself is not restored from the checkpoint.

Resume preserves weights, optimizer, PyTorch CPU/accelerator RNG, loader RNG,
metrics, best-checkpoint reference, and cumulative training time. Keep the data,
model, optimizer, batch size, seed, evaluation sample count, device and PyTorch
version unchanged; incompatible resumes are rejected. If the old cumulative time
cap is exhausted, increase it or disable it explicitly. Keep referenced best
checkpoint files when moving runs. Exact replay is tested on CPU; accelerator
kernel nondeterminism may prevent bitwise replay even with restored RNG state.

Checkpoint writes are atomic. Version-2 checkpoints identify the corrected RoPE
and MLA math. Legacy checkpoints are rejected by inference/evaluation/resume:
use the original code revision to inspect them, or train a fresh corrected model.
There is no lossless automatic conversion for the old MLA head-mixing behavior.

Memory reports identify their meaning. CUDA reports peak allocated tensors;
MPS reports current allocated tensors because no peak counter is available here;
CPU RSS is the process-lifetime high-water mark. These are not interchangeable,
and MPS current allocation must not be called peak memory.
