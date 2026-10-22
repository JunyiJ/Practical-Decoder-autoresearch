# Practical-Decoder
A repo to implement the decoder-only GPT-style model and play around with different attention mechanism and MoE structures


## To train the model
`python -m src.train.train`

## To generate with the trained checkpoint
`python -m src.utils.generate --checkpoint checkpoints/final.pt --data-path data/raw/shakespeare.txt --prompt "To be, or not to be" --device mps`

## To run autonomous config research
Point your coding agent at [program.md](program.md). The agent loop is config-only to start: it mutates `src/config/mac_tinyshakespeare.yaml`, creates per-experiment git branches, runs short screening experiments, and writes untracked reports and summaries.

## Budgets and resumption

Set a step budget and optional cumulative training-step time cap:

```bash
python -m src.train.train training.max_iters=1000 training.max_train_seconds=1200
```

Evaluation and checkpoint I/O do not count toward the time cap. A step already
started completes before stopping. Set `training.max_train_seconds=null` for
fixed-step/token experiments. Every run saves a final checkpoint and a new best
checkpoint whenever validation improves; summary JSON includes both paths.
Use separate checkpoint/output directories per experiment.

Resume with the desired **total** number of steps (not additional steps):

```bash
python -m src.train.train training.resume_from=/absolute/path/to/final.pt training.max_iters=2000
```

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
