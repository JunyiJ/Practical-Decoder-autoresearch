# Practical-Decoder
A repo to implement the decoder-only GPT-style model and play around with different attention mechanism and MoE structures


## To train the model
`python -m src.train.train`

## To generate with the trained checkpoint
`python -m src.utils.generate --checkpoint checkpoints/final.pt --data-path data/raw/shakespeare.txt --prompt "To be, or not to be" --device mps`

## To run autonomous config research
Point your coding agent at [program.md](program.md). The agent loop is config-only to start: it mutates `src/config/mac_tinyshakespeare.yaml`, creates per-experiment git branches, runs short screening experiments, and writes untracked reports and summaries.
