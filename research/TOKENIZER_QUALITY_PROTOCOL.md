# GPT-2 tokenizer quality transfer

The human authorized this new ongoing campaign on October 26, 2026 (Los Angeles):
sync GitHub learnings, use the best model as baseline, switch to a real tokenizer,
and continue improving quality until human interruption. The checkout was
fast-forwarded from `63d3970` to `ce274e5`. Historical character studies and the
unfinished null-attention campaign remain closed. Research runs on
`codex/tokenizer-quality`; the previous heartbeat remains paused. A new heartbeat
in the current chat checks progress every five minutes and reports meaningful
results, learnings and failures.

## Baseline and transferred evidence

Use the final retained recipe from [the handoff](RESEARCH_HANDOFF.md), derived
from its exact saved request: four-layer dense SwiGLU21D/8, standard MHA with
head dimension32, RoPE, RMSNorm blocks, affine final LayerNorm, untied tables,
no output bias, attention score multiplier0.5, dropout0, Muon/AdamW with the
reviewed width rules and independent decay0.00006. Peak LR0.01, 3,000 updates,
context128, batch8, paired512 sampling, cooldown from update2000 to a final
0.1 multiplier, and spaced-tail EMA with half-life200 (H200).

This transfers architecture and training choices, **not character checkpoint
weights**. Vocabulary dimensions change and fresh paired seeds train from
scratch. Prior evidence supports testing depth4, cooldown and H200; it does not
establish their superiority with subwords. The previous failures of greater
depth, longer training, MoE and various normalization changes narrow hypotheses
but do not prohibit a justified new test under the changed tokenizer.

The compact no-bias tokenizer policy retains the complete inherited model
contract without embedding an invalid 65-character unigram prior. It permits
only an absent output bias, binds actual token IDs and tokenizer metadata, and
leaves the historical character-prior contract unchanged.

## Data and measurement

Use the existing Shakespeare corpus (SHA256
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`).
Freeze the same contiguous raw-text80/10/10 boundaries **before** independently
encoding each split with tiktoken GPT-2 BPE, vocabulary50,257. Literal special
marker strings are ordinary text. Record implementation/version, vocabulary
hash, regex, special-token policy and split policy in raw checkpoints; propagate
tokenizer identity to H200 exports and validate it on generation/evaluation.
Legacy configurations without `data.tokenizer` retain character behavior.
The normal training configuration now explicitly selects `gpt2`.

Primary metric: full-validation H200 cross-entropy in nats per **subword token**,
every target except the first scored once, disjoint context128 windows with an
unpadded tail. Record raw endpoint CE, perplexity, total target bytes and bits
per byte (BPB) as diagnostics. BPB excludes the first token's bytes; context
resets and effective text context still differ from character evaluation.
Do not compare subword CE with character CE or claim a fair cross-tokenizer win.
The test loss is never computed. Online sampled validation is diagnostic and
never chooses checkpoints. Fixed-prompt samples are descriptive, not selectors.

The same3,072,000 target-token budget now represents more raw text exposure and
fewer distinct corpus tokens than before. Vocabulary tables and logits are much
larger. Equal subword-token budgets are not equal character exposure or compute.

## First finite experiment: inherited learning-rate transfer

Hypothesis: LR0.01 may not transfer to the enlarged embedding/readout tables;
0.005 or0.0025 may improve final validation. Direction is uncertain; no numerical
quality gain is forecast. Within-width LR choices should have similar costs,
but actual costs and failures will be measured. This experiment tests a new
vocabulary condition, not an extension of an old closed character LR grid.

1. Run separate40-update GPU correctness/resource pilots at widths64/256,
   seed2591. Full validation and H200 export/load must work. These short pilots
   are excluded from quality rankings and do not establish training stability
   over3,000 updates. Their short EMA tail is only a correctness check.
2. Screen widths64 at LR0.01/0.005/0.0025, seeds2601/2602, for3,000 updates each.
   Reverse LR order on the second seed. A candidate qualifies only if mean
   H200 CE improves by more than0.02 and both paired seeds improve versus0.01.
   Among qualifiers choose lowest mean CE (then lower LR for ties); otherwise
   retain0.01. Save this choice before confirmation.
3. Confirm the frozen LR versus0.01 at widths64/256, fresh seeds2611/2612/2613.
   If the retained choice is0.01, run it once per cell; those are baseline
   replication cells, not independent duplicate controls. Total12 or18 full
   trajectories. No larger-width reselection or grid extension.
4. Adopt only if mean gain exceeds0.02 and every paired seed improves at both
   widths, plus negative upper bounds for each predeclared conditional text
   bootstrap (blocks2048/4096/8192,2,000 draws, seed20261024). Otherwise retain
   the inherited LR. These intervals condition on trained seeds and reused
   validation text; they do not measure seed-population uncertainty.
5. Review all evidence and write a separate finite protocol for the next
   justified hypothesis. A completed round is a transition, not the end of the
   human-authorized research loop. No automatic promotion based on samples.

## Execution, safety and provenance

Use one local MPS worker, four CPU threads, no CPU fallback and no paid remote
compute. Freeze source hashes, runtime, config, corpus hash, prior evidence and
plan digest before dispatch. The controller uses process locks, an exclusive
trial directory and immutable requests. Raw and H200 checkpoints stay separate.
A failed or incomplete trial is retained and stops the finite round for diagnosis;
no silent retries or replacement seeds. A new source fix needs a new evidence
contract. A corpus or source change blocks further dispatch.

Pilot timeout600s; full-trial timeout7,200s including validation/export. Existing
sampled worker watchdog limits RSS4GiB and MPS driver6GiB. Keep at least8GiB free;
cap round output at12GiB. These sampled guards are not instantaneous memory
limits. Resource projections after the pilots determine whether the planned
round fits before launch. Preserve raw failure evidence; do not delete it to
make a run appear clean.

Create a STOP file in the active run directory to stop at a trial boundary.
SIGINT/SIGTERM to the controller records the same request. No later invocation
clears STOP automatically. Abruptly interrupted trials require explicit review.
The heartbeat checks actual processes/logs, records results and scoped learnings,
and commits/pushes the research branch without force. It stays quiet when state
has not changed. A human stop pauses the new heartbeat and prevents new trials.

CLI: `.venv/bin/python -m src.research.tokenizer_study prepare --output <new-path>`;
add `--pilot` for the pilot plan. Then `run --output <path>`; `status` is read-only.
Generated plans and raw evidence live under ignored `research_runs/`; reviewed
results and portable metadata are checked into Git. Checkpoint backups require
separate storage; no binary training checkpoints are pushed to GitHub.
