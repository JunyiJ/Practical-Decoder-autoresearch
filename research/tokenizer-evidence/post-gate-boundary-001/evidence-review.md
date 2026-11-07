# Post-gate research boundary: depth-scaling applicability question

Local facts: the native elementwise gate loses to unchanged MLA on all3 fresh
RAW and H200 endpoints. Close this intervention with no rescue. The original
body-allocation panel's7×192 candidate loses to4×256 by+0.007499234CE, despite
beating its nearly parameter-matched shallow192 control by−0.015588741.
Depth, FFN width, table capacity and optimization geometry differ; this is not
a pure-depth result. The prior MoE panel's dense1344 control has descriptively
lower loss than dense672 on all3 paired seeds (mean approximately−0.004876),
but was not an adoption candidate. No promotion or pooling of old scores.
Sources: TOKENIZER_BODY_ALLOCATION_RESULTS.md and its original final review;
TOKENIZER_MOE_RESULTS.md and post-prefix-boundary-001/quality001-final-review.json;
TOKENIZER_MLA_GATE_RESULTS.md and original gate final/acceptance/curve records.

Literature read at this boundary:
- Bordelon et al., ICLR2024 proceedings version, §2 equation1/Table1, §3/Figure3
  Adam/ViT discussion and §6 limitations. [Primary paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/5f5f7b6080dcadced61cf5d96f7c6dde-Paper-Conference.pdf)
- Yang et al., Tensor Programs VI, arXiv2310.02244v1, §9.1–9.2 and §11/Figures15–16.
  [Primary paper](https://arxiv.org/html/2310.02244v1)

External evidence: residual scaling can help depthwise hyperparameter transfer
in studied ResNets/ViTs. TPVI cautions that multi-layer residual branches have
limitations and transformer transfer depends on training time. Neither proves
our shallow FP32 MLA/SwiGLU/Muon-Adam recipe gains from added depth. The vision
settings and entrywise-optimizer assumptions differ materially. A simple
1/sqrt(depth) multiplier is not sufficient justification for a local campaign.

Inference and ranking: (1) body allocation with explicitly depth-coupled
optimization is the next applicability question, not an admitted experiment;
(2) recipe horizon/EMA/batch alternatives remain deferred because sparse curves
and RAW-H200 gaps do not diagnose their optimal values, and historical negative
results must be reconciled; (3) norm/residual interactions remain possible but
lack a new specific mechanism, while matched-epsilon LN's earlier small gain
failed its practical gate; (4) gate, MoE and vocabulary replays are rejected.
No candidate yet qualifies for a fresh numerical panel. Carry proposed
exploration allocation forward rather than fabricate novelty or relabel analysis.

Next executable bounded work: reconcile the actual body-allocation source's
residual multipliers, width/readout rules and matrix-Muon versus AdamW grouping
with these two papers' depth/LR assumptions. Determine whether a discriminating
depth×scaling comparison has a new local rationale distinct from the closed
4×192/7×192 and fixed-half residual studies. Read original fixed-half residual
acceptance/curves before any design. Produce one derivation/applicability decision,
then select or reject; do not extend optimizer instrumentation or replay proofs.
This is source/existing-data analysis with zero model updates/scoring/holdout,
no checkpoint loads or quality seeds, a120s/2MiB metadata bound. If a hypothesis
qualifies, declare exact controls, finite equal tuning, fresh confirmation and
complete retained-output/time forecast before implementation or scientific launch.
No training budget is currently frozen. All scientific1084bindings are immutable.
