# Evidence review for the next structural exploration

Decision: prepare a measured shape and cost feasibility comparison before
selecting another numerical campaign. Neither greater depth nor MoE is admitted
for training by this review. The proposed exploration percentage does not make
either a mandatory winner, and no new seeds, dataset or protocol are frozen.

## Local observations

The [read-only diagnostic](analysis001.json), reproduced by [its source](analysis001.py),
checks the exact metrics hashes in all six accepted final-scale anchors. It
loads no checkpoints and performs no new quality scoring. There are only five
saved online observations per trajectory.

- Training contains 267,688 GPT-2 tokens. Each trajectory samples 1,228,800
  targets, a ratio of 4.5904 to that count. This is exposure relative to corpus
  size, not literal epochs for the random-window sampler.
- All six sampled validation curves improve at every saved observation.
  From saved step900 to1199, mean sampled validation CE changes by -0.07712
  for learned scale and -0.08579 for fixed unit. These observations do not
  establish that this current recipe is already overfitting at1200 updates.
  They also do not authorize longer training or choose an intermediate endpoint.
- The two untied 50,257-by-256 vocabulary tables contain25,731,584 parameters,
  about90.31% of the incumbent's28,492,416. This algebraic parameter fraction
  is not a FLOP, optimizer-memory or runtime fraction. A total-parameter match
  alone could obscure a large change in the smaller transformer body.
- [Historical dense capacity](../../CAPACITY_QUALITY_RESULTS.md) found a
  six-layer mean delta of -0.000674 at character width256, with a worse seed
  and46.94% more training time. [Historical MoE](../../MOE_QUALITY_RESULTS.md)
  had a +0.002608 mean delta at character width256 and78.70% more training time.
  Both findings argue against assuming that more stored capacity is beneficial;
  neither establishes the answer for the present subword MLA model.
- The [subword GQA horizon study](../../TOKENIZER_HORIZON_RESULTS.md) adopted a
  shorter1200-update combined duration/cooldown/EMA recipe versus3000 updates,
  with mean delta -0.710852. That result is specific to its backbone and coupled
  schedule change. It is contrary evidence to blindly applying large-data
  training-length prescriptions here.

Training-batch loss and sampled validation loss use different observations;
their gap is not a matched generalization estimate. These diagnoses are
separate from the frozen final-scale full-validation statistical decision.
Historical character and subword losses are never pooled.

The completed [final-scale review](../final-ln-scale-preparation-001/quality002-final-review.json)
found fixed-unit scale worse on all three fresh H200 comparisons, with mean
delta +0.0165041 nat/token. Keep learned scale. An online loss snapshot cannot
replace that complete frozen decision, and another tiny readout ablation would
remain local refinement rather than structural exploration.

## Primary papers reviewed

**Kaplan et al., arXiv2001.08361v1, sections3.1 and4, figures5/6/9.**
The paper varies shape while controlling non-embedding parameters and reports
relatively weak shape sensitivity over its tested range. It also studies
finite-data limits. This supports separating body allocation from vocabulary
tables and checking data limitations before adding capacity. Its WebText2
regime and models do not supply our optimal depth or a transferable exponent.
[Primary paper](https://arxiv.org/pdf/2001.08361v1).

**Hoffmann et al., arXiv2203.15556v1, sections3.1 and3.2, figure2.**
The authors compare model size and training-token budgets using curve envelopes
and fixed-FLOP profiles. That motivates a declared compute axis instead of
calling equal updates equal cost. Their large-model, large-data analysis does
not imply that repeated exposure to our small corpus should increase, or set
our best token-to-parameter ratio.
[Primary paper](https://arxiv.org/pdf/2203.15556v1).

**Fedus et al., JMLR23(2022), sections2.2/2.3, equations3/4.**
Switch's sparse routing requires capacity/load-balancing controls, and its
pretraining setting uses a T5-style masked objective. This informs diagnostics
for an expert proposal, but top-one routing, distributed TPU execution and that
objective differ from the local top-two causal/MPS proposal. Its reported
efficiency does not establish a local speed or quality advantage.
[Primary paper](https://jmlr.org/papers/volume23/21-0998/21-0998.pdf).

## Inference and falsifiable questions

Our inference is that the next useful structural study should distinguish body
allocation and measured compute from simply adding parameters. The evidence
does not yet identify a capacity, optimization or context bottleneck causally.

A candidate shape study would ask whether a different depth/width allocation
improves quality or a predeclared quality/cost tradeoff at a comparable body
budget. A complete fresh comparison showing no practical benefit, or loss of
benefit under the declared cost axis, would count against that candidate. It
would not establish that every depth/width ratio is exhausted.

MoE remains an alternative investigation, with a harder burden from its local
width reversal and overhead. Before admitting it, quantify expert/router work,
per-expert exposure and full retained-output cost against dense controls. A
balanced router by itself is not evidence of better quality. The old storage
forecast is historical and is not a newly measured launch gate.

## Next bounded work

Audit the actual current model and optimizer geometry, then derive a small
set of valid depth/width allocations and their separate embedding/body parameter
counts, training-compute estimates and output high-water costs. Review depth
scaling and initialization applicability before choosing exact dimensions;
do not assume the inherited width rules establish depth transfer. Compare the
information and feasibility of that design with a refreshed dense/MoE design.
If neither has a justified feasible comparison, complete the missing diagnostic
instead of spending the exploration quota arbitrarily.

This is source/analysis work. A numerical study still requires explicit additive
contracts, correctness/resource pilots, equally bounded tuning where justified,
a frozen selection rule and full three-pair fresh confirmation. Existing
holdout, retention, process and resource safeguards remain in force. The70/30
portfolio remains proposed; this evidence requirement is already binding.
