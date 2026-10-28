# Wider hybrid staged tuning draft — not frozen or launch-ready

October27 Los Angeles / October28 UTC. This selects a prospective preparation
direction using closed prior rounds and anchor geometry, before any new quality
screening. It does not reopen hybrid001, wide001, or the failed combined CPU
continuation fixture. The storage and validation gates in
[the preparation record](TOKENIZER_WIDE_HYBRID_PREPARATION.md) remain binding.

## Question and bounded tuning choice

At width512, does a3:1 KDA/gated-MLA recipe preserve or improve quality against
BOTH independently tuned homogeneous controls? Use three LR candidates per
family, then two additional bottleneck-rank candidates at that family's selected
LR. This is broader than hybrid001's two-rate scan, but is still a small staged
search rather than detailed optimization of every gate, schedule or layout.

Choose rank over the alternative shared horizon or decay-bound probes for this
draft because the anchor hybrid has almost the same state size as all-KDA and
all-gated-MLA at context128. Rank offers an explicit capacity/cache tradeoff in
global layers and a decay-parameterization tradeoff in recurrent layers. The
directions of quality and latency effects are unknown. Keep the independently
motivated horizon/regularization and MoE/activation/norm/residual/depth agenda;
this round does not make attention tuning a permanent priority.

The rank divisor b is4 for the anchor; stage2 candidates are8 and2. KDA decay
bottleneck rank is D/b; gated-MLA query and KV ranks are both D/b. The hybrid
changes these bottlenecks in both component types together. This is a combined
rank recipe, not a causal separation of recurrent versus global rank effects.
KDA recurrent state shape is unchanged by this rank. Gated-MLA latent cache size
changes. The other K3-style choices remain fixed: bounded log-decay−5, initial
A=0, convolution4, normalized Q/K, full-rank sigmoid output gates, NoPE, head32,
shared unrotated key32 and four layers. These are local adaptations, distinct
from Kimi Linear's low-rank output gate. No paper-scale reproduction claim.

Retain legacy untied GPT-2 tables, dense SwiGLU21D/8 and inherited normalizations,
score multiplier.5 for gated-MLA, Muon/AdamW width scaling, decay.00006, batch8,
context128 and paired512 sampling. All quality trajectories use1200 updates,
cooldown800→1200 to.1, H200start800/interval10. These are a fixed comparison
budget, not a proven wider/hybrid optimum. Tokens/updates are matched; parameter
counts, FLOPs, elapsed time and initialization draws are not matched.

## Proposed finite order and selection

No new tokenizer request currently uses the proposed pilot3491, screen3501 or
fresh3511/3512/3513 seeds. Repeat the freshness audit before a final freeze.

Three resource pilots would use rank D/2, the largest parameter geometry in each
family, seed3491 and20 updates. Exact pilot schedule/export checks remain to be
specified and tested. These are feasibility checks excluded from selection.
The lower-rank configurations still need their own numerical/cache prerequisites;
the largest-rank pilot does not establish every configuration's resource peak.

Stage1: nine anchor-rank LR screens, seed3501:

| LR | Family order |
|---|---|
| .000625 | hybrid, gated_mla, kda |
| .00125 | kda, hybrid, gated_mla |
| .0025 | gated_mla, kda, hybrid |

Independently select the lowest final H200 validation CE for each family;
exact ties choose lower LR. Save selection before stage2.

Stage2: six rank probes, also seed3501, using each family's selected LR:
D/8 in hybrid/gated_mla/kda order, then D/2 in kda/gated_mla/hybrid order.
For each family compare its selected D/4 anchor with its D/8 and D/2 probes;
select lowest final H200 CE, exact ties choose fewer parameters (larger divisor).
Save final choices before confirmations. Do not re-tune LR for each rank or
expand the grid. LR/rank interactions remain a limitation of this staged design.

Confirmation: nine fresh trajectories, preserving the selected family recipes:

| Seed | Family order |
|---|---|
| 3511 | hybrid, gated_mla, kda |
| 3512 | kda, hybrid, gated_mla |
| 3513 | gated_mla, kda, hybrid |

Total27 retained trajectories:3 pilots+9 LR screens+6 rank probes+9 fresh runs.
All raw/H200 artifacts remain. There is no conditional omission or duplicate
control, and no selecting the easier homogeneous comparator after seeing scores.

## Proposed gates, encoded and synthetically tested before freezing

Primary contrasts are hybrid minus all-KDA and hybrid minus all-gated-MLA on
full-validation H200 CE. Both use the same three fresh seeds. A quality route
requires mean delta<−.02, every seed delta<0 and all declared conditional upper
bounds<0 against BOTH controls. A memory route requires mean delta, every seed
delta and every upper bound<=+.01 against BOTH, plus at least10% smaller actual
allocated total recurrent/convolution/latent/shared-key state against BOTH in
EVERY fresh seed at batches1/8. The+.01/−.02 and10% thresholds are prospective
engineering conventions, not numeric tolerances supplied by the human. A
memory route has no speed-win gate; measured latency tradeoffs must be disclosed.
Missing artifacts, incorrect caches or incomplete trials prohibit adoption.

Use paired text-block bootstrap2048/4096/8192,2000 draws, seed20261028; identical
resamples across models, averaging paired deltas across the three trained seeds.
For the six contrast/block-size upper bounds, allocate.05/6 per bound and use
the corresponding one-sided quantile (99.1667%) with an explicitly tested
quantile implementation. This Bonferroni-style allocation is conditional on
trained seeds and adaptively reused validation. It does not provide independent
test evidence, population uncertainty across seeds, or repair adaptive reuse.
Report ordinary descriptive intervals separately; do not swap the gate bounds.

The pure `src/research/tokenizer_wide_hybrid_rules.py` preparation module now
encodes these selection orders and gate arithmetic. Its bootstrap consumes
aligned per-seed/per-family 2048-token block-loss sums and shared target counts;
it resamples grouped blocks with replacement, weighting the short tail by its
target count. It uses NumPy default_rng(20261028), reset for each block size,
2000 draws and the same draw indices for every seed/model/contrast. Linear
quantile interpolation is fixed at sorted position (n−1)q. Block-size keys are
strings so persisted JSON retains its contract. Twenty-nine synthetic tests
pass, including an independent scalar bootstrap reference, exact thresholds,
both-control requirements, corrupt/incomplete panels and JSON round-trip.
The module does not validate saved tokenizer/config/checkpoint identity, dispatch
workers, audit artifacts or adopt recipes. Those remain mandatory integration
steps. The draft is still unfrozen and no real quality records were analyzed.

The separate saved-score adapter now checks independently expected requests,
training/tokenizer metadata, raw/H200 hashes and meta-device tensor geometry,
plus saved score arithmetic. Its36 synthetic checks pass alongside the29 rules
checks. It supplies selection rows/block sums; it never declares tensor values
audited or a recipe adopted. Frozen controller provenance, full tensor-value
audits and the remaining numerical/serving prerequisites are still required.

After quality and saved-artifact audits, plan three serial fresh-seed inference
triples: FP32 MPS/context128, batches1/8,32prefill/64teacher-forced decode tokens,
two warmups and eight rotating repeats. Cache equality precedes timing. Freeze
the exact rotation, input hashes, limits and runner before launch. Report each
model's actual parameters/state and latency, and all-resident memory separately
from deployment peaks. The old RoPE MLA option is not a fresh control here.

## Work remaining before any launch

The separate rank policy now implements D/8,D/4,D/2 with explicit model/cache/
optimizer/checkpoint signatures. Selected geometry/equation/cache/lineage checks
passed; optimizer-resume/EMA and resource-pilot prerequisites remain incomplete.
Constructed meta accounting for every variant and worst-case selections gives
53.2948GiB for the width512 draft including reserve, retained pilot/quality
artifacts and conservative metadata/serialization allowances. Measure
actual pilot resource costs only after the complete prospective panel fits.
The approximate60GiB storage target exceeds this prospective estimate, but actual
filesystem fit and pilot costs still require verification before dispatch.

The failed combined CPU fixture remains closed. New diagnostic and synthetic
buffer tests do not complete missing model continuation checks or authorize a
failed-guard retry. Resolve the validation procedure explicitly while preserving
its failure record and unchanged4GiB RSS/8GiB reserve limits. Quality dispatch
also needs7200s/trial,6GiB MPS-driver, output bounds and global single-worker locks.
Finally integrate/test the serial controllers and audited score adapter, record source/runtime/
data/config hashes, freeze both finite pilot and quality plans, and verify fit.
This draft may change before freeze; it cannot change in response to new quality
outcomes. Test holdout remains closed and no MPS launch follows from this file.
