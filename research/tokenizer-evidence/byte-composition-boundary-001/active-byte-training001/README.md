# Guarded active rows: exact model numerics, insufficient training gain

The full-model comparison completed successfully. CPU/MPS B1/B8 cached logits and
reset matched exactly. B8T128 full logits/loss and all105 parameter gradients were
exact. CPU complete states after updates1/2, and restore-at1 continuation to2,
matched exactly. All20 MPS paired histories and the complete final states matched,
including eight averaging snapshots. Sources are mapped in
../active-byte-integration-source001/source-map.json. They remain unadopted.

The prospective timing gate was not met:

- Median candidate/reference latency0.970417:2.96% lower, below the required5%.
- Candidate faster in12/16 measured pairs; this gate passed.
- Phase medians0.977422 and0.965973; neither phase regressed.
- Measured totals13.212334s reference versus12.851886s candidate.

The candidate is not adopted. Do not relax the5% gate, rerun timing until favorable,
or use the component speedup to claim a training improvement meeting that gate.
The adopted baseline remains bulk-hash-adoption001/recipe.json. Public loader and
saved-artifact qualification for this candidate were not run after the failed gate.

The component result needs its absolute scale: isolated median saving was8.540ms,
not the entire0.195s previously attributed to byte backward by cProfile. The latter
includes dependency waits and profiling effects, so treating it as isolated kernel
cost was unjustified. Actual mean measured training saving was22.528ms per update.
These differing workloads/timing boundaries do not establish a causal breakdown.
A larger component percentage alone did not predict a useful full-training gain.

The guard finished in59.713s with3,847,782,400bytes peak RSS, within4GiB. There were
45 updates, four full-model gradient-only passes, no quality evaluation and no new
weight files. CPU sampler roles62101500001/2 and MPS62101600001/2 are now closed.
The diagnostic, primitive and training guards together used about73.14seconds.

Next research should use full-training evidence rather than the misleading profile
percentage. The user's embedding discussion also identifies a comparison issue:
the original native baseline was untied. Its28,492,416 parameters include separate
12,865,792-parameter input/output matrices. A tied version of the same body would
have15,626,624 parameters; the byte model has16,351,859, or725,235more. This is a
parameter-accounting deduction, not a measured quality result. Before proposing
a new tied-baseline campaign, inspect previous tying experiments/restrictions and
relevant primary literature under the evidence-grounded research skill.

No worker is active. Closed numerical failures and acceptance gates remain fixed.
