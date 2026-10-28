# Prospective subword gate-activation comparison

October28,2026. Deferred by later human steering to a wider direct MLA/GQA comparison; see TOKENIZER_WIDE_ATTENTION_PREPARATION.md. Preparation only: no new pilot/quality training is authorized by
an executable freeze yet. Table001 is closed and audited; do not restart it.

## Hypothesis and choice among directions

Test whether exact GELU rather than SiLU as the FFN gate improves quality on the
retained width256 GPT-2 GQA1200-update/cooldown/H200 recipe. Keep legacy untied
vocabulary tables: the new table-sharing alternatives failed their fresh gates.
The gate comparison isolates a nonlinearity at identical shapes/counts rather
than changing readout coordinates or capacity at the same time. The enlarged
subword vocabulary, shorter schedule and GQA make this a different setting from
the closed character study. Direction is uncertain; no improvement is predicted.

Prior numeric evidence in GATED_ACTIVATION_RESULTS.md favored retaining SwiGLU:
GEGLU was essentially tied at character width64 but worse at256, and ReGLU was
worse at both. Equal-rate screens and independently tuned comparisons disagreed,
which supports bounded independent LR selection here. These are priors only;
no character tensor is available or needed and no character CE is compared with
subword CE. ReGLU is deferred to bound the initial experiment.

MoE remains a substantive alternative, with four-expert/top-two active/total
capacity, router overhead and load accounting required before a fair panel.
Normalization, serial/parallel residual topology and depth/scaling are also
untested in this subword recipe and remain on the agenda. Standalone LR,
regularization and EMA/schedule studies remain valid: the existing horizon result
is a combined recipe, not a causal isolation of each component. Choose the gate
comparison now for a narrow structural contrast with unchanged matrix budget;
revisit the other families at its boundary, not an architecture-only mandate.

## Proposed model and comparison

For normalized FFN input x, use y = W_down(activation(W_gate x) * W_up x).
Compare SiLU(z)=z*sigmoid(z) and exact GELU(z)=z*Phi(z), without approximation.
Both have three bias-free matrices, hidden width21D/8=672 at D256, identical
initialization draws and optimizer grouping. Initial functions/gradients differ.
Keep four layers, head32/half-KV RoPE GQA, existing block/final norms, score scale,
Muon/AdamW width rules, decay0.00006, context128, batch8, paired512 sampling and
raw80/10/10 split before GPT-2 encoding. No auxiliary loss or new parameter.

Propose two LR screens per arm (0.00125/0.0025), independently select lowest H200
full-validation CE (exact ties lower LR), then three fresh paired seeds. Main
comparison is equal1200 updates/token exposure; cooldown800–1200 to0.1 and
H200start800/interval10 remain fixed. Four screens plus six confirmations; two
retained short feasibility pilots before quality. Pilot scores never select.
Seeds, order, configs, resources and source must be reserved and frozen in a
separate executable protocol before dispatch; none are fixed by this note.

Prospective quality route: adopt GELU only if mean fresh H200 delta<-0.02 nat,
every paired seed favors it and all conditional text-bootstrap95% upper bounds
are negative (blocks2048/4096/8192,2000 draws, seed20261024). Equal parameter
storage gives no table-memory adoption route. Neutral quality alone is not an
adoption in this proposed round. Report training and measured cached inference
costs with the same workload; do not add a post-result speed-based fallback.
Validation reuse and conditional intervals retain their limitations. Test closed.

## Required implementation and launch gates

Introduce an explicit tokenizer activation policy permitting only SiLU/exact GELU
on the exact reviewed GQA backbone. Preserve historical policies and default
SiLU behavior. Checkpoint/model/optimizer lineage must reject wrong activation
or policy even when tensor shapes agree. Do not use an unreviewed override.

CPU validation must cover exact default-SiLU equivalence and initialization RNG,
parameter count/groups, analytic gate forward/gradient agreement, causal cached
parity, policy rejection, exact interrupted resume and H200 export/load. Reuse
existing model/cache machinery only after these checks. Prepare and test bounded
serial pilot/quality/timing controllers before source/plan freeze.

Keep one worker, strict MPS/no fallback, four CPU threads,600-second pilots,
7200-second quality trials,4GiB RSS/6GiB driver sampled guards,8GiB disk reserve,
exclusive run directories, locks and no automatic retries. Retain every new
checkpoint and failure. A conservative12-trajectory allocation at659,621,888
bytes each plus4MiB metadata and8GiB reserve is16,509,591,552 bytes. This is a
projection, not observed cost or a frozen launch plan; verify actual pilot costs
and all remaining allocations before every launch. No further cleanup needed.
