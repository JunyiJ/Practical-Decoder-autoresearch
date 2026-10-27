# Null-attention prototype: unregistered

Primitive, narrow model wiring, finite controller, final-only predecessor resolver and independent report auditor are implemented in a separate development archive. The256 metadata/scalar cases passed;28 actual model/trainer cases are written but unexecuted. Final logit-z completion was independently audited and its resolver retains the zero objective, RoPE/local values/unit gains/SiLU/attention0.5/LR0.01. Integration and all CPU/MPS/resource/publication/preregistration gates remain pending.

# Prospective fixed-null attention normalization

Design only. No source integration, model tests, pilots or research registration.

## Hypothesis

Allowing attention probability mass on a fixed zero-valued null option may improve fresh validation CE. Local benefit, attention-sink pathology and outlier reduction are unknown.

## Formula

{
  "scores": "s_j = inherited attention scale * Q_i K_j / sqrt(32), after inherited positional processing and any positional score bias; only causally visible real keys are valid.",
  "standard": "p_j=exp(s_j)/sum_valid exp(s_k), historical exact implementation path.",
  "prefix_control": "p_j=(n/(n+1))*softmax(s)_j, where n is the number of visible real keys for this query, not chunk size or allocated cache capacity.",
  "null_zero": "p_j=exp(s_j)/(1+sum_valid exp(s_k)); equivalent to appending one fixed score 0 and zero value per query/head, normalizing, and discarding its probability before the real-value product."
}

## Stable implementation

Use softmax over the masked real logits plus a fixed zero column and then slice real probabilities. The fixed null score is appended AFTER all content/positional scaling and is never causally masked. Do not subtract a maximum from real scores without applying the identical shift to the null score. Reject zero-visible-key rows; no learned null logit, key, value or parameter.

## Gradient

For null_zero real probabilities p, dp_j/ds_k=p_j*(delta_jk-p_k). For output y=sum_j p_j*v_j and upstream g, dL/ds_k=p_k*dot(g,v_k-y). The real-probability Jacobian does not generally annihilate an all-ones score shift. Prefix control multiplies the ordinary softmax Jacobian by n/(n+1). Independent full graph VJP checks required.

## Control

At all real scores zero, prefix_control and null_zero both give p_j=1/(n+1), while standard gives 1/n. For nonuniform scores or common score shifts they generally differ. The control matches zero-score attenuation, not observed mean attention mass, full initial logits, subsequent trajectories or a pure mechanism.

## Placement

All four fused MHA layers, before inherited attention dropout and weighted-value product. Keep real-value reuse, projection biases, output gate and complete-branch gains inherited. Null attenuates the weighted-value term; output projection bias is unchanged, so this does NOT make the whole attention residual branch exactly zero.

## Coordinate caveat

Null attention is not invariant to a common shift of real attention scores when its own score stays zero. Coordinate convention, inherited attention temperature and positional biases must remain fixed. No diagnosis of outliers, attention sinks, quantization or numerical instability from this design.

## Cache

No null K/V is persisted or appended to real cache; real cache geometry and positions unchanged. Derive n from the actual causal visibility mask for full, token, uneven chunks, reset, interleaving and nonzero offsets. Cached query n includes the existing prefix. Zero-score attenuation at n=1 is 1/2 and at n=128 is 128/129: algebra only.

## Inheritance

Resolve ALL objective kind/coefficient/pure-CE evaluation semantics, position/value/gain/activation/attention/topology/table/bias/prior/readout/input/LR/decay/warmup/H200 exclusively from FINAL complete independently audited logit-z-quality-001. No assumption of zero-objective retention and no partial-screen inheritance.

## Selection

Each family chooses minimum mean H200 pure CE, tie lower LR. Best selected family by mean, tie standard/prefix_control/null_zero. Changed best requires gain>.005 and every seed negative against both tuned standard and fixed incumbent; standard LR-only versus fixed separately. Failed best retains, no fallback. Freeze before any fresh confirmation.

## Adoption

Only the frozen eligible family: BOTH widths mean pure-CE gain>.005, every fresh seed negative, all 2048/4096/8192 conditional text interval upper bounds<0 against required references. Null-minus-prefix-control is a separate explanatory contrast, not a hidden additional adoption gate. No larger-width reselection, grid extension, pruning, historical pooling or holdout.

## Validation

Full111538 once per raw/H200 endpoint, H200 primary; raw correlated diagnostic. H200101 snapshots2000..3000 interval10, raw resumes/derived inference only. Paired conditional block bootstrap2048/4096/8192,2000 draws seed20261024; adaptive validation, not training-seed population uncertainty.

## Prerequisites

[
  "Final logit-z full audit/documentation; final-only resolver independently rebuilds whole report/bootstrap/adoption.",
  "Separate reviewed narrow named normalization policy and unchanged default path, strict saved/current/constructed/deletion/parameter absence guards. No generic relaxation of legacy MHA policy.",
  "Independent scalar and full-model forward/VJP including masks, extreme finite scores, common shifts, null probability mass, zero-score control, causal prefix counts and all inherited objective/readout gradients.",
  "Actual CPU shared learned initialization/RNG/parameter counts/optimizer groups and steps, resumed training/H200/private RNG, pure-CE evaluators and strict serialization; bounded full regression after current worker exits.",
  "Six serial40-update upper-LR.01 MPS pilots3families x64/256 with checkpoint-bound full graph/cache/primitive audits, raw/H200 fullcoverage, objective logging and pure-CE checks, optimizer/RNG and50percent resource margin.",
  "Pilot240sec/job1800sec/round1024MiB; research caps unchanged. Clean reviewed source committed/pushed, complete knowledge bank, fresh seed scan and unique finite preregistration before launch."
]

## Limitations

[
  "Zero new learned parameters/buffers/init draws intended, not yet verified; no model or tensor implementation has been executed.",
  "Miller blog is a primary proposal/opinion, not empirical local evidence. Quantizable Transformers studies clipped softmax/gating, not this exact null-zero mechanism. No replication claim.",
  "Earlier output-gate and complete-branch gain failures are contrary context, not evidence that this distinct normalization works. Learned score dependence and prefix attenuation both affect gradients and compute.",
  "Equal updates/targets is not equal compute, effective update norm, pure regularization, optimal muP or inference speed."
]

The finite grid is three families × LR .0025/.005/.01 × width64 seeds2521/2522 (18 screens), then all selected families × widths64/256 × fresh2531/2532/2533 (18 confirmation), plus0/6 exact fixed references. Total36/42 physical trajectories and72/84 correlated raw/H200 endpoints. Proposed seeds are unreserved. Budgets remain42 trials,2400s/job,21600s/round,4096MiB artifacts/RSS and6144MiB current MPS driver; one worker, first failure stop, no automatic retry. All older studies and006 permanent stop remain closed.
