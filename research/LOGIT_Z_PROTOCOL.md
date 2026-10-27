# Prospective output log-normalizer penalty

Integrated implementation and the complete CPU assertion audit are finished; no MPS pilot or research registration has run. All5,689 unique assertions passed, including27 actual geometry/evaluator/trainer cases and535 metadata cases. The complete audit preserves two monitoring failures and their reviewed continuations: orphan batch84 has no recoverable OS exit code; batch99 contains four preserved passing assertions with unavailable prefix durations/warning summary plus two fresh completed cases. No successful case was rerun. The original900/5400-second budgets include the600-second orphan reservation and interrupted work. These are CPU correctness checks, not quality or MPS evidence. Six fresh serial MPS pilots with independent linked objective/gradient/cache/pure-CE/resource checks remain mandatory before registration. Final audited position retention supplies RoPE/local values/unit gains/SiLU/attention scale0.5/peakLR0.01 and all other inherited controls.

A small training penalty on the output softmax log-normalizer may improve fresh validation CE. Local benefit and relevant instability are unknown; no measured logit-drift diagnosis.

Formula: For each valid training target t, a_t=logsumexp(final returned FP32 logits_t). Total training loss = mean_t CE(logits_t,y_t) + lambda*mean_t(a_t*a_t). Use identical valid-target mask and denominator for CE and penalty; current paired blocks have no padding. No log(vocab) subtraction, logit centering, clipping, detach, per-width coefficient or learned temperature.

Gradient: For N valid tokens, d penalty / d logit_tj = (2*lambda/N)*a_t*softmax(logits_t)_j. Differentiate through final readout, inherited table/width scaling and any inherited output bias. This is an auxiliary objective, not router z-loss or label smoothing.

Coordinate caveat: CE is invariant to a common per-token logit shift; squared log-normalizer is not. Fix final-logit convention, prior/readout bias, table scale and width scaling across arms. No new trainable centering/bias parameter. Gradient and optimizer trajectory change even if a common shift alone would preserve probabilities; no pure confidence or numerical-stability attribution.

Dtype and zero path: Research FP32 final logits; stable logsumexp without naive exponentiation. Coefficient zero follows historical CE path with no extra loss arithmetic or RNG; model logits/init/cache/parameter groups remain unchanged. Strict finite nonnegative coefficients and explicit objective policy, saved/current/constructed semantics and resume/deletion checks passed actual CPU validation; MPS verification remains required.

Validation contract: All sampled validation, checkpoint full validation, screening, selection, confirmation and reported primary/raw CE exclude the penalty. Training logs separately record pure CE, unweighted mean squared log-normalizer, weighted penalty, total loss and coefficient; record mean log-normalizer on the existing training batch only. No new corpus evaluation or historical checkpoint scoring. Existing aux/router metrics must not silently acquire a different meaning. No reliance on model.train()/eval() alone; explicit training-loss flag and evaluator tests.

Inheritance: Resolve ALL position/value/gain/activation/attention/topology/table/bias/prior/readout-norm/input/LR/decay/warmup/H200 only from FINAL complete independently audited position-quality-001. Never partial screening or confirmation. The final complete position review retained RoPE; its entire report and bootstrap were reconstructed before inheritance.

Selection: Each coefficient selects lowest mean H200 pure CE, tie lower LR. Best family by mean, tie none/z1e4/z1e3. Nonzero best requires mean gain>.005 and every screen seed negative against both tuned none and fixed incumbent; none LR-only versus fixed separately. Failed best retains incumbent without alternate fallback. Freeze before fresh confirmation.

Adoption: Only frozen eligible winner, at BOTH widths: mean pure-CE gain>.005, every fresh seed negative and all2048/4096/8192 conditional text interval upper bounds<0 versus required references. No grid extension, pruning, larger-width reselection, alternative promotion, historical pooling or holdout.

Returned weights: Inherit H200,101 snapshots2000..3000 every10; no new averaged state/parameters. Raw parents resume; H200 inference only. Raw and H200 correlated, not independent training.

Validation: Full111538targets once per raw/H200 output ctx128B8; H200 primary. 2000 paired conditional block bootstrap draws seed20261024 for2048/4096/8192. Adaptive validation, not holdout/population seed uncertainty.

Arms: lambda=0, 0.0001, 0.001; LR=.0025/.005/.01. Exactly18 width64 screens2501/2502, then18 selected confirmations2511/2512/2513 at64/256 plus0/6 exact fixed references:36/42 physical trajectories,72/84 correlated outputs. Proposed seeds scanned but not reserved.

Hard caps:42 trajectories,2400seconds/job,21600seconds/round,4096MiB artifacts,4096MiB sampled process RSS,6144MiB current MPS driver allocation. Serial, first failure stops, no automatic retries. Actual CPU assertions are verified; actual MPS prerequisites and measured resource projections remain pending.

- Final position full audit/docs; strict final-only predecessor resolver reconstructs full report/bootstrap/adoption
- Separate reviewed objective integration/controller/independent report audit; historical zero path/default semantics unchanged
- Independent scalar/NumPy loss and gradient, extreme finite logits, common-shift behavior, exact coefficient application once, valid-mask denominator, zero path; complete model/readout gradients and finite optimizer states
- Actual CPU training/resume before and during cooldown/H200; RNG, loss logging, all evaluators pure CE, strict saved coefficient/deletion and derived lineage; bounded full regression only after current worker exits
- Six serial MPS40update upperLR.01 pilots3coefficients x64/256, checkpoint-bound objective/readout/fullgraph gradients and cache, raw/H200 fullcoverage, optimizer/RNG and50percentmargin resource gate
- Clean reviewed published source/full bank/fresh seed scan/new unique finite preregistration; one worker first failure stop/no automatic retry
- PaLM and T5X supply motivation and a coefficient precedent, not evidence on this FP32 character model/Muon/H200 recipe. The .001 arm is a prospective tenfold contrast, not a published optimal value.
- No assumed bfloat16 issue, diagnosed training instability, quality gain, reproducibility, speed or optimal muP claim. No added parameters or initialization draws were verified in actual CPU cases; MPS verification remains pending.
- The inherited final-layer normalization and no-output-bias path may constrain common logit shifts; changing logits need not be a harmless gauge adjustment. Keep full logit convention fixed.
- More per-update arithmetic and altered gradients; equal updates/tokens is not equal compute or effective update norms. Existing decay/output-bias/horizon failures do not establish a causal need for this penalty.
- Integrated objective/controller implementation passed27 actual CPU cases and the complete5689 assertion audit. Original pre-integration prose is preserved in the tested-protocol archive. MPS and registration remain pending; all closed studies and006 permanent stop are preserved.

Primary motivation: PaLM reports adding 1e-4 times squared log-normalizer during training; official T5X implements that penalty and its softmax-weighted gradient, and T5.1.1 base config uses1e-4 without label smoothing. These are external results/designs, not local measurements.

- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/html/2204.02311v5) — Section5 Loss function paragraph
- [Official T5X losses implementation](https://github.com/google-research/t5x/blob/main/t5x/losses.py) — cross_entropy_with_logits forward and backward
- [Official T5.1.1 base configuration](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/base.gin) — Z_LOSS and LABEL_SMOOTHING

Implementation clarification: zero coefficient returns the exact historical CE tensor and does not compute logZ diagnostics; its logZ mean, square and penalty fields are null (coefficient zero is recorded). Nonzero arms log these values on the existing training batch. GPT defaults include_training_objective=False independently of train/eval mode; only the configured trainer call opts in. All sampled/full validation remains pure CE. Exact current/saved/constructed kind/coefficient/policy is checked even when loading inference-only weights.

Historical preparation revision003 added logit_z_study, logit_z_predecessor and logit_z_audit. The controller freezes18 pure-CE screen choices before18/24 fresh confirmation cells, with exact zero-objective fixed-reference aliases only. The final-only predecessor reconstructs the entire completed position report, selection and independent bootstrap; objective families never overwrite inherited position/value/gain. The report auditor independently reconstructs all primary/raw means, seed deltas, conditional text intervals, fixed-LR contrasts and adoption. All validation ranking remains pure CE.

Historical metadata preparation:255 unique synthetic controller/resolver/report cases passed with GPT construction, training and real worker entry points blocked. First154 passed before a new inheritance fixture incorrectly requested reference LR.01 despite its own incumbent.005; the controller correctly rejected it. The fixture now requests its recorded incumbent LR; only failed/unexecuted101 cases ran afterward, not the154 passed cases. Production controller bytes were unchanged by this fixture repair. Earlier280 primitive metadata cases were not repeated. These535 metadata checks and six earlier scalar methods are not actual model or MPS correctness evidence. Those previously deferred27 actual cases now passed after position completion; the full current CPU audit supersedes that preparation-only status, without treating CPU results as MPS or quality evidence.
