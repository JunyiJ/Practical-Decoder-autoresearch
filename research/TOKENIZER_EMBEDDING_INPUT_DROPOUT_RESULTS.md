# Input-only embedding-feature dropout: not adopted

The complete frozen comparison found that dropout .1 worsens full-validation
H200CE on all three fresh seeds. Keep the incumbent and input dropout0.

| Fresh seed | Control CE | Dropout .1 CE | Candidate−control |
|---:|---:|---:|---:|
|200090011|4.678187664|4.690571686|+.012384022|
|200090012|4.687661526|4.704242143|+.016580617|
|200090013|4.699654617|4.714795477|+.015140860|

Mean delta is+.014701832892nat/token. All frozen adoption gates fail: mean is
not<−.02, no paired delta is negative, and all three simultaneous conditional-text
upper bounds are positive (.017432391493/.017859467856/.018062963938 for
2048/4096/8192-token blocks). These reused-text bounds condition on three trained
seeds; they are not holdout or population-seed confidence. No rescue, rate/site
extension, seed replacement, partial-panel selection or gate change occurred.

Both arms independently selected nominalLR.00125 from the same three-rate
allowance. All12screening/fresh endpoints independently passed1200updates,
1,228,800targets,41H200snapshots and complete RAW/H200/optimizer/EMA/RNG/sampler/
clipping/full-validation/cache acceptance. Independent final grouping/bootstrap/
quantiles agreed within1e-12. The original7200sec/RSS4GiB/MPS6GiB/reserve8GiB/
output guards passed. Every pilot/screen/fresh/bootstrap200090* seed is used.

Mean fresh training-loop times are273.79s control and275.60s candidate. Parameters
and cache are identical:28,492,416parameters/88tensors and196,608B/item. Serving
measurements use one resident model,2warm+5timed repeats; full128median ratios are
.9466/.9530 and decode32ratios1.0526/1.0224 at batches1/8. These are descriptive
local timings; eval equations are identical, and they do not justify a model-cost
benefit or override failed quality gates. This scoped result does not show that
all dropout rates/sites fail or that a measured train/validation gap was overfit.

Root `research_runs/tokenizer-embedding-input-dropout-quality-v1-001` has terminal
STOP closing ALL original training/controller/inference/final commands. Never
rerun them or clear STOP. The16selected/fresh RAW/H200 payloads4,474,027,344B remain
protected in original paths. Eight losing-screen payloads were permanently
retired under the original selection/audit contract; resolve absent files through
its exact manifest. Original failed fixtures/source/log/config/metrics/receipts
remain immutable; all972 frozen bindings are unchanged.

Read-only [closure](tokenizer-evidence/embedding-input-dropout-preparation-001/quality001-closure001.json)
verified all original records/guards/payload hashes. The complete metadata mirror
copied122additional files with ZERO weight copies. Follow actual state and
[the next evidence review](tokenizer-evidence/tokenizer-capacity-boundary-001/evidence-review.md):
input-capacity diagnostics do not establish a low-rank/frozen-input gain. A
lossless smaller-vocabulary structural component and common-byte metric now have
actual correctness/data proofs; real transport/worker/resource gates and a new
finite design still precede any quality comparison. This is ongoing research,
not adoption of a tokenizer or a new scientific result from compression metrics.
