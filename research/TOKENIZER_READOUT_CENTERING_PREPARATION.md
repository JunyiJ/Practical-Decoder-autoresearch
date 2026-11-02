# Tokenizer final-readout centering preparation

Current status: full-model model001 and serialized model-resume001 PASS both
layer_affine/rms_affine_bias arms. All710 source/config bindings unchanged,
695 predecessors preserved. No scientific quality result, seed reservation,
pilot/quality freeze or active worker. Next actual bounded worker/independent
parent validation/inference integration; read capacity001 for the concrete recovery
route after successful resume tensor dependencies close.

The accepted incumbent has affine LayerNorm at the final readout, epsilon1e-5,
256 learned scale and256 bias parameters. Read-only inspection of accepted
clip1 H200 and raw optimizer confirms28,492,416 total parameters and44 AdamW
groups with betas(.9,.999). Earlier deferred wording claiming final RMSNorm was
incorrect; the original decision remains immutable, and completed campaigns
always validated the actual LayerNorm. Eight pre-branch RMSNorms stay unchanged.

Select a centering-only final-readout comparison:
- Control: (x-mean(x))/sqrt(mean((x-mean(x))^2)+1e-5)*weight+bias.
- Candidate: x/sqrt(mean(x^2)+1e-5)*weight+bias.

Both retain512 learned affine parameters, scale initialized1 and bias0, no RNG
changes. This candidate deliberately retains bias, unlike a default bias-free
RMSNorm. No gain assumed. Earlier block LayerNorm failures are an adverse prior
at a different placement, not a reopened trial or pooled quality evidence.
Both architecture and bounded HP reviewed: defer AdamW beta2 .95-versus-.999,
keeping beta1 .9, Muon and independent decay unchanged.

Component001 passed CPU FP64 arithmetic/finite differences, then failed an
incumbent FP32 width7 batch3 gradient elementwise bound: absolute9.8711e-5 versus
5e-5+relative5e-5 at one derivative. Its source/plan/result/log/guard remain closed.
Separate component002 keeps the normalization source and every candidate tolerance
unchanged. It adds exact native nn.LayerNorm forward and all-gradient parity and
uses a documented tensorwise max-norm comparison for the FP32 incumbent analytic
oracle; CPU FP64 finite differences and forward bounds remain unchanged. This is
a harness acceptance change, not a claim that component001 passed. All resource
limits remain unchanged. See component002-correction.json for the precise rule.

Component002 passed all24 cases at widths7/256, batches1/3 on CPU FP64/FP32 and
MPS FP32. Exact native LayerNorm forward/all-gradient parity, identical affine
initialization/construction RNG, explicit independent CPU FP64 equations, finite
differences for input/scale/bias, suffix independence and exact future-gradient
zero passed. Maximum absolute forward/gradient errors3.5333644197166336e-6 /
9.871093956093091e-5; maximum gradient normwise error2.9709798822580305e-5.
Native control parity is exact; candidate retains original elementwise5e-5 checks.
These are standalone normalization and read-only incumbent observations, not
candidate MLA/full-model/optimizer/cache validation or scientific quality results.

Historical next action (completed below): narrow additive explicit full-model final-normalization
policy, unchanged optimizer groups/shared initialization, checkpoint/cache identity
and CPU/MPS raw-model/optimizer/EMA/RNG repeat/resume/final-H200/full-cache-reset
integration outside frozen namespaces. Do not bypass existing final-LayerNorm or
MHA-only readout guards. Then actual bounded worker/independent parent validation/
inference, fresh scientific census, complete measured retained-output high-water
including serialization overlap/metadata/8GiBreserve, exact authorized recovery
if needed, finite pilots and final quality freeze. Prospective exactly THREE
fresh pairs at common LR0.00125, equal zero LR-search, original strict gates;
no screens/rescue/selection/retry/partial panels. Standard-recipe scope only.

Retain clipping1.0/unit gains/D256 MLA score0.5/SwiGLU672/SiLU/blockRMS/serial/no
product norm/local mixing/warmup, decay0.00006/cooldown800/H200 and final LayerNorm.
Clipping quality002 closed with mean+0.004133149591824823, all seeds worse/all gates
failed. All earlier campaigns/fixtures stay closed. Preserve every selected/fresh
model, original failure evidence and114 protected lossless RAW archives. Archived
RAWs are recoverable, not retired; all H200 original paths and storage004 intact.
No permanently retired tensor restoration or closed-fixture reruns. One worker/
four threads/all resource guards, no paid compute/subagents, closed holdout.
Current free space is not panel readiness; small source/numerical work proceeds.


## Explicit full-model integration and serialized resume complete

A new policy in research/readout_centering_model_v1 binds final kind/epsilon/bias
to construction, architecture, optimizer, raw/H200 checkpoints and request caches.
Existing final-LayerNorm and MHA-only contracts still reject the new fields. No
frozen namespace or guard was bypassed. Both arms have28,492,416 parameters and
88 EMA tensors with exactly identical shared initialization/RNG and Muon/AdamW
groups; LayerNorm initial logits exactly match the incumbent. RMS-with-bias
intentionally changes the function. Explicit actual MLA readout equations, CPU/MPS
forward/all-gradient checks, full/cache/reset/causality and foreign readout
checkpoint/cache rejection passed. Cache196,608B per item remains unchanged.

Both generated four-update arms repeated and resumed from serialized step2 with
exact raw-model/optimizer/EMA/RNG/clipping-history state (maximum error0). Step2
was serialized only in capped memory; final-step4 raw/H200 exports have3 snapshots.
H200 CPU/MPS and full/cache/reset checks passed batches1/8; maximum CPU/MPS logits
error8.940696716308594e-7 and cached error9.5367431640625e-7. These are generated
prerequisites, not scientific quality or long-run stability results. Actual
run_training/output/worker and independent parent full validation remain untested.

Storage012 resolved the complete resume-pair shortfall by losslessly archiving
exactly six closed branch-scale fresh RAWs:654,730,043B saved,50,525 other files
verified unchanged, exact restoration PASS. Original paths resolve through
storage-recovery-012/raw-completed.json and protected-raw-storage-012 archives.
All RAW bytes remain protected/recoverable, NOT retired; H200 originals unchanged.
Together storage006–012 preserve120 RAWs. After recovery, actualfree10,287,386,624B
exceeded the complete9,781,116,928B resume requirement, including8GiBreserve.

Next executable action: minimal additive actual run_training/output/worker and
independent parent full-validation/inference adapters outside710 frozen sources,
with explicit readout-policy data routing and complete clipping-history acceptance.
Re-audit actual retained resume raw/H200/optimizer/EMA/RNG/history/cache and close
tensor dependencies. Then independent exact successful numerical retirement if
needed and actual full worker-pair capacity recheck. Four successful generated
weights total1,118,317,844B and remain retained.
Capacity001 observedfree8,974,565,376B versus complete9,814,671,360B;
conditional post-retirement10,092,883,220B is a concrete recovery route after
dependencies close and exact audit, not dispatch readiness or missing consent.
Source work proceeds. Scientific-panel projection is separate and requires fresh
census, measured complete retained-output high-water, finite pilots and quality freeze.


## Readout-centering real worker and paired inference complete

Additive research/readout_centering_worker_v1 explicitly routes the readout policy
through actual run_training, data loading, bounded raw/H200 output and independent
parent full validation. Both generated four-update layer/rms_bias arms passed:
4,096 training targets,3 H200 snapshots and88 EMA tensors each. Parent validation
was exact; CPU/MPS CE differences were2.733889967743153e-7 and2.62410942752922e-7.
Complete clipping histories agree across raw/summary/metrics. Integration001 also
rejected missing/sequence/threshold/coefficient/nonfinite history forgeries and
foreign-readout checkpoints. Paired full/cache/reset inference passed batches1/8;
cache remains196,608B per item in both arms. All717 runtime/acceptance bindings
unchanged,710 predecessors preserved. This is generated-fixture engineering
evidence, not a scientific quality result or seed/pilot/quality reservation.
The prerequisite plan's inherited kind string says generated_gradient_clip_worker_parent_prerequisite;
its explicit readout configs, source hashes, requests and accepted artifacts bind
the actual readout comparison. This descriptive label does not change execution.

Resume-retirement001 independently audited and retired exactly four obsolete
generated resume weights,1,118,317,844B, after real payload/cache integration closed
their dependencies.41,499 retained files verified unchanged; original sources,
configs, logs and receipts remain immutable. Absent weights resolve through the
later retirement manifest; never restore or rerun them. Actual post-recovery
free10,253,819,904B exceeded complete worker-pair9,814,671,360B including8GiBreserve.
Both fresh generated workers then completed. Their new raw/H200 weights total
1,118,323,284B and remain retained until pilot integration closes dependencies
and a new exact audit. Current free space is not scientific-panel readiness.

Next: minimal additive finite20-update resource-pilot orchestration and complete
THREE-fresh-pair fixed-LR design outside717 frozen sources. Re-audit actual worker
payload/cache, test finite contracts, close dependencies, fresh scientific census,
measure complete retained-output high-water including serialization overlap,
metadata and8GiBreserve, and take exact authorized recovery if necessary before
finite pilots and final quality freeze. CommonLR0.00125/equal zero LR-search,
original strict gates/no rescue/selection/partial panel. Retain final LayerNorm
and the adopted recipe; AdamW beta2 .95-versus-.999 remains deferred.
Preserve120 recoverable historical RAW archives, all H200 originals, storage004
and every selected/fresh byte. All earlier campaigns/fixtures remain closed.

Evidence: tokenizer-evidence/readout-centering-preparation-001/trainer-worker001.json,
trained-inference001.json, worker-integration001.json and resume-retirement001/completed.json.


## Finite resource pilots complete; quality source preparation next

Both20-update readout pilots passed independent parent acceptance and full/cache/
reset paired inference at batches1/8:20,480 training targets,8 H200 snapshots and
88 EMA tensors each,196,608B cache per item. Complete clipping histories were
checked; pilot observations are descriptive resource evidence, not quality or
LR selection. Independent final003 resource review passed all original time/RSS/
MPS/disk/output guards and the conservative1200-step projection. All729 campaign
sources remain unchanged, including717 accepted runtime/acceptance predecessors.

Integration001 stopped at import before numerical work or deletion because a
relative validation import had no campaign module. Original source/plan/log remain
closed. Additive campaign_v2 explicitly imports the tested worker validation;
separate integration002 passed finite six-trajectory contracts and actual worker
raw/H200/optimizer/EMA/RNG/clipping/full-validation/cache audits. Original723
bindings remain unchanged. Retention-boundary002 passed exact pilot-only eligibility
and protected-fresh rejection using new disposable inputs. No closed test reran. Final002 reviewer launch omitted two required MPS environment
flags and stopped before audit. Original source/plan/log remain closed; separate
final003 uses the prescribed environment, with no training/inference repeated.

Worker-retirement001 then retired exactly1,118,323,284B of obsolete generated
weights after dependency closure, with41,696 survivors unchanged. Original
receipts stay immutable; absent tensors resolve through its later exact manifest.
Storage013 preserved27 closed table/MLA/horizon selected/fresh RAWs in verified
lossless archives, reclaiming2,574,573,563B and verifying50,958 other
files unchanged; exact restoration passed. The copy-only probes separately measured
18,501,652B saving on a tied-table RAW and108,837,823B on an MLA RAW; neither
probe changed its original. All147 historical archived RAWs remain recoverable,
NOT retired; all H200 originals and storage004 remain intact.

The measured complete capacity allowance is12,381,585,408B: six retained fresh
raw/H200 pairs,256MiB metadata and8GiB reserve. Same-inode streaming publication
adds no full on-disk checkpoint copy; raw/H200 coexist within the bound. Fresh
census scanned50,526 files/4,493,124,543B with no collisions. Panel199630 pilot001
and evaluation003 are now used; fresh011/012/013 and bootstrap004 remain unused.
Final review observed free11,575,418,880B with1,118,335,248B retained pilots,
conditionally12,693,754,128B after future dependency closure and NEW exact audit.
This is not permission to delete pilots now or assume future capacity.

Next: additive research/readout_centering_quality_v1 outside729 frozen sources,
exactly three fresh pairs at commonLR0.00125, equal zero LR-search budgets, original
strict gates and no screens/selection/rescue/retry/partial panel. Verify actual
pilot payload/cache and six-trajectory contracts, close dependencies, independently
audit pilot retirement and remeasure complete capacity/unused seeds before final
quality freeze. Final LayerNorm versus affine RMS WITH bias at epsilon1e-5 and
512 parameters each; eight pre-branch RMSNorms unchanged. Retain incumbent LayerNorm
and adopted recipe pending final gates; AdamW beta2 .95-versus-.999 deferred.
No scientific quality result, quality freeze or active worker.

Evidence: pilot001-final003-review.json, pilot001-review.json, pilot-integration002.json,
worker-retirement001/completed.json, campaign-capacity001.json and
scientific-seed-census-001/result.json in tokenizer-evidence/readout-centering-preparation-001.
