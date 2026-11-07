# Single readout projection: tiny component admitted only

Read TOKENIZER_READOUT_MIXTURE_RESULTS.md and post-readout-boundary-001/
evidence-review.md/source-design001.json. Old MoS panel CLOSED_NOT_ADOPTED: beats
native but fails matchedMoC comparison. No control fallback/rescue/scorepool/replay.

The source/design derivation is complete. Proposed nativeWz/4, matchedlinearWAz/4
and tanhW(tanh(Az))/4 preserve one shared untied biasfreeW and genuineCE. Single
A[256,256] adds65536params; new arms28557952elements/89tensors/45Muon44AdamW.
Preserve genuine nativechild/common88order/weights/CPUdraws, private A AFTERnative
construction and identical A for linear/tanh. NEWinitializer domain is explicit
in design001; no initializer/quality roles assigned. Linear map equivalence is
not optimizer equivalence. All ranks<=256; norm/derivative changes are not causes.

Implement ONLY new additive readout_projection_v1 pure functions and independent
math.fsum/stablelogsumexp forward+analyticgradient oracle. NEWdistinct literal cases,
N<=16/D<=8/V<=31/4096elements,CPU64/CPU32/MPS32; normalization/targetCE/discrimination/
signed/saturating/invalid cases and input/globalCPU-MPSRNG preservation. Fixed
FP64forward1e-11/grad1e-10;FP32atol4e-6/rtol4e-5. Pin/review source before first
execution;120sCPU/600sMPS/4GiBRSS/6GiBMPS/reserve8GiB/2MiBmetadata. No models,
RNGdraws/updates/corpus/tensorfiles/holdout/qualityseeds. Failure closes original
command; no retry/source/tolerance rescue. Preserve1127predecessor source bindings
(including closed existing-analysis source), without extending old tiny guards.

Success cannot admit actualD256/V50257states. Genuine own model/optimizer/config/
weights/cache/completeRAW-H200-resume/private samplers/EMA/clipping/exposure/scorer,
workers/pilots/full-quality gates remain unqualified. No new full model/panel/
budget/scientificseed/training admitted. LOCALarchitecture follow-up does not fill
broad exploration quota; carry proposed allocations forward. No active worker now.
