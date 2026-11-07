# Closed readout result; single-projection applicability review

The original MoS panel is CLOSED_NOT_ADOPTED. Read TOKENIZER_READOUT_MIXTURE_RESULTS.md,
readout-quality001-closure001/completed.json/review.json and the new existing-analysis001.
All18accepted/all9serving/independentstatistics passed. MoS−native mean−.033083090568
is insufficient: MoS−matchedMoC−.001065236022, one positive seed and all three MoC
upper bounds positive. Alljoint gates fail. Keepnative; no MoC fallback, moreLR/K/
initializer/horizon rescue, oldscorepool or oldcommand replay. Allweights protected.

Both mixtures beat native on every RAW seed too, but their means are smaller;
MoS−MoC retains one positive seed. H200 improves mixtures while worsening native;
mixtures have more clipped steps. These complete-panel metadata facts, five online
points and independently selected LRs do not diagnose averaging, clipping, rank,
amplitude or optimizer conditioning. RAW is not a replacement selector.

Re-read Yangetal arXiv1711.03953v1 sections2.3–2.5 and section3/Table4 at
https://arxiv.org/html/1711.03953v1. Its tanh context maps and matched feature-mixture
baseline motivate a discriminator. MoC retains the rank bound, and Table4 shows
mixed generalization across PTB/WT2 despite added features. RNN/word data,
regularization and optimization differ locally. Published gains are not local
gains; neither an algebraic rank bound nor added expressivity diagnoses a true
conditional-rank deficiency. This boundary claims actual retrieval of those sections.

Rank structural AND recipe alternatives from current evidence:

1. Select bounded single-projection source/design and tiny correctness ONLY.
   Test whether single tanh(Az) differs usefully from equally parameterized linearAz,
   with native as a third control. Remove the old two-map mixture/router, preserve
   one untied vocabulary projection, distinguish nonlinear transformation from
   linear factorization/conditioning. This is a NEWreduced structure, not an old
   MoC fallback or closedK2replay. Classify as LOCALreadout-architecture follow-up,
   in proposed incumbent-improvement allocation, not broad20%exploration quota.
2. Defer scale/conditioning/averaging recipe changes. New averaging differences and
   analytic initial norm reduction suggest questions, not causes. Prior final-scale
   and final-centering findings are contrary. No EMA/horizon/LR/clipping rescue or
   new optimizer instrumentation is admitted. Matchedlinear is a direct first control.
3. Defer broad depth/shared-expert/context/table proposals. Body depth comparisons
   also change width/tables/readout/headgeometry; completed depth-scaling applicability
   rejected immediate transfer to localMuon. MoE/gate/vocabulary/dropout and norm/
   parallel/branchscale negative results remain contrary/unpooled. New output results
   supply no targeted shared-expert/depth/context mechanism. Carry broad/interaction
   allocation forward rather than choose arbitrary novelty.
4. Reject immediate batch/optimizer diagnostic extensions and closed-family replays.
   Inconclusive gradient ratios and failed MPS response proof are not batch optima
   or training defects. No new scientific need justifies those instruments.

Actual native/body/readout/optimizer sources were inspected. Source-design001.json
is complete: nativeWz/4, linearWAz/4, tanhW(tanh(Az))/4. Both new arms add65536params,
becoming28557952elements/89tensors/45Muon44AdamW vs28492416/88/44/44 native. Native
TRUEcfg/common88constructor/CPUdraws exact; shared privately initialized A after
construction, new audited roles before model use. Both A groups explicitMuoneta/2
RMSmatch, commonAdamW.9,.999eps1e-6/independentdecay.00006; not claimed optimal.

LinearWA can be absorbed into W as a represented map, but not as optimizer dynamics.
All logits retain rank<=256. Var(Aij)=1/(3D) gives E||Az||²=||z||²/3 for independent
fixedz; ||tanh(Az)||<=||Az|| pointwise. Nonlinear operation changes amplitude and
derivatives jointly; matchedlinear does not identify a unique saturation/amplitude
cause. No local bottleneck is claimed.

Future prediction IF all gates qualify: tanh projection improves completeH200nativeCE
versus BOTHnative and matchedlinear under equally bounded finite tuning. Otherwise
native stays, no linear fallback. Positive evidence supports separate consumer review;
negative rejects this smaller exact recipe without attributing a cause. New fresh
roles/complete retained-output-time budget/correctness/resources BEFORE any future
freeze. No full model, new quality panel/role/budget admitted now.

Immediate executable work: smallest additive readout_projection_v1 functional
native/linear/tanh logits plus independent literal math.fsum/stablelogsumexp forward
and analyticbackward oracle. Pin/review new source before first execution. TinyN<=16,
D<=8,V<=31/4096elements,CPU64/CPU32/MPS32; fixed tolerances/source-design001. No models,
RNGdraws/optimizersteps/corpus/checkpoints/qualityseeds/holdout;120sCPU/600sMPS/4GiBRSS/
6GiBMPS/reserve8GiB/2MiBmetadata. Failure closes its command; no retry/source/tolerance
rescue or oldproof replay. Proposed70/20/10 remains a scheduling proposal; preparation
is not numerical exploration. No numerical worker active after original closure.
