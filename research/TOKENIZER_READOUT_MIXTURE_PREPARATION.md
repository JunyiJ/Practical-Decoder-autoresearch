# Native readout probability-mixture preparation

Current status: exact algebra/shape/resource design complete; **only the tiny
generated component is admitted**. No component numerical proof, full model,
optimizer/transport integration, worker/pilot or scientific panel has run.
Keep the unchanged native dense MLA incumbent. All1084 prior sources and all
closed studies/failures/STOPs/checkpoints remain immutable and protected.

Read post-gate-boundary-001/evidence-review002.md, readout-mixture-direction001.json
and exact readout-mixture-design001.json. Yangetal arXiv1711.03953v1 was actually
read (§2.1–2.5,§3Tables1–2/4–5). Its RNN word-level results motivate a controlled
question, not a local softmax-rank diagnosis or promised quality gain.

Three prospective arms: unchanged native; K2 mixture of contexts; K2 mixture of
separately normalized vocabulary probabilities. All use one untied native
W[50257,256], the unchanged native backbone/finalLN and readout divisor4.
Both mixtures use h_k=tanh(A_k z) and pi=softmax(Rz). A has two256x256 matrices;
R is2x256. Standard independent uniform context initialization uses a private,
later-audited CPU generator after exact native construction. R starts at zero
without RNG; no identity/antithetic or initialization search. Matched mixtures
share the same complete initialization. No initializer/quality roles used now.

Native remains28,492,416parameters/88tensors/44Muon44AdamW. Each mixture has
28,624,000parameters/91tensors/46Muon45AdamW. The two context maps use explicit
matrixMuon eta/2 atD256 with match_rms_adamw; R uses nominal-LR AdamW with the
incumbent beta/epsilon and independent decay. This is a declared policy, not
an asserted optimum or paper reproduction. Own model/config/loss/checkpoint/
scorer contracts are mandatory; old validators or logits APIs cannot be spoofed.

MoS training uses per-component target CE and stable target logsumexp, avoiding
full mixture distributions during training. Full normalized distributions remain
required for inference. OneB8x128x50257FP32 array costs205,852,672bytes; autograd
and temporaries can dominate. Tiny proofs cannot qualify the full4GiBRSS/6GiBMPS
limits. Future full-state/model/pilot resource gates are mandatory.

The conservative complete24-pair retention envelope (six prerequisites plus
18quality pairs;560/640/640MiB caps,metadata/sidecars/reserve/external/entry) is
26,845,642,752bytes, recovery target28,993,126,400. A new nonblocking external-lock
capacity pass observed66,994,397,184bytes free, target covered, zero deletions or
inventory needed. Caps are unqualified projections, not serialization acceptance
or a frozen budget. No old payload or future deletion credit. Retain every output.

Next implement the smallest additive pure-functional component and independent
literal scalar forward/backward oracle outside all frozen namespaces. Pin sources
before first numerical execution. Generated scope K2/N<=16/D<=8/V<=31,<=4096elements
per tensor,zero models/optimizer/corpus/checkpoint/qualityseeds/RNGdraws. Check CPU
FP64 and CPU/MPS FP32 NLL/log-prob normalization and gradients for z,W,A,R,
identity/permutation/extreme cases, MoS-versus-MoC discrimination, rejection and
input/global-RNG invariance under the exact design's frozen tolerances/resources.
Failure closes the command; no retry/rescue/tolerance changes. Only afterward
consider full-model integration. Preparation is not numerical exploration.
