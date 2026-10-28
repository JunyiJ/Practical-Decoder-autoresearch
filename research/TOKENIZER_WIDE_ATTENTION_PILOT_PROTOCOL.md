# Finite width512 direct GQA/MLA feasibility pilots

Prospective October28,2026, following explicit human steering to wider models and
fresh direct MLA/GQA comparison. Two serial20-update pilots, GQA then MLA, width512,
seed3391, LR0.00125. No width/arm/rate selection from pilot CE. No quality claim.
The full quality comparison still requires a separate tested executable freeze.

Both use the exact new wider tokenizer policy, four dense SwiGLU21D/8 layers,
head32, RoPE, RMSNorm blocks/final affine LayerNorm, legacy untied GPT-2 tables,
base64 Muon/AdamW width scaling, attention multiplier0.5, decay0.00006,
batch8/context128 and paired512 sampler. GQA has16 query and8 KV heads. MLA has
16 heads, query/KV latent rank128, and shared rotary component32. MLA score
normalization uses sqrt64 (content plus rotary) versus GQA sqrt32; both retain
the0.5 multiplier. No latent-only cache claim. No hybrid or KDA in this pilot.

Cooldown starts10 and ends20 at0.1 multiplier; H200start10/interval5 yields3
snapshots. Run the actual trainer, retain raw/H200 separately and verify load,
tokenizer/config lineage,20updates,3snapshots, parameter counts and full/cached
parity at batches1/8 using first12 training tokens in3/4/5 chunks. No test loss.
All raw logs, requests, results, checkpoints and failures are retained. Current
CPU continuation fixture uses memory serialization; these pilots must additionally
validate real disk serialization and H200 export/load at the wider geometry.

Prospective parameter counts: GQA62,877,696; MLA62,424,192. Actual allocated cache
expected per batch item/context128: GQA1,048,576 bytes, MLA327,680 bytes, including
shared rotary key. Confirm these measurements; they are not a quality result.
For the subsequent direct quality round, proposed memory adoption requires MLA
neutrality versus fresh equally tuned GQA and at least60% actual allocated KV
reduction in every seed/batch. The +0.01-nat neutrality margin is an agent-chosen
convention; full mean/every-seed/all-conditional-upper-bound requirements remain.
No decode-speed win is required. Final quality/timing protocol will freeze all
selection/order details before any quality launch; no pilot CE can alter them.

## Resource and stopping contract

One strict MPS worker, four CPU threads, no fallback, global/run locks inherited
by the child.600seconds per pilot, sampled4GiB RSS/6GiB driver watchdog and8GiB
free-disk reserve. Full prospective panel is two retained pilots plus12quality
trials (3LR screens and3fresh seeds per family). Bound each GQA trial at
1,737,836,135 bytes and MLA at1,725,319,424 bytes:24bytes/parameter conservative
tensor accounting plus2MiB, then15% headroom. Full allocation including4MiB
metadata and reserve is32,836,217,809 bytes. Before each dispatch recheck all
remaining pilots and every quality trial. Measure actual pilot costs and review
projection before quality; do not lower reserve or silently retry a guard.

Freeze source/runtime/corpus/recipe/protocol/requests before launch from a clean
commit. Exclusive run directory; no reentry after running, stopped, failed or
completed state. Monitor free disk/output every2seconds and stop/terminate on
failure or STOP. Preserve failure evidence; no replacement seed or automatic retry.
Width1024 full-panel projection exceeds current free disk; it is not in this pilot.
Historical controllers and stop markers stay closed. No additional cleanup.

Commands: python -m src.research.tokenizer_wide_pilot prepare|run --output <new-run>.
Only after CPU tests and clean source/plan commits. No quality controller dispatch
is implied by successful pilots. All broader directions remain eligible later.
