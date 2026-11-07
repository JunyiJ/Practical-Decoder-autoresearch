# New serialized proof admission needed after five-file source preparation

The new relu_squared_resume_v1 five-file implementation is pinned by
serialized-state-source002/sources.json:1189 bindings preserve1184 predecessors.
No runtime imports, models, RNG draws, corpus reads, checkpoint reads/writes or
serialization occurred. All new state/scorer behavior remains UNTESTED.
serialized-state-source001 is separately CLOSED_FAILED: the inline metadata
review used ast.literal_eval on512*1024 and stopped before window arithmetic or
publication. Its guard/failure receipt remain intact. Distinct review002 checked
the unchanged implementation with AST structure checks; no old fixture replay.

Next implement/pin/review NEW generated serialized-state/scorer proof and NEW
six-role census sources. New roles must cover initializer, derived privateungated,
train sampler, eval sampler, globalCPU and globalMPS, with no role reuse. The
data.config API now requires explicit keyword global_cpu_seed/global_mps_seed;
do not silently default globals to sampler roles. Source MPS RNG preflight uses
the version-pinned default generator initial_seed decoder transactionally and
restores the caller stream in finally. Include role/encoding rejection checks.

Use NEW literal token/sampler fixtures, independently retained uninterrupted4
versus2+2 CPU-archive-restored fresh-destination SAME-MPS trajectories, complete
RAW/H200/optimizer/EMA/sampler/globalRNG/LR/clipping/exposure exactness, independent
short generated per-target CE/logprob comparisons and new34278 literal-loss
math.fsum/window/block/byte controls. Generated losses and literal denominators
do not validate the public physical scorer or rank quality. Later real4/20
CPUANDMPS FULL34278 physical validation and cache/acceptance remain separate gates.

Important new source consumers to test:

- CPU archive moments preflight, then stage each non-step moment to its parameter
  device and retain CPU Adam steps; frozen optimizer loader preflights and deep
  clones SAME-backend tensors. Validate all state BEFORE destination mutation.
- Retained RAW ancestry now reads the hash-bound CPU archive and checks actual
  kind/model/config/initializer/final horizon/model/EMA. H200 producer requires
  liveRAW/EMA to equal that parent; H200 loader requires exported tensors to equal
  parentEMA plus persistent buffers. Test mismatched parent/liveEMA/export values
  without editing original archives. H200 remains inference-only.
- The frozen projection_validation_pair_v1 publisher has a projection-specific
  POLICY. New scoring.publish_pair_then_require has an OWN matched-FFN policy,
  fixed3e-5 gate,512KiB cap and exclusive fsync of BOTH finite caller-validated
  reports BEFORE pair rejection. No foreign report relabeling. Include NEW
  literal metadata equal/within/boundary/above/target/coverage/type/nonfinite/cap/
  provenance/overwrite/input-preservation checks before any acceptance use.
- Canonical.pt bounded exclusive.tmp/no-overwrite/weights_only/hash-before-after
  IO, source/runtime/data identities, transactional malformed complete state and
  report rejection, and caller CPU/MPS RNG restoration. Keep every partial output
  and original error; no source/tolerance/seed/fixture rescue after attempt.

Source-only scalar accounting (NOT observed torch.save sizes): learned FP32
113969664B, persistentRoPE65536B, model tensors114035200B, optimizer moments plus
44 CPU steps216929456B, oneEMA113969664B; RAW tensor subtotal444934320B before
RNG/metadata/pickle/alignment, H200 subtotal114035200B. RAW512MiB/H200128MiB caps
remain hard limits. Before numerical admission, qualify implemented serialization
counts/caps and the COMPLETE retained-output high-water for every required RAW/
H200/partial/metadata artifact. Include CPU parent decoding (up toRAW cap), CPU
live/saved EMA, optimizer staging AND deep clone in working peak and IO/time.
Publication aliases share one allocation; do not count speculative cleanup or
drop confirmations/controls. Preserve reserve8GiB plus extra2GiB headroom.

No new numeric roles, archive count/budget, census, models, serialization, worker,
pilot, scientific panel or quality launch is admitted by this handoff. Next is
concrete proof/census source+scope+fixed-predicate+complete-resource qualification;
standing human authorization covers execution once those gates actually pass.
