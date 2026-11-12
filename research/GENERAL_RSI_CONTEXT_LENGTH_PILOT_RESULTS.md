# Context128/context256 integration and pilot results

The isolated native context path passed its new integration and execution checks.
All 24 paired updates per arm were finite. A complete context256 checkpoint
roundtrip and next-update replay matched exactly, including optimizer, sampler,
RNG, clock and history. Wrong-context restoration failed before mutation.
This qualifies the bounded execution path. It does not establish a quality gain.

## Implementation and direct comparison

`general_rsi_context_length_v1/` gives native context128/context256 distinct
identities, configurations, causal masks and RoPE buffers. Its separate runtime
namespace reuses the existing model sources and supplies a context-aware contract.
Historical wrappers, fixed128 contracts and source snapshots are unchanged.
The pilot supports uncached logits, with at most1,024 input tokens per call.

Both lengths retain28,492,416 parameters and88 parameter tensors. The new128
initializer, parameter ordering, logits, one optimizer update and complete child
optimizer states matched the original128 path exactly. The256 initializer also
matched all88 parameter tensors. Its short-input logits and the first128 positions
under a changed future tail had zero observed maximum difference. Frozen
engineering tolerances for those two checks were absolute1e-6/relative1e-5.
They are unrelated to the historical composer numerical qualification.

The new optimizer source is AST-equivalent to the original after only import and
class-name substitution. Muon/AdamW grouping, clipping1, independent decay0.00006
and LR0.0025 are unchanged. The sampler consumes two512-token draws per update,
reshaping the same1,024 inputs and targets to8×128 or4×256. All25 fixture draws
and all24 pilot draws matched in flattened order and sampler progression.

## Measured pilot

| Context | Updates | Targets | Training seconds | Mean seconds/update |
|---|---:|---:|---:|---:|
| 128 | 24 | 24,576 | 8.462507377 | 0.352604474 |
| 256 | 24 | 24,576 | 8.362129587 | 0.348422066 |

The256 run then made one reference update and one restored replay, for50 total
pilot updates and51,200 training targets. The replay used the reference update's
measured duration to check exact clock/history recovery. It does not claim that
independent live-clock runs have identical durations. Cooldown and EMA did not
activate in this short pilot and remain outside this qualification's scope.

The first integration256 update took0.605769458seconds. The warmer pilot means
above are short-run measurements under serial execution, not sustained throughput
estimates or evidence of a256 speed advantage. First/last losses and every update
record remain available; they were not used for quality selection. Validation
and test data were not opened by these new fixtures or the pilot.

The pilot took24.660790208 guarded seconds. Peak RSS was1,299,709,952bytes and
peak observed MPS driver allocation2,493,300,736bytes. Retained payloads total
776,020,777bytes: one128 inference file and two256 restart/reference files.
The step24 restart is331,011,687bytes, and write/read/hash verification took
1.081173416seconds. The complete admission reserved15,099,494,400 free bytes,
including retained output ceilings, serialization overlap,10GiB reserve and2GiB
extra headroom. No capacity cleanup was needed for this pilot.

## Preserved harness failure and repair

`general-rsi-context-length-fixture-001` stopped after the first old128 update.
The audit digest requires CPU tensors; the harness supplied the optimizer's MPS
state. No model comparison completed. Its source snapshot, plan, guard and
traceback remain unchanged. The first harness did not emit that update's scalar
loss before the exception; none was reconstructed or substituted.

Fixture002 clones state to CPU before hashing and durably records each completed
update. It preserves the same seeds, model source, checks and tolerances. It passed
all three updates. All attempts together used33.603963915 guarded seconds and54
optimizer updates, including the one update before the original audit exception.
This was an implementation/harness failure, not a negative quality result.

## Next executable action

Implement only the remaining changed paths described in
`GENERAL_RSI_CONTEXT_LENGTH_QUALITY_INTEGRATION_NEXT.md`: context-aware endpoint
configuration, active time-EMA recovery, identical-target evaluation and fixed
uncached generation. Preserve the completed pilot and do not rerun it. No quality
panel or new model is adopted. No worker remains active.

Original evidence: `tokenizer-evidence/general-rsi-context-length-fixture-001/`,
`general-rsi-context-length-fixture-002/`, and `general-rsi-context-length-pilot-001/`.
The pilot plan SHA256 is
`d6479a624c0506f4aa0e050b6571ade5b25b78600dcf615865fe011da40367a1`.
Its completion review verifies all5,214 source bindings and all retained receipts.
