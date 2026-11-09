# General-corpus execution pilot

The first new-data pilot passed:20 training updates each for native and raw-byte
input models, paired batches and identical initial body/readout weights. Total
guarded elapsed time25.612 seconds; peak process RSS1,054,359,552 bytes. This is
execution evidence, not a quality comparison or model adoption.

| Measurement | Native input table | Byte input composer |
|---|---:|---:|
| Parameters | 28,492,416 | 16,351,859 |
| Median update, last15 | .3602 s | .6477 s |
| Total20 updates | 7.586 s | 13.376 s |
| Peak observed MPS driver allocation | 2,493,300,736 B | 3,606,888,448 B |
| Saved inference weights | 113,996,635 B | 65,439,213 B |
| Max CPU/MPS logit difference | 9.54e-7 | 2.15e-6 |

The byte model was1.80x slower per update in this short run, despite fewer
parameters. Do not extrapolate a quality winner from the pilot losses: the
rates differ by the prior recipe, there are only20 updates, and no validation
loss was scored. The model bodies, vocabulary and training math are reused
from the previously qualified implementation; the new data adapter is isolated
from frozen Shakespeare code. The inherited unused no-output-bias provenance
inside the native model config is not a claim that this corpus is Shakespeare;
the new data manifest is bound separately and no old prior is applied.

Verified: all training gradients finite, global clipping stayed within its
fixed limit, all20 batch hashes matched between arms, validation sampler RNG
was unchanged, saved parameters round-tripped exactly, corrected FP32 CPU/MPS
logits passed atol3e-5/rtol3e-4. The validation chunk inventory covers100,416
within-document targets once. Test payloads were never opened. The scalar-loss
logging emitted a PyTorch detach warning; it did not fail or affect training.
Use explicit `.detach()` in the new long-run worker, preserving this executed
pilot source and evidence.

The complete preliminary four-screen/six-confirmation forecast is9.75 hours
of training at these short-run medians, or13.67 hours including a30% margin
and a provisional one-hour evaluation allowance. This is an estimate, not
measured end-to-end timing. It falls below the initial36-hour tranche ceiling.

Forecast incremental retained/overlap high water is2,672,755,286 bytes, plus
10 GiB reserve; current free space was13,414,297,600 bytes. Only4,124,074 bytes
remain above that conservative forecast and reserve. Re-measure after actual
restart serialization and evaluator timing, and perform authorized exact
cleanup if needed before admitting the full panel. No screen or confirmation
has started. Keep the complete requirement; do not launch only the first run
to hide a deficit.

Next executable action: qualify the minimal general-data worker's complete
checkpoint restart and scoring with a short bounded fixture, including
optimizer, schedule, EMA and sampler state. Then freeze the panel manifest,
roles, source hashes and measured complete cost. The existing five-minute
continuation is ACTIVE and now targets this chat. Other research heartbeats
remain paused. The older native Goal badge is still paused because the Goal
tool cannot resume it; scheduled continuation is the active mechanism.

Evidence: `research/tokenizer-evidence/general-rsi-pilot-001/{plan,attempt,guard,result,forecast}.json`
and per-arm step logs/results. Selected old weights and historical failed
experiments are unchanged. All pilot weight files remain retained locally.
