"""Recompute the precomputed-inference report from saved measurements only."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'research/tokenizer-evidence/byte-composition-boundary-001/precomputed-inference005'


def read(name): return json.loads((OUT/name).read_text())


def main():
    acceptance=read('acceptance.json');plan=read('plan.json')
    assert acceptance['accepted'] and acceptance['no_training'] and not acceptance['holdout_scored']
    assert acceptance['plan_sha256']==hashlib.sha256((OUT/'plan.json').read_bytes()).hexdigest()
    artifact=acceptance['artifact'];path=Path(artifact['path'])
    assert path.stat().st_size==artifact['size']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==artifact['sha256']
    for name in acceptance['completed']:
        g=read(name+'-guard.json');assert g['status']=='finished' and g['reason'] is None
    lines=['# Precomputed byte embeddings: verified inference and measured tradeoffs','',
        'The v3 snapshot materializes all 50,257 byte-composed inputs and owns independent copies of the existing transformer and readout. It retains neither composer nor codebook. The original selected H200 checkpoint and default consumer are unchanged. This is an inference implementation result, not a new quality experiment.','',
        '## Verification','',
        '| Backend | Vocabulary rows checked | Validation targets | Largest logit error | Largest CE error | Changed validation argmax |',
        '|---|---:|---:|---:|---:|---:|']
    for backend in plan['backends']:
        v=read(f'verify-{backend}-result.json')
        assert v['completed'] and v['source_weights_rng_unchanged'] and v['greedy_exact'] and v['reset_exact']
        assert v['invalid_inputs_rejected'] and v['mutation_rejected']
        rows=v['all_vocabulary_checks'];assert rows[0]['lo']==0 and rows[-1]['hi']==50257
        assert all(a['hi']==b['lo'] for a,b in zip(rows,rows[1:]))
        assert all(r['max_ratio']<=1 for r in rows)
        validation=v['validation'];windows=validation['windows']
        assert [w['start'] for w in windows]==list(range(0,34278,128))
        assert sum(w['targets'] for w in windows)==validation['targets']==34278
        assert all(w['max_ratio']<=1 for w in windows)
        assert validation['max_ce_error']<=plan['max_ce_atol'] and validation['mean_absolute_ce_error']<=plan['mean_ce_atol']
        assert len(v['cache_checks'])==132
        assert all(r[k]['max_ratio']<=1 for r in v['cache_checks'] for k in ['vs_current','vs_full'])
        lines.append(f"| {backend.upper()} | 50,257 | 34,278 | {max(w['max_abs'] for w in windows):.9g} | {validation['max_ce_error']:.9g} | {validation['argmax_changes']} |")
    lines += ['', 'The CPU-built exported table was reloaded on both CPU and MPS. Verification includes all vocabulary rows against independent 1,024-row composition (construction used 256-row chunks), every original validation target, B1/B8 cached prefill96 plus decode32, exact reset/replay, and 16 greedy autoregressive steps. Invalid IDs, dtype, cache position, context overflow, training, and table mutation are rejected. Source weights, gradients and CPU/MPS RNG are unchanged.', '',
        'Tolerances were fixed before execution: logits/embeddings atol 3e-5 and rtol 3e-4; maximum per-target CE difference 3e-4 and mean absolute CE difference 3e-5. All comparisons are against the same backend’s current public no-grad inference. The separate RMSNorm default-epsilon repair is not included or validated by this work.', '',
        '## Latency', '',
        'Four CPU threads, FP32, two warmups and seven measured repeats; alternating paired order and MPS synchronization. Reset, loading and precomputation are excluded from steady-state latency. Decode uses the same fixed 32 continuation tokens in both arms; greedy generation is verified separately. Both paths produce the full vocabulary logits. Raw samples remain in the result JSON files. CPU latency comes from the completed sandboxed CPU pass; GPU latency and isolated memory use approved host execution because MPS is unavailable in the sandbox. Each speedup compares two arms in the same process/environment.', '',
        '| Backend | Batch | Workload | Current median ms | Precomputed median ms | Speedup | Precomputed min–max ms |',
        '|---|---:|---|---:|---:|---:|---:|']
    summary=[]
    for backend in plan['backends']:
        v=read(f'verify-{backend}-result.json')
        for row in v['timings']:
            a,b=row['seconds'];assert len(a)==len(b)==7 and min(a+b)>0
            ma,mb=statistics.median(a),statistics.median(b)
            assert [ma,mb]==row['median_seconds']
            summary.append(dict(backend=backend,batch=row['batch'],workload=row['workload'],current_ms=ma*1000,
                                precomputed_ms=mb*1000,speedup=ma/mb))
            lines.append(f"| {backend.upper()} | {row['batch']} | {row['workload']} | {ma*1000:.3f} | {mb*1000:.3f} | {ma/mb:.3f}× | {min(b)*1000:.3f}–{max(b)*1000:.3f} |")
    lines += ['', '## Conversion and storage', '']
    for backend in plan['backends']:
        v=read(f'verify-{backend}-result.json')
        lines.append(f"- {backend.upper()} conversion: {v['conversion_seconds']:.3f} seconds; conversion-process peak RSS {v['conversion_memory']['process_peak_rss_bytes']/1024**2:.2f} MiB (includes original loading and simultaneous source/snapshot).")
    source_size=plan['checkpoint']['size']
    lines += [f"- Saved H200: {source_size:,} bytes ({source_size/1024**2:.2f} MiB). Precomputed export: {artifact['size']:,} bytes ({artifact['size']/1024**2:.2f} MiB); increase {(artifact['size']-source_size)/1024**2:.2f} MiB ({artifact['size']/source_size:.3f}×).", '',
        '## Isolated memory', '',
        'Each backend/arm runs in a fresh process. Tensor storage counts unique live parameter and buffer storage; it is the precise architectural comparison. RSS also includes Python, libraries and allocator retention. MPS current allocation and driver allocation are point observations, not peaks. Process peak RSS includes loading and warmups and must not be interpreted as an isolated inference peak. CPU and MPS share physical memory, so these measures must not be summed.', '',
        '| Backend | Arm | Model tensor MiB | Loaded RSS MiB | Loaded MPS current MiB | Loaded MPS driver MiB | Process peak RSS MiB |',
        '|---|---|---:|---:|---:|---:|---:|']
    memory=[]
    for backend in plan['backends']:
        for arm in ['current','precomputed']:
            r=read(f'memory-{arm}-{backend}-result.json');m=r['loaded_memory']
            fmt=lambda x:'—' if x is None else f'{x/1024**2:.2f}'
            lines.append(f"| {backend.upper()} | {arm} | {r['model_tensor_bytes']/1024**2:.2f} | {fmt(m['current_rss_bytes'])} | {fmt(m['mps_current_bytes'])} | {fmt(m['mps_driver_bytes'])} | {fmt(r['final_memory']['process_peak_rss_bytes'])} |")
            memory.append(dict(backend=backend,arm=arm,model_tensor_bytes=r['model_tensor_bytes'],loaded=m,observations=r['observations']))
    lines += ['', 'Prefill/decode observations with live logits and idle KV caches are retained in each memory result. KV storage is unchanged: 196,608 bytes at B1 and 1,572,864 bytes at B8. The largest B8 prefill output itself contains 154,389,504 bytes; embedding precomputation does not remove that output cost.', '',
        'The dense table is 51,463,168 bytes. It replaces 2,900,940 composer-weight bytes plus 52,669,336 codebook-buffer bytes in the current implementation. This predicts a 4,107,108-byte resident tensor reduction, despite the larger export. A future compact-codebook implementation could change this memory tradeoff.', '',
        '## Use', '',
        'Run from the repository root with the project environment. The caller should own the shared research lock and appropriate resource guard, as for the current consumer.', '',
        '```python', 'import json', 'from pathlib import Path',
        'from research.byte_precomputed_inference_v3 import load, Session', '',
        'evidence = Path("research/tokenizer-evidence/byte-composition-boundary-001/precomputed-inference005")',
        'plan = json.loads((evidence / "plan.json").read_text())',
        'receipt = json.loads((evidence / "artifact.json").read_text())',
        'model = load(receipt, plan["checkpoint"], "mps")',
        '# token_ids: int64 MPS tensor, batch 1 or 8, context at most 128',
        'logits = model(token_ids)',
        'session = Session(model, batch=1)',
        'prefix_logits = session.logits(token_ids[:, :96], start_pos=0)',
        '# Append contiguous chunks; call session.reset() before a new sequence.', '```', '',
        'The artifact is local under `research_runs/byte-precomputed-inference003/precomputed.pt`; Git retains sources, receipts and measurements. Loading verifies the full artifact hash and expected source-checkpoint identity. Default checkpoint consumers and the training model remain unchanged.', '',
        '## Preserved failure and scope', '',
        'Attempt001 stopped on the inherited normalization contract because v1 disabled parameter requires_grad flags. Attempt002 retained the original flags and passed CPU numerical/cache/generation checks, then detected a loader RNG side effect: torch.manual_seed also seeded MPS despite the CPU-only RNG scope. Attempt003 uses the CPU default generator directly and retains no-grad execution at public APIs. Both failed sources, plans, guards, artifacts and tracebacks remain preserved. Attempt003 completed all CPU verification and timings, then its MPS monitor crashed in allocator initialization before model execution. Attempt004 exposed the underlying sandbox restriction on the first MPS allocation: is_available=False in sandbox, True with successful synchronized allocation on the host (macOS26.3). Attempt005 imports the completed CPU receipts unchanged, explicitly checks availability, initializes MPS on the main thread, and runs the remaining workers through approved host execution. No model math, tolerance or case was relaxed. These measurements are one local machine/runtime/checkpoint benchmark, not a CUDA, quantized, long-context or production-throughput claim.', '']
    cpu_a=read('memory-current-cpu-result.json')['model_tensor_bytes']
    cpu_b=read('memory-precomputed-cpu-result.json')['model_tensor_bytes']
    assert cpu_a-cpu_b==4107108
    (OUT/'report.md').write_text('\n'.join(lines))
    (OUT/'summary.json').write_text(json.dumps(dict(timings=summary,memory=memory,
        source_checkpoint_bytes=source_size,precomputed_checkpoint_bytes=artifact['size'],
        tensor_saving_bytes=cpu_a-cpu_b,accepted=True),indent=2)+'\n')
    print(OUT/'report.md')


if __name__=='__main__': main()
