"""Finite-polynomial precision controls on saved arrays, not optimizer replay."""
import fcntl
import hashlib
import os
import shutil
import time
import numpy as np
import torch
from research.general_rsi_transfer_data_v1 import ROOT, read, put, sha
from research.general_rsi_training_v2 import read_checkpoint
from research import general_rsi_transfer_worker_v1 as checks
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

OUT = ROOT/'research/tokenizer-evidence/general-rsi-momentum-precision-001'


def polynomial(momentum, options, device, dtype):
    value = momentum.to(device=device, dtype=dtype).clone()
    transposed = momentum.shape[0] > momentum.shape[1]
    if transposed:
        value = value.T
    value.div_(value.norm().clamp(min=options['eps']))
    a, b, c = options['ns_coefficients']
    for _ in range(options['ns_steps']):
        gram = value @ value.T
        update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        value = torch.addmm(value, update, value, beta=a)
    return value.T if transposed else value


def relative(a, b):
    return float((a.double()-b.double()).norm()/a.double().norm())


def main():
    cfg = read(OUT/'plan.json')
    assert sha(__file__) == cfg['source_sha256']
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
    assert torch.__version__ == '2.14.1' and torch.backends.mps.is_available()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checks.check()
        assert shutil.disk_usage(ROOT).free > 10*1024**3+64*1024**2
        with CpuPrerequisiteGuard(OUT/'guard.json', ROOT, timeout_seconds=240,
                                 rss_limit_bytes=4*1024**3, reserve_bytes=10*1024**3):
            start = time.monotonic()
            payloads = {name: read_checkpoint(rec) for name, rec in cfg['receipts'].items()}
            results = []
            for name in cfg['matrices']:
                outputs, repeats, times = {}, {}, {}
                for label, device, dtype in [('cpu_fp64', 'cpu', torch.float64), ('mps_fp32', 'mps', torch.float32), ('mps_bf16', 'mps', torch.bfloat16)]:
                    for arm in ['baseline', 'reuse']:
                        checks.check()
                        child = payloads[arm]['children'][0]
                        options = next(g['options'] for g in child['groups'] if g['name'] == name)
                        assert options['ns_steps'] == 5 and tuple(options['ns_coefficients']) == (3.4445, -4.775, 2.0315)
                        x = child['state'][name]['momentum_buffer']
                        key = label+'_'+arm
                        times[key], repeats[key] = [], True
                        for rep in range(3):
                            torch.mps.synchronize()
                            t = time.monotonic()
                            y = polynomial(x, options, device, dtype)
                            torch.mps.synchronize()
                            times[key].append(time.monotonic()-t)
                            y = y.detach().cpu().to(torch.float64 if dtype == torch.float64 else torch.float32)
                            assert torch.isfinite(y).all()
                            if rep == 0:
                                outputs[key] = y
                            else:
                                repeats[key] &= torch.equal(y, outputs[key])
                path = OUT/(name+'-first-outputs.npz')
                with path.open('xb') as f:
                    np.savez(f, **{k: v.numpy() for k, v in outputs.items()})
                spreads = {label: relative(outputs[label+'_baseline'], outputs[label+'_reuse']) for label in ['cpu_fp64','mps_fp32','mps_bf16']}
                errors = {arm: relative(outputs['cpu_fp64_'+arm], outputs['mps_fp32_'+arm]) for arm in ['baseline','reuse']}
                row = dict(name=name, pair_relative_spreads=spreads, fp32_vs_fp64_relative_errors=errors,
                    bitwise_repeats=repeats, seconds=times, outputs_sha256=sha(path), output_bytes=path.stat().st_size,
                    options=options, precision_reduces_spread_tenfold=spreads['mps_fp32'] <= spreads['mps_bf16']/10)
                put(OUT/(name+'.json'), row)
                results.append(row)
                print(name, spreads, 'repeatable', all(repeats.values()), flush=True)
            selected = [r for r in results if r['name'] in cfg['failed_gate_matrices']]
            qualified = all(r['precision_reduces_spread_tenfold'] for r in selected) and all(all(r['bitwise_repeats'].values()) and max(r['fp32_vs_fp64_relative_errors'].values()) < 1e-4 for r in results)
            put(OUT/'summary.json', dict(source_sha256=sha(__file__), plan_sha256=sha(OUT/'plan.json'),
                qualifies_distinct_precision_hypothesis=qualified, results=results, elapsed_seconds=time.monotonic()-start,
                new_training_updates=0, test_opened=False,
                limitation='Controlled finite-polynomial maps of saved momentum, not exact Nesterov-input replay or a model/optimizer qualification. CPU FP64 is a finite-polynomial reference, not an exact-polar target. No quality or whole-update speed claim. Closed reuse v1 remains failed.'))


if __name__ == '__main__':
    main()
