"""CPU-only descriptive analysis of retained first-update momentum matrices."""
import fcntl
import os
import shutil
import time
import torch
from research.general_rsi_transfer_data_v1 import ROOT, read, put, sha
from research.general_rsi_training_v2 import read_checkpoint
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

OUT = ROOT/'research/tokenizer-evidence/general-rsi-momentum-spectrum-001'
PRIOR = ROOT/'research/tokenizer-evidence/general-rsi-reuse-full-model-001/attempt-01'


def spectrum(matrix):
    singular = torch.linalg.svdvals(matrix.double())
    assert torch.isfinite(singular).all() and singular[0] > 0
    relative = singular/singular[0]
    energy = singular.square()
    energy /= energy.sum()
    return dict(singular_values=singular.tolist(), relative_min=float(relative[-1]),
                stable_rank=float(1/energy[0]),
                ranks_above_relative={str(t): int((relative > t).sum()) for t in [1e-3, 1e-6, 1e-9]},
                dimensions_for_99pct_energy=int(torch.searchsorted(energy.cumsum(0), .99))+1)


def difference(a, b):
    delta = a.double()-b.double()
    return dict(changed=int((a != b).sum()), max_abs=float(delta.abs().max()),
                relative_frobenius=float(delta.norm()/a.double().norm()))


def main():
    cfg = read(OUT/'plan.json')
    assert sha(__file__) == cfg['source_sha256']
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not read(ROOT/'research_state/general-rsi-state.json').get('stop_requested', False)
        assert shutil.disk_usage(ROOT).free > 10*1024**3 + cfg['output_cap_bytes']
        with CpuPrerequisiteGuard(OUT/'guard.json', ROOT, timeout_seconds=120,
                                 rss_limit_bytes=4*1024**3, reserve_bytes=10*1024**3):
            start = time.monotonic()
            payloads = []
            for name in ['baseline', 'reuse']:
                receipt = read(PRIOR/f'first-failure-{name}-receipt.json')
                assert receipt == cfg['receipts'][name]
                payloads.append(read_checkpoint(receipt))
            a, b = payloads
            assert a['step'] == b['step'] == 1
            rows = []
            names = [n for n in a['children'][0]['state'] if n.startswith('composer.')]
            assert names == cfg['matrices'] and len(names) == 9
            gate = read(PRIOR/'first-update-parameter-comparison.json')
            for name in names:
                assert not read(ROOT/'research_state/general-rsi-state.json').get('stop_requested', False)
                x = a['children'][0]['state'][name]['momentum_buffer']
                y = b['children'][0]['state'][name]['momentum_buffer']
                assert x.dtype == y.dtype == torch.float32 and x.device.type == y.device.type == 'cpu'
                qx, qy = x.bfloat16().float(), y.bfloat16().float()
                row = dict(name=name, shape=list(x.shape), elements=x.numel(),
                    first_update_parameter_gate_passed=gate[name]['passed'],
                    first_update_parameter_max_abs_delta=gate[name]['max_abs_delta'],
                    fp32_pair_difference=difference(x, y), bf16_pair_difference=difference(qx, qy),
                    baseline_quantization_error=difference(x, qx), reuse_quantization_error=difference(y, qy),
                    baseline_spectrum=spectrum(x), reuse_spectrum=spectrum(y))
                put(OUT/(name+'.json'), row)
                rows.append(row)
                print(name, 'BF16 unequal', row['bf16_pair_difference']['changed'], '/', x.numel(),
                      'relative sigma_min', row['baseline_spectrum']['relative_min'], flush=True)
            put(OUT/'summary.json', dict(source_sha256=sha(__file__), plan_sha256=sha(OUT/'plan.json'),
                torch_version=torch.__version__, matrices=len(rows),
                total_elements=sum(x['elements'] for x in rows),
                fp32_unequal_elements=sum(x['fp32_pair_difference']['changed'] for x in rows),
                bf16_unequal_elements=sum(x['bf16_pair_difference']['changed'] for x in rows),
                cases=[{k: row[k] for k in ['name','shape','elements','first_update_parameter_gate_passed',
                    'first_update_parameter_max_abs_delta','fp32_pair_difference','bf16_pair_difference']} for row in rows],
                elapsed_seconds=time.monotonic()-start, new_training_updates=0, test_opened=False,
                limitations='Saved momentum is not the exact Nesterov input to Newton-Schulz. CPU BF16 casts are descriptive of these stored arrays, not replay of MPS optimizer arithmetic. Singular spectra do not establish causality or quality. One first-step paired case, all nine composer Muon matrices, no selection by result.'))


if __name__ == '__main__':
    main()
