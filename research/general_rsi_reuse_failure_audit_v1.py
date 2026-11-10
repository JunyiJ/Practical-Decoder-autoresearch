"""Read-only localization of retained first-step failure; no optimizer execution."""
import inspect
import torch
import torch.optim._muon as muon
from research.general_rsi_transfer_data_v1 import ROOT, read, put, sha
from research.general_rsi_training_v2 import read_checkpoint
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

OUT = ROOT/'research/tokenizer-evidence/general-rsi-reuse-full-model-001'


def stats(a, b):
    delta = (a.double()-b.double()).abs()
    return dict(elements=a.numel(), changed=int((delta != 0).sum()), max_abs_delta=float(delta.max()))


def main():
    with CpuPrerequisiteGuard(OUT/'failure-audit-guard.json', ROOT, timeout_seconds=60,
                             rss_limit_bytes=4*1024**3, reserve_bytes=10*1024**3):
        dest = OUT/'attempt-01'
        a = read_checkpoint(read(dest/'first-failure-baseline-receipt.json'))
        b = read_checkpoint(read(dest/'first-failure-reuse-receipt.json'))
        assert a['step'] == b['step'] == 1 and a['history'] == b['history']
        assert a['immutable_buffers'] == b['immutable_buffers'] and a['schedule'] == b['schedule']
        assert all(torch.equal(a['samplers'][n], b['samplers'][n]) for n in a['samplers'])
        parameter_deltas = {n: stats(p, b['parameters'][n]) for n, p in a['parameters'].items()}
        state_deltas = {}
        for ac, bc in zip(a['children'], b['children'], strict=True):
            assert ac['kind'] == bc['kind'] and ac['groups'] == bc['groups']
            for n, fields in ac['state'].items():
                state_deltas[n] = {k: stats(v, bc['state'][n][k]) for k, v in fields.items()}
        source = inspect.getsource(muon._zeropower_via_newtonschulz)
        source += '\n'+inspect.getsource(muon._single_tensor_muon)
        (OUT/'torch-muon-runtime-excerpt.py.txt').write_text(source)
        result = dict(audit_source_sha256=sha(__file__), torch_version=torch.__version__,
            runtime_path=inspect.getfile(muon), runtime_file_sha256=sha(inspect.getfile(muon)),
            runtime_excerpt_sha256=sha(OUT/'torch-muon-runtime-excerpt.py.txt'),
            parameter_deltas=parameter_deltas, optimizer_state_deltas=state_deltas,
            changed_parameters=[n for n, v in parameter_deltas.items() if v['changed']],
            noncomposer_parameters_bitwise=all(v['changed'] == 0 for n, v in parameter_deltas.items() if not n.startswith('composer.')),
            sampler_history_clip_and_group_options_equal=True,
            inference='Differences are confined to composer parameters. The bound torch Muon implementation casts its update to BF16 before normalization and Newton-Schulz iterations. This is consistent with the historical optimizer-sensitivity finding; this audit does not rerun the optimizer or prove which individual rounding operation caused amplification.',
            next_action='Close reuse v1 as a failed drop-in engineering candidate. Before any different numerical recipe, use retained momentum/parameter evidence and primary optimizer analysis to decide whether a distinct precision-controlled hypothesis is justified. No retry, tolerance change, quality panel or production integration is admitted.',
            test_opened=False, new_training_updates=0)
        put(OUT/'failure-localization.json', result)
        print(result['changed_parameters'])
        print('Noncomposer bitwise', result['noncomposer_parameters_bitwise'])


if __name__ == '__main__':
    main()
