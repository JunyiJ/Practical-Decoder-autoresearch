"""Once-only supplied-logit finite proof. No model, corpus, seed or optimizer.

The scalar oracle consumes declared literals, never tested outputs. This file
must first pass its separate AST/literal source gate. All reports precede gates.
"""
import contextlib
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
R = Path(__file__).with_suffix('')
GATE = R.with_name('top-objective-proof-source001')
BACKENDS = ('CPU64', 'CPU32', 'MPS32')
SPECIAL = {'pn_sparse_layout': ('CPU32',), 'pn_device_meta': ('CPU32',),
           'pn_cpu_autocast_active': ('CPU64', 'CPU32'),
           'pn_mps_autocast_active': ('MPS32',)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    data = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    retained = sum(p.stat().st_size for root in (R, GATE)
                   for p in root.glob('*') if p.is_file())
    if retained + len(data) + 65536 > 2097152:
        raise RuntimeError('Whole two-root JSON cap including guard/STOP headroom')
    with (R / name).open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def represented(value, backend):
    # Independent IEEE binary32 rounding; no tensor or tested intermediate.
    return value if backend == 'CPU64' else struct.unpack('f', struct.pack('f', value))[0]


def scalar_oracle(fixture, backend):
    rows = [row for batch in fixture['logits'] for row in batch]
    count = len(rows)
    losses, gradients, represented_rows, represented_targets = [], [], [], []
    for declared_logits, declared_targets in zip(rows, fixture['targets']):
        z = [represented(x, backend) for x in declared_logits]
        pairs = [(token, represented(mass, backend)) for token, mass in declared_targets]
        pivot = max(z)
        shifted = [math.exp(x - pivot) for x in z]
        normalizer = math.fsum(shifted)
        log_sum = math.log(normalizer)
        losses.append(-math.fsum(w * ((z[token] - pivot) - log_sum)
                                 for token, w in pairs))
        mass = math.fsum(w for _, w in pairs)
        by_id = dict(pairs)
        gradients.append([(mass * e / normalizer - by_id.get(i, 0.)) / count
                          for i, e in enumerate(shifted)])
        represented_rows.append(z)
        represented_targets.append(pairs)
    return dict(loss=math.fsum(losses) / count, gradient=gradients,
                logits=represented_rows, targets=represented_targets,
                row_mass_sums=[math.fsum(w for _, w in row) for row in represented_targets])


def close(actual, expected, atol, rtol):
    return abs(actual - expected) <= atol + rtol * abs(expected)


def tensor_values(tensor):
    # CPU transfer MUST precede FP64 reporting conversion, including on MPS.
    return tensor.detach().to('cpu').to(dtype=torch.float64).tolist()


def snapshot(value):
    if type(value) is torch.Tensor:
        metadata = (str(value.dtype), str(value.device), str(value.layout),
                    tuple(value.shape), bool(value.requires_grad))
        if value.device.type == 'meta':
            content = 'meta-no-storage'
        elif value.layout == torch.sparse_coo:
            sparse = value.detach().coalesce().cpu()
            content = (snapshot(sparse.indices()), snapshot(sparse.values()))
        else:
            content = snapshot(value.detach().cpu().tolist())
        grad = snapshot(value.grad) if value.is_leaf and value.grad is not None else None
        return ('tensor', metadata, content, grad)
    if type(value) is float:
        return ('float', value.hex())  # stable NaN/inf preservation too
    if type(value) in (tuple, list):
        return (type(value).__name__, tuple(snapshot(x) for x in value))
    return (type(value).__name__, value)


def rng_snapshot():
    return torch.get_rng_state().clone(), torch.mps.get_rng_state().clone()


def rng_equal(saved):
    return bool(torch.equal(saved[0], torch.get_rng_state())
                and torch.equal(saved[1], torch.mps.get_rng_state()))


def positive(fixture, backend, oracle):
    dtype = torch.float64 if backend == 'CPU64' else torch.float32
    device = 'mps' if backend == 'MPS32' else 'cpu'
    x = torch.tensor(fixture['logits'], dtype=dtype, device=device)
    if fixture.get('noncontiguous_input'):
        x = x.transpose(1, 2).contiguous().transpose(1, 2)
    x = x.detach().requires_grad_(True)
    x.grad = torch.full_like(x, .375)
    targets = tuple(tuple((token, float(mass)) for token, mass in row)
                    for row in fixture['targets'])
    before, rng = snapshot((x, targets)), rng_snapshot()
    loss = objective.sparse_proximity_ce(x, targets)
    gradient, = torch.autograd.grad(loss, (x,))
    actual_loss = float(tensor_values(loss))
    actual_gradient = tensor_values(gradient.reshape(-1, fixture['vocabulary']))
    atol_loss, atol_grad, rtol = ((1e-11, 1e-10, 1e-10) if backend == 'CPU64'
                                 else (4e-6, 4e-6, 4e-5))
    masks = [[close(a, e, atol_grad, rtol) for a, e in zip(ar, er)]
             for ar, er in zip(actual_gradient, oracle['gradient'])]
    checks = dict(loss=close(actual_loss, oracle['loss'], atol_loss, rtol),
                  all_gradient=all(all(row) for row in masks),
                  input_targets_grad_slots=snapshot((x, targets)) == before,
                  global_rng=rng_equal(rng),
                  declared_shape=tuple(x.shape) == (fixture['batch'], fixture['time'], fixture['vocabulary']),
                  declared_stride=(not x.is_contiguous()) if fixture.get('noncontiguous_input') else x.is_contiguous())
    ce = None
    if fixture.get('genuine_onehot_CE_comparison'):
        other = torch.tensor(fixture['logits'], dtype=dtype, device=device, requires_grad=True)
        other.grad = torch.full_like(other, -.625)
        other_before = snapshot(other)
        labels = torch.tensor([row[0][0] for row in fixture['targets']], dtype=torch.int64, device=device)
        genuine = F.cross_entropy(other.reshape(-1, fixture['vocabulary']), labels, reduction='mean')
        genuine_gradient, = torch.autograd.grad(genuine, (other,))
        ce_loss = float(tensor_values(genuine))
        ce_gradient = tensor_values(genuine_gradient.reshape(-1, fixture['vocabulary']))
        ce_masks = [[close(a, e, atol_grad, rtol) for a, e in zip(ar, er)]
                    for ar, er in zip(actual_gradient, ce_gradient)]
        ce = dict(loss=ce_loss, gradient=ce_gradient, gradient_masks=ce_masks,
                  loss_pass=close(actual_loss, ce_loss, atol_loss, rtol),
                  gradient_pass=all(all(row) for row in ce_masks),
                  reference_input_grad_slots=snapshot(other) == other_before)
        checks['genuine_CE'] = ce['loss_pass'] and ce['gradient_pass'] and ce['reference_input_grad_slots']
        checks['global_rng'] = rng_equal(rng)
    report = dict(id=backend + '_' + fixture['id'], backend=backend, fixture=fixture['id'],
                  actual_loss=actual_loss, actual_gradient=actual_gradient, oracle=oracle,
                  gradient_masks=masks, checks=checks, genuine_CE=ce,
                  loss_atol=atol_loss, gradient_atol=atol_grad, rtol=rtol)
    write(report['id'] + '.json', report)
    assert all(checks.values()), report['id']
    return report


def invalid_payload(family, backend):
    """Every branch constructs a concrete payload before the tested call.

    Sparse and meta are CPU32-only contract checks. No MPS sparse/FP64 fixture.
    All other ordinary tensors reside on the declared backend (except not-tensor).
    """
    dtype = torch.float64 if backend == 'CPU64' else torch.float32
    device = 'mps' if backend == 'MPS32' else 'cpu'
    x = torch.full((1, 2, 5), .125, dtype=dtype, device=device, requires_grad=True)
    targets = (((1, 1.),), ((3, 1.),))
    autocast = None
    if family == 'pn_logits_not_tensor': x = [[[.125] * 5] * 2]
    elif family == 'pn_sparse_layout':
        x = torch.sparse_coo_tensor(torch.tensor([[0], [0], [1]]),
                                    torch.tensor([.125]), (1, 2, 5))
    elif family == 'pn_rank2': x = torch.full((2, 5), .125, dtype=dtype, device=device)
    elif family == 'pn_dtype_int64': x = torch.full((1, 2, 5), 1, dtype=torch.int64, device=device)
    elif family == 'pn_dtype_float16': x = torch.full((1, 2, 5), .125, dtype=torch.float16, device=device)
    elif family == 'pn_device_meta': x = torch.empty((1, 2, 5), dtype=torch.float32, device='meta')
    elif family in ('pn_batch_zero', 'pn_time_zero', 'pn_vocab_one', 'pn_batch_nine', 'pn_time_129', 'pn_vocab_50258'):
        shape = {'pn_batch_zero': (0, 2, 5), 'pn_time_zero': (1, 0, 5),
                 'pn_vocab_one': (1, 2, 1), 'pn_batch_nine': (9, 2, 5),
                 'pn_time_129': (1, 129, 5), 'pn_vocab_50258': (1, 1, 50258)}[family]
        x = torch.full(shape, .125, dtype=dtype, device=device)
    elif family in ('pn_logits_nan', 'pn_logits_inf'):
        x = torch.tensor([[[float('nan') if family == 'pn_logits_nan' else float('inf'), .125, .25, .5, 1.],
                           [.125] * 5]], dtype=dtype, device=device, requires_grad=True)
    elif family == 'pn_targets_list': targets = list(targets)
    elif family == 'pn_target_count_short': targets = targets[:1]
    elif family == 'pn_target_count_long': targets = targets + (targets[0],)
    elif family == 'pn_empty_row': targets = ((), targets[1])
    elif family == 'pn_row_list': targets = (list(targets[0]), targets[1])
    elif family == 'pn_pair_list': targets = (([1, 1.],), targets[1])
    elif family == 'pn_pair_length3': targets = (((1, 1., 0),), targets[1])
    elif family == 'pn_ID_bool': targets = (((True, 1.),), targets[1])
    elif family == 'pn_ID_negative': targets = (((-1, 1.),), targets[1])
    elif family == 'pn_ID_out_of_vocab': targets = (((5, 1.),), targets[1])
    elif family == 'pn_ID_duplicate': targets = (((1, .5), (1, .5)), targets[1])
    elif family.startswith('pn_mass_') and family != 'pn_mass_tensor':
        mass = {'pn_mass_integer': 1, 'pn_mass_bool': True, 'pn_mass_zero': 0.,
                'pn_mass_negative': -.25, 'pn_mass_nan': float('nan'), 'pn_mass_inf': float('inf'),
                'pn_mass_above_one': 1.25, 'pn_mass_sum_wrong': .75}[family]
        targets = (((1, mass),), targets[1])
    elif family == 'pn_row_oversized': targets = (tuple((i, 1. / 6) for i in range(6)), targets[1])
    elif family == 'pn_cpu_autocast_active': autocast = 'cpu'
    elif family == 'pn_mps_autocast_active': autocast = 'mps'
    elif family == 'pn_late_second_row_invalid': targets = (targets[0], ((False, 1.),))
    elif family == 'pn_targets_tensor': targets = torch.tensor([[1., 1.], [3., 1.]], dtype=dtype, device=device)
    elif family == 'pn_pair_tensor': targets = ((torch.tensor([1., 1.], dtype=dtype, device=device),), targets[1])
    elif family == 'pn_row_tensor': targets = (torch.tensor([[1., 1.]], dtype=dtype, device=device), targets[1])
    elif family == 'pn_mass_tensor': targets = (((1, torch.tensor(1., dtype=dtype, device=device)),), targets[1])
    else: raise AssertionError('Unimplemented family: ' + family)
    if type(x) is torch.Tensor and x.device.type != 'meta' and x.layout == torch.strided and x.is_floating_point():
        x.grad = torch.full_like(x, .625)
    return x, targets, autocast


def negative(family, backend):
    x, targets, autocast = invalid_payload(family, backend)
    before, rng = snapshot((x, targets)), rng_snapshot()
    context = torch.autocast(autocast, dtype=torch.bfloat16 if autocast == 'cpu' else torch.float16) if autocast else contextlib.nullcontext()
    # Constructor/context errors are NOT credited as tested-function rejections.
    rejection = None
    with context:
        try:
            objective.sparse_proximity_ce(x, targets)
        except ValueError as error:
            rejection = str(error)
    checks = dict(ValueError=rejection is not None, input_targets_grad_slots=snapshot((x, targets)) == before,
                  global_rng=rng_equal(rng), autocast_restored=not torch.is_autocast_enabled('cpu') and not torch.is_autocast_enabled('mps'))
    report = dict(id=backend + '_' + family, family=family, backend=backend,
                  payload_constructed=True, rejection=rejection, checks=checks)
    write(report['id'] + '.json', report)
    assert all(checks.values()), report['id']
    return report


def run(plan):
    global torch, F, objective
    with CpuPrerequisiteGuard(R / 'startup-guard.json', ROOT, timeout_seconds=120, poll_seconds=.25):
        import torch
        from torch.nn import functional as F
        from research.block_norm_v1.mps_guard import StrictMpsGuard
        torch.set_num_threads(4)
        assert torch.__version__ == plan['runtime']['torch_version']
        assert torch.version.git_version == plan['runtime']['torch_git']
        assert torch.get_num_threads() == 4
    with StrictMpsGuard(R / 'mps-guard.json', ROOT):
        saved_rng = rng_snapshot()
        try:
            objective = importlib.import_module('research.top_proximity_objective_v1')
            assert rng_equal(saved_rng)
            with CpuPrerequisiteGuard(R / 'oracle-guard.json', ROOT, timeout_seconds=120, poll_seconds=.25):
                oracles = {(backend, f['id']): scalar_oracle(f, backend)
                           for backend in BACKENDS for f in plan['fixtures']}
            positives, negatives = {}, []
            for backend in BACKENDS:
                guard = (CpuPrerequisiteGuard(R / (backend + '-guard.json'), ROOT, timeout_seconds=120, poll_seconds=.25)
                         if backend != 'MPS32' else contextlib.nullcontext())
                with guard:
                    for fixture in plan['fixtures']:
                        result = positive(fixture, backend, oracles[backend, fixture['id']])
                        positives[backend, fixture['id']] = result
                    for family in plan['families']:
                        if backend in SPECIAL.get(family, BACKENDS):
                            negatives.append(negative(family, backend))
            for fixture in plan['fixtures']:
                cpu, mps = (positives[backend, fixture['id']] for backend in ('CPU32', 'MPS32'))
                checks = dict(scalar_loss=close(cpu['actual_loss'], mps['actual_loss'], 3e-5, 0.))
                report = dict(id='PAIR_' + fixture['id'], CPU32_loss=cpu['actual_loss'], MPS32_loss=mps['actual_loss'],
                              checks=checks, atol=3e-5, rtol=0.)
                write(report['id'] + '.json', report)
                assert all(checks.values()), report['id']
            final_rng_equal = rng_equal(saved_rng)
            write('rng-preservation.json', dict(global_cpu_mps_rng_unchanged=final_rng_equal))
            assert final_rng_equal
            ids = [p.stem for p in R.glob('*.json') if p.stem.startswith(('CPU64_', 'CPU32_', 'MPS32_', 'PAIR_'))]
            assert sorted(ids) == sorted(plan['output_ids']) and len(ids) == len(set(ids))
            assert all(sha(ROOT / path) == digest for path, digest in plan['sources'].items())
            write('result.json', dict(status='complete_finite_supplied_logit_objective_only',
                                     positive_reports=len(positives), negative_reports=len(negatives),
                                     paired_reports=len(plan['fixtures']), unique_output_ids=len(ids),
                                     no_model_corpus_optimizer_scoring_or_quality=True, no_retry=True, no_replay=True))
        finally:
            torch.set_rng_state(saved_rng[0])
            torch.mps.set_rng_state(saved_rng[1])
        assert rng_equal(saved_rng)


def main():
    plan_path = R / 'plan.json'
    assert len(sys.argv) == 2 and sha(plan_path) == sys.argv[1]
    plan = json.loads(plan_path.read_text())
    assert not any((R / name).exists() for name in ('attempt.json', 'STOP', 'result.json'))
    assert all(sha(ROOT / path) == digest for path, digest in plan['sources'].items())
    gate = json.loads((GATE / 'review.json').read_text())
    assert gate['accepted'] and gate['proof_plan_sha256'] == sha(plan_path)
    assert json.loads((GATE / 'guard.json').read_text())['status'] == 'finished'
    assert json.loads((GATE / 'STOP').read_text())['status'] == 'complete_source_gate_only'
    assert sys.version.split()[0] == plan['runtime']['python']
    assert all(os.environ.get(key) == value for key, value in plan['environment'].items())
    assert __import__('shutil').disk_usage(ROOT).free >= plan['entry_target_bytes']
    with (ROOT / 'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write('attempt.json', dict(pid=os.getpid(), plan_sha256=sha(plan_path), started=time.time(), no_retry=True, no_replay=True))
        try:
            run(plan)
        except BaseException as error:
            write('STOP', dict(status='closed_failed', error=repr(error), no_retry=True, no_replay=True))
            raise
        else:
            write('STOP', dict(status='complete_finite_functional_fixture_only', no_retry=True, no_replay=True))


if __name__ == '__main__':
    main()
