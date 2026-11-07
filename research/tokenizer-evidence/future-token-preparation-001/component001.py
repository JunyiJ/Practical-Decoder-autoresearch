"""NEW seed-free target alignment and objective/gradient oracle, tiny scope only."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import torch
from research.future_token_objective_v1 import ARMS, align, objective
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
R = Path(__file__).with_suffix('')
TOLS = {'cpu64': (1e-11, 1e-12, 1e-10), 'cpu32': (3e-5, 4e-6, 4e-5),
        'mps32': (3e-5, 4e-6, 4e-5)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(name, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    assert sum(p.stat().st_size for p in R.iterdir() if p.is_file())+len(raw) <= 2*1024**2
    with (R/name).open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())


def cases():
    # All new literal families; no trained weights, sampled IDs, or old stress rows.
    specs = [('one_position', 1, 1, 3, 1), ('two_batches', 2, 3, 7, 2),
             ('flat_heads', 2, 5, 11, 0), ('signed_heads', 1, 7, 17, 4),
             ('sharp_targets', 2, 4, 19, 8), ('shifted_heads', 1, 9, 23, 3)]
    rows = []
    for name, b, t, v, scale in specs:
        seq = [[(i*5+j*3+1) % v for j in range(t+2)] for i in range(b)]
        first = [[[(((i*7+j*11+k*5) % 29)-14)*scale/16 for k in range(v)]
                  for j in range(t)] for i in range(b)]
        second = [[[(((i*13+j*7+k*3) % 31)-15)*scale/16 for k in range(v)]
                   for j in range(t)] for i in range(b)]
        if name == 'sharp_targets':
            for i in range(b):
                for j in range(t):
                    first[i][j][seq[i][j+1]] = 12.
                    second[i][j][seq[i][j+2]] = 11.
        if name == 'shifted_heads':
            first = [[[z+32 for z in row] for row in batch] for batch in first]
            second = [[[z-16 for z in row] for row in batch] for batch in second]
        rows.append((name, seq, first, second))
    return rows


def scalar_oracle(values, targets):
    # Independent scalar LSE and analytic derivative, no Torch or autograd.
    count = sum(len(batch) for batch in values); losses = []; gradients = []
    for batch, ys in zip(values, targets):
        gs = []
        for row, y in zip(batch, ys):
            m = max(row); exps = [math.exp(x-m) for x in row]; s = math.fsum(exps)
            losses.append((m-row[y])+math.log(s))
            g = [x/s/count for x in exps]; g[y] -= 1/count; gs.append(g)
        gradients.append(gs)
    return math.fsum(losses)/count, gradients


def preserve(tensors, action):
    snapshots = [(x, x.detach().clone(), None if x.grad is None else x.grad.clone())
                 for x in tensors]
    try:
        return action()
    finally:
        for x, value, grad in snapshots:
            torch.testing.assert_close(x, value, atol=0., rtol=0., equal_nan=True)
            assert (x.grad is None) == (grad is None)
            if grad is not None:
                torch.testing.assert_close(x.grad, grad, atol=0., rtol=0., equal_nan=True)


def evaluate(label, device, dtype):
    f_atol, g_atol, g_rtol = TOLS[label]; reports = []; comparisons = []
    seq = torch.tensor([[2, 0, 4, 1, 3]], device=device, dtype=torch.int64)
    a = align(seq, 5)
    assert [x.cpu().tolist() for x in a] == [[[2, 0, 4]], [[0, 4, 1]], [[4, 1, 3]]]
    a[0][0, 0] = 1; assert seq[0, 0].item() == 2  # outputs own their storage
    for name, sequence, first, second in cases():
        y1 = [row[1:-1] for row in sequence]; y2 = [row[2:] for row in sequence]
        main_ref, main_grad = scalar_oracle(first, y1)
        for arm in ARMS:
            x = torch.tensor(first, device=device, dtype=dtype, requires_grad=True)
            z = None if arm == 'native_next' else torch.tensor(second, device=device, dtype=dtype, requires_grad=True)
            ids = torch.tensor(sequence, device=device, dtype=torch.int64)
            x.grad = torch.full_like(x, .125)
            tensors = [x, ids]
            if z is not None:
                z.grad = torch.full_like(z, -.25); tensors.append(z)
            def action():
                output = objective(arm, x, z, ids)
                gs = torch.autograd.grad(output['training_loss'], [x] if z is None else [x, z])
                return output, gs
            output, gs = preserve(tensors, action)
            aux_ref, aux_grad = (None, None) if z is None else scalar_oracle(second, y1 if arm == 'duplicate_next' else y2)
            expected = main_ref if aux_ref is None else .5*(main_ref+aux_ref)
            expected_g = [torch.tensor(main_grad, dtype=torch.float64)*(1. if z is None else .5)]
            if z is not None:
                expected_g.append(torch.tensor(aux_grad, dtype=torch.float64)*.5)
            actual_g = [g.detach().cpu().to(torch.float64) for g in gs]
            errors = [float((g-e).abs().max()) for g, e in zip(actual_g, expected_g)]
            flags = [bool(((g-e).abs() <= g_atol+g_rtol*e.abs()).all()) for g, e in zip(actual_g, expected_g)]
            actual = float(output['training_loss'].detach().cpu())
            main_error = abs(float(output['primary_ce'].detach().cpu())-main_ref)
            aux_error = 0. if aux_ref is None else abs(float(output['auxiliary_ce'].detach().cpu())-aux_ref)
            reports.append(dict(case=name, arm=arm, training_loss=actual, oracle=expected,
                forward_abs_error=abs(actual-expected), primary_abs_error=main_error,
                auxiliary_abs_error=aux_error, gradient_abs_max=errors,
                forward_pass=max(abs(actual-expected), main_error, aux_error) <= f_atol,
                gradient_elementwise_pass=all(flags)))
            comparisons.append((name, arm, actual, actual_g))
    publish(label+'.json', dict(rows=reports, tolerance=TOLS[label], input_gradslot_preservation=True,
        exact_literal_alignment_and_output_storage=True))
    return comparisons, reports


def invalids(device, dtype, label):
    x = torch.full((1, 2, 5), .25, device=device, dtype=dtype, requires_grad=True)
    z = torch.full_like(x, -.125, requires_grad=True)
    ids = torch.tensor([[1, 4, 0, 3]], device=device, dtype=torch.int64)
    x.grad = torch.full_like(x, .5); z.grad = torch.full_like(z, .75)
    bad_ids = ids.clone(); bad_ids[0, -1] = 5
    negative_ids = ids.clone(); negative_ids[0, 0] = -1
    nan = x.detach().clone(); nan[0, 0, 0] = float('nan')
    inf = z.detach().clone(); inf[0, 0, 1] = float('inf')
    checks = [
        ('arm_type', lambda: objective(1, x, z, ids)), ('arm_unknown', lambda: objective('future3', x, z, ids)),
        ('primary_type', lambda: objective('future_two', [], z, ids)),
        ('primary_rank', lambda: objective('future_two', x[0], z, ids)),
        ('primary_half', lambda: objective('future_two', x.to(torch.float16), z, ids)),
        ('sequence_type', lambda: objective('future_two', x, z, [])),
        ('sequence_dtype', lambda: objective('future_two', x, z, ids.float())),
        ('sequence_rank', lambda: objective('future_two', x, z, ids[0])),
        ('sequence_too_short', lambda: objective('future_two', x, z, ids[:, :2])),
        ('sequence_misaligned', lambda: objective('future_two', x, z, ids[:, :3])),
        ('upper_id', lambda: objective('future_two', x, z, bad_ids)),
        ('negative_id', lambda: objective('future_two', x, z, negative_ids)),
        ('aux_missing', lambda: objective('future_two', x, None, ids)),
        ('aux_native', lambda: objective('native_next', x, z, ids)),
        ('aux_shape', lambda: objective('future_two', x, z[:, :1], ids)),
        ('aux_type', lambda: objective('future_two', x, [], ids)),
        ('aux_dtype', lambda: objective('future_two', x, z.to(torch.float16), ids)),
        ('primary_nonfinite', lambda: objective('future_two', nan, z, ids)),
        ('aux_nonfinite', lambda: objective('future_two', x, inf, ids)),
        ('sequence_device', lambda: objective('future_two', x, z, torch.empty((1, 4), device='meta', dtype=torch.int64))),
        ('align_vocab_type', lambda: align(ids, True)), ('align_vocab_low', lambda: align(ids, 1)),
        ('align_vocab_high', lambda: align(ids, 50258)),
        ('batch_zero', lambda: objective('future_two', x[:0], z[:0], ids[:0])),
        ('batch_nine', lambda: objective('future_two', x.repeat(9, 1, 1), z.repeat(9, 1, 1), ids.repeat(9, 1))),
        ('length_over128', lambda: align(torch.zeros((1, 131), device=device, dtype=torch.int64), 5)),
    ]
    assert len(checks) == 26
    names = []
    for name, call in checks:
        try:
            preserve([x, z, ids, bad_ids, negative_ids, nan, inf], call)
        except ValueError:
            names.append(name)
        else:
            raise AssertionError('Invalid objective accepted: '+name)
    with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == 'cpu' else torch.float16):
        try:
            objective('future_two', x, z, ids)
        except ValueError:
            names.append('autocast')
        else:
            raise AssertionError('Autocast accepted')
    publish(label+'-invalids.json', dict(rejected=names, count=len(names), transactional=True))
    return len(names)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--plan-sha', required=True); args = ap.parse_args()
    assert sha(R/'plan.json') == args.plan_sha and not (R/'attempt.json').exists() and not (R/'STOP').exists()
    plan = json.loads((R/'plan.json').read_text())
    assert shutil.disk_usage(ROOT).free >= plan['limits']['entry_with_extra_headroom_bytes']
    for path, expected in plan['execution_sources'].items():
        assert sha(ROOT/path) == expected
    torch.set_num_threads(4)
    assert os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] == '0' and os.environ['PYTORCH_MPS_FAST_MATH'] == '0'
    assert torch.__version__ == plan['torch_version'] and torch.version.git_version == plan['torch_git']
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish('attempt.json', dict(pid=os.getpid(), plan_sha256=args.plan_sha))
        try:
            with StrictMpsGuard(R/'mps-guard.json', ROOT):
                cpu_rng = torch.get_rng_state().clone(); mps_rng = torch.mps.get_rng_state().clone()
                all_reports = []; result = {}; rejects = {}
                try:
                    for label, device, dtype in [('cpu64', 'cpu', torch.float64), ('cpu32', 'cpu', torch.float32), ('mps32', 'mps', torch.float32)]:
                        if device == 'cpu':
                            with CpuPrerequisiteGuard(R/(label+'-guard.json'), ROOT, timeout_seconds=120, poll_seconds=.25):
                                result[label], reports = evaluate(label, device, dtype); rejects[label] = invalids(device, dtype, label)
                        else:
                            result[label], reports = evaluate(label, device, dtype); rejects[label] = invalids(device, dtype, label)
                        all_reports.extend(reports)
                    pairs = []
                    for left, right in zip(result['cpu32'], result['mps32']):
                        assert left[:2] == right[:2]
                        gs = [bool(((a-b).abs() <= 4e-6+4e-5*b.abs()).all()) for a, b in zip(left[3], right[3])]
                        pairs.append(dict(case=left[0], arm=left[1], ce_abs=abs(left[2]-right[2]),
                                          gradient_elementwise_pass=all(gs), forward_pass=abs(left[2]-right[2]) <= 3e-5))
                    publish('paired.json', dict(rows=pairs, CE_atol=3e-5, CE_rtol=0., grad_atol=4e-6, grad_rtol=4e-5))
                finally:
                    torch.set_rng_state(cpu_rng); torch.mps.set_rng_state(mps_rng)
                assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(mps_rng, torch.mps.get_rng_state())
                for path, expected in plan['execution_sources'].items():
                    assert sha(ROOT/path) == expected
                passed = all(r['forward_pass'] and r['gradient_elementwise_pass'] for r in all_reports+pairs)
                publish('result.json', dict(status='passed' if passed else 'failed_fixed_numerical_predicate',
                    comparisons=len(all_reports), rejects=rejects, caller_cpu_mps_rng_unchanged=True,
                    all_fixed_elementwise_predicates=passed, scope='Tiny objective/target alignment only; no models/corpus/checkpoints/quality claim'))
                assert passed, 'Original fixed numerical predicate failed; no retry or rescue'
            publish('STOP', dict(status='complete_original_tiny_future_token_objective', no_replay=True, no_retry=True))
        except BaseException as error:
            if not (R/'STOP').exists():
                publish('STOP', dict(status='closed_failed', error=repr(error), no_retry=True))
            raise


if __name__ == '__main__':
    main()
